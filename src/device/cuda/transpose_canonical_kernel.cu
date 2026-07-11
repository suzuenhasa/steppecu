// src/device/cuda/transpose_canonical_kernel.cu
//
// GPU primitive that rewrites a SNP-major genotype source into the individual-major
// canonical layout the rest of the pipeline consumes, doing the transpose, the
// selected-individual gather, and the code re-encoding in a single pass. A CUDA TU
// private to the device layer.
//
// Reference: docs/reference/src_device_cuda_transpose_canonical_kernel.cu.md
#include "device/cuda/transpose_canonical_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"
#include "core/internal/launch_config.hpp"
#include "device/cuda/check.cuh"

namespace steppe::device {

namespace {

constexpr int kTransposeBlock = 256;

// The encoding map — reference §6
__device__ __forceinline__ std::uint8_t apply_encoding(std::uint8_t code,
                                                       TransposeEncoding enc) {
    switch (enc) {
        case TransposeEncoding::Identity:
        default:
            return code;
    }
}

// The transpose kernel: one thread per output byte — reference §4.
// Produces `out_bytes_per_record` byte-columns per individual (a block width or the whole
// width) and scatters into `out` at row stride `dst_row_stride`, byte-column offset
// `dst_col_off`: block byte (g, b) -> out[g*dst_row_stride + dst_col_off + b].
__global__ void transpose_to_canonical_kernel(
    const std::uint8_t* __restrict__ snp_major, std::size_t src_bytes_per_record,
    const std::size_t* __restrict__ sel_rows, std::size_t n_individuals,
    std::size_t n_snp, std::size_t out_bytes_per_record, TransposeEncoding encoding,
    std::uint8_t* __restrict__ out, std::size_t dst_row_stride, std::size_t dst_col_off) {
    const std::size_t total = n_individuals * out_bytes_per_record;
    for (std::size_t tid =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         tid < total; tid += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t g = tid / out_bytes_per_record;
        const std::size_t b = tid % out_bytes_per_record;

        const std::size_t src_row = sel_rows[g];
        const std::size_t src_byte_in_snp =
            src_row / static_cast<std::size_t>(core::kCodesPerByte);
        const int src_pos =
            static_cast<int>(src_row % static_cast<std::size_t>(core::kCodesPerByte));

        std::uint8_t packed = 0;
        const std::size_t s0 = b * static_cast<std::size_t>(core::kCodesPerByte);
        for (int k = 0; k < core::kCodesPerByte; ++k) {
            const std::size_t s = s0 + static_cast<std::size_t>(k);
            if (s >= n_snp) break;
            const std::uint8_t src_byte =
                snp_major[s * src_bytes_per_record + src_byte_in_snp];
            const std::uint8_t code = core::genotype_code(src_byte, src_pos);
            const std::uint8_t canon = apply_encoding(code, encoding);
            const int shift =
                (core::kCodesPerByte - 1 - k) * core::kBitsPerCode;
            packed = static_cast<std::uint8_t>(
                packed | static_cast<std::uint8_t>(canon << shift));
        }
        out[g * dst_row_stride + dst_col_off + b] = packed;
    }
}

}  // namespace

// Launch wrapper: grid-stride coverage + grid-size clamp — reference §5
void launch_transpose_to_canonical(const std::uint8_t* d_snp_major,
                                   std::size_t src_bytes_per_record,
                                   const std::size_t* d_sel_rows,
                                   std::size_t n_individuals, std::size_t n_snp,
                                   std::size_t out_bytes_per_record,
                                   TransposeEncoding encoding, std::uint8_t* d_out,
                                   cudaStream_t stream, std::size_t dst_row_stride,
                                   std::size_t dst_col_off) {
    const std::size_t total = n_individuals * out_bytes_per_record;
    if (total == 0) return;
    // dst_row_stride == 0 -> contiguous whole-tile write (row stride == this launch's width),
    // byte-identical to the pre-scatter kernel; the streamed device path passes the whole
    // tile's stride + the block's col_off to scatter a block straight into the resident tile.
    if (dst_row_stride == 0) dst_row_stride = out_bytes_per_record;
    const long grid_x = core::grid_stride_extent(static_cast<long>(total), kTransposeBlock);
    transpose_to_canonical_kernel<<<static_cast<unsigned>(grid_x), kTransposeBlock,
                                    0, stream>>>(
        d_snp_major, src_bytes_per_record, d_sel_rows, n_individuals, n_snp,
        out_bytes_per_record, encoding, d_out, dst_row_stride, dst_col_off);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

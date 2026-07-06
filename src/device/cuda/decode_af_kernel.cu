// src/device/cuda/decode_af_kernel.cu
//
// GPU kernel that decodes packed 2-bit genotypes into per-population allele
// frequencies (Q/V/N), plus the thin host wrapper that launches it. The bit-unpack
// and counting math is shared with the CPU oracle via core/internal/decode_af.hpp,
// so the GPU and reference paths cannot diverge on the unpack/missing/divide.
//
// Reference: docs/reference/src_device_cuda_decode_af_kernel.cu.md
#include "device/cuda/decode_af_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"
#include "core/internal/launch_config.hpp"
#include "device/cuda/check.cuh"

namespace steppe::device {

using core::accumulate_genotype_ploidy;
using core::AfResult;
using core::finalize_af_counts;
using core::genotype_code;
using core::kDecodeBlockX;
using core::kDecodeBlockY;

namespace {

// The decode kernel, one thread per population-SNP pair (compute + two-phase coalesced stores) — reference §3
__global__ void __launch_bounds__(kDecodeBlockX * kDecodeBlockY)
decode_af_kernel(const std::uint8_t* __restrict__ packed,
                                 std::size_t bytes_per_record,
                                 const std::size_t* __restrict__ pop_offsets,
                                 int P, long M, int ploidy,
                                 const int* __restrict__ sample_ploidy,
                                 double* __restrict__ Q,
                                 double* __restrict__ V,
                                 double* __restrict__ N) {
    __shared__ double tQ[kDecodeBlockY][kDecodeBlockX + 1];
    __shared__ double tV[kDecodeBlockY][kDecodeBlockX + 1];
    __shared__ double tN[kDecodeBlockY][kDecodeBlockX + 1];

    const int pop0 = static_cast<int>(blockIdx.y) * blockDim.y;
    const long snp0 = static_cast<long>(blockIdx.x) * blockDim.x;

    const long s = snp0 + threadIdx.x;
    const int i = pop0 + static_cast<int>(threadIdx.y);
    if (s < M && i < P) {
        const std::size_t seg_begin = pop_offsets[static_cast<std::size_t>(i)];
        const std::size_t seg_end = pop_offsets[static_cast<std::size_t>(i) + 1];
        const std::size_t byte_in_record =
            static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
        const int pos_in_byte = static_cast<int>(s % core::kCodesPerByte);

        double ac = 0.0;
        std::int64_t n = 0;
        for (std::size_t g = seg_begin; g < seg_end; ++g) {
            const std::uint8_t byte = packed[g * bytes_per_record + byte_in_record];
            const std::uint8_t code = genotype_code(byte, pos_in_byte);
            const int pl = (sample_ploidy != nullptr) ? sample_ploidy[g] : ploidy;
            accumulate_genotype_ploidy(code, pl, ac, n);
        }
        const AfResult r = finalize_af_counts(ac, n);
        tQ[threadIdx.y][threadIdx.x] = r.q;
        tV[threadIdx.y][threadIdx.x] = r.v;
        tN[threadIdx.y][threadIdx.x] = r.n;
    }

    __syncthreads();

    const int bdx = static_cast<int>(blockDim.x);
    const int bdy = static_cast<int>(blockDim.y);
    const int tid = static_cast<int>(threadIdx.y) * bdx + static_cast<int>(threadIdx.x);
    const int p = tid % bdy;
    const int q = tid / bdy;
    const int io = pop0 + p;
    const long so = snp0 + static_cast<long>(q);
    if (io < P && so < M) {
        const std::size_t off = static_cast<std::size_t>(io) +
                                static_cast<std::size_t>(P) * static_cast<std::size_t>(so);
        Q[off] = tQ[p][q];
        V[off] = tV[p][q];
        N[off] = tN[p][q];
    }
}

}  // namespace

// The host launch wrapper: block and grid geometry — reference §7
void launch_decode_af(const std::uint8_t* d_packed,
                      std::size_t bytes_per_record,
                      const std::size_t* d_pop_offsets,
                      int P, long M, int ploidy,
                      const int* d_sample_ploidy,
                      double* d_Q, double* d_V, double* d_N,
                      cudaStream_t stream) {
    const dim3 block(kDecodeBlockX, kDecodeBlockY);
    const int grid_x = core::grid_for_x(M, kDecodeBlockX,
                                        "decode gridDim.x (SNP/M axis) exceeds kMaxGridX "
                                        "(architecture.md §7; cleanup X-7/B6) — tile the SNP axis");
    const dim3 grid(static_cast<unsigned>(grid_x),
                    static_cast<unsigned>(core::grid_for(P, kDecodeBlockY)));
    decode_af_kernel<<<grid, block, 0, stream>>>(d_packed, bytes_per_record,
                                                 d_pop_offsets, P, M, ploidy,
                                                 d_sample_ploidy,
                                                 d_Q, d_V, d_N);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

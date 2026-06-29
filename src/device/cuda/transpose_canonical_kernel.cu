// src/device/cuda/transpose_canonical_kernel.cu
//
// M-FR-1 — the SNP-major -> canonical individual-major TRANSPOSE+GATHER+ENCODING
// primitive ON THE GPU (the format-reader engine; docs/design/format-readers.md
// §2.4, M-FR-1). After this pass the gathered selection is in the canonical
// individual-major packing decode_af / detect_ploidy already consume, so the entire
// downstream genotype pipeline is UNCHANGED — only the read-time layout transposes.
//
// What this TU owns:
//   1. transpose_to_canonical_kernel — ONE thread per OUTPUT BYTE, grid-stride over
//      the output-byte axis (input-size-agnostic coverage); assembles the <=4
//      canonical codes of each owned byte by gathering+encoding from the SNP-major
//      source. Race-free by construction (each output byte has exactly one writer).
//   2. launch_transpose_to_canonical — the narrow launch wrapper (no <<<>>> in host
//      code; the established detect_ploidy_kernel.cu / decode_af_kernel.cu pattern).
//
// CORRECTNESS (the M-FR-1 gate): the transpose is INTEGER / BIT ops only — the 2-bit
// unpack (core::genotype_code, the SAME MSB-first extractor decode_af uses), the
// encoding remap (P0 = IDENTITY), and the MSB-first re-pack into the output byte.
// No floating point, no reduction order, no precision lane — so the device tile is
// bit-identical to the CpuBackend host-loop oracle AND, for PA, to the TGENO tile
// of the same data, by construction (the emulated-FP64 policy is matmul-only and
// does NOT apply here; format-readers.md §2.2 precision note).
//
// THE GATHER (selection + pop-contiguous reorder): output column g maps to source
// individual row d_sel_rows[g] — the IndPartition's selected, Q/V/N-ordered set
// (format-readers.md OQ-6). The SNP-major source interleaves ALL individuals within
// each SNP byte, so the selection happens HERE (output-driven), exactly mirroring
// the TGENO read_tile's per-individual gather (geno_reader.cpp:199-217) with the
// axis swapped.
//
// THE rlen-FLOOR (format-readers.md §3.4): GENO bytes_per_record = max(48,
// ceil(n_ind/4)) means a small-n_ind source has PADDING bytes per SNP record beyond
// the real individuals. Individuals are bounded by the EXPLICIT src rows (each <
// n_ind), so a padding byte is NEVER read as a phantom individual — the kernel
// indexes src_row/4 with src_row < n_ind, never byte-stride*4.
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). It includes the
// SHARED host/device decode primitive (core::genotype_code) so the transpose's
// bit-extraction cannot diverge from the decode / host reader / CpuBackend oracle.
#include "device/cuda/transpose_canonical_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"      // core::genotype_code, kCodesPerByte, kBitsPerCode
#include "core/internal/launch_config.hpp"  // cdiv, kMaxGridX
#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK_KERNEL

namespace steppe::device {

namespace {

constexpr int kTransposeBlock = 256;

/// Apply the native-code -> canonical-code map. P0 PACKEDANCESTRYMAP/GENO is
/// IDENTITY (the source already uses the canonical raw-value 0/1/2/3 convention and
/// MSB-first bit order). A device-side switch (not a host LUT upload) so later
/// encodings (EIGENSTRAT/PLINK) extend this ONE site without a new kernel.
__device__ __forceinline__ std::uint8_t apply_encoding(std::uint8_t code,
                                                       TransposeEncoding enc) {
    switch (enc) {
        case TransposeEncoding::Identity:
        default:
            return code;
    }
}

/// ONE thread per OUTPUT BYTE, GRID-STRIDE over the output-byte axis so coverage is
/// input-size-agnostic (every output byte is written even when the launch grid is
/// clamped below one-thread-per-byte — see launch_transpose_to_canonical; matches the
/// repack_target_kernel / qpadm_fit grid-stride idiom). For each owned byte (gathered
/// individual g, output byte b) the thread assembles the up-to-4 canonical codes of
/// output SNPs {4b, 4b+1, 4b+2, 4b+3} (the codes packed into output byte b of g's
/// canonical record) and writes the single output byte. Each code is gathered from the
/// SNP-major source at its OWN SNP record and the gathered individual's source row,
/// then encoded and packed MSB-first. SNPs >= n_snp (the partial last byte) contribute
/// 0 bits — the same all-zero tail the host packer produces. Race-free: distinct
/// (g, b) own distinct output bytes.
__global__ void transpose_to_canonical_kernel(
    const std::uint8_t* __restrict__ snp_major, std::size_t src_bytes_per_record,
    const std::size_t* __restrict__ sel_rows, std::size_t n_individuals,
    std::size_t n_snp, std::size_t out_bytes_per_record, TransposeEncoding encoding,
    std::uint8_t* __restrict__ out) {
    const std::size_t total = n_individuals * out_bytes_per_record;
    for (std::size_t tid =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         tid < total; tid += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        // Decompose the flat thread id into (output individual g, output byte b). The
        // output is individual-major: record g occupies out[g*out_bpr .. (g+1)*out_bpr).
        const std::size_t g = tid / out_bytes_per_record;
        const std::size_t b = tid % out_bytes_per_record;

        // The source individual ROW for this output column (selection + pop-contiguous
        // reorder). Read once per output byte (4 SNPs share the same source row).
        const std::size_t src_row = sel_rows[g];
        const std::size_t src_byte_in_snp =
            src_row / static_cast<std::size_t>(core::kCodesPerByte);  // which source byte holds src_row
        const int src_pos =
            static_cast<int>(src_row % static_cast<std::size_t>(core::kCodesPerByte));  // its in-byte position

        // Assemble the 4 codes of output SNPs s = 4b + k (k in 0..3), MSB-first: SNP
        // 4b in bits 7-6, 4b+1 in 5-4, ... (the canonical code_in_byte order). A SNP
        // past n_snp contributes 0 (the partial-last-byte tail).
        std::uint8_t packed = 0;
        const std::size_t s0 = b * static_cast<std::size_t>(core::kCodesPerByte);
        for (int k = 0; k < core::kCodesPerByte; ++k) {
            const std::size_t s = s0 + static_cast<std::size_t>(k);
            if (s >= n_snp) break;  // remaining SNPs of this byte are 0 (all-zero tail)
            // Gather: source SNP record s, byte holding src_row, in-byte position src_pos.
            const std::uint8_t src_byte =
                snp_major[s * src_bytes_per_record + src_byte_in_snp];
            const std::uint8_t code = core::genotype_code(src_byte, src_pos);
            const std::uint8_t canon = apply_encoding(code, encoding);
            // Pack canon into output position k MSB-first: bits (6 - 2k)..(7 - 2k).
            const int shift =
                (core::kCodesPerByte - 1 - k) * core::kBitsPerCode;  // 6,4,2,0
            packed = static_cast<std::uint8_t>(
                packed | static_cast<std::uint8_t>(canon << shift));
        }
        out[tid] = packed;  // tid == g*out_bytes_per_record + b (the canonical byte offset)
    }
}

}  // namespace

void launch_transpose_to_canonical(const std::uint8_t* d_snp_major,
                                   std::size_t src_bytes_per_record,
                                   const std::size_t* d_sel_rows,
                                   std::size_t n_individuals, std::size_t n_snp,
                                   std::size_t out_bytes_per_record,
                                   TransposeEncoding encoding, std::uint8_t* d_out,
                                   cudaStream_t stream) {
    const std::size_t total = n_individuals * out_bytes_per_record;
    if (total == 0) return;  // empty tile (no individuals or no SNPs) ⇒ nothing to pack
    // One thread per OUTPUT BYTE ideally, but CLAMP the grid extent to kMaxGridX so the
    // launch is always valid; the kernel's grid-stride loop then covers every output
    // byte regardless (the launch_grid_stride idiom in qpadm_fit_kernels.cu). This is
    // the safety net the old gridDim.x assert was not: that STEPPE_ASSERT compiled out
    // under NDEBUG (Release = the gate build), so an over-cap `total` would otherwise
    // truncate at the unsigned cast below and SILENTLY under-cover the output.
    const long grid_l =
        core::cdiv(static_cast<long>(total), static_cast<long>(kTransposeBlock));
    const long grid_cap = static_cast<long>(core::kMaxGridX);
    const long grid_x = grid_l > grid_cap ? grid_cap : grid_l;
    transpose_to_canonical_kernel<<<static_cast<unsigned>(grid_x), kTransposeBlock,
                                    0, stream>>>(
        d_snp_major, src_bytes_per_record, d_sel_rows, n_individuals, n_snp,
        out_bytes_per_record, encoding, d_out);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

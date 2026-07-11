// src/device/cuda/decode_compact_kernel.cu
//
// Device-resident decode-and-compact kernels: build a per-SNP keep mask, then gather
// the kept [P×M] columns into a gap-free packed arena — the CPU keep-and-pack loop
// moved onto the GPU, bit-for-bit identical to the host result.
// Private CUDA TU: only the launch functions are visible to the rest of steppe_device.
//
// Reference: docs/reference/src_device_cuda_decode_compact_kernel.cu.md
#include "device/cuda/decode_compact_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"  // genotype_code, kCodesPerByte, kBitsPerCode, kCodeMask
#include "core/internal/launch_config.hpp"
#include "device/cuda/check.cuh"
#include "io/filter/snp_summary_reduce.hpp"
#include "steppe/config.hpp"

namespace steppe::device {

namespace {

constexpr int kKeepBlock = 256;

// Autosome keep mask — reference §3
__global__ void autosome_keep_mask_kernel(const int* __restrict__ chrom, long M,
                                          int chrom_min, int chrom_max,
                                          std::uint8_t* __restrict__ flags) {
    const long s = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= M) return;
    const int chr = chrom[s];
    flags[s] = (chr >= chrom_min && chr <= chrom_max) ? std::uint8_t{1} : std::uint8_t{0};
}

// Full extract_f2 keep mask — reference §4
__global__ void regimeb_keep_mask_kernel(const double* __restrict__ Q,
                                         const double* __restrict__ N, int P, long M,
                                         const char* __restrict__ ref,
                                         const char* __restrict__ alt,
                                         const int* __restrict__ chrom,
                                         steppe::FilterConfig cfg, double ploidy_d,
                                         double total_indiv_d, double maxmiss,
                                         std::uint8_t* __restrict__ flags) {
    const long s = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= M) return;

    const steppe::io::filter::PooledSnpSummary ps =
        steppe::io::filter::derive_pooled_summary_one(Q, N, P, s, ploidy_d,
                                                      total_indiv_d);
    bool keep = steppe::io::filter::keep_decision_pooled(
        ps, ref[s], alt[s], chrom[s], cfg, /*membership_ok=*/true);

    if (keep && maxmiss < 1.0) {
        const long base = static_cast<long>(P) * s;
        int n_missing_pops = 0;
        for (int p = 0; p < P; ++p) {
            if (N[base + static_cast<long>(p)] <= 0.0) ++n_missing_pops;
        }
        const double frac =
            static_cast<double>(n_missing_pops) / static_cast<double>(P);
        if (frac > maxmiss) keep = false;
    }

    flags[s] = keep ? std::uint8_t{1} : std::uint8_t{0};
}

// Pooled-per-SNP summary reduction — reference §7.
// One thread per SNP calls the SHARED derive_pooled_summary_one over the resident
// [P×M] Q/N; the reduction runs the exact p = 0..P-1 order + the FMA-safe
// pooled_ref_fma intrinsics of the host path, so the four written planes are
// bit-for-bit identical to the host reduction over the same (D2H-copied) Q/N bits.
__global__ void pooled_summary_kernel(const double* __restrict__ Q,
                                      const double* __restrict__ N, int P, long M,
                                      double ploidy_d, double total_indiv_d,
                                      double* __restrict__ ref_af,
                                      double* __restrict__ minor_af,
                                      double* __restrict__ missing,
                                      double* __restrict__ allele_count) {
    const long s = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= M) return;
    const steppe::io::filter::PooledSnpSummary ps =
        steppe::io::filter::derive_pooled_summary_one(Q, N, P, s, ploidy_d,
                                                      total_indiv_d);
    ref_af[s] = ps.pooled_ref_af;
    minor_af[s] = ps.pooled_minor_af;
    missing[s] = ps.missing_frac;
    allele_count[s] = ps.pooled_allele_count;
}

// Column-gather compaction — reference §5
__global__ void compact_columns_gather_kernel(const double* __restrict__ in, int P,
                                              long M,
                                              const std::uint8_t* __restrict__ flags,
                                              const long* __restrict__ keep_idx,
                                              double* __restrict__ out) {
    const long s = static_cast<long>(blockIdx.x) * blockDim.y + threadIdx.y;
    const int i = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (s >= M || i >= P) return;
    if (flags[s] == 0) return;
    const std::size_t src = static_cast<std::size_t>(i) +
                            static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
    const std::size_t dst = static_cast<std::size_t>(i) +
                            static_cast<std::size_t>(P) *
                                static_cast<std::size_t>(keep_idx[s]);
    out[dst] = in[src];
}

// Packed 2-bit column compaction (apply_snp_filter device path).
// One thread per (individual row r, output byte b) over the grid-stride flat index
// idx = r*out_bpr + b. Each thread composes its output byte from the (up to) four kept source
// columns 4b..4b+3: source column s = kept_cols[4b+t] (the ascending select output, == the host
// repack's kept_cols), code = genotype_code(src[r*src_bpr + s/4], s), placed MSB-first at bit
// (kCodesPerByte-1-t)*kBitsPerCode. This reproduces repack_tile_columns' 2-bit gather bit-for-bit
// (same genotype_code extract, same shift), reading the resident source in place and writing each
// output byte exactly once (no atomics, order-independent).
__global__ void compact_packed_columns_kernel(const std::uint8_t* __restrict__ src,
                                              std::size_t src_bpr,
                                              const long* __restrict__ kept_cols, long n_kept,
                                              std::uint8_t* __restrict__ out,
                                              std::size_t out_bpr, std::size_t n_individuals) {
    const std::size_t total = n_individuals * out_bpr;
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t r = idx / out_bpr;
        const std::size_t b = idx % out_bpr;
        const std::uint8_t* rec = src + r * src_bpr;
        std::uint8_t byte = 0;
        for (int t = 0; t < core::kCodesPerByte; ++t) {
            const long outcol = static_cast<long>(b) * core::kCodesPerByte + t;
            if (outcol >= n_kept) break;
            const long s = kept_cols[outcol];
            const std::uint8_t code = core::genotype_code(
                rec[static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte)],
                static_cast<int>(s));
            const int shift = (core::kCodesPerByte - 1 - t) * core::kBitsPerCode;
            byte = static_cast<std::uint8_t>(
                byte | static_cast<std::uint8_t>((code & core::kCodeMask) << shift));
        }
        out[idx] = byte;
    }
}

}  // namespace

// Launch geometry + grid-size guard — reference §6
void launch_autosome_keep_mask(const int* d_chrom, long M, int chrom_min,
                               int chrom_max, std::uint8_t* d_flags,
                               cudaStream_t stream) {
    if (M <= 0) return;
    const int grid_x = core::grid_for_x(M, kKeepBlock,
                                        "autosome keep-mask gridDim.x (SNP axis) exceeds kMaxGridX "
                                        "(architecture.md §7) — tile the SNP axis");
    autosome_keep_mask_kernel<<<static_cast<unsigned>(grid_x), kKeepBlock, 0, stream>>>(
        d_chrom, M, chrom_min, chrom_max, d_flags);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_regimeb_keep_mask(const double* d_Q, const double* d_N, int P, long M,
                              const char* d_ref, const char* d_alt,
                              const int* d_chrom, const steppe::FilterConfig& cfg,
                              double ploidy, double total_indiv, double maxmiss,
                              std::uint8_t* d_flags, cudaStream_t stream) {
    if (M <= 0) return;
    const int grid_x = core::grid_for_x(M, kKeepBlock,
                                        "regime-B keep-mask gridDim.x (SNP axis) exceeds kMaxGridX "
                                        "(architecture.md §7) — tile the SNP axis");
    regimeb_keep_mask_kernel<<<static_cast<unsigned>(grid_x), kKeepBlock, 0, stream>>>(
        d_Q, d_N, P, M, d_ref, d_alt, d_chrom, cfg, ploidy, total_indiv, maxmiss,
        d_flags);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_compact_columns_gather(const double* d_in, int P, long M,
                                   const std::uint8_t* d_flags,
                                   const long* d_keep_idx, double* d_out,
                                   cudaStream_t stream) {
    if (P <= 0 || M <= 0) return;
    const int bx = core::kDecodeBlockX, by = core::kDecodeBlockY;
    const int grid_x = core::grid_for_x(M, by,
                                        "compact-columns gridDim.x (SNP axis) exceeds kMaxGridX "
                                        "(architecture.md §7) — tile the SNP axis");
    const dim3 block(static_cast<unsigned>(bx), static_cast<unsigned>(by));
    const dim3 grid(static_cast<unsigned>(grid_x),
                    static_cast<unsigned>(core::grid_for(P, bx)));
    compact_columns_gather_kernel<<<grid, block, 0, stream>>>(d_in, P, M, d_flags,
                                                              d_keep_idx, d_out);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_compact_packed_columns(const std::uint8_t* d_src, std::size_t src_bytes_per_record,
                                   const long* d_kept_cols, long n_kept, std::uint8_t* d_out,
                                   std::size_t out_bytes_per_record, std::size_t n_individuals,
                                   cudaStream_t stream) {
    if (n_individuals == 0 || out_bytes_per_record == 0 || n_kept <= 0) return;
    const long total =
        static_cast<long>(n_individuals) * static_cast<long>(out_bytes_per_record);
    const int grid = core::grid_stride_extent(total, kKeepBlock);
    compact_packed_columns_kernel<<<static_cast<unsigned>(grid), kKeepBlock, 0, stream>>>(
        d_src, src_bytes_per_record, d_kept_cols, n_kept, d_out, out_bytes_per_record,
        n_individuals);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pooled_summary(const double* d_Q, const double* d_N, int P, long M,
                           double ploidy_d, double total_indiv_d,
                           double* d_ref_af, double* d_minor_af, double* d_missing,
                           double* d_allele_count, cudaStream_t stream) {
    if (P <= 0 || M <= 0) return;
    const int grid_x = core::grid_for_x(M, kKeepBlock,
                                        "pooled-summary gridDim.x (SNP axis) exceeds kMaxGridX "
                                        "(architecture.md §7) — tile the SNP axis");
    pooled_summary_kernel<<<static_cast<unsigned>(grid_x), kKeepBlock, 0, stream>>>(
        d_Q, d_N, P, M, ploidy_d, total_indiv_d, d_ref_af, d_minor_af, d_missing,
        d_allele_count);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

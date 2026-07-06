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

}  // namespace steppe::device

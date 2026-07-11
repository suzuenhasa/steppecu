// src/device/cuda/cuda_backend_ld_prune.cu
//
// CudaBackend override for the windowed-r2 LD pruner (`--ld-prune WIN:STEP:R2`) — the plink2
// --indep-pairwise analogue. It uploads the packed diploid tile device-resident, streams the SNP
// axis by tile (the KING/FST s_lo idiom) decoding a SNP-major dosage pane ONCE per tile, and per
// tile computes (a) the per-variant global {nm, Σg, Σg²} and (b) every within-window same-
// chromosome pair's r^2 > threshold decision on the GPU. Only the per-SNP stats + the per-pair
// boolean flags cross PCIe; the genotype-scale reductions stay on the device. The greedy backward-
// scan selection that turns those flags into a kept set is a cheap host loop faithfully mirroring
// plink2's IndepPairwise (default --indep-order 2): within each window, scanning the later variant
// of each pair first and each earlier partner backward, remove the higher-major-allele-frequency
// (lower-MAF) variant of every over-threshold pair (ties remove the later variant), with
// monomorphic variants pre-removed; the window slides by STEP and resets at each chromosome
// boundary. A CUDA TU private to steppe_device, mirroring cuda_backend_king.cu.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <vector>

#include "core/internal/launch_config.hpp"  // kMaxGridX
#include "device/cuda/check.cuh"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/ld_prune_kernel.cuh"

namespace steppe::device {

namespace {

// plink2's kSmallEpsilon (2^-44): the r^2 threshold is scaled up by (1 + eps) and the major-
// allele-frequency tie-break compares maj[first] <= maj[second]*(1 + eps). Replicated verbatim so
// the boundary (exact-tie) decisions match plink2.
constexpr double kPlinkSmallEpsilon = 0x1p-44;

// Fraction of free VRAM the resident SNP-major dosage pane may occupy (leaving room for the packed
// tile, the chrom vector, and the small per-tile stat/flag slabs).
constexpr std::size_t kLdDosBudgetDivisor = 4;
constexpr std::size_t kLdFreeVramFallbackBytes = std::size_t{1} << 30;  // 1 GiB

}  // namespace

std::vector<std::uint8_t> CudaBackend::ld_prune_windowed(const DecodeTileView& tile,
                                                         std::span<const int> chrom, int window,
                                                         int step, double r2_thresh) {
    guard_device();
    const long M = static_cast<long>(tile.n_snp);
    const int N = static_cast<int>(tile.n_individuals);

    // A degenerate request keeps everything (no pair can be formed).
    if (M <= 0) return {};
    if (N <= 0 || window <= 1 || step <= 0 || r2_thresh <= 0.0 ||
        chrom.size() != static_cast<std::size_t>(M)) {
        return std::vector<std::uint8_t>(static_cast<std::size_t>(M), std::uint8_t{1});
    }

    const long W1 = static_cast<long>(window) - 1;
    const double r2_thresh_eps = r2_thresh * (1.0 + kPlinkSmallEpsilon);

    // --- Upload the packed tile + the per-SNP chromosome vector (both persistent) ---
    DeviceBuffer<std::uint8_t> dPacked;  // staging; empty when the tile is device-resident
    const std::uint8_t* packed_dev = packed_device_ptr(tile, dPacked);
    DeviceBuffer<int> dChrom(static_cast<std::size_t>(M));
    h2d_async(dChrom, chrom.data(), static_cast<std::size_t>(M), stream_.get());

    // --- Host result accumulators (filled tile-by-tile) ---
    std::vector<long> nm(static_cast<std::size_t>(M));
    std::vector<long> sum(static_cast<std::size_t>(M));
    std::vector<long> ssq(static_cast<std::size_t>(M));
    std::vector<std::uint8_t> over(static_cast<std::size_t>(M) * static_cast<std::size_t>(W1), 0);

    // --- SNP tile size: bound the resident (tileM + window - 1) x N dosage pane to ~free/divisor.
    std::size_t free_b = capabilities().free_vram_bytes;
    if (free_b == 0) free_b = kLdFreeVramFallbackBytes;
    const std::size_t row_bytes = static_cast<std::size_t>(N);  // one byte per sample per SNP
    std::size_t budget_rows = (free_b / kLdDosBudgetDivisor) / (row_bytes == 0 ? 1u : row_bytes);
    long tileM = static_cast<long>(budget_rows) - (static_cast<long>(window) - 1);
    if (const char* env = std::getenv("STEPPE_LDPRUNE_TILE")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0) tileM = v;
    }
    if (tileM < 1) tileM = 1;
    if (tileM > M) tileM = M;
    // Keep the pairwise grid (n_tgt*(window-1)) under the x-grid limit.
    while (tileM > 1 && static_cast<unsigned long long>(tileM) * static_cast<unsigned long long>(W1) >
                            static_cast<unsigned long long>(core::kMaxGridX)) {
        tileM /= 2;
    }

    const std::size_t dec_cap = static_cast<std::size_t>(tileM + window - 1) * static_cast<std::size_t>(N);
    DeviceBuffer<std::uint8_t> dDos(dec_cap == 0 ? 1u : dec_cap);
    DeviceBuffer<long> dNm(static_cast<std::size_t>(tileM));
    DeviceBuffer<long> dSum(static_cast<std::size_t>(tileM));
    DeviceBuffer<long> dSsq(static_cast<std::size_t>(tileM));
    DeviceBuffer<std::uint8_t> dOver(static_cast<std::size_t>(tileM) * static_cast<std::size_t>(W1));

    for (long s_lo = 0; s_lo < M; s_lo += tileM) {
        const long n_tgt = std::min<long>(tileM, M - s_lo);
        const long n_dec = std::min<long>(n_tgt + window - 1, M - s_lo);
        launch_ld_dosage_decode_snpmajor(packed_dev, tile.bytes_per_record, N, s_lo, n_dec,
                                         dDos.data(), stream_.get());
        launch_ld_variant_stats(dDos.data(), N, n_tgt, dNm.data(), dSum.data(), dSsq.data(),
                                stream_.get());
        launch_ld_pairwise_over(dDos.data(), N, s_lo, n_tgt, n_dec, M, dChrom.data(), window,
                                r2_thresh_eps, dOver.data(), stream_.get());
        d2h_async(nm.data() + s_lo, dNm, static_cast<std::size_t>(n_tgt), stream_.get());
        d2h_async(sum.data() + s_lo, dSum, static_cast<std::size_t>(n_tgt), stream_.get());
        d2h_async(ssq.data() + s_lo, dSsq, static_cast<std::size_t>(n_tgt), stream_.get());
        d2h_async(over.data() + static_cast<std::size_t>(s_lo) * static_cast<std::size_t>(W1), dOver,
                  static_cast<std::size_t>(n_tgt) * static_cast<std::size_t>(W1), stream_.get());
        // The per-tile device slabs are reused next iteration; drain before overwriting.
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    }

    // --- Per-variant major-allele frequency + monomorphic pre-removal (plink2 removes zero-
    // variance variants before any pairwise comparison; they never enter the kept set). ---
    std::vector<double> maj(static_cast<std::size_t>(M), 0.0);
    std::vector<char> removed(static_cast<std::size_t>(M), 0);
    for (long s = 0; s < M; ++s) {
        const long nn = nm[static_cast<std::size_t>(s)];
        const long SS = sum[static_cast<std::size_t>(s)];
        const long QQ = ssq[static_cast<std::size_t>(s)];
        const double var = static_cast<double>(QQ * nn - SS * SS);
        if (nn <= 0 || var <= 0.0) {
            removed[static_cast<std::size_t>(s)] = 1;  // all-missing or monomorphic
            continue;
        }
        const double af = static_cast<double>(SS) / (2.0 * static_cast<double>(nn));
        maj[static_cast<std::size_t>(s)] = af > 0.5 ? af : (1.0 - af);
    }

    // --- Greedy backward-scan selection, per chromosome, sliding by STEP (plink2 --indep-order 2). ---
    long c0 = 0;
    while (c0 < M) {
        long c1 = c0;
        while (c1 < M && chrom[static_cast<std::size_t>(c1)] == chrom[static_cast<std::size_t>(c0)]) {
            ++c1;
        }
        long win_start = c0;
        long prev_end = c0;  // plink winpos_split: end of the PRIOR window's scanned range
        while (true) {
            const long win_end = std::min<long>(win_start + window, c1);
            // plink compacts removed/expired variants out of the window buffer and rescans only
            // the NEW variants (those that just entered) as `second` (winpos_split). Model that by
            // ranging `second` over [new_lo, win_end): new_lo is where the prior window ended, so a
            // variant already scanned as `second` (and possibly removed) is never re-scanned.
            const long new_lo = std::max<long>(prev_end, win_start);
            for (long second = win_end - 1; second >= new_lo; --second) {
                // Within a single window scan the OUTER `second` does NOT skip an already-removed
                // variant — a `second` removed earlier in THIS scan still prunes its earlier LD
                // partners (transitive pruning). Only the inner `first` loop skips removed.
                for (long first = second - 1; first >= win_start; --first) {
                    if (removed[static_cast<std::size_t>(first)]) continue;
                    const long d = second - first;  // in [1, window-1]
                    if (!over[static_cast<std::size_t>(first) * static_cast<std::size_t>(W1) +
                              static_cast<std::size_t>(d - 1)]) {
                        continue;
                    }
                    if (maj[static_cast<std::size_t>(first)] <=
                        maj[static_cast<std::size_t>(second)] * (1.0 + kPlinkSmallEpsilon)) {
                        removed[static_cast<std::size_t>(second)] = 1;  // drop the later variant
                        break;                                          // second gone; next second
                    }
                    removed[static_cast<std::size_t>(first)] = 1;       // drop the earlier variant
                    // no break: keep scanning earlier partners of this same `second`
                }
            }
            if (win_end >= c1) break;
            prev_end = win_end;
            win_start += step;
        }
        c0 = c1;
    }

    std::vector<std::uint8_t> keep(static_cast<std::size_t>(M));
    for (long s = 0; s < M; ++s) {
        keep[static_cast<std::size_t>(s)] = removed[static_cast<std::size_t>(s)] ? std::uint8_t{0}
                                                                                 : std::uint8_t{1};
    }
    return keep;
}

}  // namespace steppe::device

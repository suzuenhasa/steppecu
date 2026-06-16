// src/core/internal/f2_estimator.hpp
//
// THE shared, per-element, bias-corrected f2 primitive — the single source of
// the f2 formula so the CPU reference oracle and the GPU feeder CANNOT diverge
// (architecture.md §5 S2, §7, §13; ROADMAP §4, §5).
//
// Lifted verbatim (logic, not structure) from the validated spike: the het
// correction `hc = q(1-q)/max(N-1,1)` (f2_emu_spike.cu build_host_arrays /
// reference_f2; f2_prec_acc.cu / f2_timing.cu --load builders) and the unbiased
// per-SNP f2 summand `(p_i - p_j)² - hc_i - hc_j` (reference_f2). The spike's
// hardcoded denominators (`max(N-1,1)` floor, the bug-shaped `/198.0` and
// `N=100`) are replaced here by the per-SNP `N` argument and the named
// `kHetCorrDenomFloor` — sample size is NEVER hardcoded (ROADMAP §4).
//
// These are pure scalar functions, CPU-unit-testable with no GPU. The GPU
// production hot path is the 3-GEMM reformulation (architecture.md §5 S2); these
// primitives are the per-element numerics invoked inside the fused feeder/oracle,
// NOT the structure of the hot path. The diagonal-of-large-sums cancellation
// lives in the small assembled numerator (`assemble_f2_numerator`), kept in
// native FP64 in every precision mode (architecture.md §12).
//
// ---------------------------------------------------------------------------
// __host__ __device__ portability
// ---------------------------------------------------------------------------
// This header is included BOTH by host-pure `core` TUs (compiled by the C++
// host compiler, where `__host__`/`__device__` are unknown) AND by device TUs
// (compiled by nvcc). We define STEPPE_HD to the CUDA qualifiers under nvcc and
// to nothing otherwise, so the exact SAME functions compile and unit-test on the
// CPU and run on the GPU — that is the whole point of one shared primitive.
#ifndef STEPPE_CORE_INTERNAL_F2_ESTIMATOR_HPP
#define STEPPE_CORE_INTERNAL_F2_ESTIMATOR_HPP

#include "steppe/config.hpp"  // kHetCorrDenomFloor, kCdivBlock

#if defined(__CUDACC__)
#  define STEPPE_HD __host__ __device__
#else
#  define STEPPE_HD
#endif

namespace steppe::core {

// ===========================================================================
// Per-element f2 numerics — the shared primitive.
// ===========================================================================

/// Within-population heterozygosity bias correction at one (population, SNP):
/// `hc = q(1-q) / max(N-1, kHetCorrDenomFloor)`.
///
/// `q` is the reference-allele frequency in [0,1]; `n` is the NON-MISSING
/// HAPLOID count (2 × non-missing diploids, or 1 × non-missing pseudo-haploids
/// for ancient DNA — the Q/V/N contract, views.hpp). The floor (the `1` in
/// AT2's `max(N-1,1)`) prevents divide-by-zero with a single non-missing
/// haploid. This is the AT2 bias-correction convention, pinned to a golden.
///
/// Returns 0 when invalid (caller passes the validity bit, so an invalid entry
/// contributes nothing — matching the zero-fill that makes the masked GEMM
/// correct). Sample size is per-SNP `n`, never a hardcoded constant.
[[nodiscard]] STEPPE_HD inline double het_correction(double q, double n, bool valid) noexcept {
    if (!valid) return 0.0;
    const double denom = (n - 1.0 > kHetCorrDenomFloor) ? (n - 1.0) : kHetCorrDenomFloor;
    return q * (1.0 - q) / denom;
}

/// The unbiased per-SNP f2 summand for a population pair (i, j) at one SNP, in
/// the CANCELLATION-FREE form the reference oracle uses:
/// `(p_i - p_j)² - hc_i - hc_j`.
///
/// This forms the difference `(p_i - p_j)` and squares it directly (never the
/// expanded `p_i² - 2 p_i p_j + p_j²`), so the catastrophic cancellation is
/// avoided at the per-SNP level. `hc_i`, `hc_j` are het corrections from
/// `het_correction`. The caller sums these over SNPs jointly valid in both i and
/// j, then divides by the joint valid count to get f2(i,j). Used by the CPU
/// reference backend; the GPU path computes the same statistic via the 3-GEMM
/// reformulation + `assemble_f2_numerator`.
[[nodiscard]] STEPPE_HD inline double f2_term(double p_i, double p_j,
                                              double hc_i, double hc_j) noexcept {
    const double d = p_i - p_j;
    return d * d - hc_i - hc_j;
}

/// Assemble the f2 numerator for one pair from the 3-GEMM reduced statistics —
/// the EXPANDED form the GPU GEMMs build (architecture.md §5 S2 line 240; spike
/// assemble_f2_kernel / assemble): `R_diag(i,j) + R_diag(j,i) − 2·G(i,j)
/// − H(i,j) − H(j,i)`.
///
///   sumsq_i = R_diag(i,j) = Σ_s p_{i,s}² over SNPs valid in both i,j
///   sumsq_j = R_diag(j,i)
///   cross   = G(i,j)      = Σ_s p_{i,s} p_{j,s}
///   hsum_i  = H(i,j)      = Σ_s hc_{i,s} over jointly-valid SNPs
///   hsum_j  = H(j,i)
///
/// This is the difference-of-large-like-magnitude-sums where catastrophic
/// cancellation lands; it is held in NATIVE FP64 in every precision mode (the
/// `Precision` knob governs only the GEMMs that produce the inputs). The `2` is
/// a true mathematical constant (a²−2ab+b²), not a magic number (ROADMAP §4).
[[nodiscard]] STEPPE_HD inline double assemble_f2_numerator(double sumsq_i, double sumsq_j,
                                                            double cross,
                                                            double hsum_i, double hsum_j) noexcept {
    return sumsq_i + sumsq_j - 2.0 * cross - hsum_i - hsum_j;
}

/// Finalize f2(i,j) = numerator / pairwise-valid-SNP-count, with the AT2
/// `Vpair == 0 ⇒ 0` guard (spike assemble_f2_kernel:239 / assemble:37). `vpair`
/// is an exact integer count carried as a double; when it is 0 the numerator is
/// also 0, so the result is 0. (`vpair >= 1.0` is the equivalent integer test;
/// `> 0.0` is kept for exactness.) The per-block divide here and the S4
/// jackknife weighting by `vpair` must COMPOSE to AT2's f2_blocks definition,
/// not double-normalize (architecture.md §5 S2 caveat (a), §12).
[[nodiscard]] STEPPE_HD inline double finalize_f2(double numerator, double vpair) noexcept {
    return (vpair > 0.0) ? (numerator / vpair) : 0.0;
}

// ===========================================================================
// Launch-config helpers (architecture.md §7, §8 — one home for grid math).
//
// These mirror core/internal/launch_config.hpp's `cdiv`/`grid_for`; they live
// alongside the f2 primitive so the f2 feeder kernels never recompute grid math
// (replacing the spike's inline `(P + block.x - 1) / block.x`,
// f2_emu_spike.cu:312). Host-callable; constexpr so they fold at compile time.
// ===========================================================================

/// Ceiling division `ceil(n / b)` — the launch-grid building block. `b` must be
/// > 0 (a block dimension). Replaces the spike's open-coded `(n + b - 1) / b`.
[[nodiscard]] STEPPE_HD constexpr int cdiv(int n, int b) noexcept {
    return (n + b - 1) / b;
}

/// Long overload of `cdiv` for SNP-count-scale dimensions (M can exceed 2^31).
[[nodiscard]] STEPPE_HD constexpr long cdiv(long n, long b) noexcept {
    return (n + b - 1) / b;
}

/// Number of `block`-sized tiles needed to cover `n` elements along one axis —
/// the grid extent for a 1-D-per-axis launch. Alias of `cdiv` named for the
/// launch-config call site (`grid_for(P, kCdivBlock)` for each of x/y over the
/// [P × P] output). Defaults `block` to the project's standard square edge.
[[nodiscard]] STEPPE_HD constexpr int grid_for(int n, int block = kCdivBlock) noexcept {
    return cdiv(n, block);
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_F2_ESTIMATOR_HPP

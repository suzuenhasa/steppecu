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
// (compiled by nvcc). STEPPE_HD (from core/internal/host_device.hpp — the single
// home, shared with decode_af.hpp) expands to the CUDA qualifiers under nvcc and
// to nothing otherwise, so the exact SAME functions compile and unit-test on the
// CPU and run on the GPU — that is the whole point of one shared primitive.
#ifndef STEPPE_CORE_INTERNAL_F2_ESTIMATOR_HPP
#define STEPPE_CORE_INTERNAL_F2_ESTIMATOR_HPP

#include "core/internal/host_device.hpp"    // STEPPE_HD
#include "core/internal/launch_config.hpp"  // cdiv, grid_for (the launch-math home)
#include "steppe/config.hpp"                // kHetCorrDenomFloor

namespace steppe::core {

// ===========================================================================
// Stacked-S layout factor — the single source of the [2P × …] row-block count.
// ===========================================================================

/// Number of row-blocks in the stacked S matrix of the S2 3-GEMM reformulation:
/// S = [Qsq (rows 0..P-1) ; Hc (rows P..2P-1)], so S / R have `kF2StackedBlocks`
/// vertically-stacked [P × …] blocks and a leading dimension of 2P. Single-homed
/// here (the CUDA-FREE STEPPE_HD f2-estimator home, consumed by BOTH the single-
/// block feeder f2_block_kernel.cu and the grouped f2_blocks_kernel.cu) so the
/// `2P` factor is defined ONCE rather than open-coded as `2 * P` in each TU — if a
/// third stacked block were ever added the sites cannot drift (DRY;
/// NAMING-STYLE-STANDARD §2.5 single-source; group-5 5.3). This is a structural
/// constant of the [2P × M] / [2P × P] layout, not a tunable.
inline constexpr int kF2StackedBlocks = 2;

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
    // `n - 1.0` bound once (DRY; NAMING-STYLE-STANDARD §2.5; findings group-7 7.2).
    // Value-identical to the prior twice-computed form (the compiler already CSEs it).
    const double nm1 = n - 1.0;
    const double denom = (nm1 > kHetCorrDenomFloor) ? nm1 : kHetCorrDenomFloor;
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
///
/// NB (F1 / OQ-12): a `Vpair == 0` pair-block returns f2 = 0 here, NOT a NaN — so
/// the f2 VALUE alone cannot distinguish "no data" from "true zero f4". The
/// MISSING-block test is therefore on `Vpair` itself (`pair_block_is_missing`
/// below), not on the finalized f2; assemble_f4 reads Vpair to drop the block.
[[nodiscard]] STEPPE_HD inline double finalize_f2(double numerator, double vpair) noexcept {
    return (vpair > 0.0) ? (numerator / vpair) : 0.0;
}

/// THE single-source missing-block predicate (F1 / OQ-12; AT2 `read_f2(remove_na
/// =TRUE)` `keep = apply(f2,3,sum(!is.finite)==0)`). A jackknife block is MISSING
/// for a population pair (i, j) when no SNP in the block is jointly valid in both
/// pops — i.e. `Vpair(i,j,b) == 0` — and AT2 then treats that pair's per-block f2
/// as NA. AT2 DROPS any block in which ANY loaded pair is NA (it imputes nothing):
/// the f2 0-fill of `finalize_f2` would otherwise bias the f4 toward 0 and inflate
/// the jackknife variance (the OQ-12 defect). `vpair` is an exact integer count
/// carried as a double; `<= 0.0` catches the 0 count exactly (and any malformed
/// negative). Single-homed here so the CpuBackend oracle and the GPU keep-mask
/// kernel cannot diverge on the rule (architecture.md §8 single-source, §13).
[[nodiscard]] STEPPE_HD inline bool pair_block_is_missing(double vpair) noexcept {
    return !(vpair > 0.0);
}

// The launch-config helpers `cdiv`/`grid_for` now live in their own single home,
// core/internal/launch_config.hpp (architecture.md §4 line 466, §8 line 525),
// included above and re-exported through this header for the f2 feeder kernels
// and the host unit test that include it for both the numerics and the grid math.

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_F2_ESTIMATOR_HPP

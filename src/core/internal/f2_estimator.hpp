// src/core/internal/f2_estimator.hpp
//
// The single source of the per-element, bias-corrected f2 numerics — one set of
// scalar formulas shared so the CPU reference oracle and the GPU feeder cannot
// diverge. Marked STEPPE_HD so the exact same functions compile on the host and
// run on the device.
//
// Reference: docs/reference/src_core_internal_f2_estimator.hpp.md
#ifndef STEPPE_CORE_INTERNAL_F2_ESTIMATOR_HPP
#define STEPPE_CORE_INTERNAL_F2_ESTIMATOR_HPP

#include "core/internal/host_device.hpp"
#include "core/internal/launch_config.hpp"
#include "steppe/config.hpp"

namespace steppe::core {

// Stacked-S layout factor — reference §3
inline constexpr int kF2StackedBlocks = 2;

// Heterozygosity bias correction — reference §4
[[nodiscard]] STEPPE_HD inline double het_correction(double q, double n, bool valid) noexcept {
    if (!valid) return 0.0;
    const double nm1 = n - 1.0;
    const double denom = (nm1 > kHetCorrDenomFloor) ? nm1 : kHetCorrDenomFloor;
    return q * (1.0 - q) / denom;
}

// Per-SNP f2 summand, cancellation-free form — reference §5
[[nodiscard]] STEPPE_HD inline double f2_term(double p_i, double p_j,
                                              double hc_i, double hc_j) noexcept {
    const double d = p_i - p_j;
    return d * d - hc_i - hc_j;
}

// f2 numerator from the GPU sums, expanded form — reference §6
[[nodiscard]] STEPPE_HD inline double assemble_f2_numerator(double sumsq_i, double sumsq_j,
                                                            double cross,
                                                            double hsum_i, double hsum_j) noexcept {
    return sumsq_i + sumsq_j - 2.0 * cross - hsum_i - hsum_j;
}

// Finalize f2 for one pair — reference §7
[[nodiscard]] STEPPE_HD inline double finalize_f2(double numerator, double vpair) noexcept {
    return (vpair > 0.0) ? (numerator / vpair) : 0.0;
}

// Missing-block predicate — reference §8
[[nodiscard]] STEPPE_HD inline bool pair_block_is_missing(double vpair) noexcept {
    return !(vpair > 0.0);
}

// Re-exported launch-config helpers cdiv, grid_for — reference §9

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_F2_ESTIMATOR_HPP

// Reference: docs/reference/src_core_readv2_readv2_classify.hpp.md
// src/core/readv2/readv2_classify.hpp
//
// The pure-host READv2 numeric heart: the degree classifier and the z statistic that
// the concord gate measures. No device, no I/O — takes only per-pair scalars, so it is
// directly unit-testable (tests/unit/test_readv2_classify.cpp pins it at the cut points).
//
// The cut points are the canonical READ/READv2 normalized-P0 boundaries (expected
// normalized P0 of 0.5 / 0.75 / 1.0 for first / second / unrelated, with the
// twin/identical floor below first). steppe's frozen 4-enum has no "third degree"; the
// READv2 third band collapses into second/unrelated per the fixture recipe. They are a
// named constexpr Phase-1 tunable pinned to reproduce the oracle's degree column, not
// frozen contract.
#ifndef STEPPE_CORE_READV2_READV2_CLASSIFY_HPP
#define STEPPE_CORE_READV2_READV2_CLASSIFY_HPP

namespace steppe::core::readv2 {

// Normalized-P0 degree cut points.
inline constexpr double kCutIdentical = 0.625;    // < -> identical
inline constexpr double kCutFirst = 0.8125;        // < -> first
inline constexpr double kCutSecond = 0.90625;      // < -> second, else unrelated

// The four frozen lowercase degree tokens (exact schema spelling).
inline constexpr const char* kDegreeIdentical = "identical";
inline constexpr const char* kDegreeFirst = "first";
inline constexpr const char* kDegreeSecond = "second";
inline constexpr const char* kDegreeUnrelated = "unrelated";

// Classify a normalized P0 into one of the four frozen degree tokens.
[[nodiscard]] const char* degree_from_p0norm(double p0_norm) noexcept;

// The boundary the call was made against: the cut point separating the called degree
// from the next-more-distant class. "unrelated" has no more-distant class, so it uses
// the nearest boundary (kCutSecond). z is measured against this (Z_upper convention).
[[nodiscard]] double boundary_for_degree(const char* degree) noexcept;

// The z statistic: normalized distance from P0_norm to the classification boundary,
// z = (boundary - P0_norm) / se_P0_norm, where se_P0_norm is the window block-jackknife
// SE of P0_mean divided by `background` (the same native-FP64 scale). Returns NaN when
// n_windows < 2 (SE undefined) so the emitter writes NA. Inputs are the per-pair
// scalars the reduction already produced: p0_mean, p0_norm, background, n_windows, and
// sum_p0_sq (= sum over windows of P0[w]^2).
[[nodiscard]] double readv2_z(double p0_mean, double p0_norm, double background,
                              int n_windows, double sum_p0_sq) noexcept;

}  // namespace steppe::core::readv2

#endif  // STEPPE_CORE_READV2_READV2_CLASSIFY_HPP

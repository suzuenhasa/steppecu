// src/io/gl_normalize.hpp
//
// Pure host math: FORMAT/PL, /GL, /GP -> a numerically-stable NORMALIZED LINEAR
// genotype-likelihood triplet (FP64), in the field's VCF-native order
// (RR, RA, AA) = (0, 1, 2) ALT-copies. Header-only, standard library only, no VCF
// text and no device — so the normalization is unit-testable in isolation (the
// PL->linear hand-computation gate) with no reader or GPU dependency.
//
// Precision: this is the GENOTYPE-LIKELIHOOD compute path (it feeds PCAngsd /
// ancIBD, NOT the f2 cache), so the emulated-FP64-matmul parity policy does NOT
// bind here — native FP64 throughout (the reference methods want double).
//
// Portability: uses std::pow(10.0, x) rather than the POSIX exp10 extension so the
// build carries no dependency on a glibc-only symbol (critic fix #4).
#ifndef STEPPE_IO_GL_NORMALIZE_HPP
#define STEPPE_IO_GL_NORMALIZE_HPP

#include <algorithm>
#include <array>
#include <cmath>

namespace steppe::io::glnorm {

// The uninformative triplet a MISSING (field-absent / wrong-arity / unparseable /
// non-finite) genotype maps to. (1/3, 1/3, 1/3) is numerically safe for every GPU
// consumer (no NaN sentinels — those are barred); the parallel present_mask=0 lets
// a consumer distinguish "observed-uninformative" from "absent".
inline constexpr std::array<double, 3> kUninformative = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};

// PL (Phred-scaled genotype likelihoods, non-negative ints) -> normalized linear.
//   m = min PL ; p[j] = 10^(-(PL[j]-m)/10) ; L[j] = p[j] / sum
// Subtracting the min (the ML genotype, whose PL is ~0) before the exp keeps every
// exponent <= 0, so no overflow and the sum is >= 1 (the min term is exactly 1).
[[nodiscard]] inline std::array<double, 3> normalize_pl(long long p0, long long p1,
                                                        long long p2) {
    const long long m = std::min({p0, p1, p2});
    const double e0 = std::pow(10.0, -static_cast<double>(p0 - m) / 10.0);
    const double e1 = std::pow(10.0, -static_cast<double>(p1 - m) / 10.0);
    const double e2 = std::pow(10.0, -static_cast<double>(p2 - m) / 10.0);
    const double s = e0 + e1 + e2;
    return {e0 / s, e1 / s, e2 / s};
}

// GL (log10-scaled genotype likelihoods, may be negative) -> normalized linear.
//   M = max GL ; p[j] = 10^(GL[j]-M) ; L[j] = p[j] / sum
// Subtracting the max keeps every exponent <= 0 (the max term is exactly 1).
[[nodiscard]] inline std::array<double, 3> normalize_gl(double g0, double g1, double g2) {
    const double M = std::max({g0, g1, g2});
    const double e0 = std::pow(10.0, g0 - M);
    const double e1 = std::pow(10.0, g1 - M);
    const double e2 = std::pow(10.0, g2 - M);
    const double s = e0 + e1 + e2;
    return {e0 / s, e1 / s, e2 / s};
}

// GP (linear genotype POSTERIORS, summing ~1 e.g. from GLIMPSE/Beagle) -> renormalized.
//   L[j] = GP[j] / sum ; guard sum > 0. GP is a posterior — passed through (the
// artifact header records that the field was GP so a consumer does not conflate it
// with a likelihood), only renormalized so file rounding drift sums to exactly 1.
[[nodiscard]] inline std::array<double, 3> normalize_gp(double g0, double g1, double g2) {
    const double s = g0 + g1 + g2;
    if (!(s > 0.0)) return kUninformative;  // degenerate (all-zero / negative) -> uninformative
    return {g0 / s, g1 / s, g2 / s};
}

}  // namespace steppe::io::glnorm

#endif  // STEPPE_IO_GL_NORMALIZE_HPP

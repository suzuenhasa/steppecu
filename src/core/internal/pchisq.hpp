// src/core/internal/pchisq.hpp
//
// Host-pure, native-FP64 upper-tail chi-squared probability — the qpAdm rank-test
// p-value. CUDA-free, header-only, standard-library-only, so it compiles into both
// the device target and core/qpadm without dragging in the toolkit.
//
// Reference: docs/reference/src_core_internal_pchisq.hpp.md
#ifndef STEPPE_CORE_INTERNAL_PCHISQ_HPP
#define STEPPE_CORE_INTERNAL_PCHISQ_HPP

#include <cmath>

namespace steppe::core::internal {

// Named constants — reference §2
inline constexpr int    kPchisqMaxIter = 1000;
inline constexpr double kPchisqEps     = 1e-15;

inline constexpr double kPchisqFpMin   = 1e-300;

// The shared normalizing prefactor — reference §4
[[nodiscard]] inline double pchisq_gamma_prefactor(double a, double x) {
    return std::exp(-x + a * std::log(x) - std::lgamma(a));
}

// The two incomplete-gamma methods — reference §5
[[nodiscard]] inline double pchisq_gammp_series(double a, double x) {
    double ap = a;
    double sum = 1.0 / a;
    double del = sum;
    for (int n = 0; n < kPchisqMaxIter; ++n) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (std::fabs(del) < std::fabs(sum) * kPchisqEps) break;
    }
    return sum * pchisq_gamma_prefactor(a, x);
}

[[nodiscard]] inline double pchisq_gammq_cf(double a, double x) {
    double b = x + 1.0 - a;
    double c = 1.0 / kPchisqFpMin;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= kPchisqMaxIter; ++i) {
        const double di = static_cast<double>(i);
        const double an = -di * (di - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < kPchisqFpMin) d = kPchisqFpMin;
        c = b + an / c;
        if (std::fabs(c) < kPchisqFpMin) c = kPchisqFpMin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < kPchisqEps) break;
    }
    return pchisq_gamma_prefactor(a, x) * h;
}

// The entry point: pchisq_upper — reference §6
[[nodiscard]] inline double pchisq_upper(double x, int dof) {
    if (dof <= 0) return std::nan("");
    if (x <= 0.0) return 1.0;
    const double a = 0.5 * static_cast<double>(dof);
    const double xx = 0.5 * x;
    if (xx < a + 1.0) return 1.0 - pchisq_gammp_series(a, xx);
    return pchisq_gammq_cf(a, xx);
}

}  // namespace steppe::core::internal

#endif  // STEPPE_CORE_INTERNAL_PCHISQ_HPP

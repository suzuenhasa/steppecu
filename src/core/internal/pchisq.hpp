// src/core/internal/pchisq.hpp
//
// Host-pure, native-FP64 upper-tail chi-squared probability — the qpAdm rank-test
// p-value special function (the loose `p` tier, OQ-13). CUDA-FREE, header-only,
// standard-library-only — exactly like core/internal/small_linalg.hpp, so it
// compiles into the device target (where CpuBackend's rank_sweep lives, M(fit-2))
// AND into core/qpadm (the orchestrator + run_impl) without dragging in any
// toolkit, keeping the formula in ONE place (DRY: qpadm_fit.cpp delegates here).
//
// The math is the standard Numerical-Recipes regularized incomplete gamma: the
// upper tail P(X > x | dof) = Q(dof/2, x/2) via the series form for x < a+1 and
// the continued-fraction form otherwise. Double precision is ample for the loose
// p tier; the rank DECISION (p > alpha) is what gates, not the p bits.
#ifndef STEPPE_CORE_INTERNAL_PCHISQ_HPP
#define STEPPE_CORE_INTERNAL_PCHISQ_HPP

#include <cmath>

namespace steppe::core::internal {

/// Regularized lower incomplete gamma P(a, x) by series (good for x < a+1).
[[nodiscard]] inline double pchisq_gammp_series(double a, double x) {
    const int kMaxIter = 1000;
    const double kEps = 1e-15;
    double ap = a;
    double sum = 1.0 / a;
    double del = sum;
    for (int n = 0; n < kMaxIter; ++n) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (std::fabs(del) < std::fabs(sum) * kEps) break;
    }
    return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

/// Regularized upper incomplete gamma Q(a, x) by continued fraction (x >= a+1).
[[nodiscard]] inline double pchisq_gammq_cf(double a, double x) {
    const int kMaxIter = 1000;
    const double kEps = 1e-15;
    const double kFpMin = 1e-300;
    double b = x + 1.0 - a;
    double c = 1.0 / kFpMin;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= kMaxIter; ++i) {
        const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < kFpMin) d = kFpMin;
        c = b + an / c;
        if (std::fabs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < kEps) break;
    }
    return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

/// Upper-tail chi-squared probability P(X > x | dof) = Q(dof/2, x/2). dof <= 0 ⇒
/// NaN (the AT2 rankdrop "NA" row, e.g. the last nested diff); x <= 0 ⇒ 1.0.
[[nodiscard]] inline double pchisq_upper(double x, int dof) {
    if (dof <= 0) return std::nan("");
    if (x <= 0.0) return 1.0;
    const double a = 0.5 * static_cast<double>(dof);
    const double xx = 0.5 * x;
    if (xx < a + 1.0) return 1.0 - pchisq_gammp_series(a, xx);  // Q = 1 - P
    return pchisq_gammq_cf(a, xx);
}

}  // namespace steppe::core::internal

#endif  // STEPPE_CORE_INTERNAL_PCHISQ_HPP

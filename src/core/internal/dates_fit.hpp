// src/core/internal/dates_fit.hpp
//
// CUDA-FREE, header-only DATES host primitives shared by steppe_core (run_dates, dates.cpp)
// and steppe_device (the CpuBackend dates_repack / dates_fit reference oracle). These are the
// bit-exact host bodies the device backend's GPU kernels (dates_kernel.cu) are diffed against:
//   - dates_repack_host: the target-genotype repack onto the kept SNP axis (M5; the
//     dates.cpp:296-313 bit-shuffle), INTEGER/BIT-EXACT.
//   - dates_fit_one / fit_exp_decay: the single-exponential decay fit (M6; the long-double
//     2×2 normal-equation coarse-to-fine search), the loose-2%-tier DATES date oracle.
//
// Header-only + inline so BOTH libraries can call them with NO inter-library link dependency
// (steppe_device must NOT link steppe_core — the dependency is one-way, core→device via
// ComputeBackend). Both libs link steppe::core_internal (this header's include root), so the
// single definition is shared without a TU-to-TU symbol reference.
#ifndef STEPPE_CORE_INTERNAL_DATES_FIT_HPP
#define STEPPE_CORE_INTERNAL_DATES_FIT_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/internal/decode_af.hpp"  // core::genotype_code / kCodesPerByte / kBitsPerCode

namespace steppe::core::dates {

/// One DATES single-exponential + affine fit outcome (date in generations + residual sd + ok).
struct ExpFitHost {
    double date_gen = std::nan("");
    double error_sd = std::nan("");
    bool ok = false;
};

/// Solve the 2×2 (or 1×1 non-affine) normal equations for (co0, c) minimizing
/// Σ(y[i] - co0·v^i - c)²; return the residual sum of squares (NOT divided by n). The
/// accumulators are long double (the cancellation carve-out; the device FP64 mirror lives in
/// dates_kernel.cu). EXACT mirror of the prior dates.cpp anonymous-namespace linfit_2x2.
inline double linfit_2x2(const std::vector<double>& y, double v, bool affine, double& co0,
                         double& c) {
    const std::size_t n = y.size();
    long double Sbb = 0.0L, Sb1 = 0.0L, S11 = 0.0L, Sby = 0.0L, Sy = 0.0L;
    double bi = 1.0;  // v^0
    for (std::size_t i = 0; i < n; ++i) {
        const long double b = static_cast<long double>(bi);
        const long double yi = static_cast<long double>(y[i]);
        Sbb += b * b;
        Sby += b * yi;
        if (affine) { Sb1 += b; S11 += 1.0L; Sy += yi; }
        bi *= v;
    }
    if (affine) {
        const long double det = Sbb * S11 - Sb1 * Sb1;
        if (det == 0.0L) { co0 = std::nan(""); c = std::nan(""); return std::nan(""); }
        co0 = static_cast<double>((Sby * S11 - Sb1 * Sy) / det);
        c = static_cast<double>((Sbb * Sy - Sb1 * Sby) / det);
    } else {
        co0 = (Sbb != 0.0L) ? static_cast<double>(Sby / Sbb) : std::nan("");
        c = 0.0;
    }
    long double rss = 0.0L;
    double bb = 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double pred = co0 * bb + c;
        const long double r = static_cast<long double>(y[i]) - pred;
        rss += r * r;
        bb *= v;
    }
    return static_cast<double>(rss);
}

/// Fit A·exp(-λ·d)+c (affine adds c) over the windowed corr curve `y`: the DATES coarse-to-fine
/// 1-D search over the per-bin decay factor v∈(0,1) (4000-point grid requiring co0>0, with the
/// any-sign fallback, then a 200-iter ternary refine) + the inner (co0,c) 2×2 linear solve.
/// `step` is the bin width in MORGANS; date_gen = λ = -log(v)/step (generations). EXACT mirror
/// of the prior dates.cpp fit_exp_decay (single home now here).
inline ExpFitHost fit_exp_decay(const std::vector<double>& y, double step, bool affine) {
    ExpFitHost out;
    if (y.size() < 3 || step <= 0.0) return out;
    auto score = [&](double v, double& co0, double& c) -> double {
        if (!(v > 0.0) || !(v < 1.0)) return std::nan("");
        return linfit_2x2(y, v, affine, co0, c);
    };
    double best_v = 0.5, best_rss = std::numeric_limits<double>::infinity(), best_co0 = 0.0,
           best_c = 0.0;
    const int coarse = 4000;
    for (int k = 1; k < coarse; ++k) {
        const double v = static_cast<double>(k) / static_cast<double>(coarse);
        double co0 = 0.0, c = 0.0;
        const double rss = score(v, co0, c);
        if (std::isnan(rss)) continue;
        if (co0 <= 0.0) continue;
        if (rss < best_rss) { best_rss = rss; best_v = v; best_co0 = co0; best_c = c; }
    }
    if (!std::isfinite(best_rss)) {
        for (int k = 1; k < coarse; ++k) {
            const double v = static_cast<double>(k) / static_cast<double>(coarse);
            double co0 = 0.0, c = 0.0;
            const double rss = score(v, co0, c);
            if (std::isnan(rss)) continue;
            if (rss < best_rss) { best_rss = rss; best_v = v; best_co0 = co0; best_c = c; }
        }
        if (!std::isfinite(best_rss)) return out;
    }
    double lo = std::max(1e-9, best_v - 1.0 / static_cast<double>(coarse));
    double hi = std::min(1.0 - 1e-9, best_v + 1.0 / static_cast<double>(coarse));
    const int ternary_iters = 200;
    for (int it = 0; it < ternary_iters; ++it) {
        const double m1 = lo + (hi - lo) / 3.0;
        const double m2 = hi - (hi - lo) / 3.0;
        double c1a = 0.0, c1b = 0.0, c2a = 0.0, c2b = 0.0;
        const double r1 = score(m1, c1a, c1b);
        const double r2 = score(m2, c2a, c2b);
        const double rr1 = std::isnan(r1) ? std::numeric_limits<double>::infinity() : r1;
        const double rr2 = std::isnan(r2) ? std::numeric_limits<double>::infinity() : r2;
        if (rr1 < rr2) hi = m2; else lo = m1;
    }
    best_v = 0.5 * (lo + hi);
    best_rss = score(best_v, best_co0, best_c);
    if (std::isnan(best_rss) || !(best_v > 0.0) || !(best_v < 1.0)) return out;
    const double lambda = -std::log(best_v) / step;
    out.date_gen = lambda;
    out.error_sd = std::sqrt(best_rss / static_cast<double>(y.size()));
    out.ok = std::isfinite(out.date_gen) && out.date_gen > 0.0;
    return out;
}

/// M5 — target-genotype repack onto the kept SNP axis (the dates.cpp:296-313 bit-shuffle),
/// INTEGER/BIT-EXACT. Per target individual i, per kept SNP ks: read the 2-bit code at
/// original SNP index `kept_src[ks]` from `src` record i (MSB-first via core::genotype_code)
/// and OR it into dense bit position ks of `dst` record i. `dst` MUST be pre-zeroed.
inline void dates_repack_host(const std::uint8_t* src, std::size_t src_bpr,
                              const long* kept_src, long M_kept, int n_target,
                              std::size_t dst_bpr, std::uint8_t* dst) {
    for (int i = 0; i < n_target; ++i) {
        const std::uint8_t* src_rec = src + static_cast<std::size_t>(i) * src_bpr;
        std::uint8_t* dst_rec = dst + static_cast<std::size_t>(i) * dst_bpr;
        for (long ks = 0; ks < M_kept; ++ks) {
            const long s = kept_src[static_cast<std::size_t>(ks)];
            const std::size_t sb =
                static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
            const int sp = static_cast<int>(s % core::kCodesPerByte);
            const std::uint8_t code = core::genotype_code(src_rec[sb], sp);
            const std::size_t db =
                static_cast<std::size_t>(ks) / static_cast<std::size_t>(core::kCodesPerByte);
            const int dp = static_cast<int>(ks % core::kCodesPerByte);
            const int shift = (core::kCodesPerByte - 1 - dp) * core::kBitsPerCode;
            dst_rec[db] = static_cast<std::uint8_t>(dst_rec[db] | (code << shift));
        }
    }
}

}  // namespace steppe::core::dates

#endif  // STEPPE_CORE_INTERNAL_DATES_FIT_HPP

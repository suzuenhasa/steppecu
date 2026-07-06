// src/core/internal/dates_fit.hpp
//
// Header-only, CUDA-free host primitives for steppe's DATES admixture-date
// computation: the ExpFitHost result, the linfit_2x2 / fit_exp_decay decay fit,
// and the dates_repack_host genotype repack. Shared inline by both steppe_core
// (to run DATES) and steppe_device (as the CpuBackend reference oracle the GPU
// kernels are diffed against), with no cross-library link dependency.
//
// Reference: docs/reference/src_core_internal_dates_fit.hpp.md
#ifndef STEPPE_CORE_INTERNAL_DATES_FIT_HPP
#define STEPPE_CORE_INTERNAL_DATES_FIT_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/internal/decode_af.hpp"
#include "core/internal/index_cast.hpp"

namespace steppe::core::dates {

// DATES decay-fit result — reference §2
struct ExpFitHost {
    double date_gen = std::nan("");
    double error_sd = std::nan("");
    bool ok = false;
};

// Inner linear solve for a fixed decay factor — reference §3
inline double linfit_2x2(const std::vector<double>& y, double v, bool affine, double& co0,
                         double& c) {
    const std::size_t n = y.size();
    long double Sbb = 0.0L, Sb1 = 0.0L, S11 = 0.0L, Sby = 0.0L, Sy = 0.0L;
    double bi = 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        const long double b = ld(bi);
        const long double yi = ld(y[i]);
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
        const long double r = ld(y[i]) - pred;
        rss += r * r;
        bb *= v;
    }
    return static_cast<double>(rss);
}

// Decay-factor search and date fit — reference §4
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
    auto coarse_grid_scan = [&](bool require_positive) {
        for (int k = 1; k < coarse; ++k) {
            const double v = static_cast<double>(k) / static_cast<double>(coarse);
            double co0 = 0.0, c = 0.0;
            const double rss = score(v, co0, c);
            if (std::isnan(rss)) continue;
            if (require_positive && co0 <= 0.0) continue;
            if (rss < best_rss) { best_rss = rss; best_v = v; best_co0 = co0; best_c = c; }
        }
    };
    coarse_grid_scan(/*require_positive=*/true);
    if (!std::isfinite(best_rss)) {
        coarse_grid_scan(/*require_positive=*/false);
        if (!std::isfinite(best_rss)) return out;
    }
    double lo = std::max(1e-9, best_v - 1.0 / static_cast<double>(coarse));
    double hi = std::min(1.0 - 1e-9, best_v + 1.0 / static_cast<double>(coarse));
    const int ternary_iters = 200;
    for (int it = 0; it < ternary_iters; ++it) {
        const double m1 = lo + (hi - lo) / 3.0;
        const double m2 = hi - (hi - lo) / 3.0;
        double co0_m1 = 0.0, c_m1 = 0.0, co0_m2 = 0.0, c_m2 = 0.0;
        const double r1 = score(m1, co0_m1, c_m1);
        const double r2 = score(m2, co0_m2, c_m2);
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

// Target-genotype repack onto the kept SNP axis — reference §5
inline void dates_repack_host(const std::uint8_t* src, std::size_t src_bpr,
                              const long* kept_src, long M_kept, int n_target,
                              std::size_t dst_bpr, std::uint8_t* dst) {
    for (int i = 0; i < n_target; ++i) {
        const std::uint8_t* src_rec = src + idx(i) * src_bpr;
        std::uint8_t* dst_rec = dst + idx(i) * dst_bpr;
        for (long ks = 0; ks < M_kept; ++ks) {
            const long s = kept_src[idx(ks)];
            const std::size_t sb =
                idx(s) / idx(core::kCodesPerByte);
            const int sp = static_cast<int>(s % core::kCodesPerByte);
            const std::uint8_t code = core::genotype_code(src_rec[sb], sp);
            const std::size_t db =
                idx(ks) / idx(core::kCodesPerByte);
            const int dp = static_cast<int>(ks % core::kCodesPerByte);
            const int shift = (core::kCodesPerByte - 1 - dp) * core::kBitsPerCode;
            dst_rec[db] = static_cast<std::uint8_t>(dst_rec[db] | (code << shift));
        }
    }
}

}  // namespace steppe::core::dates

#endif  // STEPPE_CORE_INTERNAL_DATES_FIT_HPP

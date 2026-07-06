// src/core/internal/qpfstats_jackknife.hpp
//
// The two qpfstats block-jackknife reference estimates (host long double): the
// one definition shared by the CpuBackend oracle and the on-device GPU kernel it
// is diffed against. Header-only, no CUDA.
//
// Reference: docs/reference/src_core_internal_qpfstats_jackknife.hpp.md
#ifndef STEPPE_CORE_INTERNAL_QPFSTATS_JACKKNIFE_HPP
#define STEPPE_CORE_INTERNAL_QPFSTATS_JACKKNIFE_HPP

#include <cmath>
#include <cstddef>
#include <vector>

#include "core/internal/host_device.hpp"
#include "core/internal/index_cast.hpp"

namespace steppe::core {

// Global per-combination estimate (matrix_jackknife_est_full) — reference §4
[[nodiscard]] inline double matrix_jackknife_est_col(const double* numer, const double* cnt,
                                                     int c, int n_block) {
    const std::size_t base = idx(c) * idx(n_block);
    long double sum_n_all = 0.0L;
    for (int b = 0; b < n_block; ++b) sum_n_all += static_cast<long double>(cnt[base + b]);
    if (sum_n_all <= 0.0L) return std::nan("");

    long double tot_num = 0.0L, tot_w = 0.0L;
    for (int b = 0; b < n_block; ++b) {
        const double nu = numer[base + b];
        const double cn = cnt[base + b];
        if (!std::isfinite(nu) || cn <= 0.0) continue;
        tot_num += static_cast<long double>(nu) * static_cast<long double>(cn);
        tot_w += static_cast<long double>(cn);
    }
    if (tot_w <= 0.0L) return std::nan("");
    const long double tot = tot_num / tot_w;

    long double sum_loo = 0.0L, sum_omrb = 0.0L, sum_loo_omrb = 0.0L;
    long double sum_loo_cnt = 0.0L, sum_cnt_finite = 0.0L;
    int n_finite = 0;
    for (int b = 0; b < n_block; ++b) {
        const double nu = numer[base + b];
        const double cn = cnt[base + b];
        if (!std::isfinite(nu) || cn <= 0.0) continue;
        const long double rel = static_cast<long double>(cn) / sum_n_all;
        if (rel >= 1.0L) continue;
        const long double loo = (tot - static_cast<long double>(nu) * rel) / (1.0L - rel);
        if (!std::isfinite(static_cast<double>(loo))) continue;
        const long double omrb = 1.0L - rel;
        sum_loo += loo;
        sum_omrb += omrb;
        sum_loo_omrb += loo * omrb;
        sum_loo_cnt += loo * static_cast<long double>(cn);
        sum_cnt_finite += static_cast<long double>(cn);
        ++n_finite;
    }
    if (n_finite == 0 || sum_omrb <= 0.0L || sum_cnt_finite <= 0.0L) return std::nan("");
    const long double tot2 = sum_loo_omrb / sum_omrb;
    const long double weighted_loo_mean = sum_loo_cnt / sum_cnt_finite;
    return static_cast<double>(static_cast<long double>(n_finite) * tot2 - sum_loo +
                               weighted_loo_mean);
}

// Per-pair recentering estimate (f2(array)$est) — reference §5
[[nodiscard]] inline double f2blocks_pair_est(const std::vector<double>& arr,
                                              const std::vector<int>& bl) {
    STEPPE_ASSERT(bl.size() == arr.size(), "f2blocks_pair_est: arr/bl length mismatch");
    const int nb = static_cast<int>(arr.size());
    long double sum_bl = 0.0L;
    for (int b = 0; b < nb; ++b)
        if (std::isfinite(arr[idx(b)]))
            sum_bl += static_cast<long double>(bl[idx(b)]);
    if (sum_bl <= 0.0L) return 0.0;

    long double tot_num = 0.0L;
    for (int b = 0; b < nb; ++b) {
        const double a = arr[idx(b)];
        if (!std::isfinite(a)) continue;
        tot_num += static_cast<long double>(a) *
                   static_cast<long double>(bl[idx(b)]);
    }
    const long double tot = tot_num / sum_bl;

    long double num = 0.0L, den = 0.0L;
    for (int b = 0; b < nb; ++b) {
        const double a = arr[idx(b)];
        if (!std::isfinite(a)) continue;
        const long double blb = static_cast<long double>(bl[idx(b)]);
        const long double rel = blb / sum_bl;
        if (rel >= 1.0L) continue;
        const long double loo = (tot - static_cast<long double>(a) * rel) / (1.0L - rel);
        const long double h = sum_bl / blb;
        const long double w = 1.0L - 1.0L / h;
        num += loo * w;
        den += w;
    }
    if (den <= 0.0L) return 0.0;
    return static_cast<double>(num / den);
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_QPFSTATS_JACKKNIFE_HPP

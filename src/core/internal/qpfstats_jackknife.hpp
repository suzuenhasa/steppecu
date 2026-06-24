// src/core/internal/qpfstats_jackknife.hpp
//
// The qpfstats per-comb / per-pair BLOCK-JACKKNIFE REFERENCE primitives (long double).
// SINGLE-SOURCE the host long-double jackknife the CpuBackend oracle runs AND the math the
// CUDA on-device jackknife kernel (qpfstats_jackknife_kernel.cu) is diffed against. These were
// the file-local helpers in src/core/stats/qpfstats.cpp; they are lifted here so the CpuBackend
// fused oracle (qpfstats_blocks_smooth) and the qpfstats driver share ONE definition (no copy).
//
// matrix_jackknife_est_col — AT2 matrix_jackknife_est_full per popcomb column (R/io.R): the
//   GLOBAL per-comb estimate `y` (the bglob target). NaN-masked, cnt-weighted leave-one-out.
// f2blocks_pair_est        — AT2 f2(array)$est for one pair series (the recentering target):
//   est_to_loo (block_lengths weights) then jack_vec_stats2 = weighted.mean(loo, 1-1/h).
//
// LONG DOUBLE accumulation (the §12 cancellation carve-out — a leave-one-out jackknife
// DIFFERENCE, not a matmul). The GPU kernel reproduces this in native FP64 with the EXACT
// ascending-b operand order (the only delta is the carry precision), gated on the 9-pop golden.
//
// LAYERING: pure host C++20, header-only, NO CUDA. Lives in core/internal (shared by the
// CUDA-free driver and the CPU reference backend).
#ifndef STEPPE_CORE_INTERNAL_QPFSTATS_JACKKNIFE_HPP
#define STEPPE_CORE_INTERNAL_QPFSTATS_JACKKNIFE_HPP

#include <cmath>
#include <cstddef>
#include <vector>

namespace steppe::core {

/// matrix_jackknife_est_full per popcomb column (AT2 R/io.R; the GLOBAL per-comb estimate
/// `y` used for bglob). For column-c per-block means `numer[c,b]` (= numsum/cnt, NaN where
/// invalid) + counts `cnt[c,b]`:
///   valid = is.finite(numer); rel_bl = cnt/Σcnt;
///   tot = Σ(numer·cnt over valid)/Σ(cnt over valid);
///   loo = (tot - numer·rel_bl)/(1-rel_bl), masked to NA where !valid OR !finite;
///   tot2 = Σ(loo·(1-rel_bl))/Σ(1-rel_bl) over finite loo;
///   weighted_loo_mean = Σ(loo·cnt)/Σ(cnt) over finite loo;
///   y = n_finite·tot2 - Σloo + weighted_loo_mean.
/// `numer`/`cnt` are ROW-MAJOR [npopcomb × n_block] (numer[c*nb+b], the dstat output shape).
[[nodiscard]] inline double matrix_jackknife_est_col(const double* numer, const double* cnt,
                                                     int c, int n_block) {
    const std::size_t base = static_cast<std::size_t>(c) * static_cast<std::size_t>(n_block);
    long double sum_n_all = 0.0L;  // Σ cnt (all blocks)
    for (int b = 0; b < n_block; ++b) sum_n_all += static_cast<long double>(cnt[base + b]);
    if (sum_n_all <= 0.0L) return std::nan("");

    // tot = Σ(numer·cnt over valid) / Σ(cnt over valid).
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
        const long double rel = static_cast<long double>(cn) / sum_n_all;  // rel_bl
        if (rel >= 1.0L) continue;  // 1-rel==0 ⇒ loo not finite
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

/// f2(f2blocks)$est for one pair series (the recentering target). AT2 f2(array): est_to_loo
/// (block_lengths weights) then jack_vec_stats2 = weighted.mean(loo, 1-1/h), h=n/bl. For the
/// per-block estimate vector `arr[b]` (length n_block) + block_lengths `bl[b]`:
///   tot = weighted.mean(arr, bl); rel_bl = bl/Σbl; loo = (tot - arr·rel_bl)/(1-rel_bl);
///   h = Σbl/bl; est = weighted.mean(loo, 1 - 1/h).
/// NaN-safe (na.rm=TRUE: skip non-finite blocks throughout). All-zero/empty → 0.
[[nodiscard]] inline double f2blocks_pair_est(const std::vector<double>& arr,
                                              const std::vector<int>& bl) {
    const int nb = static_cast<int>(arr.size());
    long double sum_bl = 0.0L;
    for (int b = 0; b < nb; ++b)
        if (std::isfinite(arr[static_cast<std::size_t>(b)]))
            sum_bl += static_cast<long double>(bl[static_cast<std::size_t>(b)]);
    if (sum_bl <= 0.0L) return 0.0;

    long double tot_num = 0.0L;
    for (int b = 0; b < nb; ++b) {
        const double a = arr[static_cast<std::size_t>(b)];
        if (!std::isfinite(a)) continue;
        tot_num += static_cast<long double>(a) *
                   static_cast<long double>(bl[static_cast<std::size_t>(b)]);
    }
    const long double tot = tot_num / sum_bl;  // weighted.mean(arr, bl)

    // est = weighted.mean(loo, 1 - 1/h), h = Σbl/bl, loo = (tot - arr·rel_bl)/(1-rel_bl).
    long double num = 0.0L, den = 0.0L;
    for (int b = 0; b < nb; ++b) {
        const double a = arr[static_cast<std::size_t>(b)];
        if (!std::isfinite(a)) continue;
        const long double blb = static_cast<long double>(bl[static_cast<std::size_t>(b)]);
        const long double rel = blb / sum_bl;       // rel_bl
        if (rel >= 1.0L) continue;
        const long double loo = (tot - static_cast<long double>(a) * rel) / (1.0L - rel);
        const long double h = sum_bl / blb;          // h
        const long double w = 1.0L - 1.0L / h;       // 1 - 1/h
        num += loo * w;
        den += w;
    }
    if (den <= 0.0L) return 0.0;
    return static_cast<double>(num / den);
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_QPFSTATS_JACKKNIFE_HPP

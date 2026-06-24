// src/device/cuda/qpfstats_jackknife_kernel.cu
//
// The qpfstats ON-DEVICE block-JACKKNIFE kernels (the PERF path; include/steppe/qpfstats.hpp).
// These MOVE the former host long-double per-comb / per-pair jackknife loops (the GPU-idle "0"
// half of the measured 100/0 alternation: ~305k combs × ~711 blocks ≈ 217M single-core
// long-double iterations) ONTO the GPU, embarrassingly parallel over the combs / pairs and
// reducing over the n_block axis from the RESIDENT numsum/cnt (no ~1.7GB-each D2H).
//
// PRECISION (the §12 cancellation carve-out): NATIVE FP64 with the EXACT ascending-b operand
// order of the host long-double reference (core/internal/qpfstats_jackknife.hpp) — the ONLY
// delta is the carry precision (long double 64-bit mantissa → FP64 53-bit). This is the SAME
// leave-one-out block-jackknife family already proven on-device in native FP64 by the qpAdm fit
// (qpadm_fit_kernels.cu f4_loo_total_row, which holds the qpAdm golden). NEVER EmulatedFp64 here
// (a cancellation-sensitive jackknife difference, not a matmul). Gated on the 9-pop genotype
// golden at rtol 1e-6 + CpuBackend==CudaBackend.
//
// SHAPE: ONE thread per comb (numer/ymat/y) or per pair (recenter), grid-stride over the comb /
// pair axis (each owns a SHORT ~711-block reduction in registers). This matches the proven
// one-thread-per-row f4_loo_total_kernel idiom (nr-row jackknife); a per-block cub::BlockReduce
// is unnecessary at this segment length and would only add launch/shared-mem overhead. The
// combs/pairs are contiguous in numsum/cnt (row-major), so a thread reads a contiguous run.
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). The kernel bodies + <<<>>>
// live ONLY here; the backend reaches them through the narrow launch wrapper
// (qpfstats_jackknife_kernel.cuh), never includes this body (architecture.md §7).
#include <cuda_runtime.h>

#include "core/internal/launch_config.hpp"  // core::kMaxGridX (the single-source grid-dim cap)
#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK_KERNEL
#include "device/cuda/qpfstats_jackknife_kernel.cuh"  // the narrow seam

namespace steppe::device {

namespace {

/// One thread per comb c: numer = numsum/cnt (row-major), ymat (col-major), and the GLOBAL
/// block-jackknife est y[c]. Transliterates core::matrix_jackknife_est_col EXACTLY (same masks,
/// same rel>=1 skip, same loo isfinite guard, same ascending-b accumulation, same final
/// n_finite*tot2 - Σloo + weighted_loo_mean), in native FP64. numsum/cnt are ROW-MAJOR
/// [npopcomb × n_block] (cell (c,b) at c*n_block+b).
__global__ void qpfstats_numer_jackknife_kernel(const double* __restrict__ numsum,
                                                const double* __restrict__ cnt, int npopcomb,
                                                int n_block, double* __restrict__ numer,
                                                double* __restrict__ ymat,
                                                double* __restrict__ y) {
    const double knan = nan("");
    for (long c = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         c < static_cast<long>(npopcomb);
         c += static_cast<long>(gridDim.x) * blockDim.x) {
        const long base = c * static_cast<long>(n_block);

        // numer = numsum/cnt (NaN where cnt<=0); ymat is the same value, COL-MAJOR.
        for (int b = 0; b < n_block; ++b) {
            const double cn = cnt[base + b];
            const double mean = (cn > 0.0) ? (numsum[base + b] / cn) : knan;
            numer[base + b] = mean;
            ymat[c + static_cast<long>(npopcomb) * b] = mean;
        }

        // sum_n_all = Σ cnt (all blocks).
        double sum_n_all = 0.0;
        for (int b = 0; b < n_block; ++b) sum_n_all += cnt[base + b];
        if (!(sum_n_all > 0.0)) { y[c] = knan; continue; }

        // tot = Σ(numer·cnt over valid) / Σ(cnt over valid).
        double tot_num = 0.0, tot_w = 0.0;
        for (int b = 0; b < n_block; ++b) {
            const double nu = numer[base + b];
            const double cn = cnt[base + b];
            if (!isfinite(nu) || cn <= 0.0) continue;
            tot_num += nu * cn;
            tot_w += cn;
        }
        if (!(tot_w > 0.0)) { y[c] = knan; continue; }
        const double tot = tot_num / tot_w;

        double sum_loo = 0.0, sum_omrb = 0.0, sum_loo_omrb = 0.0;
        double sum_loo_cnt = 0.0, sum_cnt_finite = 0.0;
        int n_finite = 0;
        for (int b = 0; b < n_block; ++b) {
            const double nu = numer[base + b];
            const double cn = cnt[base + b];
            if (!isfinite(nu) || cn <= 0.0) continue;
            const double rel = cn / sum_n_all;  // rel_bl
            if (rel >= 1.0) continue;           // 1-rel==0 ⇒ loo not finite
            const double loo = (tot - nu * rel) / (1.0 - rel);
            if (!isfinite(loo)) continue;
            const double omrb = 1.0 - rel;
            sum_loo += loo;
            sum_omrb += omrb;
            sum_loo_omrb += loo * omrb;
            sum_loo_cnt += loo * cn;
            sum_cnt_finite += cn;
            ++n_finite;
        }
        if (n_finite == 0 || !(sum_omrb > 0.0) || !(sum_cnt_finite > 0.0)) {
            y[c] = knan;
            continue;
        }
        const double tot2 = sum_loo_omrb / sum_omrb;
        const double weighted_loo_mean = sum_loo_cnt / sum_cnt_finite;
        y[c] = static_cast<double>(n_finite) * tot2 - sum_loo + weighted_loo_mean;
    }
}

/// One thread per pair p: the recenter shift = bglob[p] - f2blocks_pair_est(b[p,:], block_sizes).
/// Transliterates core::f2blocks_pair_est EXACTLY (the AT2 f2(array)$est) in native FP64. b is
/// COL-MAJOR [npairs × n_block] (b[p + npairs*blk]); block_sizes is the per-block SNP count.
__global__ void qpfstats_recenter_shift_kernel(const double* __restrict__ b,
                                               const double* __restrict__ bglob,
                                               const int* __restrict__ block_sizes, int npairs,
                                               int n_block, double* __restrict__ shift) {
    for (long p = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < static_cast<long>(npairs);
         p += static_cast<long>(gridDim.x) * blockDim.x) {
        const auto arr = [&](int blk) -> double {
            return b[p + static_cast<long>(npairs) * blk];
        };

        // sum_bl = Σ bl over finite-arr blocks.
        double sum_bl = 0.0;
        for (int blk = 0; blk < n_block; ++blk)
            if (isfinite(arr(blk))) sum_bl += static_cast<double>(block_sizes[blk]);
        if (!(sum_bl > 0.0)) { shift[p] = bglob[p] - 0.0; continue; }

        // tot = weighted.mean(arr, bl).
        double tot_num = 0.0;
        for (int blk = 0; blk < n_block; ++blk) {
            const double a = arr(blk);
            if (!isfinite(a)) continue;
            tot_num += a * static_cast<double>(block_sizes[blk]);
        }
        const double tot = tot_num / sum_bl;

        // est = weighted.mean(loo, 1 - 1/h), h = Σbl/bl, loo = (tot - arr·rel)/(1-rel).
        double num = 0.0, den = 0.0;
        for (int blk = 0; blk < n_block; ++blk) {
            const double a = arr(blk);
            if (!isfinite(a)) continue;
            const double blb = static_cast<double>(block_sizes[blk]);
            const double rel = blb / sum_bl;  // rel_bl
            if (rel >= 1.0) continue;
            const double loo = (tot - a * rel) / (1.0 - rel);
            const double h = sum_bl / blb;    // h
            const double w = 1.0 - 1.0 / h;   // 1 - 1/h
            num += loo * w;
            den += w;
        }
        const double est = (den > 0.0) ? (num / den) : 0.0;
        shift[p] = bglob[p] - est;
    }
}

}  // namespace

void launch_qpfstats_numer_jackknife(const double* d_numsum, const double* d_cnt,
                                     int npopcomb, int n_block, double* d_numer,
                                     double* d_ymat, double* d_y, cudaStream_t stream) {
    if (npopcomb <= 0 || n_block <= 0) return;
    constexpr int kThreads = 128;
    long blocks = (static_cast<long>(npopcomb) + kThreads - 1) / kThreads;
    if (blocks > static_cast<long>(core::kMaxGridX)) blocks = core::kMaxGridX;
    qpfstats_numer_jackknife_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        d_numsum, d_cnt, npopcomb, n_block, d_numer, d_ymat, d_y);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpfstats_recenter_shift(const double* d_b, const double* d_bglob,
                                    const int* d_block_sizes, int npairs, int n_block,
                                    double* d_shift, cudaStream_t stream) {
    if (npairs <= 0 || n_block <= 0) return;
    constexpr int kThreads = 128;
    long blocks = (static_cast<long>(npairs) + kThreads - 1) / kThreads;
    if (blocks < 1) blocks = 1;
    if (blocks > static_cast<long>(core::kMaxGridX)) blocks = core::kMaxGridX;
    qpfstats_recenter_shift_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        d_b, d_bglob, d_block_sizes, npairs, n_block, d_shift);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

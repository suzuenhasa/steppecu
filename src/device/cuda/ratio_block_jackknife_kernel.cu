// src/device/cuda/ratio_block_jackknife_kernel.cu
//
// The SHARED on-device RATIO-block-jackknife kernel (the f4ratio M1 + qpDstat M2 common
// engine; backend.hpp ratio_block_jackknife). MOVES the former host long-double per-item
// jackknife loops (f4ratio.cpp ratio_jackknife driven per tuple; dstat.cpp dstat_jackknife
// driven per quadruple) ONTO the GPU, embarrassingly parallel over the N items and reducing
// over the n_block axis from the RESIDENT per-(item,block) num/den/weight (f4ratio's dLoo +
// dX; dstat's dNum/dDen/dCnt) — dropping the [m·nb] x_blocks/x_loo D2H (f4ratio) and the
// numsum/densum/cnt D2H (dstat).
//
// PRECISION (the §12 cancellation carve-out): NATIVE FP64 with the EXACT ascending-b operand
// order of the two host long-double references — the ONLY delta is the carry precision (long
// double 64-bit mantissa → FP64 53-bit). This is the SAME leave-one-out block-jackknife family
// already proven on-device in native FP64 (qpfstats_numer_jackknife_kernel, qpadm_fit
// f4_loo_total_row). NEVER EmulatedFp64 here. Gated on the f4ratio golden + the qpDstat
// Part-A/Part-B goldens at rtol 1e-6 + CpuBackend==CudaBackend.
//
// SHAPE: ONE thread per item, grid-stride over the item axis (each owns a SHORT ~n_block
// reduction in registers). Mirrors qpfstats_numer_jackknife_kernel verbatim. The two host
// jackknives are the SAME ratio block-jackknife differing on THREE parameterized axes — WEIGHT
// (caller-supplied), SURVIVOR MASK (setmiss_thresh), TOT_MODE — applied as in-thread branches,
// NOT two kernels.
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). The kernel body + <<<>>>
// live ONLY here; the backend reaches it through the narrow launch wrapper
// (ratio_block_jackknife_kernel.cuh), never includes this body (architecture.md §7).
#include <cuda_runtime.h>

#include "core/internal/launch_config.hpp"  // core::kMaxGridX (the single-source grid-dim cap)
#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK_KERNEL
#include "device/cuda/ratio_block_jackknife_kernel.cuh"  // the narrow seam + DRatioJackArray

namespace steppe::device {

namespace {

/// Index element (item k, block b) of a descriptor: data[base + k*item_stride + b*block_stride].
/// A 0 item_stride broadcasts across items (the f4ratio per-block block_sizes weight).
__device__ __forceinline__ double rj_at(const DRatioJackArray& a, long k, int b) {
    return a.data[a.base + k * a.item_stride + static_cast<long>(b) * a.block_stride];
}

/// One thread per item k. tot_mode 0 = f4ratio (block-sum tot; num/den ARE the est_to_loo
/// replicates; reads xblk_num/xblk_den; survivor mask |xblk_den|<setmiss_thresh). tot_mode
/// 1 = dstat (LOO-ratio tot; num/den ARE the per-block sums with weight=cnt; the LOO replicate
/// is built in-thread; survivor mask weight>0). Native FP64, ascending-b operand order matching
/// the host long-double references.
__global__ void ratio_block_jackknife_kernel(DRatioJackArray num, DRatioJackArray den,
                                             DRatioJackArray weight, DRatioJackArray xblk_num,
                                             DRatioJackArray xblk_den, int N, int n_block,
                                             int tot_mode, double setmiss_thresh, bool compute_p,
                                             double* __restrict__ d_est, double* __restrict__ d_se,
                                             double* __restrict__ d_z, double* __restrict__ d_p) {
    const double knan = nan("");
    const double kInvSqrt2 = 0.7071067811865475244;  // 1/sqrt(2) (the AT2 ztop / f4_two_sided_p).
    for (long k = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         k < static_cast<long>(N);
         k += static_cast<long>(gridDim.x) * blockDim.x) {

        // Per-block survivor predicate (the SAME mask reused in every pass, ascending b).
        // f4ratio (tot_mode 0): |xblk_den(k,b)| >= setmiss_thresh. dstat (tot_mode 1): cnt>0.
        // Captured by recomputing inline per b (matching the host's per-pass mask).
        double est = knan, se = knan, z = knan;

        if (tot_mode == 0) {
            // ===== f4ratio: ratio_jackknife (f4ratio.cpp:79-155) =====
            // Pass 1 — survivor: Σ block_sizes (=n), nb_surv, totnum/totden (Σ xblk·bl).
            double n_w = 0.0;       // Σ survivor weight (== AT2 n)
            int nb_surv = 0;
            double totnum = 0.0, totden = 0.0;
            for (int b = 0; b < n_block; ++b) {
                const double den_blk = rj_at(xblk_den, k, b);
                if (fabs(den_blk) < setmiss_thresh) continue;  // AT2 setmiss: missing block
                const double bl = rj_at(weight, k, b);
                n_w += bl;
                ++nb_surv;
                totnum += rj_at(xblk_num, k, b) * bl;
                totden += den_blk * bl;
            }
            if (nb_surv <= 1 || !(n_w > 0.0) || totden == 0.0) {
                d_est[k] = knan; d_se[k] = knan; d_z[k] = knan;
                if (compute_p) d_p[k] = knan;
                continue;
            }
            const double nb = static_cast<double>(nb_surv);
            const double tot = totnum / totden;  // the variance-centering term.

            // Pass 2 — est = mean(tot − R_b)·nb + weighted_mean(R_b, bl).
            double diffsum = 0.0, wmean_num = 0.0;
            for (int b = 0; b < n_block; ++b) {
                const double den_blk = rj_at(xblk_den, k, b);
                if (fabs(den_blk) < setmiss_thresh) continue;
                const double bl = rj_at(weight, k, b);
                const double Rb = rj_at(num, k, b) / rj_at(den, k, b);
                diffsum += tot - Rb;
                wmean_num += Rb * bl;
            }
            const double term1 = (diffsum / nb) * nb;  // (Σ(tot−R)/nb_surv)·nb_surv
            const double term2 = wmean_num / n_w;
            est = term1 + term2;

            // Pass 3 — xtau variance: h=n/bl; tau=h·tot−(h−1)·R; xtau=(tau−est)/sqrt(h−1).
            double var_acc = 0.0;
            for (int b = 0; b < n_block; ++b) {
                const double den_blk = rj_at(xblk_den, k, b);
                if (fabs(den_blk) < setmiss_thresh) continue;
                const double bl = rj_at(weight, k, b);
                const double Rb = rj_at(num, k, b) / rj_at(den, k, b);
                const double h = n_w / bl;
                const double tau = h * tot - (h - 1.0) * Rb;
                const double xtau = (tau - est) / sqrt(h - 1.0);
                var_acc += xtau * xtau;
            }
            const double var = var_acc / nb;
            se = (var > 0.0) ? sqrt(var) : knan;
            z = est / se;

        } else {
            // ===== dstat: dstat_jackknife (dstat.cpp:70-157) =====
            // Pass 1 — survivor: Σcnt, nb_surv.
            double sum_cnt = 0.0;
            int nb_surv = 0;
            for (int b = 0; b < n_block; ++b) {
                const double cn = rj_at(weight, k, b);
                if (cn > 0.0) { sum_cnt += cn; ++nb_surv; }
            }
            if (nb_surv <= 1 || !(sum_cnt > 0.0)) {
                d_est[k] = knan; d_se[k] = knan; d_z[k] = knan;
                if (compute_p) d_p[k] = knan;
                continue;
            }
            const double nb = static_cast<double>(nb_surv);

            // Pass 2a — tot_num/tot_den = weighted.mean(est_t, cnt).
            double tot_num_w = 0.0, tot_den_w = 0.0;
            for (int b = 0; b < n_block; ++b) {
                const double cn = rj_at(weight, k, b);
                if (cn <= 0.0) continue;
                const double est_num = rj_at(num, k, b) / cn;
                const double est_den = rj_at(den, k, b) / cn;
                tot_num_w += est_num * cn;
                tot_den_w += est_den * cn;
            }
            const double tot_num = tot_num_w / sum_cnt;
            const double tot_den = tot_den_w / sum_cnt;

            // Pass 2b — R_b = loo_num/loo_den; tot = weighted.mean(R_b, 1−rel); Σ R_b·cnt.
            double tot_w_num = 0.0, tot_w_den = 0.0, wmean_R_num = 0.0;
            for (int b = 0; b < n_block; ++b) {
                const double cn = rj_at(weight, k, b);
                if (cn <= 0.0) continue;
                const double est_num = rj_at(num, k, b) / cn;
                const double est_den = rj_at(den, k, b) / cn;
                const double rel = cn / sum_cnt;
                const double loo_num = (tot_num - est_num * rel) / (1.0 - rel);
                const double loo_den = (tot_den - est_den * rel) / (1.0 - rel);
                const double R = loo_num / loo_den;
                const double w = 1.0 - rel;
                tot_w_num += R * w;
                tot_w_den += w;
                wmean_R_num += R * cn;
            }
            const double tot = tot_w_num / tot_w_den;

            // Pass 2c — diffsum = Σ (tot − R_b). (R_b recomputed bit-identically from the SAME
            // tot_num/tot_den — the host stores Rb[]; recompute gives the identical double.)
            double diffsum = 0.0;
            for (int b = 0; b < n_block; ++b) {
                const double cn = rj_at(weight, k, b);
                if (cn <= 0.0) continue;
                const double est_num = rj_at(num, k, b) / cn;
                const double est_den = rj_at(den, k, b) / cn;
                const double rel = cn / sum_cnt;
                const double loo_num = (tot_num - est_num * rel) / (1.0 - rel);
                const double loo_den = (tot_den - est_den * rel) / (1.0 - rel);
                const double R = loo_num / loo_den;
                diffsum += tot - R;
            }
            est = (diffsum / nb) * nb + wmean_R_num / sum_cnt;

            // Pass 3 — xtau variance: h=Σcnt/cnt; tau=h·tot−(h−1)·R−est; var=Σ(tau²/(h−1))/nb.
            double var_acc = 0.0;
            for (int b = 0; b < n_block; ++b) {
                const double cn = rj_at(weight, k, b);
                if (cn <= 0.0) continue;
                const double est_num = rj_at(num, k, b) / cn;
                const double est_den = rj_at(den, k, b) / cn;
                const double rel = cn / sum_cnt;
                const double loo_num = (tot_num - est_num * rel) / (1.0 - rel);
                const double loo_den = (tot_den - est_den * rel) / (1.0 - rel);
                const double R = loo_num / loo_den;
                const double h = sum_cnt / cn;
                const double tau = h * tot - (h - 1.0) * R - est;
                var_acc += (tau * tau) / (h - 1.0);
            }
            const double var = var_acc / nb;
            se = (var > 0.0) ? sqrt(var) : knan;
            z = est / se;
        }

        d_est[k] = est;
        d_se[k] = se;
        d_z[k] = z;
        if (compute_p) d_p[k] = erfc(fabs(z) * kInvSqrt2);  // AT2 ztop == f4_two_sided_p.
    }
}

}  // namespace

void launch_ratio_block_jackknife(const DRatioJackArray& num, const DRatioJackArray& den,
                                  const DRatioJackArray& weight, const DRatioJackArray& xblk_num,
                                  const DRatioJackArray& xblk_den, int N, int n_block,
                                  int tot_mode, double setmiss_thresh, bool compute_p,
                                  double* d_est, double* d_se, double* d_z, double* d_p,
                                  cudaStream_t stream) {
    if (N <= 0 || n_block <= 0) return;
    constexpr int kThreads = 128;
    long blocks = (static_cast<long>(N) + kThreads - 1) / kThreads;
    if (blocks < 1) blocks = 1;
    if (blocks > static_cast<long>(core::kMaxGridX)) blocks = core::kMaxGridX;
    ratio_block_jackknife_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        num, den, weight, xblk_num, xblk_den, N, n_block, tot_mode, setmiss_thresh, compute_p,
        d_est, d_se, d_z, d_p);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

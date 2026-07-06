// src/device/cuda/ratio_block_jackknife_kernel.cu
//
// The shared on-device ratio block-jackknife kernel serving both the f4-ratio and
// D-statistic jackknives: one thread per item, one kernel selecting the statistic via a
// tot_mode branch behind one launch wrapper.
//
// Reference: docs/reference/src_device_cuda_ratio_block_jackknife_kernel.cu.md
#include <cuda_runtime.h>

#include "core/internal/launch_config.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/ratio_block_jackknife_kernel.cuh"

namespace steppe::device {

namespace {

// Per-(item, block) descriptor indexing — reference §4
__device__ __forceinline__ double rj_at(const DRatioJackArray& a, long k, int b) {
    return a.data[a.base + k * a.item_stride + static_cast<long>(b) * a.block_stride];
}

// The shared ratio block-jackknife kernel (f4-ratio + D-statistic modes) — reference §6
__global__ void ratio_block_jackknife_kernel(DRatioJackArray num, DRatioJackArray den,
                                             DRatioJackArray weight, DRatioJackArray xblk_num,
                                             DRatioJackArray xblk_den, int N, int n_block,
                                             int tot_mode, double setmiss_thresh, bool compute_p,
                                             double* __restrict__ d_est, double* __restrict__ d_se,
                                             double* __restrict__ d_z, double* __restrict__ d_p) {
    const double knan = nan("");
    const double kInvSqrt2 = 0.7071067811865475244;
    for (long k = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         k < static_cast<long>(N);
         k += static_cast<long>(gridDim.x) * blockDim.x) {

        double est = knan, se = knan, z = knan;

        if (tot_mode == 0) {
            double n_w = 0.0;
            int nb_surv = 0;
            double totnum = 0.0, totden = 0.0;
            for (int b = 0; b < n_block; ++b) {
                const double den_blk = rj_at(xblk_den, k, b);
                if (fabs(den_blk) < setmiss_thresh) continue;
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
            const double tot = totnum / totden;

            double diffsum = 0.0, wmean_num = 0.0;
            for (int b = 0; b < n_block; ++b) {
                const double den_blk = rj_at(xblk_den, k, b);
                if (fabs(den_blk) < setmiss_thresh) continue;
                const double bl = rj_at(weight, k, b);
                const double Rb = rj_at(num, k, b) / rj_at(den, k, b);
                diffsum += tot - Rb;
                wmean_num += Rb * bl;
            }
            const double term1 = (diffsum / nb) * nb;
            const double term2 = wmean_num / n_w;
            est = term1 + term2;

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
        if (compute_p) d_p[k] = erfc(fabs(z) * kInvSqrt2);
    }
}

}  // namespace

// Launch geometry and early-out wrapper — reference §5
void launch_ratio_block_jackknife(const DRatioJackArray& num, const DRatioJackArray& den,
                                  const DRatioJackArray& weight, const DRatioJackArray& xblk_num,
                                  const DRatioJackArray& xblk_den, int N, int n_block,
                                  int tot_mode, double setmiss_thresh, bool compute_p,
                                  double* d_est, double* d_se, double* d_z, double* d_p,
                                  cudaStream_t stream) {
    if (N <= 0 || n_block <= 0) return;
    constexpr int kThreads = 128;
    const long blocks = core::grid_stride_extent(N, kThreads);
    ratio_block_jackknife_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        num, den, weight, xblk_num, xblk_den, N, n_block, tot_mode, setmiss_thresh, compute_p,
        d_est, d_se, d_z, d_p);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

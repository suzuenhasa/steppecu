// src/device/cuda/qpfstats_jackknife_kernel.cu
//
// The qpfstats on-device block-jackknife kernels: they reduce the resident per-block
// numerator/count sums down the block axis, one output per population combination or
// per pair, in native FP64. A CUDA TU private to steppe_device — the backend reaches
// the kernels only through the launch wrappers in the companion .cuh.
//
// Reference: docs/reference/src_device_cuda_qpfstats_jackknife_kernel.cu.md
#include <cuda_runtime.h>

#include "core/internal/launch_config.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/qpfstats_jackknife_kernel.cuh"

namespace steppe::device {

namespace {

// Per-combination global block-jackknife estimate — reference §3
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

        for (int b = 0; b < n_block; ++b) {
            const double cn = cnt[base + b];
            const double mean = (cn > 0.0) ? (numsum[base + b] / cn) : knan;
            numer[base + b] = mean;
            ymat[c + static_cast<long>(npopcomb) * b] = mean;
        }

        double sum_n_all = 0.0;
        for (int b = 0; b < n_block; ++b) sum_n_all += cnt[base + b];
        if (!(sum_n_all > 0.0)) { y[c] = knan; continue; }

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
            const double rel = cn / sum_n_all;
            if (rel >= 1.0) continue;
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

// Per-pair recenter shift — reference §4
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

        double sum_bl = 0.0;
        for (int blk = 0; blk < n_block; ++blk)
            if (isfinite(arr(blk))) sum_bl += static_cast<double>(block_sizes[blk]);
        if (!(sum_bl > 0.0)) { shift[p] = bglob[p] - 0.0; continue; }

        double tot_num = 0.0;
        for (int blk = 0; blk < n_block; ++blk) {
            const double a = arr(blk);
            if (!isfinite(a)) continue;
            tot_num += a * static_cast<double>(block_sizes[blk]);
        }
        const double tot = tot_num / sum_bl;

        double num = 0.0, den = 0.0;
        for (int blk = 0; blk < n_block; ++blk) {
            const double a = arr(blk);
            if (!isfinite(a)) continue;
            const double blb = static_cast<double>(block_sizes[blk]);
            const double rel = blb / sum_bl;
            if (rel >= 1.0) continue;
            const double loo = (tot - a * rel) / (1.0 - rel);
            const double h = sum_bl / blb;
            const double w = 1.0 - 1.0 / h;
            num += loo * w;
            den += w;
        }
        const double est = (den > 0.0) ? (num / den) : 0.0;
        shift[p] = bglob[p] - est;
    }
}

}  // namespace

// Launch wrappers — reference §6
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

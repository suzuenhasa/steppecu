// src/device/cuda/roh_fb_kernel.cu
//
// The GPU hapROH copying forward-backward kernel (the `steppe roh` FB core). One CUDA
// block per target: the M-sequential (K+1)-state scan lives entirely inside the block
// and __syncthreads() is the per-column barrier. The K ROH copying states spread over
// threadIdx.x (grid-stride over K); the distinguished non-ROH state 0 is carried in a
// shared scalar. Batching over targets = blocks is the parallelism.
//
// This forks the Li-Stephens kernel's machinery (block_reduce_sum as the pooled ROH-mass
// reduction, native-FP64 per-column rescale, and the always-on checkpoint/recompute with
// stride C = ceil(sqrt(M))) and swaps in the ROH pooled recursion:
//   f       = sum_{k>=1} alpha_prev[k]                                   (pooled ROH mass)
//   a0_new  = e0(l) * ( a0_prev*t00 + f*t10 )                            (non-ROH state 0)
//   ak_new  = ek(l) * ( a0_prev*t01 + f*t12 + alpha_prev[k]*stay )       (ROH state k)
// mirroring core::roh_fb_target (the CPU oracle) column for column. Native FP64
// throughout (the underflow-prone sub-one product scan — the reduction carve-out, NOT
// the emulated-matmul default).
//
// Reference: docs/planning/haproh-face-spec.md §3
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/stats/roh_model.hpp"  // roh_emission0 / roh_emission_copy (STEPPE_HD)
#include "device/cuda/check.cuh"
#include "device/cuda/roh_fb_kernel.cuh"

namespace steppe::device {

namespace {

inline constexpr int kBlock = 256;  // power of two: the tree reduction requires it

// Block-wide sum reduction over `val` (one partial per thread). Result broadcast to
// every thread via sh[0]. Inactive threads (tid >= K) pass val == 0. The trailing
// __syncthreads() leaves `sh` free for the next reduction. blockDim.x a power of two.
__device__ inline double block_reduce_sum(double val, double* __restrict__ sh) {
    const int tid = threadIdx.x;
    sh[tid] = val;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) sh[tid] += sh[tid + off];
        __syncthreads();
    }
    const double total = sh[0];
    __syncthreads();
    return total;
}

// One forward column step l>=1 (block-cooperative), factored so the forward sweep and
// the backward recompute call the IDENTICAL instruction stream (bit-identical replay).
// a0_prev is the previous column's state-0 value (broadcast, identical on every thread);
// aroh_prev the previous ROH column; aroh_out receives the normalized ROH column. Returns
// the normalized state-0 value (identical on every thread).
__device__ inline double roh_forward_step(double a0_prev, const double* __restrict__ aroh_prev,
                                          double* __restrict__ aroh_out, long l, int K, long M,
                                          const std::uint8_t* __restrict__ ob,
                                          const std::uint8_t* __restrict__ refhaps,
                                          const double* __restrict__ p, const double* __restrict__ T,
                                          double e_rate, double* __restrict__ sh) {
    const int tid = threadIdx.x;
    const std::size_t Ms = static_cast<std::size_t>(M);
    double loc = 0.0;
    for (int k = tid; k < K; k += blockDim.x) loc += aroh_prev[k];
    const double f = block_reduce_sum(loc, sh);  // pooled ROH mass at l-1

    const std::uint8_t obl = ob[static_cast<std::size_t>(l)];
    const double e0 = steppe::core::roh_emission0(obl, p[static_cast<std::size_t>(l)], e_rate);
    const std::size_t Tb = static_cast<std::size_t>(l) * 9;
    const double t00 = T[Tb + 0], t01 = T[Tb + 1];
    const double t10 = T[Tb + 3], t11 = T[Tb + 4], t12 = T[Tb + 5];
    const double stay = t11 - t12;
    const double x1 = a0_prev * t01;
    const double x2 = f * t12;

    double locs = 0.0;
    for (int k = tid; k < K; k += blockDim.x) {
        const std::uint8_t rk = refhaps[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)];
        const double ek = steppe::core::roh_emission_copy(rk, obl, e_rate);
        const double v = ek * (x1 + x2 + aroh_prev[k] * stay);
        aroh_out[k] = v;
        locs += v;
    }
    const double na0 = e0 * (a0_prev * t00 + f * t10);
    if (tid == 0) locs += na0;
    const double s = block_reduce_sum(locs, sh);
    const double inv = (s > 0.0) ? (1.0 / s) : 0.0;  // degenerate-column guard
    for (int k = tid; k < K; k += blockDim.x) aroh_out[k] *= inv;
    __syncthreads();
    return na0 * inv;  // identical on every thread
}

// The (K+1)-state forward-backward kernel — one block per target (blockIdx.x = tid).
__global__ void roh_fb_kernel(const std::uint8_t* __restrict__ ob_all,
                              const std::uint8_t* __restrict__ refhaps, const double* __restrict__ p,
                              const double* __restrict__ T, int K, long M, int C, int nck,
                              double e_rate, double in_val, double* __restrict__ proh_all,
                              double* __restrict__ check_roh_all, double* __restrict__ check0_all,
                              double* __restrict__ alphaA_all, double* __restrict__ alphaB_all,
                              double* __restrict__ alpha_blk_all, double* __restrict__ a0_blk_all,
                              double* __restrict__ betaA_all, double* __restrict__ betaB_all) {
    const int tgt = blockIdx.x;
    const int tid = threadIdx.x;
    const std::size_t Ks = static_cast<std::size_t>(K);
    const std::size_t Ms = static_cast<std::size_t>(M);

    // Per-target views into the backend-owned buffers.
    const std::uint8_t* ob = ob_all + static_cast<std::size_t>(tgt) * Ms;
    double* proh = proh_all + static_cast<std::size_t>(tgt) * Ms;
    double* check_roh = check_roh_all + static_cast<std::size_t>(tgt) * static_cast<std::size_t>(nck) * Ks;
    double* check0 = check0_all + static_cast<std::size_t>(tgt) * static_cast<std::size_t>(nck);
    double* alphaA = alphaA_all + static_cast<std::size_t>(tgt) * Ks;
    double* alphaB = alphaB_all + static_cast<std::size_t>(tgt) * Ks;
    double* alpha_blk = alpha_blk_all + static_cast<std::size_t>(tgt) * static_cast<std::size_t>(C) * Ks;
    double* a0_blk = a0_blk_all + static_cast<std::size_t>(tgt) * static_cast<std::size_t>(C);
    double* betaA = betaA_all + static_cast<std::size_t>(tgt) * Ks;
    double* betaB = betaB_all + static_cast<std::size_t>(tgt) * Ks;

    __shared__ double sh[kBlock];

    // ---- Forward sweep: store ONLY the normalized checkpoint columns ---------
    double* cur = alphaA;
    double* nxt = alphaB;
    // Column 0: the in_val prior, NO emission, NO rescale (sum is exactly 1).
    double a0 = 1.0 - static_cast<double>(K) * in_val;
    for (int k = tid; k < K; k += blockDim.x) cur[k] = in_val;
    __syncthreads();
    for (int k = tid; k < K; k += blockDim.x) check_roh[k] = cur[k];  // checkpoint 0
    if (tid == 0) check0[0] = a0;
    __syncthreads();

    for (long l = 1; l < M; ++l) {
        const double a0_new =
            roh_forward_step(a0, cur, nxt, l, K, M, ob, refhaps, p, T, e_rate, sh);
        double* tmp = cur;
        cur = nxt;
        nxt = tmp;  // now cur = alpha_l, nxt = alpha_{l-1}
        a0 = a0_new;
        if (l % C == 0) {
            const int bi = static_cast<int>(l / C);
            double* ck = check_roh + static_cast<std::size_t>(bi) * Ks;
            for (int k = tid; k < K; k += blockDim.x) ck[k] = cur[k];
            if (tid == 0) check0[bi] = a0;
            __syncthreads();
        }
    }

    // ---- Backward sweep + recompute, block by block right -> left ------------
    double* beta_cur = betaA;
    double* beta_nxt = betaB;
    for (int k = tid; k < K; k += blockDim.x) beta_cur[k] = 1.0;
    double beta0 = 1.0;  // identical on every thread
    __syncthreads();

    for (int bi = nck - 1; bi >= 0; --bi) {
        const long b0 = static_cast<long>(bi) * static_cast<long>(C);
        long b1 = b0 + static_cast<long>(C);
        if (b1 > M) b1 = M;
        const int len = static_cast<int>(b1 - b0);

        // Recompute the forward tile [b0, b1) from the stored checkpoint. block[0] IS the
        // normalized alpha at column b0 (bit-identical to the forward sweep).
        const double* ck = check_roh + static_cast<std::size_t>(bi) * Ks;
        for (int k = tid; k < K; k += blockDim.x) alpha_blk[k] = ck[k];
        double a0_col = check0[bi];
        if (tid == 0) a0_blk[0] = a0_col;
        __syncthreads();
        for (int j = 1; j < len; ++j) {
            const long l = b0 + j;
            a0_col = roh_forward_step(a0_col, alpha_blk + static_cast<std::size_t>(j - 1) * Ks,
                                      alpha_blk + static_cast<std::size_t>(j) * Ks, l, K, M, ob,
                                      refhaps, p, T, e_rate, sh);
            if (tid == 0) a0_blk[j] = a0_col;
            __syncthreads();
        }

        // Descend within the block: posterior at column l, then step beta_l -> beta_{l-1}.
        for (long l = b1 - 1; l >= b0; --l) {
            const int j = static_cast<int>(l - b0);
            const double* aroh_l = alpha_blk + static_cast<std::size_t>(j) * Ks;
            const double a0_l = a0_blk[j];

            // gamma_0(l) = a0_l*beta0 / denom, denom = sum_s alpha_s(l)*beta_s(l).
            double locd = 0.0;
            for (int k = tid; k < K; k += blockDim.x) locd += aroh_l[k] * beta_cur[k];
            const double denom = block_reduce_sum(locd, sh) + a0_l * beta0;
            const double gamma0 = (denom > 0.0) ? (a0_l * beta0 / denom) : 0.0;
            if (tid == 0) proh[static_cast<std::size_t>(l)] = 1.0 - gamma0;
            __syncthreads();  // finish reading beta_cur before the beta step overwrites it

            if (l == 0) break;

            // beta_l -> beta_{l-1} using T[l], emissions at column l.
            const std::uint8_t obl = ob[static_cast<std::size_t>(l)];
            const double e0 = steppe::core::roh_emission0(obl, p[static_cast<std::size_t>(l)], e_rate);
            const std::size_t Tb = static_cast<std::size_t>(l) * 9;
            const double t00 = T[Tb + 0], t01 = T[Tb + 1];
            const double t10 = T[Tb + 3], t11 = T[Tb + 4], t12 = T[Tb + 5];
            const double stay = t11 - t12;
            double locf = 0.0;
            for (int k = tid; k < K; k += blockDim.x) {
                const std::uint8_t rk = refhaps[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)];
                locf += beta_cur[k] * steppe::core::roh_emission_copy(rk, obl, e_rate);
            }
            const double fl = block_reduce_sum(locf, sh);
            const double nb0 = beta0 * t00 * e0 + fl * t01;
            const double x1 = e0 * beta0 * t10;
            double locs = 0.0;
            for (int k = tid; k < K; k += blockDim.x) {
                const std::uint8_t rk = refhaps[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)];
                const double ek = steppe::core::roh_emission_copy(rk, obl, e_rate);
                const double v = x1 + fl * t12 + ek * beta_cur[k] * stay;
                beta_nxt[k] = v;
                locs += v;
            }
            if (tid == 0) locs += nb0;
            const double s = block_reduce_sum(locs, sh);
            const double inv = (s > 0.0) ? (1.0 / s) : 0.0;
            for (int k = tid; k < K; k += blockDim.x) beta_nxt[k] *= inv;
            beta0 = nb0 * inv;  // identical on every thread
            __syncthreads();
            double* tmp = beta_cur;
            beta_cur = beta_nxt;
            beta_nxt = tmp;
        }
    }
}

}  // namespace

void launch_roh_fb(const std::uint8_t* d_ob, const std::uint8_t* d_refhaps, const double* d_p,
                   const double* d_T, int K, long M, int n_target, int C, int nck, double e_rate,
                   double in_val, double* d_proh, double* d_check_roh, double* d_check0,
                   double* d_alphaA, double* d_alphaB, double* d_alpha_blk, double* d_a0_blk,
                   double* d_betaA, double* d_betaB, cudaStream_t stream) {
    if (K <= 0 || M <= 0 || n_target <= 0) return;
    roh_fb_kernel<<<n_target, kBlock, 0, stream>>>(
        d_ob, d_refhaps, d_p, d_T, K, M, C, nck, e_rate, in_val, d_proh, d_check_roh, d_check0,
        d_alphaA, d_alphaB, d_alpha_blk, d_a0_blk, d_betaA, d_betaB);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

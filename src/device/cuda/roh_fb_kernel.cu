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
// ONE templated FB body (`roh_fb_kernel_t<Ref>`), TWO thin launchers over the SAME math:
//   - launch_roh_fb       — the non-batch oracle (CudaBackend::roh_fb, tests): a shared
//                           compacted K*M refhaps via the Compact accessor, uniform M/C/nck.
//   - launch_roh_fb_wave  — the batch path (CudaBackend::roh_fb_batch): a single grid of W
//                           item-blocks, each indexing the ONCE-resident panel DIRECTLY via
//                           the Panel accessor (donor_map/site_map indirection folded in),
//                           with per-item M/C/nck arrays. Feeds hundreds of blocks per wave.
// The Panel accessor returns panel[donor_map[k]*Mp + site_map[l]] — byte-for-byte the value
// the old per-item roh_gather wrote into the compacted refhaps — so both paths are bit-
// identical: the FB recursions, the grid-stride reduction order over K, the native-FP64
// per-column rescale, the column-0 prior, emissions and transitions are ALL unchanged; only
// WHERE a refhaps byte is fetched and the per-block buffer base offsets differ.
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

// Refhaps accessors — the ONLY difference between the two FB paths. operator()(blk, k, l)
// returns the reference-haplotype byte for donor k at kept site l of block/item blk. Both
// return exactly the byte the CPU oracle / old roh_gather placed at refhaps[k*M + l], so
// no floating-point op sees a different input in either path.

// Non-batch: a shared compacted donor-major refhaps, row stride = M (blk unused — every
// target scans the same panel-compacted axis).
struct CompactRefhaps {
    const std::uint8_t* r;
    long M;
    __device__ std::uint8_t operator()(int /*blk*/, int k, long l) const {
        return r[static_cast<std::size_t>(k) * static_cast<std::size_t>(M) +
                 static_cast<std::size_t>(l)];
    }
};

// Wave: the per-item site_map indexes the ONCE-resident donor-major panel directly. site_map
// rows are strided by Mstride (= the batch-wide Mmax); Mp is the panel column stride.
struct PanelRefhaps {
    const std::uint8_t* panel;
    const int* donor_map;
    const int* site_map_all;
    long Mp;
    long Mstride;
    __device__ std::uint8_t operator()(int blk, int k, long l) const {
        const int* sm =
            site_map_all + static_cast<std::size_t>(blk) * static_cast<std::size_t>(Mstride);
        return panel[static_cast<std::size_t>(donor_map[k]) * static_cast<std::size_t>(Mp) +
                     static_cast<std::size_t>(sm[l])];
    }
};

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
// the normalized state-0 value (identical on every thread). Templated on the refhaps
// accessor `Ref` (Compact / Panel) — the only per-path difference is where ref(blk,k,l)
// reads the reference-haplotype byte; every arithmetic op is identical.
template <class Ref>
__device__ inline double roh_forward_step(double a0_prev, const double* __restrict__ aroh_prev,
                                          double* __restrict__ aroh_out, long l, int K, int blk,
                                          const std::uint8_t* __restrict__ ob, const Ref& ref,
                                          const double* __restrict__ p, const double* __restrict__ T,
                                          double e_rate, double* __restrict__ sh) {
    const int tid = threadIdx.x;
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
        const std::uint8_t rk = ref(blk, k, l);
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

// The unified (K+1)-state forward-backward kernel — one block per item (blockIdx.x = blk).
// Per-block M/C/nck come from the *_all arrays when non-null (wave), else the uniform
// scalars Muni/Cuni/nckuni (non-batch). Buffer bases are per-block via the strides:
// ob/proh by Mstride, p/T by pTstride (0 => shared across blocks, the non-batch case),
// check_roh/check0 by nckstride, the C-tile (alpha_blk/a0_blk) by Cstride, the ping-pong
// K-columns by K. The FB math below is byte-identical to the historical roh_fb_kernel.
template <class Ref>
__global__ void roh_fb_kernel_t(
    const std::uint8_t* __restrict__ ob_all, Ref ref, const double* __restrict__ p_all,
    const double* __restrict__ T_all, const int* __restrict__ M_all, const int* __restrict__ C_all,
    const int* __restrict__ nck_all, int Muni, int Cuni, int nckuni, long Mstride, long Cstride,
    long nckstride, long pTstride, int K, double e_rate, double in_val,
    double* __restrict__ proh_all, double* __restrict__ check_roh_all,
    double* __restrict__ check0_all, double* __restrict__ alphaA_all,
    double* __restrict__ alphaB_all, double* __restrict__ alpha_blk_all,
    double* __restrict__ a0_blk_all, double* __restrict__ betaA_all,
    double* __restrict__ betaB_all) {
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    const int M = M_all ? M_all[blk] : Muni;
    if (M <= 0) return;  // empty item — nothing to do (skipped by the consumer)
    const int C = C_all ? C_all[blk] : Cuni;
    const int nck = nck_all ? nck_all[blk] : nckuni;
    const std::size_t Ks = static_cast<std::size_t>(K);
    const std::size_t blkz = static_cast<std::size_t>(blk);

    // Per-block views into the backend-owned buffers.
    const std::uint8_t* ob = ob_all + blkz * static_cast<std::size_t>(Mstride);
    const double* p = p_all + blkz * static_cast<std::size_t>(pTstride);
    const double* T = T_all + blkz * static_cast<std::size_t>(pTstride) * 9;
    double* proh = proh_all + blkz * static_cast<std::size_t>(Mstride);
    double* check_roh = check_roh_all + blkz * static_cast<std::size_t>(nckstride) * Ks;
    double* check0 = check0_all + blkz * static_cast<std::size_t>(nckstride);
    double* alphaA = alphaA_all + blkz * Ks;
    double* alphaB = alphaB_all + blkz * Ks;
    double* alpha_blk = alpha_blk_all + blkz * static_cast<std::size_t>(Cstride) * Ks;
    double* a0_blk = a0_blk_all + blkz * static_cast<std::size_t>(Cstride);
    double* betaA = betaA_all + blkz * Ks;
    double* betaB = betaB_all + blkz * Ks;

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
        const double a0_new = roh_forward_step(a0, cur, nxt, l, K, blk, ob, ref, p, T, e_rate, sh);
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
                                      alpha_blk + static_cast<std::size_t>(j) * Ks, l, K, blk, ob,
                                      ref, p, T, e_rate, sh);
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
                const std::uint8_t rk = ref(blk, k, l);
                locf += beta_cur[k] * steppe::core::roh_emission_copy(rk, obl, e_rate);
            }
            const double fl = block_reduce_sum(locf, sh);
            const double nb0 = beta0 * t00 * e0 + fl * t01;
            const double x1 = e0 * beta0 * t10;
            double locs = 0.0;
            for (int k = tid; k < K; k += blockDim.x) {
                const std::uint8_t rk = ref(blk, k, l);
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
    // Non-batch: n_target blocks, a shared compacted refhaps / p / T (pTstride = 0), uniform
    // M/C/nck; per-target ob/proh/checkpoints/tiles via Mstride = M, Cstride = C, nckstride = nck.
    roh_fb_kernel_t<CompactRefhaps><<<static_cast<unsigned>(n_target), kBlock, 0, stream>>>(
        d_ob, CompactRefhaps{d_refhaps, M}, d_p, d_T, /*M_all*/nullptr, /*C_all*/nullptr,
        /*nck_all*/nullptr, /*Muni*/static_cast<int>(M), /*Cuni*/C, /*nckuni*/nck, /*Mstride*/M,
        /*Cstride*/C, /*nckstride*/nck, /*pTstride*/0, K, e_rate, in_val, d_proh, d_check_roh,
        d_check0, d_alphaA, d_alphaB, d_alpha_blk, d_a0_blk, d_betaA, d_betaB);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_roh_fb_wave(const std::uint8_t* d_ob, const std::uint8_t* d_panel,
                        const int* d_donor_map, const int* d_site_map, const double* d_p,
                        const double* d_T, const int* d_M, const int* d_C, const int* d_nck,
                        int K, long Mp, long Mstride, long Cstride, int W, double e_rate,
                        double in_val, double* d_proh, double* d_check_roh, double* d_check0,
                        double* d_alphaA, double* d_alphaB, double* d_alpha_blk, double* d_a0_blk,
                        double* d_betaA, double* d_betaB, cudaStream_t stream) {
    if (K <= 0 || W <= 0) return;
    // Wave: W item-blocks in ONE grid, each indexing the resident panel via the site_map;
    // per-item M/C/nck from the arrays; every buffer strided by the batch-wide Mstride/Cstride
    // (nck bound = Cmax, so nckstride == Cstride), p/T per-item (pTstride = Mstride).
    roh_fb_kernel_t<PanelRefhaps><<<static_cast<unsigned>(W), kBlock, 0, stream>>>(
        d_ob, PanelRefhaps{d_panel, d_donor_map, d_site_map, Mp, Mstride}, d_p, d_T, d_M, d_C,
        d_nck, /*Muni*/0, /*Cuni*/0, /*nckuni*/0, Mstride, Cstride, /*nckstride*/Cstride,
        /*pTstride*/Mstride, K, e_rate, in_val, d_proh, d_check_roh, d_check0, d_alphaA, d_alphaB,
        d_alpha_blk, d_a0_blk, d_betaA, d_betaB);
    STEPPE_CUDA_CHECK_KERNEL();
}

namespace {

// Compacted device gather: refhaps[k*M + l] = panel[donor_map[k]*Mp + site_map[l]], flat
// grid-strided over the K*M output — the batch-overlap replacement for the per-item host
// K*M panel gather + PCIe re-upload. Pure indirection over bytes; touches no FB math.
// (Retained for launch_roh_gather; the wave path now folds this indexing into PanelRefhaps.)
__global__ void roh_gather_kernel(const std::uint8_t* __restrict__ panel,
                                  const int* __restrict__ donor_map,
                                  const int* __restrict__ site_map, int K, long M, long Mp,
                                  std::uint8_t* __restrict__ refhaps) {
    const long total = static_cast<long>(K) * M;
    const long stride = static_cast<long>(gridDim.x) * blockDim.x;
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
        const int k = static_cast<int>(i / M);
        const long l = i - static_cast<long>(k) * M;
        refhaps[static_cast<std::size_t>(i)] =
            panel[static_cast<std::size_t>(donor_map[k]) * static_cast<std::size_t>(Mp) +
                  static_cast<std::size_t>(site_map[static_cast<std::size_t>(l)])];
    }
}

}  // namespace

void launch_roh_gather(const std::uint8_t* d_panel, const int* d_donor_map,
                       const int* d_site_map, int K, long M, long Mp, std::uint8_t* d_refhaps,
                       cudaStream_t stream) {
    if (K <= 0 || M <= 0) return;  // empty item — no gather (matches the FB launch guard)
    const long total = static_cast<long>(K) * M;
    long grid = (total + kBlock - 1) / kBlock;
    if (grid > 65535) grid = 65535;  // the grid-stride loop covers the remainder
    roh_gather_kernel<<<static_cast<unsigned>(grid), kBlock, 0, stream>>>(
        d_panel, d_donor_map, d_site_map, K, M, Mp, d_refhaps);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

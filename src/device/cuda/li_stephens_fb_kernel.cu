// src/device/cuda/li_stephens_fb_kernel.cu
//
// The GPU Li-Stephens copying forward-backward kernel (the `steppe paint` FB
// core, Phase 1). One CUDA block per recipient: the M-sequential scan lives
// entirely inside the block and __syncthreads() is the per-column barrier, so
// the recurrence needs no grid-wide synchronization and no per-column kernel
// launches. Donors spread over threadIdx.x (grid-stride over K).
//
// PRECISION: the whole recurrence runs in NATIVE FP64 (NOT emulated-FP64 — the
// FB is a long product of sub-one probabilities that underflows, so each column
// is rescaled by its own sum; the precision policy differs from the matmul-heavy
// default here). Reductions are shared-memory tree reductions in double; gamma is
// per-column scale-invariant so reduction order is not load-bearing.
//
// MEMORY: an always-on checkpoint/recompute scheme (stride C = ceil(sqrt(M))).
// The forward sweep stores only the normalized alpha at each checkpoint column
// (nck*K doubles) — never the full K*M table. The backward sweep, block by block
// right->left, reloads a checkpoint and recomputes the C-column forward tile via
// the SAME single-column forward step used by the forward sweep, so the replay is
// bit-identical (the complete Markov state carried column-to-column is the
// normalized alpha column; recomputing from it replays the identical FP64 ops).
//
// NOTE: d_gamma is materialized here only for the Phase-1 posterior gate. Phase
// 2's paint face folds gamma into an online N x P_donor reduction and never
// stores the full K*M posterior — the resident gamma is not the steady-state
// design.
//
// Reference: docs/planning/li-stephens-engine-scope.md §2a.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/launch_config.hpp"
#include "core/internal/nvtx.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/li_stephens_fb_kernel.cuh"

namespace steppe::device {

namespace {

inline constexpr int kBlock = 256;  // power of two: the tree reduction requires it

// Watterson emission e_l(k), matching CpuBackend::ls_forward_backward exactly
// (cpu_backend.cpp:564-571): an allele byte > 1 is missing -> uninformative (1.0);
// otherwise match -> 1-mu[l], mismatch -> mu[l].
__device__ inline double ls_emission(long l, int k, const std::uint8_t* __restrict__ recipient,
                                     const std::uint8_t* __restrict__ donors,
                                     const double* __restrict__ mu, long M) {
    const std::uint8_t r = recipient[l];
    const std::uint8_t d = donors[static_cast<std::size_t>(k) * static_cast<std::size_t>(M) +
                                  static_cast<std::size_t>(l)];
    if (r > 1u || d > 1u) return 1.0;
    const double m = mu[l];
    return (r == d) ? (1.0 - m) : m;
}

// Block-wide sum reduction over `val` (one partial per thread). Result broadcast
// to every thread via sh[0]. Inactive threads (tid >= K) pass val == 0.0, so the
// shared scratch is fully seeded. The trailing __syncthreads() leaves `sh` free
// for the next column's reduction. Requires blockDim.x a power of two.
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

// Forward column l=0: alpha_0(k) = pi[k]*e_0(k), then rescale by the column sum
// (cpu_backend.cpp:579-588). a_out receives the normalized alpha_0 column.
__device__ inline void forward_init(int K, long M, const std::uint8_t* __restrict__ recipient,
                                    const std::uint8_t* __restrict__ donors,
                                    const double* __restrict__ pi, const double* __restrict__ mu,
                                    double* __restrict__ a_out, double* __restrict__ sh) {
    const int tid = threadIdx.x;
    double loc = 0.0;
    for (int k = tid; k < K; k += blockDim.x) {
        const double a = pi[k] * ls_emission(0, k, recipient, donors, mu, M);
        a_out[k] = a;
        loc += a;
    }
    const double s = block_reduce_sum(loc, sh);
    const double inv = (s > 0.0) ? (1.0 / s) : 0.0;  // degenerate-column guard (cpu:586)
    for (int k = tid; k < K; k += blockDim.x) a_out[k] *= inv;
    __syncthreads();
}

// One forward column step l>=1, factored so the forward sweep and the backward
// recompute call the IDENTICAL instruction stream (the bit-identical-replay
// guarantee): prev_sum reduce -> transition+emission -> s reduce -> normalize.
// a_prev is the (normalized) previous column; a_out receives the normalized alpha_l.
// Matches cpu_backend.cpp:589-611.
__device__ inline void forward_step(long l, int K, long M,
                                    const std::uint8_t* __restrict__ recipient,
                                    const std::uint8_t* __restrict__ donors,
                                    const double* __restrict__ pi, const double* __restrict__ mu,
                                    double rho_l, const double* __restrict__ a_prev,
                                    double* __restrict__ a_out, double* __restrict__ sh) {
    const int tid = threadIdx.x;
    double loc = 0.0;
    for (int k = tid; k < K; k += blockDim.x) loc += a_prev[k];
    const double prev_sum = block_reduce_sum(loc, sh);
    const double om = 1.0 - rho_l;
    double locs = 0.0;
    for (int k = tid; k < K; k += blockDim.x) {
        const double e = ls_emission(l, k, recipient, donors, mu, M);
        const double trans = om * a_prev[k] + rho_l * pi[k] * prev_sum;
        const double a = e * trans;
        a_out[k] = a;
        locs += a;
    }
    const double s = block_reduce_sum(locs, sh);
    const double inv = (s > 0.0) ? (1.0 / s) : 0.0;  // degenerate-column guard (cpu:608)
    for (int k = tid; k < K; k += blockDim.x) a_out[k] *= inv;
    __syncthreads();
}

// The forward-backward kernel — one block per recipient (blockIdx.x = rid).
//
// Two output modes, gated by null pointers so the Phase-1 gate path stays byte-
// identical (§2 of the paint-face spec):
//   - GATE mode  (gamma_all != nullptr, acc_* == nullptr): stores the K*M copying
//     posterior exactly as Phase 1 (the parity gate).
//   - PAINT mode (gamma_all == nullptr, acc_* != nullptr): allocates NO K*M posterior;
//     folds each gamma_l(k) online into the two per-recipient N*K coancestry
//     accumulators (chunkcounts, chunklengths). Single writer per donor k within a
//     block (k = tid, tid+blockDim, ...), so acc_*[rid*K+k] needs no atomics. The
//     chunkcount switch term needs a_{l-1}: within a recomputed tile that is
//     alpha_blk[j-1]; at a block's first column (j==0, b0>0) it is the companion
//     checkpoint checkpts_prev[bi] (the normalized alpha at column b0-1).
__global__ void ls_fb_kernel(const std::uint8_t* __restrict__ recipient_all,
                             const std::uint8_t* __restrict__ donors,
                             const double* __restrict__ pi_all, const double* __restrict__ rho,
                             const double* __restrict__ mu, int K, long M, int C, int nck,
                             double* __restrict__ gamma_all, double* __restrict__ checkpts_all,
                             double* __restrict__ alphaA_all, double* __restrict__ alphaB_all,
                             double* __restrict__ alpha_blk_all, double* __restrict__ betaA_all,
                             double* __restrict__ betaB_all,
                             const double* __restrict__ w, double* __restrict__ acc_cnt_all,
                             double* __restrict__ acc_len_all,
                             double* __restrict__ checkpts_prev_all) {
    const int rid = blockIdx.x;
    const int tid = threadIdx.x;
    const std::size_t Ks = static_cast<std::size_t>(K);
    const std::size_t Ms = static_cast<std::size_t>(M);
    const bool paint = (acc_cnt_all != nullptr);  // paint sink vs gate posterior

    // Per-recipient views into the backend-owned buffers.
    const std::uint8_t* recipient = recipient_all + static_cast<std::size_t>(rid) * Ms;
    const double* pi = pi_all + static_cast<std::size_t>(rid) * Ks;
    double* gamma = paint ? nullptr : gamma_all + static_cast<std::size_t>(rid) * Ks * Ms;
    double* checkpts = checkpts_all + static_cast<std::size_t>(rid) * static_cast<std::size_t>(nck) * Ks;
    double* checkpts_prev =
        paint ? checkpts_prev_all + static_cast<std::size_t>(rid) * static_cast<std::size_t>(nck) * Ks
              : nullptr;
    double* acc_cnt = paint ? acc_cnt_all + static_cast<std::size_t>(rid) * Ks : nullptr;
    double* acc_len = paint ? acc_len_all + static_cast<std::size_t>(rid) * Ks : nullptr;
    double* alphaA = alphaA_all + static_cast<std::size_t>(rid) * Ks;
    double* alphaB = alphaB_all + static_cast<std::size_t>(rid) * Ks;
    double* alpha_blk = alpha_blk_all + static_cast<std::size_t>(rid) * static_cast<std::size_t>(C) * Ks;
    double* betaA = betaA_all + static_cast<std::size_t>(rid) * Ks;
    double* betaB = betaB_all + static_cast<std::size_t>(rid) * Ks;

    __shared__ double sh[kBlock];

    // Paint mode: zero the per-recipient accumulators (each thread owns its k's).
    if (paint) {
        for (int k = tid; k < K; k += blockDim.x) { acc_cnt[k] = 0.0; acc_len[k] = 0.0; }
        __syncthreads();
    }

    // ---- Forward sweep: store ONLY the normalized checkpoint columns ---------
    double* cur = alphaA;
    double* nxt = alphaB;
    forward_init(K, M, recipient, donors, pi, mu, cur, sh);
    for (int k = tid; k < K; k += blockDim.x) checkpts[k] = cur[k];  // checkpoint 0
    __syncthreads();
    for (long l = 1; l < M; ++l) {
        forward_step(l, K, M, recipient, donors, pi, mu, rho[l], cur, nxt, sh);
        double* tmp = cur;
        cur = nxt;
        nxt = tmp;  // now cur = alpha_l, nxt = alpha_{l-1}
        if (l % C == 0) {
            const int bi = static_cast<int>(l / C);
            double* ck = checkpts + static_cast<std::size_t>(bi) * Ks;
            for (int k = tid; k < K; k += blockDim.x) ck[k] = cur[k];  // alpha_{bi*C}
            if (paint) {
                // Companion checkpoint: the column just before this one (bi*C - 1),
                // needed by the chunkcount switch term at the block's first column.
                double* ckp = checkpts_prev + static_cast<std::size_t>(bi) * Ks;
                for (int k = tid; k < K; k += blockDim.x) ckp[k] = nxt[k];  // alpha_{bi*C-1}
            }
            __syncthreads();
        }
    }

    // ---- Backward sweep + recompute, block by block right -> left ------------
    // Seed beta_{M-1} = 1 (the base case, cpu_backend.cpp:615-616) BEFORE the
    // first (rightmost) block reduces gamma at column M-1.
    double* beta_cur = betaA;
    double* beta_nxt = betaB;
    for (int k = tid; k < K; k += blockDim.x) beta_cur[k] = 1.0;
    __syncthreads();

    for (int bi = nck - 1; bi >= 0; --bi) {
        const long b0 = static_cast<long>(bi) * static_cast<long>(C);
        long b1 = b0 + static_cast<long>(C);
        if (b1 > M) b1 = M;
        const int len = static_cast<int>(b1 - b0);

        // Recompute the forward tile [b0, b1) from the stored checkpoint. block[0]
        // IS the normalized alpha at column b0 (bit-identical to the forward sweep);
        // steps 1..len-1 replay forward_step. Never recomputes the l=0 init.
        const double* ck = checkpts + static_cast<std::size_t>(bi) * Ks;
        for (int k = tid; k < K; k += blockDim.x) alpha_blk[k] = ck[k];
        __syncthreads();
        for (int j = 1; j < len; ++j) {
            const long l = b0 + j;
            forward_step(l, K, M, recipient, donors, pi, mu, rho[l],
                         alpha_blk + static_cast<std::size_t>(j - 1) * Ks,
                         alpha_blk + static_cast<std::size_t>(j) * Ks, sh);
        }

        // Descend within the block: gamma at column l, then step beta_l -> beta_{l-1}.
        for (long l = b1 - 1; l >= b0; --l) {
            const int j = static_cast<int>(l - b0);
            const double* alpha_l = alpha_blk + static_cast<std::size_t>(j) * Ks;

            // gamma_l(k) = alpha_l(k)*beta_l(k) / denom (cpu:641-654).
            double locd = 0.0;
            for (int k = tid; k < K; k += blockDim.x) locd += alpha_l[k] * beta_cur[k];
            const double denom = block_reduce_sum(locd, sh);
            const double ginv = (denom > 0.0) ? (1.0 / denom) : 0.0;  // guard (cpu:649)
            if (!paint) {
                // GATE mode: store the full K*M posterior (byte-identical to Phase 1).
                for (int k = tid; k < K; k += blockDim.x)
                    gamma[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)] =
                        alpha_l[k] * beta_cur[k] * ginv;
            } else {
                // PAINT mode: fold gamma_l(k) into the two coancestry accumulators. The
                // previous-column normalized forward a_{l-1}(k) for the switch term is the
                // recomputed tile column j-1, or (at the block head, b0>0) the companion
                // checkpoint; l==0 is the initial chunk (no switch).
                const double* a_prev = (l == 0) ? nullptr
                                       : (j >= 1 ? alpha_blk + static_cast<std::size_t>(j - 1) * Ks
                                                 : checkpts_prev + static_cast<std::size_t>(bi) * Ks);
                const double rl = rho[static_cast<std::size_t>(l)];
                const double wl = w[static_cast<std::size_t>(l)];
                for (int k = tid; k < K; k += blockDim.x) {
                    const double g = alpha_l[k] * beta_cur[k] * ginv;  // gamma_l(k)
                    acc_len[k] += g * wl;
                    double sw;
                    if (l == 0) {
                        sw = g;  // the initial chunk
                    } else {
                        const double pk = pi[k];
                        const double den = (1.0 - rl) * a_prev[k] + rl * pk;
                        sw = (den > 0.0) ? g * rl * pk / den : 0.0;  // guard: pi_k=0/rho=0 -> 0
                    }
                    acc_cnt[k] += sw;
                }
            }
            __syncthreads();  // finish reading beta_cur before the beta step overwrites it

            if (l == 0) break;  // beta_{-1} does not exist (cpu stops at l>=0 producing beta_l)

            // Step beta_l -> beta_{l-1} using rho[l], e_l (cpu:617-639; reindexed:
            // producing beta_{l-1} uses rho[(l-1)+1]=rho[l], e_{(l-1)+1}=e_l).
            const double r = rho[l];
            double loct = 0.0;
            for (int k = tid; k < K; k += blockDim.x)
                loct += pi[k] * ls_emission(l, k, recipient, donors, mu, M) * beta_cur[k];
            const double T = block_reduce_sum(loct, sh);
            const double om = 1.0 - r;
            double locs = 0.0;
            for (int k = tid; k < K; k += blockDim.x) {
                const double e = ls_emission(l, k, recipient, donors, mu, M);
                const double b = om * e * beta_cur[k] + r * T;
                beta_nxt[k] = b;
                locs += b;
            }
            const double s = block_reduce_sum(locs, sh);
            const double binv = (s > 0.0) ? (1.0 / s) : 0.0;  // guard (cpu:636)
            for (int k = tid; k < K; k += blockDim.x) beta_nxt[k] *= binv;
            __syncthreads();
            double* tmp = beta_cur;
            beta_cur = beta_nxt;
            beta_nxt = tmp;
        }
    }
}

}  // namespace

void launch_ls_forward_backward(const std::uint8_t* d_recipient, const std::uint8_t* d_donors,
                                const double* d_pi, const double* d_rho, const double* d_mu,
                                int K, long M, int n_recip, int C, int nck, double* d_gamma,
                                double* d_checkpts, double* d_alphaA, double* d_alphaB,
                                double* d_alpha_blk, double* d_betaA, double* d_betaB,
                                cudaStream_t stream, const double* d_w, double* d_acc_cnt,
                                double* d_acc_len, double* d_checkpts_prev) {
    if (K <= 0 || M <= 0 || n_recip <= 0) return;
    ls_fb_kernel<<<n_recip, kBlock, 0, stream>>>(d_recipient, d_donors, d_pi, d_rho, d_mu, K, M,
                                                 C, nck, d_gamma, d_checkpts, d_alphaA, d_alphaB,
                                                 d_alpha_blk, d_betaA, d_betaB, d_w, d_acc_cnt,
                                                 d_acc_len, d_checkpts_prev);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

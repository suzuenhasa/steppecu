// src/device/cuda/admixture_kernels.cu
//
// The ADMIXTURE Q/F block-EM elementwise kernels (`steppe admixture`) — everything that is
// NOT a cuBLAS GEMM: the per-individual dosage decode, per-SNP mean allele frequency, the
// native-FP64 binomial responsibility map, the multiplicative F/Q updates with the row-
// simplex renormalize, and the native-FP64 log-likelihood reduction. The bit-unpack reuses
// the shared core::genotype_code primitive so the GPU and CPU-oracle decode cannot drift.
#include "device/cuda/admixture_kernels.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"
#include "device/cuda/check.cuh"

namespace steppe::device {

namespace {

constexpr int kBlk = 256;

// Decode ONE packed genotype (individual i, GLOBAL SNP s) into dosage g + validity v. The
// single home for the bit-unpack -> (g, v) map, shared by the full-matrix decode and the
// per-SNP-tile decode so the two CANNOT drift: the SNP-tiled Tier-1 path (decode from the
// resident 2-bit dPacked per tile instead of materializing the full N x M FP64 G/V) is only
// parity-safe because it produces byte-identical (g, v) for the same (i, s) as the full decode.
__device__ __forceinline__ void admix_decode_one(const std::uint8_t* __restrict__ packed,
                                                 std::size_t bytes_per_record, long i, long s,
                                                 double& g, double& v) {
    const std::size_t byte_in_rec =
        static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
    const int pos = static_cast<int>(s % core::kCodesPerByte);
    const std::uint8_t byte =
        packed[static_cast<std::size_t>(i) * bytes_per_record + byte_in_rec];
    const std::uint8_t code = core::genotype_code(byte, pos);
    const bool valid = core::genotype_valid(code);
    g = valid ? static_cast<double>(code) : 0.0;  // diploid: code == dosage 0/1/2
    v = valid ? 1.0 : 0.0;
}

__global__ void admix_decode_kernel(const std::uint8_t* __restrict__ packed,
                                    std::size_t bytes_per_record, long N, long M,
                                    double* __restrict__ G, double* __restrict__ V) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = N * M;
    if (idx >= total) return;
    const long i = idx % N;   // individual (column-major: i fastest)
    const long s = idx / N;   // SNP
    admix_decode_one(packed, bytes_per_record, i, s, G[idx], V[idx]);
}

// Decode one SNP TILE [N x t] at global SNP offset s0 into a tile-local column-major buffer
// (element (i, j) at i + N*j, global SNP s0 + j). Byte-identical to the full decode for the
// same (i, s0 + j) — it routes through the SAME admix_decode_one. This is the Tier-1 wall
// fix: the caller keeps dPacked resident and decodes a [N x tileM] slice per SNP-tile inside
// the existing s0 loops instead of holding two full N x M FP64 buffers (16 B/genotype).
__global__ void admix_decode_tile_kernel(const std::uint8_t* __restrict__ packed,
                                         std::size_t bytes_per_record, long N, long s0, long t,
                                         double* __restrict__ G, double* __restrict__ V) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = N * t;
    if (idx >= total) return;
    const long i = idx % N;        // individual (column-major within the tile: i fastest)
    const long j = idx / N;        // local SNP within the tile
    admix_decode_one(packed, bytes_per_record, i, s0 + j, G[idx], V[idx]);
}

// One block per SNP, block-reduce over individuals -> phat[s].
__global__ void admix_snp_mean_kernel(const double* __restrict__ G, const double* __restrict__ V,
                                      long N, long M, double* __restrict__ phat) {
    const long s = blockIdx.x;
    if (s >= M) return;
    __shared__ double sg[kBlk];
    __shared__ double sv[kBlk];
    double lg = 0.0, lv = 0.0;
    const double* Gs = G + static_cast<std::size_t>(N) * static_cast<std::size_t>(s);
    const double* Vs = V + static_cast<std::size_t>(N) * static_cast<std::size_t>(s);
    for (long i = threadIdx.x; i < N; i += blockDim.x) {
        lg += Vs[i] * Gs[i];
        lv += Vs[i];
    }
    sg[threadIdx.x] = lg;
    sv[threadIdx.x] = lv;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            sg[threadIdx.x] += sg[threadIdx.x + off];
            sv[threadIdx.x] += sv[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double denom = 2.0 * sv[0];
        phat[s] = (denom > 0.0) ? (sg[0] / denom) : 0.5;
    }
}

__global__ void admix_responsibility_kernel(const double* __restrict__ Gt,
                                            const double* __restrict__ Vt,
                                            const double* __restrict__ A, long N, long t,
                                            double eps, double* __restrict__ R2,
                                            double* __restrict__ R1) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = N * t;
    if (idx >= total) return;
    double a = A[idx];
    if (a < eps) a = eps;
    if (a > 1.0 - eps) a = 1.0 - eps;
    const double v = Vt[idx];
    const double g = Gt[idx];
    R2[idx] = v * g / a;
    R1[idx] = v * (2.0 - g) / (1.0 - a);
}

__global__ void admix_update_f_kernel(double* __restrict__ F, const double* __restrict__ S2,
                                      const double* __restrict__ S1, long M, int K, double eps) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = M * K;
    if (idx >= total) return;
    const long s = idx % M;   // column-major F(s,k) = F[s + M*k]
    const int k = static_cast<int>(idx / M);
    const double f = F[idx];
    const std::size_t soff =
        static_cast<std::size_t>(k) + static_cast<std::size_t>(K) * static_cast<std::size_t>(s);
    const double s2 = S2[soff];
    const double s1 = S1[soff];
    const double num = f * s2;
    const double den = f * s2 + (1.0 - f) * s1;
    double fn = (den > 0.0) ? (num / den) : f;
    if (fn < eps) fn = eps;
    if (fn > 1.0 - eps) fn = 1.0 - eps;
    F[idx] = fn;
}

__global__ void admix_complement_kernel(const double* __restrict__ F, double* __restrict__ Fc,
                                        long MK) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= MK) return;
    Fc[idx] = 1.0 - F[idx];
}

// One block per individual; K threads (K small) reduce the row sum in shared memory.
__global__ void admix_update_q_kernel(double* __restrict__ Q, const double* __restrict__ T2,
                                      const double* __restrict__ T1, long N, int K) {
    const long i = blockIdx.x;
    if (i >= N) return;
    extern __shared__ double srow[];
    double qk = 0.0;
    if (threadIdx.x < static_cast<unsigned>(K)) {
        const std::size_t off = static_cast<std::size_t>(i) +
                                static_cast<std::size_t>(N) * static_cast<std::size_t>(threadIdx.x);
        qk = Q[off] * (T2[off] + T1[off]);
        srow[threadIdx.x] = qk;
    }
    __syncthreads();
    // Serial reduce in thread 0's view (K small) into srow[0]'s slot after copy.
    __shared__ double rowsum;
    if (threadIdx.x == 0) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) s += srow[k];
        rowsum = s;
    }
    __syncthreads();
    if (threadIdx.x < static_cast<unsigned>(K)) {
        const std::size_t off = static_cast<std::size_t>(i) +
                                static_cast<std::size_t>(N) * static_cast<std::size_t>(threadIdx.x);
        Q[off] = (rowsum > 0.0) ? (qk / rowsum) : (1.0 / static_cast<double>(K));
    }
}

__global__ void admix_loglik_kernel(const double* __restrict__ Gt, const double* __restrict__ Vt,
                                    const double* __restrict__ A, long N, long t, double eps,
                                    double* __restrict__ ll) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = N * t;
    __shared__ double sred[kBlk];
    double local = 0.0;
    if (idx < total) {
        const double v = Vt[idx];
        if (v > 0.0) {
            double a = A[idx];
            if (a < eps) a = eps;
            if (a > 1.0 - eps) a = 1.0 - eps;
            const double g = Gt[idx];
            local = g * log(a) + (2.0 - g) * log(1.0 - a);
        }
    }
    sred[threadIdx.x] = local;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sred[threadIdx.x] += sred[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(ll, sred[0]);
}

// ---------------------------------------------------------------------------------------------
// Batched (S-seed) variants for the SQUAREM + batched-restart accelerator. blockIdx.z = seed;
// per-seed buffers carry an explicit stride. G/V stay seed-independent (shared decode). An
// optional d_active mask (nullptr = all active) freezes settled/converged seeds to no-ops.

__global__ void admix_responsibility_b_kernel(const double* __restrict__ Gt,
                                              const double* __restrict__ Vt,
                                              const double* __restrict__ A, long N, long t,
                                              double eps, double* __restrict__ R2,
                                              double* __restrict__ R1, long stride_tile) {
    const long s = blockIdx.z;
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = N * t;
    if (idx >= total) return;
    const long o = s * stride_tile + idx;
    double a = A[o];
    if (a < eps) a = eps;
    if (a > 1.0 - eps) a = 1.0 - eps;
    const double v = Vt[idx];  // seed-independent
    const double g = Gt[idx];
    R2[o] = v * g / a;
    R1[o] = v * (2.0 - g) / (1.0 - a);
}

__global__ void admix_update_f_b_kernel(double* __restrict__ F, const double* __restrict__ S2,
                                        const double* __restrict__ S1, long M, int K, double eps,
                                        long stride_F, long stride_S,
                                        const double* __restrict__ active) {
    const long s = blockIdx.z;
    if (active && active[s] == 0.0) return;
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = M * K;
    if (idx >= total) return;
    const long sM = idx % M;
    const int k = static_cast<int>(idx / M);
    const long fo = s * stride_F + idx;
    const double f = F[fo];
    const long so = s * stride_S +
                    (static_cast<long>(k) + static_cast<long>(K) * sM);
    const double s2 = S2[so];
    const double s1 = S1[so];
    const double num = f * s2;
    const double den = f * s2 + (1.0 - f) * s1;
    double fn = (den > 0.0) ? (num / den) : f;
    if (fn < eps) fn = eps;
    if (fn > 1.0 - eps) fn = 1.0 - eps;
    F[fo] = fn;
}

__global__ void admix_complement_b_kernel(const double* __restrict__ F, double* __restrict__ Fc,
                                          long MK, long stride, const double* __restrict__ active) {
    const long s = blockIdx.z;
    if (active && active[s] == 0.0) return;
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= MK) return;
    const long o = s * stride + idx;
    Fc[o] = 1.0 - F[o];
}

__global__ void admix_update_q_b_kernel(double* __restrict__ Q, const double* __restrict__ T2,
                                        const double* __restrict__ T1, long N, int K,
                                        long stride_Q, const double* __restrict__ active) {
    const long s = blockIdx.z;
    if (active && active[s] == 0.0) return;
    const long i = blockIdx.x;
    if (i >= N) return;
    extern __shared__ double srow[];
    const long base = s * stride_Q;
    double qk = 0.0;
    if (threadIdx.x < static_cast<unsigned>(K)) {
        const long off = base + i + static_cast<long>(N) * threadIdx.x;
        qk = Q[off] * (T2[off] + T1[off]);
        srow[threadIdx.x] = qk;
    }
    __syncthreads();
    __shared__ double rowsum;
    if (threadIdx.x == 0) {
        double sm = 0.0;
        for (int k = 0; k < K; ++k) sm += srow[k];
        rowsum = sm;
    }
    __syncthreads();
    if (threadIdx.x < static_cast<unsigned>(K)) {
        const long off = base + i + static_cast<long>(N) * threadIdx.x;
        Q[off] = (rowsum > 0.0) ? (qk / rowsum) : (1.0 / static_cast<double>(K));
    }
}

__global__ void admix_loglik_b_kernel(const double* __restrict__ Gt, const double* __restrict__ Vt,
                                      const double* __restrict__ A, long N, long t, double eps,
                                      double* __restrict__ ll, long stride_tile) {
    const long s = blockIdx.z;
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = N * t;
    __shared__ double sred[kBlk];
    double local = 0.0;
    if (idx < total) {
        const double v = Vt[idx];
        if (v > 0.0) {
            double a = A[s * stride_tile + idx];
            if (a < eps) a = eps;
            if (a > 1.0 - eps) a = 1.0 - eps;
            const double g = Gt[idx];
            local = g * log(a) + (2.0 - g) * log(1.0 - a);
        }
    }
    sred[threadIdx.x] = local;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sred[threadIdx.x] += sred[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(ll + s, sred[0]);
}

__global__ void admix_squarem_norms_kernel(const double* __restrict__ t0,
                                           const double* __restrict__ t1,
                                           const double* __restrict__ t2, long len, long stride,
                                           double* __restrict__ rr, double* __restrict__ vv) {
    const long s = blockIdx.z;
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    __shared__ double sr[kBlk];
    __shared__ double sv[kBlk];
    double lr = 0.0, lv = 0.0;
    if (idx < len) {
        const long o = s * stride + idx;
        const double a = t1[o] - t0[o];                    // r  = theta1 - theta0
        const double b = t2[o] - 2.0 * t1[o] + t0[o];      // v  = theta2 - 2 theta1 + theta0
        lr = a * a;
        lv = b * b;
    }
    sr[threadIdx.x] = lr;
    sv[threadIdx.x] = lv;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            sr[threadIdx.x] += sr[threadIdx.x + off];
            sv[threadIdx.x] += sv[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(rr + s, sr[0]);
        atomicAdd(vv + s, sv[0]);
    }
}

__global__ void admix_squarem_combine_kernel(const double* __restrict__ t0,
                                             const double* __restrict__ t1,
                                             const double* __restrict__ t2, double* __restrict__ out,
                                             long len, long stride, const double* __restrict__ alpha,
                                             const double* __restrict__ active) {
    const long s = blockIdx.z;
    if (active && active[s] == 0.0) return;
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    const double a = alpha[s];
    const double c0 = (1.0 + a) * (1.0 + a);
    const double c1 = -2.0 * a * (1.0 + a);
    const double c2 = a * a;
    const long o = s * stride + idx;
    out[o] = c0 * t0[o] + c1 * t1[o] + c2 * t2[o];
}

__global__ void admix_project_f_kernel(double* __restrict__ F, double* __restrict__ Fc, long MK,
                                       long stride, double eps, const double* __restrict__ active) {
    const long s = blockIdx.z;
    if (active && active[s] == 0.0) return;
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= MK) return;
    const long o = s * stride + idx;
    double f = F[o];
    if (f < eps) f = eps;
    if (f > 1.0 - eps) f = 1.0 - eps;
    F[o] = f;
    Fc[o] = 1.0 - f;
}

__global__ void admix_project_q_kernel(double* __restrict__ Q, long N, int K, long stride_Q,
                                       const double* __restrict__ active) {
    const long s = blockIdx.z;
    if (active && active[s] == 0.0) return;
    const long i = blockIdx.x;
    if (i >= N) return;
    extern __shared__ double srow[];
    const long base = s * stride_Q;
    double qk = 0.0;
    if (threadIdx.x < static_cast<unsigned>(K)) {
        const long off = base + i + static_cast<long>(N) * threadIdx.x;
        qk = Q[off];
        if (qk < 0.0) qk = 0.0;  // clamp-to-nonneg retraction
        srow[threadIdx.x] = qk;
    }
    __syncthreads();
    __shared__ double rowsum;
    if (threadIdx.x == 0) {
        double sm = 0.0;
        for (int k = 0; k < K; ++k) sm += srow[k];
        rowsum = sm;
    }
    __syncthreads();
    if (threadIdx.x < static_cast<unsigned>(K)) {
        const long off = base + i + static_cast<long>(N) * threadIdx.x;
        Q[off] = (rowsum > 0.0) ? (qk / rowsum) : (1.0 / static_cast<double>(K));
    }
}

}  // namespace

void launch_admix_decode(const std::uint8_t* d_packed, std::size_t bytes_per_record, long N,
                         long M, double* d_G, double* d_V, cudaStream_t stream) {
    const long total = N * M;
    if (total <= 0) return;
    const int grid = static_cast<int>((total + kBlk - 1) / kBlk);
    admix_decode_kernel<<<grid, kBlk, 0, stream>>>(d_packed, bytes_per_record, N, M, d_G, d_V);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_decode_tile(const std::uint8_t* d_packed, std::size_t bytes_per_record, long N,
                              long s0, long t, double* d_Gt, double* d_Vt, cudaStream_t stream) {
    const long total = N * t;
    if (total <= 0) return;
    const int grid = static_cast<int>((total + kBlk - 1) / kBlk);
    admix_decode_tile_kernel<<<grid, kBlk, 0, stream>>>(d_packed, bytes_per_record, N, s0, t, d_Gt,
                                                        d_Vt);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_snp_mean(const double* d_G, const double* d_V, long N, long M, double* d_phat,
                           cudaStream_t stream) {
    if (M <= 0) return;
    admix_snp_mean_kernel<<<static_cast<int>(M), kBlk, 0, stream>>>(d_G, d_V, N, M, d_phat);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_responsibility(const double* d_Gtile, const double* d_Vtile, const double* d_A,
                                 long N, long t, double eps, double* d_R2, double* d_R1,
                                 cudaStream_t stream) {
    const long total = N * t;
    if (total <= 0) return;
    const int grid = static_cast<int>((total + kBlk - 1) / kBlk);
    admix_responsibility_kernel<<<grid, kBlk, 0, stream>>>(d_Gtile, d_Vtile, d_A, N, t, eps, d_R2,
                                                           d_R1);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_update_f(double* d_F, const double* d_S2, const double* d_S1, long M, int K,
                           double eps, cudaStream_t stream) {
    const long total = M * K;
    if (total <= 0) return;
    const int grid = static_cast<int>((total + kBlk - 1) / kBlk);
    admix_update_f_kernel<<<grid, kBlk, 0, stream>>>(d_F, d_S2, d_S1, M, K, eps);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_complement(const double* d_F, double* d_Fc, long M, int K, cudaStream_t stream) {
    const long total = M * K;
    if (total <= 0) return;
    const int grid = static_cast<int>((total + kBlk - 1) / kBlk);
    admix_complement_kernel<<<grid, kBlk, 0, stream>>>(d_F, d_Fc, total);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_update_q(double* d_Q, const double* d_T2, const double* d_T1, long N, int K,
                           cudaStream_t stream) {
    if (N <= 0 || K <= 0) return;
    // block dim = next pow2 >= K, at least 32; shared = K doubles for the row values.
    int bd = 32;
    while (bd < K) bd <<= 1;
    const std::size_t shmem = static_cast<std::size_t>(K) * sizeof(double);
    admix_update_q_kernel<<<static_cast<int>(N), bd, shmem, stream>>>(d_Q, d_T2, d_T1, N, K);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_loglik(const double* d_Gtile, const double* d_Vtile, const double* d_A, long N,
                         long t, double eps, double* d_ll, cudaStream_t stream) {
    const long total = N * t;
    if (total <= 0) return;
    const int grid = static_cast<int>((total + kBlk - 1) / kBlk);
    admix_loglik_kernel<<<grid, kBlk, 0, stream>>>(d_Gtile, d_Vtile, d_A, N, t, eps, d_ll);
    STEPPE_CUDA_CHECK_KERNEL();
}

// --- batched (S-seed) launchers -------------------------------------------------------------

void launch_admix_responsibility_b(const double* d_Gtile, const double* d_Vtile, const double* d_A,
                                   long N, long t, double eps, double* d_R2, double* d_R1, int S,
                                   long stride_tile, cudaStream_t stream) {
    const long total = N * t;
    if (total <= 0 || S <= 0) return;
    const dim3 grid(static_cast<unsigned>((total + kBlk - 1) / kBlk), 1u,
                    static_cast<unsigned>(S));
    admix_responsibility_b_kernel<<<grid, kBlk, 0, stream>>>(d_Gtile, d_Vtile, d_A, N, t, eps, d_R2,
                                                             d_R1, stride_tile);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_update_f_b(double* d_F, const double* d_S2, const double* d_S1, long M, int K,
                             double eps, int S, long stride_F, long stride_S,
                             const double* d_active, cudaStream_t stream) {
    const long total = M * K;
    if (total <= 0 || S <= 0) return;
    const dim3 grid(static_cast<unsigned>((total + kBlk - 1) / kBlk), 1u,
                    static_cast<unsigned>(S));
    admix_update_f_b_kernel<<<grid, kBlk, 0, stream>>>(d_F, d_S2, d_S1, M, K, eps, stride_F,
                                                       stride_S, d_active);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_complement_b(const double* d_F, double* d_Fc, long MK, int S, long stride,
                               const double* d_active, cudaStream_t stream) {
    if (MK <= 0 || S <= 0) return;
    const dim3 grid(static_cast<unsigned>((MK + kBlk - 1) / kBlk), 1u, static_cast<unsigned>(S));
    admix_complement_b_kernel<<<grid, kBlk, 0, stream>>>(d_F, d_Fc, MK, stride, d_active);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_update_q_b(double* d_Q, const double* d_T2, const double* d_T1, long N, int K,
                             int S, long stride_Q, const double* d_active, cudaStream_t stream) {
    if (N <= 0 || K <= 0 || S <= 0) return;
    int bd = 32;
    while (bd < K) bd <<= 1;
    const std::size_t shmem = static_cast<std::size_t>(K) * sizeof(double);
    const dim3 grid(static_cast<unsigned>(N), 1u, static_cast<unsigned>(S));
    admix_update_q_b_kernel<<<grid, bd, shmem, stream>>>(d_Q, d_T2, d_T1, N, K, stride_Q, d_active);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_loglik_b(const double* d_Gtile, const double* d_Vtile, const double* d_A, long N,
                           long t, double eps, double* d_ll, int S, long stride_tile,
                           cudaStream_t stream) {
    const long total = N * t;
    if (total <= 0 || S <= 0) return;
    const dim3 grid(static_cast<unsigned>((total + kBlk - 1) / kBlk), 1u,
                    static_cast<unsigned>(S));
    admix_loglik_b_kernel<<<grid, kBlk, 0, stream>>>(d_Gtile, d_Vtile, d_A, N, t, eps, d_ll,
                                                     stride_tile);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_squarem_norms(const double* d_t0, const double* d_t1, const double* d_t2,
                                long len, long stride, double* d_rr, double* d_vv, int S,
                                cudaStream_t stream) {
    if (len <= 0 || S <= 0) return;
    const dim3 grid(static_cast<unsigned>((len + kBlk - 1) / kBlk), 1u, static_cast<unsigned>(S));
    admix_squarem_norms_kernel<<<grid, kBlk, 0, stream>>>(d_t0, d_t1, d_t2, len, stride, d_rr, d_vv);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_squarem_combine(const double* d_t0, const double* d_t1, const double* d_t2,
                                  double* d_out, long len, long stride, const double* d_alpha,
                                  const double* d_active, int S, cudaStream_t stream) {
    if (len <= 0 || S <= 0) return;
    const dim3 grid(static_cast<unsigned>((len + kBlk - 1) / kBlk), 1u, static_cast<unsigned>(S));
    admix_squarem_combine_kernel<<<grid, kBlk, 0, stream>>>(d_t0, d_t1, d_t2, d_out, len, stride,
                                                            d_alpha, d_active);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_project_f(double* d_F, double* d_Fc, long MK, int S, long stride, double eps,
                            const double* d_active, cudaStream_t stream) {
    if (MK <= 0 || S <= 0) return;
    const dim3 grid(static_cast<unsigned>((MK + kBlk - 1) / kBlk), 1u, static_cast<unsigned>(S));
    admix_project_f_kernel<<<grid, kBlk, 0, stream>>>(d_F, d_Fc, MK, stride, eps, d_active);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_admix_project_q(double* d_Q, long N, int K, int S, long stride_Q,
                            const double* d_active, cudaStream_t stream) {
    if (N <= 0 || K <= 0 || S <= 0) return;
    int bd = 32;
    while (bd < K) bd <<= 1;
    const std::size_t shmem = static_cast<std::size_t>(K) * sizeof(double);
    const dim3 grid(static_cast<unsigned>(N), 1u, static_cast<unsigned>(S));
    admix_project_q_kernel<<<grid, bd, shmem, stream>>>(d_Q, N, K, stride_Q, d_active);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

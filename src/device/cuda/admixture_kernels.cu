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

__global__ void admix_decode_kernel(const std::uint8_t* __restrict__ packed,
                                    std::size_t bytes_per_record, long N, long M,
                                    double* __restrict__ G, double* __restrict__ V) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = N * M;
    if (idx >= total) return;
    const long i = idx % N;   // individual (column-major: i fastest)
    const long s = idx / N;   // SNP
    const std::size_t byte_in_rec =
        static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
    const int pos = static_cast<int>(s % core::kCodesPerByte);
    const std::uint8_t byte =
        packed[static_cast<std::size_t>(i) * bytes_per_record + byte_in_rec];
    const std::uint8_t code = core::genotype_code(byte, pos);
    const bool valid = core::genotype_valid(code);
    G[idx] = valid ? static_cast<double>(code) : 0.0;  // diploid: code == dosage 0/1/2
    V[idx] = valid ? 1.0 : 0.0;
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

}  // namespace

void launch_admix_decode(const std::uint8_t* d_packed, std::size_t bytes_per_record, long N,
                         long M, double* d_G, double* d_V, cudaStream_t stream) {
    const long total = N * M;
    if (total <= 0) return;
    const int grid = static_cast<int>((total + kBlk - 1) / kBlk);
    admix_decode_kernel<<<grid, kBlk, 0, stream>>>(d_packed, bytes_per_record, N, M, d_G, d_V);
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

}  // namespace steppe::device

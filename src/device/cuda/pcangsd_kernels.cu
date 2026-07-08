// src/device/cuda/pcangsd_kernels.cu
//
// PCAngsd (`steppe pcangsd`) native-FP64 elementwise kernels + host launchers. The
// SYRK gram / GEMM reconstruction / cuSOLVER eigen live in cuda_backend_pcangsd.cu;
// this TU is the E-build, emMAF, dCov, reconstruction-diff, cov-finalize, and pi
// kernels. The scalar posterior arithmetic mirrors core::pcangsd_reference exactly
// (pcangsd_em.hpp), so the GPU and reference paths cannot drift on the EM math.
#include "device/cuda/pcangsd_kernels.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/launch_config.hpp"  // grid_for_x, kDecodeBlockX/Y
#include "device/cuda/check.cuh"

namespace steppe::device {

using core::kDecodeBlockX;
using core::kDecodeBlockY;

namespace {

constexpr int kBlock = kDecodeBlockX * kDecodeBlockY;  // 256 threads / block

// Expected allele-2 dosage under the HWE prior + the posterior weights (device twin
// of pcangsd_detail::posterior). Lrr/Lhet/Laa are P(0/1/2 copies of A2).
__device__ __forceinline__ double edosage(double Lrr, double Lhet, double Laa, double q,
                                          double& p0, double& p1, double& p2, double& psum) {
    const double omq = 1.0 - q;
    p0 = Lrr * omq * omq;
    p1 = Lhet * 2.0 * q * omq;
    p2 = Laa * q * q;
    psum = p0 + p1 + p2;
    return (psum > 0.0) ? (p1 + 2.0 * p2) / psum : 2.0 * q;
}

__global__ void emmaf_kernel(const double* __restrict__ l, long n_site, int N, int maf_iter,
                             double maf_tol, double* __restrict__ f_all) {
    const long s = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= n_site) return;
    const double invN2 = 1.0 / (2.0 * static_cast<double>(N));
    const std::size_t row = static_cast<std::size_t>(s) * static_cast<std::size_t>(N) * 3;
    double fj = 0.25;
    for (int it = 0; it < maf_iter; ++it) {
        double acc = 0.0;
        for (int i = 0; i < N; ++i) {
            const std::size_t b = row + static_cast<std::size_t>(i) * 3;
            double p0, p1, p2, ps;
            acc += edosage(l[b + 2], l[b + 1], l[b + 0], fj, p0, p1, p2, ps);
        }
        const double fnew = acc * invN2;
        const double d = fnew - fj;
        fj = fnew;
        if (d * d < maf_tol * maf_tol) break;  // rmse1d over a scalar < maf_tol
    }
    f_all[s] = fj;
}

// One thread per (individual i, kept-site jj): write dE[i + jj*N] (column-major).
__global__ void build_E_kernel(const double* __restrict__ l, const int* __restrict__ kept,
                               const double* __restrict__ fk, const double* __restrict__ P,
                               long Nl, long Mw, int N, bool standardize,
                               double* __restrict__ E) {
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = static_cast<long>(N) * Mw;
    if (t >= total) return;
    const int i = static_cast<int>(t % static_cast<long>(N));
    const long jj = t / static_cast<long>(N);
    const double fj = fk[jj];
    const long site = static_cast<long>(kept[jj]);
    const std::size_t b = (static_cast<std::size_t>(site) * static_cast<std::size_t>(Nl) +
                           static_cast<std::size_t>(i)) * 3;
    double q = fj;
    if (P != nullptr) {
        q = (P[static_cast<std::size_t>(t)] + 2.0 * fj) * 0.5;
        q = fmin(fmax(q, 1e-4), 1.0 - 1e-4);
    }
    double p0, p1, p2, ps;
    const double ed = edosage(l[b + 2], l[b + 1], l[b + 0], q, p0, p1, p2, ps);
    double e = ed - 2.0 * fj;  // centered by the POPULATION freq f_j
    if (standardize) e *= 1.0 / sqrt(2.0 * fj * (1.0 - fj));
    E[static_cast<std::size_t>(t)] = e;
}

// One thread per individual: reduce dCov[i] over the kept sites.
__global__ void dcov_kernel(const double* __restrict__ l, const int* __restrict__ kept,
                            const double* __restrict__ fk, const double* __restrict__ P, long Nl,
                            long Mw, int N, double* __restrict__ dcov) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double acc = 0.0;
    for (long jj = 0; jj < Mw; ++jj) {
        const double fj = fk[jj];
        const double denom = 2.0 * fj * (1.0 - fj);
        const std::size_t idx = static_cast<std::size_t>(jj) * static_cast<std::size_t>(N) +
                                static_cast<std::size_t>(i);
        double q = (P[idx] + 2.0 * fj) * 0.5;
        q = fmin(fmax(q, 1e-4), 1.0 - 1e-4);
        const long site = static_cast<long>(kept[jj]);
        const std::size_t b = (static_cast<std::size_t>(site) * static_cast<std::size_t>(Nl) +
                               static_cast<std::size_t>(i)) * 3;
        double p0, p1, p2, ps;
        (void)edosage(l[b + 2], l[b + 1], l[b + 0], q, p0, p1, p2, ps);
        if (ps > 0.0) {
            const double t0 = (-2.0 * fj) * (-2.0 * fj) * (p0 / ps);
            const double t1 = (1.0 - 2.0 * fj) * (1.0 - 2.0 * fj) * (p1 / ps);
            const double t2 = (2.0 - 2.0 * fj) * (2.0 - 2.0 * fj) * (p2 / ps);
            acc += (t0 + t1 + t2) / denom;
        }
    }
    dcov[i] = acc;
}

__global__ void sqdiff_kernel(const double* __restrict__ P, const double* __restrict__ Pprev,
                              long n, double* __restrict__ acc) {
    extern __shared__ double sdata[];
    const unsigned tid = threadIdx.x;
    double local = 0.0;
    for (long k = static_cast<long>(blockIdx.x) * blockDim.x + tid; k < n;
         k += static_cast<long>(gridDim.x) * blockDim.x) {
        const double d = P[k] - Pprev[k];
        local += d * d;
    }
    sdata[tid] = local;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(acc, sdata[0]);
}

__global__ void finalize_cov_kernel(double* __restrict__ C, const double* __restrict__ dcov, int N,
                                    double inv_M) {
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = static_cast<long>(N) * static_cast<long>(N);
    if (t >= total) return;
    const int a = static_cast<int>(t % static_cast<long>(N));
    const int b = static_cast<int>(t / static_cast<long>(N));
    C[static_cast<std::size_t>(t)] = (a == b) ? dcov[a] * inv_M : C[static_cast<std::size_t>(t)] * inv_M;
}

__global__ void pi_kernel(const double* __restrict__ P, const double* __restrict__ fk, long Mw,
                          int N, double* __restrict__ pi) {
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = static_cast<long>(N) * Mw;
    if (t >= total) return;
    const int i = static_cast<int>(t % static_cast<long>(N));
    const long jj = t / static_cast<long>(N);
    const double fj = fk[jj];
    double q = (P[static_cast<std::size_t>(t)] + 2.0 * fj) * 0.5;
    q = fmin(fmax(q, 1e-4), 1.0 - 1e-4);
    pi[static_cast<std::size_t>(jj) * static_cast<std::size_t>(N) + static_cast<std::size_t>(i)] = q;
}

}  // namespace

void launch_pcangsd_emmaf(const double* d_l, long n_site, int n_sample, int maf_iter,
                          double maf_tol, double* d_f_all, cudaStream_t stream) {
    if (n_site <= 0 || n_sample <= 0) return;
    const int grid =
        core::grid_for_x(n_site, kBlock, "pcangsd emMAF gridDim.x (site axis) exceeds kMaxGridX");
    emmaf_kernel<<<static_cast<unsigned>(grid), kBlock, 0, stream>>>(d_l, n_site, n_sample,
                                                                     maf_iter, maf_tol, d_f_all);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pcangsd_build_E(const double* d_l, const int* d_kept, const double* d_fk,
                            const double* d_P, long n_sample_l, long Mw, int n_sample,
                            bool standardize, double* d_E, cudaStream_t stream) {
    if (Mw <= 0 || n_sample <= 0) return;
    const long total = static_cast<long>(n_sample) * Mw;
    const int grid =
        core::grid_for_x(total, kBlock, "pcangsd build_E gridDim.x (N*Mw axis) exceeds kMaxGridX");
    build_E_kernel<<<static_cast<unsigned>(grid), kBlock, 0, stream>>>(
        d_l, d_kept, d_fk, d_P, n_sample_l, Mw, n_sample, standardize, d_E);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pcangsd_dcov(const double* d_l, const int* d_kept, const double* d_fk,
                         const double* d_P, long n_sample_l, long Mw, int n_sample,
                         double* d_dcov, cudaStream_t stream) {
    if (Mw <= 0 || n_sample <= 0) return;
    const int grid = core::grid_for_x(n_sample, kBlock,
                                      "pcangsd dCov gridDim.x (individual axis) exceeds kMaxGridX");
    dcov_kernel<<<static_cast<unsigned>(grid), kBlock, 0, stream>>>(d_l, d_kept, d_fk, d_P,
                                                                    n_sample_l, Mw, n_sample,
                                                                    d_dcov);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pcangsd_sqdiff(const double* d_P, const double* d_Pprev, long n, double* d_acc,
                           cudaStream_t stream) {
    if (n <= 0) return;
    long blocks_l = (n + kBlock - 1) / kBlock;
    if (blocks_l > 65535) blocks_l = 65535;
    sqdiff_kernel<<<static_cast<unsigned>(blocks_l), kBlock, kBlock * sizeof(double), stream>>>(
        d_P, d_Pprev, n, d_acc);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pcangsd_finalize_cov(double* d_C, const double* d_dcov, int N, double inv_M,
                                 cudaStream_t stream) {
    if (N <= 0) return;
    const long total = static_cast<long>(N) * static_cast<long>(N);
    const int grid = core::grid_for_x(total, kBlock,
                                      "pcangsd finalize_cov gridDim.x (N*N axis) exceeds kMaxGridX");
    finalize_cov_kernel<<<static_cast<unsigned>(grid), kBlock, 0, stream>>>(d_C, d_dcov, N, inv_M);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pcangsd_pi(const double* d_P, const double* d_fk, long Mw, int n_sample, double* d_pi,
                       cudaStream_t stream) {
    if (Mw <= 0 || n_sample <= 0) return;
    const long total = static_cast<long>(n_sample) * Mw;
    const int grid =
        core::grid_for_x(total, kBlock, "pcangsd pi gridDim.x (N*Mw axis) exceeds kMaxGridX");
    pi_kernel<<<static_cast<unsigned>(grid), kBlock, 0, stream>>>(d_P, d_fk, Mw, n_sample, d_pi);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

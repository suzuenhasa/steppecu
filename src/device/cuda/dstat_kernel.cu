// src/device/cuda/dstat_kernel.cu
//
// The GPU genotype-path normalized-D / qpfstats-f4 per-SNP block reduction: two kernels
// computing identical per-(quadruple, block) num/den/count partial sums — a tiled
// shared-memory hot path and a legacy per-cell fallback — behind one launch wrapper.
//
// Reference: docs/reference/src_device_cuda_dstat_kernel.cu.md
#include <cuda_runtime.h>

#include <cstddef>

#include "core/internal/launch_config.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/dstat_kernel.cuh"

namespace steppe::device {

namespace {

// Pairwise upper-triangle index encoding — reference §3
__device__ __forceinline__ int pair_index_lo_hi(int lo, int hi, int P) {
    return lo * P - (lo * (lo + 1)) / 2 + (hi - lo - 1);
}

// Legacy per-cell reduction (cold fallback) — reference §5
__global__ void dstat_block_reduce_legacy_kernel(
    const double* __restrict__ Q, const double* __restrict__ V, int P, long M,
    const int* __restrict__ quad, int N, const int* __restrict__ block_begin,
    const int* __restrict__ block_size, int n_block, double* __restrict__ numsum,
    double* __restrict__ densum, double* __restrict__ cnt) {
    (void)M;
    const long cell = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    const long total = static_cast<long>(N) * static_cast<long>(n_block);
    if (cell >= total) return;

    const int k = static_cast<int>(cell / n_block);
    const int b = static_cast<int>(cell % n_block);

    const int p1 = quad[4 * k + 0];
    const int p2 = quad[4 * k + 1];
    const int p3 = quad[4 * k + 2];
    const int p4 = quad[4 * k + 3];

    const int s0 = block_begin[b];
    const int sz = block_size[b];

    double nsum = 0.0;
    double dsum = 0.0;
    double c = 0.0;
    for (int s = s0; s < s0 + sz; ++s) {
        const long col = static_cast<long>(P) * static_cast<long>(s);
        if (V[col + p1] == 0.0 || V[col + p2] == 0.0 ||
            V[col + p3] == 0.0 || V[col + p4] == 0.0) {
            continue;
        }
        const double a = Q[col + p1];
        const double bb = Q[col + p2];
        const double cc = Q[col + p3];
        const double dd = Q[col + p4];
        nsum += (a - bb) * (cc - dd);
        dsum += (a + bb - 2.0 * a * bb) * (cc + dd - 2.0 * cc * dd);
        c += 1.0;
    }

    const long out = static_cast<long>(k) * static_cast<long>(n_block) + b;
    numsum[out] = nsum;
    densum[out] = dsum;
    cnt[out] = c;
}

// Tiled pairwise-difference-reuse reduction (hot path) — reference §4
__global__ void dstat_block_reduce_tiled_kernel(
    const double* __restrict__ Q, const double* __restrict__ V, int P, long M,
    const int* __restrict__ quad, int N, const int* __restrict__ block_begin,
    const int* __restrict__ block_size, int n_block, double* __restrict__ numsum,
    double* __restrict__ densum, double* __restrict__ cnt) {
    (void)M;

    const int b = static_cast<int>(blockIdx.x);
    const long k = blockIdx.y * static_cast<long>(blockDim.x) + threadIdx.x;

    const int npairs = P * (P - 1) / 2;

    extern __shared__ double s_mem[];
    double* Qsh = s_mem;
    double* Vsh = s_mem + P;
    double* diff = s_mem + 2 * P;
    double* het = diff + npairs;

    const bool active = (k < static_cast<long>(N));
    int p1 = 0, p2 = 0, p3 = 0, p4 = 0;
    if (active) {
        p1 = quad[4 * k + 0];
        p2 = quad[4 * k + 1];
        p3 = quad[4 * k + 2];
        p4 = quad[4 * k + 3];
    }

    const int s0 = block_begin[b];
    const int sz = block_size[b];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    double nsum = 0.0;
    double dsum = 0.0;
    double c = 0.0;

    for (int s = s0; s < s0 + sz; ++s) {
        const long col = static_cast<long>(P) * static_cast<long>(s);

        for (int i = tid; i < P; i += nthreads) {
            Qsh[i] = Q[col + i];
            Vsh[i] = V[col + i];
        }
        __syncthreads();

        for (int idx = tid; idx < npairs; idx += nthreads) {
            int i = 0;
            int rem = idx;
            int row = P - 1;
            while (rem >= row) { rem -= row; ++i; --row; }
            const int j = i + 1 + rem;
            const double qi = Qsh[i];
            const double qj = Qsh[j];
            diff[idx] = qi - qj;
            het[idx] = qi + qj - 2.0 * qi * qj;
        }
        __syncthreads();

        if (active) {
            if (Vsh[p1] != 0.0 && Vsh[p2] != 0.0 && Vsh[p3] != 0.0 && Vsh[p4] != 0.0) {
                double f1;
                if (p1 < p2)       f1 = diff[pair_index_lo_hi(p1, p2, P)];
                else if (p1 > p2)  f1 = -diff[pair_index_lo_hi(p2, p1, P)];
                else               f1 = 0.0;
                double f2;
                if (p3 < p4)       f2 = diff[pair_index_lo_hi(p3, p4, P)];
                else if (p3 > p4)  f2 = -diff[pair_index_lo_hi(p4, p3, P)];
                else               f2 = 0.0;
                nsum += f1 * f2;
                double h1;
                if (p1 != p2) h1 = het[pair_index_lo_hi(min(p1, p2), max(p1, p2), P)];
                else { const double q = Qsh[p1]; h1 = q + q - 2.0 * q * q; }
                double h2;
                if (p3 != p4) h2 = het[pair_index_lo_hi(min(p3, p4), max(p3, p4), P)];
                else { const double q = Qsh[p3]; h2 = q + q - 2.0 * q * q; }
                dsum += h1 * h2;
                c += 1.0;
            }
        }
        __syncthreads();
    }

    if (active) {
        const long out = k * static_cast<long>(n_block) + b;
        numsum[out] = nsum;
        densum[out] = dsum;
        cnt[out] = c;
    }
}

}  // namespace

// Path choice + shared-memory budget + capped grid geometry — reference §6, §7
void launch_dstat_block_reduce(const double* d_Q, const double* d_V, int P, long M,
                               const int* d_quad, int N,
                               const int* d_block_begin, const int* d_block_size,
                               int n_block,
                               double* d_numsum, double* d_densum, double* d_cnt,
                               cudaStream_t stream) {
    const long total = static_cast<long>(N) * static_cast<long>(n_block);
    if (total <= 0) return;
    constexpr int kThreads = 256;

    const long npairs = static_cast<long>(P) * (P - 1) / 2;
    const std::size_t smem = static_cast<std::size_t>(2L * P + 2L * npairs) * sizeof(double);

    constexpr std::size_t kDefaultSmem = 48u * 1024u;
    constexpr std::size_t kOptinSmem = 99u * 1024u;

    if (P >= 2 && smem <= kOptinSmem) {
        if (smem > kDefaultSmem) {
            STEPPE_CUDA_CHECK(cudaFuncSetAttribute(
                dstat_block_reduce_tiled_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem)));
        }
        const unsigned tilesY = static_cast<unsigned>(core::grid_for(N, kThreads));
        const dim3 grid(static_cast<unsigned>(n_block), tilesY, 1);
        dstat_block_reduce_tiled_kernel<<<grid, kThreads, smem, stream>>>(
            d_Q, d_V, P, M, d_quad, N, d_block_begin, d_block_size, n_block,
            d_numsum, d_densum, d_cnt);
        STEPPE_CUDA_CHECK_KERNEL();
        return;
    }

    const long blocks = (total + kThreads - 1) / kThreads;
    dstat_block_reduce_legacy_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        d_Q, d_V, P, M, d_quad, N, d_block_begin, d_block_size, n_block,
        d_numsum, d_densum, d_cnt);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

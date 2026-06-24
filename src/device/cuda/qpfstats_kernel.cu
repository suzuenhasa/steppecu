// src/device/cuda/qpfstats_kernel.cu
//
// The qpfstats smoothing-solve PREP kernels (the genotype-path joint f2 smoother;
// include/steppe/qpfstats.hpp). NOT the f2 GEMM, NOT the f2 cache: two tiny device
// kernels that prepare the shared-factor batched least-squares the backend's
// qpfstats_smooth drives:
//   (1) zero the non-finite ymat entries IN PLACE + count NaN comb-rows per block
//       (the AT2 ymat_chunk[nan]=0 + k_i = sum(nan_i)). A zeroed RHS column yields
//       the AT2 all-NaN→b=0 result through the SAME shared Dtrsm pair (A·b=0 ⇒ b=0).
//   (2) add the ridge constant to the diagonal (A_shared = x'x + ridge·I).
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). The kernel bodies +
// <<<>>> live ONLY here; the backend reaches them through the narrow launch wrapper
// (qpfstats_kernel.cuh), never includes this body (architecture.md §7).
#include <cuda_runtime.h>

#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK_KERNEL
#include "device/cuda/qpfstats_kernel.cuh"  // the narrow seam

namespace steppe::device {

namespace {

/// One thread per (comb, block) cell: zero the non-finite ymat entries in place and
/// atomically count the NaN comb-rows per block. ymat is COLUMN-MAJOR [npopcomb × n_block]
/// (cell (c,b) at c + npopcomb*b). d_nan_per_block must be pre-zeroed.
__global__ void qpfstats_zero_nan_ymat_kernel(double* __restrict__ ymat, int npopcomb,
                                              int n_block, int* __restrict__ nan_per_block) {
    const long cell = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    const long total = static_cast<long>(npopcomb) * static_cast<long>(n_block);
    if (cell >= total) return;
    const int b = static_cast<int>(cell / npopcomb);
    const double v = ymat[cell];
    if (!isfinite(v)) {
        ymat[cell] = 0.0;
        atomicAdd(&nan_per_block[b], 1);
    }
}

/// One thread per diagonal entry: A[i + n*i] += ridge.
__global__ void qpfstats_add_ridge_diag_kernel(double* __restrict__ A, int n, double ridge) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    A[static_cast<long>(i) + static_cast<long>(n) * static_cast<long>(i)] += ridge;
}

}  // namespace

void launch_qpfstats_zero_nan_ymat(double* d_ymat, int npopcomb, int n_block,
                                   int* d_nan_per_block, cudaStream_t stream) {
    const long total = static_cast<long>(npopcomb) * static_cast<long>(n_block);
    if (total <= 0) return;
    STEPPE_CUDA_CHECK(cudaMemsetAsync(d_nan_per_block, 0,
                                      static_cast<std::size_t>(n_block) * sizeof(int), stream));
    constexpr int kThreads = 256;
    const long blocks = (total + kThreads - 1) / kThreads;
    qpfstats_zero_nan_ymat_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        d_ymat, npopcomb, n_block, d_nan_per_block);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpfstats_add_ridge_diag(double* d_A, int n, double ridge, cudaStream_t stream) {
    if (n <= 0) return;
    constexpr int kThreads = 64;
    const int blocks = (n + kThreads - 1) / kThreads;
    qpfstats_add_ridge_diag_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        d_A, n, ridge);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

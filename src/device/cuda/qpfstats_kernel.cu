// src/device/cuda/qpfstats_kernel.cu
//
// Two tiny prep kernels for the qpfstats smoothing solve: zero the non-finite
// ymat entries (counting NaN rows per block) and add the ridge to the coefficient
// diagonal. CUDA TU private to steppe_device; the backend reaches the kernels only
// through the narrow launch wrappers in qpfstats_kernel.cuh.
//
// Reference: docs/reference/src_device_cuda_qpfstats_kernel.cu.md
#include <cuda_runtime.h>

#include "core/internal/launch_config.hpp"
#include "core/internal/nvtx.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/qpfstats_kernel.cuh"

namespace steppe::device {

namespace {

// NaN-zeroing kernel — reference §3
__global__ void qpfstats_zero_nan_ymat_kernel(double* __restrict__ ymat, int npopcomb,
                                              int n_block, int* __restrict__ nan_per_block) {
    const long total = static_cast<long>(npopcomb) * static_cast<long>(n_block);
    for (long cell = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; cell < total;
         cell += static_cast<long>(gridDim.x) * blockDim.x) {
        const int b = static_cast<int>(cell / npopcomb);
        const double v = ymat[cell];
        if (!isfinite(v)) {
            ymat[cell] = 0.0;
            atomicAdd(&nan_per_block[b], 1);
        }
    }
}

// Ridge-diagonal kernel — reference §5
__global__ void qpfstats_add_ridge_diag_kernel(double* __restrict__ A, int n, double ridge) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    A[static_cast<long>(i) + static_cast<long>(n) * static_cast<long>(i)] += ridge;
}

}  // namespace

// Launch wrappers — reference §6
void launch_qpfstats_zero_nan_ymat(double* d_ymat, int npopcomb, int n_block,
                                   int* d_nan_per_block, cudaStream_t stream) {
    const long total = static_cast<long>(npopcomb) * static_cast<long>(n_block);
    if (total <= 0) return;
    STEPPE_NVTX_RANGE("qpfstats_smooth_prep");
    STEPPE_CUDA_CHECK(cudaMemsetAsync(d_nan_per_block, 0,
                                      static_cast<std::size_t>(n_block) * sizeof(int), stream));
    constexpr int kZeroNanThreads = 256;
    const long blocks = core::cdiv(total, static_cast<long>(kZeroNanThreads));
    qpfstats_zero_nan_ymat_kernel<<<static_cast<unsigned>(blocks), kZeroNanThreads, 0, stream>>>(
        d_ymat, npopcomb, n_block, d_nan_per_block);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpfstats_add_ridge_diag(double* d_A, int n, double ridge, cudaStream_t stream) {
    if (n <= 0) return;
    constexpr int kRidgeThreads = 64;
    const int blocks = core::cdiv(n, kRidgeThreads);
    qpfstats_add_ridge_diag_kernel<<<static_cast<unsigned>(blocks), kRidgeThreads, 0, stream>>>(
        d_A, n, ridge);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

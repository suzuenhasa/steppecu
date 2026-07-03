// src/device/cuda/qpfstats_kernel.cuh — narrow launch seam for the two qpfstats
// smoothing-solve prep kernels; the CUDA-private bodies live in qpfstats_kernel.cu.
// Reference: docs/reference/src_device_cuda_qpfstats_kernel.cuh.md
#ifndef STEPPE_DEVICE_CUDA_QPFSTATS_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_QPFSTATS_KERNEL_CUH

#include <cuda_runtime.h>

namespace steppe::device {

// Zero non-finite ymat entries + per-block count — reference §2
void launch_qpfstats_zero_nan_ymat(double* d_ymat, int npopcomb, int n_block,
                                   int* d_nan_per_block, cudaStream_t stream);

// Add ridge to the diagonal — reference §3
void launch_qpfstats_add_ridge_diag(double* d_A, int n, double ridge, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_QPFSTATS_KERNEL_CUH

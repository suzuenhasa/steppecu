// src/device/cuda/dstat_kernel.cuh
//
// Declares the host-callable launch wrapper for the GPU normalized-D per-SNP
// block reduction (qpDstat Part B). Names a CUDA type (cudaStream_t), so it is
// PRIVATE to steppe_device — the seam between the backend and the D kernel TU,
// not the CUDA-free public seam (include/steppe/dstat.hpp).
//
// Reference: docs/reference/src_device_cuda_dstat_kernel.cuh.md
#ifndef STEPPE_DEVICE_CUDA_DSTAT_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_DSTAT_KERNEL_CUH

#include <cuda_runtime.h>

namespace steppe::device {

// Launch wrapper: normalized-D per-SNP block reduction — reference §6
void launch_dstat_block_reduce(const double* d_Q, const double* d_V, int P, long M,
                               const int* d_quad, int N,
                               const int* d_block_begin, const int* d_block_size,
                               int n_block,
                               double* d_numsum, double* d_densum, double* d_cnt,
                               cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_DSTAT_KERNEL_CUH

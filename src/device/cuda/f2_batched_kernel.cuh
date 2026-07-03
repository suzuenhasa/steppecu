// src/device/cuda/f2_batched_kernel.cuh
//
// Launch-wrapper declarations for the batched, size-grouped f2 path — one gather,
// one GEMM, and one assemble wrapper per block size-group. Names a cuBLAS handle
// type, so it is private to the device layer (never on the CUDA-free public seam).
//
// Reference: docs/reference/src_device_cuda_f2_batched_kernel.cuh.md
#ifndef STEPPE_DEVICE_CUDA_F2_BATCHED_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_F2_BATCHED_KERNEL_CUH

#include <cublas_v2.h>

#include "steppe/config.hpp"

namespace steppe::device {

// Gather a size-group into padded slabs — reference §4
void launch_gather_group(const double* dQ_all, const double* dV_all, const double* dS_all,
                         const int* d_block_ids_in_group, const long* d_block_offsets,
                         const int* d_block_sizes,
                         int P, int s_pad, int n_in_group,
                         double* dQg, double* dVg, double* dSg,
                         cudaStream_t stream);

// The three f2 GEMMs for one size-group — reference §5
void run_f2_gemms_group(cublasHandle_t handle, const Precision& precision,
                        int P, int s_pad, int n_in_group,
                        const double* dQg, const double* dVg, const double* dSg,
                        double* dGg, double* dVpairg, double* dRg);

// Assemble + scatter a size-group into resident tensors — reference §6
void launch_assemble_blocks_group(const double* dGg, const double* dVpairg, const double* dRg,
                                  const int* d_block_ids_in_group,
                                  int P, int n_in_group,
                                  double* dF2_all, double* dVpair_all,
                                  cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_F2_BATCHED_KERNEL_CUH

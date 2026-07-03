// src/device/cuda/f2_block_kernel.cuh
//
// Launch-wrapper declarations for the f2 3-GEMM pipeline over one SNP tile —
// declarations only (kernel bodies and <<<>>> live in the .cu). Names CUDA
// types (cublasHandle_t), so this header is private to steppe_device.
//
// Reference: docs/reference/src_device_cuda_f2_block_kernel.cuh.md
#ifndef STEPPE_DEVICE_CUDA_F2_BLOCK_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_F2_BLOCK_KERNEL_CUH

#include <cublas_v2.h>

#include "steppe/config.hpp"

namespace steppe::device {

// Feeder pre-pass — reference §4
void launch_f2_feeder(const double* dQ_raw, const double* dV_raw, const double* dN_raw,
                      double* dQ_masked, double* dV_out, double* dS,
                      int P, long M, cudaStream_t stream);

// f2 GEMM precision policy — reference §5
[[nodiscard]] bool emulation_honorable(const Precision& precision) noexcept;

void engage_f2_precision(cublasHandle_t handle, const Precision& precision);

[[nodiscard]] cublasComputeType_t f2_compute_type(const Precision& precision);

// The three f2 GEMMs — reference §6
void run_f2_gemms(cublasHandle_t handle, const Precision& precision,
                  int P, long M,
                  const double* dQ, const double* dV, const double* dS,
                  double* dG, double* dVpair, double* dR);

// Assemble final f2 values — reference §7
void launch_assemble_f2(const double* dG, const double* dVpair, const double* dR,
                        double* dF2, int P, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_F2_BLOCK_KERNEL_CUH

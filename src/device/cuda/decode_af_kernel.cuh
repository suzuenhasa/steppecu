// src/device/cuda/decode_af_kernel.cuh
//
// Declares launch_decode_af, the host-callable wrapper for the GPU genotype
// decode → allele-freq reduction. The kernel body and <<<>>> live only in
// decode_af_kernel.cu; this header names cudaStream_t and so is private to the
// device layer.
//
// Reference: docs/reference/src_device_cuda_decode_af_kernel.cuh.md
#ifndef STEPPE_DEVICE_CUDA_DECODE_AF_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_DECODE_AF_KERNEL_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// GPU genotype decode → allele-freq reduction — reference §2
void launch_decode_af(const std::uint8_t* d_packed,
                      std::size_t bytes_per_record,
                      const std::size_t* d_pop_offsets,
                      int P, long M, int ploidy,
                      const int* d_sample_ploidy,
                      double* d_Q, double* d_V, double* d_N,
                      cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_DECODE_AF_KERNEL_CUH

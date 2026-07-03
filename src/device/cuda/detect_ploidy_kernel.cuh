// src/device/cuda/detect_ploidy_kernel.cuh
//
// Declares the GPU launch wrapper for the pseudo-haploid per-sample ploidy
// prepass. Names cudaStream_t, so it is private to the steppe_device library
// (not the CUDA-free public backend seam); the kernel body lives in the .cu file.
//
// Reference: docs/reference/src_device_cuda_detect_ploidy_kernel.cuh.md
#ifndef STEPPE_DEVICE_CUDA_DETECT_PLOIDY_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_DETECT_PLOIDY_KERNEL_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// GPU ploidy-detection launch contract — reference §3
void launch_detect_ploidy(const std::uint8_t* d_packed,
                          std::size_t bytes_per_record,
                          std::size_t n_individuals, std::size_t n_snp,
                          int* d_ploidy, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_DETECT_PLOIDY_KERNEL_CUH

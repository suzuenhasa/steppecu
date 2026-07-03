// src/device/cuda/transpose_canonical_kernel.cuh
//
// Launch-wrapper declaration for the SNP-major -> canonical individual-major
// transpose+gather+encode primitive (the format-reader engine). Names cudaStream_t,
// so it is device-private: the kernel body and <<<>>> live in the paired .cu.
//
// Reference: docs/reference/src_device_cuda_transpose_canonical_kernel.cuh.md
#ifndef STEPPE_DEVICE_CUDA_TRANSPOSE_CANONICAL_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_TRANSPOSE_CANONICAL_KERNEL_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// Source 2-bit code -> canonical encoding map — reference §4
enum class TransposeEncoding : int {
    Identity = 0,
};

// SNP-major -> canonical transpose+gather+encode launch — reference §5
void launch_transpose_to_canonical(const std::uint8_t* d_snp_major,
                                   std::size_t src_bytes_per_record,
                                   const std::size_t* d_sel_rows,
                                   std::size_t n_individuals, std::size_t n_snp,
                                   std::size_t out_bytes_per_record,
                                   TransposeEncoding encoding,
                                   std::uint8_t* d_out, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_TRANSPOSE_CANONICAL_KERNEL_CUH

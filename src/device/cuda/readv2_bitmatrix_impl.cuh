// src/device/cuda/readv2_bitmatrix_impl.cuh
//
// The CUDA-side definition of Readv2Bitmatrix::Impl — the DeviceBuffer<Readv2Word>
// owner. Shared by readv2_bitmatrix.cu (the special members) and
// cuda_backend_readv2.cu (which allocates/packs/reduces it), mirroring
// device_decode_result_impl.cuh. Private to steppe_device.
//
// Reference: docs/reference/src_device_cuda_readv2_bitmatrix_impl.cuh.md
#ifndef STEPPE_DEVICE_CUDA_READV2_BITMATRIX_IMPL_CUH
#define STEPPE_DEVICE_CUDA_READV2_BITMATRIX_IMPL_CUH

#include "device/readv2_bitmatrix.hpp"
#include "device/cuda/device_buffer.cuh"
#include "device/cuda/readv2_layout.cuh"

namespace steppe::device {

struct Readv2Bitmatrix::Impl {
    DeviceBuffer<Readv2Word> words;  // [n_samples * words_per_sample], zeroed at alloc
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_READV2_BITMATRIX_IMPL_CUH

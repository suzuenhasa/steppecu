// src/device/cuda/device_partial_impl.cuh — the CUDA-private header holding the
// single definition of struct DevicePartial::Impl: the resident f2/vpair GPU
// buffers. Shared by device_partial.cu and every CUDA consumer so the layout
// never drifts.
//
// Reference: docs/reference/src_device_cuda_device_partial_impl.cuh.md
#ifndef STEPPE_DEVICE_CUDA_DEVICE_PARTIAL_IMPL_CUH
#define STEPPE_DEVICE_CUDA_DEVICE_PARTIAL_IMPL_CUH

#include "device/device_partial.hpp"
#include "device/cuda/device_buffer.cuh"

namespace steppe::device {

// DevicePartial::Impl: resident f2/vpair device buffers — reference §2
struct DevicePartial::Impl {
    DeviceBuffer<double> f2;
    DeviceBuffer<double> vpair;
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_DEVICE_PARTIAL_IMPL_CUH

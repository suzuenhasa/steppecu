// src/device/cuda/device_decode_result_impl.cuh
//
// The CUDA half of the DeviceDecodeResult handle/Impl split: the one struct that
// names the real GPU buffer type, so this is the only file in the pair the CUDA
// toolchain must compile. Private to steppe_device.
//
// Reference: docs/reference/src_device_cuda_device_decode_result_impl.cuh.md
#ifndef STEPPE_DEVICE_CUDA_DEVICE_DECODE_RESULT_IMPL_CUH
#define STEPPE_DEVICE_CUDA_DEVICE_DECODE_RESULT_IMPL_CUH

#include "device/device_decode_result.hpp"
#include "device/cuda/device_buffer.cuh"

namespace steppe::device {

// The three resident device buffers (n is regime-B only) — reference §2
struct DeviceDecodeResult::Impl {
    DeviceBuffer<double> q;
    DeviceBuffer<double> v;
    DeviceBuffer<double> n;
};

}  // namespace steppe::device
#endif  // STEPPE_DEVICE_CUDA_DEVICE_DECODE_RESULT_IMPL_CUH

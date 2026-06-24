// src/device/cuda/device_decode_result_impl.cuh — the CUDA side of
// DeviceDecodeResult: the Impl holding the resident DeviceBuffer<double> q/v owners.
// PRIVATE to steppe_device. Shared by cuda/device_decode_result.cu (special members +
// q_device/v_device) and cuda_backend.cu (builds the result device-resident).
#ifndef STEPPE_DEVICE_CUDA_DEVICE_DECODE_RESULT_IMPL_CUH
#define STEPPE_DEVICE_CUDA_DEVICE_DECODE_RESULT_IMPL_CUH

#include "device/device_decode_result.hpp"
#include "device/cuda/device_buffer.cuh"

namespace steppe::device {

struct DeviceDecodeResult::Impl {
    DeviceBuffer<double> q;  ///< [P × M_kept] resident on device_id (column-major).
    DeviceBuffer<double> v;  ///< [P × M_kept] resident on device_id (column-major).
};

}  // namespace steppe::device
#endif  // STEPPE_DEVICE_CUDA_DEVICE_DECODE_RESULT_IMPL_CUH

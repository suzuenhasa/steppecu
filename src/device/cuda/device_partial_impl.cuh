// src/device/cuda/device_partial_impl.cuh — the CUDA side of DevicePartial: the
// Impl holding the resident DeviceBuffer<double> f2/vpair owners. A tiny CUDA
// header so the SINGLE definition of `struct DevicePartial::Impl` is shared by
// BOTH device_partial.cu (the out-of-line special members) and the consumers
// (cuda_backend.cu wraps the resident buffers; p2p_combine.cu reads impl->f2/
// impl->vpair device pointers). PRIVATE to steppe_device (a CUDA header,
// architecture.md §4).
#ifndef STEPPE_DEVICE_CUDA_DEVICE_PARTIAL_IMPL_CUH
#define STEPPE_DEVICE_CUDA_DEVICE_PARTIAL_IMPL_CUH

#include "device/device_partial.hpp"
#include "device/cuda/device_buffer.cuh"

namespace steppe::device {

struct DevicePartial::Impl {
    DeviceBuffer<double> f2;     ///< [P*P*n_block_local] resident, on device_id.
    DeviceBuffer<double> vpair;  ///< [P*P*n_block_local] resident, on device_id.
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_DEVICE_PARTIAL_IMPL_CUH

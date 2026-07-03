// src/device/cuda/device_f2_blocks_impl.cuh — the CUDA definition of
// DeviceF2Blocks::Impl, holding the resident f2/vpair device buffers.
// Private to steppe_device.
#ifndef STEPPE_DEVICE_CUDA_DEVICE_F2_BLOCKS_IMPL_CUH
#define STEPPE_DEVICE_CUDA_DEVICE_F2_BLOCKS_IMPL_CUH

#include "device/device_f2_blocks.hpp"
#include "device/cuda/device_buffer.cuh"

namespace steppe::device {

struct DeviceF2Blocks::Impl {
    DeviceBuffer<double> f2;
    DeviceBuffer<double> vpair;
};

}  // namespace steppe::device
#endif  // STEPPE_DEVICE_CUDA_DEVICE_F2_BLOCKS_IMPL_CUH

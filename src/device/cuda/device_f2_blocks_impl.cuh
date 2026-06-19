// src/device/cuda/device_f2_blocks_impl.cuh — the CUDA side of DeviceF2Blocks: the
// Impl holding the resident DeviceBuffer<double> f2/vpair owners. PRIVATE to
// steppe_device. Shared by cuda/device_f2_blocks.cu (special members + to_host +
// f2_device/vpair_device), cuda_backend.cu (wraps the resident buffers), and
// cuda/p2p_combine.cu (builds the full result device-resident).
#ifndef STEPPE_DEVICE_CUDA_DEVICE_F2_BLOCKS_IMPL_CUH
#define STEPPE_DEVICE_CUDA_DEVICE_F2_BLOCKS_IMPL_CUH

#include "device/device_f2_blocks.hpp"
#include "device/cuda/device_buffer.cuh"

namespace steppe::device {

struct DeviceF2Blocks::Impl {
    DeviceBuffer<double> f2;     ///< [P*P*n_block] resident on device_id.
    DeviceBuffer<double> vpair;  ///< [P*P*n_block] resident on device_id.
};

}  // namespace steppe::device
#endif  // STEPPE_DEVICE_CUDA_DEVICE_F2_BLOCKS_IMPL_CUH

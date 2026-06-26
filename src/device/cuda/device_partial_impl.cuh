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
    // EXTENT-WIDENING CONTRACT: the [P*P*n_block_local] element count below MUST be
    // formed in std::size_t at the SIZING call site (the consumer, cuda_backend.cu),
    // e.g. `std::size_t(P)*P*n_block_local` — NOT in the int shape fields' type.
    // The shape fields P/n_block_local (device_partial.hpp:40-41) are POD `int` seam
    // contract fields and stay `int` (a 32-bit int product overflows: at P~2500,
    // n_block~757 the extent ~4.7e9 exceeds 2^31). DeviceBuffer's ctor takes size_t;
    // the consumer already widens (cuda_backend.cu sizes from a size_t `total`).
    DeviceBuffer<double> f2;     ///< [P*P*n_block_local] resident, on device_id.
    DeviceBuffer<double> vpair;  ///< [P*P*n_block_local] resident, on device_id.
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_DEVICE_PARTIAL_IMPL_CUH

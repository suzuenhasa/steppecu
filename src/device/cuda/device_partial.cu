// src/device/cuda/device_partial.cu — the CUDA side of DevicePartial: defines the
// out-of-line special members (so the unique_ptr<Impl> in the CUDA-free header has a
// COMPLETE Impl type at the point its destructor/move are instantiated). The Impl
// definition itself lives in device_partial_impl.cuh, shared with cuda_backend.cu /
// p2p_combine.cu. PRIVATE to steppe_device (a CUDA TU, architecture.md §4).
#include "device/cuda/device_partial_impl.cuh"

namespace steppe::device {

DevicePartial::DevicePartial() = default;
DevicePartial::~DevicePartial() = default;
DevicePartial::DevicePartial(DevicePartial&&) noexcept = default;
DevicePartial& DevicePartial::operator=(DevicePartial&&) noexcept = default;

}  // namespace steppe::device

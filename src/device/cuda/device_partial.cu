// src/device/cuda/device_partial.cu — out-of-line DevicePartial special members,
// emitted in a CUDA TU where the complete Impl type (device_partial_impl.cuh) is
// visible so the unique_ptr<Impl> in the CUDA-free header can be destroyed and moved.
#include "device/cuda/device_partial_impl.cuh"

namespace steppe::device {

DevicePartial::DevicePartial() = default;
DevicePartial::~DevicePartial() = default;
DevicePartial::DevicePartial(DevicePartial&&) noexcept = default;
DevicePartial& DevicePartial::operator=(DevicePartial&&) noexcept = default;

}  // namespace steppe::device

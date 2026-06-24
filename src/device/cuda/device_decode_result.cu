// src/device/cuda/device_decode_result.cu — the CUDA side of DeviceDecodeResult.
// Out-of-line special members (so unique_ptr<Impl> sees a complete Impl at
// instantiation) + the device-pointer accessors. PRIVATE to steppe_device (a CUDA
// TU, architecture.md §4). Mirrors device_f2_blocks.cu's defaulted-special-members
// pattern: the DeviceBuffer<double> q/v free device-agnostically in the dtor
// (cudaFree carries the pointer's device).
#include "device/cuda/device_decode_result_impl.cuh"

#include <memory>

namespace steppe::device {

DeviceDecodeResult::DeviceDecodeResult() = default;
DeviceDecodeResult::~DeviceDecodeResult() = default;
DeviceDecodeResult::DeviceDecodeResult(DeviceDecodeResult&&) noexcept = default;
DeviceDecodeResult& DeviceDecodeResult::operator=(DeviceDecodeResult&&) noexcept = default;

const double* DeviceDecodeResult::q_device() const noexcept {
    return impl ? impl->q.data() : nullptr;
}
const double* DeviceDecodeResult::v_device() const noexcept {
    return impl ? impl->v.data() : nullptr;
}

}  // namespace steppe::device

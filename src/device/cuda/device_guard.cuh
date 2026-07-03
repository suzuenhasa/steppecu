// src/device/cuda/device_guard.cuh
//
// DeviceGuard — the one shared scoped RAII helper that restores the CUDA current
// device at scope exit (even on a throwing / early-return path). Move-only, with a
// never-throwing teardown; private to the device (GPU) layer.
//
// Reference: docs/reference/src_device_cuda_device_guard.cuh.md
#ifndef STEPPE_DEVICE_CUDA_DEVICE_GUARD_CUH
#define STEPPE_DEVICE_CUDA_DEVICE_GUARD_CUH

#include <cuda_runtime.h>

#include <utility>

#include "device/cuda/check.cuh"

namespace steppe::device {

// DeviceGuard — scoped CUDA current-device restore RAII helper — reference §1
class DeviceGuard {
public:
    explicit DeviceGuard(int dev) noexcept : dev_(dev) {}

    ~DeviceGuard() { restore(); }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

    DeviceGuard(DeviceGuard&& other) noexcept : dev_(other.dev_) {
        other.dev_ = kNoDevice;
    }
    DeviceGuard& operator=(DeviceGuard&& other) noexcept {
        if (this != &other) {
            restore();
            dev_ = std::exchange(other.dev_, kNoDevice);
        }
        return *this;
    }

private:
    static constexpr int kNoDevice = -1;

    void restore() noexcept {
        if (dev_ != kNoDevice) (void)STEPPE_CUDA_WARN(cudaSetDevice(dev_));
    }

    int dev_;
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_DEVICE_GUARD_CUH

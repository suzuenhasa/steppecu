// src/device/device_partial.hpp
//
// Move-only, CUDA-free handle owning one GPU's f2/Vpair partial left resident on
// the device that computed it (the M4.5 device-resident combine input). Names no
// CUDA type: the DeviceBuffer<double> owners live in a nested Impl defined on the
// GPU side.
//
// Reference: docs/reference/src_device_device_partial.hpp.md
#ifndef STEPPE_DEVICE_DEVICE_PARTIAL_HPP
#define STEPPE_DEVICE_DEVICE_PARTIAL_HPP

#include <memory>
#include <vector>

#include "steppe/config.hpp"

namespace steppe::device {

// Move-only opaque handle to one GPU's resident f2/Vpair partial — reference §1
class DevicePartial {
public:
    // Lifetime & ownership: move-only, survive-then-free — reference §3
    DevicePartial();
    ~DevicePartial();
    DevicePartial(DevicePartial&&) noexcept;
    DevicePartial& operator=(DevicePartial&&) noexcept;
    DevicePartial(const DevicePartial&) = delete;
    DevicePartial& operator=(const DevicePartial&) = delete;

    // Shape (host scalars; CUDA-free) — reference §4
    int P = 0;
    int n_block_local = 0;
    int b0 = 0;
    int device_id = kInvalidDeviceId;

    std::vector<int> block_sizes;

    // Empty-shard predicate — reference §5
    [[nodiscard]] bool empty() const noexcept { return n_block_local <= 0; }

    // Opaque CUDA payload (the DeviceBuffer<double> owners) — reference §2
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace steppe::device
#endif  // STEPPE_DEVICE_DEVICE_PARTIAL_HPP

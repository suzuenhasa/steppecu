// src/device/resources.cpp
//
// build_resources — turns a DeviceConfig into a Resources bundle (one
// device-bound, capability-probed compute backend per GPU). Structural note: this
// file is CUDA-free — it touches the GPU only through the make_cuda_backend /
// visible_device_count / capabilities() seam and includes no <cuda_runtime.h>.
//
// Reference: docs/reference/src_device_resources.cpp.md

#include "device/resources.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "device/backend_factory.hpp"

namespace steppe::device {

// Shared fail-fast message prefix — reference §3
static constexpr const char* kBuildErrPrefix = "steppe::device::build_resources: ";

namespace {

// resolve_device_order: pick the device set + fixed order — reference §4
[[nodiscard]] std::vector<int> resolve_device_order(const DeviceConfig& config, int visible) {
    if (!config.devices.empty()) {
        return config.devices;
    }
    if (visible < 1) {
        throw std::runtime_error(
            std::string(kBuildErrPrefix) +
            "auto-enumeration found no visible CUDA "
            "device (cudaGetDeviceCount < 1) — cannot build Resources");
    }
    std::vector<int> order;
    order.reserve(static_cast<std::size_t>(visible));
    for (int ordinal = 0; ordinal < visible; ++ordinal) {
        order.push_back(ordinal);
    }
    return order;
}

}  // namespace

// validate_device_order: reject out-of-range / duplicate ordinals — reference §5
void validate_device_order(std::span<const int> order, int visible) {
    const std::size_t span_len =
        (visible > 0) ? static_cast<std::size_t>(visible) : 0;
    std::vector<char> seen(span_len, 0);
    for (const int ord : order) {
        if (ord < 0 || ord >= visible) {
            throw std::runtime_error(
                std::string(kBuildErrPrefix) + "configured device ordinal " +
                std::to_string(ord) + " is not among the " + std::to_string(visible) +
                " visible CUDA devices (architecture.md §9 build() validation)");
        }
        if (std::exchange(seen[static_cast<std::size_t>(ord)], char{1}) != 0) {
            throw std::runtime_error(
                std::string(kBuildErrPrefix) + "configured device ordinal " +
                std::to_string(ord) +
                " is duplicated in DeviceConfig::devices — the fixed g=0..G-1 combine "
                "order must pin DISTINCT devices (architecture.md §9, §11.4/§12)");
        }
    }
}

// build_resources: assemble the Resources bundle — reference §6
Resources build_resources(const DeviceConfig& config) {
    const int visible = visible_device_count();
    const std::vector<int> device_order = resolve_device_order(config, visible);

    if (device_order.empty()) {
        throw std::runtime_error(
            std::string(kBuildErrPrefix) +
            "resolved an empty device order — "
            "the SPMG precompute requires at least one device (architecture.md §9)");
    }

    validate_device_order(device_order, visible);

    Resources resources;
    resources.config = config;
    resources.gpus.reserve(device_order.size());

    for (const int ordinal : device_order) {
        PerGpuResources entry;
        entry.device_id = ordinal;
        entry.backend = make_cuda_backend(ordinal);
        entry.caps = entry.backend->capabilities();
        resources.gpus.push_back(std::move(entry));
    }

    return resources;
}

}  // namespace steppe::device

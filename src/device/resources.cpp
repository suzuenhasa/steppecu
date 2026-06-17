// src/device/resources.cpp
//
// build_resources — the SPMG device-resource bundle BUILDER (architecture.md §9
// "Dependency injection of resources", §11.4 SPMG; cleanup 00-overview §(2).1
// "One capability probe, owned at Resources construction").
//
// CUDA-FREE BODY ON PURPOSE (so resources.cpp stays a plain .cpp, design §1/§7):
// it reaches the GPU ONLY through the CUDA-free seam — make_cuda_backend (the
// device-binding factory, backend_factory.hpp) and ComputeBackend::capabilities()
// (the CUDA-free probe POD). It includes NO <cuda_runtime.h>: even device
// enumeration is obtained from the probe's `caps.device_count`
// (cudaGetDeviceCount captured by the CUDA backend's capabilities() override),
// never by calling the CUDA runtime here. This file links into steppe_device
// (alongside cuda_backend.cu) so the factory symbol resolves, but it is itself
// device-toolkit-free.
//
// FAIL-FAST (architecture.md §2): make_cuda_backend throws (via STEPPE_CUDA_CHECK)
// if a configured ordinal cannot be cudaSetDevice-bound / constructed; that
// propagates. The capabilities() probe is NON-throwing on the can_access_peer
// answer (a "no peer access" device is a tagged degrade to the host-staged
// combine baseline, NOT a fault; backend.hpp capabilities() doc, §11.4) — so a
// budget GeForce builds a valid Resources whose caps.can_access_peer == false.

#include "device/resources.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "device/backend_factory.hpp"  // steppe::device::make_cuda_backend (CUDA-free factory, X-9/B8)

namespace steppe::device {

namespace {

/// Resolve the FIXED device ordering the combine sums over (architecture.md §11.4,
/// §12). A non-empty `config.devices` PINS both the set AND the g=0..G-1 order
/// verbatim. EMPTY ⇒ auto-enumerate every visible CUDA device in ordinal order
/// 0..device_count-1 (the §9 DeviceConfig::devices contract). The visible count is
/// read OUT-OF-BAND from a throwaway device-0 backend's capabilities().device_count
/// (the CUDA backend's capabilities() override captures cudaGetDeviceCount), so
/// this stays CUDA-free — no <cuda_runtime.h> here (design §1).
[[nodiscard]] std::vector<int> resolve_device_order(const DeviceConfig& config) {
    if (!config.devices.empty()) {
        return config.devices;  // explicit list pins the set AND the order
    }
    // Auto-enumerate: probe device 0 once for the visible device_count, then build
    // the dense ordinal list 0..device_count-1. The throwaway backend is destroyed
    // at scope exit (RAII), releasing its device-0 stream/handle before the real
    // per-device backends are constructed below.
    auto probe = make_cuda_backend(0);
    const int visible = probe->capabilities().device_count;
    if (visible < 1) {
        throw std::runtime_error(
            "steppe::device::build_resources: auto-enumeration found no visible CUDA "
            "device (capabilities().device_count < 1) — cannot build Resources");
    }
    std::vector<int> order;
    order.reserve(static_cast<std::size_t>(visible));
    for (int ordinal = 0; ordinal < visible; ++ordinal) {
        order.push_back(ordinal);
    }
    return order;
}

}  // namespace

Resources build_resources(const DeviceConfig& config) {
    const std::vector<int> device_order = resolve_device_order(config);

    // Fail-fast (architecture.md §2): a resolved Resources must own >= 1 device.
    if (device_order.empty()) {
        throw std::runtime_error(
            "steppe::device::build_resources: resolved an empty device order — "
            "the SPMG precompute requires at least one device (architecture.md §9)");
    }

    Resources resources;
    resources.config = config;  // freeze the resolved intent levers (§9)
    resources.gpus.reserve(device_order.size());

    // One backend PER device in FIXED g=0..G-1 order, each cudaSetDevice-bound by the
    // device_id ctor (per-device-instance contract, backend.hpp), capabilities()
    // probed ONCE and recorded out-of-band (cleanup §(2).1). gpus[0] is the combine
    // root (GPU 0). On a budget tier a device's caps.can_access_peer == false is the
    // EXPECTED non-throwing tagged degrade — build still succeeds (§11.4).
    for (const int ordinal : device_order) {
        PerGpuResources entry;
        entry.device_id = ordinal;
        entry.backend = make_cuda_backend(ordinal);  // throws fail-fast on a bad ordinal
        entry.caps = entry.backend->capabilities();   // the ONE probe, owned here
        resources.gpus.push_back(std::move(entry));
    }

    return resources;
}

}  // namespace steppe::device

// src/device/resources.cpp
//
// build_resources — the SPMG device-resource bundle BUILDER (architecture.md §9
// "Dependency injection of resources", §11.4 SPMG; cleanup 00-overview §(2).1
// "One capability probe, owned at Resources construction").
//
// CUDA-FREE BODY ON PURPOSE (so resources.cpp stays a plain .cpp, design §1/§7):
// it reaches the GPU ONLY through the CUDA-free seam — make_cuda_backend (the
// device-binding factory) + visible_device_count() (the CUDA-free count query, both
// in backend_factory.hpp) + ComputeBackend::capabilities() (the CUDA-free probe POD).
// It includes NO <cuda_runtime.h>: device enumeration is obtained from the CUDA-free
// visible_device_count() factory query (a one-line cudaGetDeviceCount in
// cuda_backend.cu), never by calling the CUDA runtime here. This file links into
// steppe_device (alongside cuda_backend.cu) so those symbols resolve, but it is
// itself device-toolkit-free.
//
// FAIL-FAST (architecture.md §2): make_cuda_backend throws (via STEPPE_CUDA_CHECK)
// if a configured ordinal cannot be cudaSetDevice-bound / constructed; that
// propagates. The capabilities() probe is NON-throwing on the can_access_peer
// answer (a "no peer access" device is a tagged degrade to the host-staged
// combine baseline, NOT a fault; backend.hpp capabilities() doc, §11.4) — so a
// budget GeForce builds a valid Resources whose caps.can_access_peer == false.

#include "device/resources.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "device/backend_factory.hpp"  // make_cuda_backend + visible_device_count (CUDA-free factory, X-9/B8)

namespace steppe::device {

/// Shared fail-fast message prefix for every build_resources throw (group-7: single-
/// source so a function rename can't silently drift the four messages). A file-static
/// constexpr is the right home — it is a message string, not a config.hpp tuning value
/// (§2.5 names it kPascalCase; it stays local, not promoted to config.hpp). The
/// trailing space is intentional: each throw concatenates its specific cause after it,
/// so the emitted messages stay byte-identical to the prior copy-pasted literals.
static constexpr const char* kBuildErrPrefix = "steppe::device::build_resources: ";

namespace {

/// Resolve the FIXED device ordering the combine sums over (architecture.md §11.4,
/// §12). A non-empty `config.devices` PINS both the set AND the g=0..G-1 order
/// verbatim. EMPTY ⇒ auto-enumerate every visible CUDA device in ordinal order
/// 0..visible-1 (the §9 DeviceConfig::devices contract). `visible` is the ONE
/// CUDA-free visible_device_count() query, taken once by build_resources and passed
/// in (it also feeds validate_device_order), so the count is read exactly once per
/// build — no <cuda_runtime.h> here, and NO throwaway device-0 backend / leaked
/// cudaSetDevice(0) (cleanup B8; group-7 7.2 single-query). This stays a pure
/// resolution with no device side effect.
[[nodiscard]] std::vector<int> resolve_device_order(const DeviceConfig& config, int visible) {
    if (!config.devices.empty()) {
        return config.devices;  // explicit list pins the set AND the order
    }
    // Auto-enumerate: the dense ordinal list 0..visible-1 from the single CUDA-free
    // count query build_resources already took (no context spin-up, no workspace
    // alloc, no device selection — unlike the old throwaway make_cuda_backend(0)).
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

void validate_device_order(std::span<const int> order, int visible) {
    // §9 build()-validation (architecture.md §9; cleanup B8/C1): reject out-of-range
    // or duplicate ordinals fail-fast (§2). Pure host arithmetic over (order, visible)
    // — no CUDA, GPU-free-unit-testable (T1). The fixed g=0..G-1 combine order must
    // pin DISTINCT, present devices (§11.4/§12), or a run silently mis-binds.
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
        // ord is now in [0, visible) so the index is in-bounds for `seen`.
        if (std::exchange(seen[static_cast<std::size_t>(ord)], char{1}) != 0) {
            throw std::runtime_error(
                std::string(kBuildErrPrefix) + "configured device ordinal " +
                std::to_string(ord) +
                " is duplicated in DeviceConfig::devices — the fixed g=0..G-1 combine "
                "order must pin DISTINCT devices (architecture.md §9, §11.4/§12)");
        }
    }
}

Resources build_resources(const DeviceConfig& config) {
    // ONE CUDA-free count query per build, shared by both the auto-enumerate sizing
    // (resolve_device_order) and the §9 validation (validate_device_order) below — so
    // the "one count query serves both" contract is literally true (cleanup group-7
    // 7.2). visible_device_count() is the CUDA-free cudaGetDeviceCount factory query.
    const int visible = visible_device_count();
    const std::vector<int> device_order = resolve_device_order(config, visible);

    // Fail-fast (architecture.md §2): a resolved Resources must own >= 1 device.
    if (device_order.empty()) {
        throw std::runtime_error(
            std::string(kBuildErrPrefix) +
            "resolved an empty device order — "
            "the SPMG precompute requires at least one device (architecture.md §9)");
    }

    // §9 build() validation BEFORE binding any device: reject duplicate / out-of-range
    // ordinals (cleanup B8/C1). The single `visible` count query taken above serves
    // both the auto-enumerate sizing (resolve_device_order) and this validation — and
    // on the auto path the dense 0..visible-1 order it produced trivially passes.
    // Validating first turns the silent {0,0} footgun (two lanes on one GPU) and the
    // deep cudaSetDevice "invalid device ordinal" throw into a §9-grade fail-fast.
    validate_device_order(device_order, visible);

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
        entry.backend = make_cuda_backend(ordinal);  // ordinal already §9-validated above
        entry.caps = entry.backend->capabilities();   // the ONE probe, owned here
        resources.gpus.push_back(std::move(entry));
    }

    return resources;
}

}  // namespace steppe::device

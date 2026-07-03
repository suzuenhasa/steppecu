// src/device/resources.hpp
//
// The RAII-owned bundle of per-GPU device resources the single-node multi-GPU
// f2 precompute is injected with — built once, with no global mutable device
// state. CUDA-free by contract: it names only CUDA-free seam types
// (ComputeBackend, BackendCapabilities, DeviceConfig), so it compiles into core
// and the tests without the CUDA toolkit.
//
// Reference: docs/reference/src_device_resources.hpp.md
#ifndef STEPPE_DEVICE_RESOURCES_HPP
#define STEPPE_DEVICE_RESOURCES_HPP

#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "device/backend.hpp"

namespace steppe::device {

// CombinePath — the which-transport tag for the last combine — reference §2
enum class CombinePath {
    None,
    HostStaged,
    P2pDeviceResident
};

// MultiGpuTimings — phase timing of the last multi-GPU run — reference §3
struct MultiGpuTimings {
    double compute_wall_ms = 0.0;
    double combine_wall_ms = 0.0;
    double combine_peer_ms = 0.0;
    double combine_d2h_ms = 0.0;
    std::size_t h2d_bytes = 0;
    std::size_t d2h_bytes = 0;
    std::size_t peer_bytes = 0;
};

// PerGpuResources — one GPU's owned resources — reference §4
struct PerGpuResources {
    int device_id = 0;

    std::unique_ptr<ComputeBackend> backend;

    BackendCapabilities caps{};
};

// Resources — the injected per-run device bundle — reference §5
struct Resources {
    std::vector<PerGpuResources> gpus;

    DeviceConfig config;

    CombinePath last_combine_path = CombinePath::None;

    MultiGpuTimings last_multigpu_timings{};

    [[nodiscard]] std::size_t device_count() const noexcept { return gpus.size(); }
};

// validate_device_order — pure device-order check — reference §6
void validate_device_order(std::span<const int> order, int visible);

// build_resources — construct the G-device bundle — reference §7
[[nodiscard]] Resources build_resources(const DeviceConfig& config);

// device_fault_status — classify a caught device OOM fault — reference §8
[[nodiscard]] std::optional<Status> device_fault_status(
    const std::exception& e) noexcept;

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_RESOURCES_HPP

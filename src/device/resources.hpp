// src/device/resources.hpp
//
// Resources / PerGpuResources — the injected, RAII-owned device-resource bundle
// the single-node multi-GPU (SPMG) precompute drives against (architecture.md §9
// "Dependency injection of resources", §11.4 SPMG; cleanup 00-overview §(2).1
// "One capability probe, owned at Resources construction"). No global mutable
// state: these are constructed once by `build_resources` and injected into the
// multi-GPU entry point (architecture.md §2 "No global mutable state").
//
// CUDA-FREE BY CONTRACT, like device/backend.hpp: it names only the CUDA-free
// `ComputeBackend` interface (held by unique_ptr), the CUDA-free `BackendCapabilities`
// POD, and the CUDA-free `DeviceConfig`. So this header compiles into `core`/the
// tests without the CUDA toolkit. The BUILDER (build_resources, resources.cpp) is
// device-layer: it calls make_cuda_backend, which `new`s a CudaBackend bound to a
// device (the per-device-instance contract, backend.hpp:193-202).
//
// PER-DEVICE-INSTANCE CONTRACT (backend.hpp): one ComputeBackend == one CUDA device,
// constructed with that device's id and cudaSetDevice-bound. The per-device CUDA
// stream + cuBLAS handle + emulated-FP64 workspace already live INSIDE CudaBackend
// (stream_, blas_, workspace_); PerGpuResources does NOT re-own them (DRY, §8). It
// adds only the device id + the out-of-band capability TAG (never on F2BlockTensor;
// architecture.md §12, cleanup §(2).2).
#ifndef STEPPE_DEVICE_RESOURCES_HPP
#define STEPPE_DEVICE_RESOURCES_HPP

#include <cstddef>
#include <memory>
#include <vector>

#include "steppe/config.hpp"   // steppe::DeviceConfig (CUDA-free)
#include "device/backend.hpp"  // steppe::ComputeBackend, steppe::BackendCapabilities (CUDA-free seam)

namespace steppe::device {

/// One per device in DeviceConfig::devices, RAII-owned (architecture.md §9
/// PerGpuResources). M4.5-precompute-focused: the per-device backend (which itself
/// owns the device's single statistic stream + cuBLAS handle + emulated-FP64
/// determinism workspace, cuda_backend.cu) + that device's PROBED capability tier
/// + its physical CUDA ordinal. cuSOLVER / NCCL are intentionally absent: the
/// precompute combine is cudaMemcpyPeer (P2P tier) or host-staged (baseline),
/// NEVER an NCCL AllReduce (architecture.md §11.4, §12); cuSOLVER lands with the
/// S4-S8 fit engine (a later workflow).
///
/// MOVE-ONLY: it owns a unique_ptr<ComputeBackend>, so the whole struct is
/// move-only (default-move/no-copy synthesized) — exactly the §9 move-only
/// ownership of concrete backends. Copying would clone the (deleted-copy) backend,
/// which is correctly ill-formed.
struct PerGpuResources {
    /// The physical CUDA device ordinal this entry owns (== DeviceConfig::devices[g]).
    int device_id = 0;

    /// The compute backend BOUND to `device_id` (constructed via the device_id ctor;
    /// per-device-instance contract, backend.hpp). Owns the device's stream/handle/
    /// workspace. Held as the abstract interface so this struct stays CUDA-free.
    std::unique_ptr<ComputeBackend> backend;

    /// The capability tier PROBED from `backend->capabilities()` at build time
    /// (compute cap, total/free VRAM, can_access_peer, emulated_fp64_honorable).
    /// Recorded OUT-OF-BAND here, never on the numeric F2BlockTensor (architecture.md
    /// §12; cleanup §(2).2). Drives the combine-path selection + the run tag.
    BackendCapabilities caps{};
};

/// The G-of-them bundle injected into the multi-GPU precompute entry point
/// (architecture.md §9 Resources). Construct ONCE via build_resources; const after.
///
/// MOVE-ONLY by composition (it holds move-only PerGpuResources in a vector).
struct Resources {
    /// One PerGpuResources per device in `device_order`, in g=0..G-1 order — the
    /// FIXED combine order (architecture.md §11.4, §12). gpus[0].device_id is the
    /// combine root (GPU 0) for the P2P device-resident combine.
    std::vector<PerGpuResources> gpus;

    /// The resolved, frozen device config (the fixed device set/order + the
    /// prefer_p2p_combine / enable_peer_access / deterministic intent levers, §9).
    DeviceConfig config;

    /// Convenience: number of devices G (== gpus.size() == resolved devices count).
    [[nodiscard]] std::size_t device_count() const noexcept { return gpus.size(); }
};

/// Build the G-device Resources for the SPMG precompute (cleanup 00-overview §(2).1
/// "One capability probe, owned at Resources construction"). For each ordinal in
/// `config.devices` (in order; empty ⇒ auto-enumerate every visible CUDA device in
/// cudaGetDeviceCount order — the §9 DeviceConfig::devices contract) it constructs a
/// CudaBackend bound to that ordinal (make_cuda_backend(ordinal)) and probes its
/// capabilities() ONCE, storing both in a PerGpuResources. The probe is non-throwing
/// on the P2P answer (a "no peer access" device degrades to the host-staged combine
/// baseline; backend.hpp capabilities() doc, architecture.md §11.4) — but a genuine
/// fault (cannot enumerate / construct on a configured device) throws (fail-fast, §2).
///
/// `config.devices` here IS the fixed combine order: g indexes gpus[g], and the
/// combine sums g=0..G-1 (architecture.md §12). devices=={0} (or empty on a 1-GPU
/// box) yields a single-entry Resources whose one backend is the exact current
/// single-GPU path.
[[nodiscard]] Resources build_resources(const DeviceConfig& config);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_RESOURCES_HPP

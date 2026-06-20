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
#include <span>
#include <vector>

#include "steppe/config.hpp"   // steppe::DeviceConfig (CUDA-free)
#include "device/backend.hpp"  // steppe::ComputeBackend, steppe::BackendCapabilities (CUDA-free seam)

namespace steppe::device {

/// Which combine transport the LAST multi-GPU precompute actually ran — the
/// OUT-OF-BAND "which path did this run take" tag (architecture.md §11.4 tagged
/// combine; config.hpp prefer_p2p_combine override-knob banner: "the recorded
/// which-path tag ... live in Resources / the result metadata, NEVER on
/// DeviceConfig and NEVER on the pure-numeric F2BlockTensor", cleanup §(2).2). It is
/// DISCOVERED RUNTIME STATE, not intent: the §4 gate (defined ONCE at the `use_p2p`
/// computation in f2_blocks_multigpu.cpp — "THE §4 COMBINE GATE" — §8 single-source)
/// selects P2P; everything else degrades to the host-staged baseline. Both tiers are
/// BIT-IDENTICAL (parity-NEUTRAL, §12), so this
/// tag is observability — it lets a caller/test confirm WHICH transport ran without
/// inspecting the numeric tensor (the parity test reads it to verify the P2P arm
/// actually exercised P2P rather than silently falling back).
enum class CombinePath {
    /// No multi-GPU combine has run on this Resources yet — the value-initialized
    /// default. The G==1 single-GPU fast path (no shard, no combine) does NOT touch
    /// this field, so it is NOT reset to `None` by a G==1 run (the entry point only
    /// ever ASSIGNS `HostStaged` / `P2pDeviceResident`, never `None`): after a prior
    /// G>=2 run on the same Resources the tag stale-reads the last combine, which is
    /// correct since a G==1 run combines nothing.
    None,
    /// The last G>=2 run combined via the host-staged fixed-order combine
    /// (combine_f2_partials_host) — the portable parity baseline (architecture.md
    /// §11.4). Taken when prefer_p2p_combine is false, OR enable_peer_access is false
    /// (the user forbade the cudaDeviceEnablePeerAccess the P2P path needs), OR peer
    /// access is unavailable on the device.
    HostStaged,
    /// The last G>=2 run combined via the device-resident cudaMemcpyPeer combine
    /// (combine_f2_partials_p2p) — the opt-in fast-path (architecture.md §11.4).
    /// Taken when the four-term §4 gate holds (the MAY-WE permission AND the
    /// WHICH-PATH preference both granted, AND the device can peer, AND G >= 2). The
    /// gate predicate is defined ONCE — "THE §4 COMBINE GATE" at the `use_p2p`
    /// computation in f2_blocks_multigpu.cpp (§8 single-source).
    P2pDeviceResident
};

/// Out-of-band phase timing of the LAST multi-GPU run (M4.5 Item 4; OBSERVABILITY
/// ONLY). It is a CUDA-free POD of plain wall-clock milliseconds + measured DMA byte
/// totals, recorded the SAME out-of-band way as `last_combine_path`: NEVER on the
/// numeric `F2BlockTensor`, NEVER on `DeviceConfig`, NEVER on the pure-numeric path —
/// it lives here on `Resources` so a bench/test can attribute the per-run cost
/// (compute fan-out vs combine, and the bus traffic) WITHOUT inspecting the tensor
/// (mirrors the `last_combine_path` discipline; architecture.md §12, cleanup §(2).2).
///
/// Zeroed at the START of each `compute_f2_blocks_multigpu` G>=2 run (the G==1 fast
/// path leaves it at default — it runs no combine). The orchestrator
/// (`compute_f2_blocks_multigpu`, steppe::core, CUDA-FREE) fills the two host
/// `std::chrono::steady_clock` wall brackets (`compute_wall_ms`, `combine_wall_ms`)
/// and the three arithmetic byte totals (derivable host-side from P, M, n_block, G —
/// see field docs); it does NOT thread CUDA event timers through the CUDA-free seam,
/// so the finer device-internal fields (`combine_peer_ms`, `combine_d2h_ms`) stay 0
/// unless a future combine returns them.
struct MultiGpuTimings {
    /// Fan-out wall: shard plan + the CONCURRENT per-device partial compute, host
    /// steady_clock bracket (0 on the G==1 fast path / before any G>=2 run).
    double compute_wall_ms = 0.0;
    /// Combine call wall: the whole combine (peer/D2D placement + the single final
    /// full-result D2H), host steady_clock bracket. 0 if no combine ran.
    double combine_wall_ms = 0.0;
    /// Cross-device DMA portion of the combine (root D2D + peer copies), ms.
    /// Device-internal — left 0 unless the combine reports it (no CUDA timer is
    /// threaded through the CUDA-free seam).
    double combine_peer_ms = 0.0;
    /// The single final full-result D2H portion of the combine, ms. Device-internal —
    /// left 0 unless the combine reports it.
    double combine_d2h_ms = 0.0;
    /// Measured H2D bytes (inputs Q/V/N), per-run total summed over devices:
    /// 3 * P * M * sizeof(double) (each device receives its column sub-view; the
    /// columns partition M, so the summed H2D volume is the full 3*P*M doubles).
    std::size_t h2d_bytes = 0;
    /// Measured D2H bytes, per-run total: the SINGLE full-result copy
    /// 2 * P * P * n_block * sizeof(double) (f2 + Vpair). Under the device-resident
    /// combine there is exactly one full D2H (no per-device partial D2H).
    std::size_t d2h_bytes = 0;
    /// Measured cross-device DMA bytes: 2 * P * P * (n_block - n_block_on_root) *
    /// sizeof(double) (the peer partials pulled to the root; the root's own slabs are
    /// a same-device D2D, not counted as peer traffic).
    std::size_t peer_bytes = 0;
};

/// One per device in DeviceConfig::devices, RAII-owned (architecture.md §9
/// PerGpuResources). M4.5-precompute-focused: the per-device backend (which itself
/// owns the device's single statistic stream + cuBLAS handle + cuSOLVER handle +
/// emulated-FP64 determinism workspace, cuda_backend.cu) + that device's PROBED
/// capability tier + its physical CUDA ordinal. NCCL is intentionally absent: the
/// precompute combine is cudaMemcpyPeer (P2P tier) or host-staged (baseline),
/// NEVER an NCCL AllReduce (architecture.md §11.4, §12). The cuSOLVER handle for the
/// S4-S8 fit engine landed with M(fit-4): it lives INSIDE CudaBackend (like blas_/
/// stream_/workspace_, the DRY rule §8) — NO struct field is added here
/// (PerGpuResources holds the backend by unique_ptr<ComputeBackend>; the handle is
/// owned one layer down).
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

    /// Which combine transport the LAST compute_f2_blocks_multigpu run used — the
    /// out-of-band which-path tag (CombinePath doc above; architecture.md §11.4,
    /// cleanup §(2).2). DISCOVERED runtime state, recorded by the entry point's §4
    /// fork, NEVER on the numeric F2BlockTensor. `None` until the first multi-GPU run
    /// (or after a G==1 fast-path run). Mutated by the entry point (which takes
    /// `Resources&`); read by callers/tests to confirm the chosen tier.
    CombinePath last_combine_path = CombinePath::None;

    /// Out-of-band phase timing of the LAST multi-GPU run (M4.5 Item 4; observability
    /// only — see `MultiGpuTimings` doc). Filled by the entry point's G>=2 path
    /// (`compute_f2_blocks_multigpu`, which takes `Resources&`), read by the bench /
    /// tests to attribute per-run cost. NEVER on the numeric `F2BlockTensor`. Stays at
    /// default after a G==1 fast-path run (no combine to time).
    MultiGpuTimings last_multigpu_timings{};

    /// Convenience: number of devices G (== gpus.size() == resolved devices count).
    [[nodiscard]] std::size_t device_count() const noexcept { return gpus.size(); }
};

/// Validate a resolved device ordering against the visible device count — the §9
/// build()-validation contract realized as a PURE, CUDA-FREE, GPU-free predicate
/// (architecture.md §9 "Validation at build() rejects ... any device id in
/// DeviceConfig::devices that is absent or duplicated"; §2 fail-fast). It rejects:
///   * an OUT-OF-RANGE ordinal (`ord < 0 || ord >= visible`) — a configured device
///     that is not among the `visible` CUDA devices the process sees; and
///   * a DUPLICATE ordinal — the fixed g=0..G-1 combine order must pin DISTINCT
///     devices (architecture.md §11.4/§12); a repeated member would silently run two
///     lanes serially on ONE GPU, masking a real second device and ignoring the
///     user's intent (config.hpp DeviceConfig::devices "PINS both the set AND the
///     ordering"). §9 lists "duplicated" as a reject; this is the enforcing site
///     until ConfigBuilder::build() exists.
/// It is `noexcept(false)` on purpose (it THROWS std::runtime_error fail-fast on a
/// bad ordinal). EMPTY `order` is NOT rejected here (the empty-order fail-fast lives
/// in build_resources, which carries the §9 "at least one device" message); this is
/// a no-op for an empty span. Factored out (vs inlined in build_resources) so the
/// §13 validation logic is unit-testable GPU-free over (order, visible) — it touches
/// no CUDA and no device (T1).
///
/// @throws std::runtime_error if any ordinal is out-of-range or duplicated.
void validate_device_order(std::span<const int> order, int visible);

/// Build the G-device Resources for the SPMG precompute (cleanup 00-overview §(2).1
/// "One capability probe, owned at Resources construction"). The visible CUDA device
/// count is read ONCE via the CUDA-free visible_device_count() factory query
/// (backend_factory.hpp; cleanup B8 — replacing the old throwaway device-0 backend
/// build) and the resolved ordering is validated against it (validate_device_order:
/// reject out-of-range / duplicate ordinals, §9 fail-fast). Then, for each ordinal in
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
///
/// STRONGLY EXCEPTION-SAFE: a mid-loop fail-fast (e.g. ordinal 1 of {0,1} cannot be
/// bound) unwinds all already-bound backends via RAII (the partial `Resources` local
/// + the in-progress entry destruct), leaking no device handle or VRAM (§7).
///
/// @throws std::runtime_error on an empty resolved order, no visible device, or a
///         duplicate/out-of-range ordinal (validate_device_order, §9).
/// @throws steppe::device::CudaError if a configured ordinal cannot be cudaSetDevice-
///         bound / a backend constructed on it (via make_cuda_backend, fail-fast §2).
[[nodiscard]] Resources build_resources(const DeviceConfig& config);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_RESOURCES_HPP

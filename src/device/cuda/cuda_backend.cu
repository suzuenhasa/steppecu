// src/device/cuda/cuda_backend.cu
//
// CudaBackend — the GPU implementation of ComputeBackend (architecture.md §4, §8;
// ROADMAP §2, M0). Implements `compute_f2` via the f2 3-GEMM reformulation:
// upload the Q/V/N contract, run the fused feeder + three GEMMs + fused
// numerator/divide (f2_block_kernel.cu), and copy f2 + Vpair back across the
// CUDA-free ComputeBackend seam (architecture.md §4).
//
// CudaBackend is INTENTIONALLY a single seam TU: a C++ class cannot be partial
// across translation units, and the launch wrappers / kernels are co-located here on
// purpose (architecture.md §4, §7; ASSESSMENT §5.5). The cross-TU split is REJECTED by
// decision. (X3's thin-aggregate-header + per-subsystem `.inc`-include split — still
// ONE class, so it sidesteps the "can't be partial" objection — is PARKED, never
// separately evaluated; revisit only if the file size becomes a maintenance burden.
// DECISIONS minute: docs/kimiactions/01-open-worth-doing.md §F2.)
//
// MULTI-GPU-READY (M4.5 capability-tier SCAFFOLD; architecture.md §9, §11.4, §12;
// cleanup device-cuda-cuda_backend F19/F20, X-6/B2; 00-overview §(2)). One backend
// instance is BOUND to ONE CUDA device (the per-device-instance contract,
// backend.hpp): the ctor takes an `int device_id` (default 0 ⇒ single-GPU,
// unchanged), `cudaSetDevice`-selects it (so the cuBLAS context binds there, cuBLAS
// §2.1.2), and every compute entry re-selects it (`guard_device`). `capabilities()`
// probes the bound device — compute capability, total+free VRAM, P2P reachability
// (the NON-throwing tagged degrade: "no peer access" is EXPECTED on the budget
// tier, not a fault), and EmulatedFp64-honorability (the X-6/B2 predicate). This is
// the SCAFFOLD only: SNP sharding + the host-side fixed-order combine across the G
// devices are orchestrated ABOVE this seam by `Resources` (architecture.md §11.4) —
// that combine algorithm is the NEXT workflow, not implemented here.
//
// STANDARDS (ROADMAP §1, §5):
//   * RAII for ALL device memory and handles — DeviceBuffer<T> + CublasHandle, no
//     raw cudaMalloc / cublasCreate here (architecture.md §2, §7). This TU is NOT
//     on the allocation allowlist.
//   * Precision is typed config: forwarded unchanged into run_f2_gemms, which
//     engages FIXED-slice Ozaki / native FP64 via the ONE honorability predicate —
//     an unhonorable EmulatedFp64 request DEGRADES to native Fp64 with a logged
//     capability tag, never silently runs DYNAMIC (architecture.md §12; X-6/B2).
//   * The numerator/divide stays native FP64 (in the assemble kernel).
//   * The formula lives ONCE in core/internal/f2_estimator.hpp, shared with the
//     CPU oracle, so CPU and GPU cannot diverge (architecture.md §13).
//   * Every capability lever is PARITY-NEUTRAL (data-movement / observability only),
//     so §12 parity holds identically on both tiers (architecture.md §11.4, §12).
//   * DEVICE-AGNOSTIC FREE (the M4.5 escape invariant; cleanup [17.5]). Every
//     compute entry runs `guard_device()` so allocations land on `device_id_`, but
//     teardown — `~CudaBackend`'s member RAII (there is NO explicit `~CudaBackend`),
//     the per-call DeviceBuffer/Event scope-exit frees, and CRITICALLY the
//     resident `rb.f2`/`rb.vpair` that MOVE OUT into a DeviceF2Blocks/DevicePartial
//     and are freed LATER by the host-side combine under a possibly-device-0 ambient
//     — runs under whatever device is current, and NONE of these free sites
//     re-selects `device_id_`. This is SOUND BY DESIGN: cudaFree / cudaStreamDestroy
//     / cudaEventDestroy / cublasDestroy / cusolverDnDestroy are all DEVICE-
//     ASSOCIATION-AGNOSTIC — the pointer/object carries its own device/context, so
//     the free works regardless of the current device (cudaFree VERIFIED silent on
//     the current device, CUDA 13.x Runtime API CUDART_MEMORY — it neither requires
//     nor forbids the alloc device be current). The RAII wrappers therefore
//     INTENTIONALLY omit a record-and-restore cudaSetDevice (the design bars hidden
//     global state in the owner — selecting the device is the caller's / `Resources`'
//     job, architecture.md §7). The single-home statement + the cudaMallocAsync
//     re-trigger warning live on DeviceBuffer::reset (device_buffer.cuh); this is the
//     backend-side restatement so the multi-GPU escape seam is self-documenting.
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). It is the only
// place a host caller meets the GPU f2 path; `core` reaches it solely through the
// CUDA-free ComputeBackend interface in device/backend.hpp.
#include <cublas_v2.h>
#include <cufft.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cub/device/device_select.cuh>  // cub::DeviceSelect::Flagged — the GPU-only sweep
                                         // survivor stream-compaction (CUDA 13.x / CCCL CUB;
                                         // two-call temp-storage idiom, num_items NumItemsT,
                                         // no debug_synchronous param — verified vs the docs)
#include <cub/device/device_radix_sort.cuh>  // cub::DeviceRadixSort::SortPairsDescending — the
                                             // bounded device top-K reservoir compaction
                                             // (KeyT=double |z|, ValueT=int perm index; classic
                                             // stream overload, two-call idiom, NumItemsT 64-bit)
#include <cub/device/device_scan.cuh>        // cub::DeviceScan::ExclusiveSum — the decode-seam
                                             // keep-flag → compacted-column-index prefix sum
                                             // (CUDA 13.x / CCCL CUB; two-call temp-storage idiom)

#include <algorithm>
#include <climits>    // INT_MIN — the AT2 rankdrop "NA" sentinel (M(fit-2))
#include <cstdlib>    // std::getenv / std::strtol — the STEPPE_FSTAT_CHUNK sweep chunk lever
#include <cmath>      // std::numeric_limits / quiet_NaN for the rankdrop NA encoding (M(fit-2))
#include <cstddef>
#include <cstdint>
#include <cstring>    // std::memcpy — staging->result slice copy (P5/d2h-speed)
#include <exception>  // std::exception — the device_fault_status translator param (B2)
#include <limits>     // std::numeric_limits<int>::max — the M0 k-narrowing guard (B22)
#include <memory>
#include <optional>   // std::optional / std::nullopt — device_fault_status return (B2)
#include <span>
#include <stdexcept>  // std::runtime_error — the M0 M>INT_MAX fail-fast (B22)
#include <string>     // std::to_string — diagnostic message for the B22 guard
#include <vector>

#include "core/domain/block_partition_rule.hpp" // core::block_ranges, core::BlockRange (the X-3/B3 single-source inverse)
#include "core/internal/pchisq.hpp"         // core::internal::pchisq_upper (M(fit-2) rank-test p; the ONE shared special fn)
#include "core/internal/nvtx.hpp"           // STEPPE_NVTX_RANGE (coarse phase-boundary markers; empty unless -DSTEPPE_NVTX)
#include "core/internal/qpfstats_jackknife.hpp"  // core::f2blocks_pair_est (the partial-NaN fallback recenter; cold path)
#include "core/internal/small_linalg.hpp"   // core::solve (the host downdated-A partial-NaN solve; rank_Q is now ON-DEVICE — L1)
#include "core/qpadm/qpadm_bounds.hpp"       // core::qpadm::model_fits_small_path — the SINGLE-SOURCE small-path envelope (kQpMax*)
#include "device/backend.hpp"               // ComputeBackend, F2Result, F2BlockTensor, MatView
#include "device/backend_factory.hpp"       // steppe::device::make_cuda_backend (the single-source decl, X-9/B8)
#include "device/resources.hpp"             // steppe::device::device_fault_status (the CUDA-free fault-taxonomy seam this TU defines, B2)
#include "device/device_partial.hpp"        // steppe::device::DevicePartial (the M4.5 resident handle)
#include "device/cuda/device_partial_impl.cuh" // DevicePartial::Impl (the DeviceBuffer<double> owners)
#include "device/device_f2_blocks.hpp"      // steppe::device::DeviceF2Blocks (the M4.5 device-resident FULL result handle)
#include "device/cuda/device_f2_blocks_impl.cuh" // DeviceF2Blocks::Impl (the DeviceBuffer<double> owners)
#include "device/device_decode_result.hpp"  // steppe::device::DeviceDecodeResult (the device-resident autosome-compacted Q/V handle)
#include "device/cuda/device_decode_result_impl.cuh" // DeviceDecodeResult::Impl (the DeviceBuffer<double> q/v owners)
#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK
#include "device/cuda/decode_af_kernel.cuh" // launch_decode_af
#include "device/cuda/detect_ploidy_kernel.cuh" // launch_detect_ploidy (M-FR-0: on-device AT2 per-sample ploidy prepass)
#include "device/cuda/transpose_canonical_kernel.cuh" // launch_transpose_to_canonical (M-FR-1: SNP-major -> canonical individual-major transpose+gather+encoding)
#include "device/cuda/decode_compact_kernel.cuh" // launch_autosome_keep_mask / _compact_columns_gather (the device-resident decode seam)
#include "device/cuda/dstat_kernel.cuh"     // launch_dstat_block_reduce (qpDstat Part B)
#include "device/cuda/dates_kernel.cuh"     // DATES cuFFT autocorrelation LD engine kernels
#include "device/cuda/qpfstats_kernel.cuh"  // launch_qpfstats_zero_nan_ymat / _add_ridge_diag (the smoother prep)
#include "device/cuda/qpfstats_jackknife_kernel.cuh"  // launch_qpfstats_numer_jackknife / _recenter_shift (the fused PERF path)
#include "device/cuda/ratio_block_jackknife_kernel.cuh"  // launch_ratio_block_jackknife (the SHARED f4ratio M1 + qpDstat M2 engine)
#include "device/cuda/device_buffer.cuh"    // DeviceBuffer<T> (RAII)
#include "device/cuda/f2_block_kernel.cuh"  // launch_f2_feeder, engage_f2_precision, emulation_honorable (the X-6/B2 probe)
#include "device/cuda/f2_batched_kernel.cuh" // launch_gather_group, run_f2_gemms_group, launch_assemble_blocks_group
#include "device/cuda/block_sink.cuh"       // M5: BlockSink, HostRamSink, DiskSink, kStreamStagingSlots
#include "device/stream_f2_blocks.hpp"      // M5: StreamTarget (the CUDA-free streamed-tier request)
#include "device/f2_blocks_out.hpp"         // M5: DiskF2Blocks (the Disk descriptor DiskSink populates)
#include "device/cuda/handles.hpp"          // CublasHandle, CusolverDnHandle (RAII)
#include "device/cuda/qpadm_fit_kernels.cuh" // M(fit-4): f4 gather + loo/total + xtau + small-LA launch wrappers
#include "device/cuda/qpgraph_fit_kernels.cuh" // qpGraph: the on-device IDEA-1 fleet launcher + the L3 on-device edge/f3 recovery
#include "device/cuda/pinned_buffer.cuh"    // PinnedRegistryCache (amortized in-place pin for async H2D overlap — P4/L2); PinnedBuffer (persistent D2H staging — P5/d2h-speed)
#include "device/cuda/stream.hpp"           // Stream (RAII, owning non-blocking per-device stream — P2/F1)
#include "device/vram_budget.hpp"           // max_blocks_per_chunk (host-pure VRAM budget; X-5/B5 + X-13/B26)
#include "steppe/config.hpp"                // Precision, kDefaultMantissaBits, kBlockGroupPadBase, kMaxVramUtilizationFraction, kStreamTileBudgetFraction, kFitBudget*
#include "steppe/fstats.hpp"                // F2BlockTensor
#include "device/cuda/cuda_backend.cuh"  // the CudaBackend class declaration (split T0; this TU out-of-lines the method bodies)

namespace steppe::device {

// ===========================================================================
// CudaBackend -- out-of-line method definitions (split T0).
// Class declaration lives in cuda_backend.cuh; bodies below are verbatim (modulo
// a uniform 4-space dedent). Still ONE TU at T0; redistributed across per-
// subsystem TUs in T1..T9.
// ===========================================================================

CudaBackend::CudaBackend(int device_id)
    : device_id_(set_and_return_device(device_id)) {
    // RAII handle, created once ON device_id_ (its cuBLAS context binds here,
    // cuBLAS §2.1.2, because device_id_'s initializer set the device first). Bind
    // the (stream, workspace) invariant ONCE here
    // (architecture.md §12; cleanup X-1/B1): the workspace pins emulated-FP64
    // reproducibility (cuBLAS §2.1.4) and the single statistic stream is bound
    // through `set_stream`, which re-applies the workspace after the
    // `cublasSetStream` reset (cuBLAS §2.4.7). The GEMM routines never touch the
    // stream again, so the workspace survives for every GEMM batch on BOTH the
    // M0 and M4 paths. Order matters: bind the workspace FIRST so the
    // `set_stream` re-apply has it.
    blas_.set_workspace(workspace_.data(), workspace_.bytes());
    blas_.set_stream(stream_.get());
    // cuSOLVER shares the ONE per-device statistic stream (§12 single-stream
    // determinism); no workspace re-apply hazard (CusolverDnHandle::set_stream).
    solver_.set_stream(stream_.get());
}

BackendCapabilities CudaBackend::capabilities() const {
    BackendCapabilities caps;

    // Visible CUDA devices in this process — the SPMG combine fans out over the
    // subset pinned by DeviceConfig::devices (architecture.md §9, §11.4); this is
    // the upper bound the probe saw. A failure here is a real fault (a process
    // with a CUDA backend must be able to enumerate its devices).
    int count = 0;
    STEPPE_CUDA_CHECK(cudaGetDeviceCount(&count));
    caps.device_count = count;

    // Probe THIS backend's device. Save/restore the ambient device so the const
    // probe leaves no cudaSetDevice side effect (it is a query, not a select).
    int entry_device = 0;
    STEPPE_CUDA_CHECK(cudaGetDevice(&entry_device));
    STEPPE_CUDA_CHECK(cudaSetDevice(device_id_));

    // Compute capability (sm_120 ⇒ {12, 0} on the Blackwell box). One sm_120
    // build serves both boxes (architecture.md §0), so this is observability,
    // not a dispatch key (cleanup ⚡ box-role split).
    cudaDeviceProp prop{};
    STEPPE_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
    caps.compute_major = prop.major;
    caps.compute_minor = prop.minor;

    // Total + free VRAM. `cudaMemGetInfo` yields BOTH — the M0/M4 paths
    // (cuda_backend.cu) historically captured only `free` and DISCARDED `total`
    // (cleanup 00-overview §(2).1: "exactly the datum needed"); the probe keeps
    // both. `total` feeds the §11.2 VRAM budget / per-box P_max; `free` is the
    // live headroom (CUDA Runtime API: cudaMemGetInfo writes free then total).
    std::size_t free_b = 0, total_b = 0;
    STEPPE_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    caps.free_vram_bytes = free_b;
    caps.total_vram_bytes = total_b;

    // P2P reachability from THIS device to every OTHER visible device. The
    // capability-tier law (architecture.md §11.4; measured on rtxbox: both
    // directions == 1, byte-exact, 55.6 GB/s): a "yes" enables the device-
    // resident cudaMemcpyPeer combine fast-path; a "no" (EXPECTED on consumer
    // GeForce — P2P driver-disabled) is a tagged degrade to the host-staged
    // fixed-order baseline, NOT a fault. We report can_access_peer == true iff
    // the bound device can reach AT LEAST ONE peer (the combine root needs a
    // peer to pull from; a lone device, or all-isolated devices, ⇒ false ⇒
    // host-staged baseline). The self-pair (peer == device_id_) is skipped:
    // cudaDeviceCanAccessPeer is defined for distinct devices.
    bool can_peer = false;
    for (int peer = 0; peer < count; ++peer) {
        if (peer == device_id_) continue;
        int access = 0;
        // NON-throwing: a probe failure / "cannot access" is an expected degrade
        // (U1; check.cuh CAP-1). On a non-success status `access` is left 0 and
        // the WARN line tags it; we never throw out of the capability probe.
        const cudaError_t s =
            STEPPE_CUDA_WARN(cudaDeviceCanAccessPeer(&access, device_id_, peer));
        if (s == cudaSuccess && access != 0) {
            can_peer = true;
            break;
        }
    }
    caps.can_access_peer = can_peer;

    // EmulatedFp64 honorable on THIS build? Routes through the ONE predicate
    // (emulation_honorable, f2_block_kernel.cu X-6/B2) — true iff the fixed-slice
    // Ozaki tuning API is compiled in (STEPPE_HAVE_EMU_TUNING); false ⇒ an
    // EmulatedFp64 request DEGRADES to native Fp64 with a logged capability tag
    // (never the rejected DYNAMIC ~60-bit trap; architecture.md §12). Probing
    // with a default EmulatedFp64 Precision asks exactly "would an EmulatedFp64
    // run be honored as fixed-slice Ozaki, or downgraded?" — the same predicate
    // the GEMM path consults, so the probe and the compute path can never report
    // different honorability.
    const Precision emu_probe{Precision::Kind::EmulatedFp64,
                              steppe::kDefaultMantissaBits};
    caps.emulated_fp64_honorable = emulation_honorable(emu_probe);

    // Restore the device current at entry (no side effect leaks from the probe).
    STEPPE_CUDA_CHECK(cudaSetDevice(entry_device));
    return caps;
}

void CudaBackend::set_solve_precision(const Precision& precision) {
    solve_precision_ = precision;
}

std::size_t CudaBackend::batched_dispatch_count() const {
    return batched_dispatch_count_;
}

/// Factory for the GPU backend (declared in device/backend_factory.hpp, X-9/B8;
/// architecture.md §9 — backend chosen at build()). Returns the abstract interface
/// so `core` / `Resources` never name the concrete type or touch a CUDA header
/// (architecture.md §4, §8). `device_id` BINDS the backend to one physical CUDA
/// device (the per-device-instance contract, backend.hpp; architecture.md §11.4
/// SPMG — `Resources` passes `DeviceConfig::devices[g]`). The default 0 keeps the
/// single-GPU path and every existing zero-arg call site bound to device 0,
/// unchanged (ZERO regression).
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cuda_backend(int device_id) {
    return std::make_unique<CudaBackend>(device_id);
}

/// Visible CUDA device count (declared CUDA-free in device/backend_factory.hpp, B8).
/// A single `cudaGetDeviceCount` — it does NOT create a context/cuBLAS handle, does
/// NOT allocate the workspace, and does NOT change the current device, so it is the
/// cheap process-global count query `Resources` auto-enumeration + the §9 ordinal
/// validation need (replacing the old throwaway device-0 backend build, resources
/// P1/P5/E5). Returns the same value `capabilities().device_count` reports
/// (cudaGetDeviceCount, cuda_backend.cu capabilities()). A query FAILURE is a real
/// fault (a process with a CUDA backend must enumerate its devices) and throws via
/// STEPPE_CUDA_CHECK; a zero count is returned as 0 (the no-device policy is the
/// caller's, §9).
[[nodiscard]] int visible_device_count() {
    int count = 0;
    STEPPE_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

// Definition of the CUDA-free fault-taxonomy seam declared in device/resources.hpp
// (B2). This TU is the ONLY layer that may inspect the typed device exceptions:
// CudaError / CublasError / CusolverError + their cudaError_t/cublasStatus_t/
// cusolverStatus_t status enums are PRIVATE to steppe_device (cuda/check.cuh, a .cuh),
// so the CUDA-free app cannot dynamic_cast to them — it calls this instead. A genuine
// device ALLOCATION failure (cudaErrorMemoryAllocation == 2, verified against the
// CUDA 13.x Runtime API cudaError_t enum; CUBLAS_STATUS_ALLOC_FAILED /
// CUSOLVER_STATUS_ALLOC_FAILED, the same enumerators cuda/check.cuh already renders)
// maps to Status::DeviceOom → kExitDeviceOom (3); every other exception (including a
// non-alloc CUDA/cuBLAS/cuSOLVER fault, or a host std::bad_alloc — host RAM, not
// device VRAM) yields std::nullopt so the app keeps its catch-all kExitRuntimeError
// (5). dynamic_cast is well-defined: one statically-linked executable ⇒ one typeinfo
// per type. noexcept (RTTI + integral compares only; no allocation, no throw).
[[nodiscard]] std::optional<Status> device_fault_status(
    const std::exception& e) noexcept {
    if (const auto* ce = dynamic_cast<const CudaError*>(&e)) {
        return ce->status() == cudaErrorMemoryAllocation
                   ? std::optional<Status>(Status::DeviceOom)
                   : std::nullopt;
    }
    if (const auto* be = dynamic_cast<const CublasError*>(&e)) {
        return be->status() == CUBLAS_STATUS_ALLOC_FAILED
                   ? std::optional<Status>(Status::DeviceOom)
                   : std::nullopt;
    }
    if (const auto* se = dynamic_cast<const CusolverError*>(&e)) {
        return se->status() == CUSOLVER_STATUS_ALLOC_FAILED
                   ? std::optional<Status>(Status::DeviceOom)
                   : std::nullopt;
    }
    return std::nullopt;
}

}  // namespace steppe::device

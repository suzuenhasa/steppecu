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

namespace {

// ---- Shared AT2 res$rankdrop nested-table fill ([7.1] dedup) -----------------
// rank_sweep (writing rs.rd_*) and assemble_result (writing res.rankdrop_*) built the
// nested table with the SAME descending r=rmax-k loop, the SAME dof/chisq nested diff +
// pchisq_upper(cd, dd), and the SAME INT_MIN / quiet_NaN NA tail for the last row (rank
// 0). They differed ONLY by destination field names + the source arrays. Single-homed
// here so the AT2 ranktest.cpp mirror cannot drift between the single-model sweep and
// the batched assemble. The per-rank source arrays `dof`/`chisq`/`p` are indexed by
// rank r in [0, rmax]; the out arrays are written by ROW k in [0, n) (n = rmax+1, the
// row for rank rmax-k). PARITY: byte-identical to the two former inline copies — same
// loop order, same INT subtraction, same double subtraction, same pchisq_upper call.
void fill_rankdrop(int rmax,
                   const std::vector<int>& dof, const std::vector<double>& chisq,
                   const std::vector<double>& p,
                   std::vector<int>& rd_f4rank, std::vector<int>& rd_dof,
                   std::vector<int>& rd_dofdiff, std::vector<double>& rd_chisq,
                   std::vector<double>& rd_p, std::vector<double>& rd_chisqdiff,
                   std::vector<double>& rd_p_nested) {
    const std::size_t n = static_cast<std::size_t>(rmax) + 1;
    rd_f4rank.resize(n); rd_dof.resize(n); rd_chisq.resize(n);
    rd_p.resize(n); rd_dofdiff.resize(n);
    rd_chisqdiff.resize(n); rd_p_nested.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        const int r = rmax - static_cast<int>(k);  // this row's rank (descending)
        rd_f4rank[k] = r;
        rd_dof[k] = dof[static_cast<std::size_t>(r)];
        rd_chisq[k] = chisq[static_cast<std::size_t>(r)];
        rd_p[k] = p[static_cast<std::size_t>(r)];
        if (r - 1 >= 0) {  // nested diff to the next-lower rank (r-1) — NATIVE
            const int dd = dof[static_cast<std::size_t>(r - 1)] - dof[static_cast<std::size_t>(r)];
            const double cd = chisq[static_cast<std::size_t>(r - 1)] -
                              chisq[static_cast<std::size_t>(r)];
            rd_dofdiff[k] = dd;
            rd_chisqdiff[k] = cd;
            rd_p_nested[k] = core::internal::pchisq_upper(cd, dd);
        } else {  // the last row (rank 0): NA
            rd_dofdiff[k] = INT_MIN;
            rd_chisqdiff[k] = std::numeric_limits<double>::quiet_NaN();
            rd_p_nested[k] = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

}  // namespace

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

JackknifeCov CudaBackend::jackknife_cov(const F4Blocks& x,
                                         std::span<const int> block_sizes,
                                         double fudge,
                                         const Precision& precision) {
    // PRECISION POLICY (unified with the f2 precompute; fit-engine.md §1.4). The
    // passed `precision` (default EmulatedFp64{40}) NOW governs the well-
    // conditioned covariance SYRK below — it engages the SAME
    // `engage_f2_precision` + `emulation_honorable` path the f2 GEMMs use, with
    // automatic native fallback when the build/device cannot honor emulation. The
    // two stages here that `precision` does NOT govern, by the §12 carve-outs:
    //   * the xtau CENTERING kernel — catastrophic-cancellation (loo − est),
    //     held native ALWAYS (emulation faithfully forms a product but cannot
    //     recover bits a prior subtraction annihilated), exactly the f2-numerator
    //     carve-out;
    //   * the ill-conditioned SPD inverse (potrf/potri) — pinned native FP64 by
    //     the §12 conditioning rule; the d6d3cbb cuSOLVER promotion seam
    //     (`solve_precision_` → `engage_solver_precision`) is live but FALLS BACK
    //     to native because CUDA 13.0 / cuSOLVER 12.0 exposes no FP64-emulated
    //     cuSOLVER math mode. `precision` does not drive that solve.
    guard_device();

    const int m = x.nl * x.nr;
    const int nb = x.n_block;
    JackknifeCov out;
    out.m = m;
    if (m <= 0 || nb <= 0) { out.status = Status::Ok; return out; }
    const std::size_t m_sz = static_cast<std::size_t>(m);

    // n = Σ block_sizes.
    long long n_ll = 0;
    for (int b = 0; b < nb; ++b) n_ll += block_sizes[static_cast<std::size_t>(b)];
    const double n = static_cast<double>(n_ll);

    // Upload loo / est / tot_line / block_sizes; form xtau (col-major k + m*b).
    DeviceBuffer<double> dLoo(m_sz * static_cast<std::size_t>(nb));
    DeviceBuffer<double> dEst(m_sz);
    DeviceBuffer<double> dTotLine(m_sz);
    DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb));
    DeviceBuffer<double> dXtau(m_sz * static_cast<std::size_t>(nb));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dLoo.data(), x.x_loo.data(),
                                      m_sz * static_cast<std::size_t>(nb) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dEst.data(), x.x_total.data(),
                                      m_sz * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dTotLine.data(), tot_line_.data(),
                                      m_sz * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), block_sizes.data(),
                                      static_cast<std::size_t>(nb) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    launch_f4_xtau(dLoo.data(), dEst.data(), dTotLine.data(), dBlockSizes.data(),
                   m, nb, n, dXtau.data(), stream_.get());

    // Q = xtau·xtauᵀ / nb (UNFUDGED, symmetric m×m), cublasDsyrk LOWER, OP_N
    // (n=m, k=nb, A=dXtau lda=m). The well-conditioned covariance SYRK now ENGAGES
    // the passed `precision` exactly like the f2 GEMMs: default EmulatedFp64{40}
    // (Ozaki fixed-slice tensor-core path), automatic native fallback when not
    // honorable. cublasDsyrk is the legacy API (no per-call compute-type arg), so
    // emulation is driven by the handle's MATH MODE — set by `engage_f2_precision`
    // (the single source of the f2 precision engagement). The MathModeScope
    // brackets the change: it captures the handle's current math mode, lets
    // engage_f2_precision apply the emulated (or PEDANTIC-native fallback) mode for
    // the SYRK, then RESTORES the captured mode at scope exit — so this never leaks
    // its mode into the next solve on the shared handle (the §12 determinism
    // contract; mirrors the f2 scoped pattern). The handle's stream + emulated-FP64
    // determinism workspace are bound ONCE in the ctor and are NOT touched here.
    DeviceBuffer<double> dQ(m_sz * m_sz);
    const double alpha = 1.0 / static_cast<double>(nb);
    const double beta = 0.0;
    {
        const MathModeScope syrk_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas_.get(), precision);
        CUBLAS_CHECK(cublasDsyrk(blas_.get(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                 m, nb, &alpha, dXtau.data(), m, &beta, dQ.data(), m));
    }  // syrk_mode_scope restores the prior math mode here (no leak to the SPD inverse)
    launch_symmetrize_lower_to_full(dQ.data(), m, stream_.get());
    out.Q.assign(m_sz * m_sz, 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.Q.data(), dQ.data(), m_sz * m_sz * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    // tr(Q) (host; m tiny) → fudged copy Qf = Q; diag += fudge*tr.
    double tr = 0.0;
    for (int k = 0; k < m; ++k) tr += out.Q[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(k)];
    DeviceBuffer<double> dQf(m_sz * m_sz);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQf.data(), dQ.data(), m_sz * m_sz * sizeof(double),
                                      cudaMemcpyDeviceToDevice, stream_.get()));
    launch_add_fudge_diag(dQf.data(), m, fudge, tr, stream_.get());

    // Qinv = inverse(Qf), cusolverDn Cholesky potrf + potri (the §12 SPD path).
    // potrf factors in place; potri overwrites with the inverse (filling LOWER);
    // symmetrize to full. devInfo>0 ⇒ NonSpdCovariance.
    //
    // PROMOTION SEAM (ROADMAP §6). Engage the cuSOLVER math mode for the SOLVE
    // through `engage_solver_precision`, routed via the ONE `emulation_honorable`
    // predicate (the f2 path's single source). DEFAULT `solve_precision_` is
    // native Fp64 ⇒ honorable==false ⇒ the scope sets native CUSOLVER_DEFAULT_MATH
    // and restores it at scope exit (a real, behavior-preserving cusolverDnSet/Get
    // round-trip; the golden parity is unchanged). A future per-stage policy can
    // set `solve_precision_` to EmulatedFp64{bits} to PROMOTE this ill-conditioned
    // inverse to the emulated tensor-core path — validated per stage against the
    // native oracle at S8 scale before being made the default. The scope is local
    // so it never leaks the mode into the next (native, oracle) solve on the shared
    // handle. The exploratory parity measurement (tests) flips `solve_precision_`.
    const CusolverMathModeScope solve_scope =
        engage_solver_precision(solver_.get(), solve_precision_, &emulation_honorable);
    DeviceBuffer<int> dInfo(1);
    int lwork_f = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver_.get(), CUBLAS_FILL_MODE_LOWER,
                                               m, dQf.data(), m, &lwork_f));
    // POOLED potrf/potri workspace (B5): one persistent solver_work_ grown to
    // lwork_f (monotonic, never-shrink), reused by BOTH the potrf below and the
    // potri further down (same lwork_f) instead of two per-call allocations. Same
    // bytes fed to each routine ⇒ bit-identical (§12); queried AFTER the math-mode
    // scope so lwork_f reflects the engaged mode (CUDA 13.x cuSOLVER).
    const std::size_t lwork_need = static_cast<std::size_t>(lwork_f > 0 ? lwork_f : 1);
    if (solver_work_.size() < lwork_need) solver_work_ = DeviceBuffer<double>(lwork_need);
    CUSOLVER_CHECK(cusolverDnDpotrf(solver_.get(), CUBLAS_FILL_MODE_LOWER, m,
                                    dQf.data(), m, solver_work_.data(), lwork_f, dInfo.data()));
    int info = 0;
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&info, dInfo.data(), sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    if (info != 0) {  // not SPD / singular pivot → domain outcome
        out.status = Status::NonSpdCovariance;
        return out;
    }
    CUSOLVER_CHECK(cusolverDnDpotri(solver_.get(), CUBLAS_FILL_MODE_LOWER, m,
                                    dQf.data(), m, solver_work_.data(), lwork_f, dInfo.data()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&info, dInfo.data(), sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    if (info != 0) {
        out.status = Status::NonSpdCovariance;
        return out;
    }
    launch_symmetrize_lower_to_full(dQf.data(), m, stream_.get());
    out.Qinv.assign(m_sz * m_sz, 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.Qinv.data(), dQf.data(), m_sz * m_sz * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    out.status = Status::Ok;
    return out;
}

JackknifeDiag CudaBackend::jackknife_diag(const F4Blocks& x,
                                           std::span<const int> block_sizes,
                                           const Precision& precision) {
    (void)precision;  // native FP64 (the xtau centering + per-item sum-of-squares)
    guard_device();

    const int m = x.nl * x.nr;
    const int nb = x.n_block;
    JackknifeDiag out;
    out.m = m;
    if (m <= 0 || nb <= 0) { out.status = Status::Ok; return out; }
    const std::size_t m_sz = static_cast<std::size_t>(m);

    // n = Σ block_sizes.
    long long n_ll = 0;
    for (int b = 0; b < nb; ++b) n_ll += block_sizes[static_cast<std::size_t>(b)];
    const double n = static_cast<double>(n_ll);

    // Upload loo / est / tot_line / block_sizes; form xtau (col-major k + m*b) — the
    // EXACT jackknife_cov prologue, minus the SYRK + Cholesky inverse below it.
    DeviceBuffer<double> dLoo(m_sz * static_cast<std::size_t>(nb));
    DeviceBuffer<double> dEst(m_sz);
    DeviceBuffer<double> dTotLine(m_sz);
    DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb));
    DeviceBuffer<double> dXtau(m_sz * static_cast<std::size_t>(nb));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dLoo.data(), x.x_loo.data(),
                                      m_sz * static_cast<std::size_t>(nb) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dEst.data(), x.x_total.data(),
                                      m_sz * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dTotLine.data(), tot_line_.data(),
                                      m_sz * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), block_sizes.data(),
                                      static_cast<std::size_t>(nb) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    launch_f4_xtau(dLoo.data(), dEst.data(), dTotLine.data(), dBlockSizes.data(),
                   m, nb, n, dXtau.data(), stream_.get());

    // The DIAGONAL reduce: var[k] = (1/nb)·Σ_b xtau[k+m*b]² (O(m·nb) work, O(m) memory).
    DeviceBuffer<double> dVar(m_sz);
    launch_f4_diag_var(dXtau.data(), m, nb, dVar.data(), stream_.get());

    out.var.assign(m_sz, 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.var.data(), dVar.data(), m_sz * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    out.status = Status::Ok;
    return out;
}

bool CudaBackend::model_fits_small_path(int nl, int nr, int r) {
    return core::qpadm::model_fits_small_path(nl, nr, r);
}

bool CudaBackend::gesvdj_applicable(int nl, int nr) {
    return nl <= kGesvdjMaxDim && nr <= kGesvdjMaxDim;
}

CudaBackend::SvdScratchSizes CudaBackend::large_svd_scratch_sizes(int nl, int nr) {
    SvdScratchSizes sz;
    const int rows = (nr >= nl) ? nr : nl;
    const int cols = (nr >= nl) ? nl : nr;  // rows>=cols ✓ (the §1.3 orientation)
    sz.s  = static_cast<std::size_t>(cols);
    sz.u  = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);  // == nl*nr
    sz.vt = static_cast<std::size_t>(cols) * static_cast<std::size_t>(cols);
    sz.a2 = (nr >= nl) ? 0
                       : static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr);
    solver_.set_stream(stream_.get());  // bufferSize is data-independent; stream irrelevant
    if (gesvdj_applicable(nl, nr)) {
        // RAII gesvdj params (group-14 [14.5]): the throwing bufferSize between a
        // bare create/destroy would leak the handle on unwind; GesvdjInfo frees it.
        GesvdjInfo params;
        // bufferSize ignores the device pointers (VERIFIED, see SvdScratchSizes),
        // so nullptr A/S/U/VT here is fine — we only want lwork for the shape.
        CUSOLVER_CHECK(cusolverDnDgesvdj_bufferSize(
            solver_.get(), CUSOLVER_EIG_MODE_VECTOR, /*econ=*/1, rows, cols,
            nullptr, rows, nullptr, nullptr, rows, nullptr, cols, &sz.lwork,
            params.get()));
    } else {
        CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(solver_.get(), rows, cols, &sz.lwork));
    }
    if (sz.lwork <= 0) sz.lwork = 1;
    return sz;
}

void CudaBackend::large_svd_V(const double* dXmat, int nl, int nr, int r,
                 double* dVout, double* dXt,
                 double* sS, double* sU, double* sVt, double* sA2,
                 int* sInfo, double* sWork, int lwork, cudaStream_t stream) {
    if (r <= 0) return;
    solver_.set_stream(stream);
    if (nr >= nl) {
        // Xt = transpose(xmat): nr×nl, rows(=nr) >= cols(=nl). U(Xt) == V(xmat).
        launch_transpose_small(dXmat, nl, nr, dXt, stream);
        const int rows = nr, cols = nl;  // rows>=cols ✓
        if (gesvdj_applicable(nl, nr)) {
            // gesvdj (one-sided Jacobi, single matrix), economy U (rows×cols).
            // RAII gesvdj params (group-14 [14.5]): the throwing gesvdj solve
            // between a bare create/destroy would leak the handle on unwind.
            GesvdjInfo params;
            CUSOLVER_CHECK(cusolverDnDgesvdj(
                solver_.get(), CUSOLVER_EIG_MODE_VECTOR, /*econ=*/1, rows, cols,
                dXt, rows, sS, sU, rows, sVt, cols,
                sWork, lwork, sInfo, params.get()));
        } else {
            // gesvd: jobu='S' (economy U), jobvt='N' (V of Xt unused). Descending.
            CUSOLVER_CHECK(cusolverDnDgesvd(
                solver_.get(), 'S', 'N', rows, cols, dXt, rows, sS,
                sU, rows, /*VT*/nullptr, cols, sWork, lwork,
                /*rwork*/nullptr, sInfo));
        }
        // sU is nr×cols (rows×cols) col-major; copy its leading r columns to dVout
        // (nr×r col-major). Contiguous prefix since lda(sU)=rows=nr.
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(
            dVout, sU,
            static_cast<std::size_t>(nr) * static_cast<std::size_t>(r) * sizeof(double),
            cudaMemcpyDeviceToDevice, stream));
    } else {
        // nl>nr: hand xmat itself (nl×nr, rows>=cols). V(xmat) = VT^T leading r rows.
        // cuSOLVER gesvd/gesvdj OVERWRITE A, so copy dXmat into a non-const scratch
        // (sA2, sized nl*nr) — the const dXmat must survive for the seed/ALS kernels.
        const int rows = nl, cols = nr;  // rows>=cols ✓
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(
            sA2, dXmat,
            static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr) * sizeof(double),
            cudaMemcpyDeviceToDevice, stream));
        if (gesvdj_applicable(nl, nr)) {
            // RAII gesvdj params (group-14 [14.5]): the throwing gesvdj solve
            // between a bare create/destroy would leak the handle on unwind.
            GesvdjInfo params;
            CUSOLVER_CHECK(cusolverDnDgesvdj(
                solver_.get(), CUSOLVER_EIG_MODE_VECTOR, /*econ=*/1, rows, cols,
                sA2, rows, sS, sU, rows, sVt, cols,
                sWork, lwork, sInfo, params.get()));
        } else {
            CUSOLVER_CHECK(cusolverDnDgesvd(
                solver_.get(), 'N', 'S', rows, cols, sA2, rows, sS,
                /*U*/nullptr, rows, sVt, cols, sWork, lwork,
                /*rwork*/nullptr, sInfo));
        }
        // V[:,p] = VT[p,:] (VT is cols×cols col-major): dVout[j + nr*p] = sVt[p + cols*j].
        launch_transpose_small(sVt, cols, cols, dXt, stream);
        // dXt now holds V (cols×cols col-major); copy its leading r columns to dVout.
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(
            dVout, dXt,
            static_cast<std::size_t>(nr) * static_cast<std::size_t>(r) * sizeof(double),
            cudaMemcpyDeviceToDevice, stream));
    }
}

void CudaBackend::large_svd_V(const double* dXmat, int nl, int nr, int r,
                 double* dVout, double* dXt, cudaStream_t stream) {
    if (r <= 0) return;
    const SvdScratchSizes sz = large_svd_scratch_sizes(nl, nr);
    // POOLED gesvd/gesvdj scratch (B5): grow each member arena to its per-shape
    // count (monotonic, never-shrink). The bytes fed to gesvd/gesvdj are UNCHANGED
    // (data-independent bufferSize; the scratch is fully overwritten) ⇒ bit-
    // identical (§12). solver_work_ doubles as the gesvd `lwork` buffer, shared
    // with the potrf/potri path (sequential, single-stream, one backend per device).
    // svd_a2_: sz.a2==0 for nr>=nl ⇒ the grow is skipped and sA2 is unread in that
    // branch (matches the prior DeviceBuffer(0)==nullptr behavior); nl>nr grows it.
    // svd_info_: REQUIRED non-null device out-arg for cusolverDnDgesvd/Dgesvdj
    // (info<0 = bad param; gesvdj info>0 = non-converged Jacobi). INTENTIONAL
    // DISCARD on this off-bit-parity large (NRBIG) path — not D2H-copied/checked; a
    // non-converged SVD would flow into seed/ALS. Cannot drop the arg. ([3.4])
    const std::size_t lwork_need = static_cast<std::size_t>(sz.lwork);
    if (svd_s_.size()       < sz.s)       svd_s_       = DeviceBuffer<double>(sz.s);
    if (svd_u_.size()       < sz.u)       svd_u_       = DeviceBuffer<double>(sz.u);
    if (svd_vt_.size()      < sz.vt)      svd_vt_      = DeviceBuffer<double>(sz.vt);
    if (svd_a2_.size()      < sz.a2)      svd_a2_      = DeviceBuffer<double>(sz.a2);
    if (svd_info_.size()    < sz.info)    svd_info_    = DeviceBuffer<int>(sz.info);
    if (solver_work_.size() < lwork_need) solver_work_ = DeviceBuffer<double>(lwork_need);
    large_svd_V(dXmat, nl, nr, r, dVout, dXt,
                svd_s_.data(), svd_u_.data(), svd_vt_.data(), svd_a2_.data(),
                svd_info_.data(), solver_work_.data(), sz.lwork, stream);
}

std::size_t CudaBackend::large_dbl_scratch(int nl, int nr, int r) {
    const std::size_t m = static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr);
    const std::size_t t = static_cast<std::size_t>(nl > nr ? nl : nr) *
                          static_cast<std::size_t>(r > 0 ? r : 1);
    const std::size_t als = m + m * t + t * t + t + t + t * t + t;
    const std::size_t weight_chisq =
                          static_cast<std::size_t>(nl) * static_cast<std::size_t>(nl)
                          + static_cast<std::size_t>(nl)
                          + static_cast<std::size_t>(nl) * static_cast<std::size_t>(nl)
                          + static_cast<std::size_t>(nl) + m;
    return als > weight_chisq ? als : weight_chisq;
}

std::size_t CudaBackend::large_int_scratch(int nl, int nr, int r) {
    const std::size_t t = static_cast<std::size_t>(nl > nr ? nl : nr) *
                          static_cast<std::size_t>(r > 0 ? r : 1);
    return (t > static_cast<std::size_t>(nl) ? t : static_cast<std::size_t>(nl));
}

std::size_t CudaBackend::large_loo_dbl_refit(int nl, int nr, int r) {
    const std::size_t m  = static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr);
    const std::size_t rr = static_cast<std::size_t>(r > 0 ? r : 1);
    const std::size_t a  = static_cast<std::size_t>(nl) * rr;          // A[nl*r]
    const std::size_t bb = rr * static_cast<std::size_t>(nr);          // B[r*nr]
    return m + a + bb + large_dbl_scratch(nl, nr, r);
}

void CudaBackend::large_fit_one(const double* dXmat, const double* dQinv, int nl, int nr, int r,
                   double fudge, int als_iters, double* dA, double* dB, double* dW,
                   double* dchisq, int* dStatus, double* dVout, double* dXt,
                   double* dScratch, int* dIntScratch, cudaStream_t stream) {
    if (r > 0) {
        large_svd_V(dXmat, nl, nr, r, dVout, dXt, stream);
        launch_qpadm_seed_from_V(dXmat, dVout, nl, nr, r, dA, dB, stream);
        launch_qpadm_als_large(dXmat, dQinv, nl, nr, r, fudge, als_iters,
                               dA, dB, dScratch, dIntScratch, stream);
    }
    launch_qpadm_weights_chisq_large(dXmat, dQinv, dA, dB, nl, nr, r,
                                     dW, dchisq, dStatus, dScratch, dIntScratch, stream);
}

bool CudaBackend::provides_rank_sweep() const { return true; }

RankSweep CudaBackend::rank_sweep(const F4Blocks& x,
                                   const JackknifeCov& cov,
                                   double alpha,
                                   const QpAdmOptions& opts,
                                   const Precision& precision) {
    (void)precision;  // native FP64 by carve-out (§4) — match the oracle exactly
    guard_device();
    RankSweep rs;
    const int nl = x.nl, nr = x.nr, m = nl * nr;
    const int rmax = (nl < nr ? nl : nr) - 1;
    rs.svd_path = gesvdj_applicable(nl, nr) ? 1 : 2;  // SAME predicate as large_svd_V's dispatch (always set)
    if (rmax < 0) {  // a degenerate (0-row/0-col) model — no candidate rank
        rs.status = Status::RankDeficient;
        return rs;
    }
    // The widest fit in the sweep is rank rmax — it decides the path (small vs
    // large) for the whole sweep (the bit-parity envelope is monotone in r).
    const bool small = model_fits_small_path(nl, nr, rmax);
    if (m <= 0) { rs.status = Status::Ok; return rs; }

    // ---- upload x_total + Qinv ONCE; build dXmat ONCE (V/A/B reread per r) ----
    DeviceBuffer<double> dTotal(static_cast<std::size_t>(m));
    DeviceBuffer<double> dXmat(static_cast<std::size_t>(m));
    DeviceBuffer<double> dQinv(static_cast<std::size_t>(m) * static_cast<std::size_t>(m));
    const std::size_t cap = static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr);  // ≥ nl*r and r*nr for r≤rmax
    DeviceBuffer<double> dA(cap > 0 ? cap : 1);
    DeviceBuffer<double> dB(cap > 0 ? cap : 1);
    DeviceBuffer<double> dW(static_cast<std::size_t>(nl > 0 ? nl : 1));
    // One chisq/status slot PER rank (slot r ← the rank-r fit). Lets the rmax+1
    // fits enqueue back-to-back on stream_ and the host pay ONE D2H after the
    // loop, instead of a per-rank D2H+sync ladder (each launch→sync→launch
    // stall blocked rank r+1's seed on rank r's tiny D2H draining). dW stays
    // size nl — it is overwritten and never read back in the sweep.
    DeviceBuffer<double> dchisq(static_cast<std::size_t>(rmax) + 1);
    DeviceBuffer<int> dStatus(static_cast<std::size_t>(rmax) + 1);
    // LARGE-path VRAM scratch (sized for the widest rank rmax; reused every r).
    // Allocated even on the small path (size 1) — trivial and keeps the code one
    // shape; the large branch is the only consumer.
    DeviceBuffer<double> dVout(small ? 1 : static_cast<std::size_t>(nr) * static_cast<std::size_t>(rmax > 0 ? rmax : 1));
    DeviceBuffer<double> dXt(small ? 1 : static_cast<std::size_t>(nr) * static_cast<std::size_t>(nl > 0 ? nl : 1));
    DeviceBuffer<double> dScratch(small ? 1 : large_dbl_scratch(nl, nr, rmax));
    DeviceBuffer<int>    dIntScratch(small ? 1 : large_int_scratch(nl, nr, rmax));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dTotal.data(), x.x_total.data(),
                                      static_cast<std::size_t>(m) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQinv.data(), cov.Qinv.data(),
                                      static_cast<std::size_t>(m) * static_cast<std::size_t>(m) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    launch_qpadm_xmat_from_rowmajor(dTotal.data(), nl, nr, dXmat.data(), stream_.get());

    // ---- per-rank sweep (chisq via the ALS-refined rank-r fit, on the GPU) ----
    const std::size_t n = static_cast<std::size_t>(rmax) + 1;
    rs.chisq.assign(n, 0.0);
    rs.dof.assign(n, 0);
    rs.p.assign(n, 0.0);
    bool degenerate = false;
    for (int r = 0; r <= rmax; ++r) {
        // seed → ALS (r>0) then the constrained weight solve + chisq — exactly
        // the M(fit-4) gls_weights device flow (reused, no new chisq math).
        if (small) {
            if (r > 0) {
                launch_qpadm_seed_ab(dXmat.data(), nl, nr, r, dA.data(), dB.data(), stream_.get());
                launch_qpadm_als(dXmat.data(), dQinv.data(), nl, nr, r, opts.fudge,
                                 opts.als_iterations, dA.data(), dB.data(), stream_.get());
            }
            launch_qpadm_weights_chisq(dXmat.data(), dQinv.data(), dA.data(), dB.data(),
                                       nl, nr, r, dW.data(),
                                       dchisq.data() + r, dStatus.data() + r,
                                       stream_.get());
        } else {  // LARGE path (cuSOLVER SVD seed + VRAM-scratch ALS + weight/chisq).
            large_fit_one(dXmat.data(), dQinv.data(), nl, nr, r, opts.fudge,
                          opts.als_iterations, dA.data(), dB.data(), dW.data(),
                          dchisq.data() + r, dStatus.data() + r, dVout.data(), dXt.data(),
                          dScratch.data(), dIntScratch.data(), stream_.get());
        }
        // NO per-rank D2H/sync — slot r is written; the host reads all slots once
        // after the loop (below), so the fits stay back-to-back on stream_.
    }
    // ---- ONE D2H of both [rmax+1] arrays + one sync (same-stream issue order
    //      guarantees every per-rank chisq/status write has completed) ----
    std::vector<double> chisq_h(n);
    std::vector<int> status_h(n);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(status_h.data(), dStatus.data(),
                                      n * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(chisq_h.data(), dchisq.data(),
                                      n * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    for (int r = 0; r <= rmax; ++r) {
        const std::size_t rr = static_cast<std::size_t>(r);
        if (status_h[rr] != 0) degenerate = true;
        rs.chisq[rr] = chisq_h[rr];
        rs.dof[rr] = core::qpadm::qpadm_dof(nl, nr, r);
        rs.p[rr] = core::internal::pchisq_upper(chisq_h[rr], rs.dof[rr]);
    }

    // ---- AT2 res$rankdrop nested table (rows: rank rmax, rmax-1, ..., 0) ----
    // Shared fill_rankdrop ([7.1] dedup): source = the per-rank rs.dof/chisq/p; dest
    // = the rs.rd_* field set. Byte-identical to the former inline copy (§12).
    fill_rankdrop(rmax, rs.dof, rs.chisq, rs.p,
                  rs.rd_f4rank, rs.rd_dof, rs.rd_dofdiff, rs.rd_chisq,
                  rs.rd_p, rs.rd_chisqdiff, rs.rd_p_nested);

    // ---- f4rank: the smallest non-rejected rank (p(r) > alpha, ASCENDING) ----
    rs.f4rank = rmax;  // fallback: the highest rank if all rejected
    for (int r = 0; r <= rmax; ++r) {
        if (rs.p[static_cast<std::size_t>(r)] > alpha) { rs.f4rank = r; break; }
    }

    // ---- rank_Q: numerical rank of the (unfudged) covariance Q (m×m) ----
    // Observability only (the golden model_well_determined.rank_Q diagnostic, NOT
    // gated; it is a derived property of Q, off the rank-test math path). L1: moved
    // ON-DEVICE (was a host core::jacobi_svd). launch_qpadm_rank_via_jacobi runs the
    // SAME one-sided-Jacobi sweep dev_jacobi_svd_V transliterates over VRAM scratch
    // and counts the singular values above smax*m*eps — BIT-IDENTICAL to the
    // CpuBackend oracle (eps passed from the host so the constant matches; the
    // V-accumulation is dropped, irrelevant to sigma). GPU rank_Q == CPU rank_Q
    // exactly (the localizer holds). Native FP64 (the §4 SVD carve-out).
    if (!cov.Q.empty() && cov.m == m && m > 0) {
        DeviceBuffer<double> dQfull(static_cast<std::size_t>(m) * static_cast<std::size_t>(m));
        DeviceBuffer<double> dRankScratch(static_cast<std::size_t>(m) * static_cast<std::size_t>(m)
                                          + static_cast<std::size_t>(m));  // W[m*m] | sigma[m]
        DeviceBuffer<int> dRankOrder(static_cast<std::size_t>(m));          // order[m]
        DeviceBuffer<int> dRank(1);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQfull.data(), cov.Q.data(),
                                          static_cast<std::size_t>(m) * static_cast<std::size_t>(m)
                                              * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        launch_qpadm_rank_via_jacobi(dQfull.data(), m,
                                     std::numeric_limits<double>::epsilon(),
                                     dRankScratch.data(), dRankOrder.data(),
                                     dRank.data(), stream_.get());
        int rk_h = m;
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(&rk_h, dRank.data(), sizeof(int),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        rs.rank_Q = rk_h;
    } else {
        rs.rank_Q = m;
    }

    rs.status = degenerate ? Status::RankDeficient
                           : (cov.status != Status::Ok ? cov.status : Status::Ok);
    return rs;
}

GlsWeights CudaBackend::gls_weights(const F4Blocks& x,
                                     const JackknifeCov& cov,
                                     int r,
                                     const QpAdmOptions& opts,
                                     const Precision& precision) {
    (void)precision;
    guard_device();
    GlsWeights gw;
    gw.r = r;
    const int nl = x.nl, nr = x.nr, m = nl * nr;
    const bool small = model_fits_small_path(nl, nr, r);
    gw.A.assign(static_cast<std::size_t>(nl) * static_cast<std::size_t>(r), 0.0);
    gw.B.assign(static_cast<std::size_t>(r) * static_cast<std::size_t>(nr), 0.0);
    gw.w.assign(static_cast<std::size_t>(nl), 0.0);
    if (m <= 0 || nl <= 0) { gw.status = Status::Ok; return gw; }

    DeviceBuffer<double> dTotal(static_cast<std::size_t>(m));
    DeviceBuffer<double> dXmat(static_cast<std::size_t>(m));
    DeviceBuffer<double> dQinv(static_cast<std::size_t>(m) * static_cast<std::size_t>(m));
    DeviceBuffer<double> dA(static_cast<std::size_t>(nl) * static_cast<std::size_t>(r > 0 ? r : 1));
    DeviceBuffer<double> dB(static_cast<std::size_t>(r > 0 ? r : 1) * static_cast<std::size_t>(nr));
    DeviceBuffer<double> dW(static_cast<std::size_t>(nl));
    DeviceBuffer<double> dchisq(1);
    DeviceBuffer<int> dStatus(1);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dTotal.data(), x.x_total.data(),
                                      static_cast<std::size_t>(m) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQinv.data(), cov.Qinv.data(),
                                      static_cast<std::size_t>(m) * static_cast<std::size_t>(m) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    launch_qpadm_xmat_from_rowmajor(dTotal.data(), nl, nr, dXmat.data(), stream_.get());
    if (small) {
        if (r > 0) {
            launch_qpadm_seed_ab(dXmat.data(), nl, nr, r, dA.data(), dB.data(), stream_.get());
            launch_qpadm_als(dXmat.data(), dQinv.data(), nl, nr, r, opts.fudge,
                             opts.als_iterations, dA.data(), dB.data(), stream_.get());
        }
        launch_qpadm_weights_chisq(dXmat.data(), dQinv.data(), dA.data(), dB.data(),
                                   nl, nr, r, dW.data(), dchisq.data(), dStatus.data(),
                                   stream_.get());
    } else {  // LARGE path (drives the NRBIG popdrop reduced fits).
        DeviceBuffer<double> dVout(static_cast<std::size_t>(nr) * static_cast<std::size_t>(r > 0 ? r : 1));
        DeviceBuffer<double> dXt(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nl > 0 ? nl : 1));
        DeviceBuffer<double> dScratch(large_dbl_scratch(nl, nr, r));
        DeviceBuffer<int>    dIntScratch(large_int_scratch(nl, nr, r));
        large_fit_one(dXmat.data(), dQinv.data(), nl, nr, r, opts.fudge,
                      opts.als_iterations, dA.data(), dB.data(), dW.data(),
                      dchisq.data(), dStatus.data(), dVout.data(), dXt.data(),
                      dScratch.data(), dIntScratch.data(), stream_.get());
    }
    int status_i = 0;
    double chisq = 0.0;
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&status_i, dStatus.data(), sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&chisq, dchisq.data(), sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(gw.w.data(), dW.data(),
                                      static_cast<std::size_t>(nl) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    if (r > 0) {
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(gw.A.data(), dA.data(),
                                          gw.A.size() * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(gw.B.data(), dB.data(),
                                          gw.B.size() * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
    }
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    gw.chisq = chisq;
    // NOTE: the matching kernel EMIT sites (qpadm_fit_kernels.cu small/large/loo)
    // still emit the bare 6 — they are group-7 device-side leftovers (DEFERRED) and
    // adopt kQpStatusRankDeficient when group-7 runs. Host decode uses the symbol now.
    if (status_i == core::qpadm::kQpStatusRankDeficient) { gw.status = Status::RankDeficient; return gw; }
    gw.status = Status::Ok;
    return gw;
}

std::vector<double> CudaBackend::gls_weights_loo_batched(
    const F4Blocks& x, const JackknifeCov& cov, int r,
    const QpAdmOptions& opts, const Precision& precision) {
    guard_device();
    const int nl = x.nl, nb = x.n_block;
    const int m = nl * x.nr;
    std::vector<double> wmat(static_cast<std::size_t>(nb < 0 ? 0 : nb) *
                             static_cast<std::size_t>(nl), 0.0);
    if (m <= 0 || nb <= 0 || nl <= 0) return wmat;
    // Populate the resident dWmat (the shared producer; M7 carves this out so the
    // SE reduction in se_from_wmat can consume the SAME resident dWmat WITHOUT a
    // D2H bounce). Here (the backend-primitive caller) we keep the D2H — this seam
    // is no longer on the se_from_loo hot path, but stays as a reusable primitive.
    DeviceBuffer<double> dWmat(static_cast<std::size_t>(nb) * static_cast<std::size_t>(nl));
    populate_loo_wmat_resident(x, cov, r, opts, precision, dWmat);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(wmat.data(), dWmat.data(),
                                      static_cast<std::size_t>(nb) * static_cast<std::size_t>(nl) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return wmat;
}

std::vector<double> CudaBackend::se_from_wmat(
    const F4Blocks& x, const JackknifeCov& cov, int r,
    const QpAdmOptions& opts, const Precision& precision) {
    guard_device();
    const int nl = x.nl, nb = x.n_block;
    const int m = nl * x.nr;
    std::vector<double> se(static_cast<std::size_t>(nl < 0 ? 0 : nl), 0.0);
    constexpr int kMinJackknifeBlocks = 2;
    if (m <= 0 || nb < kMinJackknifeBlocks || nl <= 0) return se;

    DeviceBuffer<double> dWmat(static_cast<std::size_t>(nb) * static_cast<std::size_t>(nl));
    populate_loo_wmat_resident(x, cov, r, opts, precision, dWmat);
    // Reduce the resident (UNSCALED) dWmat on-device with the EXISTING SE kernel
    // (the same kernel the S8 batched path uses), n_models=1. Only the nl-length se
    // is D2H'd — the sum-of-squares variance reduction never leaves the device.
    DeviceBuffer<double> dSe(static_cast<std::size_t>(nl));
    launch_qpadm_se_from_wmat_batched(dWmat.data(), nl, nb, /*n_models=*/1,
                                      dSe.data(), stream_.get());
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(se.data(), dSe.data(),
                                      static_cast<std::size_t>(nl) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    // Reintroduce the AT2 (nb-1)/sqrt(nb) scale the UNSCALED single-model dWmat lacks
    // (exact, linear — see the method note). Same expression as the S8 path / the
    // pre-M7 host se_from_loo scale.
    const double scale = static_cast<double>(nb - 1) / std::sqrt(static_cast<double>(nb));
    for (double& v : se) v *= scale;
    return se;
}

void CudaBackend::populate_loo_wmat_resident(const F4Blocks& x, const JackknifeCov& cov, int r,
                                const QpAdmOptions& opts, const Precision& precision,
                                DeviceBuffer<double>& dWmat) {
    (void)precision;
    const int nl = x.nl, nr = x.nr, m = nl * nr, nb = x.n_block;
    const bool small = model_fits_small_path(nl, nr, r);
    if (m <= 0 || nb <= 0 || nl <= 0) return;

    DeviceBuffer<double> dLoo(static_cast<std::size_t>(m) * static_cast<std::size_t>(nb));
    DeviceBuffer<double> dQinv(static_cast<std::size_t>(m) * static_cast<std::size_t>(m));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dLoo.data(), x.x_loo.data(),
                                      static_cast<std::size_t>(m) * static_cast<std::size_t>(nb) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQinv.data(), cov.Qinv.data(),
                                      static_cast<std::size_t>(m) * static_cast<std::size_t>(m) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    if (small) {
        launch_qpadm_loo_batched(dLoo.data(), dQinv.data(), nl, nr, r, opts.fudge,
                                 opts.als_iterations, nb, dWmat.data(), stream_.get());
    } else {
        // LARGE-path LOO — NOW PARALLEL (was a serial nb-loop, the ~371 s NRBIG
        // wall). The nb leave-one-block-out refits are INDEPENDENT, so run them
        // CONCURRENTLY (gpu-large-models: runtime-sized device workspace, not fixed
        // per-thread local). TWO stages:
        //   Stage A — per-block cuSOLVER SVD seed (large_svd_V + seed_from_V) into
        //     nb-strided dAseed/dBseed arenas, all async-enqueued on stream_ with NO
        //     per-block cudaStreamSynchronize AND NO per-block device-wide cudaFree
        //     (the gesvd scratch is hoisted ONCE above the loop — §14.2; both the
        //     explicit-sync and the implicit-cudaFree-sync host round-trips are gone).
        //     The SVD stays cuSOLVER gesvd per block ⇒ the seed is BIT-IDENTICAL to
        //     the serial path (an on-device Jacobi seed would shift the ALS fixed
        //     point in the LSBs ⇒ SE not bit-identical).
        //   Stage B — ONE many-thread launch (one thread per (model,block)) runs the
        //     EXACT als_large + weight-solve math from a per-thread VRAM-arena slice.
        // The wmat is bit-identical to the serial loop (only the parallelism + scratch
        // location change); se_from_loo's host long-double variance reduction is
        // untouched ⇒ the NRBIG se/z stay bit-identical and G1==G2 holds.
        const int n_models = 1;  // single-model SE here; the model axis is the S8 seam
        // nb-strided Stage-A scratch + seed arenas.
        DeviceBuffer<double> dXmatB(static_cast<std::size_t>(m) * static_cast<std::size_t>(nb));
        DeviceBuffer<double> dVout(static_cast<std::size_t>(nr) * static_cast<std::size_t>(r > 0 ? r : 1) *
                                   static_cast<std::size_t>(nb));
        DeviceBuffer<double> dXt(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nl > 0 ? nl : 1) *
                                 static_cast<std::size_t>(nb));
        DeviceBuffer<double> dAseed(static_cast<std::size_t>(nl) * static_cast<std::size_t>(r > 0 ? r : 1) *
                                    static_cast<std::size_t>(nb));
        DeviceBuffer<double> dBseed(static_cast<std::size_t>(r > 0 ? r : 1) * static_cast<std::size_t>(nr) *
                                    static_cast<std::size_t>(nb));
        // Stage-B per-refit VRAM arenas (one slice per (model,block) thread).
        const std::size_t dbl_refit = large_loo_dbl_refit(nl, nr, r);
        const std::size_t int_refit = large_int_scratch(nl, nr, r);
        const std::size_t n_refit   = static_cast<std::size_t>(nb) * static_cast<std::size_t>(n_models);
        DeviceBuffer<double> dScratch(dbl_refit * n_refit);
        DeviceBuffer<int>    dIntScratch(int_refit * n_refit);
        // ---- Stage A: per-block cuSOLVER SVD seed (NO per-block sync, NO per-block
        //      device-wide cudaFree) ----
        // The cuSOLVER gesvd scratch is hoisted ONCE out of the per-block loop (the
        // §14.2 fix). Every block's SVD is the SAME (nl,nr) shape, so a single arena
        // per scratch field serves all nb blocks: the Stage-A SVDs are async-enqueued
        // SERIALLY on stream_, so they never overlap and can share one workspace (the
        // next block's gesvd cannot start until the previous one — same stream —
        // finishes; its result is already copied into the per-block dVout slice before
        // the buffer is reused). Allocating per block instead would pay nb× device-wide
        // `cudaFree` syncs (DeviceBuffer::reset ⇒ cudaFree, VERIFIED device-wide-
        // synchronizing for non-async allocations against the CUDA 13.x Runtime API
        // docs), which is exactly the serialization this PARALLEL stage was rewritten to
        // remove. The scratch frees at this block's scope exit; that final `cudaFree`'s
        // device-wide sync drains the in-flight gesvd/seed work BEFORE the buffers free
        // (same use-after-free guarantee §17.4 audited for the per-call form). The gesvd
        // MATH is unchanged ⇒ golden_fit1_NRBIG (nr=39, svd_path==2) stays bit-identical.
        if (r > 0) {
            const SvdScratchSizes sz = large_svd_scratch_sizes(nl, nr);
            DeviceBuffer<double> dSvdS(sz.s);
            DeviceBuffer<double> dSvdU(sz.u);
            DeviceBuffer<double> dSvdVt(sz.vt);
            DeviceBuffer<double> dSvdA2(sz.a2);  // 0 ⇒ no alloc (nr>=nl)
            // dSvdInfo: REQUIRED non-null device out-arg for cusolverDnDgesvd/Dgesvdj
            // (info<0 = bad param; gesvdj info>0 = non-converged Jacobi). INTENTIONAL
            // DISCARD on this off-bit-parity large (NRBIG) Stage-A LOO sweep — never
            // D2H-copied/checked across the nb blocks. Cannot drop the arg. [3.4]
            DeviceBuffer<int>    dSvdInfo(sz.info);
            DeviceBuffer<double> dSvdWork(static_cast<std::size_t>(sz.lwork));
            const int svd_lwork = sz.lwork;
            for (int b = 0; b < nb; ++b) {
                const std::size_t ob = static_cast<std::size_t>(b);
                double* xmat_b = dXmatB.data() + static_cast<std::size_t>(m) * ob;
                launch_qpadm_xmat_from_rowmajor(
                    dLoo.data() + static_cast<std::size_t>(m) * ob, nl, nr, xmat_b,
                    stream_.get());
                large_svd_V(xmat_b, nl, nr, r,
                            dVout.data() + static_cast<std::size_t>(nr) * r * ob,
                            dXt.data() + static_cast<std::size_t>(nr) * nl * ob,
                            dSvdS.data(), dSvdU.data(), dSvdVt.data(), dSvdA2.data(),
                            dSvdInfo.data(), dSvdWork.data(), svd_lwork,
                            stream_.get());
                launch_qpadm_seed_from_V(
                    xmat_b, dVout.data() + static_cast<std::size_t>(nr) * r * ob,
                    nl, nr, r,
                    dAseed.data() + static_cast<std::size_t>(nl) * r * ob,
                    dBseed.data() + static_cast<std::size_t>(r) * nr * ob,
                    stream_.get());
            }
        }
        // ---- Stage B: ONE parallel kernel — all nb refits concurrent ----
        launch_qpadm_loo_large_batched(
            dLoo.data(), dQinv.data(), dAseed.data(), dBseed.data(),
            nl, nr, r, opts.fudge, opts.als_iterations, nb, n_models,
            static_cast<long>(dbl_refit), static_cast<long>(int_refit),
            dScratch.data(), dIntScratch.data(), dWmat.data(), stream_.get());
    }
    // Producer-only: enqueue on stream_, no D2H, no sync. The dWmat is left RESIDENT
    // for the caller (gls_weights_loo_batched D2Hs it; se_from_wmat reduces it
    // on-device). The caller owns the cudaStreamSynchronize.
}

bool CudaBackend::provides_batched_fit() const { return true; }

std::vector<QpAdmResult> CudaBackend::fit_models_batched(
    const steppe::device::DeviceF2Blocks& f2,
    std::span<const QpAdmModel> models,
    const QpAdmOptions& opts,
    const Precision& precision) {
    guard_device();
    STEPPE_NVTX_RANGE("qpadm_fit");  // coarse phase boundary: model-batched qpAdm fit
    std::vector<QpAdmResult> results(models.size());
    if (models.empty()) return results;

    const Precision::Kind tag =
        (precision.kind == Precision::Kind::EmulatedFp64 &&
         capabilities().emulated_fp64_honorable)
            ? Precision::Kind::EmulatedFp64
            : Precision::Kind::Fp64;

    // ---- bucket by (nl, nr, r): each bucket is one strided/pointer-array arena --
    // The rotation common case (all k-subsets of one pool, one right set) yields a
    // FEW buckets (one per left-size). Within a bucket every model is the same
    // shape, so all arenas are dense strided.
    struct Key { int nl, nr, r; };
    std::vector<std::vector<std::size_t>> bucket_members;  // indices into models[]
    std::vector<Key> bucket_keys;
    for (std::size_t mi = 0; mi < models.size(); ++mi) {
        const QpAdmModel& mdl = models[mi];
        const int nl = static_cast<int>(mdl.left.size());
        const int nr = static_cast<int>(mdl.right.size()) - 1;
        const int r = (opts.rank < 0) ? (nl - 1) : opts.rank;
        std::size_t bk = bucket_keys.size();
        for (std::size_t k = 0; k < bucket_keys.size(); ++k)
            if (bucket_keys[k].nl == nl && bucket_keys[k].nr == nr &&
                bucket_keys[k].r == r) { bk = k; break; }
        if (bk == bucket_keys.size()) {
            bucket_keys.push_back(Key{nl, nr, r});
            bucket_members.emplace_back();
        }
        bucket_members[bk].push_back(mi);
    }

    // F1 / OQ-12 — the SURVIVOR block set for the rotation (computed ONCE; the
    // keep-mask is a property of the resident f2, shared by every model/bucket). A
    // missing block (Vpair==0 for any pair) is dropped, exactly as the single-model
    // assemble_f4 does (AT2 read_f2(remove_na=TRUE)). With no missing blocks (every
    // existing golden) surv is identity and dSurv stays nullptr ⇒ the batched gather
    // runs its bit-identical no-drop arm. The survivor block_sizes + resident dSurv
    // are built here and threaded into every bucket/chunk.
    const std::vector<int> surv = device_survivor_blocks(f2, f2.n_block, f2.P);
    const int nb_s = static_cast<int>(surv.size());
    const bool dropped = (nb_s != f2.n_block);
    std::vector<int> surv_block_sizes(static_cast<std::size_t>(nb_s), 0);
    for (int bs = 0; bs < nb_s; ++bs)
        surv_block_sizes[static_cast<std::size_t>(bs)] =
            f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
    DeviceBuffer<int> dSurv(static_cast<std::size_t>(dropped ? nb_s : 1));
    if (dropped)
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSurv.data(), surv.data(),
                                          static_cast<std::size_t>(nb_s) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
    const int* d_surv = dropped ? dSurv.data() : nullptr;

    for (std::size_t bk = 0; bk < bucket_keys.size(); ++bk) {
        fit_one_bucket(f2, models, bucket_members[bk], bucket_keys[bk].nl,
                       bucket_keys[bk].nr, bucket_keys[bk].r, nb_s, surv_block_sizes,
                       d_surv, opts, precision, tag, results);
    }
    return results;
}

void CudaBackend::fit_one_bucket(const steppe::device::DeviceF2Blocks& f2,
                    std::span<const QpAdmModel> models,
                    const std::vector<std::size_t>& mem, int nl, int nr, int r,
                    int nb, const std::vector<int>& survivor_block_sizes,
                    const int* d_surv, const QpAdmOptions& opts,
                    const Precision& precision,
                    Precision::Kind tag, std::vector<QpAdmResult>& results) {
    if (mem.empty()) return;
    const int m = nl * nr;
    // F1 / OQ-12: `nb` is the SURVIVOR block count + `survivor_block_sizes` the
    // survivor SNP counts (the f2 source's full set MINUS the missing blocks), and
    // `d_surv` (nullptr when no drop) maps the compacted survivor block to its
    // resident block id. Everything below weights/normalizes by the SURVIVOR set,
    // mirroring the single-model assemble_f4 (AT2 read_f2(remove_na=TRUE)).
    const int P = f2.P;
    const int rmax = (nl < nr ? nl : nr) - 1;
    const int r_fit = r;
    const std::size_t m_sz = static_cast<std::size_t>(m);
    const std::size_t Mm = m_sz * m_sz;

    // n = Σ SURVIVOR block_sizes.
    long long n_ll = 0;
    for (int v : survivor_block_sizes) n_ll += v;
    const double n = static_cast<double>(n_ll);

    // SURVIVOR block_sizes resident once (reused across chunks).
    DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb > 0 ? nb : 1));
    if (nb > 0)
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), survivor_block_sizes.data(),
                                          static_cast<std::size_t>(nb) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));

    // ---- VRAM-budget the chunk size B (design §6) ---------------------------
    // Per-model device bytes (double): dX + dLoo + dXtau (3·m·nb) + dQ + dQinv +
    // dQf + dI (4·m²) + small fit outputs (≈ nl + se + chisq + rank_chisq + pop).
    //
    // SE pass-2 COEXISTS with pass-1 (one flat RAII scope in fit_chunk: the pass-1
    // arenas above are still live when the LOO SE arenas allocate on top). Under
    // jackknife≥1 the worst-case compacted pass-2 path (Sn==B) allocates per model:
    // dLooS (m·nb) + dQinvS (m²) + dWmatS (nb·nl) + dSeS (nl) doubles + dSurv (1 int).
    // The pre-policy budgeter counted ONLY pass-1 (same class as the :612 ~2×-under-
    // budget note), so jk1/jk2 over-committed and cudaMalloc OOM'd above ~pool_50.
    // Fold the pass-2 term in here, JK-GATED to 0 for None so jk0's B_max is bit-for-
    // bit unchanged. A smaller B under jk≥1 makes the EXISTING chunk loop (:2480) tile
    // BOTH passes — chunk width moves no bits (architecture.md §12; the LOO SE is per-
    // model + chunk-independent), so the pools that already ran jk1 stay byte-identical.
    const bool se_pass = (opts.jackknife != JackknifePolicy::None);
    const std::size_t se_per_model_dbl =
        se_pass
            ? (m_sz * static_cast<std::size_t>(nb)                            // dLooS
               + Mm                                                          // dQinvS
               + static_cast<std::size_t>(nb) * static_cast<std::size_t>(nl) // dWmatS
               + static_cast<std::size_t>(nl))                               // dSeS
            : 0;
    const std::size_t per_model_dbl =
        3 * m_sz * static_cast<std::size_t>(nb) + 4 * Mm +
        static_cast<std::size_t>(2 * nl + 1 + (rmax + 1) + (nl + 1) + nl) +
        se_per_model_dbl;  // SE pass-2 arenas coexist with pass-1 (JK-gated)
    const std::size_t per_model_bytes = per_model_dbl * sizeof(double) +
        static_cast<std::size_t>((nl + 1) + (nr + 1)) * sizeof(int) +
        sizeof(int) /*status*/ + 3 * sizeof(double*) /*ptr arrays*/ +
        (se_pass ? sizeof(int) : 0) /*dSurv (JK-gated)*/;
    std::size_t free_b = capabilities().free_vram_bytes;
    if (free_b == 0) free_b = kFitBudgetFreeVramFallbackBytes;  // 4 GB free-VRAM fallback
    const std::size_t headroom = kFitBudgetHeadroomBytes;       // 512 MB headroom
    std::size_t budget = (free_b > headroom) ? (free_b - headroom) : free_b / 2;
    std::size_t B_max = (per_model_bytes > 0) ? (budget / per_model_bytes) : mem.size();
    if (B_max < 1) B_max = 1;
    if (B_max > mem.size()) B_max = mem.size();

    for (std::size_t off = 0; off < mem.size(); off += B_max) {
        const std::size_t B = std::min(B_max, mem.size() - off);
        fit_chunk(f2, models, mem, off, B, nl, nr, r_fit, rmax, m, nb, P, n,
                  dBlockSizes.data(), d_surv, opts, precision, tag, results);
    }
}

void CudaBackend::fit_chunk(const steppe::device::DeviceF2Blocks& f2,
               std::span<const QpAdmModel> models,
               const std::vector<std::size_t>& mem, std::size_t off, std::size_t B,
               int nl, int nr, int r_fit, int rmax, int m, int nb, int P, double n,
               const int* dBlockSizes, const int* d_surv, const QpAdmOptions& opts,
               const Precision& precision, Precision::Kind tag,
               std::vector<QpAdmResult>& results) {
    ++batched_dispatch_count_;  // S8 observability: one BATCHED dispatch per chunk
    const std::size_t m_sz = static_cast<std::size_t>(m);
    const std::size_t Mm = m_sz * m_sz;
    const std::size_t Bnb = B * m_sz * static_cast<std::size_t>(nb);

    // ---- per-model index arenas (left [nl+1], right [nr+1]) -------------------
    std::vector<int> h_left(B * static_cast<std::size_t>(nl + 1));
    std::vector<int> h_right(B * static_cast<std::size_t>(nr + 1));
    for (std::size_t j = 0; j < B; ++j) {
        const QpAdmModel& mdl = models[mem[off + j]];
        int* lp = h_left.data() + j * (nl + 1);
        lp[0] = mdl.target;
        for (int i = 0; i < nl; ++i) lp[i + 1] = mdl.left[static_cast<std::size_t>(i)];
        int* rp = h_right.data() + j * (nr + 1);
        for (int i = 0; i <= nr; ++i) rp[i] = mdl.right[static_cast<std::size_t>(i)];
    }
    DeviceBuffer<int> dLeft(h_left.size());
    DeviceBuffer<int> dRight(h_right.size());
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dLeft.data(), h_left.data(),
                                      h_left.size() * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dRight.data(), h_right.data(),
                                      h_right.size() * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));

    // ---- strided arenas ------------------------------------------------------
    DeviceBuffer<double> dX(Bnb);
    DeviceBuffer<double> dLoo(Bnb);
    DeviceBuffer<double> dXtau(Bnb);
    DeviceBuffer<double> dTotal(B * m_sz);
    DeviceBuffer<double> dTotLine(B * m_sz);
    DeviceBuffer<double> dQ(B * Mm);
    DeviceBuffer<double> dQf(B * Mm);

    // ---- S3 gather (model-batched, reads resident f2; F1 survivor-compacted) -
    launch_assemble_f4_gather_models_batched(f2.f2_device(), P, dLeft.data(),
                                             dRight.data(), nl, nr, nb,
                                             static_cast<int>(B), d_surv, dX.data(),
                                             stream_.get());
    launch_f4_loo_total_models_batched(dX.data(), dBlockSizes, m, nb, n,
                                       static_cast<int>(B), dLoo.data(),
                                       dTotal.data(), dTotLine.data(), stream_.get());
    launch_f4_xtau_models_batched(dLoo.data(), dTotal.data(), dTotLine.data(),
                                  dBlockSizes, m, nb, n, static_cast<int>(B),
                                  dXtau.data(), stream_.get());

    // ---- S4 Q = xtau·xtauᵀ / nb (strided-batched GEMM; ENGAGES precision) -----
    // A = dXtau (m×nb col-major), Q = A·Aᵀ/nb (m×m). Strided over models. cublasD-
    // gemmStridedBatched honors the handle MATH MODE (set by engage_f2_precision),
    // exactly the jackknife_cov SYRK policy, now batched. Scoped so the emulated/
    // PEDANTIC mode never leaks into the native cuSOLVER inverse below.
    {
        const double alpha = 1.0 / static_cast<double>(nb);
        const double beta = 0.0;
        const MathModeScope gemm_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas_.get(), precision);
        CUBLAS_CHECK(cublasDgemmStridedBatched(
            blas_.get(), CUBLAS_OP_N, CUBLAS_OP_T, m, m, nb, &alpha,
            dXtau.data(), m, static_cast<long long>(m_sz) * nb,
            dXtau.data(), m, static_cast<long long>(m_sz) * nb, &beta,
            dQ.data(), m, static_cast<long long>(Mm),
            static_cast<int>(B)));
    }

    // ---- S4 fudge diag (per-model trace) → dQf -------------------------------
    launch_add_fudge_diag_models_batched(dQ.data(), dQf.data(), m, opts.fudge,
                                         static_cast<int>(B), stream_.get());

    // ---- S4 Qinv via cuSOLVER potrfBatched + potrsBatched (vs batched I) ------
    // potrfBatched factors each dQf_model (LOWER) in place. cusolverDnDpotrsBatched
    // supports ONLY nrhs==1, so the m-column inverse is built COLUMN-BY-COLUMN: dQinv
    // starts as the per-model identity, and for each column c we solve Qf·x = e_c
    // (the c-th identity column) IN PLACE on column c across ALL B models (one
    // batched potrsBatched, B systems, nrhs=1) — so column c of every model's dQinv
    // becomes column c of Qinv. m batched solves total (m<=50). Per-model potrf
    // devInfo>0 ⇒ NonSpdCovariance (record-and-continue). Native FP64.
    DeviceBuffer<double> dQinv(B * Mm);  // identity RHS → the inverse, in place
    launch_fill_identity_batched(dQinv.data(), m, static_cast<int>(B), stream_.get());
    // A-pointer array (the factored dQf) — reused for every column solve.
    std::vector<double*> h_Aptr(B);
    for (std::size_t j = 0; j < B; ++j) h_Aptr[j] = dQf.data() + j * Mm;
    DeviceBuffer<double*> dAptr(B);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dAptr.data(), h_Aptr.data(),
                                      B * sizeof(double*), cudaMemcpyHostToDevice,
                                      stream_.get()));
    DeviceBuffer<int> dInfo(B);
    const CusolverMathModeScope solve_scope =
        engage_solver_precision(solver_.get(), solve_precision_, &emulation_honorable);
    CUSOLVER_CHECK(cusolverDnDpotrfBatched(solver_.get(), CUBLAS_FILL_MODE_LOWER, m,
                                           dAptr.data(), m, dInfo.data(),
                                           static_cast<int>(B)));
    std::vector<int> h_info(B, 0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_info.data(), dInfo.data(), B * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    // All m B-pointer arrays (m blocks of B device pointers each, column-major over
    // c) built once and uploaded in ONE H2D — mirroring the single dAptr upload above.
    // The pointers are fully deterministic (dQinv.data() + j*Mm + c*m_sz), so the per-
    // column loop below only issues the B-batched solve, slicing dBptrAll at c*B.
    DeviceBuffer<double*> dBptrAll(static_cast<std::size_t>(m) * B);
    std::vector<double*> h_BptrAll(static_cast<std::size_t>(m) * B);
    for (int c = 0; c < m; ++c) {
        for (std::size_t j = 0; j < B; ++j)
            h_BptrAll[static_cast<std::size_t>(c) * B + j] =
                dQinv.data() + j * Mm + static_cast<std::size_t>(c) * m_sz;
    }
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(
        dBptrAll.data(), h_BptrAll.data(),
        static_cast<std::size_t>(m) * B * sizeof(double*), cudaMemcpyHostToDevice,
        stream_.get()));
    for (int c = 0; c < m; ++c) {
        // solve_info: REQUIRED non-null HOST out-arg for cusolverDnDpotrsBatched
        // (reports parameter validity: info<0 ⇒ i-th arg invalid — NOT per-system SPD
        // status). INTENTIONAL DISCARD: per-column SPD status is already gated by the
        // potrfBatched dInfo array checked above. Cannot drop the arg. [3.4]
        int solve_info = 0;
        CUSOLVER_CHECK(cusolverDnDpotrsBatched(
            solver_.get(), CUBLAS_FILL_MODE_LOWER, m, 1 /*nrhs*/, dAptr.data(), m,
            dBptrAll.data() + static_cast<std::size_t>(c) * B, m, &solve_info,
            static_cast<int>(B)));
    }
    // The batched cuSOLVER solve writes its host `info` arg and the stream must be
    // drained before the next stream op (an undrained batched-potrs lane returns
    // cudaErrorInvalidValue on the following cudaMemcpyAsync — MEASURED on box5090).
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    // ---- S5/S6/S7 + rankdrop + popdrop (model-batched, one thread per model) --
    DeviceBuffer<double> dWeight(B * static_cast<std::size_t>(nl));
    DeviceBuffer<double> dSe(B * static_cast<std::size_t>(nl));
    DeviceBuffer<double> dChisq(B);
    DeviceBuffer<int> dStatus(B);
    DeviceBuffer<double> dRankChisq(B * static_cast<std::size_t>(rmax + 1));
    DeviceBuffer<double> dPopChisq(B * static_cast<std::size_t>(nl + 1));
    DeviceBuffer<double> dPopWfull(B * static_cast<std::size_t>(nl));
    launch_qpadm_fit_models_batched(
        dTotal.data(), dQinv.data(), dLoo.data(), dBlockSizes, nl, nr, r_fit, rmax,
        opts.fudge, opts.als_iterations, nb, static_cast<int>(B),
        dWeight.data(), dSe.data(), dChisq.data(), dStatus.data(),
        dRankChisq.data(), dPopChisq.data(), dPopWfull.data(), stream_.get());

    // ============================ THE TWO-PASS SE POLICY ============================
    // Pass 1 (the CHEAP point estimate: gather → Q → Qinv → rank-sweep → weights →
    // chisq → popdrop → feasibility) is COMPLETE for ALL B models above. Now D2H the
    // cheap fields, decide the SURVIVOR set from those host outputs (zero extra D2H —
    // the bytes the filter reads are already on the host), and run Pass 2 (the
    // expensive LOO jackknife SE) ONLY over the survivors. The SE math/kernels are
    // UNCHANGED — only WHICH models reach them changes (fit-engine.md §M(fit-3)).
    // ALL mode runs the SE block VERBATIM over every valid model ⇒ bit-identical to
    // the pre-policy code path (the parity gate). NONE skips it for all; FEASIBLE-ONLY
    // skips it for the infeasible majority. (design: host filter, per-chunk.)

    // ---- D2H the CHEAP fields (NOT se yet) -----------------------------------
    std::vector<double> h_weight(B * nl), h_se(B * nl, 0.0), h_chisq(B),
        h_rankchisq(B * (rmax + 1)), h_popchisq(B * (nl + 1)), h_popwfull(B * nl);
    std::vector<int> h_status(B);
    auto d2h = [&](double* dst, const DeviceBuffer<double>& src, std::size_t cnt) {
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dst, src.data(), cnt * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
    };
    d2h(h_weight.data(), dWeight, B * nl);
    d2h(h_chisq.data(), dChisq, B);
    d2h(h_rankchisq.data(), dRankChisq, B * (rmax + 1));
    d2h(h_popchisq.data(), dPopChisq, B * (nl + 1));
    d2h(h_popwfull.data(), dPopWfull, B * nl);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_status.data(), dStatus.data(), B * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    // ---- the host survivor filter (per-chunk) --------------------------------
    // A model is SE-ELIGIBLE only if it produced a valid Ok fit (potrf ok AND rank
    // ok) — a domain-failed model has no weights, so no SE regardless of mode (this
    // matches today: failed models early-return with empty se in assemble_result).
    // Among the eligible: ALL ⇒ all; NONE ⇒ none; FEASIBLE-ONLY ⇒ feasible (the same
    // all-pop_wfull-in-[0,1] test assemble_result uses), optionally AND p>=threshold.
    const int dof_full = core::qpadm::qpadm_dof(nl, nr, r_fit);
    auto feasible_from_wfull = [nl](const double* w) {
        bool any = false;
        for (int i = 0; i < nl; ++i) {
            const double v = w[i];
            if (std::isnan(v)) continue;
            any = true;
            if (v < 0.0 || v > 1.0) return false;
        }
        return any;
    };
    std::vector<char> se_computed(B, 0);   // per-model: did this model get the SE?
    std::vector<std::size_t> surv;         // SE-survivor positions within the chunk
    surv.reserve(B);
    std::size_t n_eligible = 0;            // valid-Ok models (the ALL-mode SE set)
    for (std::size_t j = 0; j < B; ++j) {
        const bool ok_fit = (h_info[j] == 0) && (h_status[j] == 0);
        if (!ok_fit) continue;
        ++n_eligible;
        bool survivor = false;  // structural no-uninitialized-read, not switch-coverage-dependent ([10.1][LOW])
        switch (opts.jackknife) {
            case JackknifePolicy::None:
                survivor = false;
                break;
            case JackknifePolicy::FeasibleOnly: {
                const bool feas = feasible_from_wfull(h_popwfull.data() + j * nl);
                const double p = core::internal::pchisq_upper(h_chisq[j], dof_full);
                survivor = feas && (!opts.se_require_p || p >= opts.p_se_threshold);
                break;
            }
            case JackknifePolicy::All:
            default:
                survivor = true;
                break;
        }
        if (survivor) { surv.push_back(j); se_computed[j] = 1; }
    }

    // ---- Pass 2: the SE ONLY over the survivor set ---------------------------
    // ALL-mode FAST PATH (every eligible model is a survivor): run the SE block
    // VERBATIM over the full chunk into dSe — provably the pre-policy code path, so
    // ALL mode is byte-for-byte identical to today (the parity pin). Otherwise gather
    // the survivor dLoo/dQinv slices into compact arenas (pure D2D copies, parity-
    // neutral) and run the UNCHANGED SE kernels with n_models=surv.size(), then
    // scatter the compact dSe back into h_se at the survivor positions.
    if (nb >= 2 && !surv.empty()) {
        const double jackknife_scale =
            static_cast<double>(nb - 1) / std::sqrt(static_cast<double>(nb));
        const bool all_survive = (surv.size() == n_eligible) &&
                                 (n_eligible == B);
        if (all_survive) {
            // VERBATIM full-chunk SE (the ALL-mode bit-identical path).
            DeviceBuffer<double> dWmat(B * static_cast<std::size_t>(nb) *
                                       static_cast<std::size_t>(nl));
            launch_qpadm_loo_models_batched(dLoo.data(), dQinv.data(), nl, nr, r_fit,
                                            opts.fudge, opts.als_iterations, nb,
                                            static_cast<int>(B), jackknife_scale,
                                            dWmat.data(),
                                            stream_.get());
            launch_qpadm_se_from_wmat_batched(dWmat.data(), nl, nb,
                                              static_cast<int>(B), dSe.data(),
                                              stream_.get());
            d2h(h_se.data(), dSe, B * nl);
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        } else {
            // COMPACTED survivor SE: gather per-survivor dLoo (m*nb) + dQinv (m*m)
            // slices into dense arenas (ascending survivor order ⇒ a survivor's
            // slice is bit-identical to its full-arena slice), run the SAME kernels
            // over n_models=Sn, D2H the compact dSe, scatter into h_se. The gather is
            // ONE kernel (parity-neutral data movement), not a per-survivor D2D loop.
            const std::size_t Sn = surv.size();
            std::vector<int> h_surv(Sn);
            for (std::size_t k = 0; k < Sn; ++k)
                h_surv[k] = static_cast<int>(surv[k]);
            DeviceBuffer<int> dSurv(Sn);
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSurv.data(), h_surv.data(),
                                              Sn * sizeof(int), cudaMemcpyHostToDevice,
                                              stream_.get()));
            DeviceBuffer<double> dLooS(Sn * m_sz * static_cast<std::size_t>(nb));
            DeviceBuffer<double> dQinvS(Sn * Mm);
            launch_qpadm_gather_loo_qinv(dLoo.data(), dQinv.data(), dSurv.data(),
                                         m, nb, static_cast<int>(Sn), dLooS.data(),
                                         dQinvS.data(), stream_.get());
            DeviceBuffer<double> dWmatS(Sn * static_cast<std::size_t>(nb) *
                                        static_cast<std::size_t>(nl));
            DeviceBuffer<double> dSeS(Sn * static_cast<std::size_t>(nl));
            launch_qpadm_loo_models_batched(dLooS.data(), dQinvS.data(), nl, nr, r_fit,
                                            opts.fudge, opts.als_iterations, nb,
                                            static_cast<int>(Sn), jackknife_scale,
                                            dWmatS.data(),
                                            stream_.get());
            launch_qpadm_se_from_wmat_batched(dWmatS.data(), nl, nb,
                                              static_cast<int>(Sn), dSeS.data(),
                                              stream_.get());
            std::vector<double> h_seS(Sn * nl);
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_seS.data(), dSeS.data(),
                                              Sn * nl * sizeof(double),
                                              cudaMemcpyDeviceToHost, stream_.get()));
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
            for (std::size_t k = 0; k < Sn; ++k) {
                const std::size_t j = surv[k];
                for (int i = 0; i < nl; ++i)
                    h_se[j * nl + i] = h_seS[k * nl + i];
            }
        }
    }

    // ---- assemble QpAdmResult per model (host post-process; EXACTLY run_impl) --
    // Write POSITIONALLY into results[pos] where pos = the model's position in the
    // `models` span this fit_models_batched call received (mem[] indexes that span).
    // The result ECHOES model_index (set in assemble_result), so the orchestrator's
    // pre-sized-slot re-sort by model_index yields the deterministic output order —
    // this positional write must NOT use the global model_index (the span may be a
    // compacted small-path sub-list whose model_index values exceed its size).
    for (std::size_t j = 0; j < B; ++j) {
        const std::size_t pos = mem[off + j];
        const QpAdmModel& mdl = models[pos];
        assemble_result(mdl, nl, nr, r_fit, rmax, tag,
                        AssembleFlags{ .nonspd = h_info[j] != 0,        // potrf failed ⇒ NonSpd
                                       .se_computed = se_computed[j] != 0 },  // SE computed?
                        h_status[j],
                        h_weight.data() + j * nl, h_se.data() + j * nl, h_chisq[j],
                        h_rankchisq.data() + j * (rmax + 1),
                        h_popchisq.data() + j * (nl + 1),
                        h_popwfull.data() + j * nl,
                        opts.rank_alpha,          // f4rank threshold (= single-model path)
                        results[pos]);
    }
}

void CudaBackend::assemble_result(const QpAdmModel& mdl, int nl, int nr, int r_fit, int rmax,
                     Precision::Kind tag, AssembleFlags flags, int fit_status,
                     const double* weight, const double* se, double chisq,
                     const double* rank_chisq, const double* pop_chisq,
                     const double* pop_wfull, double rank_alpha, QpAdmResult& res) {
    res.model_index = mdl.model_index;
    res.precision_tag = tag;
    res.est_rank = r_fit;
    res.dof = core::qpadm::qpadm_dof(nl, nr, r_fit);
    if (flags.nonspd) { res.status = Status::NonSpdCovariance; return; }
    if (fit_status != 0) { res.status = Status::RankDeficient; return; }

    res.weight.assign(weight, weight + nl);
    // SE POLICY: fill se/z ONLY when the SE was computed for this model (the
    // survivor set under the chosen JackknifePolicy). A non-survivor leaves se/z
    // EMPTY — the sentinel for "not computed" (NEVER a fake 0/NaN). A survivor's
    // se/z is bit-identical to ALL mode (the SE math/kernels are unchanged).
    if (flags.se_computed) {
        res.se.assign(se, se + nl);
        res.z.assign(static_cast<std::size_t>(nl), 0.0);
        for (int i = 0; i < nl; ++i)
            res.z[static_cast<std::size_t>(i)] =
                (se[i] > 0.0) ? weight[i] / se[i] : 0.0;
    }
    res.chisq = chisq;
    // For dof<=0 pchisq_upper returns NaN (the tail-p is undefined); flagged
    // below as Status::ChisqUndefined so the batched rotation surfaces the
    // same domain outcome the single-model host path does (qpadm_fit.cpp;
    // architecture.md §10 STEPPE_ERR_CHISQ_UNDEFINED), not a NaN-p with Ok.
    res.p = core::internal::pchisq_upper(chisq, res.dof);
    res.rank_p.assign(static_cast<std::size_t>(r_fit) + 1, 0.0);
    if (r_fit >= 0 && static_cast<std::size_t>(r_fit) < res.rank_p.size())
        res.rank_p[static_cast<std::size_t>(r_fit)] = res.p;

    // rank sweep chisq(r)/dof(r)/p(r) for r = 0..rmax.
    const std::size_t nrk = static_cast<std::size_t>(rmax) + 1;
    res.rank_chisq.assign(nrk, 0.0);
    res.rank_dof.assign(nrk, 0);
    std::vector<double> rankp(nrk, 0.0);
    for (int rr = 0; rr <= rmax; ++rr) {
        res.rank_chisq[static_cast<std::size_t>(rr)] = rank_chisq[rr];
        res.rank_dof[static_cast<std::size_t>(rr)] = core::qpadm::qpadm_dof(nl, nr, rr);
        rankp[static_cast<std::size_t>(rr)] =
            core::internal::pchisq_upper(rank_chisq[rr],
                                         res.rank_dof[static_cast<std::size_t>(rr)]);
    }
    // f4rank = smallest non-rejected rank (p(r) > rank_alpha, ascending). The
    // threshold is the caller's opts.rank_alpha (default 0.05) — the SAME value
    // the single-model rank_sweep uses (line ~1844), so the batched rotation no
    // longer diverges from the single-model path for any non-default rank_alpha.
    res.f4rank = rmax;
    for (int rr = 0; rr <= rmax; ++rr)
        if (rankp[static_cast<std::size_t>(rr)] > rank_alpha) { res.f4rank = rr; break; }

    // AT2 res$rankdrop nested table (rows rank rmax..0; the nested diff). Shared
    // fill_rankdrop ([7.1] dedup): source = res.rank_dof/res.rank_chisq + the local
    // rankp (assemble_result re-derives rankp; the single-model sweep pre-stored
    // rs.rd_p — that re-derivation is preserved by passing rankp as the p source).
    // Byte-identical to the former inline copy (§12).
    fill_rankdrop(rmax, res.rank_dof, res.rank_chisq, rankp,
                  res.rankdrop_f4rank, res.rankdrop_dof, res.rankdrop_dofdiff,
                  res.rankdrop_chisq, res.rankdrop_p, res.rankdrop_chisqdiff,
                  res.rankdrop_p_nested);

    // AT2 res$popdrop: the full row (all sources) then each single-source drop.
    // Row 0 = full model (pattern "0..0"), fitted rank nl-1, weights pop_wfull
    // (the feasibility source). Rows 1.. = drop source (nl-1),(nl-2),..,0 (pat
    // "..1..") fitted at rank (nl_red-1); chisq from pop_chisq[1+...]. dof =
    // (nl_red - r)*(nr - r). Mirrors ranktest.cpp run_popdrop EXACTLY.
    auto push_pop = [&](const std::string& pat, int wt, int nl_red, double cq,
                        const double* w_for_feas) {
        const int rr = nl_red - 1;
        const int dof = core::qpadm::qpadm_dof(nl_red, nr, rr);
        res.popdrop_pat.push_back(pat);
        res.popdrop_wt.push_back(wt);
        res.popdrop_dof.push_back(dof);
        res.popdrop_f4rank.push_back(rr);
        res.popdrop_chisq.push_back(cq);
        res.popdrop_p.push_back(core::internal::pchisq_upper(cq, dof));
        // feasibility: all non-NaN reported weights in [0,1], at least one.
        bool any = false, feas = true;
        if (w_for_feas) {
            for (int i = 0; i < nl; ++i) {
                const double w = w_for_feas[i];
                if (std::isnan(w)) continue;
                any = true;
                if (w < 0.0 || w > 1.0) { feas = false; break; }
            }
        }
        res.popdrop_feasible.push_back((any && feas) ? char{1} : char{0});
    };
    if (nl >= 1) {
        std::string pat_full(static_cast<std::size_t>(nl), '0');
        push_pop(pat_full, 0, nl, pop_chisq[0], pop_wfull);
        // drops: row index 1 + (nl-1-drop) in the kernel's pop_chisq layout.
        if (nl >= 2) {
            for (int drop = nl - 1; drop >= 0; --drop) {
                std::string pat(static_cast<std::size_t>(nl), '0');
                pat[static_cast<std::size_t>(drop)] = '1';
                const int row = 1 + (nl - 1 - drop);
                // the dropped-row weights are not returned per-source (only the full
                // row's are needed for the model feasibility decision, which is
                // popdrop[0]); the drop rows carry chisq/dof/p only, feasibility on
                // the surviving set is not gated here (matches the test, which reads
                // popdrop_feasible[0]). Pass null ⇒ feasible recorded false.
                push_pop(pat, 1, nl - 1, pop_chisq[row], nullptr);
            }
        }
    }
    // dof<=0 ⇒ ChisqUndefined (the per-model status VALUE; CPU-then-GPU
    // completeness — the batched rotation must return it too). The fit
    // populated normally; only the undefined tail-p is flagged. Behavior-
    // neutral for normal models (dof>0 ⇒ Ok; goldens unchanged).
    res.status = (res.dof <= 0) ? Status::ChisqUndefined : Status::Ok;
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

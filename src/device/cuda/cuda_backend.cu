// src/device/cuda/cuda_backend.cu
//
// CudaBackend — the GPU implementation of ComputeBackend (architecture.md §4, §8;
// ROADMAP §2, M0). Implements `compute_f2` via the f2 3-GEMM reformulation:
// upload the Q/V/N contract, run the fused feeder + three GEMMs + fused
// numerator/divide (f2_block_kernel.cu), and copy f2 + Vpair back across the
// CUDA-free ComputeBackend seam (architecture.md §4).
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
#include <limits>     // std::numeric_limits<int>::max — the M0 k-narrowing guard (B22)
#include <memory>
#include <span>
#include <stdexcept>  // std::runtime_error — the M0 M>INT_MAX fail-fast (B22)
#include <string>     // std::to_string — diagnostic message for the B22 guard
#include <vector>

#include "core/domain/block_partition_rule.hpp" // core::block_ranges, core::BlockRange (the X-3/B3 single-source inverse)
#include "core/internal/pchisq.hpp"         // core::internal::pchisq_upper (M(fit-2) rank-test p; the ONE shared special fn)
#include "core/internal/qpfstats_jackknife.hpp"  // core::f2blocks_pair_est (the partial-NaN fallback recenter; cold path)
#include "core/internal/small_linalg.hpp"   // core::jacobi_svd (M(fit-2) rank_Q diagnostic; bit-identical to the CPU oracle)
#include "core/qpadm/qpadm_bounds.hpp"       // core::qpadm::model_fits_small_path — the SINGLE-SOURCE small-path envelope (kQpMax*)
#include "device/backend.hpp"               // ComputeBackend, F2Result, F2BlockTensor, MatView
#include "device/backend_factory.hpp"       // steppe::device::make_cuda_backend (the single-source decl, X-9/B8)
#include "device/device_partial.hpp"        // steppe::device::DevicePartial (the M4.5 resident handle)
#include "device/cuda/device_partial_impl.cuh" // DevicePartial::Impl (the DeviceBuffer<double> owners)
#include "device/device_f2_blocks.hpp"      // steppe::device::DeviceF2Blocks (the M4.5 device-resident FULL result handle)
#include "device/cuda/device_f2_blocks_impl.cuh" // DeviceF2Blocks::Impl (the DeviceBuffer<double> owners)
#include "device/device_decode_result.hpp"  // steppe::device::DeviceDecodeResult (the device-resident autosome-compacted Q/V handle)
#include "device/cuda/device_decode_result_impl.cuh" // DeviceDecodeResult::Impl (the DeviceBuffer<double> q/v owners)
#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK
#include "device/cuda/decode_af_kernel.cuh" // launch_decode_af
#include "device/cuda/decode_compact_kernel.cuh" // launch_autosome_keep_mask / _compact_columns_gather (the device-resident decode seam)
#include "device/cuda/dstat_kernel.cuh"     // launch_dstat_block_reduce (qpDstat Part B)
#include "device/cuda/dates_kernel.cuh"     // DATES cuFFT autocorrelation LD engine kernels
#include "device/cuda/qpfstats_kernel.cuh"  // launch_qpfstats_zero_nan_ymat / _add_ridge_diag (the smoother prep)
#include "device/cuda/qpfstats_jackknife_kernel.cuh"  // launch_qpfstats_numer_jackknife / _recenter_shift (the fused PERF path)
#include "device/cuda/ratio_block_jackknife_kernel.cuh"  // launch_ratio_block_jackknife (the SHARED f4ratio M1 + qpDstat M2 engine)
#include "device/cuda/device_buffer.cuh"    // DeviceBuffer<T> (RAII)
#include "device/cuda/f2_block_kernel.cuh"  // launch_f2_feeder, engage_f2_precision, emulation_honorable (the X-6/B2 probe)
#include "device/cuda/f2_blocks_kernel.cuh" // launch_gather_group, run_f2_gemms_group, launch_assemble_blocks_group
#include "device/cuda/block_sink.cuh"       // M5: BlockSink, HostRamSink, DiskSink, kStreamStagingSlots
#include "device/stream_f2_blocks.hpp"      // M5: StreamTarget (the CUDA-free streamed-tier request)
#include "device/f2_blocks_out.hpp"         // M5: DiskF2Blocks (the Disk descriptor DiskSink populates)
#include "device/cuda/handles.hpp"          // CublasHandle, CusolverDnHandle (RAII)
#include "device/cuda/qpadm_fit_kernels.cuh" // M(fit-4): f4 gather + loo/total + xtau + small-LA launch wrappers
#include "device/cuda/qpgraph_fit_kernels.cuh" // qpGraph: the on-device IDEA-1 fleet launcher
#include "core/qpadm/qpgraph_objective.hpp" // qpGraph host edge-recovery at the best theta (ONE eval, not per-iteration)
#include "device/cuda/pinned_buffer.cuh"    // PinnedRegistryCache (amortized in-place pin for async H2D overlap — P4/L2); PinnedBuffer (persistent D2H staging — P5/d2h-speed)
#include "device/cuda/stream.hpp"           // Stream (RAII, owning non-blocking per-device stream — P2/F1)
#include "device/vram_budget.hpp"           // max_blocks_per_chunk (host-pure VRAM budget; X-5/B5 + X-13/B26)
#include "steppe/config.hpp"                // Precision, kDefaultMantissaBits, kBlockGroupPadBase, kMaxVramUtilizationFraction, kStreamTileBudgetFraction, kFitBudget*
#include "steppe/fstats.hpp"                // F2BlockTensor

namespace steppe::device {

namespace {

// ---- Shared f2-blocks layout + size-bucketing prologue ([7.1] dedup) --------
// run_f2_blocks_resident and stream_f2_blocks_impl opened with a BYTE-IDENTICAL
// block-layout derivation (core::block_ranges -> per-block begin/size) and a
// BYTE-IDENTICAL size-bucketing pass (ceil-pow{kBlockGroupPadBase}(size) ->
// Bucket{s_pad, blocks}, sorted by s_pad). The 744/761 comments literally read
// "IDENTICAL to run_f2_blocks_resident". Single-homed here so a layout/bucket-rule
// change is one edit that cannot drift the resident and streamed paths apart (§8).
// PARITY-NEUTRAL (§12): pure host index math, no device work, byte-identical to the
// two former inline copies.

/// A size bucket: all blocks padded to the SAME ceil-pow{kBlockGroupPadBase} width
/// `s_pad`, fed as ONE strided-batched call padded only to that width. Hoisted to a
/// file-local struct (was a method-local definition in each of the two paths).
struct Bucket { int s_pad = 0; std::vector<int> blocks; };

/// The per-block layout from block_id (contiguous in file order). The SINGLE-SOURCE
/// inverse of assign_blocks: block_ranges validates the partition contract ONCE
/// (0 <= id < n_block, non-decreasing, block_id long enough) and returns each block's
/// half-open column range [begin, end). Fills block_offsets[b] = begin and
/// block_sizes[b] = size (the S4 F2BlockTensor metadata / weighting denominator base).
struct BlockLayout {
    std::vector<long> block_offsets;  ///< each block's first SNP column (= range.begin)
    std::vector<int>  block_sizes;    ///< each block's SNP count (= range.size())
};

[[nodiscard]] BlockLayout compute_block_layout(const int* block_id, long M, int n_block) {
    BlockLayout out;
    out.block_offsets.assign(static_cast<std::size_t>(n_block < 0 ? 0 : n_block), 0);
    out.block_sizes.assign(static_cast<std::size_t>(n_block < 0 ? 0 : n_block), 0);
    const std::vector<core::BlockRange> ranges =
        core::block_ranges(std::span<const int>(block_id, static_cast<std::size_t>(M)),
                           M, n_block);
    for (int b = 0; b < n_block; ++b) {
        out.block_offsets[static_cast<std::size_t>(b)] = ranges[static_cast<std::size_t>(b)].begin;
        out.block_sizes[static_cast<std::size_t>(b)] =
            static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
    }
    return out;
}

/// Group blocks by ceil-pow{kBlockGroupPadBase}(size) (the spike rule): one
/// strided-batched call per bucket, padded only to the bucket width; bounds pad waste
/// < base× WITHIN a bucket while keeping the call count O(log max_size). Buckets sorted
/// by width (cosmetic / smallest first). Byte-identical to the two former inline copies.
[[nodiscard]] std::vector<Bucket> size_buckets(const std::vector<int>& block_sizes,
                                               int n_block) {
    auto ceil_bucket = [](int x) {
        int p = 1;
        while (p < x) p *= steppe::kBlockGroupPadBase;
        return p;
    };
    std::vector<Bucket> buckets;
    for (int b = 0; b < n_block; ++b) {
        const int sp = ceil_bucket(block_sizes[static_cast<std::size_t>(b)]);
        int gi = -1;
        for (std::size_t k = 0; k < buckets.size(); ++k)
            if (buckets[k].s_pad == sp) { gi = static_cast<int>(k); break; }
        if (gi < 0) { gi = static_cast<int>(buckets.size()); buckets.push_back({sp, {}}); }
        buckets[static_cast<std::size_t>(gi)].blocks.push_back(b);
    }
    std::sort(buckets.begin(), buckets.end(),
              [](const Bucket& a, const Bucket& c) { return a.s_pad < c.s_pad; });
    return buckets;
}

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

/// GPU compute backend. The 3-GEMM f2 reformulation; one CublasHandle created
/// Rebuild the minimal core::qpadm::QpGraphModel the host edge-recovery objective
/// consumes from the CUDA-free arena (the final ONE-eval edge solve at the winning theta;
/// NOT the per-iteration objective — that runs on-device in the fleet kernel).
namespace {
[[nodiscard]] core::qpadm::QpGraphModel topo_to_model(const QpGraphTopoArena& topo) {
    core::qpadm::QpGraphModel m;
    m.npop = topo.npop; m.nedge_norm = topo.nedge_norm; m.nadmix = topo.nadmix;
    m.npair = topo.npair; m.npath = topo.npath; m.base_leaf = topo.base_leaf;
    m.pwts0 = topo.pwts0;
    m.pe_edge = topo.pe_edge; m.pe_leaf = topo.pe_leaf; m.pe_path = topo.pe_path;
    m.pae_path = topo.pae_path; m.pae_admixedge = topo.pae_admixedge;
    m.cmb1 = topo.cmb1; m.cmb2 = topo.cmb2;
    return m;
}
}  // namespace

/// once (architecture.md §7) and reused, with its workspace set for emulated-FP64
/// determinism. Move-only via the ComputeBackend base (architecture.md §8).
class CudaBackend final : public ComputeBackend {
public:
    /// Construct a backend BOUND to one CUDA device (the per-device-instance
    /// contract, backend.hpp; architecture.md §9 PerGpuResources, §11.4 SPMG).
    /// `device_id` is the physical CUDA ordinal this instance owns — `Resources`
    /// passes `DeviceConfig::devices[g]`; the default 0 keeps the single-GPU path
    /// (and every existing zero-arg call site) bound to device 0, unchanged.
    ///
    /// RECORD-AND-SET (architecture.md §7 — the wrapper RAII types record-and-ASSERT
    /// the device, never `cudaSetDevice`; the BACKEND is the owner that legitimately
    /// SELECTS it). The `cudaSetDevice(device_id)` must run BEFORE the `blas_` /
    /// `workspace_` members construct, because "a cuBLAS library context is tightly
    /// coupled with the CUDA context that is current at the time of the
    /// `cublasCreate()` call" (cuBLAS §2.1.2, CUDA 13.x), and `DeviceBuffer`'s
    /// `cudaMalloc` allocates on the current device — so both must see `device_id`
    /// current, not the ambient entry device. C++ initializes members in DECLARATION
    /// order, and `device_id_` is declared FIRST (below); we set the device while
    /// initializing it via `set_and_return_device`, so the set is sequenced before
    /// `blas_`/`workspace_` are built. Result: the handle's cuBLAS context (and
    /// `CublasHandle::device_id()`, which records `cudaGetDevice` at creation) and the
    /// workspace VRAM are both bound to `device_id`, and the per-call `guard_device()`
    /// re-selects it on every compute entry (so a later ambient-device change cannot
    /// run this backend's work on the wrong GPU, and the §11.4 CublasHandle
    /// device-ordinal debug assert holds).
    explicit CudaBackend(int device_id = 0)
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

    /// The factored GEMM body's output (run_f2_blocks_resident): the resident
    /// [P × P × n_block] f2/Vpair tensors returned BY VALUE (move) with the host-side
    /// block_sizes — NO D2H, NO copy-back. Both public overrides (compute_f2_blocks
    /// host D2H, compute_f2_blocks_resident DevicePartial wrap) consume this. The
    /// per-block bits are identical regardless of caller (§12). Declared BEFORE the
    /// methods that name it as a return type (the member-function declaration's return
    /// type is NOT in the complete-class context, so it must be visible here first).
    struct ResidentBlocks {
        DeviceBuffer<double> f2;     // [P*P*n_block] resident on device_id_ (or empty)
        DeviceBuffer<double> vpair;  // [P*P*n_block] resident on device_id_ (or empty)
        std::vector<int> block_sizes;
        int P = 0;
        int n_block = 0;
    };

    [[nodiscard]] F2Result compute_f2(const core::MatView& Q,
                                      const core::MatView& V,
                                      const core::MatView& N,
                                      const Precision& precision) override {
        guard_device();
        const int P = Q.P;
        const long M = Q.M;

        // ---- Degenerate-extent early return (cleanup X-7/E-3, B12) -----------
        // Sibling-consistency: compute_f2_blocks and decode_af both early-return
        // an empty result on a zero/negative extent (architecture.md §2 fail-
        // fast). Without this guard a zero/negative P or M throws from deep in
        // the runtime / cuBLAS instead of returning cleanly: M==0 makes
        // core::cdiv(0,16)==0, a zero-extent grid the CUDA driver rejects with
        // cudaErrorInvalidConfiguration (CUDA Runtime API §kernel-launch: a grid
        // dimension of 0 is an invalid launch configuration), and a non-negative
        // M==0 feeds k==0 into cublasGemmEx → CUBLAS_STATUS_INVALID_VALUE (cuBLAS
        // §2.x: k must be >= 0 and the GEMM is a no-op only for k==0 with defined
        // alpha/beta, but our downstream assemble assumes a populated G). A
        // negative P/M would also wrap the size_t extent products below into a
        // ~1.8e19 allocation reported as DeviceOom, not the InvalidConfig the
        // contract intends. Returning an empty F2Result (out.P carries the given
        // P, f2/vpair empty) makes the degenerate case a clean no-op — the shape
        // M4.5 multi-GPU sharding needs when a device is handed an empty SNP
        // shard.
        F2Result out;
        out.P = P;
        if (P <= 0 || M <= 0) return out;

        // ---- M0 contraction-extent narrowing guard (cleanup f2_block_kernel E-1,
        //      B22) -------------------------------------------------------------
        // This is the M0 WHOLE-matrix path: run_f2_gemms issues the three GEMMs over
        // ALL M SNPs in one shot via cublasGemmEx, whose contraction extent `k` is a
        // signed `int` (cuBLAS Library API: cublasGemmEx takes `int m, n, k`; only
        // the cublas*Ex_64 variants take int64_t). MatView::M is deliberately `long`
        // (views.hpp: "so a large SNP block does not overflow 32-bit"), and the M0
        // path does NOT tile — it uploads all M — so at AADR/streaming scale (§11.1
        // pushes 1e6-1e7 SNPs; the only ceiling here is VRAM for the [P×M] uploads,
        // which at P=768 admits M well past 2^31) the `static_cast<int>(M)` in
        // run_f2_gemms is reachable. For M > INT_MAX that conversion is
        // implementation-defined (a wrapped/negative `k`) → cuBLAS either rejects
        // with CUBLAS_STATUS_INVALID_VALUE or, worse, silently contracts over fewer
        // SNPs → a wrong-but-plausible f2. Fail FAST here (architecture.md §2) BEFORE
        // any device allocation or the narrowing, with a typed, descriptive error —
        // never a silent overflow. (The M4 compute_f2_blocks path is NOT affected:
        // its cublasGemmStridedBatchedEx contracts over `s_pad` = one block's padded
        // SNP count, which is small by construction.)
        if (M > static_cast<long>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "steppe::device::CudaBackend::compute_f2: SNP count M=" +
                std::to_string(M) + " exceeds INT_MAX (" +
                std::to_string(std::numeric_limits<int>::max()) +
                "); the M0 whole-matrix cublasGemmEx contraction extent k is a 32-bit "
                "int. Tile the SNP axis or route this path through cublasGemmEx_64 "
                "(cleanup B22).");
        }

        const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
        const std::size_t two_pm = 2u * pm;
        const std::size_t pp = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
        const std::size_t two_pp = 2u * pp;

        // ---- Device allocations (RAII; freed on scope exit) ------------------
        // Inputs (raw Q/V/N from the contract) + feeder outputs (masked Q, V,
        // stacked S=[Qsq;Hc]) + GEMM outputs (G, Vpair, R) + final f2. Only
        // [P×M], [2P×M], [P×P], [2P×P] buffers — never the [SNP×pop×pop]
        // intermediate (architecture.md §5 S2, §11.1).
        DeviceBuffer<double> dQ_raw(pm), dV_raw(pm), dN_raw(pm);
        DeviceBuffer<double> dQ(pm), dV(pm), dS(two_pm);
        DeviceBuffer<double> dG(pp), dVpair(pp), dR(two_pp), dF2(pp);

        // ---- Upload the Q/V/N contract (column-major [P × M]) ----------------
        // AMORTIZED-pin the H2D source pages (P4/L2) so these are genuine async DMAs
        // that overlap the peer device's copies/compute under the §11.4 fan-out.
        // `cudaMemcpyAsync` is host-async ONLY from page-locked memory (CUDA Runtime
        // API); from pageable host memory it falls back to a synchronous staging copy.
        // The cache registers each (ptr,bytes) ONCE and reuses it across calls — the
        // register cost is paid once, not per call (the bench/M5 reuse the same Q/V/N).
        // A failed pin degrades to pageable with a debug warning. PARITY-NEUTRAL (same
        // bytes; architecture.md §11.4, §12).
        const std::size_t raw_bytes = pm * sizeof(double);
        pinned_in_.ensure(Q.data, raw_bytes);
        pinned_in_.ensure(V.data, raw_bytes);
        pinned_in_.ensure(N.data, raw_bytes);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ_raw.data(), Q.data, raw_bytes,
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV_raw.data(), V.data, raw_bytes,
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dN_raw.data(), N.data, raw_bytes,
                                          cudaMemcpyHostToDevice, stream_.get()));

        // ---- Fused feeder -> 3 GEMMs -> fused numerator/divide ---------------
        launch_f2_feeder(dQ_raw.data(), dV_raw.data(), dN_raw.data(),
                         dQ.data(), dV.data(), dS.data(), P, M, stream_.get());
        // The handle's stream + emulated-FP64 workspace are bound ONCE in the ctor
        // (architecture.md §12; cleanup X-1/B1); run_f2_gemms never re-binds the
        // stream, so the determinism workspace survives every GEMM.
        run_f2_gemms(blas_.get(), precision, P, M,
                     dQ.data(), dV.data(), dS.data(),
                     dG.data(), dVpair.data(), dR.data());
        launch_assemble_f2(dG.data(), dVpair.data(), dR.data(), dF2.data(), P, stream_.get());

        // ---- Copy results back across the CUDA-free seam --------------------
        // `out` (with out.P set) was constructed before the degenerate guard. The D2H
        // result vectors are FRESHLY allocated each call (a new base pointer every
        // time), so a per-call cudaHostRegister would pay the page-locking tax with
        // ZERO amortization — a strict loss (MEASURED, perf-discovery P4). The result
        // copies therefore stay PAGEABLE; only the stable, reused H2D inputs above are
        // worth pinning.
        out.f2.resize(pp);
        out.vpair.resize(pp);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.f2.data(), dF2.data(), pp * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.vpair.data(), dVpair.data(),
                                          pp * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        return out;
    }

    /// M4 — PER-BLOCK f2 via the SPIKE-CHOSEN size-grouped strided-batched design
    /// (architecture.md §5 S2, §11.1; ROADMAP M4). One fused feeder over ALL SNPs
    /// (the existing block-agnostic launch_f2_feeder), then per power-of-2 size
    /// bucket: gather the bucket's blocks into a padded slab → 3 strided-batched
    /// GEMMs → fused assemble (native FP64) scattered into the resident
    /// [P × P × n_block] f2 + Vpair tensors. Only ONE bucket's padded slabs + GEMM
    /// outputs are resident at a time (VRAM-frugal; the spike's grouped design),
    /// alongside the persistent feeder outputs and the resident f2/Vpair tensors.
    /// M4 host F2BlockTensor — now a THIN WRAPPER over the device-resident primary +
    /// the opt-in to_host materialization (the M4.5 cure). It runs
    /// compute_f2_blocks_device (result stays in VRAM) then .to_host() (the ONE D2H +
    /// host alloc). Bit-identical to the prior body (same GEMM body, same D2H over
    /// total doubles): the only change is the host alloc/zero/copy is now the opt-in
    /// to_host, not forced inline. The hot path (the fit handoff) does NOT call this.
    [[nodiscard]] F2BlockTensor compute_f2_blocks(const core::MatView& Q,
                                                  const core::MatView& V,
                                                  const core::MatView& N,
                                                  const int* block_id,
                                                  int n_block,
                                                  const Precision& precision) override {
        return compute_f2_blocks_device(Q, V, N, block_id, n_block, precision).to_host();
    }

    /// M4.5 device-resident PRIMARY (the cure). Runs the SAME GEMM body
    /// (run_f2_blocks_resident), then MOVES the resident f2/Vpair DeviceBuffers into a
    /// DeviceF2Blocks — NO D2H, NO free, NO host alloc. The full result ESCAPES into
    /// the handle (VRAM-resident on device_id_). Bit-identical to compute_f2_blocks's
    /// resident bits (same run_f2_blocks_resident; §12).
    [[nodiscard]] DeviceF2Blocks compute_f2_blocks_device(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, const Precision& precision) override {
        ResidentBlocks rb = run_f2_blocks_resident(Q, V, N, block_id, n_block, precision);
        DeviceF2Blocks h;
        h.P = rb.P;
        h.n_block = rb.n_block;
        h.device_id = device_id_;
        h.block_sizes = std::move(rb.block_sizes);
        if (h.n_block > 0 && h.P > 0) {
            h.impl = std::make_unique<DeviceF2Blocks::Impl>();
            // Buffers ESCAPE: ownership -> handle. They were cudaMalloc'd on
            // device_id_ (guard_device) but free LATER, in DeviceF2Blocks's dtor run
            // by the host-side combine under a possibly-different (device-0) ambient
            // device. That cross-device free is SOUND because cudaFree is device-
            // agnostic (it carries the pointer's device), so the wrappers omit a
            // record-and-restore cudaSetDevice by design (cleanup [17.5]; STANDARDS
            // block above; single home device_buffer.cuh).
            h.impl->f2 = std::move(rb.f2);
            h.impl->vpair = std::move(rb.vpair);
        }
        return h;  // NO D2H, NO free — the result STAYS in VRAM
    }

    /// M4.5 device-resident override (the cure, doc §4 Item 1). Runs the SAME GEMM
    /// body, then MOVES the resident f2/Vpair DeviceBuffers into a DevicePartial —
    /// NO D2H, NO free. The buffers ESCAPE this call into the returned handle (they
    /// survive the jthread join and free only AFTER the combine consumed them, §7).
    [[nodiscard]] DevicePartial compute_f2_blocks_resident(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, int b0, const Precision& precision) override {
        ResidentBlocks rb = run_f2_blocks_resident(Q, V, N, block_id, n_block, precision);
        DevicePartial h;
        h.P = rb.P;
        h.n_block_local = rb.n_block;
        h.b0 = b0;
        h.device_id = device_id_;
        h.block_sizes = std::move(rb.block_sizes);
        if (h.n_block_local > 0) {
            h.impl = std::make_unique<DevicePartial::Impl>();
            // Buffers ESCAPE: ownership -> handle; they survive the jthread join and
            // free only AFTER the combine consumed them (§7). The free runs in
            // DevicePartial's dtor under a possibly-device-0 ambient device, NOT
            // device_id_ — SOUND because cudaFree is device-agnostic (carries the
            // pointer's device), so the owner omits a record-and-restore cudaSetDevice
            // by design (cleanup [17.5]; STANDARDS block above).
            h.impl->f2 = std::move(rb.f2);
            h.impl->vpair = std::move(rb.vpair);
        }
        return h;  // NO D2H, NO free — the cure (doc §4 Item 1)
    }

    /// M4.5 host-staged override (the d2h-speed cure). Runs the SAME GEMM body as
    /// compute_f2_blocks (so the per-block bits are bit-identical; §12), then D2Hs the
    /// compact f2/vpair slabs into PERSISTENT per-backend pinned staging buffers
    /// (stage_f2_/stage_vpair_, cudaHostAlloc'd ONCE and reused), and a host std::memcpy
    /// copies the exact bytes into the caller's shared result at the disjoint block
    /// offset slab_off = P*P*b0. The persistent staging replaces the prior per-call
    /// RegisteredHostRegion pin of the ~3 GB result slice: cudaHostRegister/Unregister
    /// took the device-wide driver lock and serialized the two workers' D2Hs (~570 ms
    /// serial tail, MEASURED nsys box5090). Now the two devices D2H into their OWN
    /// buffers as concurrent pinned DMAs, and the staging->result memcpy is CPU
    /// bandwidth (no driver lock) running concurrently on the two worker threads.
    /// block_sizes for this device's blocks are placed at the shared result's
    /// [b0, b0+n_block) (host int copy, mirrors f2_combine.cpp's std::copy_n at offset
    /// b0). PARITY-NEUTRAL: same doubles, same disjoint offset, exact memcpy (§12).
    void compute_f2_blocks_into(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, int b0,
        double* dst_f2, double* dst_vpair, int* block_sizes_dst,
        const Precision& precision) override {
        ResidentBlocks rb = run_f2_blocks_resident(Q, V, N, block_id, n_block, precision);
        // block_sizes for this device's blocks -> the shared host result at [b0, b0+nb).
        // (host int copy; mirrors f2_combine.cpp std::copy_n at offset b0.)
        for (int b = 0; b < rb.n_block; ++b)
            block_sizes_dst[static_cast<std::size_t>(b0) + static_cast<std::size_t>(b)] =
                rb.block_sizes[static_cast<std::size_t>(b)];
        const std::size_t total = rb.f2.size();  // P*P*n_block (0 on degenerate/empty shard)
        if (total > 0) {
            guard_device();  // rb lives on device_id_
            const std::size_t slab_off =
                static_cast<std::size_t>(rb.P) * static_cast<std::size_t>(rb.P) *
                static_cast<std::size_t>(b0);
            const std::size_t bytes = total * sizeof(double);

            // Grow the PERSISTENT pinned staging to this partial ONCE (reused thereafter).
            // The partial size P*P*n_block_local is constant across a run's repeated
            // calls, so this allocates on the first call and is a no-op after. Never
            // shrink. Two separate buffers so the f2 and vpair D2Hs stage independently
            // (one buffer would falsely serialize the two copies on the one stream).
            if (stage_f2_.size()    < total) stage_f2_    = PinnedBuffer<double>(total);
            if (stage_vpair_.size() < total) stage_vpair_ = PinnedBuffer<double>(total);

            // D2H into the PERSISTENT pinned staging (genuine async DMA; NO per-call
            // register/unregister, so NO device-wide driver lock). The two devices stage
            // into their OWN per-backend buffers -> concurrent DMAs.
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(stage_f2_.data(), rb.f2.data(), bytes,
                                              cudaMemcpyDeviceToHost, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(stage_vpair_.data(), rb.vpair.data(), bytes,
                                              cudaMemcpyDeviceToHost, stream_.get()));
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

            // Host copy into the caller's disjoint result slice. CPU bandwidth, NO driver
            // lock, so the two worker threads' copies run concurrently. Disjoint
            // block-aligned slices => race-free into the one shared result. Exact bytes
            // => PARITY (architecture.md §12). The trailing sync above is the
            // happens-before that guarantees the DMA fully drained into staging first.
            std::memcpy(dst_f2    + slab_off, stage_f2_.data(),    bytes);
            std::memcpy(dst_vpair + slab_off, stage_vpair_.data(), bytes);
        }
        // rb's DeviceBuffers free here (same free point as compute_f2_blocks).
    }

    /// M5 STREAMED (out-of-core) override — the HostRam + Disk tiers (backend.hpp
    /// compute_f2_blocks_streamed). Builds the concrete sink the CUDA-free StreamTarget
    /// selects (HostRamSink into target.host_dst, or DiskSink to target.disk_path),
    /// then drives the block-stream loop (stream_f2_blocks_impl), which REUSES the
    /// run_f2_blocks_resident prologue + per-block gather/GEMM/assemble VERBATIM and
    /// spills each block's [P²] slab through the triple-buffered sink. The per-block
    /// bits are BIT-IDENTICAL to the device-resident path (§12); the ONLY difference is
    /// the result is spilled block-by-block instead of left whole. Resident NEVER routes
    /// here (the orchestrator calls compute_f2_blocks_device directly).
    void compute_f2_blocks_streamed(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, const Precision& precision,
        StreamTarget& target) override {
        if (target.tier == OutputTier::HostRam) {
            if (!target.host_dst)
                throw std::runtime_error("compute_f2_blocks_streamed: HostRam tier needs a "
                                         "host_dst destination");
            HostRamSink sink(*target.host_dst);
            stream_f2_blocks_impl(Q, V, N, block_id, n_block, precision, sink);
        } else if (target.tier == OutputTier::Disk) {
            if (!target.disk_dst)
                throw std::runtime_error("compute_f2_blocks_streamed: Disk tier needs a "
                                         "disk_dst descriptor");
            DiskSink sink(target.disk_path);
            stream_f2_blocks_impl(Q, V, N, block_id, n_block, precision, sink);
            sink.take_descriptor(*target.disk_dst);
        } else {
            throw std::runtime_error("compute_f2_blocks_streamed: Resident tier must not "
                                     "route through the stream seam (it bypasses it)");
        }
    }

    /// M4 — PER-BLOCK f2 via the SPIKE-CHOSEN size-grouped strided-batched design
    /// (architecture.md §5 S2, §11.1; ROADMAP M4). The factored GEMM body SHARED by
    /// compute_f2_blocks (host) and compute_f2_blocks_resident: it produces the two
    /// resident [P × P × n_block] f2/Vpair DeviceBuffers (returned BY MOVE in
    /// ResidentBlocks) + the host-side block_sizes — WITHOUT any D2H and WITHOUT
    /// freeing the buffers. Both public overrides differ ONLY in what they do with the
    /// returned buffers (host D2H, or wrap in a DevicePartial); the per-block bits are
    /// identical regardless of caller (§12 — same run_f2_gemms_group calls, same fixed
    /// bucket order, same block_sizes, same per-chunk sync).
    [[nodiscard]] ResidentBlocks run_f2_blocks_resident(const core::MatView& Q,
                                                        const core::MatView& V,
                                                        const core::MatView& N,
                                                        const int* block_id,
                                                        int n_block,
                                                        const Precision& precision) {
        guard_device();
        const int P = Q.P;
        const long M = Q.M;

        ResidentBlocks rb;
        rb.P = P;
        rb.n_block = n_block;
        const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
        const std::size_t total = slab * static_cast<std::size_t>(n_block < 0 ? 0 : n_block);
        rb.block_sizes.assign(static_cast<std::size_t>(n_block < 0 ? 0 : n_block), 0);
        if (P <= 0 || M <= 0 || n_block <= 0) return rb;

        // ---- Block layout + size-bucketing (shared prologue; [7.1] dedup) ----
        // compute_block_layout is the SINGLE-SOURCE inverse of assign_blocks (validates
        // the partition contract ONCE, closing the OOB write/read this scan used to risk
        // on a malformed partition — cleanup X-3/B3) returning block_offsets (= each
        // range's begin, copied to dOffsets + dereferenced by f2_blocks_kernel.cu) and
        // block_sizes (= each range's size, the S4 metadata / weighting denominator
        // base). size_buckets then groups into the strided-batched buckets. Both are
        // shared verbatim with stream_f2_blocks_impl so the two paths cannot drift (§8).
        const BlockLayout layout = compute_block_layout(block_id, M, n_block);
        const std::vector<long>& block_offsets = layout.block_offsets;
        rb.block_sizes = layout.block_sizes;
        const std::vector<Bucket> buckets = size_buckets(rb.block_sizes, n_block);

        // ---- VRAM budget for one strided-batched chunk (architecture.md §11.2).
        // Query free VRAM BEFORE committing the resident set, so the budget helper
        // genuinely subtracts the resident f2+Vpair tensors AND the cuBLAS workspace
        // from a gross-free figure rather than relying on allocation ordering
        // (cleanup X-5/B5 + X-13/B26). The helper accounts for BOTH P²·n_block FP64
        // tensors (the prior path counted one ⇒ ~2× under-budget) and reserves the
        // determinism workspace before applying kMaxVramUtilizationFraction; the
        // resulting per-bucket `max_blocks` bounds how many blocks of a bucket fit
        // in one strided-batched call. A single big bucket's slabs would otherwise
        // OOM (at P=768 the 2048-pad bucket alone exceeds what is left). The helper
        // is host-pure + unit-tested GPU-free (device/vram_budget.hpp).
        std::size_t free_b = 0, total_b = 0;
        STEPPE_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

        // ---- Resident, run-long device tensors + the feeder outputs over ALL M.
        // VRAM is the binding constraint at scale (architecture.md §11.2): the raw
        // inputs (dQ_raw/dV_raw/dN_raw, 3·P·M doubles) are needed ONLY by the
        // feeder, so they live in an inner scope and free BEFORE the bucket loop —
        // otherwise at P=768/M=584k the raw (10.8 GB) + feeder outputs (17.9 GB) +
        // resident f2/Vpair (7.1 GB) sum to 35.8 GB > 32 GB and OOM. After the
        // feeder runs, only dQ/dV/dS (the masked Q, V, S=[Qsq;Hc]) and the resident
        // f2/Vpair tensors persist (~25 GB), leaving headroom for the bucket slabs.
        const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
        const std::size_t two_pm = 2u * pm;
        DeviceBuffer<double> dQ(pm), dV(pm), dS(two_pm);     // feeder outputs (persist)
        // The resident tensors — owned by `rb` so they ESCAPE this helper (the host
        // override D2H-copies + frees them; the resident override moves them into a
        // DevicePartial). They are the EXACT prior dF2_all/dVpair_all (same `total`).
        rb.f2 = DeviceBuffer<double>(total);
        rb.vpair = DeviceBuffer<double>(total);
        DeviceBuffer<long>   dOffsets(static_cast<std::size_t>(n_block));
        DeviceBuffer<int>    dSizes(static_cast<std::size_t>(n_block));

        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dOffsets.data(), block_offsets.data(),
                                          static_cast<std::size_t>(n_block) * sizeof(long),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSizes.data(), rb.block_sizes.data(),
                                          static_cast<std::size_t>(n_block) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));

        {
            // Raw inputs — feeder-only; freed at the end of this scope.
            DeviceBuffer<double> dQ_raw(pm), dV_raw(pm), dN_raw(pm);
            // ---- AMORTIZED-pin the Q/V/N H2D SOURCE pages (P4/L2; perf-discovery).
            // These are the dominant copies (3·P·M doubles — ~5.4 GB at P=768/M=584k
            // single-GPU, ~half that per device under the fan-out). `cudaMemcpyAsync`
            // is host-async ONLY from page-locked memory; from the caller's PAGEABLE
            // Q/V/N it falls back to a synchronous host-blocking staging copy (CUDA
            // Runtime API, `cudaMemcpyAsync`: "For transfers from pageable host memory
            // ... the function is synchronous"), so under the §11.4 fan-out the two
            // devices' H2Ds contend on the driver's internal staging instead of
            // running as concurrent DMAs (the measured ~44 % pageable copy stall,
            // perf-discovery §2; MEASURED ~109 ms/iter pageable vs ~51 ms/iter pinned
            // per device, two-device concurrent, on rtxbox). The cache registers each
            // (ptr,bytes) ONCE and reuses it across calls — registering a multi-GB
            // range is itself a ~50–360 ms page-walk, so per-call registration would
            // be a net loss; amortizing it across the repeated calls (the bench/M5
            // reuse the same host Q/V/N) is what makes the two concurrent pinned DMAs
            // a net win. A failed pin (RLIMIT_MEMLOCK) degrades to pageable with a
            // debug warning — never a crash. PARITY-NEUTRAL: pinned vs pageable moves
            // the identical bytes (architecture.md §11.4, §12).
            const std::size_t raw_bytes = pm * sizeof(double);
            pinned_in_.ensure(Q.data, raw_bytes);
            pinned_in_.ensure(V.data, raw_bytes);
            pinned_in_.ensure(N.data, raw_bytes);
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ_raw.data(), Q.data, raw_bytes,
                                              cudaMemcpyHostToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV_raw.data(), V.data, raw_bytes,
                                              cudaMemcpyHostToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dN_raw.data(), N.data, raw_bytes,
                                              cudaMemcpyHostToDevice, stream_.get()));
            // ONE fused feeder over ALL SNPs (block-agnostic; native FP64).
            launch_f2_feeder(dQ_raw.data(), dV_raw.data(), dN_raw.data(),
                             dQ.data(), dV.data(), dS.data(), P, M, stream_.get());
            // Sync so the feeder finishes before dQ_raw/dV_raw/dN_raw free (the
            // gather then reads dQ/dV/dS, and the freed raw VRAM is reused by the
            // bucket slabs).
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        }

        // Engage the precision policy ONCE (the SHARED engagement; the grouped GEMM
        // routine then sets only the per-call compute type). The handle's workspace
        // was bound in the ctor for emulated-FP64 determinism (architecture.md §12).
        engage_f2_precision(blas_.get(), precision);

        // ---- Pre-size the per-chunk slabs ONCE, reuse across every chunk (L4b;
        //      perf-discovery.md P3). --------------------------------------------
        // The prior path allocated+freed dIds/dQg/dVg/dSg/dGg/dVpairg/dRg INSIDE the
        // chunk loop — 645 `cudaMalloc` + 648 `cudaFree` on the P=768 run. Both
        // `cudaMalloc` and `cudaFree` are DEVICE-WIDE-SYNCHRONIZING and take the
        // global allocator/driver lock (CUDA C Programming Guide §3.2.2 "Device
        // Memory"; CUDA Runtime API: cudaMalloc/cudaFree synchronize the device), so
        // under the §11.4 SPMG fan-out the two per-device worker threads serialized
        // against each other on that lock — a measured 7.1 % `cudaFree` + 2.6 %
        // `cudaMalloc` CUDA-API time and the 18 % kernel overlap (perf-discovery P3).
        //
        // FIX: allocate each slab ONCE at the MAX single-chunk footprint over all
        // buckets, OUTSIDE the loop, and REUSE it for every chunk. The kernel
        // wrappers index strictly by the passed `P`/`s_pad`/`nb` (never by buffer
        // size; f2_blocks_kernel.cuh), so a chunk uses only the leading
        // `P·s_pad·nb` / `P²·nb` elements — over-sizing changes only WHEN the VRAM is
        // committed, not a single bit (PARITY-SAFE: same buffers, same math, every
        // used element fully written before read; the gather writes all `s_pad`
        // columns incl. pad, the GEMM writes its outputs, before any read).
        //
        // §11.2 VRAM budget (architecture.md §11.2) is UPHELD: the peak resident set
        // is unchanged — `max_blocks_per_chunk` already bounded EACH chunk's slabs to
        // `kMaxVramUtilizationFraction` of the post-resident-set/post-workspace VRAM,
        // and the loop already reached that peak transiently. Pre-allocating at the
        // MAX over buckets (the largest single chunk, NOT the sum of all buckets)
        // commits exactly that one-chunk peak once and holds it for the loop — never
        // more than `resident + one max-chunk` is co-resident, the same ceiling the
        // budget helper enforced per chunk. The two slab families scale differently
        // (Qg/Vg/Sg with `P·s_pad·nb`, Gg/Vpairg/Rg with `P²·nb`), and the bucket
        // maximizing each may differ, so each cap is tracked independently for a
        // tight (sound, non-over-committed) bound.
        std::size_t max_nb = 0;       // max blocks-per-chunk over all buckets (dIds, *_g)
        std::size_t max_psp_nb = 0;   // max P·s_pad·nb over all buckets (Qg/Vg/Sg)
        std::vector<int> bucket_max_blocks(buckets.size(), 0);
        for (std::size_t bi = 0; bi < buckets.size(); ++bi) {
            const int s_pad = buckets[bi].s_pad;
            const int nb_total = static_cast<int>(buckets[bi].blocks.size());
            const int mb = max_blocks_per_chunk(free_b, P, n_block, s_pad, nb_total);
            bucket_max_blocks[bi] = mb;
            const std::size_t nb = static_cast<std::size_t>(mb < 0 ? 0 : mb);
            max_nb = std::max(max_nb, nb);
            max_psp_nb = std::max(
                max_psp_nb,
                static_cast<std::size_t>(P) * static_cast<std::size_t>(s_pad) * nb);
        }
        // The reused slabs (RAII; freed once at function scope exit, NOT per chunk).
        DeviceBuffer<int>    dIds(max_nb);
        DeviceBuffer<double> dQg(max_psp_nb), dVg(max_psp_nb), dSg(2u * max_psp_nb);
        const std::size_t max_pp_nb = slab * max_nb;
        DeviceBuffer<double> dGg(max_pp_nb), dVpairg(max_pp_nb), dRg(2u * max_pp_nb);

        // ---- Per bucket (chunked): gather → strided-batched GEMMs → assemble ----
        // Each bucket is processed in CHUNKS of at most `max_blocks` blocks (the
        // host-pure, GPU-free budget helper: it reserves BOTH resident tensors +
        // the cuBLAS workspace, applies kMaxVramUtilizationFraction, and clamps in
        // size_t before the int narrowing — device/vram_budget.hpp). This keeps the
        // grouped strided-batched design while bounding the working set (the M5
        // out-of-core generalization is a superset of this; here it is the
        // single-GPU VRAM guard ROADMAP M4 requires).
        for (std::size_t bi = 0; bi < buckets.size(); ++bi) {
            const Bucket& bk = buckets[bi];
            const int s_pad = bk.s_pad;
            const int nb_total = static_cast<int>(bk.blocks.size());
            const int max_blocks = bucket_max_blocks[bi];

            for (int start = 0; start < nb_total; start += max_blocks) {
                const int nb = std::min(max_blocks, nb_total - start);

                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dIds.data(), bk.blocks.data() + start,
                                                  static_cast<std::size_t>(nb) * sizeof(int),
                                                  cudaMemcpyHostToDevice, stream_.get()));

                launch_gather_group(dQ.data(), dV.data(), dS.data(),
                                    dIds.data(), dOffsets.data(), dSizes.data(),
                                    P, s_pad, nb, dQg.data(), dVg.data(), dSg.data(), stream_.get());
                // The handle's stream + emulated-FP64 workspace are bound ONCE in
                // the ctor (architecture.md §12; cleanup X-1/B1). The M4 grouped
                // path previously reset the stream — and thus the workspace — per
                // CHUNK; it no longer touches the stream, so the determinism
                // workspace survives every chunk's strided-batched GEMMs.
                run_f2_gemms_group(blas_.get(), precision, P, s_pad, nb,
                                   dQg.data(), dVg.data(), dSg.data(),
                                   dGg.data(), dVpairg.data(), dRg.data());
                launch_assemble_blocks_group(dGg.data(), dVpairg.data(), dRg.data(),
                                             dIds.data(), P, nb,
                                             rb.f2.data(), rb.vpair.data(), stream_.get());
                // Sync before the NEXT chunk overwrites the reused slabs (dQg…dRg),
                // so each chunk's gather/GEMM/assemble fully completes before the
                // next chunk's gather writes into the SAME buffers — the §12
                // single-stream-per-device serialization that keeps the reuse
                // bit-identical to the prior fresh-per-chunk buffers.
                STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
            }
        }

        // ---- Drain the stream so the resident tensors are fully written ------
        // The bucket loop's per-chunk syncs already drained each chunk; this trailing
        // sync ensures rb.f2/rb.vpair are COMPLETE before the caller reads them (the
        // host override's D2H, or the resident override's peer/D2D reads in the
        // combine). It plays the role the former final cudaStreamSynchronize before
        // the D2H played. NO D2H here — the host override does that; the resident
        // override leaves the buffers resident (the cure, doc §4 Item 1).
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        return rb;  // f2/vpair DeviceBuffers MOVE out to the caller (no free here)
    }

    /// M5 STREAMED block-stream loop (the §5 mechanism), with SNP-TILE INPUT STREAMING
    /// (m5-input-streaming). REUSES run_f2_blocks_resident's block_ranges + size-buckets +
    /// engage_f2_precision setup and the per-block gather/GEMM/assemble VERBATIM — the
    /// per-block bits are BIT-IDENTICAL to the device-resident path (§12). TWO differences
    /// from the resident path: (1) instead of the all-M feeder (the 7·P·M wall that OOM'd
    /// full-autosome at P≳768 on a 32 GB card), each chunk decodes ONLY its own SNP-column
    /// tile [s_lo,s_hi) — uploading Q/V/N[:, s_lo:s_hi] from the HOST [P×M] MatView, running
    /// the SAME launch_f2_feeder over `tile` columns into per-tile feeder buffers, then
    /// gathering via REBASED offsets (block_offsets[gid]-s_lo) into LOCAL ids; the GPU
    /// footprint is O(P·max_tile + P²·max_nb), INDEPENDENT of M (the full [P×M] stays in
    /// host RAM, owned by the caller). The tile is the SAME host columns the all-M gather
    /// read, fed per-column elementwise ⇒ bit-identical (§5.1-5.2). (2) instead of the full
    /// [P²·n_block] resident tensors, it allocates a SMALL DEVICE RING of kStreamDeviceChunks
    /// per-chunk [P²·max_nb] f2 + vpair buffers, computes each chunk into the next ring buffer
    /// (in the SAME fixed bucket→chunk order ⇒ same batchCount per group ⇒ same bits, even
    /// native Fp64; the tile-width split valve is a NO-OP at the parity sizes, so the split
    /// is identical there — §5.4), and spills each of that chunk's blocks through `sink`
    /// (which triple-buffers the D2H→tier-write). The assemble writes chunk-local slabs
    /// 0..nb-1 (a LOCAL id array), so the destination OFFSET differs from the resident path
    /// but the VALUE does not — the spill places each slab under its GLOBAL block id. No new
    /// GEMM; no recompute; only WHEN each block's columns upload moves (§5.5).
    void stream_f2_blocks_impl(const core::MatView& Q, const core::MatView& V,
                               const core::MatView& N, const int* block_id, int n_block,
                               const Precision& precision, BlockSink& sink) {
        guard_device();
        const int P = Q.P;
        const long M = Q.M;
        const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);

        // ---- Block layout + size-bucketing (shared prologue; [7.1] dedup) --------
        // The SAME compute_block_layout + size_buckets prologue as run_f2_blocks_resident
        // (single-homed above so the resident and streamed paths cannot drift; §8). The
        // degenerate empty-result early-return below predates the layout and uses the
        // all-zeros block_sizes, so it stays before the layout build.
        std::vector<int> block_sizes(static_cast<std::size_t>(n_block < 0 ? 0 : n_block), 0);
        if (P <= 0 || M <= 0 || n_block <= 0) {
            sink.begin(P, n_block, block_sizes);  // degenerate empty result -> empty tier
            sink.finish();
            return;
        }
        const BlockLayout layout = compute_block_layout(block_id, M, n_block);
        const std::vector<long>& block_offsets = layout.block_offsets;
        block_sizes = layout.block_sizes;
        const std::vector<Bucket> buckets = size_buckets(block_sizes, n_block);

        // The sink allocates the tier destination + its persistent pinned staging ONCE.
        sink.begin(P, n_block, block_sizes);

        // ---- VRAM budget (RETAINED) ----------------------------------------------
        std::size_t free_b = 0, total_b = 0;
        STEPPE_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

        // SNP-TILE INPUT STREAMING (m5-input-streaming). The all-M feeder prologue is
        // GONE: we NO LONGER allocate dQ/dV/dS over all M (the 7·P·M wall) nor upload
        // the whole [P×M] Q/V/N. Instead each chunk decodes ONLY its own SNP-column
        // tile [s_lo, s_hi) into per-tile raw+feeder buffers (allocated ONCE below at
        // the max tile width over all chunks, reused across chunks — never a per-chunk
        // cudaMalloc/cudaFree, mirroring the L4b slab pre-sizing). The full [P×M] Q/V/N
        // stay in HOST RAM (the caller's MatView); we slice sub-columns H2D per chunk.
        // PARITY (§12): block gid's columns are the SAME host columns whether read from
        // an all-M feeder (gather src = block_offsets[gid]+c) or a per-tile feeder
        // (gather src = (block_offsets[gid]-s_lo)+c built from host columns [s_lo,s_hi));
        // the feeder is block-agnostic / per-column, so feeding a column in isolation is
        // bit-identical to feeding it inside the all-M sweep. Only WHEN columns upload
        // moves — never the values.
        const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
        DeviceBuffer<int>    dSizes(static_cast<std::size_t>(n_block));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSizes.data(), block_sizes.data(),
                                          static_cast<std::size_t>(n_block) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        // AMORTIZED-pin the FULL host [P×M] Q/V/N ONCE (P4/L2): the per-chunk tile
        // slices are sub-ranges of these already-pinned regions, so a single base pin
        // covers every slice (do NOT pin per-tile sub-ranges). cudaMemcpyAsync from a
        // pinned source runs as a true async DMA; from pageable it host-blocks.
        const std::size_t raw_bytes = pm * sizeof(double);
        pinned_in_.ensure(Q.data, raw_bytes);
        pinned_in_.ensure(V.data, raw_bytes);
        pinned_in_.ensure(N.data, raw_bytes);
        engage_f2_precision(blas_.get(), precision);

        // ---- STREAMED VRAM BUDGET (NOT the resident chunk budget) ----------------
        // The resident max_blocks_per_chunk reserves resident_tensor_bytes (the FULL
        // [P²·n_block] result, co-resident on the resident tier). The STREAMED path
        // holds NO resident result — it spills block-by-block through a SMALL device
        // RING. Using the resident budget here is doubly wrong: at low-mid P it reserves
        // a 12-48 GB phantom result while the actual co-resident footprint is the
        // gather/GEMM slabs PLUS the ring (which the resident per_block_chunk_bytes does
        // NOT count), so max_nb grows to fill a budget that omits the ring and the path
        // OOMs (the measured P=1000/1500 HostRam/Disk OOM); at high P it saturates to a
        // 1-block chunk. So budget against the REAL streamed per-block footprint:
        //   gather slabs dQg+dVg+dSg : 4·P·s_pad      (per block)
        //   GEMM out dGg+dVpairg+dRg : 4·P²           (per block)
        //   device ring (×2 buffers, f2+vpair each P²): 4·P²  (per block, scaled by nb)
        //   ⇒ per-block streamed = (4·P·s_pad + 8·P²)·8 bytes
        // plus a FIXED tile-feeder reservation (7·P·max_tile_cols·8). The envelope is
        // kMaxVramUtilizationFraction of net free VRAM (free − the cuBLAS workspace),
        // split so the tile feeder takes a bounded slice and the slabs+ring take the
        // rest — guaranteeing feeder + slabs + ring ≤ fraction·free. Independent of M.
        const std::size_t net_free_b =
            (free_b > kCublasWorkspaceBytes) ? (free_b - kCublasWorkspaceBytes) : 0u;
        const std::size_t envelope_b = static_cast<std::size_t>(
            kMaxVramUtilizationFraction * static_cast<double>(net_free_b));
        // The tile feeder is a SMALL contributor (one chunk's column union); reserve a
        // kStreamTileBudgetFraction slice of the envelope for it (bounding max_tile_cols)
        // and leave the rest for the slabs+ring. At P=512/M=584k both are vast vs the
        // need, so the chunking is unconstrained and the split is a NO-OP (parity §5.4).
        const std::size_t tile_budget_b = static_cast<std::size_t>(
            kStreamTileBudgetFraction * static_cast<double>(envelope_b));
        const std::size_t slab_budget_b = envelope_b - tile_budget_b;
        const std::size_t per_col_feeder_b =
            7u * static_cast<std::size_t>(P) * sizeof(double);  // 7·P doubles per tile col
        const long max_tile_cols = static_cast<long>(std::max<std::size_t>(
            per_col_feeder_b > 0u ? (tile_budget_b / per_col_feeder_b) : 1u, 1u));

        // Streamed per-chunk block cap for one bucket of padded width `s_pad`: the
        // largest nb whose (4·P·s_pad + 8·P²)·8·nb fits slab_budget_b, clamped to
        // [1, min(nb_total, kMaxGridZ)] (a single block always attempted; the grid-z
        // batch limit mirrors max_blocks_per_chunk's cap).
        auto streamed_max_blocks = [&](int s_pad, int nb_total) -> int {
            if (nb_total <= 0) return 0;
            const std::size_t p = static_cast<std::size_t>(P);
            const std::size_t sp = static_cast<std::size_t>(s_pad < 0 ? 0 : s_pad);
            const std::size_t per_block_b =
                (4u * p * sp + 8u * p * p) * sizeof(double);
            const std::size_t fit = (per_block_b > 0u) ? (slab_budget_b / per_block_b) : 0u;
            const std::size_t capped = std::min({fit, static_cast<std::size_t>(nb_total),
                                                 static_cast<std::size_t>(core::kMaxGridZ)});
            return static_cast<int>(std::max<std::size_t>(capped, 1u));
        };

        // Compute, for ONE chunk starting at `start` of bucket `bk` with a per-chunk
        // block cap `max_blocks`, the actual block count `nb` (≤ max_blocks, ≥1) whose
        // column union [s_lo, s_hi) fits within `max_tile_cols`, plus that extent. This
        // is the SINGLE source of the bucket→chunk split — both the pre-sizing pre-pass
        // and the streaming loop call it, so the device buffer sizing and the runtime
        // chunking can never drift. Always admits at least the first block (a single
        // block must be attempted even if its own width exceeds the cap — it then OOMs
        // cleanly rather than silently producing nothing, matching max_blocks_per_chunk).
        auto chunk_extent = [&](const Bucket& bk, int start, int max_blocks,
                                long& s_lo, long& s_hi) -> int {
            const int nb_total = static_cast<int>(bk.blocks.size());
            const int cap = std::min(max_blocks, nb_total - start);
            int nb = 0;
            s_lo = 0;
            s_hi = 0;
            for (int kk = 0; kk < cap; ++kk) {
                const int gid = bk.blocks[static_cast<std::size_t>(start + kk)];
                const long b_lo = block_offsets[static_cast<std::size_t>(gid)];
                const long b_hi = b_lo + block_sizes[static_cast<std::size_t>(gid)];
                const long new_lo = (nb == 0) ? b_lo : std::min(s_lo, b_lo);
                const long new_hi = (nb == 0) ? b_hi : std::max(s_hi, b_hi);
                if (nb > 0 && (new_hi - new_lo) > max_tile_cols) break;  // tile valve
                s_lo = new_lo;
                s_hi = new_hi;
                ++nb;
            }
            return nb;  // ≥1 (the first block always admitted)
        };

        // ---- Per-chunk slab + tile sizing (the gather/GEMM scratch + tile feeder) --
        std::size_t max_nb = 0;
        std::size_t max_psp_nb = 0;
        long        max_tile = 0;
        std::vector<int> bucket_max_blocks(buckets.size(), 0);
        for (std::size_t bi = 0; bi < buckets.size(); ++bi) {
            const int s_pad = buckets[bi].s_pad;
            const int nb_total = static_cast<int>(buckets[bi].blocks.size());
            const int mb = streamed_max_blocks(s_pad, nb_total);
            bucket_max_blocks[bi] = mb;
            // Walk this bucket's chunks EXACTLY as the streaming loop will (same
            // chunk_extent split), accumulating the max nb / P·s_pad·nb / tile width.
            for (int start = 0; start < nb_total; ) {
                long s_lo = 0, s_hi = 0;
                const int nb = chunk_extent(buckets[bi], start, mb, s_lo, s_hi);
                const std::size_t nbz = static_cast<std::size_t>(nb < 0 ? 0 : nb);
                max_nb = std::max(max_nb, nbz);
                max_psp_nb = std::max(
                    max_psp_nb,
                    static_cast<std::size_t>(P) * static_cast<std::size_t>(s_pad) * nbz);
                max_tile = std::max(max_tile, s_hi - s_lo);
                start += (nb > 0 ? nb : 1);
            }
        }
        DeviceBuffer<double> dQg(max_psp_nb), dVg(max_psp_nb), dSg(2u * max_psp_nb);
        const std::size_t max_pp_nb = slab * max_nb;
        DeviceBuffer<double> dGg(max_pp_nb), dVpairg(max_pp_nb), dRg(2u * max_pp_nb);

        // ---- Per-tile raw + feeder buffers, sized at the MAX tile width over all
        //      chunks, allocated ONCE and REUSED (mirror the slab pre-sizing — never a
        //      per-chunk cudaMalloc/cudaFree, the L4b churn the P3 commit eliminated).
        //      GPU footprint is now O(P·max_tile), INDEPENDENT of M. -----------------
        const std::size_t max_tile_z = static_cast<std::size_t>(max_tile < 0 ? 0 : max_tile);
        const std::size_t p_tile = static_cast<std::size_t>(P) * max_tile_z;
        DeviceBuffer<double> dQ_raw(p_tile), dV_raw(p_tile), dN_raw(p_tile);  // raw tile H2D
        DeviceBuffer<double> dQt(p_tile), dVt(p_tile), dSt(2u * p_tile);      // tile feeder out
        // Rebased per-tile offsets: dOffsetsTile[localid] = block_offsets[gid] - s_lo,
        // dSizesTile[localid] = block_sizes[gid], and dIdsTile = [0..nb) (LOCAL into the
        // chunk's block list). The gather then reads the [P×tile] tile feeder at
        // dOffsetsTile[id] + c — exactly block gid's columns, SAME feeder bits as the
        // all-M path (which read block_offsets[gid]+c from the all-M feeder).
        DeviceBuffer<long> dOffsetsTile(max_nb);
        DeviceBuffer<int>  dSizesTile(max_nb);
        std::vector<long>  h_offsets_tile(max_nb);
        std::vector<int>   h_sizes_tile(max_nb);

        // LOCAL id array [0,1,...,max_nb-1], uploaded ONCE: BOTH the gather and the
        // assemble use it. The gather indexes the rebased per-tile dOffsetsTile/
        // dSizesTile by this LOCAL id (its first nb entries are this chunk's blocks in
        // bucket-list order, so dOffsetsTile[localid] = block_offsets[gid]-s_lo selects
        // exactly block gid's columns in the [P×tile] tile feeder — same SNPs, same
        // bits as the all-M path). The assemble writes chunk-local slabs 0..nb-1 of the
        // small per-chunk f2/vpair buffer (instead of the resident path's global P²·id
        // offset). So the COMPUTED bits are identical; only the destination OFFSET is
        // chunk-local (a write location, not a value) and the spill restores the GLOBAL id.
        std::vector<int> local_ids(max_nb);
        for (std::size_t k = 0; k < max_nb; ++k) local_ids[k] = static_cast<int>(k);
        DeviceBuffer<int> dIdsLocal(max_nb);
        if (max_nb > 0)
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dIdsLocal.data(), local_ids.data(),
                                              max_nb * sizeof(int),
                                              cudaMemcpyHostToDevice, stream_.get()));

        // SMALL DEVICE RING of kStreamDeviceChunks (2) per-chunk [P²·max_nb] f2/vpair
        // buffers (the §5 device ring). Chunk c computes into ring buffer c%2; an event per
        // ring buffer is recorded after the chunk's per-block spill D2Hs are issued, so a
        // ring buffer is reused only after its prior chunk's D2Hs drained — the
        // device-side half of the overlap. The SINK's background writer (its pinned ring)
        // is the host-side half: it does the slow host-copy/pwrite CONCURRENTLY with this
        // loop's next-chunk compute. Two device buffers (not three) keep the device ring's
        // VRAM small (the device buffer only needs to survive its own D2H, NOT the slow
        // write — the pinned ring absorbs that latency), so a chunk-tile fits alongside the
        // feeder envelope even at large M. The ring depth is single-homed in config.hpp
        // (kStreamDeviceChunks) so this real alloc and the tier-select budget term in
        // tier_select.hpp cannot drift apart (group-5 5.3).
        struct Ring {
            DeviceBuffer<double> f2;
            DeviceBuffer<double> vpair;
            Event reuse;  // RAII disable-timing event, recorded after this buffer's chunk D2Hs issued
            bool used = false;
        };
        // Event (stream.hpp) is the owning, move-only disable-timing event wrapper:
        // its default ctor creates the cudaEventDisableTiming event on construction of
        // each Ring slot and its warn-not-throw dtor tears every slot down on unwind, so
        // no bespoke create loop or teardown guard is needed (standard §2.12 RAII
        // template, §5 non-goal #9 keep behavior). This also closes the [14.5] partial-
        // construction leak window structurally: a throw mid-array still unwinds the
        // already-constructed slots' events (the DeviceBuffer members free via their own
        // RAII regardless).
        Ring ring[kStreamDeviceChunks];
        for (Ring& r : ring) {
            r.f2 = DeviceBuffer<double>(max_pp_nb);
            r.vpair = DeviceBuffer<double>(max_pp_nb);
        }

        int chunk_idx = 0;
        for (std::size_t bi = 0; bi < buckets.size(); ++bi) {
            const Bucket& bk = buckets[bi];
            const int s_pad = bk.s_pad;
            const int nb_total = static_cast<int>(bk.blocks.size());
            const int max_blocks = bucket_max_blocks[bi];

            for (int start = 0; start < nb_total; ) {
                // The tile-width split valve fixes BOTH nb and the column extent
                // [s_lo,s_hi) of THIS chunk (same split the pre-pass sized for).
                long s_lo = 0, s_hi = 0;
                const int nb = chunk_extent(bk, start, max_blocks, s_lo, s_hi);
                const long tile = s_hi - s_lo;
                Ring& r = ring[chunk_idx % kStreamDeviceChunks];
                // Reuse-after-drain: wait until the prior chunk that used this device
                // buffer has had its D2Hs issued+drained (its event), so chunk c's
                // assemble cannot overwrite slabs chunk c-slots is still D2H-ing.
                if (r.used)
                    STEPPE_CUDA_CHECK(cudaEventSynchronize(r.reuse.get()));

                // ---- PER-CHUNK SNP-TILE DECODE (the new mechanism) -----------------
                // Upload ONLY this chunk's column union Q/V/N[:, s_lo:s_hi] from the
                // host MatView. Column-major [P×M] makes the slice a SINGLE contiguous
                // run of P·tile doubles starting at i + P·s_lo (views.hpp), so it is ONE
                // cudaMemcpyAsync per matrix (the source is a sub-range of the FULL host
                // region already pinned above — no per-tile pin).
                const std::size_t tile_elems =
                    static_cast<std::size_t>(P) * static_cast<std::size_t>(tile);
                const std::size_t tile_bytes = tile_elems * sizeof(double);
                const std::size_t src_off = static_cast<std::size_t>(P) *
                                            static_cast<std::size_t>(s_lo);
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ_raw.data(), Q.data + src_off, tile_bytes,
                                                  cudaMemcpyHostToDevice, stream_.get()));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV_raw.data(), V.data + src_off, tile_bytes,
                                                  cudaMemcpyHostToDevice, stream_.get()));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dN_raw.data(), N.data + src_off, tile_bytes,
                                                  cudaMemcpyHostToDevice, stream_.get()));
                // SAME feeder wrapper, M arg = tile (a long); per-column elementwise so
                // feeding [s_lo,s_hi) in isolation is bit-identical to feeding it inside
                // the all-M sweep (§5.2 — no cross-column dependency).
                launch_f2_feeder(dQ_raw.data(), dV_raw.data(), dN_raw.data(),
                                 dQt.data(), dVt.data(), dSt.data(), P, tile, stream_.get());

                // Rebase this chunk's offsets/sizes into LOCAL ids [0..nb): the gather
                // reads the [P×tile] tile feeder at dOffsetsTile[id] + c = block gid's
                // columns within the tile — exactly the columns the all-M gather read at
                // block_offsets[gid]+c, so the gathered slab is bit-identical.
                for (int kk = 0; kk < nb; ++kk) {
                    const int gid = bk.blocks[static_cast<std::size_t>(start + kk)];
                    h_offsets_tile[static_cast<std::size_t>(kk)] =
                        block_offsets[static_cast<std::size_t>(gid)] - s_lo;
                    h_sizes_tile[static_cast<std::size_t>(kk)] =
                        block_sizes[static_cast<std::size_t>(gid)];
                }
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dOffsetsTile.data(), h_offsets_tile.data(),
                                                  static_cast<std::size_t>(nb) * sizeof(long),
                                                  cudaMemcpyHostToDevice, stream_.get()));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSizesTile.data(), h_sizes_tile.data(),
                                                  static_cast<std::size_t>(nb) * sizeof(int),
                                                  cudaMemcpyHostToDevice, stream_.get()));
                // dIdsLocal is [0..max_nb); its first nb entries [0..nb) are the LOCAL
                // ids the gather indexes into dOffsetsTile/dSizesTile.
                launch_gather_group(dQt.data(), dVt.data(), dSt.data(),
                                    dIdsLocal.data(), dOffsetsTile.data(), dSizesTile.data(),
                                    P, s_pad, nb, dQg.data(), dVg.data(), dSg.data(), stream_.get());
                run_f2_gemms_group(blas_.get(), precision, P, s_pad, nb,
                                   dQg.data(), dVg.data(), dSg.data(),
                                   dGg.data(), dVpairg.data(), dRg.data());
                // Assemble into the chunk-LOCAL slabs 0..nb-1 of this ring buffer (the
                // dIdsLocal array). Same finalize_f2(num, vp) -> same VALUE bits as the
                // resident path; only the destination OFFSET is chunk-local.
                launch_assemble_blocks_group(dGg.data(), dVpairg.data(), dRg.data(),
                                             dIdsLocal.data(), P, nb,
                                             r.f2.data(), r.vpair.data(), stream_.get());

                // Spill each of this chunk's nb blocks under its GLOBAL id. The sink
                // issues the async D2H on stream_ (in issue order, after the assemble) and
                // triple-buffers the D2H->tier-write. NO cudaStreamSynchronize here — the
                // overlap is what hides the spill behind the next chunk's compute.
                for (int kk = 0; kk < nb; ++kk) {
                    const int gid = bk.blocks[static_cast<std::size_t>(start + kk)];
                    const std::size_t off = slab * static_cast<std::size_t>(kk);
                    sink.spill_block(gid, r.f2.data() + off, r.vpair.data() + off,
                                     slab, stream_.get());
                }
                // Record the reuse event AFTER this chunk's spill D2Hs are enqueued, so a
                // later chunk that reuses this device buffer waits for them. Event::record
                // (stream.hpp) wraps cudaEventRecord(e_, stream.get()) + its own CHECK.
                r.reuse.record(stream_);
                r.used = true;
                ++chunk_idx;
                start += (nb > 0 ? nb : 1);  // advance by the split-determined nb
            }
        }

        // Drain the last in-flight slabs + finalize the tier (Disk: trailer+fsync+reopen).
        sink.finish();
    }

    /// Shared decode front-end: upload the packed tile + partition (+ per-sample
    /// ploidy when supplied) and run launch_decode_af into the caller's RESIDENT
    /// dQ/dV/dN [P × M]. NO D2H — the caller decides whether the result crosses the
    /// CUDA-free seam (decode_af: full D2H, host/oracle path) or STAYS resident
    /// (decode_af_compact_autosome: the device-resident seam). The packed/offsets/
    /// ploidy uploads are local RAII (freed on return); dQ/dV/dN are caller-owned.
    void decode_af_resident(const DecodeTileView& tile, int P, long M,
                            DeviceBuffer<double>& dQ, DeviceBuffer<double>& dV,
                            DeviceBuffer<double>& dN) {
        // Only tile-sized and [P×M] buffers — never a [SNP×ind] decode-all
        // (architecture.md §11.1; tile-shaped for the M5 loop).
        const std::size_t packed_bytes =
            tile.n_individuals * tile.bytes_per_record;
        const std::size_t n_off = static_cast<std::size_t>(P) + 1u;
        DeviceBuffer<std::uint8_t> dPacked(packed_bytes);
        DeviceBuffer<std::size_t> dOffsets(n_off);

        // PER-SAMPLE ploidy (AT2 adjust_pseudohaploid): upload the per-sample vector
        // when the caller supplied one (the auto-detect path); when NULL the kernel
        // falls back to the uniform scalar tile.ploidy (the legacy all-diploid path),
        // and we pass a null device pointer (no alloc/copy). A zero-length DeviceBuffer
        // yields a null .data(), so the same handle serves both cases.
        const bool have_sample_ploidy = (tile.sample_ploidy != nullptr);
        DeviceBuffer<int> dSamplePloidy(have_sample_ploidy ? tile.n_individuals : 0u);

        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dPacked.data(), tile.packed,
                                          packed_bytes * sizeof(std::uint8_t),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dOffsets.data(), tile.pop_offsets,
                                          n_off * sizeof(std::size_t),
                                          cudaMemcpyHostToDevice, stream_.get()));
        if (have_sample_ploidy) {
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSamplePloidy.data(), tile.sample_ploidy,
                                              tile.n_individuals * sizeof(int),
                                              cudaMemcpyHostToDevice, stream_.get()));
        }

        // S0 unpack + S1 segmented reduction → Q/V/N (RESIDENT).
        launch_decode_af(dPacked.data(), tile.bytes_per_record, dOffsets.data(),
                         P, M, tile.ploidy,
                         have_sample_ploidy ? dSamplePloidy.data() : nullptr,
                         dQ.data(), dV.data(), dN.data(), stream_.get());
        // dPacked/dOffsets/dSamplePloidy are still in-flight uploads; they free at
        // scope exit. The decode kernel and the uploads are all on stream_, so the
        // ordering is correct and the caller's later stream_ work observes dQ/dV/dN.
    }

    [[nodiscard]] DecodeResult decode_af(const DecodeTileView& tile) override {
        guard_device();
        const int P = tile.n_pop;
        const long M = static_cast<long>(tile.n_snp);

        DecodeResult out;
        out.P = P;
        out.M = M;
        const std::size_t pm =
            static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
        out.q.assign(pm, 0.0);
        out.v.assign(pm, 0.0);
        out.n.assign(pm, 0.0);
        if (P <= 0 || M <= 0) return out;

        // Decode → dQ/dV/dN RESIDENT (the shared front-end), then the C1 full D2H
        // across the CUDA-free seam (this is the host/oracle path; the resident
        // device-resident seam, decode_af_compact_autosome, drops this D2H).
        DeviceBuffer<double> dQ(pm), dV(pm), dN(pm);
        decode_af_resident(tile, P, M, dQ, dV, dN);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.q.data(), dQ.data(), pm * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.v.data(), dV.data(), pm * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.n.data(), dN.data(), pm * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        return out;
    }

    /// DEVICE-RESIDENT decode + AUTOSOME compaction (the host-compute audit C1/C2/
    /// M3/M4 cure for qpfstats / dstat — regime (A)). Decodes the tile to dQ/dV/dN
    /// RESIDENT (NO C1 D2H), runs the on-device per-SNP autosome keep-mask
    /// (INTEGER-EXACT, the C2 host loop GONE), and stream-compacts Q/V onto the kept
    /// axis: CUB DeviceSelect::Flagged compacts the 1-D chrom/genpos (reusing the
    /// sweep's two-call idiom, whose documented "maintain their original relative
    /// ordering" pins FILE ORDER), and a CUB ExclusiveSum + a scan-keyed column
    /// gather compacts the [P×M] Q/V (Flagged is 1-D; the [P×M] tensor uses
    /// scan+gather, cleaner and still device-resident). Only the small kept
    /// chrom/genpos cross to host (for the CUDA-free assign_blocks). The resident
    /// compacted Q/V ESCAPE in the returned DeviceDecodeResult.
    [[nodiscard]] steppe::device::DeviceDecodeResult decode_af_compact_autosome(
        const DecodeTileView& tile, std::span<const int> chrom,
        std::span<const double> genpos, int chrom_min, int chrom_max) override {
        guard_device();
        steppe::device::DeviceDecodeResult out;
        out.device_id = device_id_;
        const int P = tile.n_pop;
        const long M = static_cast<long>(tile.n_snp);
        out.P = P;
        if (P <= 0 || M <= 0) { out.M_kept = 0; return out; }

        const std::size_t pm =
            static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
        const std::size_t Mz = static_cast<std::size_t>(M);

        // ---- 1. Decode → dQ/dV/dN RESIDENT (dN is decoded but the regime-(A) Q/V
        // consumers ignore it; it is NOT compacted/kept). NO host D2H. ----------
        DeviceBuffer<double> dQ(pm), dV(pm), dN(pm);
        decode_af_resident(tile, P, M, dQ, dV, dN);

        // ---- 2. The per-SNP autosome keep-mask (INTEGER-EXACT) → d_flags. -------
        DeviceBuffer<int> dChrom(Mz);
        DeviceBuffer<double> dGenpos(Mz);
        DeviceBuffer<std::uint8_t> dFlags(Mz);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dChrom.data(), chrom.data(), Mz * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dGenpos.data(), genpos.data(), Mz * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        launch_autosome_keep_mask(dChrom.data(), M, chrom_min, chrom_max, dFlags.data(),
                                  stream_.get());

        // ---- 3. The compacted column index = EXCLUSIVE prefix sum of d_flags, and
        // the kept count M_kept = keep_idx[M-1] + flags[M-1] (CUB DeviceScan). -----
        DeviceBuffer<long> dKeepIdx(Mz);
        DeviceBuffer<int> dNumSel(1);
        {
            // ExclusiveSum over the uint8 flags into a long index buffer (the
            // compacted column position). Two-call temp-storage idiom (CUDA 13.x
            // CCCL CUB; d_temp_storage=nullptr query then the sized buffer).
            std::size_t scan_bytes = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, dFlags.data(),
                                          dKeepIdx.data(), static_cast<std::int64_t>(M),
                                          stream_.get());
            DeviceBuffer<unsigned char> dScanTemp(scan_bytes == 0 ? 1 : scan_bytes);
            std::size_t sb = scan_bytes;
            cub::DeviceScan::ExclusiveSum(dScanTemp.data(), sb, dFlags.data(),
                                          dKeepIdx.data(), static_cast<std::int64_t>(M),
                                          stream_.get());
        }

        // ---- 4. Compact the 1-D chrom/genpos with CUB DeviceSelect::Flagged
        // (the SAME idiom run_fstat_sweep_device uses; ordering preserved). The
        // num_selected_out is M_kept — read it back small. ----------------------
        DeviceBuffer<int> dChromKept(Mz);
        DeviceBuffer<double> dGenposKept(Mz);
        {
            // CUB DeviceSelect::Flagged temp size depends on the VALUE TYPE (the
            // internal tile-state carries the data type), so query BOTH the int
            // (chrom) and double (genpos) calls and size dSelTemp to the MAX — the
            // double query is the larger; reusing an int-sized temp for the double
            // Flagged silently undersizes it (the observed all-zero genpos bug).
            std::size_t sel_bytes_i = 0, sel_bytes_d = 0;
            cub::DeviceSelect::Flagged(nullptr, sel_bytes_i, dChrom.data(), dFlags.data(),
                                       dChromKept.data(), dNumSel.data(),
                                       static_cast<std::int64_t>(M), stream_.get());
            cub::DeviceSelect::Flagged(nullptr, sel_bytes_d, dGenpos.data(), dFlags.data(),
                                       dGenposKept.data(), dNumSel.data(),
                                       static_cast<std::int64_t>(M), stream_.get());
            const std::size_t sel_bytes = std::max(sel_bytes_i, sel_bytes_d);
            DeviceBuffer<unsigned char> dSelTemp(sel_bytes == 0 ? 1 : sel_bytes);
            std::size_t sb = sel_bytes;
            cub::DeviceSelect::Flagged(dSelTemp.data(), sb, dChrom.data(), dFlags.data(),
                                       dChromKept.data(), dNumSel.data(),
                                       static_cast<std::int64_t>(M), stream_.get());
            sb = sel_bytes;
            cub::DeviceSelect::Flagged(dSelTemp.data(), sb, dGenpos.data(), dFlags.data(),
                                       dGenposKept.data(), dNumSel.data(),
                                       static_cast<std::int64_t>(M), stream_.get());
        }

        // Read M_kept back (one int) — needed to size the resident Q/V buffers.
        int m_kept = 0;
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(&m_kept, dNumSel.data(), sizeof(int),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        out.M_kept = static_cast<long>(m_kept);
        if (m_kept <= 0) return out;

        // ---- 5. Allocate the RESIDENT compacted Q/V [P × M_kept] and gather the
        // kept columns (scan-keyed; FILE ORDER preserved). The handle ESCAPES. ---
        const std::size_t pmk =
            static_cast<std::size_t>(P) * static_cast<std::size_t>(m_kept);
        out.impl = std::make_unique<steppe::device::DeviceDecodeResult::Impl>();
        out.impl->q = DeviceBuffer<double>(pmk);
        out.impl->v = DeviceBuffer<double>(pmk);
        launch_compact_columns_gather(dQ.data(), P, M, dFlags.data(), dKeepIdx.data(),
                                      out.impl->q.data(), stream_.get());
        launch_compact_columns_gather(dV.data(), P, M, dFlags.data(), dKeepIdx.data(),
                                      out.impl->v.data(), stream_.get());

        // ---- 6. The small kept chrom/genpos D2H (for the CUDA-free assign_blocks). --
        out.chrom_kept.assign(static_cast<std::size_t>(m_kept), 0);
        out.genpos_kept.assign(static_cast<std::size_t>(m_kept), 0.0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.chrom_kept.data(), dChromKept.data(),
                                          static_cast<std::size_t>(m_kept) * sizeof(int),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.genpos_kept.data(), dGenposKept.data(),
                                          static_cast<std::size_t>(m_kept) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        return out;
    }

    /// DEVICE-RESIDENT decode + REGIME-B (extract_f2 FULL filter) compaction (the
    /// host-compute audit regime (B) cure). SIBLING of decode_af_compact_autosome:
    /// it REUSES decode_af_resident verbatim for dQ/dV/dN, REPLACES the autosome-only
    /// keep-mask with the regime-B keep-mask kernel (pooled-MAF Σ_pop Q·N [FFMA-immune]
    /// + the shared keep_decision_pooled + the SEPARATE pop-coverage maxmiss), REUSES
    /// the IDENTICAL CUB ExclusiveSum + Flagged + scan-gather idiom, and adds the THIRD
    /// lockstep gather of N. The resident compacted Q/V/N escape (n_device() non-null);
    /// only the small kept chrom/genpos cross to host (for assign_blocks).
    [[nodiscard]] steppe::device::DeviceDecodeResult decode_af_compact_filter(
        const DecodeTileView& tile, std::span<const char> ref, std::span<const char> alt,
        std::span<const int> chrom, std::span<const double> genpos,
        const FilterConfig& cfg, std::span<const std::size_t> pop_individuals,
        int ploidy, double maxmiss) override {
        guard_device();
        steppe::device::DeviceDecodeResult out;
        out.device_id = device_id_;
        const int P = tile.n_pop;
        const long M = static_cast<long>(tile.n_snp);
        out.P = P;
        if (P <= 0 || M <= 0) { out.M_kept = 0; return out; }

        const std::size_t pm =
            static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
        const std::size_t Mz = static_cast<std::size_t>(M);

        // ---- 1. Decode → dQ/dV/dN RESIDENT (the SHARED front-end; NO host D2H). The
        // regime-B keep-mask reads the UNCOMPACTED dQ/dN; the gather then compacts
        // Q/V/N from the SAME resident dQ/dV/dN — no second decode. ----------------
        DeviceBuffer<double> dQ(pm), dV(pm), dN(pm);
        decode_af_resident(tile, P, M, dQ, dV, dN);

        // ---- 2. The regime-B per-SNP keep-mask (the SHARED reduction + decision +
        // the SEPARATE maxmiss) → d_flags. The sample-axis geno predicate is the
        // no-op (geno_max_missing FORCED to 1.0); the pop-axis maxmiss is separate. -
        FilterConfig kernel_cfg = cfg;
        kernel_cfg.geno_max_missing = 1.0;  // pop-coverage maxmiss is the `maxmiss` arg.

        // total_indiv = Σ_pop pop_individuals (the SNP-independent missing-frac
        // denominator). pop_individuals length == P (the caller's contract); if it is
        // short/empty, fall back to the segment sizes from the tile pop_offsets so the
        // reduction never reads OOB. (Mirrors snp_filter.cpp's total_indiv loop.)
        double total_indiv_d = 0.0;
        if (pop_individuals.size() == static_cast<std::size_t>(P)) {
            std::size_t total = 0;
            for (int p = 0; p < P; ++p) total += pop_individuals[static_cast<std::size_t>(p)];
            total_indiv_d = static_cast<double>(total);
        } else {
            std::size_t total = 0;
            for (int p = 0; p < P; ++p)
                total += tile.pop_offsets[static_cast<std::size_t>(p) + 1] -
                         tile.pop_offsets[static_cast<std::size_t>(p)];
            total_indiv_d = static_cast<double>(total);
        }

        // Upload the per-SNP ref/alt allele chars + chrom (genpos rides only the
        // Flagged companion). dChrom is needed BOTH by the keep-mask (autosomes_only)
        // and by the Flagged compaction, so it is uploaded once.
        DeviceBuffer<char> dRef(Mz), dAlt(Mz);
        DeviceBuffer<int> dChrom(Mz);
        DeviceBuffer<double> dGenpos(Mz);
        DeviceBuffer<std::uint8_t> dFlags(Mz);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dRef.data(), ref.data(), Mz * sizeof(char),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dAlt.data(), alt.data(), Mz * sizeof(char),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dChrom.data(), chrom.data(), Mz * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dGenpos.data(), genpos.data(), Mz * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        launch_regimeb_keep_mask(dQ.data(), dN.data(), P, M, dRef.data(), dAlt.data(),
                                 dChrom.data(), kernel_cfg,
                                 static_cast<double>(ploidy), total_indiv_d, maxmiss,
                                 dFlags.data(), stream_.get());

        // ---- 3. The compacted column index = EXCLUSIVE prefix sum of d_flags (the
        // IDENTICAL CUB DeviceScan idiom as decode_af_compact_autosome). -----------
        DeviceBuffer<long> dKeepIdx(Mz);
        DeviceBuffer<int> dNumSel(1);
        {
            std::size_t scan_bytes = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, dFlags.data(),
                                          dKeepIdx.data(), static_cast<std::int64_t>(M),
                                          stream_.get());
            DeviceBuffer<unsigned char> dScanTemp(scan_bytes == 0 ? 1 : scan_bytes);
            std::size_t sb = scan_bytes;
            cub::DeviceScan::ExclusiveSum(dScanTemp.data(), sb, dFlags.data(),
                                          dKeepIdx.data(), static_cast<std::int64_t>(M),
                                          stream_.get());
        }

        // ---- 4. Compact the 1-D chrom/genpos with CUB DeviceSelect::Flagged (the
        // IDENTICAL idiom + the int/double MAX-temp-size guard). -------------------
        DeviceBuffer<int> dChromKept(Mz);
        DeviceBuffer<double> dGenposKept(Mz);
        {
            std::size_t sel_bytes_i = 0, sel_bytes_d = 0;
            cub::DeviceSelect::Flagged(nullptr, sel_bytes_i, dChrom.data(), dFlags.data(),
                                       dChromKept.data(), dNumSel.data(),
                                       static_cast<std::int64_t>(M), stream_.get());
            cub::DeviceSelect::Flagged(nullptr, sel_bytes_d, dGenpos.data(), dFlags.data(),
                                       dGenposKept.data(), dNumSel.data(),
                                       static_cast<std::int64_t>(M), stream_.get());
            const std::size_t sel_bytes = std::max(sel_bytes_i, sel_bytes_d);
            DeviceBuffer<unsigned char> dSelTemp(sel_bytes == 0 ? 1 : sel_bytes);
            std::size_t sb = sel_bytes;
            cub::DeviceSelect::Flagged(dSelTemp.data(), sb, dChrom.data(), dFlags.data(),
                                       dChromKept.data(), dNumSel.data(),
                                       static_cast<std::int64_t>(M), stream_.get());
            sb = sel_bytes;
            cub::DeviceSelect::Flagged(dSelTemp.data(), sb, dGenpos.data(), dFlags.data(),
                                       dGenposKept.data(), dNumSel.data(),
                                       static_cast<std::int64_t>(M), stream_.get());
        }

        int m_kept = 0;
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(&m_kept, dNumSel.data(), sizeof(int),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        out.M_kept = static_cast<long>(m_kept);
        if (m_kept <= 0) return out;

        // ---- 5. The RESIDENT compacted Q/V/N [P × M_kept], the THREE lockstep
        // scan-keyed gathers (Q, V, AND N — N is the regime-B addition; the SAME
        // dFlags + dKeepIdx, so N is compacted in EXACT lockstep with Q/V, FILE ORDER
        // preserved by the monotone exclusive scan). ------------------------------
        const std::size_t pmk =
            static_cast<std::size_t>(P) * static_cast<std::size_t>(m_kept);
        out.impl = std::make_unique<steppe::device::DeviceDecodeResult::Impl>();
        out.impl->q = DeviceBuffer<double>(pmk);
        out.impl->v = DeviceBuffer<double>(pmk);
        out.impl->n = DeviceBuffer<double>(pmk);
        launch_compact_columns_gather(dQ.data(), P, M, dFlags.data(), dKeepIdx.data(),
                                      out.impl->q.data(), stream_.get());
        launch_compact_columns_gather(dV.data(), P, M, dFlags.data(), dKeepIdx.data(),
                                      out.impl->v.data(), stream_.get());
        launch_compact_columns_gather(dN.data(), P, M, dFlags.data(), dKeepIdx.data(),
                                      out.impl->n.data(), stream_.get());

        // ---- 6. The small kept chrom/genpos D2H (for the CUDA-free assign_blocks). --
        out.chrom_kept.assign(static_cast<std::size_t>(m_kept), 0);
        out.genpos_kept.assign(static_cast<std::size_t>(m_kept), 0.0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.chrom_kept.data(), dChromKept.data(),
                                          static_cast<std::size_t>(m_kept) * sizeof(int),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.genpos_kept.data(), dGenposKept.data(),
                                          static_cast<std::size_t>(m_kept) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        return out;
    }

    /// qpDstat Part B — the genotype-path NORMALIZED-D per-SNP reduction on the GPU (the
    /// S2 divergence; backend.hpp dstat_block_reduce / include/steppe/dstat.hpp). Uploads
    /// the resident Q/V [P × M] + the [4 × N] quadruple index table + the per-block
    /// begin/size layout (from core::block_ranges, the X-3/B3 single-source inverse of
    /// assign_blocks), runs the SNP-tile-batched kernel (one thread per (quadruple, block)
    /// cell), and copies the tiny [N × n_block] numsum/densum/cnt back. Device-resident
    /// (the deliverable): Q/V live in VRAM; the output is row-major like F4Blocks::x_blocks.
    /// The dstat_block_reduce CORE over ALREADY-RESIDENT device Q/V pointers (the
    /// single-source body both the host-pointer and the DeviceDecodeResult overloads
    /// share). Builds the block begin/size layout from block_id (H2D small), runs
    /// the kernel, D2Hs the tiny numsum/densum/cnt. dQ/dV are borrowed device
    /// pointers (NOT freed here): the host overload owns DeviceBuffers it uploaded;
    /// the device overload passes the resident DeviceDecodeResult Q/V — NO Q/V H2D.
    void dstat_block_reduce_device(const double* dQ, const double* dV, int P, long M,
                                   const int* block_id, int n_block,
                                   std::span<const int> quadruples,
                                   double* numsum, double* densum, double* cnt) {
        const int N = static_cast<int>(quadruples.size() / 4);
        if (P <= 0 || M <= 0 || N <= 0 || n_block <= 0) return;

        // Per-block contiguous SNP layout from block_id (the SINGLE-SOURCE inverse of
        // assign_blocks; same primitive the f2 path uses). begin/size as int (M fits int
        // by the same B22 guard the f2 path enforces upstream).
        const std::vector<core::BlockRange> ranges =
            core::block_ranges(std::span<const int>(block_id, static_cast<std::size_t>(M)),
                               M, n_block);
        std::vector<int> begin(static_cast<std::size_t>(n_block));
        std::vector<int> size(static_cast<std::size_t>(n_block));
        for (int b = 0; b < n_block; ++b) {
            begin[static_cast<std::size_t>(b)] = static_cast<int>(ranges[static_cast<std::size_t>(b)].begin);
            size[static_cast<std::size_t>(b)]  = static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
        }

        const std::size_t nq = static_cast<std::size_t>(N) * 4u;
        const std::size_t nb_out = static_cast<std::size_t>(N) * static_cast<std::size_t>(n_block);

        DeviceBuffer<int> dQuad(nq);
        DeviceBuffer<int> dBegin(static_cast<std::size_t>(n_block));
        DeviceBuffer<int> dSize(static_cast<std::size_t>(n_block));
        DeviceBuffer<double> dNum(nb_out), dDen(nb_out), dCnt(nb_out);

        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQuad.data(), quadruples.data(), nq * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBegin.data(), begin.data(),
                                          static_cast<std::size_t>(n_block) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSize.data(), size.data(),
                                          static_cast<std::size_t>(n_block) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));

        launch_dstat_block_reduce(dQ, dV, P, M, dQuad.data(), N,
                                  dBegin.data(), dSize.data(), n_block,
                                  dNum.data(), dDen.data(), dCnt.data(), stream_.get());

        STEPPE_CUDA_CHECK(cudaMemcpyAsync(numsum, dNum.data(), nb_out * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(densum, dDen.data(), nb_out * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(cnt, dCnt.data(), nb_out * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    }

    void dstat_block_reduce(const double* Q, const double* V, int P, long M,
                            const int* block_id, int n_block,
                            std::span<const int> quadruples,
                            double* numsum, double* densum, double* cnt) override {
        guard_device();
        const int N = static_cast<int>(quadruples.size() / 4);
        if (P <= 0 || M <= 0 || N <= 0 || n_block <= 0) return;
        // Host-pointer path (the CpuBackend-parity entry / non-resident callers): H2D
        // Q/V into resident buffers, then the shared core. The DeviceDecodeResult
        // overload below SKIPS this H2D (the M4 cure).
        const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
        DeviceBuffer<double> dQ(pm), dV(pm);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ.data(), Q, pm * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV.data(), V, pm * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        dstat_block_reduce_device(dQ.data(), dV.data(), P, M, block_id, n_block,
                                  quadruples, numsum, densum, cnt);
    }

    /// DEVICE-RESIDENT dstat_block_reduce — reads Q/V already RESIDENT in VRAM from
    /// a DeviceDecodeResult (the M4 cure: NO Q/V H2D). Math byte-identical to the
    /// host-pointer overload (the SAME launch_dstat_block_reduce over the SAME Q/V
    /// the decode wrote, compacted onto the autosome axis).
    void dstat_block_reduce(const steppe::device::DeviceDecodeResult& dec,
                            const int* block_id, int n_block,
                            std::span<const int> quadruples,
                            double* numsum, double* densum, double* cnt) override {
        guard_device();
        if (dec.empty() || dec.q_device() == nullptr) return;
        dstat_block_reduce_device(dec.q_device(), dec.v_device(), dec.P, dec.M_kept,
                                  block_id, n_block, quadruples, numsum, densum, cnt);
    }

    /// DATES weighted-LD CURVE on the GPU — the cuFFT autocorrelation LD engine (backend.hpp
    /// dates_curve / include/steppe/dates.hpp). The whole per-admixed-sample pipeline, GPU-bound
    /// and FLAT in M: the ~10^12 SNP-pair object is NEVER formed (cov(lag) = IFFT(|FFT(grid)|²),
    /// the ALDER FFT trick). Device-resident inputs; the host drives only the per-sample loop
    /// (n_target ~ 100) + the ONE-TIME plan setup. PER sample: scatter onto the fine per-chrom
    /// grid (weight/residual kernels), then batched cufftExecD2Z over all chroms -> |F|²/cross
    /// power kernels -> batched cufftExecZ2D inverse -> extract+accumulate lags into the per-chrom
    /// dd moments (summed across samples). Then re-bin lag->output-bin into the per-(chrom,bin)
    /// CORR sufficient statistics. Native double FFT (cuFFT UNNORMALIZED -> explicit /n_fft,
    /// IDENTICAL to the FFTW reference). The weight/residual is the §12 cancellation carve-out.
    [[nodiscard]] DatesMoments dates_curve(
        const double* src1_freq, const double* src2_freq, const double* src_valid,
        const std::uint8_t* packed, std::size_t bytes_per_record, int n_target,
        const int* target_ploidy, const int* grid_cell, long M,
        const int* chrom_first, const int* chrom_last, int n_chrom,
        int numqbins, int n_bin, int diffmax, double binsize, int qbin,
        const Precision& precision) override {
        (void)target_ploidy;  // DATES dosage = code/2 (ploidy-independent); see dates.c getgtypes.
        (void)binsize;        // the output-bin width is folded into the lag->bin re-bin (floor(d/qbin)).
        (void)precision;      // native double FFT + native weight carve-out (§12).
        guard_device();
        DatesMoments out;
        out.n_chrom = n_chrom;
        out.n_bin = n_bin;
        if (n_chrom <= 0 || n_bin <= 0 || M <= 0 || n_target <= 0 || numqbins <= 0 ||
            diffmax <= 0) {
            out.status = Status::Ok;
            return out;
        }

        // ---- FFT length: pad ALL chroms to a COMMON power-of-2 n_fft >= 2·max_len so ONE
        // cufftPlanMany batches every chrom. The linear autocorrelation Σ_g W[g]·W[g+lag]
        // (lag<len) is INDEPENDENT of n_fft once n_fft >= 2·len (zero-pad avoids circular
        // wrap), and the /n_fft scale matches the FFTW reference — DATES uses a per-chrom
        // 2^(ceil(log2(len))+1); a common larger n is bit-equivalent for the autocorr values.
        int max_len = 1;
        for (int kc = 0; kc < n_chrom; ++kc) {
            const int slo = chrom_first[kc], shi = chrom_last[kc];
            if (slo >= 0 && shi >= slo) max_len = std::max(max_len, shi - slo + 1);
        }
        // ensure n_fft covers the linear autocorr support AND the requested lags.
        long need = std::max<long>(2L * max_len, static_cast<long>(diffmax) + 1L);
        int n_fft = 1;
        while (static_cast<long>(n_fft) < need) n_fft <<= 1;
        const int n_cplx = n_fft / 2 + 1;

        // ---- cuFFT plans (created ONCE, reused across samples): batched D2Z + Z2D over n_chrom.
        auto cufft_ok = [](cufftResult r, const char* what) {
            if (r != CUFFT_SUCCESS)
                throw std::runtime_error(std::string("cuFFT error in ") + what + ": code " +
                                         std::to_string(static_cast<int>(r)));
        };
        cufftHandle plan_fwd = 0, plan_inv = 0;
        int n_dim = n_fft;
        cufft_ok(cufftPlanMany(&plan_fwd, 1, &n_dim, nullptr, 1, n_fft, nullptr, 1, n_cplx,
                               CUFFT_D2Z, n_chrom), "cufftPlanMany(D2Z)");
        cufft_ok(cufftPlanMany(&plan_inv, 1, &n_dim, nullptr, 1, n_cplx, nullptr, 1, n_fft,
                               CUFFT_Z2D, n_chrom), "cufftPlanMany(Z2D)");
        cufft_ok(cufftSetStream(plan_fwd, stream_.get()), "cufftSetStream(fwd)");
        cufft_ok(cufftSetStream(plan_inv, stream_.get()), "cufftSetStream(inv)");

        // ---- device-resident inputs + scratch ----
        const std::size_t Mu = static_cast<std::size_t>(M);
        const std::size_t nq = static_cast<std::size_t>(numqbins);
        const std::size_t pad = static_cast<std::size_t>(n_chrom) * static_cast<std::size_t>(n_fft);
        const std::size_t cpx = static_cast<std::size_t>(n_chrom) * static_cast<std::size_t>(n_cplx);
        const std::size_t dm = static_cast<std::size_t>(n_chrom) *
                               (static_cast<std::size_t>(diffmax) + 1);
        const std::size_t cb = static_cast<std::size_t>(n_chrom) * static_cast<std::size_t>(n_bin);
        const std::size_t pk = static_cast<std::size_t>(n_target) * bytes_per_record;

        DeviceBuffer<double> dS1(Mu), dS2(Mu), dValid(Mu);
        DeviceBuffer<std::uint8_t> dPacked(pk);
        DeviceBuffer<int> dCell(Mu), dCfirst(static_cast<std::size_t>(n_chrom)),
            dClast(static_cast<std::size_t>(n_chrom));
        DeviceBuffer<double> dZ0(nq), dZ1(nq), dZ2(nq);
        DeviceBuffer<double> dPad0(pad), dPad1(pad), dPad2(pad);
        DeviceBuffer<double> dInv(pad);
        // complex scratch (cufftDoubleComplex == double2): forward outs for z0,z1,z2 + power.
        DeviceBuffer<double2> dF0(cpx), dF1(cpx), dF2(cpx), dPow(cpx);
        // per-chrom dd lag moments (summed across samples): dd00,dd11,dd02,dd20.
        DeviceBuffer<double> dDd00(dm), dDd11(dm), dDd02(dm), dDd20(dm);
        // output corr sufficient stats.
        DeviceBuffer<double> dS0(cb), dS11(cb), dS12(cb), dS22(cb);
        DeviceBuffer<double> dDot12(1), dDot22(1);

        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dS1.data(), src1_freq, Mu * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dS2.data(), src2_freq, Mu * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dValid.data(), src_valid, Mu * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dPacked.data(), packed, pk,
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dCell.data(), grid_cell, Mu * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dCfirst.data(), chrom_first,
                                          static_cast<std::size_t>(n_chrom) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dClast.data(), chrom_last,
                                          static_cast<std::size_t>(n_chrom) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dDd00.data(), 0, dm * sizeof(double), stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dDd11.data(), 0, dm * sizeof(double), stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dDd02.data(), 0, dm * sizeof(double), stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dDd20.data(), 0, dm * sizeof(double), stream_.get()));

        auto fft_fwd = [&](double* in, double2* fr) {
            cufft_ok(cufftExecD2Z(plan_fwd, in, reinterpret_cast<cufftDoubleComplex*>(fr)),
                     "cufftExecD2Z");
        };
        auto fft_inv = [&](double2* fr, double* outr) {
            cufft_ok(cufftExecZ2D(plan_inv, reinterpret_cast<cufftDoubleComplex*>(fr), outr),
                     "cufftExecZ2D");
        };

        // ---- per-sample loop (host-driven; the FFT/scatter are GPU-bound) -----------------
        for (int i = 0; i < n_target; ++i) {
            STEPPE_CUDA_CHECK(cudaMemsetAsync(dDot12.data(), 0, sizeof(double), stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemsetAsync(dDot22.data(), 0, sizeof(double), stream_.get()));
            launch_dates_regress_dots(dS1.data(), dS2.data(), dValid.data(), dPacked.data(),
                                      bytes_per_record, i, M, dDot12.data(), dDot22.data(),
                                      stream_.get());
            double h_dot12 = 0.0, h_dot22 = 0.0;
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(&h_dot12, dDot12.data(), sizeof(double),
                                              cudaMemcpyDeviceToHost, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(&h_dot22, dDot22.data(), sizeof(double),
                                              cudaMemcpyDeviceToHost, stream_.get()));
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
            const double yreg = (h_dot22 != 0.0) ? (h_dot12 / h_dot22) : 0.0;

            STEPPE_CUDA_CHECK(cudaMemsetAsync(dZ0.data(), 0, nq * sizeof(double), stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemsetAsync(dZ1.data(), 0, nq * sizeof(double), stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemsetAsync(dZ2.data(), 0, nq * sizeof(double), stream_.get()));
            launch_dates_scatter(dS1.data(), dS2.data(), dValid.data(), dPacked.data(),
                                 bytes_per_record, i, M, dCell.data(), yreg, dZ0.data(),
                                 dZ1.data(), dZ2.data(), stream_.get());

            // pack chrom segments -> FFT rows (z0,z1,z2).
            launch_dates_pack_segments(dZ0.data(), dCfirst.data(), dClast.data(), n_chrom,
                                       numqbins, n_fft, dPad0.data(), stream_.get());
            launch_dates_pack_segments(dZ1.data(), dCfirst.data(), dClast.data(), n_chrom,
                                       numqbins, n_fft, dPad1.data(), stream_.get());
            launch_dates_pack_segments(dZ2.data(), dCfirst.data(), dClast.data(), n_chrom,
                                       numqbins, n_fft, dPad2.data(), stream_.get());

            // forward transforms.
            fft_fwd(dPad0.data(), dF0.data());
            fft_fwd(dPad1.data(), dF1.data());
            fft_fwd(dPad2.data(), dF2.data());

            // dd00 = autocorr(z0) = IFFT(|F0|²)/n.
            launch_dates_power_spectrum(dF0.data(), n_cplx, n_chrom, dPow.data(), stream_.get());
            fft_inv(dPow.data(), dInv.data());
            launch_dates_extract_lags(dInv.data(), n_fft, n_chrom, diffmax, dDd00.data(),
                                      stream_.get());
            // dd11 = autocorr(z1).
            launch_dates_power_spectrum(dF1.data(), n_cplx, n_chrom, dPow.data(), stream_.get());
            fft_inv(dPow.data(), dInv.data());
            launch_dates_extract_lags(dInv.data(), n_fft, n_chrom, diffmax, dDd11.data(),
                                      stream_.get());
            // dd02 = crosscorr(z0,z2) = IFFT(conj(F0)·F2)/n.
            launch_dates_cross_power(dF0.data(), dF2.data(), n_cplx, n_chrom, dPow.data(),
                                     stream_.get());
            fft_inv(dPow.data(), dInv.data());
            launch_dates_extract_lags(dInv.data(), n_fft, n_chrom, diffmax, dDd02.data(),
                                      stream_.get());
            // dd20 = crosscorr(z2,z0) = IFFT(conj(F2)·F0)/n.
            launch_dates_cross_power(dF2.data(), dF0.data(), n_cplx, n_chrom, dPow.data(),
                                     stream_.get());
            fft_inv(dPow.data(), dInv.data());
            launch_dates_extract_lags(dInv.data(), n_fft, n_chrom, diffmax, dDd20.data(),
                                      stream_.get());
        }

        // ---- re-bin lag -> output bin into the corr sufficient stats ----------------------
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dS0.data(), 0, cb * sizeof(double), stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dS11.data(), 0, cb * sizeof(double), stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dS12.data(), 0, cb * sizeof(double), stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dS22.data(), 0, cb * sizeof(double), stream_.get()));
        launch_dates_accumulate_bins(dDd00.data(), dDd11.data(), dDd02.data(), dDd20.data(),
                                     n_chrom, diffmax, n_bin, qbin, dS0.data(), dS11.data(),
                                     dS12.data(), dS22.data(), stream_.get());

        out.s0.assign(cb, 0.0);  out.s1.assign(cb, 0.0);  out.s2.assign(cb, 0.0);
        out.s11.assign(cb, 0.0); out.s12.assign(cb, 0.0); out.s22.assign(cb, 0.0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.s0.data(), dS0.data(), cb * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.s11.data(), dS11.data(), cb * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.s12.data(), dS12.data(), cb * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.s22.data(), dS22.data(), cb * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

        cufftDestroy(plan_fwd);
        cufftDestroy(plan_inv);
        out.status = Status::Ok;
        return out;
    }

    /// M5 — DATES target-genotype REPACK onto the kept SNP axis ON THE DEVICE (host-compute
    /// audit; backend.hpp dates_repack). Replaces the dates.cpp host bit-shuffle hot loop
    /// (O(n_target × M_kept)): upload the n_target FULL target records + the kept-index map
    /// ONCE, run the device gather (one thread per dest byte, race-free; reads via the SAME
    /// MSB-first core::genotype_code, writes the SAME shift/OR as the host), D2H only the
    /// dense repacked buffer. INTEGER/BIT-EXACT ⇒ bit-identical to the CpuBackend repack ⇒
    /// dates_curve moments unchanged ⇒ the DATES date golden held exactly.
    void dates_repack(const std::uint8_t* src, std::size_t src_bpr, const long* kept_src,
                      long M_kept, int n_target, std::size_t dst_bpr,
                      std::uint8_t* dst) override {
        guard_device();
        if (n_target <= 0 || M_kept <= 0 || dst_bpr == 0) return;
        const std::size_t src_bytes = static_cast<std::size_t>(n_target) * src_bpr;
        const std::size_t dst_bytes = static_cast<std::size_t>(n_target) * dst_bpr;
        DeviceBuffer<std::uint8_t> dSrc(src_bytes);
        DeviceBuffer<long> dKept(static_cast<std::size_t>(M_kept));
        DeviceBuffer<std::uint8_t> dDst(dst_bytes);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSrc.data(), src, src_bytes,
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dKept.data(), kept_src,
                                          static_cast<std::size_t>(M_kept) * sizeof(long),
                                          cudaMemcpyHostToDevice, stream_.get()));
        launch_dates_repack_target(dSrc.data(), src_bpr, dKept.data(), M_kept, n_target,
                                   dst_bpr, dDst.data(), stream_.get());
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dst, dDst.data(), dst_bytes,
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    }

    /// M6 — DATES exponential-decay FIT ON THE DEVICE, batched over the n_curves windowed
    /// corr curves (host-compute audit; backend.hpp dates_fit). One thread per curve runs
    /// the EXACT DATES coarse-to-fine 1-D search (4000-point v-grid + 200-iter ternary
    /// refine + the inner 2×2 FP64 normal-equation solve) — the host no longer runs the
    /// 4000×win_len×(n_chrom+1) fit arithmetic. The inner normal-eq accumulators are the
    /// NATIVE FP64 cancellation carve-out (the device has no long double; the DATES date
    /// golden is the loose 2% tier, held by the FP64 fit). Upload curves once, D2H the
    /// n_curves (date,sd,ok).
    [[nodiscard]] std::vector<DatesExpFit> dates_fit(const double* curves, int win_len,
                                                     int n_curves, double step,
                                                     bool affine) override {
        guard_device();
        std::vector<DatesExpFit> out(static_cast<std::size_t>(n_curves > 0 ? n_curves : 0));
        if (n_curves <= 0 || win_len <= 0 || !(step > 0.0)) return out;
        const std::size_t nc = static_cast<std::size_t>(n_curves);
        const std::size_t total = nc * static_cast<std::size_t>(win_len);
        DeviceBuffer<double> dCurves(total);
        DeviceBuffer<double> dDate(nc), dSd(nc);
        DeviceBuffer<int> dOk(nc);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dCurves.data(), curves, total * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        launch_dates_fit_curves(dCurves.data(), win_len, n_curves, step, affine,
                                dDate.data(), dSd.data(), dOk.data(), stream_.get());
        std::vector<double> h_date(nc, 0.0), h_sd(nc, 0.0);
        std::vector<int> h_ok(nc, 0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_date.data(), dDate.data(), nc * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_sd.data(), dSd.data(), nc * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_ok.data(), dOk.data(), nc * sizeof(int),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        for (std::size_t c = 0; c < nc; ++c) {
            out[c].date_gen = h_date[c];
            out[c].error_sd = h_sd[c];
            out[c].ok = h_ok[c];
        }
        return out;
    }

    /// qpfstats SMOOTHING SOLVE on the GPU (the genotype-path joint f2 smoother; backend.hpp
    /// qpfstats_smooth / include/steppe/qpfstats.hpp). The shared-factor batched
    /// least-squares — AT2 qpfstats_regression REFORMULATED with NO host per-block loop
    /// (the CPU-bound trap the R io.R CHUNK_SIZE=64 block loop is). The GPU-bound shape:
    ///   A_shared = x'x + ridge·I   (ONE cublasDsyrk, EmulatedFp64 via the f2 policy)
    ///   RHS = x' · [ymat_zeroed | y_zeroed]  (ONE cublasDgemm, [npairs × (n_block+1)])
    ///   L = chol(A_shared)         (ONE cusolverDnDpotrf LOWER, native FP64 carve-out)
    ///   solve over ALL (n_block+1) columns: a cublasDtrsm PAIR (forward L then back Lᵀ).
    /// The NaN handling: zeroing the ymat NaN entries (launch_qpfstats_zero_nan_ymat) makes
    /// an ALL-NaN block's RHS column 0 ⇒ the shared solve yields b=0 (the AT2 all-NaN→b=0
    /// policy, EXACT, via A·b=0). A PARTIAL-NaN block (0 < k_i < npopcomb; ABSENT on the
    /// real 9-pop golden — only block 536 is all-NaN) needs the per-row downdate
    /// A_b = A_shared - x[nan]'x[nan]; those few blocks (if any) are re-solved with a
    /// downdated A via the host small_linalg solve over ONLY the partial-NaN block set (NOT
    /// a per-block loop over all blocks — zero host solves when no block is partial-NaN, the
    /// production case). Native FP64 carve-out for the Cholesky + the Dtrsm + the downdate.
    [[nodiscard]] QpfstatsSmooth qpfstats_smooth(std::span<const double> x,
                                                 std::span<const double> ymat,
                                                 std::span<const double> y,
                                                 int npopcomb, int npairs,
                                                 int n_block, double ridge,
                                                 const Precision& precision) override {
        guard_device();
        QpfstatsSmooth out;
        out.npairs = npairs;
        out.n_block = n_block;
        if (npopcomb <= 0 || npairs <= 0 || n_block <= 0) { out.status = Status::Ok; return out; }

        const std::size_t nc = static_cast<std::size_t>(npopcomb);
        const std::size_t np = static_cast<std::size_t>(npairs);
        const std::size_t nb = static_cast<std::size_t>(n_block);
        const int ncols = n_block + 1;  // the n_block ymat columns + the bglob (y) column
        const std::size_t ncols_sz = static_cast<std::size_t>(ncols);

        // ---- Upload x [npopcomb × npairs] + the RHS source [npopcomb × ncols] -----------
        // RHS source columns 0..n_block-1 = ymat; column n_block = y. NaN entries are zeroed
        // on device (launch_qpfstats_zero_nan_ymat); the per-column NaN count rides back so
        // the host can detect any PARTIAL-NaN column for the downdate fallback.
        DeviceBuffer<double> dX(nc * np);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dX.data(), x.data(), nc * np * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        DeviceBuffer<double> dRhsSrc(nc * ncols_sz);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dRhsSrc.data(), ymat.data(), nc * nb * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dRhsSrc.data() + nc * nb, y.data(),
                                          nc * sizeof(double), cudaMemcpyHostToDevice,
                                          stream_.get()));
        DeviceBuffer<int> dNanPerCol(ncols_sz);
        launch_qpfstats_zero_nan_ymat(dRhsSrc.data(), npopcomb, ncols, dNanPerCol.data(),
                                      stream_.get());
        std::vector<int> nan_per_col(ncols_sz, 0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(nan_per_col.data(), dNanPerCol.data(),
                                          ncols_sz * sizeof(int), cudaMemcpyDeviceToHost,
                                          stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

        // ---- A_shared = x'x + ridge·I  (cublasDsyrk, EmulatedFp64 via the f2 policy) -----
        // x is [npopcomb × npairs] COLUMN-MAJOR ⇒ x'x is the OP_T·OP_N gram = syrk(OP_T,
        // n=npairs, k=npopcomb, A=dX lda=npopcomb). The well-conditioned gram ENGAGES
        // `precision` exactly like the f2/jackknife SYRK (engage_f2_precision +
        // MathModeScope; the §12 scoped emulated-FP64 pattern). Then symmetrize + add ridge.
        DeviceBuffer<double> dA(np * np);
        {
            const double alpha = 1.0, beta = 0.0;
            const MathModeScope syrk_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
            engage_f2_precision(blas_.get(), precision);
            CUBLAS_CHECK(cublasDsyrk(blas_.get(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                     npairs, npopcomb, &alpha, dX.data(), npopcomb,
                                     &beta, dA.data(), npairs));
        }  // restores the math mode before the native Cholesky/Dtrsm
        launch_symmetrize_lower_to_full(dA.data(), npairs, stream_.get());
        launch_qpfstats_add_ridge_diag(dA.data(), npairs, ridge, stream_.get());

        // ---- RHS = x' · RhsSrc  (cublasDgemm, [npairs × ncols], EmulatedFp64) -----------
        // x' [npairs × npopcomb] · RhsSrc [npopcomb × ncols] = RHS [npairs × ncols].
        DeviceBuffer<double> dRhs(np * ncols_sz);
        {
            const double alpha = 1.0, beta = 0.0;
            const MathModeScope gemm_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
            engage_f2_precision(blas_.get(), precision);
            CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                                     npairs, ncols, npopcomb, &alpha,
                                     dX.data(), npopcomb, dRhsSrc.data(), npopcomb,
                                     &beta, dRhs.data(), npairs));
        }

        // ---- L = chol(A_shared)  (cusolverDnDpotrf LOWER; native FP64 carve-out) ---------
        // A_shared = L·Lᵀ. The ill-conditioned solve DEFAULTS native (solve_precision_=Fp64);
        // promotable via set_solve_precision under per-stage S8 validation (the f2-cache fit
        // policy). The scope is local so it never leaks the mode to the next op.
        const CusolverMathModeScope solve_scope =
            engage_solver_precision(solver_.get(), solve_precision_, &emulation_honorable);
        DeviceBuffer<int> dInfo(1);
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver_.get(), CUBLAS_FILL_MODE_LOWER,
                                                   npairs, dA.data(), npairs, &lwork));
        DeviceBuffer<double> dWork(static_cast<std::size_t>(lwork > 0 ? lwork : 1));
        CUSOLVER_CHECK(cusolverDnDpotrf(solver_.get(), CUBLAS_FILL_MODE_LOWER, npairs,
                                        dA.data(), npairs, dWork.data(), lwork, dInfo.data()));
        int info = 0;
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(&info, dInfo.data(), sizeof(int),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        if (info != 0) { out.status = Status::NonSpdCovariance; return out; }

        // ---- The multi-column solve A·B = RHS via a cublasDtrsm PAIR (NOT potrsBatched,
        // which is nrhs=1-only): A = L·Lᵀ ⇒ forward L·Z = RHS (lower, no-trans) then back
        // Lᵀ·B = Z (lower, trans). ALL ncols columns in ONE Dtrsm each — the whole n_block
        // axis in two calls, NO host per-block loop. dRhs is overwritten with the solution B.
        {
            const double one = 1.0;
            CUBLAS_CHECK(cublasDtrsm(blas_.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, npairs, ncols, &one,
                                     dA.data(), npairs, dRhs.data(), npairs));
            CUBLAS_CHECK(cublasDtrsm(blas_.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, npairs, ncols, &one,
                                     dA.data(), npairs, dRhs.data(), npairs));
        }

        // ---- D2H the solution: b = columns 0..n_block-1, bglob = column n_block ----------
        std::vector<double> sol(np * ncols_sz, 0.0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(sol.data(), dRhs.data(), np * ncols_sz * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        out.b.assign(np * nb, 0.0);
        std::copy(sol.begin(), sol.begin() + static_cast<std::ptrdiff_t>(np * nb), out.b.begin());
        out.bglob.assign(np, 0.0);
        std::copy(sol.begin() + static_cast<std::ptrdiff_t>(np * nb), sol.end(),
                  out.bglob.begin());

        // ---- PARTIAL-NaN downdate fallback (0 < k_col < npopcomb): re-solve ONLY those
        // columns with A_b = A_shared - x[nan]'x[nan]. ABSENT on the real golden (block 536
        // is all-NaN, handled by the zero-RHS shared solve above) ⇒ this loop runs ZERO
        // iterations in production. When present it is a host small_linalg solve over the few
        // partial-NaN columns (NOT all blocks), the AT2 generic-solve branch — bit-faithful.
        bool any_partial = false;
        for (int col = 0; col < ncols; ++col)
            if (nan_per_col[static_cast<std::size_t>(col)] > 0 &&
                nan_per_col[static_cast<std::size_t>(col)] < npopcomb) { any_partial = true; break; }
        if (any_partial) {
            // A_shared (host): x'x + ridge·I (recompute on host; tiny npairs×npairs).
            const auto xat = [&](int c, int p) -> double {
                return x[static_cast<std::size_t>(c) + nc * static_cast<std::size_t>(p)];
            };
            std::vector<double> Abase(np * np, 0.0);
            for (int i = 0; i < npairs; ++i)
                for (int j = 0; j < npairs; ++j) {
                    long double s = 0.0L;
                    for (int c = 0; c < npopcomb; ++c)
                        s += static_cast<long double>(xat(c, i)) * static_cast<long double>(xat(c, j));
                    double v = static_cast<double>(s);
                    if (i == j) v += ridge;
                    Abase[static_cast<std::size_t>(i) + np * static_cast<std::size_t>(j)] = v;
                }
            std::vector<double> rhs(np), bcol, Adown;
            for (int col = 0; col < ncols; ++col) {
                const int kc = nan_per_col[static_cast<std::size_t>(col)];
                if (kc <= 0 || kc >= npopcomb) continue;  // skip no-NaN + all-NaN (already correct)
                // rhs = x' · (this column's source with NaN→0). The source is dRhsSrc col
                // (already zeroed on device) — recompute from the host inputs (col<n_block:
                // ymat; col==n_block: y), zeroing NaN, to match exactly.
                std::fill(rhs.begin(), rhs.end(), 0.0);
                Adown = Abase;
                for (int c = 0; c < npopcomb; ++c) {
                    const double sv = (col < n_block)
                        ? ymat[nc * static_cast<std::size_t>(col) + static_cast<std::size_t>(c)]
                        : y[static_cast<std::size_t>(c)];
                    if (!std::isfinite(sv)) {
                        // downdate A by this NaN row's outer product.
                        for (int i = 0; i < npairs; ++i)
                            for (int j = 0; j < npairs; ++j)
                                Adown[static_cast<std::size_t>(i) + np * static_cast<std::size_t>(j)] -=
                                    xat(c, i) * xat(c, j);
                        continue;
                    }
                    for (int p = 0; p < npairs; ++p) rhs[static_cast<std::size_t>(p)] += xat(c, p) * sv;
                }
                const core::LinAlgStatus st = core::solve(Adown, npairs, rhs, bcol);
                if (!st.ok) { out.status = Status::NonSpdCovariance; return out; }
                if (col < n_block)
                    std::copy(bcol.begin(), bcol.end(),
                              out.b.begin() + static_cast<std::ptrdiff_t>(np * static_cast<std::size_t>(col)));
                else
                    out.bglob = bcol;
            }
        }

        out.status = Status::Ok;
        return out;
    }

    /// qpfstats FUSED reduce→jackknife→smooth→recenter ON THE GPU (the PERF path; backend.hpp
    /// qpfstats_blocks_smooth). ONE device residency: the genotype-f4 numerator reduce
    /// (launch_dstat_block_reduce → dNum/dCnt RESIDENT), the per-comb block-JACKKNIFE
    /// (launch_qpfstats_numer_jackknife → dNumer/dYmat/dY, native FP64, the §12 carve-out —
    /// formerly the host ~305k×711 long-double loop, the GPU-idle "0" half of the 100/0
    /// alternation), the shared-factor batched smoothing SOLVE (the SAME syrk+potrf+gemm+Dtrsm
    /// pair as qpfstats_smooth, reading dYmat/dY straight from VRAM — NO ymat/y H2D re-upload),
    /// and the per-pair RECENTER jackknife (launch_qpfstats_recenter_shift over the resident
    /// solved dRhs → dShift). The ~1.7GB-each numsum/cnt D2H is ELIMINATED; only the small
    /// b[npairs×n_block] / bglob[npairs] / recenter_shift[npairs] cross back. matmul sub-steps
    /// run `precision`; the jackknives + Cholesky/solve are native FP64.
    ///
    /// HOST-POINTER overload (the CpuBackend-parity entry / non-resident callers): H2D
    /// Q/V into resident buffers, then the shared core. The DeviceDecodeResult overload
    /// SKIPS this H2D (the M4 cure — Q/V already resident from decode_af_compact_autosome).
    [[nodiscard]] QpfstatsSmooth qpfstats_blocks_smooth(
        const double* Q, const double* V, int P, long M, const int* block_id, int n_block,
        std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
        std::span<const int> block_sizes, double ridge, const Precision& precision) override {
        guard_device();
        if (npopcomb <= 0 || npairs <= 0 || n_block <= 0 || P <= 0 || M <= 0 ||
            quadruples.size() < 4) {
            QpfstatsSmooth out;
            out.npairs = npairs;
            out.n_block = n_block;
            out.status = Status::Ok;
            return out;
        }
        const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
        DeviceBuffer<double> dQ(pm), dV(pm);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ.data(), Q, pm * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV.data(), V, pm * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        return qpfstats_blocks_smooth_device(dQ.data(), dV.data(), P, M, block_id, n_block,
                                             quadruples, x, npopcomb, npairs, block_sizes,
                                             ridge, precision);
    }

    /// DEVICE-RESIDENT overload — reads Q/V already RESIDENT in VRAM from a
    /// DeviceDecodeResult (the M4 cure: NO Q/V H2D). Math byte-identical to the
    /// host-pointer overload (the SAME fused core over the SAME compacted Q/V).
    [[nodiscard]] QpfstatsSmooth qpfstats_blocks_smooth(
        const steppe::device::DeviceDecodeResult& dec, const int* block_id, int n_block,
        std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
        std::span<const int> block_sizes, double ridge, const Precision& precision) override {
        guard_device();
        if (dec.empty() || dec.q_device() == nullptr) {
            QpfstatsSmooth out;
            out.npairs = npairs;
            out.n_block = n_block;
            out.status = Status::Ok;
            return out;
        }
        return qpfstats_blocks_smooth_device(dec.q_device(), dec.v_device(), dec.P,
                                             dec.M_kept, block_id, n_block, quadruples, x,
                                             npopcomb, npairs, block_sizes, ridge, precision);
    }

    /// The qpfstats_blocks_smooth CORE over ALREADY-RESIDENT device Q/V pointers
    /// (the single-source fused body both the host-pointer and the DeviceDecodeResult
    /// overloads share). dQ/dV are borrowed device pointers (NOT freed here): the
    /// host overload owns DeviceBuffers it uploaded; the device overload passes the
    /// resident DeviceDecodeResult Q/V — NO Q/V H2D (the M4 cure). Math byte-identical.
    [[nodiscard]] QpfstatsSmooth qpfstats_blocks_smooth_device(
        const double* dQ, const double* dV, int P, long M, const int* block_id, int n_block,
        std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
        std::span<const int> block_sizes, double ridge, const Precision& precision) {
        QpfstatsSmooth out;
        out.npairs = npairs;
        out.n_block = n_block;
        const int N = static_cast<int>(quadruples.size() / 4);
        if (npopcomb <= 0 || npairs <= 0 || n_block <= 0 || N <= 0 || P <= 0 || M <= 0) {
            out.status = Status::Ok;
            return out;
        }

        const std::size_t nc = static_cast<std::size_t>(npopcomb);
        const std::size_t np = static_cast<std::size_t>(npairs);
        const std::size_t nbb = static_cast<std::size_t>(n_block);
        const std::size_t nb_out = nc * nbb;

        // ---- Per-block contiguous SNP layout (the SINGLE-SOURCE inverse of assign_blocks; the
        // SAME primitive dstat_block_reduce uses). begin/size as int (M fits int by B22). ----
        const std::vector<core::BlockRange> ranges =
            core::block_ranges(std::span<const int>(block_id, static_cast<std::size_t>(M)),
                               M, n_block);
        std::vector<int> begin(nbb), size(nbb);
        for (int b = 0; b < n_block; ++b) {
            begin[static_cast<std::size_t>(b)] = static_cast<int>(ranges[static_cast<std::size_t>(b)].begin);
            size[static_cast<std::size_t>(b)]  = static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
        }

        const std::size_t nq = static_cast<std::size_t>(N) * 4u;

        // ---- Upload the quad table + block layout + the design x (col-major). Q/V are
        // BORROWED resident pointers (the host overload uploaded them; the device
        // overload passes the resident DeviceDecodeResult Q/V — NO Q/V H2D here). ----
        DeviceBuffer<int> dQuad(nq), dBegin(nbb), dSize(nbb);
        DeviceBuffer<double> dX(nc * np);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQuad.data(), quadruples.data(), nq * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBegin.data(), begin.data(), nbb * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSize.data(), size.data(), nbb * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dX.data(), x.data(), nc * np * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));

        // ---- 1. The genotype-f4 numerator reduce → dNum/dCnt RESIDENT (densum ignored). ----
        DeviceBuffer<double> dNum(nb_out), dDen(nb_out), dCnt(nb_out);
        launch_dstat_block_reduce(dQ, dV, P, M, dQuad.data(), N,
                                  dBegin.data(), dSize.data(), n_block,
                                  dNum.data(), dDen.data(), dCnt.data(), stream_.get());

        // ---- 2. numer + ymat (col-major) + the GLOBAL per-comb jackknife y, ON-DEVICE. -----
        // The RHS source for the solve is [npopcomb × (n_block+1)]: columns 0..n_block-1 = ymat,
        // column n_block = y. We write ymat into the first nc*nb of dRhsSrc and y into the last.
        const int ncols = n_block + 1;
        const std::size_t ncols_sz = static_cast<std::size_t>(ncols);
        DeviceBuffer<double> dNumer(nb_out);
        DeviceBuffer<double> dRhsSrc(nc * ncols_sz);
        launch_qpfstats_numer_jackknife(dNum.data(), dCnt.data(), npopcomb, n_block,
                                        dNumer.data(), dRhsSrc.data(),
                                        dRhsSrc.data() + nc * nbb, stream_.get());

        // ---- The shared-factor batched smoothing solve, reading dRhsSrc straight from VRAM
        // (IDENTICAL math to qpfstats_smooth; NO ymat/y H2D re-upload). Zero the NaN entries +
        // count NaN comb-rows per column (the AT2 ymat_chunk[nan]=0).
        DeviceBuffer<int> dNanPerCol(ncols_sz);
        launch_qpfstats_zero_nan_ymat(dRhsSrc.data(), npopcomb, ncols, dNanPerCol.data(),
                                      stream_.get());
        std::vector<int> nan_per_col(ncols_sz, 0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(nan_per_col.data(), dNanPerCol.data(),
                                          ncols_sz * sizeof(int), cudaMemcpyDeviceToHost,
                                          stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

        // A_shared = x'x + ridge·I  (cublasDsyrk, EmulatedFp64 via the f2 policy).
        DeviceBuffer<double> dA(np * np);
        {
            const double alpha = 1.0, beta = 0.0;
            const MathModeScope syrk_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
            engage_f2_precision(blas_.get(), precision);
            CUBLAS_CHECK(cublasDsyrk(blas_.get(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                     npairs, npopcomb, &alpha, dX.data(), npopcomb,
                                     &beta, dA.data(), npairs));
        }
        launch_symmetrize_lower_to_full(dA.data(), npairs, stream_.get());
        launch_qpfstats_add_ridge_diag(dA.data(), npairs, ridge, stream_.get());

        // RHS = x' · RhsSrc  (cublasDgemm, [npairs × ncols], EmulatedFp64).
        DeviceBuffer<double> dRhs(np * ncols_sz);
        {
            const double alpha = 1.0, beta = 0.0;
            const MathModeScope gemm_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
            engage_f2_precision(blas_.get(), precision);
            CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                                     npairs, ncols, npopcomb, &alpha,
                                     dX.data(), npopcomb, dRhsSrc.data(), npopcomb,
                                     &beta, dRhs.data(), npairs));
        }

        // L = chol(A_shared) (cusolverDnDpotrf LOWER; native FP64 carve-out).
        const CusolverMathModeScope solve_scope =
            engage_solver_precision(solver_.get(), solve_precision_, &emulation_honorable);
        DeviceBuffer<int> dInfo(1);
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver_.get(), CUBLAS_FILL_MODE_LOWER,
                                                   npairs, dA.data(), npairs, &lwork));
        DeviceBuffer<double> dWork(static_cast<std::size_t>(lwork > 0 ? lwork : 1));
        CUSOLVER_CHECK(cusolverDnDpotrf(solver_.get(), CUBLAS_FILL_MODE_LOWER, npairs,
                                        dA.data(), npairs, dWork.data(), lwork, dInfo.data()));
        int info = 0;
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(&info, dInfo.data(), sizeof(int),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        if (info != 0) { out.status = Status::NonSpdCovariance; return out; }

        // The multi-column solve A·B = RHS via a cublasDtrsm PAIR over ALL ncols at once.
        {
            const double one = 1.0;
            CUBLAS_CHECK(cublasDtrsm(blas_.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, npairs, ncols, &one,
                                     dA.data(), npairs, dRhs.data(), npairs));
            CUBLAS_CHECK(cublasDtrsm(blas_.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, npairs, ncols, &one,
                                     dA.data(), npairs, dRhs.data(), npairs));
        }
        // dRhs now holds the solution B: columns 0..n_block-1 = b, column n_block = bglob.
        // It STAYS RESIDENT so the recenter kernel reads b/bglob straight from VRAM.

        // ---- The per-pair RECENTER jackknife ON-DEVICE → dShift (reads the resident b/bglob).
        DeviceBuffer<int> dBlockSizes(nbb);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), block_sizes.data(),
                                          nbb * sizeof(int), cudaMemcpyHostToDevice,
                                          stream_.get()));
        DeviceBuffer<double> dShift(np);
        launch_qpfstats_recenter_shift(dRhs.data(), dRhs.data() + np * nbb, dBlockSizes.data(),
                                       npairs, n_block, dShift.data(), stream_.get());

        // ---- D2H ONLY the small outputs: b, bglob, recenter_shift. --------------------------
        std::vector<double> sol(np * ncols_sz, 0.0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(sol.data(), dRhs.data(), np * ncols_sz * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        out.recenter_shift.assign(np, 0.0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.recenter_shift.data(), dShift.data(),
                                          np * sizeof(double), cudaMemcpyDeviceToHost,
                                          stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        out.b.assign(np * nbb, 0.0);
        std::copy(sol.begin(), sol.begin() + static_cast<std::ptrdiff_t>(np * nbb), out.b.begin());
        out.bglob.assign(np, 0.0);
        std::copy(sol.begin() + static_cast<std::ptrdiff_t>(np * nbb), sol.end(),
                  out.bglob.begin());

        // ---- PARTIAL-NaN downdate fallback (0 < k_col < npopcomb): the AT2 generic-solve
        // branch. ABSENT on the real golden (block 536 is all-NaN ⇒ the zero-RHS shared solve
        // is exact) ⇒ ZERO iterations in production. When present, re-solve ONLY those columns
        // host-side with A_b = A_shared - x[nan]'x[nan]; the recenter is then recomputed on the
        // corrected b. Bit-faithful to qpfstats_smooth's fallback. We D2H ymat/y ONLY here
        // (the cold path) to feed the host recompute — the hot path never pays this.
        bool any_partial = false;
        for (int col = 0; col < ncols; ++col)
            if (nan_per_col[static_cast<std::size_t>(col)] > 0 &&
                nan_per_col[static_cast<std::size_t>(col)] < npopcomb) { any_partial = true; break; }
        if (any_partial) {
            std::vector<double> ymat(nb_out, 0.0), yv(nc, 0.0);
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(ymat.data(), dRhsSrc.data(),
                                              nb_out * sizeof(double), cudaMemcpyDeviceToHost,
                                              stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(yv.data(), dRhsSrc.data() + nc * nbb,
                                              nc * sizeof(double), cudaMemcpyDeviceToHost,
                                              stream_.get()));
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
            const auto xat = [&](int c, int p) -> double {
                return x[static_cast<std::size_t>(c) + nc * static_cast<std::size_t>(p)];
            };
            std::vector<double> Abase(np * np, 0.0);
            for (int i = 0; i < npairs; ++i)
                for (int j = 0; j < npairs; ++j) {
                    long double s = 0.0L;
                    for (int c = 0; c < npopcomb; ++c)
                        s += static_cast<long double>(xat(c, i)) * static_cast<long double>(xat(c, j));
                    double v = static_cast<double>(s);
                    if (i == j) v += ridge;
                    Abase[static_cast<std::size_t>(i) + np * static_cast<std::size_t>(j)] = v;
                }
            std::vector<double> rhs(np), bcol, Adown;
            for (int col = 0; col < ncols; ++col) {
                const int kc = nan_per_col[static_cast<std::size_t>(col)];
                if (kc <= 0 || kc >= npopcomb) continue;
                std::fill(rhs.begin(), rhs.end(), 0.0);
                Adown = Abase;
                for (int c = 0; c < npopcomb; ++c) {
                    const double sv = (col < n_block)
                        ? ymat[nc * static_cast<std::size_t>(col) + static_cast<std::size_t>(c)]
                        : yv[static_cast<std::size_t>(c)];
                    if (!std::isfinite(sv)) {
                        for (int i = 0; i < npairs; ++i)
                            for (int j = 0; j < npairs; ++j)
                                Adown[static_cast<std::size_t>(i) + np * static_cast<std::size_t>(j)] -=
                                    xat(c, i) * xat(c, j);
                        continue;
                    }
                    for (int p = 0; p < npairs; ++p) rhs[static_cast<std::size_t>(p)] += xat(c, p) * sv;
                }
                const core::LinAlgStatus st = core::solve(Adown, npairs, rhs, bcol);
                if (!st.ok) { out.status = Status::NonSpdCovariance; return out; }
                if (col < n_block)
                    std::copy(bcol.begin(), bcol.end(),
                              out.b.begin() + static_cast<std::ptrdiff_t>(np * static_cast<std::size_t>(col)));
                else
                    out.bglob = bcol;
            }
            // Recompute the recenter on the corrected b (host; the same f2blocks_pair_est ref).
            std::vector<int> bl(block_sizes.begin(), block_sizes.end());
            std::vector<double> series(nbb, 0.0);
            for (int p = 0; p < npairs; ++p) {
                for (int b = 0; b < n_block; ++b)
                    series[static_cast<std::size_t>(b)] =
                        out.b[static_cast<std::size_t>(p) + np * static_cast<std::size_t>(b)];
                out.recenter_shift[static_cast<std::size_t>(p)] =
                    out.bglob[static_cast<std::size_t>(p)] - core::f2blocks_pair_est(series, bl);
            }
        }

        out.status = Status::Ok;
        return out;
    }

    /// Probe the capability tier of THE device this backend is bound to (the
    /// per-device-instance contract, backend.hpp; architecture.md §9, §11.4, §12;
    /// cleanup 00-overview §(2).1; device-cuda-cuda_backend F20). Pure observability
    /// + data-movement enablement — every field is PARITY-NEUTRAL (§12), so this is
    /// never on the statistic path and never changes a reported number.
    ///
    /// NON-THROWING TAGGED-DEGRADE for the P2P probe (the capability-tier law,
    /// architecture.md §11.4; workflow wxz1fiiln; cleanup 00-overview §(2).4):
    /// `cudaDeviceCanAccessPeer` answering "no" is EXPECTED on the budget GeForce
    /// tier (P2P driver-disabled) and on this device vs itself — it is a tagged
    /// degrade, not a fault, so it routes through the NON-throwing STEPPE_CUDA_WARN
    /// (U1, check.cuh CAP-1/CAP-2), never the throwing STEPPE_CUDA_CHECK. The genuine
    /// faults (device-count / properties / mem-info queries on the bound device) DO
    /// throw — a backend that cannot read its own device is a real error.
    ///
    /// `const` + DEVICE-NEUTRAL: the probe makes `device_id_` current only for the
    /// duration of the queries and RESTORES the entry device, so calling it never
    /// leaks a `cudaSetDevice` side effect (it is a pure query of THIS backend's
    /// device, callable from any ambient-device context — e.g. `Resources` probing
    /// each per-device backend in turn).
    [[nodiscard]] BackendCapabilities capabilities() const override {
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

    // =====================================================================
    // qpAdm fit-engine virtuals ON THE GPU (the FROZEN CONTRACT §2; M(fit-4)).
    // The PRODUCTION GPU path: f2 stays RESIDENT in VRAM (the gather kernel
    // reads f2.f2_device() directly — no D2H, no host round-trip of the big
    // tensor), and every step runs on the device (gather/loo/xtau kernels,
    // cublasDsyrk for Q, cusolverDn Cholesky for Qinv, the on-device
    // transliterated small-LA for the SVD seed / ALS / weight / chisq, and the
    // BATCHED on-device LOO re-fits for S7). The only host transfers are the
    // small fit intermediates that cross the CUDA-free F4Blocks/JackknifeCov/
    // GlsWeights seam (X/loo/total m*nb, Q/Qinv m²) — KB-scale, inherent to the
    // existing host-vector seam. The numbers reproduce the bit-exact FP64
    // CpuBackend oracle (the parity anchor) so the GPU result matches the
    // af6a8c2 golden. Native FP64 end-to-end (the parity gate; §12).
    // =====================================================================

    /// SOLVE-PRECISION promotion-seam setter (ROADMAP §6; base no-op overridden
    /// here). Records the per-stage solve-precision request that drives the
    /// cuSOLVER `CusolverMathModeScope` at the solve sites (currently the S4
    /// jackknife SPD inverse potrf/potri). DEFAULT (unset) is native Fp64 ⇒ the
    /// scope targets native and the af6a8c2 golden parity is unchanged; promoting
    /// to EmulatedFp64{bits} routes the honorable solve through the emulated
    /// tensor-core path (validated per stage against the native oracle). The matmul
    /// `Precision` the qpAdm virtuals receive is a SEPARATE axis and is unaffected.
    void set_solve_precision(const Precision& precision) override {
        solve_precision_ = precision;
    }

    /// S8 instrumentation (observability): how many BATCHED chunk-dispatches this
    /// backend has issued through fit_models_batched (one per same-shape bucket chunk,
    /// NOT one per model). The rotation test asserts this is a FEW (one per left-size
    /// bucket), proving the rotation ran GPU-BATCHED, not as a per-model host loop.
    [[nodiscard]] std::size_t batched_dispatch_count() const override {
        return batched_dispatch_count_;
    }

    /// F1 / OQ-12 — the ASCENDING SURVIVOR block id list for a resident f2 (AT2
    /// read_f2(remove_na=TRUE)). Runs the on-device keep kernel over the resident Vpair
    /// (shared predicate core::pair_block_is_missing), reads the tiny [nb] keep vector
    /// down, and returns `which(keep)`. When the resident handle carries NO Vpair (an
    /// upload path that left it empty — the global-intersection invariant: no missing
    /// blocks) OR Vpair is empty, EVERY block survives (identity 0..nb-1) and no kernel
    /// runs — the bit-identical no-drop path. The result is model-INDEPENDENT (a property
    /// of the loaded f2), so callers compute it once per assemble/bucket.
    [[nodiscard]] std::vector<int> device_survivor_blocks(
        const steppe::device::DeviceF2Blocks& f2, int nb, int P) {
        std::vector<int> surv;
        if (nb <= 0) return surv;
        surv.reserve(static_cast<std::size_t>(nb));
        // No resident Vpair ⇒ keep every block (no missing-block info ⇒ no drop).
        if (f2.vpair_device() == nullptr) {
            for (int b = 0; b < nb; ++b) surv.push_back(b);
            return surv;
        }
        DeviceBuffer<int> dKeep(static_cast<std::size_t>(nb));
        launch_f2_block_keep(f2.vpair_device(), P, nb, dKeep.data(), stream_.get());
        std::vector<int> keep(static_cast<std::size_t>(nb), 1);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(keep.data(), dKeep.data(),
                                          static_cast<std::size_t>(nb) * sizeof(int),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        for (int b = 0; b < nb; ++b)
            if (keep[static_cast<std::size_t>(b)] != 0) surv.push_back(b);
        return surv;
    }

    /// GPU-ONLY f-stat SWEEP core (shared by f4_sweep k=4 / f3_sweep k=3). The fix for the
    /// CPU-bound host-enumeration disaster: the HOST drives ONLY the chunk loop; EVERYTHING
    /// per item runs on the device. Per chunk [c0, c0+C):
    ///   (1) UNRANK — launch_sweep_unrank_* maps thread t -> its quartet/triple (combinatorial
    ///       number system) and WRITES the device index list (4*C / 3*C ints) — the EXACT
    ///       buffer the gather reads; REPLACES the per-chunk H2D (NO host enumeration).
    ///   (2) GATHER + loo/total + xtau + diag_var — the SAME device kernels the explicit
    ///       run_f4 path runs, verbatim (zero fork). Native FP64 (the f-stat cancellation
    ///       carve-out + the well-conditioned diagonal SE).
    ///   (3) |z| FILTER — launch_sweep_zfilter writes est/se/z + the uint8 survivor flag.
    ///   (4) COMPACT — cub::DeviceSelect::Flagged stream-compacts the survivor rows (est/se/z +
    ///       the deinterleaved key columns) ON THE DEVICE (two-call temp-storage idiom; the
    ///       num_selected count is written on device). NO host per-item filter.
    ///   (5) D2H ONLY the compacted survivors (num_selected, small) across the CUDA-free seam.
    /// The chunk is sized from free VRAM (bounds per-chunk VRAM + the host survivor mirror).
    /// Single-GPU (multi-GPU PARKED). The maxcomb cap is enforced in the CUDA-free driver
    /// BEFORE this is reached; this also re-checks for safety.
    [[nodiscard]] SweepSurvivors run_fstat_sweep_device(
        const steppe::device::DeviceF2Blocks& f2, const SweepConfig& cfg,
        const Precision& precision, int k) {
        (void)precision;  // native FP64 by the f-stat cancellation carve-out (consistent policy)
        guard_device();

        SweepSurvivors out;
        const int P = f2.P;
        const int nb_full = f2.n_block;
        const int range = cfg.pop_subset.empty() ? P : static_cast<int>(cfg.pop_subset.size());

        // Total enumerated C(range, k) (saturating; the host driver already capped, but echo it).
        unsigned long long enumerated = 1ULL;
        if (k < 0 || range < k) {
            enumerated = 0ULL;
        } else {
            for (int i = 1; i <= k; ++i) {
                const unsigned long long num = static_cast<unsigned long long>(range - k + i);
                if (num != 0 && enumerated > (~0ULL) / num) { enumerated = ~0ULL; break; }
                enumerated = enumerated * num / static_cast<unsigned long long>(i);
            }
        }
        out.enumerated = static_cast<std::size_t>(
            enumerated > static_cast<unsigned long long>(SIZE_MAX) ? SIZE_MAX : enumerated);

        if (range < k || k < 1 || nb_full <= 0 || f2.f2_device() == nullptr || enumerated == 0ULL) {
            out.status = Status::Ok;  // nothing to sweep — a clean empty Ok result.
            return out;
        }

        // F1/OQ-12 SURVIVOR-block drop (ONCE for the whole sweep — block-missingness is a
        // property of the f2, not the item). The gather/loo/xtau/diag_var all use nb_s.
        const std::vector<int> surv = device_survivor_blocks(f2, nb_full, P);
        const int nb_s = static_cast<int>(surv.size());
        if (nb_s <= 0) { out.status = Status::Ok; return out; }
        const bool dropped = (nb_s != nb_full);

        // Survivor block sizes + n = Σ block_sizes (the jackknife weight total).
        std::vector<int> surv_block_sizes(static_cast<std::size_t>(nb_s), 0);
        long long n_ll = 0;
        for (int bs = 0; bs < nb_s; ++bs) {
            const int orig = surv[static_cast<std::size_t>(bs)];
            const int sz = f2.block_sizes[static_cast<std::size_t>(orig)];
            surv_block_sizes[static_cast<std::size_t>(bs)] = sz;
            n_ll += sz;
        }
        const double n = static_cast<double>(n_ll);

        // Persistent device buffers (uploaded ONCE): survivor block sizes + survivor map + subset.
        DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb_s));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), surv_block_sizes.data(),
                                          static_cast<std::size_t>(nb_s) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        DeviceBuffer<int> dSurv(static_cast<std::size_t>(dropped ? nb_s : 1));
        if (dropped)
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSurv.data(), surv.data(),
                                              static_cast<std::size_t>(nb_s) * sizeof(int),
                                              cudaMemcpyHostToDevice, stream_.get()));
        const bool use_subset = !cfg.pop_subset.empty();
        DeviceBuffer<int> dSubset(static_cast<std::size_t>(use_subset ? range : 1));
        if (use_subset)
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSubset.data(), cfg.pop_subset.data(),
                                              static_cast<std::size_t>(range) * sizeof(int),
                                              cudaMemcpyHostToDevice, stream_.get()));

        // The bounded top-K reservoir target K (clamped to INT_MAX + the enumerated total). Sized
        // FIRST so its FIXED device footprint (CAP=2K reservoir + 2K gather scratch + sort
        // outputs ≈ 5·CAP·~52 B) is subtracted from free VRAM before the chunk is sized.
        std::size_t K_budget = (cfg.top_k > 0) ? cfg.top_k : kFstatDefaultSweepTopK;
        if (K_budget > static_cast<std::size_t>(enumerated)) K_budget = static_cast<std::size_t>(enumerated);
        if (K_budget > static_cast<std::size_t>(0x40000000)) K_budget = 0x40000000;
        if (K_budget < 1) K_budget = 1;
        const std::size_t cap_budget = K_budget * 2;
        // Reservoir + gather-scratch + sort-out per CAP slot: 4 doubles + 4 ints (reservoir) +
        // 4 doubles + 4 ints (gather scratch) + 1 double (sort keys) + 2 ints (perm in/out) ≈
        // 9·8 + 10·4 = 112 B/slot. Round up generously for the CUB sort temp.
        const std::size_t reservoir_bytes = cap_budget * 160;

        // CHUNK size from free VRAM. Per item the device holds: the k-int key (k·4B) + x_blocks
        // (nb_s·8) + x_loo (nb_s·8) + xtau (nb_s·8) + est/loo-total/tot_line (3·8) + diag var
        // (8) + est/se/z/absz (4·8) + 4 key columns (4·4) + flag (1) + the compacted outputs
        // (5·8 + 4·4). Bound by 0.4·(free − reservoir) to leave headroom for the CUB temp storage
        // + the resident f2 + the fixed reservoir footprint.
        std::size_t free_b = 0, total_b = 0;
        STEPPE_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        const std::size_t free_after_res = (free_b > reservoir_bytes) ? (free_b - reservoir_bytes)
                                                                      : (free_b / 4);
        const std::size_t per_item_bytes =
            static_cast<std::size_t>(k) * sizeof(int) +                       // dItems
            static_cast<std::size_t>(nb_s) * sizeof(double) * 3 +             // dX + dLoo + dXtau
            sizeof(double) * 4 +                                              // dTotal,dTotLine,dVar,(pad)
            sizeof(double) * 4 +                                             // dEst,dSe,dZ,dAbsZ
            static_cast<std::size_t>(4) * sizeof(int) +                       // 4 key cols
            sizeof(unsigned char) +                                          // flag
            sizeof(double) * 5 + static_cast<std::size_t>(4) * sizeof(int);   // compacted outs (+absz)
        std::size_t budget = static_cast<std::size_t>(static_cast<double>(free_after_res) * 0.4);
        if (budget < per_item_bytes) budget = per_item_bytes;  // at least one item
        std::size_t chunk_sz = budget / (per_item_bytes == 0 ? 1 : per_item_bytes);
        // Optional override (the chunk lever) via STEPPE_FSTAT_CHUNK.
        if (const char* env = std::getenv("STEPPE_FSTAT_CHUNK")) {
            const long v = std::strtol(env, nullptr, 10);
            if (v > 0) chunk_sz = static_cast<std::size_t>(v);
        }
        if (chunk_sz < 1) chunk_sz = 1;
        // Clamp a chunk to INT_MAX (CUB num_items + our int kernels) and to the total.
        if (chunk_sz > static_cast<std::size_t>(0x40000000)) chunk_sz = 0x40000000;
        if (static_cast<unsigned long long>(chunk_sz) > enumerated)
            chunk_sz = static_cast<std::size_t>(enumerated);
        const int C_max = static_cast<int>(chunk_sz);

        // Allocate the per-chunk device working set ONCE (sized to C_max); reuse across chunks.
        const std::size_t Cm = static_cast<std::size_t>(C_max);
        const std::size_t kk = static_cast<std::size_t>(k);
        DeviceBuffer<int> dItems(Cm * kk);
        DeviceBuffer<double> dX(Cm * static_cast<std::size_t>(nb_s));
        DeviceBuffer<double> dLoo(Cm * static_cast<std::size_t>(nb_s));
        DeviceBuffer<double> dXtau(Cm * static_cast<std::size_t>(nb_s));
        DeviceBuffer<double> dTotal(Cm);
        DeviceBuffer<double> dTotLine(Cm);
        DeviceBuffer<double> dVar(Cm);
        DeviceBuffer<double> dEst(Cm), dSe(Cm), dZ(Cm), dAbsZ(Cm);
        DeviceBuffer<int> dC0(Cm), dC1(Cm), dC2(Cm), dC3(Cm);
        DeviceBuffer<unsigned char> dFlags(Cm);
        // Per-chunk compacted survivors (above the CURRENT tau; bounded by C_max).
        DeviceBuffer<double> dEstSel(Cm), dSeSel(Cm), dZSel(Cm), dAbsZSel(Cm);
        DeviceBuffer<int> dC0Sel(Cm), dC1Sel(Cm), dC2Sel(Cm), dC3Sel(Cm);
        DeviceBuffer<int> dNumSel(1);

        // ---- BOUNDED DEVICE TOP-K reservoir (the fix for the unbounded-host-vector OOM) ----
        // The host NEVER accumulates survivors. Instead a FIXED CAP=O(K) device reservoir keeps
        // the running top-K (largest |z|) with a MONOTONICALLY RISING threshold tau: the zfilter
        // pre-drops items at-or-below the current K-th |z|, so the per-chunk survivor count only
        // SHRINKS as the sweep proceeds. Host RAM is O(K) (~40 MB at K=1e6) regardless of the
        // billions computed. mode 1 = TopK (tau rises); mode 0 = MinZ (tau pinned at min_z, but
        // the reservoir still caps to K as a hard safety ceiling so a MinZ sweep cannot OOM).
        const int zmode = cfg.filter_mode;  // 0 = fixed-tau MinZ; 1 = rising-tau TopK.
        // K = the reservoir target; CAP = 2K slack so a chunk's survivors always fit before a
        // compact (each compact returns the fill to <=K, leaving >=K free headroom). K is also
        // clamped to INT_MAX (CUB num_items + our int kernels) and to the enumerated total.
        std::size_t K_sz = (cfg.top_k > 0) ? cfg.top_k : kFstatDefaultSweepTopK;
        if (K_sz > static_cast<std::size_t>(enumerated)) K_sz = static_cast<std::size_t>(enumerated);
        if (K_sz > static_cast<std::size_t>(0x40000000)) K_sz = 0x40000000;
        if (K_sz < 1) K_sz = 1;
        const int K = static_cast<int>(K_sz);
        const std::size_t CAP_sz = K_sz * 2;  // slack = K (proven: chunk always fits pre-compact).
        const int CAP = (CAP_sz > static_cast<std::size_t>(INT_MAX)) ? INT_MAX
                                                                     : static_cast<int>(CAP_sz);

        // Persistent reservoir state allocated ONCE (fixed device footprint, ~80-100 B/slot).
        DeviceBuffer<double> dResEst(CAP_sz), dResSe(CAP_sz), dResZ(CAP_sz), dResAbsZ(CAP_sz);
        DeviceBuffer<int> dResC0(CAP_sz), dResC1(CAP_sz), dResC2(CAP_sz), dResC3(CAP_sz);
        // Double-buffered sort outputs (CUB SortPairsDescending, non-overwrite) + gather scratch.
        DeviceBuffer<double> dSortAbsZ(CAP_sz);    // sorted keys (descending).
        DeviceBuffer<int> dPermIn(CAP_sz), dPermOut(CAP_sz);  // iota -> sorted permutation.
        DeviceBuffer<double> dGEst(CAP_sz), dGSe(CAP_sz), dGZ(CAP_sz), dGAbsZ(CAP_sz);
        DeviceBuffer<int> dGC0(CAP_sz), dGC1(CAP_sz), dGC2(CAP_sz), dGC3(CAP_sz);
        DeviceBuffer<double> dTau(1);
        // tau floor = min_z (for TopK this is just the floor; for MinZ it stays fixed here).
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dTau.data(), &cfg.min_z, sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        int res_n = 0;  // host mirror of the reservoir fill (advanced by num_sel, reset on compact).

        // CUB temp storage — size ONCE at the max num_items each call ever sees (C_max for the
        // per-chunk Flagged compaction; CAP for the reservoir SortPairsDescending) and reuse.
        // Two-call idiom: d_temp_storage=nullptr first to query temp_storage_bytes.
        std::size_t sel_bytes = 0;
        cub::DeviceSelect::Flagged(nullptr, sel_bytes, dEst.data(), dFlags.data(),
                                   dEstSel.data(), dNumSel.data(),
                                   static_cast<std::int64_t>(C_max), stream_.get());
        std::size_t sort_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, sort_bytes, dResAbsZ.data(), dSortAbsZ.data(), dPermIn.data(),
            dPermOut.data(), static_cast<std::int64_t>(CAP), 0, sizeof(double) * 8, stream_.get());
        const std::size_t cub_bytes = std::max(sel_bytes, sort_bytes);
        DeviceBuffer<unsigned char> dCubTemp(cub_bytes == 0 ? 1 : cub_bytes);

        // COMPACT-AND-RAISE: sort the reservoir by |z| descending, truncate to K, raise tau to the
        // new K-th |z|. Fires when the reservoir would overflow CAP (and once at the end). After
        // it, res_n <= K. Because |z| is monotone in significance and tau only RISES, no kept item
        // is ever wrongly evicted relative to a GLOBAL top-K (an item dropped at the zfilter when
        // |z|<=tau could never make a top-K whose K-th value is already tau).
        auto compact_and_raise = [&](int keep) {
            if (res_n <= 0) return;
            const int m = res_n;
            launch_sweep_topk_iota(dPermIn.data(), m, stream_.get());
            std::size_t sb = cub_bytes;
            cub::DeviceRadixSort::SortPairsDescending(
                dCubTemp.data(), sb, dResAbsZ.data(), dSortAbsZ.data(), dPermIn.data(),
                dPermOut.data(), static_cast<std::int64_t>(m), 0, sizeof(double) * 8,
                stream_.get());
            const int newn = (m < keep) ? m : keep;  // truncate to <=K.
            // Gather the top `newn` rows (by the sorted permutation) into scratch, then swap back.
            launch_sweep_topk_gather(
                dPermOut.data(), newn, dResEst.data(), dResSe.data(), dResZ.data(),
                dResAbsZ.data(), dResC0.data(), dResC1.data(), dResC2.data(), dResC3.data(),
                dGEst.data(), dGSe.data(), dGZ.data(), dGAbsZ.data(),
                dGC0.data(), dGC1.data(), dGC2.data(), dGC3.data(), stream_.get());
            const std::size_t db = static_cast<std::size_t>(newn) * sizeof(double);
            const std::size_t ib = static_cast<std::size_t>(newn) * sizeof(int);
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResEst.data(), dGEst.data(), db, cudaMemcpyDeviceToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResSe.data(),  dGSe.data(),  db, cudaMemcpyDeviceToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResZ.data(),   dGZ.data(),   db, cudaMemcpyDeviceToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResAbsZ.data(),dGAbsZ.data(),db, cudaMemcpyDeviceToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC0.data(),  dGC0.data(),  ib, cudaMemcpyDeviceToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC1.data(),  dGC1.data(),  ib, cudaMemcpyDeviceToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC2.data(),  dGC2.data(),  ib, cudaMemcpyDeviceToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC3.data(),  dGC3.data(),  ib, cudaMemcpyDeviceToDevice, stream_.get()));
            // Raise tau to the new K-th |z| (only when the reservoir was actually FULL to K, and
            // only in TopK mode — MinZ keeps tau at the floor). dSortAbsZ is |z| descending, so
            // position K-1 is the K-th-largest. A no-op when newn < K (we have not yet seen K items).
            if (zmode == 1 && newn >= K)
                launch_sweep_topk_raise_tau(dSortAbsZ.data(), K, zmode, dTau.data(), stream_.get());
            res_n = newn;
        };

        // The HOST chunk loop: advance c0, run the chunk pipeline, APPEND survivors into the
        // device reservoir (D2D — NO host transfer of survivors). NO per-item host work (no
        // enumeration, no filter — those are device kernels); the only host touch is the num_sel
        // int readback (one int per chunk).
        for (unsigned long long c0 = 0; c0 < enumerated; c0 += static_cast<unsigned long long>(C_max)) {
            const unsigned long long remaining = enumerated - c0;
            const int C = (remaining < static_cast<unsigned long long>(C_max))
                              ? static_cast<int>(remaining) : C_max;
            if (C <= 0) break;

            // (1) UNRANK -> dItems (the device index list; NO host enumeration).
            if (k == 4)
                launch_sweep_unrank_quartets(static_cast<long long>(c0), C, range,
                                             use_subset ? dSubset.data() : nullptr,
                                             dItems.data(), stream_.get());
            else
                launch_sweep_unrank_triples(static_cast<long long>(c0), C, range,
                                            use_subset ? dSubset.data() : nullptr,
                                            dItems.data(), stream_.get());

            // (2) GATHER (the SAME device kernel the explicit path uses) + loo/total + xtau +
            // diag_var. nl=C, nr=1, m=C.
            if (k == 4)
                launch_assemble_f4_quartets_gather(f2.f2_device(), P, dItems.data(), C, nb_s,
                                                   dropped ? dSurv.data() : nullptr,
                                                   dX.data(), stream_.get());
            else
                launch_assemble_f3_triples_gather(f2.f2_device(), P, dItems.data(), C, nb_s,
                                                  dropped ? dSurv.data() : nullptr,
                                                  dX.data(), stream_.get());
            launch_f4_loo_total(dX.data(), dBlockSizes.data(), C, nb_s, n,
                                dLoo.data(), dTotal.data(), dTotLine.data(), stream_.get());
            launch_f4_xtau(dLoo.data(), dTotal.data(), dTotLine.data(), dBlockSizes.data(),
                           C, nb_s, n, dXtau.data(), stream_.get());
            launch_f4_diag_var(dXtau.data(), C, nb_s, dVar.data(), stream_.get());

            // (3) |z| FILTER against the LIVE rising tau (read from dTau, not a host constant)
            // -> dEst/dSe/dZ/dAbsZ + the uint8 survivor flag (|z| > tau). Device-side, no host.
            launch_sweep_zfilter_tau(dTotal.data(), dVar.data(), C, dTau.data(),
                                     dEst.data(), dSe.data(), dZ.data(), dAbsZ.data(),
                                     dFlags.data(), stream_.get());

            // Deinterleave the key columns for compaction.
            launch_sweep_deinterleave_keys(dItems.data(), C, k,
                                           dC0.data(), dC1.data(), dC2.data(), dC3.data(),
                                           stream_.get());

            // (4) COMPACT the chunk's above-tau survivors ON THE DEVICE (CUB Flagged; num_selected
            // written on device). The SAME 8 columns (est/se/z/absz + 4 keys) with the SAME flags.
            std::size_t sb = cub_bytes;
            cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dEst.data(), dFlags.data(),
                                       dEstSel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get());
            sb = cub_bytes;
            cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dSe.data(), dFlags.data(),
                                       dSeSel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get());
            sb = cub_bytes;
            cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dZ.data(), dFlags.data(),
                                       dZSel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get());
            sb = cub_bytes;
            cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dAbsZ.data(), dFlags.data(),
                                       dAbsZSel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get());
            sb = cub_bytes;
            cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dC0.data(), dFlags.data(),
                                       dC0Sel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get());
            sb = cub_bytes;
            cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dC1.data(), dFlags.data(),
                                       dC1Sel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get());
            sb = cub_bytes;
            cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dC2.data(), dFlags.data(),
                                       dC2Sel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get());
            sb = cub_bytes;
            cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dC3.data(), dFlags.data(),
                                       dC3Sel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get());

            int num_sel = 0;
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(&num_sel, dNumSel.data(), sizeof(int),
                                              cudaMemcpyDeviceToHost, stream_.get()));
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
            if (num_sel <= 0) continue;

            // (5) APPEND the chunk's survivors into the device reservoir (D2D). If they would
            // overflow CAP, compact-and-raise FIRST (frees >=K headroom), then append. A single
            // chunk may yield more than K survivors (the first chunks, before tau rises); append
            // in pieces of <=(CAP-res_n) and compact between, so the reservoir never overruns.
            int off = 0;
            while (off < num_sel) {
                if (res_n >= CAP) compact_and_raise(K);  // reservoir full -> truncate to K.
                int take = num_sel - off;
                if (take > CAP - res_n) take = CAP - res_n;  // fill to CAP, then compact.
                const std::size_t db = static_cast<std::size_t>(take) * sizeof(double);
                const std::size_t ib = static_cast<std::size_t>(take) * sizeof(int);
                const std::size_t doff = static_cast<std::size_t>(off);
                const std::size_t rdoff = static_cast<std::size_t>(res_n);
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResEst.data() + rdoff, dEstSel.data() + doff, db, cudaMemcpyDeviceToDevice, stream_.get()));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResSe.data()  + rdoff, dSeSel.data()  + doff, db, cudaMemcpyDeviceToDevice, stream_.get()));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResZ.data()   + rdoff, dZSel.data()   + doff, db, cudaMemcpyDeviceToDevice, stream_.get()));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResAbsZ.data()+ rdoff, dAbsZSel.data()+ doff, db, cudaMemcpyDeviceToDevice, stream_.get()));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC0.data()  + rdoff, dC0Sel.data()  + doff, ib, cudaMemcpyDeviceToDevice, stream_.get()));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC1.data()  + rdoff, dC1Sel.data()  + doff, ib, cudaMemcpyDeviceToDevice, stream_.get()));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC2.data()  + rdoff, dC2Sel.data()  + doff, ib, cudaMemcpyDeviceToDevice, stream_.get()));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC3.data()  + rdoff, dC3Sel.data()  + doff, ib, cudaMemcpyDeviceToDevice, stream_.get()));
                res_n += take;
                off += take;
            }
        }

        // FINAL compact: sort the reservoir by |z| descending + truncate to min(res_n,K). The
        // device now holds EXACTLY the top-K rows (sorted) — D2H ONLY these <=K rows.
        compact_and_raise(K);
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        const std::size_t ns = static_cast<std::size_t>(res_n < 0 ? 0 : res_n);

        std::vector<double> h_est(ns), h_se(ns), h_z(ns);
        std::vector<int> h_c0(ns), h_c1(ns), h_c2(ns), h_c3(ns);
        if (ns > 0) {
            const std::size_t db = ns * sizeof(double);
            const std::size_t ib = ns * sizeof(int);
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_est.data(), dResEst.data(), db, cudaMemcpyDeviceToHost, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_se.data(),  dResSe.data(),  db, cudaMemcpyDeviceToHost, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_z.data(),   dResZ.data(),   db, cudaMemcpyDeviceToHost, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_c0.data(),  dResC0.data(),  ib, cudaMemcpyDeviceToHost, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_c1.data(),  dResC1.data(),  ib, cudaMemcpyDeviceToHost, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_c2.data(),  dResC2.data(),  ib, cudaMemcpyDeviceToHost, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_c3.data(),  dResC3.data(),  ib, cudaMemcpyDeviceToHost, stream_.get()));
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        }

        // Pack the bounded top-K survivor set onto the CUDA-free POD (host RAM is O(K)).
        out.keys.resize(ns);
        out.est = std::move(h_est);
        out.se = std::move(h_se);
        out.z = std::move(h_z);
        for (std::size_t r = 0; r < ns; ++r)
            out.keys[r] = {h_c0[r], h_c1[r], h_c2[r], (k >= 4 ? h_c3[r] : 0)};
        out.status = Status::Ok;
        return out;
    }

    /// S3 — assemble the per-block f4 matrix X from DEVICE-RESIDENT f2 (zero D2H of
    /// the big tensor; the FROZEN CONTRACT §2a). The gather kernel reads
    /// f2.f2_device() in VRAM; the est_to_loo/x_total/tot_line reduction runs
    /// on-device; only the small X/loo/total cross the seam. Native FP64.
    [[nodiscard]] F4Blocks assemble_f4(const steppe::device::DeviceF2Blocks& f2,
                                       std::span<const int> left_idx,
                                       std::span<const int> right_idx,
                                       const Precision& precision) override {
        // CANCELLATION CARVE-OUT (unified precision policy; fit-engine.md §1.4, §OQ-5).
        // The f4 4-slab combine is a catastrophic-cancellation f-stat difference, held
        // native FP64 ALWAYS regardless of the requested `precision` — exactly the
        // f2-numerator carve-out: Ozaki emulation faithfully forms a product but cannot
        // recover bits annihilated by a prior subtraction. So even with the fit default
        // now EmulatedFp64{40}, this stage stays native.
        (void)precision;  // native FP64 (the cancellation-sensitive f-stat diff)
        guard_device();

        const int nl = static_cast<int>(left_idx.size()) - 1;
        const int nr = static_cast<int>(right_idx.size()) - 1;
        const int nb = f2.n_block;
        const int P = f2.P;

        F4Blocks out;
        out.nl = nl;
        out.nr = nr;
        const std::size_t m = static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr);
        out.x_total.assign(m, 0.0);
        tot_line_.assign(m, 0.0);
        if (nl <= 0 || nr <= 0 || nb <= 0 || f2.f2_device() == nullptr) {
            out.n_block = 0;
            return out;
        }

        // F1 / OQ-12 — MISSING-block drop (AT2 read_f2(remove_na=TRUE)), the GPU mirror
        // of the CpuBackend oracle. The keep-mask is computed ON-DEVICE from the resident
        // Vpair (launch_f2_block_keep, sharing core::pair_block_is_missing with the
        // oracle), read down as a tiny [nb] int vector, and the host builds the ASCENDING
        // SURVIVOR id list + the survivor block_sizes. The gather then COMPACTS dX onto
        // the survivor axis (d_surv maps compacted→original block). With NO Vpair (the
        // legacy upload path that leaves it empty) or no missing block, every block
        // survives ⇒ d_surv is identity and the path is bit-identical to pre-F1.
        const std::vector<int> surv = device_survivor_blocks(f2, nb, P);
        const int nb_s = static_cast<int>(surv.size());
        out.n_block = nb_s;
        out.block_sizes.assign(static_cast<std::size_t>(nb_s), 0);
        for (int bs = 0; bs < nb_s; ++bs)
            out.block_sizes[static_cast<std::size_t>(bs)] =
                f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
        out.x_blocks.assign(m * static_cast<std::size_t>(nb_s), 0.0);
        out.x_loo.assign(m * static_cast<std::size_t>(nb_s), 0.0);
        if (nb_s <= 0) return out;  // all blocks missing — degenerate (caller-gated)

        // Whether the survivor set differs from the full set (a real drop occurred);
        // identity survivor ⇒ pass d_surv = nullptr (the bit-identical no-drop kernel arm).
        const bool dropped = (nb_s != nb);

        // H2D the small model index vectors (length nl+1 / nr+1) + the SURVIVOR block
        // sizes + (only when a drop occurred) the survivor map.
        DeviceBuffer<int> dLeft(left_idx.size());
        DeviceBuffer<int> dRight(right_idx.size());
        DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb_s));
        DeviceBuffer<int> dSurv(static_cast<std::size_t>(dropped ? nb_s : 1));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dLeft.data(), left_idx.data(),
                                          left_idx.size() * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dRight.data(), right_idx.data(),
                                          right_idx.size() * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), out.block_sizes.data(),
                                          static_cast<std::size_t>(nb_s) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        if (dropped)
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSurv.data(), surv.data(),
                                              static_cast<std::size_t>(nb_s) * sizeof(int),
                                              cudaMemcpyHostToDevice, stream_.get()));

        // Device-resident X / loo / total / tot_line (sized to the SURVIVOR block count).
        DeviceBuffer<double> dX(m * static_cast<std::size_t>(nb_s));
        DeviceBuffer<double> dLoo(m * static_cast<std::size_t>(nb_s));
        DeviceBuffer<double> dTotal(m);
        DeviceBuffer<double> dTotLine(m);

        // S3 gather (reads RESIDENT f2; native FP64 4-slab combine), compacted onto the
        // survivor axis via d_surv (nullptr when no drop ⇒ identity, bit-identical).
        launch_assemble_f4_gather(f2.f2_device(), P, dLeft.data(), dRight.data(),
                                  nl, nr, nb_s, dropped ? dSurv.data() : nullptr,
                                  dX.data(), stream_.get());

        // n = Σ SURVIVOR block_sizes (host int → double; the jackknife normalizer).
        long long n_ll = 0;
        for (int v : out.block_sizes) n_ll += v;
        const double n = static_cast<double>(n_ll);

        // est_to_loo + x_total + tot_line (on-device reduction over SURVIVORS; FP64 order).
        launch_f4_loo_total(dX.data(), dBlockSizes.data(),
                            static_cast<int>(m), nb_s, n,
                            dLoo.data(), dTotal.data(), dTotLine.data(), stream_.get());

        // D2H the small fit intermediates across the CUDA-free seam (SURVIVOR-sized).
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_blocks.data(), dX.data(),
                                          m * static_cast<std::size_t>(nb_s) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_loo.data(), dLoo.data(),
                                          m * static_cast<std::size_t>(nb_s) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_total.data(), dTotal.data(),
                                          m * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(tot_line_.data(), dTotLine.data(),
                                          m * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        return out;
    }

    /// S3 host-oracle overload — NOT the GPU path. The CudaBackend implements only
    /// the DeviceF2Blocks form (zero D2H of the resident tensor); a host
    /// F2BlockTensor would force the big tensor onto the device only to read it back,
    /// which is the CpuBackend's oracle door, not the production GPU path.
    [[nodiscard]] F4Blocks assemble_f4(const F2BlockTensor& f2,
                                       std::span<const int> left_idx,
                                       std::span<const int> right_idx,
                                       const Precision& precision) override {
        (void)f2; (void)left_idx; (void)right_idx; (void)precision;
        throw std::runtime_error(
            "CudaBackend::assemble_f4(host): the GPU path reads DEVICE-RESIDENT f2 "
            "(assemble_f4(DeviceF2Blocks)); the host-tensor overload is the CpuBackend "
            "oracle door");
    }

    /// STANDALONE f4 quartet assemble (run_f4 seam; fit-engine §6) — the GPU mirror of the
    /// CpuBackend oracle. Build ONE F4Blocks whose m axis is the N input QUARTETS (nl=N,
    /// nr=1), reading the RESIDENT f2 (zero D2H of the tensor). Each quartet column gathers
    /// its OWN four-slab combine via launch_assemble_f4_quartets_gather; the est_to_loo /
    /// x_total / tot_line reduction REUSES launch_f4_loo_total (it reads only m), so the
    /// jackknife pipeline downstream is byte-identical to the fit path. The SURVIVOR-block
    /// drop (F1/OQ-12) reuses device_survivor_blocks. Native FP64 (cancellation carve-out).
    [[nodiscard]] F4Blocks assemble_f4_quartets(
        const steppe::device::DeviceF2Blocks& f2,
        std::span<const int> quartets,
        const Precision& precision) override {
        (void)precision;  // native FP64 (the cancellation-sensitive f-stat diff)
        guard_device();

        const int N = static_cast<int>(quartets.size()) / 4;  // quartet count (m axis)
        const int nb = f2.n_block;
        const int P = f2.P;

        F4Blocks out;
        out.nl = N;  // m = nl*nr = N (the batched f4 m-axis convention; matches the oracle)
        out.nr = 1;
        const std::size_t m = static_cast<std::size_t>(N);
        out.x_total.assign(m, 0.0);
        tot_line_.assign(m, 0.0);
        if (N <= 0 || nb <= 0 || f2.f2_device() == nullptr) {
            out.n_block = 0;
            return out;
        }

        // F1 / OQ-12 SURVIVOR drop (the SAME on-device keep-mask the fit path uses).
        const std::vector<int> surv = device_survivor_blocks(f2, nb, P);
        const int nb_s = static_cast<int>(surv.size());
        out.n_block = nb_s;
        out.block_sizes.assign(static_cast<std::size_t>(nb_s), 0);
        for (int bs = 0; bs < nb_s; ++bs)
            out.block_sizes[static_cast<std::size_t>(bs)] =
                f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
        out.x_blocks.assign(m * static_cast<std::size_t>(nb_s), 0.0);
        out.x_loo.assign(m * static_cast<std::size_t>(nb_s), 0.0);
        if (nb_s <= 0) return out;  // all blocks missing — degenerate (caller-gated)

        const bool dropped = (nb_s != nb);

        // H2D the flattened quartet quad array (4*N) + the SURVIVOR block sizes + (only
        // on a real drop) the survivor map.
        DeviceBuffer<int> dQuartets(quartets.size());
        DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb_s));
        DeviceBuffer<int> dSurv(static_cast<std::size_t>(dropped ? nb_s : 1));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQuartets.data(), quartets.data(),
                                          quartets.size() * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), out.block_sizes.data(),
                                          static_cast<std::size_t>(nb_s) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        if (dropped)
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSurv.data(), surv.data(),
                                              static_cast<std::size_t>(nb_s) * sizeof(int),
                                              cudaMemcpyHostToDevice, stream_.get()));

        DeviceBuffer<double> dX(m * static_cast<std::size_t>(nb_s));
        DeviceBuffer<double> dLoo(m * static_cast<std::size_t>(nb_s));
        DeviceBuffer<double> dTotal(m);
        DeviceBuffer<double> dTotLine(m);

        // S3 per-quartet gather (reads RESIDENT f2; native FP64 4-slab combine).
        launch_assemble_f4_quartets_gather(f2.f2_device(), P, dQuartets.data(), N, nb_s,
                                           dropped ? dSurv.data() : nullptr,
                                           dX.data(), stream_.get());

        long long n_ll = 0;
        for (int v : out.block_sizes) n_ll += v;
        const double n = static_cast<double>(n_ll);

        // est_to_loo + x_total + tot_line (REUSE the fit kernel; it reads only m = N).
        launch_f4_loo_total(dX.data(), dBlockSizes.data(),
                            static_cast<int>(m), nb_s, n,
                            dLoo.data(), dTotal.data(), dTotLine.data(), stream_.get());

        // D2H the small fit intermediates across the CUDA-free seam (SURVIVOR-sized).
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_blocks.data(), dX.data(),
                                          m * static_cast<std::size_t>(nb_s) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_loo.data(), dLoo.data(),
                                          m * static_cast<std::size_t>(nb_s) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_total.data(), dTotal.data(),
                                          m * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(tot_line_.data(), dTotLine.data(),
                                          m * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        return out;
    }

    /// FUSED f4ratio assemble→ratio-jackknife (M1 host-compute-audit cure) — the device-resident
    /// producer. SAME assemble as assemble_f4_quartets (the interleaved 2N-quartet f4 X: num
    /// rows [0,N), den rows [N,2N)), BUT keeps dX (x_blocks) and dLoo (x_loo) RESIDENT and feeds
    /// them straight to launch_ratio_block_jackknife — DROPPING the [m·nb_s] x_blocks + x_loo D2H
    /// pair the assemble_f4_quartets path does (cuda_backend.cu:2915-2920). Only the small N-length
    /// est/se/z cross back. tot_mode=0, weight=block_sizes broadcast, setmiss_thresh given,
    /// compute_p=false. Native FP64 (the §12 carve-out; the kernel matches the host long-double
    /// ratio_jackknife operand order). `flat` is the interleaved 2N-quartet flat array (length 8N).
    [[nodiscard]] RatioBlockJackknife f4ratio_blocks_jackknife(
        const steppe::device::DeviceF2Blocks& f2, std::span<const int> flat, int N,
        double setmiss_thresh, const Precision& precision) override {
        (void)precision;  // native FP64 (the cancellation-sensitive ratio diff)
        guard_device();
        RatioBlockJackknife jk;
        jk.N = N;
        const int nb = f2.n_block;
        const int P = f2.P;
        const std::size_t m = static_cast<std::size_t>(2) * static_cast<std::size_t>(N);  // 2N rows
        if (N <= 0 || nb <= 0 || f2.f2_device() == nullptr) {
            jk.est.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
            jk.se.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
            jk.z.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
            jk.status = Status::Ok;
            return jk;
        }

        // F1 / OQ-12 SURVIVOR drop (the SAME on-device keep-mask the assemble path uses).
        const std::vector<int> surv = device_survivor_blocks(f2, nb, P);
        const int nb_s = static_cast<int>(surv.size());
        if (nb_s <= 0) {
            jk.est.assign(static_cast<std::size_t>(N), std::nan(""));
            jk.se.assign(static_cast<std::size_t>(N), std::nan(""));
            jk.z.assign(static_cast<std::size_t>(N), std::nan(""));
            jk.status = Status::Ok;
            return jk;
        }
        std::vector<int> block_sizes(static_cast<std::size_t>(nb_s));
        std::vector<double> block_sizes_d(static_cast<std::size_t>(nb_s));
        for (int bs = 0; bs < nb_s; ++bs) {
            const int v = f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
            block_sizes[static_cast<std::size_t>(bs)] = v;
            block_sizes_d[static_cast<std::size_t>(bs)] = static_cast<double>(v);
        }
        const bool dropped = (nb_s != nb);

        // The interleaved num/den quartet flat array is the SAME `flat` (length 8N == 4*2N).
        const int M2 = 2 * N;  // the m-axis quartet count (num then den halves).
        DeviceBuffer<int> dQuartets(flat.size());
        DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb_s));
        DeviceBuffer<double> dBlockSizesD(static_cast<std::size_t>(nb_s));
        DeviceBuffer<int> dSurv(static_cast<std::size_t>(dropped ? nb_s : 1));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQuartets.data(), flat.data(),
                                          flat.size() * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), block_sizes.data(),
                                          static_cast<std::size_t>(nb_s) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizesD.data(), block_sizes_d.data(),
                                          static_cast<std::size_t>(nb_s) * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        if (dropped)
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSurv.data(), surv.data(),
                                              static_cast<std::size_t>(nb_s) * sizeof(int),
                                              cudaMemcpyHostToDevice, stream_.get()));

        DeviceBuffer<double> dX(m * static_cast<std::size_t>(nb_s));
        DeviceBuffer<double> dLoo(m * static_cast<std::size_t>(nb_s));
        DeviceBuffer<double> dTotal(m);
        DeviceBuffer<double> dTotLine(m);

        // S3 per-quartet gather (RESIDENT f2; native FP64 4-slab combine) + est_to_loo/total.
        launch_assemble_f4_quartets_gather(f2.f2_device(), P, dQuartets.data(), M2, nb_s,
                                           dropped ? dSurv.data() : nullptr,
                                           dX.data(), stream_.get());
        long long n_ll = 0;
        for (int v : block_sizes) n_ll += v;
        const double n = static_cast<double>(n_ll);
        launch_f4_loo_total(dX.data(), dBlockSizes.data(),
                            static_cast<int>(m), nb_s, n,
                            dLoo.data(), dTotal.data(), dTotLine.data(), stream_.get());

        // The SHARED ratio-block-jackknife over the RESIDENT dX/dLoo — NO [m·nb_s] D2H.
        // num/den ARE the est_to_loo replicates dLoo (rows k / N+k); xblk_num/xblk_den ARE the
        // per-block f4 dX (rows k / N+k); weight = block_sizes broadcast. dX/dLoo are
        // COLUMN-MAJOR in the m axis: element [row + m*b] ⇒ item_stride=1, block_stride=m.
        const long ms = static_cast<long>(m);
        DRatioJackArray dnum{dLoo.data(), 0, 1, ms};
        DRatioJackArray dden{dLoo.data(), static_cast<long>(N), 1, ms};
        DRatioJackArray dweight{dBlockSizesD.data(), 0, 0, 1};  // broadcast across items.
        DRatioJackArray dxbn{dX.data(), 0, 1, ms};
        DRatioJackArray dxbd{dX.data(), static_cast<long>(N), 1, ms};

        DeviceBuffer<double> dEst(static_cast<std::size_t>(N));
        DeviceBuffer<double> dSe(static_cast<std::size_t>(N));
        DeviceBuffer<double> dZ(static_cast<std::size_t>(N));
        launch_ratio_block_jackknife(dnum, dden, dweight, dxbn, dxbd, N, nb_s,
                                     /*tot_mode=*/0, setmiss_thresh, /*compute_p=*/false,
                                     dEst.data(), dSe.data(), dZ.data(), /*d_p=*/nullptr,
                                     stream_.get());

        jk.est.assign(static_cast<std::size_t>(N), 0.0);
        jk.se.assign(static_cast<std::size_t>(N), 0.0);
        jk.z.assign(static_cast<std::size_t>(N), 0.0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(jk.est.data(), dEst.data(),
                                          static_cast<std::size_t>(N) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(jk.se.data(), dSe.data(),
                                          static_cast<std::size_t>(N) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(jk.z.data(), dZ.data(),
                                          static_cast<std::size_t>(N) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        jk.status = Status::Ok;
        return jk;
    }

    /// FUSED dstat block-reduce→ratio-jackknife (M2 host-compute-audit cure) — the device-resident
    /// producer. Runs the per-(quadruple,block) D reduce keeping dNum/dDen/dCnt RESIDENT and feeds
    /// them straight to launch_ratio_block_jackknife — DROPPING the numsum/densum/cnt D2H the
    /// dstat_block_reduce path does (cuda_backend.cu:1445-1450). Only the small N-length
    /// est/se/z/p cross back. tot_mode=1, weight=cnt, setmiss_thresh=0 (cnt>0 mask), compute_p=true
    /// (p = f4_two_sided_p(z) on-device). Native FP64 (the §12 carve-out; matches the host
    /// long-double dstat_jackknife operand order).
    [[nodiscard]] RatioBlockJackknife dstat_blocks_jackknife(
        const steppe::device::DeviceDecodeResult& dec, const int* block_id, int n_block,
        std::span<const int> quadruples) override {
        guard_device();
        const int N = static_cast<int>(quadruples.size() / 4);
        RatioBlockJackknife jk;
        jk.N = N;
        if (dec.empty() || dec.q_device() == nullptr || N <= 0 || n_block <= 0) {
            jk.est.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
            jk.se.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
            jk.z.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
            jk.p.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
            jk.status = Status::Ok;
            return jk;
        }
        const double* dQ = dec.q_device();
        const double* dV = dec.v_device();
        const int P = dec.P;
        const long M = dec.M_kept;

        // Per-block contiguous SNP layout from block_id (the SAME inverse of assign_blocks the
        // dstat_block_reduce_device path uses).
        const std::vector<core::BlockRange> ranges =
            core::block_ranges(std::span<const int>(block_id, static_cast<std::size_t>(M)),
                               M, n_block);
        std::vector<int> begin(static_cast<std::size_t>(n_block));
        std::vector<int> size(static_cast<std::size_t>(n_block));
        for (int b = 0; b < n_block; ++b) {
            begin[static_cast<std::size_t>(b)] = static_cast<int>(ranges[static_cast<std::size_t>(b)].begin);
            size[static_cast<std::size_t>(b)]  = static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
        }
        const std::size_t nq = static_cast<std::size_t>(N) * 4u;
        const std::size_t nb_out = static_cast<std::size_t>(N) * static_cast<std::size_t>(n_block);

        DeviceBuffer<int> dQuad(nq);
        DeviceBuffer<int> dBegin(static_cast<std::size_t>(n_block));
        DeviceBuffer<int> dSize(static_cast<std::size_t>(n_block));
        DeviceBuffer<double> dNum(nb_out), dDen(nb_out), dCnt(nb_out);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQuad.data(), quadruples.data(), nq * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBegin.data(), begin.data(),
                                          static_cast<std::size_t>(n_block) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSize.data(), size.data(),
                                          static_cast<std::size_t>(n_block) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));

        launch_dstat_block_reduce(dQ, dV, P, M, dQuad.data(), N,
                                  dBegin.data(), dSize.data(), n_block,
                                  dNum.data(), dDen.data(), dCnt.data(), stream_.get());

        // The SHARED ratio-block-jackknife over the RESIDENT dNum/dDen/dCnt — NO [N·nb] D2H.
        // numsum/densum/cnt are ROW-MAJOR [N × n_block], element [k*nb+b] ⇒ item_stride=n_block,
        // block_stride=1. xblk pair is null (dstat tot_mode=1 never reads it).
        const long nbl = static_cast<long>(n_block);
        DRatioJackArray dnum{dNum.data(), 0, nbl, 1};
        DRatioJackArray dden{dDen.data(), 0, nbl, 1};
        DRatioJackArray dweight{dCnt.data(), 0, nbl, 1};
        DRatioJackArray dnull{nullptr, 0, 0, 0};

        DeviceBuffer<double> dEst(static_cast<std::size_t>(N));
        DeviceBuffer<double> dSe(static_cast<std::size_t>(N));
        DeviceBuffer<double> dZ(static_cast<std::size_t>(N));
        DeviceBuffer<double> dP(static_cast<std::size_t>(N));
        launch_ratio_block_jackknife(dnum, dden, dweight, dnull, dnull, N, n_block,
                                     /*tot_mode=*/1, /*setmiss_thresh=*/0.0, /*compute_p=*/true,
                                     dEst.data(), dSe.data(), dZ.data(), dP.data(),
                                     stream_.get());

        jk.est.assign(static_cast<std::size_t>(N), 0.0);
        jk.se.assign(static_cast<std::size_t>(N), 0.0);
        jk.z.assign(static_cast<std::size_t>(N), 0.0);
        jk.p.assign(static_cast<std::size_t>(N), 0.0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(jk.est.data(), dEst.data(),
                                          static_cast<std::size_t>(N) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(jk.se.data(), dSe.data(),
                                          static_cast<std::size_t>(N) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(jk.z.data(), dZ.data(),
                                          static_cast<std::size_t>(N) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(jk.p.data(), dP.data(),
                                          static_cast<std::size_t>(N) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        jk.status = Status::Ok;
        return jk;
    }

    /// FUSED f4ratio host-oracle overload — NOT the GPU path. The CudaBackend reads
    /// DEVICE-RESIDENT f2 (the DeviceF2Blocks form); the host-tensor overload is the CpuBackend
    /// oracle door (sibling of assemble_f4_quartets(host) throwing).
    [[nodiscard]] RatioBlockJackknife f4ratio_blocks_jackknife(
        const F2BlockTensor& f2, std::span<const int> flat, int N, double setmiss_thresh,
        const Precision& precision) override {
        (void)f2; (void)flat; (void)N; (void)setmiss_thresh; (void)precision;
        throw std::runtime_error(
            "CudaBackend::f4ratio_blocks_jackknife(host): the GPU path reads DEVICE-RESIDENT f2 "
            "(the DeviceF2Blocks overload); the host-tensor overload is the CpuBackend oracle door");
    }

    /// FUSED dstat host-pointer overload — NOT the GPU path. The CudaBackend reads the
    /// DEVICE-RESIDENT decode (the DeviceDecodeResult form, dropping the numsum/densum/cnt D2H);
    /// the host-pointer Q/V overload is the CpuBackend oracle door.
    [[nodiscard]] RatioBlockJackknife dstat_blocks_jackknife(
        const double* Q, const double* V, int P, long M, const int* block_id, int n_block,
        std::span<const int> quadruples) override {
        (void)Q; (void)V; (void)P; (void)M; (void)block_id; (void)n_block; (void)quadruples;
        throw std::runtime_error(
            "CudaBackend::dstat_blocks_jackknife(host): the GPU path reads the DEVICE-RESIDENT "
            "decode (the DeviceDecodeResult overload); the host-pointer overload is the CpuBackend "
            "oracle door");
    }

    /// STANDALONE f4 host-oracle overload — NOT the GPU path (sibling of assemble_f4(host));
    /// the CudaBackend implements only the DeviceF2Blocks form (zero D2H of the tensor).
    [[nodiscard]] F4Blocks assemble_f4_quartets(const F2BlockTensor& f2,
                                                std::span<const int> quartets,
                                                const Precision& precision) override {
        (void)f2; (void)quartets; (void)precision;
        throw std::runtime_error(
            "CudaBackend::assemble_f4_quartets(host): the GPU path reads DEVICE-RESIDENT "
            "f2 (assemble_f4_quartets(DeviceF2Blocks)); the host-tensor overload is the "
            "CpuBackend oracle door");
    }

    /// STANDALONE f3 triple assemble (run_f3 seam; fit-engine §6) — the THREE-slab clone of
    /// assemble_f4_quartets, the GPU mirror of the CpuBackend oracle. Build ONE F4Blocks
    /// whose m axis is the N input TRIPLES (nl=N, nr=1), reading the RESIDENT f2 (zero D2H
    /// of the tensor). Each triple column gathers its OWN three-slab combine via
    /// launch_assemble_f3_triples_gather; the est_to_loo / x_total / tot_line reduction
    /// REUSES launch_f4_loo_total (it reads only m), so the jackknife pipeline downstream is
    /// byte-identical to the f4/fit path. The SURVIVOR-block drop (F1/OQ-12) reuses
    /// device_survivor_blocks. Native FP64 (cancellation carve-out).
    [[nodiscard]] F4Blocks assemble_f3_triples(
        const steppe::device::DeviceF2Blocks& f2,
        std::span<const int> triples,
        const Precision& precision) override {
        (void)precision;  // native FP64 (the cancellation-sensitive f-stat diff)
        guard_device();

        const int N = static_cast<int>(triples.size()) / 3;  // triple count (m axis)
        const int nb = f2.n_block;
        const int P = f2.P;

        F4Blocks out;
        out.nl = N;  // m = nl*nr = N (the batched f3 m-axis convention; matches the oracle)
        out.nr = 1;
        const std::size_t m = static_cast<std::size_t>(N);
        out.x_total.assign(m, 0.0);
        tot_line_.assign(m, 0.0);
        if (N <= 0 || nb <= 0 || f2.f2_device() == nullptr) {
            out.n_block = 0;
            return out;
        }

        // F1 / OQ-12 SURVIVOR drop (the SAME on-device keep-mask the fit/f4 path uses).
        const std::vector<int> surv = device_survivor_blocks(f2, nb, P);
        const int nb_s = static_cast<int>(surv.size());
        out.n_block = nb_s;
        out.block_sizes.assign(static_cast<std::size_t>(nb_s), 0);
        for (int bs = 0; bs < nb_s; ++bs)
            out.block_sizes[static_cast<std::size_t>(bs)] =
                f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
        out.x_blocks.assign(m * static_cast<std::size_t>(nb_s), 0.0);
        out.x_loo.assign(m * static_cast<std::size_t>(nb_s), 0.0);
        if (nb_s <= 0) return out;  // all blocks missing — degenerate (caller-gated)

        const bool dropped = (nb_s != nb);

        // H2D the flattened triple array (3*N) + the SURVIVOR block sizes + (only on a real
        // drop) the survivor map.
        DeviceBuffer<int> dTriples(triples.size());
        DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb_s));
        DeviceBuffer<int> dSurv(static_cast<std::size_t>(dropped ? nb_s : 1));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dTriples.data(), triples.data(),
                                          triples.size() * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), out.block_sizes.data(),
                                          static_cast<std::size_t>(nb_s) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
        if (dropped)
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSurv.data(), surv.data(),
                                              static_cast<std::size_t>(nb_s) * sizeof(int),
                                              cudaMemcpyHostToDevice, stream_.get()));

        DeviceBuffer<double> dX(m * static_cast<std::size_t>(nb_s));
        DeviceBuffer<double> dLoo(m * static_cast<std::size_t>(nb_s));
        DeviceBuffer<double> dTotal(m);
        DeviceBuffer<double> dTotLine(m);

        // S3 per-triple gather (reads RESIDENT f2; native FP64 3-slab combine).
        launch_assemble_f3_triples_gather(f2.f2_device(), P, dTriples.data(), N, nb_s,
                                          dropped ? dSurv.data() : nullptr,
                                          dX.data(), stream_.get());

        long long n_ll = 0;
        for (int v : out.block_sizes) n_ll += v;
        const double n = static_cast<double>(n_ll);

        // est_to_loo + x_total + tot_line (REUSE the fit/f4 kernel; it reads only m = N).
        launch_f4_loo_total(dX.data(), dBlockSizes.data(),
                            static_cast<int>(m), nb_s, n,
                            dLoo.data(), dTotal.data(), dTotLine.data(), stream_.get());

        // D2H the small fit intermediates across the CUDA-free seam (SURVIVOR-sized).
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_blocks.data(), dX.data(),
                                          m * static_cast<std::size_t>(nb_s) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_loo.data(), dLoo.data(),
                                          m * static_cast<std::size_t>(nb_s) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_total.data(), dTotal.data(),
                                          m * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(tot_line_.data(), dTotLine.data(),
                                          m * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        return out;
    }

    /// STANDALONE f3 host-oracle overload — NOT the GPU path (sibling of
    /// assemble_f4_quartets(host)); the CudaBackend implements only the DeviceF2Blocks form.
    [[nodiscard]] F4Blocks assemble_f3_triples(const F2BlockTensor& f2,
                                               std::span<const int> triples,
                                               const Precision& precision) override {
        (void)f2; (void)triples; (void)precision;
        throw std::runtime_error(
            "CudaBackend::assemble_f3_triples(host): the GPU path reads DEVICE-RESIDENT "
            "f2 (assemble_f3_triples(DeviceF2Blocks)); the host-tensor overload is the "
            "CpuBackend oracle door");
    }

    /// qpGraph FLEET — the PRODUCTION GPU path (the productized IDEA-1 optimizer spike).
    /// The resident f3 basis (f_obs[npair] + qinv[npair*npair], assembled by the core
    /// driver via assemble_f3_triples + jackknife_cov) + the topology arenas upload to
    /// VRAM; the fleet kernel runs `numstart` restarts, ONE thread each, the WHOLE
    /// multistart x maxit projected-Newton loop ON-DEVICE (GPU-BOUND: ONE launch, NO host
    /// objective per iteration — the AT2 optim() host-loop trap designed out). Only the
    /// per-restart {theta, score} come back; the best-of-restarts + the bracket + the final
    /// edge-length recovery (ONE host eval at the winning theta) are tiny host work. The
    /// inner SPD edge solve + the GLS form are native FP64 (the cancellation carve-out).
    [[nodiscard]] QpGraphFleet qpgraph_fit_fleet(const QpGraphTopoArena& topo,
                                                 std::span<const double> f_obs,
                                                 std::span<const double> qinv,
                                                 int numstart, int maxit, double tol,
                                                 const Precision& precision) override {
        (void)precision;  // the in-thread objective is native FP64 (the carve-out); the
                          // emulated GEMM seam is the production batched-cc path.
        guard_device();
        QpGraphFleet out;
        const int D = topo.nadmix;
        const int npair = topo.npair, ne = topo.nedge_norm;

        // ---- upload the resident basis + the topology arenas ----------------------
        DeviceBuffer<double> dFobs(static_cast<std::size_t>(npair));
        DeviceBuffer<double> dQinv(static_cast<std::size_t>(npair) * static_cast<std::size_t>(npair));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dFobs.data(), f_obs.data(), f_obs.size() * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQinv.data(), qinv.data(), qinv.size() * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        DeviceBuffer<double> dPwts0(topo.pwts0.size());
        DeviceBuffer<int> dPeEdge(topo.pe_edge.size() ? topo.pe_edge.size() : 1);
        DeviceBuffer<int> dPeLeaf(topo.pe_leaf.size() ? topo.pe_leaf.size() : 1);
        DeviceBuffer<int> dPePath(topo.pe_path.size() ? topo.pe_path.size() : 1);
        DeviceBuffer<int> dPaePath(topo.pae_path.size() ? topo.pae_path.size() : 1);
        DeviceBuffer<int> dPaeEdge(topo.pae_admixedge.size() ? topo.pae_admixedge.size() : 1);
        DeviceBuffer<int> dCmb1(topo.cmb1.size());
        DeviceBuffer<int> dCmb2(topo.cmb2.size());
        auto up_d = [&](DeviceBuffer<double>& d, const std::vector<double>& h) {
            if (!h.empty()) STEPPE_CUDA_CHECK(cudaMemcpyAsync(d.data(), h.data(), h.size() * sizeof(double),
                                                              cudaMemcpyHostToDevice, stream_.get()));
        };
        auto up_i = [&](DeviceBuffer<int>& d, const std::vector<int>& h) {
            if (!h.empty()) STEPPE_CUDA_CHECK(cudaMemcpyAsync(d.data(), h.data(), h.size() * sizeof(int),
                                                              cudaMemcpyHostToDevice, stream_.get()));
        };
        up_d(dPwts0, topo.pwts0);
        up_i(dPeEdge, topo.pe_edge); up_i(dPeLeaf, topo.pe_leaf); up_i(dPePath, topo.pe_path);
        up_i(dPaePath, topo.pae_path); up_i(dPaeEdge, topo.pae_admixedge);
        up_i(dCmb1, topo.cmb1); up_i(dCmb2, topo.cmb2);

        cuda::QpGraphDeviceTopo dt{};
        dt.npop = topo.npop; dt.nedge_norm = ne; dt.nadmix = D; dt.npair = npair;
        dt.npath = topo.npath; dt.base_leaf = topo.base_leaf;
        dt.n_pe = static_cast<int>(topo.pe_edge.size());
        dt.n_pae = static_cast<int>(topo.pae_path.size());
        dt.constrained = topo.constrained ? 1 : 0;
        dt.fudge = topo.fudge;
        dt.pwts0 = dPwts0.data();
        dt.pe_edge = dPeEdge.data(); dt.pe_leaf = dPeLeaf.data(); dt.pe_path = dPePath.data();
        dt.pae_path = dPaePath.data(); dt.pae_admixedge = dPaeEdge.data();
        dt.cmb1 = dCmb1.data(); dt.cmb2 = dCmb2.data();

        const int ns = numstart > 0 ? numstart : 1;
        DeviceBuffer<double> dTheta(static_cast<std::size_t>(ns) * static_cast<std::size_t>(D > 0 ? D : 1));
        DeviceBuffer<double> dScore(static_cast<std::size_t>(ns));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));  // arenas ready

        if (D == 0) {
            // pure tree: no fleet — one host edge solve (the GPU fleet adds nothing at D=0).
            core::qpadm::QpGraphModel m = topo_to_model(topo);
            std::vector<double> bl, fit(static_cast<std::size_t>(npair), 0.0);
            const double sc = core::qpadm::qpgraph_score(m, nullptr,
                std::vector<double>(f_obs.begin(), f_obs.end()),
                std::vector<double>(qinv.begin(), qinv.end()), topo.fudge, topo.constrained, &bl, &fit);
            out.score = sc; out.edge_length = bl; out.f3_fit = fit;
            out.status = std::isfinite(sc) ? Status::Ok : Status::NonSpdCovariance;
            return out;
        }

        // ---- the on-device fleet (ONE launch; whole multistart x maxit in-kernel) ---
        cuda::launch_qpgraph_fleet(dt, ns, maxit, tol, dFobs.data(), dQinv.data(),
                                   dTheta.data(), dScore.data(), stream_.get());

        std::vector<double> h_theta(static_cast<std::size_t>(ns) * static_cast<std::size_t>(D));
        std::vector<double> h_score(static_cast<std::size_t>(ns));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_theta.data(), dTheta.data(), h_theta.size() * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_score.data(), dScore.data(), h_score.size() * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

        // best-of-restarts + the per-weight bracket (tiny host work over ns rows).
        double best = std::numeric_limits<double>::infinity();
        double smin = best, smax = -best;
        int best_i = 0;
        std::vector<double> thmin(static_cast<std::size_t>(D), 1.0), thmax(static_cast<std::size_t>(D), 0.0);
        for (int i = 0; i < ns; ++i) {
            const double s = h_score[static_cast<std::size_t>(i)];
            if (std::isfinite(s)) { if (s < smin) smin = s; if (s > smax) smax = s; }
            for (int d = 0; d < D; ++d) {
                const double v = h_theta[static_cast<std::size_t>(i) * static_cast<std::size_t>(D) + static_cast<std::size_t>(d)];
                if (v < thmin[static_cast<std::size_t>(d)]) thmin[static_cast<std::size_t>(d)] = v;
                if (v > thmax[static_cast<std::size_t>(d)]) thmax[static_cast<std::size_t>(d)] = v;
            }
            if (s < best) { best = s; best_i = i; }
        }
        out.score = best;
        out.restart_spread = (std::isfinite(smax) && std::isfinite(smin)) ? (smax - smin) : 0.0;
        out.theta.assign(static_cast<std::size_t>(D), 0.0);
        for (int d = 0; d < D; ++d)
            out.theta[static_cast<std::size_t>(d)] = h_theta[static_cast<std::size_t>(best_i) * static_cast<std::size_t>(D) + static_cast<std::size_t>(d)];
        out.theta_lo = thmin; out.theta_hi = thmax;

        // recover the edge lengths + f3_fit at the best theta (ONE host eval; the same
        // native FP64 objective the kernel runs — a localization the test diffs to 1e-6).
        core::qpadm::QpGraphModel m = topo_to_model(topo);
        std::vector<double> bl, fit(static_cast<std::size_t>(npair), 0.0);
        const double sc_final = core::qpadm::qpgraph_score(m, out.theta.data(),
            std::vector<double>(f_obs.begin(), f_obs.end()),
            std::vector<double>(qinv.begin(), qinv.end()), topo.fudge, topo.constrained, &bl, &fit);
        if (std::isfinite(sc_final)) { out.edge_length = bl; out.f3_fit = fit; out.status = Status::Ok; }
        else { out.edge_length.assign(static_cast<std::size_t>(ne), 0.0); out.f3_fit.assign(static_cast<std::size_t>(npair), 0.0); out.status = Status::NonSpdCovariance; }
        return out;
    }

    /// GPU-ONLY f4 sweep — enumerate EVERY C(P,4) quartet on the device, compute f4 + diagonal
    /// jackknife SE + |z|, filter + CUB-compact survivors ON THE DEVICE, return ONLY survivors.
    /// k=4; delegates to the shared run_fstat_sweep_device. (The fix for the CPU-bound disaster.)
    [[nodiscard]] SweepSurvivors f4_sweep(const steppe::device::DeviceF2Blocks& f2,
                                          const SweepConfig& cfg,
                                          const Precision& precision) override {
        return run_fstat_sweep_device(f2, cfg, precision, /*k=*/4);
    }

    /// GPU-ONLY f3 sweep — the three-slab sibling (every C(P,3) triple). k=3.
    [[nodiscard]] SweepSurvivors f3_sweep(const steppe::device::DeviceF2Blocks& f2,
                                          const SweepConfig& cfg,
                                          const Precision& precision) override {
        return run_fstat_sweep_device(f2, cfg, precision, /*k=*/3);
    }

    /// S4 — weighted block-jackknife covariance Q + Qinv on the GPU (the FROZEN
    /// CONTRACT §2b). xtau kernel (native; cancellation carve-out) → cublasDsyrk Q
    /// (well-conditioned matmul; ENGAGES `precision` — emulated{40} by default, auto-
    /// native fallback) → fudge diag → cusolverDn Cholesky potrf/potri Qinv (ill-
    /// conditioned; native FP64, the cuSOLVER fallback). devInfo>0 ⇒
    /// NonSpdCovariance (value, not throw).
    [[nodiscard]] JackknifeCov jackknife_cov(const F4Blocks& x,
                                             std::span<const int> block_sizes,
                                             double fudge,
                                             const Precision& precision) override {
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
        DeviceBuffer<double> dWork(static_cast<std::size_t>(lwork_f > 0 ? lwork_f : 1));
        CUSOLVER_CHECK(cusolverDnDpotrf(solver_.get(), CUBLAS_FILL_MODE_LOWER, m,
                                        dQf.data(), m, dWork.data(), lwork_f, dInfo.data()));
        int info = 0;
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(&info, dInfo.data(), sizeof(int),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        if (info != 0) {  // not SPD / singular pivot → domain outcome
            out.status = Status::NonSpdCovariance;
            return out;
        }
        CUSOLVER_CHECK(cusolverDnDpotri(solver_.get(), CUBLAS_FILL_MODE_LOWER, m,
                                        dQf.data(), m, dWork.data(), lwork_f, dInfo.data()));
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

    /// S4 DIAGONAL-only jackknife variance on the GPU (the per-item f-stat SE production
    /// shape; the OOM fix for the sweep). REUSES the EXACT xtau seam of jackknife_cov (the
    /// SAME launch_f4_xtau over the SAME loo/est/tot_line/block_sizes — native carve-out),
    /// then reduces var[k] = (1/nb)·Σ_b xtau[k,b]² with launch_f4_diag_var. NO dense m×m Q,
    /// NO cublasDsyrk, NO potrf/potri ⇒ O(m·nb) work, O(m) memory (no N²/OOM at sweep scale).
    /// var[k] is the IDENTICAL FP64 arithmetic the deleted dense diagonal Q[k+m*k] computed,
    /// so f4/f3 re-pass the existing goldens BY CONSTRUCTION. Native FP64 (this IS the
    /// diagonal jackknife_cov computes; `precision` acknowledged, not consumed — the xtau
    /// centering + sum-of-squares are the cancellation/well-conditioned carve-outs).
    [[nodiscard]] JackknifeDiag jackknife_diag(const F4Blocks& x,
                                               std::span<const int> block_sizes,
                                               const Precision& precision) override {
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

    // =====================================================================
    // LARGE-path helpers (the FROZEN CONTRACT §1/§2): arbitrary nl/nr via cuSOLVER
    // SVD + dynamic VRAM scratch. Used when a model exceeds the bit-parity small
    // envelope (model_fits_small_path == false, e.g. NRBIG nr=39). The small path is
    // UNTOUCHED (the 9-pop golden takes the on-device Jacobi). Native FP64 (§4: the
    // SVD + Qinv quadratic form are ill-conditioned/oracle-grade; no emulation).
    // =====================================================================

    /// True iff the model fits the on-device small-LA bit-parity envelope
    /// (kQpMaxNl/Nr/R). Inside ⇒ the untouched small path (byte-for-byte 9-pop
    /// golden parity); outside ⇒ the cuSOLVER large path. Delegates to the SINGLE
    /// SOURCE (core/qpadm/qpadm_bounds.hpp) that ALSO sizes the kernel per-thread
    /// arrays and gates the host core partition (model_search.cpp) — so this
    /// dispatch gate cannot drift wider than the kernel arrays it routes into.
    static bool model_fits_small_path(int nl, int nr, int r) {
        return core::qpadm::model_fits_small_path(nl, nr, r);
    }

    /// True iff the LARGE-path dense SVD routes to cuSOLVER's one-sided Jacobi
    /// (gesvdj) rather than gesvd: BOTH dims <= kGesvdjMaxDim (32). The SINGLE source
    /// of the dispatch predicate so large_svd_V's branch AND the svd_path
    /// observability report (rank_sweep) cannot disagree on the executed routine — a
    /// drift the NRBIG parity test (svd_path==2) would otherwise silently mask
    /// (group-5 5.3/5.5). svd_path = gesvdj_applicable ? 1 : 2.
    static bool gesvdj_applicable(int nl, int nr) {
        return nl <= kGesvdjMaxDim && nr <= kGesvdjMaxDim;
    }

    /// The cuSOLVER gesvd/gesvdj DEVICE scratch a single `large_svd_V` call consumes,
    /// hoistable to ONE arena reused across many same-shape SVDs (the §14.2 fix —
    /// see `large_svd_V` and the Stage-A LOO loop). Element COUNTS (not bytes), sized
    /// to whichever orientation+routine branch the fixed (nl,nr) selects: only one
    /// branch ever runs, so each field is its branch's exact need (no over-allocation):
    ///   * dS   = min(nl,nr)        (singular values, cols of the rows>=cols matrix)
    ///   * dU   = nl*nr             (economy U, rows×cols == nl*nr both orientations)
    ///   * dVt  = nr>=nl ? nl*nl : nr*nr   (right vectors, cols×cols; only the gesvdj
    ///                                       branch reads it for nr>=nl)
    ///   * dA2  = nl>nr ? nl*nr : 0 (non-const copy of A; gesvd/gesvdj OVERWRITE A and
    ///                               the const dXmat must survive — needed ONLY when
    ///                               nl>nr, where A=dXmat is handed in directly)
    ///   * dInfo= 1
    ///   * lwork= cuSOLVER's bufferSize for the selected routine (depends ONLY on the
    ///            dims/params, NOT on the matrix values — VERIFIED against the CUDA
    ///            13.x cuSOLVER docs "xyz_bufferSize ... only depends on some
    ///            parameters ... device pointer is not used to decide the size of
    ///            workspace" — so it is queried ONCE per shape and reused).
    struct SvdScratchSizes {
        std::size_t s = 0, u = 0, vt = 0, a2 = 0, info = 1;
        int lwork = 0;
    };

    /// Size (and query lwork for) the gesvd/gesvdj scratch of an nl×nr, leading-r SVD.
    /// Caller allocates ONE DeviceBuffer per field at these counts, then feeds the
    /// slices to the scratch-taking `large_svd_V` overload — so a sweep of many
    /// same-shape SVDs allocates/frees the scratch ONCE, not per call (each
    /// `DeviceBuffer` free is a device-wide `cudaFree` sync; per-call frees serialize
    /// the otherwise-async stream — the §14.2 wall).
    [[nodiscard]] SvdScratchSizes large_svd_scratch_sizes(int nl, int nr) {
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

    /// Compute the leading-r right singular vectors V[:,0:r] (nr×r, col-major) of the
    /// nl×nr col-major `dXmat` via cuSOLVER, native FP64, deterministic, on stream_.
    /// Writes dVout[nr*r] col-major (descending singular value order). The §1.3
    /// orientation rule (cuSOLVER gesvd is deterministic only for rows>=cols):
    ///   * common large case nr>=nl: hand Xt = transpose(xmat) (nr×nl, rows>=cols) to
    ///     cuSOLVER; then U(Xt) == V(xmat) ⇒ read the leading r cols of U into dVout.
    ///   * rare nl>nr: hand xmat itself (nl×nr, rows>=cols); V(xmat) = VT's leading r
    ///     rows ⇒ transpose them into dVout.
    /// Routine (§1.4): gesvdj (one-sided Jacobi) when gesvdj_applicable (both dims
    /// <= kGesvdjMaxDim); gesvd otherwise. The dispatch + the svd_path report share
    /// that ONE predicate so they cannot drift.
    /// `dXt` is the nr×nl transpose scratch used in BOTH branches (Xt = transpose(xmat)
    /// for nr>=nl, V = transpose(dVt) for nl>nr). The math-mode is native (no
    /// CusolverMathModeScope for emulation — §1.5/§4).
    ///
    /// SCRATCH-TAKING overload (the §14.2 hoist): the gesvd scratch (dS/dU/dVt/dA2/
    /// dInfo/dWork) is CALLER-OWNED — slices of arenas the caller allocated ONCE at
    /// `large_svd_scratch_sizes` counts and reuses across many same-shape SVDs. The
    /// math (cuSOLVER gesvd/gesvdj, native FP64) is BIT-IDENTICAL to the
    /// allocate-per-call form; only the scratch LOCATION moves out of the call, so a
    /// per-block sweep no longer pays nb× device-wide `cudaFree` syncs. The `dWork`
    /// arena must be `lwork` (from the same `large_svd_scratch_sizes`) doubles.
    /// `sInfo` is 1 int. `sA2` is read ONLY when nl>nr (may be nullptr for nr>=nl).
    void large_svd_V(const double* dXmat, int nl, int nr, int r,
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

    /// Single-shot convenience overload: allocate the gesvd scratch ONCE on the stack
    /// (RAII DeviceBuffers, freed at scope exit) and delegate to the scratch-taking
    /// form above. For the per-CALL large fit (`large_fit_one`) this allocate-once is
    /// within design (one small-LA burst per fit, not a per-block inner loop — see the
    /// §14.2 finding's "OTHER hot paths" note); the Stage-A LOO SWEEP must NOT use this
    /// overload — it hoists the scratch above its loop and calls the scratch-taking
    /// form, so it does not pay a device-wide `cudaFree` per block.
    void large_svd_V(const double* dXmat, int nl, int nr, int r,
                     double* dVout, double* dXt, cudaStream_t stream) {
        if (r <= 0) return;
        const SvdScratchSizes sz = large_svd_scratch_sizes(nl, nr);
        DeviceBuffer<double> dS(sz.s);
        DeviceBuffer<double> dU(sz.u);
        DeviceBuffer<double> dVt(sz.vt);
        DeviceBuffer<double> dA2(sz.a2);          // 0 ⇒ no alloc (nr>=nl); else nl*nr
        // dInfo: REQUIRED non-null device out-arg for cusolverDnDgesvd/Dgesvdj (info<0 =
        // bad param; gesvdj info>0 = non-converged Jacobi). INTENTIONAL DISCARD on this
        // off-bit-parity large (NRBIG) path — not D2H-copied/checked; a non-converged SVD
        // would flow into seed/ALS. Cannot drop the arg. (Cholesky dInfo IS checked; [3.4])
        DeviceBuffer<int>    dInfo(sz.info);
        DeviceBuffer<double> dWork(static_cast<std::size_t>(sz.lwork));
        large_svd_V(dXmat, nl, nr, r, dVout, dXt,
                    dS.data(), dU.data(), dVt.data(), dA2.data(),
                    dInfo.data(), dWork.data(), sz.lwork, stream);
    }

    /// Dynamic VRAM scratch sizes for the large-path ALS + weight/chisq (§2.3).
    /// dbl: the union of the ALS layout (xvec[m] | Wm[m*t] | coeffs[t*t] | rhs[t] |
    /// tmp[t] | lu[t*t] | y[t], t = max(nl,nr)*r) and the weight/chisq layout
    /// (RHS[nl*nl] | wv[nl] | lu[nl*nl] | y[nl] | e[m]); take the max so ONE buffer
    /// serves both kernels. int: max(t, nl) pivots.
    static std::size_t large_dbl_scratch(int nl, int nr, int r) {
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
    static std::size_t large_int_scratch(int nl, int nr, int r) {
        const std::size_t t = static_cast<std::size_t>(nl > nr ? nl : nr) *
                              static_cast<std::size_t>(r > 0 ? r : 1);
        return (t > static_cast<std::size_t>(nl) ? t : static_cast<std::size_t>(nl));
    }

    /// Per-refit DOUBLE scratch stride for the PARALLEL large-LOO kernel. Each thread
    /// owns its xmat[m] + A[nl*r] + B[r*nr] + the als/weights union (large_dbl_scratch),
    /// so one thread's slice is self-contained (no cross-thread aliasing ⇒ deterministic).
    static std::size_t large_loo_dbl_refit(int nl, int nr, int r) {
        const std::size_t m  = static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr);
        const std::size_t rr = static_cast<std::size_t>(r > 0 ? r : 1);
        const std::size_t a  = static_cast<std::size_t>(nl) * rr;          // A[nl*r]
        const std::size_t bb = rr * static_cast<std::size_t>(nr);          // B[r*nr]
        return m + a + bb + large_dbl_scratch(nl, nr, r);
    }

    /// LARGE-path single-model fit: cuSOLVER SVD seed (r>0) → ALS opt_A/opt_B → the
    /// constrained weight solve + chisq, all RESIDENT on stream_. The caller supplies
    /// the device buffers (dXmat/dQinv/dA/dB/dW/dchisq/dStatus + the VRAM scratch
    /// dVout/dXt/dScratch/dIntScratch). Mirrors the small-path seed→als→weights triple.
    void large_fit_one(const double* dXmat, const double* dQinv, int nl, int nr, int r,
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

    /// S5 — rank test / SVD seed + chisq on the GPU (the FROZEN CONTRACT §2c).
    /// xmat from x_total, on-device Jacobi SVD seed, chisq quadratic form. Native FP64.
    [[nodiscard]] GlsWeights rank_test(const F4Blocks& x,
                                       const JackknifeCov& cov,
                                       int r,
                                       const Precision& precision) override {
        (void)precision;
        guard_device();
        GlsWeights gw;
        gw.r = r;
        const int nl = x.nl, nr = x.nr, m = nl * nr;
        const bool small = model_fits_small_path(nl, nr, r);
        gw.A.assign(static_cast<std::size_t>(nl) * static_cast<std::size_t>(r), 0.0);
        gw.B.assign(static_cast<std::size_t>(r) * static_cast<std::size_t>(nr), 0.0);
        if (m <= 0) { gw.status = Status::Ok; return gw; }

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
            launch_qpadm_seed_ab(dXmat.data(), nl, nr, r, dA.data(), dB.data(), stream_.get());
            // chisq with the SEED factors (rank_test is the seed; design §4 S5).
            launch_qpadm_weights_chisq(dXmat.data(), dQinv.data(), dA.data(), dB.data(),
                                       nl, nr, r, dW.data(), dchisq.data(), dStatus.data(),
                                       stream_.get());
        } else {  // LARGE path: cuSOLVER SVD seed + VRAM-scratch weight/chisq (no ALS).
            DeviceBuffer<double> dVout(static_cast<std::size_t>(nr) * static_cast<std::size_t>(r > 0 ? r : 1));
            DeviceBuffer<double> dXt(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nl > 0 ? nl : 1));
            DeviceBuffer<double> dScratch(large_dbl_scratch(nl, nr, r));
            DeviceBuffer<int>    dIntScratch(large_int_scratch(nl, nr, r));
            if (r > 0) {
                large_svd_V(dXmat.data(), nl, nr, r, dVout.data(), dXt.data(), stream_.get());
                launch_qpadm_seed_from_V(dXmat.data(), dVout.data(), nl, nr, r,
                                         dA.data(), dB.data(), stream_.get());
            }
            launch_qpadm_weights_chisq_large(dXmat.data(), dQinv.data(), dA.data(), dB.data(),
                                             nl, nr, r, dW.data(), dchisq.data(), dStatus.data(),
                                             dScratch.data(), dIntScratch.data(), stream_.get());
        }
        double chisq = 0.0;
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(&chisq, dchisq.data(), sizeof(double),
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
        gw.status = Status::Ok;
        return gw;
    }

    /// S5 SWEEP — the qpWave / qpAdm RANK TEST over r = 0..rmax (rmax = min(nl,nr)-1)
    /// ON THE GPU (M(fit-2), THE deliverable). Mirrors the CpuBackend oracle
    /// (cpu_backend.cpp rank_sweep) op-for-op but runs the per-rank fit on the device:
    /// upload x_total + Qinv ONCE; build dXmat ONCE; then for each r run the SAME
    /// M(fit-4) device machinery (seed → ALS → weights+chisq) and read back chisq(r).
    /// The host then forms dof(r) = (nl-r)*(nr-r), p(r) = pchisq_upper, the AT2
    /// res$rankdrop nested table (rows f4rank DESCENDING; the nested diff to the
    /// next-lower rank; the last row NA), f4rank (the smallest non-rejected rank), and
    /// rank_Q (numerical rank of Q via core::jacobi_svd — observability only, computed
    /// host-side bit-identically to the oracle; NOT on the rank-test math path). The
    /// per-rank SVD is recomputed each r. Native FP64 throughout (the §4 carve-out:
    /// SVD + the Qinv quadratic form are ill-conditioned/cuSOLVER ⇒ native; precision
    /// is ignored here, exactly like the oracle). DISPATCH (the FROZEN CONTRACT §3.2,
    /// now the TRUE EXECUTED path): a model inside the bit-parity envelope (nl<=5,
    /// nr<=10, r<=4) runs the on-device Jacobi small path UNCHANGED; a model outside it
    /// (e.g. NRBIG nr=39) runs the cuSOLVER LARGE path (gesvdj when gesvdj_applicable,
    /// i.e. both dims <= kGesvdjMaxDim; gesvd otherwise) + VRAM scratch. svd_path
    /// REPORTS the executed routine via the SAME gesvdj_applicable predicate the
    /// dispatch uses: 1 = gesvdj / Jacobi (both dims <= kGesvdjMaxDim); 2 = gesvd. The
    /// test asserts svd_path==2 for NRBIG
    /// and ==1 for the 9-pop. NRBIG now GATES on the GPU (no longer a PENDING seam).
    [[nodiscard]] RankSweep rank_sweep(const F4Blocks& x,
                                       const JackknifeCov& cov,
                                       double alpha,
                                       const QpAdmOptions& opts,
                                       const Precision& precision) override {
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
        // gated; it is a derived property of Q, off the rank-test math path). Computed
        // host-side via core::jacobi_svd — BIT-IDENTICAL to the CpuBackend oracle, so
        // the GPU rank_Q == CPU rank_Q exactly (the localizer holds).
        if (!cov.Q.empty() && cov.m == m) {
            const core::SvdResult sq = core::jacobi_svd(cov.Q, m, m);
            const double smax = sq.S.empty() ? 0.0 : sq.S.front();
            const double tol = smax * static_cast<double>(m) *
                               std::numeric_limits<double>::epsilon();
            int rk = 0;
            for (double s : sq.S) if (s > tol) ++rk;
            rs.rank_Q = rk;
        } else {
            rs.rank_Q = m;
        }

        rs.status = degenerate ? Status::RankDeficient
                               : (cov.status != Status::Ok ? cov.status : Status::Ok);
        return rs;
    }

    /// S6 — GLS weights via AT2 ALS on the GPU (the FROZEN CONTRACT §2d). xmat from
    /// x_total → on-device SVD seed → opts.als_iterations (default 20) ALS opt_A/opt_B
    /// iters → constrained weight
    /// solve → normalize Σw=1 → chisq. All on-device, native FP64.
    [[nodiscard]] GlsWeights gls_weights(const F4Blocks& x,
                                         const JackknifeCov& cov,
                                         int r,
                                         const QpAdmOptions& opts,
                                         const Precision& precision) override {
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

    /// S7 — BATCHED leave-one-block-out weight re-fits on the GPU (the FROZEN
    /// CONTRACT §2e). Upload x_loo + Qinv ONCE; one batched device launch runs all
    /// nb per-block fits (xmat from loo[:,:,b] → seed → ALS → weight solve →
    /// normalize), REUSING Qinv unchanged (the AT2 parity pin). Returns wmat
    /// [nb*nl] row-major. This replaces the 708 host gls_weights calls with a single
    /// batched device kernel (NOT a host loop). Native FP64.
    [[nodiscard]] std::vector<double> gls_weights_loo_batched(
        const F4Blocks& x, const JackknifeCov& cov, int r,
        const QpAdmOptions& opts, const Precision& precision) override {
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

    /// S7 — the on-device leave-one-block-out SE reduction (M7 — the LAST host-compute
    /// move). Keeps the resident dWmat that populate_loo_wmat_resident produces and runs
    /// the EXISTING qpadm_se_from_wmat kernel (n_models=1, native FP64 — the §1.4
    /// cancellation carve-out; emulated/Ozaki is matmul-only and would not help a
    /// cancellation-sensitive reduction) — NO dWmat D2H, NO host long-double reduction.
    /// The single-model dWmat is UNSCALED (qpadm_fit_kernels.cu:1292 small / loo_large
    /// :1216 large); SE is LINEAR in the wmat scale (var(s·w)=s²·var(w) ⇒
    /// sqrt(var(s·w)/(nb-1)) = s·sqrt(var(w)/(nb-1)) EXACTLY), so the AT2
    /// s=(nb-1)/sqrt(nb) factor is reintroduced as a final multiply on the nl-length
    /// OUTPUT se (algebraically exact; the variance reduction itself stays FULLY on the
    /// device — only nl scalars are scaled host-side, NO dWmat bounce, NO new kernel).
    /// Returns the nl-length SCALED se. Native FP64.
    [[nodiscard]] std::vector<double> se_from_wmat(
        const F4Blocks& x, const JackknifeCov& cov, int r,
        const QpAdmOptions& opts, const Precision& precision) override {
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

private:
    /// S7 PRODUCER — fill the resident dWmat[nb*nl] (row-major b*nl+i) with the UNSCALED
    /// per-block leave-one-out weights, REUSING cov.Qinv unchanged (the AT2 parity pin).
    /// Uploads x_loo + Qinv ONCE; the small path runs launch_qpadm_loo_batched, the large
    /// (NRBIG) path the two-stage cuSOLVER-gesvd-seed + parallel loo_large kernel. This is
    /// the body that used to live inline in gls_weights_loo_batched — carved out (M7) so
    /// se_from_wmat consumes the SAME resident dWmat WITHOUT a D2H. Enqueues on stream_;
    /// the caller syncs. Native FP64.
    void populate_loo_wmat_resident(const F4Blocks& x, const JackknifeCov& cov, int r,
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

public:
    /// S8 — the BATCHED MODEL-SPACE ROTATION on the GPU (the M(fit-6) deliverable; the
    /// FROZEN CONTRACT §2.2). Fits a BUCKET of same-shape SMALL-path models (nl<=5,
    /// nr<=10, r<=4 — the rotation common case) in ONE batched dispatch:
    ///   - S3 f4 gather + loo/total + xtau over a (k,b,MODEL) grid reading the resident
    ///     f2 with per-model d_left/d_right index arenas (zero D2H of the tensor);
    ///   - the covariance Q = xtau·xtauᵀ/nb via cublasDgemmStridedBatched (ENGAGES the
    ///     passed `precision` — emulated{40} default, auto-native fallback — exactly the
    ///     jackknife_cov SYRK policy, now strided-batched across the model axis);
    ///   - the SPD inverse Qinv across models via cuSOLVER potrfBatched + potrsBatched
    ///     (vs a batched identity RHS); per-model devInfo>0 ⇒ that result's status =
    ///     NonSpdCovariance (record-and-continue, NOT a throw);
    ///   - the rank sweep + constrained weight solve + chisq + LOO-SE + popdrop via a
    ///     MODEL-batched kernel (one thread per model; the proven loo_batched_kernel
    ///     lift), filling weights/chisq/se/z/p/f4rank/rankdrop*/popdrop*/feasible per
    ///     model.
    /// The host then assembles the QpAdmResult fields (pchisq tail-p, the AT2 rankdrop
    /// nested table, popdrop pattern strings + feasibility) EXACTLY as run_impl/ranktest
    /// do — same math, batched. This is NOT a per-model host loop and NOT the
    /// CpuBackend: the per-bucket SVD seed/ALS/weight/chisq all run in one launch across
    /// the model axis, the covariance + inverse in batched cuSOLVER/cuBLAS. The caller
    /// (run_qpadm_search) passes ONLY small-path models here (it routes the large/>32
    /// tail to the per-model fit_models_batched_default). Native FP64 on the
    /// SVD/Qinv/chi² (the §1.4 carve-out); `precision` drives only the covariance GEMM.
    [[nodiscard]] std::vector<QpAdmResult> fit_models_batched(
        const steppe::device::DeviceF2Blocks& f2,
        std::span<const QpAdmModel> models,
        const QpAdmOptions& opts,
        const Precision& precision) override {
        guard_device();
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

private:
    /// One same-shape bucket of the S8 rotation (helper for fit_models_batched). `mem`
    /// holds the indices into `models` (and into `results`) for this bucket; nl/nr/r is
    /// the shared shape. VRAM-budgeted: it sub-chunks the bucket so the per-chunk arena
    /// (f4 + Q + Qinv + the small fit outputs) fits free VRAM minus the resident f2 +
    /// headroom; each sub-chunk is still ONE batched dispatch (the chunking only bites
    /// at very large nr·B, design §6). Writes results[models[mem[j]].model_index].
    void fit_one_bucket(const steppe::device::DeviceF2Blocks& f2,
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

    /// One BATCHED chunk of B same-shape models — the genuine batched dispatch.
    /// F1 / OQ-12: `nb`/`dBlockSizes` are the SURVIVOR set; `d_surv` (nullptr=no drop)
    /// compacts the gather onto the survivor blocks (AT2 read_f2(remove_na=TRUE)).
    void fit_chunk(const steppe::device::DeviceF2Blocks& f2,
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

    /// Per-model outcome sentinels for assemble_result, named so the two booleans can
    /// never transpose at the call site (designated initializers, see :2470). nonspd =
    /// potrf failed ⇒ NonSpdCovariance; se_computed = the SE was computed for this model
    /// (survivor under the JackknifePolicy) ⇒ fill se/z, else leave them EMPTY.
    struct AssembleFlags {
        bool nonspd;
        bool se_computed;
    };

    /// Host post-process of one model's batched outputs into a QpAdmResult — the exact
    /// run_impl + ranktest assembly (pchisq tail-p, the AT2 rankdrop nested table, the
    /// popdrop pattern strings + feasibility). Same math the single-model path does.
    void assemble_result(const QpAdmModel& mdl, int nl, int nr, int r_fit, int rmax,
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

private:
    /// Make THIS backend's device current at every compute entry (architecture.md
    /// §11.4 SPMG: `cudaSetDevice` to switch per device). One backend is bound to one
    /// device (backend.hpp per-device-instance contract); a single host process may
    /// hold one backend per device and interleave their calls, so each entry
    /// re-selects its own device rather than trusting the ambient current device.
    /// On the single-GPU path this re-selects device 0 — a cheap no-op-equivalent
    /// (the runtime short-circuits a redundant set) and ZERO behavior change. It also
    /// satisfies the `CublasHandle` debug device-ordinal assert (handles.hpp): the
    /// handle's cuBLAS context is bound to `device_id_`, and this makes that device
    /// current before any GEMM. Parity-NEUTRAL (device selection moves no bits of the
    /// arithmetic; §12).
    void guard_device() const { STEPPE_CUDA_CHECK(cudaSetDevice(device_id_)); }

    /// Make `device_id` the current CUDA device and RETURN it — the member-init-list
    /// hook that selects the device BEFORE `blas_`/`workspace_` construct (see the
    /// ctor). `static` so it is callable while initializing the first member (no
    /// `this`/no other member touched yet). Throws (STEPPE_CUDA_CHECK) on an invalid
    /// ordinal — fail-fast: a backend cannot be bound to a device that does not exist
    /// (architecture.md §2; the §9 build() device-id validation is the layer above).
    [[nodiscard]] static int set_and_return_device(int device_id) {
        STEPPE_CUDA_CHECK(cudaSetDevice(device_id));
        return device_id;
    }

    // The physical CUDA device this backend instance is bound to (backend.hpp
    // per-device-instance contract; architecture.md §9, §11.4). Set once in the
    // ctor (from the factory arg; default 0 = single-GPU), then `cudaSetDevice`-
    // selected at every compute entry (`guard_device`) and recorded into the
    // capability probe. Declared FIRST so it is initialized before the ctor body's
    // `cudaSetDevice(device_id_)` and before `blas_` (whose cuBLAS context binds to
    // the device current at that set, cuBLAS §2.1.2). A plain int — no teardown
    // ordering concern.
    int device_id_ = 0;

    // The ONE statistic stream PER DEVICE for bit-stability (architecture.md §12
    // single-stream-per-device determinism rule). An OWNING, non-blocking RAII
    // `Stream` (stream.hpp; `cudaStreamNonBlocking`), created in the ctor AFTER
    // `device_id_`'s initializer made this device current — so the stream is
    // associated with `device_id_` (CUDA Runtime API: a stream binds to the device
    // current at create time). Every launch, `cudaMemcpyAsync`, and the trailing
    // `cudaStreamSynchronize` route through `stream_.get()`. The handle is bound to
    // this stream + the workspace ONCE in the ctor and is never re-`cublasSetStream`'d
    // (cleanup X-1/B1), so the emulated-FP64 determinism workspace persists for every
    // GEMM (cuBLAS §2.4.7).
    //
    // WHY NON-BLOCKING (M4.5 SPMG, perf-discovery.md P2/F1). The prior member was
    // the NULL *legacy default* stream and the build is NOT compiled
    // `--default-stream per-thread`, so under the §11.4 multi-GPU fan-out the two
    // per-device worker threads' launches implicitly serialized against the single
    // process-wide legacy default stream (CUDA C Programming Guide §3.2.8.5.2
    // "Default Stream") — the measured 18% kernel overlap. A `cudaStreamNonBlocking`
    // stream does NOT implicitly synchronize with the legacy default stream (CUDA
    // Runtime API, `cudaStreamCreateWithFlags`), so each device's backend now issues
    // on its own independent lane and the two devices' GEMMs can overlap. This is a
    // pure scheduling change: stream choice moves no arithmetic bits, so §12 parity
    // is unaffected. §12 mandates ONE stream PER DEVICE on the statistic path — this
    // single per-device non-blocking stream satisfies it (we do NOT add a second
    // statistic stream).
    //
    // Declaration order is load-bearing at teardown (reverse-order destruction):
    //   1. `workspace_` declared AFTER `blas_` so it is freed FIRST — `blas_` holds a
    //      NON-owning pointer into it; `cublasDestroy` only synchronizes (it does not
    //      read the workspace), so freeing the workspace VRAM before the handle is
    //      destroyed is safe (architecture.md §7).
    //   2. `stream_` declared BEFORE `blas_` so it is destroyed AFTER it — the handle's
    //      cuBLAS context is bound to this stream, so the stream must outlive
    //      `cublasDestroy` (which synchronizes the bound stream); destroying the stream
    //      only after the handle is gone avoids tearing down a stream the handle still
    //      references (architecture.md §7 RAII teardown ordering).
    // Construction order (declaration order) is also load-bearing: `device_id_` is
    // declared FIRST, and its initializer makes the device current, so both `stream_`
    // (created on the current device) and `blas_`'s cuBLAS context (cuBLAS §2.1.2)
    // bind to `device_id_`.
    Stream stream_{};
    CublasHandle blas_{};
    // Dense-cuSOLVER handle for the qpAdm fit small-LA (Qinv Cholesky potrf/potri,
    // the FROZEN CONTRACT §1b). Declared AFTER blas_ so it constructs on device_id_
    // (its context binds to the device current at cusolverDnCreate, like cuBLAS
    // §2.1.2 — device_id_'s initializer made it current) and is destroyed BEFORE
    // stream_ (declaration-order teardown: stream_ outlives the handle whose stream
    // it is). Shares the ONE per-device statistic stream (§12 single-stream
    // determinism); no second stream, no search-stream pool for this single-model
    // milestone (the FROZEN CONTRACT §1b: the batched S7 runs on stream_).
    CusolverDnHandle solver_{};
    DeviceBuffer<std::byte> workspace_{steppe::kCublasWorkspaceBytes};

    // SOLVE-PRECISION promotion knob (ROADMAP §6 the fit-solve promotion seam; set
    // via set_solve_precision). DEFAULT native Fp64 ⇒ emulation_honorable()==false ⇒
    // the CusolverMathModeScope at the cuSOLVER solve sites targets native
    // CUSOLVER_DEFAULT_MATH, so the af6a8c2 golden parity is byte-for-byte the
    // M(fit-4) behavior. A caller (the S8 per-stage policy, or the non-gating
    // emulated-40 parity probe) promotes it to EmulatedFp64{bits} to route the
    // ill-conditioned SPD inverse through the emulated tensor-core path. This is a
    // SEPARATE axis from the matmul `Precision` the qpAdm virtuals receive (which
    // governs the f2/SYRK stages); it governs the cuSOLVER solve math mode only.
    Precision solve_precision_{Precision::Kind::Fp64};

    // S8 observability counter: BATCHED chunk-dispatches issued by fit_models_batched
    // (one per same-shape bucket chunk, NOT one per model). The rotation test reads it
    // via batched_dispatch_count() to PROVE the rotation ran GPU-BATCHED, not as a
    // per-model host loop. Off the numeric path (a plain counter; §12 parity-neutral).
    std::size_t batched_dispatch_count_ = 0;

    // tot_line_ caches the AT2 weighted.mean(loo, 1-bl/n) centering line (length m)
    // produced by assemble_f4 and consumed by jackknife_cov (the xtau term) — the
    // GPU mirror of the CpuBackend's private tot_line_ member (cpu_backend.cpp:531).
    // One model is fit at a time on this backend instance; rebuilt per assemble_f4.
    std::vector<double> tot_line_{};

    // AMORTIZED H2D pinned-input registry (P4/L2; perf-discovery.md). Holds the
    // persistent `cudaHostRegister`s of the Q/V/N H2D source pages so the page-locking
    // cost is paid ONCE per (ptr,bytes) and reused across the many compute_f2_blocks
    // calls a run issues — the precondition for the two devices' H2Ds to run as
    // CONCURRENT pinned DMAs (MEASURED ~2× per-device copy speedup vs contending
    // pageable). Declared LAST so it is destroyed FIRST (reverse-order destruction):
    // it only `cudaHostUnregister`s host pages — no dependency on stream_/blas_, and
    // unregistering the caller's pages before the device-context members tear down is
    // clean. The registrations reference caller-owned host memory (the contract's
    // Q/V/N); the cache must not outlive that memory, which it cannot — the backend is
    // owned by `Resources`, scoped within the caller's compute call tree. PARITY-
    // NEUTRAL (pinning moves no arithmetic bits; §12).
    PinnedRegistryCache pinned_in_{};

    // PERSISTENT pinned D2H staging (P5/d2h-speed). Sized to the largest partial this
    // backend has D2H'd, allocated ONCE via cudaHostAlloc and REUSED across every
    // compute_f2_blocks_into call — so the page-locking cost is paid once, not per call.
    // The prior path pinned the caller's ~3 GB result slice EVERY call
    // (RegisteredHostRegion), and cudaHostRegister/Unregister take the device-wide driver
    // lock, serializing the two worker threads' D2Hs (~570 ms serial tail, MEASURED nsys
    // box5090). With persistent pinned staging the two devices' D2Hs run as concurrent
    // pinned DMAs into per-backend buffers; the host memcpy to the disjoint result slice
    // is CPU-bandwidth, takes NO driver lock, and runs concurrently on the two worker
    // threads. Declared LAST (destroyed FIRST): cudaFreeHost has no dependency on
    // stream_/blas_. PARITY-NEUTRAL: same doubles, same disjoint offset, exact memcpy
    // (architecture.md §12).
    PinnedBuffer<double> stage_f2_{};
    PinnedBuffer<double> stage_vpair_{};
};

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

}  // namespace steppe::device

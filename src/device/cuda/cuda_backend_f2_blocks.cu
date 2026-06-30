// src/device/cuda/cuda_backend_f2_blocks.cu
//
// CudaBackend — f2 block-tensor compute subsystem TU (cuda_backend.cu split T7;
// docs/kimiactions/05-cuda-backend-split.md §2.3 TU-B). Out-of-line homes of the f2
// 3-GEMM door `CudaBackend::compute_f2` (the FIRST non-inline virtual override → the
// vtable + typeinfo KEY-FUNCTION home for the whole class) plus the block-tensor
// family: `compute_f2_blocks` / `compute_f2_blocks_device` / `compute_f2_blocks_resident`
// / `compute_f2_blocks_into` / `compute_f2_blocks_streamed` and the two engines
// `run_f2_blocks_resident` (the resident strided-batched tensor) + `stream_f2_blocks_impl`
// (the M5 streamed/spilling tier). Bodies MOVED VERBATIM from cuda_backend.cu; nothing
// about codegen / math / precision / file-order changed by the split.
//
// SINGLE-OWNER anon-namespace helpers (§2.2): the shared block-layout + size-bucketing
// prologue `Bucket` / `BlockLayout` / `compute_block_layout` / `size_buckets` — used ONLY
// by `run_f2_blocks_resident` + `stream_f2_blocks_impl` (both in THIS TU) — live in this
// TU's anonymous namespace (no shared internal header). The method-local `struct Ring`
// stays inside `stream_f2_blocks_impl`. Uses members `stage_f2_` / `stage_vpair_` /
// `pinned_in_` (declared in cuda_backend.cuh).
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). Joins the SAME
// steppe_device target, so it inherits identical codegen/macros/RDC.
#include <algorithm>  // std::sort — size_buckets bucket ordering (anon helper)
#include <cmath>      // (per the §2.3 TU-B include set; block sizing / budget math)
#include <cstring>    // std::memcpy — staging->result slice copy (P5/d2h-speed)
#include <stdexcept>  // std::runtime_error — the M0 M>INT_MAX / VRAM fail-fast
#include <vector>     // std::vector — block layout / bucket / staging host containers

#include "core/domain/block_partition_rule.hpp"   // core::block_ranges / core::BlockRange (compute_block_layout single-source inverse)
#include "core/internal/nvtx.hpp"                  // STEPPE_NVTX_RANGE (coarse phase-boundary marker; empty unless -DSTEPPE_NVTX)
#include "device/cuda/cuda_backend.cuh"            // the CudaBackend class declaration (split T0)
#include "device/cuda/block_sink.cuh"              // M5: BlockSink, HostRamSink, DiskSink, kStreamStagingSlots
#include "device/cuda/check.cuh"                   // STEPPE_CUDA_CHECK
#include "device/cuda/device_f2_blocks_impl.cuh"   // DeviceF2Blocks::Impl (the DeviceBuffer<double> f2/vpair owners)
#include "device/cuda/device_partial_impl.cuh"     // DevicePartial::Impl (the DeviceBuffer<double> owners)
#include "device/cuda/f2_batched_kernel.cuh"       // launch_gather_group, run_f2_gemms_group, launch_assemble_blocks_group
#include "device/cuda/f2_block_kernel.cuh"         // launch_f2_feeder, run_f2_gemms, engage_f2_precision
#include "device/f2_blocks_out.hpp"                // M5: DiskF2Blocks (the Disk descriptor DiskSink populates)
#include "device/stream_f2_blocks.hpp"             // M5: StreamTarget (the CUDA-free streamed-tier request)
#include "device/vram_budget.hpp"                  // max_blocks_per_chunk (host-pure VRAM budget; X-5/B5 + X-13/B26)

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

}  // namespace

F2Result CudaBackend::compute_f2(const core::MatView& Q,
                                  const core::MatView& V,
                                  const core::MatView& N,
                                  const Precision& precision) {
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

F2BlockTensor CudaBackend::compute_f2_blocks(const core::MatView& Q,
                                              const core::MatView& V,
                                              const core::MatView& N,
                                              const int* block_id,
                                              int n_block,
                                              const Precision& precision) {
    return compute_f2_blocks_device(Q, V, N, block_id, n_block, precision).to_host();
}

DeviceF2Blocks CudaBackend::compute_f2_blocks_device(
    const core::MatView& Q, const core::MatView& V, const core::MatView& N,
    const int* block_id, int n_block, const Precision& precision) {
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

DevicePartial CudaBackend::compute_f2_blocks_resident(
    const core::MatView& Q, const core::MatView& V, const core::MatView& N,
    const int* block_id, int n_block, int b0, const Precision& precision) {
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

void CudaBackend::compute_f2_blocks_into(
    const core::MatView& Q, const core::MatView& V, const core::MatView& N,
    const int* block_id, int n_block, int b0,
    double* dst_f2, double* dst_vpair, int* block_sizes_dst,
    const Precision& precision) {
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
        //
        // CAP (B5): the staging grows monotonically and NEVER shrinks, so a
        // pathological `total` would silently pin an unbounded non-pageable host
        // buffer. Assert `total` equals the expected per-device partial bound
        // P²·n_block_local — which is exactly rb.f2's element count and is already
        // VRAM-bounded (rb.f2/rb.vpair are device-resident DeviceBuffers that passed
        // the §11.2 budget), so the pinned staging can never exceed what already fit
        // in VRAM. Debug-only (compiles out under NDEBUG); cross-checks rb.f2.size()
        // against the independently-formed P*P*n_block before any host pin.
        STEPPE_ASSERT(total == static_cast<std::size_t>(rb.P) *
                                   static_cast<std::size_t>(rb.P) *
                                   static_cast<std::size_t>(rb.n_block),
                      "compute_f2_blocks_into: pinned staging exceeds the "
                      "P*P*n_block partial bound");
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

void CudaBackend::compute_f2_blocks_streamed(
    const core::MatView& Q, const core::MatView& V, const core::MatView& N,
    const int* block_id, int n_block, const Precision& precision,
    StreamTarget& target) {
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

CudaBackend::ResidentBlocks CudaBackend::run_f2_blocks_resident(const core::MatView& Q,
                                                    const core::MatView& V,
                                                    const core::MatView& N,
                                                    const int* block_id,
                                                    int n_block,
                                                    const Precision& precision) {
    guard_device();
    STEPPE_NVTX_RANGE("f2_gemm");  // coarse phase boundary: the batched f2 GEMM body
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
    // range's begin, copied to dOffsets + dereferenced by f2_batched_kernel.cu) and
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
    // size; f2_batched_kernel.cuh), so a chunk uses only the leading
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

void CudaBackend::stream_f2_blocks_impl(const core::MatView& Q, const core::MatView& V,
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

}  // namespace steppe::device

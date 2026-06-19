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
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). It is the only
// place a host caller meets the GPU f2 path; `core` reaches it solely through the
// CUDA-free ComputeBackend interface in device/backend.hpp.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>     // std::numeric_limits<int>::max — the M0 k-narrowing guard (B22)
#include <memory>
#include <span>
#include <stdexcept>  // std::runtime_error — the M0 M>INT_MAX fail-fast (B22)
#include <string>     // std::to_string — diagnostic message for the B22 guard
#include <vector>

#include "core/domain/block_partition_rule.hpp" // core::block_ranges, core::BlockRange (the X-3/B3 single-source inverse)
#include "device/backend.hpp"               // ComputeBackend, F2Result, F2BlockTensor, MatView
#include "device/backend_factory.hpp"       // steppe::device::make_cuda_backend (the single-source decl, X-9/B8)
#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK
#include "device/cuda/decode_af_kernel.cuh" // launch_decode_af
#include "device/cuda/device_buffer.cuh"    // DeviceBuffer<T> (RAII)
#include "device/cuda/f2_block_kernel.cuh"  // launch_f2_feeder, engage_f2_precision, emulation_honorable (the X-6/B2 probe)
#include "device/cuda/f2_blocks_kernel.cuh" // launch_gather_group, run_f2_gemms_group, launch_assemble_blocks_group
#include "device/cuda/handles.hpp"          // CublasHandle (RAII)
#include "device/cuda/stream.hpp"           // Stream (RAII, owning non-blocking per-device stream — P2/F1)
#include "device/vram_budget.hpp"           // max_blocks_per_chunk (host-pure VRAM budget; X-5/B5 + X-13/B26)
#include "steppe/config.hpp"                // Precision, kDefaultMantissaBits, kBlockGroupPadBase, kMaxVramUtilizationFraction
#include "steppe/fstats.hpp"                // F2BlockTensor

namespace steppe::device {

/// GPU compute backend. The 3-GEMM f2 reformulation; one CublasHandle created
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
    }

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
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ_raw.data(), Q.data, pm * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV_raw.data(), V.data, pm * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dN_raw.data(), N.data, pm * sizeof(double),
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
        // `out` (with out.P set) was constructed before the degenerate guard.
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
    [[nodiscard]] F2BlockTensor compute_f2_blocks(const core::MatView& Q,
                                                  const core::MatView& V,
                                                  const core::MatView& N,
                                                  const int* block_id,
                                                  int n_block,
                                                  const Precision& precision) override {
        guard_device();
        const int P = Q.P;
        const long M = Q.M;

        F2BlockTensor out;
        out.P = P;
        out.n_block = n_block;
        const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
        const std::size_t total = slab * static_cast<std::size_t>(n_block < 0 ? 0 : n_block);
        out.f2.assign(total, 0.0);
        out.vpair.assign(total, 0.0);
        out.block_sizes.assign(static_cast<std::size_t>(n_block < 0 ? 0 : n_block), 0);
        if (P <= 0 || M <= 0 || n_block <= 0) return out;

        // ---- Block layout from block_id (contiguous in file order) -----------
        // The SINGLE-SOURCE inverse of assign_blocks: block_ranges validates the
        // partition contract ONCE (0 <= id < n_block, non-decreasing, block_id
        // long enough) and returns each block's half-open column range
        // [begin, end) — closing the OOB write/read this scan used to risk on a
        // malformed partition (cleanup X-3/B3). The device wants block_offsets
        // (= each range's begin) it copies to dOffsets and the kernel dereferences
        // (f2_blocks_kernel.cu), plus block_sizes (= each range's size, the S4
        // F2BlockTensor metadata / weighting denominator base).
        const std::vector<core::BlockRange> ranges =
            core::block_ranges(std::span<const int>(block_id, static_cast<std::size_t>(M)),
                               M, n_block);
        std::vector<long> block_offsets(static_cast<std::size_t>(n_block), 0);
        for (int b = 0; b < n_block; ++b) {
            block_offsets[static_cast<std::size_t>(b)] = ranges[static_cast<std::size_t>(b)].begin;
            out.block_sizes[static_cast<std::size_t>(b)] =
                static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
        }

        // ---- Group blocks by ceil-pow{kBlockGroupPadBase}(size) (the spike rule).
        // One strided-batched call per bucket, padded only to the bucket width;
        // bounds pad waste < base× WITHIN a bucket while keeping the call count
        // O(log max_size). Buckets sorted by width (cosmetic / smallest first).
        auto ceil_bucket = [](int x) {
            int p = 1;
            while (p < x) p *= steppe::kBlockGroupPadBase;
            return p;
        };
        struct Bucket { int s_pad = 0; std::vector<int> blocks; };
        std::vector<Bucket> buckets;
        for (int b = 0; b < n_block; ++b) {
            const int sp = ceil_bucket(out.block_sizes[static_cast<std::size_t>(b)]);
            int gi = -1;
            for (std::size_t k = 0; k < buckets.size(); ++k)
                if (buckets[k].s_pad == sp) { gi = static_cast<int>(k); break; }
            if (gi < 0) { gi = static_cast<int>(buckets.size()); buckets.push_back({sp, {}}); }
            buckets[static_cast<std::size_t>(gi)].blocks.push_back(b);
        }
        std::sort(buckets.begin(), buckets.end(),
                  [](const Bucket& a, const Bucket& c) { return a.s_pad < c.s_pad; });

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
        DeviceBuffer<double> dF2_all(total), dVpair_all(total);  // the resident tensors
        DeviceBuffer<long>   dOffsets(static_cast<std::size_t>(n_block));
        DeviceBuffer<int>    dSizes(static_cast<std::size_t>(n_block));

        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dOffsets.data(), block_offsets.data(),
                                          static_cast<std::size_t>(n_block) * sizeof(long),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSizes.data(), out.block_sizes.data(),
                                          static_cast<std::size_t>(n_block) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));

        {
            // Raw inputs — feeder-only; freed at the end of this scope.
            DeviceBuffer<double> dQ_raw(pm), dV_raw(pm), dN_raw(pm);
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ_raw.data(), Q.data, pm * sizeof(double),
                                              cudaMemcpyHostToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV_raw.data(), V.data, pm * sizeof(double),
                                              cudaMemcpyHostToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dN_raw.data(), N.data, pm * sizeof(double),
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

        // ---- Per bucket (chunked): gather → strided-batched GEMMs → assemble ----
        // Each bucket is processed in CHUNKS of at most `max_blocks` blocks (the
        // host-pure, GPU-free budget helper: it reserves BOTH resident tensors +
        // the cuBLAS workspace, applies kMaxVramUtilizationFraction, and clamps in
        // size_t before the int narrowing — device/vram_budget.hpp). This keeps the
        // grouped strided-batched design while bounding the working set (the M5
        // out-of-core generalization is a superset of this; here it is the
        // single-GPU VRAM guard ROADMAP M4 requires).
        for (const Bucket& bk : buckets) {
            const int s_pad = bk.s_pad;
            const int nb_total = static_cast<int>(bk.blocks.size());
            const int max_blocks =
                max_blocks_per_chunk(free_b, P, n_block, s_pad, nb_total);

            for (int start = 0; start < nb_total; start += max_blocks) {
                const int nb = std::min(max_blocks, nb_total - start);

                DeviceBuffer<int> dIds(static_cast<std::size_t>(nb));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dIds.data(), bk.blocks.data() + start,
                                                  static_cast<std::size_t>(nb) * sizeof(int),
                                                  cudaMemcpyHostToDevice, stream_.get()));

                const std::size_t psp_nb =
                    static_cast<std::size_t>(P) * static_cast<std::size_t>(s_pad) *
                    static_cast<std::size_t>(nb);
                DeviceBuffer<double> dQg(psp_nb), dVg(psp_nb), dSg(2u * psp_nb);
                const std::size_t pp_nb = slab * static_cast<std::size_t>(nb);
                DeviceBuffer<double> dGg(pp_nb), dVpairg(pp_nb), dRg(2u * pp_nb);

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
                                             dF2_all.data(), dVpair_all.data(), stream_.get());
                // Sync before this chunk's slabs (dQg…dRg) free at scope exit, so
                // the next chunk reuses the VRAM (one chunk resident at a time).
                STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
            }
        }

        // ---- Copy the resident tensors back across the CUDA-free seam ---------
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.f2.data(), dF2_all.data(),
                                          total * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.vpair.data(), dVpair_all.data(),
                                          total * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        return out;
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

        // ---- Device allocations (RAII; freed on scope exit) ------------------
        // Packed tile bytes + the P+1 segment offsets + the three [P×M] outputs.
        // Only tile-sized and [P×M] buffers — never a [SNP×ind] decode-all
        // (architecture.md §11.1; tile-shaped for the M5 loop).
        const std::size_t packed_bytes =
            tile.n_individuals * tile.bytes_per_record;
        const std::size_t n_off = static_cast<std::size_t>(P) + 1u;
        DeviceBuffer<std::uint8_t> dPacked(packed_bytes);
        DeviceBuffer<std::size_t> dOffsets(n_off);
        DeviceBuffer<double> dQ(pm), dV(pm), dN(pm);

        // ---- Upload the packed tile + the population partition ---------------
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dPacked.data(), tile.packed,
                                          packed_bytes * sizeof(std::uint8_t),
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dOffsets.data(), tile.pop_offsets,
                                          n_off * sizeof(std::size_t),
                                          cudaMemcpyHostToDevice, stream_.get()));

        // ---- Decode (S0 unpack + S1 segmented reduction → Q/V/N) -------------
        launch_decode_af(dPacked.data(), tile.bytes_per_record, dOffsets.data(),
                         P, M, tile.ploidy, dQ.data(), dV.data(), dN.data(), stream_.get());

        // ---- Copy results back across the CUDA-free seam ---------------------
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.q.data(), dQ.data(), pm * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.v.data(), dV.data(), pm * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.n.data(), dN.data(), pm * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
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
    DeviceBuffer<std::byte> workspace_{steppe::kCublasWorkspaceBytes};
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

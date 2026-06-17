// src/device/cuda/cuda_backend.cu
//
// CudaBackend — the GPU implementation of ComputeBackend (architecture.md §4, §8;
// ROADMAP §2, M0). Implements `compute_f2` via the f2 3-GEMM reformulation:
// upload the Q/V/N contract, run the fused feeder + three GEMMs + fused
// numerator/divide (f2_block_kernel.cu), and copy f2 + Vpair back across the
// CUDA-free ComputeBackend seam (architecture.md §4).
//
// STANDARDS (ROADMAP §1, §5):
//   * RAII for ALL device memory and handles — DeviceBuffer<T> + CublasHandle, no
//     raw cudaMalloc / cublasCreate here (architecture.md §2, §7). This TU is NOT
//     on the allocation allowlist.
//   * Precision is typed config: forwarded unchanged into run_f2_gemms, which
//     engages FIXED-slice Ozaki / native FP64 (architecture.md §12; ROADMAP §0).
//   * The numerator/divide stays native FP64 (in the assemble kernel).
//   * The formula lives ONCE in core/internal/f2_estimator.hpp, shared with the
//     CPU oracle, so CPU and GPU cannot diverge (architecture.md §13).
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). It is the only
// place a host caller meets the GPU f2 path; `core` reaches it solely through the
// CUDA-free ComputeBackend interface in device/backend.hpp.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/domain/block_partition_rule.hpp" // core::block_ranges, core::BlockRange (the X-3/B3 single-source inverse)
#include "device/backend.hpp"               // ComputeBackend, F2Result, F2BlockTensor, MatView
#include "device/backend_factory.hpp"       // steppe::device::make_cuda_backend (the single-source decl, X-9/B8)
#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK
#include "device/cuda/decode_af_kernel.cuh" // launch_decode_af
#include "device/cuda/device_buffer.cuh"    // DeviceBuffer<T> (RAII)
#include "device/cuda/f2_block_kernel.cuh"  // launch_f2_feeder, engage_f2_precision
#include "device/cuda/f2_blocks_kernel.cuh" // launch_gather_group, run_f2_gemms_group, launch_assemble_blocks_group
#include "device/cuda/handles.hpp"          // CublasHandle (RAII)
#include "device/vram_budget.hpp"           // max_blocks_per_chunk (host-pure VRAM budget; X-5/B5 + X-13/B26)
#include "steppe/config.hpp"                // Precision, kBlockGroupPadBase, kMaxVramUtilizationFraction
#include "steppe/fstats.hpp"                // F2BlockTensor

namespace steppe::device {

/// GPU compute backend. The 3-GEMM f2 reformulation; one CublasHandle created
/// once (architecture.md §7) and reused, with its workspace set for emulated-FP64
/// determinism. Move-only via the ComputeBackend base (architecture.md §8).
class CudaBackend final : public ComputeBackend {
public:
    CudaBackend() {
        // RAII handle, created once. Bind the (stream, workspace) invariant ONCE
        // here (architecture.md §12; cleanup X-1/B1): the workspace pins emulated-
        // FP64 reproducibility (cuBLAS §2.1.4) and the single statistic stream is
        // bound through `set_stream`, which re-applies the workspace after the
        // `cublasSetStream` reset (cuBLAS §2.4.7). The GEMM routines never touch
        // the stream again, so the workspace survives for every GEMM batch on
        // BOTH the M0 and M4 paths. Order matters: bind the workspace FIRST so the
        // `set_stream` re-apply has it.
        blas_.set_workspace(workspace_.data(), workspace_.bytes());
        blas_.set_stream(stream_);
    }

    [[nodiscard]] F2Result compute_f2(const core::MatView& Q,
                                      const core::MatView& V,
                                      const core::MatView& N,
                                      const Precision& precision) override {
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
                                          cudaMemcpyHostToDevice, stream_));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV_raw.data(), V.data, pm * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dN_raw.data(), N.data, pm * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_));

        // ---- Fused feeder -> 3 GEMMs -> fused numerator/divide ---------------
        launch_f2_feeder(dQ_raw.data(), dV_raw.data(), dN_raw.data(),
                         dQ.data(), dV.data(), dS.data(), P, M, stream_);
        // The handle's stream + emulated-FP64 workspace are bound ONCE in the ctor
        // (architecture.md §12; cleanup X-1/B1); run_f2_gemms never re-binds the
        // stream, so the determinism workspace survives every GEMM.
        run_f2_gemms(blas_.get(), precision, P, M,
                     dQ.data(), dV.data(), dS.data(),
                     dG.data(), dVpair.data(), dR.data());
        launch_assemble_f2(dG.data(), dVpair.data(), dR.data(), dF2.data(), P, stream_);

        // ---- Copy results back across the CUDA-free seam --------------------
        // `out` (with out.P set) was constructed before the degenerate guard.
        out.f2.resize(pp);
        out.vpair.resize(pp);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.f2.data(), dF2.data(), pp * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.vpair.data(), dVpair.data(),
                                          pp * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_));
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
                                          cudaMemcpyHostToDevice, stream_));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSizes.data(), out.block_sizes.data(),
                                          static_cast<std::size_t>(n_block) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_));

        {
            // Raw inputs — feeder-only; freed at the end of this scope.
            DeviceBuffer<double> dQ_raw(pm), dV_raw(pm), dN_raw(pm);
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ_raw.data(), Q.data, pm * sizeof(double),
                                              cudaMemcpyHostToDevice, stream_));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV_raw.data(), V.data, pm * sizeof(double),
                                              cudaMemcpyHostToDevice, stream_));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dN_raw.data(), N.data, pm * sizeof(double),
                                              cudaMemcpyHostToDevice, stream_));
            // ONE fused feeder over ALL SNPs (block-agnostic; native FP64).
            launch_f2_feeder(dQ_raw.data(), dV_raw.data(), dN_raw.data(),
                             dQ.data(), dV.data(), dS.data(), P, M, stream_);
            // Sync so the feeder finishes before dQ_raw/dV_raw/dN_raw free (the
            // gather then reads dQ/dV/dS, and the freed raw VRAM is reused by the
            // bucket slabs).
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_));
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
                                                  cudaMemcpyHostToDevice, stream_));

                const std::size_t psp_nb =
                    static_cast<std::size_t>(P) * static_cast<std::size_t>(s_pad) *
                    static_cast<std::size_t>(nb);
                DeviceBuffer<double> dQg(psp_nb), dVg(psp_nb), dSg(2u * psp_nb);
                const std::size_t pp_nb = slab * static_cast<std::size_t>(nb);
                DeviceBuffer<double> dGg(pp_nb), dVpairg(pp_nb), dRg(2u * pp_nb);

                launch_gather_group(dQ.data(), dV.data(), dS.data(),
                                    dIds.data(), dOffsets.data(), dSizes.data(),
                                    P, s_pad, nb, dQg.data(), dVg.data(), dSg.data(), stream_);
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
                                             dF2_all.data(), dVpair_all.data(), stream_);
                // Sync before this chunk's slabs (dQg…dRg) free at scope exit, so
                // the next chunk reuses the VRAM (one chunk resident at a time).
                STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_));
            }
        }

        // ---- Copy the resident tensors back across the CUDA-free seam ---------
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.f2.data(), dF2_all.data(),
                                          total * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.vpair.data(), dVpair_all.data(),
                                          total * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return out;
    }

    [[nodiscard]] DecodeResult decode_af(const DecodeTileView& tile) override {
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
                                          cudaMemcpyHostToDevice, stream_));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dOffsets.data(), tile.pop_offsets,
                                          n_off * sizeof(std::size_t),
                                          cudaMemcpyHostToDevice, stream_));

        // ---- Decode (S0 unpack + S1 segmented reduction → Q/V/N) -------------
        launch_decode_af(dPacked.data(), tile.bytes_per_record, dOffsets.data(),
                         P, M, tile.ploidy, dQ.data(), dV.data(), dN.data(), stream_);

        // ---- Copy results back across the CUDA-free seam ---------------------
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.q.data(), dQ.data(), pm * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.v.data(), dV.data(), pm * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.n.data(), dN.data(), pm * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return out;
    }

private:
    // Single statistic stream for bit-stability (architecture.md §12). The
    // default stream suffices at M0 (one f2 call at a time); a dedicated RAII
    // Stream lands with the streaming pipeline (architecture.md §11.1). The
    // handle is bound to this stream + the workspace ONCE in the ctor and is
    // never re-`cublasSetStream`'d (cleanup X-1/B1), so the emulated-FP64
    // determinism workspace persists for every GEMM (cuBLAS §2.4.7).
    //
    // Declaration order is load-bearing at teardown (reverse-order destruction):
    // `blas_` holds a NON-owning pointer into `workspace_`, so `workspace_` must
    // be declared AFTER `blas_` to be freed first — `cublasDestroy` only
    // synchronizes (it does not read the workspace), so freeing the workspace
    // VRAM before the handle is destroyed is safe (architecture.md §7).
    cudaStream_t stream_ = nullptr;
    CublasHandle blas_{};
    DeviceBuffer<std::byte> workspace_{steppe::kCublasWorkspaceBytes};
};

/// Factory for the GPU backend (declared in device/backend_factory.hpp, X-9/B8;
/// architecture.md §9 — backend chosen at build()). Returns the abstract interface
/// so `core` / `Resources` never name the concrete type or touch a CUDA header
/// (architecture.md §4, §8).
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cuda_backend() {
    return std::make_unique<CudaBackend>();
}

}  // namespace steppe::device

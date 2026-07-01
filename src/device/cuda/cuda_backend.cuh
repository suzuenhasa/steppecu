#pragma once
// src/device/cuda/cuda_backend.cuh
//
// PRIVATE-to-steppe_device declaration of CudaBackend (the GPU ComputeBackend).
// Extracted from cuda_backend.cu by the cuda_backend split T0 (header extract +
// out-of-line method bodies; docs/kimiactions/05-cuda-backend-split.md). The class
// layout is byte-faithful to the pre-split inline class: the 3 nested types in
// their access regions (ResidentBlocks/SvdScratchSizes PUBLIC, AssembleFlags
// PRIVATE), the 2 hot inline one-liners (guard_device / set_and_return_device), and
// the 17 private data members in EXACT declaration order (teardown-order load-
// bearing -- DO NOT reorder). Method bodies are defined out-of-line in the per-
// subsystem .cu TUs that all compile into the same steppe_device target.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "device/backend.hpp"               // ComputeBackend + the f-stat / fit interface types named in the overrides (MatView, F2Result, ...)
#include "device/device_partial.hpp"        // steppe::device::DevicePartial (compute_f2_blocks_resident return)
#include "device/device_f2_blocks.hpp"      // steppe::device::DeviceF2Blocks (resident-tensor handle in the sweep/assemble sigs)
#include "device/device_decode_result.hpp"  // steppe::device::DeviceDecodeResult (the device-resident decode handle)
#include "device/stream_f2_blocks.hpp"      // StreamTarget (compute_f2_blocks_streamed param)
#include "device/f2_blocks_out.hpp"         // DiskF2Blocks (the Disk descriptor in the streamed-tier sig)
#include "device/vram_budget.hpp"           // (the VRAM-budget seam these subsystems share)
#include "steppe/config.hpp"                // Precision (the typed precision config in every compute sig)
#include "steppe/fstats.hpp"                // F2BlockTensor (the host-tensor oracle overloads)
#include "device/cuda/handles.hpp"          // CublasHandle, CusolverDnHandle (data members; pull in cublas/cusolver decls)
#include "device/cuda/device_buffer.cuh"    // DeviceBuffer<T> (data members + nested-type fields)
#include "device/cuda/pinned_buffer.cuh"    // PinnedRegistryCache, PinnedBuffer<T> (the H2D pin + D2H staging members)
#include "device/cuda/stream.hpp"           // Stream (the one per-device statistic stream member)
#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK (the inline guard_device / set_and_return_device bodies)

namespace steppe::device {

/// Forward declaration: the M5 block-spill sink (defined in cuda/block_sink.cuh, which
/// travels with TU-B per §2.1). Named only as a reference parameter of
/// stream_f2_blocks_impl below, so the incomplete type suffices in this header.
class BlockSink;

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
    explicit CudaBackend(int device_id = 0);

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
                                      const Precision& precision) override;

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
                                                  const Precision& precision) override;

    /// M4.5 device-resident PRIMARY (the cure). Runs the SAME GEMM body
    /// (run_f2_blocks_resident), then MOVES the resident f2/Vpair DeviceBuffers into a
    /// DeviceF2Blocks — NO D2H, NO free, NO host alloc. The full result ESCAPES into
    /// the handle (VRAM-resident on device_id_). Bit-identical to compute_f2_blocks's
    /// resident bits (same run_f2_blocks_resident; §12).
    [[nodiscard]] DeviceF2Blocks compute_f2_blocks_device(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, const Precision& precision) override;

    /// M4.5 device-resident override (the cure, doc §4 Item 1). Runs the SAME GEMM
    /// body, then MOVES the resident f2/Vpair DeviceBuffers into a DevicePartial —
    /// NO D2H, NO free. The buffers ESCAPE this call into the returned handle (they
    /// survive the jthread join and free only AFTER the combine consumed them, §7).
    [[nodiscard]] DevicePartial compute_f2_blocks_resident(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, int b0, const Precision& precision) override;

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
        const Precision& precision) override;

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
        StreamTarget& target) override;

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
                                                        const Precision& precision);

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
                               const Precision& precision, BlockSink& sink);

    /// Shared decode front-end: upload the packed tile + partition (+ per-sample
    /// ploidy when supplied) and run launch_decode_af into the caller's RESIDENT
    /// dQ/dV/dN [P × M]. NO D2H — the caller decides whether the result crosses the
    /// CUDA-free seam (decode_af: full D2H, host/oracle path) or STAYS resident
    /// (decode_af_compact_autosome: the device-resident seam). The packed/offsets/
    /// ploidy uploads are local RAII (freed on return); dQ/dV/dN are caller-owned.
    void decode_af_resident(const DecodeTileView& tile, int P, long M,
                            DeviceBuffer<double>& dQ, DeviceBuffer<double>& dV,
                            DeviceBuffer<double>& dN);

    [[nodiscard]] DecodeResult decode_af(const DecodeTileView& tile) override;

    /// M-FR-0: the AT2 per-sample ploidy prepass ON THE GPU (the L2 host-compute fix).
    /// Upload the packed tile, run launch_detect_ploidy (one thread/individual over the
    /// SAME bytes decode_af reads), and D2H the [n_individuals] int ploidy vector. A
    /// literal port of io::detect_sample_ploidy over the shared core decode primitives
    /// — bit-identical to the host detector by construction (integer/bit ops). This is
    /// the standalone entry; decode_af_resident runs the SAME prepass inline when the
    /// tile sets detect_ploidy_on_device (no second upload there).
    [[nodiscard]] std::vector<int> detect_sample_ploidy_device(
        const DecodeTileView& tile) override;

    /// M-FR-1: the SNP-major -> canonical individual-major TRANSPOSE+GATHER+ENCODING
    /// ON THE GPU (the format-reader engine; the on-device override of the CUDA-free
    /// base oracle). Upload the SNP-major source bytes + the selection (sel_rows /
    /// pop_offsets), run launch_transpose_to_canonical (ONE thread per output byte:
    /// gather source row sel_rows[g], encode, MSB-first re-pack into the canonical
    /// individual-major byte), and D2H the canonical tile. Integer/bit-exact to the
    /// base host-loop oracle BY CONSTRUCTION (same core::genotype_code extractor, same
    /// encoding, same packer); the M-FR-1 unit gate diffs this device tile against the
    /// CpuBackend oracle AND a hand-built expected tile (memcmp==0).
    [[nodiscard]] CanonicalTile transpose_to_canonical(
        const SnpMajorTileView& view) override;

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
        std::span<const double> genpos, std::span<const double> physpos,
        int chrom_min, int chrom_max) override;

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
        std::span<const double> physpos, const FilterConfig& cfg,
        std::span<const std::size_t> pop_individuals, int ploidy, double maxmiss) override;

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
                                   double* numsum, double* densum, double* cnt);

    void dstat_block_reduce(const double* Q, const double* V, int P, long M,
                            const int* block_id, int n_block,
                            std::span<const int> quadruples,
                            double* numsum, double* densum, double* cnt) override;

    /// DEVICE-RESIDENT dstat_block_reduce — reads Q/V already RESIDENT in VRAM from
    /// a DeviceDecodeResult (the M4 cure: NO Q/V H2D). Math byte-identical to the
    /// host-pointer overload (the SAME launch_dstat_block_reduce over the SAME Q/V
    /// the decode wrote, compacted onto the autosome axis).
    void dstat_block_reduce(const steppe::device::DeviceDecodeResult& dec,
                            const int* block_id, int n_block,
                            std::span<const int> quadruples,
                            double* numsum, double* densum, double* cnt) override;

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
        const Precision& precision) override;

    /// M5 — DATES target-genotype REPACK onto the kept SNP axis ON THE DEVICE (host-compute
    /// audit; backend.hpp dates_repack). Replaces the dates.cpp host bit-shuffle hot loop
    /// (O(n_target × M_kept)): upload the n_target FULL target records + the kept-index map
    /// ONCE, run the device gather (one thread per dest byte, race-free; reads via the SAME
    /// MSB-first core::genotype_code, writes the SAME shift/OR as the host), D2H only the
    /// dense repacked buffer. INTEGER/BIT-EXACT ⇒ bit-identical to the CpuBackend repack ⇒
    /// dates_curve moments unchanged ⇒ the DATES date golden held exactly.
    void dates_repack(const std::uint8_t* src, std::size_t src_bpr, const long* kept_src,
                      long M_kept, int n_target, std::size_t dst_bpr,
                      std::uint8_t* dst) override;

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
                                                     bool affine) override;

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
                                                 const Precision& precision) override;

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
        std::span<const int> block_sizes, double ridge, const Precision& precision) override;

    /// DEVICE-RESIDENT overload — reads Q/V already RESIDENT in VRAM from a
    /// DeviceDecodeResult (the M4 cure: NO Q/V H2D). Math byte-identical to the
    /// host-pointer overload (the SAME fused core over the SAME compacted Q/V).
    [[nodiscard]] QpfstatsSmooth qpfstats_blocks_smooth(
        const steppe::device::DeviceDecodeResult& dec, const int* block_id, int n_block,
        std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
        std::span<const int> block_sizes, double ridge, const Precision& precision) override;

    /// The qpfstats_blocks_smooth CORE over ALREADY-RESIDENT device Q/V pointers
    /// (the single-source fused body both the host-pointer and the DeviceDecodeResult
    /// overloads share). dQ/dV are borrowed device pointers (NOT freed here): the
    /// host overload owns DeviceBuffers it uploaded; the device overload passes the
    /// resident DeviceDecodeResult Q/V — NO Q/V H2D (the M4 cure). Math byte-identical.
    [[nodiscard]] QpfstatsSmooth qpfstats_blocks_smooth_device(
        const double* dQ, const double* dV, int P, long M, const int* block_id, int n_block,
        std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
        std::span<const int> block_sizes, double ridge, const Precision& precision);

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
    [[nodiscard]] BackendCapabilities capabilities() const override;

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
    void set_solve_precision(const Precision& precision) override;

    /// S8 instrumentation (observability): how many BATCHED chunk-dispatches this
    /// backend has issued through fit_models_batched (one per same-shape bucket chunk,
    /// NOT one per model). The rotation test asserts this is a FEW (one per left-size
    /// bucket), proving the rotation ran GPU-BATCHED, not as a per-model host loop.
    [[nodiscard]] std::size_t batched_dispatch_count() const override;

    /// F1 / OQ-12 — the ASCENDING SURVIVOR block id list for a resident f2 (AT2
    /// read_f2(remove_na=TRUE)). Runs the on-device keep kernel over the resident Vpair
    /// (shared predicate core::pair_block_is_missing), reads the tiny [nb] keep vector
    /// down, and returns `which(keep)`. When the resident handle carries NO Vpair (an
    /// upload path that left it empty — the global-intersection invariant: no missing
    /// blocks) OR Vpair is empty, EVERY block survives (identity 0..nb-1) and no kernel
    /// runs — the bit-identical no-drop path. The result is model-INDEPENDENT (a property
    /// of the loaded f2), so callers compute it once per assemble/bucket.
    [[nodiscard]] std::vector<int> device_survivor_blocks(
        const steppe::device::DeviceF2Blocks& f2, int nb, int P);

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
        const Precision& precision, int k);

    /// S3 — assemble the per-block f4 matrix X from DEVICE-RESIDENT f2 (zero D2H of
    /// the big tensor; the FROZEN CONTRACT §2a). The gather kernel reads
    /// f2.f2_device() in VRAM; the est_to_loo/x_total/tot_line reduction runs
    /// on-device; only the small X/loo/total cross the seam. Native FP64.
    [[nodiscard]] F4Blocks assemble_f4(const steppe::device::DeviceF2Blocks& f2,
                                       std::span<const int> left_idx,
                                       std::span<const int> right_idx,
                                       const Precision& precision) override;

    /// S3 host-oracle overload — NOT the GPU path. The CudaBackend implements only
    /// the DeviceF2Blocks form (zero D2H of the resident tensor); a host
    /// F2BlockTensor would force the big tensor onto the device only to read it back,
    /// which is the CpuBackend's oracle door, not the production GPU path.
    [[nodiscard]] F4Blocks assemble_f4(const F2BlockTensor& f2,
                                       std::span<const int> left_idx,
                                       std::span<const int> right_idx,
                                       const Precision& precision) override;

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
        const Precision& precision) override;

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
        double setmiss_thresh, const Precision& precision) override;

    /// FUSED dstat block-reduce→ratio-jackknife (M2 host-compute-audit cure) — the device-resident
    /// producer. Runs the per-(quadruple,block) D reduce keeping dNum/dDen/dCnt RESIDENT and feeds
    /// them straight to launch_ratio_block_jackknife — DROPPING the numsum/densum/cnt D2H the
    /// dstat_block_reduce path does (cuda_backend.cu:1445-1450). Only the small N-length
    /// est/se/z/p cross back. tot_mode=1, weight=cnt, setmiss_thresh=0 (cnt>0 mask), compute_p=true
    /// (p = f4_two_sided_p(z) on-device). Native FP64 (the §12 carve-out; matches the host
    /// long-double dstat_jackknife operand order).
    [[nodiscard]] RatioBlockJackknife dstat_blocks_jackknife(
        const steppe::device::DeviceDecodeResult& dec, const int* block_id, int n_block,
        std::span<const int> quadruples) override;

    /// FUSED f4ratio host-oracle overload — NOT the GPU path. The CudaBackend reads
    /// DEVICE-RESIDENT f2 (the DeviceF2Blocks form); the host-tensor overload is the CpuBackend
    /// oracle door (sibling of assemble_f4_quartets(host) throwing).
    [[nodiscard]] RatioBlockJackknife f4ratio_blocks_jackknife(
        const F2BlockTensor& f2, std::span<const int> flat, int N, double setmiss_thresh,
        const Precision& precision) override;

    /// FUSED dstat host-pointer overload — NOT the GPU path. The CudaBackend reads the
    /// DEVICE-RESIDENT decode (the DeviceDecodeResult form, dropping the numsum/densum/cnt D2H);
    /// the host-pointer Q/V overload is the CpuBackend oracle door.
    [[nodiscard]] RatioBlockJackknife dstat_blocks_jackknife(
        const double* Q, const double* V, int P, long M, const int* block_id, int n_block,
        std::span<const int> quadruples) override;

    /// STANDALONE f4 host-oracle overload — NOT the GPU path (sibling of assemble_f4(host));
    /// the CudaBackend implements only the DeviceF2Blocks form (zero D2H of the tensor).
    [[nodiscard]] F4Blocks assemble_f4_quartets(const F2BlockTensor& f2,
                                                std::span<const int> quartets,
                                                const Precision& precision) override;

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
        const Precision& precision) override;

    /// STANDALONE f3 host-oracle overload — NOT the GPU path (sibling of
    /// assemble_f4_quartets(host)); the CudaBackend implements only the DeviceF2Blocks form.
    [[nodiscard]] F4Blocks assemble_f3_triples(const F2BlockTensor& f2,
                                               std::span<const int> triples,
                                               const Precision& precision) override;

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
                                                 const Precision& precision) override;

    /// HETEROGENEOUS-TOPOLOGY FLEET (the qpGraph topology SEARCH; qpgraph_search.hpp). Pack
    /// EVERY candidate topology's path-table arena into ONE device buffer + a per-topology
    /// index table, size the per-thread scratch to the BATCH-MAX layout, and fit ALL topos
    /// in ONE launch reading the SAME resident f_obs/qinv (the basis is pop-set-bound, not
    /// topology-bound — uploaded ONCE here). The host reduces the per-(topo,restart) scores
    /// to the per-topology best + spread (a reduction, NOT a per-candidate host fit — the
    /// AT2 optim()-per-candidate CPU trap designed out). D==0 trees are scored IN-KERNEL.
    [[nodiscard]] QpGraphFleetBatch qpgraph_fit_fleet_batch(
        const std::vector<QpGraphTopoArena>& topos, std::span<const double> f_obs,
        std::span<const double> qinv, int numstart, int maxit, double tol,
        const Precision& precision) override;

    /// GPU-ONLY f4 sweep — enumerate EVERY C(P,4) quartet on the device, compute f4 + diagonal
    /// jackknife SE + |z|, filter + CUB-compact survivors ON THE DEVICE, return ONLY survivors.
    /// k=4; delegates to the shared run_fstat_sweep_device. (The fix for the CPU-bound disaster.)
    [[nodiscard]] SweepSurvivors f4_sweep(const steppe::device::DeviceF2Blocks& f2,
                                          const SweepConfig& cfg,
                                          const Precision& precision) override;

    /// GPU-ONLY f3 sweep — the three-slab sibling (every C(P,3) triple). k=3.
    [[nodiscard]] SweepSurvivors f3_sweep(const steppe::device::DeviceF2Blocks& f2,
                                          const SweepConfig& cfg,
                                          const Precision& precision) override;

    /// S4 — weighted block-jackknife covariance Q + Qinv on the GPU (the FROZEN
    /// CONTRACT §2b). xtau kernel (native; cancellation carve-out) → cublasDsyrk Q
    /// (well-conditioned matmul; ENGAGES `precision` — emulated{40} by default, auto-
    /// native fallback) → fudge diag → cusolverDn Cholesky potrf/potri Qinv (ill-
    /// conditioned; native FP64, the cuSOLVER fallback). devInfo>0 ⇒
    /// NonSpdCovariance (value, not throw).
    [[nodiscard]] JackknifeCov jackknife_cov(const F4Blocks& x,
                                             std::span<const int> block_sizes,
                                             double fudge,
                                             const Precision& precision) override;

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
                                               const Precision& precision) override;

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
    static bool model_fits_small_path(int nl, int nr, int r);

    /// True iff the LARGE-path dense SVD routes to cuSOLVER's one-sided Jacobi
    /// (gesvdj) rather than gesvd: BOTH dims <= kGesvdjMaxDim (32). The SINGLE source
    /// of the dispatch predicate so large_svd_V's branch AND the svd_path
    /// observability report (rank_sweep) cannot disagree on the executed routine — a
    /// drift the NRBIG parity test (svd_path==2) would otherwise silently mask
    /// (group-5 5.3/5.5). svd_path = gesvdj_applicable ? 1 : 2.
    static bool gesvdj_applicable(int nl, int nr);

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
    [[nodiscard]] SvdScratchSizes large_svd_scratch_sizes(int nl, int nr);

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
                     int* sInfo, double* sWork, int lwork, cudaStream_t stream);

    /// Single-shot convenience overload: grow the persistent per-backend gesvd scratch
    /// members to the queried per-shape sizes (B5 pool; monotonic-grow, never-shrink)
    /// and delegate to the scratch-taking form above — instead of a fresh stack
    /// DeviceBuffer set freed via a device-wide `cudaFree` on every call. For the
    /// per-CALL large fit (`large_fit_one`) this is one small-LA burst per fit, not a
    /// per-block inner loop (see the §14.2 finding's "OTHER hot paths" note); the
    /// Stage-A LOO SWEEP must NOT use this overload — it hoists its OWN scratch above
    /// its loop and calls the scratch-taking form, so it neither pays a device-wide
    /// `cudaFree` per block nor touches these pooled members.
    void large_svd_V(const double* dXmat, int nl, int nr, int r,
                     double* dVout, double* dXt, cudaStream_t stream);

    /// Dynamic VRAM scratch sizes for the large-path ALS + weight/chisq (§2.3).
    /// dbl: the union of the ALS layout (xvec[m] | Wm[m*t] | coeffs[t*t] | rhs[t] |
    /// tmp[t] | lu[t*t] | y[t], t = max(nl,nr)*r) and the weight/chisq layout
    /// (RHS[nl*nl] | wv[nl] | lu[nl*nl] | y[nl] | e[m]); take the max so ONE buffer
    /// serves both kernels. int: max(t, nl) pivots.
    static std::size_t large_dbl_scratch(int nl, int nr, int r);
    static std::size_t large_int_scratch(int nl, int nr, int r);

    /// Per-refit DOUBLE scratch stride for the PARALLEL large-LOO kernel. Each thread
    /// owns its xmat[m] + A[nl*r] + B[r*nr] + the als/weights union (large_dbl_scratch),
    /// so one thread's slice is self-contained (no cross-thread aliasing ⇒ deterministic).
    static std::size_t large_loo_dbl_refit(int nl, int nr, int r);

    /// LARGE-path single-model fit: cuSOLVER SVD seed (r>0) → ALS opt_A/opt_B → the
    /// constrained weight solve + chisq, all RESIDENT on stream_. The caller supplies
    /// the device buffers (dXmat/dQinv/dA/dB/dW/dchisq/dStatus + the VRAM scratch
    /// dVout/dXt/dScratch/dIntScratch). Mirrors the small-path seed→als→weights triple.
    void large_fit_one(const double* dXmat, const double* dQinv, int nl, int nr, int r,
                       double fudge, int als_iters, double* dA, double* dB, double* dW,
                       double* dchisq, int* dStatus, double* dVout, double* dXt,
                       double* dScratch, int* dIntScratch, cudaStream_t stream);

    /// CAPABILITY QUERY (backend.hpp): the CudaBackend deliverable DOES override
    /// `rank_sweep` (below) ⇒ true, so the host orchestrator runs the rankdrop/popdrop
    /// path on the GPU (the explicit replacement for the old sentinel-catch detection).
    [[nodiscard]] bool provides_rank_sweep() const override;

    /// S5 SWEEP — the qpWave / qpAdm RANK TEST over r = 0..rmax (rmax = min(nl,nr)-1)
    /// ON THE GPU (M(fit-2), THE deliverable). Mirrors the CpuBackend oracle
    /// (cpu_backend.cpp rank_sweep) op-for-op but runs the per-rank fit on the device:
    /// upload x_total + Qinv ONCE; build dXmat ONCE; then for each r run the SAME
    /// M(fit-4) device machinery (seed → ALS → weights+chisq) and read back chisq(r).
    /// The host then forms dof(r) = (nl-r)*(nr-r), p(r) = pchisq_upper, the AT2
    /// res$rankdrop nested table (rows f4rank DESCENDING; the nested diff to the
    /// next-lower rank; the last row NA), f4rank (the smallest non-rejected rank), and
    /// rank_Q (numerical rank of Q via the ON-DEVICE one-sided Jacobi — observability
    /// only, bit-identical to the oracle; NOT on the rank-test math path; L1). The
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
                                       const Precision& precision) override;

    /// S6 — GLS weights via AT2 ALS on the GPU (the FROZEN CONTRACT §2d). xmat from
    /// x_total → on-device SVD seed → opts.als_iterations (default 20) ALS opt_A/opt_B
    /// iters → constrained weight
    /// solve → normalize Σw=1 → chisq. All on-device, native FP64.
    [[nodiscard]] GlsWeights gls_weights(const F4Blocks& x,
                                         const JackknifeCov& cov,
                                         int r,
                                         const QpAdmOptions& opts,
                                         const Precision& precision) override;

    /// S7 — BATCHED leave-one-block-out weight re-fits on the GPU (the FROZEN
    /// CONTRACT §2e). Upload x_loo + Qinv ONCE; one batched device launch runs all
    /// nb per-block fits (xmat from loo[:,:,b] → seed → ALS → weight solve →
    /// normalize), REUSING Qinv unchanged (the AT2 parity pin). Returns wmat
    /// [nb*nl] row-major. This replaces the 708 host gls_weights calls with a single
    /// batched device kernel (NOT a host loop). Native FP64.
    [[nodiscard]] std::vector<double> gls_weights_loo_batched(
        const F4Blocks& x, const JackknifeCov& cov, int r,
        const QpAdmOptions& opts, const Precision& precision) override;

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
        const QpAdmOptions& opts, const Precision& precision) override;

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
                                    DeviceBuffer<double>& dWmat);

public:
    /// CAPABILITY QUERY (backend.hpp): the CudaBackend DOES override
    /// `fit_models_batched` (below) with the genuine model-BATCHED rotation ⇒ true, so
    /// the S8 shard dispatch (model_search.cpp) routes small-path models to the batched
    /// override (the explicit replacement for the old sentinel-catch detection).
    [[nodiscard]] bool provides_batched_fit() const override;

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
        const Precision& precision) override;

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
                        Precision::Kind tag, std::vector<QpAdmResult>& results);

    /// One BATCHED chunk of B same-shape models — the genuine batched dispatch.
    /// F1 / OQ-12: `nb`/`dBlockSizes` are the SURVIVOR set; `d_surv` (nullptr=no drop)
    /// compacts the gather onto the survivor blocks (AT2 read_f2(remove_na=TRUE)).
    void fit_chunk(const steppe::device::DeviceF2Blocks& f2,
                   std::span<const QpAdmModel> models,
                   const std::vector<std::size_t>& mem, std::size_t off, std::size_t B,
                   int nl, int nr, int r_fit, int rmax, int m, int nb, int P, double n,
                   const int* dBlockSizes, const int* d_surv, const QpAdmOptions& opts,
                   const Precision& precision, Precision::Kind tag,
                   std::vector<QpAdmResult>& results);

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
                         const double* pop_wfull, double rank_alpha, QpAdmResult& res);

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
    // single-stream-per-device determinism). An OWNING, non-blocking RAII `Stream`
    // (stream.hpp; cudaStreamNonBlocking) created in the ctor AFTER device_id_ made this
    // device current; every launch / cudaMemcpyAsync / cudaStreamSynchronize routes through
    // it. Why non-blocking + the M4.5 SPMG overlap rationale:
    // see docs/design/cuda-backend-stream.md (architecture.md §7, §11.4, §12).
    //
    // Declaration order is LOAD-BEARING at teardown (reverse-order destruction): stream_ is
    // declared BEFORE blas_/solver_ so it is destroyed AFTER them — their cuBLAS/cuSOLVER
    // contexts are bound to this stream, so the stream must outlive their cublas/cusolver
    // destroy (workspace_ is declared after blas_ so it frees first; blas_ holds a non-owning
    // pointer into it). Stream choice moves no arithmetic bits, so §12 parity is unaffected.
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

    // POOLED per-backend cuSOLVER scratch (B5 — group-8 robustness polish). The
    // potrf/potri `lwork` workspace and the single-shot gesvd/gesvdj scratch were
    // each allocated PER CALL as RAII DeviceBuffers, every one paying a device-wide
    // cudaMalloc + a device-wide cudaFree sync. cuSOLVER itself does NO internal
    // cudaMalloc — "the user must allocate the device workspace explicitly" (CUDA
    // 13.x cuSOLVER) — and that workspace is uninitialized scratch the routine fully
    // OVERWRITES, while `*_bufferSize` is data-independent ("device pointer is not
    // used to decide the size of workspace"). So pooling these into persistent
    // members that grow MONOTONICALLY to the max size seen (never shrink) drops the
    // cuSOLVER malloc/free count to near-zero after warmup while the bytes fed to
    // every routine are UNCHANGED ⇒ bit-identical (§12); only the allocation count
    // moves. SAFE without a lock: each device owns its OWN CudaBackend
    // (model_search.cpp fans out resources.gpus[g].backend per jthread) and §12 pins
    // ONE statistic stream per backend, so the pooled scratch is used strictly
    // SEQUENTIALLY on a single thread, never shared across threads. These are a
    // DISTINCT set from the f2 D2H staging (stage_f2_/stage_vpair_, pinned host) so
    // the two never alias. `solver_work_` is the shared `lwork` buffer for potrf,
    // potri AND the single-shot gesvd/gesvdj solve; svd_s_/svd_u_/svd_vt_/svd_a2_/
    // svd_info_ are that solve's output/scratch set (see large_svd_scratch_sizes).
    // The Stage-A LOO sweep keeps its OWN hoisted arenas (§14.2) and the
    // scratch-taking large_svd_V overload — it does NOT use these members.
    DeviceBuffer<double> solver_work_{};  // potrf/potri/gesvd lwork scratch (max lwork seen)
    DeviceBuffer<double> svd_s_{};        // gesvd singular values  (min(nl,nr))
    DeviceBuffer<double> svd_u_{};        // gesvd economy U        (nl*nr)
    DeviceBuffer<double> svd_vt_{};       // gesvd right vectors    (cols*cols)
    DeviceBuffer<double> svd_a2_{};       // gesvd non-const A copy  (nl*nr when nl>nr)
    DeviceBuffer<int>    svd_info_{};     // gesvd devInfo          (1 int)
};

}  // namespace steppe::device

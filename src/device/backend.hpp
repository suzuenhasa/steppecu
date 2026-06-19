// src/device/backend.hpp
//
// ComputeBackend — the dependency-injection seam between `core` orchestration
// and the device layer (architecture.md §4, §8; ROADMAP §2, M0).
//
// THIS HEADER IS CUDA-FREE BY CONTRACT. It is the only door `core` uses to reach
// the GPU: `core` is pure host C++20 and never includes a CUDA header or calls
// cuBLAS/cuSOLVER directly (architecture.md §2, §4). CUDA is PRIVATE to
// steppe_device, so this interface compiles into `core` and the CLI without
// dragging in the device toolkit. Two implementations satisfy it — `CudaBackend`
// (the 3-GEMM reformulation on the GPU) and `CpuBackend` (the scalar reference
// oracle) — and the compute layer is written once against the interface, never
// branching on GPU-vs-CPU (architecture.md §8). The CPU backend is both the DRY
// reference and the correctness anchor the GPU is continuously diffed against
// (architecture.md §13; ROADMAP §5).
//
// M0 scope: the minimal-but-real method is `compute_f2` — compute the f2 matrix
// from the Q/V/N contract at a given Precision, returning f2 [P × P] and the
// pairwise-valid-SNP count Vpair [P × P]. Vpair is RETAINED, not discarded: it
// is the weighted-block-jackknife weight at S4 (architecture.md §5 S2 caveat
// (a)). M1/M4 add `decode_af` / `compute_f2_blocks`; M4.5 adds the CUDA-free
// `capabilities()` probe + `BackendCapabilities` POD that make this seam
// single-node-multi-GPU-ready (architecture.md §9, §11.4; cleanup 00-overview
// §(2)). Later milestones add gemm / jackknife / svd methods here.
#ifndef STEPPE_DEVICE_BACKEND_HPP
#define STEPPE_DEVICE_BACKEND_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "steppe/config.hpp"        // steppe::Precision
#include "steppe/fstats.hpp"        // steppe::F2BlockTensor (the M4 deliverable)
#include "core/internal/views.hpp"  // steppe::core::MatView (Q/V/N contract)
#include "device/device_partial.hpp"  // steppe::device::DevicePartial (CUDA-free opaque resident handle)
#include "device/device_f2_blocks.hpp"  // steppe::device::DeviceF2Blocks (CUDA-free opaque FULL device-resident result handle)

namespace steppe {

/// Output of an f2 computation: the symmetric f2 matrix and the pairwise-valid
/// SNP counts, both column-major [P × P] (element (i,j) at `i + P·j`). Plain
/// host vectors so the type crosses the CUDA-free seam (architecture.md §4);
/// the CUDA backend copies its device results into these before returning.
struct F2Result {
    /// f2 matrix, column-major [P × P]. Bias-corrected, AT2-unbiased estimator
    /// f2(i,j) = mean over jointly-valid SNPs of (p_i − p_j)² − hc_i − hc_j.
    ///
    /// DIAGONAL CONVENTION (pinned; cleanup X-2/B4): f2(i,i) carries the FULL
    /// (i,i) computation, NOT a forced 0. With p_i == p_i the per-SNP summand is
    /// (p_i − p_i)² − hc_i − hc_i = −2·hc_i, so f2(i,i) = −2·mean within-pop het
    /// correction (generally nonzero, and a within-pop het quantity, NOT a
    /// between-pop f2). BOTH backends agree on the diagonal by construction: the
    /// GPU assemble_f2_kernel writes every (i,j) with no i==j guard, and the CPU
    /// oracle loops j = i. The diagonal is never consumed downstream (f3/f4 read
    /// off-diagonal f2 only) but is kept consistent across backends so the §13
    /// oracle≡GPU diff at the F2Result seam compares the FULL matrix, diagonal
    /// included, and an M7 F2Result round-trip cannot reintroduce a backend split.
    std::vector<double> f2;

    /// Pairwise-valid SNP count, column-major [P × P]: Vpair(i,j) = number of
    /// SNPs valid in BOTH i and j. RETAINED as the S4 jackknife weight
    /// (architecture.md §5 S2 caveat (a)); the per-pair divide and S4 weighting
    /// must compose to AT2's f2_blocks definition, not double-normalize. The
    /// DIAGONAL Vpair(i,i) is i's own valid-SNP count (the i==j case of "valid in
    /// both"); like f2(i,i) it is filled, not 0, and agrees across backends
    /// (cleanup X-2/B4).
    std::vector<double> vpair;

    /// Number of populations P (the leading dimension of both matrices).
    int P = 0;
};

/// A CUDA-FREE, NON-OWNING view of one packed genotype tile + its population
/// partition — the input to `decode_af` (architecture.md §5 S0/S1; ROADMAP M1).
/// This is the plain-data seam between the `io` leaf (which produces an
/// io::GenotypeTile of the same shape) and the device/CPU decode backend: `io`
/// does NOT depend on `device`, and `backend.hpp` does NOT depend on the `io`
/// leaf — `app`/the test bridge them by filling this view from a GenotypeTile.
/// All pointers are borrowed (the caller owns the storage for the call's
/// duration); no CUDA type appears.
///
/// TGENO individual-major packing: `packed` holds `n_individuals` records of
/// `bytes_per_record` bytes each, gathered POPULATION-CONTIGUOUS; SNP `s` of
/// gathered individual `g` is the 2-bit code at byte `g*bytes_per_record + s/4`,
/// position `s%4` (MSB-first). Population `p` owns gathered individuals
/// `[pop_offsets[p], pop_offsets[p+1])`. `n_pop == pop_offsets length − 1`.
struct DecodeTileView {
    const std::uint8_t* packed = nullptr;   ///< packed bytes, pop-contiguous records
    std::size_t bytes_per_record = 0;       ///< stride between individual records
    std::size_t n_snp = 0;                  ///< SNPs in the tile (the column axis M)
    std::size_t n_individuals = 0;          ///< gathered individuals across all pops
    const std::size_t* pop_offsets = nullptr;  ///< P+1 segment boundaries (sample axis)
    int n_pop = 0;                          ///< number of populations P (row axis)

    /// Per-sample PLOIDY factor for N: 2 diploid (the AADR case — every sample is
    /// diploid, so every N is even), 1 pseudo-haploid (ancient DNA). A METADATA
    /// parameter, NEVER auto-detected from genotypes (ROADMAP Q/V/N contract).
    /// Default 2 (diploid), documented.
    int ploidy = 2;
};

/// Output of a genotype-decode: the Q/V/N contract as plain column-major [P × M]
/// host arrays (element (pop i, snp s) at `i + P·s` — the views.hpp layout that
/// drops straight into MatView/compute_f2). Plain host vectors so the type
/// crosses the CUDA-free seam (architecture.md §4); the CUDA backend copies its
/// device results into these before returning.
struct DecodeResult {
    /// Reference-allele frequency in [0,1], 0 where invalid. Column-major [P × M].
    std::vector<double> q;
    /// Validity mask (1.0 valid / 0.0 missing). Column-major [P × M].
    std::vector<double> v;
    /// Non-missing haploid count (ploidy × non-missing individuals). [P × M].
    std::vector<double> n;
    /// Number of populations P (the leading dimension of Q/V/N).
    int P = 0;
    /// Number of SNPs M (the column count of Q/V/N).
    long M = 0;
};

/// CUDA-FREE probe result describing the ONE device a backend instance is bound
/// to, plus whether that device can peer-access the other visible devices — the
/// capability-tier datum the M4.5 single-node multi-GPU machinery reads
/// (architecture.md §9 DeviceConfig/Resources, §11.4 SPMG capability-tiered
/// combine, §12 parity; cleanup 00-overview §(2).1 "the ONE unified design",
/// device-backend §11.1/§11.2). Plain old data so it crosses the CUDA-free seam
/// (architecture.md §4) and slots by value into `Resources`/a result envelope.
///
/// WHY OUT-OF-BAND, NEVER ON `F2BlockTensor`. The capability TAG (which combine
/// path a run took, why it degraded) is recorded in `Resources`/the run record,
/// NEVER on the numeric payload (`F2BlockTensor` stays pure numeric storage so
/// the §12 parity diff compares only bits that the math produced). This struct is
/// the *probe input* to that out-of-band tag; it carries no statistic.
///
/// PARITY-NEUTRAL BY CONSTRUCTION. Every field here drives a data-movement /
/// observability lever only (which combine transport, which precision lane is
/// honorable) — never the arithmetic. §12 parity holds identically whether
/// `can_access_peer` is true (PRO 6000 device-resident `cudaMemcpyPeer` combine)
/// or false (the host-staged fixed-order combine baseline): both sum the same
/// fixed `g=0..G-1` device order, so the reported numbers are bit-identical on
/// both tiers (architecture.md §11.4, §12).
///
/// The real probe (the values these fields actually take on a device) is asserted
/// in the CUDA path's probe test; this header only fixes the SHAPE and the
/// value-initialized "unknown" defaults (every field zero/false ⇒ "nothing
/// probed yet", the contract `ComputeBackend::capabilities()`'s base returns).
struct BackendCapabilities {
    /// Number of CUDA devices VISIBLE to this process (`cudaGetDeviceCount`). The
    /// SPMG combine fans out over the subset pinned by `DeviceConfig::devices`
    /// (architecture.md §9, §11.4); this is the upper bound the probe saw. 0 ⇒
    /// unknown / no CUDA device (the CPU backend / value-initialized default).
    int device_count = 0;

    /// Compute capability of THE device this backend instance is bound to
    /// (`cudaDeviceProp::major`/`minor`; sm_120 ⇒ {12, 0} on the Blackwell box).
    /// One build (sm_120) serves both boxes (architecture.md §0; cleanup ⚡
    /// box-role split), so this is observability, not a dispatch key. {0,0} ⇒
    /// unknown.
    int compute_major = 0;
    int compute_minor = 0;

    /// Total / currently-free VRAM in bytes on the bound device (`cudaMemGetInfo`,
    /// which yields BOTH — `cuda_backend.cu` historically discarded `total`; the
    /// M4.5 probe captures it, cleanup 00-overview §(2).1). `total` feeds the
    /// §11.2 VRAM budget / per-box P_max; `free` is the live headroom. 0 ⇒ unknown.
    std::size_t total_vram_bytes = 0;
    std::size_t free_vram_bytes  = 0;

    /// Whether the bound device (conventionally GPU 0, the combine root) can
    /// PEER-ACCESS the other visible devices — `cudaDeviceCanAccessPeer`
    /// (architecture.md §11.4, §9 `enable_peer_access`). The capability-tier law
    /// (architecture.md §11.4; workflow wxz1fiiln): true is the PRO 6000 /
    /// datacenter-Blackwell stock-driver case (the device-resident `cudaMemcpyPeer`
    /// combine fast-path); false is EXPECTED on consumer GeForce (P2P
    /// driver-disabled) and triggers the NON-throwing tagged degrade to the
    /// host-staged fixed-order combine baseline — never a fault. false ⇒ no peer
    /// access (also the value-initialized default).
    bool can_access_peer = false;

    /// Whether `EmulatedFp64` is HONORABLE on this build/device — i.e. the
    /// fixed-slice Ozaki tuning API is present so cuBLAS pins the FIXED mantissa
    /// rather than silently falling back to the DYNAMIC ~60-bit trap (architecture
    /// .md §12; cleanup X-6/B2 `emulation_honorable`, `STEPPE_HAVE_EMU_TUNING`).
    /// false ⇒ an `EmulatedFp64` request must degrade to native `Fp64` with a
    /// logged capability tag (never run dynamic under the `EmulatedFp64` tag).
    /// false is also the value-initialized "unknown" default.
    bool emulated_fp64_honorable = false;
};

/// Abstract compute backend. One interface, two implementations (CUDA, CPU
/// reference). All device operations route through here; `core` never issues a
/// GEMM/SVD/Cholesky itself (architecture.md §2, §8). Move-only ownership of
/// concrete backends is by `std::unique_ptr<ComputeBackend>` in `Resources`
/// (architecture.md §9).
///
/// PER-DEVICE-INSTANCE CONTRACT (architecture.md §9 PerGpuResources, §11.4 SPMG;
/// cleanup device-backend §11.2). One backend instance is bound to exactly ONE
/// CUDA device — the single-process-multi-GPU model is ONE backend (and one
/// `PerGpuResources`: stream + cuBLAS/cuSOLVER handle) PER device in
/// `DeviceConfig::devices`, constructed with that device's id and
/// `cudaSetDevice`-bound to it. The interface is therefore SINGLE-DEVICE: SNP
/// sharding across the G devices and the host-side fixed-order combine of their
/// per-device full-shape partials are orchestrated ABOVE this seam (by
/// `Resources`/the streamer, architecture.md §11.4), NOT inside any one method —
/// that combine algorithm is the next workflow, not implemented here.
class ComputeBackend {
public:
    ComputeBackend() = default;
    ComputeBackend(const ComputeBackend&) = delete;
    ComputeBackend& operator=(const ComputeBackend&) = delete;
    ComputeBackend(ComputeBackend&&) = delete;
    ComputeBackend& operator=(ComputeBackend&&) = delete;
    virtual ~ComputeBackend() = default;

    /// Compute the bias-corrected f2 matrix and pairwise-valid counts from the
    /// Q/V/N contract (column-major [P × M] views, views.hpp) at the requested
    /// precision.
    ///
    /// @param Q  reference-allele frequencies in [0,1], zero-filled where
    ///           invalid (the zero is what makes the masked GEMM correct).
    /// @param V  validity mask (1.0 valid / 0.0 missing).
    /// @param N  non-missing haploid count (2 × diploids, or 1 × pseudo-haploids
    ///           for ancient DNA). Enters only the het correction.
    /// @param precision  governs the matmul-heavy f2 GEMMs ONLY (default
    ///           EmulatedFp64{40} ⇒ ≈ native, MEASURED 7–17× faster on real
    ///           AADR; ROADMAP §0). The small numerator/divide stays native
    ///           FP64 regardless (architecture.md §12). The CPU backend computes
    ///           the native-FP64 reference and ignores the matmul mode.
    ///
    /// Preconditions: Q, V, N share the same P and M and refer to the same SNP
    /// block. Returns f2 [P × P] + Vpair [P × P] (column-major). Both backends
    /// must agree at the f2/Vpair seam within the tight tolerance tier
    /// (architecture.md §12, §13) — the formula is shared via
    /// core/internal/f2_estimator.hpp so they cannot diverge.
    [[nodiscard]] virtual F2Result compute_f2(const core::MatView& Q,
                                              const core::MatView& V,
                                              const core::MatView& N,
                                              const Precision& precision) = 0;

    /// Compute the PER-BLOCK f2 tensor `f2_blocks [P × P × n_block]` + the retained
    /// per-block `Vpair` from the FULL per-SNP Q/V/N contract and a SNP→block
    /// assignment — the M4 deliverable (architecture.md §5 S2, §11.1; ROADMAP M4).
    ///
    /// The inputs Q/V/N are the SAME column-major [P × M] contract as `compute_f2`,
    /// but spanning ALL M SNPs (not one block). `block_id` is an M-vector from the
    /// shared `assign_blocks` (core/domain/block_partition_rule.hpp): `block_id[s]`
    /// is the dense jackknife block of SNP s, in `0 .. n_block-1`, non-decreasing in
    /// file order (so a block's SNPs are CONTIGUOUS — the property the batched
    /// reorder relies on). The GPU backend runs the spike-chosen SIZE-GROUPED
    /// strided-batched design over the blocks; the CPU backend is the per-block
    /// long-double oracle. `precision` governs only the matmul-heavy GEMMs
    /// (architecture.md §12); the numerator/divide stays native FP64.
    ///
    /// @param Q         reference-allele frequencies in [0,1], 0 where invalid,
    ///                  column-major [P × M] over ALL SNPs.
    /// @param V         validity mask (1.0 valid / 0.0 missing), [P × M].
    /// @param N         non-missing haploid count, [P × M] (enters het correction).
    /// @param block_id  per-SNP dense block id (length M), from assign_blocks;
    ///                  non-decreasing ⇒ each block's columns are contiguous.
    /// @param n_block   number of distinct blocks (== max(block_id)+1).
    /// @param precision governs the f2 GEMMs only (default EmulatedFp64{40}).
    /// @return  F2BlockTensor with f2 + Vpair [P × P × n_block] (block-major outer,
    ///          column-major within a block: `i + P·j + P·P·b`) + block_sizes.
    [[nodiscard]] virtual F2BlockTensor compute_f2_blocks(const core::MatView& Q,
                                                          const core::MatView& V,
                                                          const core::MatView& N,
                                                          const int* block_id,
                                                          int n_block,
                                                          const Precision& precision) = 0;

    /// M4.5 DEVICE-RESIDENT primary: compute the FULL per-block f2 tensor
    /// [P × P × n_block] + Vpair EXACTLY as compute_f2_blocks does (same GEMM body,
    /// bit-identical bits; §12), but LEAVE the result RESIDENT in VRAM and return a
    /// move-only DeviceF2Blocks handle — NO forced D2H, NO host alloc, NO host
    /// zero-fill. THE PRIMARY OUTPUT (architecture.md §11.1 "f2_blocks stays on the
    /// device"). compute_f2_blocks (host) is now a thin wrapper = this + .to_host().
    /// The ONLY difference from compute_f2_blocks is the result stays on the device
    /// instead of being copied to a host F2BlockTensor and freed.
    ///
    /// NON-PURE: the base throws (the device-resident output is a CUDA-backend
    /// concept; nothing routes the CPU backend / a GPU-free fake through it). Only
    /// the CUDA backend overrides it — CpuBackend / any fake need NOT.
    [[nodiscard]] virtual steppe::device::DeviceF2Blocks compute_f2_blocks_device(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, const Precision& precision) {
        (void)Q; (void)V; (void)N; (void)block_id; (void)n_block; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::compute_f2_blocks_device: not supported by this backend "
            "(device-resident output requires a CUDA backend)");
    }

    /// M4.5 device-resident variant of compute_f2_blocks: compute the per-block
    /// [P × P × n_block] partial EXACTLY as compute_f2_blocks does, but LEAVE the
    /// f2/Vpair tensors RESIDENT on this backend's device (NO D2H, NO free) and
    /// return an opaque, move-only DevicePartial owning them — the input to the
    /// device-resident cudaMemcpyPeer combine (device/p2p_combine.hpp). Bit-identical
    /// per-block bits to compute_f2_blocks over the same inputs (same GEMM body; §12):
    /// the ONLY difference is the result stays on the device instead of being copied
    /// to a host F2BlockTensor and freed.
    ///
    /// @param b0  the GLOBAL block placement offset for this partial (== shard.b0),
    ///            carried on the returned handle so the combine knows the disjoint
    ///            destination slice [b0, b0+n_block) without consulting the shard.
    /// NON-PURE: the base throws (it is reached only behind the four-term §4 gate,
    /// which requires caps.can_access_peer == true — only the CUDA backend reports
    /// that). CpuBackend / any fake need NOT override it.
    [[nodiscard]] virtual steppe::device::DevicePartial compute_f2_blocks_resident(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, int b0, const Precision& precision) {
        (void)Q; (void)V; (void)N; (void)block_id; (void)n_block; (void)b0; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::compute_f2_blocks_resident: not supported by this backend "
            "(device-resident combine requires a peer-capable CUDA backend; the §4 gate "
            "must have routed a non-CUDA/non-peer backend to the host-staged path)");
    }

    /// M4.5 host-staged-direct variant of compute_f2_blocks: compute the per-block
    /// [P × P × n_block] partial EXACTLY as compute_f2_blocks does (same GEMM body,
    /// bit-identical per-block bits; §12), but instead of allocating a fresh host
    /// F2BlockTensor and D2H-copying into it, D2H the compact f2/vpair slabs DIRECTLY
    /// into the caller-provided PINNED host destination at the disjoint block offset
    /// slab_off = (size_t)P*P*b0. The destination is ONE shared result owned by the
    /// orchestrator; this device owns the disjoint slice [slab_off, slab_off +
    /// P*P*n_block) and writes ONLY it (block-aligned shards are disjoint, so two
    /// devices' slices never overlap — concurrent D2H into one buffer is race-free).
    ///
    /// The backend page-locks [dst_f2+slab_off, ... +P*P*n_block) and the vpair slice
    /// for the D2H window (RegisteredHostRegion, graceful pageable degrade) so the two
    /// devices' D2Hs run as concurrent pinned DMAs. block_sizes for this device's blocks
    /// are written into block_sizes_dst[b0 .. b0+n_block) (host int).
    ///
    /// @param dst_f2          base pointer of the shared result f2 buffer (length >=
    ///                        P*P*n_block_full); the device writes [slab_off, +slab*nb).
    /// @param dst_vpair       base pointer of the shared result vpair buffer (same shape).
    /// @param block_sizes_dst base pointer of the shared result block_sizes (length >=
    ///                        n_block_full); the device writes [b0, b0+n_block).
    /// @param b0              the GLOBAL block placement offset for this partial (== shard.b0).
    /// NON-PURE: default base throws (only the CUDA backend implements it; the CPU
    /// backend / fakes need not override — but see §(4): the host test fake MUST).
    virtual void compute_f2_blocks_into(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, int b0,
        double* dst_f2, double* dst_vpair, int* block_sizes_dst,
        const Precision& precision) {
        (void)Q; (void)V; (void)N; (void)block_id; (void)n_block; (void)b0;
        (void)dst_f2; (void)dst_vpair; (void)block_sizes_dst; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::compute_f2_blocks_into: not supported by this backend");
    }

    /// Decode a packed genotype tile into the Q/V/N contract (architecture.md §5
    /// S0 Format decode + S1 Allele-freq reduction; ROADMAP M1). Unpacks the
    /// 2-bit codes (raw-value mapping: 0/1/2 ref-allele copies, 3 missing),
    /// reduces over the individuals within each population segment into integer
    /// AC (ref-allele count) / AN (non-missing individual count), then a single
    /// FP64 divide yields Q = AC/(ploidy·AN), N = ploidy·AN, V = (AN>0) — the EXACT
    /// oracle math, shared with the CPU reference via core/internal/decode_af.hpp
    /// so the two cannot diverge (architecture.md §13).
    ///
    /// @param tile  the packed tile + population partition + ploidy (CUDA-free
    ///              view; the `io` leaf fills it from an io::GenotypeTile).
    /// @return Q/V/N as column-major [P × M] host arrays (i + P·s), dropping
    ///         straight into MatView/compute_f2. Both backends must agree exactly
    ///         (N, V integer-valued ⇒ zero diff; Q via integer-accumulate AC/AN +
    ///         single FP64 divide ⇒ exact is the goal/gate).
    [[nodiscard]] virtual DecodeResult decode_af(const DecodeTileView& tile) = 0;

    /// Probe the capability tier of THE device this backend instance is bound to
    /// (the per-device-instance contract above): compute capability, total/free
    /// VRAM, whether this device can peer-access the other visible devices, and
    /// whether `EmulatedFp64` is honorable on this build (architecture.md §9,
    /// §11.4, §12; cleanup 00-overview §(2).1; device-backend §11.1). The result
    /// is recorded OUT-OF-BAND in `Resources`/the run record so the M4.5 combine
    /// picks the device-resident `cudaMemcpyPeer` fast-path vs the host-staged
    /// fixed-order baseline and logs which/why — the TAG never lands on the
    /// numeric `F2BlockTensor` (kept pure; see `BackendCapabilities`).
    ///
    /// NON-VIRTUAL-PURE on purpose: this has a DEFAULT base implementation
    /// returning a value-initialized (all-zero/false ⇒ "unknown") `BackendCapabilities`,
    /// so (a) `CpuBackend` — which has no device tier to report — need not override
    /// it, and (b) `backend.hpp` compiles standalone with no CUDA. Only the CUDA
    /// backend overrides it with the real probe (the probe values are asserted in
    /// the CUDA path's probe test, not here). The probe is NON-throwing: a
    /// "no peer access" answer (`cudaDeviceCanAccessPeer` == 0, EXPECTED on the
    /// budget GeForce tier) is a tagged degrade, not a fault (architecture.md
    /// §11.4 capability-tier law; cleanup 00-overview §(2).4).
    [[nodiscard]] virtual BackendCapabilities capabilities() const {
        return BackendCapabilities{};
    }
};

}  // namespace steppe

#endif  // STEPPE_DEVICE_BACKEND_HPP

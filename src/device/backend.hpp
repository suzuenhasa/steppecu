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
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "steppe/config.hpp"        // steppe::Precision
#include "steppe/error.hpp"         // steppe::Status (qpAdm domain-outcome POD fields)
#include "steppe/fstats.hpp"        // steppe::F2BlockTensor (the M4 deliverable)
#include "steppe/qpadm.hpp"         // steppe::QpAdmOptions (CUDA-free; the fit seam config)
#include "core/internal/views.hpp"  // steppe::core::MatView (Q/V/N contract)
#include "device/device_partial.hpp"  // steppe::device::DevicePartial (CUDA-free opaque resident handle)
#include "device/device_f2_blocks.hpp"  // steppe::device::DeviceF2Blocks (CUDA-free opaque FULL device-resident result handle)
#include "device/stream_f2_blocks.hpp"  // steppe::device::StreamTarget (CUDA-free M5 streamed-tier request)

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

// ---------------------------------------------------------------------------
// qpAdm fit-engine PODs (design docs/design/fit-engine.md §2; the M(fit-1)
// FROZEN CONTRACT §2). Plain CUDA-free PODs that cross the ComputeBackend seam by
// value, adjacent to F2Result. Flattening convention is fixed here ONCE so the
// jackknife/ALS/cross-checks cannot disagree on the vectorization order.
//
// VECTORIZATION ORDER (binding — it is AT2's `c(t(xmat))` order, design §4 S4/S6;
// verified against the golden by the R prototype): the m = nl*nr entries of the
// per-block f4 matrix are flattened ROW-MAJOR over (i,j): k = j + nr*i, with i in
// 0..nl-1 over left[-target] and j in 0..nr-1 over right[-R0]. AT2 builds Q with
// this same row-major order (jack_pairarr_stats: aperm(c(2,1,3)) then row-fill),
// and opt_A/opt_B index qinv with c(t(xmat)) = the same order. Using i + nl*j
// would silently transpose Q vs the X vector and break parity.
// ---------------------------------------------------------------------------

/// S3 output: the per-block f4 matrix X, flattened m=nl*nr per block, plus the
/// jackknife point estimate x_total (the AT2 $est vector).
struct F4Blocks {
    /// [m * n_block]: x_blocks[k + m*b] = vec(f4)[k] for block b, with the
    /// row-major k = j + nr*i convention above.
    std::vector<double> x_blocks;

    /// [m]: AT2 weighted-jackknife point estimate of f4 (the $est vector), same
    /// k = j + nr*i flatten.
    std::vector<double> x_total;

    /// [m * n_block]: the per-entry leave-one-out (est_to_loo) replicate values
    /// loo[k + m*b], carried so S7 re-fits per LOO block without recomputing S4.
    std::vector<double> x_loo;

    int nl = 0;       ///< = left.size()       (rows of the f4 matrix)
    int nr = 0;       ///< = right.size() - 1  (cols of the f4 matrix)
    int n_block = 0;
};

/// S4 output: the covariance Q and the inverse of the FUDGED Q.
struct JackknifeCov {
    /// [m*m] (symmetric, layout-agnostic), UNFUDGED (the golden Q convention).
    std::vector<double> Q;

    /// [m*m] inverse of the FUDGED Q (diag += fudge*tr(Q)).
    std::vector<double> Qinv;

    int m = 0;

    /// NonSpdCovariance if the fudged Q is not invertible (value, not throw).
    Status status = Status::Ok;
};

/// S6 output: the GLS weight fit.
struct GlsWeights {
    std::vector<double> w;  ///< [nl] normalized admixture weights (Σw = 1).
    std::vector<double> A;  ///< [nl*r] col-major refined left factor (rows=left, cols=rank).
    std::vector<double> B;  ///< [r*nr] col-major refined right factor.
    double chisq = 0.0;     ///< vec(E)' Qinv vec(E), E = X - A*B.
    int r = 0;

    /// RankDeficient if svd(X)/the constrained solve degenerates (value, not throw).
    Status status = Status::Ok;
};

/// S5 sweep output: the qpWave / qpAdm rank test over r = 0..rmax (rmax =
/// min(nl,nr)-1). Per-rank chisq/dof; the AT2 res$rankdrop nested table; f4rank
/// (the smallest non-rejected rank). All host-pure value data (CUDA-free seam).
/// Native FP64 (SVD + Qinv quadratic form are cuSOLVER/ill-conditioned; §1.4).
struct RankSweep {
    // ---- per-rank sweep (index = candidate rank r, 0..rmax) ----
    std::vector<double> chisq;   ///< [rmax+1] chisq(r) = vec(E_r)'Qinv vec(E_r) at the ALS-refined rank-r fit.
    std::vector<int>    dof;     ///< [rmax+1] (nl-r)*(nr-r).
    std::vector<double> p;       ///< [rmax+1] pchisq_upper(chisq(r), dof(r)).

    // ---- AT2 res$rankdrop nested table (rows = r descending: rmax..0) ----
    //   The AT2 rankdrop row order is f4rank DESCENDING (top row = highest rank
    //   tested = rmax; last row = 0). dofdiff/chisqdiff/p_nested compare row k to
    //   the NEXT row (k+1), i.e. rank (rmax-k) vs (rmax-k-1); the last row is NA.
    std::vector<int>    rd_f4rank;    ///< [rmax+1] the rank of this row (rmax, rmax-1, ..., 0).
    std::vector<int>    rd_dof;       ///< [rmax+1] dof at this rank.
    std::vector<double> rd_chisq;     ///< [rmax+1] chisq at this rank.
    std::vector<double> rd_p;         ///< [rmax+1] p at this rank.
    std::vector<int>    rd_dofdiff;   ///< [rmax+1] dof(next) - dof(this); INT_MIN ⇒ NA (last row).
    std::vector<double> rd_chisqdiff; ///< [rmax+1] chisq(next) - chisq(this); NaN ⇒ NA (last row).
    std::vector<double> rd_p_nested;  ///< [rmax+1] pchisq_upper(chisqdiff, dofdiff); NaN ⇒ NA (last row).

    int f4rank = 0;  ///< the smallest non-rejected rank (the AT2 res$f4rank); see §3 decision rule.
    int rank_Q = 0;  ///< numerical rank of Q (the golden model_well_determined.rank_Q).

    /// DISPATCH report (SPEC §5:231; observability only). 0 = on-device Jacobi
    /// (the path EXECUTED at these sizes by M(fit-4)); 1 = gesvdjBatched WOULD be
    /// selected (nl,nr<=32); 2 = per-model gesvd WOULD be selected (else). The
    /// executed SVD is the deterministic on-device Jacobi at all sizes in M(fit-2);
    /// the cuSOLVER routing is the documented PENDING seam (fit-engine.md §1.4).
    int svd_path = 0;

    Status status = Status::Ok;  ///< NonSpdCovariance / RankDeficient propagate as values.
};

/// S5 popdrop output: AT2 res$popdrop — leave-one-LEFT-SOURCE-out feasibility.
/// One row per pattern over the nl left SOURCES (AT2 keeps the full model row
/// "0..0" plus each single-drop). Built by the HOST orchestrator (it re-runs the
/// fit on the reduced left set), NOT a backend virtual — recorded here for the
/// contract shape (src/core/qpadm/ranktest.cpp).
struct PopDropRow {
    std::string         pat;        ///< AT2 bit pattern over the nl sources (e.g. "00","01","10"); "1"=dropped.
    int                 wt = 0;     ///< number of sources dropped (popcount(pat)).
    int                 dof = 0;
    double              chisq = 0.0;
    double              p = 0.0;
    int                 f4rank = 0;
    std::vector<double> weight;     ///< per-source weight for the SURVIVING sources (NaN for a dropped slot).
    bool                feasible = false;  ///< AT2 feasible: all surviving weights in [0,1] (the §3 rule).
    Status              status = Status::Ok;
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

class ComputeBackend;  // fwd for the fit_models_batched default delegate below

namespace core::qpadm {
/// The CUDA-FREE per-model default body of ComputeBackend::fit_models_batched
/// (defined in steppe_core, src/core/qpadm/model_search.cpp). It drives each model's
/// assemble_f4 → run_impl chain through `be`'s OWN device virtuals — so the device
/// backend runs resident batched GPU kernels and the CPU backend runs the scalar
/// oracle, with NO per-backend override. Declared here so the inline base virtual
/// can delegate to it without backend.hpp depending on the core qpadm headers
/// (the symbol is pulled in ONLY when the rotation is actually used).
[[nodiscard]] std::vector<QpAdmResult> fit_models_batched_default(
    ComputeBackend& be, const steppe::device::DeviceF2Blocks& f2,
    std::span<const QpAdmModel> models, const QpAdmOptions& opts);

/// CUDA-FREE host gate for the kQpMax* bit-parity envelope (nl<=kQpMaxNl=5,
/// nr<=kQpMaxNr=10, r<=kQpMaxR=4 — the SINGLE SOURCE: core/qpadm/qpadm_bounds.hpp).
/// The S8 orchestrator (run_qpadm_search, core) partitions the model list by this
/// predicate: small-path models route to the device-BATCHED virtual
/// fit_models_batched (the cuSOLVER-batched rotation primitive); the large/>32 tail
/// routes to the per-model fit_models_batched_default (one device dispatch per model
/// is correct for the tail, design §5). It delegates to the SAME qpadm_bounds.hpp
/// predicate that CudaBackend::model_fits_small_path dispatches on AND that sizes the
/// kernel per-thread arrays, so this host gate cannot drift wider than those arrays
/// (a wider gate would overflow them — UB). Declared here so the host orchestrator can
/// bucket WITHOUT naming the CUDA backend. nl = left.size(), nr = right.size()-1,
/// r = (opts.rank<0 ? nl-1 : opts.rank).
[[nodiscard]] bool model_in_small_path(const QpAdmModel& model, const QpAdmOptions& opts);

}  // namespace core::qpadm

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

    /// M5 STREAMED (out-of-core) per-block f2 — the HostRam + Disk tiers. Computes the
    /// FULL per-block [P × P × n_block] f2/Vpair EXACTLY as compute_f2_blocks_device does
    /// (same run_f2_blocks_resident prologue + per-block gather/GEMM/assemble; the
    /// per-block bits are BIT-IDENTICAL; §12), but instead of leaving the whole result
    /// resident it SPILLS each block's [P²] slab block-by-block through a triple-buffered
    /// sink into the tier `target` selects (HostRam: into target.host_dst; Disk: to
    /// target.disk_path, reopened read-only into target.disk_dst). The ONLY difference
    /// from the resident path is WHEN/WHERE a slab lands, never its bits. TIER 0
    /// (Resident) NEVER routes here — the orchestrator calls compute_f2_blocks_device.
    ///
    /// NON-PURE: the base throws (streaming is a CUDA-backend concept; the CPU backend /
    /// any fake need NOT override it). Only the CUDA backend overrides it — exactly the
    /// compute_f2_blocks_device pattern.
    virtual void compute_f2_blocks_streamed(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, const Precision& precision,
        steppe::device::StreamTarget& target) {
        (void)Q; (void)V; (void)N; (void)block_id; (void)n_block; (void)precision; (void)target;
        throw std::runtime_error(
            "ComputeBackend::compute_f2_blocks_streamed: not supported by this backend "
            "(out-of-core block streaming requires a CUDA backend)");
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

    // -----------------------------------------------------------------------
    // qpAdm fit-engine virtuals (design §1.7 + the M(fit-1) FROZEN CONTRACT §2).
    // BATCHED-CAPABLE by signature (a leading n_block / model axis) so the CUDA
    // backend (M(fit-4)) never needs a per-model/per-block host-loop retrofit.
    // NON-PURE: the base throws (the established backend.hpp pattern); CpuBackend
    // overrides all four with the native-FP64 reference. Each takes a `Precision`
    // that governs ONLY matmul sub-steps; the SVD/Cholesky/weight solves DEFAULT to
    // native FP64 (the §12 backend invariant) and are PROMOTABLE to EmulatedFp64 via
    // `set_solve_precision` under a validated per-stage policy (ROADMAP §6; the CUDA
    // backend's CusolverMathModeScope) — native FP64 remains the oracle/fallback. In
    // M(fit-1) the CpuBackend ignores `Precision` and is unconditionally FP64.
    // -----------------------------------------------------------------------

    /// SOLVE-PRECISION promotion knob (ROADMAP §6 the fit-solve promotion seam).
    /// The §12 conditioning rule pins the small ill-conditioned solves
    /// (SVD/Cholesky/GLS) native FP64 by DEFAULT — the matmul `Precision` passed to
    /// the virtuals above does NOT govern them. This setter is the per-stage seam
    /// that lets a caller PROMOTE a backend's solve stages to an emulated-FP64
    /// tensor-core path for the S8 rotation throughput wall (millions of small
    /// solves), validated per stage against the native oracle before any default
    /// flip. DEFAULT is native, so a backend that never calls this is byte-for-byte
    /// the M(fit-4) behavior (the af6a8c2 golden parity is unchanged). The base is a
    /// NO-OP (the CpuBackend oracle is unconditionally FP64 and ignores it); the
    /// CUDA backend overrides it to drive `CusolverMathModeScope` at the solve sites.
    /// EXPLORATORY/measurement use today (the non-gating emulated-40 parity probe);
    /// the S8 milestone wires the validated per-stage policy.
    virtual void set_solve_precision(const Precision& precision) { (void)precision; }

    /// S3 — assemble the per-block f4 matrix X from device-resident f2 (zero D2H on
    /// the CUDA path). BATCHED over n_block (the whole [m × n_block] tensor in one
    /// call) and resident-capable. left_idx[0] is the TARGET (L_0, prepended);
    /// right_idx[0] is R_0. AT2 identity (per block b), for i in 0..nl-1 (left
    /// source i), j in 0..nr-1 (right j):
    ///   X[i,j,b] = 0.5*( f2(left_idx[i+1], right_idx[0], b)
    ///                  + f2(left_idx[0],   right_idx[j+1], b)
    ///                  - f2(left_idx[0],   right_idx[0],   b)
    ///                  - f2(left_idx[i+1], right_idx[j+1], b) )
    /// nl = left_idx.size()-1, nr = right_idx.size()-1, m = nl*nr; flatten
    /// k = j + nr*i (row-major; the F4Blocks vectorization). Native FP64 (OQ-5).
    [[nodiscard]] virtual F4Blocks assemble_f4(const steppe::device::DeviceF2Blocks& f2,
                                               std::span<const int> left_idx,
                                               std::span<const int> right_idx,
                                               const Precision& precision) {
        (void)f2; (void)left_idx; (void)right_idx; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f4: not implemented by this backend");
    }

    /// S3 host-oracle overload — the SAME AT2 f4 combine reading a host
    /// F2BlockTensor directly (the M(fit-1) parity/oracle door; design §6 table:
    /// the CpuBackend reads host memory). Identical math to the DeviceF2Blocks
    /// overload; only the f2 storage differs (host vs VRAM). The GPU backend
    /// overrides the DeviceF2Blocks form (zero D2H); the CpuBackend overrides this.
    [[nodiscard]] virtual F4Blocks assemble_f4(const F2BlockTensor& f2,
                                               std::span<const int> left_idx,
                                               std::span<const int> right_idx,
                                               const Precision& precision) {
        (void)f2; (void)left_idx; (void)right_idx; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f4(host): not implemented by this backend");
    }

    /// S4 — weighted block-jackknife covariance Q[m × m] from the per-block X and
    /// the per-block jackknife WEIGHTS. OQ-3 RESOLVED: the weight is block_sizes[b]
    /// (AT2 block_lengths, the per-block SNP count) — NOT Vpair. Pipeline (AT2
    /// jack_pairarr_stats / est_to_loo, design §4 S4): the est_to_loo conversion to
    /// LOO replicates (carried on F4Blocks.x_loo), the AT2 jackknife `est` total,
    /// the xtau pseudo-values, Q = xtau·xtauᵀ/n_block (UNFUDGED), then Qinv =
    /// inverse(Q with diag += fudge·tr(Q)). BATCHED over the m axis. Native FP64.
    [[nodiscard]] virtual JackknifeCov jackknife_cov(const F4Blocks& x,
                                                     std::span<const int> block_sizes,
                                                     double fudge,
                                                     const Precision& precision) {
        (void)x; (void)block_sizes; (void)fudge; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::jackknife_cov: not implemented by this backend");
    }

    /// S5 — rank test: chisq = vec(E)'·Qinv·vec(E), E = X_total - A·B, dof =
    /// (nl-r)·(nr-r). Returns the SVD seed factors A (nl×r), B (r×nr) and the
    /// statistic for rank r. In M(fit-1) this is the SEED for S6's ALS at the
    /// single fitted rank; the qpWave rank sweep (multiple r) is M(fit-2).
    /// Deterministic; native FP64 (oracle-grade).
    [[nodiscard]] virtual GlsWeights rank_test(const F4Blocks& x,
                                               const JackknifeCov& cov,
                                               int r,
                                               const Precision& precision) {
        (void)x; (void)cov; (void)r; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::rank_test: not implemented by this backend");
    }

    /// S5 SWEEP — the qpWave / qpAdm rank test over ALL candidate ranks r = 0..rmax
    /// (rmax = min(nl,nr)-1), producing the per-rank chisq/dof/p, the AT2 res$rankdrop
    /// nested table, and f4rank (the smallest non-rejected rank). Reuses the rank-r ALS
    /// machinery (the rank_test seed → als refine → chisq quadratic form) per r; the
    /// SVD is recomputed per r (the deterministic on-device Jacobi is bit-identical on
    /// the same input, so per-r recompute is exact; the SVD-once kernel is an S8
    /// follow-on). `alpha` is the rank-decision significance (AT2 default 0.05; §3).
    /// The popdrop table is built by the HOST orchestrator (it re-gathers a reduced X),
    /// NOT here. NON-PURE: base throws (the established backend.hpp pattern). DISPATCH
    /// (SPEC §5:231): the backend REPORTS which SVD path the model WOULD take
    /// (gesvdjBatched iff nl,nr<=32 else per-model gesvd) on RankSweep.svd_path —
    /// observability; the executed SVD is the deterministic on-device Jacobi (M(fit-4))
    /// at these sizes. Native FP64 (SVD + Qinv quadratic form).
    [[nodiscard]] virtual RankSweep rank_sweep(const F4Blocks& x,
                                               const JackknifeCov& cov,
                                               double alpha,
                                               const QpAdmOptions& opts,
                                               const Precision& precision) {
        (void)x; (void)cov; (void)alpha; (void)opts; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::rank_sweep: not implemented by this backend");
    }

    /// S6 — GLS weights via AT2 ALS (OQ-1, the load-bearing primitive). Seed A,B
    /// from svd(X_total) at rank r; refine by opt_A/opt_B for opts.als_iterations
    /// (default 20) with the fudge ridge; then the CONSTRAINED weight solve on the
    /// (r+1)×(r+1) SPD system; normalize Σw=1. Returns w, refined A,B, and chisq.
    /// AT2 `qpadm.R` reproduced verbatim (design §4). Native FP64 throughout.
    [[nodiscard]] virtual GlsWeights gls_weights(const F4Blocks& x,
                                                 const JackknifeCov& cov,
                                                 int r,
                                                 const QpAdmOptions& opts,
                                                 const Precision& precision) {
        (void)x; (void)cov; (void)r; (void)opts; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::gls_weights: not implemented by this backend");
    }

    /// S7 — BATCHED leave-one-block-out weight re-fits (design §1.2 S7; AT2
    /// get_weights_covariance). For each block b in 0..n_block-1, re-solve the GLS
    /// weights using x.x_loo[:,:,b] as the xmat (REUSING cov.Qinv unchanged — the
    /// AT2 parity pin: AT2 does NOT re-invert per replicate). Returns the AT2
    /// replicate matrix wmat[n_block*nl], ROW-MAJOR (b*nl + i). The CUDA backend
    /// runs all n_block re-fits as BATCHED on-device solves (getrfBatched /
    /// getrsBatched, batchCount=n_block — NOT a host loop); the CpuBackend overrides
    /// with the per-block host loop (the bit-exact oracle). Native FP64. NON-PURE:
    /// the base throws (the established backend.hpp pattern). This is the batched-
    /// capable S7 seam that makes se_from_loo backend-agnostic (the host loop body
    /// moves into the CpuBackend override).
    [[nodiscard]] virtual std::vector<double> gls_weights_loo_batched(
        const F4Blocks& x, const JackknifeCov& cov, int r,
        const QpAdmOptions& opts, const Precision& precision) {
        (void)x; (void)cov; (void)r; (void)opts; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::gls_weights_loo_batched: not implemented by this backend");
    }

    /// S8 — BATCHED qpAdm fit over MANY same-shape models against ONE device-resident
    /// f2 (the rotation primitive; design §1.6 / the M(fit-6) FROZEN CONTRACT §2.2).
    /// Fits all `models` (a bucket of identical (nl,nr,r)) WHOLLY on this backend's
    /// device in ONE batched dispatch — NOT a per-model host loop: the S3 f4 gather +
    /// loo/total/xtau over a (k,b,MODEL) grid reading the resident f2 with per-model
    /// index arenas; the covariance Q via cublasDgemmStridedBatched (engages
    /// `precision`); the SPD inverse Qinv via cuSOLVER potrfBatched + potrsBatched
    /// across models; the rank sweep + weight solve + chisq + LOO-SE + popdrop via a
    /// MODEL-batched kernel (one thread per model, the proven loo_batched_kernel lift).
    /// `run_qpadm_search` calls THIS for the SMALL-path bucket (the rotation common
    /// case); the >32 / large tail it routes to fit_models_batched_default (one device
    /// dispatch per model is correct for the tail). Each result's model_index ECHOES
    /// models[i].model_index. Domain outcomes are per-result `status`
    /// (RankDeficient/NonSpdCovariance), NEVER a throw — record-and-continue (a search of thousands must not abort on one
    /// degenerate model). `precision` governs the matmul sub-steps (default
    /// EmulatedFp64{40}); SVD/Qinv/chi^2 native by the §1.4 carve-out + the
    /// set_solve_precision promotion seam.
    ///
    /// NON-PURE: the base provides a per-model DEFAULT (it delegates to
    /// core::qpadm::fit_models_batched_default — assemble_f4 → run_impl per model
    /// through THIS backend's own device virtuals). The CudaBackend OVERRIDES this with
    /// the genuine model-BATCHED device path (the deliverable). The CpuBackend does NOT
    /// override — it inherits the per-model default (a host loop is the CORRECT shape
    /// for the parity oracle, design §2.3). The orchestrator run_qpadm_search shards
    /// `models` across Resources::gpus and calls this once per device sub-span.
    ///
    /// The base body cannot name core::qpadm::fit_models_batched_default inline (that
    /// would pull steppe_core into every TU that instantiates the vtable, incl. host-
    /// only unit TUs), so it throws a SENTINEL; run_qpadm_search NEVER reaches it (it
    /// dispatches small-path models to the override and large-path models to
    /// fit_models_batched_default directly). The sentinel only fires if a future caller
    /// invokes the virtual on a backend that did not override it.
    [[nodiscard]] virtual std::vector<QpAdmResult> fit_models_batched(
        const steppe::device::DeviceF2Blocks& f2,
        std::span<const QpAdmModel> models,
        const QpAdmOptions& opts,
        const Precision& precision) {
        (void)f2; (void)models; (void)opts; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::fit_models_batched: this backend has no batched override "
            "(route through core::qpadm::fit_models_batched_default instead)");
    }

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

    /// S8 instrumentation (observability only; NOT on the numeric path): the number of
    /// BATCHED model-dispatches this backend has issued through fit_models_batched —
    /// ONE per same-shape (nl,nr,r) bucket (or VRAM sub-chunk), NOT one per model. The
    /// rotation test asserts this is << the model count (a few buckets, not thousands
    /// of per-model launches) to PROVE the rotation ran GPU-BATCHED, not as a per-model
    /// host loop. Base returns 0 (a backend with no batched override never increments
    /// it — the CpuBackend oracle). The CUDA backend increments it once per bucket
    /// chunk. Monotonic across calls on one instance.
    [[nodiscard]] virtual std::size_t batched_dispatch_count() const { return 0; }
};

}  // namespace steppe

#endif  // STEPPE_DEVICE_BACKEND_HPP

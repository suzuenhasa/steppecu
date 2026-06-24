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
    int n_block = 0;  ///< the SURVIVOR block count (== block_sizes.size()).

    /// [n_block]: the per-SURVIVOR-block SNP count (AT2 block_lengths) — the S4
    /// jackknife weight. F1 / OQ-12 (est_to_loo_nafix): a jackknife block in which
    /// ANY population pair has Vpair == 0 (no SNP jointly valid in both pops) is a
    /// MISSING block; AT2's `read_f2(remove_na=TRUE)` DROPS such blocks ENTIRELY
    /// (`keep = apply(f2,3,sum(!is.finite)==0)`) before the jackknife, then runs the
    /// standard est_to_loo / jack_pairarr_stats over the survivors — NOT impute-0
    /// (which biases f4 toward 0 + inflates variance). assemble_f4 therefore COMPACTS
    /// the f4 block arrays (x_blocks/x_loo) to the survivor blocks and carries the
    /// survivor SNP counts HERE (not the f2 source's full block_sizes), so jackknife_cov
    /// / the rank sweep / S7 consume the survivor set with no further change. With NO
    /// missing blocks (every existing maxmiss=0 golden) this equals the full f2.block_sizes
    /// and the path is byte-identical (the keep-mask is all-true). [AT2 resampling.R /
    /// io.R read_f2 remove_na; fit-engine.md OQ-12]
    std::vector<int> block_sizes;
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

/// qpfstats SMOOTHING-SOLVE output (the genotype-path joint f2 smoother;
/// include/steppe/qpfstats.hpp). The shared-factor batched least-squares solution: for the
/// design x [npopcomb × npairs] and the per-block numerator ymat [npopcomb × n_block],
/// A_shared = x'x + ridge·I (SPD), and the per-block smoothed f-stat-basis coefficients
///   b[:,block] = solve(A_shared, x'·ymat[:,block])
/// (a NaN-comb-row block downdates A_b = A_shared - x[nan]'x[nan] before the solve; an
/// ALL-NaN block → b=0). bglob is the SAME shared solve over the GLOBAL jackknife-est y
/// [npopcomb] (AT2 matrix_jackknife_est). The CUDA backend runs ONE syrk (A_shared), ONE
/// potrf (the shared factor), ONE gemm (x'·ymat) + a cublasDtrsm PAIR over ALL n_block
/// columns (no host per-block loop), with the NaN/missing blocks grouped (CUB segmented) +
/// per-group batched potrf/potrs downdate; the CpuBackend is the small_linalg native oracle.
struct QpfstatsSmooth {
    /// [npairs * n_block] COLUMN-MAJOR: b[pair + npairs*block] — the per-block smoothed
    /// coefficient on the outgroup-f basis pair (the AT2 `b` matrix, npairs × nblocks).
    std::vector<double> b;

    /// [npairs] — the GLOBAL smoothed coefficients (AT2 `bglob`), the recentering target.
    std::vector<double> bglob;

    int npairs = 0;    ///< n(n-1)/2 (the f-stat basis dim == ncol(x)).
    int n_block = 0;   ///< the jackknife block count (== ncol(ymat)).

    /// NonSpdCovariance if A_shared (ridge-regularized) is not factorable (value, not throw;
    /// should not occur with ridge>0 over a real basis).
    Status status = Status::Ok;
};

/// S4 DIAGONAL-only output: the per-item jackknife VARIANCE vector — the production-scale
/// shape for the per-item f-stats (f4/f3), which read ONLY diag(Q) and NEVER invert Q.
/// var[k] == JackknifeCov.Q[k + m*k] == (1/nb)·Σ_b xtau[k,b]², computed WITHOUT forming the
/// dense m×m Q / its Cholesky inverse (O(m·nb) work, O(m) memory — no N² OOM at sweep scale).
/// se[k] = sqrt(var[k]); the est is read separately from F4Blocks.x_total[k]. Native FP64 (it
/// IS the diagonal jackknife_cov computes; a tiny per-item sum-of-squares — bit-equal to the
/// deleted dense path BY CONSTRUCTION, so the f4/f3 goldens do not move). Mined from
/// wip/fstats-massive-overbuild (backend.hpp JackknifeDiag).
struct JackknifeDiag {
    std::vector<double> var;  ///< [m] the per-item diagonal variance (== diag(Q)).
    int m = 0;
    Status status = Status::Ok;  ///< Ok always (no Q invert ⇒ no NonSpdCovariance).
};

/// qpGraph TOPOLOGY ARENAS carried INTO the fleet virtual (the CUDA-free device-facing
/// twin of core::qpadm::QpGraphModel — backend.hpp need not include the model header).
/// All flat int/double arenas, uploaded ONCE per topology (the qpAdm per-model index-
/// arena pattern). The device kernel runs the SAME fill_pwts (path tables) -> centered
/// pwts -> ppwts_2d -> native SPD edge solve -> GLS quadratic form per (restart) thread.
struct QpGraphTopoArena {
    int npop = 0, nedge_norm = 0, nadmix = 0, npair = 0, npath = 0, base_leaf = 0;
    std::vector<double> pwts0;          ///< [nedge_norm x npop] col-major (theta-independent).
    std::vector<int> pe_edge, pe_leaf, pe_path;        ///< fill_pwts path-edge table.
    std::vector<int> pae_path, pae_admixedge;          ///< fill_pwts path-admixedge table.
    std::vector<int> cmb1, cmb2;        ///< [npair] centered-column pair indices (0-based).
    bool constrained = true;            ///< drift>=0 (the box-constrained edge solve).
    double fudge = 1e-4;                ///< the cc trace-scaled ridge (AT2 `diag`).
};

/// qpGraph FLEET output: the best-of-restarts {score, theta} + the per-weight restart
/// bracket + the fitted edge lengths at the optimum. NonSpdCovariance/RankDeficient ⇒
/// a degenerate inner solve at every restart (value, not throw).
struct QpGraphFleet {
    std::vector<double> theta;       ///< [nadmix] the best restart's mixture weights.
    std::vector<double> theta_lo;    ///< [nadmix] min over restarts (the bracket low).
    std::vector<double> theta_hi;    ///< [nadmix] max over restarts (the bracket high).
    std::vector<double> edge_length; ///< [nedge_norm] the fitted drift edge lengths at theta.
    std::vector<double> f3_fit;      ///< [npair] the fitted f3 (ppwts_2d * edge_length).
    double score = 0.0;              ///< the best (min) GLS score.
    double restart_spread = 0.0;     ///< max-min score across restarts (convergence witness).
    Status status = Status::Ok;
};

/// GPU-ONLY f-stat SWEEP survivors (the CUDA-free seam between the core sweep driver and the
/// CUDA backend's on-device pipeline). The backend enumerates+computes+filters+compacts EVERY
/// C(P,k) item ON THE DEVICE and returns ONLY the survivors here (the full N-row table is never
/// host-materialized). Parallel arrays, one slot per survivor; keys[r] holds the k P-axis
/// indices of survivor r (k<=4; the unused slot is 0). `enumerated` is the total C(P,k) (echoed
/// even when capped); `capped` ⇒ the maxcomb cap refused (no compute ran). Status is Ok unless
/// capped (InvalidConfig). Native FP64 (the f-stat cancellation carve-out + the diagonal SE).
struct SweepSurvivors {
    std::vector<std::array<int, 4>> keys;  ///< survivor index tuples (k<=4; unused slot 0).
    std::vector<double> est;               ///< f4/f3 point estimate per survivor.
    std::vector<double> se;                ///< diagonal jackknife SE per survivor.
    std::vector<double> z;                 ///< est/se per survivor.

    std::size_t enumerated = 0;  ///< total C(P,k) the sweep would enumerate.
    bool capped = false;         ///< refused by the maxcomb cap (no compute ran).
    Status status = Status::Ok;
};

/// CUDA-FREE request carried INTO the backend sweep virtual (the public SweepRequest's device-
/// facing twin; the core driver fills it from the public type so backend.hpp need not include
/// the public sweep header). k is the arity (4 quartet / 3 triple). filter_mode: 0 = MinZ
/// (on-device |z|>=min_z); 1 = keep-all (TopK/All — the device keeps every item, the host ranks
/// the compacted set). pop_subset empty ⇒ sweep [0,P).
struct SweepConfig {
    int k = 4;                    ///< item arity (quartet/triple).
    /// 0 = MinZ: a fixed-threshold |z|>=min_z filter; tau does NOT rise (but the device reservoir
    /// still caps to top_k as a hard safety ceiling so even a billions-item MinZ sweep cannot OOM
    /// the host). 1 = TopK: a DEVICE-BOUNDED rising-tau reservoir — keep the top_k most-significant
    /// |z| in a fixed CAP=O(top_k) buffer; tau RISES to the running K-th |z| (min_z is its floor),
    /// monotonically shrinking the per-chunk survivor count. Either way the host receives <=top_k
    /// rows, sorted by |z| descending, INDEPENDENT of how many billions are computed.
    int filter_mode = 0;
    double min_z = 3.0;          ///< |z| threshold / the rising-tau FLOOR.
    std::size_t top_k = 1000000; ///< device reservoir cap K (the bounded top-K target).
    std::vector<int> pop_subset; ///< optional subset of f2 indices; empty ⇒ all P.
    bool sure = false;           ///< lift the maxcomb cap.
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
    /// executed SVD is the deterministic on-device Jacobi at all sizes in M(fit-4);
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

    /// PER-SAMPLE PLOIDY (length n_individuals, parallel to the GATHERED sample axis
    /// — sample_ploidy[g] is the ploidy of gathered individual g): 2 diploid, 1
    /// pseudo-haploid (ancient DNA). AT2's adjust_pseudohaploid=TRUE auto-detects this
    /// PER SAMPLE (a het call ⇒ diploid; none ⇒ pseudo-haploid), so MIXED-PLOIDY
    /// populations (real for aDNA) are handled correctly. The io leaf
    /// (geno_reader::detect_sample_ploidy) fills it; the decode folds it via
    /// core::accumulate_genotype_ploidy (AC += code/(3-ploidy), N += ploidy).
    ///
    /// NULL ⇒ use the scalar `ploidy` for EVERY sample (the legacy uniform-ploidy
    /// path; the existing all-diploid callers/tests that never set this stay
    /// bit-identical). When non-null it OVERRIDES the scalar.
    const int* sample_ploidy = nullptr;

    /// The uniform fallback ploidy when `sample_ploidy == nullptr` (2 diploid / 1
    /// pseudo-haploid). Default 2 (diploid). With `sample_ploidy` set this is unused.
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

/// DATES weighted-LD moment sufficient-statistics (include/steppe/dates.hpp; the cuFFT
/// autocorrelation LD engine output). The per-(chromosome, output-bin) DATES CORR
/// sufficient statistics summed over EVERY admixed sample — the tiny reduced result the host
/// finishes the date with (fixjcorr leave-one-chrom + the corr curve + the exp fit). The
/// ~10^12 SNP-pair object is NEVER formed: these are the IFFT(|FFT|^2) autocorrelation
/// moments (dates.c ddadd/ddcorr -> addcorr2). ROW-MAJOR [n_chrom × n_bin] per moment.
/// Chromosomes are indexed 0..n_chrom-1 in the SAME order as the caller's `chrom_present`
/// list (the per-chrom segmentation), bins 0..n_bin-1 are the output distance bins
/// (bin s spans [s*binsize, (s+1)*binsize) Morgans). The six moments map to the DATES CORR
/// struct accumulators (addcorr2(corr, dd00, dd01, dd10, dd11, dd02, dd20)):
///   s0  += dd00 (count autocorr  = the per-lag pair COUNT, the denominator)
///   s1  += dd01, s2 += dd10 (count×signal cross — for the mean terms, unused at mode 1)
///   s12 += dd11 (signal autocorr = the covariance NUMERATOR)
///   s11 += dd02, s22 += dd20 (count×signal² cross — the variance moments)
/// The reported curve is corr = (s12/s0)/sqrt((s11/s0)*(s22/s0)) (calccorr mode 1, datacol 3).
struct DatesMoments {
    int n_chrom = 0;  ///< number of present chromosomes (== chrom_present length).
    int n_bin = 0;    ///< number of output distance bins.
    std::vector<double> s0;   ///< [n_chrom × n_bin] Σ dd00 (pair-count autocorr / denominator).
    std::vector<double> s1;   ///< [n_chrom × n_bin] Σ dd01.
    std::vector<double> s2;   ///< [n_chrom × n_bin] Σ dd10.
    std::vector<double> s11;  ///< [n_chrom × n_bin] Σ dd02 (signal² variance moment).
    std::vector<double> s12;  ///< [n_chrom × n_bin] Σ dd11 (signal autocorr / covariance numerator).
    std::vector<double> s22;  ///< [n_chrom × n_bin] Σ dd20 (signal² variance moment).
    Status status = Status::Ok;
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

    /// STANDALONE f4 (the run_f4 seam; fit-engine §6) — assemble ONE F4Blocks whose m
    /// axis is the N input QUARTETS (heterogeneous (L0,R0) per column, unlike assemble_f4
    /// above whose single (L0,R0) yields an nl×nr grid). For quartet k = (p1,p2,p3,p4),
    /// per block b:
    ///   X[k,b] = 0.5*( f2(p2,p3,b) + f2(p1,p4,b) - f2(p1,p3,b) - f2(p2,p4,b) )
    /// i.e. the SAME four-slab AT2 identity specialized to left={p1,p2}, right={p3,p4},
    /// nl=1, nr=1 — ZERO new math. The returned F4Blocks sets nl = N, nr = 1 (so m =
    /// nl*nr = N), and carries x_blocks / x_loo / x_total / block_sizes EXACTLY like
    /// assemble_f4, so jackknife_cov (S4) consumes the whole m-batch with NO change and
    /// se[k] = sqrt(Q[k + m*k]). Native FP64 (the cancellation carve-out, OQ-5; the
    /// `precision` argument is acknowledged but the 4-slab diff stays native). The GPU
    /// backend overrides the DeviceF2Blocks form (zero D2H); the CpuBackend overrides the
    /// host F2BlockTensor form (the oracle door). `quartets` is the flattened P-axis index
    /// quad array (length 4*N, quad k at [4*k .. 4*k+3] = {p1,p2,p3,p4}).
    [[nodiscard]] virtual F4Blocks assemble_f4_quartets(
        const steppe::device::DeviceF2Blocks& f2,
        std::span<const int> quartets,
        const Precision& precision) {
        (void)f2; (void)quartets; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f4_quartets: not implemented by this backend");
    }

    /// STANDALONE f4 host-oracle overload — the SAME per-quartet four-slab combine reading
    /// a host F2BlockTensor (the CpuBackend oracle door; sibling of assemble_f4(host)).
    /// Identical math to the DeviceF2Blocks overload; only the f2 storage differs.
    [[nodiscard]] virtual F4Blocks assemble_f4_quartets(
        const F2BlockTensor& f2,
        std::span<const int> quartets,
        const Precision& precision) {
        (void)f2; (void)quartets; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f4_quartets(host): not implemented by this backend");
    }

    /// STANDALONE f3 (the run_f3 seam; fit-engine §6) — the THREE-slab clone of
    /// assemble_f4_quartets. Assemble ONE F4Blocks (reused as the generic per-block X
    /// carrier) whose m axis is the N input TRIPLES. For triple k = (C=p1, A=p2, B=p3),
    /// per block b:
    ///   X[k,b] = 0.5*( f2(C,A,b) + f2(C,B,b) - f2(A,B,b) )
    /// i.e. the THREE-slab AT2 f3 identity (vs the four-slab quartet diff) — the ONE new
    /// math seam f3 adds over f4. The returned F4Blocks sets nl = N, nr = 1 (so m = nl*nr =
    /// N), and carries x_blocks / x_loo / x_total / block_sizes EXACTLY like
    /// assemble_f4_quartets, so jackknife_cov (S4) consumes the whole m-batch with NO change
    /// and se[k] = sqrt(Q[k + m*k]). Native FP64 (the cancellation carve-out, OQ-5; the
    /// `precision` argument is acknowledged but the 3-slab diff stays native). The GPU
    /// backend overrides the DeviceF2Blocks form (zero D2H); the CpuBackend overrides the
    /// host F2BlockTensor form (the oracle door). `triples` is the flattened P-axis index
    /// triple array (length 3*N, triple k at [3*k .. 3*k+2] = {p1=C, p2=A, p3=B}).
    [[nodiscard]] virtual F4Blocks assemble_f3_triples(
        const steppe::device::DeviceF2Blocks& f2,
        std::span<const int> triples,
        const Precision& precision) {
        (void)f2; (void)triples; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f3_triples: not implemented by this backend");
    }

    /// STANDALONE f3 host-oracle overload — the SAME per-triple three-slab combine reading
    /// a host F2BlockTensor (the CpuBackend oracle door; sibling of assemble_f4_quartets(host)).
    /// Identical math to the DeviceF2Blocks overload; only the f2 storage differs.
    [[nodiscard]] virtual F4Blocks assemble_f3_triples(
        const F2BlockTensor& f2,
        std::span<const int> triples,
        const Precision& precision) {
        (void)f2; (void)triples; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f3_triples(host): not implemented by this backend");
    }

    /// qpGraph FLEET (the productized IDEA-1 optimizer spike + the path-algebra
    /// objective; docs/research/qpgraph-gpu-design.md). Given the resident f3 basis
    /// (f_obs[npair] + qinv[npair*npair] col-major, assembled ONCE by the core driver via
    /// assemble_f3_triples + jackknife_cov and kept device-resident on the CUDA path) and
    /// the topology arenas, run `numstart` projected-Newton restarts. On the CUDA backend
    /// each restart is ONE GPU thread running the WHOLE multistart x maxit loop in-kernel
    /// (the GPU-BOUND shape: NO host objective per iteration — the AT2 optim() host-loop
    /// trap designed out); the host launches ONCE and gets back only {score, theta,
    /// edges}. The CpuBackend is the native small_linalg fleet oracle (the bit-exact diff
    /// reference). `precision` governs the (production-scale) cc design GEMM; the inner SPD
    /// edge solve + the GLS form are native FP64 (the cancellation carve-out). NON-PURE:
    /// the base throws (the established backend.hpp pattern).
    [[nodiscard]] virtual QpGraphFleet qpgraph_fit_fleet(const QpGraphTopoArena& topo,
                                                         std::span<const double> f_obs,
                                                         std::span<const double> qinv,
                                                         int numstart, int maxit,
                                                         double tol,
                                                         const Precision& precision) {
        (void)topo; (void)f_obs; (void)qinv; (void)numstart; (void)maxit; (void)tol;
        (void)precision;
        throw std::runtime_error(
            "ComputeBackend::qpgraph_fit_fleet: not implemented by this backend");
    }

    /// qpDstat Part B (the genotype-path NORMALIZED-D reduction; include/steppe/dstat.hpp).
    /// The S2 DIVERGENCE — NOT the f2 GEMM, NOT the f2 cache. Per BLOCK b, per QUADRUPLE k =
    /// (p1,p2,p3,p4), a segmented reduction over the block's SNP columns accumulating
    ///   numsum[k,b] += (a-b)*(c-d),  densum[k,b] += (a+b-2ab)*(c+d-2cd),  cnt[k,b] += 1
    /// over ONLY the SNPs where V==1 for all 4 pops (a,b,c,d = Q[p1,s]..Q[p4,s]; the
    /// allsnps=TRUE per-(block,quadruple) finiteness mask). Q/V are the column-major [P × M]
    /// decode_af outputs over ALL M SNPs; `block_id` is the per-SNP dense block id from
    /// assign_blocks (length M, non-decreasing ⇒ each block's columns are contiguous);
    /// `quadruples` is the flattened P-axis index quad array (length 4*N, quad k at
    /// [4*k .. 4*k+3] = {p1,p2,p3,p4}). Outputs are ROW-MAJOR [N × n_block]: numsum[k*nb+b]
    /// etc. (tiny, like F4Blocks::x_blocks). The CUDA backend uploads Q/V to VRAM and runs
    /// the SNP-tile-streamed batched-over-N kernel (device-resident); the CpuBackend is the
    /// long-double host oracle. Native FP64 / long-double accumulation (cancellation in
    /// num/den per §12). NON-PURE: the base throws (the established backend.hpp pattern).
    ///
    /// @param Q          reference-allele frequencies in [0,1], 0 where invalid, [P × M].
    /// @param V          validity mask (1.0 valid / 0.0 missing), [P × M].
    /// @param P          number of populations (the leading dimension of Q/V).
    /// @param M          number of SNPs (the column count of Q/V).
    /// @param block_id   per-SNP dense block id (length M), from assign_blocks.
    /// @param n_block    number of distinct blocks (== max(block_id)+1).
    /// @param quadruples flattened P-axis index quad array (length 4*N).
    /// @param numsum     OUT [N * n_block] row-major Σnum per (quadruple, block).
    /// @param densum     OUT [N * n_block] row-major Σden per (quadruple, block).
    /// @param cnt        OUT [N * n_block] row-major used-SNP count per (quadruple, block).
    virtual void dstat_block_reduce(const double* Q, const double* V, int P, long M,
                                    const int* block_id, int n_block,
                                    std::span<const int> quadruples,
                                    double* numsum, double* densum, double* cnt) {
        (void)Q; (void)V; (void)P; (void)M; (void)block_id; (void)n_block;
        (void)quadruples; (void)numsum; (void)densum; (void)cnt;
        throw std::runtime_error(
            "ComputeBackend::dstat_block_reduce: not implemented by this backend");
    }

    /// DATES weighted-LD CURVE — the cuFFT autocorrelation LD engine (include/steppe/dates.hpp).
    /// The whole GPU-bound per-admixed-sample pipeline reduced to the tiny per-(chrom,bin)
    /// CORR sufficient statistics, with the ~10^12 SNP-pair object NEVER materialized (the
    /// ALDER FFT trick: cov(d=lag) = IFFT(|FFT(signal-grid)|²) / IFFT(|FFT(count-grid)|²)).
    /// Summed over EVERY admixed-target sample (DATES per-sample main loop, dates.c:585-740):
    ///   1. per sample i, per valid SNP s: regress dosage w0 = genotype(i,s)/2 on the two
    ///      source freqs {w1=Q[src1,s], w2=Q[src2,s]}: y = Σ(w0-w2)(w1-w2)/Σ(w1-w2)², residual
    ///      res = w0 - (y·w1 + (1-y)·w2); the per-SNP scattered value is res·wt,
    ///      wt = w1-w2 (the population-delta weight; native FP64 cancellation carve-out).
    ///   2. scatter-add onto the FINE per-chrom grid at cell grid_cell[s] (spacing binsize/qbin):
    ///      z0q[g] += 1, z1q[g] += res·wt, z2q[g] += (res·wt)².
    ///   3. batched cuFFT autocorrelation over each chrom's grid segment -> the six dd moments
    ///      (fftauto/fftcv2; native double FFT, UNNORMALIZED so an explicit 1/n scale matches
    ///      the FFTW reference bit-for-bit), accumulated per chrom.
    ///   4. re-bin lag d (distance d·binsize/qbin) to the output bin s = floor(d·dbinsize/binsize)
    ///      and add the six moments into the per-(chrom, bin) DatesMoments (dates.c ddcorr +
    ///      addcorr2). The host finishes: fixjcorr leave-one-chrom + the corr curve + the fit.
    ///
    /// INPUTS (device-resident on the CUDA path; the host never loops over SNP pairs):
    /// @param src1_freq  per-SNP source-1 allele freq Q[src1,s], length M (decode_af row).
    /// @param src2_freq  per-SNP source-2 allele freq Q[src2,s], length M.
    /// @param src_valid  per-SNP validity (1.0 iff BOTH sources valid at s), length M.
    /// @param packed     the admixed-target packed genotype bytes (TGENO/GENO individual-major
    ///                   sub-tile for the target samples), length n_target·bytes_per_record.
    /// @param bytes_per_record  per-target-individual record stride in `packed`.
    /// @param n_target   number of admixed-target individuals (the per-sample loop count).
    /// @param target_ploidy per-target-sample ploidy (length n_target; dosage = code/ploidy·...)
    ///                   — DATES uses g/2 (diploid); a pseudo-haploid sample is folded the same
    ///                   way the decode does. NULL ⇒ uniform diploid.
    /// @param grid_cell  per-SNP fine-grid cell index (DATES setqbins tagnumber; cumulative
    ///                   genpos/qb with a +5 Morgan inter-chrom gap), length M.
    /// @param M          number of (autosome-kept) SNPs.
    /// @param chrom_first per-present-chromosome first grid cell (slo), length n_chrom.
    /// @param chrom_last  per-present-chromosome last grid cell (shi), length n_chrom.
    /// @param n_chrom    number of present chromosomes.
    /// @param numqbins   total fine-grid length (max grid_cell + 1).
    /// @param n_bin      number of output distance bins (== round(maxdis/binsize)).
    /// @param diffmax    max FFT lag = round(qbin·maxdis/binsize).
    /// @param binsize    output bin width (Morgans).
    /// @param qbin       fine-grid refinement factor (dbinsize = binsize/qbin).
    /// @param precision  governs ONLY any matmul-heavy sub-step; the cuFFT + the weight/residual
    ///                   accumulation are NATIVE double (FFT well-conditioned; weight is the
    ///                   cancellation carve-out). The §12 invariant.
    /// NON-PURE: the base throws (the established backend.hpp pattern). The CpuBackend gives the
    /// native FFT-free direct-autocorrelation oracle; the CUDA backend the batched cuFFT path.
    [[nodiscard]] virtual DatesMoments dates_curve(
        const double* src1_freq, const double* src2_freq, const double* src_valid,
        const std::uint8_t* packed, std::size_t bytes_per_record, int n_target,
        const int* target_ploidy, const int* grid_cell, long M,
        const int* chrom_first, const int* chrom_last, int n_chrom,
        int numqbins, int n_bin, int diffmax, double binsize, int qbin,
        const Precision& precision) {
        (void)src1_freq; (void)src2_freq; (void)src_valid; (void)packed;
        (void)bytes_per_record; (void)n_target; (void)target_ploidy; (void)grid_cell;
        (void)M; (void)chrom_first; (void)chrom_last; (void)n_chrom; (void)numqbins;
        (void)n_bin; (void)diffmax; (void)binsize; (void)qbin; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::dates_curve: not implemented by this backend "
            "(the cuFFT autocorrelation LD engine requires a CUDA backend; the "
            "CpuBackend provides the FFT-free reference oracle)");
    }

    /// qpfstats SMOOTHING SOLVE (the genotype-path joint f2 smoother; the shared-factor
    /// batched least-squares; include/steppe/qpfstats.hpp). REFORMULATES the AT2
    /// qpfstats_regression host block loop (R io.R CHUNK_SIZE=64 per-block solve, the
    /// CPU-bound trap) into ONE shared SPD factor + a batched multi-column solve over the
    /// WHOLE n_block axis — NO host per-block loop on the CUDA path.
    ///
    ///   A_shared = x'x + ridge·I  (SPD; ONE cublasDsyrk, EmulatedFp64 via the f2 policy)
    ///   L = chol(A_shared)        (ONE cusolverDnDpotrf, native FP64 carve-out)
    ///   RHS = x' · ymat            (ONE cublasDgemm, [npairs × n_block], EmulatedFp64)
    ///   b[:,blk] = solve(A_shared, RHS[:,blk])   for blocks with NO NaN comb-row:
    ///                                            a cublasDtrsm PAIR (forward t(L) then back L)
    ///                                            over ALL n_block columns at once.
    ///   NaN-comb-row block: A_b = A_shared - x[nan]'x[nan]; b = solve(A_b, RHS[:,blk])
    ///                       (grouped by missing-row-set + batched potrf/potrs downdate).
    ///   ALL-NaN block: b = 0 (the AT2 nan_chunk policy; must be ZERO, not dropped).
    ///   bglob = solve(A_shared (or A_y downdated by y's NaNs), x'·y)   (the recentering target).
    ///
    /// @param x        the design matrix, COLUMN-MAJOR [npopcomb × npairs] (x[c + npopcomb*p]).
    /// @param ymat     the per-(comb,block) numerator, COLUMN-MAJOR [npopcomb × n_block]
    ///                 (ymat[c + npopcomb*b] = numsum[c,b]/cnt[c,b], or NaN where cnt==0).
    /// @param y        the GLOBAL per-comb jackknife estimate, length npopcomb (NaN allowed).
    /// @param npopcomb the number of population combinations (rows of x / ymat).
    /// @param npairs   the f-stat basis dim n(n-1)/2 (cols of x).
    /// @param n_block  the jackknife block count (cols of ymat).
    /// @param ridge    the L2 ridge added to diag(A_shared) (AT2 constant 1e-5).
    /// @param precision governs ONLY the matmul sub-steps (SYRK / GEMM); the Cholesky +
    ///                 triangular solve default native FP64 (the §12 carve-out).
    /// NON-PURE: the base throws (the established backend.hpp pattern).
    [[nodiscard]] virtual QpfstatsSmooth qpfstats_smooth(std::span<const double> x,
                                                         std::span<const double> ymat,
                                                         std::span<const double> y,
                                                         int npopcomb, int npairs,
                                                         int n_block, double ridge,
                                                         const Precision& precision) {
        (void)x; (void)ymat; (void)y; (void)npopcomb; (void)npairs; (void)n_block;
        (void)ridge; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::qpfstats_smooth: not implemented by this backend");
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

    /// S4 DIAGONAL-only — the per-item jackknife VARIANCE (== diag(Q)) for the per-item
    /// f-stats (f4/f3) that read only diag(Q) and never invert Q. The production-scale
    /// shape (the OOM fix; a sweep of ~36k+ items would form a dense m×m Q in jackknife_cov
    /// ⇒ ~10GB+ OOM). var[k] = (1/nb)·Σ_b xtau[k,b]² over the SAME est_to_loo / x_total /
    /// tot_line xtau seam jackknife_cov uses — minus the dense SYRK + Cholesky inverse:
    /// O(m·nb) work, O(m) memory. fudge is NOT consumed (a bare f-stat SE is the UNFUDGED
    /// diagonal; the qpAdm ridge is a Q-invert concern only). Native FP64 (it IS the diagonal
    /// jackknife_cov computes ⇒ bit-equal to the deleted dense path, golden-exact). NON-PURE:
    /// base throws (the established backend.hpp pattern). Mined from wip/fstats-massive-overbuild.
    [[nodiscard]] virtual JackknifeDiag jackknife_diag(const F4Blocks& x,
                                                       std::span<const int> block_sizes,
                                                       const Precision& precision) {
        (void)x; (void)block_sizes; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::jackknife_diag: not implemented by this backend");
    }

    /// GPU-ONLY f-stat SWEEP (the production-scale fix for the CPU-bound host-enumeration
    /// disaster). Enumerate EVERY C(P,k) item (k from cfg.k; subset from cfg.pop_subset),
    /// compute its f4/f3 point estimate + diagonal jackknife SE + |z| ON THE DEVICE, filter
    /// (on-device |z|>=min_z flag, or keep-all for TopK) + cub::DeviceSelect::Flagged stream-
    /// compact ON THE DEVICE, and return ONLY the survivors. The HOST does NOT enumerate, NOT
    /// filter, NOT loop per item — it drives ONLY the chunk loop + receives survivors. The
    /// per-chunk pipeline REUSES the SAME device kernels as the explicit run_f4 path (the
    /// unrank kernel WRITES the device quartet list assemble_f4_quartets_gather reads, then
    /// loo/total/xtau/diag_var run verbatim) — NO forked compute. The maxcomb cap
    /// (kFstatMaxComb) fires up-front (capped ⇒ no compute). `precision` engages the
    /// emulated-FP64 policy where it applies (the f-stat combine + diagonal SE are the native
    /// cancellation carve-out, so the sweep is native-by-carve-out end-to-end — consistent
    /// with the policy, not a violation). NON-PURE: base throws (the established pattern; only
    /// the CUDA backend overrides — the sweep is a GPU-only product, no CPU runtime).
    [[nodiscard]] virtual SweepSurvivors f4_sweep(const steppe::device::DeviceF2Blocks& f2,
                                                  const SweepConfig& cfg,
                                                  const Precision& precision) {
        (void)f2; (void)cfg; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::f4_sweep: not implemented by this backend "
            "(the GPU-only sweep requires a CUDA backend)");
    }

    /// GPU-ONLY f3 sweep — the three-slab sibling of f4_sweep (cfg.k == 3; reuses the f3
    /// triple gather instead of the quartet gather). Same on-device pipeline + cap + filter.
    [[nodiscard]] virtual SweepSurvivors f3_sweep(const steppe::device::DeviceF2Blocks& f2,
                                                  const SweepConfig& cfg,
                                                  const Precision& precision) {
        (void)f2; (void)cfg; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::f3_sweep: not implemented by this backend "
            "(the GPU-only sweep requires a CUDA backend)");
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

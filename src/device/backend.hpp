// src/device/backend.hpp
//
// ComputeBackend — the CUDA-free interface every compute operation routes
// through, plus the plain-data structs that cross it. Two implementations
// satisfy it: CudaBackend (the GPU deliverable) and CpuBackend (the reference
// oracle). CUDA stays private to the device layer, behind this one door.
//
// Reference: docs/reference/src_device_backend.hpp.md
#ifndef STEPPE_DEVICE_BACKEND_HPP
#define STEPPE_DEVICE_BACKEND_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/fstats.hpp"
#include "steppe/qpadm.hpp"
#include "core/internal/views.hpp"
#include "core/internal/decode_af.hpp"
#include "device/device_partial.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/device_decode_result.hpp"
#include "device/likelihood_tensor.hpp"
#include "device/readv2_bitmatrix.hpp"
#include "device/stream_f2_blocks.hpp"

namespace steppe {

// f2 matrix result — reference §6
struct F2Result {
    std::vector<double> f2;

    std::vector<double> vpair;

    int P = 0;
};

// Per-block f4 + jackknife estimate — reference §7
struct F4Blocks {
    std::vector<double> x_blocks;

    std::vector<double> x_total;

    std::vector<double> x_loo;

    int nl = 0;
    int nr = 0;
    int n_block = 0;

    std::vector<int> block_sizes;
};

// Jackknife covariance and its inverse — reference §7
struct JackknifeCov {
    std::vector<double> Q;

    std::vector<double> Qinv;

    int m = 0;

    Status status = Status::Ok;
};

// GLS weight fit — reference §7
struct GlsWeights {
    std::vector<double> w;
    std::vector<double> A;
    std::vector<double> B;
    double chisq = 0.0;
    int r = 0;

    Status status = Status::Ok;
};

// qpfstats smoothing-solve output — reference §9
struct QpfstatsSmooth {
    std::vector<double> b;

    std::vector<double> bglob;

    std::vector<double> recenter_shift;

    int npairs = 0;
    int n_block = 0;

    Status status = Status::Ok;
};

// Diagonal-only jackknife variance — reference §7
struct JackknifeDiag {
    std::vector<double> var;
    int m = 0;
    Status status = Status::Ok;
};

// Ratio block-jackknife output — reference §8
struct RatioBlockJackknife {
    std::vector<double> est;
    std::vector<double> se;
    std::vector<double> z;
    std::vector<double> p;
    int N = 0;
    Status status = Status::Ok;
};

// Ratio-jackknife input array descriptor — reference §8
struct RatioJackArray {
    const double* data = nullptr;
    long base = 0;
    long item_stride = 0;
    long block_stride = 0;
};

// qpGraph topology arena — reference §10
struct QpGraphTopoArena {
    int npop = 0, nedge_norm = 0, nadmix = 0, npair = 0, npath = 0, base_leaf = 0;
    std::vector<double> pwts0;
    std::vector<int> pe_edge, pe_leaf, pe_path;
    std::vector<int> pae_path, pae_admixedge;
    std::vector<int> cmb1, cmb2;
    bool constrained = true;
    double fudge = 1e-4;
};

// qpGraph best-of-restarts fit — reference §10
struct QpGraphFleet {
    std::vector<double> theta;
    std::vector<double> theta_lo;
    std::vector<double> theta_hi;
    std::vector<double> edge_length;
    std::vector<double> f3_fit;
    double score = 0.0;
    double restart_spread = 0.0;
    Status status = Status::Ok;
};

// qpGraph batched best scores — reference §10
struct QpGraphFleetBatch {
    std::vector<double> best_score;
    std::vector<double> restart_spread;
    Status status = Status::Ok;
};

// f-stat sweep survivors — reference §11
struct SweepSurvivors {
    std::vector<std::array<int, 4>> keys;
    std::vector<double> est;
    std::vector<double> se;
    std::vector<double> z;

    std::size_t enumerated = 0;
    bool capped = false;
    Status status = Status::Ok;
};

// Sweep filter-mode constants — reference §11
inline constexpr int kSweepFilterMinZ = 0;
inline constexpr int kSweepFilterTopK = 1;

// f-stat sweep request — reference §11
struct SweepConfig {
    int k = 4;
    int filter_mode = kSweepFilterMinZ;
    double min_z = 3.0;
    std::size_t top_k = 1000000;
    std::vector<int> pop_subset;
    bool sure = false;
};

// Rank-test sweep output — reference §7
struct RankSweep {
    std::vector<double> chisq;
    std::vector<int>    dof;
    std::vector<double> p;

    std::vector<int>    rd_f4rank;
    std::vector<int>    rd_dof;
    std::vector<double> rd_chisq;
    std::vector<double> rd_p;
    std::vector<int>    rd_dofdiff;
    std::vector<double> rd_chisqdiff;
    std::vector<double> rd_p_nested;

    int f4rank = 0;
    int rank_Q = 0;

    int svd_path = 0;

    Status status = Status::Ok;
};

// Leave-one-source-out feasibility row — reference §7
struct PopDropRow {
    std::string         pat;
    int                 wt = 0;
    int                 dof = 0;
    double              chisq = 0.0;
    double              p = 0.0;
    int                 f4rank = 0;
    std::vector<double> weight;
    bool                feasible = false;
    Status              status = Status::Ok;
};

// Genotype decode input tile view — reference §5
struct DecodeTileView {
    const std::uint8_t* packed = nullptr;
    std::size_t bytes_per_record = 0;
    std::size_t n_snp = 0;
    std::size_t n_individuals = 0;
    const std::size_t* pop_offsets = nullptr;
    int n_pop = 0;

    const int* sample_ploidy = nullptr;

    int ploidy = 2;

    bool detect_ploidy_on_device = false;
};

// Real-valued (BGEN) dosage tile view — the FP32 analogue of DecodeTileView, parallel
// to the 2-bit path so the byte-exact hardcall decode and all PCA goldens stay
// untouched. dosage is INDIVIDUAL-MAJOR: dosage[i * n_snp + s] is sample i's ALT
// dosage at variant s (a real number in [0, 2]), or a quiet NaN for a missing call
// (the float analogue of the 2-bit tile's missing == code 3). Produced by the io-leaf
// bgen_reader; consumed by pca_covariance_eig_dosage's dosage standardize kernels.
struct DosageTileView {
    const float* dosage = nullptr;  // n_individuals * n_snp, individual-major; NaN = missing
    std::size_t n_snp = 0;
    std::size_t n_individuals = 0;
};

// Genotype decode output — reference §5
struct DecodeResult {
    std::vector<double> q;
    std::vector<double> v;
    std::vector<double> n;
    int P = 0;
    long M = 0;
};

// Native→canonical code map — reference §5
enum class TileEncoding : int {
    Identity = 0,
};

// SNP-major source tile view — reference §5
struct SnpMajorTileView {
    const std::uint8_t* snp_major = nullptr;
    std::size_t src_bytes_per_record = 0;
    std::size_t n_snp = 0;

    const std::size_t* sel_rows = nullptr;
    std::size_t n_individuals = 0;

    const std::size_t* pop_offsets = nullptr;
    int n_pop = 0;

    TileEncoding encoding = TileEncoding::Identity;
};

// Canonical individual-major tile — reference §5
struct CanonicalTile {
    std::vector<std::uint8_t> packed;
    std::size_t bytes_per_record = 0;
    std::size_t n_snp = 0;
    std::size_t n_individuals = 0;
    std::vector<std::size_t> pop_offsets;
};

// Backend capability probe — reference §13
struct BackendCapabilities {
    int device_count = 0;

    int compute_major = 0;
    int compute_minor = 0;

    std::size_t total_vram_bytes = 0;
    std::size_t free_vram_bytes  = 0;

    bool can_access_peer = false;

    bool emulated_fp64_honorable = false;
};

// DATES weighted-LD moments — reference §12
struct DatesMoments {
    int n_chrom = 0;
    int n_bin = 0;
    std::vector<double> s0;
    std::vector<double> s1;
    std::vector<double> s2;
    std::vector<double> s11;
    std::vector<double> s12;
    std::vector<double> s22;
    Status status = Status::Ok;
};

// DATES exponential-decay fit — reference §12
struct DatesExpFit {
    double date_gen = 0.0;
    double error_sd = 0.0;
    int ok = 0;
};

// Li-Stephens forward-backward copying posterior (the `steppe paint` FB core).
// gamma is the per-column copying posterior gamma_l(k) = P(recipient copies donor k
// at SNP l), donor-major: gamma[k*M + l], each column l summing to 1. In Phase 0
// only the CpuBackend reference oracle computes it (native FP64, per-column rescaled);
// the GPU override lands in Phase 1. Reference: li-stephens-engine-scope.md §2a.
struct LsPosterior {
    std::vector<double> gamma;  // K*M, donor-major, each SNP column normalized
    int K = 0;
    long M = 0;
    Status status = Status::Ok;
};

// Li-Stephens ChromoPainter coancestry (the `steppe paint` FACE, Phase 2). For each
// recipient, the two per-donor coancestry summaries folded on-device from the copying
// posterior gamma WITHOUT ever materializing the K*M posterior to host:
//   chunklengths[r*K+k] = sum_l gamma_l(k)*w_l   (Morgans; expected copied length)
//   chunkcounts [r*K+k] = gamma_0(k) + sum_{l>=1} switch_l(k)   (expected #chunks)
// Both are recipient-major (N rows of K). Native FP64 (scope §2c reduction carve-out).
// Reference: docs/planning/li-stephens-phase2-paint-face-spec.md §1, §2.
struct LsCoancestry {
    std::vector<double> chunkcounts;   // N*K, recipient-major [r*K + k]
    std::vector<double> chunklengths;  // N*K, recipient-major [r*K + k], Morgans
    int K = 0;
    long N = 0;
    Status status = Status::Ok;
};

// Li-Stephens LOCAL-ANCESTRY posterior (the `steppe paint --face localanc` output,
// Phase 3). Per recipient, the per-SNP posterior over ancestry LABELS, folded on-device
// from the copying posterior gamma WITHOUT materializing the K*M posterior:
//   post[(r*M + l)*P + g] = sum_{k : group(k)==g} gamma_l(k)
// Each SNP column sums to 1 over g when the column is informative (gamma is column-
// normalized and the labels partition the donors); a degenerate all-missing column whose
// FB denominator underflows to 0 has its gamma (and therefore post_l) zeroed — the same
// guard the FB applies. Native FP64 (scope §2c reduction carve-out — a reduction, not a
// GEMM; do NOT emulate). Only the CpuBackend oracle and the CUDA override implement it.
// Reference: docs/planning/li-stephens-phase3-localanc-face-spec.md §2.
struct LsLocalAncestry {
    std::vector<double> post;  // N*M*P, layout post[(r*M + l)*P + g]
    int P = 0;                 // number of ancestry labels
    long M = 0;
    long N = 0;
    Status status = Status::Ok;
};

// ancIBD per-pair IBD posterior (the `steppe ibd` FB output). p_ibd is the per-SNP
// IBD posterior 1 - post[0] for each pair, pair-major p_ibd[pair*M + l], over the M
// run sites. Only the small n_pair*M posterior returns to host; the derived
// haplotype-probability table + the 5-state scan stay device-resident. Native FP64
// (the FB is a sub-one product scan — the reduction carve-out, NOT the emulated
// default). Only the CpuBackend oracle and the CUDA override implement it.
// Reference: docs/planning/ancibd-face-spec.md §3.
struct AncibdPosterior {
    std::vector<double> p_ibd;  // n_pair*M, pair-major [pair*M + l]
    int n_pair = 0;
    long M = 0;
    Status status = Status::Ok;
};

// hapROH per-target ROH posterior (the `steppe roh` FB output). p_roh is the per-SNP
// ROH posterior 1 - gamma_0(l) for each target, target-major p_roh[target*M + l], over
// the M kept sites. Only the small n_target*M posterior returns to host; the (K+1)-state
// copying scan (with checkpoint/recompute over the K-donor panel) stays device-resident.
// Native FP64 (the FB is a sub-one product scan — the reduction carve-out, NOT the
// emulated default). Only the CpuBackend oracle and the CUDA override implement it.
// Reference: docs/planning/haproh-face-spec.md §3.
struct RohPosterior {
    std::vector<double> p_roh;  // n_target*M, target-major [target*M + l]
    int n_target = 0;
    long M = 0;
    Status status = Status::Ok;
};

// RohPanelHandle — an opaque handle to the ONCE-uploaded, run-resident hapROH reference
// panel (the `steppe roh` batch-overlap residency). `host_panel` is the donor-major
// decoded panel bytes (Kpanel rows of Mp, the CLI's `panel_hap`); `donor_map[k]` (length
// K) is the resident-panel ROW of selected donor k (the CLI's `panel_cols`). The CUDA
// override also parks the panel + donor_map in device memory (owned via `device`); the
// CpuBackend leaves `device` null and gathers from `host_panel` in the fallback. Both the
// backing arrays outlive every roh_fb_batch call (owned by the caller for the whole run).
struct RohPanelHandle {
    std::shared_ptr<void> device;              // CUDA: resident panel + donor_map; CPU: null
    const std::uint8_t* host_panel = nullptr;  // Kpanel*Mp donor-major decoded panel bytes
    const int* donor_map = nullptr;            // K, resident-panel row per selected donor
    int Kpanel = 0;                            // donor columns (rows) in the resident panel
    int K = 0;                                 // selected donors (donor_map length)
    long Mp = 0;                               // panel site count (column stride)
};

// RohItemBuilder — fill work-item i's GPU-facing inputs into the provided (pinned)
// destinations, each sized to Mmax, and RETURN the item's kept-site count M (<= Mmax; 0
// skips the item). ob[l] = target observed allele {0,1,missing}; site_map[l] = the panel
// column (KeptSite::panel_l) of kept site l; p[l] = per-SNP panel allele frequency;
// T[l*9+..] = per-SNP transition tensor. Called AHEAD of the GPU (phase P) by the pipeline.
using RohItemBuilder =
    std::function<long(long i, std::uint8_t* ob, int* site_map, double* p, double* T)>;

// RohPosteriorConsumer — invoked in STRICT item order (i = 0,1,2,...) with item i's ROH
// posterior 1 - gamma_0 (M values); the segment-caller (phase C). Strict order makes the
// emitted segment table byte-stable regardless of GPU completion order.
using RohPosteriorConsumer = std::function<void(long i, const double* p_roh, long M)>;

// Per-site Weir & Cockerham 1984 FST over one population pair (the `steppe fst` output).
// num/den/fst/valid are per-SNP in the tile's kept order (length M); a site is `valid`
// when both pops are sampled, the combined size exceeds 2, and the denominator is finite
// and non-zero (an invalid site has fst == NaN and num == den == 0.0). sum_num/sum_den
// are the device-reduced (native-FP64) numerator/denominator sums over the VALID sites
// flagged for the genome-wide summary (autosomes), and n_valid counts them; the ratio-of-
// averages fst_ratio = sum_num/sum_den is plink2's `.fst.summary` WC_FST.
struct FstPerSite {
    std::vector<double> num;
    std::vector<double> den;
    std::vector<double> fst;
    std::vector<std::uint8_t> valid;
    double sum_num = 0.0;
    double sum_den = 0.0;
    long n_valid = 0;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// FstMatrix — the device result of the all-pairs WC FST combine (`steppe fst --all-pairs`).
// pair_num/pair_den/pair_cnt are the per-pair genome-wide sums over the C(P,2) upper-
// triangular pairs, indexed by the flat rank r that readv2_unrank_pair(r, P) inverts to
// (i, j) with i < j (r = j*(j-1)/2 + i). The host expands these into the symmetric P x P
// matrix. Native FP64 (the reduction carve-out). Only the small 3*C(P,2) vectors cross PCIe
// — the per-(pop,SNP) sufficient-stat decode + the P^2 combine stay device-resident.
struct FstMatrix {
    std::vector<double> pair_num;   // C(P,2), Σ WC numerator a per pair
    std::vector<double> pair_den;   // C(P,2), Σ WC denominator per pair
    std::vector<long>   pair_cnt;   // C(P,2), valid-site count per pair
    int P = 0;
    std::size_t enumerated = 0;     // C(P,2)
    bool capped = false;            // C(P,2) > the maxcomb cap and sure==false
    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// KingMatrix — the device result of the KING-robust kinship pair sweep (`steppe kinship`).
// The five per-pair integer counts (Manichaikul et al. 2010) over the swept pairs: nsnp
// (considered sites), hethet, ibs0, het_i, het_j. For the all-pairs mode the pairs are the
// C(N,2) upper-triangular set indexed by the flat rank r that readv2_unrank_pair(r, N)
// inverts to (i, j), i < j; for the explicit-pair mode they are indexed 0..K-1 in the given
// list order. The host derives phi = (hethet - 2*ibs0)/(het_i + het_j) and the degree band.
// Integer counting on-device -> bit-exact, order-independent (native FP64 for the tag only).
// Only the five small per-pair vectors cross PCIe; the N x tile decode + the pair combine
// stay device-resident. Only the CUDA backend implements it (GPU-only, like fst_wc_all_pairs).
struct KingMatrix {
    std::vector<long> nsnp;    // per pair, considered (both non-missing) autosomal sites
    std::vector<long> hethet;  // per pair, #{ci==1 & cj==1}
    std::vector<long> ibs0;    // per pair, #{opposite homozygotes}
    std::vector<long> het_i;   // per pair, #{ci==1} over the considered set
    std::vector<long> het_j;   // per pair, #{cj==1} over the considered set
    int N = 0;
    std::size_t enumerated = 0;  // pairs swept (C(N,2) all-pairs, or the explicit-list length)
    bool capped = false;         // C(N,2) > the maxcomb cap and sure==false (all-pairs only)
    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// SfsJoint — the 2D joint site-frequency spectrum over a population pair (`steppe sfs`).
// grid is the row-major (extA x extB) integer joint histogram: cell (i, j) at
// grid[i*extB + j] counts sites with pop-A category i and pop-B category j. A pure
// integer-count stat (gated BIT-EXACT vs scikit-allel joint_sfs/joint_sfs_folded, NOT
// AT2). Only sites COMPLETE in both pops (no missing) are histogrammed (§4); folded uses
// each pop's own within-population minor allele (scikit-allel np.amin(ac, axis=1)).
struct SfsJoint {
    std::vector<std::int64_t> grid;   // row-major, length extA*extB
    long extA = 0, extB = 0;          // category extents (2N+1 unfolded, N+1 folded)
    long NA = 0, NB = 0;              // individuals per pop (chromosome count = 2N)
    long n_total = 0;                 // SNPs in the tile
    long n_complete = 0;              // sites histogrammed (complete in both pops)
    long n_dropped_incomplete = 0;    // n_total - n_complete
    bool folded = false;
    Precision::Kind precision_tag = Precision::Kind::Fp64;  // integer stat; tag informational
};

// PcaEig — the device result of the standalone genotype PCA (`steppe pca`): the top-K
// sample PC coordinates plus the eigen spectrum, produced over the device-resident tile by
// the Patterson-standardize kernel -> cuBLAS SYRK covariance -> cuSOLVER symmetric eigen
// -> coord projection. `coords` is row-major N x K with coord(i,k) = eigenvector_i *
// sqrt(eigenvalue) (== scikit-allel U*S; sign arbitrary per PC); `eigenvalues`/`var_explained`
// are the top-K descending. n_snp_used counts polymorphic non-empty SNPs (standardized
// columns); n_snp_monomorphic is the rest (mono/all-missing, contributing zero columns).
// Only the small N*K + K + counters cross PCIe — the covariance/eigen compute stays on the
// GPU. precision_tag reflects the covariance SYRK (emulated-FP64 default); the eigen is
// always native FP64. Gated vs scikit-allel/sklearn PCA, NOT ADMIXTOOLS2.
struct PcaEig {
    std::vector<double> coords;         // N*K row-major
    std::vector<double> eigenvalues;    // K, descending
    std::vector<double> var_explained;  // K, ratio
    int N = 0;
    int K = 0;
    long n_snp_used = 0;
    long n_snp_monomorphic = 0;
    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// PcaProject — the device result of the lsqproject PCA (`steppe pca --project-*`). The
// eigenbasis is built on the REFERENCE rows only (targets excluded from the covariance AND
// the per-SNP allele frequencies); `coords_ref` (N_ref x K, row-major, == U*S) keeps the
// plain-PCA numbers for the references, and `coords_tgt` (N_tgt x K, row-major, SAME U*S
// units) holds each target's least-squares coordinate a_j = (W_O^T W_O)^{-1} W_O^T z_O (mode
// Lsq) or the diagonal ratio (M_used/m_obs)*W_O^T z_O (mode Scaled / rank-deficient fallback).
// `m_obs[j]` is target j's usable observed-site count; `status_per_target[j]` is 0 (full lsq
// solve), 1 (fell back to the diagonal ratio — A_j rank-deficient / non-SPD), or 2 (target has
// no usable sites, coords zeroed). Eigenvalues/var_explained are the REFERENCE spectrum. Only
// the small coords + spectrum + per-target counters cross PCIe. precision_tag reflects the
// W/b matmuls (emulated-FP64 default); the K x K assembly/solve is native FP64.
struct PcaProject {
    std::vector<double> coords_ref;      // N_ref*K row-major (== U*S)
    std::vector<double> coords_tgt;      // N_tgt*K row-major (lsq / scaled placement)
    std::vector<double> eigenvalues;     // K, descending (reference spectrum)
    std::vector<double> var_explained;   // K, ratio (reference spectrum)
    std::vector<long>   m_obs;           // N_tgt usable observed-site counts
    std::vector<char>   status_per_target;  // N_tgt: 0 lsq, 1 fallback-ratio, 2 no-data
    int N_ref = 0;
    int N_tgt = 0;
    int K = 0;
    long n_snp_used = 0;
    long n_snp_monomorphic = 0;
    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// PcangsdFit — the device result of the PCAngsd genotype-likelihood PCA (`steppe
// pcangsd`): the GL-weighted, individual-allele-frequency covariance C (the .cov),
// the top-e PC coordinates + eigen spectrum, the per-site population allele-2
// frequency, and (optionally) the individual allele-2 frequencies. Produced over
// the device-resident GL tensor by the IAF EM (emMAF -> updateNormal init ->
// updatePCAngsd loop -> covPCAngsd), reusing the cuBLAS SYRK gram + cuSOLVER Dsyevd
// eigen from the `steppe pca` path. `cov` is N*N row-major; `coords` is N*e row-
// major (eigenvector*sqrt(eigenvalue), sign arbitrary per PC). Gated vs the
// reference pcangsd package (NOT ADMIXTOOLS2); concordance, not bit-exact, since
// pcangsd uses float32 internally. precision_tag reflects the gram SYRK
// (emulated-FP64 default); the EM elementwise + the eigen are native FP64.
struct PcangsdFit {
    std::vector<double> cov;            // N*N row-major
    std::vector<double> coords;         // N*e row-major
    std::vector<double> eigenvalues;    // e, descending
    std::vector<double> var_explained;  // e, ratio
    std::vector<double> freq;           // M_used, population allele-2 freq (kept sites)
    std::vector<double> pi;             // M_used*N individual allele-2 freq (empty unless requested)
    int N = 0;
    int e = 0;
    long M_used = 0;
    long M_total = 0;
    int iters_run = 0;
    double final_rmse = 0.0;
    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// AdmixtureFit — the device result of the ADMIXTURE Q/F ML fit (`steppe admixture`). The
// frappe/ADMIXTURE block-EM (GEMM A=QF^T -> native-FP64 binomial responsibility g/A,(2-g)/
// (1-A) -> GEMM) runs SNP-tiled over the device-resident per-individual dosage matrix G,
// alternating the multiplicative F-update (given Q) and Q-update+row-renormalize (given F).
// Q is row-major N x K (rows on the simplex); F is row-major M x K (allele-2 freq in [0,1]).
// When `fixed_F` is supplied (supervised/projection) the F-update is skipped and only Q is
// solved (deterministic, single seed). Otherwise the seed axis is a multi-restart loop, the
// best final log-likelihood winning. The GEMMs run emulated-FP64 (matmul-heavy default); the
// responsibility elementwise + the log-likelihood reduction run native FP64 (the near-0/1
// cancellation carve-out). Gated vs ADMIXTURE (NOT AT2), concordance up to label-switching.
struct AdmixtureFit {
    std::vector<double> Q;             // N*K row-major, rows sum to 1
    std::vector<double> F;             // M*K row-major, in [0,1]
    std::vector<double> seed_loglik;   // per-restart final log-likelihood
    std::vector<int>    seed_iters;    // per-restart iterations run
    std::vector<char>   seed_converged;
    double best_loglik = 0.0;
    int    best_seed = 0;
    int    N = 0, M = 0, K = 0;
    int    iters_run = 0;             // accel steps (squarem) or EM iters (em) for the winner
    int    base_map_evals = 0;        // total base-EM-map evaluations for the winner
    bool   converged = false;
    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

class ComputeBackend;

// Host helper functions — reference §15
namespace core::qpadm {
[[nodiscard]] std::vector<QpAdmResult> fit_models_batched_default(
    ComputeBackend& be, const steppe::device::DeviceF2Blocks& f2,
    std::span<const QpAdmModel> models, const QpAdmOptions& opts);

[[nodiscard]] bool model_in_small_path(const QpAdmModel& model, const QpAdmOptions& opts);

}  // namespace core::qpadm

// The ComputeBackend interface — reference §14
class ComputeBackend {
public:
    ComputeBackend() = default;
    ComputeBackend(const ComputeBackend&) = delete;
    ComputeBackend& operator=(const ComputeBackend&) = delete;
    ComputeBackend(ComputeBackend&&) = delete;
    ComputeBackend& operator=(ComputeBackend&&) = delete;
    virtual ~ComputeBackend() = default;

    [[nodiscard]] virtual F2Result compute_f2(const core::MatView& Q,
                                              const core::MatView& V,
                                              const core::MatView& N,
                                              const Precision& precision) = 0;

    [[nodiscard]] virtual F2BlockTensor compute_f2_blocks(const core::MatView& Q,
                                                          const core::MatView& V,
                                                          const core::MatView& N,
                                                          const int* block_id,
                                                          int n_block,
                                                          const Precision& precision) = 0;

    [[nodiscard]] virtual steppe::device::DeviceF2Blocks compute_f2_blocks_device(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, const Precision& precision) {
        (void)Q; (void)V; (void)N; (void)block_id; (void)n_block; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::compute_f2_blocks_device: not supported by this backend "
            "(device-resident output requires a CUDA backend)");
    }

    [[nodiscard]] virtual steppe::device::DevicePartial compute_f2_blocks_resident(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, int b0, const Precision& precision) {
        (void)Q; (void)V; (void)N; (void)block_id; (void)n_block; (void)b0; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::compute_f2_blocks_resident: not supported by this backend "
            "(device-resident combine requires a peer-capable CUDA backend; the §4 gate "
            "must have routed a non-CUDA/non-peer backend to the host-staged path)");
    }

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

    virtual void compute_f2_blocks_streamed(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, const Precision& precision,
        steppe::device::StreamTarget& target,
        const steppe::device::RedecodeSource* redecode = nullptr) {
        (void)Q; (void)V; (void)N; (void)block_id; (void)n_block; (void)precision; (void)target;
        (void)redecode;
        throw std::runtime_error(
            "ComputeBackend::compute_f2_blocks_streamed: not supported by this backend "
            "(out-of-core block streaming requires a CUDA backend)");
    }

    [[nodiscard]] virtual DecodeResult decode_af(const DecodeTileView& tile) = 0;

    [[nodiscard]] virtual std::vector<int> detect_sample_ploidy_device(
        const DecodeTileView& tile) {
        std::vector<int> ploidy(tile.n_individuals, core::kPloidyPseudoHaploid);
        const std::size_t window =
            (static_cast<std::size_t>(core::kPloidyDetectSnps) < tile.n_snp)
                ? static_cast<std::size_t>(core::kPloidyDetectSnps)
                : tile.n_snp;
        if (window == 0 || tile.n_individuals == 0) return ploidy;
        for (std::size_t g = 0; g < tile.n_individuals; ++g) {
            const std::uint8_t* rec = tile.packed + g * tile.bytes_per_record;
            for (std::size_t s = 0; s < window; ++s) {
                const std::size_t byte_in_rec =
                    s / static_cast<std::size_t>(core::kCodesPerByte);
                const int pos_in_byte =
                    static_cast<int>(s % static_cast<std::size_t>(core::kCodesPerByte));
                if (core::genotype_code(rec[byte_in_rec], pos_in_byte) ==
                    core::kHeterozygousGenotypeCode) {
                    ploidy[g] = core::kPloidyDiploid;
                    break;
                }
            }
        }
        return ploidy;
    }

    [[nodiscard]] virtual CanonicalTile transpose_to_canonical(
        const SnpMajorTileView& view) {
        CanonicalTile out;
        out.n_snp = view.n_snp;
        out.n_individuals = view.n_individuals;
        const std::size_t cpb = static_cast<std::size_t>(core::kCodesPerByte);
        out.bytes_per_record =
            view.n_snp == 0 ? 0 : (view.n_snp + cpb - 1) / cpb;
        if (view.pop_offsets != nullptr && view.n_pop >= 0) {
            out.pop_offsets.assign(
                view.pop_offsets,
                view.pop_offsets + static_cast<std::size_t>(view.n_pop) + 1u);
        }
        const std::size_t total = view.n_individuals * out.bytes_per_record;
        out.packed.assign(total, std::uint8_t{0});
        if (total == 0) return out;

        for (std::size_t g = 0; g < view.n_individuals; ++g) {
            const std::size_t src_row = view.sel_rows[g];
            const std::size_t src_byte_in_snp = src_row / cpb;
            const int src_pos = static_cast<int>(src_row % cpb);
            for (std::size_t b = 0; b < out.bytes_per_record; ++b) {
                std::uint8_t out_byte = 0;
                const std::size_t s0 = b * cpb;
                for (int k = 0; k < core::kCodesPerByte; ++k) {
                    const std::size_t s = s0 + static_cast<std::size_t>(k);
                    if (s >= view.n_snp) break;
                    const std::uint8_t src_byte =
                        view.snp_major[s * view.src_bytes_per_record + src_byte_in_snp];
                    const std::uint8_t code = core::genotype_code(src_byte, src_pos);
                    std::uint8_t canon = code;
                    switch (view.encoding) {
                        case TileEncoding::Identity:
                        default:
                            canon = code;
                            break;
                    }
                    const int shift =
                        (core::kCodesPerByte - 1 - k) * core::kBitsPerCode;
                    out_byte = static_cast<std::uint8_t>(
                        out_byte | static_cast<std::uint8_t>(canon << shift));
                }
                out.packed[g * out.bytes_per_record + b] = out_byte;
            }
        }
        return out;
    }

    [[nodiscard]] virtual steppe::device::DeviceDecodeResult decode_af_compact_autosome(
        const DecodeTileView& tile, std::span<const int> chrom,
        std::span<const double> genpos, std::span<const double> physpos,
        int chrom_min, int chrom_max) {
        (void)tile; (void)chrom; (void)genpos; (void)physpos;
        (void)chrom_min; (void)chrom_max;
        throw std::runtime_error(
            "ComputeBackend::decode_af_compact_autosome: not implemented by this backend "
            "(the device-resident decode seam requires a CUDA backend)");
    }

    [[nodiscard]] virtual steppe::device::DeviceDecodeResult decode_af_compact_filter(
        const DecodeTileView& tile, std::span<const char> ref, std::span<const char> alt,
        std::span<const int> chrom, std::span<const double> genpos,
        std::span<const double> physpos, const FilterConfig& cfg,
        std::span<const std::size_t> pop_individuals, int ploidy, double maxmiss,
        long s_lo) {
        (void)tile; (void)ref; (void)alt; (void)chrom; (void)genpos; (void)physpos;
        (void)cfg; (void)pop_individuals; (void)ploidy; (void)maxmiss; (void)s_lo;
        throw std::runtime_error(
            "ComputeBackend::decode_af_compact_filter: not implemented by this backend "
            "(the regime-B device-resident decode seam requires a CUDA backend)");
    }

    // fst_wc_per_site: the per-site Weir & Cockerham 1984 FST over the population pair
    // (tile pop indices popA, popB) computed by a GPU kernel over the device-resident
    // genotype tile — a distinct per-site variance-component reduction (NOT the AF grid,
    // which drops the allele count n and the observed het h that WC needs). Native FP64.
    // `summary_include[s] == 1` marks a SNP eligible for the genome-wide summary
    // (autosomes, mirroring plink2's WC .fst.summary); the device Σ-reduction sums num/den
    // over sites that are BOTH valid AND summary-included. Default: throw (CUDA/CPU only).
    [[nodiscard]] virtual FstPerSite fst_wc_per_site(
        const DecodeTileView& tile, int popA, int popB,
        std::span<const std::uint8_t> summary_include) {
        (void)tile; (void)popA; (void)popB; (void)summary_include;
        throw std::runtime_error(
            "ComputeBackend::fst_wc_per_site: not implemented by this backend");
    }

    // fst_wc_all_pairs: the all-pairs WC FST matrix (`steppe fst --all-pairs`). Decodes the
    // per-(pop, SNP) sufficient statistic {n, ac, het} ONCE (streamed by SNP-tile) and folds
    // every C(P,2) pair's wc_finalize into the genome-wide per-pair Σnum/Σden/n_valid on the
    // GPU (sweep_unrank k=2 -> the SAME shared wc_finalize the single-pair path uses).
    // `summary_include[s] == 1` marks a SNP eligible for the genome-wide sum (autosomes,
    // matching the single-pair path). `sure` lifts the C(P,2) maxcomb cap. Native FP64.
    // Default: throw (CUDA only; the product is GPU-only for the all-pairs sweep).
    [[nodiscard]] virtual FstMatrix fst_wc_all_pairs(
        const DecodeTileView& tile, std::span<const std::uint8_t> summary_include, bool sure) {
        (void)tile; (void)summary_include; (void)sure;
        throw std::runtime_error(
            "ComputeBackend::fst_wc_all_pairs: not implemented by this backend "
            "(the GPU-only all-pairs sufficient-stat combine requires a CUDA backend)");
    }

    // king_robust_all_pairs: the KING-robust kinship pair sweep (`steppe kinship`). Decodes
    // the per-individual diploid dosage ONCE per SNP-tile (an N x tileM byte tensor) and folds
    // every pair's five KING counts on-device via the SHARED king_classify. When `pairs_i` /
    // `pairs_j` are BOTH empty it sweeps the full C(N,2) upper-triangular set (indexed by the
    // readv2_unrank_pair rank); otherwise it walks exactly those explicit (i, j) index pairs
    // (indexed 0..K-1). `summary_include[s] == 1` marks an autosomal SNP eligible for the count
    // (indexed by the GLOBAL SNP position). `sure` lifts the C(N,2) maxcomb cap (all-pairs only;
    // the explicit-pair list is never capped). Native FP64 tag (integer counts). Default: throw
    // (CUDA only; the product is GPU-only for the pair sweep, like fst_wc_all_pairs).
    [[nodiscard]] virtual KingMatrix king_robust_all_pairs(
        const DecodeTileView& tile, std::span<const std::uint8_t> summary_include,
        std::span<const int> pairs_i, std::span<const int> pairs_j, bool sure) {
        (void)tile; (void)summary_include; (void)pairs_i; (void)pairs_j; (void)sure;
        throw std::runtime_error(
            "ComputeBackend::king_robust_all_pairs: not implemented by this backend "
            "(the GPU-only KING pair sweep requires a CUDA backend)");
    }

    // ld_prune_windowed: windowed-r2 LD pruning (`--ld-prune WIN:STEP:R2`), the plink2
    // --indep-pairwise analogue. Uploads the packed diploid tile device-resident and SNP-tiles a
    // transposed (SNP-major) dosage decode; per variant it reduces the global {nm, Σg, Σg²} (=>
    // major-allele frequency + a monomorphic flag) and, for every within-`window` same-chromosome
    // pair (index distance 1..window-1), computes plink2's exact pairwise-complete integer r^2
    // decision cov12² > r2*(1+eps)*var1*var2 (pairwise deletion, no mean-imputation). Only the
    // per-SNP stats + the per-pair boolean flags cross PCIe. It then runs plink2's greedy backward
    // within-window scan (default --indep-order 2): remove the higher-major-freq (lower-MAF)
    // variant of each over-threshold pair, ties remove the later variant, monomorphic variants are
    // pre-removed; the window slides by `step` and resets at each chromosome boundary. Returns one
    // keep byte per SNP (1 = retained, in tile SNP order). `chrom` is the per-SNP chromosome code
    // (the window-reset key). Default: throw (GPU-only, like the KING/FST pair sweeps).
    [[nodiscard]] virtual std::vector<std::uint8_t> ld_prune_windowed(
        const DecodeTileView& tile, std::span<const int> chrom, int window, int step,
        double r2_thresh) {
        (void)tile; (void)chrom; (void)window; (void)step; (void)r2_thresh;
        throw std::runtime_error(
            "ComputeBackend::ld_prune_windowed: not implemented by this backend "
            "(the GPU-only windowed-r2 LD pruner requires a CUDA backend)");
    }

    // joint_sfs_2pop: the 2D joint site-frequency spectrum over the population pair (tile
    // pop indices popA, popB) accumulated by a GPU joint-histogram over the device-resident
    // genotype tile — the SAME per-pop A1-copy fold the FST path uses, fed into a joint
    // grid instead of the WC algebra. `folded` selects the polarity-free per-pop minor
    // fold (extent N+1) vs the unfolded A1-copy count (extent 2N+1). Only sites complete in
    // both pops are histogrammed. Default: throw (CUDA/CPU only).
    [[nodiscard]] virtual SfsJoint joint_sfs_2pop(const DecodeTileView& tile, int popA,
                                                  int popB, bool folded) {
        (void)tile; (void)popA; (void)popB; (void)folded;
        throw std::runtime_error(
            "ComputeBackend::joint_sfs_2pop: not implemented by this backend");
    }

    // pca_covariance_eig: standalone genotype PCA (`steppe pca`) over the device-resident
    // tile. Patterson-2006 standardizes ALL individuals' diploid dosages on the GPU
    // (per-SNP center 2p + scale 1/sqrt(p(1-p)), missing -> 0, monomorphic columns zeroed)
    // and projects to the top-K PCs (coord = eigenvector*sqrt(eigenvalue)); `k` is clamped
    // to min(k, N). `solver_mode` picks the covariance path: 0 = EXACT (form the dense
    // N x N Gram via cuBLAS SYRK, then the randomized top-K on that resident Gram — the
    // byte-unchanged reference path), 1 = RANDOMIZED (matrix-free: never form N x N; the
    // top-K left singular vectors of the standardized Z via a streamed Z(Zᵀv) matvec range
    // finder, so N scales to biobank size), 2 = AUTO (the backend resolves to randomized
    // when the would-be dense Gram is large / near the VRAM wall, else exact). The
    // covariance matmuls follow `precision` (emulated-FP64 default); the L x L B-eigen /
    // QR are the native-FP64 carve-out. Default: throw (CUDA/CPU-oracle only).
    [[nodiscard]] virtual PcaEig pca_covariance_eig(const DecodeTileView& tile, int k,
                                                    int solver_mode,
                                                    const Precision& precision) {
        (void)tile; (void)k; (void)solver_mode; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::pca_covariance_eig: not implemented by this backend");
    }

    // pca_covariance_eig_dosage: standalone genotype PCA over a REAL-VALUED dosage tile
    // (`steppe pca --bgen`). The exact-path twin of pca_covariance_eig: uploads the FP32
    // DosageTileView device-resident, Patterson-standardizes the ALT dosages on the GPU
    // (per-SNP center 2p + scale 1/sqrt(p(1-p)) over the dosage sum, NaN mean-imputed to 0,
    // monomorphic columns zeroed), forms the dense N x N Gram via cuBLAS SYRK, and runs the
    // SAME truncated top-K eigensolve + coord projection tail. Only the standardize input
    // changes (a fractional dosage in place of a {0,1,2} code); the covariance/eigen math is
    // identical. `k` is clamped to min(k, N); the covariance SYRK follows `precision`
    // (emulated-FP64 default), the eigen is the native-FP64 carve-out. v1 runs the exact
    // dense path only (region/AADR scale); the matrix-free randomized path is a follow-on.
    // Default: throw (CUDA / CPU-oracle only).
    [[nodiscard]] virtual PcaEig pca_covariance_eig_dosage(const DosageTileView& tile, int k,
                                                           const Precision& precision) {
        (void)tile; (void)k; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::pca_covariance_eig_dosage: not implemented by this backend");
    }

    // pca_project_lsq: the smartpca lsqproject PCA (`steppe pca --project-*`). Builds the
    // eigenbasis over the REFERENCE rows only (`ref_rows`, N_ref indices into the tile),
    // folding the per-SNP freq / center / inv_scale over those rows ALONE, then places each
    // TARGET row (`tgt_rows`, N_tgt indices) by a least-squares fit over its non-missing,
    // ref-polymorphic sites. Pass 1 = the ref-only standardize -> SYRK covariance -> truncated
    // top-K eigen (identical numbers to pca_covariance_eig restricted to the references). Pass
    // 2 re-standardizes each SNP tile to form the SNP loadings W = Z_ref^T U diag(1/S_k) and
    // accumulates b = W_O^T z_O + the per-target normal matrix A = W_O^T W_O, then solves the
    // batched K x K SPD system per target. `project_mode` 0 = full lsq (default), 1 = diagonal
    // ratio. Default: throw (CUDA / CPU-oracle only).
    [[nodiscard]] virtual PcaProject pca_project_lsq(const DecodeTileView& tile, int k,
                                                     std::span<const int> ref_rows,
                                                     std::span<const int> tgt_rows,
                                                     int project_mode,
                                                     const Precision& precision) {
        (void)tile; (void)k; (void)ref_rows; (void)tgt_rows; (void)project_mode; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::pca_project_lsq: not implemented by this backend");
    }

    // admixture_fit: the ADMIXTURE Q/F ML fit over the device-resident per-individual dosage
    // matrix (`steppe admixture`). Decodes G [N x M] + validity mask V from the packed tile,
    // seeds Q/F deterministically (admix_init), and runs the block-EM SNP-tiled: A=Q F^T (GEMM,
    // emulated-FP64), binomial responsibilities (native FP64), the multiplicative F-update /
    // Q-update+renormalize (GEMMs), to |dL| < tol*max(1,|L|) or max_iter. When `fixed_F` != null
    // (supervised/projection) the F-update is skipped, F is held at the passed M x K row-major
    // table, and only Q is solved (single deterministic seed); K then comes from fixed_F's
    // columns. Otherwise it runs `seeds` random restarts and keeps the best log-likelihood.
    // Default: throw (CUDA / CPU-oracle only).
    [[nodiscard]] virtual AdmixtureFit admixture_fit(const DecodeTileView& tile, int K,
                                                     const double* fixed_F, long fixed_F_M,
                                                     unsigned long long seed, int seeds,
                                                     int max_iter, double tol, int init_mode,
                                                     int accel_mode, const Precision& precision) {
        (void)tile; (void)K; (void)fixed_F; (void)fixed_F_M; (void)seed; (void)seeds;
        (void)max_iter; (void)tol; (void)init_mode; (void)accel_mode; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::admixture_fit: not implemented by this backend");
    }

    virtual void set_solve_precision(const Precision& precision) { (void)precision; }

    [[nodiscard]] virtual F4Blocks assemble_f4(const steppe::device::DeviceF2Blocks& f2,
                                               std::span<const int> left_idx,
                                               std::span<const int> right_idx,
                                               const Precision& precision) {
        (void)f2; (void)left_idx; (void)right_idx; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f4: not implemented by this backend");
    }

    [[nodiscard]] virtual F4Blocks assemble_f4(const F2BlockTensor& f2,
                                               std::span<const int> left_idx,
                                               std::span<const int> right_idx,
                                               const Precision& precision) {
        (void)f2; (void)left_idx; (void)right_idx; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f4(host): not implemented by this backend");
    }

    [[nodiscard]] virtual F4Blocks assemble_f4_quartets(
        const steppe::device::DeviceF2Blocks& f2,
        std::span<const int> quartets,
        const Precision& precision) {
        (void)f2; (void)quartets; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f4_quartets: not implemented by this backend");
    }

    [[nodiscard]] virtual F4Blocks assemble_f4_quartets(
        const F2BlockTensor& f2,
        std::span<const int> quartets,
        const Precision& precision) {
        (void)f2; (void)quartets; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f4_quartets(host): not implemented by this backend");
    }

    [[nodiscard]] virtual F4Blocks assemble_f3_triples(
        const steppe::device::DeviceF2Blocks& f2,
        std::span<const int> triples,
        const Precision& precision) {
        (void)f2; (void)triples; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f3_triples: not implemented by this backend");
    }

    [[nodiscard]] virtual F4Blocks assemble_f3_triples(
        const F2BlockTensor& f2,
        std::span<const int> triples,
        const Precision& precision) {
        (void)f2; (void)triples; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f3_triples(host): not implemented by this backend");
    }

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

    [[nodiscard]] virtual QpGraphFleetBatch qpgraph_fit_fleet_batch(
        const std::vector<QpGraphTopoArena>& topos, std::span<const double> f_obs,
        std::span<const double> qinv, int numstart, int maxit, double tol,
        const Precision& precision) {
        (void)topos; (void)f_obs; (void)qinv; (void)numstart; (void)maxit; (void)tol;
        (void)precision;
        throw std::runtime_error(
            "ComputeBackend::qpgraph_fit_fleet_batch: not implemented by this backend");
    }

    virtual void dstat_block_reduce(const double* Q, const double* V, int P, long M,
                                    const int* block_id, int n_block,
                                    std::span<const int> quadruples,
                                    double* numsum, double* densum, double* cnt) {
        (void)Q; (void)V; (void)P; (void)M; (void)block_id; (void)n_block;
        (void)quadruples; (void)numsum; (void)densum; (void)cnt;
        throw std::runtime_error(
            "ComputeBackend::dstat_block_reduce: not implemented by this backend");
    }

    virtual void dstat_block_reduce(const steppe::device::DeviceDecodeResult& dec,
                                    const int* block_id, int n_block,
                                    std::span<const int> quadruples,
                                    double* numsum, double* densum, double* cnt) {
        (void)dec; (void)block_id; (void)n_block;
        (void)quadruples; (void)numsum; (void)densum; (void)cnt;
        throw std::runtime_error(
            "ComputeBackend::dstat_block_reduce(DeviceDecodeResult): not implemented "
            "by this backend (the device-resident decode seam requires a CUDA backend)");
    }

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

    virtual void dates_repack(const std::uint8_t* src, std::size_t src_bpr,
                              const long* kept_src, long M_kept, int n_target,
                              std::size_t dst_bpr, std::uint8_t* dst) {
        (void)src; (void)src_bpr; (void)kept_src; (void)M_kept; (void)n_target;
        (void)dst_bpr; (void)dst;
        throw std::runtime_error(
            "ComputeBackend::dates_repack: not implemented by this backend "
            "(CpuBackend uses core::dates::dates_repack_default; CUDA the device gather)");
    }

    virtual std::vector<DatesExpFit> dates_fit(const double* curves, int win_len,
                                               int n_curves, double step, bool affine) {
        (void)curves; (void)win_len; (void)n_curves; (void)step; (void)affine;
        throw std::runtime_error(
            "ComputeBackend::dates_fit: not implemented by this backend "
            "(CpuBackend uses core::dates::dates_fit_default; CUDA the device fit)");
    }

    // ls_forward_backward: the Li-Stephens copying forward-backward for ONE recipient
    // against a K-donor panel over M SNPs — the `steppe paint` FB core. Alleles are
    // haploid {0,1} bytes (any other value == missing -> uninformative emission);
    // `donors` is donor-major (K rows of M). `pi[k]` is the copying prior over donors
    // (uniform 1/K, or leave-one-out with the self donor zeroed), `rho[l]` the
    // recombination probability across the l-1 -> l gap (rho[0] ignored; 1.0 == an
    // unlinked chromosome boundary), `mu[l]` the per-site emission/mutation rate
    // (match -> 1-mu[l], mismatch -> mu[l]). Returns the per-column copying posterior
    // gamma. Phase 0: only the CpuBackend reference oracle implements it (exact,
    // per-column rescaled, scalar, native FP64); the GPU override is Phase 1.
    [[nodiscard]] virtual LsPosterior ls_forward_backward(
        const std::uint8_t* recipient, const std::uint8_t* donors, const double* pi,
        const double* rho, const double* mu, int K, long M, const Precision& precision) {
        (void)recipient; (void)donors; (void)pi; (void)rho; (void)mu; (void)K; (void)M;
        (void)precision;
        throw std::runtime_error(
            "ComputeBackend::ls_forward_backward: not implemented by this backend "
            "(the batched GPU forward-backward is Phase 1; the CpuBackend provides the "
            "reference oracle)");
    }

    // ls_paint_coancestry: the Li-Stephens ChromoPainter coancestry FACE (the `steppe
    // paint` Phase-2 output) over N recipient haplotypes against a K-donor panel. Runs
    // the forward-backward per recipient and folds the copying posterior gamma into the
    // two per-donor coancestry accumulators ON-DEVICE (the K*M posterior is never
    // materialized in this path — the fused steady-state sink). `recipients` is
    // recipient-major (N rows of M), `donors` donor-major (K rows of M); `pi` is the
    // per-recipient copying prior (N*K, self donor zeroed for leave-one-out), `rho`/`mu`
    // as in ls_forward_backward, `w` the per-SNP genetic-length weight (Morgans,
    // build_genetic_weights). Native FP64 (scope §2c). Only the CpuBackend reference
    // oracle and the CUDA override implement it. Reference: li-stephens-phase2 §2.
    [[nodiscard]] virtual LsCoancestry ls_paint_coancestry(
        const std::uint8_t* recipients, const std::uint8_t* donors, const double* pi,
        const double* rho, const double* mu, const double* w, int K, long M, int N,
        const Precision& precision) {
        (void)recipients; (void)donors; (void)pi; (void)rho; (void)mu; (void)w;
        (void)K; (void)M; (void)N; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::ls_paint_coancestry: not implemented by this backend "
            "(the batched GPU coancestry sink is Phase 2; the CpuBackend provides the "
            "reference oracle)");
    }

    // ls_localanc: the Li-Stephens LOCAL-ANCESTRY face over N recipient haplotypes against
    // a K-donor panel partitioned into P ancestry labels (donor_group[k] in [0,P)). Runs
    // the SAME forward-backward as ls_paint_coancestry and folds gamma_l(k) into the
    // per-SNP per-label posterior ON-DEVICE (the K*M gamma is never materialized).
    // `recipients` is recipient-major (N rows of M), `donors` donor-major (K rows of M),
    // `pi` the per-recipient copying prior (N*K), `rho`/`mu` as in ls_forward_backward. No
    // genetic weight and no switch term (localanc is the per-position marginal, not the
    // chunk statistic). Native FP64 (scope §2c). Only the CpuBackend reference oracle and
    // the CUDA override implement it. Reference: li-stephens-phase3-localanc-face-spec §2.
    [[nodiscard]] virtual LsLocalAncestry ls_localanc(
        const std::uint8_t* recipients, const std::uint8_t* donors, const double* pi,
        const double* rho, const double* mu, const int* donor_group,
        int K, long M, int N, int P, const Precision& precision) {
        (void)recipients; (void)donors; (void)pi; (void)rho; (void)mu; (void)donor_group;
        (void)K; (void)M; (void)N; (void)P; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::ls_localanc: not implemented by this backend "
            "(the batched GPU local-ancestry sink is Phase 3; the CpuBackend provides the "
            "reference oracle)");
    }

    // ancibd_fb: the ancIBD 5-state pairwise forward-backward (the `steppe ibd` FB
    // core). Derives the per-haplotype ancestral-allele table from the device GP
    // tensor + phased-GT bits (LoadH5Multi2.get_haplo_prob) and runs the 5-state
    // scaled scan block-per-pair, returning the per-SNP IBD posterior 1 - post[0].
    //   gp3      M*n_sample*3  site-major GP triplet (VCF-native order; g0,g1 used)
    //   phased2  M*n_sample*2  site-major phased GT allele bits {0,1}
    //   p        M             per-SNP derived (ALT) allele freq
    //   T        M*9           per-SNP transition tensor (ancibd_build_transition)
    //   pair_idx n_pair*2      the two sample indices per pair
    // Native FP64. Default: throw (the CpuBackend oracle + the CUDA override implement it).
    [[nodiscard]] virtual AncibdPosterior ancibd_fb(const double* gp3, const std::uint8_t* phased2,
                                                    const double* p, const double* T,
                                                    const int* pair_idx, int n_sample, long M,
                                                    int n_pair, double in_val, double p_min,
                                                    double min_error, const Precision& precision) {
        (void)gp3; (void)phased2; (void)p; (void)T; (void)pair_idx; (void)n_sample; (void)M;
        (void)n_pair; (void)in_val; (void)p_min; (void)min_error; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::ancibd_fb: not implemented by this backend (the CpuBackend "
            "provides the reference oracle; the CUDA override runs the device kernel)");
    }

    // pcangsd_fit: the PCAngsd GL-PCA (the `steppe pcangsd` core). Takes the host
    // site-major likelihood tile (l[(site*n_sample+i)*3 + g], g = copies of A1) +
    // the present mask; the CUDA override uploads it to a resident LikelihoodTensor
    // (residency-checksummed) and runs the IAF EM (emMAF -> updateNormal init ->
    // updatePCAngsd loop -> covPCAngsd) with the cuBLAS SYRK gram + cuSOLVER Dsyevd
    // eigen device-resident; the CpuBackend runs the reference oracle. `e` is the
    // number of PCs / IAF rank; the emMAF (maf_iter/maf_tol) + loop (max_iter/tol)
    // budgets mirror pcangsd's defaults. `want_pi` also returns the individual
    // allele-2 frequencies. Native FP64 EM; emulated-FP64 gram SYRK (via precision).
    [[nodiscard]] virtual PcangsdFit pcangsd_fit(const double* host_l,
                                                 const std::uint8_t* host_present, long n_site,
                                                 int n_sample, int e, int max_iter, double tol,
                                                 double maf, int maf_iter, double maf_tol,
                                                 bool want_pi, const Precision& precision) {
        (void)host_l; (void)host_present; (void)n_site; (void)n_sample; (void)e; (void)max_iter;
        (void)tol; (void)maf; (void)maf_iter; (void)maf_tol; (void)want_pi; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::pcangsd_fit: not implemented by this backend (the CpuBackend "
            "provides the reference oracle; the CUDA override runs the device kernels)");
    }

    // roh_fb: the hapROH (K+1)-state copying forward-backward (the `steppe roh` FB core).
    // Runs the pooled O(K)-per-column scaled scan block-per-target over the device-
    // resident reference-haplotype panel (with checkpoint/recompute), returning the
    // per-SNP ROH posterior 1 - gamma_0(l).
    //   ob       n_target*M   target observed alleles {0,1,missing} (target-major)
    //   refhaps  K*M          reference-haplotype panel bytes {0,1,missing} (donor-major)
    //   p        M            per-SNP panel allele frequency
    //   T        M*9          per-SNP transition tensor (roh_build_transition)
    // Native FP64. Default: throw (the CpuBackend oracle + the CUDA override implement it).
    [[nodiscard]] virtual RohPosterior roh_fb(const std::uint8_t* ob, const std::uint8_t* refhaps,
                                              const double* p, const double* T, int K, long M,
                                              int n_target, double e_rate, double in_val,
                                              const Precision& precision) {
        (void)ob; (void)refhaps; (void)p; (void)T; (void)K; (void)M; (void)n_target;
        (void)e_rate; (void)in_val; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::roh_fb: not implemented by this backend (the CpuBackend "
            "provides the reference oracle; the CUDA override runs the device kernel)");
    }

    // roh_upload_panel: park the hapROH reference panel ONCE for the whole batch run (the
    // batch-overlap residency). `host_panel` is Kpanel rows of Mp donor-major decoded bytes;
    // `donor_map[k]` (length K) is the resident-panel row of selected donor k. Returns a
    // handle the batch driver reuses across every target — the K*Mp panel is uploaded/gathered
    // ONCE, never per target. Default: a host-only handle (no device residency) — the CpuBackend
    // fallback gathers straight from host_panel. Pure residency plumbing (no compute).
    [[nodiscard]] virtual RohPanelHandle roh_upload_panel(const std::uint8_t* host_panel,
                                                          int Kpanel, long Mp, const int* donor_map,
                                                          int K) {
        RohPanelHandle h;
        h.host_panel = host_panel;
        h.donor_map = donor_map;
        h.Kpanel = Kpanel;
        h.K = K;
        h.Mp = Mp;
        return h;
    }

    // roh_fb_batch: run the (K+1)-state FB over N work-items against the ONCE-resident panel,
    // OVERLAPPING each item's host input-build (phase P) and segment-calling (phase C) with the
    // GPU forward-backward of neighbouring items (the batch-overlap pipeline). `build` fills each
    // item's GPU inputs (phase P, run AHEAD); `consume` receives each item's posterior in STRICT
    // item order (phase C). Bit-identical to calling roh_fb per item and consuming in order — it
    // changes only WHEN work happens, never WHAT is computed. Default: the SERIAL CpuBackend
    // oracle path (build -> gather refhaps from host_panel -> roh_fb -> consume, in order).
    virtual void roh_fb_batch(const RohPanelHandle& panel, long N, long Mmax, double e_rate,
                              double in_val, const Precision& precision,
                              const RohItemBuilder& build, const RohPosteriorConsumer& consume) {
        const int K = panel.K;
        if (K <= 0 || Mmax <= 0) return;
        const std::size_t Mm = static_cast<std::size_t>(Mmax);
        std::vector<std::uint8_t> ob(Mm);
        std::vector<int> site_map(Mm);
        std::vector<double> p(Mm), T(Mm * 9);
        std::vector<std::uint8_t> refhaps(static_cast<std::size_t>(K) * Mm);
        for (long i = 0; i < N; ++i) {
            const long M = build(i, ob.data(), site_map.data(), p.data(), T.data());
            if (M <= 0) continue;  // empty item — skipped, exactly like the serial `continue`
            const std::size_t Ms = static_cast<std::size_t>(M);
            for (int k = 0; k < K; ++k) {
                const std::size_t base =
                    static_cast<std::size_t>(panel.donor_map[k]) * static_cast<std::size_t>(panel.Mp);
                for (long l = 0; l < M; ++l)
                    refhaps[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)] =
                        panel.host_panel[base + static_cast<std::size_t>(site_map[static_cast<std::size_t>(l)])];
            }
            RohPosterior post =
                roh_fb(ob.data(), refhaps.data(), p.data(), T.data(), K, M, 1, e_rate, in_val, precision);
            consume(i, post.p_roh.data(), M);
        }
    }

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

    [[nodiscard]] virtual QpfstatsSmooth qpfstats_blocks_smooth(
        const double* Q, const double* V, int P, long M, const int* block_id, int n_block,
        std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
        std::span<const int> block_sizes, double ridge, const Precision& precision) {
        (void)Q; (void)V; (void)P; (void)M; (void)block_id; (void)n_block; (void)quadruples;
        (void)x; (void)npopcomb; (void)npairs; (void)block_sizes; (void)ridge; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::qpfstats_blocks_smooth: not implemented by this backend");
    }

    [[nodiscard]] virtual QpfstatsSmooth qpfstats_blocks_smooth(
        const steppe::device::DeviceDecodeResult& dec, const int* block_id, int n_block,
        std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
        std::span<const int> block_sizes, double ridge, const Precision& precision) {
        (void)dec; (void)block_id; (void)n_block; (void)quadruples;
        (void)x; (void)npopcomb; (void)npairs; (void)block_sizes; (void)ridge; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::qpfstats_blocks_smooth(DeviceDecodeResult): not implemented "
            "by this backend (the device-resident decode seam requires a CUDA backend)");
    }

    [[nodiscard]] virtual JackknifeCov jackknife_cov(const F4Blocks& x,
                                                     std::span<const int> block_sizes,
                                                     double fudge,
                                                     const Precision& precision) {
        (void)x; (void)block_sizes; (void)fudge; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::jackknife_cov: not implemented by this backend");
    }

    [[nodiscard]] virtual JackknifeDiag jackknife_diag(const F4Blocks& x,
                                                       std::span<const int> block_sizes,
                                                       const Precision& precision) {
        (void)x; (void)block_sizes; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::jackknife_diag: not implemented by this backend");
    }

    [[nodiscard]] virtual RatioBlockJackknife ratio_block_jackknife(
        const RatioJackArray& num, const RatioJackArray& den, const RatioJackArray& weight,
        const RatioJackArray& xblk_num, const RatioJackArray& xblk_den, int N, int n_block,
        int tot_mode, double setmiss_thresh, bool compute_p, const Precision& precision) {
        (void)num; (void)den; (void)weight; (void)xblk_num; (void)xblk_den;
        (void)N; (void)n_block; (void)tot_mode; (void)setmiss_thresh; (void)compute_p;
        (void)precision;
        throw std::runtime_error(
            "ComputeBackend::ratio_block_jackknife: not implemented by this backend");
    }

    [[nodiscard]] virtual RatioBlockJackknife f4ratio_blocks_jackknife(
        const steppe::device::DeviceF2Blocks& f2, std::span<const int> flat, int N,
        double setmiss_thresh, const Precision& precision) {
        (void)f2; (void)flat; (void)N; (void)setmiss_thresh; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::f4ratio_blocks_jackknife(DeviceF2Blocks): not implemented by this "
            "backend (the device-resident assemble seam requires a CUDA backend)");
    }

    [[nodiscard]] virtual RatioBlockJackknife f4ratio_blocks_jackknife(
        const F2BlockTensor& f2, std::span<const int> flat, int N, double setmiss_thresh,
        const Precision& precision) {
        (void)f2; (void)flat; (void)N; (void)setmiss_thresh; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::f4ratio_blocks_jackknife(host): not implemented by this backend");
    }

    [[nodiscard]] virtual RatioBlockJackknife dstat_blocks_jackknife(
        const steppe::device::DeviceDecodeResult& dec, const int* block_id, int n_block,
        std::span<const int> quadruples) {
        (void)dec; (void)block_id; (void)n_block; (void)quadruples;
        throw std::runtime_error(
            "ComputeBackend::dstat_blocks_jackknife(DeviceDecodeResult): not implemented by this "
            "backend (the device-resident decode seam requires a CUDA backend)");
    }

    [[nodiscard]] virtual RatioBlockJackknife dstat_blocks_jackknife(
        const double* Q, const double* V, int P, long M, const int* block_id, int n_block,
        std::span<const int> quadruples) {
        (void)Q; (void)V; (void)P; (void)M; (void)block_id; (void)n_block; (void)quadruples;
        throw std::runtime_error(
            "ComputeBackend::dstat_blocks_jackknife(host): not implemented by this backend");
    }

    [[nodiscard]] virtual SweepSurvivors f4_sweep(const steppe::device::DeviceF2Blocks& f2,
                                                  const SweepConfig& cfg,
                                                  const Precision& precision) {
        (void)f2; (void)cfg; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::f4_sweep: not implemented by this backend "
            "(the GPU-only sweep requires a CUDA backend)");
    }

    [[nodiscard]] virtual SweepSurvivors f3_sweep(const steppe::device::DeviceF2Blocks& f2,
                                                  const SweepConfig& cfg,
                                                  const Precision& precision) {
        (void)f2; (void)cfg; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::f3_sweep: not implemented by this backend "
            "(the GPU-only sweep requires a CUDA backend)");
    }

    // READv2 windowed-mismatch kinship — reference §11 (parallel to the f-stat sweep).
    // Three seams over the resident [sample x SNP-window] bit-matrix: allocate+zero it,
    // stream 2-bit genotype chunks into it, then run the all-pairs __popc reduction.
    // The host never touches packed bits; it receives the four per-pair reductions.
    [[nodiscard]] virtual steppe::device::Readv2Bitmatrix readv2_alloc_bitmatrix(
        int n_samples, int window_snps, long m0) {
        (void)n_samples; (void)window_snps; (void)m0;
        throw std::runtime_error(
            "ComputeBackend::readv2_alloc_bitmatrix: not implemented by this backend "
            "(the READv2 packed bit-matrix requires a CUDA backend)");
    }

    virtual void readv2_pack_chunk(steppe::device::Readv2Bitmatrix& bits,
                                   const std::uint8_t* chunk_packed,
                                   std::size_t chunk_bytes_per_record, int n_samples,
                                   long snp0, long snp_count) {
        (void)bits; (void)chunk_packed; (void)chunk_bytes_per_record; (void)n_samples;
        (void)snp0; (void)snp_count;
        throw std::runtime_error(
            "ComputeBackend::readv2_pack_chunk: not implemented by this backend "
            "(the READv2 pack kernel requires a CUDA backend)");
    }

    [[nodiscard]] virtual steppe::device::Readv2Pairs readv2_mismatch(
        const steppe::device::Readv2Bitmatrix& bits, long long n_pairs, bool tiled) {
        (void)bits; (void)n_pairs; (void)tiled;
        throw std::runtime_error(
            "ComputeBackend::readv2_mismatch: not implemented by this backend "
            "(the READv2 all-pairs __popc mismatch sweep requires a CUDA backend)");
    }

    // GL/PL/GP likelihood tensor (the low-coverage PCA / IBD substrate). Two seams:
    // upload the host [n_site x n_sample x 3] FP64 tile (+ present mask) into a
    // resident DeviceBuffer, and a device reduction over it whose sum, compared to
    // the host sum, proves the tensor is genuinely on-device (a device pointer
    // consumed by a kernel), not host-only.
    [[nodiscard]] virtual steppe::device::LikelihoodTensor upload_likelihood_tensor(
        const double* host_l, const std::uint8_t* host_present, long n_site, int n_sample) {
        (void)host_l; (void)host_present; (void)n_site; (void)n_sample;
        throw std::runtime_error(
            "ComputeBackend::upload_likelihood_tensor: not implemented by this backend "
            "(the resident GL tensor requires a CUDA backend)");
    }

    [[nodiscard]] virtual double likelihood_tensor_checksum(
        const steppe::device::LikelihoodTensor& t) {
        (void)t;
        throw std::runtime_error(
            "ComputeBackend::likelihood_tensor_checksum: not implemented by this backend "
            "(the GL tensor device reduction requires a CUDA backend)");
    }

    [[nodiscard]] virtual RankSweep rank_sweep(const F4Blocks& x,
                                               const JackknifeCov& cov,
                                               double alpha,
                                               const QpAdmOptions& opts,
                                               const Precision& precision) {
        (void)x; (void)cov; (void)alpha; (void)opts; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::rank_sweep: not implemented by this backend");
    }

    [[nodiscard]] virtual bool provides_rank_sweep() const { return false; }

    [[nodiscard]] virtual GlsWeights gls_weights(const F4Blocks& x,
                                                 const JackknifeCov& cov,
                                                 int r,
                                                 const QpAdmOptions& opts,
                                                 const Precision& precision) {
        (void)x; (void)cov; (void)r; (void)opts; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::gls_weights: not implemented by this backend");
    }

    [[nodiscard]] virtual std::vector<double> gls_weights_loo_batched(
        const F4Blocks& x, const JackknifeCov& cov, int r,
        const QpAdmOptions& opts, const Precision& precision) {
        (void)x; (void)cov; (void)r; (void)opts; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::gls_weights_loo_batched: not implemented by this backend");
    }

    [[nodiscard]] virtual std::vector<double> se_from_wmat(
        const F4Blocks& x, const JackknifeCov& cov, int r,
        const QpAdmOptions& opts, const Precision& precision) {
        (void)x; (void)cov; (void)r; (void)opts; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::se_from_wmat: not implemented by this backend");
    }

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

    [[nodiscard]] virtual bool provides_batched_fit() const { return false; }

    [[nodiscard]] virtual BackendCapabilities capabilities() const {
        return BackendCapabilities{};
    }

    [[nodiscard]] virtual std::size_t batched_dispatch_count() const { return 0; }
};

}  // namespace steppe

#endif  // STEPPE_DEVICE_BACKEND_HPP

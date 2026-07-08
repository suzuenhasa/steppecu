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
    // (per-SNP center 2p + scale 1/sqrt(p(1-p)), missing -> 0, monomorphic columns zeroed),
    // forms the sample x sample covariance via cuBLAS SYRK (`precision` = emulated-FP64
    // default, matmul-heavy), eigendecomposes it with cuSOLVER Dsyevd (native FP64
    // carve-out), and projects to the top-K PCs (coord = eigenvector*sqrt(eigenvalue)).
    // `k` is clamped to min(k, N). Default: throw (CUDA/CPU-oracle only).
    [[nodiscard]] virtual PcaEig pca_covariance_eig(const DecodeTileView& tile, int k,
                                                    const Precision& precision) {
        (void)tile; (void)k; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::pca_covariance_eig: not implemented by this backend");
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

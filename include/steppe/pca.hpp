// include/steppe/pca.hpp
//
// Public, CUDA-free entry point for standalone genotype PCA (`steppe pca`). Reads a
// .geno/.snp/.ind triple through the shared genotype decode front-end, Patterson-2006
// standardizes the diploid dosages on the device (per-SNP center 2p + scale 1/sqrt(p(1-p)),
// missing mean-imputed to 0, monomorphic SNPs excluded), forms the sample x sample
// covariance via cuBLAS SYRK, eigendecomposes it with cuSOLVER, and projects to the top-K
// principal components — an engine-independent standalone stat (no f2 cache, no fit engine).
//
// Gated against scikit-allel `allel.pca(scaler='patterson')` / sklearn PCA (NOT ADMIXTOOLS2,
// which has no PCA), sign-aligned per PC (|Pearson r| ~ 1) + explained-variance ratio, with a
// Procrustes/principal-angle subspace check for the near-degenerate trailing PCs. Precision
// split: the matmul-heavy covariance SYRK runs the emulated-FP64 default (matches the fit
// policy); the eigendecomposition is the native-FP64 cuSOLVER carve-out.
//
// v1 scope: the exact dense N x N covariance + symmetric eigensolve (right at AADR fixture
// scale, hundreds-to-few-thousand samples). A nonlinear UMAP embedding (--embed umap), EMU
// imputation, randomized SVD for biobank N, and shrinkage projection of new samples are
// documented follow-ups — the coord output + the HTML axisNames/coords schema are
// UMAP-ready (add two axis entries, no schema break).
#ifndef STEPPE_PCA_HPP
#define STEPPE_PCA_HPP

#include <span>
#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace steppe {

namespace device {
struct Resources;
}  // namespace device

// Projection mode for `steppe pca --project-*` (smartpca lsqproject semantics). Lsq is the
// full K x K least-squares placement (== lsqproject:YES); Scaled is the diagonal-only ratio
// a = (M_used/m_obs)*W^T z (the fast / extreme-low-coverage robustness path).
enum class PcaProjectMode : int { Lsq = 0, Scaled = 1 };

// Covariance solver for `steppe pca` (--pca-solver). Exact forms the dense N x N sample Gram
// (right for AADR-fixture N); Randomized is the MATRIX-FREE randomized SVD that never builds
// N x N (top-K left singular vectors of the standardized Z via a streamed Z(Zᵀv) range finder),
// so PCA scales to biobank N past the ~23k quadratic wall; Auto (default) resolves to
// randomized once N is large / the dense Gram nears the VRAM wall, else exact. The two paths
// agree to tight tolerance where both run (Auto keeps small N on the byte-identical exact path,
// so the exact-path goldens are untouched). The int values are the backend `solver_mode` codes.
enum class PcaSolver : int { Exact = 0, Randomized = 1, Auto = 2 };

// PcaResult — the per-sample PC coordinates plus the eigen spectrum and provenance counts.
struct PcaResult {
    // Row-major N x K sample PC coordinates: coord(sample i, PC k) = coords[i*K + k].
    // coord = eigenvector * sqrt(eigenvalue) (== scikit-allel's U*S), sign arbitrary per PC.
    std::vector<double> coords;
    int N = 0;                         // samples
    int K = 0;                         // principal components reported
    int n_ref = 0;                     // reference samples (== N when no projection)

    // Per-sample projection flag in tile (row) order: 1 == placed by lsqproject (target),
    // 0 == a reference sample that BUILT the eigenbasis. All 0 when no --project-* set.
    std::vector<char> is_projected;

    std::vector<double> eigenvalues;   // length K, descending (== allel singular-value^2)
    std::vector<double> var_explained; // length K, eigenvalue_k / Σ_all eigenvalues

    long n_snp_total = 0;              // SNPs in the kept tile (AFTER the QC filter subset)
    long n_snp_used = 0;              // polymorphic, non-empty SNPs (standardized columns)
    long n_snp_monomorphic = 0;       // n_snp_total - n_snp_used (mono/all-missing, dropped)

    // The SNP ids retained by the QC filter (in kept order); empty when no filter was active.
    // Surfaced for --emit-kept-snps (the bring-your-own prune.in / gate hand-off).
    std::vector<std::string> kept_snp_ids;

    // Per-sample identity in tile (row) order — the join key for the oracle gate.
    std::vector<std::string> sample_id;
    std::vector<std::string> sample_pop;
    std::vector<std::string> pop_labels;   // unique populations, first-seen order

    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// run_pca — the standalone genotype-PCA driver. `pops` selects populations to include
// (color groups); an empty span keeps ALL populations. `k` is the number of principal
// components. `precision` governs the covariance SYRK (emulated-FP64 default); the eigen
// solve is always native FP64.
//
// PROJECTION (smartpca lsqproject:YES). `project_pops` (whole populations) and
// `project_samples` (individual Genetic IDs) are placed by least-squares PROJECTION only:
// they are excluded from the eigenbasis AND from the per-SNP allele frequencies, then each
// is coordinated by a per-target K x K least-squares fit over its non-missing sites so low
// coverage does NOT shrink it toward the origin. The two sets combine (union). When BOTH are
// empty the output is identical to the plain (no-projection) path (is_projected all 0, n_ref
// == N). `project_mode` selects the full lsq solve (default) or the diagonal ratio path.
[[nodiscard]] PcaResult run_pca(const std::string& geno,
                                const std::string& snp,
                                const std::string& ind,
                                std::span<const std::string> pops,
                                int k,
                                const Precision& precision,
                                device::Resources& resources,
                                std::span<const std::string> project_pops = {},
                                std::span<const std::string> project_samples = {},
                                PcaProjectMode project_mode = PcaProjectMode::Lsq,
                                const FilterConfig& filter = FilterConfig{},
                                PcaSolver pca_solver = PcaSolver::Auto);

// run_pca_bgen — the BGEN-v1.2 real-valued-dosage PCA driver (`steppe pca --bgen`). Reads a
// biallelic-diploid BGEN into a DosageTile (per-(variant,sample) ALT dosage in [0,2], NaN =
// missing) and runs the Patterson standardize -> SYRK covariance -> top-K eigen on the device
// over the FROM-DOSAGE standardized operand (be.pca_covariance_eig_dosage). Samples are labeled
// by the BGEN sample identifiers, all in a single "ALL" color group (a --pops sidecar is a
// follow-on). `k` is the number of principal components; `precision` governs the covariance
// SYRK (emulated-FP64 default), the eigen solve is always native FP64. The output PcaResult has
// the SAME schema as run_pca, so every emit path (coords/scree/HTML) is reused unchanged.
[[nodiscard]] PcaResult run_pca_bgen(const std::string& bgen_path,
                                     int k,
                                     const Precision& precision,
                                     device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_PCA_HPP

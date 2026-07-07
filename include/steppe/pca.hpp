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

// PcaResult — the per-sample PC coordinates plus the eigen spectrum and provenance counts.
struct PcaResult {
    // Row-major N x K sample PC coordinates: coord(sample i, PC k) = coords[i*K + k].
    // coord = eigenvector * sqrt(eigenvalue) (== scikit-allel's U*S), sign arbitrary per PC.
    std::vector<double> coords;
    int N = 0;                         // samples
    int K = 0;                         // principal components reported

    std::vector<double> eigenvalues;   // length K, descending (== allel singular-value^2)
    std::vector<double> var_explained; // length K, eigenvalue_k / Σ_all eigenvalues

    long n_snp_total = 0;              // SNPs in the kept tile
    long n_snp_used = 0;              // polymorphic, non-empty SNPs (standardized columns)
    long n_snp_monomorphic = 0;       // n_snp_total - n_snp_used (mono/all-missing, dropped)

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
[[nodiscard]] PcaResult run_pca(const std::string& geno,
                                const std::string& snp,
                                const std::string& ind,
                                std::span<const std::string> pops,
                                int k,
                                const Precision& precision,
                                device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_PCA_HPP

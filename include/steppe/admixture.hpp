// include/steppe/admixture.hpp
//
// Public, CUDA-free entry point for model-based ancestry (`steppe admixture`) — the
// ADMIXTURE model of Alexander & Lange 2009: individual ancestry proportions Q [N x K]
// (rows on the simplex) and per-cluster allele-2 frequencies F [M x K], by maximum
// likelihood under the binomial model g_ij ~ Binomial(2, (Q F^T)_ij). The engine is the
// frappe/ADMIXTURE block-EM (GEMM -> native-FP64 responsibility -> GEMM), SNP-tiled and
// device-resident, reusing the pca/pcangsd GEMM + precision substrate.
//
// Three modes, one shared core:
//   - UNSUPERVISED: estimate BOTH Q and F for a given K (multi-seed restarts, best log-lik).
//   - SUPERVISED:   F fixed from labeled reference populations' per-SNP allele frequencies,
//                   solve only Q (the deterministic fixed-P convex solve; gated vs
//                   ADMIXTURE's fixed-P/projection oracle, NOT `-supervised`).
//   - PROJECTION:   F fixed entirely from a prior reference fit (F.tsv), solve only the new
//                   sample(s) Q — the consumer-DNA n=1 payoff where qpAdm structurally fails.
//
// Gated against ADMIXTURE (Alexander, Novembre & Lange 2009), NOT ADMIXTOOLS 2. GPU-vs-CPU
// reference at TIGHT TOLERANCE (emulated-FP64 GEMMs vs scalar native are not bit-identical);
// bit-reproducibility is reserved for the GPU-vs-GPU determinism golden at a fixed
// {seed, init, max-iter, tol, precision}.
#ifndef STEPPE_ADMIXTURE_HPP
#define STEPPE_ADMIXTURE_HPP

#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace steppe {

class ComputeBackend;

// Which of the three modes a run is in.
enum class AdmixtureMode : int { Unsupervised = 0, Supervised = 1, Projection = 2 };

// Init strategy for the unsupervised Q/F EM. v1 ships Random (seeded uniform-Dirichlet Q +
// per-SNP-mean F); Svd (the randomized-subspace warm start) is a Phase-2 follow-up.
enum class AdmixtureInit : int { Random = 0, Svd = 1 };

// EM acceleration. Squarem (default) wraps the untouched base EM map M in the SqS3 quasi-
// Newton control layer (~8x fewer OUTER steps) and runs all restarts concurrently as one
// batched-GEMM stack; Em forces the plain fixed-point loop (bit-identical to the base map,
// one seed at a time). The accelerator is a SWAPPABLE layer over M, never baked into it.
enum class AdmixtureAccel : int { Squarem = 0, Em = 1 };

// AdmixtureParams — the run knobs (mode, K, RNG, loop caps, precision). Mode-specific inputs
// (supervised labels, projection F + SNP ids) travel in AdmixtureInputs below.
struct AdmixtureParams {
    AdmixtureMode mode = AdmixtureMode::Unsupervised;
    int K = 0;                                  // ancestral components (>=1, <N); from F cols in projection
    unsigned long long seed = 42;               // RNG seed (recorded in meta)
    int seeds = 1;                              // random-restart count, best-loglik wins (unsupervised)
    AdmixtureInit init = AdmixtureInit::Random;
    AdmixtureAccel accel = AdmixtureAccel::Squarem;  // SQUAREM (default) | plain EM
    int max_iter = 200;
    double tol = 1e-6;                          // stop on |dL| < tol*max(1,|L|)
    Precision precision = Precision::emulated_fp64();
    // Per-SNP QC filter, applied to the SNP axis before the EM (inactive default = no-op).
    FilterConfig filter = FilterConfig{};
};

// AdmixtureResult — the per-run output tables + provenance. Q is row-major N x K (row per
// retained individual, in tile order, sums to 1); F is row-major M x K (row per retained
// SNP). seed_loglik/iters/converged carry the per-restart trace (winner at best_seed).
struct AdmixtureResult {
    std::vector<double> Q;                       // N*K row-major, rows sum to 1
    std::vector<double> F;                       // M*K row-major, in [0,1]
    int N = 0, M = 0, K = 0;

    std::vector<std::string> sample_id;          // N, tile order (join key)
    std::vector<std::string> sample_pop;         // N, tile order

    std::vector<double> seed_loglik;             // per-restart final log-likelihood
    std::vector<int>    seed_iters;              // per-restart iterations run
    std::vector<char>   seed_converged;          // per-restart converged flag
    double best_loglik = 0.0;
    int    best_seed = 0;
    int    iters_run = 0;                         // accel steps (squarem) or EM iters (em)
    int    base_map_evals = 0;                    // total base-EM-map evaluations (winner)
    bool   converged = false;

    long n_snp_total = 0;
    long n_indiv = 0;

    // SNP ids retained by the QC filter (kept order); empty when no filter was active.
    std::vector<std::string> kept_snp_ids;

    AdmixtureMode mode = AdmixtureMode::Unsupervised;
    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// Mode-specific inputs. For SUPERVISED, `supervised_labels` names the reference population
// labels that define the K clusters (F column k = that pop's per-SNP allele-2 frequency;
// the labels must be among the selected --pops). For PROJECTION, `fixed_F` is the row-major
// M x K reference frequency table (M rows = the retained SNP count; K = its columns), and
// `fixed_F_snps` — when non-empty — are its per-row SNP ids for the intersection guard.
struct AdmixtureInputs {
    std::vector<std::string> supervised_labels;
    std::vector<double>      fixed_F;            // M*K row-major (projection)
    long                     fixed_F_M = 0;
    int                      fixed_F_K = 0;
    std::vector<std::string> fixed_F_snps;       // optional M ids (projection SNP-set guard)
};

// run_admixture — the host-pure orchestrator: reads the triple through the shared genotype
// front-end, decodes per-individual dosage device-resident, builds the mode's fixed F (if
// any), runs the EM / Q-only solve on the backend, and fills AdmixtureResult. `pops` selects
// populations (empty = all with MinN>=1).
[[nodiscard]] AdmixtureResult run_admixture(const std::string& geno, const std::string& snp,
                                            const std::string& ind,
                                            const std::vector<std::string>& pops,
                                            const AdmixtureParams& params,
                                            const AdmixtureInputs& inputs, ComputeBackend& be);

}  // namespace steppe

#endif  // STEPPE_ADMIXTURE_HPP

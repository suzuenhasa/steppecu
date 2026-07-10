// src/app/cmd_admixture.hpp
//
// The `steppe admixture` command entry point: model-based ancestry Q/F (the ADMIXTURE model
// of Alexander & Lange 2009) — individual ancestry proportions Q [N x K] + per-cluster allele
// frequencies F [M x K] by maximum likelihood, via the GPU block-EM. Three modes share one
// core: unsupervised (estimate Q and F), supervised (F fixed from labeled reference pops,
// solve Q), and projection (F fixed from a prior fit, solve one/many samples' Q — the
// consumer-DNA n=1 payoff). Reads a genotype triple through the shared front-end and runs the
// EM device-resident (GPU when visible, else the CpuBackend reference oracle). Writes
// out-dir/{Q.tsv, F.tsv (--emit-F), loglik.txt, meta.json}.
//
// Self-contained args (like `pcangsd`/`ibd`/`roh`; no shared RunConfig). Gated vs ADMIXTURE
// (NOT ADMIXTOOLS2). Reference: docs/planning/admixture-qp-scope.md.
#ifndef STEPPE_APP_CMD_ADMIXTURE_HPP
#define STEPPE_APP_CMD_ADMIXTURE_HPP

#include <string>

namespace steppe::app {

struct AdmixtureArgs {
    std::string prefix;             // PREFIX.{geno,snp,ind} [required]
    std::string pops;               // --pops FILE (one population label per line; empty = all)
    int K = 0;                      // -K ancestral components [required for unsupervised]
    unsigned long long seed = 42;   // RNG seed (recorded in meta)
    int seeds = 1;                  // random-restart count, best-loglik wins (unsupervised)
    std::string init = "random";    // init: random (v1 default) | svd (Phase 2)
    int max_iter = 200;
    double tol = 1e-6;
    std::string supervised;         // --supervised FILE: labeled reference pop labels (one/line)
    std::string project_onto;       // --project-onto F.tsv: fixed reference F table
    std::string project_samples;    // --project-samples FILE (optional; ids to project)
    bool emit_F = false;            // also write F.tsv (large)
    std::string precision = "emu";  // GEMM precision: emu | fp64
    std::string device;             // CUDA device ordinal (default auto)
    std::string out_dir;            // output directory [required]
    std::string format = "tsv";     // field separator: tsv | csv
};

[[nodiscard]] int run_admixture_cmd(const AdmixtureArgs& args);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_ADMIXTURE_HPP

// src/app/cmd_pcangsd.hpp
//
// The `steppe pcangsd` command entry point: PCAngsd (Meisner & Albrechtsen 2018)
// genotype-likelihood PCA — individual allele frequencies + a GL-weighted covariance
// + top-e PCA from per-site genotype LIKELIHOODS for low-coverage samples, via the
// iterative IAF EM. Reads a beagle GL file through the shipped LikelihoodTile ->
// resident LikelihoodTensor path and runs the EM device-resident (GPU when visible,
// else the CpuBackend reference oracle). Writes PREFIX.{cov,eigenvec,eigenval} and,
// optionally, PREFIX.freq / PREFIX.pi — PCAngsd file conventions.
//
// Self-contained args (like `ibd`/`roh`; no shared RunConfig). Gated vs the reference
// pcangsd package (NOT ADMIXTOOLS2). Reference: docs/planning/pcangsd-gl-scope.md.
#ifndef STEPPE_APP_CMD_PCANGSD_HPP
#define STEPPE_APP_CMD_PCANGSD_HPP

#include <string>

namespace steppe::app {

struct PcangsdArgs {
    std::string beagle;          // beagle GL file (.beagle.gz / plain) [required]
    int eig = 0;                 // -e: number of PCs / IAF rank [required >= 1 in v1]
    int iter = 100;              // main-loop iteration cap (pcangsd --iter)
    double tole = 1e-5;          // main-loop convergence (pcangsd --tole)
    double maf = 0.05;           // per-site MAF filter (pcangsd --maf)
    int maf_iter = 500;          // emMAF iteration cap (pcangsd --maf-iter)
    double maf_tole = 1e-6;      // emMAF convergence (pcangsd --maf-tole)
    bool emit_freq = false;      // also write PREFIX.freq (population allele-2 freq)
    bool emit_iaf = false;       // also write PREFIX.pi (N*M individual allele-2 freq; large)
    std::string precision = "emu";  // gram SYRK precision: emu | fp64
    std::string device;          // CUDA device ordinal (default auto)
    std::string out;             // output PREFIX (required)
    std::string format = "tsv";  // matrix field separator: tsv | csv
};

[[nodiscard]] int run_pcangsd_cmd(const PcangsdArgs& args);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_PCANGSD_HPP

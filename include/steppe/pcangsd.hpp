// include/steppe/pcangsd.hpp
//
// Public, CUDA-free entry point for PCAngsd genotype-likelihood PCA (`steppe
// pcangsd`, Meisner & Albrechtsen 2018). Reads a beagle GL file into the shipped
// LikelihoodTile, uploads it to a resident device LikelihoodTensor, and runs the
// individual-allele-frequency EM + GL-weighted covariance + top-e PCA on the device
// (emMAF -> updateNormal init -> updatePCAngsd loop -> covPCAngsd), reusing the
// cuBLAS SYRK gram + cuSOLVER eigen from the `steppe pca` path.
//
// Gated against the reference pcangsd package (NOT ADMIXTOOLS2, which has no GL
// path), on CONCORDANCE (covariance relative-Frobenius + sign-aligned per-PC
// |Pearson r| ~ 1 + IAF/freq correlation), since pcangsd uses float32 internally.
// Precision split: native FP64 for the EM elementwise (the GL cancellation
// carve-out) + every eigen; emulated-FP64 default for the matmul-heavy gram SYRK.
//
// v1 scope: the core covariance + PCA + individual allele frequencies (the headline
// PCAngsd output). The admixture (Q/P), selection scan, HWE/inbreeding, kinship, VCF
// PL/GL input, and auto -e (MAP test) extensions are documented follow-ups.
#ifndef STEPPE_PCANGSD_HPP
#define STEPPE_PCANGSD_HPP

#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace steppe {

class ComputeBackend;

// PCAngsd knobs; defaults mirror the reference pcangsd package.
struct PcangsdParams {
    int e = 0;                 // number of PCs / IAF rank (pcangsd -e); required >= 1 in v1
    int iter = 100;            // main-loop iteration cap (pcangsd --iter)
    double tol = 1e-5;         // main-loop convergence (pcangsd --tole; rmse2d of the IAF)
    double maf = 0.05;         // per-site minor-allele-freq filter (pcangsd --maf)
    int maf_iter = 500;        // emMAF iteration cap (pcangsd --maf-iter)
    double maf_tol = 1e-6;     // emMAF convergence (pcangsd --maf-tole)
    bool want_pi = false;      // also return the individual allele-2 frequencies
    Precision precision = Precision::emulated_fp64();  // gram SYRK precision (default emu)
};

// PcangsdResult — the GL-weighted covariance, PCA coords + spectrum, per-site
// population allele-2 frequency, and (optionally) individual allele-2 frequencies,
// plus provenance. `cov` is N*N row-major; `coords` is N*e row-major (eigenvector *
// sqrt(eigenvalue), sign arbitrary per PC). All frequencies are allele-2 (the second
// beagle allele), matching pcangsd's freq/IAF sign.
struct PcangsdResult {
    std::vector<double> cov;            // N*N row-major (the .cov)
    std::vector<double> coords;         // N*e row-major (the .eigenvec)
    std::vector<double> eigenvalues;    // e, descending (the .eigenval)
    std::vector<double> var_explained;  // e, ratio
    std::vector<double> freq;           // M_used population allele-2 freq (the .freq)
    std::vector<double> pi;             // M_used*N individual allele-2 freq (the .pi), if requested

    std::vector<std::string> sample_ids;  // N, beagle header order (join key for the gate)

    int N = 0;                  // individuals
    int e = 0;                  // PCs reported
    long M_used = 0;            // sites kept after the MAF filter
    long M_total = 0;           // sites in the beagle file
    int iters_run = 0;         // main-loop iterations to convergence (or the cap)
    double final_rmse = 0.0;   // final IAF rmse2d

    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// run_pcangsd — read the beagle GL file, upload the GL tensor, and run the PCAngsd
// fit on `be` (a CUDA backend for the device path, or the CpuBackend reference
// oracle). Throws std::runtime_error on a beagle parse/IO failure.
[[nodiscard]] PcangsdResult run_pcangsd(const std::string& beagle, const PcangsdParams& params,
                                        ComputeBackend& be);

}  // namespace steppe

#endif  // STEPPE_PCANGSD_HPP

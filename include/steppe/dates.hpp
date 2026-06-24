// include/steppe/dates.hpp
//
// PUBLIC, CUDA-FREE admixture-DATING entry point — the DATES tool (Loh 2013 ALDER /
// Chintalapati-Patterson-Moorjani 2022 DATES). UNLIKE the f2-cache tools, run_dates reads
// the GENOTYPE TRIPLE (.geno/.snp/.ind) directly through the SAME extract-f2 decode
// front-end (io reader + decode_af [per-SNP source allele freqs] + the per-SNP genpos seam),
// then DIVERGES into a NEW cuFFT autocorrelation LD engine — it NEVER touches the f2 cache.
//
// THE STATISTIC. Admixture creates ancestry tracts that recombination fragments each
// generation, so the weighted ancestry-COVARIANCE between SNP PAIRS decays exponentially
// with the genetic distance d between them: cov(d) ~ A*exp(-lambda*d) + c, lambda = the
// generations since admixture. PER admixed sample i (summed over the admixed population):
//   wt[s]  = freq(source1, s) - freq(source2, s)            // population allele-freq delta
//   w0[s]  = genotype(i, s) / 2                              // the admixed sample dosage
//   y      = dot(w0-w2, w1-w2) / dot(w1-w2, w1-w2)           // regress dosage on the sources
//   res[s] = w0[s] - (y*w1[s] + (1-y)*w2[s])                // residual ancestry signal
//   signal[s] = res[s] * wt[s]
// scattered onto a FINE genetic-map grid (spacing binsize/qbin) per chromosome as three
// moment arrays z0q (count), z1q (signal), z2q (signal^2). The binned cov(d) is the
// AUTOCORRELATION of z1q normalized by the autocorrelation of z0q (the ALDER FFT trick:
// cov(d=lag) = sum_g z1q[g]*z1q[g+lag] / sum_g z0q[g]*z0q[g+lag] = IFFT(|FFT|^2)/IFFT(|FFT|^2)),
// so the ~10^12 SNP-pair object is NEVER materialized — O(G log G) cuFFT, flat in M.
// The reported curve column is the NORMALIZED correlation v12/sqrt(v11*v22) (DATES datacol 3):
//   S0 += dd00 (count autocorr), S12 += dd11 (signal autocorr), S11 += dd02, S22 += dd20,
//   corr = (S12/S0) / sqrt((S11/S0)*(S22/S0)). A*exp(-lambda*d)+c is fit over d in
//   [lovalfit, maxdis] (DATES default 0.45 cM .. 100 cM), lambda -> date in generations;
//   a LEAVE-ONE-CHROMOSOME weighted block jackknife gives the SE (DATES dowtjack / weightjack).
//
// PINNED to the DATES reference C source (github.com/priyamoorjani/DATES, Version 750; built
// on box5090): dates.c:604 (wt = w1-w2), :620-630 (residual regression), :655-665 (grid
// scatter z0q/z1q/z2q), :1280-1345 (ddadd FFT moment assembly), :1352-1380 (ddcorr lag->bin),
// :896-945 (dumpit -> the corr curve), fitexp.c (the exp+affine fit), statsubs.c weightjack
// (the leave-one-chrom SE). GOLDEN: aadr_packed PUR<-{CEU,YRI} -> 9.742 gen, SE 0.317
// (tests/reference/goldens/dates/aadr_PUR_CEU_YRI). The default-weight (population-delta)
// form is RESOLVED — NO per-sample likelihood K_i kernel.
//
// CUDA-FREE BY CONTRACT (architecture.md §4): standard-C++ only; the per-SNP weight/residual,
// the grid scatter, and the batched cuFFT autocorrelation route through the ComputeBackend
// seam (the kernels are .cu PRIVATE to steppe_device; the CpuBackend gives the native
// FFT-free reference oracle). device::Resources is forward-declared CUDA-free; the app/binding
// reach the GPU ONLY through this seam (like run_dstat).
#ifndef STEPPE_DATES_HPP
#define STEPPE_DATES_HPP

#include <string>
#include <vector>

#include "steppe/config.hpp"  // steppe::Precision
#include "steppe/error.hpp"   // steppe::Status

namespace steppe {

namespace device {
struct Resources;  // CUDA-free fwd-decl (real decl: src/device/resources.hpp)
}  // namespace device

/// The knobs that govern a DATES run (DATES par.dates fields; defaults == the reference
/// par.dates the goldens used). All distances in MORGANS internally; the CLI exposes cM where
/// the reference does (lovalfit is cM in par.dates -> converted here).
struct DatesOptions {
    /// Output bin width in MORGANS (DATES `binsize`, default 0.001 == 0.1 cM).
    double binsize_morgans = 0.001;
    /// Fine-grid refinement factor (DATES `qbin`, default 10): the scatter grid spacing is
    /// binsize/qbin == 0.0001 Morgans (0.01 cM). The cuFFT autocorrelation runs on this grid.
    int qbin = 10;
    /// Max genetic distance the curve covers in MORGANS (DATES `maxdis`, default 1.0 == 100 cM).
    double maxdis_morgans = 1.0;
    /// Fit-window low edge in cM (DATES `lovalfit`, default 0.45 cM) — bins below this are
    /// dropped from the exp fit (avoids background LD). The high edge is maxdis.
    double lovalfit_cm = 0.45;
    /// Affine mode (DATES `afffit: YES`): fit A*exp(-lambda*d) + c (constant term). Default on.
    bool affine = true;
    /// RNG seed for the fitter's multi-start initialization (DATES `seed`, default 77). The
    /// final date is deterministic given the curve (the GSL refine converges to the same min),
    /// but the seed pins the multi-start path for bit-comparability with the reference.
    long seed = 77;
};

/// One DATES result: the date (generations since admixture), its block-jackknife SE, the
/// goodness-of-fit (the fit residual sd, DATES "error sd"), and the binned covariance curve.
struct DatesResult {
    /// Generations since admixture (the weighted-jackknife point estimate; DATES .jout col 1).
    double date_gen = 0.0;
    /// Standard error of `date_gen` (leave-one-chromosome weighted block jackknife; .jout col 2).
    double se = 0.0;
    /// Fit residual standard deviation over the fit window (DATES expfit "error sd") — the
    /// goodness datum (smaller = cleaner decay). NRMSD-style gate lives in the caller.
    double fit_error_sd = 0.0;
    /// The binned covariance/correlation curve: distance in cM (curve_cm[k]) vs the normalized
    /// correlation (curve_corr[k]) — the DATES .out columns 1 & 4 (datacol 3). Length n_bin.
    std::vector<double> curve_cm;
    std::vector<double> curve_corr;
    /// The per-chromosome leave-one-out date estimates (DATES .jin col 3) and their SNP-count
    /// weights (.jin col 2 == total - chrom count) — surfaced for inspection/parity.
    std::vector<double> loo_date_gen;   ///< length n_chrom_present (the LOO replicates).
    std::vector<double> loo_weight;     ///< parallel weights (count_total - count[chrom]).

    /// PER-CALL outcome. A degenerate run (no decay, fit failure) is Status, not an exception
    /// (architecture.md §10; record-and-continue). NaN date on a degenerate curve.
    Status status = Status::Ok;
    /// Which arithmetic produced the curve (the cuFFT autocorrelation is native double FFT;
    /// the weight/residual is the native cancellation carve-out — always Fp64).
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

/// Admixture dating over the genotype triple (.geno/.snp/.ind) — the DATES tool.
/// `geno`/`snp`/`ind` are the EIGENSTRAT/packedancestrymap triple paths (steppe reads
/// TGENO or GENO packed; the .snp MUST carry a real genetic map in its genpos column —
/// DATES needs cM positions). `source1`/`source2` are the two reference (ancestral) source
/// population labels; `target` is the admixed population whose individuals are dated. The
/// covariance is summed over EVERY individual of `target` (DATES per-sample main loop), the
/// two sources enter as the population-delta weight wt = freq(source1) - freq(source2).
/// `opts` carries the bin/grid/fit knobs (defaults == the reference par.dates the goldens
/// used). Routes the weight/residual + grid scatter + batched cuFFT autocorrelation through
/// the ComputeBackend seam (CpuBackend oracle / CudaBackend cuFFT). A degenerate run is a
/// Status outcome, never an exception (an io fault PROPAGATES as an exception for the app to
/// map to a nonzero exit).
[[nodiscard]] DatesResult run_dates(const std::string& geno, const std::string& snp,
                                    const std::string& ind, const std::string& target,
                                    const std::string& source1, const std::string& source2,
                                    const DatesOptions& opts, device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_DATES_HPP

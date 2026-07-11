// src/core/internal/pca_standardize.hpp
//
// The shared Patterson-2006 PCA standardization primitive: given a diploid SNP's
// allele-dosage sum and non-missing count, it folds them into the per-SNP center (2p)
// and the variance p(1-p), and standardizes one diploid code to (g - 2p)/sqrt(p(1-p))
// with missing mean-imputed to 0. Every helper is a STEPPE_HD inline so the GPU kernel
// (pca_standardize_kernel.cu) and the CPU reference oracle (cpu_backend.cpp
// pca_covariance_eig) call the exact same native-FP64 arithmetic and cannot drift on the
// centering, the scaling, the missing convention, or the monomorphic exclusion.
//
// This is the scikit-allel `allel.pca(scaler='patterson')` normalization: center by the
// per-SNP mean 2p (p = ref-allele freq over non-missing samples), scale by 1/sqrt(p(1-p)),
// and mean-impute a missing genotype (which, pre-filled to the mean 2p, centers to 0). A
// monomorphic or all-missing SNP (p(1-p) == 0) has inv_scale forced to 0, zeroing its whole
// standardized column — equivalent to dropping it. NO 1/M scaling (the coords stay equal to
// allel's U*S; any global scale is |r|- and variance-ratio-invariant).
//
// sqrt is deliberately kept OUT of this header (a __host__ __device__ inline calling sqrt
// portably is a footgun): pca_snp_scale returns the variance pq and the `used` flag, and
// each call site computes inv_scale = used ? 1.0/sqrt(pq) : 0.0 with its own sqrt (the CUDA
// device builtin in the kernel, std::sqrt in the host oracle/test).
#ifndef STEPPE_CORE_INTERNAL_PCA_STANDARDIZE_HPP
#define STEPPE_CORE_INTERNAL_PCA_STANDARDIZE_HPP

#include <cstdint>

#include "core/internal/decode_af.hpp"    // kMissingGenotypeCode, genotype_valid
#include "core/internal/host_device.hpp"  // STEPPE_HD

namespace steppe::core {

// Per-SNP Patterson standardization parameters, folded from the diploid allele-count.
struct PcaSnpScale {
    double p = 0.0;        // ref-allele frequency over non-missing samples
    double center = 0.0;   // 2p (the per-SNP mean of the diploid dosage)
    double pq = 0.0;       // p(1-p) (the per-SNP variance; inv_scale = 1/sqrt(pq))
    bool used = false;     // pq > 0 (polymorphic AND some non-missing data); false => drop
};

// Fold the per-SNP allele-dosage sum ac (Σ diploid codes over non-missing samples) and the
// non-missing sample count n_nonmiss into the Patterson center/variance. An all-missing SNP
// (n_nonmiss == 0) or a monomorphic SNP (p == 0 or p == 1) yields used == false.
[[nodiscard]] STEPPE_HD inline PcaSnpScale pca_snp_scale(long ac, long n_nonmiss) noexcept {
    PcaSnpScale s;
    if (n_nonmiss > 0) {
        const double p = static_cast<double>(ac) / (2.0 * static_cast<double>(n_nonmiss));
        s.p = p;
        s.center = 2.0 * p;
        s.pq = p * (1.0 - p);
        s.used = s.pq > 0.0;
    }
    return s;
}

// Standardize one diploid code given the SNP's center (2p) and inv_scale (1/sqrt(p(1-p)),
// or 0 for a monomorphic/all-missing SNP). A missing code maps to 0 (mean-imputed after
// centering — the allel PattersonScaler convention); a monomorphic column (inv_scale == 0)
// is 0 for every sample.
[[nodiscard]] STEPPE_HD inline double pca_standardize_one(std::uint8_t code, double center,
                                                          double inv_scale) noexcept {
    if (!genotype_valid(code)) return 0.0;
    return (static_cast<double>(code) - center) * inv_scale;
}

// ---- Real-valued (BGEN) dosage overloads ----
//
// The DosageTile path feeds the SAME Patterson standardization a fractional ALT dosage
// (a real number in [0, 2]) instead of a decoded {0,1,2} code. The arithmetic is
// IDENTICAL — p is the mean dosage / 2, center 2p, variance p(1-p), z = (dosage-2p)/
// sqrt(p(1-p)) — so the dosage kernels and the integer kernels cannot drift. A missing
// dosage is a NaN sentinel; dosage_valid is the float analogue of genotype_valid.
// (isnan is deliberately kept out of a __host__ __device__ inline — the self-comparison
// d == d is false iff d is NaN and is portable across host and device.)
[[nodiscard]] STEPPE_HD inline bool dosage_valid(float d) noexcept { return d == d; }

// Fold the per-SNP dosage SUM (Σ ALT dosage over non-missing samples, a real number)
// and the non-missing count into the Patterson center/variance. Mirrors pca_snp_scale
// but takes a double sum instead of an integer allele count.
[[nodiscard]] STEPPE_HD inline PcaSnpScale pca_snp_scale_f(double dosage_sum,
                                                           long n_nonmiss) noexcept {
    PcaSnpScale s;
    if (n_nonmiss > 0) {
        const double p = dosage_sum / (2.0 * static_cast<double>(n_nonmiss));
        s.p = p;
        s.center = 2.0 * p;
        s.pq = p * (1.0 - p);
        s.used = s.pq > 0.0;
    }
    return s;
}

// Standardize one ALT dosage given the SNP's center (2p) and inv_scale. A missing
// (NaN) dosage maps to 0 (mean-imputed after centering); a monomorphic column
// (inv_scale == 0) is 0 for every sample — exactly the integer convention.
[[nodiscard]] STEPPE_HD inline double pca_standardize_one_f(float dosage, double center,
                                                            double inv_scale) noexcept {
    if (!dosage_valid(dosage)) return 0.0;
    return (static_cast<double>(dosage) - center) * inv_scale;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_PCA_STANDARDIZE_HPP

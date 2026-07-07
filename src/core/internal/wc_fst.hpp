// src/core/internal/wc_fst.hpp
//
// The shared Weir & Cockerham (1984) per-site FST primitive: given the per-population
// non-missing count, allele-dosage sum, and observed-heterozygote count at one SNP for
// two populations (r = 2), it folds them into the WC variance components a/b/c and the
// per-site FST = a / (a + b + c). Every helper is a STEPPE_HD inline so the GPU kernel
// and the CPU reference oracle call the exact same native-FP64 arithmetic and cannot
// drift (the reduction/cancellation carve-out — the EmulatedFp64 matmul policy does NOT
// bind FST; this is a per-site reduction, not a GEMM).
//
// The estimator matches plink2 `--fst method=wc report-variants` term-for-term:
//   FST_NUMER == a,  FST_DENOM == a + b + c,  WC_FST == a / (a + b + c).
// Every term is symmetric under the REF<->ALT swap (p -> 1-p, h -> h), so EIGENSTRAT vs
// PLINK allele polarity does not change num/den/fst — a row join needs only the SNP id.
//
// Diploid interpretation is forced: each 2-bit code is a diploid dosage 0/1/2 (missing =
// kMissingGenotypeCode), and a heterozygote is code == kHeterozygousGenotypeCode.
#ifndef STEPPE_CORE_INTERNAL_WC_FST_HPP
#define STEPPE_CORE_INTERNAL_WC_FST_HPP

#include <cstdint>

#include "core/internal/decode_af.hpp"     // kMissingGenotypeCode, kHeterozygousGenotypeCode
#include "core/internal/host_device.hpp"   // STEPPE_HD

namespace steppe::core {

// Per-population per-SNP accumulator: non-missing individual count n, allele-dosage sum
// ac (sum of 0/1/2 codes), and observed-heterozygote count het.
struct WcPerPop {
    double n = 0.0;
    double ac = 0.0;
    double het = 0.0;
};

// The finalized per-site WC record. On an invalid site (a pop with no data, a combined
// sample size <= 2, or a zero/degenerate denominator — e.g. a site monomorphic across
// the pair) num and den are forced to EXACTLY 0.0 (never NaN/inf, so a device
// Σ-reduction masked by valid can never be poisoned) and fst is the NaN sentinel.
struct WcSite {
    double num = 0.0;   // a
    double den = 0.0;   // a + b + c
    double fst = 0.0;   // a / (a + b + c)
    bool valid = false;
};

// A portable quiet NaN usable in both host and device code.
[[nodiscard]] STEPPE_HD inline double wc_nan() noexcept { return __builtin_nan(""); }

// Fold one diploid code into a population accumulator (missing excluded per-site
// per-sample: a missing code advances nothing).
STEPPE_HD inline void wc_accumulate(std::uint8_t code, WcPerPop& acc) noexcept {
    if (code == kMissingGenotypeCode) return;
    acc.n += 1.0;
    acc.ac += static_cast<double>(code);                                  // 0 / 1 / 2
    acc.het += (code == kHeterozygousGenotypeCode) ? 1.0 : 0.0;
}

// Combine two population accumulators into the WC (r = 2) variance components and FST.
// The algebra is the textbook Weir & Cockerham 1984 (eqns 2-4) specialized to r = 2
// (so (r-1)/r = 1/2 and the (r-1)*n_bar weight is n_bar):
//   n_bar = (n1 + n2)/2,  n_c = n_sum - (n1^2 + n2^2)/n_sum,
//   p_bar = (n1 p1 + n2 p2)/n_sum,  h_bar = (n1 h1 + n2 h2)/n_sum,
//   s2 = (n1 (p1-p_bar)^2 + n2 (p2-p_bar)^2)/n_bar,
//   a = (n_bar/n_c)[ s2 - (1/(n_bar-1))( p_bar(1-p_bar) - s2/2 - h_bar/4 ) ],
//   b = (n_bar/(n_bar-1))[ p_bar(1-p_bar) - s2/2 - ((2 n_bar-1)/(4 n_bar)) h_bar ],
//   c = h_bar/2.
[[nodiscard]] STEPPE_HD inline WcSite wc_finalize(const WcPerPop& A, const WcPerPop& B) noexcept {
    WcSite r;  // defaults: num=0, den=0, fst=0, valid=false

    const double n1 = A.n;
    const double n2 = B.n;
    // Both populations must be sampled at this site.
    if (!(n1 > 0.0) || !(n2 > 0.0)) { r.fst = wc_nan(); return r; }

    const double n_sum = n1 + n2;
    const double n_bar = 0.5 * n_sum;
    // Need n_bar > 1 so the 1/(n_bar - 1) factors are finite (excludes the degenerate
    // n1 == n2 == 1 site up front — no 0*inf).
    if (!(n_bar > 1.0)) { r.fst = wc_nan(); return r; }

    // n_c = 2 n1 n2 / n_sum > 0 whenever n1, n2 > 0.
    const double n_c = n_sum - (n1 * n1 + n2 * n2) / n_sum;
    if (!(n_c > 0.0)) { r.fst = wc_nan(); return r; }

    const double p1 = A.ac / (2.0 * n1);
    const double p2 = B.ac / (2.0 * n2);
    const double h1 = A.het / n1;
    const double h2 = B.het / n2;

    const double p_bar = (n1 * p1 + n2 * p2) / n_sum;
    const double h_bar = (n1 * h1 + n2 * h2) / n_sum;

    const double dp1 = p1 - p_bar;
    const double dp2 = p2 - p_bar;
    const double s2 = (n1 * dp1 * dp1 + n2 * dp2 * dp2) / n_bar;

    const double pq = p_bar * (1.0 - p_bar);
    const double nm1 = n_bar - 1.0;

    const double a = (n_bar / n_c) * (s2 - (1.0 / nm1) * (pq - 0.5 * s2 - 0.25 * h_bar));
    const double b = (n_bar / nm1) * (pq - 0.5 * s2 - ((2.0 * n_bar - 1.0) / (4.0 * n_bar)) * h_bar);
    const double c = 0.5 * h_bar;

    const double den = a + b + c;
    // A zero (monomorphic-across-the-pair -> a=b=c=0) or non-finite denominator is invalid:
    // force num=den=0 so it contributes nothing to a masked summary reduction.
    if (den == 0.0 || !(den == den) || (den - den) != 0.0) {
        r.num = 0.0;
        r.den = 0.0;
        r.fst = wc_nan();
        r.valid = false;
        return r;
    }

    r.num = a;
    r.den = den;
    r.fst = a / den;
    r.valid = true;
    return r;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_WC_FST_HPP

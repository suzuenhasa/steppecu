// src/io/filter/snp_summary_reduce.hpp
//
// THE shared __host__ __device__ per-SNP pooled-MAF reduction primitive — the
// SINGLE source of the across-pop Σ that both the host keep-mask builder
// (snp_filter.cpp derive_per_snp_summary) AND the device regime-B keep-mask kernel
// (decode_compact_kernel.cu) call, so neither path can diverge on a boundary
// (architecture.md §8 DRY; the decode_af.hpp / f2_estimator.hpp single-source
// precedent). This is the FIRST non-integer host/device reduction in the filter
// path, so the bit-exactness pin lives HERE, once.
//
// THE BIT-EXACT PIN (the regime-B GOLDEN-EXACT requirement). The reduction is
//   pooled_ref_count   = Σ_pop Q(pop,s) · N(pop,s)
//   pooled_allele_count= Σ_pop N(pop,s)
//   nonmissing_indiv   = Σ_pop N(pop,s) / ploidy
// summed SEQUENTIALLY p = 0..P-1 (one SNP per thread on the device ⇒ the order is
// trivially identical to the host loop). The ONLY FP-fragile op is the product
// Q·N folded into the running sum: nvcc defaults --fmad=true, so a naive device
// `acc + q*n` fuses to ONE rounding (FFMA) while the host (GCC -std=c++20 ISO mode,
// -ffp-contract=off by default) does TWO. To make the two paths bit-identical:
//   * DEVICE: use __dmul_rn(q,n) then __dadd_rn(acc,prod) — round-to-nearest, NO
//     contraction, regardless of the --fmad flag.
//   * HOST: the SAME two-step expression; -std=c++20 ISO mode does not contract
//     `a*b` then `+c` across statements (and CMAKE_CXX_EXTENSIONS is OFF so this is
//     the build's mode), so each op rounds separately — matching the device.
// is_monomorphic (== 0.0 exact) is FMA-immune by construction: at a monomorphic
// site every per-pop Q is exactly 0.0 or exactly 1.0, so Q·N is an exact
// integer-valued double and the sum carries no rounding error — the same bit on
// both paths whether or not the product fused (filter_decision.hpp:120-122).
//
// LAYERING: an `io`-leaf header — it includes only core/internal/host_device.hpp
// (the header-only STEPPE_HD macro, no CUDA, no link edge — the SAME single-source
// macro decode_af.hpp/f2_estimator.hpp use; co-included here so the macro is NOT
// open-coded a third time) and the standard library. No core/device library dep.
#ifndef STEPPE_IO_FILTER_SNP_SUMMARY_REDUCE_HPP
#define STEPPE_IO_FILTER_SNP_SUMMARY_REDUCE_HPP

#include "core/internal/host_device.hpp"  // STEPPE_HD (header-only macro; no link edge)
#include "io/filter/filter_decision.hpp"  // the shared (now __host__ __device__) predicates
#include "steppe/config.hpp"              // FilterConfig

namespace steppe::io::filter {

/// The four pooled per-SNP scalars derived by the across-pop reduction (the inputs
/// the threshold predicates need). Mirrors PerSnpSummary but is a POD usable in a
/// device kernel (no std::vector). Both the host loop and the device kernel fill it
/// via the same primitive below.
struct PooledSnpSummary {
    double pooled_ref_af = 0.0;        ///< Σ_pop Q·N / Σ_pop N  (0 if no data)
    double pooled_minor_af = 0.0;      ///< min(pooled_ref_af, 1 - pooled_ref_af)
    double missing_frac = 1.0;         ///< 1 - (Σ_pop N/ploidy) / total_indiv
    double pooled_allele_count = 0.0;  ///< Σ_pop N (0 ⇒ SNP all-missing)
};

/// FFMA-IMMUNE fused-multiply-add of one (pop, snp) cell into the running pooled
/// ref-count sum: `acc + q*n` with the product and the add SEPARATELY rounded
/// (round-to-nearest), so the host (-ffp-contract=off) and the device (--fmad=true)
/// produce the IDENTICAL bits. On the device this is __dadd_rn(acc, __dmul_rn(q,n));
/// on the host it is the plain two-step expression (ISO C++ does not contract it).
[[nodiscard]] STEPPE_HD inline double pooled_ref_fma(double acc, double q,
                                                     double n) noexcept {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(acc, __dmul_rn(q, n));
#else
    const double prod = q * n;  // separately rounded (ISO C++, -ffp-contract=off)
    return acc + prod;          // separately rounded
#endif
}

/// Reduce the P-length column of one SNP `s` (column-major [P × M] Q/N, element
/// (pop p, snp s) at p + P·s) ACROSS the kept populations into the four pooled
/// scalars — the SINGLE derivation shared by snp_filter.cpp and the regime-B
/// device kernel. `total_indiv` is the SNP-independent Σ_pop pop_individuals (the
/// missing-fraction denominator); `ploidy_d` is the (double) ploidy. The Σ runs
/// SEQUENTIALLY p = 0..P-1 (one SNP per device thread ⇒ identical order to the host
/// loop), and the product Q·N folds in via pooled_ref_fma (FFMA-immune) so the
/// host and device sums are bit-identical (the regime-B GOLDEN-EXACT pin).
///
/// MATCHES snp_filter.cpp:79-107 EXACTLY: pooled_ref_count += q·n (FFMA-immune),
/// pooled_allele_count += n, nonmissing_indiv += n/ploidy; then pooled_ref_af =
/// (allele>0)?ref/allele:0, pooled_minor_af = folded, missing_frac =
/// (total>0)?1-nonmiss/total:1.
[[nodiscard]] STEPPE_HD inline PooledSnpSummary derive_pooled_summary_one(
    const double* q, const double* n, int P, long s, double ploidy_d,
    double total_indiv_d) noexcept {
    double pooled_ref_count = 0.0;     // Σ_pop Q·N
    double pooled_allele_count = 0.0;  // Σ_pop N
    double nonmissing_indiv = 0.0;     // Σ_pop N/ploidy
    const long base = static_cast<long>(P) * s;
    for (int p = 0; p < P; ++p) {
        const long off = base + static_cast<long>(p);
        const double npn = n[off];
        // Q is zero where invalid and N is 0 there, so a missing cell contributes 0
        // to both Q·N and N — no branch needed (matches the host loop).
        pooled_ref_count = pooled_ref_fma(pooled_ref_count, q[off], npn);
        pooled_allele_count += npn;
        nonmissing_indiv += npn / ploidy_d;
    }
    PooledSnpSummary sm;
    sm.pooled_allele_count = pooled_allele_count;
    sm.pooled_ref_af =
        (pooled_allele_count > 0.0) ? (pooled_ref_count / pooled_allele_count) : 0.0;
    // folded_maf inline (filter_decision.hpp folded_maf, kept here to avoid a
    // CUDA-include dependency through that header on the device path; identical body).
    sm.pooled_minor_af = (sm.pooled_ref_af < 1.0 - sm.pooled_ref_af)
                             ? sm.pooled_ref_af
                             : (1.0 - sm.pooled_ref_af);
    sm.missing_frac =
        (total_indiv_d > 0.0) ? (1.0 - nonmissing_indiv / total_indiv_d) : 1.0;
    return sm;
}

/// The SINGLE per-SNP keep/drop decision over a PooledSnpSummary — the __host__
/// __device__ body that BOTH snp_filter.hpp::snp_keep_decision (host) and the
/// regime-B device keep-mask kernel call, so the host mask-builder and the GPU path
/// cannot diverge. Byte-for-byte the same DROP-NOT-FLIP order as snp_filter.hpp
/// (filter_decision.hpp predicates): (1) unconditional class drop (multiallelic /
/// strand-ambiguous); (2) MAF >= maf_min; (3) geno <= geno_max_missing; (4)
/// flag-gated drop_monomorphic (UNFOLDED pooled_ref_af, exact ==0.0) /
/// transversions_only / autosomes_only; (5) the precomputed membership bit.
[[nodiscard]] STEPPE_HD inline bool keep_decision_pooled(const PooledSnpSummary& sm,
                                                         char ref, char alt, int chrom,
                                                         const FilterConfig& cfg,
                                                         bool membership_ok) noexcept {
    if (is_multiallelic(ref, alt) || is_strand_ambiguous(ref, alt)) return false;
    if (!snp_passes_maf(sm.pooled_minor_af, cfg.maf_min)) return false;
    if (!snp_passes_geno(sm.missing_frac, cfg.geno_max_missing)) return false;
    if (cfg.drop_monomorphic && is_monomorphic(sm.pooled_ref_af)) return false;
    if (cfg.transversions_only && !is_transversion(ref, alt)) return false;
    if (cfg.autosomes_only && !is_autosome(chrom)) return false;
    return membership_ok;
}

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_SNP_SUMMARY_REDUCE_HPP

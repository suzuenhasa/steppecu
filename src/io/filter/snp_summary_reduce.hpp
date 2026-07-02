// src/io/filter/snp_summary_reduce.hpp
//
// The shared host/device per-SNP pooled-MAF reduction and keep/drop decision — the
// single source both the host keep-mask builder (snp_filter.cpp) and the regime-B
// device kernel (decode_compact_kernel.cu) call, so the two paths stay bit-for-bit
// identical.
//
// Reference: docs/reference/src_io_filter_snp_summary_reduce.hpp.md
#ifndef STEPPE_IO_FILTER_SNP_SUMMARY_REDUCE_HPP
#define STEPPE_IO_FILTER_SNP_SUMMARY_REDUCE_HPP

#include "core/internal/host_device.hpp"
#include "io/filter/filter_decision.hpp"
#include "steppe/config.hpp"

namespace steppe::io::filter {

// PooledSnpSummary — reference §3
struct PooledSnpSummary {
    double pooled_ref_af = 0.0;
    double pooled_minor_af = 0.0;
    double missing_frac = 1.0;
    double pooled_allele_count = 0.0;
};

// pooled_ref_fma — reference §4
[[nodiscard]] STEPPE_HD inline double pooled_ref_fma(double acc, double q,
                                                     double n) noexcept {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(acc, __dmul_rn(q, n));
#else
    const double prod = q * n;
    return acc + prod;
#endif
}

// derive_pooled_summary_one — reference §5
[[nodiscard]] STEPPE_HD inline PooledSnpSummary derive_pooled_summary_one(
    const double* q, const double* n, int P, long s, double ploidy_d,
    double total_indiv_d) noexcept {
    double pooled_ref_count = 0.0;
    double pooled_allele_count = 0.0;
    double nonmissing_indiv = 0.0;
    const long base = static_cast<long>(P) * s;
    for (int p = 0; p < P; ++p) {
        const long off = base + static_cast<long>(p);
        const double npn = n[off];
        pooled_ref_count = pooled_ref_fma(pooled_ref_count, q[off], npn);
        pooled_allele_count += npn;
        nonmissing_indiv += npn / ploidy_d;
    }
    PooledSnpSummary sm;
    sm.pooled_allele_count = pooled_allele_count;
    sm.pooled_ref_af =
        (pooled_allele_count > 0.0) ? (pooled_ref_count / pooled_allele_count) : 0.0;
    sm.pooled_minor_af = (sm.pooled_ref_af < 1.0 - sm.pooled_ref_af)
                             ? sm.pooled_ref_af
                             : (1.0 - sm.pooled_ref_af);
    sm.missing_frac =
        (total_indiv_d > 0.0) ? (1.0 - nonmissing_indiv / total_indiv_d) : 1.0;
    return sm;
}

// keep_decision_pooled — reference §6
[[nodiscard]] STEPPE_HD inline bool keep_decision_pooled(const PooledSnpSummary& sm,
                                                         char ref, char alt, int chrom,
                                                         const FilterConfig& cfg,
                                                         bool membership_ok) noexcept {
    if (is_multiallelic(ref, alt)) return false;
    if (cfg.strand_mode == StrandMode::Drop && is_strand_ambiguous(ref, alt)) return false;
    if (!snp_passes_maf(sm.pooled_minor_af, cfg.maf_min)) return false;
    if (!snp_passes_geno(sm.missing_frac, cfg.geno_max_missing)) return false;
    if (cfg.drop_monomorphic && is_monomorphic(sm.pooled_ref_af)) return false;
    if (cfg.transversions_only && !is_transversion(ref, alt)) return false;
    if (cfg.autosomes_only && !is_autosome(chrom)) return false;
    return membership_ok;
}

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_SNP_SUMMARY_REDUCE_HPP

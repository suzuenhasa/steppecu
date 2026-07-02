// src/io/filter/snp_filter.hpp
//
// Builds one per-SNP keep/drop mask for a decoded tile: derives the pooled folded
// MAF and per-SNP missing fraction across the kept populations, then applies the
// shared QC-filter predicates. The output is one bool per SNP — SNP-global, never
// per-(pop, SNP) — and failing SNPs are dropped, never strand-flipped. An io-leaf
// header: host C++20, no CUDA, no core/device dependency.
//
// Reference: docs/reference/src_io_filter_snp_filter.hpp.md
#ifndef STEPPE_IO_FILTER_SNP_FILTER_HPP
#define STEPPE_IO_FILTER_SNP_FILTER_HPP

#include <cstddef>
#include <vector>

#include "io/filter/filter_decision.hpp"
#include "io/filter/snp_summary_reduce.hpp"
#include "io/filter/include_exclude.hpp"
#include "io/snp_reader.hpp"
#include "steppe/config.hpp"

namespace steppe::io::filter {

// Decoded per-pop Q/V/N + sizing for one tile — reference §2
struct DecodedTileSummaryInput {
    const double* q = nullptr;
    const double* v = nullptr;
    const double* n = nullptr;
    int P = 0;
    long M = 0;

    std::vector<std::size_t> pop_individuals;

    int ploidy = 2;
};

// Per-SNP derived summary (reductions across kept populations) — reference §3
struct PerSnpSummary {
    double pooled_ref_af = 0.0;
    double pooled_minor_af = 0.0;
    double missing_frac = 1.0;
    double pooled_allele_count = 0.0;
};

// derive_per_snp_summary — reference §4
[[nodiscard]] std::vector<PerSnpSummary> derive_per_snp_summary(
    const DecodedTileSummaryInput& in);

// snp_keep_decision — reference §5
[[nodiscard]] inline bool snp_keep_decision(const PerSnpSummary& sm,
                                            char ref,
                                            char alt,
                                            int chrom,
                                            const FilterConfig& cfg,
                                            bool membership_ok) noexcept {
    PooledSnpSummary ps;
    ps.pooled_ref_af = sm.pooled_ref_af;
    ps.pooled_minor_af = sm.pooled_minor_af;
    ps.missing_frac = sm.missing_frac;
    ps.pooled_allele_count = sm.pooled_allele_count;
    return keep_decision_pooled(ps, ref, alt, chrom, cfg, membership_ok);
}

// build_snp_keep_mask — reference §6
[[nodiscard]] std::vector<bool> build_snp_keep_mask(
    const DecodedTileSummaryInput& in,
    const SnpTable& snps,
    const FilterConfig& cfg,
    const SnpMembership& mem);

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_SNP_FILTER_HPP

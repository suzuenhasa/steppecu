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

// build_snp_keep_mask_from_summary — the same keep decision as build_snp_keep_mask, but over
// an ALREADY-derived per-SNP pooled summary (length M) instead of the O(P*M) per-pop Q/V/N.
// It runs the identical snps-table validation + snp_keep_decision loop; only the derive step
// (derive_per_snp_summary) is skipped because the caller streamed the decode SNP-tile by
// SNP-tile and pooled each tile on the host as it went. Bit-identical to build_snp_keep_mask
// fed the whole decode, so the two paths cannot drift.
[[nodiscard]] std::vector<bool> build_snp_keep_mask_from_summary(
    const std::vector<PerSnpSummary>& summary,
    const SnpTable& snps,
    const FilterConfig& cfg,
    const SnpMembership& mem);

// filter_is_active — true when the config requests ANY SNP subsetting: a MAF floor, a per-SNP
// missing cap, autosomes-only, drop-monomorphic, transversions-only, or an include/exclude/
// prune membership. Strand-mode is a sub-policy of the filter (it only takes effect once the
// filter is otherwise engaged), NOT an independent trigger — so a default-constructed
// FilterConfig (strand-mode Drop, everything else off) is INACTIVE and the front-end leaves
// the tile byte-identical. Used by the shared apply_snp_filter seam to skip all work (and
// stay bit-exact) when nothing is asked.
[[nodiscard]] bool filter_is_active(const FilterConfig& cfg) noexcept;

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_SNP_FILTER_HPP

// src/io/filter/snp_filter.hpp
//
// SNP-level keep-mask producer (architecture.md §1 filter SCOPE, §5 S0' "cheap
// filters decidable from one tile"; ROADMAP M2). Given a decoded tile's per-pop
// Q/V/N (M1's contract) + the per-pop sample sizes + the .snp allele/chrom
// metadata, derive per-SNP pooled folded MAF + per-SNP missing fraction + the
// allele-pair class, then apply the shared filter_decision.hpp predicates +
// include/exclude membership + the flag-gated monomorphic/transversion/autosome
// filters into ONE per-SNP keep mask. DROP-NOT-FLIP: failing SNPs are dropped,
// never altered.
//
// LAYERING: an `io`-leaf header (architecture.md §4) — host C++20, no CUDA, no
// core/device dependency. It consumes the SAME plain column-major [P × M] Q/V/N
// the decode backend produces (the test fills these), but never reaches into
// core/device — it takes plain double pointers, not core::MatView (which lives in
// core/internal). The decision logic is the single source in filter_decision.hpp.
//
// THE FILTER INVARIANT (architecture.md §1, §5 S2): the output is one bool PER
// SNP — SNP-GLOBAL, never per-(pop, SNP). The pooled-MAF and missing-fraction
// derivations reduce ACROSS the kept populations into a single per-SNP scalar, so
// a drop zeroes a whole SNP column for the V·Vᵀ-masked GEMM, never a single cell.
#ifndef STEPPE_IO_FILTER_SNP_FILTER_HPP
#define STEPPE_IO_FILTER_SNP_FILTER_HPP

#include <cstddef>
#include <vector>

#include "io/filter/include_exclude.hpp"  // SnpMembership
#include "io/snp_reader.hpp"              // SnpTable (id / chrom / ref / alt)
#include "steppe/config.hpp"              // FilterConfig

namespace steppe::io::filter {

/// The decoded per-pop Q/V/N for one tile, as plain column-major [P × M] double
/// arrays (element (pop i, snp s) at i + P·s — the M1 Q/V/N contract), plus the
/// per-population KEPT-individual count (segment size after the --mind pre-pass).
/// This is exactly what the decode backend returns (DecodeResult.q/.v/.n) plus the
/// pop sizes the partition / kept-sample plan supplies; snp_filter consumes plain
/// pointers so the `io` leaf does not depend on core::MatView.
struct DecodedTileSummaryInput {
    const double* q = nullptr;   ///< [P × M] ref-allele freq, column-major (i + P·s)
    const double* v = nullptr;   ///< [P × M] validity mask (1 valid / 0 missing)
    const double* n = nullptr;   ///< [P × M] non-missing haploid count (ploidy × indiv)
    int P = 0;                   ///< number of (kept) populations
    long M = 0;                  ///< number of SNPs in the tile

    /// Per-population KEPT-individual count, length P. Used as the denominator of
    /// the per-SNP missing fraction (Σ kept individuals across pops). For the
    /// diploid AADR case this is N_pop/2 at a fully-non-missing SNP, but it is the
    /// total individuals (missing + non-missing) PER POP, supplied by the caller
    /// from the partition (pop_offsets diffs) intersected with the kept-sample set.
    std::vector<std::size_t> pop_individuals;

    /// Per-sample ploidy (2 diploid / 1 pseudo-haploid), to convert N (haploid
    /// count) back to the non-missing INDIVIDUAL count for the missing fraction:
    /// nonmissing_individuals_pop = N_pop / ploidy. Default 2 (the AADR case).
    int ploidy = 2;
};

/// Per-SNP derived summary (the inputs the threshold predicates need), exposed so
/// the oracle test can recompute and compare it exactly. All quantities are
/// reductions ACROSS the kept populations — SNP-global scalars.
struct PerSnpSummary {
    double pooled_ref_af = 0.0;       ///< Σ_pop Q·N / Σ_pop N  (pooled ref-allele freq; 0 if no data)
    double pooled_minor_af = 0.0;     ///< min(pooled_ref_af, 1 - pooled_ref_af) (folded MAF)
    double missing_frac = 1.0;        ///< 1 - (Σ_pop nonmissing indiv) / (Σ_pop indiv) (sample axis)
    double pooled_allele_count = 0.0; ///< Σ_pop N_pop (pooled allele count; 0 ⇒ SNP all-missing)
};

/// Derive the per-SNP pooled folded MAF + per-SNP missing fraction from the
/// decoded per-pop Q/V/N + pop sizes (the single derivation, shared by snp_filter
/// and pinnable by the oracle). For SNP `s`:
///   pooled_ref_count   = Σ_pop Q(pop,s) · N(pop,s)      (= Σ ssum_pop, the oracle's)
///   pooled_allele_count= Σ_pop N(pop,s)                 (= Σ 2·scnt_pop)
///   pooled_ref_af      = pooled_ref_count / pooled_allele_count   (0 if denom 0)
///   nonmissing_indiv   = Σ_pop N(pop,s) / ploidy
///   total_indiv        = Σ_pop pop_individuals[pop]
///   missing_frac       = 1 - nonmissing_indiv / total_indiv       (1 if total 0)
/// Length-M result, parallel to the SNP axis.
[[nodiscard]] std::vector<PerSnpSummary> derive_per_snp_summary(
    const DecodedTileSummaryInput& in);

/// Build the per-SNP keep mask for the tile: a SNP is KEPT iff it passes EVERY
/// active filter — MAF (>= maf_min), geno (<= geno_max_missing), include/exclude
/// membership, and (flag-gated) drop_monomorphic / transversions_only /
/// autosomes_only — AND it is not strand-ambiguous or multiallelic (DROPPED
/// unconditionally, drop-not-flip; architecture.md §1). All thresholds come from
/// `cfg`; the allele-pair class and chromosome come from `snps` (parallel to the M
/// SNP axis); membership is `mem` (resolved include/exclude/prune.in).
///
/// With a default FilterConfig (maf_min=0, geno_max_missing=1, all flags false,
/// empty membership), this keeps EVERY SNP whose allele pair is a clean biallelic
/// ACGT non-palindrome — which for the real AADR HO panel is every SNP (it has no
/// A/T·C/G palindromes and no non-ACGT alleles). The unconditional ambiguous/
/// multiallelic drop is the ONLY non-no-op at default, and it is the correct
/// drop-not-flip behavior; the oracle test confirms it drops zero on the HO panel,
/// so the no-op-when-default parity property holds there.
///
/// `snps.count` must be >= `in.M` (the first M ids/alleles are used). Returns a
/// length-M std::vector<bool>.
[[nodiscard]] std::vector<bool> build_snp_keep_mask(
    const DecodedTileSummaryInput& in,
    const SnpTable& snps,
    const FilterConfig& cfg,
    const SnpMembership& mem);

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_SNP_FILTER_HPP

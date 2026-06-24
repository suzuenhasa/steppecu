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

#include "io/filter/filter_decision.hpp"     // the shared per-SNP predicates (single source)
#include "io/filter/snp_summary_reduce.hpp"  // PooledSnpSummary + keep_decision_pooled (the shared HD body)
#include "io/filter/include_exclude.hpp"     // SnpMembership
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
    /// Must be 1 or 2 — `derive_per_snp_summary`/`build_snp_keep_mask` throw
    /// std::invalid_argument on any other value (fail-fast on misset metadata;
    /// cleanup B10/X-11), reconciling with decode_af.hpp::finalize_af which masks
    /// out a cell decoded with a non-positive ploidy.
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
///
/// PRECONDITIONS (fail-fast with std::invalid_argument — the `io`-leaf idiom, cf.
/// include_exclude.cpp; architecture.md §2; cleanup B20). When P>0 && M>0:
///   * `in.ploidy` must be 1 or 2 (B10/X-11; a misset ploidy fabricates a wrong
///     missing fraction otherwise);
///   * `in.pop_individuals.size()` must equal P (B20/F1): the prior code guarded
///     only the DENOMINATOR loop and ran the NUMERATOR loop over all P pops, so a
///     short `pop_individuals` understated the missing-fraction denominator and
///     could yield a NEGATIVE missing_frac that SPURIOUSLY PASSES the geno filter
///     (a missing-heavy SNP wrongly KEPT) — a silent wrong answer, not an error;
///   * `in.q` and `in.n` must be non-null (B20/F6): the reduction dereferences both
///     per cell, so a null pointer otherwise segfaults three frames deep instead of
///     surfacing a diagnosable error.
/// (`in.v` is intentionally unused: N==0 ⇔ V==0 ⇔ missing by the decode contract —
/// decode_af.hpp finalize_af — so V is redundant for these reductions.)
[[nodiscard]] std::vector<PerSnpSummary> derive_per_snp_summary(
    const DecodedTileSummaryInput& in);

/// The SINGLE per-SNP keep/drop decision, shared by build_snp_keep_mask and (the
/// M4.5 device fusion target) the future decode_af kernel so the host mask-builder
/// and the GPU path cannot diverge on a boundary — the decode_af.hpp pattern (one
/// __host__ __device__ per-element primitive both paths call). Pure: it takes the
/// per-SNP derived summary, the allele pair + chromosome, the thresholds (FROM
/// cfg), and a PRECOMPUTED membership bit (true ⇒ the id passes include/exclude/
/// prune.in, or membership is a no-op), and returns the keep decision for ONE SNP.
///
/// KEEP iff the SNP passes EVERY active filter, in this DROP-NOT-FLIP order
/// (architecture.md §1): (1) unconditional class drop — strand-ambiguous (A/T, C/G
/// palindrome) or multiallelic/non-clean-ACGT is dropped regardless of any flag and
/// never strand-flipped; (2) MAF (>= cfg.maf_min); (3) geno (<= cfg.geno_max_missing);
/// (4) flag-gated drop_monomorphic / transversions_only / autosomes_only; (5) the
/// membership bit. Every decision delegates to a filter_decision.hpp predicate (the
/// §8 single source). `inline` so it is shareable with a future device TU without an
/// out-of-line definition; `noexcept` (all callees are noexcept, no allocation).
[[nodiscard]] inline bool snp_keep_decision(const PerSnpSummary& sm,
                                            char ref,
                                            char alt,
                                            int chrom,
                                            const FilterConfig& cfg,
                                            bool membership_ok) noexcept {
    // Delegate to the SHARED __host__ __device__ body (snp_summary_reduce.hpp) so this
    // host call site and the regime-B device keep-mask kernel run the IDENTICAL
    // DROP-NOT-FLIP decision over the SAME filter_decision.hpp predicates — neither
    // path can diverge on a boundary (§8 single source). PerSnpSummary carries the
    // same three scalars the decision reads (pooled_minor_af / missing_frac /
    // pooled_ref_af); copy them into the POD the shared body takes.
    PooledSnpSummary ps;
    ps.pooled_ref_af = sm.pooled_ref_af;
    ps.pooled_minor_af = sm.pooled_minor_af;
    ps.missing_frac = sm.missing_frac;
    ps.pooled_allele_count = sm.pooled_allele_count;
    return keep_decision_pooled(ps, ref, alt, chrom, cfg, membership_ok);
}

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
/// PRECONDITIONS (fail-fast with std::invalid_argument; architecture.md §2;
/// cleanup B20/F3). When M>0: `snps.ref.size() >= M` and `snps.alt.size() >= M`
/// (the allele pair drives the unconditional class drop for EVERY SNP), and — when
/// the corresponding filter / membership is active — `snps.chrom.size() >= M`
/// (autosomes_only) and `snps.id.size() >= M` (non-no-op membership). The prior
/// code silently substituted 'N'/0/empty-id for a too-short SnpTable, which made
/// is_multiallelic('N','N') true and DROPPED the tail of the tile (or altered
/// membership) — a silent wrong answer for a mismatched .snp/decode length, which
/// is a wiring/data bug that must abort with context. Also throws (via
/// derive_per_snp_summary) on the B10/B20 preconditions there (ploidy ∉ {1,2},
/// pop_individuals.size() != P, null q/n). Returns a length-M std::vector<bool>.
[[nodiscard]] std::vector<bool> build_snp_keep_mask(
    const DecodedTileSummaryInput& in,
    const SnpTable& snps,
    const FilterConfig& cfg,
    const SnpMembership& mem);

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_SNP_FILTER_HPP

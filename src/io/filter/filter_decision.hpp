// src/io/filter/filter_decision.hpp
//
// THE shared host-pure FILTER PREDICATES — the single source of truth for every
// M2 keep/drop decision (architecture.md §1 filter SCOPE, §5 S0' "cheap filters
// decidable from one tile", §8 DRY; ROADMAP M2). This is the M2 analogue of
// core/internal/f2_estimator.hpp: pure functions, every threshold sourced FROM
// FilterConfig, no magic numbers. snp_filter / mind_prepass / include_exclude all
// call THESE — the rule lives ONCE so the in-tile path, the pre-pass, and the
// tests cannot diverge on a boundary or an allele-pair class.
//
// LAYERING: an `io`-leaf header (architecture.md §4) — pure host C++20, depends on
// nothing in core/device, no CUDA. It includes only <steppe/config.hpp> (the
// CUDA-free public config) and the standard library. The `io` leaf may consume the
// SAME plain decoded Q/V/N the test produces, but it never reaches into core/device.
//
// ===========================================================================
// FILTER SCOPE (architecture.md §1) — the non-negotiable invariants
// ===========================================================================
//   * DROP-BY-DEFAULT, FLAG-GATED. Multiallelic SNPs are ALWAYS dropped.
//     Strand-ambiguous (A/T, C/G) SNPs are dropped BY DEFAULT (StrandMode::Drop,
//     the merge-safety default) but retained under --strand-mode keep/flip (the
//     AT2-reproduction path). No Q value is ever altered by a filter; even under
//     Keep we do NOT infer strand (Flip's freq-based reorientation is a documented
//     not-yet-implemented token). The gate lives once in keep_decision_pooled
//     (snp_summary_reduce.hpp); the predicates here are pure classifiers.
//   * NO LD computation. An external prune.in is READ, never computed.
//   * NO on-disk rewrite. The filters produce a PLAN (a keep-mask / kept-sample
//     set), consumed per tile during the stream.
//   * SNP-GLOBAL or SAMPLE-GLOBAL, NEVER per-(pop, SNP). A SNP is kept or dropped
//     for ALL populations at once; a sample is kept or dropped for all SNPs. A
//     per-(pop, SNP) drop would break the symmetric V·Vᵀ masking that the 3-GEMM
//     f2 reformulation relies on (architecture.md §5 S2) — the masked GEMM
//     assumes V is a clean 0/1 mask, so any drop must zero a WHOLE column (SNP)
//     or a whole sample, never a single (pop, SNP) cell. This file's predicates
//     return one bool PER SNP (or per sample), enforcing that invariant by shape.
//
// ===========================================================================
// THE MAF CONVENTION (pinned here, the single documented source)
// ===========================================================================
// MAF = the POOLED FOLDED minor allele frequency:
//     pooled_ref_af = (Σ_pop ref_count_pop) / (Σ_pop allele_count_pop)
//     MAF           = min(pooled_ref_af, 1 - pooled_ref_af)
// pooled over ALL kept samples (and only the kept populations), matching
// build_tgeno_matrix.py's ssum/scnt pooling: there `Q_pop = ssum_pop /
// (2·scnt_pop)` and `N_pop = 2·scnt_pop`, so `ref_count_pop = ssum_pop =
// Q_pop·N_pop` and `allele_count_pop = N_pop`. Hence in the Q/V/N contract:
//     pooled_ref_count  = Σ_pop Q_pop · N_pop
//     pooled_allele_cnt = Σ_pop N_pop
// This is a POOLED, ACROSS-SAMPLE frequency — NOT a per-population frequency and
// NOT a mean of per-pop frequencies. (ADMIXTOOLS 2's minmaf/maxmaf doc does not
// state pooled vs per-pop; the pooled reading matches our oracle's ssum/scnt and
// is the documented best reading — FLAGGED in the M2 report.) snp_filter derives
// pooled_ref_af from the per-pop Q/N + the kept-pop set and calls snp_passes_maf.
#ifndef STEPPE_IO_FILTER_FILTER_DECISION_HPP
#define STEPPE_IO_FILTER_FILTER_DECISION_HPP

#include "core/internal/host_device.hpp"  // STEPPE_HD (header-only macro; no link edge)
#include "steppe/config.hpp"              // FilterConfig thresholds, kAutosomeChromMin/Max

namespace steppe::io::filter {

// ===========================================================================
// Per-SNP / per-sample threshold predicates — thresholds FROM FilterConfig.
// Each returns the KEEP decision (true ⇒ keep). The boundary side is PINNED:
//   * MAF  : keep iff pooled folded MAF >= maf_min      (>=, inclusive)
//   * geno : keep iff per-SNP missing frac <= geno_max_missing  (<=, inclusive)
//   * mind : keep iff per-sample missing frac <= mind_max_missing (<=, inclusive)
// The >= for MAF and <= for the missing caps mean a default FilterConfig
// (maf_min=0, geno_max_missing=1, mind_max_missing=1) keeps EVERYTHING:
//   MAF>=0 is always true; missing<=1 is always true. This is the no-op-when-
//   default property the parity path depends on.
// ===========================================================================

/// Folded minor allele frequency of a POOLED reference-allele frequency:
/// min(ref_af, 1 - ref_af). Pure helper used by snp_passes_maf and the
/// monomorphic check so the folding lives once.
[[nodiscard]] STEPPE_HD inline constexpr double folded_maf(double pooled_ref_af) noexcept {
    const double q = pooled_ref_af;
    return (q < 1.0 - q) ? q : (1.0 - q);
}

/// MAF filter: keep a SNP iff its POOLED folded MAF >= maf_min. `pooled_minor_af`
/// is already folded (min(q,1-q)); pass folded_maf(pooled_ref_af) for it.
/// Inclusive `>=`: maf_min == 0 keeps every SNP (the no-op default), and a SNP
/// whose folded MAF exactly equals the threshold is KEPT.
[[nodiscard]] STEPPE_HD inline constexpr bool snp_passes_maf(double pooled_minor_af, double maf_min) noexcept {
    return pooled_minor_af >= maf_min;
}

/// geno filter: keep a SNP iff its per-SNP missing fraction <= geno_max_missing.
/// `per_snp_missing_frac` is the SAMPLE-axis fraction (PLINK --geno convention,
/// see file header). Inclusive `<=`: geno_max_missing == 1 keeps every SNP (the
/// no-op default), and a SNP whose missing fraction exactly equals the threshold
/// is KEPT.
[[nodiscard]] STEPPE_HD inline constexpr bool snp_passes_geno(double per_snp_missing_frac,
                                                             double geno_max_missing) noexcept {
    return per_snp_missing_frac <= geno_max_missing;
}

/// mind filter: keep a SAMPLE iff its per-sample missing fraction <=
/// mind_max_missing. Inclusive `<=`: mind_max_missing == 1 keeps every sample
/// (the no-op default). Decided in the conditional S-1 pre-pass (mind_prepass).
[[nodiscard]] STEPPE_HD inline constexpr bool sample_passes_mind(double per_sample_missing_frac,
                                                                double mind_max_missing) noexcept {
    return per_sample_missing_frac <= mind_max_missing;
}

/// Monomorphic test: a SNP is monomorphic iff its pooled folded MAF is exactly 0
/// (no variation across the kept samples), i.e. its pooled REF-allele frequency is
/// exactly 0.0 or exactly 1.0. Used by the drop_monomorphic flag. Exact-zero by
/// design: a SNP with a single minor allele copy is NOT monomorphic. (This is the
/// strict-positive boundary of the MAF filter, kept as a separate named predicate
/// per the FilterConfig flag.)
///
/// CONTRACT (cleanup B20 / filter_decision 1.1, snp_filter F21): the parameter is
/// the POOLED REF-AF (unfolded, in [0,1]) — this predicate FOLDS it once. This
/// makes the parameter name honest about what the body does and ends the prior
/// double-fold trap, where the call site passed an already-folded minor-af into a
/// re-folding body (idempotent only because folding is the identity on [0,0.5] —
/// exactly the "two places assume different conventions" §8 single-source means to
/// prevent). Pass the UNFOLDED `pooled_ref_af` (e.g. PerSnpSummary::pooled_ref_af),
/// NOT the already-folded `pooled_minor_af`. The `== 0.0` is exact on the real
/// path: a truly monomorphic site has each per-pop Q exactly 0.0 or 1.0, so Q·N is
/// exact and the pooled ref-af is exactly 0.0 or 1.0 — do NOT change the upstream
/// pooling to a mean-of-frequencies, which rounds and would break the exactness.
[[nodiscard]] STEPPE_HD inline constexpr bool is_monomorphic(double pooled_ref_af) noexcept {
    return folded_maf(pooled_ref_af) == 0.0;
}

// ===========================================================================
// Allele-pair CLASS predicates (architecture.md §1 drop-not-flip). These read
// the .snp ref/alt allele chars (snp_reader SnpTable.ref/.alt). They classify a
// SNP by its allele pair; they NEVER alter an allele or a frequency.
// ===========================================================================

/// Normalize an allele char to uppercase A/C/G/T, or '\0' for anything else
/// (incl. 'N', '0', '-', '.', X/indel codes). Internal helper so the class
/// predicates treat lower/upper case identically and reject non-ACGT cleanly.
[[nodiscard]] STEPPE_HD inline char normalize_allele(char a) noexcept {
    // ASCII uppercase, device-safe (no <cctype>/std::toupper on the GPU): a lowercase
    // a-z differs from its uppercase form ONLY in bit 5 (0x20), so clear it. For the
    // A/C/G/T allele chars this is bit-identical to the host std::toupper path (the C
    // locale toupper is the same ASCII map), preserving the keep-set bit-exactness.
    const unsigned char uc = static_cast<unsigned char>(a);
    const char u = (uc >= 'a' && uc <= 'z') ? static_cast<char>(uc - 0x20)
                                            : static_cast<char>(uc);
    return (u == 'A' || u == 'C' || u == 'G' || u == 'T') ? u : '\0';
}

/// The Watson-Crick complement of an A/C/G/T base (A↔T, C↔G), or '\0' if `a` is
/// not a normalized base. Used by the strand-ambiguity test (the only place strand
/// complementarity appears — we DROP ambiguous pairs, never flip by it).
[[nodiscard]] STEPPE_HD inline char complement(char a) noexcept {
    switch (normalize_allele(a)) {
        case 'A': return 'T';
        case 'T': return 'A';
        case 'C': return 'G';
        case 'G': return 'C';
        default:  return '\0';
    }
}

/// Strand-ambiguous: the ref/alt pair is its OWN COMPLEMENT, so the two strands
/// are indistinguishable without strand info — i.e. the self-complementary pairs
/// {A,T} or {C,G}. The four ambiguous ordered pairs are AT, TA, CG, GC. A SNP
/// with an ambiguous pair is DROPPED, never strand-flipped (architecture.md §1
/// drop-not-flip). Non-ACGT input ⇒ not ambiguous (is_multiallelic drops it).
/// Equivalently: a == complement(b).
///
/// CONVENTION FLAG (the genuinely-ambiguous-against-the-brief point): this uses
/// the STANDARD genetics definition of strand-ambiguous/palindromic = the
/// self-complementary pairs A/T and C/G. The M2 brief's example list "GT/AC
/// dropped, GA/CT kept" describes the TRANSITION/TRANSVERSION split, not the
/// self-complementary palindrome class — GT and AC are TRANSVERSIONS, GA and CT
/// are TRANSITIONS, and NONE of them is self-complementary. We implement the
/// genetics-correct palindrome predicate here (the documented best reading) and
/// expose the transition/transversion split via is_transition / is_transversion.
/// On the real AADR HO panel there are NO A/T or C/G pairs at all (they were
/// pre-filtered upstream), so this predicate correctly drops ZERO SNPs there; the
/// "tens of thousands" the brief mentions are the TRANSVERSIONS (GT/TG/CA/AC ≈
/// 18.5k of the first 100k), handled by the transversions_only flag. FLAGGED in
/// the M2 report.
[[nodiscard]] STEPPE_HD inline bool is_strand_ambiguous(char a, char b) noexcept {
    const char na = normalize_allele(a);
    const char nb = normalize_allele(b);
    if (na == '\0' || nb == '\0') return false;  // not a clean biallelic ACGT pair
    return na == complement(nb);                 // {A,T} or {C,G}
}

/// Multiallelic / non-biallelic-ACGT: a SNP whose ref/alt are not a clean pair of
/// two DISTINCT A/C/G/T bases — either allele is non-ACGT (N/0/-/indel code) or
/// ref == alt. Such a site is DROPPED (architecture.md §1: multiallelic sites are
/// dropped, never resolved). (This reader sees only the declared ref/alt pair, so
/// "multiallelic" here means "not a declarable clean SNP"; true >2-allele records
/// arrive as a non-ACGT code or are pre-split upstream.)
[[nodiscard]] STEPPE_HD inline bool is_multiallelic(char a, char b) noexcept {
    const char na = normalize_allele(a);
    const char nb = normalize_allele(b);
    return na == '\0' || nb == '\0' || na == nb;
}

/// Transition: a purine↔purine (A↔G) or pyrimidine↔pyrimidine (C↔T) substitution.
/// The complementary classification to a transversion. Non-ACGT or equal alleles
/// are neither (is_multiallelic catches them).
[[nodiscard]] STEPPE_HD inline bool is_transition(char a, char b) noexcept {
    // The "clean biallelic ACGT pair" rule lives once, in is_multiallelic: a pair that
    // fails it is multiallelic and is neither a transition nor a transversion (cleanup
    // 7.1, §8 single-source). is_multiallelic normalizes a/b, so na/nb are re-derived
    // here only for the ring-class test.
    if (is_multiallelic(a, b)) return false;
    const char na = normalize_allele(a);
    const char nb = normalize_allele(b);
    const bool a_purine = (na == 'A' || na == 'G');
    const bool b_purine = (nb == 'A' || nb == 'G');
    return a_purine == b_purine;  // same ring class ⇒ transition
}

/// Transversion: a purine↔pyrimidine substitution (A/G ↔ C/T). The complement of
/// is_transition over clean biallelic ACGT pairs. Used by the transversions_only
/// flag (keep iff transversion). Non-ACGT or equal alleles are NOT transversions.
[[nodiscard]] STEPPE_HD inline bool is_transversion(char a, char b) noexcept {
    // Normalize once per logical check: guard the clean-pair rule via is_multiallelic,
    // then delegate the ring-class test to is_transition (its complement over clean
    // biallelic ACGT pairs). The prior body normalized a/b only to re-run the guard
    // that is_transition already applies — a redundant double-normalize (cleanup 7.2).
    if (is_multiallelic(a, b)) return false;
    return !is_transition(a, b);
}

// ===========================================================================
// Chromosome predicate.
// ===========================================================================

/// Autosome: chromosome code in [kAutosomeChromMin, kAutosomeChromMax] == 1..22.
/// ADMIXTOOLS 2's extract_f2 `auto_only = TRUE` (default) keeps chr 1-22 only, so
/// this is the AT2-parity autosome set (X→23, Y→24, MT and other codes are NOT
/// autosomes). Used by the autosomes_only flag. The range constants live in
/// config.hpp (no bare 22 here).
[[nodiscard]] STEPPE_HD inline constexpr bool is_autosome(int chrom) noexcept {
    return chrom >= kAutosomeChromMin && chrom <= kAutosomeChromMax;
}

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_FILTER_DECISION_HPP

// src/io/filter/filter_decision.hpp
//
// Pure host-C++20 filter predicates: the single source of truth for every M2
// keep/drop decision (each returns true => keep), shared by the in-tile path,
// the pre-pass, and the tests. An io-leaf header — no CUDA; includes only the
// CUDA-free public config and the standard library.
//
// Reference: docs/reference/src_io_filter_filter_decision.hpp.md
#ifndef STEPPE_IO_FILTER_FILTER_DECISION_HPP
#define STEPPE_IO_FILTER_FILTER_DECISION_HPP

#include "core/internal/host_device.hpp"
#include "steppe/config.hpp"

namespace steppe::io::filter {

// Pooled folded MAF helper — reference §3
[[nodiscard]] STEPPE_HD inline constexpr double folded_maf(double pooled_ref_af) noexcept {
    const double q = pooled_ref_af;
    return (q < 1.0 - q) ? q : (1.0 - q);
}

// Threshold predicates and their pinned boundaries — reference §4
[[nodiscard]] STEPPE_HD inline constexpr bool snp_passes_maf(double pooled_minor_af, double maf_min) noexcept {
    return pooled_minor_af >= maf_min;
}

[[nodiscard]] STEPPE_HD inline constexpr bool snp_passes_geno(double per_snp_missing_frac,
                                                             double geno_max_missing) noexcept {
    return per_snp_missing_frac <= geno_max_missing;
}

[[nodiscard]] STEPPE_HD inline constexpr bool sample_passes_mind(double per_sample_missing_frac,
                                                                double mind_max_missing) noexcept {
    return per_sample_missing_frac <= mind_max_missing;
}

// The monomorphic test — reference §5
[[nodiscard]] STEPPE_HD inline constexpr bool is_monomorphic(double pooled_ref_af) noexcept {
    return folded_maf(pooled_ref_af) == 0.0;
}

// Allele-pair class predicates — reference §6
[[nodiscard]] STEPPE_HD inline char normalize_allele(char a) noexcept {
    const unsigned char uc = static_cast<unsigned char>(a);
    const char u = (uc >= 'a' && uc <= 'z') ? static_cast<char>(uc - 0x20)
                                            : static_cast<char>(uc);
    return (u == 'A' || u == 'C' || u == 'G' || u == 'T') ? u : '\0';
}

[[nodiscard]] STEPPE_HD inline char complement(char a) noexcept {
    switch (normalize_allele(a)) {
        case 'A': return 'T';
        case 'T': return 'A';
        case 'C': return 'G';
        case 'G': return 'C';
        default:  return '\0';
    }
}

[[nodiscard]] STEPPE_HD inline bool is_strand_ambiguous(char a, char b) noexcept {
    const char na = normalize_allele(a);
    const char nb = normalize_allele(b);
    if (na == '\0' || nb == '\0') return false;
    return na == complement(nb);
}

[[nodiscard]] STEPPE_HD inline bool is_multiallelic(char a, char b) noexcept {
    const char na = normalize_allele(a);
    const char nb = normalize_allele(b);
    return na == '\0' || nb == '\0' || na == nb;
}

[[nodiscard]] STEPPE_HD inline bool is_transition(char a, char b) noexcept {
    if (is_multiallelic(a, b)) return false;
    const char na = normalize_allele(a);
    const char nb = normalize_allele(b);
    const bool a_purine = (na == 'A' || na == 'G');
    const bool b_purine = (nb == 'A' || nb == 'G');
    return a_purine == b_purine;
}

[[nodiscard]] STEPPE_HD inline bool is_transversion(char a, char b) noexcept {
    if (is_multiallelic(a, b)) return false;
    return !is_transition(a, b);
}

// The autosome (chromosome) predicate — reference §7
[[nodiscard]] STEPPE_HD inline constexpr bool is_autosome(int chrom) noexcept {
    return chrom >= kAutosomeChromMin && chrom <= kAutosomeChromMax;
}

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_FILTER_DECISION_HPP

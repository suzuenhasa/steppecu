// src/io/filter/mind_prepass.hpp
//
// The conditional S-1 --mind streaming pre-pass (architecture.md §5 S-1 "QC
// pre-pass (conditional)", §1; ROADMAP M2). ONLY runs when mind_max_missing < 1.0;
// otherwise it is skipped entirely (every sample kept). It accumulates per-sample
// NON-MISSING counts across ALL SNPs, resolves each sample's missing fraction, and
// emits the kept-sample index set via the shared sample_passes_mind predicate.
//
// Per architecture.md §5 S-1, --mind is an AGGREGATE filter not decidable from one
// tile (it needs every SNP for a sample), so it gets its own light streaming
// pre-pass — distinct from the cheap in-tile MAF/geno filters. This reuses the
// SAME byte/decode path as M1's decode (io::code_in_byte + io::kMissingCode), so
// the missing-handling is identical to the decode front-end.
//
// LAYERING: an `io`-leaf header (architecture.md §4) — host C++20, no CUDA, no
// core/device dependency. Uses io/eigenstrat_format.hpp's byte helpers (the io-side
// shared bit-extraction), NOT core/internal/decode_af.hpp (which would be an
// upward dep) — the two share the SAME pinned bit order by construction.
//
// SAMPLE-GLOBAL invariant (architecture.md §1, §5 S2): the result is a kept-sample
// INDEX set over the whole sample axis — a sample is kept for ALL SNPs or dropped
// for all. Never per-(pop, SNP).
#ifndef STEPPE_IO_FILTER_MIND_PREPASS_HPP
#define STEPPE_IO_FILTER_MIND_PREPASS_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "steppe/config.hpp"  // FilterConfig (mind_max_missing)

namespace steppe::io::filter {

/// A view of the packed genotype records to stream over for --mind: `n_individuals`
/// records of `bytes_per_record` bytes each (individual-major TGENO packing, the
/// same layout as io::GenotypeTile), covering `n_snp` SNPs. Sample `g`'s SNP `s` is
/// the 2-bit code at byte `g*bytes_per_record + s/4`, MSB-first (io::code_in_byte).
/// CUDA-free plain view; the caller owns `packed`.
struct MindPrepassInput {
    const std::uint8_t* packed = nullptr;  ///< packed bytes, one record per individual
    std::size_t bytes_per_record = 0;      ///< stride between individual records
    std::size_t n_snp = 0;                 ///< SNPs spanned per record (the prefix length)
    std::size_t n_individuals = 0;         ///< number of individual records
};

/// Per-sample missingness summary from the pre-pass, exposed so the oracle test can
/// recompute and compare exactly. Parallel to the sample (individual) axis.
struct MindSummary {
    std::vector<std::size_t> nonmissing;  ///< per-sample count of non-missing SNPs
    std::vector<double> missing_frac;     ///< per-sample missing fraction = 1 - nonmissing/n_snp
    std::vector<std::size_t> kept;        ///< ascending indices of samples that pass --mind
};

/// Run the per-sample non-missing accumulation over ALL SNPs and resolve the
/// kept-sample set against `cfg.mind_max_missing` via the shared predicate. When
/// `cfg.mind_max_missing >= 1.0` the pre-pass is a NO-OP: `kept` is every sample
/// index 0..n_individuals-1 and the missing fractions are still reported (so the
/// caller can observe them) but nothing is dropped. With n_snp == 0 (or no packed
/// data) the missing fraction is UNDEFINED — there is no SNP to base a drop on — so
/// every sample reports `missing_frac = 0` and is KEPT, even under an active filter
/// (the no-data fail-safe is keep-all, never drop-all; see the .cpp). This is the
/// opposite of snp_filter's empty-denominator convention (frac 1.0 ⇒ drop) and the
/// divergence is intentional: --mind asks "does this sample have data across SNPs?",
/// which is unanswerable with zero SNPs.
[[nodiscard]] MindSummary run_mind_prepass(const MindPrepassInput& in,
                                           const FilterConfig& cfg);

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_MIND_PREPASS_HPP

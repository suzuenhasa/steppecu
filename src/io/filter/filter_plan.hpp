// src/io/filter/filter_plan.hpp
//
// FilterPlan — the plain PLAN struct the M2 filter front-end produces (architecture
// .md §1 "merge is a plan, not an on-disk rewrite"; §5 S-1/S0'; ROADMAP M2). It is
// a PLAN, NEVER an on-disk rewrite: it carries the resolved kept-sample set, the
// resolved per-SNP keep predicate/mask, and the live in-tile FilterConfig
// thresholds — everything the S0' harmonized+filtered tile produce needs to apply
// the cheap in-tile filters and the conditional S-1 --mind result, with no second
// pass over the data and no rewritten file.
//
// LAYERING: an `io`-leaf header (architecture.md §4) — plain std::vector POD, host
// C++20, no CUDA, no core/device dependency. Crosses the layer boundary unchanged.
//
// THE FILTER INVARIANT (architecture.md §1, §5 S2): every decision in a FilterPlan
// is SNP-GLOBAL or SAMPLE-GLOBAL, never per-(pop, SNP). `snp_keep` is one bool per
// SNP (kept for ALL populations or none); `kept_samples` is one index list over
// the WHOLE sample axis. A per-(pop, SNP) drop would break the symmetric V·Vᵀ
// masking of the 3-GEMM f2 reformulation — so the plan cannot even express one.
#ifndef STEPPE_IO_FILTER_FILTER_PLAN_HPP
#define STEPPE_IO_FILTER_FILTER_PLAN_HPP

#include <cstddef>
#include <vector>

#include "steppe/config.hpp"  // steppe::FilterConfig (the live in-tile thresholds)

namespace steppe::io::filter {

/// The resolved filter PLAN over one dataset's SNP + sample axes.
///
/// Produced upstream of the S0' tile produce (snp_filter builds `snp_keep`;
/// mind_prepass builds `kept_samples`); consumed by the decode front-end to emit
/// only the surviving SNP columns and only the kept samples. A dropped SNP simply
/// contributes nothing to its jackknife block, so block identity is unchanged (the
/// §8 DRY invariant holds) — the plan carries no block info, that stays with the
/// shared block_partition_rule.
struct FilterPlan {
    /// Per-SNP keep mask, parallel to the .snp records in file order: `snp_keep[s]`
    /// is true iff SNP `s` survives ALL SNP-level filters (MAF / geno / membership
    /// / flag-gated monomorphic / transversion / autosome). SNP-GLOBAL: a false
    /// here drops the SNP for every population (never a single (pop, SNP) cell).
    /// Length == number of SNPs considered (M).
    std::vector<bool> snp_keep;

    /// Number of SNPs kept (== count of true in `snp_keep`). Cached so callers do
    /// not re-scan; `n_snp_kept <= snp_keep.size()`.
    std::size_t n_snp_kept = 0;

    /// Kept-sample INDICES over the whole sample (individual) axis, ascending. The
    /// result of the conditional S-1 --mind pre-pass; when --mind is a no-op
    /// (mind_max_missing >= 1.0) this is ALL sample indices (every sample kept).
    /// SAMPLE-GLOBAL: a sample is kept for all SNPs or dropped for all.
    std::vector<std::size_t> kept_samples;

    /// The live in-tile FilterConfig thresholds (MAF / geno + the flag-gated
    /// options) that the S0' tile produce still needs at apply time. Carried by
    /// value so the plan is self-contained; the thresholds here are the SAME ones
    /// snp_filter used to build `snp_keep` (single source — no re-reading).
    FilterConfig in_tile;
};

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_FILTER_PLAN_HPP

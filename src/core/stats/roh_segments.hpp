// src/core/stats/roh_segments.hpp
//
// Host segment-calling + per-individual summary for `steppe roh` — an exact port of
// hapROH's postprocessing.call_roh + PackagesSupport/pp_individual_roh_csvs
// (merge_called_blocks_custom, combine_ROH_df, post_process_roh_df,
// individual_roh_statistic). Pure host, CUDA-free, unit-testable; consumes the per-SNP
// ROH posterior the FB produces (1 - posterior0) and emits hapROH's segment + summary
// columns.
//
// Reference: docs/planning/haproh-face-spec.md §4
#ifndef STEPPE_CORE_STATS_ROH_SEGMENTS_HPP
#define STEPPE_CORE_STATS_ROH_SEGMENTS_HPP

#include <string>
#include <vector>

#include "core/stats/roh_model.hpp"

namespace steppe::core {

// One called ROH segment (postprocessing.create_df columns). Map values are in Morgan;
// the CLI emitter derives the cM aliases (x100) for humans.
struct RohSegment {
    std::string iid;
    int ch = 0;
    long start = 0;      // SNP index (inclusive), in the target's kept-site order
    long end = 0;        // SNP index (exclusive)
    long length = 0;     // SNP count (end - start)
    double startM = 0.0;
    double endM = 0.0;
    double lengthM = 0.0;
    long long startBP = 0;
    long long endBP = 0;
};

// Per-individual ROH summary (pp_individual_roh -> combine_ROH_df columns; sums/max in cM).
struct RohSummary {
    std::string iid;
    double max_roh = 0.0;            // cM
    std::vector<double> sum_cm;      // per bin, cM
    std::vector<long> n_seg;         // per bin, count
};

// The default summary length bins (cM). The paper reports sum_roh>4/>8/>20; the wrapper
// default is [4,8,12] — the gate passes the SAME list to steppe and pip-hapROH.
inline const std::vector<double> kRohSummaryBinsCm = {4.0, 8.0, 12.0, 20.0};

// call_roh: threshold the per-SNP ROH posterior (> cutoff_post), form contiguous runs,
// initial length cut (> roh_min_l_initial Morgan), merge (merge_called_blocks_custom:
// gap < max_gap_merge AND min(len) >= min_len1 AND max(len) >= min_len2, same chrom),
// then final length cut (> roh_min_l_final Morgan). `r_map` is the per-kept-site genetic
// position (Morgan), `bp` the physical position, both length M in the target's kept
// order; `p_roh` is the FB ROH posterior.
[[nodiscard]] std::vector<RohSegment> roh_call_segments(const std::vector<double>& r_map,
                                                        const std::vector<long long>& bp,
                                                        const std::vector<double>& p_roh, int ch,
                                                        const std::string& iid,
                                                        const RohParams& pr);

// combine_ROH_df: concatenate every chromosome's segments per target, re-merge in cM
// space (gap=0.5 cM, min_len1=2 cM, min_len2=4 cM; NO final cut — the un-custom merge),
// then per bin filter `lengthM*100 > bin` AND density `length/(lengthM*100) > snp_cm`
// (default 50) before summing sum_roh / counting n_roh; max_roh is the largest kept
// lengthM (from the smallest bin).
//
// `all_iids` is the full ordered target set: when non-empty, one summary row is emitted
// for EVERY listed target in that order — a zero-ROH individual (no called segments) gets
// an all-zeros row rather than being dropped, mirroring hapROH's pp_individual_roh which
// writes a zeros row per input sample. Any segment iid not in the list is appended after.
// When empty, the legacy first-appearance-of-a-segment order is preserved.
[[nodiscard]] std::vector<RohSummary> roh_summarize(
    const std::vector<RohSegment>& segments, const std::vector<double>& bins_cm = kRohSummaryBinsCm,
    double snp_cm = 50.0, double merge_gap_cm = 0.5, double merge_min_len1_cm = 2.0,
    double merge_min_len2_cm = 4.0, const std::vector<std::string>& all_iids = {});

}  // namespace steppe::core

#endif  // STEPPE_CORE_STATS_ROH_SEGMENTS_HPP

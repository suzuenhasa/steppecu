// src/core/stats/ancibd_segments.hpp
//
// Host segment-calling + per-pair summary for `steppe ibd` — an exact port of
// ancIBD's postprocessing.PostProcessing.call_roh (postprocessing.py:186-237) and
// IO/ind_ibd.create_ind_ibd_df / filter_ibd_df / roh_statistics_df (ind_ibd.py:14-115).
// Pure host, CUDA-free, unit-testable; consumes the per-SNP IBD posterior the FB
// produces (1 - post[0]) and emits the reference's segment + summary columns.
//
// Reference: docs/planning/ancibd-face-spec.md §3b
#ifndef STEPPE_CORE_STATS_ANCIBD_SEGMENTS_HPP
#define STEPPE_CORE_STATS_ANCIBD_SEGMENTS_HPP

#include <string>
#include <vector>

#include "core/stats/ancibd_model.hpp"

namespace steppe::core {

// One called IBD segment (postprocessing.py create_df columns). Map values are in
// Morgan; the CLI emitter derives the cM aliases (x100) for humans.
struct IbdSegment {
    std::string iid1;
    std::string iid2;
    int ch = 0;
    long start = 0;      // SNP index (inclusive), in the pair's run-site order
    long end = 0;        // SNP index (exclusive)
    long length = 0;     // SNP count (end - start)
    double startM = 0.0;
    double endM = 0.0;
    double lengthM = 0.0;
    long long startBP = 0;
    long long endBP = 0;
};

// Per-pair relatedness summary (ind_ibd.py create_ind_ibd_df columns; sums/max in cM).
struct IbdSummary {
    std::string iid1;
    std::string iid2;
    double max_IBD = 0.0;               // cM, from the >=8 cM bin
    double sum_cm[4] = {0, 0, 0, 0};    // sum IBD (cM) for cutoffs [8,12,16,20]
    long n_seg[4] = {0, 0, 0, 0};       // segment count for cutoffs [8,12,16,20]
};

// The summary cutoffs (cM), fixed to the reference default [8,12,16,20].
inline constexpr double kIbdSummaryCutoffsCm[4] = {8.0, 12.0, 16.0, 20.0};

// call_roh: threshold the per-SNP IBD posterior, form contiguous blocks, filter by
// genetic length (> min_cm cM), then merge blocks separated by < max_gap_merge
// Morgan. `r_map` is the per-run-site genetic position (Morgan), `bp` the physical
// position, both length M in the pair's run order; `p_ibd` is the FB IBD posterior.
[[nodiscard]] std::vector<IbdSegment> ancibd_call_segments(const std::vector<double>& r_map,
                                                           const std::vector<long long>& bp,
                                                           const std::vector<double>& p_ibd, int ch,
                                                           const std::string& iid1,
                                                           const std::string& iid2,
                                                           const AncibdParams& pr);

// create_ind_ibd_df: filter the concatenated segment table (density > snp_cm SNP/cM
// AND length > min_cm cM, defaults snp_cm=220, min_cm=6), then per (iid1,iid2)
// compute max_IBD and the per-cutoff sum/count. Preserves first-appearance pair order.
[[nodiscard]] std::vector<IbdSummary> ancibd_summarize(const std::vector<IbdSegment>& segments,
                                                       double snp_cm = 220.0,
                                                       double filter_min_cm = 6.0);

}  // namespace steppe::core

#endif  // STEPPE_CORE_STATS_ANCIBD_SEGMENTS_HPP

// src/core/stats/roh_segments.cpp
//
// Implementation of the hapROH segment caller + per-individual summary (see the
// header). Faithful to postprocessing.call_roh + pp_individual_roh_csvs
// (merge_called_blocks_custom, combine_ROH_df, post_process_roh_df).
#include "core/stats/roh_segments.hpp"

#include <algorithm>
#include <unordered_map>

namespace steppe::core {

namespace {

// Merge a chromosome-sorted run of segments (hapROH merge_called_blocks[_custom]): fuse
// the running block with the next iff same chromosome AND (next.startM - cur.endM <
// max_gap) AND (min(len) >= min_len1) AND (max(len) >= min_len2). Lengths are Morgan.
// `final_cut` (Morgan) applies a post-merge length filter (> final_cut); pass a value
// <= 0 to skip it (the un-custom variant used by the summary re-merge).
[[nodiscard]] std::vector<RohSegment> merge_blocks(const std::vector<RohSegment>& kept,
                                                   double max_gap, double min_len1, double min_len2,
                                                   double final_cut) {
    std::vector<RohSegment> merged;
    if (kept.empty()) return merged;
    RohSegment cur = kept[0];
    for (std::size_t i = 1; i < kept.size(); ++i) {
        const RohSegment& row = kept[i];
        const double len_c = cur.lengthM;
        const double len_n = row.lengthM;
        const bool same_c = (cur.ch == row.ch);
        const bool close = (row.startM - cur.endM) < max_gap;
        const bool gate1 = std::min(len_c, len_n) >= min_len1;
        const bool gate2 = std::max(len_c, len_n) >= min_len2;
        if (same_c && close && gate1 && gate2) {
            cur.end = row.end;
            cur.endM = row.endM;
            cur.length = cur.end - cur.start;
            cur.lengthM = cur.endM - cur.startM;
            cur.endBP = row.endBP;
        } else {
            merged.push_back(cur);
            cur = row;
        }
    }
    merged.push_back(cur);

    if (final_cut > 0.0) {
        std::vector<RohSegment> cut;
        for (const RohSegment& s : merged)
            if (s.lengthM > final_cut) cut.push_back(s);
        return cut;
    }
    return merged;
}

}  // namespace

std::vector<RohSegment> roh_call_segments(const std::vector<double>& r_map,
                                          const std::vector<long long>& bp,
                                          const std::vector<double>& p_roh, int ch,
                                          const std::string& iid, const RohParams& pr) {
    const std::size_t M = p_roh.size();

    // 1) roh status per SNP; 2) contiguous runs via pad-and-diff. `end` is exclusive.
    std::vector<RohSegment> raw;
    std::size_t l = 0;
    while (l < M) {
        if (p_roh[l] > pr.cutoff_post) {
            const std::size_t start = l;
            std::size_t end = l;
            while (end < M && p_roh[end] > pr.cutoff_post) ++end;  // [start, end)
            RohSegment seg;
            seg.iid = iid;
            seg.ch = ch;
            seg.start = static_cast<long>(start);
            seg.end = static_cast<long>(end);
            seg.length = static_cast<long>(end - start);
            seg.startM = r_map[start];
            seg.endM = r_map[end - 1];  // -1 to stay within bounds (call_roh)
            seg.lengthM = seg.endM - seg.startM;
            seg.startBP = bp[start];
            seg.endBP = bp[end - 1];
            raw.push_back(std::move(seg));
            l = end;
        } else {
            ++l;
        }
    }

    // 3) initial length cut: keep lengthM > roh_min_l_initial (Morgan). BEFORE merge.
    std::vector<RohSegment> kept;
    for (const RohSegment& s : raw)
        if (s.lengthM > pr.roh_min_l_initial) kept.push_back(s);

    // 4) merge_called_blocks_custom (gap + gates) then final cut > roh_min_l_final.
    return merge_blocks(kept, pr.max_gap_merge, pr.min_len1, pr.min_len2, pr.roh_min_l_final);
}

std::vector<RohSummary> roh_summarize(const std::vector<RohSegment>& segments,
                                      const std::vector<double>& bins_cm, double snp_cm,
                                      double merge_gap_cm, double merge_min_len1_cm,
                                      double merge_min_len2_cm,
                                      const std::vector<std::string>& all_iids) {
    const std::size_t nbin = bins_cm.size();

    // Group by iid. Seed the emit order (and empty groups) from `all_iids` so every target
    // gets a row — including zero-ROH individuals that contribute no segments (hapROH's
    // pp_individual_roh writes a zeros row per input sample). Segments whose iid is not in
    // that list are appended after in first-appearance order (also covers all_iids empty).
    std::vector<std::string> order;
    std::unordered_map<std::string, std::vector<RohSegment>> groups;
    for (const std::string& iid : all_iids) {
        if (groups.emplace(iid, std::vector<RohSegment>{}).second) order.push_back(iid);
    }
    for (const RohSegment& s : segments) {
        auto it = groups.find(s.iid);
        if (it == groups.end()) {
            order.push_back(s.iid);
            groups.emplace(s.iid, std::vector<RohSegment>{s});
        } else {
            it->second.push_back(s);
        }
    }

    const double merge_gap = merge_gap_cm / 100.0;        // Morgan
    const double merge_min1 = merge_min_len1_cm / 100.0;  // Morgan
    const double merge_min2 = merge_min_len2_cm / 100.0;  // Morgan

    std::vector<RohSummary> out;
    out.reserve(order.size());
    for (const std::string& iid : order) {
        std::vector<RohSegment>& g = groups[iid];
        // Sort by (chrom, startM) so the re-merge sees adjacency correctly.
        std::sort(g.begin(), g.end(), [](const RohSegment& a, const RohSegment& b) {
            if (a.ch != b.ch) return a.ch < b.ch;
            return a.startM < b.startM;
        });
        // Re-merge in cM space, NO final cut (the un-custom variant).
        const std::vector<RohSegment> merged =
            merge_blocks(g, merge_gap, merge_min1, merge_min2, /*final_cut=*/0.0);

        RohSummary sm;
        sm.iid = iid;
        sm.sum_cm.assign(nbin, 0.0);
        sm.n_seg.assign(nbin, 0);
        for (std::size_t j = 0; j < nbin; ++j) {
            const double bin = bins_cm[j];
            double sum = 0.0, mx = 0.0;
            long n = 0;
            for (const RohSegment& s : merged) {
                const double len_cm = s.lengthM * 100.0;
                if (!(len_cm > bin)) continue;
                // Density filter: length (SNP count) / length(cM) > snp_cm.
                const double density = (len_cm > 0.0) ? static_cast<double>(s.length) / len_cm : 0.0;
                if (!(density > snp_cm)) continue;
                sum += s.lengthM;
                mx = std::max(mx, s.lengthM);
                ++n;
            }
            sm.sum_cm[j] = sum * 100.0;
            sm.n_seg[j] = n;
            if (j == 0) sm.max_roh = mx * 100.0;  // max_roh from the smallest bin
        }
        out.push_back(std::move(sm));
    }
    return out;
}

}  // namespace steppe::core

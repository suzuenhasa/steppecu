// src/core/stats/ancibd_segments.cpp
//
// Implementation of the ancIBD segment caller + per-pair summary (see the header).
// Line-for-line faithful to postprocessing.py:186-237 and ind_ibd.py:14-115.
#include "core/stats/ancibd_segments.hpp"

#include <algorithm>
#include <unordered_map>

namespace steppe::core {

std::vector<IbdSegment> ancibd_call_segments(const std::vector<double>& r_map,
                                             const std::vector<long long>& bp,
                                             const std::vector<double>& p_ibd, int ch,
                                             const std::string& iid1, const std::string& iid2,
                                             const AncibdParams& pr) {
    const std::size_t M = p_ibd.size();

    // 1) ibd status per SNP; 2) contiguous runs via pad-and-diff (ibd_stat_to_block).
    //    starts/ends are found where the padded boolean rises/falls; `end` is exclusive.
    std::vector<IbdSegment> raw;
    std::size_t l = 0;
    while (l < M) {
        if (p_ibd[l] > pr.cutoff_post) {
            const std::size_t start = l;
            std::size_t end = l;
            while (end < M && p_ibd[end] > pr.cutoff_post) ++end;  // [start, end)
            IbdSegment seg;
            seg.iid1 = iid1;
            seg.iid2 = iid2;
            seg.ch = ch;
            seg.start = static_cast<long>(start);
            seg.end = static_cast<long>(end);
            seg.length = static_cast<long>(end - start);
            seg.startM = r_map[start];
            seg.endM = r_map[end - 1];  // -1 to stay within bounds (postprocessing.py:203)
            seg.lengthM = seg.endM - seg.startM;
            seg.startBP = bp[start];
            seg.endBP = bp[end - 1];
            raw.push_back(std::move(seg));
            l = end;
        } else {
            ++l;
        }
    }

    // 3) create_df length filter: keep lengthM > min_cm/100 (Morgan). BEFORE merge.
    const double min_len_morgan = pr.min_cm / 100.0;
    std::vector<IbdSegment> kept;
    for (const IbdSegment& s : raw)
        if (s.lengthM > min_len_morgan) kept.push_back(s);

    // 4) merge_called_blocks: fuse adjacent kept blocks whose gap StartM_next - EndM_prev
    //    is < max_gap_merge (Morgan). Preserves the reference's row_c running-merge.
    std::vector<IbdSegment> merged;
    if (pr.max_gap_merge > 0.0 && !kept.empty()) {
        IbdSegment cur = kept[0];
        for (std::size_t i = 0; i < kept.size(); ++i) {
            const IbdSegment& row = kept[i];
            if (row.startM - cur.endM < pr.max_gap_merge) {
                // extend the running block (i==0 folds the first row into itself: a no-op)
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
    } else {
        merged = kept;
    }
    return merged;
}

std::vector<IbdSummary> ancibd_summarize(const std::vector<IbdSegment>& segments, double snp_cm,
                                         double filter_min_cm) {
    // filter_ibd_df: keep rows with (lengthM*100 > filter_min_cm) AND density > snp_cm,
    // density = length / (lengthM*100)  [SNPs per cM].
    std::vector<const IbdSegment*> filt;
    for (const IbdSegment& s : segments) {
        const double len_cm = s.lengthM * 100.0;
        if (!(len_cm > filter_min_cm)) continue;
        const double density = (len_cm > 0.0) ? static_cast<double>(s.length) / len_cm : 0.0;
        if (!(density > snp_cm)) continue;
        filt.push_back(&s);
    }

    // Group by (iid1,iid2), first-appearance order.
    std::vector<std::pair<std::string, std::string>> order;
    std::unordered_map<std::string, std::vector<const IbdSegment*>> groups;
    for (const IbdSegment* s : filt) {
        const std::string key = s->iid1 + '\t' + s->iid2;
        auto it = groups.find(key);
        if (it == groups.end()) {
            order.emplace_back(s->iid1, s->iid2);
            groups.emplace(key, std::vector<const IbdSegment*>{s});
        } else {
            it->second.push_back(s);
        }
    }

    std::vector<IbdSummary> out;
    out.reserve(order.size());
    for (const auto& pr_key : order) {
        const std::string key = pr_key.first + '\t' + pr_key.second;
        const std::vector<const IbdSegment*>& g = groups[key];
        IbdSummary sm;
        sm.iid1 = pr_key.first;
        sm.iid2 = pr_key.second;
        for (int j = 0; j < 4; ++j) {
            const double cut_morgan = kIbdSummaryCutoffsCm[j] / 100.0;
            double sum = 0.0, mx = 0.0;
            long n = 0;
            for (const IbdSegment* s : g) {
                if (s->lengthM > cut_morgan) {
                    sum += s->lengthM;
                    mx = std::max(mx, s->lengthM);
                    ++n;
                }
            }
            sm.sum_cm[j] = sum * 100.0;
            sm.n_seg[j] = n;
            if (j == 0) sm.max_IBD = mx * 100.0;  // max_IBD from the >=8 cM bin (ind_ibd.py:100)
        }
        out.push_back(std::move(sm));
    }
    return out;
}

}  // namespace steppe::core

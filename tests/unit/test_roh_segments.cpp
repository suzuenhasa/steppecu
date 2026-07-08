// tests/unit/test_roh_segments.cpp
//
// Host unit gate for the hapROH segment caller + per-individual summary (`steppe roh`):
// the call_roh threshold -> initial cut -> merge_called_blocks_custom -> final cut chain
// and the combine_ROH_df cM-space re-merge + per-bin density/length filter. Runs on a
// canned per-SNP ROH posterior; device-free, self-checking; CTest gates on the exit code.
#include <cstdio>
#include <string>
#include <vector>

#include "core/stats/roh_segments.hpp"

using namespace steppe::core;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}
bool close(double a, double b, double tol) {
    double d = a - b;
    if (d < 0) d = -d;
    return d <= tol;
}
}  // namespace

int main() {
    RohParams pr;  // cutoff_post=0.999, initial 0.02M, merge gap 0.005M, gates 0.02/0.04, final 0.04M

    // --- 1) call_roh: threshold -> runs -> initial cut -> merge -> final cut ------
    {
        // 25 sites, 0.005 Morgan (0.5 cM) apart. Two ROH runs of 12 sites (5.5 cM each,
        // > final 0.04M cut), split by a below-cutoff site at index 12. The gap between
        // run ends (rm[13]-rm[11] = 0.010M) exceeds max_gap_merge 0.005M -> NO merge.
        const int N = 25;
        std::vector<double> rm(N);
        std::vector<long long> bpv(N);
        std::vector<double> post(N, 0.9999);
        for (int i = 0; i < N; ++i) {
            rm[static_cast<std::size_t>(i)] = 0.005 * i;
            bpv[static_cast<std::size_t>(i)] = 100 + i;
        }
        post[12] = 0.2;  // the break between the two runs
        auto segs = roh_call_segments(rm, bpv, post, 3, "A", pr);
        check(segs.size() == 2, "two ROH segments called");
        if (segs.size() == 2) {
            check(close(segs[0].lengthM, 0.055, 1e-9), "seg0 length 5.5cM (0..11)");
            check(segs[0].startBP == 100 && segs[0].endBP == 111, "seg0 bp span");
            check(close(segs[1].lengthM, 0.055, 1e-9), "seg1 length 5.5cM (13..24)");
        }
        // A short run (2 sites = 0.5 cM) is below the final 0.04M cut -> dropped entirely.
        std::vector<double> post2(N, 0.2);
        post2[0] = 0.9999;
        post2[1] = 0.9999;
        auto segs2 = roh_call_segments(rm, bpv, post2, 3, "A", pr);
        check(segs2.empty(), "sub-final-cut run dropped");
    }

    // --- 2) merge across a small gap (< max_gap_merge) ---------------------------
    {
        // Two 25-site runs (4.8 cM each, > min_len2 0.04M) split by a single below-cutoff
        // site at index 25. Spacing 0.002M so the gap between run ends (rm[26]-rm[24] =
        // 0.004M) is < max_gap_merge 0.005M AND both length gates hold -> MERGE into one.
        const int N = 51;
        std::vector<double> rm(N), postv(N, 0.9999);
        std::vector<long long> bpv(N);
        for (int i = 0; i < N; ++i) {
            rm[static_cast<std::size_t>(i)] = 0.002 * i;  // 0.2 cM spacing
            bpv[static_cast<std::size_t>(i)] = 100 + i;
        }
        postv[25] = 0.1;  // the single-site break
        auto segs = roh_call_segments(rm, bpv, postv, 5, "B", pr);
        check(segs.size() == 1, "small-gap runs merged into one");
        if (segs.size() == 1) check(close(segs[0].lengthM, 0.100, 1e-9), "merged length 10cM");
    }

    // --- 3) summary: cM re-merge + per-bin density/length filter ------------------
    {
        std::vector<RohSegment> segs;
        // One dense 30 cM segment: 30cM -> lengthM 0.30, density > 50 needs length >
        // 30*50 = 1500 SNPs; use 2000.
        RohSegment a;
        a.iid = "S"; a.ch = 1; a.lengthM = 0.30; a.length = 2000;
        a.startM = 0.10; a.endM = 0.40; segs.push_back(a);
        // One sparse 25 cM segment: density too low (100 SNPs / 25 cM = 4 < 50) -> filtered.
        RohSegment b = a; b.ch = 2; b.lengthM = 0.25; b.length = 100;
        b.startM = 1.00; b.endM = 1.25; segs.push_back(b);
        std::vector<double> bins = {4.0, 8.0, 20.0};
        auto sm = roh_summarize(segs, bins);
        check(sm.size() == 1, "one summary individual");
        if (sm.size() == 1) {
            check(close(sm[0].sum_cm[0], 30.0, 1e-9), "sum_roh>4 == 30cM (sparse seg filtered)");
            check(sm[0].n_seg[0] == 1, "n_roh>4 == 1");
            check(sm[0].n_seg[2] == 1, "n_roh>20 == 1 (30 > 20)");
            check(close(sm[0].max_roh, 30.0, 1e-9), "max_roh == 30cM");
        }
    }

    if (g_fail == 0) std::printf("test_roh_segments: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}

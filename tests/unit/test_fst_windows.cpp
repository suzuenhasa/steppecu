// tests/unit/test_fst_windows.cpp
//
// Host-only unit test of the bp-window LAYOUT for `steppe fst --windowed` / `--pbs`
// (core/stats/bp_windows.hpp). Window-edge alignment is the primary risk for matching
// scikit-allel's windowed_weir_cockerham_fst, so this pins build_bp_windows against the exact
// allel semantics (position_windows edge generation + windowed_statistic searchsorted counts):
//   (1) the documented end-window-clip example (pos in [5..88], size=20 step=10),
//   (2) 1-based inclusive edges + searchsorted (side='left' start, side='right' stop) counts,
//   (3) overlapping windows (step < size) — a SNP falls in ceil(size/step) windows,
//   (4) chromosome-reset — each chromosome windows independently from its own first/last pos,
//   (5) a single-position chromosome yields no windows (allel's empty range(start, stop, step)).
// Pure C++ TU: NO CUDA, NO data.
#include <cstdio>
#include <span>
#include <vector>

#include "core/stats/bp_windows.hpp"

namespace {

using steppe::core::BpWindow;
using steppe::core::build_bp_windows;

int g_failures = 0;

void check_true(const char* what, bool ok) {
    if (!ok) {
        std::printf("  [FAIL] %s\n", what);
        ++g_failures;
    }
}

void check_eq(const char* what, long got, long want) {
    if (got != want) {
        std::printf("  [FAIL] %-44s got=%ld want=%ld\n", what, got, want);
        ++g_failures;
    }
}

std::vector<BpWindow> windows_for(const std::vector<int>& chrom, const std::vector<long>& pos,
                                  long size, long step) {
    std::vector<double> physpos(pos.begin(), pos.end());
    return build_bp_windows(std::span<const int>(chrom.data(), chrom.size()),
                            std::span<const double>(physpos.data(), physpos.size()), size, step);
}

// The documented allel example: pos in [5..88], size=20 step=10 -> 8 windows, the LAST clipped
// to stop=88 ([75,88] not [75,94]). Positions 5,15,...,85,88 present so every window has SNPs.
void test_end_window_clip() {
    std::printf("-- end-window clip (allel position_windows example) --\n");
    std::vector<long> pos;
    for (long p = 5; p <= 85; p += 10) pos.push_back(p);  // 5,15,...,85
    pos.push_back(88);                                    // the last SNP defines stop=88
    std::vector<int> chrom(pos.size(), 1);

    const auto w = windows_for(chrom, pos, /*size=*/20, /*step=*/10);
    const long expect_start[] = {5, 15, 25, 35, 45, 55, 65, 75};
    const long expect_end[] = {24, 34, 44, 54, 64, 74, 84, 88};
    check_eq("window count", static_cast<long>(w.size()), 8);
    for (std::size_t k = 0; k < w.size() && k < 8; ++k) {
        check_eq("start", w[k].start, expect_start[k]);
        check_eq("end", w[k].end, expect_end[k]);
    }
    // The clipped last window [75,88] contains positions 75,85,88 -> n_snp = 3.
    if (w.size() == 8) check_eq("last window n_snp", w[7].hi - w[7].lo, 3);
}

// searchsorted bounds: a SNP at position p is in window iff start <= p <= end. n_snp counts ALL
// positions in [start,end]. size=step=10 (non-overlapping) over pos {10,10,15,20,30}.
void test_inclusive_counts() {
    std::printf("-- inclusive edges + searchsorted counts --\n");
    const std::vector<long> pos{10, 10, 15, 20, 30};  // duplicate 10 on purpose
    const std::vector<int> chrom(pos.size(), 2);
    const auto w = windows_for(chrom, pos, /*size=*/10, /*step=*/10);
    // start=10, stop=30. range(10,30,10) -> 10, 20. window0=[10,19], window1=[20,30] (clip).
    check_eq("window count", static_cast<long>(w.size()), 2);
    if (w.size() == 2) {
        check_eq("w0 start", w[0].start, 10);
        check_eq("w0 end", w[0].end, 19);
        check_eq("w0 n_snp", w[0].hi - w[0].lo, 3);  // 10,10,15
        check_eq("w1 start", w[1].start, 20);
        check_eq("w1 end", w[1].end, 30);            // clipped to stop
        check_eq("w1 n_snp", w[1].hi - w[1].lo, 2);  // 20,30
    }
}

// Overlapping windows (step < size): a SNP falls in multiple windows, each summing its own slice.
void test_overlapping() {
    std::printf("-- overlapping windows (step < size) --\n");
    std::vector<long> pos;
    for (long p = 1; p <= 40; ++p) pos.push_back(p);  // 1..40 dense
    const std::vector<int> chrom(pos.size(), 3);
    const auto w = windows_for(chrom, pos, /*size=*/20, /*step=*/10);
    // start=1, stop=40. range(1,40,10) -> 1,11,21,31, but allel CLIPS-AND-BREAKS on the first
    // window whose stop reaches `stop`: [1,20], [11,30], [21,40] (21+20=41>=40 -> clip to 40,
    // last, break). The ws=31 start is never emitted -> 3 windows, not 4.
    check_eq("window count", static_cast<long>(w.size()), 3);
    if (w.size() == 3) {
        check_eq("w0", w[0].hi - w[0].lo, 20);  // pos 1..20
        check_eq("w1", w[1].hi - w[1].lo, 20);  // pos 11..30
        check_eq("w2 end", w[2].end, 40);
        check_eq("w2", w[2].hi - w[2].lo, 20);  // pos 21..40
    }
}

// Chromosome-reset: each chromosome windows from its own first/last position independently.
void test_chromosome_reset() {
    std::printf("-- chromosome reset --\n");
    // chr1: 100,110,120  chr2: 5,15
    const std::vector<long> pos{100, 110, 120, 5, 15};
    const std::vector<int> chrom{1, 1, 1, 2, 2};
    const auto w = windows_for(chrom, pos, /*size=*/10, /*step=*/10);
    // chr1 start=100 stop=120: range(100,120,10)->100,110 => [100,109],[110,120]
    // chr2 start=5 stop=15:   range(5,15,10)->5        => [5,15]
    check_eq("total window count", static_cast<long>(w.size()), 3);
    if (w.size() == 3) {
        check_true("w0 chrom==1", w[0].chrom == 1);
        check_eq("w0 start", w[0].start, 100);
        check_eq("w1 end", w[1].end, 120);
        check_true("w2 chrom==2", w[2].chrom == 2);
        check_eq("w2 start", w[2].start, 5);
        check_eq("w2 end", w[2].end, 15);
        // chr2 slice indices are global (start at 3).
        check_eq("w2 lo", w[2].lo, 3);
        check_eq("w2 hi", w[2].hi, 5);
    }
}

// A single-position chromosome -> allel's range(start, stop, step) with start==stop is empty.
void test_single_position() {
    std::printf("-- single-position chromosome yields no windows --\n");
    const std::vector<long> pos{42};
    const std::vector<int> chrom{7};
    const auto w = windows_for(chrom, pos, 10, 10);
    check_eq("window count", static_cast<long>(w.size()), 0);
}

}  // namespace

int main() {
    std::printf("=== bp-window layout (allel-exact) ===\n");
    test_end_window_clip();
    test_inclusive_counts();
    test_overlapping();
    test_chromosome_reset();
    test_single_position();
    if (g_failures == 0) {
        std::printf("OK: all bp-window layout checks passed\n");
        return 0;
    }
    std::printf("FAILED: %d check(s)\n", g_failures);
    return 1;
}

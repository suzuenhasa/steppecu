// tests/unit/test_block_ranges.cpp
//
// Host-only unit test of the SINGLE-SOURCE inverse of assign_blocks:
// core::block_ranges (architecture.md §2 fail-fast, §5, §8, §13; ROADMAP M4;
// cleanup X-3/B3). Pure C++ TU, NO GPU, NO io.
//
// block_ranges turns the dense, non-decreasing per-SNP block_id[] into the
// per-block half-open column ranges [begin, end) that BOTH backends derive their
// per-block layout from (the CUDA path's block_offsets/block_sizes the kernel
// dereferences, the CPU oracle's begin/end). That scan used to be hand-duplicated
// in both backends (and a third near-copy in the M4 equivalence test) and NONE
// validated the partition, so a malformed block_id was a silent out-of-bounds
// write on the host range vectors and an out-of-bounds device read. This test
// pins:
//   1. the VALID layouts (the assign_blocks postcondition): empty, single-block,
//      multi-block contiguous runs, a permutation-free dense partition, and that
//      block_ranges agrees bit-for-bit with assign_blocks's own output;
//   2. the MALFORMED partitions are FAIL-FAST (throw std::runtime_error), closing
//      the OOB the B3 fix targets: empty-but-n_block>0 (short/null block_id),
//      out-of-range id (< 0 and >= n_block), and non-monotonic block_id.
//
// These cases assert no statistic — they exercise the CONTROL FLOW and the
// fail-fast contract of the inverse rule, the allowed use of a hand-built layout
// (ROADMAP §0: no synthetic data for PRECISION claims; this is control flow).
//
// Dual harness (identical to tests/unit/test_block_partition.cpp): with
// -DSTEPPE_TEST_WITH_GTEST it uses GoogleTest; otherwise it is a self-checking
// main() returning non-zero on the first failure. No CUDA here.
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/domain/block_partition_rule.hpp"  // block_ranges, BlockRange, assign_blocks, BlockPartition, block_size_cm_to_morgans
#include "steppe/config.hpp"                      // kDefaultBlockSizeCm

namespace {

using steppe::core::assign_blocks;
using steppe::core::block_ranges;
using steppe::core::block_size_cm_to_morgans;
using steppe::core::BlockPartition;
using steppe::core::BlockRange;

// Convenience: build a span over a vector<int> (the block_id[] shape).
[[nodiscard]] std::span<const int> as_span(const std::vector<int>& v) {
    return std::span<const int>(v.data(), v.size());
}

// Did calling block_ranges(block_id, M, n_block) throw std::runtime_error?
[[nodiscard]] bool throws_runtime_error(const std::vector<int>& block_id, long M, int n_block) {
    try {
        (void)block_ranges(as_span(block_id), M, n_block);
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;  // some other exception type → not the fail-fast we promise
    }
    return false;  // no throw
}

// --- VALID: empty input → empty ranges --------------------------------------
// M <= 0 OR n_block <= 0 → empty vector (the backends early-out on this too).
[[nodiscard]] bool test_empty() {
    const std::vector<int> empty;
    const auto r0 = block_ranges(as_span(empty), 0, 0);
    const auto r1 = block_ranges(as_span(empty), 0, 3);   // M=0 dominates
    // A non-empty block_id with M=0 must STILL produce an empty result (nothing
    // to scan), never touching block_id.
    const std::vector<int> some = {0, 0, 1};
    const auto r2 = block_ranges(as_span(some), 0, 2);
    return r0.empty() && r1.empty() && r2.empty();
}

// --- VALID: single block over all M columns ---------------------------------
// The n_block==1 (M0-equivalent) shape: one range [0, M).
[[nodiscard]] bool test_single_block() {
    const std::vector<int> block_id = {0, 0, 0, 0, 0};
    const auto r = block_ranges(as_span(block_id), 5, 1);
    return r.size() == 1 && r[0].begin == 0 && r[0].end == 5 && r[0].size() == 5;
}

// --- VALID: multiple contiguous blocks --------------------------------------
// block_id = [0,0,1,2,2,2] over M=6, n_block=3 → ranges [0,2),[2,3),[3,6).
[[nodiscard]] bool test_multi_block() {
    const std::vector<int> block_id = {0, 0, 1, 2, 2, 2};
    const auto r = block_ranges(as_span(block_id), 6, 3);
    if (r.size() != 3) return false;
    return r[0].begin == 0 && r[0].end == 2 && r[0].size() == 2 &&
           r[1].begin == 2 && r[1].end == 3 && r[1].size() == 1 &&
           r[2].begin == 3 && r[2].end == 6 && r[2].size() == 3;
}

// --- VALID: the inverse round-trips assign_blocks's own output --------------
// Feed a real assign_blocks partition (two chroms, a gap, a negative position)
// back through block_ranges and check the ranges partition [0, M) exactly: every
// column is covered once, the ranges are contiguous and ascending, and the sizes
// sum to M. This pins that the producer (assign_blocks) and the inverse
// (block_ranges) agree by construction — the §8 single-source property.
[[nodiscard]] bool test_roundtrip_assign_blocks() {
    const double bs = block_size_cm_to_morgans(steppe::kDefaultBlockSizeCm);  // 0.05 M
    // chr1: bin 0,0,1 ; chr2: bin 0 ; chr2 negative-pos bin -1 then bin 0.
    const std::vector<int> chrom = {1, 1, 1, 2, 2, 2};
    const std::vector<double> gp = {0.00, 0.01, 0.06, 0.02, -0.001, 0.03};
    const BlockPartition bp = assign_blocks(chrom, gp, bs);
    const long M = static_cast<long>(bp.block_id.size());
    const auto r = block_ranges(as_span(bp.block_id), M, bp.n_block);
    if (static_cast<int>(r.size()) != bp.n_block) return false;

    long covered = 0;
    long expected_begin = 0;
    for (int b = 0; b < bp.n_block; ++b) {
        // contiguous, ascending, non-empty (assign_blocks never emits an empty block)
        if (r[static_cast<std::size_t>(b)].begin != expected_begin) return false;
        if (r[static_cast<std::size_t>(b)].size() <= 0) return false;
        // every column in the range carries this block id
        for (long s = r[static_cast<std::size_t>(b)].begin;
             s < r[static_cast<std::size_t>(b)].end; ++s) {
            if (bp.block_id[static_cast<std::size_t>(s)] != b) return false;
        }
        covered += r[static_cast<std::size_t>(b)].size();
        expected_begin = r[static_cast<std::size_t>(b)].end;
    }
    return covered == M && expected_begin == M;
}

// --- MALFORMED: empty/short block_id with n_block > 0 → throw ----------------
// M=4 columns are required but block_id is empty (data() may legally be null) and
// n_block=1 > 0, so the n_block<=0 early-out is SKIPPED. block_ranges must reject
// the short partition rather than dereference a null/short pointer (the corrected
// F-1 null-deref sub-case).
[[nodiscard]] bool test_malformed_short_block_id() {
    const std::vector<int> empty;
    if (!throws_runtime_error(empty, 4, 1)) return false;        // empty, M>0
    const std::vector<int> shortv = {0, 0};                       // 2 < M=4
    return throws_runtime_error(shortv, 4, 1);
}

// --- MALFORMED: an id >= n_block → throw ------------------------------------
// block_id = [0,1,2] but n_block=2 → id 2 is out of [0,2). Unchecked, this would
// write ranges[2] past the n_block-sized vector (the OOB B3 closes).
[[nodiscard]] bool test_malformed_id_too_large() {
    const std::vector<int> block_id = {0, 1, 2};
    return throws_runtime_error(block_id, 3, 2);
}

// --- MALFORMED: a negative id → throw ---------------------------------------
// A stray -1 (e.g. an all-negative-position regression) must be rejected, not
// used to index ranges[-1].
[[nodiscard]] bool test_malformed_id_negative() {
    const std::vector<int> block_id = {-1, 0, 1};
    return throws_runtime_error(block_id, 3, 2);
}

// --- MALFORMED: non-monotonic block_id → throw ------------------------------
// block_id = [0,1,0] is in range but NOT non-decreasing, so block 0's SNPs are
// not one contiguous run — the contiguity contract every consumer relies on is
// violated. Must throw.
[[nodiscard]] bool test_malformed_non_monotonic() {
    const std::vector<int> block_id = {0, 1, 0};
    return throws_runtime_error(block_id, 3, 2);
}

struct Case {
    const char* name;
    bool (*fn)();
};

constexpr Case kCases[] = {
    {"empty input (M<=0 or n_block<=0) -> empty ranges", test_empty},
    {"single block -> one range [0, M)", test_single_block},
    {"multiple contiguous blocks -> correct [begin,end)", test_multi_block},
    {"block_ranges round-trips assign_blocks output (partition [0,M) exactly)", test_roundtrip_assign_blocks},
    {"MALFORMED: empty/short block_id with n_block>0 -> throws", test_malformed_short_block_id},
    {"MALFORMED: id >= n_block -> throws", test_malformed_id_too_large},
    {"MALFORMED: negative id -> throws", test_malformed_id_negative},
    {"MALFORMED: non-monotonic block_id -> throws", test_malformed_non_monotonic},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(BlockRanges, ValidAndMalformedCases) {
    for (const auto& c : kCases) {
        EXPECT_TRUE(c.fn()) << "failed: " << c.name;
    }
}
#else
int main() {
    int failures = 0;
    for (const auto& c : kCases) {
        const bool ok = c.fn();
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (!ok) ++failures;
    }
    if (failures != 0) {
        std::fprintf(stderr, "test_block_ranges: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_block_ranges: all checks PASS\n");
    return 0;
}
#endif

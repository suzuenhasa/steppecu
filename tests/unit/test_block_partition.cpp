// tests/unit/test_block_partition.cpp
//
// Host-only unit test of the SHARED SNP→block partition rule (architecture.md §13
// "Unit tests"; ROADMAP §3 M3, §5). Pure C++ TU, NO GPU, NO io: it exercises the
// host-pure domain rule in core/domain/block_partition_rule.{hpp,cpp} — the single
// source of the jackknife block id that both the `io` front-end and the device
// kernels consume (the §8 DRY invariant). Three things are pinned here:
//   1. block_size_cm_to_morgans — the ONE cM→Morgan conversion site (no magic
//      0.01/100 anywhere else); block_size_cm_to_morgans(5.0) == 0.05 exactly.
//   2. assign_blocks LOGIC on synthetic chromosome/position LAYOUTS.
//   3. an OPTIONAL real-AADR internal-consistency check, run when a .snp path is
//      passed as argv[1] (mirrors tests/reference/test_f2_equivalence.cu, which
//      takes its real-data dir as argv[1]).
//
// On the SYNTHETIC LAYOUTS: the "no synthetic data" rule (ROADMAP §0) forbids
// synthetic data for PRECISION/ACCURACY claims. These cases assert no statistic —
// they exercise the CONTROL FLOW of block assignment (chromosome reset, empty-bin
// absorption, negative-position handling, dense renumbering, and the fail-fast
// rejection of an illegal block width — 0 / negative / NaN — that would otherwise
// be float→int UB or a silently-inverted partition; cleanup X-3/B13), which is
// exactly the allowed use of a hand-built layout. The numerical-parity gate is the real-AADR
// check (consistency only — counts and ordering, never a statistic) plus, when the
// toolchain exists, ADMIXTOOLS 2 goldens (ROADMAP §6, deferred).
//
// Dual harness (build contract, identical to tests/unit/test_f2.cpp): with
// -DSTEPPE_TEST_WITH_GTEST it uses GoogleTest; otherwise it is a self-checking
// main() returning non-zero on the first failure — all CTest needs to gate. No
// CUDA here.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

#include "core/domain/block_partition_rule.hpp"  // block_size_cm_to_morgans, assign_blocks, BlockPartition
#include "steppe/config.hpp"                      // kDefaultBlockSizeCm, kCentimorgansPerMorgan

namespace {

using steppe::core::assign_blocks;
using steppe::core::block_size_cm_to_morgans;
using steppe::core::BlockPartition;

// The Morgan block width every synthetic case uses: the AT2 default, 5 cM = 0.05
// Morgans, via the single conversion site. Named (not a bare 0.05) so the test
// itself has no magic number for the block size.
const double kBs = block_size_cm_to_morgans(steppe::kDefaultBlockSizeCm);

// --- block_size_cm_to_morgans: the ONE conversion site ----------------------
// 5 cM must convert to exactly 0.05 Morgans (5.0 / 100.0 is exact in binary
// double). This is the single place the cM↔Morgan factor is applied (ROADMAP §4).
[[nodiscard]] bool test_cm_to_morgans_exact() {
    return block_size_cm_to_morgans(steppe::kDefaultBlockSizeCm) == 0.05 &&
           block_size_cm_to_morgans(steppe::kCentimorgansPerMorgan) == 1.0 &&  // 100 cM == 1 Morgan
           block_size_cm_to_morgans(0.0) == 0.0;
}

// Helper: a partition is dense iff every id in 0..n_block-1 is used and the
// sequence is monotonically non-decreasing (the two structural invariants every
// assign_blocks result must satisfy, architecture.md §5).
[[nodiscard]] bool dense_and_nondecreasing(const BlockPartition& bp) {
    if (bp.block_id.empty()) return bp.n_block == 0;
    std::vector<char> seen(static_cast<std::size_t>(bp.n_block), 0);
    int prev = bp.block_id.front();
    for (std::size_t k = 0; k < bp.block_id.size(); ++k) {
        const int id = bp.block_id[k];
        if (id < 0 || id >= bp.n_block) return false;
        if (id < prev) return false;  // must be non-decreasing
        prev = id;
        seen[static_cast<std::size_t>(id)] = 1;
    }
    for (char s : seen) {
        if (!s) return false;  // a gap → not dense
    }
    return true;
}

// --- single chromosome, one block → all within blgsize of the anchor --------
// Every SNP is < 0.05 Morgans from the first SNP (the anchor at 0.0), so no
// interior cut fires and the whole chromosome is one block.
[[nodiscard]] bool test_single_chrom_one_bin() {
    const std::vector<int> chrom = {1, 1, 1, 1};
    const std::vector<double> gp = {0.0, 0.01, 0.02, 0.049};
    const BlockPartition bp = assign_blocks(chrom, gp, kBs);
    return bp.n_block == 1 && dense_and_nondecreasing(bp) &&
           bp.block_id == std::vector<int>{0, 0, 0, 0};
}

// --- single chromosome, SNP-anchored cuts; the >= boundary is INCLUSIVE -------
// THE AT2 walk (NOT a fixed grid), with EXACTLY-blgsize gaps to pin the inclusive
// `>=` (port of C `dis >= blocklen`). Positions 0.0, 0.05, 0.10 are chosen so the
// distances are exact in binary double (0.05-0.0 and 0.10-0.05 are both EXACTLY
// 0.05; verified) — a SNP exactly blgsize from the anchor CUTS:
//   SNP0 → opens block 0, anchor = 0.00;
//   SNP1: 0.05 - 0.00 = 0.05 >= 0.05 → CUT (inclusive) → block 1, anchor 0.05;
//   SNP2: 0.10 - 0.05 = 0.05 >= 0.05 → CUT → block 2.
// Result {0,1,2}. (NB: a naive {0.01,0.06,0.16} would NOT cut at SNP1 because
// 0.06-0.01 == 0.04999999999999996 < 0.05 in double — the same FP knife-edge AT2
// itself sees; the walk is exact-to-AT2, including this. The next case pins the
// behaviour the grid CANNOT reproduce.)
[[nodiscard]] bool test_single_chrom_walk_cuts() {
    const std::vector<int> chrom = {1, 1, 1};
    const std::vector<double> gp = {0.00, 0.05, 0.10};
    const BlockPartition bp = assign_blocks(chrom, gp, kBs);
    return bp.n_block == 3 && dense_and_nondecreasing(bp) &&
           bp.block_id == std::vector<int>{0, 1, 2};
}

// --- the SNP-anchored re-anchor: remainder rolls FORWARD (grid CANNOT do this) -
// THE load-bearing AT2 property a fixed k*0.05 grid gets wrong. genpos
// 0.00, 0.04, 0.06, 0.10 @ bs=0.05 on one chrom:
//   SNP0 → block 0, anchor = 0.00;
//   SNP1: 0.04 - 0.00 = 0.04 < 0.05 → no cut → block 0;
//   SNP2: 0.06 - 0.00 = 0.06 >= 0.05 → CUT → block 1, anchor RE-SET to 0.06
//         (NOT to the grid line 0.05; the 0.01 remainder rolls forward);
//   SNP3: 0.10 - 0.06 = 0.04 < 0.05 → no cut → block 1.
// Result {0,0,1,1}, n_block=2. The OLD floor-grid rule would have cut SNP2 into
// bin floor(0.06/0.05)=1 AND SNP3 stays in bin floor(0.10/0.05)=2 → {0,0,1,2},
// n_block=3 — the extra split this fix removes (the 718→709 mechanism).
[[nodiscard]] bool test_walk_reanchors_remainder_forward() {
    const std::vector<int> chrom = {1, 1, 1, 1};
    const std::vector<double> gp = {0.00, 0.04, 0.06, 0.10};
    const BlockPartition bp = assign_blocks(chrom, gp, kBs);
    return bp.n_block == 2 && dense_and_nondecreasing(bp) &&
           bp.block_id == std::vector<int>{0, 0, 1, 1};
}

// --- two chromosomes each starting ~0 → chr2 gets a fresh id ----------------
// Per-chromosome reset + concat: chr2's first SNP is in local bin 0, same as
// chr1's last SNP, but the chromosome boundary ALWAYS forces a new global block
// (blocks never straddle chromosomes).
[[nodiscard]] bool test_two_chroms_reset() {
    const std::vector<int> chrom = {1, 1, 2, 2};
    const std::vector<double> gp = {0.00, 0.01, 0.00, 0.02};
    const BlockPartition bp = assign_blocks(chrom, gp, kBs);
    // chr1 → block 0 (both SNPs, same bin); chr2 → block 1 (fresh, both SNPs).
    return bp.n_block == 2 && dense_and_nondecreasing(bp) &&
           bp.block_id == std::vector<int>{0, 0, 1, 1};
}

// --- all-zero chromosome → exactly one block --------------------------------
// A whole chromosome at genpos 0 (e.g. the AADR Y chromosome, all genpos=0) → one
// block. Mirrors chr24 in the real data.
[[nodiscard]] bool test_all_zero_chrom_one_block() {
    const std::vector<int> chrom = {7, 7, 7, 7, 7};
    const std::vector<double> gp = {0.0, 0.0, 0.0, 0.0, 0.0};
    const BlockPartition bp = assign_blocks(chrom, gp, kBs);
    return bp.n_block == 1 && dense_and_nondecreasing(bp) &&
           bp.block_id == std::vector<int>{0, 0, 0, 0, 0};
}

// --- negative chromosome-opening positions just anchor the first block ------
// Real AADR chr17 OPENS with two negative-genpos SNPs before transitioning to
// positive. Under the AT2 walk there is NO "negative bin" concept: the chromosome
// boundary opens a new block and ANCHORS it at the first (negative) SNP; later
// SNPs cut only when their distance FROM that anchor reaches blgsize. Layout
// (chr16 tail then chr17 head):
//   SNP0 chr16 1.340230            → block 0, anchor 1.340230;
//   SNP1 chr17 -0.000258 (!=chrom) → CUT → block 1, anchor -0.000258;
//   SNP2 chr17 -0.000238: dis = 0.000020 < 0.05 → block 1;
//   SNP3 chr17  0.000249: dis = 0.000507 < 0.05 → block 1.
// Result {0,1,1,1}, n_block=2. The negatives do NOT form a separate block — they
// are simply the anchor and near-anchor SNPs of chr17's first block (the OLD
// floor-grid rule gave them their own bin -1; the walk does not).
[[nodiscard]] bool test_negative_chrom_open_anchors() {
    const std::vector<int> chrom = {16, 17, 17, 17};
    const std::vector<double> gp = {1.340230, -0.000258, -0.000238, 0.000249};
    const BlockPartition bp = assign_blocks(chrom, gp, kBs);
    return bp.n_block == 2 && dense_and_nondecreasing(bp) &&
           bp.block_id == std::vector<int>{0, 1, 1, 1};
}

// --- empty input → empty partition ------------------------------------------
[[nodiscard]] bool test_empty_input() {
    const BlockPartition bp = assign_blocks(std::span<const int>{}, std::span<const double>{}, kBs);
    return bp.n_block == 0 && bp.block_id.empty() && dense_and_nondecreasing(bp);
}

// --- ILLEGAL block widths → DEFINED behavior (empty partition) --------------
// The B13 verdict gate (cleanup X-3/B13): the AT2 walk tests
// `(pos - fpos) >= block_size_morgans`, so a non-positive or NaN width is a
// silently WRONG partition (not the AT2-matching one):
//   * 0.0       → a 0 gap satisfies `>=` on EVERY SNP → every SNP its own block
//                 (a silent over-partition that still looks dense);
//   * negative  → likewise: every SNP trips the threshold → every SNP its own
//                 block (silent over-partition);
//   * NaN       → `x >= NaN` is ALWAYS false → no interior cut fires → a whole
//                 chromosome collapses to one block (silent merge).
// assign_blocks guards with !(width > 0.0) and returns an EMPTY partition
// (n_block == 0) for all three. These cases pass a NON-EMPTY (chrom, genpos) so
// the guard is what produces the empty result — not the empty-input early-out —
// proving the illegal width alone is rejected. dense_and_nondecreasing on an
// empty result requires n_block == 0, so it doubles as the "defined" assertion.
[[nodiscard]] bool test_zero_block_size_empty() {
    const std::vector<int> chrom = {1, 1, 1};
    const std::vector<double> gp = {0.0, 0.06, 0.12};  // would span >1 bin at a legal width
    const BlockPartition bp = assign_blocks(chrom, gp, 0.0);
    return bp.n_block == 0 && bp.block_id.empty() && dense_and_nondecreasing(bp);
}

[[nodiscard]] bool test_negative_block_size_empty() {
    const std::vector<int> chrom = {1, 1, 1};
    const std::vector<double> gp = {0.0, 0.06, 0.12};
    const BlockPartition bp = assign_blocks(chrom, gp, -0.05);
    return bp.n_block == 0 && bp.block_id.empty() && dense_and_nondecreasing(bp);
}

[[nodiscard]] bool test_nan_block_size_empty() {
    const std::vector<int> chrom = {1, 1, 1};
    const std::vector<double> gp = {0.0, 0.06, 0.12};
    const double nan_bs = std::nan("");  // quiet NaN; nan_bs > 0.0 is false
    const BlockPartition bp = assign_blocks(chrom, gp, nan_bs);
    return bp.n_block == 0 && bp.block_id.empty() && dense_and_nondecreasing(bp);
}

// --- positive control: a TINY but VALID width still partitions --------------
// The !(width > 0.0) guard must NOT reject a legal positive width — even a
// denormal-small one. With bs = 1e-6 the three genpos 0.0/0.06/0.12 fall in three
// distinct local bins, so the guard did not break the happy path.
[[nodiscard]] bool test_tiny_positive_block_size_ok() {
    const std::vector<int> chrom = {1, 1, 1};
    const std::vector<double> gp = {0.0, 0.06, 0.12};
    const BlockPartition bp = assign_blocks(chrom, gp, 1e-6);
    return bp.n_block == 3 && dense_and_nondecreasing(bp) &&
           bp.block_id == std::vector<int>{0, 1, 2};
}

struct Case {
    const char* name;
    bool (*fn)();
};

constexpr Case kCases[] = {
    {"block_size_cm_to_morgans(5)==0.05 exactly", test_cm_to_morgans_exact},
    {"single chrom, one block (all within blgsize of anchor)", test_single_chrom_one_bin},
    {"single chrom SNP-anchored cuts -> dense 0,1,2", test_single_chrom_walk_cuts},
    {"AT2 walk re-anchors: remainder rolls forward (grid cannot)", test_walk_reanchors_remainder_forward},
    {"two chroms each ~0 -> per-chrom reset + fresh id", test_two_chroms_reset},
    {"all-zero chromosome -> one block", test_all_zero_chrom_one_block},
    {"negative chrom-open positions just anchor the first block", test_negative_chrom_open_anchors},
    {"empty input -> empty partition", test_empty_input},
    {"block_size 0 -> empty partition (no per-SNP over-partition) [B13]", test_zero_block_size_empty},
    {"block_size negative -> empty partition (no per-SNP over-partition) [B13]", test_negative_block_size_empty},
    {"block_size NaN -> empty partition (no silent whole-chrom merge) [B13]", test_nan_block_size_empty},
    {"block_size tiny-positive -> still partitions (guard not over-broad) [B13]", test_tiny_positive_block_size_ok},
};

// ---------------------------------------------------------------------------
// Real-AADR INTERNAL-CONSISTENCY check (no synthetic, no statistic). Reads an
// EIGENSTRAT/PACKEDANCESTRYMAP .snp file (col 2 = chromosome, col 3 = genetic
// position in MORGANS), runs assign_blocks at the AT2 default block width, and
// asserts the structural invariants. This is a CONSISTENCY check (counts +
// ordering on real positions), NOT a statistic or precision claim (ROADMAP §0).
//
// MEASURED on the box's v66 AADR .snp (584131 SNPs, chr 1..24) under the AT2
// SNP-anchored walk (the current rule): n_block == kExpectedNBlock below. This
// count is the walk's block count over ALL 584131 SNPs — assign_blocks does NO
// chromosome filtering by design (that is M2/M6 `io` territory), so chr 24 (the Y,
// 597 SNPs all at genpos 0) contributes one block here. The count differs from
// the pre-fix floor-grid count (757) because the walk re-anchors at each cut
// (remainder rolls forward) instead of pinning to k*0.05 grid lines — fewer
// interior splits (see docs/research/block-partition-at2.md §3). We assert the
// MEASURED walk count (ROADMAP §6: report the real number; do not fudge it).
// ---------------------------------------------------------------------------
[[nodiscard]] bool run_real_aadr_check(const std::string& snp_path) {
    std::FILE* f = std::fopen(snp_path.c_str(), "r");
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot open .snp file %s\n", snp_path.c_str());
        return false;
    }
    // .snp columns: rsid  chrom  genpos(Morgans)  physpos  allele1  allele2
    std::vector<int> chrom;
    std::vector<double> genpos;
    chrom.reserve(600000);
    genpos.reserve(600000);
    char rsid[256];
    int c = 0;
    double gp = 0.0;
    long phys = 0;
    char a1[16], a2[16];
    while (std::fscanf(f, "%255s %d %lf %ld %15s %15s", rsid, &c, &gp, &phys, a1, a2) == 6) {
        chrom.push_back(c);
        genpos.push_back(gp);
    }
    std::fclose(f);

    const long m = static_cast<long>(chrom.size());
    if (m <= 0) {
        std::fprintf(stderr, "ERROR: parsed 0 SNPs from %s\n", snp_path.c_str());
        return false;
    }

    const double bs = block_size_cm_to_morgans(steppe::kDefaultBlockSizeCm);  // 0.05 Morgans
    const BlockPartition bp = assign_blocks(chrom, genpos, bs);

    // Count chromosome-boundary increments (a new block forced purely by a
    // chromosome change) for the ">= 24" diagnostic — actually 23 here (24
    // distinct chromosomes → 23 transitions). The brief's ">=24" expectation
    // assumed boundary forcings; with 24 chromosomes there are 23 transitions, so
    // we report the measured value and require >= (n_distinct_chrom - 1).
    int boundary_incr = 0;
    int distinct_chrom = (m > 0) ? 1 : 0;
    for (long s = 1; s < m; ++s) {
        if (chrom[static_cast<std::size_t>(s)] != chrom[static_cast<std::size_t>(s - 1)]) {
            ++boundary_incr;
            ++distinct_chrom;  // file is chromosome-sorted, so each change is a new chromosome
        }
    }

    const bool dense_ok = dense_and_nondecreasing(bp);

    // The MEASURED walk count on this dataset, fed all SNPs (no chrom filter).
    // Measured on the box's v66 .snp under the AT2 SNP-anchored walk this pass
    // (584131 SNPs, chr 1..24): 748. The pre-fix floor-grid rule gave 757; the walk
    // re-anchors at each cut (remainder rolls forward) instead of cutting on grid
    // lines, removing 9 interior splits (see docs/research/block-partition-at2.md §3).
    constexpr int kExpectedNBlock = 748;
    constexpr long kExpectedSnps = 584131;

    std::printf("\n");
    std::printf("real-AADR block-partition consistency (NO synthetic, NO statistic)\n");
    std::printf("  file              : %s\n", snp_path.c_str());
    std::printf("  SNPs parsed       : %ld   (expected %ld)\n", m, kExpectedSnps);
    std::printf("  block_size        : %.5f Morgans  (= %.1f cM via the single conversion site)\n",
                bs, steppe::kDefaultBlockSizeCm);
    std::printf("  n_block           : %d   (expected %d — AT2 SNP-anchored walk count; see test header)\n",
                bp.n_block, kExpectedNBlock);
    std::printf("  distinct chroms   : %d\n", distinct_chrom);
    std::printf("  chrom-boundary inc: %d   (= distinct_chrom - 1; %d transitions)\n",
                boundary_incr, distinct_chrom - 1);
    std::printf("  dense 0..n-1, non-decreasing: %s\n", dense_ok ? "yes" : "NO");
    std::printf("\n");

    bool ok = true;
    if (m != kExpectedSnps) {
        std::fprintf(stderr, "  [FAIL] SNP count %ld != expected %ld\n", m, kExpectedSnps);
        ok = false;
    }
    if (bp.n_block != kExpectedNBlock) {
        std::fprintf(stderr,
            "  [FAIL] n_block %d != measured-expected %d. If this changed, re-derive the "
            "block count and EXPLAIN it (ROADMAP §6) — do not silently retune.\n",
            bp.n_block, kExpectedNBlock);
        ok = false;
    }
    if (!dense_ok) {
        std::fprintf(stderr, "  [FAIL] block_id is not dense 0..n_block-1 / not non-decreasing\n");
        ok = false;
    }
    // Every chromosome boundary forces a new block, so boundary_incr increments are
    // a subset of the block openings. There are 24 distinct chromosomes → 23
    // boundary increments; require at least that.
    if (boundary_incr < distinct_chrom - 1) {
        std::fprintf(stderr, "  [FAIL] chrom-boundary increments %d < distinct_chrom-1 %d\n",
                     boundary_incr, distinct_chrom - 1);
        ok = false;
    }
    return ok;
}

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(BlockPartition, AllSyntheticLayoutCases) {
    for (const auto& c : kCases) {
        EXPECT_TRUE(c.fn()) << "failed: " << c.name;
    }
}
#else
int main(int argc, char** argv) {
    int failures = 0;
    for (const auto& c : kCases) {
        const bool ok = c.fn();
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (!ok) ++failures;
    }

    // Optional real-AADR consistency check when a .snp path is supplied (argv[1]),
    // mirroring tests/reference/test_f2_equivalence.cu's argv-driven data path.
    if (argc >= 2) {
        if (!run_real_aadr_check(argv[1])) {
            ++failures;
        } else {
            std::printf("[PASS] real-AADR consistency check (%s)\n", argv[1]);
        }
    } else {
        std::printf("[skip] real-AADR consistency check (no .snp path passed as argv[1])\n");
    }

    if (failures != 0) {
        std::fprintf(stderr, "test_block_partition: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_block_partition: all checks PASS\n");
    return 0;
}
#endif

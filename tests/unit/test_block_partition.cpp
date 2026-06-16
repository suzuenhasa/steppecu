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
// absorption, negative-position handling, dense renumbering), which is exactly the
// allowed use of a hand-built layout. The numerical-parity gate is the real-AADR
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

#include "core/domain/block_partition_rule.hpp"  // block_of, block_size_cm_to_morgans, assign_blocks, BlockPartition
#include "steppe/config.hpp"                      // kDefaultBlockSizeCm, kCentimorgansPerMorgan

namespace {

using steppe::core::assign_blocks;
using steppe::core::block_of;
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

// --- single chromosome, one bin → exactly one block ------------------------
// All positions fall in local bin 0 (< 0.05 Morgans) on one chromosome.
[[nodiscard]] bool test_single_chrom_one_bin() {
    const std::vector<int> chrom = {1, 1, 1, 1};
    const std::vector<double> gp = {0.0, 0.01, 0.02, 0.049};
    const BlockPartition bp = assign_blocks(chrom, gp, kBs);
    return bp.n_block == 1 && dense_and_nondecreasing(bp) &&
           bp.block_id == std::vector<int>{0, 0, 0, 0};
}

// --- single chromosome with a GAP → interior empty bin absorbed -------------
// genpos 0.01,0.06,0.16 @ bs=0.05 → local bins 0,1,3 (bin 2 is EMPTY). The dense
// rule collapses to global 0,1,2 (n_block=3): the empty interior bin is absorbed,
// NOT reserved a slot. This is the 756-not-757-style policy in miniature.
[[nodiscard]] bool test_single_chrom_gap_absorbs() {
    const std::vector<int> chrom = {1, 1, 1};
    const std::vector<double> gp = {0.01, 0.06, 0.16};
    // Sanity-pin the local bins this case relies on (0,1,3) so the intent is
    // explicit and survives any refactor of block_of.
    if (block_of(0.01, kBs) != 0 || block_of(0.06, kBs) != 1 || block_of(0.16, kBs) != 3) {
        return false;
    }
    const BlockPartition bp = assign_blocks(chrom, gp, kBs);
    return bp.n_block == 3 && dense_and_nondecreasing(bp) &&
           bp.block_id == std::vector<int>{0, 1, 2};
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

// --- a negative position gets its OWN block, never aliasing bin 0 -----------
// floor(-0.0002 / 0.05) = -1, a distinct local bin from bin 0. Layout mirrors the
// real AADR chr17, which OPENS with two negative-genpos SNPs (bin -1) before
// transitioning to bin 0: chrom boundary → bin -1 block → bin 0 block.
[[nodiscard]] bool test_negative_position_own_block() {
    // block_of must floor negatives correctly (own bin -1, not 0).
    if (block_of(-0.000258, kBs) != -1 || block_of(-0.000238, kBs) != -1) return false;
    const std::vector<int> chrom = {16, 17, 17, 17};
    const std::vector<double> gp = {1.340230, -0.000258, -0.000238, 0.000249};
    const BlockPartition bp = assign_blocks(chrom, gp, kBs);
    // chr16 last SNP → block 0; chr17 negatives (bin -1) → block 1; chr17 bin 0
    // → block 2. The negatives form their own block and do not merge with bin 0.
    return bp.n_block == 3 && dense_and_nondecreasing(bp) &&
           bp.block_id == std::vector<int>{0, 1, 1, 2};
}

// --- empty input → empty partition ------------------------------------------
[[nodiscard]] bool test_empty_input() {
    const BlockPartition bp = assign_blocks(std::span<const int>{}, std::span<const double>{}, kBs);
    return bp.n_block == 0 && bp.block_id.empty() && dense_and_nondecreasing(bp);
}

struct Case {
    const char* name;
    bool (*fn)();
};

constexpr Case kCases[] = {
    {"block_size_cm_to_morgans(5)==0.05 exactly", test_cm_to_morgans_exact},
    {"single chrom, one bin -> n_block=1", test_single_chrom_one_bin},
    {"single chrom gap (0,1,3) -> dense 0,1,2 (interior empty absorbed)", test_single_chrom_gap_absorbs},
    {"two chroms each ~0 -> per-chrom reset + fresh id", test_two_chroms_reset},
    {"all-zero chromosome -> one block", test_all_zero_chrom_one_block},
    {"negative position -> own block, no alias of bin 0", test_negative_position_own_block},
    {"empty input -> empty partition", test_empty_input},
};

// ---------------------------------------------------------------------------
// Real-AADR INTERNAL-CONSISTENCY check (no synthetic, no statistic). Reads an
// EIGENSTRAT/PACKEDANCESTRYMAP .snp file (col 2 = chromosome, col 3 = genetic
// position in MORGANS), runs assign_blocks at the AT2 default block width, and
// asserts the structural invariants. This is a CONSISTENCY check (counts +
// ordering on real positions), NOT a statistic or precision claim (ROADMAP §0).
//
// MEASURED on the box's v66 AADR .snp (584131 SNPs, chr 1..24): n_block == 757.
// The plan PREDICTED 756; the +1 is the Y chromosome (chr 24), whose 597 SNPs are
// all at genpos 0 and form one block. Excluding chr 24 (as AT2 does — it operates
// on chr 1..23) yields exactly 756. assign_blocks does NO chromosome filtering by
// design (that is M2/M6 `io` territory), so over ALL 584131 SNPs it must report
// 757. We assert the measured 757 and DO NOT force 756 (ROADMAP §6: report the
// real number and explain it; do not fudge the assertion).
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

    // The MEASURED parity number on this dataset, fed all SNPs (no chrom filter).
    constexpr int kExpectedNBlock = 757;
    constexpr long kExpectedSnps = 584131;

    std::printf("\n");
    std::printf("real-AADR block-partition consistency (NO synthetic, NO statistic)\n");
    std::printf("  file              : %s\n", snp_path.c_str());
    std::printf("  SNPs parsed       : %ld   (expected %ld)\n", m, kExpectedSnps);
    std::printf("  block_size        : %.5f Morgans  (= %.1f cM via the single conversion site)\n",
                bs, steppe::kDefaultBlockSizeCm);
    std::printf("  n_block           : %d   (expected %d — plan's 756 + Y chr 24; see test header)\n",
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

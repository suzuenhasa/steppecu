// tests/unit/test_shard_plan.cpp
//
// HOST-PURE, GPU-FREE, DATA-FREE unit test of the M4.5 SPMG block-aligned shard
// planner steppe::device::plan_block_shards (src/device/shard_plan.{hpp,cpp};
// architecture.md §11.4 SPMG tile sharding, §12 PARITY LAW; cleanup TST-1 /
// shard_plan D-1). NO GPU, NO CUDA call, NO io, NO real data: plan_block_shards is
// a deterministic pure function of (ranges, G) over the CUDA-free core::BlockRange,
// so this exercises it in isolation — the testability the unit's own header promises
// ("equally exercisable host-only", shard_plan.hpp) but that the GPU parity test
// (test_f2_multigpu_parity.cu) could only hit transitively on near-uniform AADR
// blocks, never the skew/edge cases.
//
// This is the GPU-free coverage gate that makes the B6 parameter-drop safe to land
// without re-running the slow GPU parity check for every edit. THE FIX UNDER TEST
// (B6 / X1 / shard_plan D-1): plan_block_shards dropped its redundant `block_sizes`
// parameter and now derives each block's SNP count from `ranges[b].size()` (a `long`,
// the single source of both the balance math AND each shard's [s0, s1)). The plan it
// produces MUST be byte-identical to the old (block_sizes, ranges, G) signature —
// `block_sizes[b]` was literally `ranges[b].size()` narrowed to int, so deriving the
// count straight from `ranges[b].size()` is the same value with no narrowing. Below,
// PLAN_GREEDY_REFERENCE recomputes the documented greedy independently and every case
// asserts plan == reference, pinning the byte-identity the audit requires.
//
// WHY A PLAIN .cpp (NOT a .cu): plan_block_shards is host-pure CUDA-FREE — it names
// only core::BlockRange + std types, reaches no GPU, includes no <cuda_runtime.h>,
// and its TU (shard_plan.cpp) is a plain .cpp in steppe_device that itself references
// no CUDA symbol. So this test links the CUDA-free seam directly: steppe::core_internal
// (the header-only src/ include root that resolves "device/shard_plan.hpp" +
// "core/domain/block_partition_rule.hpp") + the one out-of-line symbol from
// steppe::device. A CUDA leak in the planner would fail this host compile (the §4
// layering proof, same discipline as test_f2_partials_validate.cpp).
//
// Self-checking main() (CTest gates on the exit code), mirroring the host-unit-test
// convention (test_f2_partials_validate.cpp): each named case returns bool, main runs
// the table and returns non-zero if any failed.
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/domain/block_partition_rule.hpp"  // steppe::core::BlockRange
#include "device/shard_plan.hpp"                  // plan_block_shards, DeviceShard

namespace {

using steppe::core::BlockRange;
using steppe::device::DeviceShard;
using steppe::device::plan_block_shards;

// ---------------------------------------------------------------------------
// Build a CONTIGUOUS [begin, end) range vector from a list of per-block SNP counts:
// block b occupies columns [Σ_{<b} sizes, Σ_{<=b} sizes). This is exactly the shape
// core::block_ranges emits for a dense non-decreasing partition (contiguous columns
// per block), so the planner sees a faithful stand-in for a real partition.
// ---------------------------------------------------------------------------
[[nodiscard]] std::vector<BlockRange> ranges_from_sizes(const std::vector<long>& sizes) {
    std::vector<BlockRange> r;
    r.reserve(sizes.size());
    long cursor = 0;
    for (long sz : sizes) {
        r.push_back(BlockRange{cursor, cursor + sz});
        cursor += sz;
    }
    return r;
}

// ---------------------------------------------------------------------------
// PLAN_GREEDY_REFERENCE — an INDEPENDENT recomputation of the documented greedy
// (shard_plan.hpp / .cpp): close device g once its cumulative SNP count reaches
// ceil(total/G) while devices remain and b is not the last block; the last device
// absorbs the tail; trailing devices stay empty. Written from the SPEC, not by
// calling plan_block_shards, so plan == reference is a true cross-check that the
// B6-refactored planner is byte-identical to the documented behavior.
// ---------------------------------------------------------------------------
[[nodiscard]] std::vector<DeviceShard> plan_greedy_reference(
    const std::vector<BlockRange>& ranges, std::size_t G) {
    std::vector<DeviceShard> plan(G);  // {0,0,0,0} each
    const std::size_t n_block = ranges.size();
    if (n_block == 0) return plan;

    long total = 0;
    for (const auto& rg : ranges) total += rg.size();
    const long Gs = static_cast<long>(G);
    const long target = (total + Gs - 1) / Gs;  // ceil(total/G)

    std::size_t g = 0, b0 = 0;
    long dev = 0;
    for (std::size_t b = 0; b < n_block; ++b) {
        dev += ranges[b].size();
        if ((g + 1 < G) && (dev >= target) && (b + 1 < n_block)) {
            const std::size_t b1 = b + 1;
            plan[g] = DeviceShard{static_cast<int>(b0), static_cast<int>(b1),
                                  ranges[b0].begin, ranges[b1 - 1].end};
            ++g;
            b0 = b1;
            dev = 0;
        }
    }
    plan[g] = DeviceShard{static_cast<int>(b0), static_cast<int>(n_block),
                          ranges[b0].begin, ranges[n_block - 1].end};
    return plan;
}

[[nodiscard]] bool shards_equal(const DeviceShard& a, const DeviceShard& b) {
    return a.b0 == b.b0 && a.b1 == b.b1 && a.s0 == b.s0 && a.s1 == b.s1;
}

// Structural invariants every valid plan must satisfy (the parity floor +
// contract): exactly G shards; non-empty shards are a prefix (empties trailing
// only); the non-empty prefix tiles [0, n_block) contiguously with no gap/overlap;
// each shard's [s0,s1) matches its block span's columns; whole-block alignment.
[[nodiscard]] bool check_structure(const std::vector<DeviceShard>& plan,
                                   const std::vector<BlockRange>& ranges,
                                   std::size_t G, const char* tag) {
    const int n_block = static_cast<int>(ranges.size());
    if (plan.size() != G) {
        std::fprintf(stderr, "[%s] plan.size()=%zu != G=%zu\n", tag, plan.size(), G);
        return false;
    }
    // Walk the contiguous prefix of non-empty shards; once empty, all rest empty.
    int next_b = 0;        // the block id the next non-empty shard must start at
    long next_s = 0;       // the SNP column it must start at
    bool seen_empty = false;
    for (std::size_t g = 0; g < G; ++g) {
        const DeviceShard& sh = plan[g];
        if (sh.empty()) { seen_empty = true; continue; }
        if (seen_empty) {  // a non-empty shard AFTER an empty one => not trailing-only
            std::fprintf(stderr, "[%s] non-empty shard g=%zu after an empty one\n", tag, g);
            return false;
        }
        if (sh.b0 != next_b) {
            std::fprintf(stderr, "[%s] g=%zu b0=%d, expected contiguous %d\n",
                         tag, g, sh.b0, next_b);
            return false;
        }
        if (sh.b1 <= sh.b0 || sh.b1 > n_block) {
            std::fprintf(stderr, "[%s] g=%zu bad block span [%d,%d) (n_block=%d)\n",
                         tag, g, sh.b0, sh.b1, n_block);
            return false;
        }
        // Whole-block-aligned column span: s0 == ranges[b0].begin, s1 == ranges[b1-1].end.
        if (sh.s0 != ranges[static_cast<std::size_t>(sh.b0)].begin ||
            sh.s1 != ranges[static_cast<std::size_t>(sh.b1 - 1)].end ||
            sh.s0 != next_s) {
            std::fprintf(stderr, "[%s] g=%zu column span [%ld,%ld) not block-aligned/contiguous\n",
                         tag, g, sh.s0, sh.s1);
            return false;
        }
        next_b = sh.b1;
        next_s = sh.s1;
    }
    // The union of the non-empty shards must tile exactly [0, n_block).
    if (next_b != n_block) {
        std::fprintf(stderr, "[%s] coverage ends at block %d, expected n_block=%d\n",
                     tag, next_b, n_block);
        return false;
    }
    return true;
}

// A full case: build ranges from sizes, plan, assert structure + byte-identity to
// the independent greedy reference, and (optional) an expected exact layout.
[[nodiscard]] bool run_case(const char* tag, const std::vector<long>& sizes, std::size_t G,
                            const std::vector<DeviceShard>* expect = nullptr) {
    const std::vector<BlockRange> ranges = ranges_from_sizes(sizes);
    const std::vector<DeviceShard> plan = plan_block_shards(
        std::span<const BlockRange>(ranges.data(), ranges.size()), G);

    if (!check_structure(plan, ranges, G, tag)) return false;

    const std::vector<DeviceShard> ref = plan_greedy_reference(ranges, G);
    for (std::size_t g = 0; g < G; ++g) {
        if (!shards_equal(plan[g], ref[g])) {
            std::fprintf(stderr,
                "[%s] g=%zu plan{%d,%d,%ld,%ld} != reference{%d,%d,%ld,%ld} (B6 byte-identity broken)\n",
                tag, g, plan[g].b0, plan[g].b1, plan[g].s0, plan[g].s1,
                ref[g].b0, ref[g].b1, ref[g].s0, ref[g].s1);
            return false;
        }
    }
    if (expect != nullptr) {
        for (std::size_t g = 0; g < G; ++g) {
            if (!shards_equal(plan[g], (*expect)[g])) {
                std::fprintf(stderr,
                    "[%s] g=%zu plan{%d,%d,%ld,%ld} != EXPECT{%d,%d,%ld,%ld}\n",
                    tag, g, plan[g].b0, plan[g].b1, plan[g].s0, plan[g].s1,
                    (*expect)[g].b0, (*expect)[g].b1, (*expect)[g].s0, (*expect)[g].s1);
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

// Even split: 6 equal blocks of 10 SNPs over G=3 => two blocks each, exact thirds.
[[nodiscard]] bool test_even_split() {
    const std::vector<DeviceShard> expect = {
        {0, 2, 0, 20}, {2, 4, 20, 40}, {4, 6, 40, 60}};
    return run_case("even split 6x10 / G=3", {10, 10, 10, 10, 10, 10}, 3, &expect);
}

// G == 1: the single range [0,n_block) over all columns (the exact single-GPU shard).
[[nodiscard]] bool test_g1() {
    const std::vector<DeviceShard> expect = {{0, 4, 0, 40}};
    return run_case("G==1 single range", {10, 10, 10, 10}, 1, &expect);
}

// n_block == 0: G empty shards {0,0,0,0}; nothing to compute/combine.
[[nodiscard]] bool test_empty_input() {
    const std::vector<DeviceShard> expect = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    return run_case("n_block==0 -> G empty", {}, 3, &expect);
}

// n_block < G: leading non-empty one-per-device, TRAILING devices empty.
[[nodiscard]] bool test_n_block_lt_g() {
    const std::vector<DeviceShard> expect = {
        {0, 1, 0, 10}, {1, 2, 10, 20}, {2, 3, 20, 30}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    return run_case("n_block<G trailing empty (3 blocks / G=5)", {10, 10, 10}, 5, &expect);
}

// All-empty blocks (total_snps==0 => target==0): every block closes a device while
// devices remain; tiling still exact (one block per device then the rest), balance
// vacuous (the documented total_snps==0 corner; shard_plan.cpp C-3). 4 zero-size
// blocks / G=3 => [0,1) [1,2) [2,4). The structure + reference check pin it.
[[nodiscard]] bool test_all_empty_blocks() {
    const std::vector<DeviceShard> expect = {{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 4, 0, 0}};
    return run_case("all-empty blocks target-0 (4x0 / G=3)", {0, 0, 0, 0}, 3, &expect);
}

// Giant FIRST block: the threshold logic commits dev0 to the giant block and the
// small tail collapses onto dev1 — the prefix-threshold greedy's known skew
// (shard_plan P-1; dev2 idle). {1000,1,1} / G=3, total=1002, target=ceil(1002/3)=334:
// b=0 dev0=1000>=334 closes dev0=[0,1); b=1 dev1=1 (<334) no close; b=2 is the last
// block so the (b+1<n_block) guard blocks any close — the final close sweeps the tail
// onto dev1=[1,3) and dev2 stays EMPTY. Pins the EXACT documented (skewed) layout so a
// balance-quality change (P-1) would be a deliberate, visible plan change — and the
// byte-identity reference confirms B6 did not move it.
[[nodiscard]] bool test_giant_first_block() {
    const std::vector<DeviceShard> expect = {
        {0, 1, 0, 1000}, {1, 3, 1000, 1002}, {0, 0, 0, 0}};
    return run_case("giant first block {1000,1,1} / G=3", {1000, 1, 1}, 3, &expect);
}

// Giant LAST block: the (b+1<n_block) close guard prevents any close ON the final
// block, so the last device absorbs the giant — all SNPs land on the last NON-empty
// run. {1,1,1,1,1000} / G=3: dev0 closes at the first cross of target=ceil(1004/3)=335
// (only the final 1000-block crosses it, but the guard blocks closing on it), so the
// whole thing collapses to dev0=[0,5). Structure+reference pin the documented behavior.
[[nodiscard]] bool test_giant_last_block() {
    const std::vector<DeviceShard> expect = {
        {0, 5, 0, 1004}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    return run_case("giant last block {1,1,1,1,1000} / G=3", {1, 1, 1, 1, 1000}, 3, &expect);
}

// Uneven-but-balanceable interior: 5,5,5,5,5 / G=2 (total 25, target ceil(25/2)=13):
// dev0 accumulates 5,10,15 -> closes at b=2 (15>=13), dev1=[3,5). Pins the close-on-
// cross point against the reference.
[[nodiscard]] bool test_uneven_two_device() {
    const std::vector<DeviceShard> expect = {{0, 3, 0, 15}, {3, 5, 15, 25}};
    return run_case("uneven 5x5 / G=2", {5, 5, 5, 5, 5}, 2, &expect);
}

// G == 0: fail-fast throw (a shard plan needs at least one device).
[[nodiscard]] bool test_g0_throws() {
    const std::vector<BlockRange> ranges = ranges_from_sizes({10, 10});
    try {
        (void)plan_block_shards(std::span<const BlockRange>(ranges.data(), ranges.size()), 0);
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        std::fprintf(stderr, "[G==0] wrong exception type\n");
        return false;
    }
    std::fprintf(stderr, "[G==0] did NOT throw\n");
    return false;
}

// A non-uniform realistic-ish partition (per-chromosome-tail-like short blocks among
// uniform interiors) over G=4 — exercises a larger table through the structure +
// reference cross-check (no hand-coded expect; the independent greedy is the oracle).
[[nodiscard]] bool test_mixed_realistic() {
    return run_case("mixed sizes / G=4",
                    {50, 50, 50, 7, 50, 50, 50, 3, 50, 50, 50, 11}, 4);
}

struct NamedTest {
    const char* name;
    bool (*fn)();
};

}  // namespace

int main() {
    const NamedTest tests[] = {
        {"even split", test_even_split},
        {"G==1 single range", test_g1},
        {"n_block==0 empty", test_empty_input},
        {"n_block<G trailing empty", test_n_block_lt_g},
        {"all-empty blocks (target 0)", test_all_empty_blocks},
        {"giant first block (skew)", test_giant_first_block},
        {"giant last block (skew)", test_giant_last_block},
        {"uneven two-device close-on-cross", test_uneven_two_device},
        {"G==0 fail-fast throw", test_g0_throws},
        {"mixed realistic vs reference", test_mixed_realistic},
    };

    int failures = 0;
    for (const NamedTest& t : tests) {
        const bool ok = t.fn();
        std::fprintf(stderr, "[%s] %s\n", ok ? "PASS" : "FAIL", t.name);
        if (!ok) ++failures;
    }
    if (failures == 0) {
        std::fprintf(stderr, "test_shard_plan: ALL %zu cases passed\n",
                     sizeof(tests) / sizeof(tests[0]));
    } else {
        std::fprintf(stderr, "test_shard_plan: %d case(s) FAILED\n", failures);
    }
    return failures == 0 ? 0 : 1;
}

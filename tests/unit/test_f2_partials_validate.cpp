// tests/unit/test_f2_partials_validate.cpp
//
// Host-only unit test of the SHARED f2-combine precondition guard
// core::validate_f2_partials (cleanup B5 / X7; architecture.md §2 fail-fast, §8
// single-source, §13). Pure C++ TU, NO GPU, NO io.
//
// validate_f2_partials is the ONE home of the partial/shard/P/storage/tiling
// contract that BOTH f2 combine tiers — the host-staged core::combine_f2_partials_host
// and the device-resident device::combine_f2_partials_p2p — call, so the two tiers
// reject malformed inputs IDENTICALLY (their parity-neutrality, architecture.md
// §11.4/§12, rests on it). It used to be DUPLICATED byte-for-byte between the two
// TUs; this test pins the single shared guard's contract directly, GPU-free, in
// milliseconds — the fast inner-loop gate for the slow GPU parity test.
//
// It pins:
//   1. the VALID cases pass (single-shard, two-shard tiling, empty/trailing-empty
//      degenerates, the all-empty n_block==0 corner);
//   2. EVERY malformed case FAILS FAST (throws std::runtime_error): partials/shards
//      count mismatch, negative P / n_block_full, n_block != shard span, P
//      disagreement, the SHORT-PARTIAL storage gap this fix closes (f2/vpair/
//      block_sizes under-sized vs P*P*n_block), and shards that do not tile
//      [0, n_block_full).
//
// These cases assert no statistic — they exercise the CONTROL FLOW and the fail-fast
// contract of the shared guard, the allowed use of a hand-built layout (ROADMAP §0:
// no synthetic data for PRECISION claims; this is control flow).
//
// Dual harness (identical to tests/unit/test_block_ranges.cpp): with
// -DSTEPPE_TEST_WITH_GTEST it uses GoogleTest; otherwise it is a self-checking
// main() returning non-zero on the first failure. No CUDA here — the CUDA-free host
// compile is itself the §4-layering proof that the shared guard names no CUDA type.
#include <cstdio>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/fstats/f2_partials_validate.hpp"  // validate_f2_partials
#include "steppe/fstats.hpp"                      // F2BlockTensor
#include "device/shard_plan.hpp"                  // DeviceShard

namespace {

using steppe::F2BlockTensor;
using steppe::core::validate_f2_partials;
using steppe::device::DeviceShard;

constexpr const char* kWho = "test";

// Build a WELL-FORMED compact partial owning `n_block` blocks at population count P:
// f2/vpair length == P*P*n_block, block_sizes length == n_block.
[[nodiscard]] F2BlockTensor make_partial(int P, int n_block) {
    F2BlockTensor t;
    t.P = n_block > 0 ? P : 0;  // an empty partial carries no slab / P
    t.n_block = n_block;
    const std::size_t slabs = static_cast<std::size_t>(P) *
                              static_cast<std::size_t>(P) *
                              static_cast<std::size_t>(n_block > 0 ? n_block : 0);
    t.f2.assign(slabs, 0.0);
    t.vpair.assign(slabs, 0.0);
    t.block_sizes.assign(static_cast<std::size_t>(n_block > 0 ? n_block : 0), 0);
    return t;
}

[[nodiscard]] DeviceShard shard(int b0, int b1) {
    DeviceShard s;
    s.b0 = b0;
    s.b1 = b1;
    return s;
}

// Did validate_f2_partials(partials, shards, P, n_block_full) throw runtime_error?
[[nodiscard]] bool throws(std::span<const F2BlockTensor> partials,
                          std::span<const DeviceShard> shards,
                          int P, int n_block_full) {
    try {
        validate_f2_partials(kWho, partials, shards, P, n_block_full);
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;  // a different exception type → not the fail-fast we promise
    }
    return false;
}

// --- VALID: one shard covering all blocks -----------------------------------
[[nodiscard]] bool test_valid_single_shard() {
    const std::vector<F2BlockTensor> partials = {make_partial(3, 4)};
    const std::vector<DeviceShard> shards = {shard(0, 4)};
    return !throws(partials, shards, 3, 4);
}

// --- VALID: two shards tiling [0, 5) contiguously ---------------------------
[[nodiscard]] bool test_valid_two_shard_tiling() {
    const std::vector<F2BlockTensor> partials = {make_partial(2, 2), make_partial(2, 3)};
    const std::vector<DeviceShard> shards = {shard(0, 2), shard(2, 5)};
    return !throws(partials, shards, 2, 5);
}

// --- VALID: a trailing EMPTY shard (n_block < G) ----------------------------
// Device 1 owns no blocks (b0 == b1): an empty partial (n_block==0) is exempt from
// the P/storage checks and contributes 0 to the cover.
[[nodiscard]] bool test_valid_trailing_empty() {
    const std::vector<F2BlockTensor> partials = {make_partial(4, 3), make_partial(4, 0)};
    const std::vector<DeviceShard> shards = {shard(0, 3), shard(3, 3)};
    return !throws(partials, shards, 4, 3);
}

// --- VALID: the all-empty (n_block_full == 0) corner ------------------------
// G empty shards {0,0}; every partial empty; cover == 0 == n_block_full.
[[nodiscard]] bool test_valid_all_empty() {
    const std::vector<F2BlockTensor> partials = {make_partial(5, 0), make_partial(5, 0)};
    const std::vector<DeviceShard> shards = {shard(0, 0), shard(0, 0)};
    return !throws(partials, shards, 5, 0);
}

// --- MALFORMED: partials count != shards count → throw ----------------------
[[nodiscard]] bool test_malformed_count_mismatch() {
    const std::vector<F2BlockTensor> partials = {make_partial(2, 2), make_partial(2, 2)};
    const std::vector<DeviceShard> shards = {shard(0, 4)};  // only 1 shard for 2 partials
    return throws(partials, shards, 2, 4);
}

// --- MALFORMED: negative P / n_block_full → throw ---------------------------
[[nodiscard]] bool test_malformed_negative() {
    const std::vector<F2BlockTensor> partials = {make_partial(2, 2)};
    const std::vector<DeviceShard> shards = {shard(0, 2)};
    return throws(partials, shards, -1, 2) && throws(partials, shards, 2, -1);
}

// --- MALFORMED: partial n_block != shard span → throw -----------------------
[[nodiscard]] bool test_malformed_span_mismatch() {
    const std::vector<F2BlockTensor> partials = {make_partial(2, 2)};  // n_block=2
    const std::vector<DeviceShard> shards = {shard(0, 3)};             // span=3
    return throws(partials, shards, 2, 3);
}

// --- MALFORMED: non-empty partial P disagrees with combined P → throw -------
[[nodiscard]] bool test_malformed_p_disagree() {
    std::vector<F2BlockTensor> partials = {make_partial(2, 2)};
    partials[0].P = 3;  // scalar P now disagrees with combined P=2
    const std::vector<DeviceShard> shards = {shard(0, 2)};
    return throws(partials, shards, 2, 2);
}

// --- MALFORMED: SHORT partial — f2 under-sized vs P*P*n_block → throw --------
// THE gap this fix (B5 / C1) closes: scalar n_block/P correct, but f2 storage too
// short. The combine would have read past it (unchecked operator[]). Now rejected.
[[nodiscard]] bool test_malformed_short_f2() {
    std::vector<F2BlockTensor> partials = {make_partial(3, 2)};  // wants 3*3*2 = 18
    partials[0].f2.resize(10);                                   // short
    const std::vector<DeviceShard> shards = {shard(0, 2)};
    return throws(partials, shards, 3, 2);
}

// --- MALFORMED: SHORT partial — vpair under-sized → throw -------------------
[[nodiscard]] bool test_malformed_short_vpair() {
    std::vector<F2BlockTensor> partials = {make_partial(3, 2)};
    partials[0].vpair.resize(5);  // short (wants 18)
    const std::vector<DeviceShard> shards = {shard(0, 2)};
    return throws(partials, shards, 3, 2);
}

// --- MALFORMED: SHORT partial — block_sizes under-sized → throw -------------
[[nodiscard]] bool test_malformed_short_block_sizes() {
    std::vector<F2BlockTensor> partials = {make_partial(3, 2)};
    partials[0].block_sizes.resize(1);  // short (wants n_block=2)
    const std::vector<DeviceShard> shards = {shard(0, 2)};
    return throws(partials, shards, 3, 2);
}

// --- MALFORMED: shards do not tile [0, n_block_full) → throw ----------------
// Spans cover 4 blocks but n_block_full is 5 (a gap/short cover).
[[nodiscard]] bool test_malformed_no_tile() {
    const std::vector<F2BlockTensor> partials = {make_partial(2, 2), make_partial(2, 2)};
    const std::vector<DeviceShard> shards = {shard(0, 2), shard(2, 4)};  // cover 4
    return throws(partials, shards, 2, 5);                                // claim 5
}

struct Case {
    const char* name;
    bool (*fn)();
};

constexpr Case kCases[] = {
    {"VALID: single shard covering all blocks", test_valid_single_shard},
    {"VALID: two shards tiling [0,5)", test_valid_two_shard_tiling},
    {"VALID: trailing empty shard (n_block<G)", test_valid_trailing_empty},
    {"VALID: all-empty (n_block_full==0)", test_valid_all_empty},
    {"MALFORMED: partials/shards count mismatch -> throws", test_malformed_count_mismatch},
    {"MALFORMED: negative P or n_block_full -> throws", test_malformed_negative},
    {"MALFORMED: partial n_block != shard span -> throws", test_malformed_span_mismatch},
    {"MALFORMED: non-empty partial P disagrees -> throws", test_malformed_p_disagree},
    {"MALFORMED: short f2 storage (the B5/C1 gap) -> throws", test_malformed_short_f2},
    {"MALFORMED: short vpair storage -> throws", test_malformed_short_vpair},
    {"MALFORMED: short block_sizes storage -> throws", test_malformed_short_block_sizes},
    {"MALFORMED: shards do not tile [0,n_block_full) -> throws", test_malformed_no_tile},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(F2PartialsValidate, ValidAndMalformedCases) {
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
        std::fprintf(stderr, "test_f2_partials_validate: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_f2_partials_validate: all checks PASS\n");
    return 0;
}
#endif

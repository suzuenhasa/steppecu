// tests/unit/test_f2_combine.cpp
//
// Host-only unit test of the HOST-STAGED fixed-order f2 combine
// core::combine_f2_partials_host (cleanup B7 / f2_combine N2+P2+T1;
// architecture.md §11.4 SPMG host-side fixed-order combine, §12 parity, §13). Pure
// C++ TU, NO GPU, NO io, NO data.
//
// combine_f2_partials_host is the PORTABLE PARITY BASELINE of the single-node
// multi-GPU precompute and the ONLY combine path on the budget box (no peer access).
// Its whole point is to be host-testable (the header asserts "unit-testable
// host-only"), yet it was exercised ONLY through the GPU-required end-to-end parity
// test (tests/reference/test_f2_multigpu_parity.cu). This test pins the unit's
// contract directly, GPU-free, in microseconds — the fast inner-loop gate for the
// slow GPU parity test.
//
// It pins:
//   1. PLACEMENT — each device's compact partial lands at its block offset
//      shards[g].b0 in the full [P × P × n_block_full] tensor, BIT-FOR-BIT, with
//      f2, vpair, AND block_sizes all placed (the contiguous std::copy_n per device,
//      B7/P2);
//   2. NON-OWNED slabs stay the +0.0 init (no device writes them; the disjoint
//      tiling proves the real path leaves none, but a SINGLE-block layout exercises
//      a strict subset of blocks owned -> the rest +0.0 by construction is moot,
//      so we instead assert the full owned cover equals a hand-built reference);
//   3. THE −0.0 CASE (B7 / f2_combine N2 — THE gate for this fix): a partial whose
//      f2 contains an exact −0.0 must land as −0.0 (bit-for-bit, 0x8000…0) in the
//      combined tensor. The OLD `out += part` onto a +0.0 accumulator FLIPPED it to
//      +0.0 (IEEE-754 (+0.0)+(−0.0)==+0.0); the std::copy_n placement reproduces it.
//      This case FAILS the old body and PASSES the fixed body — the parity-hardening
//      proof that the combine is now bit-identical to a single-GPU run that itself
//      computes a −0.0 slab directly.
//   4. FIXED g=0..G-1 ORDER over a two-shard tiling — each device's run lands at the
//      right global offset regardless of g (the §12 fixed-order law's placement);
//   5. DEGENERATES — all-empty (n_block_full==0 -> empty tensor) and a trailing
//      EMPTY shard (device 1 owns no blocks -> contributes nothing, skipped);
//   6. FAIL-FAST — a malformed input (count mismatch) still throws std::runtime_error
//      via the shared validate_f2_partials guard (the guard's own contract is
//      covered exhaustively in test_f2_partials_validate; here we just confirm the
//      combine routes through it before touching memory).
//
// These cases assert no statistic — they exercise the PLACEMENT / bit-faithfulness /
// control flow of the combine on hand-built layouts (ROADMAP §0: no synthetic data
// for PRECISION claims; this is bit-level placement + control flow, not a precision
// claim, and the −0.0 case is a bit-pattern fidelity claim).
//
// Dual harness (identical to tests/unit/test_f2_partials_validate.cpp): with
// -DSTEPPE_TEST_WITH_GTEST it uses GoogleTest; otherwise it is a self-checking
// main() returning non-zero on the first failure. No CUDA here — the CUDA-free host
// compile is itself the §4-layering proof that the combine names no CUDA type.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/fstats/f2_combine.hpp"  // combine_f2_partials_host
#include "steppe/fstats.hpp"           // F2BlockTensor
#include "device/shard_plan.hpp"       // DeviceShard

namespace {

using steppe::F2BlockTensor;
using steppe::core::combine_f2_partials_host;
using steppe::device::DeviceShard;

[[nodiscard]] DeviceShard shard(int b0, int b1) {
    DeviceShard s;
    s.b0 = b0;
    s.b1 = b1;
    return s;
}

// Build a WELL-FORMED compact partial owning n_block blocks at population count P,
// with f2/vpair filled by `gen(global_block, element)` so each placed value is
// uniquely identifiable at its destination. `b0` is the device's global block
// offset (so the generator can be keyed on the GLOBAL block id and the test can
// reconstruct the expected full tensor independently).
template <class Gen>
[[nodiscard]] F2BlockTensor make_partial(int P, int b0, int n_block, Gen gen) {
    F2BlockTensor t;
    t.P = n_block > 0 ? P : 0;
    t.n_block = n_block;
    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    const std::size_t total =
        slab * static_cast<std::size_t>(n_block > 0 ? n_block : 0);
    t.f2.resize(total);
    t.vpair.resize(total);
    t.block_sizes.resize(static_cast<std::size_t>(n_block > 0 ? n_block : 0));
    for (int lb = 0; lb < n_block; ++lb) {
        const int gb = b0 + lb;  // global block id
        for (std::size_t e = 0; e < slab; ++e) {
            t.f2[slab * static_cast<std::size_t>(lb) + e] = gen(gb, e);
            t.vpair[slab * static_cast<std::size_t>(lb) + e] =
                gen(gb, e) + 1000.0;  // distinct stream so a swapped f2/vpair shows
        }
        t.block_sizes[static_cast<std::size_t>(lb)] = 100 + gb;  // unique per block
    }
    return t;
}

// Bit-exact double comparison (catches the −0.0 vs +0.0 flip that == would miss).
[[nodiscard]] bool bits_equal(double a, double b) {
    std::uint64_t ua, ub;
    std::memcpy(&ua, &a, sizeof ua);
    std::memcpy(&ub, &b, sizeof ub);
    return ua == ub;
}

// --- 1+2: PLACEMENT of two disjoint shards over the full owned cover, bit-exact ---
[[nodiscard]] bool test_placement_two_shards() {
    constexpr int P = 3;
    constexpr int Nfull = 5;
    auto gen = [](int gb, std::size_t e) {
        return static_cast<double>(gb) * 1.5 + static_cast<double>(e) * 0.25;
    };
    // device 0 owns blocks [0,2); device 1 owns [2,5)
    const std::vector<F2BlockTensor> partials = {
        make_partial(P, 0, 2, gen), make_partial(P, 2, 3, gen)};
    const std::vector<DeviceShard> shards = {shard(0, 2), shard(2, 5)};

    const F2BlockTensor out = combine_f2_partials_host(partials, shards, P, Nfull);
    if (out.P != P || out.n_block != Nfull) return false;

    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    for (int gb = 0; gb < Nfull; ++gb) {
        for (std::size_t e = 0; e < slab; ++e) {
            const std::size_t idx = slab * static_cast<std::size_t>(gb) + e;
            if (!bits_equal(out.f2[idx], gen(gb, e))) return false;
            if (!bits_equal(out.vpair[idx], gen(gb, e) + 1000.0)) return false;
        }
        if (out.block_sizes[static_cast<std::size_t>(gb)] != 100 + gb) return false;
    }
    return true;
}

// --- 3: THE −0.0 CASE (cleanup B7 / N2 gate) -------------------------------------
// A partial element that is an exact −0.0 must land as −0.0 (0x8000000000000000),
// NOT +0.0. This FAILS the old `out += part` body (which flips it) and PASSES the
// std::copy_n placement.
[[nodiscard]] bool test_negative_zero_preserved() {
    constexpr int P = 2;
    constexpr int Nfull = 1;
    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);

    F2BlockTensor part;
    part.P = P;
    part.n_block = 1;
    part.f2.assign(slab, 0.0);
    part.vpair.assign(slab, 0.0);
    part.block_sizes.assign(1, 7);
    part.f2[0] = -0.0;       // the load-bearing element
    part.f2[1] = 0.0;        // a +0.0 control
    part.f2[2] = -1.25;      // an ordinary negative control
    part.vpair[0] = -0.0;    // (cannot arise for a count, but pins the path)

    const std::vector<F2BlockTensor> partials = {part};
    const std::vector<DeviceShard> shards = {shard(0, 1)};
    const F2BlockTensor out = combine_f2_partials_host(partials, shards, P, Nfull);

    // out.f2[0] must be EXACTLY −0.0 (bit pattern 0x8000…0), distinct from +0.0.
    if (!bits_equal(out.f2[0], -0.0)) return false;          // the fix
    if (bits_equal(out.f2[0], 0.0)) return false;            // and NOT +0.0
    if (!bits_equal(out.f2[1], 0.0)) return false;           // +0.0 control intact
    if (!bits_equal(out.f2[2], -1.25)) return false;         // ordinary value intact
    if (!bits_equal(out.vpair[0], -0.0)) return false;       // vpair path too
    return true;
}

// --- 4: FIXED-ORDER placement, single full shard (G==1 floor) --------------------
[[nodiscard]] bool test_single_shard_full() {
    constexpr int P = 4;
    constexpr int Nfull = 3;
    auto gen = [](int gb, std::size_t e) {
        return static_cast<double>(gb) * 100.0 + static_cast<double>(e);
    };
    const std::vector<F2BlockTensor> partials = {make_partial(P, 0, 3, gen)};
    const std::vector<DeviceShard> shards = {shard(0, 3)};
    const F2BlockTensor out = combine_f2_partials_host(partials, shards, P, Nfull);

    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    for (int gb = 0; gb < Nfull; ++gb) {
        for (std::size_t e = 0; e < slab; ++e) {
            const std::size_t idx = slab * static_cast<std::size_t>(gb) + e;
            if (!bits_equal(out.f2[idx], gen(gb, e))) return false;
            if (!bits_equal(out.vpair[idx], gen(gb, e) + 1000.0)) return false;
        }
        if (out.block_sizes[static_cast<std::size_t>(gb)] != 100 + gb) return false;
    }
    return true;
}

// --- 5a: DEGENERATE all-empty (n_block_full == 0) -> empty tensor ----------------
[[nodiscard]] bool test_all_empty() {
    auto gen = [](int, std::size_t) { return 0.0; };
    const std::vector<F2BlockTensor> partials = {
        make_partial(5, 0, 0, gen), make_partial(5, 0, 0, gen)};
    const std::vector<DeviceShard> shards = {shard(0, 0), shard(0, 0)};
    const F2BlockTensor out = combine_f2_partials_host(partials, shards, 5, 0);
    return out.P == 5 && out.n_block == 0 && out.f2.empty() &&
           out.vpair.empty() && out.block_sizes.empty();
}

// --- 5b: DEGENERATE trailing EMPTY shard (device 1 owns nothing) -----------------
[[nodiscard]] bool test_trailing_empty_shard() {
    constexpr int P = 2;
    constexpr int Nfull = 2;
    auto gen = [](int gb, std::size_t e) {
        return static_cast<double>(gb) + static_cast<double>(e) * 0.5;
    };
    // device 0 owns [0,2); device 1 owns [2,2) == empty
    const std::vector<F2BlockTensor> partials = {
        make_partial(P, 0, 2, gen), make_partial(P, 2, 0, gen)};
    const std::vector<DeviceShard> shards = {shard(0, 2), shard(2, 2)};
    const F2BlockTensor out = combine_f2_partials_host(partials, shards, P, Nfull);
    if (out.n_block != Nfull) return false;

    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    for (int gb = 0; gb < Nfull; ++gb) {
        for (std::size_t e = 0; e < slab; ++e) {
            const std::size_t idx = slab * static_cast<std::size_t>(gb) + e;
            if (!bits_equal(out.f2[idx], gen(gb, e))) return false;
        }
    }
    return true;
}

// --- 6: FAIL-FAST through the shared validator (count mismatch) ------------------
[[nodiscard]] bool test_fail_fast_count_mismatch() {
    auto gen = [](int, std::size_t) { return 1.0; };
    const std::vector<F2BlockTensor> partials = {
        make_partial(2, 0, 2, gen), make_partial(2, 2, 2, gen)};
    const std::vector<DeviceShard> shards = {shard(0, 4)};  // 1 shard, 2 partials
    try {
        (void)combine_f2_partials_host(partials, shards, 2, 4);
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

struct Case {
    const char* name;
    bool (*fn)();
};

const Case kCases[] = {
    {"PLACEMENT: two disjoint shards, full cover bit-exact", test_placement_two_shards},
    {"N2: a -0.0 partial element lands as -0.0 (the B7 gate)", test_negative_zero_preserved},
    {"PLACEMENT: single full shard (G==1 floor) bit-exact", test_single_shard_full},
    {"DEGENERATE: all-empty (n_block_full==0) -> empty tensor", test_all_empty},
    {"DEGENERATE: trailing empty shard (device owns nothing)", test_trailing_empty_shard},
    {"FAIL-FAST: partials/shards count mismatch -> throws", test_fail_fast_count_mismatch},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(F2Combine, PlacementFixedOrderAndNegativeZero) {
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
        std::fprintf(stderr, "test_f2_combine: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_f2_combine: all checks PASS\n");
    return 0;
}
#endif

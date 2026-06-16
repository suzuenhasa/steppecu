// tests/unit/test_vram_budget.cpp
//
// Host-only unit test of the M4 per-block f2 VRAM budget helper
// (device/vram_budget.hpp; architecture.md §11.1/§11.2, §13; ROADMAP §4; cleanup
// X-5/B5 + X-13/B26). Pure C++ TU, NO GPU — the budget is plain std::size_t
// arithmetic, extracted out of the .cu precisely so it is unit-testable GPU-free.
//
// It pins the three things B5/B26 require, against the architecture.md §11.2
// contract the helper models:
//   1. resident_tensor_bytes counts BOTH P²·n_block FP64 tensors (f2 AND vpair),
//      i.e. it is EXACTLY 2× the single-tensor footprint — the prior path counted
//      one and under-budgeted by ~2× (B26);
//   2. chunk_budget_bytes subtracts BOTH resident tensors AND the cuBLAS workspace
//      (kCublasWorkspaceBytes) from free VRAM BEFORE applying the fraction, and
//      the result NEVER exceeds kMaxVramUtilizationFraction of the net-of-reserve
//      free VRAM (B5 + the X-5 "subtract the workspace" nuance);
//   3. max_blocks_per_chunk clamps in size_t BEFORE the int narrowing — a huge
//      free figure cannot wrap negative (the X-5/F9 trap), it never returns more
//      blocks than the bucket has, it is floored at 1, and one chunk's slabs never
//      exceed the fraction-of-net budget.
//
// These cases assert no statistic — they exercise the CONTROL FLOW and the budget
// CONTRACT (ROADMAP §0: this is integer policy logic, not a precision claim).
//
// Dual harness (identical to tests/unit/test_block_ranges.cpp): with
// -DSTEPPE_TEST_WITH_GTEST it uses GoogleTest; otherwise it is a self-checking
// main() returning non-zero on the first failure. No CUDA here.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "device/vram_budget.hpp"  // resident_tensor_bytes, per_block_chunk_bytes, chunk_budget_bytes, max_blocks_per_chunk
#include "steppe/config.hpp"       // kMaxVramUtilizationFraction, kCublasWorkspaceBytes

namespace {

using steppe::device::chunk_budget_bytes;
using steppe::device::max_blocks_per_chunk;
using steppe::device::per_block_chunk_bytes;
using steppe::device::resident_tensor_bytes;
using steppe::kCublasWorkspaceBytes;
using steppe::kMaxVramUtilizationFraction;

constexpr std::size_t kDbl = sizeof(double);  // 8

// One f2/vpair tensor's bytes: P²·n_block·8. The helper must count TWO of these.
[[nodiscard]] std::size_t one_tensor_bytes(int P, int nb) {
    return static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
           static_cast<std::size_t>(nb) * kDbl;
}

// --- B26: resident_tensor_bytes counts BOTH tensors (exactly 2×) -------------
// The whole point of B26: f2 AND vpair are co-resident, each P²·n_block·8, so the
// resident footprint is 2× the single-tensor term. A helper that counts one
// under-budgets by 2× and OOMs mid-stream.
[[nodiscard]] bool test_resident_counts_both_tensors() {
    const int P = 768, nb = 757;  // the real-AADR M4 scale (ROADMAP §0)
    const std::size_t one = one_tensor_bytes(P, nb);
    const std::size_t two = resident_tensor_bytes(P, nb);
    // Exactly 2× one tensor — the f2 + vpair pair.
    if (two != 2u * one) return false;
    // Sanity at a second scale.
    const std::size_t one2 = one_tensor_bytes(4266, 757);  // §0 top end
    if (resident_tensor_bytes(4266, 757) != 2u * one2) return false;
    // Degenerate inputs are well-defined 0 (no wrap on negatives).
    if (resident_tensor_bytes(0, 757) != 0u) return false;
    if (resident_tensor_bytes(768, 0) != 0u) return false;
    if (resident_tensor_bytes(-1, 757) != 0u) return false;
    return true;
}

// --- B26: the 2× term does NOT overflow size_t at the §0 top end -------------
// At P=4266, n_block=757: one tensor ≈ 110 GB, the pair ≈ 220 GB — large, but
// the cast-before-multiply keeps it inside 64 bits (≈2.2e11 « 1.8e19), so the
// budget arithmetic stays sound rather than wrapping.
[[nodiscard]] bool test_resident_no_overflow_top_end() {
    const std::size_t two = resident_tensor_bytes(4266, 757);
    // ~220 GB, comfortably inside size_t; just assert it is the exact 2× value
    // and is far below SIZE_MAX (a wrap would make it tiny or absurd).
    const std::size_t one = one_tensor_bytes(4266, 757);
    return two == 2u * one && two > (200ull * 1024ull * 1024ull * 1024ull) &&
           two < std::numeric_limits<std::size_t>::max() / 2u;
}

// --- B5 + X-5: chunk_budget subtracts 2× tensors + workspace, THEN fraction --
// The budget the chunk slabs are sized against must be the utilization fraction
// of the free VRAM that REMAINS after the resident pair AND the cuBLAS workspace
// are reserved — never the gross free figure.
[[nodiscard]] bool test_chunk_budget_reserves_both_and_workspace() {
    const int P = 768, nb = 757;
    const std::size_t resident = resident_tensor_bytes(P, nb);  // 2× tensors
    const std::size_t free = resident + kCublasWorkspaceBytes +
                             (8ull * 1024ull * 1024ull * 1024ull);  // 8 GB headroom
    const std::size_t net = free - resident - kCublasWorkspaceBytes;
    const std::size_t expect =
        static_cast<std::size_t>(kMaxVramUtilizationFraction * static_cast<double>(net));
    const std::size_t got = chunk_budget_bytes(free, P, nb);
    if (got != expect) return false;
    // The budget must never exceed the fraction of the NET free (the B5 gate):
    // it must be <= fraction·net, and strictly < the gross free (we reserved).
    if (got > static_cast<std::size_t>(kMaxVramUtilizationFraction *
                                       static_cast<double>(net)))
        return false;
    if (got >= free) return false;  // never budgets all of free VRAM
    return true;
}

// --- B5: chunk_budget saturates to 0 when free can't cover the reserve -------
// If free VRAM cannot even hold the resident pair + workspace, the chunk budget
// is 0 (fail-fast: max_blocks then clamps to 1 and OOMs cleanly rather than
// silently sizing a chunk against a wrapped negative).
[[nodiscard]] bool test_chunk_budget_saturates_to_zero() {
    const int P = 768, nb = 757;
    const std::size_t resident = resident_tensor_bytes(P, nb);
    // free is LESS than the reserve → net would underflow; helper saturates to 0.
    if (chunk_budget_bytes(resident, P, nb) != 0u) return false;          // missing the workspace
    if (chunk_budget_bytes(resident + kCublasWorkspaceBytes - 1u, P, nb) != 0u)
        return false;                                                     // 1 byte short
    // EXACTLY the reserve → net 0 → budget 0.
    if (chunk_budget_bytes(resident + kCublasWorkspaceBytes, P, nb) != 0u) return false;
    return true;
}

// --- per_block_chunk_bytes: the 4·P·s_pad + 4·P² structural footprint --------
[[nodiscard]] bool test_per_block_bytes_formula() {
    const int P = 768, s_pad = 1024;
    const std::size_t p = static_cast<std::size_t>(P);
    const std::size_t sp = static_cast<std::size_t>(s_pad);
    const std::size_t expect = (4u * p * sp + 4u * p * p) * kDbl;
    return per_block_chunk_bytes(P, s_pad) == expect;
}

// --- B5/X-5/F9: max_blocks clamps in size_t BEFORE the int narrowing ---------
// A near-SIZE_MAX free figure makes the size_t quotient exceed INT_MAX; the OLD
// code cast that quotient to int (wrapping negative) THEN clamped to 1. The helper
// must clamp the quotient against nb_total IN size_t first, so the result is a
// sane value in [1, nb_total] — never a wrapped negative re-clamped to 1.
[[nodiscard]] bool test_max_blocks_no_int_wrap() {
    const int P = 8, nb = 64;          // tiny resident set
    const int s_pad = 8, nb_total = 50;
    // Enormous free VRAM: quotient budget/per_block » INT_MAX, but capped to
    // nb_total IN size_t before the narrow.
    const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2u;
    const int got = max_blocks_per_chunk(huge, P, nb, s_pad, nb_total);
    // Must be exactly nb_total (the bucket cap), NOT a wrapped/negative value
    // and NOT the degenerate 1 the int-wrap-then-clamp bug produced.
    return got == nb_total;
}

// --- max_blocks is floored at 1 and never exceeds nb_total -------------------
[[nodiscard]] bool test_max_blocks_floor_and_cap() {
    const int P = 768, nb = 757, s_pad = 2048;
    // Zero free → budget 0 → quotient 0 → floored to 1 (a single block always
    // attempted; it then OOMs cleanly if it truly cannot fit).
    if (max_blocks_per_chunk(0, P, nb, s_pad, 10) != 1) return false;
    // Free that fits exactly the reserve → net 0 → still floored to 1.
    const std::size_t reserve = resident_tensor_bytes(P, nb) + kCublasWorkspaceBytes;
    if (max_blocks_per_chunk(reserve, P, nb, s_pad, 10) != 1) return false;
    // Plenty of free but only 3 blocks in the bucket → capped at 3 (never > nb_total).
    const std::size_t plenty = reserve + (16ull * 1024ull * 1024ull * 1024ull);
    const int got = max_blocks_per_chunk(plenty, P, nb, s_pad, 3);
    if (got < 1 || got > 3) return false;
    // Empty bucket → 0 (nothing to chunk; the kernel never launches a 0-block grid).
    if (max_blocks_per_chunk(plenty, P, nb, s_pad, 0) != 0) return false;
    return true;
}

// --- The gate: one chunk's slabs NEVER exceed the fraction-of-net budget -----
// For a budget that admits a partial chunk (more blocks than fit), the chosen
// max_blocks·per_block must be <= chunk_budget_bytes (= fraction·net) — i.e. one
// resident chunk never overshoots the utilization fraction. (Floored-to-1 is the
// sole documented exception: a single block is always attempted.)
[[nodiscard]] bool test_chunk_never_exceeds_fraction() {
    const int P = 768, nb = 757, s_pad = 1024;
    const std::size_t resident = resident_tensor_bytes(P, nb);
    const std::size_t per_block = per_block_chunk_bytes(P, s_pad);
    const int nb_total = 4096;  // a large bucket
    // Make the NET budget exactly an integer number of blocks (avoid fraction
    // floor brittleness): set net so floor(fraction·net) == 20·per_block exactly.
    // free = resident + workspace + net, where net = 20·per_block / fraction.
    const std::size_t net =
        static_cast<std::size_t>(static_cast<double>(20u * per_block) /
                                 kMaxVramUtilizationFraction);
    const std::size_t free = resident + kCublasWorkspaceBytes + net;
    const std::size_t budget = chunk_budget_bytes(free, P, nb);  // ~= 20·per_block
    const int mb = max_blocks_per_chunk(free, P, nb, s_pad, nb_total);
    if (mb < 1) return false;
    if (mb >= nb_total) return false;  // must be a PARTIAL chunk (budget < bucket)
    // The load-bearing gate: the chosen chunk's slabs fit within fraction·net —
    // one resident chunk never overshoots the utilization fraction.
    if (static_cast<std::size_t>(mb) * per_block > budget) return false;
    // And it is the LARGEST such count: adding one more block would overshoot.
    if (static_cast<std::size_t>(mb + 1) * per_block <= budget) return false;
    return true;
}

struct Case {
    const char* name;
    bool (*fn)();
};

constexpr Case kCases[] = {
    {"B26: resident_tensor_bytes counts BOTH f2+vpair (exactly 2x)", test_resident_counts_both_tensors},
    {"B26: the 2x resident term does not overflow size_t at P=4266", test_resident_no_overflow_top_end},
    {"B5/X-5: chunk_budget reserves 2x tensors + workspace, then fraction", test_chunk_budget_reserves_both_and_workspace},
    {"B5: chunk_budget saturates to 0 when free < resident+workspace", test_chunk_budget_saturates_to_zero},
    {"per_block_chunk_bytes = (4*P*s_pad + 4*P^2)*8", test_per_block_bytes_formula},
    {"B5/X-5/F9: max_blocks clamps in size_t before the int narrowing", test_max_blocks_no_int_wrap},
    {"max_blocks floored at 1, capped at nb_total, 0 on empty bucket", test_max_blocks_floor_and_cap},
    {"GATE: one chunk's slabs never exceed fraction-of-net budget", test_chunk_never_exceeds_fraction},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(VramBudget, BudgetHelperContract) {
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
        std::fprintf(stderr, "test_vram_budget: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_vram_budget: all checks PASS\n");
    return 0;
}
#endif

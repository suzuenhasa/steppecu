// tests/unit/test_launch_config.cpp
//
// Host-only unit test of the single-source-home headers created by cleanup B7
// (architecture.md §2 DRY single-home, §4, §8 DRY-internals table, §13; ROADMAP
// §4, §5). Pure C++ TU, NO GPU: it proves the relocated launch-math helpers and
// the host/device + debug facilities compile and are correct on the host path
// (STEPPE_HD expands to empty under the host compiler), reached DIRECTLY through
// their new homes — not transitively through f2_estimator.hpp — so the move out
// of f2_estimator.hpp into core/internal/launch_config.hpp + host_device.hpp is
// pinned as a real single source, not a phantom (the X-4 finding B7 closes).
//
// Dual harness (build contract): with -DSTEPPE_TEST_WITH_GTEST it uses GoogleTest;
// otherwise it is a self-checking main() that returns non-zero on the first
// failure — all CTest needs to gate (tests/CMakeLists.txt). No CUDA here.
#include <cstdio>

#include "core/internal/host_device.hpp"    // STEPPE_HD, STEPPE_DEBUG_ONLY, STEPPE_ASSERT
#include "core/internal/launch_config.hpp"  // cdiv (int+long), grid_for, kDecodeBlockX/Y
#include "steppe/config.hpp"                // kCdivBlock

namespace {

using steppe::core::cdiv;
using steppe::core::grid_for;

// ---- cdiv: the int overload (the launch-grid building block) ----------------
[[nodiscard]] bool test_cdiv_int() {
    return cdiv(10, 4) == 3 && cdiv(8, 4) == 2 && cdiv(0, 4) == 0 && cdiv(1, 4) == 1;
}

// ---- cdiv: the long overload (the SNP/M axis can exceed 2^31) ---------------
// Exercise a value above INT_MAX so the long path is genuinely distinct from the
// int one (it must NOT silently narrow). 3_000_000_001 SNPs over a 32-wide block
// is ceil(3000000001/32) = 93750001. constexpr-folded, so this also proves the
// long overload is selected without ambiguity at the same-width call site.
[[nodiscard]] bool test_cdiv_long() {
    constexpr long m = 3'000'000'001L;
    return cdiv(m, 32L) == 93'750'001L && cdiv(0L, 32L) == 0L && cdiv(1L, 32L) == 1L;
}

// cdiv is STEPPE_HD constexpr, so its results fold at compile time — pin that
// (the strongest possible regression guard; matches the grid-helper contract).
static_assert(cdiv(10, 4) == 3, "cdiv(int) must constant-fold");
static_assert(cdiv(3'000'000'001L, 32L) == 93'750'001L, "cdiv(long) must constant-fold");

// ---- grid_for: the SQUARE-block f2 helper, defaulting to kCdivBlock ---------
[[nodiscard]] bool test_grid_for() {
    return grid_for(50) == cdiv(50, steppe::kCdivBlock) &&
           grid_for(steppe::kCdivBlock) == 1 &&
           grid_for(100, 32) == cdiv(100, 32);  // explicit block arg honored
}

// ---- decode block geometry: the named, warp-justified 32×8 home -------------
// The decode kernel's block dims now live in launch_config.hpp (not as bare
// literals re-picked in the kernel TU). x must be one full warp (32) for SNP-axis
// coalescing; 32×8 = 256 threads/block. Pin both, and the warp-alignment property.
[[nodiscard]] bool test_decode_block_geometry() {
    return steppe::core::kDecodeBlockX == 32 &&
           steppe::core::kDecodeBlockY == 8 &&
           (steppe::core::kDecodeBlockX % 32) == 0 &&  // warp-aligned SNP axis
           (steppe::core::kDecodeBlockX * steppe::core::kDecodeBlockY) == 256;
}
static_assert(steppe::core::kDecodeBlockX == 32 && steppe::core::kDecodeBlockY == 8,
              "decode block dims are the named 32x8 home");

// ---- STEPPE_HD: the one host/device qualifier compiles on the host ----------
// Under the host compiler STEPPE_HD expands to nothing; a STEPPE_HD function must
// therefore be an ordinary callable. (The whole point of the single home: the
// SAME annotated function compiles into host-pure TUs and into device kernels.)
[[nodiscard]] STEPPE_HD inline int hd_double(int x) noexcept { return 2 * x; }

[[nodiscard]] bool test_hd_qualifier() {
    return hd_double(21) == 42;
}

// ---- STEPPE_DEBUG_ONLY / STEPPE_ASSERT: the debug facility -------------------
// Both must be valid statements in BOTH build modes (NDEBUG and debug). We do not
// trip the assert (that would abort the test); we only prove the facility expands
// to a well-formed statement and that a true precondition passes. In NDEBUG the
// body is removed; in debug it runs and the side-effect below is visible — either
// way the function returns true, so the macros are exercised for compilation in
// the build mode CTest runs.
[[nodiscard]] bool test_debug_facilities() {
    int touched = 0;
    STEPPE_DEBUG_ONLY(touched = 1);  // expands to nothing under NDEBUG
    STEPPE_ASSERT(1 + 1 == 2, "arithmetic must hold");  // a true precondition
#if defined(NDEBUG)
    return touched == 0;  // release: the debug-only body was removed
#else
    return touched == 1;  // debug: the debug-only body ran
#endif
}

struct Case {
    const char* name;
    bool (*fn)();
};

constexpr Case kCases[] = {
    {"cdiv int ceiling division", test_cdiv_int},
    {"cdiv long overload (M > 2^31)", test_cdiv_long},
    {"grid_for == cdiv(n, block)", test_grid_for},
    {"decode block geometry (warp-justified 32x8)", test_decode_block_geometry},
    {"STEPPE_HD host-callable", test_hd_qualifier},
    {"STEPPE_DEBUG_ONLY / STEPPE_ASSERT facility", test_debug_facilities},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(LaunchConfig, SingleSourceHomes) {
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
        std::fprintf(stderr, "test_launch_config: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_launch_config: all %zu single-source-home checks PASS\n",
                sizeof(kCases) / sizeof(kCases[0]));
    return 0;
}
#endif

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
#include <cstddef>
#include <cstdio>

#include "core/internal/host_device.hpp"    // STEPPE_HD, STEPPE_DEBUG_ONLY, STEPPE_ASSERT
#include "core/internal/launch_config.hpp"  // cdiv (int+long), grid_for, grid_z_extent, kMaxGridX/Y/Z, kDecodeBlockX/Y
#include "device/vram_budget.hpp"           // max_blocks_per_chunk (the grid-z tiling cap, X-7/B6)
#include "steppe/config.hpp"                // kCdivBlock

namespace {

using steppe::core::cdiv;
using steppe::core::grid_for;
using steppe::core::grid_z_extent;
using steppe::core::kMaxGridX;
using steppe::core::kMaxGridY;
using steppe::core::kMaxGridZ;

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

// ====== Grid-dimension launch-failure fixes (cleanup X-7/B6) =================
// These pin the B6 fix: the y/z grid axes are hardware-capped at 65 535 and only
// x reaches 2^31−1, so the large (SNP/M-scale) extent MUST ride x, the batch axis
// (gridDim.z) must stay ≤ 65 535, and the M4 chunk loop tiles the batch so it does.
// All checks are build-mode-independent (they assert RETURN VALUES / constants, not
// the debug-only STEPPE_ASSERT side effect, which would abort rather than report).

// ---- the named hardware limits are the documented values --------------------
[[nodiscard]] bool test_grid_limit_constants() {
    return kMaxGridX == 2147483647u &&  // 2^31 − 1, the only axis past 65 535
           kMaxGridY == 65535u &&
           kMaxGridZ == 65535u;
}
static_assert(steppe::core::kMaxGridY == 65535u && steppe::core::kMaxGridZ == 65535u,
              "y/z grid axes are hardware-capped at 65 535 (architecture.md §7)");
static_assert(steppe::core::kMaxGridX == 2147483647u, "x grid axis reaches 2^31 − 1");

// ---- f2-feeder / decode RE-ORIENTATION: the large extent rides grid.x -------
// The bug B6 fixes: the f2 feeder put the SNP count M on gridDim.y (capped 65 535),
// failing at M > ~1.05M SNPs. The fix mirrors the safe decode launcher — M rides
// gridDim.x (cap 2^31−1) and P rides gridDim.y. Here we replicate EXACTLY the grid
// math both launch wrappers now perform (cdiv(M,long) for x; grid_for/cdiv(P) for y)
// at a SNP count whose y-axis extent would have BLOWN the 65 535 cap under the old
// orientation, and assert: (i) the M-derived extent lands on x and fits kMaxGridX;
// (ii) the P-derived extent lands on y and fits kMaxGridY; (iii) the M-derived
// extent really would have OVER-RUN the y cap (so the re-orientation is load-bearing,
// not cosmetic).
[[nodiscard]] bool test_feeder_orientation_large_M() {
    constexpr long kBigM = 1'050'000L;          // > 65 535·16 = 1 048 560 ⇒ trips y@16
    constexpr int  kP = 4266;                    // architecture.md §12 P_max

    // The f2 feeder (square kCdivBlock block): x = cdiv(M, kCdivBlock), y = grid_for(P).
    const long gx_feeder = cdiv(kBigM, static_cast<long>(steppe::kCdivBlock));
    const int  gy_feeder = grid_for(kP);  // == cdiv(P, kCdivBlock); asserts ≤ kMaxGridY

    // The SNP extent must fit the x cap and must NOT fit the y/z cap — the whole
    // point: it can only legally ride x.
    const bool x_holds = static_cast<unsigned long long>(gx_feeder) <=
                         static_cast<unsigned long long>(kMaxGridX);
    const bool would_overrun_y =
        static_cast<unsigned long long>(gx_feeder) > static_cast<unsigned long long>(kMaxGridY);
    const bool y_holds = gy_feeder >= 0 &&
                         static_cast<unsigned>(gy_feeder) <= kMaxGridY;

    // The decode kernel (non-square 32×8 block): x = cdiv(M, kDecodeBlockX), y =
    // grid_for(P, kDecodeBlockY). Same orientation, same invariants.
    const long gx_decode = cdiv(kBigM, static_cast<long>(steppe::core::kDecodeBlockX));
    const int  gy_decode = grid_for(kP, steppe::core::kDecodeBlockY);
    const bool decode_x_holds = static_cast<unsigned long long>(gx_decode) <=
                                static_cast<unsigned long long>(kMaxGridX);
    const bool decode_y_holds = gy_decode >= 0 &&
                                static_cast<unsigned>(gy_decode) <= kMaxGridY;

    return x_holds && would_overrun_y && y_holds && decode_x_holds && decode_y_holds;
}

// ---- grid_for keeps a square-block axis within its cap ----------------------
// For every in-scope P (≤ P_max), grid_for(P) is the y-axis extent of the f2
// kernels and must stay ≤ kMaxGridY. Sweep a range incl. P_max.
[[nodiscard]] bool test_grid_for_within_y_cap() {
    for (int P = 0; P <= 5000; P += 137) {
        const int g = grid_for(P);  // square kCdivBlock edge
        if (g < 0 || static_cast<unsigned>(g) > kMaxGridY) return false;
    }
    // exactly at the kCdivBlock·kMaxGridY boundary, grid_for == kMaxGridY (still OK)
    const long boundary = static_cast<long>(steppe::kCdivBlock) * kMaxGridY;  // 16·65535
    return cdiv(boundary, static_cast<long>(steppe::kCdivBlock)) ==
           static_cast<long>(kMaxGridY);
}

// ---- grid_z_extent: the DIRECT batch-axis guard (bypasses grid_for) ---------
// The M4 gather/scatter set gridDim.z = n_in_group directly, so they route through
// grid_z_extent, not grid_for. Within [1, kMaxGridZ] it returns the extent verbatim;
// the out-of-range cases (0, > kMaxGridZ) are guarded by the debug STEPPE_ASSERT and
// are not exercised here (an assert aborts; we test the legal-input return + that
// the legal domain reaches the cap exactly).
[[nodiscard]] bool test_grid_z_extent_legal_domain() {
    return grid_z_extent(1) == 1u &&
           grid_z_extent(40) == 40u &&                 // the spike's 40-block bucket
           grid_z_extent(static_cast<int>(kMaxGridZ)) == kMaxGridZ;  // exactly the cap
}

// ---- max_blocks_per_chunk TILES the batch so gridDim.z ≤ kMaxGridZ ----------
// The fix that makes grid_z_extent's precondition always hold: the per-bucket chunk
// budget caps blocks-per-chunk at kMaxGridZ, so each chunk's n_in_group (= gridDim.z)
// never exceeds the hardware z limit, even when VRAM would permit far more (the
// high-VRAM PRO-6000 tier where the latent failure lives). We hand it an
// effectively-unbounded free VRAM and a tiny per-block footprint and assert the
// result is capped at kMaxGridZ — not the (much larger) VRAM/nb_total figure.
[[nodiscard]] bool test_chunk_tiling_caps_grid_z() {
    using steppe::device::max_blocks_per_chunk;
    constexpr std::size_t kHugeFree = static_cast<std::size_t>(1) << 60;  // ~1 EiB free
    const int P = 2;            // tiny [2×2] slabs ⇒ huge per-block fit
    const int n_block = 200000; // > kMaxGridZ blocks in the bucket
    const int s_pad = 1;
    const int nb_total = 200000;
    const int mb = max_blocks_per_chunk(kHugeFree, P, n_block, s_pad, nb_total);
    // VRAM + nb_total would both permit > 65 535; the grid-z cap must bind.
    if (static_cast<unsigned>(mb) != kMaxGridZ) return false;
    // grid_z_extent must then accept exactly that chunk size (the invariant holds).
    return grid_z_extent(mb) == kMaxGridZ;
}

// ---- a small bucket is unaffected (the cap only bites past kMaxGridZ) --------
[[nodiscard]] bool test_chunk_tiling_small_bucket_unaffected() {
    using steppe::device::max_blocks_per_chunk;
    constexpr std::size_t kHugeFree = static_cast<std::size_t>(1) << 60;
    const int mb = max_blocks_per_chunk(kHugeFree, 768, 748, 2048, 40);  // spike shape
    return mb == 40 && static_cast<unsigned>(mb) <= kMaxGridZ;  // bucket fits, no tiling
}

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
    // --- B6: grid-dimension launch-failure fixes (X-7) ---
    {"grid limit constants (x=2^31-1, y/z=65535)", test_grid_limit_constants},
    {"feeder/decode re-orientation: large M rides grid.x", test_feeder_orientation_large_M},
    {"grid_for stays within the y cap for all in-scope P", test_grid_for_within_y_cap},
    {"grid_z_extent legal-domain identity (1..kMaxGridZ)", test_grid_z_extent_legal_domain},
    {"chunk tiling caps gridDim.z at kMaxGridZ", test_chunk_tiling_caps_grid_z},
    {"small bucket unaffected by the grid-z cap", test_chunk_tiling_small_bucket_unaffected},
    // --- B7 single-source-home facilities (unchanged) ---
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

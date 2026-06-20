// tests/unit/test_model_search_core.cpp
//
// HOST-PURE, GPU-FREE, DATA-FREE unit test of the M(fit-6) S8 ROTATION shard planner
// steppe::core::qpadm::plan_model_shards (src/core/qpadm/model_search_core.{hpp,cpp})
// + the deterministic pre-sized-slot RE-SORT discipline run_qpadm_search relies on.
// NO GPU, NO CUDA, NO io, NO real data: plan_model_shards is a pure function of
// (n_models, G), so this exercises the tiling + the re-sort in isolation (the proven
// test_f2_blocks_multigpu / test_shard_plan pattern), the GPU-free coverage gate that
// makes the shard logic safe to change without re-running the slow GPU rotation test.
//
// THE CONTRACT UNDER TEST (the FROZEN CONTRACT §1.2): plan_model_shards tiles
// [0, n_models) into G contiguous, non-overlapping, count-balanced ranges in
// g=0..G-1 order (the first n%G devices get ceil(n/G), the rest floor(n/G)); and the
// re-sort is IMPLICIT — each model is written into results[model_index], a vector
// pre-sized to n BEFORE any worker starts, so the returned order is identical for any
// G (the determinism gate). Below, a FAKE per-model "fit" (model_index -> a stamped
// value) is sharded and scattered into pre-sized slots, and the result is asserted
// identical across G — a GPU-free proof of the determinism gate's mechanism.
//
// Self-checking main() (CTest gates on the exit code).
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "core/qpadm/model_search_core.hpp"  // plan_model_shards, ModelShard

namespace sq = steppe::core::qpadm;

namespace {

int g_failures = 0;
void want(const char* what, bool cond) {
    if (!cond) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}

// Assert plan_model_shards(n, G) tiles [0, n) fully, contiguously, non-overlapping,
// in g=0..G-1 order, count-balanced (first n%G get ceil, rest floor).
void check_tiling(std::size_t n, std::size_t G) {
    const std::vector<sq::ModelShard> sh = sq::plan_model_shards(n, G);
    char nm[160];
    std::snprintf(nm, sizeof(nm), "n=%zu G=%zu: shard count == G", n, G);
    want(nm, sh.size() == G);
    if (sh.size() != G) return;

    const std::size_t base = n / G, rem = n % G;
    std::size_t cursor = 0;
    std::vector<int> covered(n, 0);
    for (std::size_t g = 0; g < G; ++g) {
        std::snprintf(nm, sizeof(nm), "n=%zu G=%zu: shard[%zu].g == g", n, G, g);
        want(nm, sh[g].g == static_cast<int>(g));
        std::snprintf(nm, sizeof(nm), "n=%zu G=%zu: shard[%zu] contiguous (lo==cursor)", n, G, g);
        want(nm, sh[g].lo == cursor);
        const std::size_t expect_count = base + (g < rem ? 1 : 0);
        std::snprintf(nm, sizeof(nm), "n=%zu G=%zu: shard[%zu] count-balanced", n, G, g);
        want(nm, (sh[g].hi - sh[g].lo) == expect_count);
        want("shard lo<=hi", sh[g].lo <= sh[g].hi);
        for (std::size_t i = sh[g].lo; i < sh[g].hi; ++i) {
            if (i < n) covered[i] += 1;  // mark coverage
        }
        cursor = sh[g].hi;
    }
    std::snprintf(nm, sizeof(nm), "n=%zu G=%zu: full cover (last hi == n)", n, G);
    want(nm, cursor == n);
    bool all_once = true;
    for (std::size_t i = 0; i < n; ++i) if (covered[i] != 1) all_once = false;
    std::snprintf(nm, sizeof(nm), "n=%zu G=%zu: every model covered EXACTLY once", n, G);
    want(nm, all_once);
}

// Simulate the pre-sized-slot RE-SORT: a fake per-model "fit" stamps model_index->
// 1000+model_index. Shard across G, scatter each shard's results into a vector
// pre-sized to n at results[model_index]. The result must be identical for any G.
std::vector<int> simulate_search(std::size_t n, std::size_t G) {
    const std::vector<sq::ModelShard> sh = sq::plan_model_shards(n, G);
    std::vector<int> results(n, -1);  // pre-sized; each slot written once
    for (const sq::ModelShard& s : sh) {
        for (std::size_t i = s.lo; i < s.hi; ++i) {
            // The "fit" of model i yields a deterministic stamp; written to slot i
            // (== model_index) — the implicit re-sort.
            results[i] = 1000 + static_cast<int>(i);
        }
    }
    return results;
}

}  // namespace

int main() {
    std::printf("=== M(fit-6) S8 plan_model_shards + re-sort unit test (GPU-free) ===\n");

    // ---- tiling across a spread of (n, G), incl. n<G (empty shards) and n%G!=0 ----
    const std::size_t Ns[] = {0, 1, 2, 3, 5, 7, 8, 84, 100, 1000, 2520};
    const std::size_t Gs[] = {1, 2, 3, 4, 8};
    for (std::size_t n : Ns)
        for (std::size_t G : Gs)
            check_tiling(n, G);

    // n < G: the trailing (G-n) devices draw EMPTY shards (lo==hi), still valid.
    {
        const std::vector<sq::ModelShard> sh = sq::plan_model_shards(3, 5);
        want("n<G: 5 shards", sh.size() == 5);
        int empty = 0; for (const auto& s : sh) if (s.lo == s.hi) ++empty;
        want("n<G: exactly (G-n)=2 empty shards", empty == 2);
    }

    // ---- G==0 fails fast (the contract: G >= 1) ----
    {
        bool threw = false;
        try { (void)sq::plan_model_shards(4, 0); } catch (const std::exception&) { threw = true; }
        want("G==0 throws (fail-fast)", threw);
    }

    // ---- the determinism gate's MECHANISM: simulate_search identical across G ----
    {
        const std::size_t n = 84;
        const std::vector<int> g1 = simulate_search(n, 1);
        const std::vector<int> g2 = simulate_search(n, 2);
        const std::vector<int> g4 = simulate_search(n, 4);
        bool ident = (g1.size() == n) && (g1 == g2) && (g1 == g4);
        want("re-sort: results identical & identically ordered for G=1,2,4", ident);
        // and correctly stamped (slot i holds the i-th model's fit).
        bool stamped = true;
        for (std::size_t i = 0; i < n; ++i) if (g1[i] != 1000 + static_cast<int>(i)) stamped = false;
        want("re-sort: every slot holds its own model's fit (no scramble)", stamped);
    }

    std::printf("=== RESULT: %s (%d failures) ===\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

# include__steppe__qpgraph_search
Files: /home/suzunik/steppe/include/steppe/qpgraph_search.hpp, /home/suzunik/steppe/src/core/qpadm/qpgraph_search.cpp
Subsystem: core-qpadm

## Findings

### G3
- [G3.dead][LOW] src/core/qpadm/qpgraph_search.cpp:207,213,217 — `second_arena` is tracked through the argmin loop (`second_arena = best_arena`, `second_arena = a`) but never read; it is explicitly discarded with `(void)second_arena;` at line 217. Only `second_s` is consumed (line 226). The variable and its two assignments in the loop are dead. Suggested: drop `second_arena` and its updates, keep only `second_s`.
- [G3.dead][LOW] src/core/qpadm/qpgraph_search.cpp:320 — `(void)score_cache;` is a redundant no-op suppression: `score_cache` IS used (captured-by-ref and mutated in the `score_of` lambda at lines 264-281, and seeded at 259-262). The `(void)` cast is misleading leftover. Suggested: remove the line.

### G5
- [G5.magic][MED] src/core/qpadm/qpgraph_search.cpp:299 — the hill-climb step cap `1000` is an unnamed magic literal buried in the loop header (`for (int step = 0; step < 1000; ++step)`). It is a tunable safety bound on best-improvement descent. Suggested: name it (e.g. `constexpr int kMaxHillClimbSteps = 1000;`).
- [G5.magic][LOW] src/core/qpadm/qpgraph_search.cpp:316 — `rtol = 1e-6, atol = 1e-9` recovery tolerances are inline literals. The header comment (hpp:11, "rtol ~1e-6") references the same value; the 1e-6 fit tolerance lives in several places across the qpgraph chain — risk of drift. Suggested: hoist to named constexpr (or source rtol from the shared fit-tolerance constant).

### G6
- [G6.naming][LOW] src/core/qpadm/qpgraph_search.cpp:78,82,84 — `ncol_f2` / `ncc` ("non-base column count") are cryptic abbreviations; `bb` (line 80) as the inner pair loop var shadows the conceptual "b" while `b` is also the `CanonicalBasis` instance name in `build_canonical_basis` (line 63). Local to a tight pair loop so impact is low. Suggested: rename `bb`->`b2` or the basis instance to avoid the visual collision.

### G8
- [G8.comment][LOW] src/core/qpadm/qpgraph_search.cpp:188,229 — comments label the per-candidate vector and best_fit as "additive exposure; NOT new compute" / "REUSES the fleet seam", but the best-fit block (lines 230-248) issues a SECOND backend call `qpgraph_fit_fleet(a, ...)` (line 232-233) for the single best topology — that IS an extra fit launch beyond the batch, not a pure reduction. The "NOT new compute" framing is slightly stale relative to the second launch. Suggested: clarify the comment that the best-fit is a separate (cheap, single-topology) re-fit to obtain the full result struct.

### G9
- [G9.const][LOW] src/core/qpadm/qpgraph_search.cpp:316 — `const double rtol = 1e-6, atol = 1e-9;` are `const` but should be `constexpr` (compile-time literals, no runtime dependency). Minor. Suggested: make `constexpr`.

### G4 / G7 / G10
- No issues. Index widening is correct: the load-bearing span sizes use `static_cast<std::size_t>(npair) * npair` (line 160) so the qinv P*P span is widened before multiply; `npair`/arena counts are bounded by the small enumerated topology space (oracle-C bounded pop-set), so `int` indexing of `arenas`/`cands` carries no P~2500/M~584k scale-overflow risk. No copy-paste blocks beyond the intentional single `make_canonical_arena` reuse in the heuristic fallback. No far-from-use or uninitialized declarations.

### G2
- N/A (no CUDA in this host-pure unit; G11-G22 not applicable).

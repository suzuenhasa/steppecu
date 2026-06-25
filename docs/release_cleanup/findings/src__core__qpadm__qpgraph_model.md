# src__core__qpadm__qpgraph_model
Files: /home/suzunik/steppe/src/core/qpadm/qpgraph_model.cpp, /home/suzunik/steppe/src/core/qpadm/qpgraph_model.hpp
Subsystem: core-qpadm

## Findings

### G3 dead/unused
- [G3.dead][LOW] qpgraph_model.hpp:28,30 — `#include <string>` appears twice (lines 28 and 30); the second is a redundant duplicate. Suggested: drop one.
- [G3.dead][LOW] qpgraph_model.cpp:13 — `#include <algorithm>` is unused (no std algorithm is called; the only "sort" in the file is in a comment at line 5). Suggested: remove.
- [G3.dead][LOW] qpgraph_model.cpp:16 — `#include <set>` is unused (only `std::map`, `std::unordered_map`, `std::vector`, `std::function` are used). Suggested: remove.

### G4 numeric / overflow
- [G4.overflow][LOW] qpgraph_model.cpp:98 — `m.npair = m.npop * (m.npop - 1) / 2` is plain `int` arithmetic. For qpGraph, `npop` is the GRAPH leaf count (small, not the P~2500 f2 axis), so this cannot overflow in practice; noting only because the multiply is not widened. No action needed unless graph leaf counts could ever approach ~46k. Suggested: none (documented as not-at-risk).

### G7 duplication / repeated work
- [G7.dup][LOW] qpgraph_model.hpp:144-154 — `fill_pwts_centered` dedups (edge,leaf) cells with an O(n_pe^2) double scan (the inner `for b < a` first-seen check at 147-148 plus the `for b = a..` sum at 151-152), re-walking `pe_edge`/`pe_leaf` for every cell. This is the host oracle / one-eval CudaBackend edge-recovery path (per the header comment at 124-127), so the path-edge table is small and it is not a hot loop, but the quadratic dedup is avoidable. Suggested: accumulate sums in a `std::map<pair<int,int>,double>` (mirrors the cpp's own `cell_cnt` pattern at 216-223) in a single pass.

### G8 comments
- [G8.stale][MED] qpgraph_model.hpp:39 (and cpp:90) — the struct doc claims a non-empty `error` is returned for "an unrooted / cyclic / malformed graph", but the only cyclicity check is the root-existence test (cpp:90), which only fires when EVERY node has a parent. A cycle that is reachable from a valid root (e.g. an admix node whose ancestor is also its descendant) is NOT detected here. The comment overstates the cycle coverage. Suggested: either tighten the parser (see G10 note below) or narrow the comment to "no-root cycle".

### G10 initialization / robustness
- [G10.robustness][MED] qpgraph_model.cpp:171-183 — the root->leaf DFS (`std::function dfs`) has no visited/recursion guard. If the input edge list contains a cycle reachable from the root (admix node with indeg==2 fed by its own descendant), `dfs` recurses indefinitely and stack-overflows; the indeg/outdeg/root checks (84-90) do not catch this case. Current callers (`qpgraph_enumerate`, `qpgraph_search`, `qpgraph_fit`) feed generated DAGs so it is not a parity bug, but a user-supplied malformed edge list would crash rather than return a clean `error`. Suggested: add a DFS color/visited guard (or a pre-pass topological-sort acyclicity check) that sets `m.error` and returns on a back-edge.

(groups checked: G2-G10; non-CUDA unit, G11-G22 N/A)

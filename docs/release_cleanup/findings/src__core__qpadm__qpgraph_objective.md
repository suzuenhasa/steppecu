# src__core__qpadm__qpgraph_objective
Files: /home/suzunik/steppe/src/core/qpadm/qpgraph_objective.hpp
Subsystem: core-qpadm

## Findings

### G5
- [G5.src__core__qpadm__qpgraph_objective][LOW] qpgraph_objective.hpp:58 — `const double eps = 1e-12;` is an unnamed tolerance magic number reused as both the gradient-free threshold (line 71/73) and the passive-drop / ratio-test floor (lines 94, 103, 106, 115). It conflates two distinct roles (KKT gradient tolerance vs. variable-magnitude floor) under one literal. Suggested: name them (e.g. `kKktTol`, `kZeroTol`) and/or split so the two semantics can drift independently.
- [G5.src__core__qpadm__qpgraph_objective][LOW] qpgraph_objective.hpp:61,78 — the active-set iteration caps `3 * n + 30` are duplicated bare literals appearing in both the outer and inner loops. Suggested: hoist to a single named `const int max_iter = 3 * n + 30;` to avoid the two drifting apart.

### G8
- [G8.src__core__qpadm__qpgraph_objective][LOW] qpgraph_objective.hpp:185-187 — the doc comment on `qpgraph_score` documents `out_bl`/`out_fit` as "Optionally returns bl + f3_fit" but omits the load-bearing precondition that `*out_fit` must be pre-sized to `npair` by the caller (it is written via `(*out_fit)[k]` at line 207, never `.assign`/`.resize`d, unlike `*out_bl` which is assigned at line 216). Callers (cpu_backend.cpp:2137, 2218) do pre-size it, so this is correct today, but the asymmetric contract is undocumented and a hazard for future callers. Suggested: note "caller must size `*out_fit` to npair" in the comment (or `assign` it inside for symmetry with `out_bl`).

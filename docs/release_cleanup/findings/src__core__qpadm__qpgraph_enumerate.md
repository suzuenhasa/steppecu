# src__core__qpadm__qpgraph_enumerate
Files: /home/suzunik/steppe/src/core/qpadm/qpgraph_enumerate.cpp, /home/suzunik/steppe/src/core/qpadm/qpgraph_enumerate.hpp
Subsystem: core-qpadm

## Findings

### G3 dead/unused
- [G3.dead][LOW] qpgraph_enumerate.cpp:302,352 — `base_tree_of(const IGraph& g, int nleaf)` never uses `nleaf`; it is silenced with `(void)nleaf;` at line 352. The parameter is dead. Suggested: drop the parameter (and its call sites at 484, 497, 498-relabel-path) or actually use it.
- [G3.dead][LOW] qpgraph_enumerate.cpp:497 — `nni_seen.insert(base_tree_of(cur, nleaf).empty() ? 0 : hash_igraph(base, nleaf));` recomputes a whole base tree just to test `.empty()`, then discards it and hashes the already-computed `base`. For any valid input the base tree is never empty (so the ternary's `0` branch is unreachable), and `base` is already in scope from line 484. The recompute is dead work and the guard is effectively never taken. Suggested: `nni_seen.insert(hash_igraph(base, nleaf));`.

### G5 magic numbers / duplicated constants
- [G5.magic][LOW] qpgraph_enumerate.cpp:238,384 — the fresh-node base ids `100000` (enumerate_admix1) and `500000` (admix1_children_of) are bare unnamed literals encoding the same concept ("an id base well above leaf ids"); two different magic values for one idea, drift-prone. Suggested: a single named `constexpr int kFreshNodeBase = ...` (the hash quotients labels out, so one shared base is safe).
- [G5.magic][LOW] qpgraph_enumerate.cpp:64 — `const int rounds = 2 * static_cast<int>(nodes.size()) + 4;` the `+4` is an unexplained magic slack constant for the WL refinement bound. Suggested: name it / add a one-line rationale for why `2*|V|+4` rounds suffice.

### G7 duplication
- [G7.dup][MED] qpgraph_enumerate.cpp:247-279 vs 385-407 — the admix-1 construction inner body (edge-set base build, the four endpoint filters `s.p==d.p || s.c==d.c || s.c==d.p || d.c==s.p`, the `reachable_from` acyclicity guard, the `insert_admix` 5-edge rewiring, the hash de-dup) is copy-pasted between `enumerate_admix1` and `admix1_children_of`. The comment at line 376 admits "Mirrors enumerate_admix1's inner body for a single tree." Two copies of the same AT2-parity-load-bearing logic will drift. Suggested: factor a single `admix1_children_of(tree, ...)` and have `enumerate_admix1` loop calling it (assigning `et.id` in the caller).
- [G7.dup][LOW] qpgraph_enumerate.cpp:127-129,247-255 — `reachable_from` rebuilds the full `adj` map on every call, and it is called inside the O(E^2) `(si,di)` double loop in `enumerate_admix1`/`admix1_children_of` where the input graph (`tree`) is loop-invariant. The adjacency is reconstructed for every candidate edge pair. Suggested: build the `tree` adjacency once outside the loops and pass it in (host-only, small n today, but it is a repeated loop-invariant rebuild).

### G8 comments
- [G8.rationale][LOW] qpgraph_enumerate.cpp:278 — `next_id += 2;  // advance after each source so split-node ids stay unique-ish;` "unique-ish" is vague and slightly misleading: within one source-edge iteration every `(si,di)` candidate reuses the SAME `x`/`a` ids, which is fine because each candidate graph is independent and the hash quotients internal labels out. The comment should state that reuse is intentional and harmless, not imply an attempt at global uniqueness. Suggested: clarify that internal-node ids need only be locally distinct per graph (hash-quotiented), so the advance is cosmetic.

### G9 constants & config
- [G9.const][LOW] qpgraph_enumerate.cpp:402,457 — neighbor topologies are emitted with `et.id = 0;`, but the header (hpp:43) documents `id` as "stable index within the nadmix level (enumeration order)." For neighbor moves `id` is always 0 and carries no enumeration-order meaning; a consumer trusting the doc could misread it. Suggested: document on `topology_neighbors` that `id` is not meaningful for neighbor results (or assign a running index).

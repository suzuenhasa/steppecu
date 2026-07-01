I read through this carefully. This is **not slop** — it's clearly written by someone who knows both the admixtools graph semantics and the graph-isomorphism problem, but a senior C++ developer would have mixed reactions. Some parts are genuinely excellent; others would raise eyebrows, mostly around performance and production-hardening.

## What's genuinely good

- **Bit-exact verification against admixtools 2.0.** The header comment documents file:line verification against `generate_all_graphs(5,1)` and an exact hash-set match against the committed fixture. That is a strong competence signal — it shows the author understands that "looks right" is not enough for an enumerator.
- **The 1-WL canonical hash is the right idea.** Using Weisfeiler-Lehman color refinement with leaf-anchored colors is a defensible isomorphism invariant for these graphs, and the comment honestly notes the exactness is proven only on the AT2 1485-set.
- **`admix1_children_of` is factored into a single shared helper.** Lines 209-211 and 361-392 show the same wiring used by both the global `enumerate_admix1` and the local `topology_neighbors`. That avoids the copy-paste drift these move-sets usually accumulate.
- **`base_tree_of` handles a genuinely subtle symmetric case.** The comment at lines 300-305 about the bug where both parents have out-degree 2 shows someone debugged this to completion, not just ported pseudocode.
- **Topology neighbor design is algorithmically thoughtful.** The comments at lines 453-461 explain *why* the move graph must be connected (monotone descent from any seed reaches the global optimum). That is the right level of reasoning for a neighborhood search.
- **Deterministic, no RNG.** The file comment is explicit about this, which matters for reproducibility in population-genetics tooling.

## What a senior developer would flag

**Performance: the hash path allocates like crazy.**

`hash_igraph` (lines 57-120) is the hot path for every enumerated graph and every neighbor. It builds:
- three `unordered_map`/set objects per call,
- string-based colors that are concatenated and re-allocated every round,
- an entire canonical string that is then FNV-1a'd.

For the small `nleaf` values used in qpGraph this is probably fine, but the implementation is `O(iterations × string churn)` rather than `O(iterations × integer vectors)`. A senior reviewer would flag this as the first place to profile if enumeration becomes a bottleneck.

**String-based color compression is heavy.**

```cpp
std::string s = color[u] + "|<";
for (const std::string& x : ins) { s += x; s += ','; }
```

Line 86-90 builds a new string for every node in every WL round. The canonical invariant only needs a comparable key; a small-vector integer encoding would be faster and avoid `std::string` entirely.

**Magic FNV constants are unnamed.**

```cpp
std::uint64_t h = 1469598103934665603ULL;
for (char ch : canon) { h ^= static_cast<unsigned char>(ch); h *= 1099511628211ULL; }
```

Lines 117-118 implement FNV-1a but leave the offset and prime bare. Name them `kFnvOffsetBasis` and `kFnvPrime`.

**`base_tree_of` takes a `[[maybe_unused]] int nleaf` it never touches.**

Line 285:
```cpp
IGraph base_tree_of(const IGraph& g, [[maybe_unused]] int nleaf) {
```

Either the parameter is needed for validation or it should be removed. Leaving it decorated with `[[maybe_unused]]` suggests a refactor that did not finish.

**Dead/paranoid branch in `base_tree_of`.**

Lines 306-319:
```cpp
if (a_par.size() == 2) { ... }
else if (!a_par.empty()) {
    x = a_par[0];
    dest_from = a_par.size() > 1 ? a_par[1] : -1;
}
```

The `else if` branch implies an admix node with in-degree not equal to 2, which contradicts the detector at lines 290-292 (`if (indeg[u] == 2)`). The `a_par.size() > 1` ternary inside it is dead code. This looks like defensive coding after a bug hunt, but it should be an assertion or removed.

**`reachable_from` rebuilds adjacency every call.**

Line 140-154 builds an `unordered_map<int, vector<int>>` from the edge list each time. `admix1_children_of` calls it inside a double loop over edges (line 375), so the same graph is re-scanned repeatedly. Pass a pre-built adjacency structure or use the existing `out_adj` if available.

**Root/outpop helpers are linear scans with silent failure modes.**

Line 123 `root_of` returns `-1` if every node has a parent (a cycle or malformed graph). Line 131 `outpop_of` returns `-1` if the root does not have exactly one leaf child. Neither is validated at the call sites, and there is no `assert` or error reporting. Production code would at least `STEPPE_ASSERT` these invariants.

**Hash collisions are silently accepted.**

`seen` is an `unordered_set<std::uint64_t>` keyed by `graph_hash`. The comment says 1-WL is exact on the 1485-set, but 64-bit FNV-1a has a non-zero collision probability as the enumerated space grows. For a correctness-critical deduplicator, a senior reviewer would ask for a tie-breaker or a formal argument that the space is small enough.

**`enumerate_trees_int` is copy-heavy.**

Lines 160-189 build each new generation of trees by copying and mutating the previous generation. For the expected `nleaf` this is fine, but the ` grown.push_back(std::move(nt))` pattern still materializes every intermediate tree. The author clearly knows move semantics, but the algorithmic memory pressure grows factorially.

**The `kFreshNodeBase = 500000` constant is explained but arbitrary.**

Line 45:
```cpp
inline constexpr int kFreshNodeBase = 500000;
```

The comment says the hash quotients these labels out, which is true, but the choice of 500000 is still a magic number. A symbolic name like `kSyntheticNodeBase` plus a comment about why it must not collide with leaf/internal ids would be cleaner.

## The "slop" test

**Not slop.** Slop is:
- Magic numbers without explanation
- Copy-pasted code with stale comments
- No error checking
- Obviously wrong algorithms that happen to pass tests

This has none of that. The comments are dense but accurate, the AT2 mappings are explicit, and the shared helper shows real care. The `500000` base and the FNV constants are the closest things to unexplained magic, and even those are documented in context.

## What it actually looks like

This looks like **high-quality research/engineering code written by a domain expert who understands admixtools graph topology and canonical graph hashing.** The algorithmic design is sound, the verification against a reference implementation is thorough, and the local-move connectivity reasoning is better than what most comp-bio tooling ships.

A senior C++ reviewer would say: "Correct and well-documented, but I would profile `hash_igraph` before calling this production-ready for large search spaces." A senior HPC person would say: "Stop allocating strings in the hot path and pre-build your adjacency." A population-genetics reviewer would say: "This is exactly the semantics we need, and the AT2 traceability is excellent."

The code could be tightened: replace string colors with integer color codes, name the FNV constants, assert invariants instead of silently returning `-1`, and remove the dead branch in `base_tree_of`. None of those are blockers; they are the difference between a solid research implementation and hardened production code.

**Verdict:** B+ to A-. Solid, correct, and well-explained, but needs a performance and invariant-hardening pass before it belongs on a hot path.

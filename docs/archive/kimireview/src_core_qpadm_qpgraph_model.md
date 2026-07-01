I read through `src/core/qpadm/qpgraph_model.cpp` carefully. This is **not slop** — it's careful, mathematically faithful work, but a senior C++ developer would flag several stylistic and idiomatic issues.

## What's genuinely good

- **The AT2 port is methodical and documented.** The comments about edge-order convention (`theta` vs `1-theta`), the verification note against admixtools 2.0.10, and the explanation of why the score is invariant to parent swap show real domain expertise and care (lines 3–10).
- **Cycle detection is explicit and safe.** The DFS uses an `on_stack` back-edge detector (lines 174–195) instead of assuming the input is a DAG. That's the right way to avoid infinite recursion on malformed graphs.
- **Error handling is non-fatal and centralized.** Every failure path returns `m.error = "..."` rather than throwing, crashing, or asserting (lines 47, 56, 79, 88, etc.). The contract is clear: the caller checks `error.empty()`.
- **Structure mirrors the reference algorithm.** The numbered sections (register nodes, degrees, leaf mapping, admixture edges, path enumeration, `pwts0`, `fill_pwts`) make it straightforward to compare against admixtools.

## What a senior developer would flag

**The `static_cast<std::size_t>` noise is relentless.**

```cpp
e_parent[static_cast<std::size_t>(e)] = reg.get(edges[static_cast<std::size_t>(e)].from);
```

Lines 54, 59, 60, 67, 70, 71, and dozens more repeat this pattern. It obscures the actual logic. Either index by `std::size_t` directly (range-based loops, `std::size_t e = 0`) or use signed `int` indices consistently and avoid the mixed-type dance. Right now it reads like defensive casting for a problem the code itself created.

**`std::function` for a local recursive lambda:**

```cpp
std::function<void(int)> dfs = [&](int node) { ... };
```

Line 176. `std::function` can heap-allocate and is overkill for a single-file helper. A named private free function, or a hand-rolled Y-combinator if you must keep it local, would be more idiomatic and avoid type erasure overhead.

**Partial-object error state:**

```cpp
QpGraphModel m;
if (edges.empty()) { m.error = "qpgraph: empty edge list"; return m; }
```

Line 46–47. Returning a half-initialized object and relying on `error` is fine, but modern C++ would prefer `std::expected<QpGraphModel, std::string>` or a dedicated `Result<T>` type. As the codebase grows, "check `.error` after every call" becomes easy to forget.

**Ordered map for an unordered problem:**

```cpp
std::map<std::pair<int, int>, int> cell_cnt;
```

Line 234. The cell counts don't need ordering. `std::unordered_map` (with a hash for `std::pair`) would be more idiomatic and avoid the tree overhead. Using `std::map` here is correct but looks like someone reaching for the first container that accepts a pair key.

**Signed/unsigned schizophrenia:**

Loop variables are `int` (line 53), but vector sizes and indices flip to `std::size_t` everywhere. Pick one. For graph indices, signed `int` is perfectly reasonable and makes `-1` sentinel values (line 35, line 99) natural. If so, size the vectors with `static_cast<int>` once and index with `int` thereafter.

**Minor nits:**
- `std::vector<char> on_stack` (line 174) works, but `std::vector<uint8_t>` or `std::vector<bool>` (despite its issues) more clearly signals a bitset intent.
- The sanity check at lines 272–275 is good, but the message "internal pair-count mismatch" is the kind of thing that should probably `assert` in debug and return in release, or at least include the actual vs expected numbers.
- The 1-based admixedge id convention (`2*j + 1`, `2*j + 2`, lines 143–144) is well commented, but the magic `+1`/`-1` offsets still invite off-by-one bugs for maintainers unfamiliar with AT2.

## The "slop" test

**Not slop.** Slop is uncommented magic numbers, copy-paste drift, no error handling, or algorithms that only work on happy paths. This file has none of that. The comments explain *why* decisions were made, the cycle detection is robust, and the AT2 conventions are explicitly acknowledged.

## What it actually looks like

This looks like **solid research/engineering code written by someone who understands the genomics algorithm better than they understand modern C++ idioms.** It's careful, verifiable, and easy to review for correctness, but it carries the stylistic habits of someone coming from Python/R/C — heavy casting, `std::function` convenience, and a "return an object with an error string" pattern. A senior C++ reviewer would trust the math but would wince at the verbosity.

## Verdict

**B+, ship after reducing cast noise and replacing `std::function`.** The core logic is sound and well-explained; the main issues are C++ craftsmanship, not correctness.
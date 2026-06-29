I read this carefully. It is **not slop** — it is clearly written by someone who understands both the admixture-graph problem and modern C++. A senior reviewer would find the overall design sound, but there are several places where the code is a little too comfortable with silent failures and implicit assumptions.

## What's genuinely good

- **The architecture is documented and well-motivated.** The file-level comment explains the "basis is pop-set-bound, NOT topology-bound" invariant, the single heterogeneous fleet launch, and the hill-climb recovery gate. That is real domain competence, not decoration.
- **Basis reuse is done correctly.** `build_canonical_basis` builds the canonical f3 pairs once, and `make_canonical_arena` remaps each topology's `cmb1`/`cmb2` into that order so the same `f_obs`/`qinv` serve every candidate (lines 73–137).
- **Template dispatch over f2 sources is clean.** `run_search_impl` handles both `DeviceF2Blocks` and `F2BlockTensor` without copy-pasting the orchestration logic (lines 144–335).
- **The recovery gate is falsifiable.** The hill-climb is not just "run a heuristic"; it is required to recover the exhaustive global best within a documented tolerance, and the result is exposed in `res.heuristic_recovered` (lines 265–331).
- **Modern C++ idioms dominate.** `std::span`, `std::chrono`, `std::unordered_map`, `std::vector`, move semantics, and `[[nodiscard]]` are all used appropriately. No raw `new`/`delete`, no `printf`, no `FILE*`.

## What a senior developer would flag

**`primary_backend` assumes GPU 0 always exists and is valid:**

```cpp
inline constexpr std::size_t kPrimaryGpu = 0;
[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}
```

This will throw `std::out_of_range` if the machine has no GPUs, and it dereferences `.backend` without a null check. The file claims to be "CUDA-FREE" host orchestration, but it still silently depends on a specific GPU resource layout.

**`build_canonical_basis` can produce silently malformed indices:**

```cpp
for (int i = 0; i < b.npop; ++i) {
    auto it = f2idx.find(pops[static_cast<std::size_t>(i)]);
    pop_f2[static_cast<std::size_t>(i)] = (it == f2idx.end()) ? -1 : it->second;
}
```

If a requested pop is not in `leaf_names`, the basis is built with `-1` f2 indices. There is no validation, no early return, and no diagnostic. The failure will surface later inside `assemble_f3_triples` or the fleet launch, making debugging harder.

**Candidate failures are swallowed with `continue`:**

```cpp
cq::QpGraphModel m = cq::parse_qpgraph(cands[i].edges, leaf_names, opts.pops.front());
if (!m.ok()) continue;
QpGraphTopoArena a;
if (!make_canonical_arena(m, basis, opts.fit, a)) continue;
```

A parse mismatch or structural failure gives the caller no hint *which* candidate failed or why. In a research/oracle context, silently dropping candidates is risky — it can make the "exhaustive" search non-exhaustive without warning.

**The `models` vector is over-allocated and default-constructs every candidate:**

```cpp
std::vector<cq::QpGraphModel> models(cands.size());
```

If there are thousands of candidates, this default-constructs a `QpGraphModel` for every one even though only the successfully-parsed subset is ever populated. Depending on the model struct's size, this can be a noticeable memory/initialization cost. A side map or `std::optional`/`std::vector<std::size_t>` would be cleaner.

**The single-graph refit of the global best does not propagate failure:**

```cpp
const QpGraphFleet fl =
    be.qpgraph_fit_fleet(a, f_obs, qinv, opts.fit.numstart, opts.fit.maxit, opts.fit.tol, prec);
res.best_fit.status = fl.status;
res.best_fit.score = fl.score;
```

If `fl.status != Status::Ok`, the overall `res.status` is still set to `Status::Ok` at line 333. The caller may think the search succeeded while the representative single-graph fit actually failed.

**The fallback path in `score_of` allocates and reparses for every miss:**

```cpp
const QpGraphFleetBatch one = be.qpgraph_fit_fleet_batch(
    std::vector<QpGraphTopoArena>{a}, f_obs, qinv, opts.fit.numstart, opts.fit.maxit,
    opts.fit.tol, prec);
```

Constructing `std::vector<QpGraphTopoArena>{a}` and re-parsing the model on every cache miss is a small performance footgun. The comment says misses "shouldn't happen," but when they do, this path is hotter than it needs to be.

**The `max_nadmix` clamp silently downgrades user input:**

```cpp
const int max_nadmix = std::min(opts.max_nadmix, 1);  // v1 bounded space.
```

If the caller asks for `max_nadmix = 2`, the search just runs with `1`. A senior reviewer would prefer `InvalidConfig` or at least a logged warning instead of quietly ignoring the request.

**Several readability/style nits:**

- Multiple member assignments on one line (line 128 and following) hurt readability.
- The comment at line 239 says the single-graph refit "REUSES the fleet seam," but it actually calls `qpgraph_fit_fleet`, not the batch seam. The wording is slightly stale.
- `kPrimaryGpu = 0` is a magic constant; in a multi-GPU codebase this should either be configurable or explicitly documented as a v1 limitation.

## The "slop" test

**Not slop.** Slop would be magic numbers without explanation, copy-pasted logic with stale comments, ignored error codes, or obviously wrong algorithms that happen to pass. This file has none of that. The comments are dense but accurate, and the design is coherent.

## What it actually looks like

This looks like **competent research-engineering host orchestration written by someone who knows the admixture-graph domain cold.** The C++ is modern enough to avoid the usual traps, and the GPU boundary is kept narrow and explicit. It is not a CUDA footgun because the GPU is reached only through two clean backend seams.

A senior C++ reviewer would say: "Solid structure, but tighten the error handling — stop silently swallowing parse failures and missing pops, and propagate the single-fit status properly." A senior systems reviewer would also ask for GPU-resource validation at the entry point rather than an unchecked dereference.

The one thing that might get side-eye is the **comment density**. The comments are mostly good — they explain invariants that are not visible in the code — but there are enough of them that a terse reviewer might wonder if the code is compensating for complexity.

## Verdict

**B+.** Strong design and clear domain expertise, held back by a handful of silent-failure paths and unchecked resource assumptions that a production review would block on.
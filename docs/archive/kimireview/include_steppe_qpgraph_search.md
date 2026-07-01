I read through this carefully. This is **not slop** — it's a thoughtful public-API header written by someone who understands both the domain (admixture-graph topology search) and modern C++ surface design. But a senior reviewer would have mixed reactions: the API contract is clean, yet the header carries too much implementation narrative and leaves some constraints implicit.

## What's genuinely good

- **The API surface is small and purposeful.** Two overloads, one options struct, one result struct, one candidate struct. No kitchen sink.
- **Forward declarations are used well.** `DeviceF2Blocks` and `Resources` are forward-declared in a nested `device` namespace (lines 34-37), avoiding a heavy include graph for a public header.
- **`[[nodiscard]]` on both entry points** (lines 118 and 124). The author treats a failed/incomplete search as a first-class result that callers must acknowledge.
- **Result struct carries rich diagnostic metadata:** per-candidate scores, canonical hashes, restart spread, second-best score, heuristic recovery witness, and wall-clock timing. This is exactly the kind of observability a topology-search oracle needs.
- **Defaults are explicit and tied to a documented "golden" fit** (`QpGraphOptions fit;` plus the comments referencing the `cc9ff69` AT2 fit). That gives reviewers a concrete baseline.
- **Documentation explains *why* the design exists** — exhaustive vs. stochastic search, the falsifiable heuristic gate, and the single-GPU-launch shape. This isn't just "what" filler.

## What a senior developer would flag

**The header file is doing too much narrative work.**

Lines 3-19 are an implementation manifesto in a public header:

```cpp
// WHY (C) and not match-AT2's-search: AT2's find_graphs is stochastic ...
// GPU SHAPE: the f3 basis (depends only on the pop-set) + Qinv are assembled ONCE ...
```

This is valuable content, but it belongs in a design doc or the `.cpp`, not at the top of a public API header. Headers should state *what* callers can depend on; the "why" can drift out of sync with the implementation.

**The member-comment numbering is out of order.**

Line 76 labels the first field group `(2) EXHAUSTIVE-COVERAGE`, line 82 `(3) GLOBAL-BEST`, and line 86 `(1) PER-CANDIDATE`. That's a stale copy-paste drift from some earlier ordering. It's harmless, but it's exactly the kind of thing that makes a reviewer wonder what else drifted.

**Implicit constraints with no enforcement or helper.**

```cpp
/// The maximum number of admixture nodes (the bounded nadmix axis). v1 supports {0,1}.
int max_nadmix = 1;
```

A senior reviewer would want either a compile-time or runtime guard for the `{0,1}` constraint, or at least a `[[nodiscard]] bool valid() const` method on `QpGraphSearchOptions`. As written, callers can pass `max_nadmix = 5` and only discover the failure mode at runtime, deep in the implementation.

**Magic seed count with no justification.**

```cpp
int heuristic_seeds = 8;
```

Why 8? It may be derived from hardware concurrency or empirical testing, but the comment only says "the recovery must hold from ALL." A named constant or a brief rationale would help.

**Default-on heuristic changes behavior silently.**

```cpp
bool run_heuristic = true;
int heuristic_seeds = 8;
```

Defaults that invoke extra compute are risky. If a downstream caller omits the field, they get a heuristic run they may not expect. That may be intentional, but it warrants a louder warning.

**Mixed precision claim is hard to verify.**

```cpp
Precision::Kind precision_tag = Precision::Kind::Fp64;
```

The result defaults to `Fp64`, but the header gives no indication whether the device path actually honors this or whether `precision_tag` is set from the `Resources`/backend at runtime. If it's always `Fp64`, it's redundant; if it's meant to reflect runtime behavior, a stale default is a bug waiting to happen.

**The `Status` member is opaque.**

```cpp
Status status = Status::Ok;
```

No documentation of which `Status` values can be produced (`Ok`, `InvalidConfig`, `OutOfMemory`, etc.), and no helper like `bool ok() const`. Callers have to know the `Status` enum by heart.

**Comments occasionally promise more than the type system guarantees.**

Lines 95-97 say the `best_fit` field is "the full single-graph FIT of the global-best," but the type is just `QpGraphResult`. That's probably fine if `QpGraphResult` is self-evident, but a reviewer new to the codebase has to take it on faith.

## The "slop" test

**Not slop.** Slop is magic numbers without context, copy-pasted code with stale comments, missing error paths, and algorithms that happen to pass tests. This header has none of that. The comments are verbose but accurate, the API is coherent, and the defaults are explicit. If anything, it's *over*-documented, not under-documented.

## What it actually looks like

This looks like the public face of a research compute project written by a domain expert who cares about reproducibility and parity with an external reference (AT2). The API design is sensible: bounded configuration in, deterministic oracle result out. The author wants the header to double as a design document, which is understandable in a small team but will become a maintenance burden as the implementation evolves.

A senior C++ reviewer would say: "Good API, but move the manifesto out of the header, add validation for the `{0,1}` constraint, and document the `Status` contract." A senior CUDA/compute reviewer would say: "The surface is clean; now show me the implementation to see if the one-GPU-launch claim actually holds."

## Verdict

**B+.** Solid, competent public API with strong domain reasoning, but dragged down by header bloat and a few implicit contracts that should be explicit.

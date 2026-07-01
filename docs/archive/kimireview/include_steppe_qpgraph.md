I read through this carefully. This is a clean, CUDA-free public header, and it's clearly written by someone who understands the project layering. But a senior C++ reviewer would have a few specific gripes, mostly around API ergonomics and the "struct of parallel arrays" result design.

## What's genuinely good

- **The CUDA-free boundary is exactly right.** Forward-declaring `device::DeviceF2Blocks` and `device::Resources` (lines 39-42) and keeping the header purely standard C++ honors the architecture.md §4 layering. This is the correct way to expose a GPU-first algorithm to core, CLI, and bindings without leaking `<cuda_runtime.h>`.
- **Domain defaults are named and sourced.** The `QpGraphOptions` struct (lines 56-80) gives every AT2 parity constant a name (`fudge`, `diag_f3`, `numstart`, `constrained`) and explicitly ties the defaults to admixtools::qpgraph() / the golden generator. That removes the usual "where did this literal come from?" problem.
- **Error handling is a value, not an exception.** Returning `Status` inside `QpGraphResult` (line 122) and documenting "NEVER an exception for a domain outcome" (lines 83-84, 120-122) is a mature contract for numerical code where non-SPD covariances and degenerate graphs are expected edge cases, not bugs.
- **`[[nodiscard]]` on both entry points** (lines 134, 142) shows awareness that ignoring a fit result is almost certainly a mistake.
- **Default member initializers everywhere** (e.g., `double score = 0.0;`, `Status status = Status::Ok;`) make the result struct safe to default-construct without surprise garbage.

## What a senior developer would flag

**The result type is a pile of parallel vectors.**

```cpp
std::vector<double> weight;
std::vector<double> weight_lo;
std::vector<double> weight_hi;
std::vector<std::string> admix_from;
std::vector<std::string> admix_to;
// ...
std::vector<double> edge_length;
std::vector<std::string> edge_from;
std::vector<std::string> edge_to;
```

This is an anti-pattern for a public API. A caller has to keep four separate arrays in sync to reason about one admixture node, and three for one edge. A `std::vector<AdmixtureFit>` and `std::vector<DriftEdgeFit>` (or even reusing `QpGraphEdge` + length) would be far less error-prone and self-documenting. Parallel arrays in a result struct tend to drift during future refactors.

**The host-oracle overload still demands `device::Resources&`.**

```cpp
[[nodiscard]] QpGraphResult run_qpgraph(const F2BlockTensor& f2_host,
                                        const std::vector<QpGraphEdge>& edges,
                                        const std::vector<std::string>& leaf_names,
                                        const QpGraphOptions& opts,
                                        device::Resources& resources);
```

The comment says "CpuBackend reads host memory," but a host-oracle API that still requires a mutable `device::Resources` is surprising. Is it allocating device scratch? Scheduling on a GPU anyway? A senior reviewer would want either a `const device::Resources&` if it's read-only, or a comment explaining exactly what mutable state is touched. Non-const `resources` for a "host" overload looks like a leak in the abstraction.

**The admixture weight semantics are fragile.**

```cpp
/// weight[j] is the mass on the edge admix_from[j] -> admix_to[j]
/// (the FIRST incident parent; the second parent carries 1 - weight[j]).
```

Lines 95-96 silently encode a "first parent wins" convention. That is a contract the implementation must enforce deterministically across the edge list, and it's easy to get wrong when parsing admixtools-format edges. A reviewer would flag this as a subtle source of silent correctness bugs if the implementation's "first incident parent" ordering isn't bulletproof.

**Zero is an overloaded sentinel.**

```cpp
double worst_residual_z = 0.0;
```

Line 113 defaults to 0.0, but 0.0 is also a perfectly valid "best possible residual." If the fit fails early or the basis is empty, a caller cannot tell "no residual computed" from "perfect fit." `std::optional<double>` or NaN would remove that ambiguity.

**No visible precision control, despite the precision tag.**

```cpp
Precision::Kind precision_tag = Precision::Kind::Fp64;
```

Line 125 records which precision produced the result, but `QpGraphOptions` has no precision field and the entry points don't take a precision argument. That means precision is inferred from the input tensor or hard-coded elsewhere. A senior reviewer would expect the public API to either expose precision as an option or document clearly why it isn't.

**The header guard is old school.**

```cpp
#ifndef STEPPE_QPGRAPH_HPP
#define STEPPE_QPGRAPH_HPP
```

Lines 27-28 work, but the rest of the project likely uses `#pragma once`. It's a minor convention mismatch; in a new file a senior dev would just use the simpler pragma.

**Milestone-heavy comments will go stale.**

The top-of-file comment (lines 3-26) is accurate and useful right now, but it's full of transient project names: "Phase 2," "qpGraph milestone," "IDEA-1 optimizer spike," "path-algebra prototype," "AT2 optim() host-loop trap." These are great context today, but in six months they'll either be wrong or require a new reader to know internal codenames. A public header should describe *what* and *why* with less roadmap archaeology.

## The "slop" test

**Not slop.** There are no magic numbers, no raw owning pointers, no copy-pasted blocks with stale comments, no `printf`/`std::cout` leakage, and no obviously wrong API shape. The file is small, focused, and self-consistent.

## What it actually looks like

This looks like the work of a **competent systems engineer who understands both the genomics domain and the project's GPU/core layering**. The public surface is conservative and well-documented, the CUDA-free boundary is respected, and the error-handling contract is appropriate for numerical fitting code.

The weaknesses are mostly **API ergonomics and future-proofing**, not correctness. Parallel arrays in a result type, a mutable resources argument on a "host" overload, and sentinel-laden diagnostics are the kinds of things that make downstream callers write boilerplate or footgun themselves later. A senior reviewer would say: "Solid foundation, but tidy the public result type before it accumulates callers."

**Verdict:** Respectable public header. Good enough to ship internally, but the result struct deserves a refactor before it's locked in. B+.

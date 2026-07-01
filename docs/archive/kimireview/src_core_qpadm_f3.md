I read through `f3.cpp` carefully. It is **not slop** — it is clearly written by someone who understands both the genomics domain and the project’s architecture — but a senior reviewer would flag a few contract and style gaps.

## What's genuinely good

- **The OOM-aware diagonal-only path is the right call.** Using `jackknife_diag` instead of the dense `jackknife_cov` for a bare f3 SE is a solid architectural decision, and the comment at lines 120–127 explains *why* (no Q invert, `O(m·nb)` work, bit-exact by construction).
- **Fail-fast on the `size_t → int` narrowing.** Lines 77–86 guard against silent truncation at population-sweep scale (`C(2500,3) > INT_MAX`) and return a `Status::InvalidConfig` value instead of throwing. That is exactly the defensive check this kind of code needs.
- **Modern C++ ownership model.** No raw pointers, no `new`/`delete`, no manual memory management. Vectors and `std::span` carry the data, and the backend is passed by reference.
- **Template deduplication of the two overloads.** `run_f3_impl` cleanly handles both `DeviceF2Blocks` and `F2BlockTensor`, mirroring the `run_f4_impl` pattern noted in the comment.
- **Result contract for degenerate batches.** Lines 111–118 return `Ok` with NaN rows rather than crashing or returning a vague error — a sensible API choice for statistics code.

## What a senior developer would flag

**Error-handling contract inconsistency — `primary_backend` can throw:**

```cpp
46 [[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
47     return *resources.gpus.at(kPrimaryGpu).backend;
48 }
```

`std::vector::at` throws `std::out_of_range` if `resources.gpus` is empty. That directly contradicts the file’s stated architecture — “Domain outcomes … are a per-call status VALUE, never an exception” (line 17). A caller of `run_f3` who expects only `Status` values could get an exception instead. This is the most obvious gotcha.

**Downstream error status is invisible.**

```cpp
105 F4Blocks X = core::qpadm::assemble_f3_triples(be, f2, std::span<const int>(flat), prec);
128 const JackknifeDiag diag =
129     core::qpadm::jackknife_diag(be, X, std::span<const int>(X.block_sizes), prec);
```

Both downstream calls return values, not `Status`. If `assemble_f3_triples` or `jackknife_diag` can fail, this file has no way to observe it. Either they throw (contradicting the status-value architecture), or they signal failure through a field this file does not check. A senior reviewer would want the contract spelled out: “these seams never fail,” “they throw on programmer error only,” or “check `X.status`/`diag.status` here.”

**Verbose, shouty comments.**

The top 18-line block is dense with all-caps emphasis: “REUSES,” “NO new infrastructure,” “NO ALS / NO rank test,” “NEVER,” “UNFUDGED,” “BY CONSTRUCTION.” Lines 60–63 even include a meta-comment justifying why `(void)opts` exists. The explanations are accurate, but the tone reads defensive. Senior reviewers often distrust code that feels like it is trying too hard to prove it is correct.

**Redundant explicit `std::span` construction.**

```cpp
105 F4Blocks X = core::qpadm::assemble_f3_triples(be, f2, std::span<const int>(flat), prec);
128     core::qpadm::jackknife_diag(be, X, std::span<const int>(X.block_sizes), prec);
```

`std::span` has an implicit constructor from contiguous containers, so `flat` and `X.block_sizes` would bind directly. The explicit casts are harmless but add noise and suggest the author is not fully comfortable with `span`’s conversion rules.

**Idiosyncratic NaN choice.**

```cpp
112 res.est.assign(static_cast<std::size_t>(N), std::nan(""));
```

Same as using `nan("")` in CUDA kernels: it works, but `std::numeric_limits<double>::quiet_NaN()` is the more conventional C++ idiom. This is a minor “C-with-classes” tells.

**Hardcoded primary GPU index.**

```cpp
44 inline constexpr std::size_t kPrimaryGpu = 0;
```

The comment acknowledges this is a TU-private convention and that multi-GPU fan-out lives above, which is fine. But it is still a magic constant in a file that otherwise tries to route everything through injected resources. A named config value or a defaulted parameter would be cleaner.

## The "slop" test

**Not slop.** Slop would be copy-pasted f4 code with stale comments, unchecked downstream status, magic numbers, and “it passes tests so ship it.” This file has none of that. The math is right, the OOM path is intentional, and the comments — despite being overwrought — explain *why*, not just *what*.

## What it actually looks like

This looks like **competent research/engineering code written by a domain expert who is actively trying to follow a project-wide architectural contract.** The author clearly understands the f3/f4 relationship, the jackknife math, and the memory/performance constraints. The C++ is mostly modern and safe. The rough edges are not competence gaps — they are contract gaps (`at()` can throw, downstream status is unobserved) and editorial choices (over-commenting, redundant casts).

A senior reviewer’s internal monologue would be: “Solid core logic, but tighten the error-handling contract and cut the comment volume by a third.”

**Verdict:** B+ — ship after resolving the exception-vs-status inconsistency and clarifying how downstream seam failures are surfaced.
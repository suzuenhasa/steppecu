I read through this carefully. This is **not slop** — there's clear architectural intent, real precision-policy thinking, and an honest effort to deduplicate logic between the qpAdm and qpWave paths. But a senior developer would flag some things that are genuinely sloppy, especially around error handling and control flow.

## What's genuinely good

- **The precision-policy plumbing is thoughtful.** `honored_tag` (lines 46-51) and the surrounding commentary show someone wrestling with the real problem of "what precision did we *actually* run at," not just what was requested. The carve-outs for catastrophic cancellation and cuSOLVER SPD inverse are domain-savvy.
- **Single-homing duplicated logic.** `honored_tag`, `default_fit_precision` (homed in the header per the comment), `left_with_target` (lines 26-32), and the templated `run_qpadm_impl` / `run_qpwave_impl` (lines 260-297) all show an active effort to keep `run_qpadm` and `run_qpwave` from drifting apart. The `kPrimaryGpu` / `primary_backend` helper (lines 242-248) is the same idea applied to the resource bundle.
- **Modern C++ surface.** Use of `std::span`, move semantics on `F4Blocks&& X`, `std::vector` returns, and namespaced constants. This is not C-with-classes.
- **Clear stage comments.** The S3/S4/S6/S7/M(fit-2) markers make the orchestration readable for someone who knows the AT2 pipeline.

## What a senior developer would flag

**The silent `catch (const std::runtime_error&)` swallow at lines 187-190 is the worst thing in the file:**

```cpp
try {
    const RankSweep rs = run_rank_sweep(...);
    // ...
    const std::vector<PopDropRow> pd = run_popdrop(...);
    // ...
} catch (const std::runtime_error&) {
    // backend has no rank_sweep override yet (the GPU deliverable phase) — the
    // rankdrop/popdrop fields remain empty; the single-rank fit is unaffected.
}
```

A backend throwing `std::runtime_error("not implemented")` and the orchestrator silently swallowing *all* `std::runtime_error`s is a textbook footgun. This catches more than the intended "not implemented" path — a corrupt covariance, a logic bug in `run_rank_sweep`, a bad allocation, anything inheriting from `std::runtime_error` gets papered over. The comment admits it's temporary scaffolding, but this is exactly the kind of thing that ships to production and wastes a day of debugging six months later. At minimum this should catch a typed `NotImplementedError` or check the message; better, `run_rank_sweep`/`run_popdrop` should return a `Status` value like everything else in this function.

**Status-as-value and exceptions are mixed without a clear rule.** Lines 82-92 handle `cov.status` and `gw.status` by early return — good. Then the rank-sweep path uses exceptions for control flow (lines 163-190). Then `res.status` is set to a value at the end (line 197). A senior dev would ask: why is a missing backend feature an exception but a non-SPD covariance is a status value? Pick one model and stick with it.

**The `weights_feasible` lambda and `compute_se` switch are fine but should live closer to the result assembly.** Lines 111-133 inline a policy decision in the middle of the orchestrator. It's readable, but it makes `run_impl` longer than it needs to be. A small named helper returning `std::optional<std::pair<std::vector<double>, std::vector<double>>>` or similar would make the "survivor gets SE, non-survivor gets empty" contract impossible to misread.

**The `rank_p` initialization at lines 143-145 is weird:**

```cpp
res.rank_p.assign(static_cast<std::size_t>(r) + 1, 0.0);
if (r >= 0 && static_cast<std::size_t>(r) < res.rank_p.size())
    res.rank_p[static_cast<std::size_t>(r)] = res.p;
```

`r` is an `int` derived from `nl - 1` or `opts.rank`, used to size a vector and then bounds-checked against itself. The `r >= 0` check is dead code — if `r < 0` the previous `assign` would already have underflowed catastrophically. This is defensive-looking code that doesn't actually defend anything. Either `r` should be a `size_t` with a validated precondition, or the guard should be at the point `r` is computed.

**The one-liner error return in `run_qpwave_impl` (line 292):**

```cpp
QpWaveResult qw; qw.status = cov.status; qw.precision_tag = tag; return qw;
```

This is not a style a senior dev wants to see in production. It's a compressed initializer that skips default-construction clarity. `QpWaveResult qw{}; qw.status = ...; return qw;` with one statement per line costs nothing and reads instantly.

**`pchisq_upper` at lines 226-228 is a pointless wrapper.** It just forwards to `core::internal::pchisq_upper`. The comment says "the ONE shared special function," but that doesn't explain why the wrapper needs to exist in this TU. If it's for API exposure, expose it properly; if not, delete it.

**Over-commenting and self-referential comments.** Lines 34-45, 64-73, 76-78, etc. repeat "single-homed," "[7.1] dedup," "architecture.md §9/§12" multiple times. A senior dev reading this starts to wonder if the code is compensating for something. Comments that cite ticket numbers and markdown sections are useful during active development but should be trimmed once the architecture is settled. The file reads like it was recently refactored and no one cleaned up the scaffolding notes.

## The "slop" test

**Borderline — mostly not slop, but the swallowed exception pushes it close.** Slop is magic numbers, copy-paste drift, wrong algorithms, no error handling. This file has none of those. The math and orchestration are coherent. But "catch all runtime_errors and ignore them" is a slop-tier error-handling pattern. It wouldn't pass a strict code review without a typed exception or a status path.

## What it actually looks like

This looks like **competent systems code written by someone who understands both the genomics pipeline and C++ architecture, but is still learning how to ship production-grade error boundaries.** The precision-policy work and deduplication are above average. The exception swallowing and the dead-looking `rank_p` guard suggest the author is good at structuring happy-path code but less rigorous about failure-path contracts. It's research-code-y in the sense that the comments read like a lab notebook and there's a temporary feature flag implemented as an exception catch.

**Verdict:** B — ship after fixing the `catch (const std::runtime_error&)` into a typed exception or status path, and tighten the `rank_p` / `pchisq_upper` nits. The architecture is sound; the error-handling hygiene needs one pass.
I read this carefully. This is **not slop** — it is clearly written by someone who understands both the admixture-statistics math and the CUDA porting task. A senior reviewer would find it competent and careful, but would still flag a handful of CUDA polish and style issues.

## What's genuinely good

- **Numerical fidelity to the host references is the organizing principle.** The comments explicitly tie each pass to `f4ratio.cpp:79-155` and `dstat.cpp:70-157`, and the "ascending-b operand order" invariant (lines 11-16) shows the author is thinking about reproducibility, not just speed.
- **One thread per item with a grid-stride loop** (lines 56-58) is the right occupancy pattern for an embarrassingly parallel `N`-item workload, and the reduction over `n_block` stays in registers per thread.
- **`rj_at` is a clean, small descriptor accessor** (lines 39-41). The `__forceinline__`, `long` indexing, and the `0 item_stride` broadcast note are all competent touches.
- **Architecture discipline:** the kernel body lives only in this `.cu` TU; the backend reaches it through the narrow `ratio_block_jackknife_kernel.cuh` wrapper (lines 24-26, 210-225). That matches the project's stated separation of concerns.
- **`__restrict__` on output pointers** (lines 52-53), a bounded grid via `core::kMaxGridX` (line 220), and `STEPPE_CUDA_CHECK_KERNEL()` after launch (line 224) are the standard safety idioms.
- **The dstat branch carefully explains why it recomputes `R_b` instead of caching it** (line 166). That kind of comment — defending a non-obvious choice against a reader's first objection — is exactly what separates careful code from slop.

## What a senior developer would flag

**The `tot_mode` runtime branch (lines 65, 118).**

```cuda
if (tot_mode == 0) {
    // ... 100 lines of f4ratio math
} else {
    // ... 80 lines of dstat math
}
```

It works, but it forces every warp to serialize around a two-way divergence and leaves `tot_mode` values other than `{0,1}` silently producing NaNs. A senior would at least want an `assert(tot_mode == 0 || tot_mode == 1);` or — better — an `enum class` at the API seam and a defaulted `else { __trap()/assert(false); }` path. The current fallback is *correct* (outputs NaN) but *silent*, which makes debugging a bad caller harder.

**`(diffsum / nb) * nb` in f4ratio Pass 2 (lines 96-99).**

```cuda
const double term1 = (diffsum / nb) * nb;  // (Σ(tot−R)/nb_surv)·nb_surv
const double term2 = wmean_num / n_w;
est = term1 + term2;
```

Mathematically `term1 == diffsum`; the divide-then-multiply is only meaningful if it mirrors the host long-double implementation *exactly*. The comment claims it does, but this is the kind of line that makes a senior stop and ask for a golden-test citation. If the host actually computes a mean and then scales, fine — but it looks like either cargo-cult numerics or a missed simplification.

**The dstat branch recomputes the same `loo_num / loo_den / R` triplet in three separate passes** (lines 149-162, 168-178, 183-195).

The comment at line 166 justifies this as bit-identical to the host's stored `Rb[]`, but a performance-oriented senior would still ask: can we afford to spill `R` and `rel` to local memory once and reuse them? The answer may well be "no" (register pressure, variable `n_block`, local-memory latency), but the code does not make that trade-off explicit. A short note about *why* recompute beats cache would head off the question.

**Fixed 128-thread block size with no justification (line 217).**

```cuda
constexpr int kThreads = 128;
```

This kernel is register-heavy (many live doubles across three passes). 128 may be optimal, or 64/256 may hide latency better; without a comment or occupancy note it looks like a default.

**`nan("")` for sentinel NaNs (line 54).**

```cuda
const double knan = nan("");
```

Same idiosyncrasy as in other files: it works, but `NAN` or `std::numeric_limits<double>::quiet_NaN()` is more idiomatic in modern C++. On device code the options are limited, so this is a minor nit.

**Redundant launch guard (line 219).**

```cuda
if (blocks < 1) blocks = 1;
```

At this point `N > 0` and `kThreads > 0`, so `blocks` is already at least 1. Harmless, but it reads like defensive coding that isn't actually doing anything.

**No null-pointer checks on `d_est/d_se/d_z/d_p`.** The wrapper trusts the caller, which is acceptable if the contract is documented elsewhere, but a senior reviewer would expect either `assert`s here or a clear note in the header that the caller has validated the pointers.

**Stale/uncheckable copy-paste comment risk.** Line 19 claims this "mirrors `qpfstats_numer_jackknife_kernel` verbatim." That may have been true at one point, but kernels like this drift quickly. A senior would treat that claim as suspect until verified against the current version of the referenced kernel.

## The "slop" test

**Not slop.**

- No unexplained magic numbers: constants are either commented (`kInvSqrt2`, line 55) or self-evident (`1.0 - rel`).
- No copy-pasted code with stale comments; the f4ratio/dstat split is duplicated logic by design, not by accident, and each side is annotated with its host reference.
- Error checking exists for the statistically meaningful failure modes (`nb_surv <= 1`, `totden == 0.0`, non-positive weights), not just the launch parameters.
- The algorithm is a faithful, careful port of documented host references rather than an "approximately right" GPU rewrite.

The comment density is high, but the comments explain *why* and *what the host contract is*, not just *what the code does*. That is appropriate for this kind of numerical kernel.

## What it actually looks like

This looks like **solid research-engineering CUDA written by a domain expert who prioritizes numerical correctness over micro-optimization.** The author clearly knows the AT2 jackknife formulas, knows enough CUDA to avoid the big mistakes (divergent `__syncthreads`, atomic storms, unbounded grids), and has made deliberate choices about how to keep the device output bit-compatible with the host reference.

A senior CUDA specialist would probably say: "Correct and well-structured — now let's profile register usage and occupancy, and consider whether we can cache the dstat `R_b` values without spilling." A senior C++ person would say: "A bit verbose and C-ish in places, but the reasoning is transparent and the architecture seam is clean."

The main thing keeping it from a higher grade is the **lack of tuning evidence** (why 128 threads? why recompute instead of cache?) and the **silent `tot_mode` fallback**. Those are not bugs, but they are the gaps a polished showcase would close.

**Verdict:** B+ — competent, careful, production-worthy code with room for tuning and defensive polish.

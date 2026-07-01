I read through this carefully. This is a **clean, well-scoped public header** — clearly written by someone thinking about API boundaries and downstream consumers. A senior developer would find a lot to like and a few things to tighten.

## What's genuinely good

- **The contract is precise and domain-informed.** The comment block (lines 1–26) states exactly what this is (standalone f4, NOT a fork of qpAdm), what it reuses (`assemble_f4`, `jackknife_cov`), and what it does not do (no ALS, no rank test). That shows architectural awareness.
- **CUDA-free public surface.** Forward declarations of `device::DeviceF2Blocks` and `device::Resources` via `qpadm.hpp` let the header stay standard C++ (lines 36–38). This is the right call for a library boundary.
- **Status-based degenerate outcomes.** Line 55 makes `Status::Ok` the default and the comment (lines 53–55) documents that `NonSpdCovariance` is returned as a value, never thrown. That is consistent, reviewable error handling.
- **`[[nodiscard]]` on all entry points** (lines 65, 70, 78). Small thing, but it signals "this result matters; don't silently drop the status."
- **Dual overload design is sound.** Device-resident primary (line 70) plus host-oracle parity door (line 78) gives testability without polluting the production path.

## What a senior developer would flag

**`F4Result` is eight parallel arrays with no invariant enforcement:**

```cpp
struct F4Result {
    std::vector<int>    p1, p2, p3, p4;
    std::vector<double> est, se, z, p;
    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};
```

A senior reviewer would immediately ask: who guarantees these fields stay the same length? There is no constructor, no `reserve`, no `emplace_back` helper, and `status = Ok` is the default even for an empty result. For a public API this is fragile; it puts the consistency burden on every implementation and every binding consumer.

**Reusing `QpAdmOptions` for an f4 call is pragmatic but leaky:**

```cpp
[[nodiscard]] F4Result run_f4(..., const QpAdmOptions& opts, ...);
```

The comment (lines 17–19) explains that `fudge = 0` for bare f4 because there is no matrix inverse to regularize. But if `fudge` is irrelevant, an f4-specific options struct would be cleaner than dragging qpAdm's GLS ridge parameter into a sibling API. A future caller will inevitably set `fudge = 1e-4` and wonder why it is ignored.

**The `precision_tag` field is a bit muddled:**

```cpp
Precision::Kind precision_tag = Precision::Kind::Fp64;
```

The comment (lines 57–59) says `est` is always native FP64 and only the covariance SYRK may vary. That means `precision_tag` does not describe the whole result. If the tag is meant to be "precision of the covariance," it should be named that; otherwise consumers may misreport the estimate precision.

**Missing preconditions and lifetime documentation.** `quartets` are P-axis indices (lines 68–69), but nothing documents that they must be in `[0, f2.pop_count())`, or whether duplicate indices are allowed, or whether the span must outlive the call. For a zero-D2H GPU path this matters.

**Resource mutability is unexplained:**

```cpp
device::Resources& resources
```

`resources` is taken by non-const reference, implying the call mutates GPU state/streams/pools. That is expected, but a senior reviewer would want a one-line comment on thread safety: can two `run_f4` calls share one `Resources`? Must they be serialized? A public header should say.

**Tiny: `f4_two_sided_p` has no documented behavior for `NaN` or infinite `z`.** Not every function needs that, but a tail-probability helper in a stats library probably should.

## The "slop" test

**Not slop.** Slop is magic numbers without explanation, stale copy-paste, inconsistent error handling, or APIs that leak implementation details carelessly. This header has none of that. The comments explain *why* the design choices exist, and the public surface is deliberately narrow.

## What it actually looks like

This looks like **a careful, domain-expert public API written by someone who has been burned by header bloat and wants a stable downstream contract.** The author knows the f4 math, knows the difference between the device and host paths, and is trying to keep the surface small. It is not flashy; it is the kind of header a senior reviewer is relieved to see after wading through implementation files.

A senior C++ reviewer would say: "Good separation of concerns, but give me a helper to keep `F4Result` fields consistent and tighten the options struct." A senior CUDA reviewer would say: "Fine surface; the real risks are in the `.cpp` launch config and error propagation, which I can't see here."

## Verdict

**A- / B+.** Solid public header with a coherent contract. The parallel-array result struct and the borrowed `QpAdmOptions` are the main things keeping it from an A.

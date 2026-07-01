I read `f4.cpp` carefully. This is **not slop** — it’s written by someone who understands the f4 seam, the AT2 math, and why the diagonal jackknife is the right OOM-safe shortcut. But a senior C++ reviewer would flag a handful of defensive gaps and one stale copy-paste comment.

## What's genuinely good

- **The architecture comment at the top is excellent.** Lines 1–16 correctly position `run_f4` as a sibling of `run_qpwave`, explain why it reuses `assemble_f4_quartets`, and justify the diagonal-only SE. That shows real design-context awareness.
- **Template deduplication of the two overloads** (lines 53–54, 158–170) is the right call: `DeviceF2Blocks` and `F2BlockTensor` share one `run_f4_impl` body instead of two near-copies.
- **Empty and degenerate batches return gracefully** with `Ok` + NaN rows rather than crashing (lines 68–72, 100–107). That’s a clean, caller-friendly contract.
- **The p-value computation is correct.** Lines 152–155 use `std::erfc(|z|/sqrt(2))`, which is exactly `2 * pnorm_upper(|z|)`.
- **Value semantics throughout** — `std::span`, `std::vector`, references, no raw owning pointers in this file.

## What a senior developer would flag

**Stale seam name in the header comment:**

```cpp
// run_f4 is the SIBLING of run_qpwave ... it REUSES the same two seams —
// assemble_f4_quartets ... and jackknife_cov (the block-jackknife covariance)
```

That says `jackknife_cov`, but the implementation uses `jackknife_diag` (lines 109–120). The inline comment later corrects itself, but the top-level copy-paste drift is the kind of thing that makes another senior dev mutter “ugh, stale header.”

**No validation of the `m == N` invariant:**

```cpp
const int m = X.nl * X.nr;  // == N (the quartet m-axis)
...
for (int k = 0; k < N; ++k) {
    const double est = X.x_total[ks];
    const double var = diag.var[ks];
```

Line 95 assumes `m == N`; if `assemble_f4_quartets` ever returns a different `m`, the loop either walks off the end or silently drops rows. A `STEPPE_ASSERT(m == N)` or an explicit status check belongs here.

**Un-guarded primary backend access:**

```cpp
[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}
```

Line 44 will throw `std::out_of_range` if `gpus` is empty, and dereference a null pointer if `.backend` is null. That directly conflicts with the file’s own claim that domain outcomes are status values, “never an exception.”

**Error propagation is effectively absent:**

The only status ever returned is `Status::Ok`. Degenerate batches become `Ok`+NaN (which may be the intended contract), but if `assemble_f4_quartets` or `jackknife_diag` can fail, that failure is swallowed. From this file alone it looks like the no-exceptions policy is asserted but not enforced.

**API ergonomics around `QpAdmOptions`:**

```cpp
(void)opts;  // line 61
```

Silencing an unused parameter is fine, but taking `QpAdmOptions` and then deliberately ignoring `opts.fudge` is a mild trap for callers. Either accept a narrower `F4Options`, or name the parameter `/*opts*/`.

**Include-what-you-use nit:**

The file uses `std::array<int, 4>` in the public signature (line 159) but does not `#include <array>`; it relies on a transitive include from `steppe/f4.hpp`. That’s brittle.

**Minor C++ style nits:**

- `std::nan("")` at line 101 works but reads C-ish; `std::numeric_limits<double>::quiet_NaN()` is more idiomatic.
- `static const double kInvSqrt2` at line 153 could be a `constexpr` literal (or `std::numbers::sqrt2_v` if the project is C++20).
- `const int m = X.nl * X.nr;` could overflow for huge batches; `int64_t` would be safer.

## The "slop" test

**Not slop.** The comments explain *why*, the math is right, and there’s no wall of unexplained magic numbers or obviously wrong algorithms. The only copy-paste drift is the `jackknife_cov` vs. `jackknife_diag` mismatch.

## What it actually looks like

This looks like **competent systems/research C++ written by someone who knows the genomics pipeline and the seam contracts well, but hasn’t fully production-hardened the error boundaries.** The author clearly thinks about edge cases (empty batches, OOM-safe diagonal SE, bit-exact FP64), but still trusts downstream invariants a little too much and hasn’t reconciled the “no exceptions” policy with `std::vector`/`std::at` usage.

## Verdict

**B+.** Ship after tightening the invariant checks (`m == N`, valid `resources.gpus`), clarifying or propagating non-`Ok` statuses from the seams, and fixing the stale `jackknife_cov` comment.
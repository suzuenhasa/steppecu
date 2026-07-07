# `readv2_classify.hpp` reference

## 1. Purpose

`src/core/readv2/readv2_classify.hpp` declares the **pure-host numeric heart of
READv2** — the piece that turns a pair's normalized P0 into a kinship *degree*
token and computes the *z statistic* the concord gate measures. It is the
type-and-contract half of a two-file module: this header names the cut points,
the four degree tokens, and the three free functions; the arithmetic behind them
lives in `src/core/readv2/readv2_classify.cpp` (see
`src_core_readv2_readv2_classify.cpp.md`). This doc describes what the header
promises; the `.cpp` doc describes the mechanics.

The whole module is **pure host, CUDA-free arithmetic**. There is no device, no
I/O, no `RunConfig` — every function takes only per-pair scalars and returns a
scalar (or a token). That isolation is the point: it makes the classifier
directly unit-testable on a machine with no GPU toolkit, and
`tests/unit/test_readv2_classify.cpp` pins each function exactly at its cut
points. Its one caller in the real pipeline is `readv2.cpp`, which walks the
surviving pairs after the GPU sweep and calls `degree_from_p0norm` and
`readv2_z` once per row to fill the `degree` and `z` columns of the frozen
output schema.

One deliberate scope decision worth stating up front: this arithmetic is done in
**native FP64**, never the emulated-FP64 path used for the matmul-heavy fit
engine. The values here are a handful of scalar operations on numbers the GPU
reduction already produced — there is no matmul to emulate — so plain `double` is
both correct and simplest.

---

## 2. The degree ladder and its cut points

READv2 sorts a pair into one of four kinship classes by where its **normalized
P0** falls. Lower normalized P0 means a closer relationship; higher means more
distant. The header names three cut points and, through them, four half-open
bands:

| Band | Condition | Token |
|---|---|---|
| Closest | `p0_norm < kCutIdentical` (0.625) | `identical` |
| | `p0_norm < kCutFirst` (0.8125) | `first` |
| | `p0_norm < kCutSecond` (0.90625) | `second` |
| Most distant | otherwise (`>= 0.90625`) | `unrelated` |

The comparisons are all strict `<`, which makes the bands **half-open on the
low side**: a value sitting exactly on a boundary belongs to the *more-distant*
class. That convention is what the unit tests pin, so it is a contract, not an
accident.

The three constants are the canonical READ / READv2 normalized-P0 boundaries.
steppe's frozen enum has only four degrees — there is no "third degree" cell —
so READv2's third band collapses into `second`/`unrelated` the way the fixture
recipe expects. The header is explicit that these numbers are a **named
`constexpr` Phase-1 tunable**, pinned to reproduce the oracle's degree column,
*not* a frozen part of the output contract. The four *tokens*, by contrast, are
frozen: `kDegreeIdentical`, `kDegreeFirst`, `kDegreeSecond`, `kDegreeUnrelated`
are the exact lowercase schema spellings, and downstream tooling (the concord
validator's confusion matrix) depends on those spellings and their order.

---

## 3. `degree_from_p0norm` — the classifier

`degree_from_p0norm(p0_norm)` walks the ladder of section 2 top to bottom and
returns the first token whose band the value falls into, defaulting to
`unrelated` when it clears the last cut. It returns a `const char*` pointing at
one of the four frozen string literals — never an allocation, never a copy — so
the caller can compare pointers or content freely.

The one edge case it handles explicitly: **a NaN normalized P0 classifies as
`unrelated`.** A pair whose normalized P0 is undefined (for instance because its
denominator vanished upstream) is mapped to the most-distant token rather than
being left in an undefined state. That choice is conservative — an
uncomputable pair is reported as "no detectable relatedness," not silently
promoted into a kinship band.

---

## 4. `boundary_for_degree` — which cut the call was made against

`boundary_for_degree(degree)` answers a narrower question the z statistic needs:
given a token, which cut point separates that class from the **next-more-distant**
one? `identical` returns `kCutIdentical`, `first` returns `kCutFirst`, `second`
returns `kCutSecond`.

`unrelated` is the wrinkle: it has no more-distant class, so there is no
"upper" boundary above it. By convention it uses the **nearest** boundary
below it, `kCutSecond` — the same value `second` reports. This keeps the z
statistic well-defined for every degree (section 5) without inventing a
boundary that doesn't exist. The function matches the token by string content,
and any unrecognized string falls through to `kCutSecond` as well.

---

## 5. `readv2_z` — the concord gate's z statistic

`readv2_z(p0_mean, p0_norm, background, n_windows, sum_p0_sq)` computes the
**normalized signed distance from a pair's normalized P0 to its classification
boundary** — how many standard errors of headroom the call has:

```
z = (boundary - p0_norm) / se_p0_norm
```

This is the **`Z_upper` convention**: a positive z means `p0_norm` sits *below*
its boundary (comfortably inside the called class, toward the closer side),
and z shrinks toward zero as the pair approaches the cut. The `boundary` is the
cut for the pair's own degree from section 4, so the statistic is measured
against the very decision that was made.

The standard error is built entirely from the per-pair scalars the GPU
reduction already produced, treating the pair's windows as jackknife blocks:

1. **Window variance of `P0_mean`.**
   `var = (sum_p0_sq - n * p0_mean^2) / (n - 1)`, where `n = n_windows` and
   `sum_p0_sq` is the sum over windows of `P0[w]^2`. A tiny negative value from
   floating-point cancellation is clamped to 0 so the following square root is
   always real.
2. **SE of the mean.** `se_p0_mean = sqrt(var / n)`.
3. **Rescale into normalized space.** `se_p0_norm = se_p0_mean / background`, the
   same `background` (the all-pairs median denominator) that normalized P0 in the
   first place, so numerator and SE live in the same units.

---

## 6. Contracts and invariants

- **`z` is NaN exactly when it is undefined, and that becomes `NA`.** `readv2_z`
  returns `NaN` when `n_windows < 2` (a variance over fewer than two blocks is
  undefined), when `background` is not strictly positive, or when `p0_mean` /
  `p0_norm` is itself NaN. It also returns NaN if the computed `se_p0_norm` is
  not strictly positive (a zero-variance pair has no scale to divide by). The
  emitter turns each of these into the literal `NA` in the output column — a pair
  whose z cannot be computed is reported as absent, never as a fabricated number.

- **Every degree has a defined boundary.** Because `boundary_for_degree` gives
  even `unrelated` a value (section 4), `readv2_z` never faces a missing
  boundary — the z statistic is well-defined for all four classes.

- **The tokens are frozen; the cut points are not.** The four `kDegree*` strings
  are the exact schema vocabulary and must not drift. The three `kCut*` constants
  are a tunable pinned to reproduce the oracle, and are documented as such — a
  future recalibration is expected to touch them, and should leave the tokens
  alone.

- **No allocation, no I/O, no throw.** All three functions are `noexcept` and
  return either a pointer into static string storage or a plain `double`. They
  can be called in a tight per-pair loop with no per-call cost beyond the
  arithmetic, which is exactly how `readv2.cpp` uses them.

- **Native FP64, on purpose.** This module is deliberately outside the
  emulated-FP64 policy that governs the matmul-heavy fit path. The scalar
  relatedness arithmetic is computed in plain `double`, matching the
  scope-locked decision recorded alongside the caller.

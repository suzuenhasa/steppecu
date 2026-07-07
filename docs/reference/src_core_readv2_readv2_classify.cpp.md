# `readv2_classify.cpp` reference

## 1. Purpose

`src/core/readv2/readv2_classify.cpp` is the tiny numeric heart of the READv2
relatedness caller. Given the per-pair scalars the windowed-mismatch reduction has
already produced, it answers two questions for each pair of samples: **what
relatedness degree is this** (identical / first / second / unrelated), and **how far
is the pair from the boundary of that call, in standard errors** (the z statistic).

It is deliberately the smallest possible surface. There is no GPU code here, no I/O,
no allocation — three free functions that take plain `double`s and return a `double`
or a token string. That isolation is what makes it directly unit-testable: the
companion test (`tests/unit/test_readv2_classify.cpp`) pins the classifier right at
the cut points and the z formula at its edge cases, without needing a genotype file
or a device.

The header note calls this "the single native-FP64 arithmetic the concord gate
measures" — meaning the whole READv2 pipeline computes its mismatch counts on the
GPU, but the final normalization and z-score math live here on the host in native
double precision, and the concordance gate that validates steppe against the READv2
oracle is measuring the numbers this file emits.

---

## 2. What the caller hands in

The reduction upstream (`readv2_mismatch`, consumed in
`src/core/readv2/readv2.cpp`) produces four scalars per sample pair. By the time
this file is called, the caller has already turned those into the derived values
these functions take:

- **`p0_mean`** — the mean per-window mismatch rate P0 for the pair (sum of the
  per-window P0 divided by the number of usable windows).
- **`background`** — the all-pairs normalizer: the median (default) or mean of
  `p0_mean` across every surviving pair. This is the "unrelated" scale everything is
  divided by.
- **`p0_norm`** — `p0_mean / background`, the normalized P0 that the whole READ /
  READv2 method classifies on. Around 1.0 for unrelated, lower as relatedness rises.
- **`n_windows`** — how many windows actually contributed (the block count for the
  jackknife-style variance).
- **`sum_p0_sq`** — the sum over windows of `P0[w]^2`, i.e. the second moment the
  variance formula needs alongside `p0_mean`.

Nothing in this file re-derives those or re-reads the genotypes; it works purely
from the five scalars.

---

## 3. The degree classifier (`degree_from_p0norm`)

`degree_from_p0norm(p0_norm)` maps a normalized P0 to one of four frozen lowercase
tokens by walking a fixed ladder of cut points:

| Condition | Degree token |
|---|---|
| `p0_norm` is NaN | `unrelated` |
| `p0_norm < kCutIdentical` (0.625) | `identical` |
| `p0_norm < kCutFirst` (0.8125) | `first` |
| `p0_norm < kCutSecond` (0.90625) | `second` |
| otherwise | `unrelated` |

The cut points come from the canonical READ / READv2 normalized-P0 boundaries: an
unrelated pair sits around P0_norm 1.0, a second-degree pair around 0.75, a
first-degree pair around 0.5, and identical/twin below that. The three constants are
the midpoints between those expected values (and a twin/identical floor), so a pair
lands in a class when it is closer to that class's expected P0 than to the next.

Two design points worth flagging:

- **NaN maps to `unrelated`, the most-distant class.** An undefined normalized P0 is
  not a crash and not a special "unknown" token — it collapses to the safest,
  least-committal call. This keeps the emitter's job simple: every pair gets a real
  token.
- **The four tokens are frozen schema spelling.** `identical` / `first` / `second`
  / `unrelated` are the exact strings the output schema promises. steppe's enum
  deliberately has no separate "third degree" band — per the fixture recipe the
  READv2 third-degree band folds into second/unrelated. The cut points are a named
  `constexpr` Phase-1 tunable pinned to reproduce the oracle's degree column; they
  are a reproduction target, not a permanent contract.

---

## 4. The boundary lookup (`boundary_for_degree`)

`boundary_for_degree(degree)` is the inverse-facing helper the z statistic needs: given
a called degree, which cut point did the call sit against?

| Degree | Boundary returned |
|---|---|
| `identical` | `kCutIdentical` (0.625) |
| `first` | `kCutFirst` (0.8125) |
| `second` | `kCutSecond` (0.90625) |
| `unrelated` (and anything unrecognized) | `kCutSecond` (0.90625) |

It matches the token by string comparison and returns the cut point that separates
that degree from the **next-more-distant** class. `unrelated` is the special case:
there is no class beyond it, so there is no "more distant" boundary to measure
against. It falls back to the nearest boundary, `kCutSecond` — the line separating it
from second-degree. The final `return kCutSecond` also acts as the catch-all for any
string that isn't one of the three named tokens, so the function is total.

This pairs exactly with the classifier: whatever `degree_from_p0norm` calls a pair,
`boundary_for_degree` returns the same cut point that call was decided by.

---

## 5. The z statistic (`readv2_z`)

`readv2_z(p0_mean, p0_norm, background, n_windows, sum_p0_sq)` returns how far the
pair's normalized P0 sits below its classification boundary, expressed in standard
errors. This is the number that tells a reader how confident the degree call is.

The computation, in order:

1. **Guard the inputs.** If there are fewer than 2 windows, or `background` is not
   strictly positive, or either `p0_mean` / `p0_norm` is NaN, it returns NaN. Fewer
   than two blocks means the variance is undefined; a non-positive background means
   the normalization scale is degenerate. Returning NaN here is how the emitter is
   told to write `NA` for this pair rather than a bogus number.
2. **Variance of P0_mean across windows.** Treating the windows as jackknife-style
   blocks, it forms the sample variance
   `var = (sum_p0_sq - n * p0_mean^2) / (n - 1)`, where `n = n_windows`. This is the
   textbook `E[X^2] - E[X]^2` variance written in sum-of-squares form, which is why
   the caller had to carry `sum_p0_sq` — the mean alone can't reconstruct it.
3. **Clamp tiny negatives.** Floating-point cancellation in the `sum_p0_sq - n·mean²`
   subtraction can push a true-zero variance slightly negative; the code clamps
   `var` up to 0 so the following `sqrt` can't produce NaN from a numerically-zero
   variance.
4. **Standard errors.** The SE of `p0_mean` is `sqrt(var / n)`; the SE of the
   *normalized* P0 is that divided by `background` (the same scale `p0_norm` itself
   was divided by, so the units line up). If the resulting SE is not strictly
   positive, it returns NaN — an SE of zero would make the z-score infinite.
5. **Z against the boundary.** It classifies `p0_norm` (section 3), looks up that
   degree's boundary (section 4), and returns `(boundary - p0_norm) / se_p0_norm`.

This is the `Z_upper` convention: the boundary minus the observed value, so a
**positive z means P0_norm sits below the boundary** — i.e. the pair is comfortably
inside the more-related side of its call. The magnitude is how many standard errors
of headroom there are.

---

## 6. Contracts and invariants

- **Pure, `noexcept`, allocation-free.** All three functions are `noexcept` and
  touch no global state, no device, no filesystem. Same inputs always give the same
  output, which is what lets the unit test pin them exactly at the cut points.
- **The classifier is total and the boundary lookup is total.** Every `double`
  (including NaN) maps to one of the four tokens; every string (including an
  unexpected one) maps to a cut point. Neither can fail or fall through.
- **NaN is the "undefined" signal, in both directions.** A NaN normalized P0 becomes
  `unrelated`; an undefined z (too few windows, degenerate background or SE, NaN
  inputs) becomes a NaN return that the emitter renders as `NA`. The rest of the
  pipeline never sees an infinity or an exception out of this file.
- **The classifier and the z-score agree on the boundary.** Because `readv2_z`
  re-runs `degree_from_p0norm` internally and feeds the result to
  `boundary_for_degree`, the z is always measured against the exact cut point that
  decided the very degree reported for the pair. There is no way for the reported
  degree and the boundary in the z formula to disagree.

---

## 7. Edge cases worth knowing

- **A single-window pair** (`n_windows < 2`) has no defined variance, so its z is
  `NA` even though it still receives a degree call.
- **Exactly-on-a-cut-point** normalized P0 falls into the *more-distant* class,
  because every comparison is a strict `<`. A `p0_norm` of exactly 0.625 is `first`,
  not `identical`.
- **A perfect-match-heavy dataset** where more than half the pairs match perfectly
  could drive `background` to zero or negative; upstream treats that as an
  invalid-configuration state, and here it independently makes `readv2_z` return NaN.
  It is called out as impossible on real AADR data but guarded rather than assumed
  away.
- **The `unrelated` z** is measured against `kCutSecond` (the only nearby boundary),
  so a very-unrelated pair with a large `p0_norm` yields a large *negative* z — it
  sits well above the second/unrelated line, which is exactly the right sign for
  "not related."

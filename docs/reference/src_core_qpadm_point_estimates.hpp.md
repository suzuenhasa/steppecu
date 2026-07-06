# `point_estimates.hpp` reference

## 1. Purpose

`src/core/qpadm/point_estimates.hpp` holds one tiny function, `fill_point_estimates()`,
that turns raw statistic results into the four numbers a user actually reads off an
f3 or f4 run: the **estimate**, its **standard error**, a **z-score**, and a
**p-value**.

Some background on what those are. When steppe computes an f-statistic (f3 or f4), it
doesn't just want a single number — it wants to know how trustworthy that number is.
So it computes the statistic once over all the data to get the point estimate, and it
also runs a **block jackknife**: it splits the genome into blocks, recomputes the
statistic leaving each block out in turn, and from how much the answer wobbles it
derives a variance. The standard error is the square root of that variance; the
z-score says how many standard errors the estimate sits away from zero; and the
p-value converts that z into a familiar "how surprising is this if the true value were
zero" probability. This file does the last, purely arithmetic, step: given the total
estimate and the jackknife variance already in hand, it fills in est/se/z/p.

The reason it exists as its own function is deduplication. Both `run_f3` and `run_f4`
had this exact read-out loop, character for character, at the end of their fit. Two
identical copies of a numeric loop are two chances to fix a bug in one and forget the
other, or to have them quietly drift apart. Pulling the loop into a single shared
helper means the f3 and f4 point-estimate read-out is defined in exactly one place and
the two paths cannot disagree.

It is a **host-pure** header — plain C++ arithmetic, no GPU code, no CUDA. The heavy
lifting (the f-statistic sums and the jackknife) happens elsewhere and on the device;
by the time we get here everything is a small array of doubles on the host.

---

## 2. The read-out loop (`fill_point_estimates`)

```
template <class Result>
fill_point_estimates(Result& res,
                     std::span<const double> x_total,
                     std::span<const double> var,
                     int N) -> void
```

The inputs are:

| Input | What it is |
|---|---|
| `res` | The result struct to fill; its `est`, `se`, `z`, `p` vectors get written |
| `x_total` | The point estimate of each statistic over all the data — length N |
| `var` | The jackknife variance of each statistic — length N |
| `N` | How many statistics there are (how many f3 targets, or f4 quartets) |

The function first sizes and zeroes the four output vectors (`est`, `se`, `z`, `p`) to
length N, then walks the N statistics one at a time. For statistic `k` it reads the
estimate `x_total[k]` and the variance `var[k]`, and computes:

```
se = (var_k > 0) ? sqrt(var_k) : NaN
z  = est / se
p  = f4_two_sided_p(z)
```

then stores all four values into their slots.

A few details worth calling out:

- **The standard error is `sqrt(variance)`**, but only when the variance is strictly
  positive. If the jackknife handed back a zero or negative variance — which means it
  had no real spread to measure (for example, no usable blocks) — the standard error is
  set to `NaN` rather than zero. That is deliberate: `NaN` marks the value as "not
  available" instead of pretending the estimate is infinitely precise. It also makes
  the next line behave sensibly, because `est / NaN` is `NaN`, so a statistic with no
  usable variance propagates a `NaN` z and p rather than a misleading number.

- **The z-score is simply `est / se`** — the estimate divided by its standard error.
  This is the plain definition of a z-score: how many standard errors the estimate is
  away from the null value of zero. A big magnitude means the statistic is far from
  zero relative to its noise (strong signal); a magnitude near zero means it is well
  within the noise (consistent with zero). No sign flip or shift is applied — the sign
  of z just follows the sign of the estimate.

- **The p-value is a two-sided normal-tail probability** of that z, computed by
  `f4_two_sided_p` (defined in `f4.cpp`). That function is `erfc(|z| / sqrt(2))` — the
  complementary error function, which is exactly the total area in both tails of a
  standard normal beyond `±|z|`. "Two-sided" means we don't care about the direction of
  the deviation, only its size, so both tails count. A z of 0 gives p = 1 (completely
  unsurprising), and as `|z|` grows p shrinks toward 0 (increasingly unlikely under the
  null that the true statistic is zero). Taking `|z|` makes the result symmetric, so a
  z of +3 and a z of −3 give the same p.

The write order — est, then se, then z, then p — is preserved verbatim from the
original f3/f4 loops, so the shared helper produces byte-for-byte the same result the
two hand-written copies did.

---

## 3. Why it is a template (`Result`)

The function is templated on `Result` so that one body can serve both `F3Result` and
`F4Result`. Those two structs are nearly the same — both carry `est`, `se`, `z`, and
`p` vectors — and differ only in their index columns (an f3 result labels each row by
its population triple, an f4 result by its quartet of `p1..pK` columns). The read-out
loop only ever touches the four numeric vectors, which both structs share, so writing
it against a template parameter lets f3 and f4 reuse the identical code without the
helper needing to know or care which one it was handed. The index columns are filled in
by the callers, not here.

---

## 4. The `idx()` casts

The loop uses `idx(N)` and `idx(k)` when sizing vectors and indexing. `idx()` (from
`core/internal/index_cast.hpp`) is the project's one-liner that converts a signed
integer to the unsigned `std::size_t` that vector sizes and subscripts expect, in a
single audited place instead of scattering `static_cast<std::size_t>` through the math.
It keeps the index arithmetic readable and keeps the signed→unsigned conversion honest
in exactly one spot. There is nothing statistical about it — it is purely to make the
counting types line up cleanly.

---

# `pchisq.hpp` reference

## 1. Purpose

`src/core/internal/pchisq.hpp` computes one special function: the upper-tail
probability of a chi-squared distribution. Given a test statistic `x` and a number
of degrees of freedom `dof`, `pchisq_upper(x, dof)` returns the probability that a
chi-squared random variable with `dof` degrees of freedom exceeds `x`. This is the
p-value the qpAdm rank test reports.

The header is deliberately minimal in what it depends on. It contains no CUDA code,
is header-only, and uses nothing beyond the C++ standard library (`<cmath>`). That
combination lets it compile into two different places without pulling in the GPU
toolkit: the device-side target (which is where the CPU reference backend's rank
sweep runs) and the core qpAdm orchestrator that drives a run. Keeping it toolkit-free
means the p-value formula lives in exactly one place, and the fit code delegates to
it rather than re-deriving the same math.

All of the arithmetic is native double precision. Double precision is more than
enough here: the rank test's outcome is the *decision* `p > alpha` (does this many
sources fit?), and that decision is not sensitive to the last few bits of `p`. For
that reason the numeric knobs in this file are treated as tunable — they are not
frozen to a bit-exact reference the way the parity-critical parts of steppe are.

---

## 2. Named constants

Three compile-time constants control convergence. They sit at namespace scope rather
than inside the individual functions so that the two different ways of evaluating the
math (see section 5) share one copy and cannot drift apart over time. They are
declared `inline constexpr` because they are compile-time values.

| Constant | Value | What it's for |
|---|---|---|
| `kPchisqMaxIter` | `1000` | The hard cap on loop iterations for both the series form and the continued-fraction form. If a computation has not converged within this many steps, the loop stops anyway and returns its best current value. In practice both forms converge in far fewer steps for any realistic input; this is a safety ceiling, not the expected count. |
| `kPchisqEps` | `1e-15` | The relative convergence tolerance. Each form stops as soon as its next incremental term is smaller than this fraction of the running total — that is, once further terms can no longer meaningfully change the answer at double precision. |
| `kPchisqFpMin` | `1e-300` | A floating-point underflow floor used only by the continued-fraction form. It is roughly the smallest positive number double precision can represent as a normal value. Any intermediate numerator or denominator in the continued fraction that falls below this magnitude is bumped back up to it, which keeps the recurrence from ever dividing by zero. |

None of these three values are held to a bit-exact reference. They are tuned to give
a solidly converged p-value, and could be adjusted without breaking any parity
guarantee, because the rank test depends on the p-value's magnitude relative to a
threshold, not on its exact bits.

---

## 3. The math: upper-tail chi-squared as a regularized incomplete gamma

The upper-tail chi-squared probability is a special case of a more general function,
the *regularized incomplete gamma function*. Concretely:

- P(X > x | dof) — the number this file ultimately returns — equals Q(dof/2, x/2),
  where Q is the **regularized upper incomplete gamma function**.
- Q is defined as `1 − P`, where P is the **regularized lower incomplete gamma
  function**.

So computing the chi-squared upper tail reduces to computing one of these two
incomplete-gamma pieces at the point `a = dof/2`, `x = x/2`.

There is no single formula that evaluates the incomplete gamma accurately everywhere.
Instead there are two classic methods, each accurate on one side of a crossover
point:

- A **power series** converges quickly for the lower piece P when the argument `x`
  is small relative to `a` (specifically `x < a + 1`).
- A **continued fraction** converges quickly for the upper piece Q when `x` is large
  relative to `a` (`x >= a + 1`).

The entry point (section 6) picks whichever method is in its good region and, if it
computed the lower piece, converts to the upper piece by subtracting from 1. This is
the standard Numerical Recipes approach to the incomplete gamma function.

---

## 4. The shared normalizing prefactor

Both incomplete-gamma methods multiply their result by the same normalizing factor:

```
exp(−x + a·log(x) − lgamma(a))
```

`pchisq_gamma_prefactor(a, x)` computes exactly this one expression, and both methods
call it rather than writing the expression out themselves.

The reason this is a shared function and not two copies is subtle but important. The
series method and the continued-fraction method have to compose into one consistent
answer — the series computes the lower piece P, the continued fraction computes the
upper piece Q, and the two are supposed to be complements (`P + Q = 1`) at the
crossover. For that relationship to hold cleanly, both methods must apply *the exact
same* prefactor: the same expression, evaluated in the same operation order with the
same standard-library calls, producing an identical value. Having one function
guarantees that. If the expression were duplicated and one copy were ever reordered
or rewritten, the two halves could subtly disagree. Renaming this function is fine;
changing *what it computes* is not.

---

## 5. The two incomplete-gamma methods

### The series form — `pchisq_gammp_series(a, x)`

Computes the regularized **lower** incomplete gamma P(a, x) by summing a power
series. This is the method used when `x` is small relative to `a` (`x < a + 1`).

It accumulates terms of the series until either the next term is negligibly small
relative to the running sum (using `kPchisqEps`) or the iteration cap
`kPchisqMaxIter` is reached, then multiplies the sum by the shared prefactor from
section 4.

### The continued-fraction form — `pchisq_gammq_cf(a, x)`

Computes the regularized **upper** incomplete gamma Q(a, x) by evaluating a continued
fraction. This is the method used when `x` is large relative to `a` (`x >= a + 1`).

It uses the modified Lentz algorithm, which walks the continued fraction from the top
down while guarding against intermediate values collapsing to zero. That guard is
where `kPchisqFpMin` is used: any numerator or denominator term that drops below that
floor is clamped back up to it, so the algorithm never divides by zero. The loop ends
when a step's multiplicative correction is within `kPchisqEps` of 1 (meaning the
value has stopped moving) or when `kPchisqMaxIter` is hit. The result is the shared
prefactor from section 4 times the accumulated continued-fraction value.

---

## 6. The entry point: `pchisq_upper(x, dof)`

`pchisq_upper` is the only function callers use directly. It returns the upper-tail
chi-squared probability P(X > x | dof) and handles two boundary cases before doing
any real work:

| Input condition | Return value | Why |
|---|---|---|
| `dof <= 0` | `NaN` | There is no valid chi-squared distribution with zero or negative degrees of freedom. This corresponds to the "not applicable" row in the rank-test drop table — for example the last, fully-nested difference where there is nothing left to test — so a not-a-number result is the honest answer rather than a made-up probability. |
| `x <= 0` | `1.0` | A chi-squared variable is always at least zero, so the probability it exceeds a non-positive value is certain. |

For a normal input it sets `a = dof/2` and `xx = x/2`, then dispatches on the
crossover from section 3:

- If `xx < a + 1`, it calls the series form to get the lower piece P and returns
  `1 − P` (converting the lower piece to the upper tail).
- Otherwise it calls the continued-fraction form, which already returns the upper
  piece Q directly.

The result in both branches is the same quantity, Q(dof/2, x/2) — the probability of
seeing a test statistic at least as extreme as `x` under the null hypothesis with
`dof` degrees of freedom.

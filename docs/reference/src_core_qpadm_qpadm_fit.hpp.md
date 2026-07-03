# `qpadm_fit.hpp` reference

## 1. Purpose

`src/core/qpadm/qpadm_fit.hpp` is the header for the single-model qpAdm fit
orchestrator — the code that takes one admixture model (a target population, a set
of source populations on the "left", and a set of reference populations on the
"right") and runs the full fit that produces mixture weights, standard errors, a
fit p-value, and the supporting rank and drop-population tables.

Two properties define this header:

1. **It is host-pure and CUDA-free.** Nothing here issues a matrix multiply, a
   singular-value decomposition, or a Cholesky solve directly. Instead it drives the
   fit through a `ComputeBackend` — an abstract seam that has a CPU implementation
   (used only as a correctness reference) and a GPU implementation (the real
   product). Because this file never touches CUDA, it can be compiled and included
   without pulling in the GPU toolchain.

2. **It declares the shared fit body plus four small supporting functions.** The two
   public `run_qpadm` entry points (one that takes GPU-resident f2 data, one that
   takes host-memory f2 data) live in the public header `steppe/qpadm.hpp`. Those
   public functions are thin wrappers: each picks the backend of the first GPU and
   then calls the shared `run_impl` declared here. This header is where the actual
   fit logic is single-homed so both entry points, and the batched model-space
   search, all run identical math.

The header declares five functions: `default_fit_precision`, `pchisq_upper`,
`honored_tag`, `left_with_target`, and `run_impl`. Each is described below.

---

## 2. What the `qpadm/` directory actually contains

Despite the name, `src/core/qpadm/` is not qpAdm-only. It is the general fit /
search / statistics engine. Alongside the qpAdm orchestrator it also houses:

- the qpGraph family (admixture-graph fitting),
- the standalone f-statistics (f3, f4, the f4-ratio, and the all-quartets sweep),
- the model-space search (fitting many candidate models, and the nested-model
  machinery).

The directory keeps its qpAdm-centric name on purpose. Renaming it to something like
`fit/`, or splitting it into separate `qpgraph/` and `fstats/` directories, was
weighed and deliberately declined because the churn it would create in the version
history was judged not worth it. So the name is a known misnomer, not an oversight —
if you are looking for f4 or qpGraph code, this is the right directory.

---

## 3. The default fit precision (`default_fit_precision`)

```cpp
constexpr Precision default_fit_precision();
```

Returns the one precision setting every stage of the fit starts from: **emulated
double precision at the standard mantissa-bit count** (`EmulatedFp64` with
`kDefaultMantissaBits`, which is 40 bits). This is deliberately the *same* precision
the upstream f2 matrix multiplications use, so the fit and the data it consumes speak
the same numeric language.

### Why it is a function in the header

The value is defined in exactly one place — here — so that every part of the fit
chain refers to a single `(kind, mantissa_bits)` pair and the two numbers can never
drift apart. Both the orchestrator (`qpadm_fit.cpp`) and the batched search
(`model_search.cpp`) call this function rather than writing out the precision by
hand.

It is `constexpr`, which means it is a compile-time value, not a runtime call:
including it in several translation units is safe (no duplicate-definition problem),
and it costs nothing at run time.

### What it does and does not change

This function only folds two identical constructor calls into one named factory. It
does **not** set policy for the whole fit. Some stages cannot honor emulated
precision and fall back to native double precision internally (see section 7 for
which ones and why). The emulated-vs-native value chosen here is frozen to match the
f2 path — treat it as a fixed default, not a knob to tune.

---

## 4. Chi-squared tail probability (`pchisq_upper`)

```cpp
double pchisq_upper(double x, int dof);
```

Computes the upper-tail probability of a chi-squared distribution: the probability
that a chi-squared random variable with `dof` degrees of freedom exceeds `x`. This is
the standard way a fit statistic is turned into a p-value — a large chi-squared
relative to the degrees of freedom yields a small p, meaning the model fits poorly.

It is implemented directly as the regularized upper incomplete gamma function
`Q(dof/2, x/2)`, a deterministic special function. This is an acceptable
own-implementation because the p-value is held only to a loose tolerance — it is a
reported diagnostic, not a quantity that is bit-compared against a reference.

Edge case: if `dof <= 0` the tail probability is undefined and the function returns
`NaN`. The caller treats a non-positive degrees-of-freedom count (an
over-parameterized model) as a domain outcome, not a crash: it still fills in the
weights and rank tables, but flags the result so a consumer filtering on "status is
OK" does not silently accept a NaN p-value.

---

## 5. Honest precision reporting (`honored_tag`)

```cpp
Precision::Kind honored_tag(const Precision& prec, ComputeBackend& be);
```

Returns the precision that **actually ran**, which can differ from the precision that
was *requested*. Every fit result carries a `precision_tag` field so a downstream
consumer knows how the numbers were produced. This function derives that tag
honestly.

The rule: report `EmulatedFp64` only when the request was emulated **and** the
backend can genuinely honor emulation on the covariance matrix multiply; otherwise
report native `Fp64`. The important guarantee is the negative one — a run that
silently fell back to native precision is never labeled as emulated. (Emulated
precision requires a GPU build capability that can be compiled out; when it is
absent, the backend downgrades to native rather than running a rejected fallback,
and this tag must reflect that.)

Like `default_fit_precision`, this is single-homed so that the several fit entry
points — the qpAdm body, the qpWave body, and the standalone-f4 body — cannot drift
apart in how they derive the tag. The standalone-f4 code in a separate file reuses
this one function.

---

## 6. The left-with-target convention (`left_with_target`)

```cpp
std::vector<int> left_with_target(const QpAdmModel& model);
```

Returns the model's target population index prepended to its list of left (source)
population indices — that is, `[target, source_0, source_1, ...]`.

This tiny helper exists to match the convention used by ADMIXTOOLS 2, where the
"left" set passed into the underlying f4 machinery is the concatenation of the target
and the sources. Isolating that convention in a named function keeps the "target goes
first" rule in one place instead of being re-open-coded wherever the combined list is
needed.

---

## 7. The shared fit body (`run_impl`)

```cpp
QpAdmResult run_impl(ComputeBackend& be, F4Blocks&& X,
                     std::span<const int> block_sizes,
                     const QpAdmModel& model, const QpAdmOptions& opts);
```

This is the heart of the header: the shared fit body that every qpAdm entry point
funnels into. It takes an already-assembled block of f4 statistics and runs the rest
of the fit to completion, returning a fully populated `QpAdmResult`.

### Inputs

| Parameter | What it is |
|---|---|
| `be` | The compute backend (CPU reference or GPU). All heavy math is issued through it, never directly. |
| `X` | The assembled f4 statistics for this model, laid out in genome blocks — the input the fit consumes. It is moved in (`F4Blocks&&`) because the fit takes ownership and reuses it across stages. The caller assembles this from either GPU-resident or host-memory f2 data before calling; that assembly step is the only part that differs between the two public entry points. |
| `block_sizes` | The per-block weights for the block jackknife — the number of SNPs (the block lengths) each genome block contributes. Used to weight the uncertainty estimate, matching ADMIXTOOLS 2's jackknife weighting. |
| `model` | The model being fit (target, sources, references, and an optional index tag). |
| `opts` | The per-call options, including the jackknife policy discussed below. |

### What it computes, in order

The fit runs a fixed chain of stages. Naming them by their internal stage labels:

1. **Covariance (the jackknife).** Builds the covariance matrix of the f4 statistics
   by leaving out one genome block at a time and weighting by `block_sizes`. This is
   the matrix-multiply-heavy step that honors the emulated-precision default. If the
   resulting covariance is not usable (not positive-definite), the fit does not throw
   — it returns a result whose `status` records that outcome.
2. **Weights (the generalized least-squares fit).** Solves for the mixture weights
   using an alternating fitting procedure that matches ADMIXTOOLS 2, producing the
   weight vector and the fit chi-squared. A rank-deficient system is again returned as
   a status value, not an exception.
3. **The p-value.** Turns the chi-squared and its degrees of freedom into a fit
   p-value via `pchisq_upper` (section 4). Computed before the standard errors so the
   optional p-value gate below can consult it.
4. **Standard errors (leave-one-out jackknife), conditionally.** Re-fits the weights
   with each genome block left out to estimate the standard error and z-score of each
   weight. Whether this expensive step runs is governed by the jackknife policy.
5. **The rank sweep and drop-population tables.** Sweeps the model over a range of
   ranks and over leaving out one source population at a time, producing the nested
   comparison tables. This runs only if the backend advertises the capability;
   otherwise these fields are simply left empty.

### Domain outcomes are values, not exceptions

A recurring rule across the chain: a bad-but-expected outcome (a non-positive-definite
covariance, a rank-deficient solve, an over-parameterized model with non-positive
degrees of freedom) is reported through the result's `status` field, not thrown. A
caller that fits thousands of candidate models can therefore record-and-continue
instead of aborting the whole batch. A genuine internal error still propagates as an
exception.

### The precision fallback

The fit starts from the emulated-precision default (section 3), but individual stages
fall back where they must:

- The covariance matrix multiply uses emulated precision **when the build and device
  can honor it**, and native double precision otherwise.
- The steps that are prone to catastrophic cancellation — combining the f4 slabs, and
  the centering inside the jackknife — stay in native double precision **always**,
  because emulation can faithfully form a product but cannot recover digits that an
  earlier subtraction has already destroyed.
- The ill-conditioned matrix inverse stays native as well, pending a future emulated
  mode the GPU toolkit does not yet expose.

### The jackknife-SE policy (`opts.jackknife`)

`opts.jackknife` selects *which* models pay for the expensive leave-one-out standard
error. This exists because the standard-error step is by far the costliest part of a
fit, and in a large model search most candidate models are not worth spending it on.

| Policy | Behavior |
|---|---|
| `All` | Always compute the standard errors. This is the default and the behavior the reference goldens were generated under. |
| `FeasibleOnly` | Compute the standard errors only when the cheap point estimate is a "survivor" — meaning the fitted weights are feasible (every non-missing weight lies in `[0, 1]`, with at least one present), and optionally also that the fit p-value clears `opts.p_se_threshold`. |
| `None` | Never compute the standard errors. |

The critical invariant: **the point estimate is identical regardless of policy.** The
weights, p-value, fit rank, feasibility, and the rank-drop and pop-drop tables come
out exactly the same no matter which policy is chosen. Only *which* models also get
standard errors changes.

When a model does not qualify for standard errors, its `se` and `z` fields are left
**empty** — an explicit sentinel meaning "not computed", never a fabricated zero. A
consumer reading a result must therefore treat empty `se`/`z` as "no standard error
available", not as "standard error of zero".

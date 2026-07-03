# `ranktest.hpp` reference

## 1. Purpose

`src/core/qpadm/ranktest.hpp` declares the small host-side layer that drives the
two final steps of a qpAdm / qpWave model fit:

1. **The rank sweep** — deciding how many independent mixing sources the data
   actually support, and how well the model fits at each candidate rank.
2. **The popdrop table** — re-running the fit with one source dropped at a time and
   reporting whether each reduced model is still admissible.

Everything here is plain host code. It contains no GPU calls — no matrix multiply,
no singular-value decomposition, no Cholesky. Instead it hands the heavy numerical
work to a compute backend through a narrow set of virtual functions. That backend
comes in two forms: a CPU implementation used as the reference oracle, and a CUDA
implementation used as the production deliverable. Both are driven through the exact
same functions declared here, so the two paths cannot drift apart.

The file exposes three functions: a thin wrapper around the backend's rank sweep, a
feasibility predicate, and the popdrop orchestrator that ties them together.

---

## 2. Background: the rank sweep and leave-one-source-out

A qpAdm fit works on a matrix of f4 statistics. It has one row per "left"
population (the target plus its candidate sources) and one column per "right"
population (the references, minus one). The method estimates the **rank** of that
matrix — informally, how many genuinely independent mixing components are present —
and then fits admixture weights consistent with that rank.

Two data structures carry the inputs:

- **`F4Blocks`** holds the f4 matrix. It carries the point estimate (`x_total`), the
  per-jackknife-block values (`x_blocks`), and the leave-one-block-out replicates
  (`x_loo`). Its dimensions are `nl` (number of left populations, the row count) and
  `nr` (number of right populations minus one, the column count); the flattened
  matrix has `m = nl * nr` entries.
- **`JackknifeCov`** holds the covariance of those f4 estimates. `Q` is the raw
  `m × m` covariance; `Qinv` is the inverse of the *fudged* covariance (a tiny amount
  added to its diagonal so it is safely invertible). The fit weights f4 residuals by
  `Qinv`, so entries that are noisy or correlated count for less.

The **rank sweep** tries each candidate rank in turn, fits the best rank-`r`
approximation to the f4 matrix, and computes a chi-squared goodness-of-fit for that
rank. From the sweep it picks the smallest rank the data do not reject. The result
comes back as a `RankSweep` struct.

**Leave-one-source-out** (the popdrop table) asks a different question: is the model
still admissible if you remove one of the candidate sources? For the full model and
for each single-source drop, it re-runs the fit on the smaller left set and records
whether the resulting weights are admissible. The result is a list of `PopDropRow`.

---

## 3. `run_rank_sweep` — the rank sweep entry point

```
RankSweep run_rank_sweep(ComputeBackend& be, const F4Blocks& x,
                         const JackknifeCov& cov, double alpha,
                         const QpAdmOptions& opts, const Precision& precision)
```

This is a thin, inlined pass-through. It forwards its arguments straight to the
backend's `rank_sweep` virtual and returns the result unchanged. The wrapper exists
so that callers (and tests using a fake backend) talk to one stable function instead
of reaching into the backend directly.

`alpha` is the significance level for the rank decision — the threshold below which a
rank is considered rejected. It comes from `QpAdmOptions::rank_alpha`, whose default
is `0.05`, matching ADMIXTOOLS 2.

The decision rule itself — which rank the sweep chooses — lives inside the backend's
`rank_sweep`, not here. This function only routes the call.

---

## 4. The popdrop contract

The popdrop table follows exactly what ADMIXTOOLS 2's `drop_pops` routine does, and
the precise mechanics matter because they are easy to get subtly wrong.

The key point: **popdrop does not re-gather f4 statistics and does not re-run the
jackknife.** Recomputing f4 for every dropped source would be enormously more
expensive and would also change the numbers slightly. Instead popdrop reuses the
already-computed full f4 estimate and the already-computed full inverse covariance,
and forms each reduced model by pure index arithmetic:

1. **Keep the rows.** For a given subset of the left sources, keep only the
   corresponding rows of the f4 matrix. The right columns are untouched.
2. **Subset the inverse covariance directly.** Take the block of `Qinv` whose row and
   column indices correspond to the surviving f4 entries — that is, `Qinv[ind, ind]`.
   This is a **sub-block of the existing inverse**, not the inverse of a smaller
   covariance. It is critical that the code does *not* build a smaller `Q` and invert
   it again; inverting a sub-matrix gives a different, wrong answer. The correct
   operation is to slice the already-inverted matrix.
3. **Fit at the reduced rank.** The reduced model is fit at rank equal to the number
   of surviving rows minus one.

Because the reduced inputs are just a smaller `F4Blocks` (the kept rows) and a
smaller `JackknifeCov` (the `Qinv` sub-block), they can be routed through the **same**
rank sweep and weight-solve functions the full model uses. That is what lets the CPU
oracle and the CUDA deliverable share a single code path for both the full fit and
every dropped-source fit.

---

## 5. `run_popdrop` — leave-one-source-out feasibility

```
std::vector<PopDropRow> run_popdrop(ComputeBackend& be, const F4Blocks& x,
                                    const JackknifeCov& cov,
                                    const QpAdmOptions& opts,
                                    const Precision& precision)
```

This orchestrates the whole popdrop table on the host, following the contract in
section 4. It operates on the already-computed full f4 matrix and covariance — it
never re-gathers or re-jackknifes.

It produces one `PopDropRow` for the full model, then one for each single-source
drop. For each row it:

- builds the reduced `F4Blocks` (the kept rows) and reduced `JackknifeCov` (the
  `Qinv` sub-block),
- routes them through the rank sweep to choose a rank, then through the GLS weight
  solve at that rank,
- and records the outcome.

Each `PopDropRow` carries:

| Field | Meaning |
|---|---|
| `pat` | A bit pattern over the left sources, e.g. `"00"`, `"01"`, `"10"`. A `1` marks a dropped source. |
| `wt` | How many sources this row dropped (the number of `1`s in `pat`). |
| `dof`, `chisq`, `p` | The goodness-of-fit at the chosen rank for this reduced model. |
| `f4rank` | The rank the sweep chose for this reduced model. |
| `weight` | The per-source admixture weights. A dropped source's slot holds `NaN`. |
| `feasible` | Whether the surviving weights are admissible (see section 6). |
| `status` | Carries a non-invertible-covariance or rank-deficient condition as a value rather than throwing. |

The numerically delicate parts of these fits run in native double precision (a
deliberate carve-out) even when the rest of the pipeline uses the faster emulated
double-precision mode, because these steps are cancellation-prone and the inverse
covariance is ill-conditioned.

All of this function's own work is host-side index arithmetic; the actual fitting is
delegated to the backend. No GPU code is issued from here.

---

## 6. `popdrop_feasible` — the admissibility predicate

```
bool popdrop_feasible(const std::vector<double>& weights)
```

This decides whether a fitted model's weights are admissible, matching ADMIXTOOLS 2's
`feasible` flag under the setting that forbids negative weights.

The rule: **every surviving weight must lie in the closed interval `[0, 1]`.** A
weight outside that range would mean a source contributes a negative fraction or more
than the whole, which is not a physically meaningful mixture.

Two details:

- **Dropped slots are skipped.** A dropped source's weight is stored as `NaN`; the
  predicate ignores those entries and only checks the surviving weights.
- **A single surviving source is trivially feasible.** When the reduced model has
  collapsed to one source, its weight is `1`, which is in range, so the model is
  admissible by definition.

Note that this in-range check is the feasibility rule specifically; the full qpAdm
options default `allow_negative_weights` to `true`, but the popdrop feasibility flag
always applies the stricter no-negative-weights test.

---

## 7. Host-only, CUDA-free by design

This header is deliberately free of any GPU code. It includes only the backend seam,
the weight-solve wrapper, the precision policy, and the qpAdm options — no CUDA
headers, and it issues no matrix multiply, decomposition, or linear solve itself.

Every heavy numerical step is reached through the compute backend's virtual
functions. That single seam is what makes the CPU reference oracle and the CUDA
production backend interchangeable: the same rank sweep and weight-solve calls drive
both, for the full model and for every dropped-source model, so their results are
held to one shared contract instead of two parallel implementations that could
diverge.

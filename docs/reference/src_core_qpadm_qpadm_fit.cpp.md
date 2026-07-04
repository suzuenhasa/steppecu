# `qpadm_fit.cpp` reference

## 1. Purpose

`src/core/qpadm/qpadm_fit.cpp` is the conductor for a single qpAdm model fit. qpAdm
models one *target* population as a mixture of a small set of *source* populations,
using a larger set of *reference* populations as a fixed backdrop. The fit answers
three questions: what mixture weights best explain the target, how well does that
model actually fit, and how uncertain are the weights.

This file does not do any of the heavy numerical work itself. It is host-side glue
that is deliberately free of GPU code. It drives a fixed sequence of stages, each of
which is implemented elsewhere and reached through a backend interface, and then
packages everything the stages produce into one result record. Its real content is
the *orchestration decisions*: the order the stages run in, which numerical precision
each stage is allowed to use, when to pay for the expensive uncertainty estimate,
and how a "the model doesn't make sense" situation is reported back to the caller
without throwing an exception.

The same machinery serves two closely related tools. qpAdm fits a target against
sources; qpWave asks the simpler question of how many distinct ancestry streams a
set of populations needs, with no target at all. Both run through the same rank
sweep; the only difference is whether a target population is prepended to the input.

---

## 2. The single-model fit pipeline

The core routine is `run_impl`. It receives an already-assembled matrix of f4
statistics (called `X` here) plus the jackknife block sizes, the model definition,
and the run options. It runs the following stages in order and fills one
`QpAdmResult`.

1. **Set up the shape.** From `X` it reads `nl` (the number of left-hand
   populations, meaning the target plus its sources) and `nr` (the number of
   reference populations). The estimated rank `r` defaults to `nl - 1` — one less
   than the number of left populations, which is the natural rank for a mixture of
   that many sources — unless the caller pins a specific rank. The degrees of
   freedom for the fit test are `dof = (nl - r) * (nr - r)`.

2. **Covariance of the statistics.** A block jackknife estimates the covariance of
   the f4 statistics across the genome. Blocks are weighted by their size (the count
   of SNPs in each block), the parity weighting[^at2]. If the resulting covariance
   matrix is not symmetric-positive-definite, the fit cannot proceed; this is
   reported as a status value (see section 7), not an exception.

3. **Weights.** A generalized-least-squares solve produces the mixture weights. This
   uses an alternating-least-squares procedure[^at2]. It yields the
   fitted weights and a chi-squared statistic measuring how far the model is from a
   perfect fit. A rank-deficient system is again reported as a status value.

4. **Fit p-value.** The chi-squared statistic and the degrees of freedom give a
   tail probability — the p-value of the fitted rank. This is computed *before* the
   uncertainty estimate so that the survivor gate in the next step can consult it.
   When `dof` is zero or negative the tail probability is undefined and comes back as
   NaN (see section 7).

5. **Standard-error gate and estimate.** The routine decides whether this particular
   model earns the expensive uncertainty estimate (section 5). If it does, a
   leave-one-out jackknife re-fits the weights with each genome block held out in
   turn and derives standard errors and z-scores. Non-survivors leave the
   standard-error and z-score fields empty rather than filling in fake zeros.

6. **Rank sweep and drop tables.** If the backend supports it, the routine runs a
   sweep over candidate ranks and two nested-model comparison tables (section 6).

7. **Final status.** The status is set last: undefined-chi-squared when `dof <= 0`,
   otherwise Ok.

Note that the assembly of `X` itself (turning the raw f2 block statistics into the
f4 matrix) happens in the caller, not inside `run_impl` — `run_impl` always receives
an already-assembled `X`. See section 9.

---

## 3. Precision policy across the stages

steppe can run its matrix math in an emulated form of double precision that is much
faster than the GPU's native double precision at essentially the same accuracy. The
fit uses **one unified default**: emulated double precision at 40 mantissa bits — the
exact same default the f2 precompute uses. That single default is defined once (in
the header, as `default_fit_precision`) so every part of the fit chain references the
same setting and none of them can drift apart.

The important subtlety is that emulated precision is not appropriate for every stage,
so several stages carve themselves out and always run in native double precision. The
distinction is about *numerical delicacy*, not size:

- **Matrix-multiply-heavy stages use the emulated default.** The covariance
  computation is dominated by a large symmetric matrix multiply, so it engages the
  emulated mode — and falls back to native automatically if the build or device
  cannot honor emulation.

- **Cancellation-prone stages always stay native.** Forming the f4 statistics from
  the underlying pieces, and the centering step inside the jackknife, both involve
  subtracting nearly equal quantities. Emulated arithmetic can faithfully form a
  product, but it cannot recover accuracy that a prior subtraction has already thrown
  away. These stages therefore ignore the requested precision and always use native
  double precision.

- **Ill-conditioned linear algebra stays native.** The symmetric-positive-definite
  matrix inverse used inside the solve is left native as well, pending a future
  emulated mode for that library routine that the toolkit does not yet expose.

Passing the emulated default down to a stage that carves itself out is harmless: the
stage simply ignores it. Doing so keeps the whole fit on one consistent policy rather
than special-casing each call site.

The reference oracle — the CPU backend that the GPU path is validated against —
ignores the precision setting entirely and always runs native double precision.

---

## 4. The honest precision tag

Every result carries a `precision_tag` recording which precision *actually ran*, not
which was requested. This matters because a run can silently downgrade: emulated
precision is only honored when both the request asks for it *and* the backend is
capable of honoring it. If the request is emulated but the build or device cannot do
the emulation, the covariance stage quietly falls back to native double precision.

The `honored_tag` helper computes the truthful tag: emulated only when the request is
emulated *and* the backend reports it can honor emulation, otherwise native. The CPU
oracle, which always runs native, reports native. The rule is deliberately
conservative — a run is never tagged as emulated when it actually ran native. This
derivation lives in one place so that the qpAdm path and the qpWave path (and the
standalone f4 path) cannot compute the tag differently.

---

## 5. The standard-error survivor gate

Computing standard errors is by far the most expensive step, because it re-fits the
weights once per genome block. In a large model search most models are not worth that
cost — a model whose weights are nonsensical will be discarded regardless of how
precise its error bars are. The gate decides, from the cheap point estimate alone,
whether this model is a *survivor* that earns the full jackknife.

The behavior is controlled by the jackknife policy option, which has three settings:

| Policy | Behavior |
|---|---|
| `All` | Always compute the standard error. This is the default and reproduces the reference test fixtures. |
| `FeasibleOnly` | Compute only for survivors — models whose weights are feasible, optionally also requiring the fit p-value to clear a threshold. |
| `None` | Never compute the standard error. |

**Feasibility** is the test of whether the fitted weights make sense as mixture
proportions: every non-missing weight lies between 0 and 1 inclusive, and at least
one weight is present (not all missing). This is the same feasibility test the result
assembler applies, so the two always agree. When the p-gate is also enabled, a
survivor must additionally have a fit p-value at or above the configured threshold.

Whatever the policy, the cheap point estimate — weights, p-value, feasibility, and
the rank and drop tables — is identical. Only *which* models receive standard errors
changes. A non-survivor leaves the standard-error and z-score fields empty (the
sentinel for "not computed"), never a fabricated zero that a downstream consumer
might mistake for a real, tiny error bar.

The leave-one-out re-fit reuses the covariance inverse already computed for the point
estimate rather than recomputing it, and survivors receive the identical full-precision
standard error the reference tool would produce.

---

## 6. The rank sweep and population-drop tables

Beyond the single fit at the chosen rank, qpAdm reports two nested-model comparison
tables. Both reuse the same `X` and covariance already in hand, so neither needs to
recompute anything expensive.

- **Rank sweep (the rank-drop table).** A sweep over candidate ranks `0` through the
  maximum, plus the inferred number of independent ancestry streams. This tells the
  user how many sources the data actually supports, with a nested chi-squared
  comparison between adjacent ranks.

- **Population drop.** A leave-one-source-out table: for each left-hand source, drop
  it and report how the fit changes. This is a pure subsetting operation — it takes
  rows of the already-computed f4 estimate and the matching block of the covariance
  inverse. It does not re-gather statistics from the underlying f2 source and it does
  not re-run the jackknife, which is why it can be produced right here inside
  `run_impl` without access to the original f2 data.

Both tables are produced only when the backend explicitly reports that it supports the
rank sweep, via a capability query. This is an intentional design choice over the
older pattern of wrapping the call in a try/catch and treating any exception as "not
supported." A backend that genuinely does not support the sweep simply leaves these
fields empty — a clean, non-breaking absence. Because the query is explicit, a real
error thrown from *inside* a working sweep now propagates as a genuine fault with a
diagnostic, instead of being silently swallowed and blanking the tables. Both real
backends — the CPU oracle and the CUDA deliverable — support the sweep.

---

## 7. Domain outcomes reported as status values

A recurring principle in this file: a model that is mathematically ill-posed is a
normal *outcome* of the analysis, not a program fault. Such outcomes are returned as
a status field on the result, so a caller filtering a large search on
`status == Ok` naturally skips them, while a caller that wants to inspect them still
can. None of these throw an exception.

| Status | Meaning |
|---|---|
| `NonSpdCovariance` | The jackknife covariance matrix was not symmetric-positive-definite, so the solve cannot proceed. Returned immediately after the covariance stage. |
| `RankDeficient` | The generalized-least-squares system was rank-deficient. Returned immediately after the weights stage. |
| `ChisqUndefined` | The degrees of freedom came out zero or negative (an over-parameterized model), so the chi-squared tail p-value is undefined (NaN). The fit itself still succeeded — weights, chi-squared, and the rank tables are all populated — but the p-value is not meaningful, so this status flags it. This keeps a consumer from silently accepting a NaN p-value as if the model were a clean pass. Normal models (positive degrees of freedom) are unaffected and report Ok. |

---

## 8. qpWave: the same machinery without a target

qpWave uses the exact same rank-sweep machinery as qpAdm, differing in one place: it
has no target population. Where qpAdm prepends the target to the source list to form
the left-hand rows (left = target followed by sources), qpWave treats the supplied
left list *as* the rows directly, with the first entry serving as the reference. So
the number of left rows is one less than in the equivalent qpAdm call.

The qpWave path assembles `X`, computes the covariance, and runs the rank sweep,
then maps the sweep output into a `QpWaveResult` via the small `qpwave_from_sweep`
helper. That helper copies across the rank-sweep and rank-drop fields and sets both
the reported number of ancestry streams and the estimated rank to the sweep's inferred
value. Like the qpAdm path, a non-positive-definite covariance is returned as a status
value with the honest precision tag attached, rather than throwing.

---

## 9. Public entry points and backend selection

The file exposes four public functions — `run_qpadm` and `run_qpwave`, each with two
overloads. The overloads differ only in where the f2 block statistics come from:

- A **device-resident** source (`DeviceF2Blocks`), used on the CUDA path, where the
  assembly reads statistics already sitting in GPU memory with no copy back to the
  host.
- A **host** source (`F2BlockTensor`), used by the CPU oracle, which reads statistics
  from host memory directly.

Each pair of overloads forwards into one shared, templated body (`run_qpadm_impl` /
`run_qpwave_impl`) so the two source types do not duplicate the orchestration logic.
For qpAdm the shared body prepends the target to the sources, assembles `X`, and hands
off to `run_impl`. For qpWave it assembles `X` from the left list directly.

**Backend selection.** All four entry points select the first GPU's backend through a
single accessor, `primary_backend`, which names the otherwise-magic device index via
the constant `kPrimaryGpu` (value `0`). Any multi-GPU fan-out lives *above* this file —
the model-batched search is what drives the other GPUs — so a single fit always runs
on GPU 0. The index constant is kept private to this file because it is a local
convention, not a cross-module tunable.

**Survivor-block subtlety.** The jackknife is run over the block sizes that the
assembly step *kept* in `X`, not the full block list of the original f2 source. If any
genome block had a missing pairwise value it was dropped during assembly, following
the "remove missing" read semantics[^at2]. When nothing is missing the two block
lists are byte-for-byte identical.

**One shared special function.** `pchisq_upper` — the upper-tail chi-squared
probability used for every p-value — is a thin wrapper delegating to a single internal
implementation, so the whole fit chain shares one definition of that function (and one
NaN-for-non-positive-degrees-of-freedom convention).

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

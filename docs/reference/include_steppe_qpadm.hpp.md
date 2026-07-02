# `qpadm.hpp` reference

## 1. Purpose

`include/steppe/qpadm.hpp` declares the value types and entry-point functions
for steppe's qpAdm fit engine — the stage that, given precomputed pairwise
population statistics, fits an admixture model and reports the mixture weights
and how well the model fits.

What qpAdm does, in one sentence: it asks whether a *target* population can be
explained as a mixture of a set of *left* source populations, judged against a
set of *right* outgroup populations, and returns mixture weights that sum to 1
along with goodness-of-fit numbers.

Three design points shape this header:

1. **CUDA-free.** It uses only the C++ standard library, so it can be included
   into the core library, the command-line tool, and the language bindings
   without any of them having to pull in the GPU toolkit. The two GPU types it
   needs (`DeviceF2Blocks` and `Resources`) are only forward-declared here; the
   implementation file includes their real definitions.
2. **Populations are referenced by integer index, never by name.** A model
   points at populations by their position along the population axis of the
   statistics tensor. Turning population names into indices is the caller's
   job (an app or binding concern), not something this compute seam does.
3. **Two ways in for every entry point.** The primary overload reads statistics
   that already live in GPU memory, so nothing is copied back to the host on the
   GPU path. A second overload of each entry point takes the same statistics as
   a plain host-memory tensor — that overload is the reference/parity door, used
   by the test that compares the GPU result against a known-good result computed
   on the CPU.

---

## 2. JackknifePolicy

`JackknifePolicy` controls, during a large model-space search, **which models
pay for the expensive standard-error computation.**

Standard errors are estimated with a block jackknife: the fit is redone many
times, each time leaving out one block of the genome, and the spread of the
answers gives the error bars. That leave-one-out pass is by far the most
expensive part of a fit, so when screening thousands of models you may not want
to pay it for every one.

The key thing to understand is that the **point estimate is identical across all
three modes** — the weights, chi-squared, p-value, feasibility, and the
rank-drop and pop-drop tables come out the same no matter which policy is set.
The policy governs *only* which models get standard errors.

The three values, with their frozen integer codes (these map directly to the
user-facing `--jackknife=0/1/2` option):

| Value | Code | Meaning |
|---|---|---|
| `None` | `0` | No standard error for any model — the fastest pure screen. The standard-error and z-score fields are left empty. |
| `FeasibleOnly` | `1` | Standard errors only for the models that survive the feasibility criterion; the rest are left empty. |
| `All` | `2` | Standard error for every model. **This is the default**, and it matches the original behavior. |

The single-model entry points (`run_qpadm`, `run_qpwave`) ignore this setting —
they always compute the standard error. Only the batched search
(`run_qpadm_search`) consults it.

---

## 3. QpAdmOptions

`QpAdmOptions` is the per-call configuration. The constants that must match
ADMIXTOOLS 2 are named fields here (and recorded in the reference metadata)
rather than bare numbers buried in the code.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `fudge` | `double` | `1e-4` | A tiny ridge constant added for numerical stability. It is applied in the same places ADMIXTOOLS 2 applies it: both alternating-least-squares systems of the weight solve, the final constrained weight normal equations, and the regularization of a matrix inversion. Chosen to match ADMIXTOOLS 2 exactly. |
| `als_iterations` | `int` | `20` | The number of iterations of the alternating-least-squares loop that fits the weights. The weight fit is this iterative loop, not a single solve. Matches ADMIXTOOLS 2's default. |
| `rank` | `int` | `-1` | The rank to fit at. `-1` means "use the default best rank," which is one less than the number of left populations. |
| `allow_negative_weights` | `bool` | `true` | Whether weights are allowed to leave the range `[0,1]`. `true` (the default) matches ADMIXTOOLS 2, which does not clip: a model whose weights fall outside `[0,1]` is "infeasible," and that is a legitimate finding about the data, not an error. |
| `rank_alpha` | `double` | `0.05` | The significance level for the rank decision. The reported best rank is the smallest candidate rank whose model is *not* rejected at this level (its p-value exceeds `rank_alpha`). Matches ADMIXTOOLS 2's default of 0.05. |
| `jackknife` | `JackknifePolicy` | `All` | The standard-error policy for a model-space search (see section 2). Ignored by the single-model entry points, which always compute the standard error. |
| `p_se_threshold` | `double` | `0.05` | Used only in `FeasibleOnly` mode together with the flag below: to earn a standard error, a survivor must have a p-value at least this large. |
| `se_require_p` | `bool` | `false` | Selects the survivor test for `FeasibleOnly` mode. `false` (the default) means feasibility alone decides who gets a standard error; `true` means a model must be *both* feasible *and* have a p-value at least `p_se_threshold`. The default is `false` on purpose: feasibility is the firm qpAdm screen, whereas the p-value boundary is statistically noisy, so a default p-gate would strip standard errors from exactly the feasible-but-marginal models a researcher most needs them for. Turn it on only for an aggressive first-pass survey. |

---

## 4. QpAdmModel

`QpAdmModel` is one model to fit, expressed entirely as integer indices into the
population axis of the statistics — no strings. It is a plain value type, so a
batched search can pass a whole array of them.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `target` | `int` | `-1` | The target population's index. Internally it is prepended to the `left` list to form ADMIXTOOLS 2's convention, where the left set begins with the target followed by the sources. |
| `left` | `vector<int>` | empty | The source population indices. The number of sources is the length of this list. |
| `right` | `vector<int>` | empty | The outgroup population indices. The first entry is the fixed reference outgroup; the remaining entries are the other outgroups. |
| `model_index` | `int` | `-1` | A stable identifier the caller assigns. The search echoes it back on the result, so the caller can match results to inputs regardless of the order they come back in. |

---

## 5. QpAdmResult

`QpAdmResult` is the result of one single-model fit. Two invariants matter
before the field list:

- **Domain outcomes are never exceptions.** A model that turns out
  rank-deficient, or that has a non-positive-definite covariance, records that
  in its `status` field and the fit continues. A search over thousands of models
  must be able to record a failure and move on rather than throwing.
- **An empty standard-error vector is the unambiguous "not computed" signal.**
  steppe never fills in a fake `0` or `NaN`. If `se` is empty, no standard error
  was computed — either the policy said not to, or the model failed. The z-score
  vector `z` follows the same rule and is empty exactly when `se` is empty.

### Core fit outputs

| Field | Type | Meaning |
|---|---|---|
| `weight` | `vector<double>` | The admixture weights, one per source; they sum to 1. |
| `se` | `vector<double>` | Jackknife standard error per weight, same length as `weight` when computed. Empty means "not computed" (see the invariant above). |
| `z` | `vector<double>` | Weight divided by its standard error. Empty exactly when `se` is empty. |
| `p` | `double` | The tail p-value at the fitted rank. |
| `chisq` | `double` | The goodness-of-fit chi-squared at the fitted rank. |
| `dof` | `int` | Degrees of freedom: `(number of left − rank) × (number of right − rank)`. |
| `rank_p` | `vector<double>` | Nested rank-test p-values, one per candidate rank. |
| `est_rank` | `int` | The rank actually used for the reported weights. |

### Rank-test / qpWave table

The fuller rank sweep, describing how the model behaves at each candidate rank
and each step of dropping to a lower rank.

| Field | Type | Meaning |
|---|---|---|
| `rank_chisq` | `vector<double>` | Chi-squared at each candidate rank. |
| `rank_dof` | `vector<int>` | Degrees of freedom at each candidate rank. |
| `f4rank` | `int` | The smallest rank not rejected — the chosen rank (matches ADMIXTOOLS 2's `f4rank`). |
| `rankdrop_f4rank`, `rankdrop_dof`, `rankdrop_dofdiff` | `vector<int>` | The integer columns of the nested rank-drop table: for each row, the rank, its degrees of freedom, and the change in degrees of freedom from the previous step. |
| `rankdrop_chisq`, `rankdrop_p`, `rankdrop_chisqdiff`, `rankdrop_p_nested` | `vector<double>` | The floating-point columns of that table: the chi-squared, its p-value, the change in chi-squared, and the nested-test p-value. Together these mirror ADMIXTOOLS 2's rank-drop table row for row. |

### Pop-drop table

The result of refitting with each subset of sources removed, mirroring
ADMIXTOOLS 2's `popdrop` table.

| Field | Type | Meaning |
|---|---|---|
| `popdrop_pat` | `vector<string>` | The pattern string naming which sources are present in each row. |
| `popdrop_wt`, `popdrop_dof`, `popdrop_f4rank` | `vector<int>` | The integer columns of the pop-drop table, matching ADMIXTOOLS 2's `popdrop`. |
| `popdrop_chisq`, `popdrop_p` | `vector<double>` | The chi-squared and p-value for each row. |
| `popdrop_feasible` | `vector<char>` | A per-row feasibility flag stored as `0`/`1`. It is a `char` rather than a `bool` on purpose, to keep the whole struct CUDA-free and avoid the packed `vector<bool>`. |

### Bookkeeping

| Field | Type | Default | Meaning |
|---|---|---|---|
| `status` | `Status` | `Ok` | The per-model outcome: `Ok`, `RankDeficient`, `NonSpdCovariance`, or `ChisqUndefined`. Never an exception. When `ChisqUndefined`, degrees of freedom is `<= 0` and `p` is a `NaN` sentinel — so filter on `status`, not on `p`. |
| `precision_tag` | `Precision::Kind` | `Fp64` | Which arithmetic produced this result. This engine always uses native double precision today. |
| `model_index` | `int` | `-1` | Echoes the input model's identifier, for deterministic ordering. |

---

## 6. Single-model entry points: run_qpadm

`run_qpadm` fits one model and returns a `QpAdmResult`. It comes in two
overloads:

- **Primary (GPU-first).** Reads statistics that already live in GPU memory (a
  `DeviceF2Blocks`) and runs on the compute backend of the first GPU in the
  supplied resources. Nothing is copied back to the host on this path.
- **Host-oracle / parity.** Takes the same statistics as a plain host-memory
  tensor (`F2BlockTensor`). The CPU reference backend reads host memory
  directly. This is the overload the parity test drives: it uploads a
  known-good statistics tensor as a host tensor and calls this to get the
  reference answer that the GPU path is checked against.

Both overloads are marked so the compiler warns if the returned result is
ignored.

---

## 7. Model-space search: run_qpadm_search

`run_qpadm_search` fits a whole pool of candidate models against the *same*
GPU-resident statistics. It is the batched search behind the model-rotation
feature.

Its contract:

- The models are batched on the GPU and split across the available GPUs.
- Each model is fit *entirely on one GPU* — there is no traffic between GPUs.
- Domain failures are per-model `status` values, never exceptions, so a search
  of thousands of models records-and-continues.
- The returned table is in the caller's **input order**, regardless of how many
  GPUs were used. `results[k]` corresponds to `models[k]`, and each result
  carries the `model_index` that identifies its input, so the ordering is
  deterministic even across different GPU counts.

Like the single-model entry point, it has a host-oracle overload that routes
every model through the CPU reference backend's per-model loop. That is the
bit-exact reference the batched GPU path is compared against.

Both overloads take the models as a `std::span`, so the caller can pass any
contiguous array of models.

---

## 8. qpWave: run_qpwave and QpWaveResult

qpWave is the rank-sufficiency test run *without a target*. Instead of asking
"is this target a mixture of these sources," it asks whether a set of left
populations is consistent with being drawn from some number of independent
sources — the same rank machinery as qpAdm, but with no target prepended. Here
the first left population acts as the reference row.

### QpWaveResult

Holds the per-rank sweep and the chosen rank.

| Field | Type | Meaning |
|---|---|---|
| `rank_chisq`, `rank_dof`, `rank_p` | `vector<double>` / `vector<int>` / `vector<double>` | The chi-squared, degrees of freedom, and p-value at each candidate rank. |
| `rankdrop_f4rank`, `rankdrop_dof`, `rankdrop_dofdiff` | `vector<int>` | The integer columns of the nested rank-drop table (as in `QpAdmResult`). |
| `rankdrop_chisq`, `rankdrop_p`, `rankdrop_chisqdiff`, `rankdrop_p_nested` | `vector<double>` | The floating-point columns of that table. |
| `f4rank` | `int` | The smallest rank not rejected. |
| `est_rank` | `int` | The rank used. |
| `status` | `Status` | The per-run outcome (never an exception). |
| `precision_tag` | `Precision::Kind` | Which arithmetic produced the result (native double precision). |

### run_qpwave

`run_qpwave` comes in the same two overloads as the other entry points: one over
GPU-resident statistics (the primary), and one over a host-memory tensor for the
CPU reference. Its `left` argument is the *full* set of left populations with no
target prepended — its first entry is the reference row — and `right` is the
outgroup set, starting with the fixed reference outgroup.

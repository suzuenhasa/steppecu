# `cmd_rotate.hpp` reference

## 1. Purpose

`src/app/cmd_rotate.hpp` declares the entry point for the `steppe qpadm-rotate`
command. It exposes exactly one function:

```cpp
[[nodiscard]] int run_qpadm_rotate_command(const config::RunConfig& config);
```

The command runs a *rotation*: it fixes one target population and one set of right
(outgroup) populations, takes a pool of candidate left (source) populations, and
then fits many qpAdm models at once — one model for every subset of the pool that
could serve as the left set. It returns a single per-model feasibility table so a
user can see, in one run, which combinations of sources plausibly explain the
target.

The header itself is deliberately thin: a single function declaration plus a
comment describing its contract. The substance lives in that contract — how the
model list is enumerated, how the fit is run, what the output looks like, and
which outcomes are treated as a normal result versus a hard failure.

The function returns a process exit code (a `steppe::config::CliExitCode` value)
rather than exiting the process itself. It owns its own stdout/stderr — the
library it calls never prints — so all user-facing output and all error messages
come from inside this one function.

---

## 2. The rotation shape and how models are enumerated

A rotation is defined by three inputs plus two bounds:

- **target** — the one population being modeled. Required.
- **right** — the fixed set of outgroup populations, shared by every model.
  At least one is required.
- **pool** — the candidate left/source populations. At least one is required.
- **min_sources** / **max_sources** — the smallest and largest left-set sizes to
  enumerate.

The command builds one qpAdm model for every subset of the pool whose size falls
between `min_sources` and `max_sources`. For each chosen subset the target and the
right set stay fixed; only the left set changes.

### Size bounds and their defaults

`min_sources` defaults to `1`. `max_sources` defaults to `-1`, which is the
sentinel meaning "use the whole pool"; internally that is resolved to the pool's
size. `max_sources` is also clamped down to the pool size if a larger number is
passed. If `min_sources` is larger than the pool size there are no models to fit,
and the command reports that as a configuration error rather than producing an
empty table.

### Enumeration order

Subsets are generated smallest-size-first, and within each size in lexicographic
order over the pool's own ordering — that is, all subsets of size `min_sources`
first (each in lexicographic order), then all subsets of size `min_sources + 1`,
and so on. Each model is given a dense, zero-based `model_index` counter in exactly
this order. This ordering is a fixed part of the contract: it is the same order the
reference (golden) generator uses[^at2], so the row `model_index` values line up
one-to-one with the reference rows and can be compared directly.

---

## 3. One batched fit, not many separate runs

Every enumerated model is fit in a **single** call into the search engine, not in a
loop of one-model-at-a-time runs. The whole model list is handed to the
GPU-batched, f2-resident fit engine in one shot. This is the reason the command
exists as its own thing rather than as a shell loop over the single-model qpAdm
command: batching lets the engine keep the shared f2 data resident on the GPU and
fit thousands of models against it without reloading, which is what makes a large
rotation practical.

The command performs no fit math of its own. It resolves population names to
indices, builds the model list, hands everything to the engine, and formats the
results. All of the numerical work — the f2 blocks, the linear algebra, the
jackknife standard errors — happens inside the engine call.

---

## 4. The per-model output table

The command emits one row per fitted model. Each row carries:

- `model_index` — the zero-based position in the enumeration order (see §2).
- `target` — the target population label.
- `left` — the model's source populations (the chosen pool subset).
- `p` — the tail probability from the model's chi-square test.
- `chisq` — the chi-square statistic.
- `dof` — the degrees of freedom.
- `f4rank` — the rank used for the fit.
- `feasible` — whether the model passes the feasibility criteria.
- `status` — the per-model outcome (see §5).
- `weights` / `se` — the fitted source weights and their standard errors.

The table can be written as CSV, TSV, or JSON. Population names are resolved back
from indices to labels only at emit time, so the labels the user sees match the
names they passed in.

---

## 5. Record-and-continue: which outcomes are rows and which are failures

This is the most important and least obvious part of the contract. A rotation has
**no single result** — it has a whole table — so it cannot map one result's status
to the process exit code the way a single-model command does. Instead it uses a
*record-and-continue* rule that splits outcomes into two categories.

### Per-model domain outcomes become a row, and the command still exits 0

If an individual model cannot be fit for a data-driven reason, that is a normal,
expected result for a rotation — the whole point is to sweep many combinations,
some of which will not work. Such an outcome is written into the model's `status`
column and the command still exits successfully. These per-model outcomes include:

- **rank deficient** — the model's system did not have full rank.
- **non-SPD covariance** — the covariance matrix was not symmetric positive
  definite.
- **chi-square undefined** — the chi-square statistic could not be computed.

None of these stop the run or change the exit code. They arrive as row statuses,
never as thrown errors.

### Only true faults return a nonzero exit code

A fault is a problem with the run as a whole, not with one model. These return a
nonzero exit code:

| Condition | Exit code |
|---|---|
| Bad names, missing/empty required inputs, an empty enumeration (invalid config) | `2` (`kExitInvalidConfig`) |
| GPU ran out of memory | `3` (`kExitDeviceOom`) |
| File or format error reading the f2 data | `4` (`kExitIoError`) |
| Any other runtime/CUDA error | `5` (`kExitRuntimeError`) |
| Success (including a table full of per-model domain statuses) | `0` (`kExitOk`) |

Concretely: if the target/right/pool inputs are empty or name populations that do
not exist, or the size band enumerates zero models, the command fails up front with
the invalid-config code and prints a message to stderr. If the device work
(building resources, uploading f2 blocks, or running the search) throws, that is a
fault and the caught error is mapped to the matching nonzero code. Everything else
— including a run where every single model came back rank-deficient — is a
success with exit code 0 and a fully populated table.

Requesting two or more GPUs is neither a fault nor silently accepted: the command
prints a warning that the rotation runs single-GPU-preferred and continues.

---

## 6. Layering and reuse

The header includes only `core/config/run_config.hpp` and pulls in no CUDA header
at all. This is intentional and enforced: the command is application-layer code
that must stay free of GPU headers. It reaches the GPU only through CUDA-free seams
— the resource builder, the f2-block uploader, and the search engine — each of
which is a plain C++ interface that hides the CUDA code behind it. This is the same
layering the single-model qpAdm command follows.

The command also reuses that single-model command's machinery wholesale: the same
f2-directory loader, the same population-name resolver, the same
build-resources-then-upload chain, and the same result-formatting primitives. The
only genuinely new logic in the rotation is the pool-subset enumerator (§2) and the
per-model table emit (§4). No fit math and no output formatting is duplicated.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

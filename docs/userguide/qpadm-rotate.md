# qpadm-rotate

Fit many competing qpAdm models — every subset of a source pool — against one target in a single batched GPU pass.

## What it does

Plain `qpadm` fits *one* model: you name a target, a fixed left set of sources, and a right outgroup set, and it tells you the mixture weights and how well that one model fits. `qpadm-rotate` instead takes a *pool* of candidate sources and automatically enumerates every subset of that pool (within your size bounds), fitting each subset as its own qpAdm model for the same target and the same right set. You get one row per candidate model, so you can rank them by feasibility and p-value and see which source combination best explains the target.

Reach for this when you don't already know which sources to use and want to let the data choose — the classic "model rotation" workflow. Because every model is fit against the *same* GPU-resident statistics, steppe batches them and splits them across the available GPUs, so screening a whole pool costs far less than running `qpadm` once per subset by hand. A pool of N sources produces up to 2^N − 1 models (e.g. 5 sources → 31 models), all in one run.

Like every f2-path command, it reads a precomputed `f2_blocks` directory. Build one first with [extract-f2](./extract-f2.md).

## Flags

| flag | what it does | default |
|---|---|---|
| `--f2-dir TEXT` | Path to the `f2_blocks` directory built by `extract-f2`. Required. | — |
| `--target TEXT` | The target population label — the population you're trying to explain as a mixture. Required. | — |
| `--pool TEXT ...` | The source pool. steppe enumerates subsets of this list and fits each subset as a model's left/source set. Space- or comma-separated list of labels. Required. | — |
| `--right TEXT ...` | The right outgroup labels, shared by every model in the run. The first entry is the fixed reference outgroup. Required. | — |
| `--min-sources INT` | Smallest subset size to enumerate (sources per model). | `1` |
| `--max-sources INT` | Largest subset size to enumerate. `-1` means "up to the whole pool." | `-1` |
| `--fudge FLOAT` | Ridge constant added for numerical stability, matching the ADMIXTOOLS 2 value. Leave it unless you're chasing a specific parity detail. | `1e-4` |
| `--als-iters INT` | Number of alternating-least-squares iterations in the weight fit. | `20` |
| `--rank INT` | The f4 rank to fit at. `-1` auto-selects (one less than the number of left populations). | `-1` |
| `--rank-alpha FLOAT` | Significance level for the rank decision; the chosen rank is the smallest one not rejected at this level. | `0.05` |
| `--allow-neg` / `--no-allow-neg` | Whether weights may fall outside `[0,1]`. On by default (matching the reference): a model with out-of-range weights is reported as *infeasible*, which is a real finding, not an error. Pass `--no-allow-neg` to forbid it. | on |
| `--jackknife INT` | Standard-error policy for the batch: `0` = none (fastest screen), `1` = SEs only for feasible survivors, `2` = SEs for every model. Mode `2` (all-model SE) is specific to `qpadm-rotate`. The point estimates (weights, chi-squared, p, feasibility) are identical across all three — this only governs which models get error bars. | `2` |
| `--p-se-threshold FLOAT` | Used only with `--jackknife 1`: a survivor must have a p-value at least this large to earn a standard error. | `0.05` |
| `--se-require-p` | With `--jackknife 1`, require a survivor to be *both* feasible *and* have p ≥ `--p-se-threshold` before it gets a standard error. Off by default (feasibility alone decides). | off |
| `--out TEXT` | Write results to this file. Omit to print to stdout. | stdout |
| `--format TEXT` | Output format: `csv`, `tsv`, or `json`. | `csv` |
| `--device TEXT` | CUDA device(s): `auto`, a single ordinal like `0`, or two ordinals like `0,1`. GPU-only — there is no `cpu` device. | `auto` |
| `--precision TEXT` | Matmul precision: `emu40`, `emu32`, `fp64`, or `tf32`. The `emu40` default is the parity-safe choice. | `emu40` |
| `--config TEXT` | **Reserved — not yet supported** (passing one currently errors). | — |

`--f2-dir`, `--target`, `--pool`, and `--right` are the required inputs; everything else has a working default.

## Examples

31 competing models (every non-empty subset of a 5-source pool) for one target, CSV to stdout:

```
steppe qpadm-rotate \
  --f2-dir /path/to/f2_30 \
  --target Sweden_Viking \
  --pool Russia_Samara_EBA_Yamnaya,Turkey_N,Serbia_IronGates_Mesolithic,France_Yonne_N,Czechia_EBA_CordedWare \
  --right Mbuti,Han,Papuan,Karitiana,Israel_Natufian,Iran_GanjDareh_N \
  --min-sources 1 --max-sources -1 \
  --format csv --device 0
```

Expect up to 2^5 − 1 = 31 rows, one per source subset, each with weights, chi-squared, p-value, and a feasibility flag.

Screen only 2- and 3-source models, and skip standard errors for a faster first pass:

```
steppe qpadm-rotate \
  --f2-dir /path/to/f2_30 \
  --target Sweden_Viking \
  --pool Russia_Samara_EBA_Yamnaya,Turkey_N,Serbia_IronGates_Mesolithic,France_Yonne_N,Czechia_EBA_CordedWare \
  --right Mbuti,Han,Papuan,Karitiana,Israel_Natufian,Iran_GanjDareh_N \
  --min-sources 2 --max-sources 3 \
  --jackknife 0 --format csv --device 0
```

The `--jackknife 0` run leaves the standard-error columns empty (an empty SE means "not computed," never a fake `0`); rerun with `--jackknife 1` to add error bars only to the models that pass feasibility.

## Gotchas

- **Build the f2-dir first.** `qpadm-rotate` reads precomputed `f2_blocks`; it does not read genotypes. See [extract-f2](./extract-f2.md).
- **Every label must exist in the f2-dir.** The `--target`, `--pool`, and `--right` labels are all resolved against the populations present in `--f2-dir`. A typo or a population you never extracted will fail the lookup.
- **The pool grows fast.** Models scale as 2^N with pool size N (before applying `--min-sources`/`--max-sources`). A 10-source pool with no bounds is 1023 models. Use the size bounds to keep a large pool tractable.
- **`--max-sources -1` means "whole pool," not "zero."** It's the default; set a positive cap to limit subset size.
- **`--jackknife 2` (all-model SE) is the default and is rotate-only.** It gives every model error bars but is the most expensive mode — the leave-one-out jackknife is the costly part of each fit. For a wide first screen, drop to `1` (feasible survivors only) or `0` (none). The weights and p-values don't change; only which rows get SEs.
- **Infeasible is a result, not a failure.** With the default `--allow-neg`, a model whose weights leave `[0,1]` is reported as infeasible rather than dropped. Filter on the feasibility flag, not on the presence of a row.
- **Filter on status, not on p.** A model that came out rank-deficient or with an undefined chi-squared records that in its status field and reports a `NaN` p-value sentinel — don't treat that as a low p-value.
- **`emu40` is the parity default.** Change `--precision` only if you know why; the other modes trade parity for speed.

## See also

- [extract-f2](./extract-f2.md) — build the `f2_blocks` directory this command reads.
- [qpadm](./qpadm.md) — fit a single, fixed model when you already know the sources.
- [qpwave](./qpwave.md) — the target-free rank-sufficiency test that shares the same machinery.
- `docs/reference/include_steppe_qpadm.hpp.md` — the fit-engine internals: `JackknifePolicy`, `QpAdmOptions`, and the batched `run_qpadm_search` behind rotation.

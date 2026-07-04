# qpadm

Fit an admixture model: can a **target** population be described as a mixture of a set of **left** source populations, judged against a set of **right** outgroups?

## What it does

`qpadm` answers one question at a time: *is my target a blend of these sources?* You give it a target label, a list of left (source) labels, and a list of right (outgroup) labels. It estimates the mixture **weights** — one per source, summing to 1 — and reports how well that model actually fits the data.

Along with the weights you get error bars (`se`), z-scores (`z`, weight ÷ its se), a chi-squared goodness-of-fit, and a **p-value**. A high p-value means the model is consistent with the data; a very low one means the data reject this particular mixture. A model is also flagged **feasible** or not: feasible means every weight lands in `[0,1]` (a real mixture), and infeasible means at least one weight went negative or above 1 — a legitimate finding that the target is *not* this blend, not an error. On top of the single fit, `qpadm` returns two diagnostic tables: a **rank-drop** table (how the fit behaves as you lower the model rank) and a **pop-drop** table (refit with each subset of sources removed, so you can see whether a source is actually pulling its weight). It reproduces ADMIXTOOLS 2's qpAdm numerically.

Reach for `qpadm` when you have a specific hypothesis to test — one target, a named source list. To race many competing source combinations at once, use [`qpadm-rotate`](./qpadm-rotate.md) instead.

`qpadm` reads a precomputed **f2-dir** (pairwise-population statistics). Build one first with [`extract-f2`](./extract-f2.md).

## Flags

| flag | what it does | default |
|---|---|---|
| `--f2-dir TEXT` | The f2_blocks directory to read (`f2.bin` + `pops.txt` + `meta.json`), as produced by `extract-f2`. Required. | — |
| `--target TEXT` | The target population label — the population you're trying to explain as a mixture. Required. | — |
| `--left TEXT ...` | The left source population labels, comma- or space-separated. These are the ingredients of the proposed mixture. Required. | — |
| `--right TEXT ...` | The right outgroup labels, comma- or space-separated. `right[0]` is the fixed reference outgroup (R0); order matters. Required. | — |
| `--fudge FLOAT` | Tiny ridge constant added for numerical stability in the weight solves and matrix inversions. Matches the reference value; leave it unless you know why you're changing it. | `1e-4` |
| `--als-iters INT` | Number of alternating-least-squares iterations used to fit the weights (the weight fit is an iterative loop, not a single solve). | `20` |
| `--rank INT` | The f4 rank to fit at. `-1` means auto: use `(number of left sources) − 1`, the standard qpAdm rank. | `-1` |
| `--rank-alpha FLOAT` | Significance level for the rank decision. The chosen rank is the smallest candidate whose p-value exceeds this. | `0.05` |
| `--allow-neg` / `--no-allow-neg` | Whether weights may fall outside `[0,1]`. On (the default) means weights are never clipped — an out-of-range weight makes the model *infeasible*, which is a real result. `--no-allow-neg` turns clipping-style restriction on. | on |
| `--jackknife INT` | Standard-error policy: `0` = none (fastest, no `se`/`z`), `1` = compute SE only for feasible survivors, `2` = compute SE for everything. Governs *only* which models get error bars; the weights and p-value are the same regardless. (Mainly relevant to searches; a single fit here computes SE.) | `2` |
| `--p-se-threshold FLOAT` | Used only with `--jackknife 1`: the p-value gate a survivor must clear to earn a standard error. | `0.05` |
| `--se-require-p` | With `--jackknife 1`, require a survivor to be *both* feasible *and* have p ≥ `--p-se-threshold` before computing its SE. Off by default on purpose — feasibility is the firm screen, the p boundary is noisy. | off |
| `--out TEXT` | Write results to this file. Omit to print to stdout. | stdout |
| `--format TEXT` | Output format: `csv`, `tsv`, or `json`. | `csv` |
| `--device TEXT` | CUDA device(s): `auto`, a single ordinal like `0`, or two like `0,1`. GPU only — there is no `cpu` device. | `auto` |
| `--precision TEXT` | Matmul precision: `emu40`, `emu32`, `fp64`, or `tf32`. The default `emu40` is the parity-grade emulated-FP64 path. | `emu40` |
| `--config TEXT` | **Reserved — not yet supported** (passing one currently errors). | — |

## Examples

Fit a single two-source model with full jackknife error bars, JSON output, on GPU 0:

```
steppe qpadm --f2-dir /path/to/steppe_f2 \
  --target Czechia_EBA_CordedWare \
  --left Russia_Samara_EBA_Yamnaya,Turkey_N \
  --right Mbuti,Russia_UstIshim_IUP,Russia_Kostenki_UP,Russia_Malta_UP,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian \
  --jackknife 2 --format json --device 0
```

Expect a JSON block with the two weights (summing to 1), their standard errors and z-scores, the chi-squared and p-value, plus the rank-drop and pop-drop tables. A high p-value means the CordedWare-as-Yamnaya+Anatolian model is consistent with the data.

Same fit against a larger f2-dir, using a leaner outgroup set. Confirm your labels first with `cat /path/to/f2dir/pops.txt`:

```
steppe qpadm --f2-dir /path/to/2m_f2_700 \
  --target Czechia_EBA_CordedWare \
  --left Russia_Samara_EBA_Yamnaya,Turkey_N \
  --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian \
  --jackknife 2 --format json --device 0
```

## Gotchas

- **The right set is ordered.** `right[0]` is the fixed reference outgroup (R0). Reordering the right list changes the parametrization; keep it stable when comparing models.
- **Feasible ≠ good p-value, and vice versa.** Feasibility only checks that the weights are a real mixture (in `[0,1]`). A model can be feasible but fit poorly (low p), or infeasible but have a decent p. Read both.
- **Infeasible is not an error.** With `--allow-neg` on (the default), a negative weight is reported, not clipped — that's the tool telling you the target isn't this blend.
- **`--jackknife 0` leaves `se` and `z` empty on purpose.** Empty means "not computed," never a fake `0`. Don't read an empty SE as zero error.
- **Filter on status, not on p.** If a model comes back with an undefined chi-squared, its degrees of freedom are `≤ 0` and `p` is a NaN sentinel. The status field is the reliable signal.
- **You need an f2-dir first.** `--f2-dir` must point at a directory built by `extract-f2`. Check the population labels in its `pops.txt` — your `--target`/`--left`/`--right` labels must match exactly.
- **GPU only.** There is no CPU device; `--device` takes `auto` or GPU ordinals.

## See also

- [`extract-f2`](./extract-f2.md) — build the f2-dir this command reads.
- [`qpadm-rotate`](./qpadm-rotate.md) — race many competing source combinations for one target in a single batched run.
- [`qpwave`](./qpwave.md) — the rank-sufficiency test without a target.
- [`docs/reference/include_steppe_qpadm.hpp.md`](../reference/include_steppe_qpadm.hpp.md) — the fit engine's types, jackknife policy, and result-table internals.

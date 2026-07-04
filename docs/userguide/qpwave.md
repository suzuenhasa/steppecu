# qpwave

Tests how many independent source lineages a set of populations needs — the rank-sufficiency test, run with no target.

## What it does

`qpwave` asks a different question than [qpadm](./qpadm.md). qpAdm takes a *target* population and asks "can this target be modeled as a mixture of these sources?" qpWave drops the target entirely. It takes a set of **left** populations and a set of **right** outgroups and asks: how many independent source lineages does the left set actually need to explain its relationship to the outgroups? In other words, it estimates the *rank* of the f4-matrix between the left and right sets.

There is no target here, and no admixture weights. The first entry of `--left` acts as the **reference row** for the rank machinery — it plays the role that the target plays in qpAdm, but it is just an ordinary member of the left set, not a population you are modeling. The command sweeps candidate ranks and reports, for each, a chi-squared statistic, degrees of freedom, and a p-value, then picks the smallest rank the data does not reject. Reach for `qpwave` when you want to know whether two or more populations form a "clade" with respect to the outgroups (rank 0 / rank 1 sufficiency), or how many streams of ancestry a group of populations requires, before you commit to a specific qpAdm mixture model.

Under the hood it reuses the same rank test as qpAdm, matching ADMIXTOOLS 2 numerically. Like the other f2-path commands, it reads a precomputed `f2_blocks` directory built by [extract-f2](./extract-f2.md) — build that first.

## Flags

| flag | what it does | default |
|---|---|---|
| `--f2-dir TEXT` | The `f2_blocks` directory produced by [extract-f2](./extract-f2.md). Required — this is the source of all statistics. | — |
| `--left TEXT ...` | The left population set. **`left[0]` is the reference row** (the population the rank test is anchored on). Give the full set with no target prepended. | — |
| `--right TEXT ...` | The right outgroup labels. The first entry is the fixed reference outgroup; the rest are the other outgroups. | — |
| `--fudge FLOAT` | A tiny ridge constant added for numerical stability, applied in the same places ADMIXTOOLS 2 applies it. Leave it alone unless you are chasing exact parity with a specific run. | `1e-4` |
| `--als-iters INT` | Iteration count for the alternating-least-squares solve inside the fit. | `20` |
| `--rank INT` | The f4 rank to fit. `-1` means auto — sweep ranks up to `nl-1` (number of left pops minus one). | `-1` |
| `--rank-alpha FLOAT` | Significance level for the rank decision: the smallest rank whose p-value is not below this is chosen. | `0.05` |
| `--allow-neg` / `--no-allow-neg` | Whether the internal weights may leave `[0,1]`. On by default, matching ADMIXTOOLS 2 (it does not clip). | on |
| `--jackknife INT` | Standard-error policy: `0` = none, `1` = feasible-only, `2` = all (rotate only). | `2` |
| `--p-se-threshold FLOAT` | With `--jackknife 1`, the p-value gate a model must clear to be a "feasible" survivor that gets a jackknife SE. | `0.05` |
| `--se-require-p` | With `--jackknife 1`, additionally require p >= `--p-se-threshold` for the survivor. | off |
| `--out TEXT` | Output file. Writes to stdout if omitted. | stdout |
| `--format TEXT` | Output format: `csv`, `tsv`, or `json`. | `csv` |
| `--device TEXT` | CUDA device(s): `auto`, a single ordinal like `0`, or two ordinals like `0,1`. GPU-only — there is no `cpu` mode. | `auto` |
| `--precision TEXT` | Matmul precision: `emu40`, `emu32`, `fp64`, or `tf32`. The default `emu40` is the accuracy-safe choice for the f2 GEMMs. | `emu40` |
| `--config TEXT` | **Reserved — not yet supported** (passing one currently errors). | — |

## Examples

Basic rank sweep — is this set of three populations consistent with a small number of sources, given nine outgroups? `left[0]` (here `Czechia_EBA_CordedWare`) is the reference row.

```
steppe qpwave \
  --f2-dir /workspace/data/haak/steppe_f2 \
  --left Czechia_EBA_CordedWare,Turkey_N,Russia_Samara_EBA_Yamnaya \
  --right Mbuti,Russia_UstIshim_IUP,Russia_Kostenki_UP,Russia_Malta_UP,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian \
  --format csv --device 0
```

Expect one row per candidate rank (0 up to `nl-1`), each with its chi-squared, degrees of freedom, and p-value, plus the chosen rank. A rank-1 fit that is not rejected means the left set behaves like a two-source system relative to the outgroups.

Write JSON to a file and add feasible-only jackknife standard errors:

```
steppe qpwave \
  --f2-dir /workspace/data/haak/steppe_f2 \
  --left Czechia_EBA_CordedWare,Turkey_N,Russia_Samara_EBA_Yamnaya \
  --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian \
  --jackknife 1 --p-se-threshold 0.05 \
  --format json --out qpwave_result.json --device 0
```

## Gotchas

- **No target.** Unlike qpAdm, you do not pass a target. Everything goes in `--left`, and `left[0]` is just the reference row the rank test anchors on — not a population you are "modeling." If you are used to qpAdm, do not prepend a target here.
- **Order matters.** `--left` is ordered (first = reference row) and `--right` is ordered (first = fixed reference outgroup). Reordering changes what anchors the test.
- **Build the f2-dir first.** `--f2-dir` must point at a real `f2_blocks` directory from [extract-f2](./extract-f2.md). qpwave does not read genotypes directly.
- **GPU only.** `--device` never accepts `cpu`. Use `auto` or a device ordinal.
- **`--rank -1` is the sweep.** The default auto value sweeps ranks; passing a fixed `--rank` pins a single rank instead of letting the command find the smallest sufficient one.
- **Jackknife SEs are on by default** (`--jackknife 2`). Pass `--jackknife 0` to skip them for a faster run. The `--p-se-threshold` / `--se-require-p` gates only apply when `--jackknife 1`.

## See also

- [qpadm](./qpadm.md) — the same rank machinery, but with a target and admixture weights.
- [extract-f2](./extract-f2.md) — builds the `f2_blocks` directory qpwave reads.
- `docs/reference/include_steppe_qpadm.hpp.md` — internals: `run_qpwave`, `QpWaveResult`, and the rank-drop table semantics.

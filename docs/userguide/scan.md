# scan

Guided proxy/model scanner: search for a better source model for a target, with a gated, best-first ranking — and, optionally, a relatedness shortlist, swap suggestions, and an outgroup-admissibility check.

## What it does

`qpadm-rotate` fits *every* subset of a pool and hands you the raw table; you rank it yourself. `scan` is the next step up: it runs a **guided search** over the source pool, applies a principled **objective** to every fit, and returns the models **best-first** with a clear winner marked — plus tools to understand *why* a model works or fails.

Four things, layered:

1. **A gated, ranked objective.** A model is a *candidate* only if it passes a hard gate — the fit succeeded, every weight is in `[0,1]` (**feasible**), and the tail p-value clears `--p-min`. Survivors are then ranked by **parsimony** (fewest sources, striking models whose extra source isn't statistically earned) → **weight stability** → **leave-one-out robustness**. Ranking is **never** by p-value magnitude (a higher p is not a better model). The single best survivor is marked `selected` — and always labelled *selected, not confirmed*.
2. **A guided search.** Instead of brute-forcing all 2^N subsets, `scan` grows models greedily or with a beam, climbing toward feasibility and stopping at the smallest feasible model. On a small pool it just runs exhaustively (so you lose nothing); on a big pool it fits far fewer models.
3. **A relatedness shortlist and swap suggestions** (`--prerank`, `--suggest-swaps`) — "which sources are closest to my target?" and "this model failed; drop *this* source, add *that* one."
4. **An outgroup-admissibility check** (`--right-search`) — "do my outgroups actually have the power to tell my sources apart?" This is the identifiability question qpAdm quietly depends on.

Like every f2-path command, it reads a precomputed `f2_blocks` directory. Build one first with [extract-f2](./extract-f2.md).

## Flags

### Core

| flag | what it does | default |
|---|---|---|
| `--f2-dir TEXT` | Path to the `f2_blocks` directory. Required. | — |
| `--target TEXT` | The population you're trying to model as a mixture. Required. | — |
| `--pool TEXT ...` | Candidate source pool. Comma/space-separated labels. Required. | — |
| `--right TEXT ...` | Right outgroup labels; the first is the fixed reference (R0). Required. | — |
| `--min-sources INT` | Smallest model (sources) to consider. | `1` |
| `--max-sources INT` | Largest model to consider; `-1` = up to the whole pool. | `-1` |
| `--p-min FLOAT` | The objective's hard-gate tail-p cutoff (α). A model must have p ≥ this to pass. **Not** a multiplicity-corrected threshold — see Gotchas. | `0.05` |
| `--allow-clade` / `--no-allow-clade` | May a **1-source** ("target is a clade of X") model be crowned the winner? On by default. `--no-allow-clade` still shows 1-source models but only ever selects a genuine ≥2-source mixture. | on |

### Search (Phase 1)

| flag | what it does | default |
|---|---|---|
| `--strategy TEXT` | `greedy` (width 1), `beam` (width N), or `exhaustive`. Small pools auto-run exhaustively regardless. | `beam` |
| `--beam-width INT` | Beam width for `--strategy beam`. | `3` |
| `--base TEXT ...` | Optional seed model (sources) to grow the search from, instead of starting from every single source. | — |
| `--sure` | Proceed with a huge **explicit** `--strategy exhaustive` enumeration (lifts the safety cap). | off |

### Relatedness + swaps (Phase 2)

| flag | what it does | default |
|---|---|---|
| `--prerank` | Instead of searching, print the pool ranked by mean outgroup-f3 relatedness to the target (`mean_k f3(right_k; target, X)`), then exit. A "who's closest" shortlist. | off |
| `--suggest-swaps` | For every model that **fails** the gate, suggest dropping the least-related source and adding the most-related unused one, then refit the swap and report whether it works (`swap_*` columns). | off |

### Outgroup admissibility (Phase 3)

| flag | what it does | default |
|---|---|---|
| `--right-search TEXT` | `none`, `check` (does the seed right set distinguish the selected model's sources?), or `add-drop` (find the minimal sufficient outgroup set from `--right-pool`, R0 pinned). | `none` |
| `--right-pool TEXT ...` | Curated outgroup pool `add-drop` may draw from. R0 (`--right[0]`) always stays pinned. | — |

### Shared fit + output options

| flag | what it does | default |
|---|---|---|
| `--rank-alpha FLOAT` | Significance level for the rank tests (the parsimony strike and the `--right-search` admissibility gate). Distinct from `--p-min`. | `0.05` |
| `--fudge` / `--als-iters` / `--rank` / `--allow-neg` / `--jackknife` / `--p-se-threshold` / `--se-require-p` | The same qpAdm fit knobs as [qpadm-rotate](./qpadm-rotate.md). | — |
| `--out TEXT` | Write to this file (stdout if omitted). | stdout |
| `--format TEXT` | `csv`, `tsv`, or `json`. | `csv` |
| `--device TEXT` | CUDA device (single ordinal; multi-GPU is parked). | `auto` |
| `--precision TEXT` | Matmul precision: `emu40` (parity default) / `emu32` / `fp64` / `tf32`. | `emu40` |

## The output table

One row per model the search fit, best-first. Key columns:

| column | meaning |
|---|---|
| `p` | qpAdm tail p-value of the fit. Reported, **not** the ranking key. |
| `feasible` | weights all in `[0,1]` (ignores p). |
| `passes` | the **full** gate: `status ok` AND `feasible` AND `p ≥ --p-min`. The header census counts these. |
| `selected` | the single best survivor (respecting `--allow-clade`). Always *selected, not confirmed*. |
| `over_param` | the within-model rank-drop test says the top source isn't statistically earned (over-specified). |
| `min_abs_z` | smallest \|z\| across interior weights — a stability signal; `NA` for a degenerate/1-source model. |
| `f4rank` | the data-driven fit rank. |
| `swap_drop` / `swap_add` / `swap_feasible` / `swap_p` | present with `--suggest-swaps`: the suggested drop/add and whether the refit passes. |

## Examples

The examples use the bundled 10-population example the installer stages at `~/.local/share/steppe/example`.

**Basic scan** — model Bell Beaker from a pool of candidate sources:

```
steppe scan --f2-dir ~/.local/share/steppe/example \
  --target England_BellBeaker \
  --pool Czechia_EBA_CordedWare,Turkey_N,Iran_GanjDareh_N,Israel_Natufian \
  --right Mbuti,Han,Papuan,Karitiana
```

**Prefer a genuine mixture** (don't crown a 1-source clade):

```
steppe scan --f2-dir ~/.local/share/steppe/example \
  --target England_BellBeaker --pool Czechia_EBA_CordedWare,Turkey_N \
  --right Mbuti,Han,Papuan,Karitiana --no-allow-clade
```

**Guided search on a big pool** — beam vs greedy fit far fewer models than exhaustive (`--strategy exhaustive` for the complete enumeration; small pools auto-exhaust anyway):

```
steppe scan --f2-dir f2_dir --target England_BellBeaker \
  --pool <30+ candidate sources> --right Mbuti,Han,Papuan,Karitiana \
  --max-sources 3 --strategy beam --beam-width 3
```

**Relatedness shortlist** — which sources are closest to the target:

```
steppe scan --f2-dir ~/.local/share/steppe/example \
  --target England_BellBeaker \
  --pool Czechia_EBA_CordedWare,Turkey_N,Iran_GanjDareh_N,Israel_Natufian \
  --right Mbuti,Han,Papuan,Karitiana --prerank
```

**Swap suggestions** — for failing models, "drop the culprit, add a related source" (refit-verified):

```
steppe scan --f2-dir ~/.local/share/steppe/example \
  --target England_BellBeaker \
  --pool Czechia_EBA_CordedWare,Turkey_N,Iran_GanjDareh_N,Israel_Natufian,Han \
  --right Mbuti,Papuan,Karitiana --min-sources 2 --max-sources 2 --suggest-swaps
```

**Outgroup admissibility** — do the outgroups distinguish the selected model's sources, and what's the minimal sufficient set:

```
# check: is the seed right set admissible for the selected model?
steppe scan --f2-dir ~/.local/share/steppe/example \
  --target England_BellBeaker --pool Czechia_EBA_CordedWare,Turkey_N \
  --right Mbuti,Han,Papuan,Karitiana --no-allow-clade --right-search check

# add-drop: minimal sufficient outgroup set (R0 pinned)
steppe scan --f2-dir ~/.local/share/steppe/example \
  --target England_BellBeaker --pool Czechia_EBA_CordedWare,Turkey_N \
  --right Mbuti,Han,Papuan,Karitiana --no-allow-clade \
  --right-search add-drop --right-pool Iran_GanjDareh_N,Israel_Natufian
```

## Gotchas

- **`selected` means *selected, not confirmed*.** The scanner fits many correlated models and surfaces the best survivor. Because "success" is *non-rejection* (`p ≥ α`), the more models you test, the more likely one wrong model survives by chance. The header reports how many models were tested — treat the winner as a hypothesis to validate, not a proven fact. There is deliberately **no** Bonferroni/FDR correction on `--p-min`: for an acceptance gate, shrinking α would admit *more* models, not fewer. Power (better/more outgroups), not α, is the lever.
- **`feasible` and `passes` are different.** `feasible` is weights-in-`[0,1]` only; `passes` is the full gate including `p ≥ --p-min`. A model can be feasible but p-rejected. "Feasible ≠ good p-value."
- **Parsimony-first, not p-first.** The winner is the *simplest* feasible model, even if a larger model has a higher p. Use `--no-allow-clade` if you don't want a 1-source clade answer to win.
- **`--right-search` never uses fit-p to choose outgroups.** Admissibility is decided by a **sources-only** qpWave test (the target never enters it), so outgroups are chosen for their power to *distinguish the sources*, never to make the model pass — the circularity qpAdm is prone to. `add-drop` keeps at least `nsource+1` outgroups so the reported fit isn't degenerate.
- **Greedy/beam can miss a model.** The search is a heuristic; if a run is truncated it prints "may have missed a model — rerun `--strategy exhaustive`." Small pools auto-run exhaustively, so the blind spot only exists on large pools.
- **Build the f2-dir first,** and every label (`--target`/`--pool`/`--right`/`--base`/`--right-pool`) must exist in it.

## See also

- [qpadm-rotate](./qpadm-rotate.md) — brute-force rotation over a pool; `scan` is the guided, gated, ranked version.
- [qpadm](./qpadm.md) — fit one fixed model when you already know the sources.
- [qpwave](./qpwave.md) — the rank-sufficiency test behind `--right-search` admissibility.

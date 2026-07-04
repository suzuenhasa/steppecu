# dates (admixture dating)

Estimate *when* two populations mixed — how many generations ago an admixture event happened — for one present-day (or ancient) admixed population.

## What it does

When two ancestral populations mix, each admixed chromosome starts out as long alternating blocks of the two ancestries. Every generation of recombination chops those blocks shorter, so the rate at which ancestry correlation falls off as you move along the genome — measured against genetic distance — tells you how much time has passed since the mixture. A slow decay means a recent event; a fast decay means an old one. `steppe dates` fits that decay curve and reads the date off it, giving you a number of generations plus a standard error.

Reach for this when you have a target population you believe is a two-way mixture and you want to date the event. You give it the target and the two ancestral sources, and it returns one row: the date in generations, its standard error, a fit-quality figure, and a status. This is the same method as the DATES tool (and ALDER before it), and steppe matches DATES numerically. Unlike most of steppe's f-statistic tools, `dates` does **not** use a precomputed f2 cache — it reads the genotype files directly and runs its own GPU pipeline (an FFT-based autocorrelation), so there is no `extract-f2` step to do first.

## Flags

| flag | what it does | default |
|---|---|---|
| `--prefix TEXT` | **Required.** Path prefix for a genotype triple. For EIGENSTRAT/ANCESTRYMAP it reads `PREFIX.geno` / `PREFIX.snp` / `PREFIX.ind`; for PLINK it reads `PREFIX.bed` / `PREFIX.bim` / `PREFIX.fam`. The `.snp` (or `.bim`) **must carry a real centimorgan genetic map** — see Gotchas. | — |
| `--target TEXT` | **Required.** The single admixed population label whose admixture date you want. The signal is summed over every individual in this population. | — |
| `--left TEXT ...` | **Required, exactly two.** The two ancestral source population labels whose mixture formed the target. Any count other than two is rejected with an error. Order does not change the date (see Gotchas). | — |
| `--out TEXT` | Output file to write the result row to. If omitted, the row is printed to stdout. | stdout |
| `--format TEXT` | Output format for the row: `csv`, `tsv`, or `json`. Undefined numbers are written as `NA` (or `null` in JSON). | `csv` |
| `--device TEXT` | Which CUDA GPU(s) to use: `auto`, a single ordinal like `0`, or two ordinals like `0,1`. GPU-only — there is no `cpu` option. | `auto` |
| `--precision TEXT` | Matmul precision mode: `emu40`, `emu32`, `fp64`, or `tf32`. Note the dating engine's cancellation-sensitive steps and the FFT always run in native double regardless of this flag, so it has little effect here. | `emu40` |
| `--config TEXT` | **Reserved — not yet supported** (passing one currently errors). | — |

The fine-grained dating knobs (bin width, fit window, seed, the affine baseline term) are not exposed as CLI flags — the command runs with the reference DATES defaults that the validation goldens were produced with. Those defaults are documented in the reference doc linked below.

## Examples

Date a Corded Ware target as a mix of Yamnaya and Anatolian Neolithic, on GPU 0:

```
steppe dates --prefix v66.p1_HO.aadr.patch.PUB \
  --target Czechia_EBA_CordedWare \
  --left Russia_Samara_EBA_Yamnaya,Turkey_N \
  --device 0
```

Expect one CSV row on stdout with the date in generations and its standard error.

Write the result as JSON to a file instead:

```
steppe dates --prefix v66.p1_HO.aadr.patch.PUB \
  --target Czechia_EBA_CordedWare \
  --left Russia_Samara_EBA_Yamnaya,Turkey_N \
  --format json --out cordedware.date.json
```

The reference validation case dates Puerto Ricans as a European (CEU) + West African (YRI) mix and lands at **9.742 generations, SE 0.317** — a useful sanity check that your build and data path agree with DATES.

## Gotchas

- **The `.snp` needs a real cM map.** Dating fundamentally needs centimorgan positions. If the genetic-position column of your `.snp`/`.bim` is all zeros — very common for data converted straight from VCF or PLINK — the run cannot be dated and will fault. Make sure the genetic map is filled in first.
- **`--left` must be exactly two sources.** This is a two-way admixture dater, not a general qpAdm-style model. One source or three sources is rejected. Pass the two labels as `--left A,B` (or `--left A B`).
- **Source order does not change the date.** Swapping the two `--left` sources flips the sign of the internal per-SNP weight but leaves the decay *rate* — and therefore the date — unchanged. `--left A,B` and `--left B,A` give the same answer.
- **A degenerate run is not an error.** If there is no measurable ancestry-covariance decay to fit (e.g. the target is not actually a mix of these two sources), the command does not fail — it writes a normal row whose date and SE are `NA`/`null` and exits `0`. Check the status column, don't assume a zero exit means a clean date.
- **No f2 cache, no `extract-f2`.** Unlike the qpAdm/f-statistic tools, dating reads the genotype files directly. You do not build an f2 directory for it.
- **GPU required.** steppe is a GPU product; there is no CPU runtime mode. If no CUDA device is present the command errors out.

## See also

- [extract-f2](./extract-f2.md) — the f2-cache builder the *other* stats need (dating does not use it; noted here so you don't go looking for it).
- `docs/reference/include_steppe_dates.hpp.md` — the dating engine internals: the decay math, the FFT autocorrelation trick, the full `DatesOptions` knob table and their defaults, and the jackknife error bar.
- `docs/reference/src_app_cmd_dates.hpp.md` — the `steppe dates` subcommand contract: inputs, output row, and exit-code behavior.

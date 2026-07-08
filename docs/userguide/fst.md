# fst

Per-SNP FST between two populations — the classic Weir & Cockerham 1984
estimator, computed on the GPU straight from a genotype triple.

## 1. What it does

FST answers a simple question: **how different are two populations at each
site?** A value near 0 means the two groups have basically the same allele
frequencies there; a value near 1 means they're almost fully differentiated.
`steppe fst` takes two populations you name and, for every autosomal SNP, reports
the Weir & Cockerham 1984 variance-component FST for that pair — plus one
genome-wide number that summarizes the whole comparison.

It's a standalone stat. It does **not** need an f2 cache, and it has nothing to
do with the Li-Stephens engine or genotype likelihoods — it reads the genotype
triple directly, decodes it on the GPU, and runs a per-site reduction over the
population pair. One thread per SNP, batched over all of them.

You get two things out:

- **A per-SNP table** (with `--per-snp`): for each site, its id / chrom /
  position / alleles, the WC numerator and denominator, the FST itself, and a
  `valid` flag. Sites that can't produce a meaningful FST (monomorphic in both
  groups, or all-missing) come back as `NaN` with `valid = 0` — never a bogus
  divide.
- **A genome-wide summary row** (the default): the two population labels, the
  count of valid sites, and the **ratio-of-averages** FST — `sum(num)/sum(den)`
  over the valid sites, which is the number you usually quote. It also reports a
  secondary plain mean of the per-SNP FSTs, but the ratio-of-averages is the
  headline.

The genome-wide summary is always echoed to stderr as a one-line human
diagnostic, whichever output mode you pick.

## 2. The method, and what it matches

The estimator is Weir & Cockerham 1984 — the standard variance-component FST
that treats the two populations as samples and corrects for finite sample size.
Every WC term is symmetric under `p -> 1-p`, so which allele a file calls REF vs
ALT doesn't matter: an EIGENSTRAT file and a PLINK file that disagree on polarity
still give identical `num`/`den`/`fst`. Sites are matched by SNP id, not by
allele orientation.

FST is gated against **plink2** (`--fst method=wc`), *not* ADMIXTOOLS 2. Because
of that, the emulated-FP64 matmul precision policy the f-statistics tools use
does not apply here — FST is a per-site reduction, so it runs in **native
FP64** throughout (the numerator is a small difference of similar terms, exactly
the cancellation case that wants full FP64).

**Gated concordance (commit edccab2).** On a real AADR PLINK dataset
(`v66_fit9_ped`, 430 samples x 584,131 variants), Han (N=153) vs Papuan (N=46),
compared per-SNP against `plink2 --fst method=wc report-variants`:

- 425,234 valid sites compared.
- Max absolute per-SNP `WC_FST` difference = 5.0e-7 — and that's entirely
  plink2's 6-significant-figure text rounding; the underlying arithmetic agrees
  to machine precision.
- Pearson r = 0.999999999999.
- Genome-wide summary FST = **0.177817**, matching plink2 to every printed digit.

## 3. All-pairs: the whole population × population FST matrix

`--all-pairs` flips `steppe fst` from a single 2-population query into a full
**P×P genome-wide WC FST matrix** — every population against every other, in one
GPU pass. It decodes the genotypes once into per-population sufficient statistics,
then sweeps all `C(P, 2)` pairs on the device and assembles a symmetric P×P matrix
with a zero diagonal. Each cell is the same genome-wide **ratio-of-averages**
(`sum(num)/sum(den)`) you'd get from the single-pair path — and it's **bit-exact**
against that path, pair for pair; `--all-pairs` is just the batched, decode-once
way to get the whole grid at once.

You pick the populations one of two ways:

- **`--pops A,B,C,…`** — an explicit set of labels.
- **`--min-n N`** — every population in the file that has at least `N` individuals
  (`--min-n 1`, the default, is *all* populations). Mutually exclusive with an
  explicit `--pops` set.

The output is a labelled matrix: a population-label header row and a leading label
column, symmetric with a zero diagonal (in JSON, `{"pops": [...], "fst": [[...]]}`).
A cell is `NaN` when a pair shares no valid site. Because a large panel can mean a
lot of pairs, there's a safety cap on the pair count; pass `--sure` to lift it (or
restrict `--pops` / raise `--min-n`). Note the cap is on the *pair count* — the
actual work is `C(P,2) × M` accumulations, so a very large P at full SNPs can still
be a long run even under the cap.

**Where this earns its keep.** The value of `--all-pairs` is the **large-panel**
regime — hundreds to thousands of populations, the kind of full-panel FST grid you
build for QC, a population-structure atlas, or curating which pops to keep. It is
*not* the tool for a single 2-pop question; for that a mature CPU tool is fine.

**Measured** (real AADR, 1.15–1.23M SNPs, vs `plink2 --fst CATPHENO --threads 384`
computing the same all-pairs matrix, one RTX 5090, `--device 0`):

| populations | pairs | steppe vs plink2 |
|---|---|---|
| 109 | 5,886 | **3.95×** |
| 267 | 35,511 | **19.9×** |
| 502 | 125,751 | **29.5×** |
| 3,898 (full panel) | 7.6M | **22 min** vs plink2's projected **~18.5 hr** (**~50×**) |

There's a **crossover around ~60 populations**: below it plink2 wins (steppe's
fixed GPU-context and small-problem overhead dominate), above it steppe pulls away
fast, and by the hundreds-of-pops range — exactly the regime you'd actually want a
full matrix for — it's an order of magnitude or more ahead.

## 4. Inputs

A genotype triple, passed as a `--prefix`: `PREFIX.geno`, `PREFIX.snp`,
`PREFIX.ind` (EIGENSTRAT), or the equivalent PLINK / other supported multi-format
inputs the shared decode front-end reads. The two `--pops` labels must be
population names that appear in the `.ind` (or the PLINK family/population
column).

## 5. Outputs

Either the genome-wide summary row or, with `--per-snp`, the full per-SNP table.

Summary columns: `popA`, `popB`, `n_valid`, `fst_ratio`, `fst_mean`.

Per-SNP columns: `snp_id`, `chrom`, `pos_bp`, `a1`, `a2`, `popA`, `popB`,
`fst_num`, `fst_den`, `fst`, `valid`. Here `fst_num` is plink2's `FST_NUMER`,
`fst_den` is `FST_DENOM`, and `fst` is `WC_FST` (`NaN` on an invalid site).

With `--all-pairs`, instead a labelled **P×P matrix**: a population-label header
row plus a leading label column, symmetric with a zero diagonal (JSON:
`{"pops": [...], "fst": [[...]]}`), each cell the genome-wide ratio-of-averages
for that pair.

## 6. Flags

| flag | what it does | default |
|---|---|---|
| `--prefix TEXT` | Genotype triple prefix — reads `PREFIX.{geno,snp,ind}` (EIGENSTRAT / PLINK / …). Required. | — |
| `--pops A,B` | The **two** populations to differentiate. With `--all-pairs`, instead the explicit set of populations for the matrix. | — |
| `--all-pairs` | Compute the whole **P×P** genome-wide WC FST matrix over the selected populations in one GPU pass, instead of a single pair. | off |
| `--min-n INT` | (`--all-pairs` only) Include every population with at least this many individuals. `1` = all pops. Mutually exclusive with an explicit `--pops` set. | `1` |
| `--sure` | (`--all-pairs` only) Lift the safety cap on the C(P,2) pair count for very large population sets. | off |
| `--method TEXT` | FST estimator. Only `wc` (Weir-Cockerham 1984) is available today; `hudson` is a documented follow-up. | `wc` |
| `--per-snp` | Emit the per-SNP FST table instead of just the genome-wide summary row. (Single-pair mode only.) | off (summary) |
| `--out TEXT` | Write to this file. | stdout |
| `--format TEXT` | Output format: `csv`, `tsv`, or `json`. | `csv` |
| `--device TEXT` | CUDA device (single ordinal; GPU-only, there is no `cpu`). Multi-GPU is parked. | `auto` |
| `--precision TEXT` | Shared matmul-precision flag. **No effect on FST** — FST always runs native FP64 regardless of what you pass. | `emu40` |

## 7. Example

The measured gate command — Han vs Papuan on the AADR PLINK fixture, per-SNP TSV:

```
steppe fst \
  --prefix /path/to/v66_fit9_ped \
  --pops Han,Papuan \
  --method wc --per-snp \
  --device 0 \
  --out fst.tsv --format tsv
```

That prints, to stderr, a line like:

```
steppe fst: Han vs Papuan — WC FST(ratio-of-averages) = 0.177817 over 425234 valid autosomal sites (mean per-SNP = ...)
```

and writes the per-SNP table to `fst.tsv`. Drop `--per-snp` to get just the
one-row genome-wide summary.

The all-pairs matrix — every population with at least 5 individuals, one GPU pass:

```
steppe fst --all-pairs \
  --prefix /path/to/v66.p1_1240K.aadr.patch.PUB \
  --min-n 5 \
  --device 0 \
  --out fst_matrix.tsv --format tsv
```

Or restrict to an explicit set with `--pops A,B,C,…`. For the full-panel matrix
(thousands of populations, millions of pairs), add `--sure` to lift the pair-count
cap.

**Measured wall-clock:** ~3.15 s on box5090 (Release build, single RTX 5090,
`--device 0`) for the 430-sample × 584,131-variant fixture above — median of 3
runs, ~505 MB peak RSS. On a gate this small, plink2 (a mature multithreaded CPU
tool) finishes sub-second, and steppe's few seconds are dominated by fixed
GPU-context startup plus the full multi-format decode. steppe's GPU advantage is
a large-model / scale story, not this one fixture; treat the ~3 s here as a
correctness-and-parity result, not a speed claim.

## 8. Caveats and gotchas

- **Two populations, or the whole matrix.** Without `--all-pairs`, `--pops` must
  name two *different* labels. With `--all-pairs` it computes the full P×P matrix
  over the selected set instead (`--per-snp` is single-pair only). Hudson's
  estimator, windowed FST, PBS, and a block-jackknife standard error are still
  documented follow-ups, not shipped.
- **Autosomes only.** The summary and the valid-site count cover autosomal sites;
  the genome-wide FST is over those.
- **Invalid sites are honest `NaN`s.** A site is valid only if both groups have at
  least one genotyped sample, the mean sample size is > 1, and the denominator is
  finite and non-zero. Monomorphic-in-both or all-missing sites come back as
  `fst = NaN`, `valid = 0` — they're excluded from the genome-wide totals, never
  silently zeroed.
- **`fst_ratio` is the number to quote,** not `fst_mean`. The ratio-of-averages
  (`sum(num)/sum(den)`) is what plink2's summary reports and what the concordance
  above was checked against; the unweighted per-SNP mean is a secondary
  diagnostic.
- **`--precision` does nothing here.** It's a shared flag across steppe commands,
  but FST is always computed in native FP64. Passing `emu40`/`fp64`/etc. changes
  nothing.
- **GPU only.** Like the rest of steppe there is no CPU runtime; the CPU WC
  implementation exists purely as the parity oracle behind the tests.

## 9. See also

- [extract-f2](./extract-f2.md) / [f-statistics](./f-statistics.md) —
  the f2-cache-based population statistics (a different family; FST needs no cache).
- [data-and-formats](./data-and-formats.md) — the genotype input
  formats the shared decode front-end accepts.

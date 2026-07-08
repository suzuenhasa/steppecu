# `steppe ibd` reference

## 1. What it is

`steppe ibd` finds **shared-by-descent DNA segments** between pairs of ancient
individuals. When two people inherit a long stretch of chromosome from a recent
common ancestor, that stretch is identical-by-descent (IBD): the closer the
relationship, the more IBD they share. `steppe ibd` scans every pair, calls those
segments, and hands you back a table of "who shares what, and how long" — which is
what you use to reconstruct a pedigree or spot relatives in a burial.

The real question it answers: *given a handful of imputed ancient genomes, which
pairs are related, and how closely?*

This is the GPU port of **ancIBD** (Ringbauer et al. 2023), run faithfully — not a
simplified two-state stand-in. It is the first face of steppe's ancient-DNA
relatedness family beyond READv2.

## 2. The method, in one paragraph

ancIBD models each pair of individuals as a hidden Markov chain walking along the
chromosome. There are five hidden states: state 0 = "not sharing IBD here", and
states 1–4 = the four ways two diploid people can share one haplotype
(mother/father × mother/father). At each SNP the model reads the imputed genotype
posteriors (the `GP` field, P(0/0), P(0/1), P(1/1)) together with the phased
genotype, turns them into per-haplotype ancestral-allele probabilities, and asks
how well "sharing" versus "not sharing" explains what it sees. A forward-backward
pass gives, at every SNP, the posterior probability that the pair is in an IBD
state. steppe runs that forward-backward on the GPU (one block per pair, batched
across all pairs), then walks the posterior to **call segments**: threshold each
SNP at the posterior cutoff, form contiguous IBD runs, drop runs shorter than the
minimum length, and merge runs separated by only a tiny gap. Finally it rolls the
segments up into a per-pair relatedness summary.

Everything numerical is ported exactly from production ancIBD: the 5-state
`FiveStateScaled` forward-backward, the symmetry-collapsed 3×3 transition-rate
table (the matrix exponential `expm(Q·dGenetic)` is precomputed on the host per
gap), the `haploid_gl2` / `l_model h5` emission built from GP + the phased GT, and
the segment post-processing (`call_roh` → `create_ind_ibd_df`). The default
parameters are the locked `run_ancIBD.py` values.

## 3. What it matches (the gate)

`steppe ibd` was gated on box5090 (Release, sm_120, real data) against **pip ancIBD
0.8**, on ancIBD's own tutorial dataset: the Hazelton Neolithic-Britain pedigree —
6 GLIMPSE-imputed individuals, chromosome 20, phased GT + GP, 15 pairs.

Against pip ancIBD 0.8 on that data:

- per-pair total IBD (≥ 8 cM): **Pearson r = 0.999986**
- segment overlap: **99.84%**
- relationship-degree agreement: **7 / 7** known relative pairs
- worst per-pair difference: **0.71 cM** (a segment-edge rounding difference)

The 7 known relatives were all recovered with substantial IBD and the 8 unrelated
pairs sat at 0, matching the published pedigree. As a bonus, the run closed the
GL/GP reader's deferred GP gate: steppe's GP decode was **171,732 / 171,732 =
100% bit-exact vs bcftools**. The GPU kernel is 100% of GPU time by nsys (no host
forward-backward loop), and GPU-vs-CPU-oracle parity is 4e-14. Full test suite:
ctest 104/104 (new: `ancibd_model_unit`, `cli_ibd`).

## 4. Speed

On the Hazelton chr20 gate data (28.5k sites × 15 pairs) the GPU forward-backward
ran in **~1.8 s**.

Honest caveat on that number: it is the figure recorded in the gate doc
(`docs/planning/ancibd-gate-results.md`), *not* a fresh re-measurement — the
Hazelton VCF/map/AF inputs were cleaned off the disk-tight box after the gate, so
the command could not be re-timed here. pip ancIBD 0.8's wall-clock was not
separately timed at the gate.

## 5. Inputs

- **`--gp-vcf`** *(required)* — the imputed VCF (`.vcf.gz`/`.vcf`) carrying both
  phased `GT` and the `GP` posterior triple per site (GLIMPSE output is the
  intended source).
- **A target-site source** *(required, choose exactly one)*:
  - `--targets` a pre-built target-site table, or
  - `--panel` an AADR EIGENSTRAT `.snp` panel **with** `--fasta` (build-matched
    `.fa` + `.fai`), plus `--lift` (rsID→pos map) for a cross-build VCF and
    optionally `--assembly GRCh37|GRCh38` to override build auto-detection.
- **`--map`** *(required)* — a per-site genetic map, `rsID <ws> position` per line.
  `--map-unit cm|morgan` says which unit the file is in (default `cm`).
- **`--af`** — a per-site derived (ALT) allele-frequency file, `rsID <ws> freq`.
  Required only under `--af-mode panel` (the default); see §7.

Optional subsetting:

- **`--samples`** — a file of IIDs (one per line) to restrict to; default is all
  samples in the VCF.
- **`--pairs`** — a file of explicit `iid1 <ws> iid2` pairs; default is every pair,
  C(n,2).

## 6. Outputs

Two tab- (or comma-) separated tables.

**Per-segment table** (`--out`, default stdout), one row per called IBD segment:

```
iid1  iid2  ch  Start  End  StartM  EndM  length  lengthM  lengthCM  StartBP  EndBP
```

`Start`/`End` are SNP indices in the pair's run order, `StartM`/`EndM`/`lengthM`
are genetic positions/length in Morgan, `lengthCM` the length in centimorgans,
`length` the SNP count, and `StartBP`/`EndBP` the physical bounds.

**Per-pair summary** (`--summary`, default `<out>.summary`, or stderr if neither is
set), one row per pair:

```
iid1  iid2  max_IBD  sum_IBD>8  n_IBD>8  sum_IBD>12  n_IBD>12  sum_IBD>16  n_IBD>16  sum_IBD>20  n_IBD>20
```

`max_IBD` is the longest segment (cM); the `sum_IBD>k` / `n_IBD>k` pairs give the
total shared cM and segment count above each of the fixed cutoffs `[8, 12, 16, 20]`
cM — the reference's relatedness columns.

## 7. The CLI

Minimal run against a pre-built target table and a panel AF file (the gate shape):

```
steppe ibd \
  --gp-vcf hazelton.chr20.imputed.vcf.gz \
  --targets 1240K.targets \
  --map 1240K.chr20.cm.map \
  --af 1240K.chr20.af \
  --min-cm 8 \
  --device 0 \
  --out steppe_ibd_chr20.tsv
```

Key flags beyond the inputs in §5:

- **`--af-mode panel|sample|half`** *(default `panel`)* — where the derived-allele
  frequency comes from: an external `--af` file (`panel`), estimated in-sample
  across the selected haplotypes (`sample`), or a flat 0.5 (`half`). `panel`
  requires `--af`.
- **`--min-cm`** *(default 8)* — the called-segment length floor in cM.
- **`--post-cutoff`** *(default 0.99)* — the per-SNP IBD-posterior calling
  threshold.
- **`--max-gap-cm`** *(default 0.75)* — merge two IBD runs separated by less than
  this genetic gap (cM).
- **`--device`** *(default auto)* — CUDA device ordinal.
- **`--format tsv|csv`** *(default `tsv`)* — output field separator.

The HMM rate/emission knobs — `--ibd-in` (1), `--ibd-out` (10), `--ibd-jump`
(400), `--in-val` (1e-4), `--min-error` (1e-3), `--p-min` (1e-3) — default to the
locked ancIBD values and normally need no touching; they exist so the model is
reproducible, not so you tune it.

## 8. Honest caveats

- **It needs imputed, phased data with GP.** ancIBD is designed for GLIMPSE-imputed
  genomes carrying phased `GT` and the `GP` posterior triple. It is not a
  pseudo-haploid / low-coverage-genotype method — feed it imputed input.
- **A whole chromosome is ~50–280 cM.** If your `--map-unit` disagrees with the
  file (e.g. cM flag on a Morgan file), positions come out 100× off and you get
  silently empty or nonsensical calls. steppe checks the per-chromosome span and
  prints a one-line warning when it looks like the unit is wrong — read it.
- **Sites must be present for every selected sample.** A SNP is dropped from a run
  if any selected sample is missing it, if it has no genetic-map entry, or (under
  `--af-mode panel`) if it has no AF entry. The run prints the dropped-site counts.
- **Known residual vs pip ancIBD (does not affect the scored metrics).** steppe
  keeps ~1.5% fewer chr20 markers than pip ancIBD (it drops no-GP / no-AF sites).
  On the gate that meant one 26 cM pair was filtered out of steppe's *summary
  rollup* by the SNP-density floor (the summary keeps segments above ~220 SNP/cM),
  even though the segment itself is detected by both tools. The three scored
  concordance metrics (Pearson r, segment overlap, degree agreement) are unaffected.

## 9. Under the hood

The CPU backend is a **reference oracle only** — used to validate the GPU, never as
a user runtime. When a CUDA device is visible steppe runs the forward-backward on
the GPU (`ancibd_fb_kernel`, block-per-pair, batched, native-FP64 per-column
rescale); with no device visible it falls back to the CPU oracle so the command
still works in a test/CI setting. The forward-backward runs in native FP64.

Source: `src/app/cmd_ibd.cpp` (CLI + driver), `src/core/stats/ancibd_model.hpp`
(emission + transition table), `src/core/stats/ancibd_segments.{hpp,cpp}` (segment
calling + summary), `src/device/cuda/ancibd_fb_kernel.cu` (the GPU kernel).
Committed in 848eb66.

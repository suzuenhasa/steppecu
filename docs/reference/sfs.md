# `steppe sfs` reference

## 1. What it is

`steppe sfs` builds the **2D joint site-frequency spectrum** for a pair of
populations — the table that answers, for two groups A and B, "at how many sites
does A carry *i* copies of an allele while B carries *j* copies?" You point it at
a genotype file and name two populations, and it hands back an integer matrix
where cell `(i, j)` is the number of SNPs with count `i` in A and count `j` in B.

That joint spectrum is a workhorse input for demographic inference — it is what
tools like `dadi`, `moments`, and `fastsimcoal` fit models against — and on its
own it is a compact, honest summary of how allele frequencies co-vary between two
populations. steppe just computes the table; it does not fit a demographic model
on top of it.

Every number in the matrix is an exact integer count. There is no estimator and
no floating-point math in the answer, so the result is checked **cell-by-cell,
bit-exact** against a reference tool rather than to a tolerance.

---

## 2. The real question it answers

Given two populations from the same SNP set, the joint SFS captures the shape of
their shared and private variation in one picture:

- Sites near the corners (both populations near 0 or near fixed) are alleles the
  two groups mostly agree on.
- Mass along one edge is variation that segregates in one population but is
  fixed in the other — a fingerprint of drift and population split.
- The off-diagonal spread tells you how correlated the two allele frequencies
  are, which is exactly what a demographic model is trying to reproduce.

You get either the **unfolded** spectrum (you know which allele is ancestral /
derived and want to count a specific allele) or the **folded** spectrum (you
don't trust the polarity and fold each population onto its own minor allele).

---

## 3. The method, and the tool it matches

Under the hood `steppe sfs` reuses the same front-end as `steppe fst`. It reads
the genotype triple into one population-contiguous device tile, then computes,
per SNP and per population, the non-missing individual count and the number of A1
(reference-axis) allele copies — the exact same per-population fold the FST path
uses. Instead of feeding those counts into a variance formula, it feeds them into
a **joint histogram**: a GPU kernel runs one thread per SNP and `atomicAdd`s a `1`
into the device-resident grid cell for that site's `(count_A, count_B)` pair. Only
the finished matrix and the complete-site counter come back from the GPU.

- **Unfolded** cell `(i, j)`: `i` = copies of A1 in population A (0…2·NA), `j` =
  copies of A1 in population B (0…2·NB). The counted allele is A1 on both sides,
  and the gate aligns the reference tool's axis to A1 explicitly so the two agree.
  Each axis has `2N + 1` categories.
- **Folded** cell `(i, j)`: each population is folded on **its own** within-group
  minor allele — `i = min(a, 2NA − a)`, `j = min(b, 2NB − b)` — not on a shared
  minor. This matches how the reference tool folds (`np.amin` per population, per
  axis), and it is invariant to swapping which allele you called "reference," so
  it is the robust, polarity-free view. Each axis has `N + 1` categories.

**Reference tool and concordance.** The result is gated bit-exact against
**scikit-allel** (`allel.joint_sfs` / `allel.joint_sfs_folded`, version 1.3.13),
**not** ADMIXTOOLS2 — the SFS is not an ADMIXTOOLS statistic, so no floating-point
parity policy applies. At the gate on real data (AADR v66 1240K, Han with NA = 153
individuals / 306 chromosomes vs Papuan with NB = 46 / 92, over 378,113
complete-data sites), all six comparisons — folded and unfolded, each across
steppe, a fresh scikit-allel oracle, and the frozen golden — came back with **zero
mismatched cells**, and both matrices summed to exactly 378,113 (every kept site
lands in exactly one cell). A GPU profile confirmed the histogram is accumulated
on the device tile, not on the host.

---

## 4. The complete-data convention (an honest caveat up front)

steppe matches scikit-allel's missing-data behaviour exactly, and it is worth
stating plainly because it shapes the result: **a site contributes only if every
individual in both populations is non-missing.** scikit-allel bins allele counts
at a fixed chromosome count and does not down-project partial-coverage sites, so
steppe drops any site with a missing genotype in either population rather than
scaling it. The run reports how many sites were kept (`complete`) and how many
were dropped for missingness (`dropped_incomplete`).

For dense, high-coverage panels this drops little. For ragged, low-coverage
ancient data it can drop a lot, and the honest fix — a hypergeometric
down-projection to a common smaller sample size — is a documented follow-up, not
something v1 does. Genotypes are also read as **diploid** dosages (0/1/2),
matching the FST path and the scikit-allel oracle.

---

## 5. Inputs and outputs

**Input.** A genotype triple given by `--prefix` (reads `PREFIX.geno`,
`PREFIX.snp`, `PREFIX.ind`; EIGENSTRAT, PLINK, and the other formats the shared
reader supports), plus exactly two population labels from the `.ind` / `.fam`
population column.

**Output.** The `(extA × extB)` integer joint matrix, where `extA = 2·NA + 1`
(unfolded) or `NA + 1` (folded), and likewise for B. Formats:

- **CSV / TSV** — a comment header block recording the two populations, the fold
  flag, the per-population individual counts, the matrix dimensions, and the
  site accounting (`sites_total`, `complete`, `dropped_incomplete`), followed by
  the raw integer grid, one matrix row per line.
- **JSON** — the same provenance fields plus `sfs` as a nested row-major array of
  integers.

Cells are exact 64-bit integers. A one-line human summary (populations, matrix
size, fold, sites kept and dropped) is also printed to stderr, separate from the
matrix stream, so you can pipe the matrix cleanly.

---

## 6. Command-line usage

```
steppe sfs --prefix PREFIX --pops A,B [--fold | --unfold] \
           [--out FILE] [--format csv|tsv|json] [--device auto|N]
```

Real flags (read from `cli_parse.cpp` and `cmd_sfs.cpp` — nothing here is
invented):

- `--prefix PREFIX` — the genotype triple prefix. Required.
- `--pops A,B` — exactly **two** population labels, and they must be different.
  Naming one, three, or two identical labels is a clear config error.
- `--fold` / `--unfold` — folded (per-population minor allele) vs unfolded
  (A1-copy count). **Default is unfolded.** Folded is the polarity-free view and
  was the primary concordance gate.
- `--out FILE` — write to a file (default is stdout).
- `--format csv|tsv|json` — output format (default `csv`).
- `--device auto|N` — CUDA device selection. steppe is GPU-only; there is no CPU
  runtime mode. Use `--device 0` to pin the first GPU.
- `--precision` — accepted for interface consistency, but it does not change the
  answer here: the SFS is a pure integer count with no matmul, so precision mode
  is irrelevant to the result.

**Example** (the folded joint SFS used for timing):

```
steppe sfs \
  --prefix /path/to/aadr/v66_fit9_ped \
  --pops Han,Papuan \
  --fold \
  --device 0 \
  --out sfs.tsv --format tsv
```

---

## 7. Performance

On a single RTX 5090 (Release build, `--device 0`), the folded Han vs Papuan run
on the AADR v66 `v66_fit9_ped` fixture — a 154×47 folded joint SFS over 380,265
complete-data sites (203,866 dropped as incomplete) — measured **2.17 s** wall
(median of three runs, 2.17 / 2.18 / 2.16 s; peak RSS 505 MB). On a fixture this
small the wall time is dominated by GPU context setup and the genotype decode —
the same decode spine `steppe fst` uses — rather than by the histogram itself,
which is a single lightweight pass. The scikit-allel reference was not separately
timed on this fixture at the gate.

---

## 8. Caveats and scope

- **Two populations only.** v1 is the 2D joint SFS. A 3D (three-population)
  extension is a documented follow-up; the internal index math already
  generalizes, but only 2D is built and gated.
- **Complete-data sites only.** Sites with any missing genotype in either
  population are dropped, not projected (see §4). Hypergeometric down-projection
  is a follow-up; if it lands, it would be gated against `moments` / `dadi`
  rather than scikit-allel.
- **Hardcall genotypes only.** This path reads called genotypes; it does not use
  genotype likelihoods. It is engine-independent — no f2 cache, no Li-Stephens.
- **Diploid.** Genotypes are read as diploid dosages, matching the reference
  oracle.
- **GPU required.** There is no CPU-only runtime; the CPU backend exists only as
  the parity oracle for testing.

# Using steppe on non-AADR datasets

steppe was developed and golden-validated against the [Allen Ancient DNA Resource
(AADR)](https://reich.hms.harvard.edu/allen-ancient-dna-resource-aadr-downloadable-genotypes-present-day-and-ancient-dna-data),
but its statistics are **not AADR-specific**. This page covers what to expect when you
run steppe on other datasets (HGDP, SGDP, 1000 Genomes, your own merges), how it differs
from ADMIXTOOLS 2 (AT2) by default, and how to reproduce AT2's numbers exactly.

## The math generalizes

On four independent modern, all-diploid panels (HumanOrigins/HGDP, SGDP, gnomAD
HGDP+1KGP, 1000 Genomes), whenever steppe and AT2 are given the **identical SNP set**,
steppe reproduces `admixtools 2.0.10` to **~1e-11 – 1e-14** on f4, f3, and qpAdm
weights/chisq/p. In particular f3 — which does not cancel the within-population diploid
heterozygosity correction the way f4 does — matches to ~1e-14. **The engine is correct on
any diploid dataset, not just ancient pseudo-haploid AADR data.**

What differs out of the box is a handful of **default SNP-selection choices** and one
**data requirement**, none of them in the compute engine. Each is listed below with the
one-line fix to match AT2.

## Supported input formats

steppe reads **EIGENSTRAT** (`.geno/.snp/.ind`), **PACKEDANCESTRYMAP**, **ANCESTRYMAP**,
**PLINK** (`.bed/.bim/.fam`), and the AADR's **TGENO**. There is **no VCF reader** —
convert first with `convertf` or `plink2` (the standard popgen workflow). The readers are
validated bit-exact against `convertf`.

---

## Default differences from ADMIXTOOLS 2

### 1. Strand-ambiguous (palindromic) SNPs — `--strand-mode`

steppe **drops** strand-ambiguous SNPs (A/T and C/G allele pairs) **by default** — a
merge-safety choice, since palindromic sites are a classic strand-flip corruption source
when merging data from different sources. **AT2 keeps them.** On a modern panel this is
~14–16% of SNPs and is the single largest out-of-the-box difference (up to ~3–4% on f4,
~1–2% on a qpAdm p-value).

- To **reproduce AT2**, keep them:

  ```
  steppe extract-f2 --strand-mode keep  ...
  ```
  ```python
  steppe.extract_f2(..., strand_mode="keep")
  ```

- `--strand-mode` values: `drop` (default), `keep` (retain palindromes — matches AT2),
  `flip` (reserved for future frequency-based strand reorientation; currently behaves as
  `keep`).

Multiallelic and non-ACGT SNPs are dropped in **every** mode — only the palindrome policy
is switchable. On a **matched** SNP set steppe and AT2 agree to ~1e-12, so this is purely
which SNPs enter the computation, not the numerics.

> Note: the AADR Human Origins panel contains **zero** palindromic SNPs by array design,
> so `drop` and `keep` are identical there — this only matters on other datasets.

### 2. Genetic map required for standard errors — the cM column

The block-jackknife standard errors (and DATES) partition SNPs into LD blocks using the
**centiMorgan genetic map** in the `.snp` third column.

- steppe now matches AT2's behavior when the map is **absent** (all-zero cM): it warns and
  falls back to a **2 Mb base-pair block window**, producing valid SEs (identical block
  count to AT2). So a map-less dataset no longer breaks — you'll see:

  ```
  steppe: No genetic linkage map found! Defining blocks by base pair distance of 2e+06
  ```

- **Important caveat:** `convertf` **silently fabricates** a synthetic uniform 1 cM/Mb map
  (`cM = bp × 1e-8`) when it writes EIGENSTRAT, so a `convertf`-derived `.snp` will have a
  *non-zero but fake* map. Your SEs will then be computed off that synthetic map, and
  **DATES in particular assumes a real recombination map**. For publication-grade SEs or
  any DATES analysis on VCF/PLINK-derived data, **supply a real recombination map** (e.g.
  interpolate HapMap or the Broad hg19/hg38 maps onto your SNP positions) rather than
  relying on the uniform fallback.

### 3. Missingness threshold — `--maxmiss` semantics

steppe's `--maxmiss` / `--geno-max-miss` is a **per-individual** missing-fraction filter.
AT2's `maxmiss` is a **per-population** (fully-missing-in-a-group) filter. They select
different SNPs at the same nominal value — set them to match your intent, and don't assume
the same number reproduces AT2's filtering.

### 4. Monomorphic-SNP definition

steppe's monomorphic drop and AT2's `poly = 'f2'` definition differ slightly (e.g. on one
SGDP subset, AT2 kept 258,802 vs steppe 219,237). Use matched SNP sets for exact
reproduction; the difference is small on well-covered data.

### 5. qpAdm weight standard error — method nuance

steppe computes the qpAdm **weight** SE with a fast analytic delta-method Jacobian; AT2
uses a leave-one-block-out re-solve jackknife. The two agree at genome-wide block counts
but can differ by a few percent on **low-block-count** runs (e.g. a single chromosome).
The qpAdm **point weights, chisq, and p-value are exact** in all cases — only the weight
SE carries this method difference.

---

## Recipe: reproduce AT2 numbers exactly on a non-AADR dataset

```
steppe extract-f2 \
    --strand-mode keep \          # keep palindromes, like AT2
    --maxmiss <matched> \         # match AT2's per-population semantics to your intent
    <prefix>                      # a .snp with a REAL cM map (for correct SEs / DATES)
```

With a matched SNP set and a real genetic map, steppe reproduces AT2 f4/f3/qpAdm to the
golden tolerance (~1e-11 – 1e-13). See `docs/MULTI-DATASET-RESULTS.md` for the full
cross-dataset validation.

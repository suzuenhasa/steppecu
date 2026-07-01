# DATES + the genotype-path-statistic seam (reference + design note)

*2026-06-22. Captures what DATES is, why it needs a genotype-path (not f2-cache) pipeline, and the Part-B decision for qpDstat. Companion to `docs/research/standalone-fstats.md`.*

## What DATES is
**DATES = admixture *dating*** — it answers "**when** did this mixture happen?" (in generations → years), complementing qpAdm's "what mixed / how much." It's the Moorjani-lab, ancient-DNA-capable member of the **ALDER / ROLLOFF** "admixture-LD-decay" family.

### Intuition
Admixture makes a descendant genome a **mosaic of ancestry tracts** (chunks from source A interleaved with source B). Each generation, **recombination chops the tracts shorter**. Rate of fragmentation = recombination rate × generations, so **how fragmented the tracts are ⇒ how long ago the admixture happened**.

### Mechanic
Low-coverage aDNA can't see tracts directly, so DATES measures them statistically: it computes an **ancestry covariance** (a weighted LD — the correlation of source-allele-frequency-differences) between **pairs of SNPs**, as a function of the **genetic distance `d`** between them. That covariance decays exponentially:

```
cov(d) ≈ A · exp(−t · d) + c
```

Bin SNP-pairs by genetic distance `d`, fit the exponential, and the decay rate **`t` is the admixture date in generations**.

## Why DATES is new machinery (effort: HIGH)
1. **Needs per-SNP data AT genetic positions, and per-SNP *pairs*.** The f2 cache (per-pair-per-block scalars) has none of it; even D's Part-B kernel is per-*single*-SNP. DATES needs the **per-SNP genetic position** (`.snp` col-3 cM/Morgans — currently used for block assignment but not surfaced per-SNP to a stat kernel; physpos col-4 is parsed-then-discarded).
2. **The pairwise kernel is the hard part.** Naively O(M²) SNP-pairs (~1.2M SNPs → ~10¹² pairs). DATES/ALDER use FFT / distance-binning tricks to make it tractable — a real algorithm, not a simple reduction.
3. **Ends in a nonlinear curve-fit** (exponential → `t`). steppe has **no general optimizer** (only the qpAdm-specific ALS) — a new fitter.

## The genotype-path-statistic seam
steppe's pipeline splits at S2: `genotypes → S0/S1 decode_af → per-SNP Q/V/N → S2 f2 GEMM → f2 cache → fits/f4/f3/qpdstat(--f2-dir)`. Per-SNP data is **discarded at S2**. Any stat needing per-SNP data (D's normalized magnitude, DATES) must run a **separate genotype-reading path** that shares the **decode front-end** (io read + `decode_af` + `assign_blocks` + SNP-tile streaming) but **diverges at the back-end** (its own kernel, not the f2 GEMM) and **never touches the f2 cache**. This mirrors AT2 (`qpdstat_geno` is separate from the f2-data path).

- **D's Part B back-end:** per-*single*-SNP `num=(a−b)(c−d)`, `denom=(a+b−2ab)(c+d−2cd)`, accumulated per block, then a num/den block-jackknife. → `D, se, z, p`.
- **DATES' back-end (different!):** per-SNP-*pair* covariance, binned by genetic distance, then an exponential-decay fit. → `t` (generations).

**Shared = only the decode front-end** (already reusable from `extract-f2`). The kernels/outputs differ structurally, so a "reusable kernel interface" designed from D's shape would likely be the *wrong* abstraction for DATES.

## Decision (Part B of qpDstat)
**Go with (i): D-specific genotype path now.** Build `run_dstat`/`steppe qpdstat --prefix` reusing the decode front-end + a separable D kernel + the num/den jackknife; gate vs AT2 genotype-path `qpdstat(prefix, f4mode=FALSE)`. **Plus a small, clearly-shared down-payment on DATES: surface/retain per-SNP genetic position through the genotype-stat decode path** (D ignores it; DATES will need it). Do NOT build a speculative kernel framework before DATES' real (pairwise + fit + genpos) requirements are known — generalize the seam when DATES is actually built.

### Effort tiers (recap)
- **D Part B** (genotype-path normalized-D): MEDIUM — decode front-end (reuse) + the D kernel + num/den jackknife + genpos surfacing.
- **DATES**: HIGH — the pairwise distance-binned covariance kernel (FFT/binning) + a nonlinear fitter + per-SNP genpos; reuses the decode front-end the D Part B establishes.

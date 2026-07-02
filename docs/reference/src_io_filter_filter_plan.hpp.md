# `filter_plan.hpp` reference

## 1. Purpose

`src/io/filter/filter_plan.hpp` defines a single small struct, `FilterPlan`. It is
the resolved description of which SNPs and which samples a filtered read of one
dataset will keep, plus the live thresholds needed to finish the cheap filters
while the data is being read.

The key idea is that a `FilterPlan` is a **plan, never an on-disk rewrite**. steppe
does not produce a new, smaller genotype file with the unwanted rows and columns
removed. Instead it computes, up front, exactly which SNPs survive and which samples
survive, records those decisions in this struct, and then applies them on the fly as
each tile of genotype data is read and processed. The plan carries three things:

1. The resolved set of kept samples (as a list of sample indices).
2. The resolved per-SNP keep decision (as one true/false flag per SNP).
3. A copy of the live filter thresholds that the tile-processing stage still needs
   at the moment it applies the remaining cheap, per-tile filters.

Together these let the filtered tile-production stage do its work in a single pass
over the data, with no second scan and no rewritten file.

---

## 2. How the plan flows through the pipeline

A `FilterPlan` is built upstream and consumed downstream; the struct itself is just
the hand-off between the two.

**Who builds it.** Two separate stages fill in two separate parts of the plan:

- The SNP-filter stage builds `snp_keep` (and the cached `n_snp_kept` count),
  deciding which SNPs survive all of the SNP-level quality filters.
- The per-sample missing-data pre-pass (the `--mind` pre-pass) builds
  `kept_samples`, deciding which samples survive. This is a conditional step: it
  only does real work when a per-sample missing-data threshold is actually set.

**Who reads it.** The decode front-end (the stage that reads and harmonizes the
genotype data into tiles) consumes the plan. It emits only the surviving SNP columns
and only the kept samples, applying the remaining in-tile filters using the carried
thresholds.

The plan deliberately carries **no jackknife-block information**. steppe estimates
uncertainty with a block jackknife that partitions SNPs into blocks along the genome.
Dropping a SNP does not renumber or reshape those blocks — a dropped SNP simply
contributes nothing to whatever block it fell in, so block identity is unchanged. The
definition of the blocks lives with the shared block-partition rule, not here, so
that there is exactly one source of truth for block layout.

---

## 3. Layering and dependencies

This is a leaf header of the input/output (`io`) layer. It is intentionally
lightweight:

- It contains only plain data — standard `std::vector` members and a plain
  configuration struct. No behavior, no methods.
- It is host-only C++20 with no CUDA and no dependency on the core compute layer or
  the GPU-device layer.
- Its only project include is the shared configuration header, from which it pulls
  in `FilterConfig` (the live in-tile threshold settings).

Because it holds no GPU types and no compute-layer types, it can cross layer
boundaries unchanged and be included freely wherever a resolved filter plan needs to
be passed around.

---

## 4. The filter invariant: whole-SNP and whole-sample only

There is one hard rule this struct is built to enforce: **every filter decision is
either SNP-global or sample-global — never per-(population, SNP).**

- `snp_keep` is one flag per SNP. A SNP is either kept for *all* populations or
  dropped for *all* of them. There is no way to keep a SNP for one population and
  drop it for another.
- `kept_samples` is one index list over the *whole* sample axis. A sample is either
  kept for all SNPs or dropped for all SNPs.

This is not merely a convention; the struct cannot even express a per-(population,
SNP) drop, and that is deliberate. The f2 statistics that steppe computes are
expressed as matrix multiplications in which each SNP corresponds to a column of the
data. Dropping a SNP is done by zeroing its entire column, which cleanly and
symmetrically removes that SNP's contribution from every population at once. A drop
that applied to a single population at a single SNP could not be expressed as zeroing
a whole column, so it would break that matrix formulation of the computation. Keeping
the plan restricted to whole-SNP and whole-sample decisions is what keeps the fast
matrix-based f2 path correct.

---

## 5. Fields of the `FilterPlan` struct

| Field | Type | Default | Meaning |
|---|---|---|---|
| `snp_keep` | `vector<bool>` | empty | The per-SNP keep mask, one entry per SNP in file order. `snp_keep[s]` is true if and only if SNP `s` survives *all* SNP-level filters — minor-allele-frequency floor, per-SNP missing-data limit, include/exclude membership, the flag-gated monomorphic and transversion filters, and the autosome-only filter. This is SNP-global: a false entry drops that SNP for every population, never for just one. Its length equals the number of SNPs considered. |
| `n_snp_kept` | `size_t` | `0` | The number of kept SNPs, i.e. the count of true entries in `snp_keep`. Cached alongside the mask so callers that need the surviving count do not have to re-scan the whole mask. Always no larger than the length of `snp_keep`. |
| `kept_samples` | `vector<size_t>` | empty | The indices of the kept samples over the whole sample (individual) axis, in ascending order. This is the result of the conditional per-sample missing-data pre-pass. When that filter is turned off (its threshold is 1.0 or greater, so no sample can exceed it), this list holds *every* sample index — nothing is dropped. Sample-global: a sample here is kept for all SNPs, and a sample absent here is dropped for all SNPs. |
| `in_tile` | `FilterConfig` | default `FilterConfig` | A by-value copy of the live filter thresholds that the tile-production stage still needs when it applies the remaining cheap, per-tile filters (the minor-allele-frequency and per-SNP missing-data thresholds, plus the flag-gated options). Carried by value so the plan is fully self-contained. These are the *same* threshold values that the SNP-filter stage used to build `snp_keep`, so there is a single source of truth and no re-reading of settings later. |

For the full meaning of each `FilterConfig` field carried in `in_tile`, see the
`config.hpp` reference.

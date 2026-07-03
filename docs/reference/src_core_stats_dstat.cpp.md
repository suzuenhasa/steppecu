# `dstat.cpp` reference

## 1. Purpose

`src/core/stats/dstat.cpp` implements `run_dstat`, the entry point that computes
the **normalized D statistic** (the classic "ABBA–BABA" test, also called
qpDstat) directly from genotype files.

The caller hands in three input files (the genotype matrix, the SNP table, and
the individual table), a list of populations to read, and a list of
**quadruples** — each quadruple is four population indices `(p1, p2, p3, p4)`.
For every quadruple, `run_dstat` returns four numbers:

- **est** — the D statistic itself (a value between −1 and +1).
- **se** — its standard error, estimated by a block jackknife over the genome.
- **z** — the z-score (`est / se`).
- **p** — the two-sided p-value derived from the z-score.

D is a *ratio* of two per-SNP sums accumulated across the genome: a numerator
and a denominator. Reading it directly from genotypes (rather than from a
precomputed f2 cache) is what makes this the "genotype path." The whole routine
is written to reproduce ADMIXTOOLS 2's `qpDstat` in genotype mode — specifically
its `allsnps=TRUE`, `f4mode=FALSE`, `blgsize=0.05` configuration — number for
number.

The result also carries a precision tag of native double precision, because the
delicate numerator/denominator accumulation in the jackknife runs in extended
precision on the host reference path and native double precision on the GPU, not
the faster emulated arithmetic used elsewhere.

---

## 2. Relationship to the f4 statistics

`run_dstat` is the close sibling of `run_f4` and `run_f4ratio`. All three read
genotype files rather than the f2 cache, and all three share large pieces of
machinery:

- **Shared front end.** They all reuse the same genotype decode front end — the
  code that opens the genotype reader, reads the requested populations and the
  SNP table, reads the packed genotype tile, and turns packed genotypes into
  per-SNP allele frequencies. This is the same decode used by the standalone f2
  extraction tool.
- **Where D diverges.** After decoding, `run_dstat` splits off into its own
  per-SNP D reduction kernel (`dstat_block_reduce` in the GPU code), which
  produces the numerator and denominator sums per genome block.
- **Shared jackknife.** The final uncertainty estimate is *not* a private host
  loop. It goes through the **same** on-device ratio block-jackknife engine that
  f4-ratio uses — one shared backend routine, reached here through
  `dstat_blocks_jackknife`. D and f4-ratio are both ratios of two sums, so they
  can share the leave-one-out math (Section 7).

The one-line way to think about it: `run_dstat` reuses the f4-ratio *decode* and
the f4-ratio *jackknife*, and only the middle step — the per-SNP D reduction — is
its own.

---

## 3. The three parity pins

Matching ADMIXTOOLS 2 exactly requires three specific choices, each verified
against the reference on real data. Getting any one of them wrong changes the
reported D. They are described here as "pins" because they are frozen: they are
what makes the output byte-identical to the reference, and they must not drift.

### Pin 1 — allele frequency uses forced diploid counts

ADMIXTOOLS 2 computes each population's allele frequency as the plain ratio
`(sum of reference-allele counts) / (2 × number of observed alleles)` — with **no
pseudo-haploid adjustment**. To reproduce that, `run_dstat` forces every sample
to be treated as diploid (ploidy = 2) when it decodes allele frequencies.

This is a deliberate departure from the f2-extraction path, which auto-detects
each sample's ploidy and applies a pseudo-haploid adjustment. That adjustment
would be *wrong here*: this kind of dataset mixes pseudo-haploid and diploid
samples, and applying the per-sample adjustment flips the **sign** of a D that
sits near zero. Forcing diploid makes the frequencies match the reference's
`gmat_to_aftable` output exactly.

### Pin 2 — genome blocks match the reference partition

The jackknife blocks are built by walking the kept SNPs in order and cutting a
new block whenever the genetic distance since the last cut exceeds the block
size, resetting at each chromosome boundary. This block assignment is
byte-identical to ADMIXTOOLS 2's block-length routine, computed over the
autosome-filtered SNP set. The block size default is `0.05` Morgans (5
centimorgans), matching the reference.

### Pin 3 — the SNP mask is `allsnps=TRUE`

In `allsnps=TRUE` mode, the only SNP filters are:

- **Autosomes only** — keep chromosomes 1 through 22, drop the sex chromosomes
  and everything else, matching the reference default.
- **Per-quadruple finiteness** — within each block, a SNP counts toward a
  quadruple only if all four of that quadruple's populations have an observed
  allele frequency at that SNP. This mask is applied inside the D kernel.

There is deliberately **no** maximum-missingness filter, **no** minor-allele
frequency cutoff, and **no** monomorphic-SNP dropping. This is what `allsnps=TRUE`
means, and it is why the finiteness check is per-quadruple rather than global —
different quadruples can keep different SNPs.

---

## 4. Named constants

Two small constants are defined at the top of the file.

| Constant | Value | What it's for |
|---|---|---|
| `kPloidyDiploid` | `2` | The forced-diploid ploidy that implements Pin 1. Every sample is decoded as diploid so that allele frequencies are the plain reference-count / (2 × observed) ratio, with no pseudo-haploid adjustment. This is the value that makes the frequencies match ADMIXTOOLS 2. |
| `kPrimaryGpu` | `0` | The index of the single GPU this entry point uses. `run_dstat` is a single-GPU routine; it always uses the primary device. This mirrors the same choice in `f4.cpp` and `f4ratio.cpp`. |

---

## 5. The population-axis contract

This is a subtle but important invariant about *which* populations get read and
how the quadruple indices line up.

`run_dstat` reads **only** the populations named in the passed-in `pop_union`
list — not the entire individual file. So a 4-population D test run against a
dataset with tens of thousands of individuals still only decodes those few
populations. The populations are read as an explicit partition, sorted ascending
by label.

The critical guarantee is that the **caller resolved the quadruple indices
against this exact same ordering.** The application layer builds its
population-index resolver over the same explicit, label-sorted partition, so
every index appearing in a quadruple is a valid row of the decoded frequency
matrix. The two orderings cannot disagree because they are produced the same way
from the same input.

**Failure handling split.** A file-level fault — a missing or unreadable input
file — propagates *out* as an exception, which the application's error handling
maps to a nonzero I/O exit code. A *domain* outcome — for example, a quadruple
that has no surviving blocks — is **not** an exception; it is reported as a
not-a-number sentinel in that row (Section 9).

---

## 6. The processing pipeline

`run_dstat` runs four stages in order.

### Stage 1 — decode front end

Open the genotype reader, read the requested populations and the SNP table, read
the packed genotype tile, and set up the decode view with forced-diploid ploidy
(Pin 1). This is the shared front end from Section 2.

### Stage 2 — decode, autosome keep, and lockstep subset

Turn packed genotypes into per-SNP allele frequencies, then keep only the
autosome SNPs (Pin 3). The frequency matrix and the finiteness mask are subset to
the kept SNPs in **lockstep** with the kept chromosome and genetic-position
arrays, so all of them stay aligned to the same kept SNP set in the same order.

This stage has two implementations that produce the **identical** kept set, kept
order, and kept coordinate arrays:

- **Device-resident path** (used whenever a real GPU is present). The decode, the
  autosome keep-mask, and the compaction of the frequency matrix all run on the
  GPU. The compacted results and the small kept coordinate arrays stay in GPU
  memory. This avoids copying the roughly gigabyte-scale full frequency matrix
  back to the host, avoids a host-side filter loop, and avoids re-uploading the
  compacted result.
- **Host path** (used only by the CPU reference backend). The full frequency
  matrix comes back to the host and a plain loop keeps the autosome rows.

A per-SNP physical-position array is carried alongside. It is used only as a
fallback for block assignment when a dataset ships with no genetic map (the
genetic-position column is all zeros); on a real map it is ignored.

### Stage 3 — assign blocks

Partition the kept autosome SNPs into jackknife blocks (Pin 2). If the dataset
has a real genetic map, blocks are cut by genetic distance; if the map is all
zeros, blocks fall back to a fixed physical-distance window instead — the same
fallback the reference uses.

### Stage 4 — fused reduce and jackknife

The per-SNP D reduction and the block jackknife happen in **one** backend call.
On the GPU, the per-quadruple, per-block numerator, denominator, and count sums
stay resident in GPU memory and feed the jackknife kernel directly — there is no
copy of the per-quadruple/per-block sums back to the host and no host-side
per-quadruple jackknife loop. On the CPU reference backend, the same call reduces
and then delegates to the extended-precision jackknife oracle. Both paths compute
the p-value in place. The returned estimate, standard error, z-score, and
p-value become the result.

---

## 7. The block jackknife

The standard error comes from a **leave-one-out block jackknife** over genome
blocks. Because D is a ratio (numerator sum over denominator sum), this is a
*ratio* jackknife — the same one f4-ratio uses. It runs per quadruple, over that
quadruple's surviving blocks (the blocks where the SNP count is greater than
zero).

The steps, keeping the real formulas:

1. Each surviving block `b` contributes a numerator sum and a denominator sum.
2. The **leave-one-out** value for block `b` is the total over all blocks with
   block `b` removed. Taking the leave-one-out numerator over the leave-one-out
   denominator gives that block's ratio estimate `R_b`.
3. The overall estimate combines the per-block estimates. With `cnt_b` the SNP
   count in block `b` and `Σcnt` the total SNP count:
   - `tot = weighted-mean(R_b, weight = 1 − cnt_b/Σcnt)`
   - `est = mean(tot − R_b) × (number of blocks) + weighted-mean(R_b, weight = cnt_b)`
4. The variance uses a per-block "pseudo-value." With `h_b = Σcnt / cnt_b`:
   - `xtau_b = (h_b·tot − (h_b−1)·R_b − est)² / (h_b − 1)`
   - `var = mean(xtau_b)`
5. The final numbers: **D = est**, **se = √var**, **z = D/se**, and **p** is the
   two-sided p-value of `z` (matching the reference's z-to-p conversion).

Two details make this match the reference and stay numerically sound:

- The block weights are the **per-block SNP counts for that quadruple**, not a
  single shared block size. Different quadruples can keep different SNPs
  (Section 3), so their block weights differ.
- The survivor mask is purely "count greater than zero" — there is no separate
  missing-block sentinel.
- All of the numerator/denominator accumulation is done in **extended precision**
  (long double on the host reference, native double precision on the GPU),
  because the numerator and denominator are cancellation-prone sums where the
  faster emulated arithmetic would lose accuracy.

---

## 8. Device-resident and CPU-reference paths

The routine runs one of two ways depending on whether a real GPU is present, and
both are guaranteed to produce the same result.

- **GPU present.** The whole compute path stays on the device. The decode, the
  autosome compaction (Stage 2), and the per-quadruple/per-block sums (Stage 4)
  live in GPU memory and are consumed by the next kernel without a round trip to
  the host. This is what avoids the large frequency-matrix copy and the
  per-quadruple host loop that an earlier design paid for.
- **No GPU (CPU reference backend).** The same entry points run on the host. This
  path exists as the correctness oracle: it computes the jackknife in extended
  precision and is what the GPU path is validated against.

The point of the seam is that the two paths are *interchangeable in output*: same
kept SNP set, same block assignment, same reference math. The GPU path is faster;
the CPU path is the thing that proves the GPU path is right.

---

## 9. Degenerate outcomes and error handling

A "degenerate" run is one where there is nothing to compute — for example, no
populations or no SNPs decoded, no SNP surviving the autosome keep, or no block
formed. These are legitimate *domain* outcomes, not errors.

The helper `fill_nan` handles them uniformly. It writes a not-a-number sentinel
into the four statistic columns (est, se, z, p) for every requested row and sets
the status to OK. It writes **only** those four columns plus the status, so the
population-label rows and the precision tag that the caller already filled in are
preserved — it fills in place rather than building a fresh result. This is
deliberately a shared helper so that all three degenerate guards (empty
population or SNP axis, no kept autosome SNP, no block) return the same clean
not-a-number result instead of repeating the fill in three places.

The contract, restated: a *domain* outcome (nothing to compute, or a single
degenerate quadruple) is a not-a-number sentinel with an OK status — never a
thrown exception. Only a genuine *file* fault (missing or unreadable input)
throws, and that is handled one level up as an I/O error.

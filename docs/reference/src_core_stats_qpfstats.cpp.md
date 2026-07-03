# `qpfstats.cpp` reference

## 1. Purpose

`src/core/stats/qpfstats.cpp` implements `run_qpfstats`, the joint f-statistic
smoother. It reads a genotype dataset directly, jointly fits *every* f2, f3, and f4
statistic over a chosen set of populations, and produces one smoothed per-block f2
tensor that the downstream tools (qpAdm, f4, qpGraph) consume.

The point of the smoother is that the individual f2/f3/f4 estimates from real data are
noisy and not perfectly self-consistent — the f-statistics obey exact linear identities
(for example every f3 and f4 can be written in terms of f2 values), but the raw
estimates violate those identities because of sampling noise. This file fits all of the
raw statistics at once against those identities using a ridge-regularized least-squares
regression, and reads back a clean, mutually consistent set of per-pair f2 values that
respects the identities exactly. The result is more accurate than estimating each f2 in
isolation.

The implementation reproduces ADMIXTOOLS 2's `qpfstats()` (its R ridge-regression path)
and is pinned bit-for-bit to that tool's reference output. It is a *genotype-path* tool:
it does not build or read an f2 cache. Instead it reuses, unchanged, the same genotype
decoding and D-statistic numerator machinery that the D-statistic tool uses, and adds
only the regression on top.

Almost all of the heavy work — the numerator reduction, the block jackknife, the
regression solve, and the recentering — happens in one fused call into the compute
backend, so the arithmetic runs on the GPU (with a CPU reference implementation used
only for validation). The code in this file itself contains no CUDA. The host-side work
that remains is small and independent of the data size: building the list of population
combinations and the regression design matrix, and scattering the final numbers into the
output tensor.

---

## 2. Named constants

| Constant | Value | What it's for |
|---|---|---|
| `kPloidyDiploid` | `2` | The forced ploidy. Every sample is treated as diploid (allele frequency computed as reference-count over allele-count over 2), which is what ADMIXTOOLS 2 does on this path. This is deliberately *not* the per-sample auto-detected ploidy that the f2-cache extraction path uses — qpfstats runs through the D-statistic numerator engine, which assumes plain diploid frequencies. |
| `kPrimaryGpu` | `0` | The index of the single GPU this tool runs on. qpfstats is a single-GPU tool; it always uses device 0. |
| `kRidge` | `1e-5` | The ridge (Tikhonov) regularization added to the regression normal-matrix diagonal. Matches ADMIXTOOLS 2's `qpfstats_regression` default of `ridge = 1e-5`. It keeps the least-squares system well-conditioned and invertible even when the design matrix is rank-deficient. |

---

## 3. The population-pair index

Every pair of distinct populations `(i, j)` maps to a single column of the regression
design matrix. `pair_index(i, j, npop)` computes that column number.

The pairs are enumerated in row-major order over the upper triangle of the population ×
population grid: `(0,1), (0,2), …, (0,npop-1), (1,2), (1,3), …`. There are
`npop·(npop-1)/2` of them. The function is symmetric — it swaps `i` and `j` first if
`i > j` — so `pair_index(i, j)` and `pair_index(j, i)` return the same column.

The closed-form index is

```
i·npop − i·(i+1)/2 + (j − i − 1)
```

which is "the number of pairs in all the earlier rows (`0` through `i-1`), plus the
offset of `j` within row `i`."

This ordering is not arbitrary: it was verified numerically to match, exactly, the
symmetric pair ordering that ADMIXTOOLS 2's `construct_fstat_matrix` uses internally
(its `indmat[i,j]` for `i<j`, zero-based). Because the design matrix and its column
meanings must line up with the reference tool, this index and that ordering must stay
in agreement.

---

## 4. Population combinations: the f2 / f3 / f4 set

### The `PopComb` struct

A `PopComb` is one population combination, four indices `{p1, p2, p3, p4}` into the
sorted population list. Every f-statistic in this file — f2, f3, and f4 alike — is
computed from the *same* numerator formula, `(a − b)(c − d)`, where `a = Q[p1]`,
`b = Q[p2]`, `c = Q[p3]`, `d = Q[p4]` are the allele frequencies of the four indexed
populations. The three statistic types differ only in how the indices alias each other:

| Statistic | Index pattern | Numerator |
|---|---|---|
| f2 | `(A, B, A, B)` | `(a − b)(a − b) = (a − b)²` |
| f3 | `(A, B, A, D)` | `(a − b)(a − d)` |
| f4 | `(A, B, C, D)` | `(a − b)(c − d)` |

Using one formula for all three is what lets the numerator engine process the entire
mixed set in a single batched pass.

### Building the combination set

`build_popcomb_and_design` builds the full set of combinations over the `npop` **sorted**
populations, in exactly the order ADMIXTOOLS 2 produces them. Reproducing that order
precisely matters, because the combinations become the rows of the design matrix and the
per-row jackknife estimates must correspond to the reference tool's rows. For `npop = 9`
the counts are `36 + 252 + 378 = 666` combinations, verified against ADMIXTOOLS 2 source.

**f2** — one combination `(i, j, i, j)` for every pair `i < j`. This is `C(npop, 2)`
combinations, in ascending pair order.

**f3** — starts from every triple `i < j < k` (`C(npop, 3)` of them, in the ascending
"column-major" order R's `combn` produces). Each triple is expanded into three rotations
`{0,1,2}`, `{1,2,0}`, `{2,0,1}`, and then each rotation `(P1, P2, P3)` is remapped to the
combination `(P1, P2, P1, P3)`. The ordering here is **block-wise**: all triples in
rotation 0, then all triples in rotation 1, then all triples in rotation 2. This
reproduces ADMIXTOOLS 2 stacking the three rotation blocks on top of each other. The
result is `C(npop, 3)·3` combinations.

**f4** — starts from every quad `i < j < k < l` (`C(npop, 4)` of them, in ascending
order). Each quad is expanded into three rotations `{0,1,2,3}`, `{0,2,1,3}`, `{0,3,1,2}`.
The ordering here is **interleaved**, not block-wise: quad-0 rotation-0, quad-0
rotation-1, quad-0 rotation-2, quad-1 rotation-0, and so on. This is the one subtle
difference from the f3 layout — ADMIXTOOLS 2 interleaves the f4 rotations per quad
(via its `slice(rep(...))`) rather than stacking them in blocks. The result is
`C(npop, 4)·3` combinations.

Getting the block-vs-interleaved distinction wrong would produce the right *set* of
combinations in the wrong *order*, which would silently misalign every design-matrix row.

---

## 5. The regression design matrix

The design matrix `x` encodes each combination's f-statistic as a linear expression in
the per-pair f2 values — the exact linear identities the regression fits against. It has
one row per combination (`npopcomb` rows) and one column per population pair (`npairs`
columns), and it is stored **column-major**, indexed as `x[c + npopcomb·p]`. This
reproduces ADMIXTOOLS 2's `construct_fstat_matrix` exactly.

### The four coefficient writes

For each combination `(p1, p2, p3, p4)`, the row gets four coefficient writes:

| Coefficient | Column |
|---|---|
| `+1` | `pair(p1, p4)` |
| `+1` | `pair(p2, p3)` |
| `−1` | `pair(p1, p3)` |
| `−1` | `pair(p2, p4)` |

This is the standard expansion of an f4 statistic into f2 terms:
`f4(p1,p2;p3,p4) = ½·[ f2(p1,p4) + f2(p2,p3) − f2(p1,p3) − f2(p2,p4) ]`.

### Two subtle rules — assignment, not accumulation

The four writes are **assignments** (`out[col] = ±1`), not additions. This exactly
mirrors ADMIXTOOLS 2, and it matters in two cases:

- **Colliding columns.** When two of the four pairs land on the *same* column, the later
  assignment overwrites the earlier one — the value stays `1`, it does not become `1+1=2`.
  This happens for a pure-f2 row, where `pair(p1,p4)` and `pair(p2,p3)` are the same pair.
- **Diagonal pairs.** A "pair" `(i, i)` of a population with itself is not a real column
  (ADMIXTOOLS 2 marks it not-available in its `indmat`). The write is simply skipped — a
  no-op. This drops the `−1` writes on `pair(p1,p3)` and `pair(p2,p4)` for f2 and f3 rows
  where those indices coincide.

### The pure-f2 normalization

After the four writes, if the row is a pure f2 row (`p1==p3` and `p2==p4`), the entire
row is multiplied by 2. This restores the f2 normalization relative to the f4 identity.
Finally, **every** entry of the whole matrix is halved (the `out/2` in
`construct_fstat_matrix`).

Worked example for an f2 row `(a, b, a, b)`: the `+1` at `pair(a,b)` and the `+1` at
`pair(b,a)` are the same column (stays `1`); both `−1` writes are on diagonal pairs and
are dropped. After the four writes the row is `1` at `pair(a,b)` and zero elsewhere. The
pure-f2 doubling makes it `2`, and the final halving makes it `1`. So an f2 row ends up
with a single nonzero coefficient, `+1` at its own pair — exactly as it should.

### The integer-overflow guard

The combination count grows roughly as `C(npop, 4)·3`. `npopcomb` is stored as a signed
`int` (kept as an `int` because it is the axis size the numerator engine expects). Before
that narrowing conversion, the code checks whether the true count exceeds `INT_MAX` and
throws a `std::runtime_error` if so. Without this guard, a very large population set could
wrap `npopcomb` negative, and that wrapped value would then size the design matrix and
index into it — a silent out-of-bounds write. The check is a hard throw (not a debug-only
assertion) so it also protects release builds, where it is exactly the case that must not
slip through.

---

## 6. `run_qpfstats`: the pipeline

`run_qpfstats` is the single public entry point. It takes the genotype/SNP/individual
file triple, the list of populations, the jackknife block size (in Morgans), a precision
policy, and the GPU resources, and returns a `QpfstatsResult` holding the smoothed f2
tensor and the sorted population labels. It runs the following steps.

### Step 0 — the sorted population set

The input population list is sorted and de-duplicated. Sorting matters because the sorted
order is the population axis order for everything downstream (it is what ADMIXTOOLS 2 uses
for its dimension names), so the output tensor's rows and columns are always in sorted
label order. At least **four** distinct populations are required — fewer cannot form a
non-degenerate f4 basis — and fewer returns an `InvalidConfig` status.

### Step 1 — decode and keep autosomes

The genotype decode front-end is the shared helper that opens the reader, reads only the
requested populations (in sorted order), reads the SNP table, and reads the genotype
tile. Ploidy is forced diploid. If the number of populations actually decoded does not
match the number requested (a requested population was missing from the data), it returns
`InvalidConfig`. An empty dataset returns `Ok` with an empty result.

Decoding to allele frequencies and then keeping only autosomal SNPs (chromosomes 1
through 22) happens by one of two paths that produce an identical kept set, kept order,
and kept chromosome/position arrays:

- **On the GPU** (a real device is present): a single fused call decodes, builds the
  autosome keep-mask, and compacts the kept allele-frequency and variance arrays entirely
  on the device. The compacted arrays stay resident in GPU memory; only the small kept
  chromosome and position vectors come back to the host. This avoids copying roughly a
  gigabyte of per-SNP data back and forth.
- **On the CPU reference backend** (no device, used only for validation): it decodes on
  the host and then loops over SNPs, keeping the autosomal ones and copying their
  frequency/variance columns into host vectors.

Three position arrays are carried for the kept SNPs: chromosome, genetic position (in
Morgans), and physical position (in base pairs). The physical position is only used as a
fallback when a dataset ships with no genetic map (all genetic positions zero); on a real
map it is ignored.

### Step 2 — assign jackknife blocks

The kept autosomal SNPs are partitioned into contiguous jackknife blocks by genetic
position, reproducing ADMIXTOOLS 2's block-length assignment. If the dataset has no
genetic map, the partition falls back to fixed-width windows of physical position. From
the partition, a per-block SNP count (`block_lengths`) is computed. That count serves two
purposes later: it is the weight for the recentering jackknife, and it is the block-size
metadata written into the output tensor.

### Step 3 — build the combinations and design matrix

`build_popcomb_and_design` (sections 4 and 5) builds the combination list and the design
matrix. The combinations are also flattened into a `4·npopcomb` integer table laid out
`{p1, p2, p3, p4}` per combination — the exact layout the D-statistic numerator kernel
reads. All of this is independent of the genotype data and is cheap host work.

### Steps 4–8 — the fused reduce → jackknife → smooth → recenter

The entire numerical core is one call into the compute backend,
`qpfstats_blocks_smooth`, which keeps its intermediate data resident on the device for
the whole computation. It performs, in order:

1. **Numerator reduction.** The D-statistic numerator engine computes, for every
   combination and every block, the finiteness-aware sum of the `(a − b)(c − d)`
   numerator over that block's SNPs, along with a count of contributing SNPs. It runs in
   "numerator only" mode (the D-statistic denominator is computed by the shared kernel but
   ignored here). The reduction is batched over both the combination axis and the block
   axis on the GPU. Per-block estimates are `numsum / count`, and are not-a-number where a
   block contributed no SNPs to a combination.
2. **Per-combination jackknife.** A leave-one-block-out block jackknife over the
   per-block numerator estimates yields one global estimate per combination.
3. **The smoothing solve.** The regression normal matrix is `A = xᵀx + ridge·I`. The
   global smoothed pair values solve `A · bglob = xᵀ · y`; the per-block smoothed pair
   values solve the corresponding per-block systems `A_blk · b[:,blk] = xᵀ · ymat[:,blk]`,
   where combinations whose block estimate is not-a-number are downdated out of that
   block's system, and an all-not-a-number case collapses to zero. This is ADMIXTOOLS 2's
   `qpfstats_regression`, reformulated to run as batched matrix operations rather than a
   host loop over blocks and combinations.
4. **The per-pair recentering shift.** For each pair, a block jackknife over the smoothed
   per-block pair values gives a jackknife estimate of the smoothed tensor; the
   recentering shift is the difference between the global smoothed value and that
   jackknife estimate. The block SNP counts are the jackknife weights.

Precision is split: the matrix-multiply-heavy sub-steps run in the requested precision
mode (emulated double precision by default), while the numerically delicate steps — the
jackknives and the Cholesky/solve — run in native double precision. Only three small
arrays cross back to the host: the per-block smoothed pair values, the global smoothed
values, and the recentering shifts. A non-`Ok` status from the seam is propagated
straight out.

The CPU reference backend implements the same seam by composing its existing reference
routines, so it produces the identical result for validation.

### Step 7–8 — scatter into the output tensor

The smoothed per-block pair values are scattered into the output f2 tensor, shaped
`[npop, npop, n_block]`. For each off-diagonal population pair `(i, j)`, the tensor entry
is the smoothed per-block value **plus** that pair's recentering shift, written
symmetrically into both `(i, j)` and `(j, i)`. The diagonal stays zero (a population's f2
with itself is zero by definition). Each block's SNP count is recorded as the block size.
This recentered tensor is the smoothed f2 tensor the downstream tools consume.

### Step 9 — the per-pair valid-count field

The output tensor also carries a per-pair, per-block valid-count field (`vpair`) that the
f2-cache writer uses to detect blocks that contributed nothing. Because the smoothing
regression imputes values, the smoothed tensor has no natural per-pair SNP count. So each
off-diagonal pair records the block's overall kept-SNP count: any block that contributed
to any combination has a positive count, which is enough for the missing-block detector
to work correctly.

---

## 7. The three parity pins

Three specific choices, inherited from the D-statistic genotype engine and verified
against real data, are what make this tool's output match ADMIXTOOLS 2 bit-for-bit. They
are load-bearing and must not drift:

1. **Forced diploid frequencies.** Allele frequency is reference-count over allele-count
   over 2 for every sample — not per-sample auto-detected ploidy.
2. **Block assignment matches ADMIXTOOLS 2.** The jackknife block partition reproduces
   ADMIXTOOLS 2's block-length rule exactly, including the physical-position fallback for
   map-less data.
3. **All-SNPs finiteness, per combination per block.** A SNP contributes to a given
   combination-and-block if and only if it is finite for that combination there — computed
   independently per combination. There is no shared missingness cap, no minor-allele-
   frequency filter, and no monomorphic-SNP dropping; autosomes-only is on.

---

## 8. What runs on the GPU versus the host

This file contains no CUDA. The whole numerical core — numerator reduction, jackknife,
smoothing solve, and recentering — goes through the single fused backend call, which is
dispatched to GPU kernels in production and to the CPU reference implementation for
validation.

The host work that remains is small and data-independent: building the combination list
and the design matrix (which depend only on the population count, not the genotypes), and
scattering the final small arrays into the output tensor. There is deliberately **no**
host loop over SNPs, over jackknife blocks, or over combinations. An earlier design
materialized the full per-combination-per-block numerator and count arrays on the host
(roughly 1.7 gigabytes each) and ran the large jackknife as a host long-double loop; that
copied enormous buffers back and left the GPU idle for half of the run. Keeping the
numerator, the jackknives, the solve, and the recentering all resident on the device
eliminates both the large copies and the host-bound loop.

The reference (long-double) versions of the two jackknives — the per-combination estimate
and the per-pair recenter estimate — live in a shared internal header so the CPU
reference oracle and the production path stay in agreement, but on the production path
they run on the device inside the fused seam and the host never loops over combinations or
pairs.

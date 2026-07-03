# `f2_estimator.hpp` reference

## 1. Purpose

`src/core/internal/f2_estimator.hpp` holds the small set of scalar functions that
define, in exactly one place, how a single element of the **f2 statistic** is
computed. f2 is a population-genetics quantity: for a pair of populations it
measures how much their allele frequencies differ, averaged over the SNPs both
populations have data for, with a bias correction subtracted so that sampling noise
doesn't inflate the estimate.

The reason these formulas live in one header is that steppe computes f2 two
different ways and the two must never disagree:

- a **CPU reference** implementation, used as the trusted oracle that results are
  validated against, and
- the **GPU production path**, which is far faster.

If each of those re-typed the f2 formula, they could quietly drift apart. Instead,
both call the functions here, so there is a single source of the numerics.

These are pure scalar functions — each takes a few doubles and returns a double or a
bool. They can be unit-tested on the CPU with no GPU involved. They are *not* the
structure of the fast GPU path; the GPU computes f2 by reformulating the whole
calculation as three large matrix multiplications (described in sections 5 and 6).
The functions here are the per-element numerics that get invoked inside that fast
path and inside the CPU oracle alike.

The formulas match ADMIXTOOLS 2's bias-correction conventions and are pinned to
saved reference values (goldens), so they are effectively frozen: you may rename
things, but changing a formula would change a reported result.

---

## 2. How this header compiles for both CPU and GPU

This header is `#include`d by two very different kinds of source file:

- ordinary host source files, compiled by the normal C++ compiler, which has never
  heard of CUDA's `__host__` / `__device__` keywords, and
- device source files, compiled by the CUDA compiler (nvcc), where those keywords
  are required for a function that must run on the GPU.

To let the exact same function text serve both, every function here is marked with
`STEPPE_HD`. That is a macro (defined once in `core/internal/host_device.hpp`, and
shared with the decode code) that expands to the CUDA qualifiers when nvcc is
compiling, and to nothing at all otherwise. The practical payoff is that one copy of
each function compiles and unit-tests on the CPU and also runs on the GPU — which is
the whole point of having one shared primitive.

---

## 3. The stacked-S layout factor (`kF2StackedBlocks`)

```
inline constexpr int kF2StackedBlocks = 2;   // value: 2
```

This is the single home for the number `2` that shows up as the row-block count in
the GPU's matrix-multiply reformulation of f2. In that reformulation the GPU builds
one tall matrix, called S, by stacking two blocks on top of each other:

- an upper block (rows 0 through P−1) holding the squared allele-frequency terms, and
- a lower block (rows P through 2P−1) holding the heterozygosity-correction terms,

where P is the number of populations. Because there are two stacked blocks each P
rows tall, S has 2P rows, and its leading dimension is 2P.

Naming this `2` once means the factor is defined in a single spot rather than being
open-coded as `2 * P` in each GPU source file that consumes it (both the
single-block feeder kernel and the grouped batched kernel). If a third stacked block
were ever added, every consumer stays consistent instead of some of them silently
still assuming two. This is a structural constant of the matrix layout, not a
performance-tuning knob.

---

## 4. The heterozygosity bias correction (`het_correction`)

```
het_correction(double q, double n, bool valid) -> double
```

Computes, for one population at one SNP, the small quantity that is subtracted to
remove sampling bias:

```
hc = q * (1 - q) / max(n - 1, kHetCorrDenomFloor)
```

- `q` is the reference-allele frequency at that SNP, a number between 0 and 1.
- `n` is the **non-missing haploid count** — that is, twice the number of
  non-missing diploid samples, or once the number of non-missing pseudo-haploid
  samples for ancient DNA. It is a per-SNP count, never a hardcoded sample size.
- `valid` is the caller's validity bit for this entry.

`kHetCorrDenomFloor` is the value `1.0` (defined in `config.hpp`). It is the floor in
the `max(n - 1, 1)` denominator, matching ADMIXTOOLS 2's convention. Its job is to
prevent a divide-by-zero when a population has only a single non-missing haploid
sample, where `n - 1` would be 0.

When `valid` is false the function returns `0.0`. Contributing zero for an invalid
entry is exactly what makes the masked matrix multiplication on the GPU come out
correct — an invalid entry is zero-filled and adds nothing to the sums.

---

## 5. The per-SNP f2 summand — the cancellation-free form (`f2_term`)

```
f2_term(double p_i, double p_j, double hc_i, double hc_j) -> double
```

Returns the unbiased contribution of one SNP to f2 for a population pair (i, j), in
the numerically careful form the CPU reference oracle uses:

```
(p_i - p_j)² - hc_i - hc_j
```

The key detail is that it forms the difference `p_i - p_j` first and then squares
it. It deliberately does **not** use the algebraically-equal expanded form
`p_i² - 2·p_i·p_j + p_j²`. The two are equal on paper, but the expanded form
subtracts large, nearly-equal numbers, which loses precision — a phenomenon called
catastrophic cancellation, where the leading digits cancel and you are left with the
noisy trailing digits. Squaring the difference directly avoids that at the per-SNP
level.

`hc_i` and `hc_j` are the two populations' heterozygosity corrections from
`het_correction` (section 4). The caller sums this term over all SNPs that are valid
in *both* populations, then divides by that joint count to get the final f2(i, j).
This is the CPU path; the GPU computes the identical statistic through the matrix
reformulation and the assembly function in section 6.

---

## 6. Assembling the f2 numerator from the GPU sums — the expanded form (`assemble_f2_numerator`)

```
assemble_f2_numerator(double sumsq_i, double sumsq_j,
                      double cross,
                      double hsum_i, double hsum_j) -> double
```

This is the counterpart to `f2_term` for the GPU path. The GPU cannot cheaply form
each per-SNP difference; instead it produces, via three matrix multiplications, four
running sums over the jointly-valid SNPs of a population pair, and this function
combines them into the f2 numerator:

```
sumsq_i + sumsq_j - 2·cross - hsum_i - hsum_j
```

where the inputs are:

| Input | What it is |
|---|---|
| `sumsq_i` | Sum over SNPs of `p_i²` (population i's squared frequencies) |
| `sumsq_j` | Sum over SNPs of `p_j²` |
| `cross`   | Sum over SNPs of `p_i · p_j` (the cross term) |
| `hsum_i`  | Sum over SNPs of population i's heterozygosity corrections |
| `hsum_j`  | Sum over SNPs of population j's heterozygosity corrections |

This is the **expanded** `a² - 2ab + b²` form — the same one `f2_term` avoids — and
this is precisely where the catastrophic cancellation lands, because
`sumsq_i + sumsq_j - 2·cross` subtracts large sums of similar magnitude.

Because of that, this assembly is held in **native double precision in every
precision mode**. steppe's precision setting controls only the big matrix
multiplications that *produce* the four input sums; the final small combination here
is always done in full native double precision so the cancellation does not throw
away accuracy. The `2` in the formula is a genuine mathematical constant from
`a² - 2ab + b²`, not a tunable.

---

## 7. Finalizing f2 for one pair (`finalize_f2`)

```
finalize_f2(double numerator, double vpair) -> double
```

Turns the numerator (from section 6) into the block-level f2 value for a pair by
dividing by the count of SNPs valid in both populations:

```
f2 = (vpair > 0) ? (numerator / vpair) : 0
```

`vpair` is that joint-valid count. It is a whole number but is carried as a double.
When `vpair` is 0 the numerator is also 0, so returning 0 avoids a 0-divided-by-0.
This matches ADMIXTOOLS 2's guard for the same situation. (`vpair > 0.0` is used
rather than an integer test purely so the exact-zero case is caught exactly.)

Two normalizations must be kept straight so they don't get applied twice: the divide
by `vpair` here, and a later jackknife weighting that also uses `vpair`. Together
they must reproduce ADMIXTOOLS 2's definition, not double-normalize.

Important subtlety: because a pair with no shared data returns f2 = 0 (not a special
"no data" marker), the f2 **value alone cannot tell apart "no data" from a genuine
zero**. That is why the missing-data test is done on `vpair` directly — see section
8 — and never by looking at the finalized f2 value.

---

## 8. The missing-block predicate (`pair_block_is_missing`)

```
pair_block_is_missing(double vpair) -> bool
```

The single-source rule for deciding whether a jackknife block has **no usable data**
for a population pair:

```
missing  ⇔  not (vpair > 0)     // i.e. vpair <= 0
```

steppe estimates uncertainty with a block jackknife, which splits the genome into
blocks. A block is missing for a pair (i, j) when no SNP inside it is valid in both
populations at once — that is, when `vpair` for that pair in that block is 0.

This matters because ADMIXTOOLS 2 drops any block in which *any* loaded pair has no
data; it never fills such a gap in. If steppe instead used the zero produced by
`finalize_f2` as if it were a real value, it would bias downstream statistics toward
zero and inflate the jackknife's variance estimate. So the drop decision must be
made on `vpair` itself, exactly as this predicate does, and not on the finalized f2.

Testing `vpair <= 0.0` catches the true zero count exactly and also rejects any
malformed negative value. This predicate is single-homed here so the CPU oracle and
the GPU keep-mask kernel cannot disagree about which blocks to drop.

---

## 9. Re-exported launch-configuration helpers

The header also pulls in `core/internal/launch_config.hpp` and passes its small
grid-math helpers (`cdiv`, `grid_for`) straight through. Those helpers compute how
many GPU thread blocks a launch needs and now live in their own dedicated home. They
are re-exported here only as a convenience, so that a file which includes
`f2_estimator.hpp` for the numerics above gets the launch math too without a second
include — used by the f2 feeder kernels and by the host unit test that exercises
both the numerics and the grid math.

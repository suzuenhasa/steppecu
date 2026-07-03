# `decode_compact_kernel.cu` reference

## 1. Purpose

`src/device/cuda/decode_compact_kernel.cu` holds the GPU kernels that decide which
SNPs to keep and then pack the kept ones together, all without leaving the GPU.

Reading genotypes produces, for every SNP, a set of per-population numbers (allele
frequencies and sample counts). Many of those SNPs then get filtered out — because
they sit on a sex chromosome, have too much missing data, don't vary, and so on.
The remaining ("kept") SNPs have to be squeezed into a smaller, gap-free array so
that later stages only see the survivors.

Historically that filtering happened on the CPU: a loop walked every SNP, tested a
keep condition, and pushed the survivors onto growing arrays one at a time. This
file replaces those CPU loops with GPU kernels so the whole keep-and-pack step runs
on the device, on data that is already sitting in GPU memory. The design goal is not
just speed — it is to produce a result that is **bit-for-bit identical** to what the
old CPU loop produced, so nothing downstream can tell the difference. Section 7
explains how that exactness is guaranteed.

The file exposes three small launch functions (`launch_autosome_keep_mask`,
`launch_regimeb_keep_mask`, `launch_compact_columns_gather`). Each one wraps a single
GPU kernel. The kernels themselves are private to this file; only the launch
functions are visible to the rest of the GPU layer, and they in turn are internal to
the GPU code — no other part of the program includes this file directly.

---

## 2. The keep-and-pack pipeline

The three kernels are meant to run as a short pipeline, in this order:

1. **Build a keep flag for every SNP.** One of the two "keep mask" kernels writes a
   one-byte flag per SNP — `1` to keep it, `0` to drop it. There are two different
   keep-mask kernels because there are two different filtering situations
   (Section 3 and Section 4). You pick whichever one matches the situation; you do
   not run both.

2. **Turn the flags into destination positions.** Between step 1 and step 3, the
   flag array is run through an *exclusive prefix sum* (a running total that, at each
   SNP, holds the count of kept SNPs strictly before it). That prefix sum is computed
   by a standard library routine, not by a kernel in this file. For a kept SNP, its
   prefix-sum value is exactly the slot it should land in inside the packed output.
   Because a running total only ever increases, the kept SNPs keep their original
   left-to-right order — the packed array is the survivors in file order, with the
   gaps removed.

3. **Copy the kept columns into the packed array.** The gather kernel (Section 5)
   reads the flags and the prefix-sum positions and copies each kept SNP's whole
   column of per-population values into its packed slot.

The one-byte flags are also what a companion library routine uses to compact the
simpler one-dimensional per-SNP arrays (chromosome number and genetic position) that
travel alongside the per-population values. Using the same flag array for both keeps
the two-dimensional value tensor and the one-dimensional side arrays perfectly in
step, so a SNP is either kept in all of them or dropped from all of them.

---

## 3. The autosome keep mask (`autosome_keep_mask_kernel`)

This is the simpler of the two keep-mask kernels. It keeps only SNPs that sit on an
autosome — a numbered (non-sex) chromosome. It runs one GPU thread per SNP. Each
thread reads that SNP's chromosome number and writes a `1` flag if the number falls
in the inclusive range `[chrom_min, chrom_max]`, otherwise a `0`.

The range bounds are passed in; in practice they are `1` and `22`, meaning
"chromosomes 1 through 22," which is the same set of autosomes ADMIXTOOLS 2 keeps by
default (dropping X, Y, mitochondrial, and other codes).

The comparison is done entirely in integers, because the chromosome number is read
straight from the `.snp` file as an integer. There is no floating-point rounding
anywhere in the decision, so the kept set is exactly the set the equivalent CPU test
(`if (chr < 1 || chr > 22) continue;`) would have kept — not merely close to it.

This mask corresponds to the case where the *only* filter being applied is
"autosomes only." When a fuller set of filters is in play, use the regime-B mask
(Section 4) instead.

---

## 4. The full keep mask (`regimeb_keep_mask_kernel`)

This is the richer keep-mask kernel. It reproduces the complete keep decision used
when building the f2 statistics — the same set of quality filters ADMIXTOOLS 2's
`extract_f2` applies — but on the GPU. Like the autosome kernel it runs one thread
per SNP and writes one flag per SNP.

Running exactly one thread per SNP is a deliberate choice, not just convenience.
Inside each thread the work across populations is done as a plain sequential loop
(population `0`, then `1`, and so on). Summing in that fixed, one-population-at-a-time
order is what makes the arithmetic match the CPU's arithmetic exactly; a parallel
sum across populations would add the same numbers in a different order and could land
on a slightly different floating-point total. See Section 7.

Each thread makes its decision in two parts:

### Part 1 — the pooled quality filters

The thread first combines this SNP's per-population frequencies and counts into a
single pooled summary (the pooled allele frequency and related quantities), then runs
the shared keep decision against it. This is where the bulk of the filters live:
dropping multiallelic sites, the strand-ambiguous-SNP policy, the minimum
minor-allele-frequency filter, the missing-data filter, dropping sites with no
variation (a pooled minor-allele frequency of exactly `0.0`, tested as an exact
equality), and the transition/transversion and autosome filters.

Two details matter here:

- Both the pooling step and the decision step call the **same shared code** that the
  CPU reference uses, rather than re-implementing the math. That shared pooling
  routine is written to add its terms in a fixed order (it does not let the compiler
  fuse a multiply-and-add into a single rounding step), so the GPU and CPU produce
  the identical pooled value.
- The membership test built into the shared decision is a no-op here — for this
  path it is always treated as passing — because `extract_f2` does not apply an
  include/exclude SNP list at this stage.

The per-sample missing-data threshold carried in the config is forced to `1.0` (its
"off" value) by the caller, because that per-sample filter is not the one being
applied here. The population-coverage filter is applied separately, in Part 2.

### Part 2 — the population-coverage filter

After Part 1, and only if the SNP is still a keeper, the thread applies a separate
"how many populations actually have data here" filter, matching what `extract_f2`
does. It counts how many populations have a sample count of zero or less (no usable
data) at this SNP, divides by the total number of populations to get a missing
fraction, and drops the SNP if that fraction is **strictly greater** than the
`maxmiss` threshold.

The exact boundary conditions are load-bearing and chosen to match the reference:

- A population counts as "missing" when its count is `<= 0.0` (less-than-or-equal to
  zero, not just equal to zero).
- The drop test is a strict greater-than (`fraction > maxmiss`), so a fraction that
  exactly equals `maxmiss` is kept, not dropped.
- If `maxmiss` is `1.0` or larger, this whole part is skipped and the SNP keeps
  whatever verdict Part 1 gave it. A threshold of `1.0` therefore means "no
  population-coverage filtering."

---

## 5. The column-gather kernel (`compact_columns_gather_kernel`)

This kernel does the actual packing. The per-population values live in a single large
tensor laid out as `[P populations × M SNPs]` in **column-major** order — that is,
all `P` values for one SNP sit next to each other in memory, forming that SNP's
"column," and consecutive SNPs' columns follow one another.

The kernel runs one thread per (population, source-SNP) pair. A thread does nothing
if its SNP's flag is `0`. If the flag is `1`, it copies that one population's value
from the source column at SNP `s` to the destination column at the SNP's packed
position (the prefix-sum value from Section 2). Every thread that handles the same
kept SNP copies a different population's entry, so together they move the whole
column. The copy is a plain value copy — no arithmetic — so the packed values are the
exact same bit patterns as the originals.

The source and destination tensors must be **different** buffers; the kernel is not
safe if they overlap.

### The deliberate axis assignment (coalescing)

There is one subtlety in this kernel that is easy to misread as a bug, and the code
calls it out explicitly. The population index rides the fastest-moving thread axis
(`threadIdx.x`) while the SNP index rides the slower one (`threadIdx.y`). At first
glance that looks like the axes are "crossed."

It is intentional and correct. Because the tensor is column-major, the population
axis is the unit-stride axis — neighboring population entries are neighboring
addresses in memory. Putting the population index on the fastest thread axis means
neighboring threads touch neighboring memory addresses, which lets the hardware
combine their reads and writes into few, wide memory transactions (a *coalesced*
access) instead of many scattered ones. This mapping is deliberately kept consistent
with the way the launch geometry is set up (Section 6) and with the matching feeder
kernel elsewhere in the GPU code, so the same thread-to-element rule is used
everywhere. It is **not** an out-of-bounds mistake to "fix" by swapping the axes.

---

## 6. Launch geometry and the grid-size guard

Each launch function sets up the grid of GPU threads and starts its kernel. A few
conventions are shared across all three.

- **Early out on empty work.** If there are no SNPs (`M <= 0`) — or, for the gather,
  no populations either (`P <= 0`) — the launch function returns immediately without
  starting a kernel.

- **Block size for the two mask kernels.** Both keep-mask kernels use a
  one-dimensional block of `kKeepBlock = 256` threads and lay a one-dimensional grid
  of those blocks over the SNP axis (enough blocks to cover all `M` SNPs).

- **Block shape for the gather.** The gather uses a two-dimensional block whose
  dimensions come from the shared decode block constants (`32` wide on the population
  axis by `8` on the SNP axis, which is again `256` threads total). It deliberately
  reuses those shared constants rather than typing fresh numbers, so the decode tile
  shape is defined in exactly one place and stays consistent across the files that
  use it. The `32`-wide population axis is what makes the accesses coalesced
  (Section 5).

- **The grid-size guard.** GPUs cap how many blocks a grid may have along its first
  dimension. Every launch here computes how many blocks the SNP axis needs and
  asserts that the count stays within that hardware limit; the assertion message says
  to "tile the SNP axis" if it is ever exceeded. In other words, a dataset with an
  enormous number of SNPs would need to be processed in slices rather than in one
  launch. This guard makes that requirement fail loudly instead of silently launching
  a malformed grid.

---

## 7. Why the GPU result matches the host exactly

The whole point of this file is that turning the CPU keep-and-pack loop into GPU
kernels changes **only where the work runs**, never the answer. Several independent
guarantees combine to make that true:

- **The keep set is the same.** The autosome test (Section 3) is done in integers, so
  there is no rounding that could flip a decision. The full filter (Section 4) reuses
  the same shared decision code as the CPU reference, adds its per-population terms in
  the same fixed order (with fused multiply-add disabled in the shared reduction),
  and applies the population-coverage filter with the same exact boundary tests
  (`<= 0.0`, strict `>`, and the `maxmiss >= 1.0` skip). A SNP is kept on the GPU if
  and only if it was kept on the CPU.

- **The packed order is the same.** The packed positions come from an exclusive
  prefix sum over the flags, which only ever increases along the SNP axis. So the
  survivors keep their original file order with the gaps removed — the same order the
  CPU produced by appending survivors one at a time.

- **The packed values are the same.** The gather copies values byte-for-byte; it does
  no arithmetic, so a kept value is unchanged.

Because the kept set, the order, and the values all match, everything computed
afterward from the packed arrays — including how SNPs are assigned to jackknife
blocks from the compacted chromosome and genetic-position arrays — comes out
identical to the old CPU path.

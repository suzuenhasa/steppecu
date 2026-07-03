# `ratio_block_jackknife_kernel.cu` reference

## 1. Purpose

`src/device/cuda/ratio_block_jackknife_kernel.cu` holds one GPU kernel that
computes a *ratio block jackknife*: given, for each item, a per-block numerator, a
per-block denominator, and a per-block weight, it produces a point estimate, a
standard error, a z-score, and (optionally) a p-value.

The same kernel serves two different statistics that used to have their own
host-side loops:

- the **f4-ratio** statistic (a ratio of two f4 statistics), and
- the **D-statistic** (the qpDstat / ABBA-BABA test).

Both statistics were originally computed one item at a time on the CPU using
`long double` math (an f4-ratio was jackknifed per tuple; a D-statistic per
quadruple). This kernel moves those loops onto the GPU and runs them for all items
at once. Each item is handled independently by one GPU thread, which walks the list
of genome blocks and reduces them down to the four output numbers. Because the
per-block inputs already live in GPU memory, doing the jackknife on the GPU also
removes two large copies back to the host that the old design required (the
per-item block replicates for the f4-ratio path, and the numerator/denominator/count
sums for the D-statistic path).

The kernel body and its launch (`<<<...>>>`) live only in this file. The rest of
the library never includes this source; it reaches the kernel through a small
launch wrapper declared in the paired `.cuh` header. That keeps all CUDA code
private to the device layer.

---

## 2. Precision: native double precision, and why not the emulated kind

This kernel runs in **native double precision (FP64)**, not the faster emulated
double precision that steppe uses for its heavy matrix multiplications.

The reason is cancellation. A jackknife works by taking differences between a
full-data estimate and each leave-one-out estimate; those differences are small
numbers formed by subtracting two nearby large numbers, which is exactly the case
where emulated precision loses accuracy. So this kernel is one of the deliberate
"native FP64" carve-outs: the numerically delicate reductions run in true double
precision even though the bulk matrix work elsewhere does not.

To reproduce the original CPU results as closely as possible, the kernel adds up
its per-block terms in **ascending block order** — the exact same operand order the
old `long double` CPU code used. The only remaining difference from the CPU
reference is the width of the running accumulator: the CPU carried a 64-bit
`long double` mantissa, and the GPU carries a 53-bit double mantissa. Everything
else — the survivor test, the formulas, the summation order — is identical.

**Frozen:** this path must stay native FP64. Do not switch it to the emulated
double-precision mode. The kernel is checked against reference ("golden") values
for both statistics at a relative tolerance of `1e-6`, and the CPU and GPU backends
are required to agree.

This is the same family of leave-one-out block jackknife that already runs on the
GPU in native FP64 elsewhere in steppe (the qpfstats numerator jackknife and the
qpAdm fit's f4 leave-one-out row), so the approach is consistent across the codebase.

---

## 3. The two modes: f4-ratio and D-statistic

The kernel is one kernel, not two. The f4-ratio jackknife and the D-statistic
jackknife are the *same* ratio block jackknife; they differ only along three axes,
and those differences are handled as in-thread branches rather than as separate
kernels:

1. **The weight.** What counts as a block's weight is caller-supplied. For the
   f4-ratio it is the per-block SNP count (block size). For the D-statistic it is a
   per-block observation count.
2. **The survivor mask.** Which blocks are dropped as "missing" differs (see below).
3. **The "tot mode."** How the full-data centering ratio is formed differs (see the
   algorithm section).

The `tot_mode` argument selects between them:

| `tot_mode` | Statistic | What the inputs mean | Survivor test |
|---|---|---|---|
| `0` | f4-ratio | `num` / `den` are already the per-block leave-one-out replicates; `xblk_num` / `xblk_den` are the per-block raw sums used to form the centering ratio; `weight` is the per-block block size. | A block survives if `\|xblk_den\| >= setmiss_thresh`. A block whose denominator is essentially zero is treated as missing. This matches ADMIXTOOLS 2's "setmiss" rule. |
| `1` | D-statistic | `num` / `den` are the per-block sums; `weight` is the per-block count. The leave-one-out replicates are rebuilt inside the thread. The `xblk_num` / `xblk_den` arrays are unused (passed as absent). | A block survives if its count `weight > 0`. |

The survivor test is not computed once and stored; it is re-evaluated inline on
every pass over the blocks, so each pass sees exactly the same set of surviving
blocks in the same order. This mirrors how the CPU reference re-applied its mask on
each pass.

---

## 4. The input descriptor: `DRatioJackArray`

Each of the five input arrays (`num`, `den`, `weight`, `xblk_num`, `xblk_den`) is
passed to the kernel as a small `DRatioJackArray` descriptor rather than a bare
pointer. The descriptor tells the kernel how to find element (item `k`, block `b`)
inside one flat device array, using the formula:

```
data[base + k * item_stride + b * block_stride]
```

| Field | Type | Meaning |
|---|---|---|
| `data` | `const double*` | Device pointer to the flat array. A **null** `data` means the array is absent — this is how the D-statistic path signals that the `xblk_num` / `xblk_den` pair is not supplied. |
| `base` | `long` | The starting offset into `data`. |
| `item_stride` | `long` | How far apart consecutive items are. A value of **0 broadcasts** the same data across all items — used for the f4-ratio's per-block block-size weight, which is the same for every item. |
| `block_stride` | `long` | How far apart consecutive blocks are. |

All pointers are device pointers. Passing an indexing descriptor instead of a raw
pointer lets one array be laid out in memory however the caller finds convenient
(item-major, block-major, or broadcast) without the kernel needing to know or care.

---

## 5. Kernel shape and launch geometry

The kernel assigns **one thread per item**. Each thread owns a short reduction over
the block axis (there are only about as many blocks as there are jackknife blocks
along the genome, which is small), and it keeps its running totals in registers. If
there are more items than threads, threads walk the item axis in a **grid-stride
loop**, so the launch works for any number of items regardless of how many blocks
were launched.

This shape deliberately matches the qpfstats numerator jackknife kernel exactly.

Launch parameters:

| Constant | Value | What it's for |
|---|---|---|
| `kThreads` | `128` | Threads per block. The number of blocks launched is the item count divided by 128, rounded up. |
| grid-dimension cap | `core::kMaxGridX` | The number of launched blocks is clamped to this single, shared maximum grid dimension. If the item count would need more blocks than the cap allows, the grid-stride loop covers the remainder. The cap is not chosen here — it comes from the one place in the codebase that defines the grid-dimension limit. |

The launch wrapper returns immediately and does nothing if there are no items or no
blocks (`N <= 0` or `n_block <= 0`). After launching, it performs the standard
kernel error check.

---

## 6. The jackknife algorithm, pass by pass

Both modes make several passes over the surviving blocks. The passes exist so that
values computed in an earlier pass (the full-data ratio, the point estimate) are
available when a later pass needs them. All sums are taken in ascending block order.

If a thread ends up with **one or zero** surviving blocks, or a zero total weight,
it cannot form a jackknife and writes `NaN` for all outputs for that item.

### f4-ratio (`tot_mode == 0`)

1. **Survivor pass.** Sum the surviving block weights into `n` (the total weight),
   count the surviving blocks `nb`, and accumulate the weighted raw numerator and
   denominator (`totnum = Σ xblk_num·weight`, `totden = Σ xblk_den·weight`). The
   full-data centering ratio is `tot = totnum / totden`.
2. **Estimate pass.** For each surviving block, form its leave-one-out ratio
   `R = num / den` (these inputs already are the replicates). Accumulate
   `Σ (tot − R)` and the weighted mean of `R`. The point estimate is
   `est = Σ(tot − R) + weighted_mean(R, weight)`.
3. **Variance pass.** For each surviving block, let `h = n / weight`, form the
   pseudo-value `tau = h·tot − (h − 1)·R`, and accumulate
   `((tau − est) / sqrt(h − 1))²`. The variance is that sum divided by the number
   of surviving blocks; the standard error is its square root.

### D-statistic (`tot_mode == 1`)

Here the leave-one-out replicates are not supplied, so the thread rebuilds them:

1. **Survivor pass.** Sum the surviving counts into `sum_cnt` and count the
   surviving blocks.
2. **Full-data ratio pass.** For each surviving block, form its per-block means
   `est_num = num / count` and `est_den = den / count`, and accumulate their
   count-weighted means into `tot_num` and `tot_den`.
3. **Leave-one-out pass.** For each block, with `rel = count / sum_cnt`, remove that
   block's contribution to get the leave-one-out numerator and denominator
   `loo_num = (tot_num − est_num·rel) / (1 − rel)` (and the same for the
   denominator), then the replicate `R = loo_num / loo_den`. The centering ratio
   `tot` is the `(1 − rel)`-weighted mean of these replicates.
4. **Estimate pass.** Recompute each `R` the same way (recomputing rather than
   storing gives the bit-identical double) and accumulate `Σ (tot − R)`. The point
   estimate is `est = Σ(tot − R) + weighted_mean(R, count)`.
5. **Variance pass.** With `h = sum_cnt / count`, form the pseudo-value
   `tau = h·tot − (h − 1)·R − est` and accumulate `tau² / (h − 1)`. The variance is
   that sum divided by the number of surviving blocks; the standard error is its
   square root.

The two variance passes are the same jackknife variance written two ways: the
f4-ratio subtracts `est` before dividing by `sqrt(h − 1)` and squaring, while the
D-statistic folds `est` into `tau` and divides the square by `(h − 1)`. Both equal
`(tau − est)² / (h − 1)`; each is written to match its own CPU reference so the
GPU output tracks the CPU output term for term.

In both modes the z-score is `z = est / se`. If the variance is not strictly
positive, the standard error and z-score are `NaN`.

---

## 7. Outputs and the p-value

The kernel writes four output arrays, each of length equal to the number of items:
`d_est` (estimate), `d_se` (standard error), `d_z` (z-score), and, when requested,
`d_p` (p-value).

The p-value is computed only when the `compute_p` flag is set; otherwise the `d_p`
array is left untouched (and may be null). When it is computed, it is the two-sided
normal-tail p-value:

```
p = erfc(|z| / sqrt(2))
```

The constant `kInvSqrt2 = 0.7071067811865475244` is `1 / sqrt(2)`, written out to
full precision. This p-value formula matches ADMIXTOOLS 2's two-sided tail
conversion used for both statistics.

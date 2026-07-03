# `ratio_block_jackknife_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/ratio_block_jackknife_kernel.cuh` is the small, public-facing
declaration ("launch seam") for one GPU kernel that two different statistics share:
the **f4-ratio** and the **D-statistic** (qpDstat). Both statistics estimate a value
and its uncertainty using the same technique — a **block jackknife**, where the SNPs
are split into blocks along the genome, each block is left out in turn, and the spread
of those leave-one-out answers becomes the standard error.

Historically each statistic ran that jackknife in a per-item loop on the CPU. This
header exposes the single GPU kernel that replaces both of those loops. The actual
kernel body lives in the paired `.cu` file; this header only declares the two things a
caller needs:

1. `DRatioJackArray` — a tiny descriptor that tells the kernel how to find one input
   array's `(item, block)` elements in GPU memory.
2. `launch_ratio_block_jackknife` — the one function that configures and launches the
   kernel.

The rest of the code base reaches the kernel **only** through this wrapper. A caller
never launches the kernel directly and never includes the kernel body.

The kernel is intentionally embarrassingly parallel across items: it runs **one thread
per output item**, and each thread does a short reduction over the block axis entirely
in registers. Because the block count is small, that per-thread loop is cheap, and
thousands of items can be computed at once.

---

## 2. Why this header is private to the GPU library

This header mentions a CUDA type, `cudaStream_t`, in the launch function's signature.
Any header that names a CUDA type is considered internal to steppe's GPU layer and is
not part of the library's public surface. The CUDA-free code (the core library, the
public API) never sees this file; it talks to the GPU layer through a CUDA-free
descriptor (`RatioJackArray` in `backend.hpp`) instead. `DRatioJackArray` here is the
device-side twin of that CUDA-free descriptor, holding the same four fields.

---

## 3. The input descriptor: `DRatioJackArray`

The kernel takes several per-`(item, block)` input arrays. Rather than assume every
input is laid out the same way, each one is described by a `DRatioJackArray`, a plain
struct of four fields. This lets two callers with completely different memory layouts
feed the same kernel with **no repacking**.

| Field | Type | Meaning |
|---|---|---|
| `data` | `const double*` | Base pointer to the array. This is always a **device** pointer. A **null** pointer means the array is absent (see below). |
| `base` | `long` | The element offset of item 0, block 0 — where the logical `(0, 0)` element actually starts inside `data`. |
| `item_stride` | `long` | How many elements to step to move from one item to the next. A value of **0 broadcasts** the array across all items (see below). |
| `block_stride` | `long` | How many elements to step to move from one block to the next. |

The kernel reads the element for item `k`, block `b` at:

```
data[base + k*item_stride + b*block_stride]
```

Two conventions carry real meaning:

- **Null `data` means "this array is not supplied."** The f4-ratio path uses every
  array; the D-statistic path leaves the raw-block pair (`xblk_num` / `xblk_den`) null
  because it does not need them. The mode flag (see section 5) tells the kernel which
  arrays it should actually read, so a null array is only ever read in the mode that
  supplies it.
- **A `0` `item_stride` broadcasts one array to every item.** The f4-ratio path uses
  this for its per-block block-size weights: the same block-size vector applies to
  every item, so instead of duplicating it per item, one copy is read by all items via
  a zero item stride.

Because the descriptor separates base, item stride, and block stride, an array that is
stored item-major, block-major, or as a shared broadcast vector all look the same to
the kernel — only the three numbers differ.

---

## 4. The launch function: parameters and outputs

```cpp
void launch_ratio_block_jackknife(
    const DRatioJackArray& num, const DRatioJackArray& den,
    const DRatioJackArray& weight, const DRatioJackArray& xblk_num,
    const DRatioJackArray& xblk_den, int N, int n_block,
    int tot_mode, double setmiss_thresh, bool compute_p,
    double* d_est, double* d_se, double* d_z, double* d_p,
    cudaStream_t stream);
```

### Inputs

| Parameter | Meaning |
|---|---|
| `num`, `den` | The per-`(item, block)` numerator and denominator arrays. What they hold depends on the mode (section 5): for f4-ratio they are the leave-one-out replicate parts; for the D-statistic they are the per-block sums. |
| `weight` | The per-`(item, block)` jackknife weight. For f4-ratio this is the block-size weight (usually broadcast); for the D-statistic it is the per-block observation count. |
| `xblk_num`, `xblk_den` | The **raw** (not-yet-left-out) per-block numerator and denominator. Used only by the f4-ratio mode; left null for the D-statistic. `xblk_den` is also the source the f4-ratio missing-block test reads. |
| `N` | The item count — the number of output rows. |
| `n_block` | The block count each item reduces over. |
| `tot_mode` | Selects the statistic: `0` = f4-ratio, `1` = D-statistic (section 5). |
| `setmiss_thresh` | Controls the missing-block rule (section 6). |
| `compute_p` | Whether to also compute a p-value (section 4, outputs). |
| `stream` | The CUDA stream to launch on. |

### Outputs

Four device output buffers, each of length `N` (one slot per item):

| Buffer | Holds |
|---|---|
| `d_est` | The jackknife point estimate. |
| `d_se` | The standard error (square root of the jackknife variance). |
| `d_z` | The z-score, `est / se`. |
| `d_p` | The two-sided p-value — written **only** when `compute_p` is true; otherwise this buffer is left untouched. |

When `compute_p` is true, each item's p-value is `erfc(|z| / sqrt(2))`, the same
two-sided tail formula ADMIXTOOLS 2 uses. The f4-ratio path passes `compute_p` false
(it reports estimate, standard error, and z only); the D-statistic path passes it true.

If an item cannot produce a valid answer — fewer than two surviving blocks, a
non-positive total weight, or a zero total denominator — all of its outputs are set to
NaN rather than a garbage number.

### Launch shape

The function launches one thread per item with a fixed block of 128 threads, and a
grid sized to cover `N` items, capped at the maximum allowed grid dimension so that
very large item counts fall back to a grid-stride loop instead of exceeding the limit.
It returns immediately (does nothing) if `N` or `n_block` is not positive.

---

## 5. The two modes: f4-ratio versus D-statistic

The two statistics are the **same** ratio block jackknife; they differ only in how the
"total" (the full-data answer that the leave-one-out replicates are compared against)
is formed and in what the `num` / `den` arrays contain. The `tot_mode` flag switches
between them inside the kernel, as an in-thread branch — there are **not** two separate
kernels.

| | `tot_mode = 0` (f4-ratio) | `tot_mode = 1` (D-statistic) |
|---|---|---|
| Total | A block-sum ratio: sum the weighted raw block numerators and denominators, then divide. | A leave-one-out ratio built inside the kernel from the per-block sums. |
| What `num` / `den` are | Already the leave-one-out replicate parts. | The per-block sums; the leave-one-out replicate is reconstructed in the kernel. |
| Raw-block arrays | Reads `xblk_num` / `xblk_den`. | Does not use them (null). |
| Weight | Block-size weight (typically broadcast across items). | Per-block observation count. |
| p-value | Not computed. | Computed. |

Keeping both statistics on one code path means the two can never silently drift apart
in how they weight blocks, order operands, or handle missing blocks.

---

## 6. Which blocks are kept: the survivor rule

Before combining blocks, each item decides which of its `n_block` blocks are usable
("survivors"). The rule is chosen by `setmiss_thresh`, and it must match the CPU
reference exactly:

- **`setmiss_thresh > 0` (f4-ratio):** a block is dropped when the magnitude of its raw
  denominator is below the threshold — that is, `|xblk_den(k, b)| < setmiss_thresh`.
  This reproduces ADMIXTOOLS 2's "set missing" behavior, where a block whose
  denominator is effectively zero is treated as absent.
- **`setmiss_thresh <= 0` (D-statistic):** a block is dropped when its weight is not
  positive — that is, `weight <= 0` (a block with no observations).

The same survivor test is recomputed on every pass over the blocks rather than cached,
so that every pass sees an identical mask. An item needs **at least two** surviving
blocks (and a positive total weight, and a nonzero total denominator) to produce a
result; otherwise its outputs are NaN.

---

## 7. How the estimate, standard error, and z-score are formed

Both modes, once their survivors and total are settled, build the answer the same way
— a weighted block jackknife.

1. **Per-block replicate.** For each surviving block `b`, the kernel forms a
   leave-one-out replicate `R_b` (the estimate computed with that block removed).
2. **Point estimate.** The reported estimate combines two pieces: the sum of how much
   each replicate deviates from the total, plus a weight-weighted average of the
   replicates. This is a bias-corrected weighted jackknife estimate, not just the
   plain full-data total. The reported number is deliberately this jackknife estimate,
   **not** the raw total: on the f4-ratio golden it matches ADMIXTOOLS 2's answer to
   about 2e-15, whereas the raw total is off by about 3e-4.
3. **Standard error.** For each surviving block the kernel forms a pseudo-value from
   the total and the replicate, scaled by that block's relative weight, subtracts the
   estimate, and accumulates the square. Averaging those squares gives the jackknife
   variance; its square root is the standard error. (This is the standard weighted
   delete-block jackknife variance that ADMIXTOOLS uses, where blocks of unequal size
   are weighted by their relative size.)
4. **z-score.** Simply `est / se`. A non-positive variance yields a NaN standard error
   and z.

Every one of these accumulations walks the blocks in **ascending block index order**,
because floating-point addition is not associative and the CPU reference sums them in
that order. Reproducing the order is what lets the GPU result match the reference to
the last bits.

---

## 8. Precision: native double precision, and why

Most of steppe's heavy math runs in an emulated, faster form of double precision. This
kernel is a deliberate exception: it runs in **native double precision** and must never
use the emulated form.

The reason is cancellation. The jackknife subtracts nearly-equal quantities (a
replicate from the total, a pseudo-value from the estimate). Those subtractions lose
precision catastrophically if the inputs are even slightly wrong, so this path keeps
the honest hardware double and reproduces the operand order of the CPU reference
exactly. The CPU reference itself uses extended (long double) precision; the **only**
intended difference between the two is the width of the running accumulator (long
double's wider mantissa versus double's), and that difference is small enough that the
results agree at the validation tolerance.

The precision setting a caller may pass through the higher-level API is honored only as
a label on the result — it does not change the arithmetic here. This kernel is always
native double precision.

# `cuda_backend_dstat.cu` reference

## 1. Purpose

`src/device/cuda/cuda_backend_dstat.cu` is one of the GPU-backend translation
units. It holds the member functions of the CUDA backend that compute the
D-statistic (qpDstat) and the f4-ratio, both broken down block-by-block along the
genome and both turned into a standard-error estimate by a block jackknife.

It was split out of the larger `cuda_backend.cu` file purely to keep that file a
manageable size; the function bodies were moved across unchanged. Nothing about
the math, the numeric precision, or the order operations run in changed in the
split.

There are three groups of functions here:

1. **The D-statistic block reduction** — one private worker that sums a numerator,
   a denominator, and a count for each population quadruple within each genome
   block, plus two public entry points that feed it data from either host memory or
   already-on-GPU memory.
2. **Two device-resident block-jackknife functions** — one for the f4-ratio and
   one for the D-statistic. Both read their inputs straight from GPU memory and run
   the same shared jackknife kernel, so no large per-block intermediate array is
   ever copied back to the host.
3. **Two "oracle door" stub functions** — the host-memory versions of the two
   jackknife functions, which deliberately throw. They exist only so the backend
   satisfies the shared interface; the real GPU work always goes through the
   device-resident versions.

Every function first calls `guard_device()` to make sure the correct GPU is the
active one, and every function returns early with empty or not-a-number results
when handed a degenerate problem (no populations, no SNPs, no blocks, or an empty
decode), rather than launching a kernel on nothing.

---

## 2. Block layout: turning block ids into contiguous SNP ranges

Both the D-statistic reduction and the D-statistic jackknife need to know, for each
genome block, which run of SNP columns belongs to it. The caller supplies a
`block_id` array — one entry per SNP saying which block that SNP falls in — and the
SNPs are already ordered so that each block's SNPs are contiguous.

`core::block_ranges` is the single shared helper that converts that per-SNP
`block_id` array into a `[begin, size)` range per block. It is the exact inverse of
the routine that originally assigned SNPs to blocks, and it is the same primitive
the f2 computation path uses, so all paths agree on block boundaries. The code then
splits those ranges into two plain integer arrays — a `begin` (first SNP column of
each block) and a `size` (SNP count of each block) — copies both up to the GPU, and
hands them to the kernel.

The SNP count `M` is carried as a `long` but the per-block begin and size are
stored as `int`. That is safe because an upstream guard already guarantees the total
SNP count fits in an `int`.

---

## 3. The D-statistic block reduction

The private worker `dstat_block_reduce_device` computes, for every genome block and
every population quadruple, three running sums:

- **numerator** `(a − b)·(c − d)`
- **denominator** `(a + b − 2ab)·(c + d − 2cd)`
- **count** the number of SNPs that contributed

where `a, b, c, d` are the four populations' allele frequencies at a SNP. A SNP only
contributes to a block/quadruple cell if all four populations have a valid
(non-missing) frequency there. This is the normalized-D form[^at2],
and all three sums are accumulated in native double precision because the numerator
and denominator each subtract nearly equal quantities and are sensitive to
cancellation.

The frequency matrix `Q` and the validity matrix `V` are stored column-major with
shape population-by-SNP (element for population `i`, SNP `s` sits at `i + P·s`). The
quadruples arrive as a flat array of four population indices each. The outputs are
laid out **row-major** with shape quadruple-by-block: the cell for quadruple `k`,
block `b` lives at `k·n_block + b`. The worker copies the quadruples, block begins,
and block sizes to the GPU, launches the reduction kernel, and copies the three
output sums back to host buffers the caller provided.

### The two public entry points

Two public overloads of `dstat_block_reduce` feed this worker:

| Overload | Where `Q`/`V` come from | What it does |
|---|---|---|
| Host-pointer | Plain host arrays | Allocates GPU buffers, copies `Q` and `V` up, then calls the worker. This is the entry used by callers whose data is not already on the GPU, and the one the CPU reference backend mirrors for parity. |
| `DeviceDecodeResult` | Already-resident GPU buffers | Reads the decode's on-GPU frequency and validity pointers directly and calls the worker, **skipping the host-to-device copy entirely**. This is the fast path — the whole point of keeping the decoded data resident is to avoid re-uploading it. |

Both overloads return immediately, doing nothing, when the problem is empty
(no populations, no SNPs, no quadruples, or no blocks; and for the resident overload,
when the decode is empty or its GPU pointer is null).

---

## 4. The shared device-resident block jackknife

A block jackknife estimates a statistic and its uncertainty by, in effect, recomputing
the statistic once per genome block with that block left out, and measuring how much
the answer moves. Both the f4-ratio and the D-statistic use the **same** jackknife
kernel, reached through `launch_ratio_block_jackknife`.

### The no-copy design

The defining choice in this file is that the jackknife runs over data that is already
on the GPU and never copies the large per-block intermediates back to the host. For
the f4-ratio that intermediate would be a `[2N × survivor-blocks]` matrix; for the
D-statistic it would be an `[N × blocks]` matrix. Both stay resident. Only the three
(or four) small final result vectors — estimate, standard error, z-score, and
optionally p-value, each of length `N` — are copied back. This is what lets these
statistics scale to large numbers of quadruples without the per-block matrices
dominating host memory or the copy time.

### The `DRatioJackArray` stride descriptor

Because the two paths lay their data out differently but share one kernel, each input
is passed as a small descriptor, `DRatioJackArray`, that tells the kernel how to
find element (item `k`, block `b`):

```
data[base + k·item_stride + b·block_stride]
```

Two special conventions make one descriptor cover every case:

- A **null `data`** pointer means the array is absent. The D-statistic path uses this
  for the per-block-value pair it does not need.
- A **zero `item_stride`** broadcasts one value across all items. The f4-ratio path
  uses this for its per-block weight, which is the same for every quadruple.

Setting the strides correctly is how each path maps its own memory layout onto the
shared kernel; the two mappings are described in sections 5 and 6.

### The two modes

The kernel takes a `tot_mode` flag that selects which statistic's jackknife formula to
use. Mode 0 is the f4-ratio; mode 1 is the D-statistic. Both reproduce the
corresponding jackknife exactly[^at2] — the same block drop rule, the same operand order,
and the same variance form — and both run in native double precision, matching the
CPU long-double reference at the tested tolerance.

---

## 5. `f4ratio_blocks_jackknife` — the f4-ratio path (mode 0)

This function computes the f4-ratio and its jackknife from a resident f2 tensor
(`DeviceF2Blocks`), which holds per-block f2 values entirely on the GPU. The steps:

1. **Drop missing blocks.** `device_survivor_blocks` returns the ascending list of
   blocks that are usable — dropping any block that is partially missing, matching
   the behavior of removing blocks with missing data[^at2]. If that leaves no
   surviving block, the function returns all-not-a-number results. The surviving
   block sizes are gathered into both an integer and a double copy (the double copy is
   the per-block weight the kernel broadcasts).
2. **Gather the four-slab f4 per block.** For each quadruple and each surviving
   block, the per-block f4 value is assembled from four f2 slabs by the identity
   `0.5·(f2(p2,p3) + f2(p1,p4) − f2(p1,p3) − f2(p2,p4))`, reading the resident f2
   directly. The numerator and denominator quadruples are interleaved in one flat
   index array, so the item axis has `2N` rows: the first `N` are numerators and the
   next `N` are denominators. This produces the per-block matrix `dX` and, via a
   second kernel, the leave-one-out replicates `dLoo` and the totals.
3. **Describe the layout and run the jackknife.** `dX` and `dLoo` are **column-major
   in the item axis**: element (row, block) sits at `row + (2N)·block`, so
   `item_stride = 1` and `block_stride = 2N`. The numerator descriptor starts at base
   `0` and the denominator descriptor at base `N` (pointing at the second half of the
   rows). The weight is the per-block size, broadcast across items. The kernel runs in
   mode 0, with `compute_p` off, and the per-block denominator threshold
   (`setmiss_thresh`) passed through so blocks with a near-zero denominator are
   dropped, again matching the reference.

The `precision` argument is accepted but deliberately ignored: the ratio difference
is cancellation-sensitive, so this path always uses native double precision rather
than the faster emulated double precision used elsewhere.

---

## 6. `dstat_blocks_jackknife` — the qpDstat path (mode 1)

This function computes the D-statistic and its jackknife from a resident decode
(`DeviceDecodeResult`). It reuses the block reduction from section 3 and then runs the
shared jackknife:

1. **Reduce per block.** Build the per-block SNP ranges (section 2), copy the
   quadruples and ranges up, and launch the same reduction kernel to produce the
   resident per-block numerator, denominator, and count arrays.
2. **Describe the layout and run the jackknife.** These arrays are **row-major** with
   shape quadruple-by-block: element (quadruple `k`, block `b`) sits at `k·n_block + b`,
   so `item_stride = n_block` and `block_stride = 1`. The numerator and denominator
   descriptors point at those arrays; the weight descriptor points at the count array.
   The per-block-value pair is passed as null descriptors, because mode 1 never reads
   it — the leave-one-out replicate is built inside the kernel. The kernel runs in
   mode 1 with `compute_p` on, so it also writes a two-sided p-value from the z-score.
   Blocks with a zero count are dropped.

Because the numerator and denominator can nearly cancel, the reduction accumulates in
native double precision, and the jackknife runs in native double precision as well.

---

## 7. Empty and degenerate inputs

Every entry point guards against inputs that would make a kernel launch meaningless,
and each guard returns a well-formed result rather than crashing or launching on
empty data:

- The block-reduction overloads return immediately (leaving the caller's output
  buffers untouched) when there are no populations, SNPs, quadruples, or blocks.
- The two jackknife functions fill their estimate, standard-error, and z-score (and,
  for the D-statistic, p-value) vectors with not-a-number and report success when the
  problem is empty — no quadruples, no blocks, a null resident pointer, or, for the
  f4-ratio, no surviving blocks after the missing-block drop. The vector lengths are
  clamped to at least zero so a negative `N` cannot produce a bad allocation size.

A not-a-number-filled result with a success status is the deliberate signal that
there was simply nothing to estimate, distinct from an error.

---

## 8. The host-overload "oracle door" throws

The shared backend interface declares host-memory versions of both jackknife
functions — one taking a host-side f2 tensor, one taking host `Q`/`V` pointers. The
GPU backend implements these only to satisfy the interface; both throw a
`runtime_error` explaining that the GPU path reads device-resident data and that the
caller must use the resident overload instead.

These host overloads are the "oracle door": in tests they route to the CPU reference
backend, which is the parity oracle. On the GPU backend they are never a valid runtime
path, so throwing (rather than silently doing a slow host round-trip) is the intended
behavior.

---

## 9. Precision policy

The heavy matrix-multiply stages elsewhere in steppe default to a fast emulated form
of double precision. The statistics in this file deliberately do **not**. The
D-statistic numerator and denominator, the f4-ratio's difference of f2 slabs, and both
jackknife variance forms all subtract nearly equal quantities and are therefore run in
**native** double precision. This is why `f4ratio_blocks_jackknife` accepts a precision
argument only to ignore it, and why the reduction kernel accumulates in double. Native
double precision here reproduces the CPU long-double reference at the tested tolerance,
which the emulated form would not reliably do for these cancellation-prone quantities.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

# `qpfstats_jackknife_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/qpfstats_jackknife_kernel.cuh` is the small, host-callable entry
point for the two GPU kernels that finish the qpfstats block-jackknife on the
device. It declares exactly two functions — `launch_qpfstats_numer_jackknife` and
`launch_qpfstats_recenter_shift` — and nothing else.

These two kernels exist to move work that used to run on a single CPU core onto
the GPU. The jackknife estimate for qpfstats is a per-item reduction over genome
blocks, repeated for every population combination and every population pair. In a
production run that is roughly 305,000 combinations, each reducing over about 711
blocks — on the order of 217 million iterations that previously ran one at a time
in host code while the GPU sat idle. The two kernels here do that same arithmetic
in parallel, reading their inputs straight from the block statistics that already
live in GPU memory, so nothing large has to be copied back to the host first.

This file contains no kernel code. The kernel bodies and their `<<<...>>>` launch
syntax live only in the matching `.cu` file. Host orchestration code calls the two
wrappers declared here and never sees the raw launch.

The declarations name a CUDA type (`cudaStream_t`) and the file includes
`<cuda_runtime.h>`, so this header is internal to the device library — the private
seam between the backend and these kernels, not part of steppe's CUDA-free public
API. Section 8 explains that boundary.

Both kernels are written to reproduce a host reference exactly. That reference —
the `long double` oracle both the CPU backend and this GPU path are checked
against — is `matrix_jackknife_est_col` and `f2blocks_pair_est` in
`src/core/internal/qpfstats_jackknife.hpp`.

---

## 2. What the two kernels compute

Each kernel reproduces a specific reference routine[^at2], and each has a host
counterpart it is diffed against.

| Launch function | Reproduces | Host reference | Produces |
|---|---|---|---|
| `launch_qpfstats_numer_jackknife` | `matrix_jackknife_est_full`, one column | `matrix_jackknife_est_col` | per-block means and the global per-combination jackknife estimate |
| `launch_qpfstats_recenter_shift` | `f2(array)$est`, one pair series | `f2blocks_pair_est` | the per-pair recentering shift |

### The numerator / global-estimate kernel

`launch_qpfstats_numer_jackknife` does three things for every population
combination `c`, all in one pass:

1. **Materializes the per-block mean.** For each block it divides the running
   numerator sum by the count, `numsum / cnt`, and writes that value out twice:
   once into `d_numer` (row-major) and once into `d_ymat` (column-major — the same
   number, transposed, ready for the downstream right-hand-side that consumes it).
   Where a block has no data (`cnt <= 0`) the mean is `NaN`.
2. **Forms the whole-combination weighted total** over the blocks that are finite
   and have positive count.
3. **Computes the global block-jackknife estimate** `d_y[c]` for that combination
   using the leave-one-out formula (each block dropped in turn, the results
   recombined). This per-combination value is what the downstream fit is centered
   on.

### The recenter-shift kernel

`launch_qpfstats_recenter_shift` computes, for every population pair `p`, a single
recentering constant:

```
shift[p] = bglob[p] − est(p)
```

where `est(p)` is the pair's own jackknife estimate over its smoothed per-block
series (`f2(array)$est`), and `bglob[p]` is the global value the shift is measured
against. The result is one number per pair, used to recenter that pair's series.

The exact leave-one-out arithmetic — which blocks are included, the weighting, the
recombination formula, and the degenerate-input behavior — is documented in full
in the host reference (`qpfstats_jackknife.hpp`). The two kernels transliterate it
line for line; the notes below cover only what is specific to the GPU launch.

---

## 3. Native FP64: the cancellation carve-out

These kernels run in **native double precision**, deliberately, and must **never**
use the emulated double-precision arithmetic steppe uses for its large matrix
multiplies elsewhere.

The reason is cancellation. A jackknife estimate is a leave-one-out *difference* —
it subtracts one block's contribution from a whole-dataset total (`tot − numer·rel`
in the kernel). Subtracting two nearly equal numbers discards low-order bits, which
is exactly the situation where the faster emulated format loses accuracy. steppe's
policy is to spend extra precision precisely where cancellation happens and use the
faster emulated math everywhere else; a cancellation-prone reduction like this one
gets true FP64, not the emulated approximation.

The host reference carries its running sums in `long double`. GPUs have no
`long double` type, so the kernels reproduce the same computation in FP64 under two
rules that keep them bit-comparable to the host oracle:

- **Same operand order.** The reference walks blocks in ascending block index. In
  floating-point, addition order changes the last bits of a sum, so the kernels
  add in the identical ascending-block order.
- **Only the carry precision differs.** The single intended difference between the
  host reference and the kernel is that the host carries its running sums with a
  64-bit `long double` mantissa while the GPU carries them with a 53-bit FP64
  mantissa. Which values are included, the order they are combined, and every
  formula are otherwise identical.

This is the same leave-one-out block-jackknife family already proven on-device in
native FP64 by the qpAdm fit's per-row jackknife kernel, so it reuses a validated
approach rather than a new one. Both this kernel and its host reference are gated
on the 9-population genotype golden test at a relative tolerance of `1e-6`, and the
CPU backend and CUDA backend are required to agree — which is what proves the two
stay in step.

---

## 4. How the work is split across GPU threads

Both kernels are embarrassingly parallel over their outer axis: one GPU thread owns
one whole combination (for the numerator kernel) or one whole pair (for the
recenter kernel). That thread performs the entire short reduction over the roughly
711 blocks for its item, keeping the running sums in registers, and writes its
results at the end.

Threads use a grid-stride loop, so a launch can cover an item count larger than the
grid it was given: each thread processes item `i`, then `i + gridDim·blockDim`, and
so on until the axis is exhausted. Both launches use 128 threads per block and cap
the grid at the single shared grid-dimension limit; the stride loop handles any
overflow past that cap.

The per-item reduction is deliberately kept as a plain per-thread loop rather than
a block-wide cooperative reduction. At a segment length of a few hundred blocks a
cooperative reduction would only add launch and shared-memory overhead without
helping. Because the blocks for one item are contiguous in memory (see section 5),
each thread reads a single contiguous run.

Both launches are no-ops when their sizes are non-positive: the numerator kernel
returns immediately if `npopcomb <= 0` or `n_block <= 0`, and the recenter kernel
if `npairs <= 0` or `n_block <= 0`.

---

## 5. Memory layouts and index formulas

Getting the layouts right matters, because the numerator kernel mixes row-major and
column-major arrays on purpose.

**Numerator / global-estimate kernel.**

- `d_numsum` and `d_cnt` (inputs) are **row-major** with shape
  `[npopcomb × n_block]`. Cell (combination `c`, block `b`) is at `c*n_block + b`,
  so all of one combination's blocks are adjacent — the contiguous run one thread
  reads.
- `d_numer` (output) is **row-major** with the same shape and index, `c*n_block + b`.
  The kernel writes it and then reads it back within the same thread as the
  materialized per-block mean.
- `d_ymat` (output) is **column-major** with the same logical shape. Cell (`c`, `b`)
  is at `c + npopcomb*b`. It holds the same per-block mean as `d_numer`, transposed,
  for the downstream stage that wants it column-major.
- `d_y` (output) has one entry per combination, `[npopcomb]`.

**Recenter kernel.**

- `d_b` (input) is **column-major** with shape `[npairs × n_block]`. Cell (pair `p`,
  block `blk`) is at `p + npairs*blk`.
- `d_bglob` (input) has one entry per pair, `[npairs]`.
- `d_block_sizes` (input) is `[n_block]` integers — the per-block SNP count, used as
  the block weights.
- `d_shift` (output) has one entry per pair, `[npairs]`.

All of these arrays already live in GPU memory. The wrappers launch the kernels and
allocate or copy nothing themselves.

---

## 6. The `launch_qpfstats_numer_jackknife` parameters

| Parameter | Type | Meaning |
|---|---|---|
| `d_numsum` | `const double*` | `[npopcomb × n_block]` row-major per-combination, per-block numerator sum from the upstream stage; cell (c, b) at `c*n_block + b`. |
| `d_cnt` | `const double*` | `[npopcomb × n_block]` row-major per-combination, per-block count (the block weight). The materialized mean is `NaN` wherever `cnt <= 0`. |
| `npopcomb` | `int` | Number of population combinations — the outer axis, one thread per value. |
| `n_block` | `int` | Number of genome blocks — the length of each thread's reduction. |
| `d_numer` | `double*` | `[npopcomb × n_block]` row-major output: `numsum / cnt`, `NaN` where `cnt <= 0`. Written, then reread inside the same thread. |
| `d_ymat` | `double*` | `[npopcomb × n_block]` column-major output: the same per-block mean as `d_numer`, transposed for the downstream right-hand-side; cell (c, b) at `c + npopcomb*b`. |
| `d_y` | `double*` | `[npopcomb]` output: the per-combination global block-jackknife estimate (the value the fit is centered on). |
| `stream` | `cudaStream_t` | The CUDA stream the launch is queued on. |

The per-block mask and recombination follow the host reference exactly: a block is
`NaN` where `cnt <= 0`; a block is skipped from the leave-one-out step when its
relative weight reaches 1 (so `1 − rel` would be zero) or when its leave-one-out
value is not finite; and the whole combination returns `NaN` when no block survives
or a required weight sum is not positive.

---

## 7. The `launch_qpfstats_recenter_shift` parameters

| Parameter | Type | Meaning |
|---|---|---|
| `d_b` | `const double*` | `[npairs × n_block]` column-major smoothed per-pair, per-block series; cell (p, blk) at `p + npairs*blk`. Non-finite entries mark blocks the pair skips. |
| `d_bglob` | `const double*` | `[npairs]` the per-pair global value the shift is measured against. |
| `d_block_sizes` | `const int*` | `[n_block]` per-block SNP count, used as the block weights. |
| `npairs` | `int` | Number of population pairs — the outer axis, one thread per value. |
| `n_block` | `int` | Number of genome blocks. |
| `d_shift` | `double*` | `[npairs]` output: `bglob[p] − est(p)`, the per-pair recentering constant. |
| `stream` | `cudaStream_t` | The CUDA stream the launch is queued on. |

Matching the host reference, a pair with no usable blocks (total weight not
positive) produces `est = 0`, so its shift is simply `bglob[p]`; a block whose
relative weight reaches 1 is skipped from the leave-one-out mean.

---

## 8. Why this header is device-private

The declarations take a `cudaStream_t`, and the file includes `<cuda_runtime.h>`.
Any header that mentions a CUDA type can only be included by code already compiled
with the CUDA toolchain. That places this header on the device library's internal
surface, not on the public API.

steppe keeps a firm line between two kinds of header:

- **Public, CUDA-free headers** any caller can include. The qpfstats public
  contract sits there and pulls in no GPU dependency.
- **Device-internal headers** like this one, which sit behind that line and are used
  only by the backend and the kernel's translation unit.

Keeping the kernel-launch declarations here — rather than in the public header — is
what lets the public API stay free of any CUDA dependency. A caller who only wants
a qpfstats result never has to pull in `<cuda_runtime.h>`. The kernel bodies and
their launch syntax stay in the `.cu` file, and the backend reaches them only
through the two wrappers declared here.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

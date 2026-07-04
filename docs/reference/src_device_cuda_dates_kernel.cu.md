# `dates_kernel.cu` reference

## 1. Purpose

`src/device/cuda/dates_kernel.cu` holds the GPU kernels for **DATES admixture
dating**[^dates] — the method that estimates *how many generations ago* two ancestral
populations mixed by measuring how the correlation between ancestry-informative
markers decays with genetic distance along the chromosome.

The core quantity is a **weighted autocorrelation**: for a target (admixed)
sample, steppe builds a per-SNP weight, lays those weights onto a fine
genetic-map grid, and then, for every genetic-distance lag, sums the product of
the weight at one position with the weight a given distance away. That decaying
curve is fit to an exponential, and the decay rate converts to a date.

The one idea that makes this tractable is that the autocorrelation
(the sum over all SNP pairs of `weight[g] · weight[g + lag]`) is exactly what a
Fast Fourier Transform computes cheaply. So the SNP pairs are **never
materialized**. Instead of an `O(M²)` double loop over SNP pairs (which would be
hopeless for hundreds of thousands of SNPs), the work is `O(G log G)` per
chromosome, where `G` is the number of grid cells — and, importantly, **flat in
the number of SNPs**.

What this file is and is not:

- It **is** the elementwise, scatter, and reduction "glue" around the FFTs: the
  per-sample weight and residual computation, the scatter onto the grid, the
  `|F|²` and cross-power frequency-domain steps, the lag extraction, and the
  re-binning into the final correlation statistics. It also contains a
  self-contained exponential-decay curve fit that runs entirely on the GPU.
- It is **not** the f2 statistic (that lives elsewhere), and it is **not** a host
  SNP-pair loop.
- The forward and inverse FFTs themselves are **not** issued here. They are
  launched from the backend using the cuFFT library; these kernels only prepare
  the FFT inputs and consume the FFT outputs.

The kernel bodies and their launch syntax live only in this translation unit.
The rest of the library reaches them through the narrow launch-wrapper functions
declared in the companion header, never by touching the kernels directly. The
whole file mirrors the reference DATES implementation stage for stage so that
steppe reproduces its numbers.

---

## 2. How the kernels fit together (the pipeline)

The kernels run in a fixed order. Reading them in pipeline order makes the file
much easier to follow:

1. **Regression dot products** (Phase A) — for one target sample, compute two
   summary dot products across its valid SNPs. These give the regression
   coefficient that predicts the sample's genotype from the two reference
   ancestries.
2. **Residual scatter** (Phase B) — using that coefficient, compute a per-SNP
   residual weight and scatter it onto the fine genetic-map grid, accumulating
   three moments per grid cell (count, sum, sum-of-squares).
3. **Pack segments** — copy each chromosome's stretch of the grid into a
   zero-padded row sized for the FFT.
4. **FFT (issued by the backend)** — forward-transform each padded row.
5. **Power / cross power** — in the frequency domain, form `|F|²` (for an
   autocorrelation) or `conj(A)·B` (for a cross-correlation between two moment
   channels).
6. **Inverse FFT (issued by the backend)** — transform back to the lag domain.
7. **Extract lags** — pull out lags `0…diffmax`, divide by the FFT length to
   undo the FFT's scaling, and accumulate into the per-lag moment array.
8. **Accumulate bins** — re-bin the fine per-lag moments into the coarser
   per-(chromosome, output-bin) correlation statistics that the curve fit
   consumes.
9. **Fit curves** — fit each chromosome's (and the genome-wide) decay curve to an
   exponential and report the date.

Two more kernels sit off to the side of this main flow:

- **Target repack** prepares the packed target genotypes on the kept-SNP axis
  before Phase A runs.
- The **exponential-decay fit** is the final numerical step (section 8).

---

## 3. Phase A — regression dot products

`regress_dots_kernel` computes two scalar sums over one target sample's valid
SNPs:

- `dot12` = Σ `(target − ref2) · (ref1 − ref2)`
- `dot22` = Σ `(ref1 − ref2)²`

Their ratio is the regression coefficient (`yreg` in Phase B) that best predicts
the target's allele dosage as a blend of the two reference ancestry frequencies.

Each thread handles one SNP. It reads the target's 2-bit genotype code from the
packed record, skips it if the SNP is flagged invalid or the genotype is
missing, and otherwise converts the code (0, 1, or 2 copies) to a dosage in
`[0, 1]` by dividing by 2. The two per-SNP products are then reduced.

### The block reduction and its frozen invariant

Rather than have every thread do its own `atomicAdd` (which would serialize
under heavy contention), each block first sums its threads' contributions in
shared memory using a **binary halving tree**, then does a single `atomicAdd`
per block into the global totals.

The halving tree folds the upper half of the active range onto the lower half
each step (`off >>= 1`). This is correct **only if the block size is a power of
two** — a non-power-of-two size would drop an odd leftover element each round and
silently lose data. The sole caller launches with a block size of 256, so the
invariant always holds.

**Do not add odd-remainder handling to this tree.** Beyond being unnecessary, it
would change the fixed order in which the partial sums are combined, and that
order is part of what makes the result bit-for-bit reproducible.

---

## 4. Phase B — residual scatter onto the fine grid

`scatter_kernel` turns each valid SNP into a weighted residual and scatters it
onto the fine genetic-map grid. One thread per SNP:

1. Reads the target dosage `w0` and the two reference frequencies `w1`, `w2` (as
   in Phase A).
2. Forms the **prediction** from the regression coefficient `yreg` computed in
   Phase A: `pred = yreg·w1 + (1 − yreg)·w2`.
3. Forms the **residual** `r = w0 − pred` and the **ancestry weight**
   `wt = w1 − w2`, then the scattered value `y = r · wt`.
4. Looks up which grid cell this SNP falls in (`grid_cell[s]`, precomputed from
   the genetic map) and atomically adds three moments into that cell:
   - `z0 += 1` — a **count** of contributing SNPs in the cell,
   - `z1 += y` — the **sum** of the weighted residuals,
   - `z2 += y²` — the **sum of squares**.

These three per-cell channels are what the FFT stage autocorrelates and
cross-correlates. Multiple SNPs can map to the same cell, which is why the writes
are atomic adds.

---

## 5. The FFT glue kernels

Four small elementwise kernels wrap the cuFFT calls that the backend issues.

### `pack_segments_kernel` — build zero-padded FFT rows

Each chromosome occupies a contiguous span of the grid. This kernel copies each
chromosome's span into its own FFT-length row, zero-filling the rest of the row.
One thread per (chromosome, position) cell: it looks up the chromosome's first
and last grid index, and if this position falls within the chromosome's length it
copies the grid value, otherwise it writes zero. Zero-padding to the FFT length is
what lets a single batched transform handle all chromosomes at once and avoids
wrap-around (circular) contamination in the autocorrelation.

### `power_spectrum_kernel` — `|F|²`

Given the complex frequency-domain output of a forward FFT, this writes the real
power spectrum `re² + im²` (with a zero imaginary part) elementwise. Inverse-
transforming a power spectrum yields the **autocorrelation** of the original
signal — this is the step that replaces the SNP-pair double loop. One thread per
complex frequency bin.

### `cross_power_kernel` — `conj(A)·B`

For a **cross**-correlation between two different moment channels, this multiplies
one channel's frequency data by the complex conjugate of the other:
`conj(A)·B = (Ar·Br + Ai·Bi) + i(Ar·Bi − Ai·Br)`. Inverse-transforming the result
gives the cross-correlation. One thread per complex bin.

### `extract_lags_kernel` — pull out and rescale the lags

After the inverse FFT, the lag-domain signal is in the first entries of each row.
This kernel copies lags `0…diffmax` out into a compact per-lag array, dividing
each by the FFT length to undo the scaling that a forward-then-inverse FFT
introduces. It **accumulates** (`+=`) rather than overwrites, so the caller must
zero the destination once before the first sample and can then run many samples
into the same buffer. One thread per (chromosome, lag).

---

## 6. Re-binning lags into the correlation statistics

`accumulate_bins_kernel` collapses the fine per-lag moments into the coarser
per-(chromosome, output-bin) correlation statistics that the curve fit reads. One
thread per (chromosome, lag). It takes four fine-grid lag arrays — the
autocorrelation and cross-correlations of the moment channels — and sums each
into its output bin.

Two skip rules match the reference implementation exactly:

- **Lag 0 is skipped** (only lags of 1 or more contribute). The zero-lag term is
  the self-correlation and is not part of the decay curve.
- A lag with a **count below 0.5** (i.e. effectively zero SNP pairs landed in it)
  is skipped, so empty lags do not dilute the statistics.

The output bin index is `floor(lag / qbin)`, where `qbin` is how many fine lags
collapse into one output bin. Bins outside the valid range `[0, n_bin)` are
dropped.

The four input arrays map onto the four output statistic streams as follows (the
mapping is deliberate and mirrors the reference — note that the middle two are
crossed):

| Input lag array | Output statistic |
|---|---|
| `dd00` (count autocorrelation) | `s0` |
| `dd11` | `s12` |
| `dd02` | `s11` |
| `dd20` | `s22` |

All four writes are atomic adds, because many fine lags fold into one output bin.

---

## 7. Target-genotype repack

`repack_target_kernel` rebuilds the packed target genotypes on the **kept-SNP
axis**: given the target individuals' records indexed by original SNP position,
it produces new dense records indexed only by the SNPs that survived filtering.
It is a bit-for-bit port of the reference host code's shift-and-OR packing loop.

The genotype codes are 2 bits each, four to a byte, packed **most-significant
first** (positions shift by 6, 4, 2, 0 within a byte). Reading uses the shared
genotype-decode helper so the read order matches the host exactly, and the write
reproduces the host's `(codes_per_byte − 1 − position) · bits_per_code` shift.

### Why one thread owns a whole destination byte

The obvious parallelization — one thread per (individual, kept SNP) — would have
four threads writing into the same destination byte (four codes share a byte),
which is a data race. To be both **race-free and bit-exact**, each thread instead
owns one complete **destination byte**: it packs the up-to-four consecutive kept
codes that land in that byte (kept indices `db·4 … db·4+3`), building the byte in
a register with the same MSB-first shifts the host uses, then writes it once.
Because the host also writes destination codes in ascending kept-index order, the
byte layout is identical.

The kernel uses a grid-stride loop so any grid size covers all (individual, byte)
cells correctly.

---

## 8. The exponential-decay curve fit

The final stage fits each correlation-decay curve to a positive decaying
exponential and converts the decay rate into a date. It is a self-contained
numerical method with no library dependency, and it runs in **native double
precision** (not the emulated double precision used for the matrix-multiply
stages), because it is small and cancellation-sensitive. It is a stage-for-stage
mirror of the reference DATES fit; the host reference uses `long double`, and the
device uses `double` with the cancellation-tolerant algebraic form.

### `dev_linfit_2x2` — the inner least-squares solve

For a fixed decay base `v` (with `0 < v < 1`), this fits the model
`y[i] ≈ co0·vⁱ + c` and returns the residual sum of squares. It accumulates the
five sums needed for the normal equations in a single pass, then:

- **Affine mode** solves the full 2×2 system for both the amplitude `co0` and the
  constant offset `c`. If the system is singular (zero determinant), it returns
  NaN so the caller skips that `v`.
- **Non-affine mode** solves the 1×1 problem for `co0` only and pins `c = 0`.

A second pass computes the residual sum of squares against the fitted model.
Amplitude and offset are returned through pointers; the sum of squares is the
return value.

### `dates_fit_curves_kernel` — the search over decay rates

One thread fits one curve. The number of curves is small (roughly one per
chromosome plus one genome-wide), so this is a tiny launch; the grid is still
sized generically so a future many-curve batch would launch correctly. Each
thread:

1. **Defaults** its outputs to "no fit" (date 0, standard deviation 0, ok flag 0)
   and bails out immediately on a too-short curve (fewer than 3 points) or a
   non-positive step size.
2. **Coarse grid search** — scans a 4000-point grid of `v` across `(0, 1)`,
   solving the least-squares fit at each and keeping the one with the lowest
   residual sum of squares, subject to requiring a genuinely **decaying positive
   exponential** (amplitude `co0 > 0`).
3. **Fallback** — if no positive-amplitude fit was found at all (a degenerate
   curve), it repeats the same scan but allows any-sign amplitude, so the fit at
   least returns something rather than failing outright. If even that finds
   nothing, it gives up (leaving the "no fit" defaults).
4. **Ternary refine** — runs 200 iterations of ternary search in the interval
   around the best grid point to polish `v` to far finer resolution than the
   4000-point grid alone.
5. **Convert to a date** — the decay rate is `lambda = −log(v) / step`, where
   `step` is the genetic distance per bin. The reported outputs are:
   - **date** = `lambda`,
   - **standard deviation** = `sqrt(residual_sum_of_squares / curve_length)`,
   - **ok flag** = 1 when `lambda` is finite and positive, else 0.

The constants `4000` (coarse grid resolution) and `200` (ternary iterations) come
straight from the reference implementation and set the fit's precision.

---

## 9. Launch wrappers and shared conventions

Everything above is `private` to this file. The backend uses it only through the
thin `launch_dates_*` wrapper functions at the bottom, one per kernel. Each
wrapper computes the launch grid, launches the kernel on the caller's stream, and
checks for a launch error. Passing the stream in lets the DATES work overlap with
other GPU work and keeps it off the default stream.

Shared conventions:

- **Block size** is a fixed 256 threads (`kBlock`). The Phase-A reduction depends
  on this being a power of two (section 3).
- **Grid size** is computed by a single ceiling-division helper (`grid_for`) so no
  kernel picks its own launch geometry ad hoc.
- The complex-data wrappers take `void*` and cast to the complex type internally,
  so the header stays free of any CUDA/cuFFT types and can be included by
  non-GPU code.
- Wrappers with nothing to do (zero curves, zero repack bytes) return without
  launching.
- The curve-fit wrapper marks a coarse profiling range around itself; the marker
  compiles to nothing unless profiling is explicitly enabled at build time.

---

[^dates]: **DATES** — admixture dating by ancestry-covariance decay. Chintalapati M, Patterson N, Moorjani P. *The spatiotemporal patterns of major human admixture events during the European Holocene.* eLife 2022;11:e77625.

# `cuda_backend_dates.cu` reference

## 1. Purpose

This file is the GPU implementation of the three DATES routines that live on the
CUDA backend. DATES estimates when an admixture event happened by measuring how a
weighted ancestry signal decays as you move apart along the genome: recent
admixture leaves long-range correlation, older admixture leaves only short-range
correlation, and the rate of decay converts to a number of generations.

The file holds the out-of-line bodies of three methods:

1. **`dates_curve`** — the heavy lifting. It turns per-sample genotypes plus two
   source-population allele frequencies into the decay curve's *sufficient
   statistics*: for every chromosome and every output distance bin, a handful of
   summed moments from which the covariance curve is later read off.
2. **`dates_repack`** — a bit-level utility that copies each target individual's
   genotype record down onto a smaller "kept SNP" axis on the GPU.
3. **`dates_fit`** — fits a single decaying exponential to each finished curve and
   reports the implied admixture date.

The bodies were split out of the larger CUDA backend translation unit verbatim;
the split changed no math, precision, or behavior. All three join the same private
GPU compile target and are never visible outside the device layer.

---

## 2. The autocorrelation idea

The naive way to build a decay curve is to look at every pair of SNPs, group the
pairs by the genetic distance between them, and average a correlation statistic
within each distance group. For a real dataset that is on the order of a trillion
pairs — far too many to form directly.

DATES avoids ever building that object by using a fast Fourier transform (the same
trick the ALDER method uses). If you lay the per-SNP signal down on a fine grid
indexed by genetic-map position, then the sum over all pairs at a given lag is
exactly the *autocorrelation* of that grid at that lag. And an autocorrelation is
cheap to compute through the frequency domain:

```
autocorrelation(z)[lag] = inverse_FFT( |FFT(z)|² )[lag]
```

So instead of a trillion pairwise products, each sample costs a few forward
transforms, an elementwise multiply, and a few inverse transforms. The trillion-pair
object is never materialized. The reported covariance curve is a ratio of two such
autocorrelations — a signal autocorrelation over a count autocorrelation — which is
what makes it a proper correlation at each distance rather than a raw sum.

---

## 3. Building the decay curve: `dates_curve`

`dates_curve` uploads its inputs once, keeps everything resident on the GPU, and
lets the host drive only a short loop over the admixed target samples (typically
around a hundred) plus a one-time setup. Every heavy step inside the loop runs on
the GPU.

At a high level the loop, per sample, does: fit a per-sample regression
coefficient, scatter a residual signal onto a fine per-chromosome grid, transform
each chromosome, form power and cross-power spectra, transform back, and pull out
the autocorrelation values at the lags of interest — accumulating those into
per-chromosome moment arrays that are summed across all samples. After the loop the
per-lag moments are re-binned into the output distance bins.

If any of the sizing inputs is non-positive (no chromosomes, no bins, no SNPs, no
targets, and so on) the function returns an empty-but-OK result immediately.

### Choosing the FFT length

All chromosomes are transformed together in one batched plan, so they must share a
single transform length. The code picks one power-of-two length `n_fft` and uses it
for every chromosome. Two requirements set the floor:

- **`n_fft` must be at least twice the longest chromosome's grid length.** A linear
  autocorrelation needs the input zero-padded to at least double its length;
  otherwise the circular nature of the FFT wraps the tail around and corrupts the
  result. Padding to `2 × max_len` guarantees the wrap-around region is all zeros.
- **`n_fft` must be at least `diffmax + 1`**, so every requested lag actually fits
  inside the transform.

The code takes the larger of those two, then rounds up to the next power of two.
The complex spectrum length is then `n_fft / 2 + 1`, the standard size for the
real-to-complex transform of a real signal.

One subtlety worth knowing: the reference DATES implementation uses a *per-chromosome*
transform length, while this code uses one common, possibly larger length for all
chromosomes. That difference does not change the answer. Once the length clears the
`2 × len` bar, the linear autocorrelation values at the lags below `len` are
identical regardless of how much extra zero padding sits beyond them, and the
`1 / n_fft` normalization (applied later, when the lags are extracted) matches the
reference exactly. So the batched common-length choice is numerically equivalent to
the reference's per-chromosome choice.

### The cuFFT plans and why they never leak

Two batched plans are created up front and reused across every sample: a
forward real-to-complex plan and an inverse complex-to-real plan, each batched over
all chromosomes. Creating them once instead of per-sample avoids repeated setup
cost.

These plans own real GPU scratch memory (workspaces), so leaking them leaks VRAM,
and the leak would compound every time the routine is retried. The function can
throw at many points *after* the plans are created — when binding the stream, on
every FFT execution inside the loop, and on every error check in between. The plans
are therefore held in a small owning wrapper whose destructor tears them down. That
destructor runs on the normal return **and** on any exception thrown along the way,
so both plans are always released no matter which exit path is taken. Keeping that
ownership intact is what guarantees the no-leak behavior; a bare handle destroyed
only at the end of the function would leak on any early throw.

### Device buffers

Everything the loop touches is allocated on the GPU before the loop starts:

- The two source-population frequencies, the per-SNP validity mask, the packed
  target genotypes, the per-SNP grid-cell index, and the per-chromosome first/last
  SNP bounds.
- Three fine per-chromosome grids (`z0`, `z1`, `z2`), one holding a count, one the
  signal, one the signal squared.
- Three padded real FFT input buffers (one per grid) and one inverse-transform
  output buffer.
- Complex scratch for the three forward spectra plus one shared power/cross-power
  buffer. (The complex type is a pair of doubles.)
- Four per-chromosome, per-lag moment accumulators (`dd00`, `dd11`, `dd02`, `dd20`)
  that are zeroed once and summed into across all samples.
- The output sufficient-statistic arrays, sized per chromosome × output bin.

The input uploads and the moment-array zeroing are queued asynchronously on the
backend's single stream.

### The per-sample loop

For each admixed target sample `i`:

1. **Regression coefficient.** Two dot products are reduced on the GPU over the
   sample's valid SNPs — one mixing the sample's dosage against the source-frequency
   difference, one being the squared source-frequency difference. Their ratio is the
   per-sample regression coefficient `y`. This is the only place the loop copies a
   scalar back to the host and synchronizes, because `y` is needed to drive the next
   kernel. A zero denominator yields `y = 0`.
2. **Scatter the residual onto the grid.** Using `y`, each valid SNP's residual and
   its signal (the residual times the source-frequency difference) are scattered by
   atomic add onto the fine grid: `z0` accumulates a count of 1, `z1` the signal, and
   `z2` the signal squared, at the grid cell that SNP maps to. The three grids are
   re-zeroed before each sample.
3. **Pack chromosome segments into FFT rows.** Each chromosome's slice of the fine
   grid is copied into its own zero-padded row of the batched FFT input, for all
   three grids.
4. **Forward transforms.** The three padded grids are transformed to their complex
   spectra `F0`, `F1`, `F2`.
5. **Power / cross-power and inverse transforms.** Four autocorrelation-style
   quantities are formed by multiplying spectra elementwise, transforming back, and
   pulling out the lag values into the four moment accumulators:
   - `dd00` — the autocorrelation of the count grid (`|F0|²` transformed back).
   - `dd11` — the autocorrelation of the signal grid.
   - `dd02` — the cross-correlation of the count grid with the signal-squared grid.
   - `dd20` — the cross-correlation of the signal-squared grid with the count grid.

   The lag-extraction step applies the `1 / n_fft` normalization and adds into the
   running per-sample sum, so after the loop each `dd` array holds the total over all
   samples.

### Re-binning into the output statistics

After the loop the fine per-lag moments are collapsed into the coarser output
distance bins. For each chromosome and each lag with a meaningful count, the lag maps
to an output bin, and the four moments are added into the output sufficient
statistics for that (chromosome, bin) cell. The finished statistics are copied back
to the host, the stream is synchronized once, and the result is returned. The two FFT
plans are released by their destructor at that point — and equally on any earlier
throw.

---

## 4. The four correlation moments and the crossed subscripts

The output carries six sufficient-statistic arrays, but only four are populated on
this path; the two "mean" terms are left at zero because the curve is computed
without subtracting means. The four that matter map from the `dd` accumulators as
follows:

| Output stat | Fed by | Role in the curve |
|---|---|---|
| `s0` | `dd00` | pair-count autocorrelation — the shared denominator |
| `s12` | `dd11` | signal autocorrelation — the covariance numerator |
| `s11` | `dd02` | one variance normalizer |
| `s22` | `dd20` | the other variance normalizer |

The reported correlation at each (chromosome, bin) is then

```
corr = (s12 / s0) / sqrt( (s11 / s0) · (s22 / s0) )
```

which is a covariance over the product of two standard deviations — a normalized
correlation.

The subscripts look scrambled on purpose, and it is easy to misread. The two digits
in a `dd` name index *moment powers* over the (count, signal) pair, while the two
digits in an `s` name index the two correlation *series* in the final formula. That
is why the numerator series-cross term `s12` is fed by the signal autocorrelation
`dd11`, while the two per-series variances `s11` and `s22` are fed by the squared
normalizers `dd02` and `dd20`. The mapping `dd11 → s12` and `dd02 → s11` is correct,
not a typo.

---

## 5. Parameters that are accepted but deliberately ignored

`dates_curve` takes three arguments it never reads. Each is intentional, and each is
marked as unused so the compiler stays quiet:

- **`target_ploidy`** — DATES treats the dosage as the genotype code divided by two,
  which is the same regardless of ploidy, so the per-sample ploidy is not needed here.
- **`binsize`** — the output-bin width is not applied as a multiplier; it is already
  folded into how a lag is turned into a bin index (an integer floor division), so the
  raw width value is redundant on this path.
- **`precision`** — this path does not consult the general precision policy. It always
  runs the transforms in native double precision (see the next section).

They stay in the signature because the shared interface passes them to every backend,
and keeping them documents what the routine could use but does not.

---

## 6. Precision on this path

Everything numerically sensitive here runs in **native double precision**. The FFTs
use double-precision transforms, and the per-sample weight and residual arithmetic is
done in native double as well. This path never uses the faster emulated-double math
that the matrix-multiply-heavy parts of steppe default to, and it never touches the
dense linear-algebra libraries — there is no general matrix multiply, no solver, and
no factorization on the DATES curve path. The residual/weight computation is one of
the deliberately carved-out spots where native double is used because it is prone to
cancellation and needs the extra headroom.

---

## 7. Repacking target genotypes: `dates_repack`

`dates_repack` rewrites each target individual's packed 2-bit genotype record from a
wide source axis down onto a dense "kept SNP" axis — the set of SNPs that survived
filtering. It uploads the full target records and the kept-index map once, runs a
single device kernel that gathers each destination bit from its source position (one
GPU thread per destination byte, so no two threads write the same byte), and copies
only the compact repacked buffer back to the host.

The work is pure integer bit-shuffling and is bit-for-bit identical to the equivalent
host routine — it reads genotype codes the same way and writes the same shift/or
pattern — so the repacked records are exactly what the host would have produced. That
in turn keeps the downstream curve moments, and the final date, unchanged. The
function is a no-op for empty or degenerate inputs.

---

## 8. Fitting the exponential decay: `dates_fit`

`dates_fit` takes a set of finished, windowed correlation curves and fits a single
decaying exponential to each one, then reports the implied admixture date. It is
batched: one GPU thread handles one curve, so the full-data curve and the
leave-one-chromosome curves (used to estimate the standard error) are all fit in a
single launch. The curves are uploaded once and only the small per-curve outputs come
back.

Each thread runs the same coarse-to-fine one-dimensional search the reference DATES
uses: a scan over a few thousand candidate decay rates, then a couple hundred rounds
of ternary refinement to home in, with an inner two-by-two normal-equation solve at
each candidate to fit the curve's amplitude (and, when the affine option is on, a
constant offset). Those inner accumulators run in native double precision — the GPU
has no extended-precision type, and native double is the cancellation carve-out that
holds the (looser) accuracy tier this fit is checked against.

For each curve the routine returns three values:

| Field | Meaning |
|---|---|
| `date_gen` | the fitted decay rate, expressed as a number of generations |
| `error_sd` | the residual standard deviation, `sqrt(residual_sum_of_squares / window_length)` |
| `ok` | `1` only if a genuinely decaying, positive exponential was found (a finite, positive rate) |

Curves that fail the sanity checks (non-positive window, non-positive step, or no
decaying fit) come back with `ok = 0` so the caller can discard them.

# `dates_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/dates_kernel.cuh` declares the GPU launch wrappers for the DATES
engine[^dates] — the part of steppe that dates an admixture event by measuring how a
weighted linkage-disequilibrium (LD) signal decays with genetic distance and
fitting an exponential to that decay curve. The decay rate is read out as the
number of generations since admixture.

The header holds only **declarations**, one per GPU step, all named `launch_*`.
Each declared function is a thin wrapper: the host orchestration code calls
`launch_something(...)`, and the wrapper is the only thing that actually launches
the matching GPU kernel. The kernel bodies themselves — the code that runs on the
GPU, and the `<<<...>>>` launch syntax — live in the companion `.cu` file, never
here. Splitting it this way lets ordinary host code call into the DATES kernels
without having to compile any GPU kernel code of its own.

There are nine launch wrappers. Seven of them form one continuous pipeline that
turns raw genotypes into a set of correlation-curve statistics; the remaining two
are a genotype-repacking helper that runs before the pipeline and the final
exponential-decay curve fit that runs after it.

---

## 2. The device-private seam and the launch-wrapper rule

This header mentions CUDA types directly — every wrapper takes a `cudaStream_t`
(the GPU work queue the launch should go on). Because it names GPU types, it is
**private to the GPU layer**. It is the internal boundary between the GPU backend
and the DATES kernels, and it must not be included by CUDA-free code. The public,
CUDA-free entry point to the DATES feature is a separate header
(`include/steppe/dates.hpp`); that one is what the rest of the library and the
command-line tool see.

The rule that host code never contains kernel launch syntax is what forces the
split described in section 1: host files include *this* header (plain function
declarations) and call the wrappers; only the paired `.cu` file contains the kernel
bodies and the `<<<...>>>` launches.

Every wrapper follows the same calling convention:

- Pointers named `d_*` are addresses in GPU memory, not host memory.
- Output buffers are **pre-allocated and pre-zeroed by the caller**. The kernels
  add into them (many use atomic adds), so they rely on the destination starting
  at zero.
- The final `cudaStream_t stream` argument is the GPU queue the launch is placed
  on, so the caller controls ordering and overlap.

---

## 3. The FFT autocorrelation strategy (the ALDER trick and the four moments)

The covariance curve steppe needs is, in principle, a sum over **pairs** of SNPs
at every genetic-distance lag. For a real dataset that is on the order of a trillion
(10^12) SNP pairs, which can never be formed directly. DATES avoids this with the
classic ALDER trick[^alder]: the covariance-at-each-lag curve is exactly the
**autocorrelation** of a signal laid out along the genetic map, and an
autocorrelation can be computed with a Fast Fourier Transform (FFT) instead of an
explicit double loop over pairs.

Concretely, if a per-position value `z` is binned along the genetic map, its
autocorrelation at lag `l` is `Σ_g z[g]·z[g+lag]`, and that whole curve equals
`inverse-FFT( |FFT(z)|² )`. So each curve steppe wants becomes: FFT the binned
grid, take a power (or cross-power) spectrum, inverse-FFT, and read off the lags.
The trillion-pair object is never built.

### The three grids and the four moments

For each admixed sample the pipeline builds **three** binned grids along the
genetic map:

| Grid | What each bin accumulates | Nickname |
|---|---|---|
| `z0` | a count of `1` per SNP that lands in the bin | the **count** grid |
| `z1` | the LD **signal** value per SNP | the **signal** grid |
| `z2` | the signal **squared** | the **signal²** grid |

From these, four autocorrelation/cross-correlation "moments" are formed per lag.
The two-digit name of each moment records which power of the signal goes into each
side of the correlation (0 = the plain count grid, 1 = the signal grid, 2 = the
signal² grid):

| Moment | Formed from | Meaning |
|---|---|---|
| `dd00` | `\|FFT(z0)\|²` | autocorrelation of the **count** grid |
| `dd11` | `\|FFT(z1)\|²` | autocorrelation of the **signal** grid |
| `dd02` | cross of `z0` and `z2` | count × signal² |
| `dd20` | cross of `z2` and `z0` | signal² × count |

The correlation curve is then assembled from these moments as

```
corr(lag) = (dd11/dd00) / sqrt( (dd02/dd00) · (dd20/dd00) )
```

This is DATES's "calccorr mode 1," which means the series means are **not**
subtracted before correlating. Because the means are not subtracted, the two
mean cross-terms that a full correlation would need (`dd01` and `dd10`) are unused,
so steppe never computes them at all.

---

## 4. Per-sample regression and residual scatter

The first two wrappers run once **per admixed target sample** and produce that
sample's contribution to the three grids. "Valid SNPs" for a sample means SNPs
where both source populations have a defined allele frequency **and** the sample's
own genotype is not missing.

Throughout these two steps the per-SNP quantities are:

- `w0` = the sample's allele dosage at the SNP (the 2-bit genotype code divided by 2).
- `w1` = source population 1's allele frequency at the SNP.
- `w2` = source population 2's allele frequency at the SNP.

### `launch_dates_regress_dots` — the mixing-proportion regression

Over the sample's valid SNPs, this reduces two dot products:

```
dot12 = Σ (w0 - w2)(w1 - w2)
dot22 = Σ (w1 - w2)²
```

The host later divides them to get the regression coefficient `y = dot12/dot22`,
which is that sample's estimated mixing proportion. `d_dot12` and `d_dot22` are
single doubles that the caller pre-zeroes; the kernel atomic-adds each thread
block's partial sum into them. This reproduces the corresponding regression in the
reference DATES program.

Key parameters:

| Parameter | Meaning |
|---|---|
| `d_src1`, `d_src2` | source-population allele frequencies (`w1`, `w2`) |
| `d_valid` | per-SNP validity mask for the two sources |
| `d_packed`, `bytes_per_record` | the packed genotype records and their per-record byte stride |
| `sample` | which target sample this launch is for |
| `M` | number of SNPs |
| `d_dot12`, `d_dot22` | OUT: the two scalar dot products (pre-zeroed) |

### `launch_dates_scatter` — residual signal onto the grid

Using the regression coefficient `y` from the previous step (passed in as `yreg`),
this computes, per valid SNP, the regression **residual** and the LD **signal**:

```
residual r = w0 - ( y·w1 + (1 - y)·w2 )
signal     = r · (w1 - w2)
```

and scatters them onto the fine genetic-map grid at that SNP's cell:

```
z0[cell] += 1
z1[cell] += signal
z2[cell] += signal²
```

The three grids `d_z0`, `d_z1`, `d_z2` each have `numqbins` bins, are pre-zeroed by
the caller, and are filled with atomic adds (`cell = d_grid_cell[s]` maps each SNP
to its bin). These are exactly the `z0`/`z1`/`z2` grids of section 3.

---

## 5. The FFT autocorrelation of the grid

The next four wrappers turn the three grids into the four per-lag moments, using the
FFT strategy of section 3. They are **batched over chromosomes** (`n_chrom`): the
autocorrelation is done independently per chromosome so that lags never cross a
chromosome boundary.

### `launch_dates_pack_segments` — lay each chromosome out as an FFT row

The FFT wants each chromosome's grid segment placed at the start of a
zero-padded row of length `n_fft` (zero padding is what makes a cyclic FFT compute
the *linear*, non-wrapping autocorrelation). For each chromosome `kc`, the segment
`d_grid[slo..shi]` (its first and last bins given by `d_chrom_first`/`d_chrom_last`)
is copied into `d_padded[kc*n_fft + 0 ...]` and the remainder of the row is zeroed.

### `launch_dates_power_spectrum` — the self-autocorrelation numerator

Given a chromosome-batched frequency-domain buffer `d_freq`, this produces the
power spectrum `out[k] = re² + im²` (a real value stored in the complex slot with a
zero imaginary part), batched `n_chrom × n_cplx`. Applied to `FFT(z0)` it yields the
frequency-domain form of `dd00`; applied to `FFT(z1)` it yields `dd11`.

### `launch_dates_cross_power` — the count×signal² cross term

Given two chromosome-batched frequency buffers `d_freq_a` and `d_freq_b`, this
produces the cross-power `out = conj(A)·B`, batched `n_chrom × n_cplx`. This is how
`dd02` (count × signal²) and `dd20` are formed in the frequency domain.

### `launch_dates_extract_lags` — read lags out of the inverse FFT

After the inverse FFT (`d_inv`, a real buffer of shape `n_chrom × n_fft`), the
autocorrelation value at lag `l` sits at inverse-FFT index `l`. This wrapper copies
lags `[0, diffmax]` into a compact array `d_dd` of shape `n_chrom × (diffmax+1)`,
applies the `1/n_fft` scale that the inverse FFT leaves off, and — importantly —
**sums across samples** by adding (`+=`) into `d_dd` rather than overwriting. So the
same `d_dd` accumulates every sample's contribution as the per-sample loop runs.

---

## 6. Re-binning lag moments into correlation statistics

### `launch_dates_accumulate_bins`

The lag moments live on a fine grid (one entry per lag up to `diffmax`). The
correlation curve that gets fit is coarser — grouped into output bins of width
`qbin`. This wrapper collapses the fine lag moments into the per-(chromosome,
output-bin) sufficient statistics of the correlation.

For each chromosome `kc` and each lag `d` in `[1, diffmax]` whose count
autocorrelation `dd00[d]` is at least `0.5` (i.e. the lag actually has data), the
output bin is `s = floor(d / qbin)`, and the statistics accumulate as:

```
s0[kc,s]  += dd00[d]     (the weight / count)
s12[kc,s] += dd11[d]     (the numerator cross-term)
s11[kc,s] += dd02[d]     (the first-series variance)
s22[kc,s] += dd20[d]     (the second-series variance)
```

The two mean-terms (`s1`, `s2`) are left at zero because mode 1 does not subtract
means. From these, the per-bin correlation is `corr = s12 / sqrt(s11·s22)`.

### The crossed subscripts are intentional

The map from `dd`-moment names to `s`-statistic names looks scrambled — `dd11`
lands in `s12`, `dd02` lands in `s11`, `dd20` lands in `s22` — and that is
**correct, not a bug**. The two naming schemes count different things:

- The **`dd` digits** index the *powers of the signal* that went into that moment
  (0 = count grid, 1 = signal grid, 2 = signal² grid) — see section 3.
- The **`s` digits** index the two *correlation series* whose correlation is being
  built (`s12` is the cross-term between series 1 and series 2; `s11` and `s22` are
  the per-series variances).

Reading it through: the correlation's numerator cross-term (`s12`) is exactly the
signal-with-signal autocorrelation, which is `dd11`. The two variances that
normalize it (`s11`, `s22`) are the two squared-normalizer cross moments `dd02` and
`dd20`. So `dd11 → s12` and `dd02 → s11` are the intended, verified mappings, even
though the digits do not line up.

---

## 7. Target-genotype repack

### `launch_dates_repack_target`

Before the pipeline runs, the target individuals' genotypes have to be repacked
from the full, file-order SNP layout down to the dense axis of only the SNPs that
survived filtering. This wrapper is the on-GPU version of a bit-shuffle that the
reference program does on the host.

One GPU thread handles one `(target individual i, kept SNP ks)` pair. It reads the
2-bit genotype code out of source record `i` at the original SNP position
`d_kept_src[ks]`, and writes that code into bit position `ks` of a new, dense
per-target record. The operation is **integer / bit-exact**: it reproduces the
host's shift-and-OR packing bit for bit, so the repacked records are bit-identical
to what the host would have produced, and every downstream number is unchanged.

| Parameter | Meaning |
|---|---|
| `d_src`, `src_bpr` | the full per-target packed records and their per-record byte stride |
| `d_kept_src`, `M_kept` | kept-index → original SNP index, and the number of kept SNPs (the dense length) |
| `n_target` | number of target individuals |
| `dst_bpr` | dest per-record byte stride (the packed byte count for `M_kept` SNPs) |
| `d_dst` | OUT: the dense repacked records (pre-zeroed) |

---

## 8. Exponential-decay curve fit

### `launch_dates_fit_curves`

The final step fits the model `A·vⁱ + c` to each windowed correlation curve, where
`i` indexes the lag bins along the curve. It is **batched over `n_curves` curves**,
with one GPU thread fitting one curve (each thread reads its own
`win_len`-long slice of `d_curves`). The `+ c` constant term is only included when
`affine` is true.

The fit follows the reference DATES **coarse-to-fine** search exactly:

1. A **4000-point scan** over the decay base `v`, requiring the leading coefficient
   `co0 > 0` (so the fit genuinely decays). If no decaying fit is found, it falls
   back to accepting any sign.
2. A **200-iteration ternary refinement** around the best grid point.
3. An inner **2×2 normal-equation solve** at each candidate `v` to get the linear
   coefficients.

For each curve it writes:

| Output | Meaning |
|---|---|
| `d_date[c]` | the fitted decay rate `λ`, in **generations** |
| `d_sd[c]` | `sqrt(rss / win_len)` — a spread estimate from the residual sum of squares |
| `d_ok[c]` | `1` only if a finite `λ > 0` was successfully fit, else `0` |

`step` is the bin width in Morgans, used to convert the fitted base into a rate in
generations.

**Precision note.** This kernel runs in **native double precision throughout**. The
GPU has no `long double`, so the normal-equation accumulators — which are the
cancellation-sensitive part — are deliberately left in native FP64 rather than the
emulated-double math used elsewhere in steppe. The date result is validated against
its reference at a loose (about 2%) tolerance, which this native-FP64 path meets.

---

[^dates]: **DATES** — admixture dating by ancestry-covariance decay. Chintalapati M, Patterson N, Moorjani P. *The spatiotemporal patterns of major human admixture events during the European Holocene.* eLife 2022;11:e77625.
[^alder]: **ALDER** — admixture-induced linkage-disequilibrium decay. Loh P-R, Lipson M, Patterson N, Moorjani P, Pickrell JK, Reich D, Berger B. *Inferring admixture histories of human populations using linkage disequilibrium.* Genetics 2013;193(4):1233–1254.

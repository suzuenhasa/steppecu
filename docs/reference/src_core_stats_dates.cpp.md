# `dates.cpp` reference

## 1. Purpose

`src/core/stats/dates.cpp` implements `run_dates`, the one function behind the
admixture-*dating* tool. Given a target (admixed) population and two reference
source populations, it estimates how many generations ago the target formed as a
mix of the two sources, along with a standard error and the raw decay curve the
estimate was fit to.

It is a close sibling of the genotype-reading D-statistic tool. It reuses the
same shared front-end — open the genotype files, read the three populations,
decode allele frequencies — and then diverges into a completely separate engine
built around a Fast Fourier Transform (FFT). It never reads or writes the
precomputed "f2 cache" that the f-statistic tools use.

The whole design turns on one point. The quantity being measured is a covariance
between *pairs* of genome positions. A genome has on the order of a million
positions, so there are on the order of a trillion pairs. That pairwise object is
never formed. Instead, the covariance at every distance at once is computed as the
autocorrelation of a signal laid out on a grid, using the identity that the
autocorrelation of an array equals the inverse FFT of the squared magnitude of its
FFT. Because of that reformulation, the host does only tiny work: build the grid,
subtract per-chromosome contributions, run about two dozen exponential fits over a
roughly thousand-point curve, and a weighted jackknife. There is deliberately **no
host loop over position pairs** — that would be the central performance trap, and
it is designed out.

The engine is pinned for parity to the reference DATES[^dates] tool (the Moorjani-lab C
source, version 750). Where a constant or a formula matches that reference exactly,
it is called out below.

---

## 2. Named constants

A handful of file-local constants carry the values that would otherwise be bare
numbers scattered through the code. Each has a specific reason to be exactly what
it is, and each matches the reference DATES tool.

| Constant | Value | What it's for |
|---|---|---|
| `kPrimaryGpu` | `0` | Which GPU the run uses. Dating runs on a single GPU, so this is always device 0. |
| `kPloidyDiploid` | `2` | The ploidy forced during the allele-frequency decode. DATES uses a plain reference/alternate frequency, and the per-sample dosage path divides the genotype by 2 directly, so it is treated as diploid regardless of the sample's real ploidy. |
| `kInterChromGapMorgans` | `5.0` | The gap, in Morgans, inserted between chromosomes when all positions are laid end-to-end on one continuous genetic-distance axis. A 5-Morgan gap is far larger than any real chromosome span, so each chromosome's fine-grid cells stay disjoint and the per-chromosome FFT segments can never overlap. |
| `kMinJackWeight` | `1e-6` | The zero-weight filter for the jackknife. A leave-one-out block whose weight falls below this is dropped before the standard-error calculation. Because the weight is a position count, any block with real data clears it easily — this only removes empty or degenerate blocks. |
| `kCorrDenomFloor` | `1.0e-20` | A divide-by-zero guard added under the square root in the correlation formula. A bin with zero variance would otherwise divide by zero and return not-a-number; with this tiny floor added it returns a correlation of 0 instead. It is negligible against any genuine variance. |

---

## 3. The weighted block jackknife (`weight_jack`)

`weight_jack` turns the per-chromosome leave-one-out dates into a single point
estimate and its standard error. It is the standard weighted block jackknife, the
same family used by the D-statistic tool, and it mirrors the reference DATES
`weightjack`/`weightjackx` routine[^dates].

**Inputs.** A list of leave-one-out dates (one per chromosome, each computed with
that chromosome held out), the paired weights (each the number of positions on all
the *other* chromosomes), and the full-data date.

**Outputs.** The jackknife point estimate `est` and the standard error `sig`. Both
come back as not-a-number if the data is too thin to support the calculation.

**How it works.**

1. Drop any block whose weight is below `kMinJackWeight` or whose date is not
   finite. If fewer than two blocks survive, return not-a-number for both outputs —
   a jackknife needs at least two replicates.
2. The point estimate combines two pieces: the sum of the differences
   `(full date − leave-one-out date)` across blocks, plus a weight-weighted average
   of the leave-one-out dates. This is the reference tool's jackknife-estimate
   equation.
3. The variance uses a per-block leverage factor `hh = (total weight) / (this
   block's weight)`. For each block it forms a pseudo-value from `hh`, the full-data
   date, and the leave-one-out date, squares it, divides by `hh − 1`, and averages
   over the blocks. The standard error is the square root of that variance (or
   not-a-number if the variance is not positive).

All of the accumulation is done in `long double` for extra precision headroom,
since it is a small sum of differences that can suffer cancellation.

A second tiny helper, `find_pop`, resolves a population label to its position index
in the decoded (alphabetically sorted) partition, returning −1 if the label is
absent.

---

## 4. The processing pipeline

`run_dates` runs as seven stages. It first validates the run knobs at entry: if the
bin size, the fine-grid factor, or the maximum distance is not strictly positive,
it returns an invalid-config status immediately. The result's precision tag is set
to native double precision up front (see section 9). Sections 5 through 8 walk the
stages; section 9 covers precision and where each heavy piece runs.

---

## 5. Decode front-end and position selection

**Reading the data (stage 1).** One shared front-end helper opens the genotype
reader, reads the three needed populations (target, source1, source2) as an
explicit partition, reads the per-position metadata table, and reads the genotype
tile in whatever on-disk format the files use. This is the identical front-end the
D-statistic and f-statistics tools use. Allele frequencies are then decoded with
forced diploid ploidy. The three population labels are resolved to their axis
indices in the decoded partition; if any is missing, the run returns an
invalid-config status.

**Keeping the autosomes (stage 2).** Positions are processed in file order, which
is sorted by chromosome and then by genetic position. Only autosomes are kept —
chromosomes 1 through 22, using the shared `kAutosomeChromMin`/`kAutosomeChromMax`
bounds, matching the parity autosomes-only default[^at2]. For each kept position the
code records, in lockstep:

- the two source allele frequencies (the frequency difference between them is the
  per-position weight the engine uses);
- a validity flag, set when both sources have a nonzero effective sample size at
  that position;
- the chromosome number and the genetic-map position.

**Re-packing the target (stage 2, continued).** The target population's packed
genotypes still carry the *full* position record. They are re-packed onto the
kept-position axis into a dense per-individual record, so that downstream the
grid-cell index lines up directly with the kept axis. This repack is a pure
integer bit-shuffle, and it runs through the compute backend rather than on the
host — a gather on the GPU, or a bit-exact CPU reference. Because it is
bit-for-bit identical either way, the FFT moments that consume it are unchanged
regardless of which path ran.

---

## 6. The genetic-map grid and curve dimensions

**The fine grid (stage 3, `setqbins`).** All kept positions are laid out on one
continuous genetic-distance axis. Walking positions in file order, the cumulative
distance grows by the gap between consecutive genetic-map positions within a
chromosome, and jumps by the fixed `kInterChromGapMorgans` gap whenever the
chromosome changes so that chromosomes never overlap. Each position's fine-grid
cell is `floor(cumulative distance / qb)`, where `qb = binsize / qbin` is the
fine-cell width — finer than an output bin by the `qbin` factor. The code also
tracks which chromosomes are present, in order of appearance, and the first and
last grid cell each one spans; those extents drive the per-chromosome bookkeeping
later.

**The curve dimensions (stage 4).** Two integer grid dimensions are derived from
the run knobs: the number of output bins, `round(maxdis / binsize)`, and a finer
dimension, `round(qbin · maxdis / binsize)`. Both are computed in `long` and then
bounds-checked to confirm they fit in an `int` before the narrowing cast, because
a pathological combination of the distance knobs could otherwise overflow the
`int` that indexes these grids. The check is behavior-neutral: the knobs are
already validated positive at entry, and the guard compiles out in release builds,
so the narrowed values are identical to a plain cast. If either dimension comes out
non-positive, the run returns an invalid-config status.

---

## 7. The FFT autocorrelation engine

Stage 5 is the divergence from the f-statistic tools and the heart of the tool. A
single backend call computes, for each combination of chromosome and output bin,
the correlation's sufficient statistics summed over *every* individual in the
target population. This is where the trillion-pair covariance is avoided: the
covariance at each separation is obtained as the inverse FFT of the squared
magnitude of the signal grid's FFT, so the cost grows with the number of grid cells
(times its logarithm), not with the number of position pairs, and not with the
number of individuals.

The call takes the source frequencies, the validity flags, the re-packed target
genotypes, each position's fine-grid cell, the per-chromosome cell extents, the two
grid dimensions, and the bin/fine-grid widths. It returns six per-(chromosome, bin)
moment arrays. Four of them feed the correlation directly (see section 8): a
count-like normalizer, the cross-moment that carries the decay signal, and two
second-moment sums that normalize it. The whole engine runs in native double
precision.

---

## 8. The correlation curve and leave-one-chromosome subtraction

Stage 6 (`fixjcorr`) exploits the fact that the moments are stored *per
chromosome*. Both the full-data curve and every leave-one-chromosome curve can be
built by cheap **subtraction** rather than recomputation:

- the total over all chromosomes is the sum of the per-chromosome moments;
- each leave-one-out curve is the total minus that one chromosome's contribution.

**The correlation at a bin.** From the four summed moments at a bin — call them
`S0` (the count-like normalizer), `S12` (the signal cross-moment), and `S11`/`S22`
(the two second moments) — the code forms the variances `S11/S0`, `S12/S0`,
`S22/S0` and returns `(S12/S0) / sqrt((S11/S0)·(S22/S0) + kCorrDenomFloor)`. A bin
whose count is below 0.5 (effectively empty) returns 0. This is the reference
tool's normalized correlation, its "datacol 3."

**The reported curve.** The full-data curve is surfaced as two parallel arrays: the
distance of each bin in centimorgans and the correlation there. Bin `k`'s center
distance is `(k + 1) · binsize` Morgans, converted to centimorgans by the shared
`kCentimorgansPerMorgan` factor. That bin-center-distance expression is defined once
and reused both here and by the fit-window edge test in the next stage, so the
curve's distance axis and the fit window can never drift apart. All computed bins
are emitted; the number of bins is already `round(maxdis / binsize)`, and the extra
padding bins the reference tool allocates internally are not created here.

---

## 9. The exponential fit and standard error

Stage 7 fits the decay curve and produces the final numbers.

**The fit window.** Only bins whose center distance is at least `lovalfit` and at
most `maxdis` enter the fit. Very short distances are excluded because ordinary
background linkage — not admixture — dominates there and would bias the date. All
curves (the full-data curve and every leave-one-out curve) share this one window,
so they can be fit together in a dense batch.

**The batched fit.** A single backend call fits `number of chromosomes + 1`
single-exponential curves at once: the full-data curve plus one leave-one-out curve
per chromosome. Each fit is an exponential-plus-optional-constant model (the affine
knob controls the constant term), solved with a coarse grid search, a refinement,
and a small two-by-two normal-equation solve. Like the repack and the FFT engine,
this runs through the compute backend — on the GPU, one thread per curve; on the CPU
reference, a bit-exact host oracle — so the host no longer runs the fit arithmetic
itself. If the full-data fit fails to converge, the run returns a rank-deficient
status with a not-a-number date.

**Assembling the standard error.** For each chromosome, the leave-one-out date is
that curve's fitted date, and its weight is the total position count minus that
chromosome's own position count. These per-chromosome dates and weights are surfaced
in the result and handed to `weight_jack` (section 3) to produce the jackknife point
estimate and its standard error. If the jackknife estimate is finite it becomes the
reported date; otherwise the full-data fitted date is used as a fallback. The
full-data fit's residual spread is reported as the goodness-of-fit number, and the
run finishes with an OK status.

---

## 10. Precision and where the work runs

The FFT autocorrelation and the per-position weight/residual computation run in
native double precision, and the host jackknife accumulates in `long double`. The
per-position weight is a difference of two allele frequencies — one of the
cancellation-sensitive quantities that always uses native double precision rather
than the faster emulated-double mode the matrix-heavy tools default to. The
emulated mode is acknowledged in the backend seam but is not used by this math.

Three heavy pieces cross into the compute backend rather than running on the host:
the target-genotype repack (section 5), the FFT moment computation (section 7), and
the batched exponential fit (section 9). Each has a GPU implementation and a
bit-exact CPU reference the GPU path is validated against. Everything the host
itself does — the grid construction, the leave-one-chromosome subtraction, and the
weighted jackknife — is small, exact arithmetic, never a loop over position pairs.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
[^dates]: **DATES** — admixture dating by ancestry-covariance decay. Chintalapati M, Patterson N, Moorjani P. *The spatiotemporal patterns of major human admixture events during the European Holocene.* eLife 2022;11:e77625.

# `dates_fit.hpp` reference

## 1. Purpose

`src/core/internal/dates_fit.hpp` holds the host (CPU) building blocks for
steppe's DATES computation[^dates] — the method that estimates how many generations ago
two ancestral populations mixed, by measuring how a correlation signal decays
along the genome. steppe reproduces its math for parity[^at2].

The file contains three things:

1. **A result struct** (`ExpFitHost`) — the outcome of one decay fit: an
   estimated date, an error estimate, and a success flag.
2. **Two math routines** (`linfit_2x2` and `fit_exp_decay`) — the numerical core
   that fits a decaying-exponential curve to the measured signal and reads a date
   off it.
3. **One data-shuffling routine** (`dates_repack_host`) — repacks the target
   population's genotypes onto the reduced set of SNPs the run actually keeps.

Two design choices shape the whole file:

**Header-only and inline.** Every routine is defined right here in the header
rather than in a separate `.cpp` file. This lets two different libraries both
call these routines without either library having to link against the other.
steppe's core library uses them to actually run DATES, and steppe's device
(GPU) library uses the exact same code as a reference oracle. Because both
libraries include this one header, there is a single shared copy of each routine
and no cross-library symbol dependency to manage.

**These CPU routines are the reference the GPU is checked against.** steppe's
real DATES runs on the GPU. The routines here are the plain, readable CPU version
of the same math, and the GPU kernels are validated by confirming they produce
the same answers. Two of the three routines are held to different standards of
"the same":

- The genotype repack is checked to be **bit-for-bit identical** — the CPU and
  GPU must produce exactly the same bytes, because it is pure integer bit
  manipulation with no rounding.
- The decay fit is checked to a **loose tolerance** (within about 2 percent) —
  it involves floating-point search and rounding, so an exact bit match is not
  expected, only a close date.

---

## 2. `ExpFitHost` — the fit result

`ExpFitHost` is the small struct returned by a single decay fit. It reports the
estimated admixture date, how well the curve fit, and whether the fit succeeded.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `date_gen` | `double` | not-a-number | The estimated admixture date in **generations**. Not-a-number until a fit succeeds. |
| `error_sd` | `double` | not-a-number | A residual standard deviation — the root-mean-square gap between the measured points and the fitted curve. Smaller means the curve fit the data more tightly. |
| `ok` | `bool` | `false` | Whether the fit produced a usable, positive date. Callers should check this before trusting the other two fields. |

A default-constructed `ExpFitHost` represents "no valid fit": both numbers are
not-a-number and `ok` is false. The fit routine returns exactly this whenever it
cannot produce an answer (too few data points, a bad step size, or a search that
never lands on a valid decay).

---

## 3. `linfit_2x2` — the inner linear solve

`linfit_2x2` is the innermost math step. Given a fixed per-bin decay factor `v`,
it finds the two numbers — an amplitude and (optionally) a constant offset — that
best fit the measured curve, and reports how good that best fit was.

### What it computes

The measured signal `y` is a list of values, one per distance bin. For a fixed
decay factor `v` between 0 and 1, the routine models the signal as an amplitude
times `v` raised to the bin number, plus an optional constant:

- **Amplitude term:** `co0 · v^i` for bin `i` — a geometric decay.
- **Constant term:** `c` — a flat baseline, present only in "affine" mode.

It solves for the `co0` and `c` that minimize the total squared gap between the
model and the measured `y`, and returns the **residual sum of squares** — that
total squared gap, *not* divided by the number of points. A smaller returned
value means a better fit for that particular `v`.

### The two modes

- **Affine mode** (`affine = true`): fit both the amplitude and the constant
  baseline. This solves a 2-by-2 system of equations (hence the name). If the
  system has no unique solution — its determinant is exactly zero — the routine
  returns not-a-number and sets both outputs to not-a-number.
- **Non-affine mode** (`affine = false`): force the baseline to zero and fit only
  the amplitude. This reduces to a single division; if the denominator is exactly
  zero, the amplitude is set to not-a-number.

### Why the internal math uses extended precision

The running sums inside this routine are accumulated in `long double` (extended
precision, wider than an ordinary double), not in ordinary `double`. This is a
deliberate carve-out. Fitting a decaying curve involves subtracting nearly-equal
large quantities, which is exactly the situation where ordinary double precision
loses accuracy to cancellation. Doing the accumulation in the wider type keeps
those cancellations from corrupting the result. The GPU version mirrors this with
its own extended-precision equivalent.

This routine is an exact reproduction of the corresponding solver that previously
lived inside DATES; the goal is to match its floating-point results, not to
re-derive the math a cleaner way.

---

## 4. `fit_exp_decay` — the decay fit and date search

`fit_exp_decay` is the top-level fit. It takes the whole measured correlation
curve and searches for the decay factor that best explains it, then converts that
decay factor into a date. It returns an `ExpFitHost`.

### Inputs

- `y` — the measured correlation curve, one value per distance bin.
- `step` — the width of one distance bin, measured in **Morgans** (a genetic-map
  unit). This is what turns a unit-less decay factor into a real date. Must be
  positive.
- `affine` — whether to fit a constant baseline in addition to the decaying term
  (passed straight through to the inner solve of section 3).

If there are fewer than 3 data points, or the step size is not positive, the
routine gives up immediately and returns the "no valid fit" result.

### How the search works — coarse to fine

The routine is looking for the single decay factor `v` (between 0 and 1) whose
best-fit curve is closest to the measured data. It finds it in two stages:

1. **Coarse grid scan.** It tries a fixed grid of **4,000** evenly-spaced
   candidate values of `v` across the open interval from 0 to 1, scoring each one
   with the inner solve from section 3 and remembering the best.
2. **Ternary refinement.** Starting from a narrow window around the best grid
   point, it runs **200** rounds of ternary search — repeatedly splitting the
   window into thirds and discarding the worse third — to home in on the exact
   best `v` between the grid points.

The final decay factor is the midpoint of the refined window.

### The positive-amplitude filter and its fallback

The coarse grid scan runs in two passes, and this two-pass structure is
deliberate DATES behavior, not an accident:

- **First pass:** only accept candidates whose fitted amplitude is positive. A
  genuine admixture-decay signal has a positive amplitude, so this filter keeps
  the search on physically meaningful solutions.
- **Fallback pass:** if the first pass found nothing at all (no positive-amplitude
  candidate anywhere on the grid), the scan runs again with the filter removed,
  accepting a best fit of any sign. Only if this second pass *also* finds nothing
  does the routine give up.

### Turning the decay factor into a date

Once the best decay factor `v` is found, the date is computed as
`-log(v) / step`. Because `v` is between 0 and 1, its logarithm is negative, so
the date comes out positive. Dividing by the bin width in Morgans converts the
per-bin decay rate into a rate per generation. The reported `error_sd` is the
square root of the best residual sum of squares divided by the number of data
points — a root-mean-square residual.

The fit is marked `ok` only if the resulting date is a finite, strictly positive
number. A zero, negative, or infinite date is treated as a failure.

Like the inner solver, this routine is an exact reproduction of the original
DATES fit, kept faithful to its floating-point operations so the results match.

### The named search constants

| Constant | Value | What it's for |
|---|---|---|
| coarse grid points | `4000` | How many evenly-spaced decay-factor candidates the coarse scan tries between 0 and 1. More points mean a finer starting search but more work. |
| ternary iterations | `200` | How many rounds of ternary search refine the best grid point. 200 rounds narrow the window far below floating-point resolution, so the refined answer is effectively exact. |
| minimum data points | `3` | Fewer than this and the fit is refused outright — a decay curve can't be meaningfully fit to one or two points. |

---

## 5. `dates_repack_host` — the genotype repack

`dates_repack_host` reshuffles the target population's genotype data so that it
lines up with the reduced set of SNPs the run actually keeps. It is pure integer
bit manipulation and is checked to be bit-for-bit identical to the GPU version.

### The problem it solves

Genotypes are stored two bits per SNP, packed tightly into bytes — four SNPs per
byte. A DATES run keeps only some of the original SNPs (the ones that pass its
filters). This routine reads the kept SNPs out of the full, original packing and
writes them into a new, dense packing where the kept SNPs sit consecutively with
no gaps.

### How it works

For each target individual, and for each position in the kept-SNP list:

1. Look up which original SNP index that kept position corresponds to.
2. Find that SNP's byte and its slot within the byte in the source data, and read
   out its two-bit code. The two-bit codes are read most-significant-first within
   each byte.
3. Compute the destination byte and slot for the new dense position, and write the
   two-bit code there by shifting it into place and OR-ing it in.

The most-significant-first ordering and the shift arithmetic are shared with the
rest of steppe's genotype handling (through the helpers it includes), so the
packing convention stays consistent everywhere.

### The pre-zeroing invariant

The destination buffer **must be zeroed before this routine is called.** The
routine writes each two-bit code by OR-ing it into the destination byte, which
only sets bits — it never clears them. If the destination still held leftover
data, the OR would merge the old and new bits and silently corrupt the result.
The routine relies on the caller having cleared the buffer first; it does not
clear the buffer itself.

### Parameters

| Parameter | Meaning |
|---|---|
| `src` | Pointer to the source (original) packed genotypes. |
| `src_bpr` | Bytes per record in the source — the stride from one individual to the next. |
| `kept_src` | For each kept-SNP position, the original SNP index it came from. |
| `M_kept` | How many kept SNPs there are (the length of the kept list). |
| `n_target` | How many target individuals to repack. |
| `dst_bpr` | Bytes per record in the destination — the stride in the new dense packing. |
| `dst` | Pointer to the destination buffer. **Must be pre-zeroed** (see the invariant above). |

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
[^dates]: **DATES** — admixture dating by ancestry-covariance decay. Chintalapati M, Patterson N, Moorjani P. *The spatiotemporal patterns of major human admixture events during the European Holocene.* eLife 2022;11:e77625.

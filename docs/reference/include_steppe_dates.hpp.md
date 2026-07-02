# `dates.hpp` reference

## 1. Purpose

`include/steppe/dates.hpp` is the public entry point for admixture *dating* — the
DATES tool. Admixture dating answers the question "how many generations ago did two
populations mix to produce a third?" Given a target (admixed) population and two
reference source populations, `run_dates` estimates the number of generations since
that admixture happened, along with a standard error and the raw decay curve the
estimate was fit to.

The header exposes three things:

1. **`DatesOptions`** — the tunable knobs for a run (bin widths, the fit window,
   the random seed). Every default matches the reference DATES parameter file that
   the validation goldens were produced with.
2. **`DatesResult`** — everything a single run produces: the date, its standard
   error, a goodness-of-fit number, the binned decay curve, and the per-chromosome
   leave-one-out replicates used for the error bar.
3. **`run_dates`** — the one function that does the work, reading a genotype triple
   off disk and returning a `DatesResult`.

Unlike the f-statistic tools, this one does **not** use the precomputed "f2 cache."
It reads the genotype files (`.geno`/`.snp`/`.ind`) directly through the same
front-end decoder those tools use, and then branches into an entirely separate
engine built around a Fast Fourier Transform (FFT). It never touches the f2 cache.

The header contains no CUDA (GPU) code. It uses only the C++ standard library, so it
is safe to include everywhere — the core library, the command-line tool, and the
language bindings — without forcing any of them to also pull in the GPU code. See
section 6 for how the GPU work is reached without appearing in this header.

---

## 2. How the dating statistic works

This section explains the math the engine implements, in plain terms. You do not
need it to *call* `run_dates`, but it explains what the numbers in `DatesResult`
mean and why the knobs in `DatesOptions` exist.

### The core idea

When two populations mix, each admixed chromosome is a patchwork of long stretches
("tracts") inherited from one source or the other. Every generation, recombination
chops these tracts into smaller pieces. So the amount of shared ancestry between two
nearby genetic positions falls off as they get farther apart on the chromosome, and
it falls off *faster* the more generations have passed. Concretely, the ancestry
covariance between a pair of positions separated by a genetic distance `d` decays
exponentially:

```
covariance(d)  ≈  A · exp(−lambda · d)  +  c
```

Here `lambda` (the decay rate) is directly the number of generations since
admixture, `A` is the amplitude, and `c` is a constant background term. Fitting this
curve and reading off `lambda` gives the date.

### Building the signal, per position

For each admixed individual (summed over the whole target population), the engine
computes a per-position "ancestry signal." Two ingredients go into it:

- A **weight** `wt` — the difference in allele frequency between the two source
  populations at that position. Positions where the two sources differ a lot carry
  more ancestry information.
- A **residual** `res` — the admixed individual's genotype at that position after
  subtracting off the best linear mix of the two sources. The engine first fits a
  single mixing proportion (regress the individual's dosage onto the two source
  frequencies), then keeps only what that mix fails to explain. That leftover is the
  ancestry signal for that position.

The product `signal = res · wt` is the per-position quantity whose covariance
between position pairs carries the decay.

### The FFT trick that makes it tractable

A genome has on the order of a million positions, so the number of *pairs* of
positions is on the order of a trillion. Forming that covariance pair by pair is
impossible. DATES avoids it with a Fourier-transform identity.

The per-position signals are scattered onto a fine, evenly spaced grid along each
chromosome's genetic map (the grid spacing is set by the `binsize` and `qbin` knobs
below). Three running totals are kept per grid cell: a count, the summed signal, and
the summed signal-squared. The covariance at a given separation `d` is then just the
*autocorrelation* of the signal grid at lag `d`, divided by the autocorrelation of
the count grid at the same lag. An autocorrelation of a whole array, at every lag at
once, is exactly what an FFT computes cheaply. So instead of a trillion-pair sum, the
cost is proportional to `G · log G` where `G` is the number of grid cells — and it
does not grow with the number of individuals. The trillion-pair object is never
actually built.

### What the reported curve is

The curve returned in the result is the *normalized* correlation — the signal
autocorrelation divided by the square root of the product of the two second-moment
autocorrelations, so that it is scaled to a comparable range across distances. In
DATES terms this is the curve's "datacol 3." The exponential-plus-constant form is
then fit over the distance window `[lovalfit, maxdis]` (see the options below), and
`lambda` is converted into a date in generations.

### The error bar

The standard error comes from a **leave-one-chromosome weighted block jackknife**:
the whole estimate is redone once per chromosome with that chromosome held out, and
the spread of those replicates (weighted by how many positions each chromosome
contributes) gives the standard error. The individual leave-one-out estimates and
their weights are surfaced in the result for inspection.

### Reference and validation

The engine is pinned to the reference DATES C source (the Moorjani-lab DATES tool,
version 750). The frozen validation case is the AADR packed dataset dating the
Puerto Rican population as a mix of European (CEU) and West African (YRI) sources,
which produces **9.742 generations with a standard error of 0.317**. The engine
implements the default population-delta weighting; there is no per-individual
likelihood-kernel variant.

---

## 3. `DatesOptions` — the run knobs

`DatesOptions` carries the settings that shape a run. Every default equals the
reference DATES parameter file the goldens were generated with, so a default-
constructed `DatesOptions` reproduces the reference behavior. Internally all
distances are in **Morgans**; the command-line tool exposes centimorgans (cM) in the
same places the reference does, and converts.

A note on units: 1 Morgan = 100 centimorgans. Both are measures of genetic distance
(expected number of recombination crossovers), not physical base pairs.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `binsize_morgans` | `double` | `0.001` (= 0.1 cM) | The width of one *output* bin of the decay curve, in Morgans. This is the resolution at which the final curve is reported. |
| `qbin` | `int` | `10` | The fine-grid refinement factor. The internal scatter grid the FFT runs on is finer than the output bins by this factor: `binsize / qbin` = `0.0001` Morgans (0.01 cM) per grid cell at the defaults. A finer internal grid gives a more accurate autocorrelation before the results are collapsed down into the output bins. |
| `maxdis_morgans` | `double` | `1.0` (= 100 cM) | The maximum genetic distance the curve spans, in Morgans. The curve is computed for separations from zero up to this value. |
| `lovalfit_cm` | `double` | `0.45` | The low edge of the *fit* window, in centimorgans. Bins closer together than this are excluded from the exponential fit, because at very short distances ordinary background linkage (not admixture) dominates and would bias the date. The high edge of the fit window is `maxdis`. |
| `affine` | `bool` | `true` | Whether to include the constant background term `c` in the fit (`A · exp(−lambda · d) + c`). On by default, matching the reference. With it off, the fit is a pure exponential with no baseline. |
| `seed` | `long` | `77` | Random seed for the fitter's multi-start search. The final date is deterministic given a fixed curve — the numerical refinement converges to the same minimum regardless of seed — but the seed pins the exact multi-start path so results are bit-comparable with the reference tool. |

---

## 4. `DatesResult` — the run output

`DatesResult` holds everything one call to `run_dates` produces.

### The headline numbers

| Field | Type | Meaning |
|---|---|---|
| `date_gen` | `double` | The estimate: generations since admixture. This is the weighted-jackknife point estimate. |
| `se` | `double` | The standard error of `date_gen`, from the leave-one-chromosome weighted block jackknife described in section 2. |
| `fit_error_sd` | `double` | Goodness-of-fit: the standard deviation of the fit residuals over the fit window (the reference tool's "error sd"). Smaller means the observed curve tracks the fitted exponential more cleanly. A pass/fail quality gate built on top of this lives in the calling code, not here. |

### The decay curve

The binned curve the date was fit to, as two parallel arrays of equal length (one
entry per output bin):

| Field | Type | Meaning |
|---|---|---|
| `curve_cm` | `vector<double>` | The distance of each bin, in centimorgans. |
| `curve_corr` | `vector<double>` | The normalized correlation at that distance (the "datacol 3" value from section 2). |

`curve_cm[k]` and `curve_corr[k]` together are one point on the curve.

### The jackknife replicates

The raw inputs to the standard-error calculation, surfaced so a caller can inspect
or independently reproduce the error bar. Both arrays have one entry per chromosome
present in the data:

| Field | Type | Meaning |
|---|---|---|
| `loo_date_gen` | `vector<double>` | The date estimate with each single chromosome left out (the leave-one-out replicates). |
| `loo_weight` | `vector<double>` | The weight paired with each replicate — the number of positions in all the *other* chromosomes (the total position count minus that chromosome's count). |

### Outcome and precision tags

| Field | Type | Default | Meaning |
|---|---|---|---|
| `status` | `Status` | `Status::Ok` | The per-call outcome. A *degenerate* run — one where the curve shows no real decay, or the fit fails to converge — is reported here as a non-`Ok` status rather than by throwing an exception, so a batch of runs can record the failure and keep going. On a degenerate curve the `date_gen` comes back as NaN (not-a-number). Note that this is distinct from an input/output fault (a missing or unreadable file), which *does* throw — see section 5. |
| `precision_tag` | `Precision::Kind` | `Fp64` | Which arithmetic produced the result. Always native double precision (`Fp64`): the FFT autocorrelation runs in native double, and the weight/residual step is one of the cancellation-sensitive computations that always uses native double rather than the faster emulated mode. |

---

## 5. `run_dates` — the API

```cpp
[[nodiscard]] DatesResult run_dates(
    const std::string& geno, const std::string& snp, const std::string& ind,
    const std::string& target,
    const std::string& source1, const std::string& source2,
    const DatesOptions& opts, device::Resources& resources);
```

One call performs a complete admixture-dating run.

### Parameters

- **`geno`, `snp`, `ind`** — the three file paths of the genotype triple, in the
  EIGENSTRAT / packed-ancestrymap layout. `geno` holds the genotype matrix, `snp`
  the per-position metadata, `ind` the individual/population labels. The reader
  accepts the transposed-genotype (TGENO) or packed GENO encodings. **Important
  requirement:** the `.snp` file must carry a real genetic map in its genetic-
  position column. Dating fundamentally needs centimorgan positions; a file whose
  genetic-position column is all zeros (common for data converted from VCF or PLINK)
  cannot be dated.
- **`target`** — the label of the admixed population to date. The covariance is
  summed over *every* individual in this population.
- **`source1`, `source2`** — the labels of the two reference (ancestral) source
  populations. They enter the math only through the frequency-difference weight
  `wt = freq(source1) − freq(source2)` from section 2. Their order sets the sign of
  that weight but does not change the date.
- **`opts`** — the knobs from section 3. Passing a default-constructed
  `DatesOptions` reproduces the reference defaults.
- **`resources`** — the handle through which the GPU work is dispatched (see
  section 6). It is a forward-declared type, so this header never needs the GPU
  headers.

### Return value and error handling

Returns a `DatesResult` (section 4). The `[[nodiscard]]` marker means the compiler
warns if a caller ignores the returned result.

There are two distinct failure channels, and it matters which is which:

- A **degenerate run** — a flat curve or a fit that will not converge — is *not* an
  error. It comes back normally as a `DatesResult` whose `status` field is non-`Ok`
  and whose `date_gen` is NaN. This lets a caller sweep many population triples and
  simply record the ones that did not produce a clean date.
- An **input/output fault** — a missing file, an unreadable file, a `.snp` with no
  genetic map — *throws an exception*. This is meant to propagate up so the command-
  line tool can turn it into a non-zero exit code.

---

## 6. The CUDA-free contract

This header is deliberately free of any GPU code, and that is a design rule, not an
accident. It includes only standard C++.

The heavy computation — the per-position weight and residual, scattering onto the
genetic-map grid, and the batched FFT autocorrelations — all runs on the GPU. But
none of that appears here. It is reached through a compute-backend seam: the actual
GPU kernels live in separate `.cu` source files that are private to the GPU library,
and this header only references them indirectly through the forward-declared
`device::Resources` type. Because `device::Resources` is only *declared* (its real
definition lives in the GPU layer), including `dates.hpp` never drags in any CUDA
headers.

The same seam also provides a pure-CPU reference implementation (an FFT-free
version) that the GPU path is validated against. The command-line tool and the
language bindings reach the GPU exclusively through this seam — they never see GPU
code directly, exactly the same arrangement the other statistics tools use.

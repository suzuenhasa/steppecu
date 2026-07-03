# `cmd_dates.hpp` reference

## 1. Purpose

`src/app/cmd_dates.hpp` declares the entry point for the `steppe dates`
command-line subcommand: a single function, `run_dates_command`. The command
estimates **when two populations mixed** — how many generations ago an admixture
event happened — for one present-day admixed population.

The method is admixture dating by ancestry-covariance decay, the same approach
used by the ALDER and DATES tools (Loh et al. 2013; Chintalapati, Patterson, and
Moorjani 2022). The intuition: when two ancestral populations mix, each admixed
chromosome starts out as long alternating blocks of the two ancestries. Every
generation of recombination chops those blocks shorter. So the rate at which
ancestry correlation falls off as you move along the genome — measured against
genetic distance — tells you how much time (in generations) has passed since the
mixture. A slow decay means a recent event; a fast decay means an old one.

The header is deliberately small — it exposes only the one function. Everything
about how the estimate is actually computed lives behind the `run_dates` seam
(section 4), so callers of this command see only its inputs, its output, and its
exit code.

---

## 2. Inputs

`run_dates_command` reads everything it needs from the frozen `RunConfig` it is
handed. Three pieces of input are required, and the command refuses to start
(returning an invalid-configuration exit code) if any is missing or malformed:

| Input | Source flag | Meaning |
|---|---|---|
| Genotype data | `--prefix` | A path prefix naming a genotype triple. The command expands it in a format-aware way: for the EIGENSTRAT family it looks for `PREFIX.geno` / `PREFIX.snp` / `PREFIX.ind`, and for PLINK it looks for `PREFIX.bed` / `PREFIX.bim` / `PREFIX.fam`. |
| The admixed population | `--target` | The single present-day population whose admixture date is being estimated. |
| The two ancestral sources | `--left` | **Exactly two** population labels — the two ancestral groups whose mixture formed the target. The command rejects any other count with a clear error. |

### Why the two sources, and why order does not matter

The dating math weights each SNP by the allele-frequency difference between the
two reference sources, `freq(source1) - freq(source2)`. That weighting is what
turns a plain along-genome correlation into a signal specifically about the
admixture between *these two* ancestries.

Swapping the two sources flips the sign of that per-SNP weight, but it does
**not** change the reported date. The decay *rate* — which is what the date is
read from — is symmetric in the source order; only the amplitude of the fitted
curve changes sign. So `--left A B` and `--left B A` produce the same date.

---

## 3. What it reports

The command produces a small one-row result describing the admixture event:

- **Date, in generations** — the estimated number of generations since the two
  sources mixed.
- **Standard error** — the uncertainty on that date, computed with a
  leave-one-chromosome block jackknife. This drops each chromosome in turn,
  re-estimates the date without it, and forms the standard error from how much
  the estimate moves across those drops. It is the honest measure of how
  well-pinned the date is.
- A fit-quality figure and a status label round out the row.

The row is written in the caller's chosen format (CSV, TSV, or JSON). A missing
or undefined number is written as `NA` (or `null` in JSON) rather than as a
misleading zero — see the degenerate-run behavior in section 6.

---

## 4. The dating engine — and what it deliberately is not

All of the real computation happens inside `run_dates`, which this header reaches
without ever naming it in its own declarations. Two design rules about that
engine are worth stating up front, because both are easy to get wrong:

- **It uses a Fourier (cuFFT) autocorrelation to measure the decay curve.**
  Computing how ancestry covariance falls off with genetic distance is, done
  naively, a comparison of every pair of SNPs — quadratic in the number of SNPs.
  Instead the engine casts that as an autocorrelation and evaluates it with a
  fast Fourier transform on the GPU, which gets the whole decay curve in roughly
  *M log M* work instead of *M²*.
- **It is NOT the f2 cache, and it is NOT a host-side all-pairs loop.** The
  block-jackknife allele-frequency-covariance store (the "f2 cache") that most of
  steppe's other statistics share is the wrong tool here and must not be used for
  dating. Likewise, a host-side O(M²) loop over SNP pairs would be correct but
  hopelessly slow and must never be the implementation. The dating path is its
  own GPU pipeline.

---

## 5. Header layering: no CUDA here

This header includes only the run-configuration header and pulls in **no CUDA
header at all**. The command is plain C++20 living in the application layer.

That is intentional and load-bearing: the GPU is reached only *indirectly*,
through the `run_dates` seam, whose interface is itself CUDA-free (it takes an
opaque device-resources handle rather than any CUDA type). Keeping this header
and the application translation units free of CUDA means they compile with an
ordinary host compiler, and the GPU code stays quarantined behind one clean
boundary. Anyone extending the command should preserve this: do not add CUDA
includes to the app layer; call through the seam.

---

## 6. The `run_dates_command` contract

```cpp
int run_dates_command(const steppe::config::RunConfig& config);
```

The function takes the already-frozen `RunConfig` and returns a **process exit
code**. The contract distinguishes real faults from an honest "no answer":

- **`0` on success** — the estimate was computed and written.
- **Nonzero only on a genuine fault.** These include: a missing or malformed
  input (no `--prefix`, no `--target`, or a `--left` that is not exactly two
  sources) → an invalid-configuration code; no CUDA-capable GPU present → a
  runtime-error code (steppe is a GPU product and requires a device); an
  input-reading or device error thrown during the run → an I/O-error code; and a
  torn or short write of the output → an I/O-error code (so a truncated file is
  never mistaken for success).
- **A degenerate run is not a fault.** If there is simply no measurable
  ancestry-covariance decay to fit — so no meaningful date exists — the command
  does **not** fail. It writes a normal result row whose date and standard error
  are `NaN` (rendered as `NA` / `null`) and exits `0`. This is the
  "record-and-continue" convention: a batch that dates many targets records the
  empty result for this one and keeps going instead of aborting the whole run.
  It matches how the other genotype-path commands behave.

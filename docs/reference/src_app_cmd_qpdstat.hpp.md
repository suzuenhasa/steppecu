# `cmd_qpdstat.hpp` reference

## 1. Purpose

`src/app/cmd_qpdstat.hpp` declares the entry point for the `steppe qpdstat`
command-line subcommand. `qpdstat` computes the D-statistic (and, on the f2
path, the closely related f4 statistic) for groups of four populations. A
D-statistic asks, for four populations arranged as a quartet, whether two of
them share more alleles with a third than expected — the classic test for gene
flow between populations.

The header is intentionally tiny: a single function declaration plus its
contract. All of the substance is in what that one command *does*, which splits
into two distinct execution paths depending on the input the user supplies. The
command adds no new statistical engine of its own — each path reuses machinery
that is already built and validated elsewhere.

The numbers, signs, z-scores, and p-values this command reports are meant to
match the reference implementation for the same inputs[^at2].

---

## 2. The two paths: `--f2-dir` reports f4, `--prefix` reports normalized D

Which statistic `qpdstat` computes depends on which input the user points it at.

### The f2-directory path (reports f4)

When the command is given a precomputed f2 blocks directory (`--f2-dir`), it
reports the **f4 statistic**. This is deliberate and follows the reference
convention[^at2]: when the reference `qpdstat` runs against an f2-data directory
rather than raw genotypes, its result *is* f4. The "D mode" toggle that would
normalize the statistic is a no-op once you are working from f2 blocks, because
normalization needs the per-SNP genotypes that an f2 directory has already
summarized away. So on this path, asking for a D-statistic and asking for f4
give byte-identical answers.

Because of that equivalence, this path introduces **no new compute and no new
output formatter**. It reuses the existing f4 machinery verbatim:

- the batched f4 point estimate together with its block-jackknife standard error
  (the standard error taken from the diagonal of the jackknife — the per-quartet
  variance, not a full cross-quartet covariance), and
- the existing f4 result emitter, which produces the columns described in
  section 4.

### The genotype path (reports the normalized D magnitude)

When the command is instead given a genotype file prefix (`--prefix`), it reads
the raw genotype triple — the `.geno`, `.snp`, and `.ind` files that share that
prefix — and computes the **normalized D-statistic**. Normalized D is the
signed *magnitude* form: for each SNP it forms a numerator and a denominator,
averages each across SNPs, and divides the mean numerator by the mean
denominator, with the uncertainty coming from the same block-jackknife method.
This path exists precisely because normalization requires the per-SNP genotypes
that the f2-directory path does not have.

Even though this path computes a different number, it emits it through the *same*
result table as the f4 path, so both paths present an identical set of columns
and can be read the same way.

### The rule of thumb

Point `qpdstat` at an f2 directory and you get f4; point it at a genotype prefix
and you get normalized D. Exactly one of the two inputs is supplied per run.

---

## 3. How quartets are specified

Whichever path runs, the four-population quartets to evaluate can be supplied in
either of two ways, and the command accepts exactly one of them per run:

- **Row-aligned columns** — four parallel lists given as `--pop1`, `--pop2`,
  `--pop3`, and `--pop4`. Row *i* of each list forms one quartet, so the four
  lists must be the same length. Each row is one quadruple of population names.
- **A single-quartet convenience form** — `--pops A,B,C,D`, exactly four names,
  which is shorthand for one quartet. This exists so a quick one-off test does
  not require four separate flags.

If the four column lists are supplied but their lengths do not line up, that is a
configuration fault (see section 5), not an empty result.

---

## 4. Output schema and the D-statistic conventions

Both paths emit one row per quartet with the same columns:

`pop1, pop2, pop3, pop4, est, se, z, p`

— the four population names, the estimate, its standard error, the z-score, and
the p-value.

The z-score and p-value follow the reference D-statistic convention exactly[^at2]:

- **z is the estimate divided by its standard error** (`z = est / se`), and
- **p is the two-sided normal tail** — `p = 2 × (1 − Φ(|z|))`, where Φ is the
  standard normal cumulative distribution. This is the probability of seeing a
  z-score at least this large in magnitude if the true value were zero.

The **sign** of the estimate carries meaning (it says which pair shares more
alleles), so it is preserved and follows the same reference sign convention.
Reusing one emitter for both paths guarantees the f4 result and the
normalized-D result cannot drift apart in their column layout or their
sign/z/p conventions.

---

## 5. Layering: plain C++20, no CUDA in this file

This header — and the command implementation behind it — is deliberately plain
C++20 with no CUDA header included. It is "app-only" code: it reaches the GPU
**only** through a small set of CUDA-free seams — the resource builder, the
device-side f2-blocks type, and the `run_f4` entry point — exactly the way the
sibling `f4` command does. Keeping CUDA out of the app layer is an enforced
boundary in the codebase (checked mechanically), and this file stays on the
correct side of it by never naming a GPU type directly.

That shared shape is also *why* the f2-directory path can reuse the f4 engine
and emitter wholesale rather than reimplementing anything: the two commands
enter the GPU through the same seams.

---

## 6. The `run_qpdstat_command` contract

```cpp
[[nodiscard]] int run_qpdstat_command(const config::RunConfig& config);
```

This is the single function the header declares. It runs `qpdstat` over an
already-frozen `RunConfig` and returns the process exit code. The `--f2-dir`
input dispatches to the f4 path; the `--prefix` input dispatches to the
genotype-path normalized-D. `[[nodiscard]]` marks the return value so a caller
cannot accidentally ignore whether the command succeeded.

- **It owns its own console output.** The command prints to stdout and stderr
  itself; the underlying library never prints. This keeps all user-facing text
  in the command layer.
- **The return value is a process exit code** drawn from steppe's `CliExitCode`
  set.

### Which outcomes exit zero, and which don't

The important, non-obvious part of the contract is how failures split into two
categories:

- **Domain outcomes exit 0 (record-and-continue).** If the statistic simply
  cannot be computed cleanly for some quartet — for example a degenerate
  quadruple (a population repeated, or a near-zero denominator on the normalized
  path) — that is treated as a legitimate result, not a crash. The condition
  rides on the result's `status` field and shows up as a `NaN` in that row's
  estimate, and the process still exits 0. One bad quartet should not abort the
  whole run.
- **Faults return a nonzero code.** Only genuine faults produce a nonzero exit:
  invalid population names, a missing or bad f2 directory, missing genotype
  files on the `--prefix` path, the GPU running out of memory, and file, format,
  or CUDA-runtime errors.

The practical rule for a caller: a nonzero exit means the command could not run,
while a zero exit means it ran and you should read each row's `status` (and its
estimate and standard error) to learn what happened for that quartet.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

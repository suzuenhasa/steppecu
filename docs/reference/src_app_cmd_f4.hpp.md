# `cmd_f4.hpp` reference

## 1. Purpose

`src/app/cmd_f4.hpp` declares the entry point for the `steppe f4`
command-line subcommand. `f4` computes the standalone f4 statistic: for a
group of four populations (a "quartet") it returns a single number that
summarizes how allele-frequency differences between two pairs of populations
covary. It is the plain statistic on its own — there is no model being fitted.

In steppe's family of commands, `f4` is the sibling of the `qpwave` command,
not a variant of `qpAdm`. Concretely that means it has:

- **no target population** being modeled,
- **no iterative fitting** step, and
- **no rank test**.

For each quartet it produces two things: the f4 **point estimate** and its
**standard error**. Both use a weighted block jackknife — the same
uncertainty method the rest of steppe uses, where the genome is cut into
blocks, the statistic is recomputed leaving each block out in turn, and the
spread of those recomputations gives the standard error. The standard error
here comes specifically from the diagonal of that jackknife (the per-quartet
variance), not a full cross-quartet covariance. The numbers are meant to match
what ADMIXTOOLS 2 produces for the same statistic.

The header itself is tiny — one function declaration plus its contract. The
substance is the contract and the surrounding rules, described below.

---

## 2. What the command does, step by step

The GPU path is the real deliverable; there is no separate CPU-only mode for a
user to select. Running `steppe f4` walks through a fixed pipeline:

1. **Read the f2 blocks directory.** The command consumes a precomputed
   directory of per-block f2 statistics (the same cached artifact the other fit
   commands read), rather than reading raw genotypes itself.
2. **Resolve population names to indices.** The names the user supplies on the
   command line are looked up against the `pops.txt` that ships with the f2
   blocks directory, turning each name into the row/column index it occupies in
   the f2 data.
3. **Build the GPU resources** from the device configuration (which GPU, the
   precision policy, and so on).
4. **Upload the f2 blocks to the device.**
5. **Run f4** over the resolved quartets against the on-device f2 blocks.
6. **Emit the result table** as tidy CSV, TSV, or JSON.

### Output schema

The emitted table has one row per quartet with these columns:

`pop1, pop2, pop3, pop4, est, se, z, p`

— the four population names, the estimate, its standard error, the z-score
(estimate divided by standard error), and the corresponding p-value. This is
the same schema the reference f4 output uses, so the two can be compared
directly.

---

## 3. How quartets are specified

A user can supply the quartets to compute in either of two ways, and the
command accepts exactly one of them per run:

- **Row-aligned columns** — four parallel lists given as `--pop1`, `--pop2`,
  `--pop3`, and `--pop4`. Row *i* of each list forms one quartet, so the four
  lists must be the same length. This mirrors ADMIXTOOLS 2's `f4` call with
  `comb = FALSE`, where the four population columns are lined up row by row
  rather than combined into every possible grouping.
- **A single quartet convenience form** — `--pops A,B,C,D`, exactly four names,
  which is just shorthand for one quartet. This exists so a quick one-off f4
  doesn't require four separate flags.

If the four column lists are given but their lengths don't match, that is a
configuration error (see section 5), not an empty result.

---

## 4. Layering: plain C++20, no CUDA in this file

This header — and the command implementation behind it — is deliberately
written in plain C++20 with no CUDA header included. It is "app-only" code: it
reaches the GPU **only** through a set of CUDA-free seams (the resource
builder, the f2-blocks uploader, and the `run_f4` call), exactly the way the
`qpadm` and `qpwave` commands do. Keeping CUDA out of the app layer is an
enforced boundary in the codebase, and this file stays on the correct side of
it by never touching a GPU type directly.

Because of that shared shape, the command **reuses** most of the existing
machinery rather than reimplementing it:

- the f2-directory loader from the `qpadm` command,
- the population-name resolver,
- the build-and-upload chain that gets f2 data onto the device, and
- the result-formatting primitives (used through the f4 result emitter).

The **only** genuinely new logic this command adds is resolving the quartets
and emitting the f4 table. Everything else is borrowed from paths that are
already validated.

---

## 5. The `run_f4_command` contract

```cpp
[[nodiscard]] int run_f4_command(const config::RunConfig& config);
```

This is the single function the header declares. It runs the standalone f4
over an already-frozen `RunConfig` and returns the process exit code.

- **It owns its own console output.** The command prints to stdout and stderr
  itself; the underlying library never prints. This keeps all user-facing text
  in the command layer.
- **The return value is a process exit code**, drawn from steppe's
  `CliExitCode` set. `[[nodiscard]]` marks it so a caller can't accidentally
  ignore whether the command succeeded.

### Which outcomes exit zero, and which don't

The important, non-obvious part of the contract is how failures are split into
two categories:

- **Domain outcomes exit 0 (record-and-continue).** If the statistic simply
  can't be computed cleanly for some quartet — for example, a covariance matrix
  over the batch of quartets that isn't symmetric-positive-definite — that is
  treated as a legitimate result, not a crash. The condition is recorded on the
  result's `status` field and reported in the output, and the process still
  exits 0. The idea is that one bad quartet shouldn't abort the whole run.
- **Faults return a nonzero code.** Only genuine faults produce a nonzero exit:
  an invalid configuration (bad population names, a missing or bad f2 directory,
  or mismatched quartet column lengths, all surfaced as `InvalidConfig`), the
  GPU running out of memory (`DeviceOom`), and file, format, or CUDA-runtime
  errors.

The practical rule for a caller: a nonzero exit means the command couldn't run,
while a zero exit means it ran and you should read the per-row `status` (and the
estimate/standard error) to learn what actually happened for each quartet.

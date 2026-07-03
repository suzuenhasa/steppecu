# `cmd_f3.hpp` reference

## 1. Purpose

`src/app/cmd_f3.hpp` declares the entry point for the `steppe f3`
command-line subcommand — the standalone f3 statistic. It exposes exactly
one function, `run_f3_command`, and nothing else. The header is deliberately
tiny; all the work lives in the matching `.cpp`.

f3 is the close sibling of the f4 statistic, not a variant of qpAdm. That
distinction matters: f3 has **no** target population being modeled, **no**
iterative fitting step, and **no** rank test. It simply computes, for each
requested triple of populations, a single f3 number (the point estimate) plus
an estimate of that number's uncertainty (the standard error). The math it
uses — a weighted block-jackknife f3 point estimate with a jackknife-derived
standard error — reproduces what ADMIXTOOLS 2 produces for the same input.

## 2. What the command computes

For every triple of populations it is given, the command produces one row of
output with these columns:

| Column | Meaning |
|---|---|
| `pop1`, `pop2`, `pop3` | The three population names of the triple. |
| `est` | The f3 point estimate for that triple. |
| `se` | The standard error of the estimate, from the block jackknife. |
| `z` | The z-score (estimate divided by its standard error). |
| `p` | The p-value derived from the z-score. |

This column layout is the fixed output schema; the reference result file that
the command is validated against uses exactly these seven columns in this
order. The table can be emitted as CSV, TSV, or JSON — all three are the same
rows in a tidy, one-row-per-triple layout.

The standard error is the jackknife **diagonal** only: each triple's own
uncertainty is reported, and the command does not build or report a full
covariance matrix across triples.

## 3. How triples are specified

There are two ways a caller names the triples, and the command accepts either.

**Row-aligned columns.** The caller supplies three equal-length lists via
`--pop1`, `--pop2`, and `--pop3`. The lists are read in lockstep: triple
number *k* is `(pop1[k], pop2[k], pop3[k])`. This is the "give me exactly
these specific triples, one per row" style, and it matches ADMIXTOOLS 2's f3
call with combination-generation turned off (`comb = FALSE`). The three lists
must be the same length; a length mismatch is a configuration error.

The ordering within a triple follows the f3 convention `f3(pop1; pop2, pop3)`:
`pop1` is the population being tested (often labeled C), and `pop2`/`pop3` are
the two reference populations (often labeled A and B).

**The `--pops` convenience form.** As a shortcut, the caller can pass a flat
list of names via `--pops`. Exactly three names means one triple. Any multiple
of three means several triples, taken three at a time in order. This is the
quick way to ask for one or a handful of triples without filling in three
separate column flags.

## 4. The GPU execution path

The GPU path is the deliverable — this command is GPU-only. When it runs it
performs these steps in order:

1. Read the directory of precomputed f2 blocks (the per-genome-block building
   material the statistic is computed from).
2. Resolve the population *names* the user supplied into the *indices* those
   populations occupy in the f2 data, using the dataset's `pops.txt`.
3. Build the GPU resources from the device configuration.
4. Upload the f2 blocks to the GPU.
5. Run the f3 computation on the GPU over all the requested triples at once.
6. Emit the result table (see section 2) in the chosen format.

The command reuses machinery that already exists for the sibling commands: the
f2-directory loader and the name-to-index resolver come from the qpWave
command, and the build-and-upload chain and the result-formatting primitives
are shared as well. The only genuinely new logic this command adds is
resolving the triples (section 3) and emitting the f3-shaped result table.

## 5. Layering: plain C++20, no CUDA in this file

This header (and its `.cpp`) is application-layer code written in plain C++20.
It contains **no** CUDA and includes **no** CUDA headers. It reaches the GPU
only through CUDA-free seams — the resource builder, the f2-block uploader, and
the f3 runner — each of which is an ordinary C++ interface that hides the GPU
code behind it. This is the same arrangement the f4 command uses, and it is
enforced so that the command layer stays free of GPU compilation concerns. The
single include this header needs is the run configuration type.

## 6. The `run_f3_command` function

```cpp
[[nodiscard]] int run_f3_command(const config::RunConfig& config);
```

The function takes the already-frozen run configuration and returns the process
exit code. It owns its own printing to standard output and standard error — the
underlying library never prints anything itself, so all user-facing text
originates here.

The return value encodes an important distinction between two kinds of
outcomes:

- **A domain outcome returns exit code 0.** If the statistic itself runs into a
  numerical condition for some triples — for example, a covariance matrix over
  the block batch that is not positive-definite — that is treated as a normal,
  recorded result, not a crash. The condition is carried on the result's status
  field, the run continues, and the process still exits 0. This is the
  record-and-continue policy: a data-driven condition is data, not a failure.

- **A fault returns a nonzero code.** Genuine failures return nonzero. These
  include an invalid configuration (bad population names, a missing or bad f2
  directory, or mismatched triple-column lengths), running out of GPU memory,
  and file, format, or CUDA-runtime errors.

The `[[nodiscard]]` marking means callers are not allowed to silently ignore
the returned exit code.

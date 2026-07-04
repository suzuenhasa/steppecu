# `cmd_f4ratio.hpp` reference

## 1. Purpose

`src/app/cmd_f4ratio.hpp` declares the entry point for the `steppe f4-ratio`
command-line subcommand — the standalone f4-ratio statistic. It exposes exactly
one function, `run_f4ratio_command`, and nothing else. The header is
deliberately tiny; all the work lives in the matching `.cpp`.

f4-ratio is a close sibling of the f4 and f3 statistics, not a variant of qpAdm.
That distinction matters: f4-ratio has **no** target population being modeled,
**no** iterative fitting step, and **no** rank test. For each requested group of
five populations it computes a single admixture-proportion number (the point
estimate, written alpha) plus an estimate of that number's uncertainty (the
standard error). The math it uses reproduces what the `qpf4ratio` function
produces for the same input[^at2].

## 2. What the command computes

f4-ratio estimates an admixture proportion as the ratio of two f4 statistics. For
each group of five populations `(pop1, pop2, pop3, pop4, pop5)` the command
computes:

```
alpha = f4(pop1, pop2; pop3, pop4) / f4(pop1, pop2; pop5, pop4)
```

The two f4 values share `pop1`, `pop2`, and `pop4`; only the third slot differs
(`pop3` in the numerator, `pop5` in the denominator). The result is a single
proportion per group.

For every group it is given, the command produces one row of output with these
columns:

| Column | Meaning |
|---|---|
| `pop1`, `pop2`, `pop3`, `pop4`, `pop5` | The five population names of the group. |
| `alpha` | The f4-ratio point estimate (the admixture proportion). |
| `se` | The standard error of that estimate, from the block jackknife. |
| `z` | The z-score (`alpha` divided by its standard error). |

This column layout is the fixed output schema. The reference result file that
the command is validated against (`golden_fit0_f4ratio_readf2.csv`) uses exactly
these eight columns in this order. The table can be emitted as CSV, TSV, or
JSON — all three are the same rows in a tidy, one-row-per-group layout.

Note there is **no `p` column** here, unlike the f3 command's output. f4-ratio
reports the estimate, its standard error, and the z-score, and stops there.

### The standard error is a jackknife of the ratio

The standard error is not derived from the two f4 statistics separately and then
combined. Instead the *ratio itself* is jackknifed: the whole ratio is
recomputed once for each block left out, and the standard error is the jackknife
spread of those recomputed ratios. This is the correct way to get the
uncertainty of a ratio (the numerator and denominator move together across
blocks), and it is what the reference implementation does[^at2]. It is a per-group diagonal only — each
group's own uncertainty is reported, and the command does not build or report a
full covariance matrix across groups.

## 3. How the five-population groups are specified

There are two ways a caller names the groups, and the command accepts either.

**Row-aligned columns.** The caller supplies five equal-length lists via
`--pop1`, `--pop2`, `--pop3`, `--pop4`, and `--pop5`. The lists are read in
lockstep: group number *k* is `(pop1[k], pop2[k], pop3[k], pop4[k], pop5[k])`.
This is the "give me exactly these specific groups, one per row" style, and it
matches the `qpf4ratio` call[^at2]. The five lists must be the same length;
a length mismatch is a configuration error.

**The `--pops` convenience form.** As a shortcut, the caller can pass a flat list
of names via `--pops`. Exactly five names means one group. Any multiple of five
means several groups, taken five at a time in order. This is the quick way to ask
for one or a handful of groups without filling in five separate column flags.

## 4. The GPU execution path

The GPU path is the deliverable — this command is GPU-only. When it runs it
performs these steps in order:

1. Read the directory of precomputed f2 blocks (the per-genome-block building
   material the statistic is computed from).
2. Resolve the population *names* the user supplied into the *indices* those
   populations occupy in the f2 data, using the dataset's `pops.txt`.
3. Build the GPU resources from the device configuration.
4. Upload the f2 blocks to the GPU.
5. Run the f4-ratio computation on the GPU over all the requested groups at once.
6. Emit the result table (see section 2) in the chosen format.

The command reuses machinery that already exists for the sibling commands: the
f2-directory loader and the name-to-index resolver come from the qpWave command,
and the build-and-upload chain and the result-formatting primitives are shared as
well. The only genuinely new logic this command adds is resolving the
five-population groups (section 3) and emitting the f4-ratio-shaped result table.

## 5. Layering: plain C++20, no CUDA in this file

This header (and its `.cpp`) is application-layer code written in plain C++20. It
contains **no** CUDA and includes **no** CUDA headers. It reaches the GPU only
through CUDA-free seams — the resource builder, the f2-block uploader, and the
f4-ratio runner — each of which is an ordinary C++ interface that hides the GPU
code behind it. This is the same arrangement the f4 and f3 commands use, and it
is enforced so that the command layer stays free of GPU compilation concerns. The
single include this header needs is the run configuration type.

## 6. The `run_f4ratio_command` function

```cpp
[[nodiscard]] int run_f4ratio_command(const config::RunConfig& config);
```

The function takes the already-frozen run configuration and returns the process
exit code. It owns its own printing to standard output and standard error — the
underlying library never prints anything itself, so all user-facing text
originates here.

The return value encodes an important distinction between two kinds of outcomes:

- **A domain outcome returns exit code 0.** If the statistic itself runs into a
  numerical condition for some groups — for example, a degenerate block batch —
  that is treated as a normal, recorded result, not a crash. The condition is
  carried on the result's status field and, for an affected row, surfaced as a
  not-a-number value in that row's estimate; the run continues and the process
  still exits 0. This is the record-and-continue policy: a data-driven condition
  is data, not a failure.

- **A fault returns a nonzero code.** Genuine failures return nonzero. These
  include an invalid configuration (bad population names, a missing or bad f2
  directory, or mismatched group-column lengths), running out of GPU memory, and
  file, format, or CUDA-runtime errors.

The `[[nodiscard]]` marking means callers are not allowed to silently ignore the
returned exit code.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

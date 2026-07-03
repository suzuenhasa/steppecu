# `cmd_f4ratio.cpp` reference

## 1. Purpose

`src/app/cmd_f4ratio.cpp` implements the `steppe f4-ratio` command: the
command-line entry point that computes the f4-ratio admixture statistic and
prints a results table.

f4-ratio is a standalone statistic — a sibling of the f4 and f3 commands, not a
variant of qpAdm. It has no admixture target, no least-squares fit, and no rank
test. For each five-population tuple it computes a single admixture proportion
(named alpha) together with a standard error and a z-score, then emits one row
per tuple.

The file is deliberately thin plumbing around a GPU compute call. Its job is to:
turn command-line options into a list of five-population tuples, resolve the
population names to numeric indices, upload the precomputed f2 data to the GPU,
call the compute routine, and write the output. The actual f4-ratio math lives in
the compute layer (`steppe/f4ratio.hpp` and its implementation), not here.

Several of the surrounding pieces — reading the f2 directory, resolving names to
indices, building GPU resources, uploading f2 data, and writing the output — are
reused verbatim from the f3 and f4 commands, so this file mostly wires those
shared parts together for the five-population case.

---

## 2. What f4-ratio computes

For a tuple of five populations `(p1, p2, p3, p4, p5)`, the statistic is a ratio
of two f4 statistics that share the same numerator populations:

```
alpha = f4(p1, p2; p3, p4) / f4(p1, p2; p5, p4)
```

The result is the admixture proportion alpha, plus a standard error obtained by
jackknifing the ratio itself (not the two f4 values separately), plus the z-score
`alpha / se`. This reproduces the behavior of ADMIXTOOLS 2's `qpf4ratio` function.

The output table has exactly these columns, one row per tuple:

```
pop1, pop2, pop3, pop4, pop5, alpha, se, z
```

There is deliberately no p-value column. Unlike qpAdm, an f4-ratio has no rank
test and reports no p-value. This column set matches the reference results file
used to validate the command.

---

## 3. The tuple-arity constant

`kTupleArity` is a file-local constant with the value `5`.

Every f4-ratio input is a group of exactly five populations, so this one named
constant is the single source of truth for that `5` everywhere it appears: the
number of population columns, the group size and divisibility check for the
`--pops` form, the stride used to slice a flat name list into tuples, the
per-tuple loop bounds, and the fixed-size arrays that hold a tuple.

The value is frozen for parity — five is the tuple shape ADMIXTOOLS 2's
`qpf4ratio` expects, and it must not change. The constant is kept local to this
file because the arity is private knowledge of this one command rather than a
setting other code needs to see.

---

## 4. Specifying tuples: the two input forms

Tuples can be supplied to the command in either of two ways. The helper that
builds the tuple name table (`build_tuple_names`) prefers the first form and
falls back to the second.

### Row-aligned column form

The five options `--pop1`, `--pop2`, `--pop3`, `--pop4`, `--pop5` are treated as
parallel columns. Tuple `k` is the `k`-th entry read across all five columns.
This form is active as soon as any one of the five options is non-empty.

Its rules:

- The five columns must be **row-aligned** — all the same length. If they differ,
  the command reports the exact lengths it saw and stops.
- The columns must not be empty.

### The `--pops` convenience form

If none of the five column options were given, the command falls back to a single
`--pops` list: a flat list of names read in groups of five. Five names make one
tuple; ten names make two tuples; and so on.

Its rule:

- The total number of names must be a multiple of five (each tuple is
  `p1, p2, p3, p4, p5`). If it is not, the command reports the count it saw and
  stops.

If neither form supplies any names, the command explains both accepted forms and
stops. All of these malformed-input cases are treated as a configuration error
(see section 7).

---

## 5. The processing pipeline

`run_f4ratio_command` runs a fixed five-step sequence.

1. **Read the f2 directory.** The `--f2-dir` option is required; if it is empty
   the command stops with a configuration error. The directory holds the
   precomputed f2 data (`f2.bin`) and the population label list (`pops.txt`),
   read through the same loader the f3 and f4 commands use. A read failure stops
   with an I/O error.

2. **Build and resolve the tuples.** The name table is built from whichever input
   form was used (section 4). A resolver maps each population name to its index on
   the population axis, using the labels from `pops.txt`. As it resolves each
   tuple, it also records the canonical spelling of each name (the exact form
   found in `pops.txt`) so the output table prints consistent labels. An
   unresolved name stops with a configuration error.

3. **Build GPU resources and upload.** GPU resources are constructed for the
   selected device. If no CUDA device is available, the command stops with a
   runtime error, because steppe requires a GPU. The f2 data is then uploaded to
   the chosen device.

4. **Run the compute routine.** `run_f4ratio` is called with the on-device f2
   data, the resolved tuples, and the options, and returns the results.

5. **Emit the table.** The results are written as CSV (the default), TSV, or JSON
   to the `--out` file or to standard output (section 8).

Steps 3 and 4 are wrapped in a single try/catch: any exception from building
resources, uploading, or running is treated as a device fault and stops with a
nonzero exit (section 7).

---

## 6. The GPU path and the CUDA-free boundary

The GPU is the deliverable — f4-ratio always computes on the GPU. There is no
CPU-only mode.

Even so, this file is plain C++ and includes no CUDA headers. It reaches the GPU
only through header interfaces that are themselves free of CUDA: the resource
builder, the on-device f2 block type and its uploader, and the compute routine.
This keeps the application layer buildable without the GPU toolchain and enforces
a clean separation between the command-line code and the GPU code. On a machine
with no GPU, this shows up as a clear failure when resources are built, rather
than a confusing lower-level error.

The command owns its own standard output and standard error; it does not print
from deeper layers.

---

## 7. Options: the fudge default

The command passes through the shared options object. The one detail specific to
f4-ratio is the `fudge` value, which defaults to `0`.

A zero fudge means the standard error is computed from the bare ratio, with no
stabilizing term added. This is intentionally different from qpAdm, which uses a
small nonzero fudge (`1e-4`). f4-ratio needs no such term, so the default of zero
is used as-is.

---

## 8. Exit codes: domain outcomes versus faults

The command draws a firm line between a **domain outcome** and a **fault**, and
maps them to different exit codes.

A domain outcome is a normal result — including a result that the statistic itself
flags as degenerate. A domain outcome always produces the output table and exits
with success (record-and-continue). The final status carried back in the result is
mapped to the success code for these cases.

A fault is an operational failure — bad configuration, an I/O problem, or a device
error — and always exits nonzero. A domain outcome never arrives as a thrown
exception; it comes back as a status field in the result. Only faults throw or
return early.

The specific exit conditions:

| Condition | Exit meaning |
|---|---|
| `--f2-dir` missing, malformed tuples, or an unresolvable name | invalid-configuration exit |
| f2 directory read failure, or an invalid resolver | I/O-error exit |
| No CUDA device available | runtime-error exit |
| Exception while building resources, uploading, or running | mapped from the caught exception (for example, a genuine device out-of-memory maps to its own code) |
| A torn or short write while emitting (full disk, closed pipe) | I/O-error exit |
| Normal completion (including a flagged-degenerate result) | success |

---

## 9. Output

The results are written through the shared destination helper, which follows an
open, write, flush, and verify sequence. If the write is torn or short — for
example because the disk filled up or a pipe closed — the command returns an
I/O error instead of silently exiting with success and leaving a truncated file.

The row formatting is shared with the other statistic commands, so the CSV, TSV,
and JSON layouts stay consistent and no formatting logic is duplicated here. The
emitter is given the resolved, canonical population labels recorded in step 2 of
the pipeline, so the printed names match the spelling in `pops.txt`. The column
set is the one described in section 2.

# `cmd_qpwave.hpp` reference

## 1. Purpose

`src/app/cmd_qpwave.hpp` declares the single entry point for the `steppe qpwave`
command — the top-level function the command-line tool calls to run a qpWave rank
sweep. It is a header in the application layer, so it sits above the compute
library and below `main()`: `main()` parses the command line into a configuration
and then hands that configuration to the one function this header exposes.

qpWave is the test that sits underneath qpAdm. Where qpAdm asks "can this target be
modeled as a mixture of these sources," qpWave asks the more basic question that
comes first: given a set of left populations and a set of right populations, how
many independent ancestry streams are needed to relate them. It answers that by
sweeping the rank of the f4 statistics that connect the two sets and finding the
smallest rank consistent with the data.

The GPU path is the product. Running the command reads a directory of precomputed
"f2 blocks," moves that data onto a GPU, runs the rank sweep on the GPU, and writes
the results out as a table. On a machine with no usable GPU, the command reports a
clear "no CUDA device" failure rather than falling back to some slower path.

The header is deliberately plain C++20 with no GPU (CUDA) code in it. It reaches the
GPU only indirectly, through a few narrow seams that are themselves free of GPU
headers. That keeps the application layer buildable and unit-testable on a machine
that has no GPU toolkit installed at all (see section 6).

---

## 2. The single entry point: `run_qpwave_command`

```cpp
[[nodiscard]] int run_qpwave_command(const config::RunConfig& config);
```

This is the only thing the header declares.

- **Input** — a `RunConfig`, passed by const reference. `RunConfig` is the frozen,
  already-validated configuration for one run: which command to run, the resolved
  device and filter settings, the input and output paths, and the population lists.
  It is immutable by the time it reaches here; this function reads it and never
  changes it.
- **Return value** — a process exit code as a plain `int`. The value is one of the
  named `CliExitCode` codes (see section 5). `main()` returns this integer straight
  to the operating system as the program's exit status. The `[[nodiscard]]` marker
  means the caller is not allowed to silently ignore the returned code.
- **Output ownership** — this function owns everything the command prints. The
  compute library underneath it never writes to `stdout` or `stderr`; all
  user-facing output (the rank-sweep table and any error messages) is produced here
  in the application layer. This is why the function can guarantee a single,
  consistent format for both results and errors.

---

## 3. What qpWave computes: the rank sweep

qpWave differs from qpAdm in three concrete ways, and these differences are the only
genuinely new logic in this file compared with the qpAdm command.

**There is no target population.** qpAdm takes a target plus a left (source) set plus
a right (reference) set. qpWave takes only a left set and a right set — there is no
population being modeled as a mixture. The question is purely about the relationship
between the two sets.

**The first left population is the reference row.** The left set is not a flat list of
equals. The population at position 0 (`left[0]`) is singled out as the reference row
against which the f4 statistics for the other left populations are formed. This is
what "no-target resolve" means when the population names are turned into indices:
instead of pulling out a separate target index, the resolver treats `left[0]`
specially and forms the differences relative to it.

**The output is a sweep over ranks, not a single fit.** The command tests a range of
possible ranks — how many independent ancestry dimensions relate the left set to the
right set — and reports, for each rank, a chi-square statistic and its associated
degrees of freedom and tail probability. The smallest rank that the data are
consistent with is the meaningful answer: it tells you how many streams of ancestry
distinguish the left populations, which is exactly the number qpAdm needs to know
before it can fit a mixture. Each row of the emitted table is one rank in that sweep.

---

## 4. The pipeline: from f2 blocks to the rank-sweep table

Calling `run_qpwave_command` runs a fixed sequence of steps. Each step is delegated
to a lower-level helper — and every step except the last two is shared verbatim with
the qpAdm command — but the order is what makes the command work:

1. **Read the f2-blocks directory.** The input is a directory of precomputed f2
   statistics (one file of per-genome-block values, plus a `pops.txt` listing the
   populations in order). This is the "compute once, fit many times" artifact — the
   expensive genotype scan has already been done, so the sweep itself is cheap.
2. **Resolve names to indices.** The user names the left populations and the right
   populations. Those names are looked up in `pops.txt` and turned into numeric
   positions along the f2 tensor's population axis. Unlike qpAdm, there is no target
   name to resolve; `left[0]` becomes the reference row (see section 3). A name that
   isn't present is a configuration fault.
3. **Build the GPU resources.** The resolved device settings are used to pick and
   set up the GPU(s) that will run the sweep.
4. **Upload the f2 blocks to the device.** The block data read in step 1 is copied
   from host memory onto the GPU.
5. **Run the rank sweep.** qpWave runs on the GPU against the uploaded blocks and the
   resolved population indices, producing, for each rank in the sweep, a chi-square
   statistic, degrees of freedom, a tail probability, and an overall status.
6. **Emit the results.** The outcome is written out as a tidy table — CSV, TSV, or
   JSON — to the configured destination, one row per swept rank.

Steps 1 through 4 reuse the exact same f2-directory loader, population resolver, and
build-and-upload chain as the qpAdm command, and the table formatting reuses the same
output primitives. The only qpWave-specific code is the no-target resolve in step 2
and the rank-sweep emit in step 6.

---

## 5. Exit-code contract: domain outcomes versus faults

The most important and least obvious rule in this file is *how the return code is
chosen*. There are two fundamentally different kinds of "bad" outcome, and they are
treated in opposite ways.

**A domain outcome is a real, reportable result — not an error.** Sometimes the math
for the sweep is simply degenerate: the linear system is rank deficient, the
covariance matrix isn't positive-definite (non-SPD), or the chi-square statistic is
undefined. These are legitimate scientific answers about the populations given. When
one happens, the command records it as the `status` value in the output and **still
exits 0 (success).** This is the "record and continue" principle: a degenerate result
is data, not a program failure, so the process succeeds and the fact is captured in
the output, where the user can see it. qpWave produces one result carrying this
status.

**A fault means the run itself could not be carried out**, and it returns a nonzero
code. Faults include: a bad configuration such as an unknown population name or a
missing input directory; the GPU running out of memory; and file, format, or
GPU-runtime errors. The specific nonzero codes are:

| Situation | Code | Name |
|---|---|---|
| Success — including a recorded domain outcome (rank-deficient, non-SPD, undefined chi-square) | `0` | `kExitOk` |
| Bad configuration — unknown population name, missing/unreadable directory | `2` | `kExitInvalidConfig` |
| GPU ran out of memory | `3` | `kExitDeviceOom` |
| File or data-format error while reading or writing | `4` | `kExitIoError` |
| Any other runtime failure, including a GPU-runtime error | `5` | `kExitRuntimeError` |

The practical takeaway: **a nonzero exit means the run failed to complete; it never
means "the populations aren't cladal" or "the rank came out high."** A degenerate or
uninteresting scientific answer is still a successful run with a status recorded in
the output.

---

## 6. The CUDA-free layering rule and reuse of the qpAdm plumbing

This header — and the command implementation behind it — contains no GPU (CUDA) code
and includes no CUDA headers. It reaches the GPU **only** through three narrow seams,
each of which is itself free of CUDA headers:

- building the GPU resources from the device configuration,
- uploading the f2 blocks to the device,
- running the qpWave sweep on the device.

Everything CUDA-specific lives on the far side of those seams, in the compute
library. Keeping the boundary clean has two payoffs. First, the whole application
layer compiles and its unit tests run on a machine with no GPU toolkit installed,
because nothing here needs the CUDA compiler. Second, the layering can be checked
mechanically: a build-time check scans these files and fails if a CUDA header ever
sneaks in, so the separation can't quietly erode over time.

By design this command is a thin sibling of the qpAdm command. It reuses the qpAdm
f2-directory loader, the population resolver, the build-and-upload chain, and the
result-formatting primitives rather than re-implementing any of them, so the two
commands stay consistent in how they read data, select GPUs, and shape their output.

On a machine that has no usable GPU, this same design is what lets the command
surface a clean, specific "no CUDA device" fault (returning a nonzero code from
section 5) instead of crashing or silently doing the wrong thing.

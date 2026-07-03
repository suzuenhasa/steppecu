# `cmd_qpadm.hpp` reference

## 1. Purpose

`src/app/cmd_qpadm.hpp` declares the single entry point for the `steppe qpadm`
command — the top-level function the command-line tool calls to run a qpAdm fit.
It is a header in the application layer, so it sits above the compute library and
below `main()`: `main()` parses the command line into a configuration and then
hands that configuration to the one function this header exposes.

The GPU path is the product. Running the command reads a directory of precomputed
"f2 blocks," moves that data onto a GPU, runs the fit on the GPU, and writes the
results out as a table. On a machine with no usable GPU, the command reports a
clear "no CUDA device" failure rather than falling back to some slower path.

The header is deliberately plain C++ with no GPU (CUDA) code in it. It reaches the
GPU only indirectly, through a few narrow seams that are themselves free of GPU
headers. That keeps the application layer buildable and unit-testable on a machine
that has no GPU toolkit installed at all (see section 5).

---

## 2. The single entry point: `run_qpadm_command`

```cpp
[[nodiscard]] int run_qpadm_command(const config::RunConfig& config);
```

This is the only thing the header declares.

- **Input** — a `RunConfig`, passed by const reference. `RunConfig` is the frozen,
  already-validated configuration for one run: which command to run, the resolved
  device and filter settings, the input and output paths, and the population lists.
  It is immutable by the time it reaches here; this function reads it and never
  changes it.
- **Return value** — a process exit code as a plain `int`. The value is one of the
  named `CliExitCode` codes (see section 4). `main()` returns this integer straight
  to the operating system as the program's exit status. The `[[nodiscard]]` marker
  means the caller is not allowed to silently ignore the returned code.
- **Output ownership** — this function owns everything the command prints. The
  compute library underneath it never writes to `stdout` or `stderr`; all
  user-facing output (the results table and any error messages) is produced here in
  the application layer. This is why the function can guarantee a single, consistent
  format for both results and errors.

---

## 3. The pipeline: from f2 blocks to results

Calling `run_qpadm_command` runs a fixed sequence of steps. Each step is delegated
to a lower-level helper, but the order is what makes the command work:

1. **Read the f2-blocks directory.** The input is a directory of precomputed f2
   statistics (one file of per-genome-block values, plus a `pops.txt` listing the
   populations in order). This is the "compute once, fit many times" artifact — the
   expensive genotype scan has already been done, so the fit itself is cheap.
2. **Resolve names to indices.** The user names populations (a target, a set of
   left/source populations, and a set of right/reference populations). Those names
   are looked up in `pops.txt` and turned into numeric positions along the f2
   tensor's population axis. A name that isn't present is a configuration fault.
3. **Build the GPU resources.** The resolved device settings are used to pick and
   set up the GPU(s) that will run the fit.
4. **Upload the f2 blocks to the device.** The block data read in step 1 is copied
   from host memory onto the GPU.
5. **Run the fit.** The qpAdm fit runs on the GPU against the uploaded blocks and
   the resolved population indices, producing weights, standard errors, a chi-square
   / p-value, and a status.
6. **Emit the results.** The outcome is written out as a tidy table — CSV or JSON —
   to the configured destination.

---

## 4. Exit-code contract: domain outcomes versus faults

The most important and least obvious rule in this file is *how the return code is
chosen*. There are two fundamentally different kinds of "bad" outcome, and they are
treated in opposite ways.

**A domain outcome is a real, reportable result — not an error.** Sometimes the
math for a particular model is simply degenerate: the linear system is rank
deficient, the covariance matrix isn't positive-definite (non-SPD), or the
chi-square statistic is undefined. These are legitimate scientific answers about
*that model*. When one happens, the command records it as the `status` value in the
output row and **still exits 0 (success).** This is the "record and continue"
principle: a degenerate model is data, not a program failure, so the process
succeeds and the fact is captured in the results, where the user can see it.

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
means "the model didn't fit well."** A poorly-fitting or degenerate model is still a
successful run with a status recorded in the output.

---

## 5. The CUDA-free layering rule

This header — and the command implementation behind it — contains no GPU (CUDA)
code and includes no CUDA headers. It reaches the GPU **only** through three narrow
seams, each of which is itself free of CUDA headers:

- building the GPU resources from the device configuration,
- uploading the f2 blocks to the device,
- running the qpAdm fit on the device.

Everything CUDA-specific lives on the far side of those seams, in the compute
library. Keeping the boundary clean has two payoffs. First, the whole application
layer compiles and its unit tests run on a machine with no GPU toolkit installed,
because nothing here needs the CUDA compiler. Second, the layering can be checked
mechanically: a build-time check scans these files and fails if a CUDA header ever
sneaks in, so the separation can't quietly erode over time.

On a machine that has no usable GPU, this same design is what lets the command
surface a clean, specific "no CUDA device" fault (returning a nonzero code from
section 4) instead of crashing or silently doing the wrong thing.

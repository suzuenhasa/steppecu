# `exit_code.hpp` reference

## 1. Purpose

`src/core/config/exit_code.hpp` is the one place that decides what number the
command-line tool returns to the shell when a run ends. It answers a single
question: given the top-level status of a run, what process exit code should the
program exit with?

It holds exactly two things:

1. **A named enum of exit codes** (`CliExitCode`) — the small, stable integers the
   process can exit with, one per outcome category, each with a name instead of a
   bare number sprinkled through the code.
2. **One mapping function** (`exit_code_for`) — turns a run's final status into the
   matching exit code.

Keeping both here means the tool cannot drift into a second, conflicting set of exit
codes: everything that needs to know "what does this outcome exit with?" reads this
one file.

The header is deliberately tiny and dependency-light. It contains no CUDA code and
pulls in only `<cstdlib>` (for the standard `EXIT_SUCCESS` value) plus the public
status enum. The library itself never calls `std::exit` — it only *computes* the
right code and hands it back. Actually pulling the trigger on process exit is the
command-line tool's job, which calls this function from the return and error-handling
paths of its `main()`.

---

## 2. The record-and-continue contract

The central rule this file enforces is what separates a *fault* from a *domain
outcome*, and it is the reason the exit codes look the way they do.

A **domain outcome** is an ordinary statistical result of trying to fit a model that
turns out to be degenerate. Three of these exist:

- `RankDeficient` — the model's inputs were not independent enough to solve.
- `NonSpdCovariance` — the covariance matrix was not positive-definite as required.
- `ChisqUndefined` — the chi-squared goodness-of-fit value could not be defined.

These are *not* failures of the program. When a run sweeps over a large model
space — potentially thousands of candidate models — some of them will be degenerate,
and that is expected. A degenerate model produces a normal result row whose `status`
column records which of the three outcomes it hit. The run keeps going. A run that
completes exits with success (`0`) even if many of its rows carry one of these
outcomes. This is the "record-and-continue" contract: record the outcome on the row,
continue the sweep, never abort the whole run because one model was degenerate.

A **fault**, by contrast, is a real error that stops the run: a bad configuration, a
device out-of-memory condition, a file or format problem, or an unexpected runtime
error. Faults are the only things that exit with a nonzero code.

So the dividing line is: domain outcomes are rows in the output and exit `0`; faults
exit nonzero. This file is where that line is drawn.

---

## 3. The exit codes (`CliExitCode`)

`CliExitCode` is an `int`-backed enum of the process exit codes. Each fault category
gets its own distinct nonzero value so that a calling script can branch on exactly
which kind of failure happened, rather than seeing a single generic "failure."

| Name | Value | Meaning |
|---|---|---|
| `kExitOk` | `0` | The run completed. Emitted even when some result rows carry a domain outcome, because those outcomes are rows, not faults. |
| `kExitInvalidConfig` | `2` | Bad configuration — the config failed the builder's validation before any real work started. |
| `kExitDeviceOom` | `3` | A device allocation or GPU-memory-budget fault (ran out of, or exceeded the budget for, VRAM). |
| `kExitIoError` | `4` | A file or format input/output fault — for example a missing f2 directory or a malformed populations file. |
| `kExitRuntimeError` | `5` | A CUDA-runtime or otherwise unexpected fault. This is the catch-all bucket for anything that does not fit the categories above. |

Notes on the values:

- They are intentionally **small and stable**. Scripts and CI pipelines can rely on
  them.
- Every fault value is guaranteed different from `kExitOk`, so "did it succeed?" is
  always simply "is the code zero?".
- Value `1` — the conventional generic-failure code on most systems — is left unused
  on purpose, so each fault category has its own specific, distinguishable number
  from `2` upward.

---

## 4. Mapping a status to a code (`exit_code_for`)

`exit_code_for(Status status)` is the function that converts a run's top-level status
into one of the exit codes above. It is declared `constexpr` and `noexcept` (it does
no allocation, throws nothing, and can be evaluated at compile time) and is marked
`[[nodiscard]]` so its return value cannot be accidentally ignored.

The mapping:

| Input status | Returned code |
|---|---|
| `Ok` | `kExitOk` (0) |
| `RankDeficient` | `kExitOk` (0) |
| `NonSpdCovariance` | `kExitOk` (0) |
| `ChisqUndefined` | `kExitOk` (0) |
| `InvalidConfig` | `kExitInvalidConfig` (2) |
| `DeviceOom` | `kExitDeviceOom` (3) |

### Why the three domain outcomes map to 0

Mapping `RankDeficient`, `NonSpdCovariance`, and `ChisqUndefined` to `0` is what makes
the record-and-continue contract structural rather than a matter of discipline. In
normal operation, one of these three should never reach the *top level* of a run at
all — it belongs on an individual model's result row. But if some caller mistakenly
routes a per-model outcome up to the top level, this function still returns `0` for
it. The result: a single degenerate model can never accidentally turn a completed run
into a process failure. The rule is baked into the code path, not just relied upon by
convention.

### The two I/O and runtime codes are for the tool's error handling

Notice that `kExitIoError` (4) and `kExitRuntimeError` (5) do not appear as results of
any explicit case above. That is because file/format and unexpected runtime problems
do not travel through the status enum — they surface as exceptions that the
command-line tool catches in `main()` and translates directly into those codes. This
function covers the status-driven outcomes; the tool's catch blocks cover the
exception-driven ones. Together they use the same `CliExitCode` values, so there is
still only one exit-code vocabulary.

### The defensive fallthrough

The status enum is *closed*: it has exactly the six values handled above, so the
switch covers every case and the final `return kExitRuntimeError;` is unreachable
today. It exists on purpose as a safety net: if someone later adds a new status value
and forgets to handle it here, the function returns a nonzero code (5) rather than
letting the new, unhandled status silently fall through and masquerade as success.
Failing safe toward "nonzero" is the correct default for an unknown outcome.

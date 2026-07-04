# `cmd_fstat_sweep.hpp` reference

## 1. Purpose

`src/app/cmd_fstat_sweep.hpp` declares the command-line entry points for
steppe's all-combinations f-statistic *sweep*: the `steppe f4-sweep` and
`steppe f3-sweep` subcommands, plus the shared sweep body they both run.

An ordinary `f4` or `f3` command scores an explicit list of quartets or triples
that the user names. A **sweep** takes no such list. Instead it enumerates
*every* combination over a population set — every group of 4 populations for an
f4 sweep, every group of 3 for an f3 sweep — scores all of them, and returns
only the ones that clear a significance filter. This is the production-scale
tool for mining a whole panel rather than checking a handful of hand-picked
population groups.

The header itself is small: two thin per-command wrappers and one shared body
function. All three take an already-frozen run configuration and return a
process exit code. The substance is the sweep design and the reuse contract on
the shared body, described below.

---

## 2. What an all-combinations sweep is

The sweep is GPU-only by design, and the design exists to keep the host out of
the hot path. The number of combinations over even a few hundred populations
runs into the billions, so the host never enumerates them, never filters them,
and never loops over individual items. Everything that scales with the
combination count happens on the device:

1. **Enumerate on the device.** For each group of `k` populations the GPU
   computes which population indices that combination is, directly from a linear
   index — no host-built list is shipped across the CPU/GPU boundary.
2. **Compute and filter on the device.** Every combination is scored (its f4/f3
   point estimate, its diagonal block-jackknife standard error, and the
   resulting z-score) and then tested against the filter.
3. **Compact and emit only survivors.** The survivors are packed together on the
   GPU and only that small set is copied back to the host. A sweep over billions
   of combinations therefore returns a small result no matter how many
   combinations were scored.

The critical, non-obvious point: **the filter limits the output, never the
work.** Every single combination is computed on the GPU in order to test it, so
a billion-combination sweep is hours of GPU work even if only a handful of rows
survive. Because of that, a sweep whose enumeration count would exceed a safety
cap refuses to start up front — before doing any GPU work — unless the user
explicitly passes `--sure` to lift the cap.

Two survivor rules are available, both driven off the frozen configuration:

- **Minimum z-score** (the default) — keep every combination whose `|z|` is at
  least a threshold. The default threshold matches the reference significance
  cut[^at2].
- **Top-K** — keep the K combinations with the largest `|z|`.

The command reads these knobs (`--min-z` / `--top-k` / `--sure`, an optional
`--pops` subset to restrict the sweep to a set of interest, and `--shard-dir`
for sharded output) off the run configuration rather than re-parsing them. The
underlying sweep engine owns the enumeration, filtering, compaction, and the
safety cap itself; this command layer only frames the request and prints the
result.

---

## 3. Layering: plain C++20, no CUDA in this file

This header — and the command implementation behind it — is deliberately written
in plain C++20 with no CUDA header included. It is application-layer code that
reaches the GPU **only** through a CUDA-free seam: the `run_f4_sweep` /
`run_f3_sweep` calls in the public sweep interface. Those calls are ordinary C++
functions that hide the GPU code behind them.

Keeping CUDA out of the command layer is an enforced boundary in the codebase,
and this file stays on the correct side of it by never touching a GPU type
directly. The only thing it needs to include is the run-configuration type.

---

## 4. The two dedicated commands: `f4-sweep` and `f3-sweep`

```cpp
[[nodiscard]] int run_f4_sweep_command(const config::RunConfig& config);
[[nodiscard]] int run_f3_sweep_command(const config::RunConfig& config);
```

These are the entry points the command-line dispatcher calls for the
`steppe f4-sweep` and `steppe f3-sweep` subcommands.

- `run_f4_sweep_command` runs the **k = 4** sweep — every group of 4 populations
  (quartets), scored with f4.
- `run_f3_sweep_command` runs the **k = 3** sweep — every group of 3 populations
  (triples), scored with f3.

Each takes the already-frozen run configuration and returns the process exit
code. Both are thin wrappers: they simply call the shared sweep body (section 5)
with the appropriate arity and a matching program-name string. `[[nodiscard]]`
marks the return so a caller cannot accidentally ignore whether the command
succeeded.

---

## 5. The shared sweep body and how the standalone commands route into it

```cpp
[[nodiscard]] int run_fstat_sweep(const config::RunConfig& config, int k,
                                  const char* cmd);
```

`run_fstat_sweep` is the single implementation both sweep commands run through.
Its two extra parameters are what let one function serve several callers.

### The `k` (arity) parameter

`k` selects the combination size: `k = 4` gives an f4 sweep over quartets,
`k = 3` gives an f3 sweep over triples. This is the only difference between the
two sweeps at this layer, so factoring it into a parameter avoids duplicating
the whole sweep body twice.

### The `cmd` (program-name) parameter

`cmd` is the program-name string used in diagnostics printed to standard error.
It exists so error messages read with the command the user actually invoked
rather than always saying "sweep." It takes different values depending on who
calls in:

- `"f4-sweep"` / `"f3-sweep"` when reached through the dedicated
  `run_f4_sweep_command` / `run_f3_sweep_command` wrappers above.
- `"f4"` / `"f3"` / `"qpdstat"` when reached through the standalone f-statistic
  commands (see routing below).

### Why the body is exported: routing from the standalone commands

`run_fstat_sweep` is exported (rather than kept private to this translation
unit) so the standalone `f4`, `f3`, and `qpdstat` commands can route into the
**same** GPU sweep when the user asks for an all-combinations run — that is, when
`--all-quartets` (for f4/qpdstat) or `--all-triples` (for f3) is set.

In that case the standalone command does not enumerate quartets/triples itself.
It hands off to `run_fstat_sweep`, which mines the combinations over the explicit
`--pops` **subset** from the frozen configuration and applies the same
`--min-z` / `--top-k` / `--sure` / `--shard-dir` knobs. Passing its own name as
`cmd` keeps the diagnostics reading with the invoked command instead of the
sweep. The result is that there is exactly one all-combinations sweep
implementation, shared by five different command entry points, with no duplicated
enumeration or filtering logic.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

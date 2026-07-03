# `cmd_qpwave.cpp` reference

## 1. Purpose

`src/app/cmd_qpwave.cpp` implements the `steppe qpwave` command. qpWave is the
rank (cladality) test that underlies qpAdm: given a set of "left" populations and a
set of outgroup ("right") populations, it sweeps for the smallest f4 rank that can
relate the two sets, and reports whether each rank is sufficient. Unlike qpAdm,
there is **no target population** — the left set stands on its own, and the first
left population (`left[0]`) is the reference row the others are compared against.
There is no admixture-weight or population-drop output; the result is a
rank-sufficiency sweep plus a rank-drop table.

The command reads a precomputed `f2_blocks` directory (produced earlier by
`steppe extract-f2`), runs the rank sweep on the GPU, and writes one tidy result —
CSV by default, or TSV/JSON — to a file or standard output.

Structurally this is the `steppe qpadm` command with one difference: qpWave has no
target. Everything else is shared. The `f2_blocks` loader, the name-to-index
resolver, the build-resources / upload-to-device chain, and the output-emit
machinery are reused verbatim from the qpAdm command, and the result-formatting
primitives are reused through a shared `emit_qpwave_result` helper. No fit compute
and no output formatting are duplicated here.

The file is plain C++20 with no CUDA header of its own. It reaches the GPU only
through three CUDA-free seams — the resource builder, the f2-upload helper, and the
library's `run_qpwave` entry — so the command can be compiled without pulling in the
GPU toolchain. Keeping the app layer free of CUDA is a deliberate layering rule the
build enforces.

The command owns all of the program's standard output and standard error. The
library it calls never prints; it either returns a result or throws, and this command
turns those outcomes into printed messages and a process exit code. The single entry
point is `run_qpwave_command`, which takes an already-parsed, frozen configuration and
returns the exit code.

---

## 2. The five-step pipeline

`run_qpwave_command` runs a fixed sequence. Each step can fail with its own exit code;
the later steps only run if the earlier ones succeeded.

| Step | What happens | On failure |
|---|---|---|
| 1 | Read the `f2_blocks` directory: its `f2.bin` tensor and its `pops.txt` labels | invalid-config if `--f2-dir` is missing; I/O error if the directory can't be read |
| 2 | Resolve the left and right population **names** to numeric indices on the population axis, using `pops.txt` (no target; `left[0]` is the reference) | I/O error if `pops.txt` itself is malformed; invalid-config if a required flag is empty or a user-supplied name is unknown |
| 3 | Build GPU resources, upload the f2 blocks to the device, and run the rank sweep | runtime error if no GPU is visible; a device fault maps through a shared helper |
| 4 | Emit the result (CSV/TSV/JSON) to `--out` or standard output | I/O error on a torn write; invalid-config on an unknown `--format` |
| 5 | Return the process exit code derived from the result's status | (see section 5) |

Step 3 is the substance; the rest is cheap validation and bookkeeping around it.

---

## 3. Population resolution: no target, `left[0]` is the reference

Step 2 turns the user-facing population **names** into the numeric **indices** the
rank-sweep engine works with, using the labels loaded from `pops.txt`. Because
qpWave has no target argument, `--left` is the full left set on its own, and its
first entry is the reference row.

The command rejects incomplete requests up front, each with a specific message:

- `--left` needs at least the reference plus one more population.
- `--right` needs at least one outgroup population.

These are non-empty checks only. The command does not itself reject a left set that
is too small to be meaningful (for example only one population); the engine gates
those degenerate cases and returns them as a domain `status` rather than a fault.

There is a deliberate two-way split in how "name problems" are reported, and it is
worth understanding:

- A **structurally broken `pops.txt`** — for example duplicate or malformed labels
  that make the resolver itself invalid — is treated as an **I/O error**. The
  on-disk artifact is at fault, not the user's request.
- A **valid `pops.txt` but a name the user got wrong** — a typo, or a population that
  isn't in this dataset — is treated as an **invalid-config error**. The artifact is
  fine; the request naming it is not.

On success the resolver produces a list of left indices (where `left[0]` is the
reference) and a list of right indices (where `right[0]` is the first outgroup,
called `R0`). This split is not visible in the command's public declaration; it
lives here in the implementation.

---

## 4. Reaching the GPU through CUDA-free seams

Step 3 is where the actual rank sweep happens, and it goes through three CUDA-free
calls in order:

1. **`build_resources`** enumerates and binds a CUDA device from the resolved device
   configuration and returns a resources handle. On a machine with no visible GPU it
   throws, because there is nothing to bind.
2. **`upload_f2_blocks_to_device`** copies the host-side f2 tensor up to the chosen
   GPU, returning a device-resident handle.
3. **`run_qpwave`** performs the rank sweep on that device-resident data and returns
   the result. The two fit knobs from the configuration — the fudge factor and the
   rank-test significance level (`rank_alpha`) — are passed through to it.

The GPU is the deliverable: there is no CPU fallback path here. If `build_resources`
returns no usable GPU, the command prints a clear "no CUDA device available" message
and returns the runtime-error exit code — steppe is a GPU product and a
CUDA-capable device is required.

**Which device is used.** The device ordinal comes from the resolved device
configuration. An empty configuration means auto-enumerate every visible GPU; the
upload then targets the *first* configured GPU's device id. This is the single-GPU
path — the same one the reference parity test exercises. The command does not attempt
multi-GPU work of its own.

**How device faults are handled.** All three calls are wrapped in a single
`try`/`catch`. Anything they throw — no device, an out-of-memory failure, or a CUDA
runtime error — is a **fault**: the command prints a "device error" message and
returns a nonzero code through a shared helper that maps the exception to the right
category (in particular, a genuine device out-of-memory becomes an out-of-memory
exit code rather than a generic error). A normal outcome never travels this path; it
arrives as a status field on the returned result and is handled in step 5.

---

## 5. Faults versus domain outcomes, and the exit-code map

The central contract of this command is the distinction between a **fault** and a
**domain outcome**.

- A **fault** is something that prevented the run: a missing or unreadable input, a
  bad population name, no GPU, a device failure, or a failed write. Every fault
  returns a **nonzero** exit code.
- A **domain outcome** is a valid answer to the question the user asked, even when
  that answer is "this doesn't resolve here" — for example the sweep was
  rank-deficient, a matrix wasn't positive-definite, or the chi-square statistic was
  undefined. A domain outcome is **recorded as a row and exits 0**. This is the
  record-and-continue rule: a rank test that fails to resolve is still a legitimate
  result to report, not an error to abort on.

The exit codes each step produces:

| Situation | Exit code |
|---|---|
| `--f2-dir` missing | invalid-config |
| `f2_blocks` directory unreadable | I/O error |
| `pops.txt` structurally invalid | I/O error |
| `--left` or `--right` empty, or a user-named population unknown | invalid-config |
| No CUDA device visible | runtime error |
| Device fault while building/uploading/running (including OOM) | mapped by the shared caught-exception helper (OOM → out-of-memory) |
| Torn or short write while emitting | I/O error |
| Unknown `--format` token | invalid-config |
| Sweep completed (any status, success or a domain outcome) | success (0) |

That last row is the record-and-continue rule in action: the final return value is
computed from the result's status through a mapping that turns every domain outcome
into success and reserves nonzero codes for the genuine fault categories.

---

## 6. Emitting the result, and the `R0` / `right_n` invariant

Step 4 writes the result in a name-readable form and guards against a silently
truncated file.

**Name round-trip.** The sweep works in numeric population indices, but the output
should read in names. Before emitting, the command maps each left index back to its
label — including `left[0]`, the reference — so the emitted rows identify
populations by name rather than by number. The right/outgroup populations are not
listed per-row; they are summarized by a count.

**The `R0` convention and the `right_n` count.** By convention `right[0]` is the
first outgroup, `R0`, which anchors the sweep rather than being counted as an
ordinary outgroup. So the number of outgroups reported in the output, `right_n`, is
the size of the right list *minus one*. That subtraction would underflow to a
negative number only if the right list were empty — but the R0 convention guarantees
it is non-empty, and that is already enforced by the non-empty `--right` check in
step 2. To make the invariant locally self-evident, a debug-only assertion restates
that the right list is non-empty right before the subtraction; it is a fail-fast
guard, not a runtime check that fires on user input.

**Torn-write safety.** The actual writing goes through a shared
`emit_to_destination` helper that follows an open → write → flush → verify sequence.
If a write is torn or short — a full disk, a closed pipe — the helper returns an I/O
error instead of letting the process exit 0 with a truncated file. The same helper
parses the `--format` token and returns an invalid-config error on an unknown one. If
the helper returns any exit code, the command returns it immediately; otherwise it
falls through to the final status-based exit code in section 5.

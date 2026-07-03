# `cmd_qpadm.cpp` reference

## 1. Purpose

`src/app/cmd_qpadm.cpp` implements the `steppe qpadm` command: the qpAdm fit that
tests whether a target population can be modeled as a mixture of a set of source
("left") populations, judged against a set of outgroup ("right") populations. The
command reads a precomputed `f2_blocks` directory (produced earlier by
`steppe extract-f2`), runs the fit on the GPU, and writes one tidy result — CSV by
default, or TSV/JSON — to a file or standard output.

The file is plain C++20 with no CUDA header of its own. It reaches the GPU only
through three CUDA-free seams — the resource builder, the f2-upload helper, and the
library's `run_qpadm` entry — so the command can be compiled without pulling in the
GPU toolchain. Keeping the app layer free of CUDA is a deliberate layering rule the
build enforces.

The command owns all of the program's standard output and standard error. The
library it calls never prints; it either returns a result or throws, and this command
turns those outcomes into printed messages and a process exit code. The single entry
point is `run_qpadm_command`, which takes an already-parsed, frozen configuration and
returns the exit code.

---

## 2. The five-step pipeline

`run_qpadm_command` runs a fixed sequence. Each step can fail with its own exit code;
the later steps only run if the earlier ones succeeded.

| Step | What happens | On failure |
|---|---|---|
| 1 | Read the `f2_blocks` directory: its `f2.bin` tensor and its `pops.txt` labels | invalid-config if `--f2-dir` is missing; I/O error if the directory can't be read |
| 2 | Resolve the target, left, and right population **names** to numeric indices on the population axis, using `pops.txt` | I/O error if `pops.txt` itself is malformed; invalid-config if a user-supplied name is unknown |
| 3 | Build GPU resources, upload the f2 blocks to the device, and run the fit | runtime error if no GPU is visible; a device fault maps through a shared helper |
| 4 | Emit the result (CSV/TSV/JSON) to `--out` or standard output | I/O error on a torn write; invalid-config on an unknown `--format` |
| 5 | Return the process exit code derived from the model's status | (see section 5) |

Steps 3 and 4 are the substance; steps 1, 2, and 5 are cheap validation and
bookkeeping around them.

---

## 3. Name resolution and building the model

Step 2 is handled by a small local helper, `resolve_model`, which turns the
user-facing population **names** into the numeric **indices** the fit engine works
with. It does two jobs.

First it rejects incomplete requests up front, each with a specific message:

- `--target` is required (exactly one target population).
- `--left` needs at least one source population.
- `--right` needs at least one outgroup population.

Then it resolves each name against the labels loaded from `pops.txt`. The target is
a single name; left and right are lists. If any name is unknown, the resolver
fail-fasts and names the offending label, the helper prints that reason, and the
command returns the invalid-config exit code. On success it fills in the model's
target index, its list of left indices, its list of right indices, and a model index
of zero (this command fits exactly one model).

There is a deliberate two-way split in how "name problems" are reported, and it is
worth understanding:

- A **structurally broken `pops.txt`** — for example duplicate or malformed labels
  that make the resolver itself invalid — is treated as an **I/O error**. The
  on-disk artifact is at fault, not the user's request.
- A **valid `pops.txt` but a name the user got wrong** — a typo in `--target` or a
  population that isn't in this dataset — is treated as an **invalid-config error**.
  The artifact is fine; the request naming it is not.

This split is not visible in the command's public declaration; it lives here in the
implementation.

---

## 4. Reaching the GPU through CUDA-free seams

Step 3 is where the actual fit happens, and it goes through three CUDA-free calls in
order:

1. **`build_resources`** enumerates and binds a CUDA device from the resolved device
   configuration and returns a resources handle. On a machine with no visible GPU it
   throws, because there is nothing to bind.
2. **`upload_f2_blocks_to_device`** copies the host-side f2 tensor up to the chosen
   GPU, returning a device-resident handle.
3. **`run_qpadm`** performs the fit on that device-resident data and returns the
   result.

The GPU is the deliverable: there is no CPU fallback path here. If `build_resources`
returns no usable GPU, the command prints a clear "no CUDA device available" message
and returns the runtime-error exit code — steppe is a GPU product and a
CUDA-capable device is required.

**Which device is used.** The device ordinal comes from the resolved device
configuration. An empty configuration means auto-enumerate every visible GPU; the
upload then targets the *first* configured GPU's device id. This is the single-GPU
fit path — the same one the reference tests exercise. The command does not attempt
multi-GPU work of its own.

**How device faults are handled.** All three calls are wrapped in a single
`try`/`catch`. Anything they throw — no device, an out-of-memory failure, or a CUDA
runtime error — is a **fault**: the command prints a "device error" message and
returns a nonzero code through a shared helper that maps the exception to the right
category (in particular, a genuine device out-of-memory becomes an out-of-memory
exit code rather than a generic error). A normal per-model outcome never travels this
path; it arrives as a status field on the returned result and is handled in step 5.

---

## 5. Faults versus domain outcomes, and the exit-code map

The central contract of this command is the distinction between a **fault** and a
**domain outcome**.

- A **fault** is something that prevented the run: a missing or unreadable input, a
  bad population name, no GPU, a device failure, or a failed write. Every fault
  returns a **nonzero** exit code.
- A **domain outcome** is a valid answer to the question the user asked, even when
  that answer is "this model doesn't work here" — for example the fit was
  rank-deficient, a matrix wasn't positive-definite, or the chi-square statistic was
  undefined. A domain outcome is **recorded as a row and exits 0**. This is the
  record-and-continue rule: a model that fails to fit is still a legitimate result to
  report, not an error to abort on.

The exit codes each step produces:

| Situation | Exit code |
|---|---|
| `--f2-dir` missing | invalid-config |
| `f2_blocks` directory unreadable | I/O error |
| `pops.txt` structurally invalid | I/O error |
| Required flag missing, or a user-named population unknown | invalid-config |
| No CUDA device visible | runtime error |
| Device fault while building/uploading/running (including OOM) | mapped by the shared caught-exception helper (OOM → out-of-memory) |
| Torn or short write while emitting | I/O error |
| Unknown `--format` token | invalid-config |
| Fit completed (any model status, success or a domain outcome) | success (0) |

That last row is the record-and-continue rule in action: the final return value is
computed from the model's status through a mapping that turns every domain outcome
into success and reserves nonzero codes for the genuine fault categories.

---

## 6. Emitting the result

Step 4 writes the result in a name-readable form and guards against a silently
truncated file.

**Name round-trip.** The fit works in numeric population indices, but the output
should read in names. Before emitting, the command maps the model's indices back to
their labels — the target index to the target name, and each left index to its
source name — so the emitted row identifies populations by name rather than by
number. (The right/outgroup populations do not appear as per-row labels in the
output, so only the target and left labels are resolved back.)

**Torn-write safety.** The actual writing goes through a shared
`emit_to_destination` helper that follows an open → write → flush → verify sequence.
If a write is torn or short — a full disk, a closed pipe — the helper returns an I/O
error instead of letting the process exit 0 with a truncated file. The same helper
parses the `--format` token and returns an invalid-config error on an unknown one. If
the helper returns any exit code, the command returns it immediately; otherwise it
falls through to the final status-based exit code in section 5.

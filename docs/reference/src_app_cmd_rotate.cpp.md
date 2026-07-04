# `cmd_rotate.cpp` reference

## 1. Purpose

`src/app/cmd_rotate.cpp` implements the `steppe qpadm-rotate` command: the entry
point `run_qpadm_rotate_command`.

A qpAdm *rotation* answers a "which sources explain this population?" question in
bulk. The caller fixes three things:

- **one target** population (the group being modeled),
- **one right set** of outgroup populations (held fixed across the whole run), and
- **a pool** of candidate left/source populations.

Instead of fitting a single hand-picked source list, the rotation enumerates every
small subset of the pool — every group of `min_sources` up to `max_sources`
candidates — turns each subset into its own model, and fits the entire list of
models in one batched GPU call. The result is a per-model feasibility table: one row
per subset, reporting the p-value, chi-square, degrees of freedom, the estimated
mixture rank, a feasibility flag, a status, and the mixture weights with their
standard errors.

Most of this file is deliberately borrowed from the single-model `qpadm` command.
The f2-directory loader, the population-name-to-index resolver, the
build-resources / upload-to-GPU chain, and the output formatting are all reused
unchanged. The one genuinely new piece of logic is the pool-subset enumerator
described in section 3; everything else is orchestration around it. No fitting math
is duplicated here — the actual statistics run inside the batched engine.

The file is plain C++ with no GPU code compiled into it. It reaches the GPU only
through a small number of CUDA-free interface functions (build the device
resources, upload the f2 blocks, run the batched search). This keeps the
command-line layer cleanly separated from the GPU layer. The command owns its own
standard-output and standard-error printing; the library itself never prints.

---

## 2. Inputs and validation

The command reads everything it needs from the run configuration and validates the
required fields up front, before touching the GPU. Each missing or bad input
produces a specific error message on standard error and a nonzero exit code.

| Input | Meaning | If missing/invalid |
|---|---|---|
| `--f2-dir` | Directory holding the precomputed f2 statistics (an `f2.bin` file plus a `pops.txt` listing the populations, in order). | Empty → invalid-config exit. Unreadable → I/O-error exit. |
| `--target` | The single population being modeled. | Empty → invalid-config exit. Name not found in `pops.txt` → invalid-config exit. |
| `--right` | The fixed set of outgroup populations. Needs at least one. | Empty → invalid-config exit. Any name not found → invalid-config exit. |
| `--pool` | The candidate source populations to enumerate subsets from. Needs at least one. | Empty → invalid-config exit. Any name not found → invalid-config exit. |
| `--min-sources` | The smallest subset size to enumerate. Must be at least 1. | If it exceeds the pool size, there are no models to fit → invalid-config exit with an explicit message. |
| `--max-sources` | The largest subset size to enumerate. | The sentinel value `-1` means "use the whole pool" (subset size equal to the pool size). Any value larger than the pool is silently clamped down to the pool size. |

Population names are resolved to numeric indices into the f2 directory's population
list. The target resolves to a single index; the right set and the pool each
resolve to a list of indices, preserving the order the user gave.

---

## 3. The pool-subset enumerator

`enumerate_pool_subsets` is the heart of the file. Given the resolved target index,
the pool indices, the right indices, and a size band `[lo, hi]`, it produces the
full list of models to fit — one `QpAdmModel` per subset.

### What it builds

For every subset size `k` from `lo` to `hi`, it walks **every** `k`-element subset
of the pool and emits one model per subset. Each model carries:

- `target` — the single target index (the same for every model),
- `left` — the `k` pool indices in this particular subset (the sources being tried),
- `right` — the fixed right index list (the same for every model), and
- `model_index` — a dense counter starting at 0 and incrementing by 1 for each model
  emitted, across all sizes.

Sizes are skipped if they fall outside the pool: any `k` below 1 or above the pool
size contributes no models.

### The frozen ordering invariant

The order in which subsets are emitted is not incidental — it is a contract. The
enumerator lists **all subsets of the smallest size first, then all of the next
size, and so on**, and within a single size it lists subsets in lexicographic order
over the pool's index order (the subset made of the earliest pool positions comes
first). For a pool `[A, B, C]` enumerated over sizes 2 then 3, the emitted order is
`{A,B}, {A,C}, {B,C}, {A,B,C}` — indices 0, 1, 2, 3.

This exact order reproduces the order used by the reference result generator[^at2]
(which builds the expected "golden" outputs using R's `combn` function: all pairs,
then all triples, each in lexicographic order). Because both sides walk the subsets the same
way, `model_index` N always refers to the same subset on both sides, so a steppe
result row can be lined up directly against the corresponding reference row. Changing
the enumeration order would silently break that alignment even though every
individual model would still be correct. Keep the order as-is.

### The combination walk

Advancing from one subset to the next uses the standard "next combination in
lexicographic order" algorithm. The subset is tracked as an ascending list of `k`
positions into the pool. To advance: find the rightmost position that can still be
incremented (it has not yet reached its maximum, which for position `i` is
`n - k + i`, where `n` is the pool size), increment it, and reset every position to
its right to the smallest ascending values that follow. When no position can be
incremented, all subsets of that size have been produced.

Note the target is not part of the enumerated left set — it is stored separately on
each model and is treated as the modeled population by the engine, not as a source.

---

## 4. The execution pipeline

`run_qpadm_rotate_command` runs six stages in order. The first four never touch the
GPU; a failure in any of them returns before any device work begins.

1. **Read the f2 directory.** Load `f2.bin` and `pops.txt`. A missing `--f2-dir` is
   an invalid-config exit; an unreadable directory is an I/O-error exit.
2. **Resolve names to indices.** Build the resolver from the population labels, then
   resolve the target, the right list, and the pool list. Any unknown name stops the
   run with an invalid-config exit.
3. **Enumerate the subsets.** Compute the size band: `lo` is `--min-sources`, `hi` is
   `--max-sources` (or the whole pool when `-1`), clamped to the pool size. Bail out
   with an invalid-config exit if `lo` exceeds the pool, or if the band somehow
   enumerated zero models. Otherwise call the enumerator from section 3 to build the
   model list.
4. **Warn on multiple devices** (see section 6) — a warning only, not a failure.
5. **Run the batched engine.** Build the device resources, confirm at least one CUDA
   device is present (steppe is a GPU product; with no device this is a
   runtime-error exit), upload the f2 blocks to the first device, and hand the whole
   model list to the batched search in one call. Any exception thrown here — no
   device, out of memory, or a CUDA runtime error — is caught and turned into a
   nonzero fault exit code (see section 5).
6. **Attach labels and emit the table.** Turn the target and each model's left
   indices back into population names for display, compute the reported right count
   (see section 7), and write the per-model table in the requested format (CSV, TSV,
   or JSON). The write goes through a shared helper that opens, writes, flushes, and
   verifies the output, so a truncated or torn write is reported as an I/O error
   rather than silently exiting successfully with a short file.

---

## 5. The exit-code contract: record-and-continue

The rotation distinguishes two very different kinds of "this model didn't work,"
and only one of them is a failure of the command.

- **Per-model domain outcomes are rows, not errors.** If an individual model comes
  out rank-deficient, non-positive-definite, or with an undefined chi-square, that is
  a legitimate scientific outcome for that subset. It is recorded as the `status` on
  that model's row, and the command still exits successfully (exit code 0). A
  rotation is expected to contain many such rows; they are data, not faults. Because
  there is no single result to derive an exit code from, the rotation never routes an
  individual model's outcome to the process exit code.

- **Faults are nonzero.** Only failures of the command as a whole return a nonzero
  code: a bad or missing input (invalid config), an unreadable f2 directory or a torn
  output write (I/O error), no CUDA device (runtime error), or a device error such as
  out-of-memory surfacing from the batched run. The caught-exception path maps the
  specific failure to its exit code — for example, a genuine device out-of-memory maps
  to the out-of-memory exit code.

In short: a rotation that completes and emits its table is a success even if every
row reports a problem; only the machinery around the rotation can fail the command.

---

## 6. The multiple-device warning

If the caller requests two or more GPUs, the command prints a warning to standard
error and continues — it does **not** treat this as an error.

The batched engine's multiple-GPU path is correct, but on consumer cards without
direct GPU-to-GPU peer access its throughput is limited by having to route partial
results back through host memory. Until a fully device-resident combine lands,
single-GPU is the supported and recommended configuration, and the warning tells the
user to prefer `--device 0`. The run still proceeds on the requested devices.

---

## 7. Output labels and the "nr" convention

Before emitting, the numeric indices carried on each model are turned back into
population names: the target label, and for each model the list of its left/source
labels in order. These names are what the table shows.

The reported count of right (outgroup) populations follows a specific convention.
The first population in the right list is treated as a distinguished base outgroup,
so the number of right populations reported in the table is the size of the right
list **minus one**. This matches the count recorded in the engine's own metadata, so
the table's reported right count and the engine's internal count always agree.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

# `cmd_fstat_sweep.cpp` reference

## 1. Purpose

`src/app/cmd_fstat_sweep.cpp` implements the two GPU-only "sweep" commands,
`steppe f4-sweep` and `steppe f3-sweep`. A sweep computes an f-statistic for
*every* combination of populations — every group of 4 for f4, every group of 3
for f3 — and reports back only the combinations that clear a significance filter.

The same body also backs the `--all-quartets` / `--all-triples` modes of the
standalone `f4`, `f3`, and `qpdstat` commands, so those commands route into the
identical GPU sweep instead of computing one combination at a time.

The single most important property to understand is that **the full table of
combinations is never built in memory or on disk.** For 500 populations there are
about 2.5 billion groups of four; writing all of them out would be a multi-terabyte
dump. Instead every combination is enumerated, computed, and filtered *on the GPU*,
and only the survivors are compacted and copied back across to the host. The filter
limits the *output*, never the amount of work — every combination is still computed
in order to test it.

The file is plain C++20 and includes no CUDA headers. It reaches the GPU only
through a small set of CUDA-free entry points (`build_resources`,
`upload_f2_blocks_to_device`, `run_f4_sweep`, `run_f3_sweep`). That keeps the
command layer free of GPU code and lets the GPU work sit behind a clean seam.

---

## 2. The pipeline, step by step

`run_fstat_sweep` is the shared body. It runs the following sequence, returning a
process exit code at each failure point:

1. **Require an f2 directory.** The sweep reads its inputs from a precomputed `--f2-dir`
   (the "f2 blocks" — per-block pairwise statistics). If none was given, it refuses
   with an invalid-config exit code.
2. **Read the f2 directory** and, on any read failure, return an I/O error.
3. **Build a population-name resolver** from the labels stored in that directory. The
   resolver maps a population name to its index along the population axis.
4. **Resolve the optional `--pops` subset.** Each name passed with `--pops` is looked
   up and turned into an index. An unknown name is an invalid-config failure. If
   `--pops` is empty, the sweep runs over the *whole* directory.
5. **Assemble the sweep request** from the frozen configuration — the subset, the
   filter choice, and the cap override (see sections 3 and 4).
6. **Build GPU resources** and confirm at least one CUDA device is present. steppe is
   a GPU product; with no CUDA device the command fails with a runtime error rather
   than falling back to a CPU path.
7. **Upload the f2 blocks to the device** so they stay resident in GPU memory for the
   whole sweep.
8. **Run the sweep** — `run_f4_sweep` for arity 4, `run_f3_sweep` for arity 3 — which
   does the on-device enumerate, compute, filter, and compact.
9. **Handle a capped result** (section 4).
10. **Emit the survivors** to the chosen destination (section 5).

Any exception thrown by the GPU steps is caught and turned into a device-error exit
code (section 6).

---

## 3. The filter policy (how many results survive)

Two independent filters decide which combinations come back, and they can be active
at the same time.

- **Minimum z-score (`--min-z Z`).** Keep any combination whose absolute z-score is at
  least `Z`. This is a *floor*: it acts as a device-side pre-filter (a "tau floor")
  that discards everything below the threshold before it can compete for a survivor
  slot. The default is `3.0`; passing `--min-z` raises it.
- **Top-K (`--top-k K`).** Keep the `K` combinations with the largest absolute z-scores.
  This is implemented as a **device-bounded reservoir**: the GPU holds only the `K`
  most-significant results in a fixed buffer of size proportional to `K`, with a
  threshold that rises as stronger results are found and displace weaker ones. Because
  only those `K` rows ever cross back to the host, host memory stays proportional to
  `K` no matter how many billions of combinations are computed.

### How the filter mode is chosen

The request's filter field is set by this decision tree:

1. **If `--top-k K` was given** (`K > 0`): use Top-K with that `K`. When both are set,
   `--min-z` still applies as the floor and coexists with the `K` cap — the floor
   pre-filters, the cap bounds the kept count.
2. **Otherwise, if this is a bare all-combinations sweep** (a standalone `f4`/`f3`/`qpdstat`
   run with `--all-quartets`/`--all-triples` and no explicit `--top-k`): default to
   Top-K with `kFstatDefaultSweepTopK` (one million). This default exists to prevent a
   host out-of-memory failure. On a signal-rich dataset a minimum-z filter alone is
   **not a real bound** — millions of combinations can clear `|z| >= 3.0`, and copying
   all of them back could exhaust host RAM. The one-million bounded reservoir (about
   40 MB) guarantees a full sweep can never blow up host memory.
3. **Otherwise** (the dedicated `f4-sweep` / `f3-sweep` subcommands with no `--top-k`):
   use the minimum-z filter with its `3.0` default. These subcommands keep the explicit
   minimum-z behavior because they are the deliberate "give me everything above this
   threshold" entry point.

---

## 4. The combination cap and `--sure`

An all-combinations sweep is guarded by an up-front safety cap on the *number of
combinations* it will attempt. The count of groups grows explosively with the
population count, and every combination costs GPU time even if almost none survive,
so a careless sweep can be hours of work.

If a requested sweep would enumerate more combinations than the cap allows, the
sweep refuses **before doing any GPU work.** The refusal comes back as a "capped"
result, which this command treats as a **hard, actionable error** — not as an empty
or partial answer. It prints how many combinations were requested, tells the user to
either pass `--sure` to override the cap or restrict the population set with `--pops`,
and returns an invalid-config exit code.

`--sure` (`config.sweep_sure()`) is the explicit override that lifts the cap and lets
an oversized sweep proceed.

The cap value itself, and the one-million default kept count, are named constants
documented with the rest of steppe's tunables; this file only reads them.

---

## 5. Output destinations and formats

### Destination precedence

The survivor table is written to exactly one destination, chosen in this order:

1. **`--shard-dir DIR`** (highest precedence). This is the sweep-scale destination: a
   stable on-disk directory the survivors are written into, as a file named
   `survivors.<ext>` (`csv`, `tsv`, or `json` matching the format). The directory is
   created if missing. This is preferred over piping a large stream to standard output
   when a sweep produces a lot of survivors.
2. **`--out FILE`** — a single output file.
3. **Standard output** — when neither of the above is set.

### The write-verify contract

Every on-disk write goes through an open → write → flush → verify sequence. A torn or
short write — a full disk, a closed pipe — returns an I/O error exit code instead of
silently exiting successfully with a truncated survivors file. This applies both to
the `--shard-dir` path and to the shared destination helper used for `--out` and
standard output.

### Row format

`emit_sweep` writes the survivors only (never the full table). Each row is the `k`
population labels (`pop1`..`popK`, where `k` is 4 for f4 and 3 for f3) followed by
four numbers: the estimate `est`, its standard error `se`, the z-score `z`, and the
p-value `p`.

- **CSV / TSV** share the same layout and differ only in the delimiter (comma vs tab).
  A header row names the columns.
- **JSON** is a single object carrying summary fields — `status`, `enumerated` (the
  number of combinations computed), `survivors` (how many came back), and `capped` —
  plus a `rows` array of row objects with the same `pop1..popK, est, se, z, p` fields.

Population indices in the results are turned back into human-readable names through the
same resolver built in step 3.

### Observability

After a successful emit, a one-line summary is written to standard error (not to the
data stream): how many combinations were enumerated and how many survived, and for the
shard path, the file it was written to. Keeping this off the data stream means the
survivor output stays clean for piping.

---

## 6. Exit codes

The command distinguishes failure classes so a caller can react to each:

| Situation | Exit code |
|---|---|
| Missing `--f2-dir`; unknown `--pops` name; unknown `--format`; a capped sweep without `--sure` | invalid-config |
| Failure reading the f2 directory; a failed resolver; cannot create `--shard-dir`; cannot open the shard file; a torn/short write | I/O error |
| No CUDA device present | runtime error |
| An exception from the GPU steps | mapped by `exit_code_for_caught` |
| Success | mapped from the sweep's own status via `exit_code_for` |

The `exit_code_for_caught` mapping is what turns a genuine device out-of-memory
condition into its own distinct exit code rather than a generic failure, so an OOM is
reported as an OOM.

A capped or empty sweep is a *clean, deliberate* outcome — the process records it and
returns the appropriate code without treating it as a crash. Only real device or I/O
faults produce the failure codes above by way of an exception or a checked write.

---

## 7. Entry points

Three functions make up the public surface:

- **`run_f4_sweep_command`** — the `steppe f4-sweep` entry point; calls the shared body
  with arity 4 and the program name `"f4-sweep"`.
- **`run_f3_sweep_command`** — the `steppe f3-sweep` entry point; arity 3, program name
  `"f3-sweep"`.
- **`run_fstat_sweep`** — the shared body both wrappers call. It takes the arity `k`
  (4 selects `run_f4_sweep`, 3 selects `run_f3_sweep`) and a `cmd` string used as the
  program name in every diagnostic message.

The `cmd` string is why the same code can serve several commands: the standalone
`f4`, `f3`, and `qpdstat` commands call `run_fstat_sweep` directly when
`--all-quartets` / `--all-triples` is set, passing their own name, so their error and
summary messages read with the command the user actually typed rather than always
saying "f4-sweep".

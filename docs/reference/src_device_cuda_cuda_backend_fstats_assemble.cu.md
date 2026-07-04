# `cuda_backend_fstats_assemble.cu` reference

## 1. Purpose

This file is the GPU backend's home for two related families of work:

1. **Assembling f4 and f3 statistics** from data that already lives in GPU
   memory. Given a set of population combinations, it produces the per-block
   f-statistic values and the leave-one-out and total reductions that the model
   fit and the block jackknife need.
2. **Sweeping over every combination** of populations (every group of 3, or
   every group of 4) and keeping only the most statistically significant
   results — without ever letting the host run out of memory, no matter how
   many billions of combinations are enumerated.

Everything here reads f2 data that is already resident on the GPU (a
`DeviceF2Blocks`). None of it recomputes f2 from raw genotypes; that happens
earlier. The functions defined here are:

- `device_survivor_blocks` — the single definition of the "which genome blocks
  are usable" predicate. It is shared by every other function in this file and
  is also called from other translation units (the D-statistic and model-fit
  code).
- `run_fstat_sweep_device` — the on-device all-combination sweep engine.
- `assemble_f4`, `assemble_f4_quartets`, `assemble_f3_triples` — the three
  device-resident assemble entry points, each with a host-tensor twin that
  simply throws.
- `f4_sweep`, `f3_sweep` — thin wrappers that call the sweep engine with the
  right combination size.

These function bodies were moved here verbatim when a larger backend file was
split into smaller translation units. Nothing about the math, the numerical
precision, or the order of operations changed in that move. This file compiles
as part of the same GPU backend target as the rest of the CUDA code, so it
inherits identical compiler settings.

One piece of shared state is worth knowing about up front: the class member
`tot_line_` is **written** by the assemble functions here (they cache the
per-combination "total line" of the jackknife) and later **read** by the
model-fit code when it builds the jackknife covariance. It is a hand-off buffer
between this file and the fit stage.

---

## 2. Precision policy: the native-FP64 cancellation carve-out

Every function here takes a `precision` argument, and every function here
**ignores it on purpose**. The f-statistic stages always run in native
double precision, regardless of what the caller asks for. Each function marks
this with a `(void)precision;` and a short comment.

The reason is numerical, not a shortcut. An f4 or f3 value is a *difference* of
products — a subtraction of two nearly equal quantities — which is exactly the
situation where floating-point catastrophic cancellation destroys accuracy. The
faster emulated double-precision path used elsewhere in steppe can faithfully
form a *product*, but it cannot recover bits that were already annihilated by a
prior *subtraction*. So even though the default arithmetic mode for the
matrix-multiply-heavy stages is emulated double precision, this cancellation-
sensitive difference is carved out and held at true native double precision. The
same carve-out is applied to the f2 numerator elsewhere; this file follows the
identical rule so the two stay consistent.

---

## 3. Missing-block survivor filtering (`device_survivor_blocks`)

The block jackknife partitions the genome into blocks. A block can be unusable
for a given run if it has missing data for a needed population pair. This
function decides, once, which blocks survive.

It is the **single definition** of that predicate. Every assemble function and
the sweep engine calls it, and so do functions in other files — keeping the
"which blocks count" decision in exactly one place means the fit path, the
D-statistic path, and the sweep can never disagree about it.

How it works:

- If the resident f2 data carries no per-pair missing-block information (the
  pointer is null), there is nothing to drop, so **every block is kept**. This
  is the legacy path and is bit-for-bit identical to doing no filtering.
- Otherwise a small GPU kernel computes a keep/drop flag per block from the
  resident missing-block information. That tiny flag vector (one int per block)
  is copied down to the host, and the host builds the list of surviving block
  ids in **ascending** order.

The output is a plain list of surviving block indices. When that list is shorter
than the full block count, a real drop occurred; when it is the same length, the
survivor list is just the identity and every downstream kernel takes its
bit-identical no-drop arm. This mirrors the behavior of removing
blocks with missing data before computing statistics[^at2].

---

## 4. The device-resident assemble doors

Three functions assemble f-statistics from resident f2 data. They all return an
`F4Blocks` result and all follow the same shape:

- `assemble_f4(DeviceF2Blocks, left_idx, right_idx)` — the general form. `left_idx`
  and `right_idx` describe a grid of left and right population groups; the result
  covers `nl × nr` combinations (where `nl` and `nr` are one less than the sizes
  of the two index arrays, because those arrays are stored as offset boundaries).
- `assemble_f4_quartets(DeviceF2Blocks, quartets)` — a flat array of population
  quadruples (four ints each). The result is `N` combinations laid out along a
  single axis, using the convention that the row count is `N` and the column
  count is `1`.
- `assemble_f3_triples(DeviceF2Blocks, triples)` — the same, for triples (three
  ints each) instead of quadruples.

Each function does the following:

1. Call `device_survivor_blocks` to get the usable blocks, then build the
   survivor block sizes and the total SNP count `n` (the sum of the surviving
   block sizes, which is the jackknife normalizer).
2. Upload the small index vectors, the survivor block sizes, and — only when a
   real drop occurred — the survivor map, to the GPU. When no block was dropped
   the survivor map is passed as null and the kernel takes its identity arm.
3. Run the gather kernel, which reads the resident f2 data and forms the native
   double-precision f-statistic combine (a four-slab combine for f4, a
   three-slab combine for f3), compacting the output onto the survivor block
   axis.
4. Run the leave-one-out and total reduction, which fills the per-block
   leave-one-out values, the total, and the total line.
5. Copy the small fit intermediates back to the host: the per-block values, the
   leave-one-out values, the total, and the total line (into the shared
   `tot_line_` member).

The quartet and triple forms deliberately **reuse the same leave-one-out/total
kernel** as the general form; that kernel only cares about the flattened
combination count, so no separate kernel is needed for the batched shapes.

Degenerate inputs (no combinations, no blocks, or no resident f2) return an
empty result with a zero block count rather than launching any kernel. If every
block is missing, the result is returned empty as well; callers are expected to
gate that case.

---

## 5. The host-tensor throw twins

For each of the three assemble functions there is a second overload that takes a
host-side `F2BlockTensor` instead of a `DeviceF2Blocks`. On the GPU backend these
overloads do nothing but throw a `runtime_error` with a clear message.

They exist because the backend interface declares both overloads. The GPU
backend only implements the device-resident door — the whole point is that f2
already lives in GPU memory — so the host-tensor overload is never a valid GPU
call. That overload is the CPU reference backend's door; on the GPU it is a
loud, immediate error rather than a silent wrong path.

---

## 6. The all-combination sweep: pipeline (`run_fstat_sweep_device`)

This is the engine behind `f4_sweep` (combination size 4) and `f3_sweep`
(combination size 3). It computes an f-statistic for **every** combination of
the requested size drawn from the population set (or from a caller-supplied
subset), scores each one, and returns only the survivors.

The central design fact: **every combination is computed on the GPU**, in order,
so that it can be tested against the significance filter. The filter bounds the
*output*, never the *work*. A sweep over a billion combinations is a billion
GPU computations even if only a handful of results survive.

The engine never enumerates combinations on the host and never accumulates
survivors in a host array. Both would be catastrophic at these scales. Instead:

- Combinations are **unranked on the device** — a kernel turns a range of global
  combination indices directly into population-index tuples, with no host loop.
- Survivors are kept in a **fixed-size device reservoir** (section 7), so host
  memory stays small no matter how many combinations are computed.

### The enumeration count

First the engine computes how many combinations exist: "range choose k", where
`range` is the population count (or the subset size). This is done with a
saturating product so an astronomically large count clamps to the maximum
instead of overflowing. An impossible request (fewer populations than the
combination size) yields zero, and a zero count returns a clean empty result.

### The chunk loop

Because the whole enumeration cannot fit in GPU memory at once, the engine
processes it in **chunks** of `C_max` combinations (chunk sizing is covered in
section 8). For each chunk it runs a five-stage pipeline:

1. **Unrank** the chunk's global index range into the actual population-index
   tuples on the device. Uses the quartet unrank kernel for size 4, the triple
   unrank kernel for size 3, remapping through the subset list when a subset was
   requested.
2. **Gather and reduce.** The same gather kernel the explicit assemble path uses
   forms the per-block f-statistic values, then the leave-one-out/total kernel,
   then a jackknife-adjusted per-block kernel, then the jackknife diagonal
   variance. This is where each combination gets its estimate and its variance.
3. **Filter by significance.** A kernel turns each combination's total and
   variance into an estimate, a standard error, a z-score, and its absolute
   value, and sets a one-byte survivor flag wherever the absolute z-score
   exceeds the current threshold `tau`. Crucially, `tau` is read live from the
   device, not passed as a host constant — it rises as the sweep proceeds (see
   section 7).
4. **Compact on the device.** The key columns are deinterleaved and then the
   survivors are compacted with a GPU stream-compaction primitive (CUB's
   `Flagged`). The number selected is written on the device; exactly one integer
   per chunk (that selected count) is read back to the host. This single int
   readback is the only per-chunk host touch in the entire loop.
5. **Append to the reservoir.** The chunk's survivors are copied device-to-device
   into the bounded reservoir. If they would overflow it, the reservoir is
   compacted first to make room (section 7). A single early chunk can produce
   more survivors than the reservoir target, so the append happens in pieces,
   compacting between pieces, and the reservoir never overruns.

After the last chunk, one final compaction leaves the device holding exactly the
top surviving rows, sorted, and only those rows are copied to the host and packed
into the result. The result carries, for each survivor, its population-key tuple,
its estimate, its standard error, and its z-score, plus the total enumerated
count.

---

## 7. The bounded device top-K reservoir

This is the core mechanism that lets a sweep over billions of combinations run in
a small, fixed amount of host memory. It replaced an earlier design that
accumulated survivors in an unbounded host array and ran out of memory.

The idea: keep a **fixed-capacity device buffer** that always holds the running
top K survivors (the K with the largest absolute z-score), together with a
threshold `tau` that **only ever rises**.

Key quantities:

- **K** — the reservoir target: how many results the sweep ultimately keeps. It
  comes from the caller's `top_k`, or the default when unspecified, and is then
  clamped (section 8). The default when the user gives no limit is
  `kFstatDefaultSweepTopK` (one million).
- **CAP = 2·K** — the actual buffer capacity. The extra K of slack guarantees a
  whole chunk's survivors always fit before a compaction is needed; each
  compaction returns the fill to at most K, leaving at least K free again.
- **tau** — the live absolute-z-score cutoff, held on the device and read by the
  filter kernel each chunk.

### Compact-and-raise

When the reservoir would overflow, a helper sorts it by absolute z-score
descending (CUB `SortPairsDescending`), gathers the top rows into scratch, swaps
them back, and truncates the fill to at most K. When the reservoir is genuinely
full to K — and only in the rising-threshold mode — it then **raises `tau` to the
new K-th largest absolute z-score**. The same helper runs once more at the very
end to produce the final sorted top-K.

### Why this is correct

Because a combination's absolute z-score is what defines significance, and
because `tau` only ever rises, a combination that the filter dropped (its
absolute z-score was at or below `tau`) could never have belonged to a global
top-K whose K-th value is already `tau`. So no kept result is ever wrongly
evicted relative to a true global top-K over the entire enumeration. The rising
threshold also means each successive chunk tends to produce *fewer* survivors, so
the work of appending and compacting shrinks as the sweep proceeds.

### Host memory stays O(K)

The host never accumulates survivors. Only the final top-K rows — about 40 MB at
K = one million — ever cross back to the host, independent of how many billions
of combinations were computed. A full multi-billion-combination sweep cannot blow
up host memory.

### Two filter modes

The sweep supports two thresholding modes, selected by the config's filter mode:

- **Top-K mode** (`kSweepFilterTopK`): `tau` starts at the configured floor and
  rises as described above. The sweep keeps the K most significant results.
- **Minimum-z mode** (`kSweepFilterMinZ`): `tau` stays pinned at the configured
  minimum z-score floor and never rises — the sweep keeps *everything* above that
  floor. But the reservoir **still caps its fill to K** as a hard safety ceiling,
  so even a minimum-z sweep that matches an enormous number of combinations
  cannot exhaust memory. In this mode the reservoir is a bounded "top K above the
  floor" rather than an exact "everything above the floor."

---

## 8. Sizing constants and VRAM budgeting

The sweep sizes both its reservoir and its per-chunk working set from the amount
of free GPU memory at run time. Two TU-local constants and one environment
variable govern this. These are kept next to the sweep, not in the shared config
header, because they are occupancy and memory-fit tuning specific to this engine.

| Name | Value | What it's for |
|---|---|---|
| `kFstatIntClampMax` | `0x40000000` (2^30) | The ceiling for anything counted with a signed 32-bit int: the CUB item counts and this file's integer-indexed kernels. Both the reservoir target K and the per-chunk size are clamped to this so no count overflows an int. |
| `kFstatReservoirBytesPerSlot` | `160` | The per-reservoir-slot byte estimate used to reserve reservoir memory before sizing the chunk. Each slot needs roughly 112 bytes (four doubles and four ints for the live reservoir, plus sort keys, an index permutation in and out, and gather scratch); 160 rounds that up generously to also cover the CUB sort's temporary storage. |

### Single-source sizing of K and CAP

The reservoir target `K` and capacity `CAP` are computed **exactly once**: `K`
starts from the caller's `top_k` (or the default), is clamped down to the
enumerated total, clamped to `kFstatIntClampMax`, and floored at 1; `CAP` is
`2·K`. That single computation is reused both to reserve the reservoir's fixed
memory footprint and to drive the live reservoir state, so a change to the
clamping policy lands in one place and the sizing and the state can never drift
apart.

### Reservoir footprint reserved first

The reservoir's fixed footprint (`CAP` slots × `kFstatReservoirBytesPerSlot`) is
subtracted from free GPU memory **before** the chunk is sized. That way the chunk
is fit into the memory that remains after the reservoir, the resident f2 data,
and the CUB scratch are accounted for.

### Chunk size from free VRAM

The engine measures the per-combination device cost — the index tuple, the
per-block value / leave-one-out / jackknife arrays (three arrays of one double
per surviving block), the scalar reductions, the estimate/se/z/absolute-z, the
four key columns, the survivor flag byte, and the compacted output columns — then
takes a budget of **0.4 of the free memory that remains after the reservoir** and
divides it by that per-combination cost. The 0.4 factor and the reservoir-first
subtraction together leave headroom for the CUB temporary storage, the resident
f2, and allocator fragmentation. The resulting chunk size is floored at one
combination, clamped to `kFstatIntClampMax`, and clamped to the enumerated total.

### The chunk override lever

The environment variable `STEPPE_FSTAT_CHUNK`, when set to a positive integer,
overrides the computed chunk size directly. It is a manual tuning lever for the
chunk size only; it does not change any reported result, and the clamps to
`kFstatIntClampMax` and to the enumerated total still apply on top of it.

### CUB temporary storage

The CUB scratch buffer is sized **once**, at the largest item count any call will
ever see — the chunk size for the per-chunk compaction, and `CAP` for the
reservoir sort — and reused for every call. The two-call CUB idiom (a first call
with a null pointer to query the required size) is used to discover that size.

---

## 9. The `f4_sweep` / `f3_sweep` wrappers

These two public functions are one-line dispatchers. `f4_sweep` calls
`run_fstat_sweep_device` with combination size 4; `f3_sweep` calls it with
combination size 3. All of the real work — sizing, the chunk pipeline, the
reservoir — lives in the shared engine, so the two sweep sizes cannot diverge in
behavior.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

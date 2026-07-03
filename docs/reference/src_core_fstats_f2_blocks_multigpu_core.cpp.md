# `f2_blocks_multigpu_core.cpp` reference

## 1. Purpose

This file is the host-only heart of the multi-GPU precompute step that builds
the f2 block statistics. When more than one GPU is available, the work of
computing per-block f2 tensors is split across the devices, run on all of them
at once, and then the pieces are combined. This file owns the two things that
happen *before* the combine:

1. **The plan** — `plan_multigpu_shards` decides which whole genome-blocks (and
   therefore which SNP columns) each GPU is responsible for.
2. **The fan-out** — three closely related functions launch one host thread per
   GPU, hand each GPU its slice of the input, and collect that GPU's partial
   result. The three differ only in *where* each GPU's partial result ends up
   (a returned host tensor, a device-resident handle, or written straight into
   a caller-supplied buffer).

All three fan-out functions share a single private helper, `fan_out_shards`,
which does the actual thread choreography. The three public entries supply only
the last step — what each worker does with its finished slice.

Nothing in this file touches CUDA, and nothing here does the final combine. The
combine (including the optional direct GPU-to-GPU fast path) lives in the
public entry point that calls into this file.

---

## 2. Host-pure and CUDA-free: why this is a separate file

This file deliberately contains no GPU code. It refers only to the CUDA-free
seams: the abstract compute backend interface, the pure-host shard planner, and
the pure-host block-range helper. It never names the device-side peer-copy
symbol.

That restriction is the whole point of the split. Because this file links
without any CUDA toolkit, a plain host-side unit test can drive the planner and
the fan-out against a *fake* compute backend — no GPU, no CUDA install, no
device linking required. That makes the multi-GPU logic (the shard math, the
sub-view transform, the thread fan-out, the error handling, the empty-shard
edge cases) fast to test in isolation. The slower end-to-end parity test that
exercises the real GPU backend and the real combine cannot be run that cheaply,
so keeping this core testable on its own is a real win.

The body of this file was lifted verbatim out of the original single-file
implementation as a pure refactor. The sequence of operations, the plan, and
the per-device partials are byte-for-byte identical to what the inline version
produced. This is a **frozen** property: the goal of the split was to change
*where the code lives*, never *what it computes*.

---

## 3. `plan_multigpu_shards` — building the block-aligned shard plan

This function decides how to divide the genome-blocks among the GPUs. It is a
pure function of its four inputs — no devices, no GPU state, no CUDA.

The inputs describe the SNP-to-block assignment (`partition`), the number of
SNPs (`M`), the number of blocks (`n_block`), and the number of GPUs (`G`). It
returns exactly `G` shards, one per GPU, in device order (device 0 first).

How it works, in two steps:

1. **Invert the partition into per-block column ranges.** The shared block-range
   helper turns the per-SNP block assignment into, for each block, the
   contiguous span of SNP columns that belong to it. This works because the
   block assignment is non-decreasing across SNPs — every block's SNPs are
   contiguous — so a contiguous run of blocks always maps to a contiguous run of
   SNP columns. That helper also validates the partition contract, and it is the
   single place that validation happens.
2. **Assign whole blocks to devices.** The resulting ranges are handed to the
   block-shard planner, which hands each device a contiguous run of *whole*
   blocks, balanced so each device gets a roughly equal number of SNPs. This
   planner is the single home of the block-to-device mapping.

The per-block ranges are the *only* input the planner needs. Each range already
carries its own SNP count (used for the balancing) and its column bounds. There
is deliberately no separate parallel array of block sizes — an earlier version
had one, but it only narrowed the true `long` count down to an `int` and forced
a fragile two-array contract on the planner, so it was removed.

Each returned shard records the block range it owns (`b0`, `b1`) and the SNP
column range that corresponds to it (`s0`, `s1`). When there are fewer blocks
than GPUs, the surplus GPUs get **empty** trailing shards (where `b0 == b1`),
which the fan-out handles gracefully (see section 10).

The function throws if `G` is zero (via the planner) or if the partition is
malformed (via the range helper).

---

## 4. The shared fan-out (`fan_out_shards`)

This private helper is the single home of the per-device concurrent fan-out.
All three public entry functions call it; before it existed, each of the three
copy-pasted the same thread-launch-and-join code.

Its job: launch one worker thread per GPU, give each worker the correct slice
of the input, run all of them at once, wait for them all to finish, and surface
any failure. What each worker actually *does* with its slice is not baked in —
that last step is supplied by the caller as a callable (see section 5).

The structure is:

- Size a per-worker error slot array to `G`, one slot per device, all initially
  empty.
- Open a scope containing a vector of auto-joining threads. **Reserve** capacity
  for `G` threads up front. This reserve is not just an optimization — if the
  vector reallocated mid-launch it would move threads that have not yet been
  joined, which is unsafe. Reserving guarantees the launch loop never
  reallocates.
- Launch `G` workers, one per device.
- Close the scope. Because the threads auto-join in their destructors, closing
  the scope is a **join barrier**: control does not continue until every worker
  has finished. This barrier is what makes it safe to read the per-worker error
  slots (and, for the callers, the per-device result slots) afterward.
- After the barrier, walk the error slots in device order and re-throw the first
  one that is set (see section 8).

---

## 5. The seam callable and the three entry points

The three public functions differ in exactly one respect: what a worker does
once its slice is ready. That single varying step is passed into `fan_out_shards`
as a callable — the "seam."

### The seam signature

The seam is invoked by each worker with everything that worker prepared:

| Argument | What it is |
|---|---|
| `g` | The device index (also the result-slot index). |
| `Qg`, `Vg`, `Ng` | Zero-copy column sub-views of the three input matrices, restricted to this device's SNP columns. |
| `block_id_local` | Pointer to this device's dense, zero-based local block ids (see section 6). |
| `n_block_local` | The number of blocks in this device's shard. |
| `sh` | The shard itself, so the seam can read its global block offset (`sh.b0`) when it needs to know where this device's blocks sit in the full result. |

The local block-id array is owned by the fan-out for the duration of the seam
call — it stays alive until the seam returns inside the worker, so the seam may
use the pointer freely but must not retain it past the call.

### The three entries

Each of the three public functions is a thin wrapper: it pre-sizes its result
container to `G`, calls `fan_out_shards` with a seam, and returns. Because every
worker writes only its own slot `g` of a container that was fully sized before
any thread started, there is no shared mutable state and no locking.

| Function | Seam calls | Result of each worker | Where the partial lands |
|---|---|---|---|
| `compute_multigpu_partials` | `compute_f2_blocks` | A compact host-side f2 tensor | Moved into `partials[g]`; the caller combines afterward. This is the host-staged baseline. |
| `compute_multigpu_partials_resident` | `compute_f2_blocks_resident` | An opaque, move-only handle to a partial that stays resident on the GPU (no copy back to the host) | Moved into `partials[g]`. The handle carries the global block offset so a later device-resident combine knows where the partial belongs. |
| `compute_multigpu_partials_into` | `compute_f2_blocks_into` | Nothing returned; the worker copies its result directly into a shared caller buffer | Written straight into the caller's pre-allocated f2/vpair/block-size arrays at this device's offset. No per-device tensor, no combine copy. |

For the resident variant, moving a handle into its slot is just a pointer swap —
no CUDA call — so it is safe from any thread. The handles (and the GPU memory
they own) survive the join barrier because the returned vector outlives the
worker threads; that memory is freed only after the later combine has consumed
it. If a worker throws, its slot is left as a default empty handle that owns
nothing, and any peer handles that did allocate are freed by normal stack
unwinding.

For the "into" variant, each worker writes only its own disjoint slice of the
shared output — the block-aligned shards tile the full block range without
overlap, so two workers never write the same bytes and the concurrent writes
into one host buffer are race-free. If a worker throws, its slice may be left
partially written, but the fan-out re-throws before the function returns, so no
partial result is ever handed back to the caller.

---

## 6. What each worker does with its shard

Every worker, regardless of which seam it will call, runs the same preparation.
This is the shared body inside `fan_out_shards`.

**Zero-copy column sub-views.** The three inputs are column-major matrices of
shape `P × M` (P populations across, M SNPs along the columns). A worker does
*not* copy its columns; it constructs a lightweight view that points at the same
memory, offset to the shard's first SNP column. The offset is `P * s0` elements
in, and the view's column count is `s1 - s0` (the shard's local SNP count). The
same offset applies to all three matrices because they share the same shape.

**Dense, zero-based local block ids.** The backend expects each device's blocks
to be numbered starting at zero. The global block ids for this shard's SNPs run
from `sh.b0` upward, so the worker builds a small local array where
`block_id_local[k] = global_block_id[s0 + k] - sh.b0`. This renumbering is done
inside the worker, off the critical path of launching the other devices, and
each worker owns its own slice.

**The seam call.** With the sub-views and the local block ids ready, the worker
invokes the seam, which drives that device's backend and stores the result.

---

## 7. Concurrency: why running the GPUs in parallel is safe

The `G` devices run their matrix multiplies at the same time, one host thread
per device. This is where the multi-GPU speedup comes from. Several facts make
it safe:

- **Each backend targets its own device.** Selecting a CUDA device is a
  per-host-thread setting, so each worker thread binds itself to its device and
  drives that device's backend. The backend for each device owns its own stream,
  handle, and scratch buffers, and finishes by synchronizing its own stream.
  Commands issued to streams on *different* devices run concurrently.
- **No shared mutable state.** Each worker writes only its own result slot, in a
  container sized before any thread started, and builds its own local block-id
  array. Nothing is shared and mutated across workers, so no locks are needed.
- **Inputs upload concurrently.** The host-to-device uploads are pinned through
  each backend's registry, so the devices' uploads run as concurrent DMA
  transfers rather than being serialized.

The combine that follows reads the per-device results in fixed device order,
and only *after* the join barrier — the barrier is the happens-before edge that
the fixed-order combine relies on.

---

## 8. Exception safety and deterministic error reporting

A `std::jthread` whose entry function lets an exception escape calls
`std::terminate` — it would crash the whole process. So every worker wraps its
body in a catch-all and, on any failure, stores the captured exception into its
own error slot rather than letting it propagate out of the thread.

After the join barrier, `fan_out_shards` walks the error slots in device order
and re-throws the **first** one that is set (lowest device index). This turns a
backend or device fault into an ordinary exception the caller can handle, and it
does so **deterministically**: no matter which device actually failed first in
wall-clock time, the caller always sees the failure from the lowest-numbered
failing device. Two runs that hit the same fault report it identically.

---

## 9. Parity-neutrality: identical results regardless of timing

Running the devices concurrently does not change any reported number. The bits
each device produces are fixed entirely by its block-aligned shard — which SNP
columns and which blocks it was given — and not at all by which wall-clock slot
it happened to run in. Every worker forwards the same precision setting
unchanged to its backend call.

Because of this, the concurrent fan-out is **bit-identical** to computing the
shards one after another in a single thread, and the final combined result is
bit-identical to a single-GPU run. This is a frozen property the refactor was
required to preserve: same plan, same per-device partials, combined in the same
fixed device order.

---

## 10. Defensive guards against malformed shards

Two clamps guard against a shard with a malformed negative range. Well-formed
block-aligned shards always tile the block space with non-negative ranges, so
these clamps never fire in normal operation — they exist purely so that a bad
range can never turn into a giant wrapped-around unsigned value:

- The column offset uses `s0 < 0 ? 0 : s0` before it is cast to an unsigned
  size, so a negative start can never wrap the byte offset.
- The local block-id array length uses `M_local < 0 ? 0 : M_local` for the same
  reason, so a negative local SNP count can never wrap the vector length.

**Empty shards** (where `b0 == b1`) are a normal, expected case — they occur
when there are fewer blocks than GPUs. An empty shard has zero local blocks and
zero local SNP columns. The sub-view offset is still computed but is harmless
because the backend early-returns before dereferencing anything, and it hands
back an empty result. The combine then simply places nothing for that device.

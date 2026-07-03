# `block_sink.cuh` reference

## 1. Purpose

`src/device/cuda/block_sink.cuh` defines the seam that gets each genome
block's finished results out of GPU memory and to wherever they need to live.

steppe processes the genome in independent blocks. For each block it computes,
entirely on the GPU, two square arrays sized P×P (where P is the number of
populations): an "f2" array and a paired-variance ("vpair") array. Together these
two arrays are one block's *slab*. Once a slab is computed on the device, something
has to decide whether it stays in GPU memory, gets copied to host RAM, or gets
written to a file on disk — and, for the latter two, do that copy or write without
stalling the GPU that is already busy computing the next block.

That "get the slab to its destination" job is the *sink*. This header holds:

1. **The sink interface** (`BlockSink`) — the small contract the block-streaming
   loop calls: start up, hand off block *b*'s device pointers, finish.
2. **Two concrete sinks** (`HostRamSink`, `DiskSink`) — one per non-resident
   destination.
3. **The shared machinery both sinks are built from** (`StagingRing`, `SinkSlot`,
   `sink_wait_slot_drained`, and the constant `kStreamStagingSlots`) — a small pool
   of pinned host buffers plus a background writer thread that lets the slow
   copy-or-write overlap the GPU computing the next block.

A defining property of this whole layer: the sink only changes **when** and
**where** a slab lands, **never its bits**. Every copy along the way is a raw byte
copy — GPU-to-host is a plain memory transfer, host-to-destination is a `memcpy` or
a `pwrite` of the same bytes — so which tier a run uses can never change a number
steppe reports. This "same bytes regardless of tier" property is what keeps results
reproducible across the three memory tiers.

This is CUDA code: it takes raw device pointers and owns pinned (page-locked) host
staging memory, so it lives in the GPU layer alongside the pinned-buffer helper it
builds on.

---

## 2. The three memory tiers and where the sink fits

steppe can hold its per-block results in one of three places, chosen by how big the
problem is:

- **Tier 0 — Resident.** Everything stays in GPU memory. This is the fastest path,
  and it **bypasses this file entirely** — it never constructs a sink at all,
  because there is nothing to stage or write; the slabs simply stay where they were
  computed.
- **Tier 1 — Host RAM.** Slabs are copied from the GPU into a host-memory tensor.
  Handled by `HostRamSink`.
- **Tier 2 — Disk.** Slabs are written to a file so that RAM stays tiny even on a
  laptop. Handled by `DiskSink`.

Only Tiers 1 and 2 use a sink, and both of them build on the exact same staging
machinery (see section 7). The block-streaming loop that drives all of this keeps a
small number of GPU chunk buffers of its own (double-buffered, so one chunk's
copy-to-host can finish while the next chunk is computed); the sink adds a second,
host-side ring on top of that so the *slow* write can overlap as well.

---

## 3. `kStreamStagingSlots` — the triple-buffer depth

| Constant | Value | What it's for |
|---|---|---|
| `kStreamStagingSlots` | `3` | How many pinned host staging slots the ring holds. Three is deliberately the smallest depth that lets three things happen at once: the GPU computes block `b+1`, the copy-to-host of block `b` is in flight, and the background writer drains block `b-1` to its destination. The pinned slots are allocated **once** when the sink starts and reused for every block. |

The three-way overlap (compute `b+1` / copy `b` / write `b-1`) is exactly why the
depth is 3 and not 2. The GPU-side chunk buffers only need to survive their own
copy-to-host, so they are double-buffered separately; it is the *slow write* — the
disk `pwrite` or the host `memcpy` — that needs the extra slot of headroom, and that
extra slot is what the third staging slot provides.

---

## 4. `BlockSink` — the abstract sink interface

`BlockSink` is the abstract base class the block-streaming loop talks to. It has a
virtual destructor and three pure-virtual methods. Both concrete sinks implement it.

| Method | What it does |
|---|---|
| `begin(P, n_block, block_sizes)` | Set up the destination and start the background writer. Allocates the destination (a host tensor or an open file), pins the reusable staging slots **once**, and records the shape information (`P`, the number of blocks, and each block's SNP count) that the destination header needs. Called before any block is spilled. |
| `spill_block(b, f2_dev, vpair_dev, slab_elems, stream)` | Hand off block `b`. `f2_dev` and `vpair_dev` are **device** pointers to that block's two P×P arrays, valid on the given CUDA `stream` after the block finished computing. The sink claims the next free staging slot, issues the asynchronous copy-to-host into it on `stream`, records a completion event, and enqueues the slot to the background writer. It does **not** block on the destination write, and it does **not** free the device slab — the streaming loop still owns that memory. |
| `finish()` | Drain the writer queue and finalize. For disk: write the trailing shape information, flush to stable storage, close, and reopen the file read-only. For host RAM: join the writer thread. |

### The spill contract in detail

The key behaviors a caller relies on:

- **Device pointers, not host data.** `spill_block` receives GPU pointers. The
  actual copy to host is the sink's job, issued asynchronously on the caller's
  stream so it is a genuine background DMA transfer, not a blocking copy.
- **Natural backpressure.** Claiming a staging slot blocks *only* when all slots are
  still being drained by the writer. In the normal case a slot is always free, so
  the call returns immediately; under a slow disk it blocks just long enough to let
  the writer catch up. This is the mechanism that keeps the GPU from racing ahead and
  overrunning the ring.
- **Non-blocking on the write.** The compute thread only *enqueues* a slot; it never
  waits for the actual disk write or host copy. That is what lets the slow write
  overlap GPU compute of the next block.
- **The loop keeps ownership of the device slab.** The sink copies out of the device
  memory but never frees it.

---

## 5. `SinkSlot` — one pinned staging slot

`SinkSlot` is one slot in the reusable ring. Both sinks use the same slot type.

| Field | Type | Meaning |
|---|---|---|
| `f2` | `PinnedBuffer<double>` | A pinned host buffer sized to hold one block's f2 array (P² doubles). Pinned memory is required for a genuine asynchronous DMA copy. |
| `vpair` | `PinnedBuffer<double>` | The matching pinned buffer for the block's paired-variance array (also P² doubles). |
| `done` | `Event` | A CUDA event recorded on the stream right after this slot's copy-to-host is issued. Waiting on it tells the writer the slot's bytes have fully arrived in host memory. |
| `block` | `int` | Which block this slot currently holds (`-1` when unused). The writer uses it to place the bytes at the right destination offset. |

The `done` field is a move-only wrapper around a raw CUDA event, chosen deliberately
over a hand-rolled event handle for two reasons: it creates the event (with timing
disabled) automatically when the slot vector is sized at startup, and it frees the
event automatically when the slot is destroyed. Because it is move-only and clears
its handle when moved, moving a `SinkSlot` can never accidentally destroy the same
CUDA event twice.

---

## 6. `sink_wait_slot_drained` — the fail-fast completion wait

`sink_wait_slot_drained(done, what)` is a tiny free function both sinks' writer
threads call before touching a slot's bytes. It waits on the slot's completion event
and, if the wait returns anything other than success, throws.

The reason it throws rather than proceeding is a correctness guarantee, not mere
caution. A non-success return means either the wait itself failed or an earlier
asynchronous operation on the stream — such as the copy-to-host itself — failed. In
either case the slot may hold **stale or partially-copied bytes**. Copying such a
slot to the destination would silently corrupt the f2/vpair results. So the only safe
action is to fail fast; the slot is never drained on a bad wait. The `what` argument
is a short tag naming the call site, so the two tiers report an identical, attributable
error.

The comment records the specific CUDA return codes this can surface (success, invalid
value, invalid resource handle, launch failure) and notes that it can also surface
errors from *previous* asynchronous launches on the stream — which is exactly the
"the copy already failed" case that makes fail-fast necessary.

---

## 7. `StagingRing` — the shared pinned ring and background writer

`StagingRing` is the single piece of machinery that both sinks are built from. It
owns the pinned staging slots and the one background writer thread, and it handles all
of the pinning, the slot bookkeeping, the copy-and-enqueue, the writer loop, the
shutdown, and the safety barrier. The *only* thing that differs between the two tiers
is the one action taken when a slot is finally written to its destination — and the
sink injects that as a callback.

### 7.1 Why it exists

The host-RAM and disk sinks previously held byte-for-byte identical ring plumbing:
the same slot pool, the same locks and condition variables, the same slot-claiming
logic, the same writer loop skeleton, the same copy-and-enqueue in `spill_block`, the
same shutdown-and-join (in four places), and the same teardown barrier. The *only*
per-tier difference was the write action itself — a `memcpy` into a host tensor for
Tier 1 versus a `pwrite` to a file for Tier 2 — plus the error tag. Consolidating all
the shared plumbing into one class, with the write action passed in as a callback,
replaces two copies of the plumbing with one, with no change to the math or the
threading behavior.

### 7.2 The drain callback

`DrainFn` is the per-slot write action: `std::function<void(SinkSlot&)>`. It runs
**on the writer thread**, after the slot's bytes have been confirmed present in host
memory, and it is what actually moves the bytes to the tier's destination. It may
throw (a disk `pwrite` can fail with an errno); a thrown error is recorded as the
writer's failure.

### 7.3 Lifecycle and methods

| Member | What it does |
|---|---|
| `begin(slab_elems, drain, what)` | Pin the `kStreamStagingSlots` slots (each holding two `slab_elems`-double buffers) **once**, store the tier's drain callback and error tag, and start the writer thread. The caller is expected to do all *non-ring* startup (allocate the destination tensor, or open the file and write its header) before calling this. If `slab_elems` is 0 the ring is a no-op and no writer starts. |
| `spill_block(b, f2_dev, vpair_dev, stream)` | Claim a free slot (blocking under backpressure), issue the two asynchronous copies of the block's f2 and vpair slabs into the slot on `stream`, record the slot's completion event, and enqueue the slot index to the writer. Does not block on the write and does not free the device slab. No-op on an empty ring. |
| `stop_and_join()` | Signal the writer to stop, wake it, and join it. Idempotent — a second call is a no-op once the thread has joined. |
| `writer_failed()` / `writer_error()` | Report whether the writer recorded a failure, and its message. The owning sink checks these after `stop_and_join()` and re-throws the error on the compute thread from `finish()`. |

`StagingRing` owns a thread and pinned memory, so it is explicitly **non-copyable and
non-movable** — copying or moving it would make no sense and could leave a dangling
thread or double-freed pinned memory.

### 7.4 The writer loop and the fail-fast policy

The background writer runs a simple loop: wait until a slot is queued (or a stop is
requested), pop it, wait for its copy-to-host to complete via
`sink_wait_slot_drained`, run the tier's drain callback, then return the slot to the
free pool and signal that a slot is available again.

The fail-fast policy from section 6 is enforced here for **both** tiers. If the
completion wait returns a failure, the writer records the error and **skips the
drain** for that slot — it never writes a possibly-undrained slot, because doing so
would silently corrupt the results. The drain callback itself may also throw (a disk
write error); that is caught the same way. In both cases the writer records only the
*first* error and keeps draining the rest of the queue, so the compute thread is never
left deadlocked waiting on a full ring. The recorded error is re-thrown by the owning
sink at `finish()`.

### 7.5 Slot bookkeeping

Internally the ring tracks slots with a mutex, two condition variables, a free-list, a
ready-queue of slot indices, and a stop flag:

- `acquire_slot()` blocks on the "a slot is free" condition variable until a slot
  returns to the free pool, then marks it taken. This is the backpressure point.
- The writer pops slot indices from the ready-queue, and after draining a slot,
  returns it to the free pool and signals the "slot free" condition variable.

The slab element count is stored once, and the byte size (`slab_bytes_`) is derived
once at startup rather than recomputed on every copy.

### 7.6 The teardown barrier

`teardown_barrier()` is a defensive `cudaDeviceSynchronize` run from the destructor,
**after** the writer has stopped and joined but **before** the pinned slots are freed.

It exists because of one narrow failure case. Normally the writer's per-slot event
wait is the guarantee that every copy-to-host finished before the pinned memory is
freed. But if `spill_block` faulted partway through issuing a block — say the second
copy or the event-record threw — the *first* copy could be left in flight on the
stream, and that slot was never enqueued, so the writer never waited on it. Freeing
the pinned host buffers while such a transfer is still running would be a
use-after-free of pinned host memory. `cudaDeviceSynchronize` blocks until the device
has completed all outstanding work on every stream, so it drains any such orphaned
transfer before the memory is released.

Because it runs in a destructor, it warns rather than throws on error (a destructor
must never throw). It is off the hot path, so its cost is irrelevant; on the normal
path every copy was already drained before the join, so it is a cheap no-op.

---

## 8. `HostRamSink` — Tier 1 (spill to host RAM)

`HostRamSink` streams each block's slab into a host-memory tensor using the shared
`StagingRing`. It is constructed with a reference to the destination host tensor.

Its drain callback simply `memcpy`s each drained slot into the correct block offset of
the host tensor (the block's f2 array to the f2 region, its vpair array to the vpair
region) while the GPU computes the next block. The copy-to-host is a raw byte copy and
the host copy is a `memcpy` of the same bytes, so this tier is result-identical to
keeping everything resident.

Because it owns a `StagingRing` (and therefore a thread and pinned memory), it is
explicitly non-copyable and non-movable. Its destructor needs no special logic — the
ring's own destructor stops the writer, joins it, and runs the teardown barrier.

---

## 9. `DiskSink` — Tier 2 (spill to disk)

`DiskSink` streams each block's slab to a file, so RAM usage stays tiny even for a
large problem on a small machine. It is constructed with the output file path.

The flow across its lifecycle:

- **`begin`** opens the file, writes a fixed 64-byte header, and starts the ring. The
  ring's drain callback `pwrite`s each drained slot to the file's f2 and vpair region
  offsets. Because the write happens on the writer thread, the slow `pwrite` overlaps
  GPU compute.
- **`spill_block`** copies the block's slabs into the next staging slot and enqueues
  it (delegated to the ring).
- **`finish`** drains the queue, writes the trailing block-size information, flushes
  to stable storage, closes the write handle, and reopens the file read-only.
- **`take_descriptor(out)`** moves the finalized on-disk descriptor (the path, the
  shape, and the reopened read handle) out to the caller. It is valid only after
  `finish()`; the orchestrator stores the result for the later fit phase to read from.

As with the host sink, every byte written is raw, so the disk tier produces the exact
same results as the other two tiers. `DiskSink` owns a file descriptor **and** a
`StagingRing`, so it too is explicitly non-copyable and non-movable, and its members
record the shape, region offsets, block sizes, and finalization state needed to build
and reopen the on-disk cache.

A forward-declared `DiskF2Blocks` (a CUDA-free descriptor type from the f2-blocks
output header) is the object `take_descriptor` fills in.

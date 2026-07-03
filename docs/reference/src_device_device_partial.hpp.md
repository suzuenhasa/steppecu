# `device_partial.hpp` reference

## 1. Purpose

`src/device/device_partial.hpp` declares one class, `DevicePartial`. It is a
lightweight handle that owns one GPU's partial result from a multi-GPU f2
computation, and keeps that result sitting in the memory of the GPU that produced
it rather than copying it back to the host.

Here is the situation it exists for. When steppe computes its f2 statistics across
several GPUs, it splits the genome's blocks into shards and gives each GPU one
shard. Every GPU independently computes a partial f2 tensor (and its paired-variance
companion) for just its own blocks. Afterward those partials have to be summed
together into one final result — the "combine" step. A `DevicePartial` is exactly
one GPU's contribution to that sum, packaged as a handle the orchestrating code can
hold, move around, and hand to the combine.

The key design choice is that the partial stays **resident** on the GPU that
computed it. It is not copied down to host memory and is not freed early. The
combine step reads it straight out of GPU memory. This avoids a wasteful round trip
to the host and back, which was the specific slowdown that motivated the
device-resident combine.

One more thing shapes the whole design: this header names **no CUDA type at all**.
It can be included by code that does not (and should not) pull in the CUDA runtime
headers. How that is achieved is the subject of the next section.

---

## 2. The CUDA-free seam and the opaque payload

The orchestrating code that decides which GPU gets which shard, and that collects
the partials to combine them, is deliberately written to be free of any CUDA types.
It needs to hold and pass along these partial-result handles without ever seeing
`<cuda_runtime.h>`. So `DevicePartial` is split into two halves:

- **The public handle** (this header) exposes only plain host data: a few `int`
  shape fields and a `std::vector<int>`. Nothing here is a CUDA type.
- **The private payload** lives in a nested `struct Impl`, which is only
  *declared* here (`struct Impl;`) and *defined* in the GPU-side file
  `cuda/device_partial_impl.cuh`. That is where the actual GPU memory buffers and
  device pointers live.

The handle points at its payload through a `std::unique_ptr<Impl> impl`. Because
`Impl` is an incomplete type in this header, the compiler never needs to know its
contents here, and the CUDA types stay entirely on the GPU side of the seam. This
is the same "declare CUDA-free, define in a `.cu`/`.cuh`" split that the peer-to-peer
combine code uses.

The payload itself holds the pair of double-precision GPU buffers — the f2 slab and
its paired-variance slab — plus the resident device pointers into them. Those
buffers are the owned resource; when the handle is destroyed, they are freed.

Invariant to remember: `impl` is **null exactly when the handle is empty** (see
section 5). A non-empty partial always has a payload; an empty one never does.

---

## 3. Lifetime and ownership contract

`DevicePartial` is **move-only**. The copy constructor and copy assignment are
deleted; only move construction and move assignment (both `noexcept`) are provided.
This is because it uniquely owns a pair of GPU buffers — there is exactly one owner,
and ownership transfers by moving, never by copying.

The four special members and what they mean:

| Member | Behavior |
|---|---|
| `DevicePartial()` | Default constructor. Produces an **empty** handle — the state used for a freshly-made, a moved-from, or an empty-shard partial. It owns no GPU buffers. |
| `~DevicePartial()` | Destructor. Frees the resident pair of GPU buffers (a no-op if the handle is empty). |
| `DevicePartial(DevicePartial&&) noexcept` | Move constructor. Transfers ownership of the buffers; the source is left empty. |
| `operator=(DevicePartial&&) noexcept` | Move assignment. Same transfer of ownership. |
| copy constructor / copy assignment | **Deleted.** |

### The survive-then-free ordering

The lifetime has a specific order that the rest of the system relies on:

1. A worker allocates the buffer pair on its assigned GPU (`device_id`) while
   computing its shard.
2. The finished `DevicePartial` is **moved out of the worker thread** into the
   vector of partials that gets returned to the caller. It has to *outlive the
   worker thread's join* — the handle keeps the GPU memory alive after the thread
   that created it has ended.
3. The handle is destroyed — and its buffers freed — only **after the combine has
   consumed it**. Freeing early would pull the data out from under the combine.

A subtle point that makes step 3 safe across multiple GPUs: freeing GPU memory is
**pointer-device-aware**. The free operation releases the memory on whichever GPU
the buffer actually lives on, no matter which GPU is the "current" one at the moment
the destructor runs. So the collecting code can destroy a vector of partials that
came from several different GPUs without first having to switch the active GPU to
match each one.

---

## 4. Shape fields

These are plain host scalars (and one host vector) describing the partial. They
carry no CUDA types, which is what lets the CUDA-free orchestrator hold and forward
the handle. Every buffer slab is a `P`-by-`P` matrix per block.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `P` | `int` | `0` | The population count. This is the leading dimension of every slab — each per-block matrix is `P` × `P`. |
| `n_block_local` | `int` | `0` | How many genome blocks this GPU owns — the size of its shard. A value of `0` marks an **empty shard** (see section 5). |
| `b0` | `int` | `0` | Where this partial's blocks start in the global block ordering — the shard's offset. The combine uses it to place these local blocks at the right position in the combined result. |
| `device_id` | `int` | `kInvalidDeviceId` (`-1`) | The physical CUDA device ordinal the buffers are resident on. The `-1` default means "no device yet," matching the shared no-device sentinel from `config.hpp`. |
| `block_sizes` | `std::vector<int>` | empty | The per-block SNP counts, one entry per owned block, so its length equals `n_block_local`. These live on the host (not the GPU) and mirror the same per-block-size list the existing host combine copies. |

The `block_sizes` vector matters even for an empty shard: the combine still reads it
to know each block's SNP weight when summing, so it is placed host-side rather than
being part of the GPU payload.

---

## 5. Empty shards

Not every GPU necessarily receives blocks. When there are more GPUs than blocks, or
when the sharding leaves one device with nothing to do, that device produces an
**empty** partial.

The `empty()` predicate reports this:

```
[[nodiscard]] bool empty() const noexcept { return n_block_local <= 0; }
```

An empty `DevicePartial`:

- has `n_block_local == 0` (a value of `0` or below counts as empty),
- owns **no** resident GPU buffers, so its `impl` payload pointer is null, and
- contributes nothing to the sum — during the combine, the loop that would place
  this partial's blocks simply runs zero iterations and does nothing.

The one thing the combine still does for an empty shard is read its `block_sizes`
(which is itself empty here), so an empty partial is handled by the exact same code
path as a non-empty one, just with nothing to iterate over. There is no special-case
branch for "this GPU was idle."

# `p2p_combine.hpp` reference

## 1. Purpose

`src/device/p2p_combine.hpp` declares the fast path for combining partial f2
results across several GPUs in one machine. When a job is split across multiple
GPUs, each GPU computes the f2 result for its own slice of the genome and leaves
that partial result sitting in its own GPU memory. This header declares the two
functions that gather those per-GPU partials back into one whole result.

The combine is done directly GPU-to-GPU. One GPU is chosen as the "combine root"
(always the first GPU in the run's device list, referred to here as GPU 0). The
root pulls each other GPU's partial straight out of that GPU's memory over a
direct peer copy and drops it into the correct place in the full result — no
detour through host RAM. This is the whole point of the file: it is the version
of the combine that avoids copying every partial down to the CPU and back up
again.

The header itself contains **no CUDA code**. It only names public,
CUDA-free types and declares two functions. All the actual GPU calls live in the
matching `.cpp`. That keeps the header safe to include from code that must not
pull in the GPU toolchain.

The two functions differ only in where the final answer ends up:

- `combine_f2_partials_resident` returns the finished result as a host-side
  tensor (one final copy down from GPU to CPU at the end).
- `combine_f2_partials_resident_device` stops one step earlier and returns the
  finished result still living in GPU memory, so the caller can decide whether
  and when to copy it down.

---

## 2. The two functions

### `combine_f2_partials_resident`

```
F2BlockTensor combine_f2_partials_resident(
    std::span<DevicePartial> partials,
    std::span<const DeviceShard> shards,
    int P, int n_block_full, int root_device_id);
```

Takes the per-GPU partial results that were left resident in GPU memory,
assembles them into the full result on the root GPU, and copies that full result
down to the host with **one** final GPU-to-CPU transfer. Returns the whole
`[P × P × n_block_full]` tensor as a host `F2BlockTensor`. Marked
`[[nodiscard]]` — the returned tensor is the entire result, so ignoring it is a
bug.

### `combine_f2_partials_resident_device`

```
DeviceF2Blocks combine_f2_partials_resident_device(
    std::span<DevicePartial> partials,
    std::span<const DeviceShard> shards,
    int P, int n_block_full, int root_device_id);
```

Does exactly the same assembly — same GPU-to-GPU placement, same bytes in the
same positions — but stops *before* the final copy to the host. It returns the
assembled result still resident in the root GPU's memory, wrapped in a
`DeviceF2Blocks` handle. There is no host tensor and no final copy down; the
caller can request `.to_host()` later if it needs the CPU copy. The per-block
sizes are recorded on the returned handle in the same fixed GPU order. Also
`[[nodiscard]]`.

Use this second form when the next stage (for example, the model fit) is going
to read the result straight from GPU memory, so copying it to the CPU only to
copy it back would be wasted work.

---

## 3. Why the combine is a placement, not a sum

The most important property of this combine is that it **copies bytes into
place** — it does not add anything up.

Each GPU owns a contiguous, non-overlapping range of genome blocks (its
"shard"). Because the shards are disjoint and together they tile the whole block
range `[0, n_block_full)` exactly, every slab of the final result is produced by
exactly one GPU. So combining is just: take each GPU's partial and write it,
unchanged, into its own slice of the full tensor. No slice is ever written
twice, so there is nothing to accumulate.

This has two concrete consequences the code relies on:

1. **No zeroing and no `+=`.** The result buffer is never cleared to zero first,
   and partials are never added onto it. Each partial is a raw byte copy into its
   own disjoint region. An earlier design that zeroed the buffer and then added
   partials onto it could silently flip a negative-zero (`-0.0`) element; doing a
   plain copy instead reproduces `-0.0` byte-for-byte.

2. **Bit-identical to the other combine paths.** Because it is a verbatim copy
   in a fixed GPU order, this direct GPU-to-GPU combine produces exactly the same
   bytes as the slower host-staged combine (which gathers every partial to the CPU
   and copies them into place there) and exactly the same bytes as a plain
   single-GPU run. This is a firm invariant: the transport only moves bytes, and
   software fixes the order. It is **never** a networked all-reduce, which would
   not give a reproducible bit-for-bit result.

The fixed order is `g = 0 .. G-1` — the GPUs are visited in the exact order of
the run's device list, and each partial goes to the block offset that its shard
owns.

---

## 4. The transport: peer copy versus device-to-device copy

The bytes move in one of two ways, chosen per partial by whether that partial
already lives on the root GPU:

- **A partial owned by some other GPU** is pulled directly out of that GPU's
  memory into its slice of the root's result using an asynchronous peer copy
  (`cudaMemcpyPeerAsync`). This is a direct GPU-to-GPU DMA — no upload from the
  host, no staging buffer. On the capable hardware this path runs at roughly
  55.6 GB/s.

- **The root GPU's own partial** (the one whose owning device *is* the root) is
  already on the right GPU, so it is moved with a plain asynchronous
  device-to-device copy (`cudaMemcpyAsync` in device-to-device mode) into its
  slice. Again, no host round-trip.

Each partial handle carries its own source device id and its own destination
block offset. Because that information travels with the handle, the caller does
not have to pass a separate parallel list of device ids alongside the partials.

After issuing all the copies, the code waits on the stream
(`cudaStreamSynchronize`) so every DMA has finished before the function returns.

---

## 5. Who turns on peer access

Direct GPU-to-GPU copies require that peer access between the GPUs is turned on.
This header's functions assume that decision has **already** been made and
approved by the caller; they do not re-check whether the hardware can do peer
access.

The decision is a single gate evaluated once by the caller (the multi-GPU
driver, `compute_f2_blocks_multigpu`) before it ever calls into this file. Part
of that gate is the user's permission flag (`enable_peer_access` in the device
config): it is the user saying "yes, you may turn on direct GPU-to-GPU access."
Reaching either function in this header therefore means that permission was
granted, so these functions go ahead and enable peer access per peer without
asking again.

Turning peer access on for a pair of GPUs can report that it was "already
enabled" — for instance if a previous call left it on. That specific status is
**expected and not an error**; it is tagged and tolerated with a warning rather
than treated as a failure. A genuine peer-enable problem does not get swallowed —
it surfaces as a thrown error on the peer copy that follows.

---

## 6. Lifetime and ownership

These functions **read** the resident partial buffers where they sit. They do
**not** take ownership of the partials and do **not** free them.

The caller keeps its vector of partial handles alive across the whole call, and
those handles are freed only *after* the function returns. Because the function
synchronizes the stream (draining every DMA) before it returns, no copy is ever
still reading a partial's memory at the moment that partial is freed. In short:
the sources stay valid for the entire duration of the copies, and cleanup happens
strictly afterward.

The `partials` span is passed as **non-const** even though the handles are not
modified. This is deliberate: it signals the "consume in place" intent — the
resident buffers are being read directly out of the partials — rather than
implying the partials are an untouched, shareable input.

---

## 7. Preconditions

Both functions fail fast if the inputs do not describe a clean, disjoint tiling.
The checks are the same shape contract the host-staged combine enforces:

- There is exactly one partial per shard: `partials.size() == shards.size()`,
  and that count equals the number of GPUs, `G`.
- Every non-empty partial reports the same population count `P`.
- Each partial covers exactly the blocks its shard covers — the partial's local
  block count equals `shards[g].b1 - shards[g].b0`, and the partial's offset
  `b0` equals `shards[g].b0`.
- The shard block ranges, taken together, tile `[0, n_block_full)` contiguously
  with no gaps and no overlaps.

Any violation throws (`std::runtime_error`). An **empty** shard is allowed and is
not a violation: its handle has a local block count of zero and a null internal
pointer, meaning that GPU placed nothing.

The `shards` argument exists specifically so this tiling can be cross-checked
against the partials — the shard plan is the authoritative statement of which
blocks belong to which GPU, and the validation compares each shard's `b0`/`b1`
against the matching handle's offset and block count.

---

## 8. Parameters and return value

| Name | Meaning |
|---|---|
| `partials` | The `G` resident partial handles, in the fixed GPU order `g = 0 .. G-1`. Handle `g` is GPU `g`'s `[P × P × (shards[g].b1 - shards[g].b0)]` partial, living in that GPU's memory. An empty shard's handle carries a zero block count and a null internal pointer. The span is non-const to reflect that these buffers are read in place (see section 6). |
| `shards` | The block-aligned shard plan, kept so the tiling can be validated against the partials (see section 7). |
| `P` | Population count — the leading dimension of every result slab. |
| `n_block_full` | Total number of genome blocks in the combined result. |
| `root_device_id` | The combine root: the first GPU in the run's device list. This is the GPU the full result lives on and the one that pulls each other GPU's partial. |
| **returns** (`combine_f2_partials_resident`) | The full `[P × P × n_block_full]` host `F2BlockTensor`, bit-identical to what the host-staged combine would produce over the same partials and shards. |
| **returns** (`combine_f2_partials_resident_device`) | The full result still resident on the root GPU as a `DeviceF2Blocks` handle — same bytes, no host copy. |

### Errors

Both functions throw on a precondition violation (`std::runtime_error`) and on a
genuine CUDA fault — a failed allocation, the peer copy, the device-to-device
copy, or the final copy down to the host. The one non-fatal case is peer access
reporting that it was "already enabled," which is tolerated with a warning rather
than thrown (see section 5).

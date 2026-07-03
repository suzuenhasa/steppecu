# `device_f2_blocks.hpp` reference

## 1. Purpose

`src/device/device_f2_blocks.hpp` declares the owning handle for the main output of
the first computation phase: the full block-by-block `f2` result and its paired
variance, left sitting in GPU memory.

steppe's first phase computes, for every pair of populations, an `f2` distance
statistic, broken out separately for each genome block (the blocks used later to
estimate uncertainty by jackknife). The result is a cube of numbers with shape
`P × P × n_block` — one `P × P` slab per block — and there is a second cube of the
same shape holding the paired variance that goes with each `f2` entry. Together they
are what the second phase (the model fit) reads.

The important design point is that this result **stays on the GPU**. The producer
computes it directly into GPU memory and hands back this handle, which owns those
GPU buffers. The fit phase then reads them where they are, in GPU memory, with no
trip through host RAM. Copying the result down to the host is a separate, opt-in
step (`to_host()`), and it is the only place in the whole pipeline that copies this
result to host memory.

The header is deliberately free of any CUDA code. It names no CUDA types at all. The
actual GPU buffer owners live in a private implementation type (`Impl`) defined in a
companion `.cu`/`.cuh` file, and this header only holds a pointer to that type plus a
few plain host-side numbers describing the shape. That keeps the header light enough
to include from the CUDA-free orchestration and public-API layers, which can hold and
pass along the handle without pulling in the GPU headers.

---

## 2. The `DeviceF2Blocks` handle

`DeviceF2Blocks` is a move-only owner of the two resident GPU buffers (the `f2` cube
and its paired-variance cube) on a specific GPU.

### Move-only ownership

- It can be **moved** (move constructor and move assignment), which is how the
  producer hands the result out and how the handle is forwarded down the pipeline.
- It **cannot be copied** — the copy constructor and copy assignment are deleted.
  There is only ever one owner of the GPU buffers, so there is no accidental double
  ownership and no accidental second allocation.

### Lifetime

The GPU buffers were allocated on the GPU named by `device_id`. This handle owns
them and frees them in its destructor. The free is aware of which GPU the pointers
belong to, so the buffers are released on the correct device even if the current
GPU has since been switched.

In practice the handle is the handoff between the two phases: the producer builds
the result into GPU memory and moves the handle out; the fit phase reads the buffers
in GPU memory; nothing is copied to the host unless `to_host()` is explicitly called.

### The three constructor states

- A default-constructed handle is **empty** — no resident buffers. This is also what
  a degenerate (zero-work) result looks like and what a moved-from handle becomes.
- A move leaves the source handle in the empty state (its buffer pointer is nulled),
  though see the moved-from caveat in section 4.

---

## 3. Shape and block metadata

The handle carries a few plain host-side numbers describing the result. These are
ordinary integers and vectors — no GPU types — so the CUDA-free layers can read them
freely.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `P` | `int` | `0` | The population count. This is the leading dimension of every slab: each block's slab is `P × P`. |
| `n_block` | `int` | `0` | The number of genome blocks in the full result. This is the count for the whole tensor, not a shard or partial count. |
| `device_id` | `int` | `-1` | The GPU ordinal the `f2` and paired-variance buffers live on. `-1` means no device (empty or not-yet-resolved). |
| `block_sizes` | `vector<int>` | empty | The per-block SNP counts — how many SNPs went into each block. Length equals `n_block` (empty on a degenerate result). This is the metadata the later jackknife needs, kept on the host side. |

### `size()`

Returns the flat element count of one resident cube: `P × P × n_block`. It is a
convenience for sizing a copy or a buffer. The computation is done in a wide integer
type so it does not overflow for large problems, and a negative `n_block` is treated
as zero.

### `empty()`

Returns true for a degenerate or empty result — that is, when `n_block` or `P` is
zero or negative, meaning there are no resident buffers to speak of. Note the
important exception in section 4: `empty()` is **not** a reliable "do I own GPU
buffers?" check on a moved-from handle.

---

## 4. The resident device pointers and memory layout

Two accessors expose borrowed (non-owning) pointers into the GPU buffers the handle
owns. The fit phase reads these in GPU memory.

- `f2_device()` — pointer to the resident `f2` cube.
- `vpair_device()` — pointer to the resident paired-variance cube.

Both are defined in the companion `.cu` file, because reaching the pointers means
dereferencing the private `Impl`.

### Memory layout

Both cubes are stored **column-major** with shape `P × P × n_block`. The element at
row `i`, column `j`, block `b` is at flat offset:

```
i + P*j + P*P*b
```

So each `P × P` slab is contiguous, one slab per block, blocks laid end to end.

### The null-pointer check and the moved-from "husk" trap

Both accessors return a null pointer when the handle has no resident buffers — that
is, when the private `Impl` is null.

This matters because of a subtle trap. When a handle is moved from, the move nulls
out the `Impl` pointer (so the GPU buffers now belong to the destination), **but it
leaves the plain scalar fields behind** — the moved-from source can still report a
stale `P`, `n_block`, and `device_id`. Such a leftover handle is a "husk": it owns
nothing, yet `empty()` and `size()` would look at those stale scalars and wrongly
claim it has a real result.

The rule that follows: to decide whether a handle actually owns GPU buffers, check
that `f2_device()` (or `vpair_device()`) is non-null. Do **not** rely on `empty()` or
`size()` for that decision on a handle that might have been moved from.

---

## 5. Copying the result to the host — `to_host()`

`to_host()` is the one and only place in the entire pipeline that copies this result
from GPU memory down to host memory, and it is opt-in. It allocates a host-side
result object and copies both the `f2` cube and the paired-variance cube down, one
device-to-host transfer each.

Some practical details:

- **Pinned staging.** For the duration of each copy, the host destination is pinned
  (page-locked) so the transfer runs on the faster path, with a graceful fall back to
  ordinary pageable memory if pinning is unavailable. Pinning only affects speed —
  the same bytes are moved either way, so the result is identical.
- **Correct device, restored.** It switches to `device_id` to perform the copy and
  then restores whatever GPU the caller had selected, using a scope guard so the
  restore happens even on an error path.
- **Bit-identical.** The host object it returns is bit-for-bit the same doubles in
  the same layout that the original host-returning code path produced, before the
  result was made device-resident.

Because copying down is opt-in, most runs never call it. It is used only by the
parity/correctness test, by the planned on-disk cache, and by any explicit host-side
or command-line caller that genuinely wants the numbers on the host.

---

## 6. Uploading a host tensor back to the device — `upload_f2_blocks_to_device`

`upload_f2_blocks_to_device(host, device_id)` is the reverse of `to_host()`: it takes
a host-side result, allocates fresh GPU buffers for the `f2` and paired-variance
cubes on the given GPU, copies the host data up, and returns a `DeviceF2Blocks`
handle owning them. The upload is a raw byte copy, so it is bit-faithful.

It exists for one specific situation in the multi-GPU path. When the machine's GPUs
cannot copy directly to each other (no peer-to-peer support), the separate
per-GPU partial results have to be assembled together by routing through host memory.
After that host-side assembly, this function re-uploads the combined tensor so that
the phase's return value is still a device-resident `DeviceF2Blocks` handle, the same
type every other path returns. The host round-trip here is the cross-GPU assembly
transport, not a forced copy of the final output.

### Why multi-GPU is currently deferred

There is a standing note on this function worth understanding. This upload, paired
with the `to_host()` copy that feeds it, is the cost of replicating the result across
GPUs when peer-to-peer is unavailable. On real data with `P = 600` it was measured at
roughly 8.72 GB moved and about 3.8 seconds cold. That transport cost caps the
multi-GPU speedup at about 1.21× — never reaching the crossover where using more GPUs
would actually pay off.

The root cause is the lack of direct GPU-to-GPU copy on the consumer GPUs this was
measured on. Because of that ceiling, the multi-GPU path is deferred. The real fix is
not to tune this transfer but to remove it: have each GPU build its own `f2` result
independently, so there is no cross-GPU copy at all.

# `stream_f2_blocks.hpp` reference

## 1. Purpose

`src/device/stream_f2_blocks.hpp` declares one small data structure,
`StreamTarget`, that describes *where* a streamed f2-block computation should send
its results. It is the hand-off point between two layers that are deliberately kept
apart:

- The **orchestrator** (the `steppe::core` layer), which decides what to compute and
  which memory tier to use, but is written to contain no CUDA code at all.
- The **CUDA backend**, which does the actual GPU work.

When the f2 result is too large to keep entirely in GPU memory, the orchestrator
needs to ask the GPU backend to compute the result and *spill* it out block by
block — either into host RAM or all the way to a file on disk. `StreamTarget` is the
CUDA-free request object the orchestrator fills in and passes across that boundary.
It names the destination without ever naming a CUDA type.

The header pulls in only three lightweight dependencies (the tier enum, the disk
descriptor, and the host-tensor type) and no CUDA headers, so the CUDA-free
orchestrator can construct a `StreamTarget` without dragging in the GPU toolkit.

---

## 2. Background: the three memory tiers

The f2-block result is a large `[P × P × n_block]` array (populations by populations
by jackknife blocks). steppe stores it in the fastest place it will fit, chosen
automatically from how much GPU and host memory is free at runtime:

- **Resident** — the whole result fits in GPU memory and stays there. This is the
  original, unchanged path.
- **HostRam** — it does not fit in GPU memory but fits in host RAM, so the blocks are
  streamed out of the GPU into a host-side tensor.
- **Disk** — it fits in neither, so the blocks are streamed to a file on disk through
  a small staging buffer.

`StreamTarget` exists only for the two *streaming* tiers, HostRam and Disk. The
Resident tier never uses this seam: for a resident result the orchestrator calls the
device compute entry point directly and the whole result is left in place. This is
why every mention of Resident below is a "never reaches here" note rather than a
supported case.

The important guarantee across all three tiers is that the numbers are the same. The
streamed path reuses the resident path's per-block gather, matrix-multiply, and
assembly steps unchanged; the only difference is that each finished block's `[P × P]`
slab is written out immediately instead of being kept in GPU memory. Choosing a tier
changes *where* and *when* bytes land, never any value steppe reports.

---

## 3. Why this header is CUDA-free

The GPU backend writes streamed blocks through a triple-buffered "sink" object, and
that sink is a CUDA type. The orchestrator must not construct or even name it,
because the orchestrator is compiled without the CUDA toolkit.

`StreamTarget` resolves this by carrying only CUDA-free destinations:

- a pointer to a host-side result tensor for the HostRam tier, and
- a file path plus a plain descriptor object for the Disk tier.

The orchestrator fills in whichever of these the resolved tier needs and passes the
`StreamTarget` to the backend. The CUDA backend then reads the tier and builds the
matching concrete sink (a host-RAM sink or a disk sink) internally, on its own side
of the boundary. In other words, `StreamTarget` says *what destination is wanted* in
CUDA-free terms, and the backend privately translates that into the CUDA machinery
that fulfills it.

---

## 4. The `StreamTarget` struct

`StreamTarget` is a plain struct with four fields. Only the subset that matches the
selected tier is used; the rest stay at their defaults.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `tier` | `OutputTier` | `HostRam` | Which streaming tier this request is for. In practice the caller always sets this explicitly to `HostRam` or `Disk`; it selects which of the fields below the backend will read. `Resident` must never appear here. |
| `host_dst` | `F2BlockTensor*` | `nullptr` | The HostRam destination. When `tier` is `HostRam`, this points at the host tensor the backend's host-RAM sink streams the blocks into. The tensor is owned elsewhere (by the unified result object); `StreamTarget` only borrows a pointer to it. |
| `disk_path` | `std::string` | empty | The Disk cache file path. When `tier` is `Disk`, this is the file the backend's disk sink writes the blocks to. |
| `disk_dst` | `DiskF2Blocks*` | `nullptr` | The Disk descriptor to populate. When `tier` is `Disk`, the backend writes the cache file at `disk_path`, then reopens it read-only and fills in this descriptor (path, shape, and an open read handle) so the result can be read back block by block afterward. |

The pairing is strict:

- **`tier == HostRam`** uses `host_dst` (must be non-null); `disk_path` and
  `disk_dst` are ignored.
- **`tier == Disk`** uses `disk_path` and `disk_dst`; `host_dst` is ignored.
- **`tier == Resident`** is never a valid value here.

---

## 5. How the seam is used

The streamed compute entry point on the backend takes the genotype-derived inputs,
the jackknife block assignment, the precision policy, and a `StreamTarget&`. Only the
CUDA backend implements it; the base and any non-GPU backend deliberately do not
support streaming and will reject the call, because out-of-core block streaming is a
GPU-only concept.

The orchestrator's side of the contract works like this:

1. It picks a tier from the runtime free-memory figures.
2. For a Resident result it does **not** build a `StreamTarget` at all — it calls the
   ordinary device compute path and keeps the result in GPU memory.
3. For **HostRam**, it constructs a `StreamTarget`, sets `tier = HostRam`, and points
   `host_dst` at the host tensor inside the unified result object.
4. For **Disk**, it constructs a `StreamTarget`, sets `tier = Disk`, resolves the
   cache path (from the config field, then the `STEPPE_F2_CACHE_PATH` environment
   variable, then the built-in default), stores it in `disk_path`, and points
   `disk_dst` at the disk descriptor inside the unified result object.

The backend then validates that the tier and the matching destination agree — a
HostRam request must carry a `host_dst`, a Disk request must carry a `disk_dst`, and
a Resident value is rejected outright — before it builds the concrete sink and runs
the block loop. Because `StreamTarget` only ever borrows pointers into the result
object the orchestrator already owns, it carries no memory of its own and needs no
special lifetime handling beyond outliving the single compute call it is passed to.

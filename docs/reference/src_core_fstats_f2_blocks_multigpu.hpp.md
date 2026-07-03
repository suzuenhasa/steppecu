# `f2_blocks_multigpu.hpp` reference

## 1. Purpose

`src/core/fstats/f2_blocks_multigpu.hpp` is the entry point for computing the
per-block **f2 tensor** across every GPU in a machine at once. The f2 tensor is
the large three-dimensional result `[P × P × n_block]` — for each pair of the `P`
populations, one f2 value per genome block — plus a retained per-block
paired-variance ("Vpair") slab alongside it. This header is the seam that the
higher-level API calls when it wants that tensor computed and spread across the
`G` devices of a machine.

The design principle here is deliberately narrow. This layer does **not**
reimplement the per-block matrix-multiply that produces one device's slice of the
result. That math already exists in the single-GPU backend
(`CudaBackend::compute_f2_blocks`), and this layer calls it unchanged, once per
device. What this layer adds is three things:

1. **Sharding** — deciding which genome blocks each device computes.
2. **Concurrency** — driving all `G` devices at the same time instead of
   one after another.
3. **Combining** — assembling the per-device partial results back into one
   whole tensor, in a fixed device order, so the answer is bit-for-bit identical
   no matter how many GPUs ran it.

The header exposes three related functions, from lowest-level to
highest-level in what they return: a combined host tensor, a device-resident
handle, and an adaptive tiered result. They all compute the same f2 tensor;
they differ only in **where the result lands** and **how it is returned**.
Sections 4, 5, and 6 cover them in turn.

This header is intentionally CUDA-free (see section 8), so it compiles into the
core library without pulling in the GPU toolkit.

---

## 2. The parity guarantee

The single most important property of this file is that the multi-GPU result is
**bit-for-bit identical** to the single-GPU result over the same inputs, and
identical no matter how many GPUs (`G`) are used. Splitting the work across two,
four, or eight devices never changes a single bit of the reported tensor.

This is a hard guarantee, not a best-effort approximation. Every design choice in
the file exists to preserve it:

- Each genome block is computed **entirely on one device**, from exactly its own
  contiguous columns of input, so the bits that device produces for that block
  equal the bits the single-GPU path would have produced for the same block.
  (See section 3 for why this holds.)
- The per-device partial results are summed onto a **zero-initialized** full
  tensor in a **fixed device order** (device 0, then device 1, and so on). A
  fixed summation order is what makes floating-point addition reproducible.
- Running the devices concurrently does not affect the result. Each device's bits
  are fixed by which blocks it was assigned, not by when it happens to finish. The
  combine step reads the partials in the fixed device order only **after** all
  devices have finished (a join barrier), so concurrency and correctness are
  fully decoupled.

There are also two different ways to physically combine the partials — a portable
host-staged path (gather all partials to host memory, sum there) and an opt-in
faster path that copies device-to-device directly. Both sum in the same fixed
order, so they produce identical bits to each other and to a single-GPU run. The
choice between them only changes how bytes move, never the answer.

---

## 3. Block-aligned sharding

The parity guarantee in section 2 rests on one structural property:
**block-aligned sharding**. Understanding it explains why the multi-GPU path can
be bit-identical to the single-GPU path at all.

The inputs are the per-SNP `Q`, `V`, and `N` arrays, laid out column-major as
`[P × M]` (P populations, M SNPs). A shared partition assigns every SNP column to
a genome block. Crucially, that partition is required to be **non-decreasing in
block id** — block 0's columns come first, then block 1's, and so on — which means
each block occupies a single **contiguous run of columns**.

Because of that contiguity, a whole block can be handed to one device as a
zero-copy column sub-view: the device sees exactly the same contiguous columns for
that block that the single-GPU path would have seen. It computes that block's
slab from precisely the same input bytes, with the same math, so it gets the same
output bits. No block is ever split across two devices, so no block's result is
ever the sum of two separately-rounded partial GEMMs.

The sharding therefore assigns **whole blocks** to devices, never fractions of a
block. Each device's partial tensor is the set of blocks it owns; the rest of its
slots are zero. Summing the partials in fixed order reconstructs the full tensor
exactly.

---

## 4. `compute_f2_blocks_multigpu` — the combined host tensor

This is the original, most concrete entry point. It computes the full f2 tensor
across all `G` devices and returns it as a single combined `F2BlockTensor` (the
`[P × P × n_block]` result plus the per-block Vpair), already assembled in host
memory.

**When `G == 1`** (one device), this is exactly the existing single-GPU path with
zero behavior change: no shard, no combine. It calls the one device's
`compute_f2_blocks` over the full `Q`/`V`/`N` and partition and returns that
result unchanged, bit-for-bit.

**When `G >= 2`**, it plans the block shards, computes each device's sub-view
partial concurrently, and combines the partials in fixed device order. The devices
are driven **at the same time**, each on its own host thread (see section 7).

### Parameters

| Parameter | What it is |
|---|---|
| `resources` | The bundle of `G` devices. `gpus[g]` are in the fixed `g = 0..G-1` order that also fixes the combine order. Passed by non-const reference because driving each device's backend mutates that backend's device-side scratch memory; the backends are move-only and owned inside the bundle. |
| `Q`, `V`, `N` | The full per-SNP input arrays, column-major `[P × M]`. Each device receives a zero-copy column sub-view — no input is copied to split it. |
| `partition` | The shared SNP-to-block assignment. Its block ids must be non-decreasing, which is exactly what makes each block's columns contiguous and block-aligned sharding possible (section 3). |
| `precision` | The arithmetic policy for the per-device f2 matrix-multiplies. It is forwarded **unchanged** to every device, so all devices use the identical precision path (the same emulated-FP64 or native-FP64 choice). It governs only the per-device GEMMs. |

**Returns** the combined full `F2BlockTensor`.

The two other entry points below build on this same machinery but change what they
return and where the result physically lives.

---

## 5. `compute_f2_blocks_multigpu_device` — the device-resident result

This is the primary entry point and the one to prefer. It computes the same
per-block f2 tensor across the `G` devices but returns it as a **VRAM handle**
(`DeviceF2Blocks`) that stays on the GPU, rather than forcing the whole tensor
back down to host memory. There is no forced device-to-host copy and no host
allocation or zeroing of the full tensor.

The function `compute_f2_blocks_multigpu` in section 4 is now just a thin wrapper
around this one plus a final "copy to host" call — so callers that genuinely need
the tensor in host memory still get it, but callers that will consume it on the
GPU (the common case) can skip the round trip entirely.

Its behavior depends on the hardware:

- **`G == 1`** — the single device's full result stays resident on device 0. This
  is the headline win: **no device-to-host copy at all**.
- **`G >= 2` with peer access** (GPUs that can talk to each other directly) — each
  device's partial stays resident on its own card, and the partials are assembled
  device-resident on the root device into one `DeviceF2Blocks`, again with no
  final copy to host.
- **`G >= 2` on hardware without peer access** — assembling a single tensor that
  spans two cards fundamentally requires either direct GPU-to-GPU transfer or a
  bounce through host memory. On such hardware the per-device partials still stay
  **resident** while they compute (no premature copy down), and this entry then
  materializes exactly **one** host bounce, only because there is no direct fabric
  to assemble across the cards. The result is re-uploaded so the returned value is
  still a device-resident `DeviceF2Blocks`. The host bounce here is the assembly
  transport, not a forced output copy. This is a documented limitation of
  peer-less consumer hardware, not the general case.

---

## 6. `compute_f2_blocks_multigpu_tiered` — adaptive memory tiers

This entry point solves a different problem: what to do when the f2 tensor is too
large to fit in GPU memory. It returns an `F2BlocksOut`, an adaptive result that
lands in the **fastest storage tier it fits in**, chosen automatically at runtime
from how much free GPU memory and free host RAM are actually available (or pinned
by a config field / an environment override, mainly for tests).

The three tiers, fastest first:

| Tier | Name | When it is used | Where the result lives |
|---|---|---|---|
| 0 | Resident | The result plus its working set fit in free GPU memory | Entirely in GPU memory — the existing device-resident path, unchanged, with no streaming (streaming is opt-in only when needed) |
| 1 | HostRam | Does not fit GPU memory, but fits free host RAM | Streamed block-by-block into a host tensor through a triple-buffered handoff |
| 2 | Disk | Fits neither GPU memory nor host RAM | Streamed to a disk cache file through a small persistent staging buffer (friendly to machines with little RAM) |

This path is **parity-neutral**: reading the result back out (`to_host()`) gives a
byte-identical answer across all three tiers, and identical to the single-GPU
device-resident reference. The tier changes only **where and when** a slab of the
result lands — never its bits.

One current scope note: the tiered path always drives device 0 regardless of how
many GPUs exist. It fails fast only if there are zero devices, and it selects its
tier from device 0's free memory. Spreading the streamed computation across
multiple GPUs is a planned follow-on, not yet implemented.

---

## 7. Concurrent per-device execution

When `G >= 2`, the `G` per-device matrix-multiplies are driven **concurrently** —
one host thread per device, each thread driving its own device's backend and
writing into its own pre-sized partial-result slot. All threads are joined before
the combine step runs.

This concurrency is the actual source of the multi-GPU speedup. Each device's
`compute_f2_blocks` call is **blocking and self-contained** — it owns its device,
its stream, and its library handle. Issuing them one at a time would serialize
work that the hardware is perfectly capable of overlapping; running device 1 only
after device 0 finished would waste device 1's idle time. Fanning them out means
the wall-clock time is the time of the **slowest single device**, not the **sum**
of all device times.

The fan-out does not affect the result (see section 2): each device's output bits
are fixed by which blocks it was assigned, independent of execution timing, and
the combine reads the partials in the fixed device order only after the join
barrier. The combine step itself moves very little data (kilobytes to megabytes),
so it sits off the bandwidth-critical path and deliberately stays serial.

---

## 8. Why the header is CUDA-free

This header lives in the core library namespace and is **host-pure**: it contains
no CUDA code and does not need the GPU toolkit to compile. It names only
toolkit-free types — the device bundle, the input views, the block partition, the
precision policy, the result tensor, the device-resident result handle, and the
adaptive tiered result — plus the host-staged combine path.

That matters for build layering. Because the declarations here don't require the
CUDA compiler, this file compiles into the core library on any machine, and the
higher layers that call it don't transitively pull in the device toolkit just by
including it. The actual GPU work happens inside the backend implementations,
which are a separate compilation unit.

The host-staged combine is the portable baseline that always works. The faster,
opt-in device-to-device combine (which uses direct GPU-to-GPU copies) lives in a
separate unit and is bit-identical to this baseline by construction — same fixed
summation order, same result (section 2).

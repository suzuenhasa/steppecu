# `f2_blocks_multigpu_core.hpp` reference

## 1. Purpose

`src/core/fstats/f2_blocks_multigpu_core.hpp` declares the host-only, CUDA-free
heart of the multi-GPU f2-block precompute.

Some background on what that precompute produces. The f2 statistic is computed for
every pair of populations, and it is computed separately for each *jackknife block*
(a contiguous stretch of SNPs used to estimate uncertainty). So the full result is a
three-dimensional tensor of shape `[P × P × n_block]`, where `P` is the number of
populations and `n_block` is the number of jackknife blocks. On a single GPU one
kernel pass fills the whole tensor.

When steppe has two or more GPUs, the work is split across them: the blocks are
partitioned into contiguous groups, each GPU computes the sub-tensor for its own
group of blocks (its **partial**), and the partials are later merged (the
**combine**) back into the one full `[P × P × n_block]` tensor. This header declares
the pure-C++ steps of that flow that contain no GPU code:

1. **Plan the split** — decide which blocks go to which device
   (`plan_multigpu_shards`).
2. **Fan the work out** — run all devices concurrently, each producing its own
   partial. Three variants exist, differing only in *where* each device's partial
   lands (`compute_multigpu_partials`, `compute_multigpu_partials_resident`,
   `compute_multigpu_partials_into`).

The final combine step is **not** declared here. The combine is the one piece that
must reference a CUDA device symbol (the peer-to-peer sum kernel), so it stays with
the public entry point (`compute_f2_blocks_multigpu`) rather than in this CUDA-free
core. The entry point is a thin composition: it calls the planner declared here, then
one of the fan-out functions declared here, then the combine.

Everything declared here lives in the `steppe::core` namespace and names only
CUDA-free types, so it compiles into the core library without the GPU toolkit
present.

---

## 2. Why this lives in its own CUDA-free header

The multi-GPU algorithm has a host-pure heart — the shard plan and the concurrent
fan-out — and one small part that genuinely needs CUDA (the peer-to-peer combine).
Separating the two into different files is deliberate.

The host-pure heart includes several pieces of real logic worth testing on their
own: the zero-copy column sub-view math, the transform that renumbers block IDs into
a dense zero-based range per device, the one-thread-per-device fan-out, the way a
worker's exception is captured and re-thrown, and the handling of empty shards (when
there are fewer blocks than devices). All of that depends only on CUDA-free seams —
the abstract compute-backend interface, the CUDA-free shard planner, and the
core block-range helper — and never on the GPU peer-to-peer symbol.

Keeping these functions in a CUDA-free header lets a GPU-free unit test drive them
against a **fake** compute backend with no GPU, no CUDA toolkit, and no device
linking step. That test is the fast inner-loop guard that the slow full-GPU parity
test cannot be, because the parity test has to link the real CUDA backend and the
peer-to-peer combine. The CUDA-free property is not just a convention: if CUDA code
leaked into this header, the host-only test build would fail to compile, which is
itself the proof that the layering holds.

Splitting these declarations out of the entry point was a pure refactor — the entry
point still calls them in the identical order over the identical inputs, so the
partials and the shard plan are byte-for-byte what the older inlined code produced,
and the combine still reads the partials in the fixed device order after the join.
The bit-for-bit results did not change.

---

## 3. `plan_multigpu_shards` — building the block-aligned split

```cpp
std::vector<device::DeviceShard> plan_multigpu_shards(
    const BlockPartition& partition, long M, int n_block, std::size_t G);
```

Builds the plan that says which jackknife blocks each device owns. It inverts the
SNP-to-block assignment (turning "SNP *i* belongs to block *b*" into "block *b*
spans SNP columns `[s0, s1)`") using the single shared block-range helper, which
validates the partition contract once, then hands those per-block column ranges to
the CUDA-free shard planner — the one and only place the block-to-device mapping
lives.

It returns exactly `G` shards, one per device, in device order `0 … G-1`. The shards
are contiguous and together tile the whole block range `[0, n_block)` with no gaps
and no overlap. When there are fewer blocks than devices (`n_block < G`), the extra
trailing shards come back **empty** (their block range is zero-width) rather than
being dropped — the count of returned shards always equals `G`.

Each returned `DeviceShard` carries a block range (`b0 … b1`) and the matching SNP
column range (`s0 … s1`) that those blocks cover. `b0` in particular is the global
block offset a device uses both to renumber its local blocks and, later, to place its
partial back into the full tensor.

This is a pure function of `(partition, M, n_block, G)` — it touches no device, no
resources, and no CUDA. Its only real input to the planner is the list of per-block
column ranges; each block's SNP count is simply the size of its range, so no separate
"block sizes" vector is needed.

**Parameters**

| Parameter | Meaning |
|---|---|
| `partition` | The shared SNP-to-block assignment. Its block IDs are non-decreasing, which is what guarantees each block's columns are contiguous. |
| `M` | The SNP count (equal to the width of the `Q` matrix). |
| `n_block` | The number of blocks (equal to `partition.n_block`). |
| `G` | The number of devices. Must be at least 1. |

**Throws** — a runtime error if `G` is 0 (from the planner), or if the partition is
malformed (from the block-range helper).

---

## 4. The shared concurrent fan-out

The three compute functions in this header (sections 5, 6, and 7) all share the same
skeleton. They differ only in where each device writes its partial. Understanding the
skeleton once explains all three.

**One thread per device, running at the same time.** Each function spawns one
`std::jthread` per device and lets all `G` devices compute concurrently. Running the
devices in parallel is the whole point of the multi-GPU path — it is where the
speedup comes from.

**Zero-copy column sub-views.** The full per-SNP inputs `Q`, `V`, and `N` are stored
column-major as `[P × M]` matrices. A device that owns SNP columns `[s0, s1)` does
**not** get a copy of that data — it gets a sub-view, which is just the original
pointer advanced by `P * s0` columns together with the sub-width. No data is copied
to hand a device its slice of the input.

**Dense, zero-based local block IDs.** Inside its shard, each device renumbers the
blocks so they start at 0. It builds a local block-ID array where every entry is the
global block ID minus its shard's starting block `b0`. That way each device's backend
sees a self-contained problem numbered from zero, independent of where the shard sits
in the global block range.

**Each device drives its own backend on its own device.** Worker `g` calls
`resources.gpus[g]`, which is bound to device `g`. Because the active CUDA device is a
per-host-thread property, each worker sets and uses its own device with no
interference from the others. There is no shared mutable state: every worker writes
only its own pre-sized output slot (distinct vector elements, never resized or
aliased) and builds its own local block-ID array.

**Empty shards do nothing.** A shard whose block range is zero-width (`b0 == b1`)
hands the backend a local block count of 0 and a local SNP count of 0; the backend
early-returns an empty result, and the later combine places nothing for that device.
This is how the fewer-blocks-than-devices case is handled cleanly.

**Parity is neutral.** The per-device matrix-multiply work is fixed entirely by the
block-aligned shard and is independent of which wall-clock moment a worker happened to
run in. So the concurrent fan-out produces bit-for-bit the same numbers as running the
devices one after another would — and the same numbers a single-GPU run would. The
returned partials are always in the fixed device order `0 … G-1`, and the caller
combines them **after** this function returns. The thread join is the barrier that
guarantees every partial is finished before the fixed-order combine reads it.

**Exception safety is deterministic.** A `std::jthread` whose worker function lets an
exception escape immediately calls `std::terminate` (a hard crash). To avoid that,
each worker catches *everything* into its own `std::exception_ptr`. After all workers
have joined, the function re-throws the **first** captured exception — the one from
the lowest-numbered device that failed. So a backend or device fault surfaces as an
ordinary C++ throw, and *which* exception you get is deterministic: it does not depend
on which worker happened to race to fail first.

**The three variants at a glance**

| Function | Where each device's partial lands | Returns |
|---|---|---|
| `compute_multigpu_partials` | Copied back to host as a separate tensor per device | `G` host-side `F2BlockTensor`s |
| `compute_multigpu_partials_resident` | Left resident in each device's own memory (no copy back) | `G` opaque `DevicePartial` handles |
| `compute_multigpu_partials_into` | Copied directly into disjoint slices of one shared, caller-owned buffer | nothing (writes through pointers) |

---

## 5. `compute_multigpu_partials` — host-staged partials

```cpp
std::vector<F2BlockTensor> compute_multigpu_partials(
    device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const device::DeviceShard> shards,
    const Precision& precision);
```

The baseline variant. Each device computes its `[P × P × (b1 - b0)]` partial through
the unmodified compute-backend method and copies it back to host memory as its own
`F2BlockTensor`. The full backend computation is reused as-is, not reimplemented.

Returns `G` compact partials in device order — `partials[g]` corresponds to
`shards[g]`. The caller (the entry point) then hands these to the host-staged combine,
which reads them in order and places each one into the full tensor.

**Parameters**

| Parameter | Meaning |
|---|---|
| `resources` | The device bundle. `gpus[g]` drives device `g`. Non-const because each backend call mutates that device's scratch memory. |
| `Q`, `V`, `N` | The full per-SNP inputs, column-major `[P × M]`. Each device receives a zero-copy column sub-view. |
| `partition` | The shared SNP-to-block partition, used to build each device's dense local block IDs. |
| `shards` | The block-aligned plan from `plan_multigpu_shards` (length `G`). |
| `precision` | Forwarded unchanged to every backend call, so all devices use the identical arithmetic. |

**Throws** — re-throws the first worker failure (a runtime error on a malformed
sub-partition, or a CUDA error on a device fault).

---

## 6. `compute_multigpu_partials_resident` — device-resident partials

```cpp
std::vector<device::DevicePartial> compute_multigpu_partials_resident(
    device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const device::DeviceShard> shards,
    const Precision& precision);
```

The device-resident sibling of section 5. It runs the exact same concurrent fan-out —
same zero-copy sub-views, same dense local block IDs, same exception handling, same
empty-shard treatment — but each worker calls the *resident* backend method, which
**leaves** the device's f2 and paired-variance partial sitting in that device's own
memory. There is no copy back to host and no free; the method returns an opaque,
move-only `DevicePartial` handle that owns the resident buffers.

Each handle is *moved* into its pre-sized output slot, which is just a pointer swap
with no CUDA call, so it is safe to do from any thread. The handles survive the thread
join: the returned vector outlives the workers, and the resident device buffers are
freed only *after* the device-resident combine has consumed them.

The point of this variant is to avoid an unnecessary round trip. The host-staged
baseline copies every partial down to host memory, and the peer-to-peer combine then
has to move data around again; keeping the partials resident lets the combine read
them directly on the devices, which removes an extra copy that was the main multi-GPU
speed wall. The returned handles feed the resident combine.

This function is still CUDA-free at the header level: the `DevicePartial` type is an
opaque handle with no CUDA in its interface, and the resident backend method is a
virtual on the CUDA-free backend interface. So it compiles into the core library
without the GPU toolkit, exactly like the host-staged sibling.

**Parameters** — identical to section 5, with one note: `shards[g].b0` is the global
block offset that gets carried on the handle so the later combine knows where to place
that device's partial.

**Throws** — re-throws the first worker failure (a runtime error on a malformed
sub-partition, or a CUDA error on a device fault).

---

## 7. `compute_multigpu_partials_into` — direct into one shared buffer

```cpp
void compute_multigpu_partials_into(
    device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const device::DeviceShard> shards,
    double* dst_f2, double* dst_vpair, int* block_sizes_dst,
    const Precision& precision);
```

The direct variant. It runs the same concurrent fan-out as section 5, but instead of
returning a per-device tensor, each worker copies its compact f2 and paired-variance
results (through pinned host memory) **directly** into its own disjoint slice of one
big shared result buffer that the caller pre-allocated. There is no per-device
`F2BlockTensor` and no separate combine copy — the result is assembled in place as
each worker writes.

Concurrent writes into a single buffer are safe here because the shards are disjoint
and block-aligned: device `g` writes only the slice `[slab * b0, slab * (b0 + nb))`
(where `slab` is the `P × P` per-block stride), and no two shards' slices overlap. The
`G` workers therefore write non-overlapping regions with no synchronization needed.
The result is parity-neutral — the same bytes at the same offsets, in the fixed device
order regardless of wall-clock timing.

**Parameters** (in addition to the shared `resources` / `Q,V,N` / `partition` /
`shards` / `precision` described in section 4):

| Parameter | Meaning |
|---|---|
| `dst_f2`, `dst_vpair` | The shared result base pointers for the f2 and paired-variance tensors. The caller (the orchestrator) must pre-size each to `P * P * n_block_full`. |
| `block_sizes_dst` | The shared per-block-size base, pre-sized by the caller to `n_block_full`. |

**Throws** — re-throws the first worker failure (the lowest-numbered device that
failed), deterministically.

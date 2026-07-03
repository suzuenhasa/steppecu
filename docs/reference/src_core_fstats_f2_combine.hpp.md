# `f2_combine.hpp` reference

## 1. Purpose

`src/core/fstats/f2_combine.hpp` declares one function,
`combine_f2_partials_host`, which stitches together the per-GPU pieces of an f2
computation into a single complete result.

When steppe spreads an f2 computation across several GPUs, each GPU works on its
own slice of the genome — a contiguous range of jackknife blocks — and produces a
**compact** partial result covering only that slice. This function takes those `G`
compact partials (one per GPU) and copies each one into its correct place inside a
single full-shape result tensor, so the caller ends up with one tensor that looks
exactly as if a single GPU had computed the whole thing.

An f2 result is a tensor of shape `P × P × n_block`: for every pair of the `P`
populations and every one of the `n_block` genome blocks, it stores an f2 statistic.
Each GPU's compact partial has the same `P × P` face but only as many blocks as that
GPU was assigned. The combine assembles the full block dimension by placing the
partials side by side along it.

The header pulls in only two lightweight, CUDA-free declarations — the public
`F2BlockTensor` type and the `DeviceShard` sharding plan. It contains no GPU code at
all, so it compiles into the core library and can be unit-tested on a machine with no
GPU. This function is also the **only** combine path available on a machine whose
GPUs cannot directly access each other's memory; the faster GPU-to-GPU combine is a
separate opt-in unit (see section 6).

---

## 2. Why the fixed device order matters

The single most important property of this combine is that it visits the GPUs in a
**fixed order** — GPU 0, then GPU 1, and so on up to GPU `G−1` — regardless of how
many GPUs there are or which ones they are. That fixed order is exactly what makes
the combined result **bit-for-bit identical** in three different situations:

- across different GPU counts (the same data split two ways versus four ways gives
  the same bits),
- against the single-GPU reference (one GPU computing the whole thing),
- and against the faster GPU-to-GPU combine path (section 6), which places the same
  shards in the same fixed order.

The order in the fixed sequence comes from the caller's device list, which is pinned
once when resources are set up and never reshuffled.

This is why the combine is a deliberate ordered placement and **not** a general
"all-reduce" style sum across GPUs. An all-reduce combines partials in whatever order
the GPUs happen to finish or however the collective library schedules them, and that
order changes with the number of GPUs. Because floating-point addition is not
perfectly associative, a different combine order can produce different low bits.
Placing each piece into its own fixed slot sidesteps that entirely.

---

## 3. Placement, not summation

Because the GPU slices are aligned to block boundaries, no two GPUs ever own the same
block — the block ranges are **disjoint** and together tile the full block range with
no gaps and no overlaps. Every block of the final tensor is written by exactly one
GPU, exactly once.

That disjointness is what lets the combine be a plain copy rather than an addition.
Each GPU owns a single contiguous run of blocks, so filling in that GPU's contribution
is just one bulk copy per array (a `memcpy`-grade operation): one copy of the f2
values, one of the paired-variance values, and one of the per-block SNP counts. There
is no arithmetic and no accumulation.

### The negative-zero subtlety

Copying, rather than adding onto a zero-initialized buffer, is not just a performance
choice — it is required for exact bit-parity, because of how IEEE-754 floating point
treats negative zero.

The single-GPU reference computes each slab of the tensor **directly**; it never adds
a slab on top of a pre-existing zero. So if the true value of some element is negative
zero (`−0.0`), the reference stores that exact `−0.0` bit pattern. A faithful copy
reproduces it exactly.

Adding onto a zero-initialized buffer would not. Under the default rounding mode,
`(+0.0) + (−0.0)` evaluates to `+0.0`, which is a *different* bit pattern from `−0.0`.
So `x + 0.0 == x` holds for every ordinary finite number, but it silently flips a
`−0.0` element to `+0.0`. Doing a straight copy avoids that flip in every case,
including that one. This is why the combine copies each partial into place instead of
accumulating it.

The full tensor is created zero-initialized (with `+0.0`), so any block that no GPU
owns would keep that `+0.0`. But the disjoint tiling guarantees no such orphan block
exists on the real path — every block has exactly one owner and is overwritten by the
copy.

### All three arrays are copied

The per-block SNP count array is copied straight from each partial, not recomputed on
the host. Each GPU already counted the SNPs in each of its blocks while doing its
compute, and because the blocks are disjoint, a GPU's local count for a block is
exactly that block's global count. So the host has nothing to recompute; it just
copies the counts into place alongside the f2 and variance values.

---

## 4. The `combine_f2_partials_host` function

```cpp
[[nodiscard]] F2BlockTensor combine_f2_partials_host(
    std::span<const F2BlockTensor> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full);
```

It returns a full `P × P × n_block_full` tensor, bit-for-bit identical to what a
single GPU would have produced over the same complete input.

| Parameter | Meaning |
|---|---|
| `partials` | The `G` compact partial tensors, in fixed GPU order — element `g` is GPU `g`'s result covering just its assigned blocks. A GPU that was assigned no blocks has an empty partial (zero blocks), which is skipped. |
| `shards` | The block-aligned sharding plan, one entry per GPU. Entry `g` records where GPU `g`'s blocks begin in the full tensor, which is the offset the combine copies GPU `g`'s partial to. |
| `P` | The population count — the leading dimension shared by every slab. It must equal the `P` of every non-empty partial. |
| `n_block_full` | The total number of blocks in the combined tensor. |

The result is marked `[[nodiscard]]` because the combined tensor is the entire point
of the call; ignoring the return value is always a mistake.

---

## 5. Preconditions and failure behavior

The function checks its inputs up front and throws `std::runtime_error` if any of the
following is violated, rather than producing a silently wrong tensor:

- **Matching counts** — the number of partials must equal the number of shards (both
  equal `G`, the GPU count).
- **Consistent `P`** — every non-empty partial must share the same population count
  `P` that was passed in.
- **A clean tiling** — the shards' block ranges must tile the full block range
  `[0, n_block_full)` contiguously, with no gaps and no overlaps.
- **Each partial matches its shard** — partial `g` must span exactly as many blocks as
  its shard claims (its block count must equal the shard's end minus its start).

These are fail-fast checks: a violation throws immediately, before any copying, so a
malformed sharding plan can never corrupt the output.

---

## 6. Relationship to the peer-to-peer combine

There are two combine implementations, and they are guaranteed to agree bit-for-bit.

This host-staged version gathers every GPU's partial into host memory and copies each
into place there. It is the portable baseline: it works on any machine, including one
whose GPUs cannot directly read each other's memory, and it is what runs whenever the
faster path is unavailable or disabled.

The other version lives in a separate GPU unit and does a direct device-to-device
combine: one GPU pulls each other GPU's partial across a direct copy. It is the opt-in
fast path, chosen when the hardware supports direct GPU-to-GPU access and the caller
has not disabled it.

Both paths do the same thing conceptually — place the same fixed-order shards onto a
zeroed full-shape accumulator — so they produce identical bits, both to each other and
to a single-GPU run. The choice between them only affects how the bytes travel, never
the numbers that come out.

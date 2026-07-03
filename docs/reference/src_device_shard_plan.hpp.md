# `shard_plan.hpp` reference

## 1. Purpose

`src/device/shard_plan.hpp` defines the planning step that decides which parts of
the genome each GPU works on when steppe splits its precompute across several GPUs
on one machine. Its one job is to take the list of jackknife blocks (contiguous
runs of SNP columns) and divide them among the available devices, so that each
device is handed a fair, contiguous slice of work.

Two things make this file more than a simple divide-by-N:

1. It never splits a block across two devices. Each whole block is computed
   entirely on one device. This is the single property that lets the multi-GPU run
   produce bit-for-bit identical results to a single-GPU run (explained in
   section 2).
2. It balances by SNP count, not by block count. Blocks vary in size, so handing
   each device an equal *number* of blocks would leave the matrix-multiply work
   lopsided. The planner instead aims to give each device roughly the same *number
   of SNPs* (explained in section 4).

The header contains no GPU code and includes no CUDA headers. It names only a
plain C++ block-range type and standard-library types. That keeps it compilable
into the core library and the tests without pulling in the CUDA toolkit. It lives
in the device layer only because it is internal plumbing for the multi-GPU
orchestrator — it is not part of the public API, and it is shared between that
orchestrator and the test that checks multi-GPU parity. It touches no GPU itself.

---

## 2. Why the plan is block-aligned (the parity guarantee)

steppe promises that running a computation across several GPUs gives the exact same
numbers as running it on one GPU — bit for bit. This file is the foundation that
promise rests on.

The reason a block must be computed entirely on one device is a chain of "same
input, same output" steps:

- A block is a fixed set of SNP columns. If one device owns that whole block, it
  sees exactly the same SNP columns that a single-GPU run would see for that block.
- Same columns means the same underlying genotype bits feed the computation.
- Same bits flow into the same size-bucketing and the same batched matrix-multiply
  work.
- So that block's partial result is identical to what a single GPU would have
  produced for it.

If a block were ever split across two devices, each half would be a *different*
computation than the whole, and the results would no longer match bit-for-bit. So
"never split a block" is not an optimization — it is the invariant the parity
guarantee depends on.

The second half of the guarantee is that the per-device partial results are always
combined in the same fixed order (device 0, then device 1, and so on) onto a
zeroed-out full result. Because the plan is a deterministic pure function (see
section 4), a given block always lands on the same device on every run, so its bits
are stable, and the fixed-order sum of stable partials is itself stable. Together,
block-alignment and fixed-order combining are what make the multi-GPU result
reproducible and identical to the single-GPU result.

---

## 3. The `DeviceShard` struct

A `DeviceShard` describes the slice of work assigned to one device. It records the
assignment in two matching forms: a range of block ids and the range of SNP columns
those blocks cover.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `b0` | `int` | `0` | The first block id this device owns (inclusive). |
| `b1` | `int` | `0` | One past the last block id this device owns (exclusive). |
| `s0` | `long` | `0` | The first SNP column this device owns (inclusive). Equals the start of block `b0`. |
| `s1` | `long` | `0` | One past the last SNP column this device owns (exclusive). Equals the end of block `b1-1`. |

Both ranges are half-open — they include the low end and exclude the high end,
the usual `[begin, end)` convention.

### Why the column range is always contiguous

Because block ids are handed out in non-decreasing order, a contiguous range of
blocks always maps to a contiguous range of SNP columns. That matters because the
multi-GPU orchestrator can then take a device's slice as a plain zero-copy sub-view
of the full genotype matrix (a pointer offset and a width), rather than gathering
scattered columns. `s0` is simply the starting column of the first owned block, and
`s1` is the ending column of the last owned block.

### The `empty()` helper

`empty()` returns true when `b0 >= b1`, meaning the device owns no blocks. An empty
shard always has `s0 == s1` as well, so it is a clean no-op: a device handed an
empty shard early-returns an empty partial result, and the combine step places
nothing for it. This is what makes the "more devices than blocks" case (section 5)
harmless — the extra devices simply get empty shards.

---

## 4. `plan_block_shards` — how the partition is computed

```
std::vector<DeviceShard> plan_block_shards(
    std::span<const steppe::core::BlockRange> ranges,
    std::size_t G);
```

`plan_block_shards` partitions the blocks into `G` contiguous shards, one per
device, balanced by SNP count. It returns exactly `G` `DeviceShard` entries, where
entry `g` is the slice device `g` owns, in order from device 0 to device `G-1`. The
non-empty ranges are contiguous and together cover every block from the first to the
last.

### Inputs

- `ranges` — the per-block column ranges (the sole description of the partition).
  Its length is the number of blocks. Each block `b` contributes a column range
  whose size (`end - begin`, always zero or greater) is that block's SNP count.
  This single array is the source of both the balancing math *and* each shard's
  `[s0, s1)`. There is deliberately no separate "block sizes" array — a size is just
  the range's width — so there is no parallel array that could drift out of sync.
- `G` — the number of devices, which the caller resolves from the detected device
  count. It must be at least 1.

### Deterministic pure function

The output depends only on `(ranges, G)`. It does not depend on how fast the devices
are, which device currently holds which data, or anything else about the runtime. The
same partition and the same `G` always produce the same plan. This determinism is
exactly what the parity guarantee in section 2 needs: a block lands on the same
device every run.

### Balancing by SNP count

The planner targets roughly `total_snps / G` SNPs per device. This target is
derived from the inputs — it is not a hardcoded number. The algorithm is a single
greedy pass: it walks the blocks in order, accumulating SNP counts into the current
device's range, and closes that device's range once its running SNP total crosses
the target, then moves on to the next device. It never splits a block — a block is
always assigned whole, preserving the block-aligned invariant from section 2.

Balancing by SNP count rather than block count matters because blocks are not all
the same size. The heavy matrix-multiply work scales with SNP count, so equalizing
SNP counts is what actually balances the load across devices; equalizing block
*counts* would leave the work lopsided.

Every block ends up in exactly one device's range, the ranges are contiguous, and
they tile the full span of blocks with no gaps and no overlap.

---

## 5. Edge cases and error handling

The function is written to fail fast on a genuine error and to degrade cleanly on
the harmless boundary cases.

| Case | Behavior |
|---|---|
| `G == 0` | Throws `std::runtime_error`. A shard plan needs at least one device. In practice the caller resolves `G` from the detected device count, which is guaranteed to be at least 1, so this only fires on misuse. |
| `G == 1` | Returns a single range covering every block (and therefore every column). This is exactly the single-GPU slice, so running the multi-GPU path with one device is a structural no-op. (The orchestrator also special-cases one device before it even plans, but the plan is correct on its own either way.) |
| `n_block == 0` (no blocks) | Returns `G` empty shards (`{0,0,0,0}`). There is nothing to compute and nothing to combine. |
| `n_block < G` (fewer blocks than devices) | The first several devices each get one block (or a contiguous few), and the trailing devices get empty shards (`b0 == b1`). A device handed an empty shard early-returns an empty partial, so the extra devices cost nothing. |

The only thrown error is the `G == 0` case; every other boundary resolves to a valid
plan.

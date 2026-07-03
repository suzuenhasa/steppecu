# `cuda_backend_f2_blocks.cu` reference

## 1. Purpose

This file is the part of the CUDA backend that computes the **f2 statistic** on
the GPU. f2 is a population-genetics quantity: for every pair of populations it
measures the average squared difference in allele frequency across many genetic
markers (SNPs). The output is a P×P matrix (P = number of populations) together
with a companion "pairwise variance" matrix that later stages use for
uncertainty estimates.

The file produces f2 in two shapes:

- **One matrix over all SNPs at once** — `compute_f2`.
- **A stack of matrices, one per genome block** — the "block tensor" family.
  Splitting the genome into blocks is what lets the rest of the pipeline
  estimate uncertainty with a block jackknife (recompute leaving out one block
  at a time).

The block tensor can be delivered three ways, depending on how big the problem
is: kept entirely in GPU memory (fastest), copied back to host RAM, or streamed
all the way to disk. Two internal engines do the real work — `run_f2_blocks_resident`
builds the whole tensor in GPU memory, and `stream_f2_blocks_impl` produces it
block-by-block and spills each block out so the full tensor never has to fit in
GPU memory at once.

Everything here was moved verbatim out of a larger backend file when it was split
up; none of the math, precision handling, or ordering changed in the move. This
file is also, by C++ rules, the home of the backend class's first out-of-line
virtual method, which makes it the anchor point the compiler uses for the class's
type information — a build detail, not a behavioral one.

A theme that recurs in every path below is **parity-neutrality**: the performance
machinery moves or reorders *bytes*, never *values*. All three tiers and the
whole-matrix path produce bit-for-bit identical f2 numbers. Section 7 collects
the invariants that make that true.

---

## 2. Shared block-layout and size-bucketing helpers

Both block engines open with the exact same two preparation steps. So the two
paths can never drift apart, those steps live once here in the file's private
(anonymous) namespace and are called from both. All three helpers are pure
host-side index arithmetic — no GPU work — and are byte-for-byte identical to
what the two engines used to do inline.

### `Bucket`

A small struct describing one size bucket. Every block in a bucket is padded to
the same width `s_pad` and fed to the GPU as one batched matrix-multiply call,
padded only up to that width. Fields: `s_pad` (the padded width) and `blocks`
(the list of block indices in this bucket).

### `BlockLayout` and `compute_block_layout`

A block is a contiguous run of SNP columns in the input. `compute_block_layout`
turns the caller's per-SNP block-id array into, for each block, its first column
(`block_offsets`) and its column count (`block_sizes`). It does this by calling
one shared routine that validates the partition once — the ids must be in range
and non-decreasing, and there must be enough of them — which closes off the
out-of-bounds reads and writes a hand-rolled scan used to risk on a malformed
partition. `block_sizes` also doubles as the per-block weight denominator the
later stages need.

### `size_buckets`

Groups blocks by rounding each block's size up to the next power of two (the base
is `kBlockGroupPadBase`, which is 2). Each group becomes one batched call, padded
only to that group's width. Rounding to powers of two keeps the wasted padding
inside a group under 2× while keeping the number of batched calls small
(proportional to the log of the largest block). The buckets are sorted
smallest-first, which is cosmetic. The padding columns carry zeros, so they
contribute nothing to the result.

---

## 3. compute_f2 — a single f2 matrix over all SNPs

`compute_f2` computes one f2 matrix using every SNP in a single pass. It is the
simplest path and the one that anchors the class's type information.

### Degenerate input

If P or M is zero or negative, it returns an empty result immediately (carrying
the given P). This matches the fail-fast behavior of the sibling routines and,
importantly, avoids feeding a zero-sized problem deep into the GPU libraries,
where it would surface as a confusing driver or matrix-library error instead of a
clean empty answer. It is exactly the shape the multi-GPU split needs when a
device is handed an empty share of SNPs.

### The INT_MAX guard

This path issues its three matrix multiplications over all M SNPs in one shot.
The matrix-multiply library call used here takes the contraction length as a
32-bit signed int, but M is deliberately stored as a 64-bit value so a very large
SNP block cannot overflow. Because this path does not tile — it uploads all M —
a run with more than about 2.1 billion SNPs (INT_MAX) would silently truncate
that length, and the library would either reject it or, worse, quietly compute f2
over fewer SNPs (a wrong-but-plausible answer). So the routine fails fast with a
descriptive error before any GPU allocation. The block path is not affected,
because it contracts over one block's small padded width, not all M.

### Pipeline

After the guards: upload the three input matrices (the raw genotype-derived Q, V,
N, stored column-major P×M); run the "feeder" kernel that turns them into the
masked Q, masked V, and a stacked helper matrix; run the three matrix multiplies;
run the assemble kernel that combines them into the final f2 and pairwise-variance
matrices; copy those two P×P results back to host; synchronize.

### Pinning choices

The three input matrices are pinned (page-locked) once through a cache so their
host-to-device copies are genuine asynchronous transfers that can overlap another
GPU's work; a copy from ordinary pageable memory would instead block the host.
The cache registers each region once and reuses it across calls, so the one-time
pinning cost is amortized. By contrast the two result buffers copied back to the
host are freshly allocated every call, so pinning them would pay the cost with no
reuse — they are deliberately left pageable.

---

## 4. The block-tensor public entry points

Five methods form the block-tensor surface. Each runs one of the two engines and
then packages the result differently.

- `compute_f2_blocks` — runs the resident engine and copies the whole tensor back
  to the host. The plain "give me the answer on the CPU" entry.
- `compute_f2_blocks_device` — runs the resident engine and hands back a handle
  whose GPU buffers stay in VRAM (no copy back, no free). Used when the next
  stage will read the tensor straight from GPU memory.
- `compute_f2_blocks_resident` — same idea, but the handle also records the global
  block offset `b0` of this device's share, for the multi-GPU combine.
- `compute_f2_blocks_into` — runs the resident engine and copies this device's
  blocks into the caller's shared host result at the right offset, through a
  persistent pinned staging buffer.
- `compute_f2_blocks_streamed` — runs the streaming engine, writing into either a
  host-RAM destination or a disk file, depending on the requested tier. It
  rejects the resident tier (that tier is supposed to bypass this seam).

### Buffer escape and late free

In the device and resident variants, the GPU buffers produced by the engine are
*moved* into the returned handle rather than freed. They were allocated on this
worker's device but are freed later, from the handle's destructor, possibly while
a different device is the active one. That cross-device free is safe because the
GPU free call carries the pointer's own device with it, so no save-and-restore of
the active device is needed.

### compute_f2_blocks_into staging

This variant stages each device's partial through two persistent pinned buffers
(one for f2, one for pairwise variance) that grow once to the partial's size and
are then reused — never shrunk. Because two devices each stage into their own
buffers, their copies back to the host run concurrently instead of serializing on
a device-wide lock. A debug-only assertion checks that the partial's element
count equals the expected P²·(block count), which is already bounded by GPU
memory, so the pinned staging can never grow past what already fit in VRAM. After
the copy-back and a synchronize (the ordering guarantee that the transfer
finished), a plain host memcpy places the bytes into the caller's slice; the
slices for different devices are disjoint, which makes the two workers' copies
race-free.

---

## 5. run_f2_blocks_resident — the in-VRAM block engine

Builds the entire block tensor in GPU memory and returns it; the caller decides
whether to copy it back or keep it resident.

### Layout and budget

Compute the shared block layout and size buckets (section 2). Query free GPU
memory *before* committing the big resident tensors, so the budget helper
subtracts the resident tensors and the matrix-library workspace from a true free
figure rather than trusting allocation order. The helper accounts for **both**
P²·(block count) double-precision tensors — an earlier version counted only one,
so it was about 2× under budget — and reserves the workspace before applying the
occupancy fraction. The result is a per-bucket cap on how many blocks fit in one
batched call. A single big bucket's slabs would otherwise run the device out of
memory.

### Freeing the raw inputs early

GPU memory is the binding constraint at scale. The raw input matrices are needed
only by the feeder, so they live in an inner scope and are freed *before* the
main loop. The concrete arithmetic: at P=768, M≈584k, the raw inputs (10.8 GB)
plus feeder outputs (17.9 GB) plus resident tensors (7.1 GB) would sum to 35.8 GB
and overflow a 32 GB card; freeing the raw inputs after the feeder leaves only
about 25 GB resident and enough headroom for the batched scratch.

### One feeder over all SNPs

Inside that inner scope the raw inputs are pinned-and-uploaded, then a single
feeder pass over all M SNPs produces the masked Q, masked V, and stacked helper.
A synchronize makes the feeder finish before the raw inputs are freed and their
memory reused.

### Slab pre-sizing (eliminating per-chunk allocate/free)

The earlier version allocated and freed the batched scratch buffers *inside* the
chunk loop — measured at 645 allocations and 648 frees on the P=768 run. Both
allocate and free take a device-wide lock and synchronize the whole device, so
under a two-GPU fan-out the two workers serialized against each other on that
lock (a measured 7.1% plus 2.6% of API time and an 18% loss of kernel overlap).
The fix: size each scratch buffer once, to the largest single chunk over all
buckets, allocate it outside the loop, and reuse it for every chunk. The kernels
index strictly by the passed dimensions, never by buffer size, so oversizing
changes only *when* memory is committed, never a single result bit. Peak memory
is unchanged, because the loop already reached that one-chunk peak transiently.
The two scratch families scale differently (one with P·s_pad·blocks, the other
with P²·blocks) and different buckets can maximize each, so each cap is tracked
independently for a tight bound.

### The chunked bucket loop

For each bucket, process its blocks in chunks of at most the budgeted count:
upload the chunk's block ids, gather each block's SNP columns into padded scratch,
run the batched matrix multiplies, and assemble the results directly into the
resident tensor at each block's slot. A synchronize after each chunk makes it
fully finish before the next chunk overwrites the reused scratch — the
single-stream serialization that keeps the reuse bit-identical to fresh-per-chunk
buffers. A final synchronize guarantees the tensor is complete before the caller
reads it. The precision policy is engaged once before the loop, and the
matrix-multiply workspace bound at construction survives every chunk because the
loop never resets the stream.

---

## 6. stream_f2_blocks_impl — the spill-to-RAM-or-disk engine

Produces the same block tensor but never holds it all in GPU memory. It writes
each block out through a sink (host RAM or disk) as the block is computed. Two
mechanisms make this possible: SNP-column tiling of the input, and a small device
ring that overlaps compute with the spill.

### SNP-tile input streaming

The resident engine's "upload all of Q/V/N and feed all M" prologue is gone.
Instead each chunk uploads only the columns its blocks actually span — the range
`[s_lo, s_hi)` — from the host matrices, which stay in host RAM. Because the
layout is column-major, that span is one contiguous run per matrix, so it is a
single copy each. The per-tile buffers are sized once to the widest tile over all
chunks and reused, mirroring the resident engine's slab pre-sizing, so there is
no per-chunk allocate/free. The result: the GPU footprint is proportional to
P × (widest tile), independent of the total SNP count M. This is value-preserving
because the feeder works one column at a time with no cross-column dependency, so
feeding a column in isolation produces the same bits as feeding it inside the
all-SNP sweep — only *when* the column is uploaded changes.

### The streamed memory budget

This path uses a different budget from the resident engine. The resident budget
reserves room for the full resident result; the streamed path holds no resident
result — it spills block-by-block through a small ring — so using the resident
budget here would reserve a phantom 12–48 GB result and then, since that budget
omits the ring, let the chunk size grow until the path runs out of memory (the
observed out-of-memory failures at P=1000 and P=1500). Instead it budgets against
the real streamed per-block footprint: the gather scratch (4·P·s_pad per block),
the matrix-multiply outputs (4·P² per block), and the device ring (another 4·P²
per block), plus a fixed reservation for the tile feeder. The envelope is the
occupancy fraction of free memory (after the workspace), split so the feeder
takes a bounded slice (`kStreamTileBudgetFraction`, one quarter) and the
scratch-plus-ring take the rest — guaranteeing feeder plus scratch plus ring stay
within the fraction. At modest P and M both slices are far larger than needed, so
the split is a no-op and chunking is unconstrained.

### chunk_extent — the tile valve

A single helper decides, for one chunk, how many blocks it takes and what column
span they cover, stopping before the span would exceed the tile-width budget.
Both the pre-pass that sizes the buffers and the streaming loop call this same
helper, so the buffer sizes and the runtime chunking can never disagree. It
always admits at least the first block, even if that one block's own width
exceeds the budget — in which case it runs out of memory cleanly rather than
silently producing nothing.

### Rebased local ids

Within a chunk, each block's offset is rebased to be relative to the tile start
(the block's global offset minus `s_lo`), and a local id array `[0, 1, …]`
indexes them. The gather then reads the tile feeder at the rebased offset —
exactly the same SNP columns, the same bits, as the all-SNP path read at the
absolute offset. The assemble writes into chunk-local slots of the ring buffer
instead of a global per-block offset. So the computed values are identical to the
resident path; only the write location is chunk-local, and the spill restores each
block's global id.

### The device ring and overlap

The path cycles through a small ring of `kStreamDeviceChunks` (2) device buffers.
Chunk c computes into buffer c mod 2. After a chunk's spill copies are issued, an
event is recorded on that buffer; a later chunk that reuses the buffer first waits
on that event, so it cannot overwrite slabs still being copied out. That is the
device-side half of the overlap. The sink's own background writer is the host-side
half: it does the slow host-copy or disk-write concurrently with the next chunk's
compute. Two device buffers (not three) are enough because a device buffer only
has to survive its own copy-to-host, not the slow write — the sink's separate
host-side ring absorbs that latency — which keeps the device ring small. The ring
depth is defined once in the shared config so this real allocation and the
tier-selection budget can't drift. The ring's events are cleaned up automatically,
so a throw partway through construction still tears down the already-built slots.

Each block is spilled under its **global** id (issued on the stream, in order,
after the assemble) with no synchronize in the loop — the missing synchronize is
exactly what lets the spill overlap the next chunk's compute. A final
`sink.finish()` drains the last in-flight blocks and finalizes the destination
(for disk: trailer, fsync, reopen).

---

## 7. Cross-cutting invariants

A few rules hold across all of the paths above.

- **Parity-neutrality.** Every performance choice here — pinning, tiling, buffer
  reuse, the ring, which tier the bytes land in — moves or reorders bytes, never
  values. The three tiers (resident, host RAM, disk) and the single whole-matrix
  path all produce bit-for-bit identical f2 and pairwise-variance numbers. The
  tile streaming and local-id rebasing are value-preserving for the same reason:
  they change only where a column is uploaded or where a slab is written, not what
  is computed.

- **Amortized input pinning.** The host input matrices are pinned once and reused
  across calls, because pinning a multi-gigabyte region is itself a 50–360 ms page
  walk that only pays off when reused. A pin that fails (for example, hitting the
  locked-memory limit) degrades to a pageable copy with a debug warning — never a
  crash. Measured effect on a two-GPU box: about 109 ms per iteration per device
  pageable versus about 51 ms pinned.

- **Precision engaged once.** The precision policy is engaged a single time per
  run; the batched matrix-multiply routine then only sets the per-call compute
  type. The determinism workspace bound when the backend was constructed survives
  every matrix multiply because none of these loops resets the stream.

- **Single stream per device.** Each device does its work on one stream, and the
  per-chunk synchronizes serialize the reuse of shared scratch. That ordering is
  what makes buffer reuse bit-identical to allocating fresh buffers every chunk,
  and it is part of the reproducibility contract.

- **Buffers escape, freed late.** The resident engine's output buffers are moved
  into the returned handle and freed later, possibly under a different active
  device — safe because the GPU free call carries its pointer's own device.

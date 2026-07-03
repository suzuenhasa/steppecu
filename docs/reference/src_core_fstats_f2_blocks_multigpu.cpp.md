# `f2_blocks_multigpu.cpp` reference

## 1. Purpose

`src/core/fstats/f2_blocks_multigpu.cpp` is the entry point that computes the
per-block **f2 tensor** across every GPU in one machine. The f2 tensor is the big
`[P × P × n_block]` block of numbers (for `P` populations partitioned into
`n_block` genome blocks) that the later model-fitting stage reads. This file's job
is to spread that computation across the available GPUs and stitch the pieces back
together so the answer is exactly the same as if a single GPU had done the whole
thing.

Three things are worth understanding up front:

- **It contains no GPU code.** The file is plain host C++ and includes no CUDA
  header. It never launches a matrix-multiply, never allocates GPU memory, and
  never re-implements the per-block computation. Instead it drives each device's
  existing single-GPU backend through a narrow, CUDA-free interface
  (`backend->compute_f2_blocks` and friends) and combines the results. All the
  multi-GPU logic sits *above* the per-device backend and reuses that backend
  unchanged.

- **It exposes three entry points**, not one — a host-returning function, a
  device-resident function, and an adaptive tiered function. They share almost all
  of their logic and delegate to one another (see section 3).

- **Every path produces bit-for-bit identical results.** Whether the work runs on
  one GPU or eight, and whether the partial results are merged over a direct
  GPU-to-GPU link or bounced through host memory, the final numbers are identical
  down to the last bit. This is the central promise the file is built to keep, and
  section 2 explains why it holds.

---

## 2. The parity guarantee (why multi-GPU matches single-GPU exactly)

The reason splitting work across GPUs does not change the answer comes down to two
design choices working together.

**Whole blocks go to one device each.** The SNP columns are partitioned into
blocks, and each *entire* block is assigned to exactly one GPU. That GPU computes
the block from precisely its own contiguous run of SNP columns — the same input
bits, the same internal bucketing, the same independent batched computation that a
single-GPU run would use for that block. So each device's per-block result is
identical, bit for bit, to the slab a single GPU would have produced for the same
block. Nothing is split *within* a block, so no block ever gets summed across
devices in a different order.

**The pieces are merged in a fixed device order.** Each device produces a compact
partial result covering only its own blocks. The merge step places every device's
partial at its correct block offset and sums onto a zero-initialized full tensor,
always walking the devices in the fixed order `0, 1, … , G-1`. Because the order
is fixed and each block lands at one known offset, the assembled tensor equals the
single-GPU tensor slab for slab.

Two consequences follow. First, the number of GPUs (`G`) never changes the result.
Second, the concurrency used to speed things up — each device is driven on its own
host thread so their work overlaps in wall-clock time — is invisible to the math,
because each device's bits are fixed by its block assignment and the merge only
reads the partials *after* all threads have finished. The overlap changes how long
the run takes, never what it computes.

---

## 3. The three entry points

The file defines three public functions in `steppe::core`. They differ only in
*where the result lives* and *how the caller wants it back*; the input contract,
the validation, and the block-sharding plan are shared.

| Function | Returns | Use when |
|---|---|---|
| `compute_f2_blocks_multigpu` | `F2BlockTensor` (a plain host-memory result) | The caller wants the tensor in ordinary host memory. |
| `compute_f2_blocks_multigpu_device` | `DeviceF2Blocks` (a handle to a result that stays in GPU memory) | The caller wants to keep the result on the GPU and hand it straight to the fitting stage without copying it back to the host. This is the primary, preferred entry. |
| `compute_f2_blocks_multigpu_tiered` | `F2BlocksOut` (an adaptive result that may live in GPU memory, host RAM, or on disk) | The problem may be too big to fit in GPU memory, so the storage location must be chosen automatically at runtime. |

### How they delegate to each other

The host-returning and device-resident functions cooperate so that neither copies
the other's body:

- On a machine that supports direct GPU-to-GPU copies, the **host-returning**
  function delegates to the **device-resident** function and then materializes the
  result once with `.to_host()`. This avoids duplicating the resident fan-out and
  merge logic.

- On a machine *without* direct GPU-to-GPU copies, the **device-resident** function
  delegates the other direction — it calls the **host-returning** function to do the
  cross-device assembly through host memory, then uploads that assembled tensor back
  to the root GPU so its own return type is still a GPU handle.

These two delegations sit on *mutually exclusive* branches (peer-access-available
versus not), so they can never call each other in a loop. There is no recursion.

The **tiered** function does not delegate to the other two; it always drives the
root GPU directly (see section 7).

---

## 4. The combine gate (choosing how partials are merged)

There are two ways to merge the per-device partial results, and both produce
identical bits:

- **Host-staged combine** — the portable baseline. Each device copies its compact
  partial down into its own slice of one shared host-memory result. This path works
  on any machine, including budget boxes where the GPUs cannot address each other's
  memory.

- **Device-resident combine** — the opt-in fast path. On hardware that supports it,
  the root GPU pulls each other GPU's partial directly over a GPU-to-GPU copy and
  assembles the whole tensor in GPU memory, with no trip through host memory and no
  final copy back to the host.

Which path runs is decided by a single predicate that lives in exactly one place, so
the three entry points can never disagree about it. The predicate is a four-way AND:

1. **The caller prefers the direct path** — a "which path?" intent flag in the
   configuration.
2. **The caller permits peer access** — a separate "may we?" permission flag. If
   this is off, the direct path is forbidden even on capable hardware, because the
   direct path would have to switch on the very peer-access feature this flag vetoes.
3. **The hardware actually supports peer access** — a fact discovered at startup by
   probing the devices, not something the caller asserts.
4. **There are at least two GPUs** — structurally always true once past the
   single-GPU fast path, but included so the predicate reads completely.

### "Requested" versus "selected", and the honest downgrade

The first three intent-and-permission terms (prefer, permit, at-least-two-GPUs)
are split into their own smaller predicate called *requested*. The full gate is
*requested* AND the discovered hardware probe. Keeping these separate lets the code
detect a **genuine downgrade** precisely: if the caller *requested* the direct path
(preferred it and permitted it on a real multi-GPU run) but the gate still came out
false, the only term that can have failed is the hardware probe — the hardware
cannot do peer access. Only in that exact case does the file log a one-line warning
that it fell back to the host-staged path. A caller who simply chose the baseline on
purpose (by not preferring the direct path, or by withholding permission) gets no
warning, because nothing went wrong.

Because both merge paths are bit-identical, this gate is purely about *how bytes
move*, never about *what is computed*. It is safe to flip either way.

---

## 5. The single-GPU fast path

When there is exactly one GPU, all three entry points short-circuit before any
sharding or merging happens. They drive the one backend over the full inputs and
return its result unchanged. This makes single-GPU behavior *structurally*
identical to the plain single-GPU code: the one-GPU case adds nothing to the value
path — no shard plan, no partial buffers, no combine — so there is no way for the
multi-GPU machinery to perturb it.

For the device-resident entry this is the headline win: with one GPU the result is
already sitting in that GPU's memory after the computation, so it is returned as a
GPU handle with **no copy back to the host at all**.

---

## 6. The multi-GPU paths (two or more devices)

Once there are at least two GPUs, all entry points build the same block-aligned
shard plan — the mapping of which block goes to which device — and then fork on the
combine gate from section 4.

- **Direct-copy machine (peer access available).** Each device's partial result
  stays in GPU memory, and the root GPU assembles them all into one GPU-resident
  tensor over direct GPU-to-GPU copies. No host bounce, no final copy back to the
  host.

- **Host-staged direct path (peer access unavailable or not chosen).** The
  full-shape result is allocated once in pinned host memory up front, and each
  device copies its compact partial *straight into its own disjoint slice* of that
  one result. There is no intermediate per-device buffer and no separate summing
  step — because the blocks are disjoint, each device simply writes its own range.
  The byte layout and offsets are identical to what a separate gather-and-sum step
  would have produced.

- **No-peer machine returning a GPU handle.** Assembling a *single* GPU-resident
  tensor spanning two cards is impossible without either direct GPU-to-GPU copies or
  a trip through host memory. On such a machine the device-resident entry keeps each
  device's computation resident (no premature copy per device), assembles the whole
  tensor in host memory using the host-staged path above, and then uploads that one
  assembled tensor to the root GPU. The single host bounce is the *cross-card
  assembly transport* forced by the missing hardware feature — it is not a wasteful
  extra copy of the output.

After the merge, each entry records which combine path it actually used
(direct-resident or host-staged) in the shared resources object, so callers and
tests can see the transport that ran.

---

## 7. The memory tiers (Resident / HostRam / Disk)

The tiered entry point handles problems that may be too large to keep entirely in
GPU memory. It always drives the root GPU (multi-GPU sharding of this streamed path
is a future addition), and it chooses one of three storage tiers:

- **Resident** — the whole result fits in GPU memory. This is the fastest path and
  is exactly the ordinary device-resident computation, with no streaming.
- **HostRam** — the result does not fit in GPU memory but does fit in host RAM.
  Blocks are streamed out into a host-memory result as they are computed.
- **Disk** — the result fits in neither. Blocks are streamed to a cache file on
  disk through a small persistent staging buffer, which keeps memory use tiny (so it
  works even on a laptop).

### How the tier is chosen

The tier is decided from **runtime measurements**, never hardcoded: the amount of
free GPU memory on the root device (measured once at startup) and the amount of free
host RAM (measured right now). A caller or test can override the automatic choice.
The override precedence is: an explicit configuration setting wins first, then the
`STEPPE_FORCE_TIER` environment variable, then the automatic decision.

### Shared streaming logic and the per-tier count rule

The HostRam and Disk arms share a small helper that runs the streamed computation
into the chosen destination and then copies the population count and the per-block
sizes from the tier's own result onto the tier-agnostic surface the caller reads.
The helper deliberately does **not** copy the block count, because each tier applies
its own rule for it: the HostRam arm trusts the streamed result's count only when it
is non-negative (otherwise it keeps the value derived from the partition), while the
Disk arm always clamps the count to be non-negative. This small difference is
preserved exactly as it was, so folding the shared parts changed no behavior.

The per-block sizes (needed later for the jackknife uncertainty estimate) are
derived once, up front, from the partition itself, so the result describes itself
completely in every tier.

---

## 8. Operator override environment variables

Two environment-variable names are defined as named constants in this file, so their
exact spellings live in one place. A typo in a spelling then fails to compile rather
than silently disabling an override at runtime. They are operator-facing keys, not
part of the protected vocabulary that guarantees result parity.

| Constant | Environment variable | What it overrides |
|---|---|---|
| `kEnvForceTier` | `STEPPE_FORCE_TIER` | Forces the memory tier (Resident / HostRam / Disk). A configuration setting wins over it; it wins over the automatic choice. |
| `kEnvF2CachePath` | `STEPPE_F2_CACHE_PATH` | Sets the on-disk cache file path used by the Disk tier. A configuration setting wins over it; if both are empty a frozen default path is used. |

---

## 9. Shared guards and defensive helpers

Every entry point opens with the same two checks, each written in exactly one place
so the three entries cannot drift apart:

- **Input validation** — a debug-only check that the three input matrices agree on
  their population count and SNP count, that those counts are non-negative, and that
  the partition describes exactly the SNP columns. These checks are assertions that
  compile away entirely in a release build. Because *every* statement in the check
  disappears on release, all of the check's parameters are marked as possibly-unused
  so the release build does not warn about them. The name of the calling function is
  baked into the assertion message text rather than passed as a runtime string,
  because the assertion mechanism needs a compile-time string literal.

- **At-least-one-device check** — a real runtime check (it throws, so it is *not*
  compiled away on release) that the resources bundle actually has a GPU, returning
  the device count. The tiered entry calls this purely for its throwing side effect
  and discards the returned count, since it only ever uses the root device.

The file also folds a small "treat a defensive negative count as zero" idiom into
two tiny helpers — one that keeps the value's `int` type (for block-count fields)
and one that also widens to the unsigned size type the allocators expect. These do
exactly the same arithmetic the inline expressions did; they only remove the risk of
mistyping the same guard at the many places it appears. The protected count and
size names themselves are never renamed — only the guard idiom is shared.

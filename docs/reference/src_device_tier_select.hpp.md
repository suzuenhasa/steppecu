# `tier_select.hpp` reference

## 1. Purpose

`src/device/tier_select.hpp` is the single home of one policy: **where the large
f2 result should live** when a run computes it. That result is a pair of
`[populations × populations × jackknife-blocks]` double-precision tensors (the f2
values and their paired variances). Depending on the size of the problem and how
much memory the machine actually has free at that moment, this result may fit
entirely in GPU memory, or it may have to spill to host RAM, or all the way to a
file on disk. This header decides which of those three homes to use and exposes
the small helpers that decision is built from.

The header is deliberately free of any CUDA code. It only uses the C++ standard
library, and every helper is plain integer arithmetic with no call into the GPU.
That has two payoffs: the whole policy is unit-testable on an ordinary machine
with no GPU present, and the CUDA-free orchestration layer can include it directly
without dragging in the GPU toolkit. It sits next to `vram_budget.hpp` in the same
directory and reuses that file's shared arithmetic helpers so the two cannot drift
apart.

The tier choice is **parity-neutral**: it changes only *where* and *when* a slab
of results lands, never the actual numbers. When the policy picks the in-GPU-memory
tier, the run takes the existing device-resident code path completely unchanged —
no streaming, no spilling. Streaming to host RAM or disk is opt-in-by-need: it is
reached only when the result genuinely does not fit in GPU memory.

---

## 2. The three tiers (`OutputTier`)

`OutputTier` is an enum naming the three possible homes for the result, fastest
first.

| Value | Meaning |
|---|---|
| `Resident` | **Tier 0.** The result plus its working set fit in free GPU memory, so everything stays on the device (in a `DeviceF2Blocks`). This is the existing GPU path, unchanged — no streaming of any kind. |
| `HostRam` | **Tier 1.** The result does not fit in GPU memory but does fit in free host RAM. Result blocks are streamed off the GPU into a host-side tensor through a triple-buffered sink (so a block's copy-to-host overlaps with the next block's computation). |
| `Disk` | **Tier 2.** The result fits neither GPU memory nor host RAM. Blocks are streamed out to a file on disk through a small, persistent pinned staging buffer. This keeps memory use tiny, so it works even on a laptop with little RAM. |

The tier is chosen automatically by the policy function (see section 6), or pinned
explicitly for tests via a config field or an environment variable (see section 7).

---

## 3. Force-tier token strings

When a tier is pinned through the `STEPPE_FORCE_TIER` environment variable, the
variable's value is one of three lowercase words. These constants are the single
home of those spellings, one per tier, so the code that parses the environment
variable and the documented, user-facing spelling can never disagree.

| Constant | Value | Maps to |
|---|---|---|
| `kForceTierTokenResident` | `"resident"` | `OutputTier::Resident` |
| `kForceTierTokenHostRam` | `"host"` | `OutputTier::HostRam` |
| `kForceTierTokenDisk` | `"disk"` | `OutputTier::Disk` |

Matching is case-insensitive, so `"Disk"` and `"DISK"` also work.

---

## 4. Working-set footprint helpers

Two helpers estimate the *transient* GPU memory a compute holds at the same time as
the result — the scratch that a phase needs while it runs, on top of the result
itself. They exist so the tier policy (and the performance bench) can predict a
footprint from one source of truth rather than re-deriving the same numbers in
several places. Every count of buffers-per-population comes from a named constant in
`config.hpp`, so these formulas cannot silently drift away from the real GPU
allocations. All arithmetic is done in `std::size_t` so no product can wrap a
32-bit count, and non-positive inputs return `0`.

### `resident_working_set_bytes(P, M)` — the in-GPU-memory tier's scratch

For the in-GPU-memory (`Resident`) tier, the compute runs a "feeder" phase that
decodes **all** SNPs at once and keeps them co-resident with the result. That phase
holds, per population, 3 raw decoded input buffers plus 4 persisted output buffers —
7 buffers total, each sized by the number of populations `P` times the total SNP
count `M`. On top of that sits the matrix-multiply library's fixed workspace. So the
estimate is:

```
7 · P · M · sizeof(double)  +  cuBLAS workspace
```

The two per-population buffer counts (3 and 4) are the named constants
`kFeederRawBufsPerPop` and `kFeederOutBufsPerPop`. The chunk slabs that the compute
uses afterward reuse the memory freed from the raw inputs, so they fit under this
same envelope and do not need a separate term. This helper is consulted **only** for
the `Resident` decision, because only that tier runs the all-at-once feeder; it must
therefore keep the `7 · P · M` term exactly.

### `streamed_working_set_bytes(P, M, max_tile, max_nb, max_s_pad)` — the streamed tiers' scratch

The streamed tiers (`HostRam` and `Disk`) do **not** run the all-at-once feeder.
They decode one block-tile at a time inside a chunk loop, so their peak GPU
footprint is essentially independent of the total SNP count `M`. Instead it is
bounded by the widest single chunk. The estimate sums four parts, then adds the
matrix-multiply workspace:

- **per-tile raw inputs**: 3 buffers of size `P × max_tile`
- **per-tile feeder outputs**: 4 buffers of size `P × max_tile`
- **gather/matrix-multiply scratch**: the per-block chunk footprint (from
  `vram_budget.hpp`) times the most blocks in any one chunk
- **the device ring**: a small fixed number of result buffers (the
  `kStreamDeviceChunks` ring), sized `P × P` per block

This works out to something proportional to `P · max_tile + P² · max_nb`, with
**no** `P · M` term at all. That missing `P · M` term is exactly why the streamed
tiers escape the memory wall that the all-at-once feeder hits — a wall that
previously capped whole-genome runs at roughly 768 populations on a 32 GB card.

This helper is **not** used by the tier policy itself: the streamed tiers are chosen
by the size of the *result*, since the feeder cost that mattered for the resident
decision is gone once you are streaming. It is exposed for the performance bench's
high-population feasibility story and for any future policy that needs to assert the
streamed path fits.

The parameters:

| Parameter | Meaning |
|---|---|
| `P` | number of populations |
| `M` | unused; kept only so the signature mirrors `resident_working_set_bytes` |
| `max_tile` | the widest single chunk's tile width, in SNP columns |
| `max_nb` | the most blocks in any one chunk (the batched matrix-multiply batch count) |
| `max_s_pad` | the widest bucket's padded SNP-block width |

---

## 5. The free-host-RAM probe (`free_host_ram_bytes`)

This function reports how much host RAM is free **right now**, in bytes, read at
runtime. It is never hardcoded, because the machines this runs on vary widely (cloud
instances in particular). It returns the **minimum** of two figures: the operating
system's free RAM and the container's cgroup memory headroom. It is only declared
here; the body lives in `host_ram.cpp`.

The first figure comes from the Linux `sysinfo` system call:
`(free RAM + buffer RAM) × memory unit` — a deliberately conservative proxy for "RAM
the operating system can hand back without swapping." It does not try to count
reclaimable page cache beyond the buffer figure. If the system call fails this figure
is treated as `0`.

The second figure is the cgroup memory headroom: the container's memory limit minus
its current usage. `host_ram.cpp` reads the cgroup v2 unified-hierarchy files first
(`/sys/fs/cgroup/memory.max` and `/sys/fs/cgroup/memory.current`), then falls back to
cgroup v1 (`/sys/fs/cgroup/memory/memory.limit_in_bytes` and `.../memory.usage_in_bytes`). A
limit of `"max"`, an unlimited/sentinel value, or a file it cannot open or parse means
"no cap," so the headroom does not constrain the result. If a valid limit is found, the
headroom is `limit − current` (clamped to `0` when usage already meets or exceeds the
limit, or when the current-usage file cannot be read).

Clamping to the cgroup headroom is the fix for a real OOM. Inside a memory-capped
container — for example a 90 GiB vast.ai instance — `sysinfo` reports the whole
**host's** free RAM, which can be hundreds of gigabytes and has nothing to do with
what the container is actually allowed to allocate. Without the clamp the tier
selector over-commits, chooses the Resident (or Host) tier for a model that cannot
possibly fit inside the cap, and the process is OOM-killed by the kernel. Taking the
minimum with the cgroup headroom makes the selector see the true ceiling and correctly
fall through to the streamed HostRam or Disk path. Both figures failing (system call
fails *and* no cgroup cap can be read) yields `0`, which safely pushes the caller
toward the disk tier.

---

## 6. The tier-select policy (`select_output_tier`)

`select_output_tier` is the frozen policy: pick the fastest tier the result fits in.
It takes the problem shape and the two runtime free-memory figures and returns an
`OutputTier`.

The rule, in order:

1. Compute the result size: `2 · P² · n_block · 8` bytes (the f2 tensor plus its
   paired-variance tensor, from `resident_tensor_bytes` in `vram_budget.hpp`).
2. **Resident** if **both** of these hold:
   - the result plus the resident working set (section 4) fits within **70%** of free
     GPU memory (`kResidentTierVramFraction`); **and**
   - the dense decode input also fits within **60%** of free host RAM
     (`kHostTierRamFraction`). The Resident output engine reads its whole dense
     `P × M` Q/V/N from **one** host buffer, so this cost is
     `kResidentHostInputStacks · P · M · sizeof(double)` — three dense `P × M` host
     arrays (the Q, V, and N stacks), where `kResidentHostInputStacks` is `3`. If that
     host input would not fit, the model is routed to a streamed tier instead of the
     Resident engine, even when the GPU side alone would have fit.
3. Otherwise **HostRam** if the result alone fits within **60%** of free host RAM
   (`kHostTierRamFraction`).
4. Otherwise **Disk**.

The second half of the Resident test is the extract-f2 host-RAM-wall clamp: a large
model's dense `P × M` Q/V/N input can be far bigger than the `P²` result, so checking
only the result would let the selector pick Resident, build the dense host input, and
OOM. Reserving `kResidentHostInputStacks · P · M · sizeof(double)` up front keeps such
models on the bounded, block-source streamed path. The streamed tiers never build the
dense host input — they re-decode one SNP-tile at a time — so the HostRam and Disk
decisions still turn only on the result size.

The two fraction-of-free-memory thresholds are computed with the shared
`budget_bytes` helper (floor of `fraction × free`, in `std::size_t`), the same
idiom the chunk-sizing math uses, so all these thresholds share one implementation.
The free-GPU and free-host figures passed in are always the runtime probes, never
hardcoded values. A degenerate empty problem (`P ≤ 0` or `n_block ≤ 0`) returns
`Resident`, which is the existing path's harmless no-op — the policy never streams
"nothing." The choice moves no bits, so it is parity-neutral.

The parameters:

| Parameter | Meaning |
|---|---|
| `P` | number of populations |
| `M` | total SNP count — feeds both the GPU resident working-set term and the dense host-input reservation, so it matters only for the Resident check |
| `n_block` | number of jackknife blocks |
| `free_vram` | free GPU memory, in bytes (from the device's reported free-VRAM figure) |
| `free_host_ram` | free host RAM, in bytes (from `free_host_ram_bytes`) |

---

## 7. Resolving the effective tier (`resolve_output_tier`)

`resolve_output_tier` decides the tier a run will actually use, layering the
overrides on top of the automatic policy in a fixed order of precedence:

1. **The config field wins.** If `DeviceConfig::force_tier` is anything other than
   `Auto`, that pinned tier is used.
2. **Then the environment variable.** Otherwise, if `STEPPE_FORCE_TIER` is set to
   `"resident"`, `"host"`, or `"disk"` (case-insensitive, per section 3), that tier
   is used.
3. **Then the automatic policy.** Otherwise it falls through to
   `select_output_tier` (section 6).

The already-read value of the environment variable is passed in as `env_value`
(`nullptr` when unset) rather than being read inside the function. That keeps the
helper pure and testable — the orchestrator does the `getenv`, honoring the rule
that this policy layer reads no global state on its own. The body lives in
`host_ram.cpp`.

The parameters:

| Parameter | Meaning |
|---|---|
| `force` | the config's `ForceTier` override (`Auto` means "no override") |
| `env_value` | the already-read `STEPPE_FORCE_TIER` value, or `nullptr` if unset |
| `P`, `M`, `n_block` | problem shape, forwarded to the automatic policy |
| `free_vram`, `free_host_ram` | the runtime free-memory figures, forwarded to the automatic policy |

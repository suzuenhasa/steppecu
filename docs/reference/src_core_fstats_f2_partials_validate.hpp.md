# `f2_partials_validate.hpp` reference

## 1. Purpose

`src/core/fstats/f2_partials_validate.hpp` is the single place that checks the
inputs to the "combine" step of an f2 computation are well-formed, before any of
that work begins.

When steppe computes f2 statistics across more than one GPU, each GPU works on its
own slice of the genome and produces a **partial** result — a chunk of the f2 tensor
covering only that GPU's range of genome blocks. A later **combine** step stitches
those partials back into one complete tensor. There are two different ways to do
that combine (described in the next section), and both of them must first confirm
that the partials they were handed actually fit together: the right number of them,
agreeing on the population count, each one sized to hold what it claims to hold, and
together covering the whole genome with no gaps or overlaps.

This header holds that pre-flight check. It exposes two public functions —
`validate_f2_partials` (for the first combine path) and `validate_resident_partials`
(for the second) — plus one internal helper they share. Each function throws a
`std::runtime_error` with a descriptive message the moment it finds a problem, and
returns normally if everything is consistent. The check is deliberately run **once,
up front**, before any device memory is allocated or any data is copied, so that a
malformed combine is blamed on its own inputs (naming the offending partial, its
block count, and the shard it should have matched) instead of silently producing a
wrong tensor or reading past the end of an undersized partial.

---

## 2. The two combine tiers and why they must agree

There are two combine paths — called "tiers" — and they produce identical results by
design. This validator is what keeps them agreeing on which inputs are legal.

- **The host-staged tier** copies each GPU's partial back to host RAM and sums them
  there. Its combine code lives in `core/fstats/f2_combine.cpp`.
- **The device-resident tier** keeps everything on the GPUs: GPU 0 pulls each other
  GPU's partial directly over a device-to-device (peer-to-peer) copy and sums them in
  a fixed device order. Its combine code lives in `device/cuda/p2p_combine.cu`.

The two tiers are meant to be interchangeable — feed either one the same partials and
it produces the same combined tensor, bit for bit. That guarantee only holds if the
two tiers accept **exactly** the same set of inputs. If one tier were stricter than
the other, there could be a set of partials that one tier combines while the other
rejects, and the "either path gives the same answer" promise would quietly break.

For that reason the precondition check must be the same for both. It used to be
copied byte-for-byte into both combine files, kept in step only by a comment asking
future editors to change both copies together — a fragile arrangement that this
header replaces. Now there is one home for the check, both tiers call into it, and
they cannot drift apart.

---

## 3. Why the header is CUDA-free and header-only

Two structural properties of this file are worth knowing because they are
intentional, not accidental.

**It contains no CUDA code.** The host-staged tier is pure host code and must not
pull in the CUDA runtime; the device-resident tier is CUDA code. To be shared by
both, this header names only types that are themselves CUDA-free: the public
`F2BlockTensor`, the plan type `device::DeviceShard`, the opaque resident handle
`device::DevicePartial`, and standard-library types. Because none of those drag in
`<cuda_runtime.h>`, the header compiles cleanly into the host core library **and**
into a CUDA translation unit. The device-resident overload references a device
handle, but that handle is itself an opaque, CUDA-free type, so the header stays
CUDA-free even though one of its functions is only ever used by the GPU tier.

**Everything is `inline` in the header — there is no separate `.cpp`.** The check is
cheap: it does a constant amount of work per partial, so its cost grows only with the
number of GPUs (a handful), and it runs off the bandwidth-critical part of the
combine, where it is effectively free. Being header-only inline means there is no
out-of-line object file to link and no link-time coupling between the two tiers —
each tier simply `#include`s the same CUDA-free header and gets its own copy.

---

## 4. The shared precondition contract

Both public functions enforce the same core contract. Some vocabulary first:

- **G** — the number of partials, which is also the number of GPUs that produced
  them and the number of shards in the plan. All three counts must match.
- **shards** — the block-aligned plan. `shards[g]` is a half-open block range
  `[b0, b1)` naming exactly which genome blocks partial `g` is responsible for. The
  shards are the authoritative tiling that the partials are checked against.
- **P** — the combined population count. It is the leading dimension of every f2
  slab, so a P-by-P block of values exists for each genome block.
- **n_block_full** — the total number of genome blocks in the finished, combined
  tensor.
- **empty partial** — a partial that owns zero blocks. A GPU can legitimately be
  assigned no blocks, in which case its partial carries no data. The per-partial
  storage and population checks are skipped for such a partial, because it has no slab
  to check.

The checks, in order:

1. **Counts line up.** `partials.size() == shards.size()` (both equal G).
2. **No negative sizes.** `P >= 0` and `n_block_full >= 0`.
3. **Each partial spans exactly its shard.** Partial `g`'s block count equals
   `shards[g].b1 - shards[g].b0`.
4. **Non-empty partials agree on P.** Every partial that owns blocks reports the same
   population count as the combined `P`.
5. **Non-empty partials are sized correctly** (details differ per tier — see sections
   5 and 6).
6. **The shards tile the genome exactly.** Summing every shard's block span gives
   `n_block_full`, with no gap and no overlap — the shards must cover
   `[0, n_block_full)` contiguously.

Every error message is prefixed with a caller-supplied name (the `who` argument), so
a thrown exception always identifies which combine raised it — for example
`steppe::core::combine_f2_partials_host: ...`.

---

## 5. `validate_f2_partials` — the host-staged tier

```
void validate_f2_partials(
    const char* who,
    std::span<const F2BlockTensor> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full);
```

This is the guard for the host-staged combine. Its partials are host-side
`F2BlockTensor` objects, each carrying its own vectors of data: `f2` and `vpair`
(the paired-variance tensor) and `block_sizes` (one count per block).

| Parameter | Meaning |
|---|---|
| `who` | The calling combine's qualified name, prefixed onto every error message so a throw names its own tier. |
| `partials` | The G host-side partials, in order `g = 0 .. G-1`. |
| `shards` | The block-aligned plan; `shards[g]` is partial `g`'s owned block range. |
| `P` | The combined population count (leading dimension of every slab). |
| `n_block_full` | The total block count of the combined tensor. |

Beyond the shared contract of section 4, this overload performs the **short-partial
out-of-bounds guard** — check 5 for this tier. For every non-empty partial it
verifies that the actual storage matches the declared extent:

- `f2.size()` and `vpair.size()` must both equal `P * P * n_block`.
- `block_sizes.size()` must equal `n_block`.

This matters because both combine tiers index into a partial's `f2` array up to
element `P*P*n_block - 1` (the host tier reads it directly; the device tier drives a
peer-to-peer copy whose byte count is computed from `P*P*n_block`). A partial that
reported the right scalar `n_block` and `P` but whose backing vector was actually too
small would be read past its end — a memory-safety bug. Checking the vector sizes up
front closes that gap for both tiers at once. The element count `P*P*n_block` is
computed in `size_t` to match how `F2BlockTensor` reports its own size, avoiding any
overflow in the multiplication.

---

## 6. `validate_resident_partials` — the device-resident tier

```
void validate_resident_partials(
    const char* who,
    std::span<const steppe::device::DevicePartial> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full);
```

This is the guard for the device-resident (peer-to-peer) combine. It enforces the
same contract as `validate_f2_partials`, but over `DevicePartial` handles instead of
host tensors. Keeping it as a **separate overload** — rather than editing the
existing host validator to also accept device handles — leaves the host tier's
reject behavior completely unchanged and risk-free, while the CUDA-free `DevicePartial`
type keeps the header CUDA-free.

A `DevicePartial` is an opaque handle to data that lives on a GPU. Its bulk `f2` and
`vpair` arrays are on the device and cannot be inspected from this host-side,
CUDA-free header. Two things it does carry that this validator uses:

- `n_block_local` — how many blocks this partial owns (the device-resident spelling
  of `n_block`).
- `b0` — the partial's placement offset: the first global block index it is placed
  at. In this tier each handle carries its own offset inline, rather than the plan
  passing a separate parallel array of offsets, so this validator can cross-check it.
- `block_sizes` — a host-side vector of per-block counts, one per local block.

| Parameter | Meaning |
|---|---|
| `who` | The calling combine's qualified name, prefixed onto every error message. |
| `partials` | The G resident handles, in order `g = 0 .. G-1`. |
| `shards` | The block-aligned plan; the authoritative tiling the handles are checked against. |
| `P` | The combined population count (leading dimension of every slab). |
| `n_block_full` | The total block count of the combined tensor. |

The per-partial checks mirror section 5 with two differences:

- **Placement offset cross-check** (extra in this tier): each handle's `b0` must equal
  its shard's `b0`. Because each device handle names its own placement offset, this
  confirms the handle is placed exactly where the plan expects it.
- **Storage check is count-only.** For every non-empty handle, `block_sizes.size()`
  must equal `n_block_local`. The full `f2`/`vpair` extent of `P * P * n_block_local`
  cannot be re-checked here because that data is on the device behind an opaque
  handle — it is instead guaranteed correct by construction where the resident
  partial is produced.

---

## 7. `validate_partials_scaffold` — the shared internal helper

The two public overloads share their outer skeleton through a single internal
template, `detail::validate_partials_scaffold`. This is the mechanism that makes the
"one home, both tiers" promise of section 2 concrete: the parts that are identical
between the tiers live here exactly once and cannot drift.

The scaffold owns the three checks that are the same regardless of which partial type
is being validated:

- the count check (partials count equals shards count),
- the negative-bound check (`P` and `n_block_full` are non-negative), and
- the tiling accumulator and its final check (summing each shard's span and confirming
  the total equals `n_block_full`).

It walks the shards once. For each shard `g` it computes that shard's block span
(`b1 - b0`) and then calls a caller-supplied callback, `check_g`, passing the error
prefix, the index `g`, the shard, and the span. That callback is where each tier
plugs in the checks that are specific to its partial type: `validate_f2_partials`
supplies a body that checks the host tensor's `n_block`, `P`, and vector storage
extents; `validate_resident_partials` supplies a body that checks the device handle's
`n_block_local`, `b0` offset, `P`, and `block_sizes` count. After the callback
returns, the scaffold adds the span to its running total and moves on. The final tile
check runs after the loop.

Keeping two separate overloads rather than one function with a type switch is a
deliberate choice — it keeps each tier's per-partial logic readable and independent.
The scaffold folds only the shared outer skeleton, not the per-partial bodies, so the
two tiers get shared structure without giving up their distinct, type-specific checks.

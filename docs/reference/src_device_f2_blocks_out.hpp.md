# `f2_blocks_out.hpp` reference

## 1. Purpose

`src/device/f2_blocks_out.hpp` defines `F2BlocksOut`, the single result type that
carries the finished f2 precompute out of the compute stage and into the model-fit
stage. steppe first computes, for every jackknife block of the genome, a `P × P`
table of f2 statistics (`P` is the number of populations) together with a matching
table of paired variances. `F2BlocksOut` is the object that holds that whole
`P × P × n_block` result.

The key idea is that the result can be **too big to keep in GPU memory**, so it may
instead live in ordinary host RAM, or on disk. `F2BlocksOut` hides which of those
three places the data actually sits in. The code that consumes the result — the fit
engine and the parity test — never asks where the data lives; it only calls a small
set of read-back accessors, and those accessors do the right thing for whichever
place the data is in.

This header contains **no GPU (CUDA) code**. It holds the GPU-resident handle, the
host tensor, and the on-disk descriptor all as plain by-value members, and it only
declares the read-back accessors — their bodies live in a separate GPU source file
(`cuda/f2_blocks_out.cu`), because reading a GPU-resident result back requires a
copy from GPU memory, which is the only part that needs CUDA. Keeping this header
CUDA-free lets the plain host-side orchestrator include it without dragging in the
GPU toolkit.

---

## 2. The three storage tiers and the bit-for-bit guarantee

A finished result lives in **exactly one** of three tiers, recorded in the `tier`
field:

| Tier | Where the data lives | Member that holds it |
|---|---|---|
| `Resident` | GPU memory (the fastest path, unchanged from before the tier system existed). | `resident` (a `DeviceF2Blocks`) |
| `HostRam` | Ordinary host RAM, as a fully materialized tensor. | `host` (an `F2BlockTensor`) |
| `Disk` | A binary file on disk. | `disk` (a `DiskF2Blocks` descriptor) |

Only the member matching the current `tier` is engaged; the other two are empty.

Two properties make this safe to rely on:

- **The tier choice is out-of-band.** Which tier a result uses is decided by a
  separate policy based on how much memory is free, and it is recorded here in the
  `tier` field — never baked into the numeric data itself. The numbers are identical
  no matter which tier carries them.
- **Read-back is bit-for-bit identical across all three tiers.** Calling `to_host()`
  on a `Resident`, `HostRam`, or `Disk` result produces byte-for-byte the same host
  tensor, and that tensor also matches what a plain single-GPU run would produce. The
  parity test depends on this: it reads the result back and compares it byte-for-byte
  against the reference. So the tier is purely about *where* and *how fast*, never
  *what number comes out*.

Every slab is stored as double precision (FP64) in all three tiers regardless of
which precision mode the compute stage ran in.

---

## 3. `slab_elems` — the per-block slab element count

`slab_elems(int P)` returns `P²`, the number of elements in one block's slab. Each
jackknife block contributes a `P × P` table laid out column-major (element `i, j`
sits at index `i + P·j`), so one block's worth of data is `P²` elements.

This tiny helper exists so that every read-back path — `read_block_to_host`,
`to_host`, and `size` — computes the slab size from one shared formula and cannot
drift apart from one another.

It matters that it widens `P` to a 64-bit `std::size_t` **before** multiplying.
At production scale, `P²` and `P² · n_block` reach roughly ten billion elements,
which overflows a 32-bit integer. Doing the widening first means the product is
always computed in 64-bit arithmetic and never wraps around.

---

## 4. `FileCloser` — the disk read-handle deleter

`FileCloser` is the cleanup helper attached to the open file handle used by the
on-disk tier. When the handle goes away, `FileCloser` closes the underlying C file
(`std::fclose`).

The one non-obvious behavior: if closing the file reports an error (a nonzero return
from `std::fclose`), `FileCloser` routes a single teardown-warning line through
steppe's logging sink instead of silently swallowing the failure. `std::fclose`
returns `0` on success and a nonzero value on failure, so "nonzero means failed" is
the correct test.

Practical points:

- It is a stateless, empty struct. Because of that, the owning smart pointer
  (`std::unique_ptr`) is exactly the size of a raw file pointer — the deleter adds no
  storage overhead.
- It checks for a null file pointer before closing, purely as a defensive measure; a
  `unique_ptr` only ever calls its deleter on a non-null pointer.
- The actual body lives in the GPU source file (`cuda/f2_blocks_out.cu`). The
  warning-logging branch is a debug-build feature and compiles away entirely in
  release (`NDEBUG`) builds, with no unused-parameter warning.

---

## 5. `DiskF2Blocks` — the on-disk cache descriptor (tier 2)

`DiskF2Blocks` describes a finished result that has been written to a binary file. It
does not hold the numbers; it holds everything needed to *read them back*: the file
path, the shape parsed out of the file header, and an open read handle. The fit
engine and the parity test read a given block by computing its byte offset in the
file and reading directly at that offset — no scanning.

| Field | Type | Meaning |
|---|---|---|
| `path` | `std::string` | Path to the on-disk cache file. |
| `P` | `int` (default `0`) | Population count — the leading dimension of every slab. |
| `n_block` | `int` (default `0`) | Number of jackknife blocks stored in the file. |
| `block_sizes` | `std::vector<int>` | The per-block SNP counts, length `n_block`, read from the trailer at the end of the file. |
| `read_handle` | `std::unique_ptr<std::FILE, FileCloser>` | An open, read-only handle to the file, used for reading slabs back. |

Ownership and lifetime:

- The struct is **move-only** — it can be moved but not copied. It owns the file
  handle through the move-only smart pointer above, and that pointer's deleter
  (`FileCloser`) supplies both the automatic close-on-destruction and the
  null-out-on-move behavior. Because those are handled for free, every special member
  (destructor, move constructor, move assignment) is simply defaulted, and copying is
  explicitly deleted.
- `read_handle` is a plain public field; read the raw pointer with
  `read_handle.get()` and test whether the file is open with `if (read_handle)`.
- Reads always go through the `F2BlocksOut` accessors below, never directly against
  this descriptor.

---

## 6. `F2BlocksOut` — the unified result and its read-back accessors

`F2BlocksOut` is the single result object. It records the tier, the shape, and holds
whichever one of the three tier members is engaged. Like the disk descriptor, it is
**move-only**: it owns either a move-only GPU handle, host vectors, or an open file,
so copying is deleted and the move operations are defaulted.

### Shape and tier fields

| Field | Type | Meaning |
|---|---|---|
| `tier` | `OutputTier` (default `Resident`) | Which of the three tiers this result lives in. |
| `P` | `int` (default `0`) | Population count. |
| `n_block` | `int` (default `0`) | Number of jackknife blocks. |
| `block_sizes` | `std::vector<int>` | Per-block SNP counts. |
| `resident` | `DeviceF2Blocks` | Engaged only when `tier == Resident`: the existing GPU-resident handle, unchanged. |
| `host` | `F2BlockTensor` | Engaged only when `tier == HostRam`: the full result materialized in host RAM. |
| `disk` | `DiskF2Blocks` | Engaged only when `tier == Disk`: the on-disk cache descriptor. |

### The read-back accessors

These are the entire surface the fit engine and the parity test use. Neither branches
on `tier` — the accessor does that internally.

- **`to_host()`** materializes the whole result into a host `F2BlockTensor`. This is
  bit-for-bit identical across all three tiers. For a `Resident` result it copies the
  data down from GPU memory (the one and only GPU-to-host copy). For a `HostRam`
  result it returns a copy of the in-RAM tensor. For a `Disk` result it reads every
  block back from the file. The parity test calls this and compares the result
  byte-for-byte against the reference.

- **`read_block_to_host(b, f2_slab_out, vpair_slab_out)`** reads a single block `b`
  into caller-provided buffers: one `P²` f2 slab and one `P²` paired-variance slab,
  both column-major (`i + P·j`). Each output buffer must be at least `P²` elements.
  This is the fit engine's per-block tile reader, and it is tier-agnostic:
  - `Resident` — copies that block's slab down from GPU memory.
  - `HostRam` — copies from the in-RAM tensor at offset `P²·b`.
  - `Disk` — two reads at the block's byte offsets in the file.

- **`size()`** returns the total element count of the result, `P² · n_block` (with
  `n_block` treated as `0` when negative). It uses `slab_elems(P)` so it can never
  disagree with the per-block slab shape the read-back paths assume.

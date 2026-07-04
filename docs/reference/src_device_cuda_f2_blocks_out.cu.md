# `f2_blocks_out.cu` reference

## 1. Purpose

`src/device/cuda/f2_blocks_out.cu` implements how steppe *reads back* its
precompute result — the per-block "f2" and paired-variance ("vpair") arrays that
the model-fit stage consumes. It also holds the cleanup code for the on-disk
result's file handle.

The result of the precompute is stored in exactly one of three places, called
**tiers**:

- **Resident** — the arrays stay in GPU memory (fastest, used when everything fits).
- **HostRam** — the arrays are spilled to ordinary host RAM.
- **Disk** — the arrays are spilled all the way to a binary file on disk.

The rest of steppe never has to know which tier a given result landed in. Two
accessor functions defined here — `read_block_to_host` (read one block) and
`to_host` (read everything) — hide the tier behind a single interface. Each
function contains a three-way branch, and every branch produces the exact same
bytes (see section 2).

This is a CUDA translation unit specifically because the Resident branch needs a
GPU-to-host copy. The HostRam and Disk branches are plain host memory and file
I/O, but they live here too so all three read-back paths sit side by side.

The layout of a single block is fixed and shared everywhere: each block is a
`P × P` square (where `P` is the number of populations), stored column-major so
that element `(i, j)` lives at index `i + P·j`. One block therefore holds `P²`
double-precision numbers — this count is called the *slab* size. The block axis
is the outermost dimension, so block `b` starts at element `P²·b`.

---

## 2. The bit-identical read-back guarantee

The central promise of this file is that both read-back functions return
**byte-for-byte identical** data no matter which tier the result was stored in,
and identical to what a plain single-GPU run would produce. A parity test relies
on this directly: it calls `to_host()` and does a raw byte comparison against the
reference result, so any drift between tiers would show up immediately.

This holds because none of the three paths ever transforms the numbers. Each one
is a pure byte movement:

- **Resident** does a raw device-to-host memory copy of the exact bytes.
- **HostRam** does a `memcpy` of the exact bytes.
- **Disk** reads the exact bytes back from the file.

There is no recomputation, no reordering, and no change of precision anywhere in
the read-back. The numbers are always stored and moved as 64-bit doubles, in
every precision mode the precompute might have used. The Resident path does
page-lock ("pin") the caller's destination buffers before copying, but pinning
only changes how the operating system treats those pages — it never touches the
bytes — so it is safe with respect to this guarantee (see section 3.1).

---

## 3. `read_block_to_host` — the fit's one-block tile reader

`read_block_to_host(b, f2_slab_out, vpair_slab_out)` copies a single block's
`P²`-element f2 slab and `P²`-element vpair slab into two caller-provided buffers
(each must be at least `P²` doubles long). The model-fit stage calls this
repeatedly to pull blocks one tile at a time rather than materializing the whole
result.

A debug-only check asserts that `b` is in range (`0 ≤ b < n_block`). This matters
because every tier computes a byte offset as `slab · b`: a negative `b` would cast
to a huge unsigned value and read out of bounds, and a `b` at or past the block
count would read past the last slab. In release builds the check compiles away
entirely — it is a trust-based contract, and the release build is byte-identical
to the checked build. If `P` is zero (an empty result) the slab size is zero and
the function returns immediately.

The three tier branches:

- **Resident** — reads the block's slab out of GPU memory. If the device pointers
  are null (nothing was stored) it returns without copying. Otherwise it copies
  the f2 and vpair slabs from device to host. The device-selection and
  buffer-pinning mechanics around these two copies are described in section 3.1.
- **HostRam** — a straight `memcpy` from the in-RAM arrays at element offset
  `P²·b` into the caller's buffers.
- **Disk** — requires that the descriptor still holds an open read handle
  (otherwise it throws). It reconstructs the file's layout (section 5) and reads
  the f2 slab and the vpair slab, each from its own computed byte offset in the
  file.

### 3.1 The Resident tier: restoring the device and pinning the destination

The Resident branch has two pieces of GPU housekeeping that the other tiers do
not need.

**Device restore.** The copy must run with the GPU that owns the data selected as
the active device, but the caller may have had a different device active. The code
records the caller's current device, switches to the result's device, and installs
a scoped guard that restores the caller's device when the function returns —
whether it returns normally or by throwing. That restore is deliberately routed
through a non-throwing path: a guard's cleanup must never throw, so a failed
restore logs a single diagnostic line (in debug builds) and yields its error
status instead of vanishing. On the normal, everything-succeeds path this is
byte-identical to doing nothing.

**Destination pinning.** Before the two copies, the code page-locks ("pins") the
caller's two destination buffers in place for the duration of the copy window,
and unpins them automatically when the function returns. Pinning lets the GPU copy
run faster. It is done defensively: the pinning helper never throws — if the
memory cannot be pinned it simply proceeds with an ordinary (pageable) copy — so
it can never cause the read-back to fail. Crucially, pinning changes only the page
state of the memory, never its contents, so it does not affect the bytes that come
back. This mirrors exactly how the device-resident handle's own full-result copy
pins its destinations.

---

## 4. `to_host` — materializing the whole result at once

`to_host()` returns the entire result as one host-side tensor, again identical
across tiers. This is what the parity test calls before its byte comparison.

- **Resident** — delegates to the device handle's existing single device-to-host
  copy. There is only ever this one copy of the resident data, so this is a
  copy of the same buffers in the same layout.
- **HostRam** — copies the already-materialized host arrays (plus the shape
  metadata: population count, block count, and per-block sizes) into a fresh
  tensor. The in-RAM arrays are already in the final block-major layout, so this
  is a plain copy.
- **Disk** — allocates the full f2 and vpair vectors (`P² · n_block` doubles
  each), then reads each region back in a single contiguous read. This works
  because the on-disk f2 region and vpair region are each stored in exactly the
  final layout, so no per-block scatter is needed — one read fills the whole
  vector. An empty result (`P² · n_block == 0`) returns early; a Disk tier with
  no open read handle throws.

---

## 5. The Disk tier internals: header reconstruction, offsets, and safe reads

The on-disk cache file starts with a fixed **64-byte header**, followed by the f2
region (every block's slab, back to back), then the vpair region (same shape),
then a small trailer listing the block sizes. Storage is always 64-bit doubles.
The per-block, block-major ordering of that layout is the parity on-disk
f2_blocks ordering[^at2]; steppe's own 64-byte header is a fixed prefix that a separate
stand-alone reader strips off. The 64-byte header size and the double-precision
storage type are frozen properties of the format.

Two internal helpers support the Disk paths.

**`disk_header` — rebuild the offsets from the shape, not the file.** The exact
byte offsets of the f2 region, the vpair region, and the block-sizes trailer are
completely determined by `P` and the block count, both of which the descriptor
already knows. So rather than re-parse the 64 header bytes from the file, this
helper recomputes the offsets from the in-memory shape. The f2 region begins right
after the 64-byte header; the vpair region begins one full region later; the
trailer begins one region after that, where a region is `P² · n_block · 8` bytes.
That region size is computed with each factor widened to a 64-bit unsigned value
*before* multiplying, because the true size reaches into the multi-gigabyte range
and a 32-bit intermediate would silently wrap. A negative block count is treated
as zero.

**`pread_all` — read a whole slab or region, or fail loudly.** Given a byte count
and a starting offset, this seeks to the offset and reads the full number of
bytes, throwing a descriptive error if the seek fails or if fewer bytes than
requested come back (a short read). It never returns partial data silently. The
seek uses a signed `long` file offset, and a compile-time check pins the
assumption that `long` is 64-bit on this platform — on a hypothetical 32-bit-`long`
target a large offset into the multi-gigabyte regions would wrap, so the build is
made to fail rather than misread.

---

## 6. `FileCloser` — the one place the disk read handle is closed

The on-disk descriptor owns its open file through a smart pointer whose custom
deleter is `FileCloser`. Making the deleter the sole owner means the descriptor
needs no hand-written copy/move/destructor code — moving it automatically leaves
the moved-from descriptor holding no file, and destroying it automatically closes
the file. Folding every close down to this one place also gives a single, tidy
spot to notice a failed close.

`FileCloser` closes the file and, if the close reports failure, emits exactly one
teardown warning. Two details are worth knowing:

- It first checks for a null pointer and returns — defensive, since the smart
  pointer only ever calls the deleter on a real, non-null handle.
- The close itself is the load-bearing action and must always happen, so the close
  call sits *outside* the warning macro. In release builds that warning macro
  compiles to nothing and does not even evaluate its arguments, so if the close
  were folded into the macro it would be dropped entirely. The status value the
  close returns is consumed only by the (release-stripped) warning, so it is
  marked as possibly-unused to keep release builds warning-clean.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

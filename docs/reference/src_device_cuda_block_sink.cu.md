# `block_sink.cu` reference

## 1. Purpose

This file implements the two streaming "sinks" that carry each genome block's
computed result out of GPU memory and into its storage tier without stalling the
compute loop:

- **`HostRamSink`** (tier 1) — copies each block's result into a plain array in
  host RAM.
- **`DiskSink`** (tier 2) — writes each block's result to an on-disk cache file,
  so the working set never has to fit in RAM at all.

Both are used only when the full result is too large to keep resident in GPU
memory. The fastest tier, where everything stays on the GPU, never constructs a
sink — it bypasses this file entirely.

Each block produces two square slabs of `P × P` double-precision values: an `f2`
slab and a paired-variance (`vpair`) slab, where `P` is the number of
populations. A slab therefore holds `P²` doubles. Blocks are numbered `0` to
`n_block − 1` and are independent of one another; a sink only changes *when* and
*where* a slab lands, never its numeric contents.

The shared plumbing that both sinks rely on — a small pool of pinned host
buffers and a background writer thread — is declared and documented alongside the
sink classes in the companion header. This file holds the parts that are unique
to each tier: what the background writer actually does with a copied slab, and,
for the disk tier, the full procedure for writing and safely finalizing the
cache file.

---

## 2. Overlapping the write with GPU compute

The reason both sinks exist in this shape is to keep the slow storage write off
the critical path. When the compute loop finishes a block, it hands the sink a
pair of *device* pointers to that block's freshly computed slabs. The sink does
three quick things and returns:

1. Claims a free pinned host buffer (a "slot") from a small pool.
2. Starts an asynchronous copy of the two slabs from the GPU into that slot.
3. Records a completion marker and hands the slot to a background writer thread.

The compute loop is then free to start the next block immediately. The
background writer waits for the copy to finish, then does the genuinely slow work
— a large host-side memory copy or a disk write — while the GPU is already busy
computing the next block. At large population counts the matrix-multiply work
dominates, so this slow write hides completely behind it.

The pool holds three slots, which is enough depth for the GPU to keep computing
and copying while the writer drains earlier blocks. If the writer falls behind
and all slots are still in its hands, the next spill simply blocks until a slot
comes free — natural backpressure, not an error.

Everything a sink does is **parity-neutral**: the copy from the GPU is a raw
byte copy, the host-side copy is a plain memory copy, and the disk write is raw
bytes. No value is recomputed, reordered, or rounded differently, so a run that
spills produces bit-for-bit the same reported numbers as a run that stays fully
resident.

---

## 3. Where each block's bytes land (the drain action)

The one place the two tiers differ is what the background writer does with a slot
once its copy from the GPU has finished. Each sink supplies this step as a small
callback when it starts its writer:

- **`HostRamSink`** copies the slot straight into its destination arrays. Block
  `b`'s slabs go at element offset `b × P²` — a plain memory copy of `P²` doubles
  each for `f2` and `vpair`.
- **`DiskSink`** writes the slot to the file. Block `b`'s slabs go at byte offset
  `(region start) + b × (P² × 8)`, one such write into the `f2` region and one
  into the `vpair` region.

In both cases every block occupies a distinct, non-overlapping region keyed by
its block number, laid out block-by-block. Because no two blocks ever touch the
same destination bytes, the background writer is free to drain slots in whatever
order it finishes them, with no risk of one block clobbering another.

`HostRamSink` also prepares its destination up front: when it starts, it sizes
its `f2` and `vpair` arrays to hold all blocks (`n_block × P²` doubles each) and
fills them with zeros, so any block that never spills is simply left as zeros.

---

## 4. Writing the disk cache file

The disk tier produces a single self-describing cache file. The exact byte
layout (a fixed 64-byte header followed by the `f2` region, the `vpair` region,
and a small trailer) is defined by the on-disk format header and documented with
it. This file is responsible for the *procedure* of writing and finalizing that
file safely.

### Opening and the header

When the disk sink starts, it computes where each region will live:

- The header sits at byte offset `0`.
- The `f2` region begins right after the header.
- The `vpair` region begins right after the `f2` region.
- The block-sizes trailer sits at the very end, after `vpair`.

Each region spans `n_block × P² × 8` bytes (eight bytes per double). The sink
opens the file for read/write, creating it and truncating any existing file, then
writes a header stamped with the format's magic string, version, data type
(always double precision), the population count `P`, the block count, and the
byte offsets of the three regions. Recording those offsets in the header lets a
later reader jump straight to any block without scanning the file.

### The file's permissions

The file is created with mode `kCacheFileMode`, which is `0644` — readable and
writable by the owner, read-only for group and others. It is given a name so the
raw permission bits are not an unexplained octal number sitting at the `open`
call.

### Finalizing

After every block has been spilled, finishing the disk sink runs a fixed
sequence:

1. Drain and join the background writer, then re-throw any error the writer hit
   (for example a failed disk write) on the calling thread.
2. Write the block-sizes trailer — one 32-bit integer per block — at its offset.
3. Call `fsync` so the whole file is durably on disk, not just in the operating
   system's write cache.
4. Close the write handle, then reopen the same file read-only. The read-only
   handle is what the rest of the program uses to read blocks back.

Finishing is idempotent: if it has already run, it returns immediately and does
nothing a second time.

---

## 5. `pwrite_all` — completing a write despite short writes and interrupts

A single positioned write to a file is allowed to transfer fewer bytes than
requested, or to be interrupted by a signal before it starts. `pwrite_all` is a
small helper that hides this: it loops, writing from wherever the last call left
off at the correct offset, until every requested byte has landed. It retries
transparently when a call is interrupted, and it fails loudly — throwing an error
that names which part of the file was being written — if a write reports an error
or reports writing zero bytes (which would otherwise spin forever).

Every write to the cache file goes through this helper: the header, each block's
two region slabs, and the trailer.

---

## 6. Shutdown ordering and handing off the read handle

The disk sink owns two things whose teardown order matters: an open file
descriptor, and a background writer thread whose job is to write *to* that file
descriptor.

### The destructor must stop the writer before closing the file

If the sink is destroyed, its destructor stops and joins the writer thread
**first**, before it closes the file. The writer's whole purpose is to write into
that file, so the file must outlive any write still in flight; closing it first
would let the writer write to a closed handle. The shared staging machinery has
its own safe shutdown as well, but because it is the last-declared member of the
sink it would otherwise be torn down *after* the sink's own destructor body has
already run and closed the file — so the sink deliberately stops the writer
itself, up front, rather than relying on that later teardown.

On the normal path, finishing has already closed the write handle, so the
destructor's file close only fires on an error path where finishing never ran.

### Handing off the read handle

Once finishing has reopened the file read-only, the sink can hand that read
handle, together with the file's shape (its path, `P`, block count, and the list
of block sizes), to a small descriptor object that outlives the sink. The
handoff moves ownership of the read handle: the descriptor takes it over, and if
the descriptor happened to already hold some other open handle, that older one is
closed first.

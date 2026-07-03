# `f2_disk_format.hpp` reference

## 1. Purpose

`src/device/f2_disk_format.hpp` defines the on-disk file format for the f2_blocks
cache — the file steppe writes when the f2 results are too large to keep in GPU
memory or host RAM and have to spill to disk. It is the "compute the expensive f2
statistics once, then fit many models against them" artifact, the same kind of
reusable cache that ADMIXTOOLS 2 keeps.

The header describes the byte layout, names the small set of constants that identify
and version the file, defines the fixed 64-byte header record that sits at the front
of every such file, and provides the offset arithmetic used to seek to any one
block's data without scanning the whole file.

Despite living under `src/device`, the file contains **no CUDA code**. It only uses
the C++ standard library. That keeps it usable by both the GPU writer and the plain
host-side reader without either pulling in the other's dependencies.

Two properties are the whole point of the design:

1. **A whole-file load is a single bulk copy.** The bytes on disk are laid out in
   exactly the same order as the in-memory f2 tensor, so reading the file back is
   equivalent to one `memcpy` into that tensor — no reshuffling, no per-element
   conversion.
2. **Any single block is directly seekable.** The fit stage can read block *b*'s f2
   data straight from a computed byte offset, without reading the blocks before it.

---

## 2. File layout

The file is **binary, little-endian, and block-major**. It has four regions laid
out back to back in this order:

```
header[64 bytes] | f2 region | vpair region | block_sizes trailer
```

| Region | Size | Contents |
|---|---|---|
| Header | 64 bytes | The fixed `F2DiskHeader` record (see section 4). Always at file offset 0. |
| f2 | `P² × n_block` doubles | Every block's f2 matrix, one after another. Each block is a `P × P` slab (P is the number of populations). |
| vpair | `P² × n_block` doubles | Every block's paired-variance matrix, in the same block-major order as the f2 region. |
| block_sizes | `n_block` 32-bit integers | The trailer: how many SNPs went into each block. |

### Block-major with column-major slabs

"Block-major" means the block index is the **outermost** axis: all of block 0's
`P²` values come first, then all of block 1's, and so on. Within a single block, the
`P × P` slab is stored **column-major** — the value for populations *i* and *j* sits
at position `i + P·j` inside that slab.

Put together, the value for populations *i*, *j* in block *b* lives at linear
position `i + P·j + P·P·b`. This is deliberately **byte-identical** to the in-memory
f2 block tensor, which uses that same `i + P·j + P·P·b` indexing. Because the two
layouts match exactly, loading the file is a straight bulk copy into the tensor and
saving is the reverse — there is no transpose or repack step in either direction.

### ADMIXTOOLS 2 compatibility

Matching ADMIXTOOLS 2's on-disk ordering is a **goal** of this format. ADMIXTOOLS 2
stores its f2_blocks as one `P²` slab per block in block-major order, which is the
same ordering used here. The one difference is the steppe file's fixed 64-byte
header prefix; a reader aiming for ADMIXTOOLS 2 compatibility strips that prefix and
reads the numeric regions that follow.

---

## 3. Named constants

Three constants identify and version the numeric payload. Defining them here, in the
one header shared by the writer and the reader, is what keeps the two from drifting
apart.

| Constant | Value | What it's for |
|---|---|---|
| `kF2DiskMagic` | the 8 bytes `S T P F 2 B K 1` (the string `"STPF2BK1"`) | The magic number stamped at the very start of the header. A reader checks these 8 bytes first to confirm the file really is a steppe f2_blocks cache before trusting anything else in it. |
| `kF2DiskVersion` | `1` | The binary format version of the numeric payload bytes. The writer stamps this into the header's `version` field, and the reader refuses a file whose version it does not understand. |
| `kF2DiskDtypeFp64` | `1` | The code meaning "the stored numbers are little-endian FP64 (double precision)." |

### Two different "versions" — do not conflate them

`kF2DiskVersion` versions the shape of the **numeric bytes** in this `.bin` file. It
is deliberately separate from the schema version of the JSON provenance sidecar that
travels alongside the cache (which describes *how the file was produced* — the
populations, the settings, and so on). One versions the raw payload; the other
versions the human-readable metadata shape. They change independently and must not
be mixed up.

### Storage is always FP64

The dtype is fixed at double precision in **every** precision mode. Even when the f2
values were computed with the faster emulated-double-precision arithmetic, the
result is stored on disk as true FP64. There is no lower-precision on-disk variant,
which is why `kF2DiskDtypeFp64` is the only dtype code defined.

---

## 4. `F2DiskHeader`

`F2DiskHeader` is the fixed record at the front of every file. It is **exactly 64
bytes**, padded, and written in little-endian order.

| Field | Type | Value / meaning |
|---|---|---|
| `magic` | `char[8]` | The 8 magic bytes — always `kF2DiskMagic`. |
| `version` | `uint32` | The binary format version — `kF2DiskVersion` (currently 1). |
| `dtype` | `uint32` | The on-disk number format — `kF2DiskDtypeFp64` (FP64). |
| `P` | `int32` | The number of populations. Each block slab is `P × P`. |
| `n_block` | `int32` | The number of jackknife blocks stored in the file. |
| `f2_offset` | `uint64` | Byte offset where the f2 region begins. Always `64` (immediately after the header). |
| `vpair_offset` | `uint64` | Byte offset where the vpair region begins. Equals `64 + P²·n_block·8` (just past the f2 region). |
| `block_sizes_offset` | `uint64` | Byte offset where the block_sizes trailer begins. Equals `vpair_offset + P²·n_block·8` (just past the vpair region). |
| `reserved` | `uint8[16]` | Reserved padding, written as zeros. This is what pads the record out to a round 64 bytes and leaves room to add fields later without changing the size. |

Storing the three region offsets explicitly (rather than making the reader
recompute them) means the reader can seek straight to any region using a value read
from the file. The relationships listed above are the invariants a well-formed file
must satisfy — each offset is the previous region's start plus that region's size,
where each region holds `P²·n_block` doubles of 8 bytes each.

### `kF2DiskHeaderSize`

`kF2DiskHeaderSize` is defined as `sizeof(F2DiskHeader)` — the single named home for
the number 64 that the layout description, the `f2_offset` value, and the reader's
header-strip all refer to. A compile-time check (`static_assert`) enforces that this
is **exactly 64 bytes**; if a future edit to the struct changed its size, the build
would fail rather than silently produce files that no existing reader could parse.
The value is frozen by the on-disk format.

---

## 5. Block offset arithmetic

Three small functions compute the byte offset of a single block's slab so a caller
can `pread` just that block instead of loading the whole file.

| Function | Returns |
|---|---|
| `f2_block_offset(header, b)` | The byte offset of block *b*'s `P × P` f2 slab. |
| `vpair_block_offset(header, b)` | The byte offset of block *b*'s `P × P` vpair slab. |
| `detail::slab_offset(base, header, b)` | The shared helper both of the above call. |

### One stride formula, two regions

The f2 region and the vpair region have the **identical** internal stride — they
differ only in where they start in the file. So the actual arithmetic lives once, in
`detail::slab_offset`, which computes:

```
base + P² · b · sizeof(double)
```

`f2_block_offset` calls it with `base = f2_offset`; `vpair_block_offset` calls it
with `base = vpair_offset`. Keeping the stride math in a single place means the two
accessors cannot disagree about how blocks are spaced.

### Overflow safety

The `P² · b` product is computed in **64-bit unsigned** arithmetic, with each factor
cast to `uint64` *before* the multiplications. This matters because `P`, `P`, and `b`
are 32-bit values, and for a large population count and many blocks their product
would overflow a 32-bit intermediate and produce a wrong (wrapped-around) offset.
Widening every factor up front guarantees the stride is computed at full 64-bit
width and cannot wrap.

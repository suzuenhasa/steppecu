# `f2_dir_writer.cpp` reference

## 1. Purpose

`src/app/f2_dir_writer.cpp` writes an **f2_blocks directory** to disk — the
precomputed cache that lets steppe compute the expensive f2 statistics once and
then fit many models against them without recomputing. It is the write side of the
same directory format the loader reads back, and the byte layout it produces round-
trips through that loader by construction. The directory format matches the f2
cache layout of the reference implementation[^at2], so the artifact is a portable
interchange rather than a private format.

The file also contains a **self-contained SHA-256 implementation**. steppe stamps
content hashes into the cache directory so a cache is content-addressed and its
recorded hashes can be checked against the original data files. Rather than pull a
whole cryptography library into the command-line tool, this file carries a small,
standard, `sha256sum`-compatible SHA-256 with both a portable scalar path and a
hardware-accelerated x86 path.

Two layering rules shape the file:

- It is **plain C++20 with no CUDA header**. The command-line-tool layer is kept
  free of GPU code. The only device-layer thing it reaches for is the CUDA-free
  on-disk header struct and its magic/version stamps, so that the writer and the
  reader share exactly one definition of the file format and cannot drift apart.
- The library never prints. Every failure is returned as a status plus a human-
  readable reason string; the command-line tool is what decides to print it.

---

## 2. The output directory layout

`write_f2_dir` creates the target directory (if absent) and writes three files
into it.

| File | What it holds |
|---|---|
| `f2.bin` | The numeric payload: a fixed 64-byte header, then the f2 tensor, then the paired-variance (`vpair`) tensor, then the per-block SNP counts. Little-endian, block-major. Its bytes match the in-memory f2 tensor exactly, so loading it is one bulk copy and any single block is directly seekable. |
| `pops.txt` | The population labels, one per line, in population-axis index order. This is the name-to-index map: line 1 is index 0, line 2 is index 1, and so on. |
| `meta.json` | Provenance: which steppe version and precision produced the cache, the filters that were applied, the source file paths and their hashes, and content hashes of `f2.bin` and `pops.txt`. |

The population count `P` and block count `n_block` written into the files come from
the f2 tensor's own shape, not from the caller-supplied metadata — the writer trusts
the tensor.

---

## 3. Shape validation

Before writing anything, `write_f2_dir` runs fail-fast checks so that a wiring or
data bug surfaces as a clear error instead of a silently wrong file on disk. Each
check that fails returns immediately with an `InvalidConfig` status and a message
naming the mismatch:

- The tensor must be non-degenerate: `P > 0` and `n_block > 0`.
- The label list length must equal `P`. The name-to-index map has to cover the
  whole population axis, or every downstream lookup would be misaligned.
- The f2 data and the `vpair` data must each contain exactly `P × P × n_block`
  elements (the full slab).
- The block-sizes list length must equal `n_block`.

A failure to create the output directory, or any later file-open or file-write
failure, is also returned as `InvalidConfig`. An unwritable output path is treated
as a configuration-level fault the user must fix, which matches how the reader
classifies the same class of problem.

The `fail` helper builds these failure results in one place. `json_escape`,
`json_str`, and `bool_str` are small helpers that keep the emitted `meta.json`
well-formed — `json_escape` backslash-escapes quotes, backslashes, and control
characters in string values (population labels and paths are usually tame, but a
label could contain a quote or backslash, and the JSON must stay valid regardless).

---

## 4. The self-contained SHA-256

A small, standard SHA-256 lives in this file so the tool can content-address its
output without a crypto dependency. It follows the FIPS 180-4 standard and produces
a digest that is byte-for-byte identical to what the `sha256sum` utility produces.
Hashing is **streaming** — data is fed in chunks and compressed as it arrives — so
a multi-gigabyte genotype file can be hashed without ever buffering the whole file
in memory.

### 4.1 The round constants

`kSha256K` is the 64-entry table of SHA-256 round constants defined by the standard.
It is shared by both the scalar and the hardware block functions — the scalar path
indexes it directly, and the hardware path loads groups of it into vector registers
— so the two implementations can never disagree about a constant. It is aligned to
64 bytes for the vector loads.

### 4.2 The scalar block path (portable fallback)

`sha256_blocks_scalar` is the plain, portable SHA-256 compression of one or more
consecutive 64-byte blocks. It runs the standard message schedule and 64-round
compression loop and folds the result back into the running state. This path always
compiles and runs correctly on any CPU, so it is the fallback when hardware
acceleration is unavailable.

### 4.3 The hardware SHA-NI path (fast path)

`sha256_blocks_shani` does the same computation using the x86 SHA extension
instructions (the `_mm_sha256rnds2` family). On a CPU that has these instructions,
the per-block round is a single hardware instruction rather than a software loop,
roughly six to seven times faster than the scalar path. This is what makes hashing a
multi-gigabyte file cheap enough to overlap with other work and cost effectively
nothing.

Some details worth knowing:

- It is compiled **only on x86**. Other architectures use the scalar path.
- Only this one function is built for the SHA instruction set, via a compiler target
  attribute (`sha,sse4.1,ssse3`). The rest of the file stays at the default
  instruction set, so merely having this function present does not require the
  extension to run the binary. The function is called only after a runtime check
  confirms the CPU supports it.
- The 64 rounds are unrolled. The middle rounds (16 through 59) are expanded through
  a **local macro, deliberately not a lambda**. A lambda's call operator would not
  inherit this function's SHA target attribute, so the hardware intrinsics inside it
  would fail to inline. The macro expands inline within the function and keeps the
  target attribute. The macro is `#undef`-ed right after use.
- The internal state is kept in the SHA-NI register layout (the interleaved ABEF /
  CDGH form) across the whole block loop, loaded once at entry and stored back once
  at exit. It maintains the same big-endian-loaded state the scalar path does, so the
  two paths are bit-identical.

### 4.4 Runtime dispatch

`sha256_blocks` is the single entry point the rest of the file calls. On x86 it
checks once, caching the result, whether the running CPU supports the SHA extension
(via `__builtin_cpu_supports("sha")`, a cheap CPU-feature query) and routes to the
hardware path if so, otherwise to the scalar path. On non-x86 it always uses the
scalar path. Callers never pick a path themselves.

### 4.5 The streaming `Sha256` class and the bulk-update design

`Sha256` is the streaming hasher: construct it, call `update` with successive chunks
of bytes, then call `hex` to finalize and get the 64-character lowercase hex digest.

The `update` method is written for throughput, and its shape is a deliberate
performance fix. An earlier version copied **every** input byte one at a time into a
64-byte staging buffer before compressing, which topped out around 190 MB/s and, on
one real run, burned roughly 37 seconds of a roughly 41-second extract while hashing
a 6.7 GB genotype file. The current `update` instead:

1. Tops up any partially filled staging buffer to a full 64-byte block and
   compresses that one block.
2. Compresses **all** whole 64-byte blocks straight from the caller's buffer in a
   single call — no per-byte copy, and the hardware block loop keeps its state in
   registers across the entire run. This is the part that removes the old
   bottleneck.
3. Stages only the leftover tail (fewer than 64 bytes) for the next `update` or for
   finalization.

The digest is unchanged by this — it is still standard and `sha256sum`-compatible;
only the path the bytes take into the compression function changed.

Finalization applies the standard SHA-256 padding (append a `0x80` byte, zero-pad,
then the 64-bit big-endian total message bit length) and emits the eight state words
big-endian. The total bit length is accumulated as bytes arrive and captured before
the padding is added, so the padding bytes never corrupt the recorded length.

### 4.6 Whole-file hashing

`sha256_file` hashes an entire file and returns its lowercase hex digest, or an
empty string if the file cannot be opened. It reads through an **8 MiB window**:
large enough to amortize the per-read overhead so throughput is bounded by the (now
bulk) compression rather than the read loop, and small enough that the file is never
fully buffered — which matters because the source genotype file can be around
6.7 GB. Bytes are compressed straight from the read buffer, with no second copy.

This is the single hashing entry point the writer uses for every hash it stamps, and
it is also exposed for the extract command to hash the large source genotype file on
a background thread (overlapping the tens-of-seconds hash with the GPU pipeline) and
pre-fill the result before calling the writer.

---

## 5. Writing `f2.bin`

`f2.bin` is written as: the 64-byte header, then the f2 slab, then the `vpair` slab,
then the block-sizes trailer.

The header (`F2DiskHeader`, defined once in the shared device-layer format header)
carries:

- the 8-byte magic string `STPF2BK1`,
- the binary format version and the data-type stamp (double precision),
- `P` and `n_block`,
- and the byte offsets where the f2 slab, the `vpair` slab, and the block-sizes
  trailer begin. The f2 slab starts right after the header, at byte 64; each later
  offset is the previous section's start plus that section's size.

The two large slabs are each `P × P × n_block` doubles, written directly from the
tensor's own memory. Their in-file layout matches the tensor's memory layout, so the
loader reads them back with one bulk copy.

### The real-`vpair` rule

The `vpair` region carries the **real** per-block pairwise-valid SNP counts from the
precompute — not zeros. This is load-bearing. The model fit itself reads the block-
sizes trailer, so it would be tempting to leave `vpair` as zeros. But the missing-
data path detects a dropped pair-block by testing `vpair == 0` for that block. If
`vpair` were written as zeros, every block would look dropped and the missing-data
filtering would misbehave. So the writer always serializes the true `vpair` values.

The block-sizes trailer is written as `n_block` 32-bit integers (the block sizes are
narrowed to `int32` for the on-disk trailer).

---

## 6. Content addressing and provenance hashes

The writer stamps three content hashes so the directory is content-addressed and
independently verifiable. Each is formatted as `sha256:` followed by the 64-character
hex digest, or an empty string if the file could not be re-read.

| Hash | Of what | Why |
|---|---|---|
| `f2_cache_id` | the complete `f2.bin` | The content address of the cache payload. This is **always** computed — `f2.bin` is small (kilobytes to a few megabytes), so hashing it is never part of any bottleneck. It is also returned to the caller in the write result. |
| `pops_sha256` | the exact `pops.txt` bytes just written | `pops.txt` is the name-to-index map. A swapped or corrupted `pops.txt` would silently reassign every name to a different population-axis index and change every downstream result undetectably. Stamping its hash lets any reader re-hash the file and catch that. It is tiny, so hashing it is free. |
| `geno_sha256`, `snp_sha256`, `ind_sha256` | the source genotype / SNP / individual files | Content hashes of the input dataset, so a cache can be tied back to the exact data it came from and matched against known-good dataset hashes. These are governed by the opt-in policy in section 7. |

---

## 7. The source-file hashing opt-in

Hashing the source dataset is **off by default**, controlled by the metadata's
`hash_source_files` flag.

The reason is cost. The three content hashes above are all of small files. The
source genotype file, by contrast, can be around 6.7 GB, and hashing it whole is a
tens-of-seconds read-and-compress that once dominated an entire extract run — while
producing only a provenance value, not anything that affects correctness. So by
default the writer skips it: it does not hash any source file whose hash is still
empty, and `meta.json` records the source hashes as empty strings together with a
marker saying the hashing was deliberately skipped, so a consumer knows the absence
is intentional rather than a failed or forgotten hash.

When source hashing is requested, the writer fills any source hash that is still
empty by hashing its file. Even then, the caller normally pre-computes the large
genotype hash on a background thread (overlapping it with the GPU pipeline) and hands
it in already filled — so this code path typically only hashes the small SNP and
individual files. The metadata is treated as read-only here: the three dataset hashes
are copied into local variables, seeded from any pre-filled value, and only the
missing ones are computed.

---

## 8. The `meta.json` provenance file

`meta.json` is emitted as hand-built JSON recording how the cache was produced. Its
fields include the format tag (`STPF2BK1`), the schema version, the steppe version,
`P` and `n_block`, the engaged precision tag and mantissa-bit count, the block size
in centimorgans, the pre-filter and post-filter SNP counts, a `filters` block
(minimum minor-allele frequency, the per-SNP and per-sample missingness caps, and the
autosomes-only / drop-monomorphic / transversions-only flags), a human-readable
echo of how the populations were selected, a `source` block (the three file paths, a
flag saying whether source hashing was done, and the three source hashes), plus
`pops_sha256` and `f2_cache_id`.

Two subtleties are worth calling out:

- **Two independent version numbers.** `meta_schema_version` versions the shape of
  this JSON sidecar — the set of provenance fields the writer emits. It is distinct
  from the binary version stamped inside `f2.bin`'s header, which versions the on-disk
  numeric layout. The two evolve independently and must not be conflated: adding a
  provenance field bumps the schema version, while changing the binary layout bumps
  the binary version. A reader gates on the binary version; `meta.json` is advisory.
  The current schema version is `1`.
- **The block size is intentionally coarse.** The centimorgan block size is written
  at the default output precision (about six significant figures) with no extra
  precision applied. It is a coarse binning knob, not a value that parity depends on,
  so a coarse serialization is deliberate.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

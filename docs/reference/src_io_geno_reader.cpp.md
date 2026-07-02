# `geno_reader.cpp` reference

## 1. Purpose

`src/io/geno_reader.cpp` implements `GenoReader`: the component that opens a
genotype file, figures out which on-disk format it is, validates it, and copies
the requested individuals and SNPs into an in-memory tile. It reads and *packs*
bytes only — it never decodes a genotype into an allele frequency. Decoding
happens later, on the GPU (or in the CPU reference path). This file is pure host
C++ with no CUDA and no dependency on the rest of the library, so it can be built
and tested on its own. Every failure it can hit surfaces as a
`std::runtime_error`.

The public method contracts (what each `read_*` function takes and returns, and
exactly which errors it throws) are already spelled out in the header,
`src/io/geno_reader.hpp`. What lives *only* here in the implementation, and what
this reference covers, is:

- how the constructor auto-detects the file format by probing in a fixed order,
- where the geometry (number of individuals, number of SNPs, record stride)
  comes from for each format,
- how the on-disk size is validated and how partial (truncated) files are
  handled,
- how the two different gather axes work, and how the four SNP-major readers all
  produce one identical packed layout,
- the overflow and out-of-memory guards that make every failure safe, and
- the small file-local helper functions.

steppe recognizes six on-disk shapes, named by the `GenoFormat` enum: `Tgeno`,
`Geno`, `Eigenstrat`, `Plink`, `Ancestrymap`, and `Unknown`. The five real
formats split into two families by their row axis — **individual-major** (TGENO)
versus **SNP-major** (everything else) — and that split is why there are two
kinds of reader.

---

## 2. Format auto-detection: the constructor probe chain

The constructor does not require the caller to say what format the file is. It
opens the file once and runs a fixed sequence of probes, each of which either
recognizes the file or leaves the format as `Unknown` and lets the next probe
try. The order matters, and it is safe precisely because each format's leading
bytes cannot be mistaken for another's.

The probes run in this order:

1. **Packed TGENO / GENO magic.** Read the first `kGenoHeaderBytes` (48) bytes.
   Only if a full 48 bytes are present does it parse them for the ASCII magic
   token `"TGENO"` or `"GENO"`. A read shorter than 48 bytes is *not* treated as
   an error here: a file that short cannot be a packed file, but it might be a
   tiny ASCII EIGENSTRAT fixture (a few individuals by a few SNPs), so a short
   read simply falls through to the next probe.
2. **ASCII EIGENSTRAT.** If still unknown, scan the file as an EIGENSTRAT `.geno`:
   one text line per SNP, one character per individual, each character `0`/`1`/`2`
   (reference-allele copy count) or `9` (missing). Its leading bytes are ASCII
   genotype characters plus a newline — never the binary magic — so a packed file
   never reaches here misclassified.
3. **PLINK `.bed`.** If still unknown, check the leading three bytes for the
   `.bed` magic `0x6c 0x1b 0x01`. `0x6c` is the character `'l'`, which is neither
   a magic token nor a `0`/`1`/`2`/`9` genotype character, so a `.bed` file always
   reaches this probe still unknown. The third byte is the mode byte: `0x01` means
   SNP-major (the only layout steppe reads); `0x00` means individual-major and is
   rejected with a clear "re-export SNP-major" error rather than silently
   mis-read.
4. **ANCESTRYMAP text triples.** If still unknown, probe for the legacy unpacked
   EIGENSOFT `.geno`: one text line per (SNP, individual) pair of the form
   `<snp_id> <sample_id> <genotype>`. This is neither a packed magic nor the dense
   `0`/`1`/`2`/`9` character matrix EIGENSTRAT uses, so the earlier probes already
   declined it. Recognition needs only the first non-blank line to have exactly
   three whitespace tokens with a valid genotype (`0`/`1`/`2`/`-1`) in the third —
   the multi-gigabyte file is *not* scanned here.
5. **Give up loudly.** If every probe declined, the constructor throws a
   `std::runtime_error` that lists all the shapes it looked for, so a malformed or
   truly unrecognized file fails immediately with a helpful message rather than
   being read with the wrong assumptions.

The key safety property is that these five leading-byte signatures are mutually
exclusive, so the fixed order never mis-assigns a file: a binary magic, an
all-`0`/`1`/`2`/`9` ASCII matrix, a `0x6c`-led `.bed`, and a three-token text
line each rule the others out.

---

## 3. Deriving geometry per format

"Geometry" means the number of individuals (`n_ind`), the number of SNPs
(`n_snp`), how many records are on disk (`n_records`), the stride between records
(`bytes_per_record`), and the size of any leading header (`header_bytes`). Where
these come from depends on the format:

- **TGENO / GENO (packed).** The 48-byte header itself carries `n_ind` and
  `n_snp`, and the record stride is derived from them. This work is done by
  `parse_geno_header` in the sibling format header, not in this file.
- **EIGENSTRAT.** The ASCII file is fully self-describing, but you must read it to
  learn its shape. `parse_eigenstrat_geometry` scans the whole file: `n_ind` is
  the character count of the first line, and `n_snp` is the number of lines. The
  scan doubles as validation — every line must be the same length and every
  character must be `0`/`1`/`2`/`9`. If any line is ragged or holds a foreign
  byte, the function returns `Unknown` so a non-EIGENSTRAT file that merely lacked
  a packed magic is not misclassified as a degenerate EIGENSTRAT. The full scan is
  a single cheap sequential pass, on par with reading the sibling `.snp`/`.ind`
  metadata. `header_bytes` is `0` — the data starts at byte zero.
- **PLINK and ANCESTRYMAP (geometry from siblings).** These `.geno`/`.bed` files
  carry no geometry of their own, so the counts come from the sibling metadata
  files' line counts: for PLINK, `n_snp` from `.bim` and `n_ind` from `.fam`; for
  ANCESTRYMAP, `n_snp` from `.snp` and `n_ind` from `.ind`. Both siblings must be
  present and non-empty, or the file is rejected as not a well-formed triple.

For every SNP-major format the record stride `bytes_per_record` is the canonical
packed width `packed_bytes(n_ind)` — four individuals per byte, rounded up — and
`n_records` equals `n_snp` (one record per SNP). PLINK's `header_bytes` is the
3-byte magic; EIGENSTRAT's and ANCESTRYMAP's are `0`.

---

## 4. File-size validation and partial files

The last thing the constructor does is decide `records_present_`: how many
complete records are actually on disk. This can be fewer than the header claims,
because a download or copy may have been truncated. Downstream selection uses
`records_present_` to ignore any individual or SNP that is not actually present.

The path splits by whether the format has a fixed-width packed data region:

- **EIGENSTRAT and ANCESTRYMAP are text** and have no fixed-stride data region
  (their lines vary in byte width), so the packed size check does not apply.
  `records_present_` is set to `n_records` (the SNP count) and the constructor
  returns. The actual text is parsed later, lazily, inside that format's gather.
- **TGENO, GENO, and PLINK have a packed data region.** The constructor first
  rejects a degenerate header (a zero stride or zero record count). It then
  measures the file size, subtracts `header_bytes`, and divides the remaining data
  bytes by `bytes_per_record`. That quotient is `records_present_`, matching the
  reference tool's rule of `n_records = (file_size - header) / bytes_per_record`.
  A file smaller than its own header, or one with no complete records at all, is
  rejected. `records_present_` is capped so it can never exceed the header's
  record count, and if it is smaller, that smaller value is the cap all later
  reads honor.

A compile-time check at the top of the file asserts that `std::streamoff` (the
type used for seek offsets) is at least 64 bits wide. Record offsets are computed
as `header_bytes + row * bytes_per_record`, and at the scale of a real dataset
(billions of bytes) that multiply must stay 64-bit. Pinning the width means a
hypothetical 32-bit-offset platform fails to *compile* rather than silently
truncating a seek target and reading the wrong record.

---

## 5. Individual-major gather (`read_tile`)

`read_tile` is the TGENO path, and TGENO is the only format whose records are laid
out one per individual. For each selected population (in the partition's order),
and each member individual within it (in ascending row order), it seeks to that
individual's record and reads the prefix of packed bytes covering the requested
SNP range, copying it into the next slot of the output tile. Because individuals
are read in partition order, they land grouped by population, and a companion
offset array marks where each population's block begins and ends.

Two details are load-bearing. First, every requested row is checked against
`records_present_` before any read; a row past the end would otherwise seek into
trailing junk (or a concatenated file) and silently read a *complete* record of
the *wrong* individual. Second, the current implementation requires the SNP range
to start at zero, so genotype codes align to byte boundaries and the per-record
byte width is simply `ceil(number_of_SNPs / 4)`; a nonzero start is reserved for a
future tiling loop and is rejected so a caller never gets misaligned codes.

---

## 6. SNP-major gather and the canonical packing

The other four readers — `read_snp_major_tile` (GENO),
`read_eigenstrat_snp_major_tile`, `read_plink_snp_major_tile`, and
`read_ancestrymap_snp_major_tile` — all read files whose records are one per SNP,
with every individual interleaved inside each SNP's bytes. This is the axis-swapped
mirror of `read_tile`: it seeks whole per-SNP records rather than per-individual
records.

Because a SNP's bytes interleave all individuals, you cannot skip an unselected
individual at read time. So the individual selection and reordering does *not*
happen during the read. Instead each reader gathers the *full* per-SNP record for
every individual and records the selected source rows in a list; a later
on-device transpose step applies the selection, the reorder, and the decode. The
readers all produce the **same canonical SNP-major byte layout**, so that single
transpose consumes any of them identically. That canonical layout packs each
individual `i` into byte `i / 4` of the SNP record, most-significant bits first
(shifts of 6, 4, 2, 0 for the four individuals in a byte).

The four readers differ only in how they turn their source into those canonical
bytes:

- **GENO** is already in the canonical packing, so the reader copies each SNP
  record straight through.
- **EIGENSTRAT** parses one ASCII line per SNP, maps each character to its 2-bit
  code (`0`/`1`/`2` unchanged, `9` to the missing code), and packs it
  most-significant-bits-first. The `.geno` line index *is* the SNP index, so a line
  whose length is not `n_ind`, or that carries a non-`0`/`1`/`2`/`9` byte, is a
  fail-fast error reported with the SNP line and column — a ragged line would
  desync the SNP axis against the `.snp` file.
- **PLINK** reads each `.bed` SNP record in the `.bed`'s own least-significant-bits-
  first order, remaps each 2-bit value through a fixed lookup table (`00` to 2,
  `01` to missing, `10` to 1, `11` to 0, in reference-allele copies), and re-packs
  it most-significant-bits-first. The bit-flip and the remap change the byte
  *contents*, never the byte *count*.
- **ANCESTRYMAP** is positionally addressed: line `L` (zero-based) is SNP `L /
  n_ind`, individual `L % n_ind`. The reader reads the leading `n_snp × n_ind`
  lines in one forward sequential pass (the text lines vary in width, so there is
  no seeking), maps each line's third whitespace token to its 2-bit code (`0`/`1`/
  `2`, or `-1` for missing — a different missing sentinel from EIGENSTRAT's `9`),
  and packs it most-significant-bits-first. A line without exactly three tokens,
  or whose third token is not a valid genotype, fails fast with the SNP and
  individual position.

When `n_ind` is not a multiple of four, the last byte of each SNP record has
unused high slots. The three readers that build codes by OR-ing them in (the two
text readers and PLINK) pre-zero the buffer so those unused slots are defined; the
raw GENO copy overwrites every byte and needs no pre-zeroing. This choice is the
`zero_init` flag threaded into the shared allocation helper.

---

## 7. Overflow-checked allocation and the exception-type contract

Every reader sizes a heap buffer from a product of two sizes — for example
`number_of_individuals × bytes_per_record`, or (for ANCESTRYMAP's line count)
`number_of_SNPs × n_ind`. In C++, unsigned size arithmetic wraps around modulo a
power of two silently rather than raising an error. On a hostile or stale input,
such a product could wrap to a small value, the buffer would be under-sized, and
the gather loop would then write past the end of the allocation — a silent
heap-buffer overflow rather than a clean failure. So before every allocation the
code performs the standard overflow test (`a > MAX / b`) and throws if the product
would wrap. The row-bounds checks make this unreachable in practice; the explicit
test is defense in depth.

There is also a subtler contract about *which exception type* escapes. The header
promises callers that all I/O failures surface as `std::runtime_error`. But a
large-but-non-wrapping request (an honestly huge tile, or an out-of-memory
machine) makes `std::vector::resize` throw `std::bad_alloc` or `std::length_error`,
*neither* of which derives from `std::runtime_error`. A caller written literally
to the documented contract (catching `std::runtime_error`) would miss those and
crash. So every allocation is wrapped in a try/catch that translates both into the
documented `std::runtime_error`, keeping the contract true for any allocation
failure whatsoever.

Two private helpers hold the parts every SNP-major reader shares, so this logic
lives in one place rather than being copied four times:

- **`build_selection`** sets the tile geometry and builds the pop-contiguous list
  of selected source rows, validating each row against `n_ind` so the later
  transpose can never read a padding byte as a phantom individual.
- **`checked_alloc_snp_major`** performs the overflow-checked allocation with the
  exception translation described above, choosing between a zero-filled and a bare
  buffer via the `zero_init` flag.

Each reader passes its own name into these helpers so the thrown message names the
right function.

---

## 8. File-local helpers

A handful of small functions in the anonymous namespace support the probes and the
mismatch messages:

- **`count_text_records`** counts the newline-terminated lines of a sibling text
  file — the `.bim`/`.fam` line counts that give a PLINK triple its geometry, and
  the `.snp`/`.ind` counts for ANCESTRYMAP. It tolerates a Windows `\r`, does not
  count a single trailing blank line at end of file (a common newline artifact),
  and does not count interior blank lines (the dedicated metadata readers reject
  those later). If the file cannot be opened it returns `0`, which the caller reads
  as "not a well-formed triple."
- **`plink_sibling`** and **`geno_sibling`** derive a sibling path by swapping the
  extension — `.bed` to `.bim`/`.fam`, or `.geno` to `.snp`/`.ind`. If the input
  path does not carry the expected extension, the new extension is appended so the
  sibling is still derived deterministically.
- **`geno_format_name`** maps a `GenoFormat` to a human-readable name plus which
  reader to use. It is shared by every reader's "wrong reader" error, so when a
  file's actual format does not match the reader it was handed to, the message
  names the real format and points at the correct reader. Keeping this in one
  place replaced a per-reader chain that had drifted stale as formats were added.

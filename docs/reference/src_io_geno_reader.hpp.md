# `geno_reader.hpp` reference

## 1. Purpose

`src/io/geno_reader.hpp` declares `GenoReader`, the class that opens a packed
`.geno` genotype file, parses its header, and serves the raw packed bytes of a
chosen slice of the data to the rest of the program.

The reader does exactly one job: read and gather packed bytes. It never *decodes*
them. Decoding — turning the 2-bit genotype codes into allele-frequency numbers —
happens later, on the GPU (or in the CPU reference path). The reader's product is
always a plain data struct full of `std::vector`s that the compute backend
consumes.

Two properties follow from where this file sits in the layering:

- **No GPU code.** This is a pure host C++ file. It includes only the standard
  library and its sibling `io` headers, never anything CUDA. That keeps it cheap
  to include and cheap to test.
- **Failures surface as `std::runtime_error`.** The reader does not use the error
  types the core/device layers use. Every way a read can fail — a bad file, an
  out-of-range request, a corrupt line, even a failed memory allocation — comes
  back to the caller as a thrown `std::runtime_error`. Section 10 covers this
  contract in detail.

A single `GenoReader` is meant to be opened once and kept open across many reads.
It owns the underlying file stream, so it is move-only (it cannot be copied, but
it can be moved).

---

## 2. The two reading paths

Genotype files come in two fundamentally different layouts, and the reader has a
separate entry point for each.

**Individual-major** files store one record per individual (sample), and inside
each record the SNPs run one after another. The AADR TGENO format is the real-world
example. Reading a chosen set of individuals is easy here: each individual is a
contiguous record, so the reader simply seeks to each selected individual's record
and copies the prefix of bytes covering the requested SNP range. The selection
happens in the seek loop — an unselected individual is just never read. This is the
`read_tile` path (Section 5), and its output is already in the canonical
individual-major shape the decoder wants.

**SNP-major** files store one record per SNP, and inside each record *all*
individuals are interleaved. GENO/PACKEDANCESTRYMAP, EIGENSTRAT, PLINK `.bed`, and
ANCESTRYMAP are all SNP-major. Here selection is not free: because every
individual's data for a SNP is packed together within that SNP's bytes, you cannot
skip an unselected individual at read time. So the SNP-major readers do something
different — they read the *whole* per-SNP records and carry the selection as a
separate list of which source rows to keep. The actual selecting, reordering, and
axis-swap (turning SNP-major into individual-major) is deferred to a transpose step
that runs on the GPU. The `io` layer cannot call a GPU kernel, so it hands the app
layer a raw SNP-major tile plus the gather list, and the app layer runs
`ComputeBackend::transpose_to_canonical` to produce the same canonical tile the
individual-major path produces directly.

So there are two output struct types:

- `GenotypeTile` — the canonical, individual-major, ready-to-decode tile that
  `read_tile` produces.
- `SnpMajorTile` — the raw SNP-major bytes plus the selection list that the four
  SNP-major readers produce, to be transposed on the GPU into a `GenotypeTile`.

All four SNP-major readers converge on the *same* canonical packing before the
transpose, so once the bytes reach the transpose, nothing downstream can tell which
format they came from. The formats differ only in how their bytes are read and
re-packed inside the `io` layer.

---

## 3. Opening a file and reading the header

### The constructor

```cpp
explicit GenoReader(const std::string& geno_path);
```

Opening a reader does three things:

1. Opens the file at `geno_path`.
2. Reads and parses the leading fixed-width header, which carries the format magic
   (`TGENO`, `GENO`, and so on) and the two counts (number of individuals, number
   of SNPs). From those, it derives the *record stride* — how many bytes each
   record occupies.
3. Validates the actual on-disk file size against what the header implies. The
   expected size is `header_bytes + n_records × bytes_per_record`. A file that
   doesn't match is rejected.

The constructor throws `std::runtime_error` if the file cannot be opened, if the
magic is missing or unrecognized, or if the file size is inconsistent with the
header.

### `header()`

```cpp
const GenoHeader& header() const noexcept;
```

Returns the parsed header: the detected format, the individual and SNP counts, and
the derived per-record stride. The record stride is always *derived* from the
counts, never a hardcoded number.

### `records_present()`

```cpp
std::size_t records_present() const noexcept;
```

Returns how many *complete* records are actually present on disk. This can be fewer
than the count the header claims, when a file is partial or truncated. For an
individual-major TGENO file this is the number of individual records available.

This value matters for population selection: it should be passed to the `.ind`
reader so that selection ignores any individuals whose records are not actually on
disk. Selecting an absent individual would otherwise ask the reader to seek past
the end of the file.

---

## 4. Tile shape and the `snp_begin == 0` rule

Every read method takes a half-open SNP range `[snp_begin, snp_end)` and returns
only that slice of SNPs, for only the selected individuals. This tile-shaped
signature exists so a future SNP-tiling loop can call the reader once per tile and
stream a dataset larger than memory. Today's callers read the whole SNP axis in one
shot, calling with `snp_begin == 0` and `snp_end` equal to the SNP count.

**All read methods currently require `snp_begin == 0`, and reject a nonzero begin.**
The reason is byte alignment. Genotypes are packed four to a byte. When the SNP
range starts at zero, the first SNP lands on a byte boundary and the packed prefix
lines up cleanly with what the decoder expects. A nonzero start would begin partway
into a byte, so the codes would be misaligned. Rather than silently hand a future
caller mis-aligned codes, the reader rejects a nonzero `snp_begin` outright. That
capability is reserved for the future tiling loop, which will handle the alignment
explicitly.

For the `snp_begin == 0` case, the per-record byte range read is
`[0 .. ceil(snp_end / 4))`, and the resulting per-record stride is
`ceil((snp_end − snp_begin) / 4)` bytes.

---

## 5. `read_tile` — the TGENO individual-major reader

```cpp
GenotypeTile read_tile(const IndPartition& part,
                       std::size_t snp_begin,
                       std::size_t snp_end);
```

This is the canonical path for TGENO (individual-major) files. For each selected
individual — walked in the partition's order, so the result comes out
population-contiguous — the reader seeks to that individual's record, copies the
packed byte prefix covering the SNP range, and appends it to the tile. It also
fills in the population segment boundaries and labels from the partition. The output
is a `GenotypeTile`, already in the shape the decoder consumes.

This method only accepts TGENO files. A SNP-major GENO file is *not* read here — it
would be mis-decoded along the wrong axis — and is rejected so the caller routes it
to `read_snp_major_tile` instead.

It throws `std::runtime_error` on a read error, a SNP range outside the file, a
non-TGENO file, a malformed partition (see Section 10), or a failed tile
allocation.

---

## 6. `read_snp_major_tile` — the GENO / PACKEDANCESTRYMAP reader

```cpp
SnpMajorTile read_snp_major_tile(const IndPartition& part,
                                 std::size_t snp_begin,
                                 std::size_t snp_end);
```

This is the axis-swapped twin of `read_tile`, for packed SNP-major
(PACKEDANCESTRYMAP / GENO) files. Two things differ from the individual-major path:

- It seeks whole per-SNP records (`header_bytes + snp × bytes_per_record`) rather
  than per-individual records. Each source record is the full stride wide, with all
  individuals interleaved.
- The selection moves out of the seek loop and into a gather *list*. Because a
  SNP-major record interleaves all individuals within a single SNP's bytes, a row
  cannot be skipped at read time — so the reader records which source rows the
  transpose should keep, in population-contiguous order, and lets the transpose do
  the selecting.

No decode and no transpose happen here. The reader produces the raw SNP-major bytes
plus the gather list (the selected rows, the population offsets, and the population
labels), and the app layer hands that to the on-device
`ComputeBackend::transpose_to_canonical`, which applies the selection, the reorder,
and the axis-swap together to produce a canonical `GenotypeTile`.

It keeps the same fail-fast guards as `read_tile` (Section 10) and the same
`snp_begin == 0` requirement (Section 4). It throws on a read error, an
out-of-range SNP range, a non-GENO file (TGENO must use `read_tile`), a malformed
partition, or a failed allocation.

---

## 7. `read_eigenstrat_snp_major_tile` — the EIGENSTRAT reader

```cpp
SnpMajorTile read_eigenstrat_snp_major_tile(const IndPartition& part,
                                            std::size_t snp_begin,
                                            std::size_t snp_end);
```

The text twin of `read_snp_major_tile`. An EIGENSTRAT `.geno` file is ASCII and
SNP-major: one line per SNP, one character per individual, where the character is
the reference-allele copy count `0`, `1`, or `2`, and `9` means a missing call.

For the requested SNP range, the reader parses each line, maps each character to its
canonical 2-bit code, and *packs* the line into exactly the same canonical
SNP-major 2-bit layout that `read_snp_major_tile` produces from raw packed bytes
(record `s` is SNP `s`, four individuals per byte, most-significant bits first). It
then builds the identical selection/gather list. Because the character-to-code map
is the identity on the value, the resulting tile flows through the same transpose
with an identity encoding — nothing downstream changes. Only the front-end parse
(ASCII plus pack, versus a raw byte copy) and the geometry source differ.

Beyond the shared guards, this reader adds a **malformed-line guard**: a `.geno`
line whose length is not the individual count, or that carries any character other
than `0`, `1`, `2`, or `9`, is rejected with the offending SNP line number and
column reported (1-based). The line index *is* the SNP index, so a length mismatch
means the SNP axis has gone out of sync, which must fail loudly rather than corrupt
the data silently.

---

## 8. `read_plink_snp_major_tile` — the PLINK `.bed` reader

```cpp
SnpMajorTile read_plink_snp_major_tile(const IndPartition& part,
                                       std::size_t snp_begin,
                                       std::size_t snp_end);
```

The binary twin of `read_snp_major_tile`, for a PLINK `.bed` file. A `.bed` is
SNP-major, packed two bits per genotype, four individuals per byte — but with two
differences from the canonical packing that this reader normalizes away:

1. **Bit order is reversed.** PLINK packs the first individual in the *low* bits of
   a byte (least-significant-bits-first), the opposite of the canonical
   most-significant-bits-first order. The reader flips the bit order as it re-packs.
2. **The code values are different.** PLINK's 2-bit codes are remapped to the
   canonical raw-value codes: `00` becomes 2, `01` becomes missing, `10` becomes 1,
   and `11` becomes 0, counting copies of the A1 allele (which is taken as the
   reference). Note that PLINK's missing sentinel is `01`, not `11` — a distinct
   encoding that does not match the GENO/TGENO identity map.

Both the value remap and the bit-flip happen here, inside the `io` layer's gather,
so the transpose body and its encoding enum stay untouched and the tile can use the
identity encoding like the EIGENSTRAT case.

The `.bed` magic (the three bytes `0x6c 0x1b 0x01`, where the last byte marks
SNP-major mode) is consumed at construction, so the header width is 3 bytes and each
SNP record sits at `3 + s × bytes_per_record`.

It keeps the same fail-fast guards, the same `snp_begin == 0` requirement, and a
short-read throw, and reports a non-PLINK file, an out-of-range range, a malformed
partition, or a failed allocation as `std::runtime_error`.

---

## 9. `read_ancestrymap_snp_major_tile` — the ANCESTRYMAP reader

```cpp
SnpMajorTile read_ancestrymap_snp_major_tile(const IndPartition& part,
                                             std::size_t snp_begin,
                                             std::size_t snp_end);
```

The text-triple twin of the EIGENSTRAT reader, for the legacy unpacked EIGENSOFT[^eigensoft]
ANCESTRYMAP format. Its `.geno` is text with one line per *(SNP, individual)* pair:
`<snp_id> <sample_id> <genotype>`. The lines are laid out SNP-major — each SNP's
individual rows appear consecutively, in `.ind` order, with SNPs in `.snp` order —
so the file is addressed positionally: line `L` (0-based) is SNP `L / n_ind`,
individual `L % n_ind`.

Because the file has no header, the reader consumes the leading
`tile_snps × n_ind` lines sequentially. It maps each line's third token to its
canonical code (`0`/`1`/`2` are copy counts; `"-1"` is the ANCESTRYMAP missing
sentinel, distinct from EIGENSTRAT's `9`), and packs each value most-significant-
bits-first into the same canonical SNP-major layout the other readers produce. It
then builds the identical gather list and flows through the same transpose with the
identity encoding — only the parse and the missing sentinel differ.

Its **malformed-line guard**: a `.geno` line that does not carry exactly three
whitespace-separated tokens, or whose third token is not `0`, `1`, `2`, or `-1`, is
rejected with the 1-based line number plus the SNP/individual position it maps to.
The line index *is* the SNP-times-individual position, so a desync there corrupts
the genotype matrix and must fail loudly.

---

## 10. Fail-fast guards and the exception-type contract

Every read method validates its inputs at the boundary and fails immediately rather
than producing a wrong result or a memory-safety bug.

### Partition validation

The population partition passed in is checked before any bytes are gathered. Three
conditions are rejected with a thrown `std::runtime_error`:

- an **empty** partition (no selected populations),
- any selected individual **row index at or beyond the records present** — a stale
  or wrong-dataset partition, which would otherwise seek past the end of the file
  or, for SNP-major sources, read a padding byte as a phantom individual, and
- a gathered tile size that would **overflow** when multiplied out — which would
  otherwise silently wrap and cause a heap buffer overflow on the copy.

### The exception-type contract

Beyond input validation, memory allocation itself can fail — a legitimately large
tile whose backing buffer cannot be allocated. The standard-library calls that
allocate can throw `std::length_error` (when the requested size exceeds the maximum)
or `std::bad_alloc`. This reader **translates those into `std::runtime_error`**.

The point of the contract is uniformity: *every* way a read can fail surfaces as
`std::runtime_error`. A caller written to `catch (std::runtime_error&)` will catch
all of them; no raw `std::bad_alloc` or `std::logic_error` leaks out through a
different type.

Put together, each read method throws `std::runtime_error` on a read error, an
out-of-range SNP range, the wrong file format for that method, a malformed
partition, a malformed input line (for the text formats), or a failed allocation.

---

## 11. Private helpers

Two private methods factor out the work the four SNP-major readers share, so the
selection logic and the allocation-safety logic each live in exactly one place.

### `build_selection`

Builds the selection-plus-reorder gather list common to all four SNP-major readers.
It sets the tile's geometry (the source record stride and the SNP count), fills in
the population offsets and labels, and pushes each selected source row into the
gather list — after checking that the row is within the source's individual count,
so the transpose can never read a padding byte as a phantom individual. The readers
differ only in a `reader_name` string that prefixes the out-of-range error message.
Throws `std::runtime_error` on a row at or beyond the individual count.

### `checked_alloc_snp_major`

Allocates the raw SNP-major source buffer (`tile_snps × src_bpr` bytes) with the
checked-multiply and allocation-failure translation shared by all four SNP-major
readers. A silent size overflow, a `std::bad_alloc`, and a `std::length_error` are
each turned into the documented `std::runtime_error`.

Its `zero_init` flag chooses how the buffer is prepared. When true, the buffer is
zero-filled first — the text and PLINK packers OR their codes in bit by bit and need
a partially filled last byte to start at zero. When false, a bare resize is used —
the raw GENO copy overwrites every byte, so pre-zeroing would be wasted work. The
`reader_name` again prefixes the error messages.

---

## 12. Ownership and move semantics

`GenoReader` owns an open file stream and its parsed header, so it is **move-only**:
the copy constructor and copy assignment are deleted, while move construction and
move assignment are defaulted and marked `noexcept`. The destructor is the default.

Its private state is just the file path, the parsed `GenoHeader`, and the
`records_present_` count. Keeping one reader open and moving it (rather than copying)
across a tile loop is the intended usage — construction parses and validates once,
and each subsequent read reuses that already-open, already-validated state.

---

[^eigensoft]: **EIGENSOFT / convertf** — the EIGENSTRAT and ANCESTRYMAP genotype formats and the `convertf` converter. Patterson N, Price AL, Reich D. *Population structure and eigenanalysis.* PLoS Genetics 2006;2(12):e190.

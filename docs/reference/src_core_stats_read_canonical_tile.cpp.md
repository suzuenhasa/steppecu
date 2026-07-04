# `read_canonical_tile.cpp` reference

## 1. Purpose

`read_canonical_tile` is the single place the genotype-path tools turn an open
genotype file plus a chosen set of populations into one uniform in-memory tile,
no matter which on-disk file format the data came in. A "tile" here is a slab of
genotypes for a range of SNPs (`[snp_begin, snp_end)`) and a selected set of
individuals, grouped by population.

The problem it solves: steppe supports several genotype file formats, and they do
not all store their data the same way. Some store the data individual-by-individual
(one individual's genotypes across all SNPs are contiguous); others store it
SNP-by-SNP (one SNP's genotypes across all individuals are contiguous). Every
computation downstream — ploidy detection, allele-frequency decoding, the f2/f3/f4
statistics, D-statistics, and the dating tool — expects exactly one layout: the
**canonical individual-major tile**. This function is the funnel that produces that
one shape from any of the supported formats, so nothing downstream has to know or
care what the file on disk looked like.

The file contains two pieces: the public dispatch function `read_canonical_tile`,
and a small private helper `transpose_snp_major` that all the SNP-major formats
share.

---

## 2. The two on-disk axes and transpose-on-read

Genotype data is a two-dimensional table: SNPs on one axis, individuals on the
other. A file can store that table in one of two orders:

- **Individual-major** — an individual's genotypes across all SNPs sit together.
- **SNP-major** — a SNP's genotypes across all individuals sit together.

The canonical tile that steppe computes on is **individual-major**. One format
(TGENO) is already stored that way, so its path just reads straight through with
no rearrangement. Every other supported format is SNP-major on disk, so its data
has to be flipped from SNP-major to individual-major before anything can use it.

That flip happens **at read time, on the GPU**, and only at read time. This is the
key design choice: rather than teach every downstream stage to handle two possible
layouts, the front-end transposes SNP-major data into the canonical individual-major
shape once, as the tile is read. After this function returns, the tile is byte-for-byte
the same regardless of source format, so nothing downstream changes — the only thing
that ever differs between formats is the read-time gather-and-transpose that happens
inside this file.

---

## 3. Format dispatch

`read_canonical_tile` reads the format tag off the open reader's header and
switches on it. There are five recognized formats plus a defensive default.

| Format | What it is | How this function handles it |
|---|---|---|
| `Tgeno` | The individual-major native format. | Reads straight through with the existing `reader.read_tile(...)`. No transpose, and the GPU backend is never touched on this path. |
| `Geno` | PACKEDANCESTRYMAP — the packed binary SNP-major format, where genotypes are stored as raw 2-bit codes. | Reads the raw SNP-major bytes with the file-reading leaf, then runs the shared SNP-major transpose path (section 4). |
| `Eigenstrat` | The text SNP-major format, where each SNP is a line of ASCII digits. | Parses and packs the ASCII line into the canonical SNP-major layout, then runs the same shared transpose path. |
| `Plink` | The PLINK `.bed` binary SNP-major format (2-bit codes, but bit-ordered and value-coded differently from PACKEDANCESTRYMAP). | Reads and **normalizes** the `.bed` records into the canonical SNP-major layout, then runs the same shared transpose path. |
| `Ancestrymap` | The unpacked legacy EIGENSOFT text format[^eigensoft] — one line per (SNP, sample, genotype) triple. | Parses the triples and packs them into the canonical SNP-major layout, then runs the same shared transpose path. |
| `Unknown` / default | — | Throws `std::runtime_error`. See section 7. |

The four non-TGENO formats all end up calling the same helper, `transpose_snp_major`.
Their only real difference is the **gather** step — how each format's raw bytes or
text get turned into the one common SNP-major layout. Once that gather is done, the
work is identical.

---

## 4. The shared SNP-major transpose path

`transpose_snp_major` is the common back half that every SNP-major format runs
through. Each format's own reader is responsible for producing the **same** canonical
SNP-major source packing; this helper then transposes that packing to individual-major
and assembles the final tile. The design payoff is that there is exactly one transpose
implementation and exactly one tile-assembly implementation, shared by four formats.

### The canonical SNP-major source packing

Before this helper runs, each format's reader has already packed its data into one
agreed-upon layout:

- One record per SNP (record `s` holds SNP `s`).
- Within a record, source individual `i`'s value lives in byte `i/4`, packed
  most-significant-bits first (four individuals per byte, since each value is 2 bits).
- The values themselves are the canonical raw codes: `0`, `1`, `2` are reference-allele
  copy counts, and `3` is the missing-data code.

Because all four formats normalize to this exact packing, the transpose and assembly
that follow are byte-for-byte identical across them.

### The transpose call

The helper builds a lightweight **view** over the reader's SNP-major buffer — it does
not copy the genotype bytes, it borrows the reader's storage and describes it with
pointers, counts, and the population offsets. It then hands that view to
`backend.transpose_to_canonical(...)`, which does the actual SNP-major-to-individual-major
flip on the GPU (or, in the CPU test backend, in a host loop used as the correctness
oracle). The result comes back as a backend-local packed tile.

The view is tagged with the encoding `Identity`, which tells the transpose that the
native codes it is reading are already the canonical codes — no per-value remapping
is needed during the flip. Section 5 explains why every format can honestly claim
identity here.

### Assembling the output tile

The transpose produces the packed genotypes, the byte stride per record, the SNP and
individual counts, and the population offsets — but **not** the population labels,
because the transpose works on a backend-local plain-data structure that never carried
labels. So the helper stitches the final `io::GenotypeTile` together from two sources:
the packed data and offsets come from the transpose output, and the population labels
come from the original file-reading gather. The genotype data is moved (not copied)
out of the transpose result to avoid an extra allocation.

---

## 5. Per-format normalization: why the encoding is always identity

The reason all four SNP-major formats can share one transpose is that each format's
reader does whatever remapping its format needs **before** handing data to the
transpose, so by the time the transpose sees the bytes they are already canonical.
That is what "the encoding is `Identity`" means: the transpose does no value
translation, only a rearrangement.

| Format | What the reader does to reach the canonical packing |
|---|---|
| `Geno` (PACKEDANCESTRYMAP) | Almost nothing. This format already uses the canonical raw-value codes and the most-significant-bits-first bit order, so the reader gathers the raw bytes as-is. The encoding is genuinely identity with no remap at all. |
| `Eigenstrat` | Parses the ASCII line and packs it. The character-to-code map is the identity on the copy counts (`0`/`1`/`2` copies pass straight through), and the format's missing sentinel `9` becomes the canonical missing code. Only the parse differs from the `Geno` path. |
| `Ancestrymap` | Parses the positionally-addressed `<snp_id> <sample_id> <genotype>` triples and packs them. Like EIGENSTRAT the copy counts are the identity, and this format's missing sentinel `-1` becomes the canonical missing code. Only the parse and the choice of missing sentinel differ from the EIGENSTRAT path. |
| `Plink` (`.bed`) | The most involved normalization. PLINK stores its 2-bit codes **least-significant-bits first** (the opposite bit order) and with a **different value coding**. The reader flips the bit order to most-significant-bits-first and remaps every code through a fixed table (`kBedToCanon`): `.bed 00 → 2`, `.bed 01 → 3` (missing), `.bed 10 → 1`, `.bed 11 → 0`. This counts copies of the `.bed` first allele (`A1`), which is treated as the canonical reference allele. After this flip-and-remap, the buffer is byte-for-byte what the `Geno` path produces, so from the transpose's point of view the encoding is again identity. |

The single takeaway: format-specific quirks — bit order, value coding, missing
sentinels, text-vs-binary — are all absorbed by each format's own reader, so the
shared transpose only ever sees one canonical layout.

---

## 6. Ploidy and labels

Two fields of the assembled tile deserve a specific note because they are populated
(or deliberately not populated) outside the normal packed-data flow:

- **Population labels** are *not* produced by the transpose (which only knows about
  packed bytes and offsets). They are carried over from the original file-reading
  gather and attached during assembly.
- **`sample_ploidy` is deliberately left empty.** This function does not detect or set
  ploidy. Ploidy is determined later, once the canonical tile exists — either by a
  separate on-device pre-pass over the tile, or by an explicit ploidy vector the caller
  supplies. The format dispatch never touches ploidy.

---

## 7. Error handling

If the format tag is `Unknown` or anything not covered by the five arms above, the
function throws `std::runtime_error` with a message naming the supported formats.

This throw is **defensive**, not a primary validation gate: the reader's constructor
already rejects any file whose format it does not recognize, so a genuinely unknown
file never reaches this switch. The arm exists to catch a future format that gets a
reader but is added without a matching dispatch arm here — a programming mistake caught
loudly rather than silently mis-handled.

Any failure inside the reader gather or the GPU transpose propagates out as an
exception, matching the same failure contract the underlying read functions already
carry.

---

[^eigensoft]: **EIGENSOFT / convertf** — the EIGENSTRAT and ANCESTRYMAP genotype formats and the `convertf` converter. Patterson N, Price AL, Reich D. *Population structure and eigenanalysis.* PLoS Genetics 2006;2(12):e190.

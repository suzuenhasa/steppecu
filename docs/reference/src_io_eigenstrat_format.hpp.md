# `eigenstrat_format.hpp` reference

## 1. Purpose

`src/io/eigenstrat_format.hpp` is the single home for the low-level format
literals and helper functions used to read genotype files. It covers the
EIGENSTRAT family (packed `TGENO`/`GENO`, ASCII EIGENSTRAT, and legacy
ANCESTRYMAP) and the PLINK `.bed` binary format.

Every "magic number" that describes how these files are laid out on disk — the
fixed header width, the packing stride, the format magic strings, the 2-bit code
values, the chromosome codes, the `.snp` column positions — lives here once with
a name and an explanation. Decode code refers to these names instead of
re-typing a bare number that could quietly drift out of sync. In particular, the
48-byte header record and the `ceil(n_snp / 4)` record stride are *derived* from
the parsed header, never hardcoded at a decode call site.

The header is deliberately lightweight: it is pure host C++20 and includes only
the standard library. It has no CUDA code and no dependency on the core or device
layers, so it can be included freely by the file-reading code without dragging in
the GPU stack.

---

## 2. The two packed layouts: TGENO and GENO

Two packed on-disk genotype layouts are recognized. They store the same two
counts (number of individuals and number of SNPs) but differ in which axis a
record represents.

- **TGENO** (magic `"TGENO"`) is **individual-major**: one record per
  individual, `ceil(n_snp / 4)` bytes long, 4 SNPs per byte. This is the real
  AADR v66 layout — a header reading `TGENO 27594 584131` means 27,594
  individuals and 584,131 SNPs. Within an individual's record, SNP `k` lives in
  byte `k / 4`, and its 2-bit code is read most-significant-bits first (see
  section 7).
- **GENO** (magic `"GENO"`) is the SNP-major PACKEDANCESTRYMAP layout: one record
  per SNP, `max(48, ceil(n_ind / 4))` bytes long, 4 individuals per byte.

GENO is recognized so the reader can tell the two apart and refuse to
mis-decode. The packed decode path targets TGENO (the real data); a GENO file is
reported rather than silently read along the wrong axis.

---

## 3. The 2-bit genotype code mapping

Every format in this header maps a genotype to the same canonical 2-bit code:

| Code | Meaning |
|---|---|
| `0` | 0 reference-allele copies |
| `1` | 1 reference-allele copy (heterozygous) |
| `2` | 2 reference-allele copies |
| `3` | MISSING |

This is the **raw-value** mapping: the code *is* the reference-allele copy count
for 0/1/2, and 3 means missing. It was verified bit-for-bit against an
independent oracle (a Python reference that builds the genotype matrix
directly). It is deliberately **not** the binary mapping
(`00→0, 10→1, 11→2, 01→missing`), which mis-decodes the data; only the raw-value
mapping reproduces the oracle exactly.

A missing code is excluded from **both** the numerator and the denominator of an
allele-frequency calculation.

---

## 4. Packed-format constants

These constants describe the packed `.geno` (TGENO/GENO) format. They are kept
here once so no decode site re-spells them and the header parse can derive the
record stride from them.

| Constant | Value | What it's for |
|---|---|---|
| `kGenoHeaderBytes` | `48` | Bytes in the leading ASCII header record of a packed `.geno` file. The header is the magic, then the two decimal counts and dataset hashes, NUL-padded to this fixed width. Verified against the real AADR v66 TGENO header. |
| `kMagicTgeno` | `"TGENO"` | The format magic for the individual-major TGENO packing — the first whitespace-delimited token of the header. |
| `kMagicGeno` | `"GENO"` | The format magic for the SNP-major GENO (PACKEDANCESTRYMAP) packing. Recognized so the two axes can be told apart and a GENO file is not mis-decoded. |
| `kCodesPerByte` | `4` | How many 2-bit genotype codes are packed per byte (4 SNPs/byte for TGENO, 4 individuals/byte for GENO). A structural constant of the 2-bit packing. |
| `kBitsPerCode` | `2` | Bits per packed genotype code. |
| `kCodeMask` | `0x3` | Low-bit mask for a single packed code, derived from `kBitsPerCode` so the mask can never desync from the packing width. Used to isolate an extracted code. |
| `kMissingCode` | `3` | The 2-bit code that denotes a missing genotype (see section 3). Excluded from both the numerator and denominator of the allele frequency. |
| `kHetCode` | `1` | The 2-bit code that denotes a heterozygous genotype (1 reference-allele copy). Used by the pseudo-haploid auto-detection below. |
| `kPloidyDetectSnps` | `1000` | The pseudo-haploid auto-detection window — the number of leading SNPs scanned per sample looking for a heterozygous call. |

### Pseudo-haploid auto-detection

steppe optionally auto-detects whether each sample is diploid or
pseudo-haploid[^at2] (the `adjust_pseudohaploid` option). The rule this header
supports: a sample is **diploid** if it has at least one heterozygous call
(`kHetCode`) within the first `kPloidyDetectSnps` SNPs, because a truly haploid
genome can never be heterozygous; otherwise it is treated as **pseudo-haploid**.
The window default of `1000` is the parity `ntest` default. A
record shorter than the window simply scans fewer SNPs. `kHetCode` and
`kPloidyDetectSnps` are mirrored from the core layer's equivalents, kept here so
the file-reading layer does not have to depend on core.

---

## 5. Chromosome codes for the `.snp` file

The EIGENSTRAT `.snp` file labels each SNP with a chromosome code. These three
constants name the non-autosomal codes so the label→code convention lives in one
place instead of as bare literals at the `.snp` parse site. They are *format*
conventions (the EIGENSOFT integer codes[^eigensoft]), not mathematical constants.

| Constant | Value | Chromosome |
|---|---|---|
| `kChromCodeX` | `23` | X (sex) chromosome |
| `kChromCodeY` | `24` | Y (sex) chromosome |
| `kChromCodeMt` | `90` | Mitochondrial ("MT") |

**Correctness coupling.** The "autosomes only" filter keeps chromosomes 1
through 22 and drops everything else. That range is defined once (in the config
header) and it is precisely these codes — 23, 24, 90, and any other
non-autosomal label — that fall outside it and get dropped. The `.snp` reader
*emits* these codes and the autosome filter *excludes* them, so both sides must
agree on the exact X = 23 / Y = 24 / MT = 90 mapping. It is single-sourced here
so the two cannot drift.

---

## 6. The `.snp` text-record columns

An EIGENSTRAT `.snp` record is a whitespace-separated text line with the layout:

```
<id> <chrom> <genpos> [<physpos> <ref> <alt>]
```

The first three fields are mandatory; the physical position and the two alleles
are an optional extension. These constants describe the layout so the field-count
gates, the column indices, and the prose can never drift apart.

| Constant | Value | What it's for |
|---|---|---|
| `kMinSnpFields` | `3` | Minimum well-formed field count (`<id> <chrom> <genpos>`). Fewer than this means the record is malformed. |
| `kFullSnpFields` | `6` | The full 6-column form that carries an explicit reference and alternate allele. |
| `kPhysposCol` | `3` | 0-based column index of the physical position (base pairs). Present when the record has 4 or more fields. It feeds the base-pair block fallback used when the genetic-position column is all zero. |
| `kRefAlleleCol` | `4` | 0-based column index of the reference allele in a full 6-column record. |
| `kAltAlleleCol` | `5` | 0-based column index of the alternate allele in a full 6-column record. |
| `kMissingAllele` | `'N'` | The EIGENSTRAT "missing/unknown base" — the default reference and alternate allele for a record with fewer than 6 columns. |
| `kFirstOtherChromCode` | `-1` | The first synthetic code assigned to a chromosome label that is neither numeric nor X/Y/MT. Subsequent distinct labels decrement (−1, −2, …). All such codes are outside the 1–22 autosome range, so these labels are dropped by the autosomes-only filter. |

---

## 7. Bit-packing helpers: `packed_bytes` and `code_in_byte`

Two `constexpr` helpers do the actual bit arithmetic, so the packing formula and
the bit order each live at exactly one site.

### `packed_bytes(n_codes)`

Returns how many bytes are needed to pack `n_codes` 2-bit codes at 4 per byte —
the ceiling division `ceil(n_codes / 4)`. This is the record-stride formula:
`ceil(n_snp / 4)` for TGENO and `ceil(n_ind / 4)` for GENO. It is computed here
and never open-coded elsewhere.

### `code_in_byte(byte, k)`

Extracts the 2-bit code at position `k` (0-based) within a packed byte,
**most-significant-bits first**:

| Position `k` (mod 4) | Bits |
|---|---|
| 0 | 7–6 |
| 1 | 5–4 |
| 2 | 3–2 |
| 3 | 1–0 |

This is the rule `(byte >> (6 − 2·(k mod 4))) & 3`. It is the single bit-order
site shared by the host reader and, through the decode primitive, the GPU and CPU
decoders — so the MSB-first convention is defined in exactly one place.

---

## 8. `GenoFormat`: the recognized on-disk layouts

`GenoFormat` is an enum naming which on-disk genotype layout a file uses — its
row axis and packing. Its values:

| Value | Layout |
|---|---|
| `Unknown` | Not yet identified, or unrecognized. |
| `Tgeno` | Packed, individual-major (see section 2). |
| `Geno` | Packed, SNP-major PACKEDANCESTRYMAP (see section 2). |
| `Eigenstrat` | ASCII SNP-major: one text line per SNP, one character per individual. |
| `Plink` | The `.bed`/`.bim`/`.fam` triple — SNP-major 2-bit binary. |
| `Ancestrymap` | The legacy unpacked EIGENSOFT text format — one line per (SNP, individual) pair. |

How the non-packed formats are handled:

- **EIGENSTRAT** is the ASCII SNP-major form: one text line per SNP, one
  character per individual (`0`/`1`/`2` reference-allele copies, `9` missing). It
  has no packed magic, so it is detected from its leading ASCII content.
- **PLINK** is the `.bed`/`.bim`/`.fam` triple. The `.bed` is SNP-major 2-bit
  data behind a 3-byte magic; its geometry (number of SNPs, number of
  individuals) comes from the `.bim` and `.fam` line counts, and its 2-bit codes
  are remapped and bit-order-flipped to the canonical convention while reading.
  EIGENSTRAT / GENO / TGENO share the sibling `.snp`/`.ind` files, while PLINK
  uses `.bim`/`.fam`.
- **ANCESTRYMAP** is the legacy unpacked EIGENSOFT format: a text `.geno` with one
  line per (SNP, individual) pair, `<snp_id> <sample_id> <genotype>`, laid out
  SNP-major (each SNP's individual rows are consecutive, in `.ind` order; SNPs in
  `.snp` order). The genotype is the reference-allele copy count `0`/`1`/`2`, or
  `-1` for missing (distinct from EIGENSTRAT's `9`). Like PLINK, its geometry
  comes from the sibling metadata files' line counts, because the `.geno` carries
  no header.

The common thread: EIGENSTRAT, GENO, PLINK, and ANCESTRYMAP are all SNP-major.
Each is read, remapped to the canonical raw-value codes of section 3, packed
SNP-major, and passed through the same on-device transpose to reach the canonical
individual-major tile. Because the codes are already canonical after the remap,
that transpose runs as an identity on the values (no further remapping) for
every one of these formats.

---

## 9. PLINK `.bed` constants

The PLINK `.bed` file has its own magic, its own packing bit order, and its own
code encoding, all single-homed here.

### The `.bed` magic

A `.bed` begins with a 3-byte magic: two fixed bytes followed by a mode byte.

| Constant | Value | Meaning |
|---|---|---|
| `kBedMagic0` | `0x6c` | First magic byte. |
| `kBedMagic1` | `0x1b` | Second magic byte. |
| `kBedModeSnpMajor` | `0x01` | The mode byte for SNP-major layout — the only mode steppe reads. A `0x00` mode byte would be legacy individual-major and is rejected. |
| `kBedMagicBytes` | `3` | Total bytes of leading magic; the SNP records start at this offset. |

After the magic come SNP-major records: one record per SNP, `ceil(n_ind / 4)`
bytes, 4 individuals per byte, packed **least-significant-bits first** — the
opposite of the canonical MSB-first packing.

### The `.bed` code remap: `kBedToCanon`

PLINK uses a distinct 2-bit encoding whose missing sentinel is `01`, not `11`, so
it does **not** match the identity map that GENO/TGENO use. `kBedToCanon` is the
4-entry lookup table that converts a `.bed` code (read LSB-first, then taken as a
plain 2-bit integer) to the canonical raw-value code. Note that PLINK counts
copies of the A1 allele, and steppe takes the reference allele to be A1, so an A1
count is a reference-allele count:

| `.bed` value | PLINK meaning | Canonical code |
|---|---|---|
| `00` (0) | homozygous A1 = 2 A1 copies | `2` |
| `01` (1) | missing | `3` (`kMissingCode`) |
| `10` (2) | heterozygous = 1 A1 copy | `1` (`kHetCode`) |
| `11` (3) | homozygous A2 = 0 A1 copies | `0` |

The table is single-homed so the reader (and any future device twin) share the
one lookup.

### `bed_code_in_byte(byte, k)`

Extracts the 2-bit `.bed` code at sample position `k` (0-based),
**least-significant-bits first** — the opposite shift direction from
`code_in_byte`:

| Position `k` (mod 4) | Bits |
|---|---|
| 0 | 1–0 |
| 1 | 3–2 |
| 2 | 5–4 |
| 3 | 7–6 |

This is the rule `(byte >> (2·(k mod 4))) & 3`, and it is the single PLINK
bit-order site.

---

## 10. EIGENSTRAT ASCII `.geno` constants

The ASCII EIGENSTRAT `.geno` is one line per SNP, one character per individual.
The character *is* the reference-allele copy count (`0`/`1`/`2`), or `9` for a
missing call. These constants and one helper define the character convention.

| Constant | Value | Meaning |
|---|---|---|
| `kEigenstratMissingChar` | `'9'` | The ASCII digit for a missing EIGENSTRAT genotype. |
| `kEigenstratMissingCode` | `3` (`kMissingCode`) | The canonical code a missing character maps to. Kept beside `kEigenstratMissingChar` so the character→code mapping cannot drift. |
| `kEigenstratMaxCopiesChar` | `'2'` | The largest reference-allele copy-count character allowed. Valid input is `'0'`, `'1'`, `'2'`, or `'9'` — nothing else. |

### `eigenstrat_char_to_code(c, out)`

Maps one `.geno` character to its canonical 2-bit code. `'0'`/`'1'`/`'2'` map to
the copy count 0/1/2 (the character value *is* the code, the same raw-value
convention TGENO/GENO use), and `'9'` maps to `kMissingCode`. It returns `true`
and writes `out` on a recognized character, and returns `false` on any other byte
so the caller can fail loudly with the line-and-column context. Because the
ASCII-to-2-bit pack happens on the host and the canonical bytes then flow through
the same transpose as GENO, there is no GPU twin — this is the single EIGENSTRAT
character→code site.

---

## 11. ANCESTRYMAP text constants

The legacy ANCESTRYMAP `.geno` is one line per (SNP, individual) pair:
`<snp_id> <sample_id> <genotype>`, whitespace-separated. (The `convertf` tool
right-justifies the fields, so arbitrary leading or intervening runs of spaces
are normal.) The genotype token is the reference-allele copy count `"0"`/`"1"`/
`"2"`, or `"-1"` for missing — a **different** missing sentinel from EIGENSTRAT's
`'9'`.

| Constant | Value | Meaning |
|---|---|---|
| `kAncestrymapFields` | `3` | The exact whitespace-separated field count of a valid line. Any other count means the line is not an ANCESTRYMAP record. |
| `kAncestrymapMissingToken` | `"-1"` | The missing-genotype token, mapping to `kMissingCode`. Distinct from EIGENSTRAT's single-character `'9'`. |

### Positional addressing

The `.geno` carries no header and no geometry; the number of SNPs and
individuals comes from the sibling `.snp` and `.ind` line counts (the same way
PLINK derives geometry from `.bim`/`.fam`). The file is addressed positionally:
0-based line index `L` is SNP `L / n_ind` and individual `L % n_ind`. Each SNP id
spans exactly `n_ind` consecutive lines in `.ind` order, with SNPs in `.snp`
order.

### `ancestrymap_token_to_code(tok, out)`

Maps one genotype token (the third whitespace field) to its canonical 2-bit code.
`"0"`/`"1"`/`"2"` map to the copy count 0/1/2, and `"-1"` maps to `kMissingCode`.
It returns `true` and writes `out` on a recognized token, and `false` on anything
else so the caller can fail loudly with the SNP/individual context. As with
EIGENSTRAT, the text-to-2-bit pack happens host-side and the canonical bytes flow
through the same transpose as GENO, so there is no GPU twin.

---

## 12. `GenoHeader` and `parse_geno_header`

### `GenoHeader`

The parsed result of reading a packed `.geno` header: the detected format, the
two dataset counts, and the *derived* record geometry.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `format` | `GenoFormat` | `Unknown` | Which packed layout was recognized (see section 8). |
| `n_ind` | `size_t` | `0` | Number of individuals (samples). |
| `n_snp` | `size_t` | `0` | Number of SNPs. |
| `n_records` | `size_t` | `0` | Records on disk: `n_ind` for TGENO, `n_snp` for GENO. |
| `bytes_per_record` | `size_t` | `0` | The stride between records — derived, never hardcoded. For TGENO this is `packed_bytes(n_snp)`; for GENO it is `max(kGenoHeaderBytes, packed_bytes(n_ind))` (the PACKEDANCESTRYMAP record-length floor). |
| `header_bytes` | `size_t` | `48` (`kGenoHeaderBytes`) | The width of the leading header record. |

**The header-offset invariant.** Any data record lives at
`header_bytes + record × bytes_per_record`. Crucially, `header_bytes` is **not**
always the `kGenoHeaderBytes` (48) constant. TGENO's header is a fixed 48-byte
record, but GENO writes its header into one full record-width slot, so for GENO
`header_bytes` equals `bytes_per_record` (for example 6899 on v66 data, not 48).
Always use the `header_bytes` field — never the `kGenoHeaderBytes` constant — to
seek to a data record.

### `parse_geno_header(head)`

Parses the leading `kGenoHeaderBytes` of a `.geno` header buffer. The argument
must point to at least `kGenoHeaderBytes` bytes. It recognizes the `"TGENO"` or
`"GENO"` magic and the two decimal counts that follow (both formats store the
same two numbers — the difference is only which axis a record represents).

On a bad magic or unparsable counts it returns a header with
`format == Unknown` and leaves the caller to decide how to fail; the function
itself is `noexcept` and never throws across the layer boundary on a format
probe.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
[^eigensoft]: **EIGENSOFT / convertf** — the EIGENSTRAT and ANCESTRYMAP genotype formats and the `convertf` converter. Patterson N, Price AL, Reich D. *Population structure and eigenanalysis.* PLoS Genetics 2006;2(12):e190.

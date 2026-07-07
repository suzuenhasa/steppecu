# `panel_merge_writer.cpp` reference

## 1. Purpose

`src/io/panel_merge_writer.cpp` is the first genotype *writer* in the tree. Its one
public entry point, `write_merged_panel`, takes an existing EIGENSTRAT-family panel
and appends **one** new trailing individual to it — a fresh sample that becomes its
own size-1 population — writing out a complete new `.geno`/`.snp`/`.ind` triple.
Everyone who was already in the panel keeps their genotypes byte-for-byte; the only
thing that grows is the new column (or, for the individual-major format, the new
record) at the end.

This is the "place the new sample among the ancients" merge — Stage 3 of the
ingestion build. The typical caller is `steppe ingest` (see
`src/app/cmd_ingest.cpp`): it decodes an incoming sample (VCF, etc.) into
panel-aligned 2-bit codes and then hands those codes here to be stitched onto the
reference panel (for the 1240K AADR panel, that's the individual-major TGENO file).

It is a pure host `io`-leaf translation unit. No CUDA, no `core`/`device`
dependency, standard library only. Every failure surfaces as a
`std::runtime_error` — there are no silent partial writes and no error return
codes; either the merged triple is written whole or an exception is thrown.

---

## 2. The one-column contract, and why it's format-preserving

The whole design turns on a single promise: **the existing panel is not
re-encoded.** The new individual is appended, and nothing else moves. That promise
is the reason the merge is fast (a mostly-verbatim byte copy) and the reason it's
trustworthy (a downstream f2 over the old populations sees identical bytes to what
it saw before the merge).

How "append one individual" is realized depends on the on-disk layout, and there
are three of them (sections 4–6). The format is auto-detected from the source
`.geno` (section 3); the caller never has to say which one it is.

Two sibling files are handled the same way across all three branches, in
`write_merged_panel` before the `.geno` work begins:

- **`.snp` is copied verbatim.** The SNP set is unchanged by adding an individual,
  so `copy_file_bytes` byte-copies it in 4 MiB chunks. This keeps `shash`-style SNP
  hashes valid for free.
- **`.ind` gains exactly one row.** `write_merged_ind` slurps the source `.ind`,
  writes it back verbatim, guarantees a trailing newline if the last line lacked
  one, then appends a single `"<label>\tU\t<label>"` row. The label is used for
  **both** the individual id and the population name, which is what makes the new
  sample a size-1 population. `U` is the sex/placeholder column.

The `label` is required to be non-empty — an empty label throws before any file is
touched.

---

## 3. Format detection (`detect_geno`)

Detection reads the first 48 bytes of the source `.geno` into a header buffer and
asks the shared `parse_geno_header` (in `eigenstrat_format.hpp`) what it is:

- If the magic says **TGENO** or **GENO/PACKEDANCESTRYMAP**, that packed format
  wins immediately, and the parsed `GenoHeader` (with its `n_ind`, `n_snp`, and
  `header_bytes`) is returned.
- Otherwise the file is probed as **EIGENSTRAT ASCII** by geometry: re-open the
  file, read it line by line, and require that every character on every line is a
  valid EIGENSTRAT genotype char (`0`/`1`/`2`/`9`). The first line's length is
  taken as `n_ind`; every later line must match that width; the line count is
  `n_snp`. Any non-genotype character, a ragged line, or an interior blank line
  makes the probe bail out to `Unknown`.

Anything that lands on `Unknown` (or on the parsed-but-unsupported `Ancestrymap` /
`Plink` formats) is rejected by `write_merged_panel` with an "unsupported source
`.geno` format" error. Only the three branches below are writable.

A note on trailing blank lines in the ASCII probe and writer: a single blank line
at true EOF is tolerated (a common trailing newline), but a blank line *between*
data rows is an error, not a silently-skipped row.

---

## 4. TGENO append — the primary AADR-1240K path (`merge_tgeno`)

TGENO is **individual-major**: each individual is one self-contained record of
`packed_bytes(n_snp)` bytes (four 2-bit SNP codes per byte). This is the cleanest
case and the one that matters most in practice, because the real 1240K `.PUB`
reference panel is TGENO.

The algorithm:

1. Rebuild the 48-byte header with `n_ind` bumped to `n_ind + 1` (section 7).
2. **Byte-copy the entire `n_ind`-record data region verbatim**, seeking past the
   original header first and streaming in 4 MiB chunks. This copy is the structural
   proof of the one-column contract: not a single existing individual's bytes are
   touched.
3. Build the new sample's single packed record from `nikki_codes` — pack each 2-bit
   code into its slot with `pack_code_into_byte` — and append it after the copied
   region.

Because the new record starts from a fresh zero-filled buffer, a plain OR into each
byte is safe (no stale bits to clear). A code greater than `kMissingCode` (3) is
clamped to missing before packing, so an out-of-range input can never corrupt an
adjacent slot.

The spec originally assumed the panel was PACKEDANCESTRYMAP and said to *refuse*
TGENO; but the real panel is TGENO, steppe already decodes TGENO correctly, and a
native TGENO append avoids a ~7 GB offline `convertf` TGENO→PA conversion. So this
branch exists deliberately, and for a TGENO source it is exact.

---

## 5. GENO / PACKEDANCESTRYMAP append (`merge_geno`)

GENO/PA is **SNP-major**: each record is one SNP, holding all individuals' 2-bit
codes packed four-to-a-byte, padded up to a record stride of at least 48 bytes. The
new individual becomes slot `n_ind` in *every* SNP record, so this is a
prefix-append done row by row:

- `meaningful = packed_bytes(n_ind)` is the count of bytes actually carrying the old
  individuals; the source record stride is `max(48, meaningful)`.
- The output stride is `max(48, packed_bytes(n_ind + 1))`. It grows by one byte
  exactly when `n_ind` was a multiple of 4 (the new slot opens a fresh byte);
  otherwise the new code lands in the previously-partial last meaningful byte.

For each SNP record: read the source record, zero the output record, copy the
`meaningful` prefix bytes across, then write the new code into its slot. The write
is deliberately a **mask-clear-then-OR**, not a bare OR: the new slot's two bits are
cleared first, so any nonzero padding bits left in a previously-partial last byte
can't bleed into the new genotype. (A bare OR was the earlier bug; the mask-clear is
the fix.) Codes above `kMissingCode` are clamped to missing, same as TGENO.

The header is rebuilt into the *output* stride's worth of bytes (section 7), so a
short GENO header region that grew a byte is still fully zero-padded.

---

## 6. EIGENSTRAT ASCII append (`merge_eigenstrat`)

The text format is the simplest: one line per SNP, one character per individual.
Appending an individual means appending one character to the end of every line.

The writer streams the source line by line, tolerates a stray `\r` (CRLF input),
enforces the same geometry rules as the probe (first line sets the width, later
lines must match, interior blanks are errors), and appends
`code_to_eigenstrat_char(code)` — mapping 0/1/2 to `'0'`/`'1'`/`'2'` and everything
else to the missing char `'9'` — before the newline. It cross-checks that the
number of `.geno` rows consumed equals `nikki_codes.size()` at the end.

---

## 7. Rebuilding the packed header (`rebuild_packed_header`)

The packed TGENO/GENO header is a short ASCII string
(`MAGIC n_ind n_snp [ihash shash]`) living in a fixed 48-byte, zero-padded region.
To change `n_ind` without disturbing the rest:

1. Recover the original tokens by reading up to the first NUL.
2. Re-emit `magic new_n_ind n_snp`, then append **every original token from index 3
   onward verbatim** — i.e. the trailing hash tokens are kept exactly as they were.
3. Write the rebuilt string into a zero-filled buffer of the requested region
   length (48 for TGENO; the output stride for GENO).

Keeping the hash tokens verbatim is a deliberate choice. The `shash` (over the
unchanged `.snp`) stays valid. The `ihash` (over the individuals) is technically
*stale* after adding a sample — but steppe's own reader (`parse_geno_header`) reads
only `magic`, `n_ind`, and `n_snp` and ignores both hashes, so a stale `ihash`
changes nothing steppe acts on, and rewriting it would only invite a subtly-wrong
recomputation. If the rebuilt string somehow exceeds 48 bytes, that's a hard error
rather than a silent truncation.

---

## 8. Contracts and invariants

- **`nikki_codes` must be panel-aligned.** Its length must equal the source
  `n_snp`; a mismatch throws (per-branch, with the two counts named). One 2-bit code
  per panel SNP row, values in `{0,1,2,3}` where 3 is `kMissingCode`. Values above 3
  are clamped to missing rather than rejected.
- **Existing individuals are byte-identical after the merge.** The TGENO region copy
  and the GENO prefix copy are the mechanical guarantees of this; the EIGENSTRAT
  writer only appends to each line.
- **`n_ind_out == n_ind_src + 1`, always.** Exactly one individual is added, ever.
- **`.snp` is byte-identical; `.ind` grows by exactly one row.**
- **All-or-nothing on failure.** Every open, every short read, every failed write
  throws `std::runtime_error`. The output files are opened with `trunc`, so a failed
  run leaves a truncated/partial output that the caller is expected to discard — the
  contract is on the *thrown exception*, not on the output being usable after a
  throw.

### What `MergeCounts` reports back

The return value carries the detected `format`, `n_snp`, `n_ind_src`,
`n_ind_out`, and the tally of the new sample's `n_called` (codes in {0,1,2}) versus
`n_missing` (code 3). The caller prints these as a one-line merge summary and a
gate can assert on them.

---

## 9. Edge cases worth calling out

- **`n_ind % 4 == 0` in GENO** grows the record stride by a byte; the output-stride
  arithmetic and the zero-fill handle it, and the new slot opens cleanly in a fresh
  byte.
- **A previously-partial last byte in GENO** (nonzero padding bits) is handled by the
  mask-clear-then-OR; a bare OR would have corrupted the appended genotype.
- **CRLF EIGENSTRAT input** is tolerated (the trailing `\r` is stripped per line).
- **A trailing blank line at EOF** is accepted in both the ASCII probe and writer; a
  blank line *between* data rows is a hard error.
- **A source `.ind` with no trailing newline** still gets a correctly-separated
  appended row (the writer inserts the missing newline first).
- **Truncated packed files** are caught: a short read of the TGENO data region or of
  any GENO SNP record throws rather than producing a malformed output.
- **Unsupported formats** — ANCESTRYMAP text, PLINK `.bed`, or anything the probe
  can't classify — are rejected up front with a clear "need TGENO, GENO/
  PACKEDANCESTRYMAP, or EIGENSTRAT ASCII" message.

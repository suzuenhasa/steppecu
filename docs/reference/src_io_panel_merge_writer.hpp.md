# `panel_merge_writer.hpp` reference

## 1. Purpose

`src/io/panel_merge_writer.hpp` declares the first genotype **writer** in the
tree. Everywhere else in steppe an EIGENSTRAT-family `.geno`/`.snp`/`.ind` triple
is something we read; here it is something we produce. The one job is narrow and
concrete: take an existing panel and append **one** new trailing individual — a
size-1 population — to it, format-preserving.

This is Stage 3 of the "place nikki among the ancients" ingestion build. A single
modern sample (nikki) has been called against the panel's SNP list; this writer
splices her in as one more column/record so the merged panel can then be used like
any other. "Format-preserving" is the load-bearing promise: whatever layout the
source panel was in — packed TGENO, packed GENO/PACKEDANCESTRYMAP, or ASCII
EIGENSTRAT — the output is the same layout with exactly one individual added, and
**every pre-existing individual's genotype is left byte-for-byte unchanged.**

The header exposes two things: a `MergeCounts` accounting struct and the single
free function `write_merged_panel`. The algorithm itself lives in the paired
`panel_merge_writer.cpp`; this document describes the whole subsystem, since the
header's own comment is where the design was written down. It is a pure host
C++20 io-leaf — no CUDA, no `core`/`device` dependency — so it costs nothing to
compile into the file-IO layer.

---

## 2. The public surface

### `MergeCounts`

The per-run tally the caller prints and any gate checks against. Every field is a
plain count filled in by the merge:

| Field | Meaning |
|---|---|
| `format` | Which on-disk branch ran (`Tgeno`, `Geno`, or `Eigenstrat`). |
| `n_snp` | Panel SNP rows — equal to `nikki_codes.size()`. |
| `n_ind_src` | Source individual count, before the merge. |
| `n_ind_out` | `n_ind_src + 1` — always exactly one more. |
| `n_called` | How many of nikki's codes were real calls, in `{0,1,2}`. |
| `n_missing` | How many were missing (`kMissingCode`, the value 3). |

`n_called + n_missing` always equals `n_snp` — every SNP row contributes exactly
one of nikki's codes, and every code is classified as called-or-missing.

### `write_merged_panel`

```
MergeCounts write_merged_panel(const std::string& panel_prefix,
                               const std::string& out_prefix,
                               const std::vector<std::uint8_t>& nikki_codes,
                               const std::string& label);
```

Read the panel at `panel_prefix.{geno,snp,ind}`, append the individual described
by `nikki_codes` + `label`, and write the merged triple to
`out_prefix.{geno,snp,ind}`. `nikki_codes` carries one 2-bit code per panel SNP
row, in `{0,1,2,3}` where 3 is missing. `label` names the new individual for
**both** its `.ind` id and its population — a size-1 population all its own.

It returns the `MergeCounts` tally on success and throws `std::runtime_error` on
any open/format/size failure (section 7).

---

## 3. Three files, three fates

A panel is a triple, and the merge treats each of the three files differently
according to what actually changed:

- **`.snp` — copied byte-identically.** Adding an individual changes nothing about
  the SNP list, so the source `.snp` is streamed verbatim to the output. No
  reformatting, no re-derivation.
- **`.ind` — copied, then one row appended.** The source `.ind` is read whole and
  written out unchanged, and a single `"<label>\tU\t<label>"` row is appended: the
  new individual, sex `U` (unknown), belonging to a population named after itself.
  The writer guarantees a newline separates the appended row even if the source
  `.ind` didn't end in one — a panel that was missing its final newline still
  merges cleanly.
- **`.geno` — the real work.** Only here does the genotype data grow, and how it
  grows depends entirely on the on-disk layout (sections 4–6).

An empty `label` is rejected up front, before any file is touched, so a bad call
never leaves a half-written output triple.

---

## 4. Format auto-detection

The writer never asks the caller what format the panel is in — it figures it out
from the source `.geno` itself:

1. Read the first 48 bytes and hand them to `parse_geno_header`. If that reports a
   packed **TGENO** or **GENO** magic, that's the branch, and its `n_ind`/`n_snp`
   come straight from the header.
2. Otherwise, probe for **EIGENSTRAT ASCII geometry**: re-scan the file line by
   line, require every character to be a valid ASCII genotype (`0`/`1`/`2`/`9`),
   take the first line's length as `n_ind`, and require every subsequent line to
   match that width. A rectangular, all-ASCII-genotype file is EIGENSTRAT; its row
   count is `n_snp`.
3. Anything else — a ragged ASCII file, an interior blank line, a non-genotype
   character, or a packed magic we don't handle here (ANCESTRYMAP, PLINK) — comes
   back `Unknown`, and `write_merged_panel` throws rather than guess.

---

## 5. The TGENO branch (the primary AADR-1240K path)

TGENO is individual-major: each individual is its **own** contiguous record of
`packed_bytes(n_snp)` bytes. That layout makes this the cleanest possible append,
and it happens to be the real AADR 1240K `.PUB` panel's format:

- The 48-byte header is rebuilt with `n_ind` bumped to `n_ind + 1`.
- The **entire** `n_ind`-record data region is copied verbatim, in 4 MiB chunks,
  straight from source to output. This byte-for-byte copy is the structural proof
  that no existing individual is disturbed — nothing is decoded, re-packed, or
  even inspected.
- Nikki's single packed record is built fresh in a zeroed buffer (so a plain OR
  suffices) and appended at the end.

Because the panel is decoded correctly as TGENO and the append is native TGENO,
this path is **exact** — no lossy round-trip. It also sidesteps a ~7 GB offline
`convertf` TGENO→PA conversion the original spec assumed would be necessary; the
spec expected to "refuse TGENO," but the real panel is TGENO and steppe reads it
faithfully, so the native append is both correct and far cheaper. See
`docs/planning/nikki-stage3-4-spec.md`.

---

## 6. The GENO and EIGENSTRAT branches

### GENO / PACKEDANCESTRYMAP (SNP-major)

Here the layout is SNP-major: one record per SNP, packing all individuals' 2-bit
codes across a record. Nikki becomes the new **last** slot, index `n_ind`, in
every record. This is a *prefix-append*: for each SNP record, the writer copies
the existing "ancient prefix" bytes into a fresh output record and writes nikki's
code into the new trailing slot.

Two details matter:

- **The record stride can grow.** A record holds `packed_bytes(n_ind)` meaningful
  bytes. When adding the `(n_ind+1)`-th slot crosses a 4-slot byte boundary (i.e.
  `n_ind` was a multiple of 4), the packed record needs one more byte, so the
  output stride is `packed_bytes(n_ind + 1)`. The header stride respects the same
  48-byte floor the packed format uses.
- **Mask-clear before OR.** Nikki's slot may land in a byte that was previously
  *partial* — a last byte whose unused low bits could hold nonzero padding. So the
  writer explicitly masks that 2-bit slot to zero before OR-ing her code in, rather
  than trusting the padding to be clean. (This is a deliberate hardening fix; a
  naive OR could otherwise corrupt her call.)

### EIGENSTRAT (ASCII)

The simplest branch: each genotype line gets exactly one character appended —
`0`/`1`/`2` for a call, `9` for missing — and a newline. The reader tolerates
CRLF line endings (a trailing `\r` is stripped), treats a single blank line at
end-of-file as a benign terminator, and rejects an interior blank line or a
ragged (wrong-width) line as a malformed panel.

Of the three branches, only the TGENO-source path is provably exact; the GENO and
EIGENSTRAT appends are covered by the by-construction unit test.

---

## 7. Contracts, invariants, and edge cases

- **Panel alignment is mandatory.** `nikki_codes.size()` must equal the panel's
  `n_snp`. Every branch checks this and throws with a message naming both counts
  if they disagree — a misaligned code vector is a hard error, never a silent
  truncation or over-run.
- **Codes are clamped to valid range.** Any code strictly greater than
  `kMissingCode` (3) is treated as missing before packing, so a stray out-of-range
  byte degrades to "missing" rather than corrupting the bit-packing.
- **Everything is one more individual.** `n_ind_out == n_ind_src + 1` on every
  path; the header's `n_ind` is bumped by exactly one; the `.ind` gains exactly one
  row. There is no batch mode — this writer appends a single individual, by design.
- **Existing data is inviolate.** The `.snp` is a verbatim byte copy; existing
  individuals' genotypes are copied without re-encoding (verbatim in TGENO, prefix
  bytes preserved in GENO, whole lines preserved in EIGENSTRAT). Nothing pre-existing
  is recomputed.
- **Stale header hashes are kept, knowingly.** When the packed header is rebuilt,
  any trailing hash tokens (`ihash`/`shash`) are preserved verbatim. The `shash`
  over the unchanged `.snp` stays valid; the `ihash` is technically stale after
  adding an individual, but steppe's own reader (`parse_geno_header`) reads only
  magic + `n_ind` + `n_snp` and ignores both, so this is harmless. A rebuilt header
  that would exceed the 48-byte budget throws.
- **Failures are exceptions, and they're specific.** Every open, short read, write
  error, format-detection failure, and geometry mismatch surfaces as a
  `std::runtime_error` whose message names the file and the condition. The caller
  (`cmd_ingest`) wraps the call in a try/catch and maps any throw to an IO-error
  exit, so a merge failure is a clean, reported abort — never a partial success.

---

## 8. Where it's called

The one caller today is the `steppe ingest` command's Stage-3 merge
(`src/app/cmd_ingest.cpp`). After nikki's calls are built into a panel-aligned code
vector, it invokes `write_merged_panel(merge_into, emit_merged, codes, sample_id)`
and prints the returned `MergeCounts` as a one-line human summary: the detected
format, the output path, `n_snp`, the `n_ind` before→after, nikki's called/missing
split, and any duplicate-rsid hits from the code-building step. That summary line is
the whole point of `MergeCounts` — it's the operator-facing receipt that the merge
did exactly one individual's worth of work and nothing more.

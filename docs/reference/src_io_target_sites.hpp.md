# `target_sites.hpp` reference

## 1. Purpose

`src/io/target_sites.hpp` (with its implementation in `target_sites.cpp`) defines
`TargetSites` — the GRCh38 target-site table the native VCF reader genotypes a
sample against. Think of it as the panel of positions steppe cares about: for each
site it knows the rsID, the chromosome, the GRCh37 and GRCh38 coordinates, the two
alleles, and the pre-fetched GRCh38 reference base. The VCF reader walks a sample's
records and, for every record that lands on one of these positions, decides which
allele the sample carries there.

This file is the **boundary artifact between two stages** of VCF ingestion. Stage 1
is the orchestrated prep — the panel has already been rsID-joined, lifted from
GRCh37 to GRCh38, and had its GRCh38 REF base looked up. Stage 2 is the native
reader that consumes the result. So the things a `TargetSites` deliberately does
*not* do are as important as what it does: this code performs **no liftover, no
dbSNP-position cross-check, and no strict-1:1 de-dup**. Those steps stay upstream
(shared with the Stage-0 oracle) and are the native Stage-2 builder's job in
`target_build.cpp`. By the time a table reaches this file, palindromes are still in,
positions may already collide, and every value is taken as given.

It is pure host C++20, standard library only — an io-leaf with no GPU code, no CUDA
header, and no dependency on anything heavier than `<fstream>` and the STL
containers.

---

## 2. What a target site is (`TargetSite`)

Each row of the panel becomes one `TargetSite`:

| Field | Meaning |
|---|---|
| `rsid` | The dbSNP rsID label, carried through verbatim. |
| `chrom` | Chromosome as an integer (autosomes 1–22 in the supported v1 mode). |
| `pos37` | GRCh37 position. Informational only for the reader; defaults to `0` on the 6-column format. |
| `pos38` | GRCh38 position. **This is the coordinate the reader joins on.** |
| `a1`, `a2` | The two panel alleles, upper-cased on read (`'N'` if unset). |
| `ref38` | The GRCh38 REF base, upper-cased and soft-mask-normalized, or `'.'` when it was unavailable. |
| `palindrome` | Set when `{a1,a2}` is a strand-ambiguous pair (section 5). |

The `ref38` base is pre-fetched upstream precisely so the reader never has to open a
FASTA: when it needs to reconcile a sample's alleles against the reference strand,
the answer is already sitting on the site. A `'.'` there is the honest "we couldn't
get a reference base" marker, and the reader treats that site accordingly rather
than guessing.

---

## 3. The table and its per-chromosome index (`TargetSites`, `ChromIndex`)

`TargetSites` holds two things:

- `sites` — every `TargetSite` in **panel order**, exactly as read. This is the full
  record list, palindromes included.
- `by_chrom` — a `chrom -> ChromIndex` map used for the fast position join, built
  over the **non-palindromic sites only** (section 5).

A `ChromIndex` is the lookup structure the VCF reader leans on for one chromosome:

- `pos` — the site positions on that chromosome, **sorted ascending, with duplicates
  kept**.
- `slot` — a `pos38 -> index-into-pos` map, **last-wins** on duplicate positions.

The reader uses these two together. It runs `std::lower_bound` / `std::upper_bound`
over `pos` to find every target position covered by a VCF record's interval (an
inclusive-END coverage sweep), and it uses `slot` to turn a matched position back
into an index. The `has_chrom(c)` helper is the cheap "do we have any targets on
this chromosome at all" check the reader does before bothering with a record.

Keeping `pos` sorted with duplicates and giving `slot` last-wins semantics is not an
accident — it is the exact shape the upstream oracle produces (its `ppos` array and
`slot` map), so that interval-join coverage and slot lookup agree with the oracle
even at colliding lifted positions. See section 6.

---

## 4. Reading a table (`read_target_sites`)

`read_target_sites(path)` parses the table off disk into a `TargetSites`. The file
is whitespace- or tab-separated text with an optional header line, in one of two
column layouts:

```
rsID  chrom  pos37  pos38  A1  A2  ref38     (7-col, canonical)
rsID  chrom  pos38  A1  A2  ref38            (6-col, pos37 defaults to 0)
```

The parse is deliberately forgiving about surrounding noise and strict about the
fields it keeps:

- **Blank lines and `#` comment lines are skipped.** So is a header line whose first
  token is literally `rsID` or `rsid`, so a table can carry column names without
  tripping the parser.
- **A row with 7 or more fields** is read as the canonical layout (extra trailing
  fields beyond the 7th are ignored). **A row with exactly 6 fields** is read as the
  shorter layout, and `pos37` is left at `0`. Any other field count is a malformed
  record.
- **Alleles and the ref base are upper-cased** as they are read (via a local
  `toupper` helper), so downstream comparisons don't have to worry about case or
  soft-masking. An empty `ref38` token becomes `'.'`.
- **The palindrome flag is computed at read time** from `a1`/`a2` and stored on the
  site, so the drop policy is decided once.

After all rows are read it calls `build_chrom_index` (section 5) to populate
`by_chrom`, then returns the finished table.

---

## 5. Palindromes, and the single-homed drop policy

A **palindrome** here is a strand-ambiguous SNP: the allele pair is `{A,T}` or
`{C,G}` (in either order, case-insensitive). These are the sites where you can't
tell which strand a variant call is reported on from the alleles alone, so the
position join can't safely trust them.

`is_palindrome(a1, a2)` is the one function that decides this, and it is shared: both
`read_target_sites` (to set the per-site flag) and the Stage-2 native builder
(`target_build`) call it, so the definition of "palindrome" lives in exactly one
place and can't drift between the reader and the builder.

The policy is **keep the row, drop it from the join index**:

- Palindromic sites stay in `TargetSites::sites` — they are still part of the panel,
  the flag is just set on them.
- `build_chrom_index` walks `sites`, skips any site whose `palindrome` flag is set,
  and only pushes the non-palindromic ones into the per-chromosome `pos` array. So
  `by_chrom` — the thing the reader actually joins against — never contains a
  palindrome.

`build_chrom_index(ts)` is exported (not just an internal detail of the reader)
because it is the shared interval-join contract: both `read_target_sites` and the
native `build_target_sites` call it so a natively-built table and a
read-from-disk table index identically. It clears and overwrites `ts.by_chrom`, so
it is safe to call to rebuild the index after mutating `sites`. For each chromosome
it sorts `pos` ascending, reserves the `slot` map generously, then assigns
`slot[pos[i]] = i` in increasing `i` — which is what makes a duplicate position
resolve to its **highest** index (last-wins).

---

## 6. Why sorted-with-duplicates and last-wins (the oracle contract)

The precise shape of `ChromIndex` exists to match the upstream Stage-0 oracle
bit-for-bit, so that the native reader and the orchestrated oracle agree on which
sites a VCF record covers and which slot a hit maps to.

- **Duplicates are kept in `pos`** because liftover can map two different rsIDs to
  the same GRCh38 position. The interval-coverage sweep (`lower_bound`/`upper_bound`
  over `pos`) must see both, exactly as the oracle's `sorted(ppos)` does, or the two
  implementations would disagree on coverage at a collision.
- **`slot` is last-wins** because the oracle builds its position→slot map by
  enumerating the sorted positions, so a later (higher-index) duplicate overwrites an
  earlier one. Assigning `slot[pos[i]] = i` in ascending `i` reproduces that
  precisely.

This is the whole reason the index is a public, shared, carefully-specified
structure rather than an ad-hoc lookup: two independent code paths have to produce
the same answer on the same panel, including the awkward colliding-position cases.

---

## 7. Errors, contracts, and edge cases

- **Open failure throws.** If the file can't be opened, `read_target_sites` throws
  `std::runtime_error` naming the path. Callers get an exception, not an empty table.
- **Malformed rows throw.** A row that is neither 6-plus nor exactly-6 fields, or a
  row whose `chrom`/`pos` fields aren't numeric (a `std::stoi`/`std::stoll`
  `invalid_argument`), throws `std::runtime_error` with the offending line number.
  Parsing is fail-fast: one bad record aborts the read rather than being silently
  dropped.
- **An empty or all-comment file is not an error.** It simply yields a `TargetSites`
  with no sites and an empty `by_chrom`; `has_chrom` returns false for every
  chromosome and the reader finds no targets. That is a clean, documented no-op.
- **`ref38 == '.'`** is a first-class value, not a failure — it flows through to the
  reader, which handles a site with no reference base rather than dropping it here.
- **Only the first character of each allele/ref token is used.** A multi-character
  allele token is truncated to its first (upper-cased) character; the panel is
  expected to be biallelic SNPs.
- **`slot` and `pos` are always consistent after a build**, because both are produced
  by the same `build_chrom_index` pass over the same filtered site list. Callers that
  mutate `sites` directly must re-run `build_chrom_index` to keep the index in step —
  the reader does this for you.

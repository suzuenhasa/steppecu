# `target_build.cpp` reference

## 1. Purpose

`src/io/target_build.cpp` builds the GRCh38 **target-site table** — the
`TargetSites` the native VCF reader genotypes a sample against — directly in
memory, from three plain inputs:

1. the AADR EIGENSTRAT `.snp` panel (`id chrom genpos physpos A1 A2`, on GRCh37),
2. an orchestrated `rsID<TAB>pos38` lift map (the GRCh37→GRCh38 coordinate
   crossing, done outside steppe), and
3. a `FaidxReader` over the GRCh38 reference FASTA, which supplies the reference
   base at each lifted position.

This is the native, in-process replacement for the old orchestrated prep chain —
the Python `oracle.load_panel` + `lift_panel` + `emit_target_table.py` steps. The
one piece that stays orchestrated is the liftover itself (there is deliberately no
coordinate-chain engine here); everything else — the filtering, the strict 1:1
de-dup, the palindrome flagging, the reference-base fetch, and the per-chromosome
index — happens in this file.

It is a pure host, standard-library-only translation unit. No GPU, no CUDA, no
zlib. It reads text files and returns a struct.

The header (`target_build.hpp`) declares only the two entry points that this
`.cpp` implements — `build_target_sites` and `write_target_table` — so this file
is the authoritative place the build contract is written down. The shared pieces
it leans on (`is_palindrome`, `build_chrom_index`, the `TargetSite`/`TargetSites`
shapes) live in `target_sites.{hpp,cpp}` and are documented with that file.

---

## 2. The parity contract (why every branch is tagged)

The load-bearing rule of this file is **line-for-line agreement with the
orchestrated oracle**. The build reproduces the oracle's row-selection decisions
exactly, and every counter and every drop is annotated in the source with the
`oracle.py` line it mirrors (and, where a subtlety was found during review, a
`critic fix #N` note explaining the correction). The reason for that discipline is
the "gate-1" validation: the native table is diffed, row for row and count for
count, against the orchestrated `emit_target_table.py` output. If a single filter
decision drifts, that diff catches it.

So when you read a comment like "M1", "Fork #1", "H3", or "oracle:54" in this
file, it is naming the exact upstream decision this line reproduces. The counters
in `TargetBuildCounts` exist to make that agreement checkable — each one maps to a
named oracle counter (see section 6).

---

## 3. Two passes over the panel

`build_target_sites` reads the panel `.snp` in **two passes**, because a de-dup
decision needs the full multiplicity of every rsID before any row can be kept.

### Pass 1 — parse, filter, count

It streams the `.snp` line by line, splitting each on whitespace, and for every
row applies the filters in a fixed order:

1. **Short rows are skipped silently.** A row with fewer than 6 fields is not a
   panel record and is not counted at all — it never even reaches `panel_total`.
2. **Autosomes only.** The chromosome field must parse as an integer in `1..22`.
   Anything else — `X`, `Y`, `MT`, a non-numeric token — is dropped and does *not*
   count as autosomal. Every row that passes bumps `panel_total`, and every
   autosomal one bumps `panel_autosomal`.
3. **rsID join only (Fork #1).** The id must begin with the literal `rs`. A
   non-`rs` id on an autosome is counted in `panel_non_rsid` and dropped — the
   panel joins to the lift map by rsID, so a row with no rsID has nothing to join
   on.
4. **Per-rsID multiplicity.** Each surviving autosomal-rs row bumps
   `rs_count[rsid]`. This count is taken over the autosomal-rs rows *only* — a
   critical detail (critic fix #1): rows already dropped for chromosome or id do
   not contribute to the duplicate test.

A surviving row's physical position (`physpos`, field 4) must parse as an integer;
a non-numeric `physpos` is a hard error, not a silent skip, because it signals a
malformed panel rather than an expected filter case. The two alleles (fields 5 and
6) are upper-cased and reduced to their first character, and the row is flagged as
a palindrome via the shared `is_palindrome` test. Palindromic rows are **counted
now** (`panel_palindromic`), pre-dedup, because that is where the oracle counts
them.

The kept rows land in a `PanelRow` vector; the multiplicity map is finished before
pass 2 begins.

### Between the passes — the de-dup set

With every rsID's multiplicity known, the build collects the **distinct** rsIDs
whose count is greater than 1 into a set. `panel_dup_rsids` records the *size* of
that set — the number of distinct offending rsIDs, `len(dup)`, not the number of
rows they span (critic fix #3). This is the strict 1:1 de-dup policy (H3): if an
rsID appears more than once anywhere in the autosomal-rs panel, *every* occurrence
of it is dropped, not just the extras.

### Pass 2 — lift join, reference fetch, assemble

Now the lift map is read (section 4), and each `PanelRow` is either emitted or
dropped:

1. **Duplicate drop.** If the row's rsID is in the de-dup set, the row is dropped
   and `lift_dropped_dup` is bumped. This counter is per-*occurrence* (it counts
   rows, `n_dropdup`), which is why it can exceed `panel_dup_rsids`.
2. **No-lift drop.** If the rsID is absent from the lift map, there is no GRCh38
   position for it, so the row is dropped and `lift_no_lift` is bumped.
3. **Emit.** Otherwise the row is lifted: `lift_ok` is bumped, a `TargetSite` is
   assembled carrying both coordinates (`pos37` from the panel, `pos38` from the
   lift map), the two alleles, the palindrome flag, and — for *every* lifted site,
   palindrome or not (critic fix #2) — the GRCh38 reference base fetched natively
   from the FASTA at `(chrom, pos38)`.

`emitted` records the final site count, which always equals `lift_ok`.

Finally the shared `build_chrom_index` runs, producing the per-chromosome sorted
`pos38` index (over the non-palindromic sites only). This is the identical index
construction `read_target_sites` uses when it loads a table from disk, so a
`TargetSites` built natively and one parsed from a file are interchangeable
downstream.

---

## 4. The lift map, and the header it tolerates

`read_lift_map` parses the orchestrated `rsID<TAB>pos38` file into an
`rsID -> pos38` map. It is intentionally lenient about what it ignores:

- Empty lines and lines beginning with `#` are skipped.
- A line with fewer than 2 whitespace-separated fields is skipped.
- **A non-numeric position field is skipped rather than errored.** This is what
  quietly swallows a leading header row (critic fix #5): the map is commonly
  produced by `cut -f1,4 nikki_target_sites.tsv`, which carries the header line
  `rsID … pos38`, whose position column is the literal text `pos38` and therefore
  fails the integer parse and is skipped. No explicit "is this a header?" check is
  needed — a non-numeric position is never a real entry.

If two lines carry the same rsID, the last one wins (plain map assignment). The
only hard failure here is being unable to open the file at all, which throws.

The build takes the reference FASTA position from this map verbatim; it does no
coordinate arithmetic of its own. The lift is the orchestrated input, consumed as
data.

---

## 5. Strict integer parsing

`parse_ll` is the one number parser this file trusts for every field that must be
an integer — the chromosome, the panel `physpos`, and the lift position. It uses
`std::from_chars` and accepts a token **only if the entire token is consumed** and
there was no error. That strictness is the point: `12abc`, `1.5`, an empty string,
`X`, or an overflowing value all fail cleanly. A partial parse that stopped at the
first non-digit is treated as a failure, not a truncated success — so `chr12` does
not silently read as `12`, and the header token `pos38` does not read as `0`.

Where a failed parse is an *expected* filter case (a non-numeric chromosome, a
header position) it means "skip this row". Where it signals a malformed panel (a
non-numeric `physpos` on a row that already passed every filter) it throws.

---

## 6. The counters, and what they attest

`TargetBuildCounts` is reset to zero at the top of every build and mirrors the
oracle's `load_panel`/`lift_panel` counters exactly, so the gate-1 assertion can
compare them field for field:

| Counter | Counts | Oracle name |
|---|---|---|
| `panel_total` | Panel rows with ≥ 6 fields | `n_total` |
| `panel_autosomal` | Rows on chromosomes 1–22 | — |
| `panel_non_rsid` | Autosomal rows whose id isn't `rs…` | — |
| `panel_palindromic` | Autosomal-rs rows that are palindromic (**pre-dedup**) | — |
| `panel_dup_rsids` | **Distinct** rsIDs with multiplicity > 1 | `len(dup)` |
| `lift_dropped_dup` | **Rows** dropped for being a duplicate rsID | `n_dropdup` |
| `lift_ok` | Rows lifted to a `pos38` (== emitted) | — |
| `lift_no_lift` | Autosomal-rs, non-dup rows absent from the lift map | — |
| `emitted` | `TargetSites.sites.size()` (== `lift_ok`) | — |

Two of these are easy to misread, and the code notes both: `panel_dup_rsids`
counts *distinct rsIDs* (a set size) while `lift_dropped_dup` counts *rows*
(per-occurrence), so they are not the same number; and `panel_palindromic` is a
pre-dedup tally, so a palindromic site that is later dropped as a duplicate still
shows up in it.

---

## 7. Contracts and invariants

- **Autosomes-only is the only supported mode.** `TargetBuildOptions.autosomes_only`
  defaults to `true` and is the only value the build accepts; passing `false`
  throws immediately rather than half-implementing a wider chromosome set.
- **`emitted == lift_ok`** always: the emitted-site count and the lift-ok counter
  are two views of the same event.
- **Palindromes are kept, not dropped.** A strand-ambiguous site (A/T or C/G)
  stays in the table with its `palindrome` flag set. Only duplicate rsIDs and
  no-lift rows are dropped. The palindrome flag is what lets downstream consumers
  (and `build_chrom_index`) exclude them from the position index without losing
  them from the site list.
- **Every emitted site carries a reference base.** `ref38` is fetched for every
  lifted site, so no emitted `TargetSite` is left with a placeholder base — the
  fetch either succeeds or the whole build throws (section 8).
- **The output is panel-ordered.** `ts.sites` preserves the order the surviving
  rows appeared in the panel; the per-chrom index is a separate structure built
  over it.

---

## 8. Edge cases and failure modes

- **Un-openable inputs throw.** A missing or unreadable panel `.snp` or lift map
  raises a `std::runtime_error` naming the path. These are hard failures — the
  build cannot proceed without either file.
- **An out-of-range lift position throws.** The reference-base fetch goes through
  `FaidxReader::base_at`, which rejects a position outside `[1, contig-length]`
  (or an unknown contig) by throwing. So a lift map that points a site off the end
  of its GRCh38 chromosome surfaces as a hard build error, not a silently bad base
  — the malformed lift is caught at fetch time.
- **A non-numeric `physpos` throws** (section 5) — a malformed panel is not
  silently tolerated.
- **A header line in the lift map is absorbed silently** (section 4) — the common
  `cut`-produced header is a non-case, not an error.
- **Short and non-matching rows are dropped silently** — under-width panel rows,
  non-autosomal chromosomes, and non-rsID ids are all expected filter outcomes,
  counted where the oracle counts them and otherwise ignored.
- **An empty result is legal.** If nothing survives the filters and lift join, the
  build returns an empty (but well-formed) `TargetSites` with an empty index — it
  does not treat "no sites" as an error.

---

## 9. `write_target_table`

`write_target_table` dumps a `TargetSites` as the 7-column tab-separated table
`rsID chrom pos37 pos38 A1 A2 ref38`, with a header line, in `ts.sites` order.
This is the gate-1 artifact: the file that gets diffed against the orchestrated
`emit_target_table.py` output to prove the native build reproduces the pipeline it
replaces. It opens the output truncating, writes the header and one row per site,
and throws a `std::runtime_error` naming the path if the file can't be opened or if
any write fails — so a partial or failed dump is reported, never passed off as a
complete table.

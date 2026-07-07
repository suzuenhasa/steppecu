# `target_build.hpp` reference

## 1. Purpose

`src/io/target_build.hpp` declares the **Stage-2 native panel harmonizer**: the one
call, `build_target_sites`, that assembles a `TargetSites` table in memory from three
inputs — the AADR EIGENSTRAT `.snp` panel (GRCh37), an orchestrated
`rsID -> pos38` lift map, and a GRCh38 FASTA read through a `FaidxReader`.

The point of this file is to *replace* the old orchestrated prep — the
`oracle.load_panel` + `lift_panel` + `emit_target_table.py` chain — with native
C++. Everything the orchestrated scripts did to turn the raw panel into the target
table steppe genotypes against, this function now does in-process. The one thing it
does **not** take over is the GRCh37 -> GRCh38 liftover itself: that stays
orchestrated (plan Fork #1 — steppe carries no chain-file engine), and its output is
consumed here as a plain `rsID<TAB>pos38` text map. Everything downstream of the lift
is native.

The rule that governs every line of this file: **reproduce the oracle exactly.** The
counters, the filter order, the de-dup policy, the palindrome handling — all of it
mirrors specific lines of the orchestrated `oracle.py` so that the native table and
the old orchestrated table are the same table. That parity is checkable (section 8).

This is a pure host C++20 io-leaf. It touches no GPU, no zlib, no CUDA — just the
standard library plus its two sibling io headers (`faidx_reader.hpp`,
`target_sites.hpp`).

---

## 2. Where Stage 2 sits, and what it hands off

The `TargetSites` this function returns is the same artifact `read_target_sites`
parses from a 7-column table (see `target_sites.hpp`). The two are deliberately
interchangeable: `build_target_sites` is the *build-it-natively* path and
`read_target_sites` is the *load-a-precomputed-table* path, and both must produce a
byte-for-byte equivalent `TargetSites` so the native VCF reader downstream can't tell
which one built it.

That is why the last thing `build_target_sites` does is call the shared
`build_chrom_index(ts)` from `target_sites.hpp` — the exact same per-chromosome
sorted-`pos38` index construction the reader uses. The interval-join contract (a
sorted `pos` array with duplicates kept, plus a last-wins `(chrom, pos38) -> slot`
map built **only over non-palindromic sites**) is single-homed in that shared helper,
so the join coverage and slot lookup match at colliding lifted positions no matter
which path built the table.

---

## 3. The two-pass algorithm

`build_target_sites` walks the panel `.snp` twice (the implementation lives in
`target_build.cpp`, but the shape is worth stating here because it's the contract):

- **Pass 1 — parse, filter, and count multiplicity.** It reads every panel row,
  keeps the autosomal rsID rows, flags palindromes, and — crucially — counts how many
  times each rsID appears. It cannot decide which rows to drop for duplication until
  it has seen the whole file, so pass 1 only *collects* (surviving rows into a vector,
  per-rsID counts into a map). This is also where all the "input shape" counters are
  filled.

- **Between the passes** it condenses the count map into the strict-1:1 de-dup set:
  every rsID seen more than once. That set drives the drops in pass 2.

- **Pass 2 — lift-join, fetch `ref38`, assemble.** It walks the surviving rows in
  panel order, drops the ones whose rsID is a duplicate, drops the ones absent from
  the lift map, and for each survivor fetches the native GRCh38 reference base and
  appends a finished `TargetSite`.

Two passes rather than one because the de-dup decision is global (multiplicity over
the whole file) while the emit decision is per-row — you can't do the second until
the first is complete.

---

## 4. The filter and de-dup rules (oracle parity)

The row filter is a fixed cascade, applied in this order, matching the oracle:

1. **At least 6 fields.** A short row is skipped and is *not* counted at all — it
   never reaches `panel_total`. (The panel format is `id chrom genpos physpos A1 A2`.)
2. **Autosomes only (M1, Fork #1).** The chromosome field must parse as an integer in
   `1..22`. Anything non-numeric or out of range — `X`, `Y`, `MT`, a scaffold name —
   is dropped and does **not** count as autosomal. `TargetBuildOptions::autosomes_only`
   is the only supported v1 mode; passing `false` throws rather than silently doing
   something unvalidated.
3. **rsID join only (Fork #1).** The id must start with the literal `"rs"`. An
   autosomal row whose id isn't an rsID is counted (`panel_non_rsid`) and dropped —
   steppe joins to the lift map by rsID, so a non-rsID site has nothing to join to.

Only rows that clear all three enter the per-rsID count and the surviving-rows vector.

**Strict 1:1 de-dup (H3).** The multiplicity count is taken over the
autosomal-rs rows *only* — the same population the oracle counts over. Any rsID that
appears more than once in that population is added to the drop set, and in pass 2
**every** occurrence of a duplicated rsID is dropped (not "keep the first" — drop them
all). This is the strict 1:1 panel guarantee: a surviving rsID maps to exactly one
panel row.

**No-lift drop.** After de-dup, a row whose rsID is absent from the lift map is
dropped (`lift_no_lift`). Only rows with a real `pos38` survive to be emitted.

---

## 5. Palindromes are kept, not dropped

Strand-ambiguous SNPs — the A/T and C/G palindromes — are a common place for pipelines
to diverge, so state it plainly: **`build_target_sites` keeps palindromic sites in the
table.** They pass the filter, they get a real `pos38` from the lift, they get a native
`ref38` base fetched like any other site, and they land in `TargetSites::sites`. The
only thing that marks them is `TargetSite::palindrome = true`.

The palindrome test itself (`is_palindrome`, `{A,T}` or `{C,G}`, case-insensitive) is
shared with `read_target_sites` via `target_sites.hpp`, so the drop policy is
single-homed. Where the flag *does* matter is the chromosome index: `build_chrom_index`
builds its interval-join structures over the **non-palindromic** sites only, so a
palindrome is present in the table but excluded from the position lookup — exactly the
oracle's `ppos`/`slot` construction.

---

## 6. The counters (`TargetBuildCounts`)

`build_target_sites` fills a caller-supplied `TargetBuildCounts` out-parameter (it
zeroes it first, so the caller need not). These mirror the oracle's `load_panel` /
`lift_panel` counters exactly, and their whole reason to exist is the **gate-1 counts
assertion**: the native counters must equal the orchestrated ones, field for field.

| Counter | What it counts |
|---|---|
| `panel_total` | Panel rows with at least 6 fields (the oracle's `n_total`). |
| `panel_autosomal` | Rows on chromosome 1..22. |
| `panel_non_rsid` | Autosomal rows whose id is not an rsID. |
| `panel_palindromic` | Autosomal-rs rows that are palindromic, counted **pre-dedup**. |
| `panel_dup_rsids` | **Distinct** rsIDs with multiplicity > 1 (the oracle's `len(dup)`). |
| `lift_dropped_dup` | **Rows** dropped because their rsID is a dup (`n_dropdup`) — per-occurrence, so it can exceed `panel_dup_rsids`. |
| `lift_ok` | Rows successfully lifted to a `pos38` — equals the number emitted. |
| `lift_no_lift` | Autosomal, non-dup rows absent from the lift map. |
| `emitted` | `TargetSites::sites.size()` — equal to `lift_ok`. |

Two subtleties worth internalizing, because they're the ones that trip a naive
reimplementation:

- **`panel_dup_rsids` counts distinct rsIDs; `lift_dropped_dup` counts rows.** An
  rsID that appears three times contributes `1` to the former and `3` to the latter.
- **`panel_palindromic` is a pre-dedup tally.** It counts palindromic autosomal-rs
  rows as they're seen in pass 1, before any dup or no-lift drop — so it does not
  equal the number of palindromes that survive into the emitted table.

---

## 7. Native `ref38` fetch

For **every** lifted site — palindrome or not — the builder fetches the GRCh38
reference base at `(chrom, pos38)` through `FaidxReader::base_at`, and stores it in
`TargetSite::ref38`. This is the native replacement for what `emit_target_table.py`
did by reading the FASTA out-of-process.

The base is stored as `FaidxReader` returns it. That reader upper-cases and
soft-mask-normalizes, so `ref38` is an upper-case A/C/G/T (or `'.'` / `'N'` where the
FASTA has no real base). Note `FaidxReader` is **not** thread-safe, which is one reason
`build_target_sites` is a straight-line single-threaded call rather than a parallel
fetch.

---

## 8. Contracts, invariants, and edge cases

**What it throws.** `build_target_sites` throws `std::runtime_error` on:

- an unopenable panel `.snp` or lift map;
- `autosomes_only == false` (the only unsupported-mode guard);
- a non-numeric `physpos` on a row that already passed the autosome + rsID filter
  (a malformed panel is a hard error, not a silent skip);
- an out-of-range lift `pos38` surfaced by `FaidxReader::base_at` when it fetches
  past the end of a contig.

Note the asymmetry: a *bad chromosome* is a quiet drop (it's just a non-autosome), but
a *bad physpos on an otherwise-valid row* is fatal — because the former is expected
input variety and the latter means the panel is corrupt.

**Lift-map tolerance.** `read_lift_map` (the internal helper) is forgiving on the way
in: it skips blank lines and `#` comments, skips any row with fewer than two
whitespace-separated fields, and skips a row whose second field doesn't parse as an
integer. That last rule is what silently swallows a leading header row — a
`cut -f1,4 nikki_target_sites.tsv` carries its header line, and a `pos` field reading
`"pos38"` simply fails the integer parse and is dropped. The map is last-wins on
duplicate rsID keys.

**Invariants that hold on return.**

- `counts.emitted == counts.lift_ok == ts.sites.size()`.
- `ts.sites` is in panel read order.
- Every emitted site has a real `pos38` from the lift map and a fetched `ref38`.
- No emitted rsID is a duplicate — the surviving rsIDs are strict-1:1 to panel rows.
- `ts.by_chrom` is freshly built (via `build_chrom_index`) over the non-palindromic
  emitted sites only.

---

## 9. `write_target_table` — the gate-1 diff dump

`write_target_table(path, ts)` writes the `TargetSites` back out as the canonical
7-column table:

```
rsID  chrom  pos37  pos38  A1  A2  ref38
```

with a header line, tab-separated, in `ts.sites` order. It exists for one job: the
**gate-1 diff**. Dump the native table with this, dump the orchestrated table from
`emit_target_table.py`, and diff them — if the native harmonizer reproduces the
oracle, the two files are identical. It throws `std::runtime_error` if the output can't
be opened or if the write fails partway (the trailing stream check catches a disk that
filled mid-write).

Because the columns it writes are exactly the 7-column form `read_target_sites` parses,
a table written by `write_target_table` round-trips cleanly back through
`read_target_sites` — closing the loop between the build path and the load path of
section 2.

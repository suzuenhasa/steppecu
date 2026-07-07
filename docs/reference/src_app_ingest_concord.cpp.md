# `ingest_concord.cpp` reference

## 1. Purpose

`src/app/ingest_concord.cpp` is the arithmetic behind `steppe ingest-concord` — the
correctness gate for steppe's VCF/gVCF ingestion. It compares two per-site report
tables and decides whether steppe's ingestion agrees, site for site, with a trusted
oracle. It is the ingest-side twin of the READv2 concordance validator
(`readv2_concord.cpp`) and shares its shape almost exactly.

The two tables are:

- **A — the TEST table.** steppe's own per-site ingestion report.
- **B — the RULER table.** The Stage-0 oracle's dosage report, the thing steppe is
  being checked against.

The job is a join over rsID and an **exact** four-field agreement test on every site
the two tables share. The output is a single PASS/FAIL verdict plus the counts that
justify it.

This is a pure host, CUDA-free translation unit. There is no GPU here, no
`RunConfig`, no device — just two text files, a `std::map` join, and some counting.
The thin CLI shim that parses flags and turns the verdict into a process exit code
lives next door in `cmd_ingest_concord.cpp`; this file is the parse-agnostic
arithmetic it calls.

---

## 2. The oracle table schema

Both tables are tab-separated and **header-driven by column name**, not by position.
The reader finds the columns it needs by matching the header row, so extra columns,
or a different column order, are fine. The six columns it requires are:

| Column | What it means |
|---|---|
| `rsID` | The join key — the variant's rsID. |
| `call` | The genotype call at the site. |
| `dosage` | The alt-allele dosage. |
| `source` | Where the call came from: `variant`, `refblock`, … |
| `drop_reason` | Why a site was dropped, or empty for a kept site. |
| `pos38` | The GRCh38 position — used to filter the oracle (section 4). |

The full oracle schema (Stage-0 spec §9) carries more columns than these —
`chrom`, `pos37`, `A1`, `A2`, `flip` — but the concordance only joins and compares
the six above. Any required column missing from a table's header is a hard
bad-input error naming the column and the file.

---

## 3. The four-field exact-match tuple

The concordance does not score fields independently or fuzzily. For each rsID present
in both tables it forms the tuple `{call, dosage, source, drop_reason}` from each side
and requires **all four to be string-equal** for the site to count as a match:

```
match  ==  (a.call == b.call) && (a.dosage == b.dosage) &&
           (a.source == b.source) && (a.drop_reason == b.drop_reason)
```

Everything is compared as trimmed strings — there is no numeric parsing of `dosage`,
no canonicalization of `call`. That is deliberate: the oracle and steppe are meant to
emit byte-identical field text, so a string compare is the strictest possible check
and the one least likely to paper over a real disagreement. A trailing-space or
formatting drift is a mismatch, on purpose.

An empty `drop_reason` is a meaningful value, not a missing one. The tab splitter
keeps empty cells (section 7), so a kept site with `drop_reason == ""` on both sides
compares equal, exactly as it should.

---

## 4. The valid-pos38 asymmetry (why B is filtered and A is not)

The two tables are read with different rules, and this is the single most important
design decision in the file:

- **A (steppe) is read whole.** steppe's report contains no null-pos38 rows, so every
  row is kept.
- **B (oracle) is filtered to valid pos38 only.** The oracle carries extra rows that
  steppe structurally cannot reproduce: sites the orchestrated lift stage dropped
  (`no_lift`, `dup_rsid`), which the oracle records with `pos38` equal to `"None"`,
  `"NA"`, or empty. Those rows are dropped from the join.

The reason is that steppe's reader never runs the orchestrated lift stage, so it
*can't* emit those drop rows — holding steppe to them would be an unfair, guaranteed
miss. The join is therefore defined over the oracle rows steppe had a fair chance to
produce: the valid-pos38 rows.

The report keeps both counts so nothing is hidden: `b_rows` is the valid-pos38 row
count that the join actually uses, and `b_rows_raw` is the full oracle row count
including the null-pos38 drops. The gap between them is exactly the lift-stage rows
that were excluded.

---

## 5. The two named subsets (and why they get their own lines)

Beyond the overall match rate, the report breaks out two subsets of the oracle rows,
each scored separately:

- **The ref-block hom-ref subset** — oracle rows where `source == "refblock"` and
  `call == "homref"`. This is the load-bearing one. These are sites called
  homozygous-reference from a gVCF reference block, and they have **no external
  decoder** — steppe's reader is itself the only independent check on them. That is
  why the subset must be surfaced on its own line rather than folded into the overall
  number: it's the carry-forward / novel-surface requirement, and burying it in an
  aggregate would hide the one place there's no second opinion. The text report even
  annotates the line with `[*] novel surface: NO external decoder`.
- **The explicit-variant subset** — oracle rows where `source == "variant"`. Sites
  that came from an explicit VCF variant record.

Each subset gets a total, a match count, and a match fraction. A site can belong to a
subset and still be a mismatch; the subset counters use the same four-field `match`
result computed for the overall tally, so the subset numbers are always consistent
with the overall one.

---

## 6. Reading a table (`read_table`)

`read_table` turns one file into a `std::map<rsID, Row>`. The steps:

1. **Find the header.** It skips leading blank lines and takes the first non-blank
   line as the header. An empty file (no header at all) is an I/O-style error.
2. **Locate columns by name.** It builds a name→index map from the header and looks
   up each of the six required columns. A missing one is a bad-input error. It also
   remembers the largest required column index (`need`) so it can reject short rows
   cheaply.
3. **Read the rows.** Blank lines are skipped. A row with fewer columns than the
   header requires (`c.size() <= need`) is a bad-input "short row" error rather than
   a silent out-of-bounds read.
4. **Apply the pos38 filter** when asked (`drop_null_pos38`, true only for B): a row
   whose `pos38` is empty, `"None"`, or `"NA"` is skipped and not counted as kept.
5. **Insert under the rsID key.** A duplicate rsID within one table is a hard
   bad-input error — the join assumes each key appears at most once per table, so a
   duplicate would make the result ambiguous and is refused outright rather than
   silently overwriting or double-counting.

It returns two counts by reference: `rows_raw` (rows seen before the pos38 filter)
and `rows_kept` (rows actually inserted). For A these are equal; for B `rows_raw` is
the oracle's full count and `rows_kept` is the valid-pos38 subset.

---

## 7. The parsing helpers (`trim`, `split_tab`)

Two small helpers do all the text handling:

- **`trim`** strips leading and trailing spaces, tabs, and carriage returns. It is
  applied to every field and every header cell, so trailing `\r` from a
  CRLF-terminated file, or incidental whitespace, never causes a spurious mismatch.
- **`split_tab`** splits a line on tab only and **keeps empty cells**. This is
  deliberate: an empty `drop_reason` is a real, comparable value (section 3), so a
  splitter that collapsed empty fields would silently corrupt the compare. It also
  drops stray `\r` characters inline so a CRLF line doesn't leave a carriage return
  glued to the last cell.

There is no CSV/quoting logic — the oracle and report schema is plain
tab-separated, so a hand-rolled tab split is all that's needed and there is no JSON
or CSV library linked here.

---

## 8. The join and the derived metrics (`run_ingest_concordance`)

The entry point reads A whole and B filtered (section 4), then walks **B's** map and
looks each key up in A:

- A B key **not** in A is a `b_only` site — an oracle site steppe missed. It is
  skipped in the loop and recovered arithmetically afterward.
- A B key **in** A is a `common` site: it runs the four-field match, updates
  `match_all` / `mismatch`, and updates whichever subset(s) the oracle row belongs
  to.
- The **first** mismatch is captured with a human-readable detail string showing both
  sides' four fields side by side, purely for eyeballing — it is not gate-load-bearing.

After the loop the derived counts fall out arithmetically:

```
a_only   = a_rows - common
b_only   = b_rows - common
coverage = common / b_rows                 (0 when b_rows == 0)
overall_match       = match_all / common   (0 when common == 0)
refblock_match_frac = refblock_match / refblock_total   (0 when total == 0)
variant_match_frac  = variant_match  / variant_total    (0 when total == 0)
```

Every fraction guards its denominator, so an empty table or an empty subset yields a
clean `0.0` rather than a division by zero.

---

## 9. The PASS gate

`pass` is the conjunction of four floors, every one of which must be met:

```
pass = (coverage       >= coverage_min)       &&
       (common > 0)                            &&
       (overall_match  >= overall_match_min)   &&
       (refblock_match_frac >= refblock_match_min) &&
       (variant_match_frac  >= variant_match_min)
```

The floors default to `1.0` across the board (`IngestThresholds`) — the default
posture is that ingestion must agree with the oracle **exactly**, on every shared
site and in both subsets. The thresholds are knobs the CLI can loosen, but out of the
box this is an all-or-nothing gate.

The `common > 0` term is a guard against a vacuous pass: if the two tables share no
sites at all, `coverage` and the match fractions would each be their guarded `0.0`,
but requiring at least one common site means an empty join can never masquerade as a
clean PASS.

---

## 10. Rendering the report (`write_ingest_report`)

The same numbers render two ways, chosen by a `format` string:

- **`"json"`** — a flat object with every count and fraction and a `"result"` of
  `"PASS"` or `"FAIL"`. Fractions go through `g6` (a `%.6g` formatter) so the JSON
  stays compact and stable.
- **`"text"`** (the default) — a human-readable block first: a one-line row summary
  (A / B-valid / B-raw / common / a_only / b_only), then one aligned line per gated
  metric showing the value, its floor, and a per-line PASS/FAIL. The ref-block line
  carries its `novel surface: NO external decoder` annotation. If there were any
  mismatches, the first one is printed in full.

The text form then emits a **stable machine trailer**: a block of `iconcord_<name>:
<value>` lines, one per field, followed by a final `RESULT: PASS|FAIL`. This grep-able
trailer is what a harness or CI script keys on — the pretty block above it is for a
human, the `iconcord_*` lines are the contract. Both the `iconcord_*` keys and the
final `RESULT:` line are meant to stay stable so downstream tooling can rely on them.

---

## 11. Status codes and how they map to exit codes

`run_ingest_concordance` reports one of three statuses:

| Status | Meaning |
|---|---|
| `kOk` | Computed cleanly. `report.pass` carries the PASS-vs-FAIL verdict. |
| `kBadInput` | A required column was missing, a row was short, or a duplicate rsID appeared. |
| `kIoError` | A table file could not be opened, or was empty (no header). |

Note the separation of concerns: **this file never talks to a process exit code.** A
clean run that FAILs the concordance is still `kOk` — the FAIL lives in
`report.pass`, not in the status. It is the CLI shim (`cmd_ingest_concord.cpp`) that
maps these to the process's exit code: `0` for PASS, a distinct literal `1` for a
clean-but-FAILing concordance, `2` (invalid-config) for bad input, and `4` for an
I/O error. Keeping the arithmetic status and the process exit code separate is what
lets the same function back both the CLI and any in-process caller (a test) without
either one inheriting the other's notion of "failure".

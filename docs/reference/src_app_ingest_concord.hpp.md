# `ingest_concord.hpp` reference

## 1. Purpose

`src/app/ingest_concord.hpp` declares the arithmetic behind the VCF-ingest
concordance validator — the **Stage-1 gVCF block-correctness gate**. It is the
layer one step below the `steppe ingest-concord` CLI shim
(`cmd_ingest_concord.{hpp,cpp}`): the shim parses flags and maps an outcome to an
exit code, and this file does the real work of reading two report tables, joining
them, and deciding PASS or FAIL.

It is pure host code. There is no CUDA header here, no `RunConfig`, no device, no
precision knob — it reads two text files and compares numbers, so it runs anywhere,
card or no card. That host-only isolation is the same rule the rest of the app
target lives under.

The header declares four small value types (`IngestReport`, `IngestThresholds`,
`IngestConcordStatus`, `IngestConcordResult`) and two functions
(`run_ingest_concordance` computes, `write_ingest_report` renders). The
implementation lives in `ingest_concord.cpp`. It mirrors its sibling
`readv2_concord.{hpp,cpp}` — same shape, different subject.

---

## 2. The two tables it compares (and which is which)

The validator diffs **two per-site report TSVs**, both in the Stage-0 oracle
schema:

- **A — the TEST table** (`a_path`): steppe's own ingest report, the thing under
  scrutiny.
- **B — the RULER table** (`b_path`): the Stage-0 oracle dosage TSV, the trusted
  reference steppe is validated against.

The A-vs-B direction is load-bearing and must not be flipped: A is what's being
checked, B is what it's checked against. Getting them backwards silently inverts
the meaning of every count in the report.

Both tables share one column schema (Stage-0 spec §9), and the reader is
**header-driven by column name**, not by position:

```
rsID  chrom  pos37  pos38  A1  A2  call  dosage  source  flip  drop_reason
```

Only six of those columns are actually required — `rsID`, `call`, `dosage`,
`source`, `drop_reason`, `pos38` — and they may appear in any order; the reader
looks each one up by name. Any of the other columns can be present, absent, or
reordered without affecting the result. A file missing one of the six required
columns is rejected as bad input (section 6).

---

## 3. The rsID join over valid-`pos38` oracle rows

The two tables are joined on **rsID** — each `rsID` is the key, and the row it maps
to is compared against the same key in the other table. Within a single table an
rsID must be unique; a duplicate key is bad input, not a silently-kept last-writer.

The subtlety is on the B (oracle) side. The Stage-0 orchestration runs a lift stage
that drops some sites — `no_lift` and `dup_rsid` cases — and those dropped rows are
still emitted into the oracle table but carry a null `pos38` (the literal `"None"`,
`"NA"`, or an empty cell). steppe's reader **cannot reproduce** those orchestrated
lift-stage drops, because it has no external decoder for them, so they must not
count against it. The join therefore keeps:

- **A (steppe):** every row. steppe's own report has no null-`pos38` rows to begin
  with, so nothing is dropped.
- **B (oracle):** only rows with a valid `pos38`. The null-`pos38` lift-stage drops
  are filtered out before the join.

That asymmetry is why the report tracks both `b_rows` (valid-`pos38` only, the join
denominator) and `b_rows_raw` (every oracle row including the nulls) — the raw count
is kept for visibility, but coverage and the match rates are measured against the
valid-`pos38` subset alone.

---

## 4. The exact per-site match

For every rsID present in **both** tables (the `common` set), the gate asserts
**exact string agreement** of a four-field tuple:

```
{ call, dosage, source, drop_reason }
```

All four must match, byte for byte, for the site to count as a match. The
comparison is literal string equality — no numeric tolerance, no canonicalization —
which is exactly right for a correctness gate: the strings the reader emits either
are what the oracle emits or they aren't. Fields are trimmed of surrounding
whitespace before comparison, and an empty cell is meaningful (an empty
`drop_reason` compares equal to `""`, not to a missing value).

The first disagreement is captured in `first_mismatch_key` and
`first_mismatch_detail` — the rsID and a side-by-side dump of both tuples — purely
for eyeballing. Those two fields are *not* gate-load-bearing; the pass/fail decision
rides on the counts, and the first-mismatch strings are a debugging convenience so a
human can see *what* diverged without diffing the whole table.

---

## 5. The `IngestReport` — every count the gate produces

`IngestReport` is the filled-in result: raw counts, derived fractions, and the final
`pass` verdict. The fields, grouped by what they measure:

### Row bookkeeping

| Field | Meaning |
|---|---|
| `a_rows` | steppe (A) rows kept. |
| `b_rows` | oracle (B) rows kept — valid-`pos38` only, the join denominator. |
| `b_rows_raw` | oracle rows including the null-`pos38` lift-stage drops. |
| `common` | rsIDs present in both A and B (over valid-`pos38` B). |
| `a_only` | `a_rows − common` — steppe rows the oracle didn't have. |
| `b_only` | `b_rows − common` — valid-`pos38` oracle rows steppe missed. |
| `coverage` | `common / b_rows` — how much of the oracle steppe covered. |

### The overall match

| Field | Meaning |
|---|---|
| `match_all` | common sites where all four fields agreed. |
| `mismatch` | common sites where at least one field disagreed. |
| `overall_match` | `match_all / common` — the headline exact-match fraction. |

### The two oracle-defined subsets

Two subsets of the common sites are counted and reported **on their own lines** —
this is the load-bearing design decision of the whole validator, not a cosmetic
breakdown:

| Field | Subset | Definition (oracle-side) |
|---|---|---|
| `refblock_total` / `refblock_match` / `refblock_match_frac` | ref-block hom-ref | `source == "refblock"` **and** `call == "homref"` |
| `variant_total` / `variant_match` / `variant_match_frac` | explicit variant | `source == "variant"` |

Both subsets are defined by the **oracle's** classification of the site, not
steppe's — the ruler decides which bucket a site belongs to, and steppe is scored
within that bucket.

The ref-block hom-ref subset is the reason the whole tool exists as a separate gate.
Those are the sites the gVCF represents as a reference block (hom-ref stretches with
no explicit variant record) — and they have **no external decoder**. For an explicit
variant, another tool can independently decode the genotype and check steppe against
it; for a ref-block hom-ref site, steppe's reader *is* the only check there is. So
that subset is the novel surface, and its match rate has to be surfaced on its own
line — folding it into the overall would let a strong explicit-variant score paper
over a weakness in the one place there's no independent oracle. This is the
carry-forward / verify-LOW-#2 requirement made concrete.

---

## 6. The PASS gate and `IngestThresholds`

`IngestThresholds` carries four PASS floors, each defaulting to **1.0**:

| Field | Gates |
|---|---|
| `overall_match_min` | the overall exact-match fraction. |
| `refblock_match_min` | the ref-block hom-ref subset. |
| `variant_match_min` | the explicit-variant subset. |
| `coverage_min` | the oracle-row coverage. |

The strict `1.0` default is the point: an un-flagged run demands **perfect**
agreement on all four axes — nothing short of a full match passes. Loosening a floor
is opt-in, per flag, never implicit.

`report.pass` is set true only when **all** of these hold:

- `coverage >= coverage_min`, and
- `common > 0` (an empty join can never pass — there was nothing to check), and
- `overall_match >= overall_match_min`, and
- `refblock_match_frac >= refblock_match_min`, and
- `variant_match_frac >= variant_match_min`.

The `common > 0` guard is what stops a degenerate "we compared nothing, so nothing
disagreed" run from reporting a vacuous PASS. Any fraction whose denominator is zero
(an absent subset, an empty join) is reported as `0.0` rather than dividing by zero;
whether that zero passes then depends on its floor.

---

## 7. The three-state status

`run_ingest_concordance` returns an `IngestConcordResult`: a status, an error string,
and the filled `IngestReport`. The status separates *how the run ended* from *what
the gate decided*:

| `IngestConcordStatus` | Meaning |
|---|---|
| `kOk` | Computed cleanly. The PASS/FAIL verdict lives in `report.pass` — a clean FAIL is still `kOk`. |
| `kBadInput` | A required column was missing, a row was shorter than the header requires, or an rsID was duplicated within a table. |
| `kIoError` | A table file couldn't be opened or read (including an empty file with no header). |

The important distinction is that a concordance **FAIL is not an error** — it's a
clean `kOk` result whose `report.pass` is false. The tool ran perfectly and simply
found a disagreement. Only a malformed table (`kBadInput`) or an unreadable file
(`kIoError`) is an actual failure to run. The CLI shim maps these three states, plus
the PASS/FAIL split inside `kOk`, onto four distinct process exit codes (see
`cmd_ingest_concord.hpp.md`).

---

## 8. The two entry points

### `run_ingest_concordance(a_path, b_path, thr)` — compute

Marked `[[nodiscard]]`. Reads both tables, applies the valid-`pos38` filter to B,
joins on rsID, counts the overall and both subset matches, derives every fraction,
and sets `report.pass` against the thresholds. Any read/parse problem short-circuits
to a `kBadInput` / `kIoError` result before the gate runs.

### `write_ingest_report(os, rep, thr, format)` — render

Renders a computed report to a stream in one of two formats:

- **`"text"`** — a human-readable block (the row tallies, then one aligned
  `coverage` / `overall exact match` / `refblock hom-ref [*]` / `explicit-variant`
  line each with its own PASS/FAIL against the floor, then the first mismatch if
  any), followed by a **stable machine trailer** of `iconcord_*` key/value lines,
  and a final `RESULT: PASS|FAIL` line. The `[*]` note on the ref-block line spells
  out "novel surface: NO external decoder" — the reason that subset stands alone
  (section 5). The `iconcord_*` trailer is the contract a CI harness greps: it is
  stable, one metric per line, and always present in text mode.
- **`"json"`** — the same numbers as a single JSON object, with a `"result"` field
  of `"PASS"` or `"FAIL"`.

Floating-point values in both formats are formatted with `%.6g`, so a rate reads as
a compact `1` / `0.998` rather than a long decimal tail.

---

## 9. Contracts and invariants

- **Directional inputs.** A is TEST, B is RULER. Swapping them inverts the report's
  meaning; nothing in the code detects or corrects a flip.
- **Header-driven, position-free.** Columns are found by name, so the six required
  columns may appear in any order and extra columns are ignored. A missing required
  column is `kBadInput`.
- **rsID is a unique key per table.** A duplicate rsID within either table is
  `kBadInput`, not a silent overwrite.
- **The asymmetric filter is fixed.** Every A row is kept; only valid-`pos38` B rows
  join. `b_rows` (the join denominator) and `b_rows_raw` are both reported so the
  drop is visible.
- **Exact string match.** The `{call, dosage, source, drop_reason}` tuple is
  compared as trimmed strings with no tolerance; an empty cell is a real value.
- **The subsets are oracle-defined and reported separately.** Ref-block hom-ref and
  explicit-variant membership come from B's `source`/`call`, and each gets its own
  floor and its own reported line — the ref-block subset is never folded into the
  overall.
- **PASS needs a non-empty join.** `common > 0` is part of the gate, so a run that
  compared nothing cannot report PASS.
- **Thresholds pass through untouched.** The gate enforces exactly the four floors
  it's handed; the strict `1.0` defaults mean an un-flagged run demands perfection.
- **Host-only, forever.** No CUDA, no `RunConfig`, no device — this translation unit
  stays GPU-free.

---

## 10. Edge cases

- **A subset the oracle never produced.** If B has no `refblock`/`homref` sites (or
  no `variant` sites), that subset's total is `0` and its fraction is reported as
  `0.0` rather than dividing by zero. Whether that `0.0` passes then rides on its
  floor — with the strict `1.0` default, a `0.0` fraction fails, but that only
  matters when the subset is genuinely expected.
- **Empty join.** If no rsID is shared, `common == 0`; every match fraction is
  `0.0`, and the `common > 0` guard forces FAIL regardless of the floors. The tool
  refuses to call "nothing compared" a pass.
- **Null-`pos38` oracle rows.** The `no_lift` / `dup_rsid` lift-stage drops (with
  `pos38` of `"None"`, `"NA"`, or empty) are filtered out of B before the join, so
  they never count against steppe — steppe has no way to reproduce them. They still
  show up in `b_rows_raw`.
- **A clean FAIL.** Both tables read fine, the join succeeds, and a floor is missed.
  The status is still `kOk`; the report renders in full so a human sees exactly what
  diverged (including the first-mismatch dump), and `report.pass` is simply false.
- **Empty file / missing header.** A file that opens but has no non-blank header line
  is `kIoError`, treated the same as a file that wouldn't open at all — either way
  there was no table to read.
- **Short row.** A data row with fewer columns than the header's highest required
  index is `kBadInput` — a schema problem, distinct from an I/O problem, so a caller
  can tell a malformed table from an unreadable one.

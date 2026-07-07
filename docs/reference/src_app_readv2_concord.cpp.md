# `readv2_concord.cpp` reference

## 1. Purpose

`src/app/readv2_concord.cpp` is the arithmetic core of the READv2 concordance
validator — the Phase-0 "ruler" that answers one question: **do steppe's READv2
relatedness results agree with the reference tool's?** It reads two READv2 output
tables, matches them up by sample pair, and reports how well they line up on two
axes — the relatedness *degree* each pair was assigned, and the numeric *P0_norm*
value behind that call.

It is host-only and CUDA-free. There is no device, no `RunConfig`, no GPU header —
just file reading and floating-point comparison. That isolation is deliberate: the
validator is a plain measuring stick, not a compute step, and this translation unit
is directly unit-testable without ever touching a GPU.

This file does the parsing and the math. The thin command-line shim that turns
`argv` into a call lives next door in `cmd_readv2_concord.cpp`; the frozen schema
and the public structs are declared in the paired header `readv2_concord.hpp`.
Crucially, this validator **never classifies** anything. It does not decide whether
a pair is "first-degree" — it only diffs the `degree` strings and `P0_norm` numbers
the two tables already carry.

---

## 2. The two tables, and what "A" and "B" mean

The validator takes two READv2 tables:

- **A** — the *test* table, the one steppe produced.
- **B** — the *oracle* table, the reference tool's ground truth.

That orientation is load-bearing and shows up everywhere downstream. In the degree
confusion matrix, **B (oracle) is the rows and A (steppe) is the columns**, so the
matrix reads "when the truth said X, what did steppe say?". The relative-tolerance
comparison also anchors on B: the P0_norm relative deviation is measured against
`|b_p0|`, because the oracle value is the reference the test is being held to.

Both tables share one frozen schema — one row per unordered sample pair, with these
columns, matched by **name** in the header (not by position):

```
sampleA  sampleB  n_windows  n_overlap_sites  P0_mean  P0_norm  degree  z
```

Only four of those columns are actually read: `sampleA`, `sampleB`, `degree`, and
`P0_norm`. The rest are tolerated and ignored — a table may carry them or not, in
any column order, and the validator still works because it looks each field up by
its header name.

---

## 3. The frozen degree enum

Relatedness degree is one of exactly four lowercase tokens, in a fixed index order:

| Index | Token |
|---|---|
| 0 | `identical` |
| 1 | `first` |
| 2 | `second` |
| 3 | `unrelated` |

That order is frozen — it's the row/column order of the confusion matrix and the
order everything is printed in. `degree_index()` maps a raw token to its index, or
`-1` if the token is outside the enum, and `degree_tokens()` hands the fixed array
back out for rendering.

A degree token outside this set is **not** tolerated — it's a bad-input error that
stops the run (section 6). The enum is small and closed on purpose: an unexpected
token means the table isn't the schema the validator was promised, and guessing
would be worse than failing.

---

## 4. Reading a table (the sniff-parser)

`read_table` slurps one READv2 file into a `key -> Readv2Row` map. It is a small,
dependency-free CSV/TSV reader — the app links no CSV library — built to be tolerant
of the shapes a real READv2 dump takes:

- **Blank leading lines are skipped** to find the header. A file that is empty or
  all-blank (no header at all) is an IO error.
- **The delimiter is sniffed from the header line.** If the header contains a TAB
  it's treated as TSV; otherwise CSV. There is no mode flag — the file tells the
  reader what it is.
- **`split_line` handles CSV double-quotes** (including `""` as an escaped quote
  inside a quoted field) for the comma case. TSV never quotes, so on the tab path
  the quote handling simply never triggers.
- **Every field is trimmed** of leading/trailing spaces, tabs, and a trailing `\r`
  — so a CRLF file, or a table with padded columns, parses cleanly.
- **Required columns are checked by name.** `sampleA`, `sampleB`, `degree`, and
  `P0_norm` must all be present in the header; a missing one is a bad-input error
  naming the column and the file. Their column indices are looked up once, and the
  reader remembers the largest index it needs so it can reject short rows.
- **Blank data lines are skipped**; a non-blank row with fewer columns than the
  header requires is a bad-input "short row" error.

Each surviving row becomes a `Readv2Row` (trimmed sampleA/sampleB, the raw degree
token, and a parsed `p0_norm`), keyed by its canonical pair key (section 5) and
inserted into the map.

---

## 5. The canonical pair key

A relatedness pair is **unordered** — (Ann, Bob) is the same pair as (Bob, Ann) —
but two tables may list the two samples in either order. So every row is keyed by a
canonicalized pair key rather than by the raw column values:

```
pair_key(a, b)  =  min(a, b) + '\x1f' + max(a, b)
```

The two names are sorted lexicographically and joined with an ASCII Unit Separator
(`\x1f`). Sorting makes the key order-independent, so a pair matches across the two
tables regardless of which sample each one put in `sampleA`. The `\x1f` separator is
a non-printable control byte that can't appear in a real sample id, so it can't be
forged by a name that happens to contain a comma or a dash — the join is
unambiguous.

Inserting into the map with `emplace` also gives duplicate detection for free: if a
single table lists the same unordered pair twice, the second `emplace` fails and
that's a bad-input "duplicate pair key" error. A table is expected to carry each
pair at most once.

---

## 6. The three outcome lanes (why a FAIL isn't an error)

The result carries a `ConcordStatus` that deliberately separates *"the tables
disagree"* from *"a table is broken"*:

| Status | Meaning |
|---|---|
| `kOk` | The comparison ran cleanly. Whether the two tables actually *agreed* is a separate boolean, `report.pass`. |
| `kBadInput` | A missing required column, a degree token outside the enum, or a duplicate pair key. |
| `kIoError` | A table file couldn't be opened or read (including an empty/header-less file). |

This split matters. A clean concordance **FAIL** — steppe and the oracle genuinely
disagree — is a real, meaningful result the validator is *supposed* to produce; it
is `kOk` with `report.pass == false`. It must never be confused with a malformed
input, where the validator couldn't even do the comparison. Callers can therefore
tell "steppe is wrong" apart from "the table is broken," which are very different
things to a human debugging a run.

---

## 7. Parsing P0_norm, and why NA becomes NaN

`parse_p0` turns a `P0_norm` cell into a `double`. The one design decision worth
calling out: **a missing or unparseable value becomes NaN, and a NaN is treated as
a real failure — never a silent pass.**

An empty cell, or the literal `NA` / `null` / `NaN` / `nan`, or any string that
`strtod` can't get a single digit out of, all map to `std::nan("")`. That NaN then
flows into the comparison in section 8, where it counts as **not** within tolerance
(a NaN compares false against any bound) and is excluded from the max-deviation
trackers. So a pair whose P0_norm is missing in either table drags the
within-tolerance fraction down rather than being quietly skipped or counted as a
match. Missing data is a failure to concord, not an absence of evidence.

Note the distinction from the header struct's default: a freshly-constructed
`Readv2Row` has `p0_norm = 0.0`, but that default is only a placeholder — every row
`read_table` produces has its `p0_norm` set by `parse_p0`, so a real missing cell is
a genuine NaN, not a stealthy zero.

---

## 8. Computing the report (`run_concordance`)

`run_concordance` is the entry point. It reads both tables (bailing early on any
IO/bad-input error), then walks the **intersection** of the two pair maps — every
metric is over the pairs the two tables share.

It iterates B (the oracle) and looks each pair up in A (the test). For each common
pair:

- **The confusion cell** `confusion[b_deg][a_deg]` is bumped — oracle degree picks
  the row, steppe degree the column. When the two degrees match, the
  `degree_agree_num` counter ticks up.
- **The P0_norm comparison** uses a combined absolute + relative tolerance:

  ```
  within  ==  |a_p0 - b_p0|  <=  p0_atol  +  p0_rtol * |b_p0|
  ```

  The absolute floor (`p0_atol`, default `5e-3`) keeps tiny values from failing on
  rounding; the relative term (`p0_rtol`, default `1e-2`) scales the allowance with
  the oracle magnitude. Pairs that clear it bump `p0_within_tol_num`.
- **The worst-case trackers** record the largest absolute deviation and the largest
  relative deviation seen, each with the pair key that produced it — so a report can
  point at *which* pair was the worst offender. The relative tracker is only updated
  when `b_p0 != 0.0`, to avoid dividing by zero.

Pairs only in A are `a_only` (steppe emitted them, the oracle didn't — reported but
not fatal); pairs only in B are `b_only` (steppe missed them — these drive coverage
down). Both are derived by subtraction from the row counts and the common count.

The three headline fractions — `coverage` (common / B rows), `degree_agreement`
(agreements / common), and `p0_within_tol_frac` (within-tol / common) — each guard
against a zero denominator by falling to `0.0`.

---

## 9. The PASS gate

`report.pass` is true only when **all three** floors are cleared at once:

```
pass  ==  coverage           >= coverage_min           (default 1.0)
      &&  degree_agreement    >= degree_agreement_min   (default 0.95)
      &&  p0_within_tol_frac  >= p0_within_tol_min       (default 0.90)
```

The default `coverage_min` of `1.0` means steppe must reproduce *every* oracle pair
— a missed pair alone fails the run. An **empty intersection is never a pass**: with
zero common pairs, `degree_agreement` and `p0_within_tol_frac` both fall to their
zero-denominator `0.0` and fail their floors, so a run that found nothing to compare
can't slip through as a vacuous success.

The thresholds live in `ConcordanceThresholds` and are supplied by the caller, so
the CLI shim can loosen or tighten any floor without this file changing.

---

## 10. Rendering the report (`write_report`)

`write_report` emits the report in one of two formats.

**`json`** writes a single object: the row/coverage counts, the flat degree numbers,
the confusion matrix as a nested 4×4 array, the max-deviation values, the
within-tolerance counts, and a final `"result": "PASS" | "FAIL"`. All floating
values go through `g6` (a `%.6g` formatter) for a compact, stable rendering.

**Any other format (the text default)** writes three blocks:

1. A **human block** — the row tallies, the labeled degree confusion matrix (rows
   annotated as oracle truth, columns as steppe test), and a PASS/FAIL line per
   floor with its threshold and the worst-offender keys. This block is for
   eyeballing and is explicitly *not* test-load-bearing.
2. A **stable machine trailer** — one `concord_<field>: <value>` line per metric,
   the confusion matrix flattened to 16 space-separated ints on a
   `concord_confusion:` line, and a final `RESULT: PASS | FAIL`.

The split is deliberate: the human block can be reworded freely, but the
`concord_*` trailer and the final `RESULT:` line are the stable contract the tests
grep for. The confusion matrix is walked in the frozen index order (section 3) in
every rendering, so the columns always mean the same thing.

---

## 11. Contracts and edge cases

- **Order-independence.** Sample order within a pair, and row order within a table,
  never affect the result — the canonical key (section 5) and the map keying handle
  both.
- **Schema tolerance.** Extra columns, reordered columns, CSV or TSV, CRLF or LF,
  padded fields, and blank lines are all accepted. Only the four named columns must
  be present.
- **A NaN is a failure, not a skip** (section 7) — missing P0_norm data lowers the
  within-tolerance fraction.
- **A broken table is distinct from a disagreement** (section 6) — `kBadInput` /
  `kIoError` vs a `kOk` FAIL.
- **Zero common pairs never passes** (section 9).
- **`report` is only valid when `status == kOk`.** On any non-`kOk` status the
  `error` string carries the human message and the report should be ignored.
- **No device, ever.** This file is host-only and holds no GPU code; it is the
  parity ruler, not a compute path.

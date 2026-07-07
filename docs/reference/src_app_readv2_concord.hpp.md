# `readv2_concord.hpp` reference

## 1. Purpose

`src/app/readv2_concord.hpp` declares the **READv2 concordance validator** — the
Phase-0 "ruler" that tells you whether steppe's READv2 output agrees with the
reference tool's. It is the type-and-contract half of a two-file module: this
header names the row struct, the report struct, the thresholds, the status enum,
and the three free functions that do the work; the arithmetic behind them lives
in `src/app/readv2_concord.cpp` (see `src_app_readv2_concord.cpp.md`). This doc
describes the contract the header promises; the `.cpp` doc describes the
mechanics.

The whole module is **pure host, CUDA-free arithmetic**. There is no device, no
`RunConfig`, no kernel — it reads two text tables off disk and diffs them. That
isolation is deliberate: it makes the validator directly unit-testable on a
machine with no GPU toolkit, and it keeps the "did we get the right answer?"
check completely independent of the GPU sweep that produced the answer. The thin
command-line shim that wires this into `steppe readv2-concord` lives separately
in `cmd_readv2_concord.{hpp,cpp}`; this module never touches `argv`.

What it does is narrow and worth stating up front: it **never classifies and
never recomputes anything**. Both tables already carry a `degree` string and a
`P0_norm` number per pair. The validator only *diffs* those two columns — the
degree tokens against each other, and the P0_norm values against a tolerance. It
is a ruler, not a second estimator.

---

## 2. The frozen schema it reads

Both inputs are READv2 output tables in steppe's frozen Phase-0 schema — one row
per unordered sample pair, columns identified by **name** in the header line, not
by position:

```
sampleA  sampleB  n_windows  n_overlap_sites  P0_mean  P0_norm  degree  z
```

The validator only reads four of those columns — `sampleA`, `sampleB`, `degree`,
`P0_norm` — and tolerates (ignores) the rest. Reading by column name, not by
index, is what lets it accept both the steppe table and the reference tool's
table even if their unused columns differ in order or count.

The `degree` field is a closed vocabulary of exactly four lowercase tokens, in a
fixed index order that everything downstream depends on:

```
0 = identical   1 = first   2 = second   3 = unrelated
```

`degree_tokens()` hands back that array in that order, and `kNumDegrees` (= 4) is
the dimension of the confusion matrix. Any degree cell outside this enum is a bad
input, not a silently-tolerated string (section 6).

---

## 3. The A-vs-B framing: test against oracle

The two tables are not symmetric. By convention:

- **A** is steppe's own table — the `--a` **TEST** side, the thing under
  scrutiny.
- **B** is the reference tool's table — the `--b` **ORACLE**, treated as ground
  truth.

That asymmetry shows up in every metric. The confusion matrix has **oracle B on
the rows (truth) and steppe A on the columns (test)**. Coverage is measured
against B — a pair the oracle emitted that steppe missed (`b_only`) drives
coverage *down*, because it is a pair steppe should have found and didn't. A pair
steppe emitted that the oracle lacks (`a_only`) is reported for visibility but is
**not fatal** — it doesn't fail the run on its own. The mental model is "did
steppe reproduce the oracle?", and every number is oriented that way.

---

## 4. The canonical pair key

`pair_key(a, b)` produces the join key both tables are matched on:

```
min(a, b) + '\x1f' + max(a, b)
```

The two sample labels are sorted lexicographically and joined with an ASCII unit
separator (`0x1f`). Sorting makes the pair **unordered**: table A listing a pair
as `(X, Y)` and table B listing it as `(Y, X)` still collide on the same key and
get compared. The `0x1f` separator is chosen because it cannot appear inside a
normal sample label, so `("a", "b\x1fc")` and `("a\x1fb", "c")` can't alias into
the same key — the separator is unambiguous. This is the *same* canonical pair
identity the READv2 emitter uses when it writes rows (`sampleA <= sampleB`), so
the keys line up by construction.

---

## 5. What the report carries

`run_concordance(a_path, b_path, thr)` is the entry point: it reads both tables,
computes every metric over their intersection, gates PASS/FAIL against the
thresholds, and returns a `ConcordResult`. That result has two layers.

The outer `ConcordResult` is the **outcome lane**. Its `ConcordStatus status`
tells you whether the computation even happened:

| Status | Meaning |
|---|---|
| `kOk` | Both tables parsed cleanly and were diffed. `report` is valid; `report.pass` carries PASS vs FAIL. |
| `kBadInput` | A required column was missing, a `degree` token fell outside the enum, or a pair key appeared twice in one table. `error` explains; `report` is meaningless. |
| `kIoError` | A table file couldn't be opened or read (or was empty / header-less). `error` explains; `report` is meaningless. |

This separation is load-bearing (section 6): **a broken input is a distinct
outcome from a clean disagreement.** A malformed table is not a concordance
FAIL — you don't want "your file has a typo" to read as "steppe disagrees with
the oracle."

The inner `ConcordanceReport` (valid only when `status == kOk`) holds the
numbers, in three groups:

- **Coverage** — `a_rows`, `b_rows`, `common_pairs` (the intersection), `a_only`,
  `b_only`, and `coverage = common_pairs / b_rows`.
- **Degree agreement** — the 4x4 `confusion` matrix (oracle rows x steppe cols),
  the agreement count `degree_agree_num` over `degree_agree_den` (= common
  pairs), and the ratio `degree_agreement`.
- **P0_norm concordance** — how many common pairs fell within tolerance
  (`p0_within_tol_num` / `_den`, and the fraction `_frac`), plus the worst
  offenders for a human to eyeball: the largest absolute deviation
  (`p0_max_abs_dev`) and largest relative deviation (`p0_max_rel_dev`), each with
  the pair key that produced it.

And the single verdict: `pass`.

---

## 6. Contracts and invariants

- **Bad input is never a silent pass.** Every parse-side problem — missing
  required column, out-of-enum degree token, duplicate pair key, unreadable /
  empty file — stops the run with `kBadInput` or `kIoError` and a human message,
  rather than being quietly skipped. The validator refuses to guess.

- **A P0_norm cell of `NA` / `null` / empty / non-numeric becomes NaN, and a NaN
  is a *failure*, not a skip.** `Readv2Row::p0_norm` documents this: a NaN counts
  as **not within tolerance** and is excluded from the max-deviation trackers. A
  missing number can't launder itself into a pass — it just isn't a match.

- **All metrics are over the intersection.** Degree agreement and P0_norm
  concordance are computed only on pairs present in *both* tables. Pairs that
  exist in only one side are accounted for in `a_only` / `b_only` and folded into
  coverage, but they contribute no confusion cell and no tolerance count.

- **An empty intersection is never a PASS.** With zero common pairs, the
  agreement and within-tolerance fractions fall to 0 (and coverage to 0 when B is
  empty), so the PASS gate can't be cleared vacuously. "We compared nothing"
  reports as FAIL, not success.

- **The three PASS floors are ANDed.** A run passes only when *all* of coverage,
  degree agreement, and P0_norm-within-tolerance clear their minimums
  simultaneously. The defaults in `ConcordanceThresholds` are the frozen bar:
  `coverage_min = 1.0` (every oracle pair must be reproduced),
  `degree_agreement_min = 0.95`, `p0_within_tol_min = 0.90`.

- **The P0_norm tolerance is combined absolute + relative.** A pair matches when
  `|a - b| <= p0_atol + p0_rtol * |b|`, with defaults `p0_atol = 5e-3` and
  `p0_rtol = 1e-2`. The oracle value `b` is the reference for the relative
  term — consistent with B being ground truth (section 3). The relative-deviation
  tracker skips pairs where `b == 0` to avoid dividing by zero.

- **The thresholds are a parameter, not a constant.** `run_concordance` takes the
  `ConcordanceThresholds` by argument, so a caller (or a test) can tighten or
  loosen any floor without touching the module. The struct's member defaults are
  the shipped policy.

---

## 7. Rendering the report

`write_report(os, rep, thr, format)` serializes a report to a stream in one of
two formats:

- **`"text"`** — a human-readable block (the confusion table and per-metric
  PASS/FAIL lines, for eyeballing) followed by a **stable machine trailer** of
  `concord_*` key/value lines and a final `RESULT: PASS|FAIL`. The trailer is the
  test-load-bearing part: tests grep those exact keys, so their names and format
  are a contract even though the human block above them is free to be pretty.
- **`"json"`** — one JSON object with the same fields, `confusion` rendered as a
  nested 4x4 array.

Both formats report the same numbers; the split is human-glance versus
machine-parse. The final verdict appears as a literal `PASS` or `FAIL` string in
each, so a caller never has to re-derive it from the individual metrics.

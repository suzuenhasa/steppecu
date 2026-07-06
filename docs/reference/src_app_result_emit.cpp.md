# `result_emit.cpp` reference

## 1. Purpose

`src/app/result_emit.cpp` turns the result of every steppe command into text —
CSV, TSV, or JSON — ready to print to standard output. It is the single place
where a computed result becomes bytes on the wire.

It sits in the command-line application, not in the core library. The library
itself never prints; it returns result structs, and the command-line tool owns
standard output. This file is that app-side serializer. It is plain C++20 and
pulls in no GPU code, so it can format a result without touching CUDA.

Two properties drive almost every decision in the file:

1. **Byte-for-byte fidelity to committed golden files.** Each output shape is
   designed so a fresh run can be compared directly, character for character,
   against a stored reference file. The column names, the section markers, the
   quoting, and the way "not computed" is written are all chosen to match those
   golden files exactly.
2. **Readability by downstream analysis scripts.** The column names mirror
   the parity field names[^at2], so an existing analysis script can read
   steppe's output unchanged.

The file serializes six different result types: a single qpAdm fit, a qpAdm
rotation (many models at once), a qpWave rank sweep, and the three standalone
statistics f4, f3, and f4-ratio. Every emitter is **total**: a result that
failed for a domain reason (for example, a model that produced no weights)
still serializes cleanly by writing sentinel values, never by throwing.

---

## 2. Number formatting and the exact value round-trip

Every floating-point number is printed with **17 significant digits**.

That specific count is not arbitrary. Seventeen significant digits is the
smallest number that round-trips an IEEE-754 double exactly: a value printed at
17 digits and then read back parses to the bit-identical double it came from.
This means the emitted CSV or JSON loses nothing — a comparison against a golden
file can hold weights and chi-squared values to a tight relative tolerance
(around 1e-6) while the text itself carries the full precision.

There are two number formatters, differing only in how they write a
not-a-number value:

| Function | A normal value | A NaN value |
|---|---|---|
| `fmt_double` (CSV/TSV) | 17 significant digits | the literal `NA` |
| `json_double` (JSON) | 17 significant digits | the literal `null` |

The split exists because `NA` is the natural "missing" marker in a CSV that R
or a spreadsheet will read, whereas `NA` is not valid JSON — so the JSON path
writes the JSON `null` literal instead. A NaN reaches these formatters whenever
a quantity is genuinely undefined, for example the p-value of a model whose
chi-squared could not be computed.

---

## 3. How "not computed" is represented

Missing or undefined outputs are written as sentinels rather than being omitted,
so every row keeps the same shape and a diff against the golden stays aligned.
Three sentinels are in play, and which one appears depends on the value's type
and the output format.

| Situation | CSV / TSV | JSON |
|---|---|---|
| A `double` that is NaN (undefined estimate, p-value, standard error, z-score) | `NA` | `null` |
| A standard error or z-score that was not computed for a model (the `se` array is empty) | `NA` | `null` |
| A degrees-of-freedom-difference that is not applicable | `NA` | `null` |

The degrees-of-freedom-difference case uses a distinct integer sentinel. Because
that column is an integer, it cannot use NaN; instead the value `INT_MIN` (the
most negative representable `int`) is the agreed "not applicable" marker. This
appears in the rank-drop table, where the first row has no nested comparison to
make. `fmt_dofdiff` turns `INT_MIN` into `NA` for CSV, and the JSON array
emitter turns it into `null`.

The guiding rule for consumers of the output: **a failed or undefined result is
a value, not an error.** To find models that failed, filter on the `status`
column (see below), not on whether a number came out as `NA`.

---

## 4. Status and precision labels

Two small columns carry a short human-readable tag describing how a result was
produced. Both label mappings now live in shared headers rather than in this
file — `status_str` in `include/steppe/error.hpp` and `precision_label` in
`src/app/precision_label.hpp` — and `result_emit.cpp` only calls them. Each
mapping is a `switch`, deliberately not a chained conditional, so that adding a
new enumerated value elsewhere in the codebase forces a visible compile-time
update at the switch rather than silently falling through.

### The `status` column

`status_str` maps the per-result outcome code to a lowercase string:

| Outcome code | Emitted string |
|---|---|
| `Ok` | `ok` |
| `DeviceOom` | `device_oom` |
| `RankDeficient` | `rank_deficient` |
| `NonSpdCovariance` | `non_spd_covariance` |
| `ChisqUndefined` | `chisq_undefined` |
| `InvalidConfig` | `invalid_config` |
| (any unexpected value) | `unknown` |

These are domain outcomes for a single model, carried as a value in the output —
never a serialization failure.

### The `precision` column

`precision_label` maps the arithmetic mode a result was computed in to a short
tag. It is defined in the shared header `src/app/precision_label.hpp`, not in
`result_emit.cpp`; this file just calls it:

| Precision mode | Emitted string |
|---|---|
| `EmulatedFp64` | `emu` |
| `Tf32` | `tf32` |
| `Fp64` | `fp64` |
| (fallback default) | `fp64` |

The fallback returns `fp64` so that native double precision stays the default
label when the mode is unrecognized.

---

## 5. Feasibility

"Feasible" is the standard qpAdm screen for whether a model's mixture weights
are physically sensible. There are two feasibility helpers, used by different
output paths.

### Whole-model feasibility (`model_feasible`)

The canonical rule[^at2]: a model is feasible when **every**
weight lies in the closed interval from 0 to 1. A model with no weights (a
domain-failed model) is not feasible. This drives the `feasible` field of the
single-model summary and JSON summary.

### Rotation feasibility (`rotation_feasible`)

The rotation path prefers the engine's own recorded decision over recomputing
it. The full-model row of a model's population-drop table (its first row) holds
exactly the feasibility flag the search engine wrote, so `rotation_feasible`
reads that flag when it is present. When the population-drop table is empty (a
model that failed before producing one), it falls back to the canonical
`model_feasible` rule. Both sources agree for well-formed models — they apply
the same screen — so either matches the golden file.

---

## 6. CSV field quoting: two distinct policies

The file contains **two different** ways to quote a CSV field, and choosing the
right one for each output is deliberate.

### Always-quote (`csv_quote`, file-local)

This wraps every string in double quotes unconditionally and doubles any
embedded quote character. It is used for the string columns of the qpAdm, qpWave,
and standalone-statistic outputs, because the committed golden CSVs for those
commands wrap **every** string column in quotes. To diff byte-for-byte against
those goldens, steppe must quote the same way, always.

### Conditional per RFC 4180 (`csv_field`, public)

This returns the string **unchanged** unless it contains the active separator, a
double quote, or a carriage return or newline; only then does it wrap the field
and double embedded quotes. It is a public helper used by the other emitters
(dates, qpGraph, the f-statistic sweep) whose golden files carry **bare**
population names and whose command-line tests split on bare tokens. For a normal
population name (no special characters), this returns the name untouched, so
that output stays byte-identical while a pathological name is still escaped
safely.

### JSON string escaping (`json_quote`, public)

A companion public helper that returns a string wrapped in double quotes with the
JSON control characters escaped: the double quote, the backslash, and the
newline, carriage-return, and tab characters. It is shared so every JSON emitter
escapes labels consistently instead of concatenating raw quotes around a label.

`csv_field` and `json_quote` are the two escaping primitives promoted to public,
namespace-scope functions precisely so that the emitters which bypass this file's
main path still route through one shared escaping seam.

---

## 7. Output schemas per command

Each command has a fixed column-and-section layout, chosen to match a specific
golden file. The CSV/TSV forms of the sectioned outputs prefix each section with
a `# section: NAME` comment line so a parser can split the single stream into
its parts. The standalone-statistic tables have **no** section prefix — they are
a single flat table (a bare header row plus data rows), so a row-for-row diff is
direct.

### Single qpAdm fit

Matches `golden_fit0`. Four CSV sections, and the parallel-array JSON mirror:

| Section | Columns |
|---|---|
| `weights` | target, left, weight, se, z (one row per left source) |
| `summary` | p, chisq, dof, f4rank, est_rank, feasible, status, precision |
| `rankdrop` | f4rank, dof, chisq, p, dofdiff, chisqdiff, p_nested |
| `popdrop` | pat, wt, dof, chisq, p, f4rank, feasible |

In the `weights` section, `se` and `z` are `NA`/`null` when the standard-error
array is empty (a model for which no standard errors were computed).

### qpAdm rotation

Matches `golden_rot`. One section, **one row per model**, in input order.

CSV columns: model_index, target, left, right_n, p, chisq, dof, f4rank,
feasible, status, weights, se. The JSON form is a top-level object carrying the
shared target and right_n plus a `models` array of per-model objects
(model_index, left, weight, se, z, p, chisq, dof, f4rank, feasible, status).

Within a row, the multiple left labels are joined into one field with
semicolons, and the per-model weights and standard errors are likewise joined
with semicolons into single sub-fields. A weight whose standard error is absent
gets `NA` in that sub-field; a model with no standard errors at all writes the
whole `se` field as `NA`.

The `f4rank` column here is subtle — see section 8.

### qpWave rank sweep

Matches `golden_qpwave`. qpWave has no target, so there are no admixture weights
and no population-drop table. Three CSV sections:

| Section | Columns |
|---|---|
| `rankdrop` | f4rank, dof, chisq, p, dofdiff, chisqdiff, p_nested (f4rank descending) |
| `per_rank` | rank, chisq, dof, p (rank ascending; `rank` is the 0-based index) |
| `summary` | f4rank, est_rank, status, precision, reference, right_n |

The JSON form is a single object with `left`, `right_n`, and the `rankdrop`,
`per_rank`, and `summary` blocks. The `rankdrop` block is byte-shaped exactly
like the qpAdm rank-drop block because both go through the same shared helpers.

### Standalone f4, f3, and f4-ratio

Each is a single flat table, one row per input tuple, in input order, with no
section prefix. Each matches a regenerated fixture golden:

| Command | Columns | Golden |
|---|---|---|
| f4 | pop1, pop2, pop3, pop4, est, se, z, p | `golden_fit0_f4_readf2` |
| f3 | pop1, pop2, pop3, est, se, z, p | `golden_fit0_f3_readf2` |
| f4-ratio | pop1, pop2, pop3, pop4, pop5, alpha, se, z | `golden_fit0_f4ratio_readf2` |

f3 is the three-population form of the f4 table (drop the fourth population).
f4-ratio adds a fifth population and reports `alpha`, `se`, and `z` — note it has
**no** `est` or `p` column, matching the f4-ratio output[^at2], which
reports only those three value columns. The JSON forms wrap the rows in a named
array (`quartets`, `triples`, or `tuples`) alongside a top-level `status` and
`precision`.

---

## 8. The rotation `f4rank` column: a deliberate mislabel

In the rotation output — both CSV and JSON — there is a column named `f4rank`,
and it does **not** carry the value most readers would assume.

The column is named `f4rank` only to mirror the golden file and the parity
field name[^at2], so a byte-for-byte diff lines up. But the value written into it is
the per-model **fitted** rank (`est_rank`), not the rank **decision** (`f4rank`).

The reason: the reference rotation output has no meaningful rank-decision field
(it is null there), and its per-model `f4rank` value equals the fitted rank
recorded on the full-model population-drop row. The rank-decision value is
meaningless for the rotation path and is deliberately not emitted. Keeping the
column named `f4rank` preserves the diff against the golden while the value
underneath is `est_rank`.

If you are reading rotation output and expecting the `f4rank` column to be the
rank decision, this is the gotcha: it is the fitted rank.

---

## 9. Shared helpers and public entry points

### Shared format helpers

To avoid copy-pasting the same loop into every emitter, several small helpers are
defined once and reused across the qpAdm and qpWave paths:

- `emit_int_arr`, `emit_dbl_arr`, `emit_dofdiff_arr` write a named JSON array of
  ints, doubles, or degrees-of-freedom-differences, applying the `null` sentinels
  where needed. A trailing-comma flag lets the caller mark the last array in a
  block.
- `emit_rankdrop_csv` writes the seven-column rank-drop table body, shared
  verbatim between the single-model and qpWave CSV outputs because that table is
  identical in both.
- `join_left` joins a model's left-source labels into one semicolon-separated
  field.
- `label_at` safely reads the k-th population label from a parallel label array,
  returning an empty string when the array is shorter than the value array. It is
  shared by the f4, f3, and f4-ratio emitters so their out-of-range fallback is
  uniform.

### Public entry points

`parse_output_format` maps a format token to the selector: `csv`, `tsv`, or
`json`, returning false for any other token so the caller can fail fast.

The six public `emit_*_result` / `emit_*_table` functions are thin dispatchers.
Each one switches on the format: **CSV and TSV share the same emitter**, called
with a different separator character (a comma for CSV, a tab for TSV), while JSON
has its own dedicated emitter. This is why the CSV and TSV outputs are structurally
identical apart from the delimiter.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

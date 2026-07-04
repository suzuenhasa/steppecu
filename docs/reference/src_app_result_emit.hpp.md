# `result_emit.hpp` reference

## 1. Purpose

`src/app/result_emit.hpp` is the command-line tool's serializer: it turns the
result structs the compute engine produces (qpAdm fits, qpWave rank sweeps, and
the standalone f4 / f3 / f4-ratio statistics) into text output — CSV, TSV, or
JSON — on an output stream.

It lives in the app layer for a deliberate reason. The library itself never
prints anything; only the command-line tool is allowed to write to standard
output. So every function here is app-side glue that takes a finished result and
formats it. The header is plain C++20 and pulls in no GPU code — it only needs
the result struct definitions and the standard library.

One property runs through every function in this file: **serialization is
total**. A run that failed to produce a real answer — no admixture weights, an
undefined chi-square, a degenerate fit — is still a valid outcome of the
analysis, not an error in formatting. Those "domain outcomes" come out as ordinary
rows with `NA` placeholders in the affected columns; no emitter ever throws. That
way a downstream script always receives a well-formed table it can parse and
filter, whether or not the underlying model was feasible.

---

## 2. Output formats and shared conventions

### `OutputFormat`

`OutputFormat` is the on-the-wire format selector, parsed from the `--format`
flag. It has three values:

| Value | Meaning |
|---|---|
| `Csv` | Comma-separated. **The default.** |
| `Tsv` | Tab-separated. Same layout as CSV, different field separator. |
| `Json` | A single JSON object. |

### How CSV and TSV are laid out

The CSV/TSV output is shaped like the reference R package's tables[^at2],
so its downstream scripts can read a steppe run unchanged. A single result can
contain several logically distinct tables — for a qpAdm fit, for example, a
weights table, a one-line summary, and two diagnostic tables. Rather than write
them to separate files, all of the tables are streamed one after another into the
same output, and **each table is introduced by a comment line of the form
`# section: NAME`**. A parser splits the stream back into its tables on those
markers. The column names within each section mirror the committed reference CSVs
exactly.

### How JSON is laid out

The JSON output is a single object that mirrors the shape of the committed
reference JSON files. Each table becomes a block of parallel arrays (one array per
column), and the summary becomes a block of scalar fields. The point of matching
that shape is diffability: a fresh run can be compared directly against a stored
reference file to confirm the numbers still match.

### The `NA` / status convention

Results that carry a status also carry a per-row `status` string. When a fit's
chi-square is undefined, its p-value is a not-a-number sentinel, and that value is
written out as the literal text `NA` (in CSV/TSV) or a JSON null. The intended way
to tell a real result from a degenerate one is therefore to **filter on the
`status` column, not on the p-value** — an `NA` p-value is a legitimate reported
outcome, not a parsing failure. The same principle covers every other place a
value can be missing: empty standard-error sentinels, the not-applicable
nested-difference cells in the last row of a rank sweep, and so on all render as
`NA` / null.

---

## 3. Format helpers

Three small free functions do the low-level formatting that the emitters share.

### `parse_output_format`

```
bool parse_output_format(const std::string& token, OutputFormat& out);
```

Maps a `--format` token (`"csv"`, `"tsv"`, or `"json"`) to the corresponding
`OutputFormat`, writing it into `out`. Returns `false` on an unrecognized token so
the caller can fail fast. The configuration builder also validates this flag
earlier, but the serializer re-maps the token here at emit time so that it is
self-contained and does not depend on the earlier validation having run.

### `csv_field` — conditional quoting

```
std::string csv_field(const std::string& s, char sep);
```

Quotes a CSV field only when it has to, following RFC 4180. If the string contains
none of the four troublesome characters — the field separator, a double quote, a
newline, or a carriage return — it is returned **completely unchanged**. Otherwise
it is wrapped in double quotes and any embedded quote is doubled.

This is intentionally different from an always-quote primitive used elsewhere (the
qpAdm and f4 golden CSVs wrap every string column in quotes). Some emitters — the
DATES, qpGraph, and f-statistic-sweep outputs — write bare, unquoted labels
because the tests and gates that consume them split on bare tokens. Conditional
quoting lets those emitters keep every real population name byte-for-byte identical
to what a bare concatenation would have written, while still safely escaping a
pathological name that happens to contain a separator or a quote.

### `json_quote` — JSON string escaping

```
std::string json_quote(const std::string& s);
```

Escapes a label or field for JSON — double quote, backslash, and the newline,
carriage-return, and tab control characters — and returns the value **with its
surrounding double quotes included**. It is shared so that the emitters which
would otherwise build a JSON string by hand (again DATES, qpGraph, and the
f-statistic sweep) escape their labels properly instead of concatenating a raw
quote, the label, and another raw quote. For any label with no quote, backslash, or
control character, the result is byte-identical to that old manual concatenation,
so it moves no existing reference output.

---

## 4. Serializing one qpAdm result

```
void emit_qpadm_result(std::ostream& os, OutputFormat fmt,
                       const QpAdmResult& result,
                       const std::string& target_label,
                       const std::vector<std::string>& left_labels);
```

Writes a single qpAdm fit to `os` in the chosen format. This is the four-section
layout:

- **`weights`** — one row per left-hand source (`target`, `left`, `weight`, `se`,
  `z`). When the standard-error sentinel is empty, `se` and `z` are written as
  `NA`.
- **`summary`** — a single row of scalars: `p`, `chisq`, `dof`, `f4rank`,
  `est_rank`, `feasible`, `status`, `precision`.
- **`rankdrop`** — the rank-drop diagnostic table, in the parity result
  order[^at2].
- **`popdrop`** — the population-drop diagnostic table, in the same order.

The JSON form is one object with `weights`, `rankdrop`, and `popdrop` blocks (each
a set of parallel arrays) plus a scalar `summary` block, so a run diffs directly
against the committed golden.

| Parameter | Meaning |
|---|---|
| `target_label` | The resolved name of the target population. |
| `left_labels` | The resolved names of the left-hand sources, one per weight, used to label the per-source rows and the popdrop columns. Its length equals the number of weights when the fit produced weights. |

A domain-outcome result (empty weights / empty standard error) emits `NA`
sentinels rather than throwing.

---

## 5. Serializing a qpAdm rotation table

```
void emit_rotation_table(std::ostream& os, OutputFormat fmt,
                         std::span<const QpAdmResult> results,
                         const std::string& target_label,
                         const std::vector<std::vector<std::string>>& left_labels_per_model,
                         int right_n);
```

Writes the result of a qpAdm **rotation** — a batched run that fits many candidate
models sharing one target — as **one row per model**, in the input (model-index)
order the engine returns. This is a sibling of `emit_qpadm_result`, not the
single-model four-section layout: a rotation reports a per-model feasibility table,
so each model collapses to a single row.

| Parameter | Meaning |
|---|---|
| `results` | The per-model fits; `results[i]` is the fit of the i-th input model, in input order. |
| `target_label` | The resolved target name, shared by every model. |
| `left_labels_per_model` | For each model, the resolved names of its left-hand sources — used for that row's `left` column and for the per-weight column headers. Its length equals the number of results. |
| `right_n` | The number of right-hand (reference) populations reported per row, by the convention of one fewer than the full right set. |

In CSV/TSV the output is a `# section: rotation` header, a column-header row, and
one data row per model. In JSON it is a `{ "models": [ ... ] }` array whose entries
carry `model_index`, `left`, `weight`, `se`, `z`, `p`, `chisq`, `dof`, `f4rank`,
`feasible`, and `status`, mirroring the committed golden's `models[]` shape. As
everywhere, a domain-outcome row emits `NA` / null sentinels rather than throwing.

---

## 6. Serializing a qpWave rank sweep

```
void emit_qpwave_result(std::ostream& os, OutputFormat fmt,
                        const QpWaveResult& result,
                        const std::vector<std::string>& left_labels,
                        int right_n);
```

Writes a single qpWave rank-sweep result. qpWave differs from qpAdm in what it
produces: it has **no target and no admixture weights**, and therefore no popdrop
table. Its result is a rank-sufficiency sweep — testing each candidate rank — plus
the rank-drop diagnostic table and a small summary.

The CSV/TSV form has three sections:

- **`rankdrop`** — the rank-drop table in descending-`f4rank` order:
  `f4rank`, `dof`, `chisq`, `p`, `dofdiff`, `chisqdiff`, `p_nested`.
- **`per_rank`** — the ascending-rank sweep: `rank`, `chisq`, `dof`, `p`.
- **`summary`** — `f4rank`, `est_rank`, `status`, `precision`.

The JSON form is one object with `left[]`, `right_n`, `rankdrop{}`, `per_rank{}`,
and `summary{}`, using the same parallel-array shape as the committed golden.

| Parameter | Meaning |
|---|---|
| `left_labels` | The resolved left-hand source names, echoed into the output header for readability; the first entry is the qpWave reference row. |
| `right_n` | The number of right-hand populations, by the same "one fewer than the full right set" convention as the rotation table. |

This emitter reuses the same numeric-formatting, quoting, status, and
parallel-array building blocks as `emit_qpadm_result`, so its rank-drop block comes
out byte-shaped exactly like the qpAdm path's. The last row of a rank sweep has no
nested-model to compare against, so its `dofdiff` / `chisqdiff` / `p_nested` cells
are the not-applicable sentinels and render as `NA` / null. It never throws.

---

## 7. Standalone f-statistic emitters (f4, f3, f4-ratio)

The three standalone statistic commands each have their own emitter. All three
write **one row per input tuple, in input order**, and all three share the exact
same formatting primitives as the qpAdm/qpWave emitters — no formatting logic is
duplicated. Each result struct carries population *indices*; the app resolves those
back to names and passes the label vectors in. In every case the label vectors have
one entry per output row, and a degenerate row (a not-a-number estimate or standard
error) emits `NA` / null sentinels without throwing.

### `emit_f4_result`

```
void emit_f4_result(std::ostream& os, OutputFormat fmt,
                    const F4Result& result,
                    const std::vector<std::string>& p1_labels,
                    const std::vector<std::string>& p2_labels,
                    const std::vector<std::string>& p3_labels,
                    const std::vector<std::string>& p4_labels);
```

The `steppe f4` command. One row per population quartet, with columns
`pop1`, `pop2`, `pop3`, `pop4`, `est`, `se`, `z`, `p` — matching the regenerated
golden schema so the parity test can diff row for row. `p1_labels` … `p4_labels`
are the resolved names of each quartet's four populations.

### `emit_f3_result`

```
void emit_f3_result(std::ostream& os, OutputFormat fmt,
                    const F3Result& result,
                    const std::vector<std::string>& p1_labels,
                    const std::vector<std::string>& p2_labels,
                    const std::vector<std::string>& p3_labels);
```

The `steppe f3` command — essentially `emit_f4_result` with the fourth population
column dropped. One row per population triple, columns
`pop1`, `pop2`, `pop3`, `est`, `se`, `z`, `p`, again matching a fixture golden
schema for row-for-row parity.

### `emit_f4ratio_result`

```
void emit_f4ratio_result(std::ostream& os, OutputFormat fmt,
                         const F4RatioResult& result,
                         const std::vector<std::string>& p1_labels,
                         const std::vector<std::string>& p2_labels,
                         const std::vector<std::string>& p3_labels,
                         const std::vector<std::string>& p4_labels,
                         const std::vector<std::string>& p5_labels);
```

The `steppe f4-ratio` command. One row per five-population tuple, with columns
`pop1` … `pop5`, `alpha`, `se`, `z`. Note there is **no `p` column** here —
the f4-ratio reports only the ratio estimate `alpha`[^at2], its standard
error, and the z-score. `p1_labels` … `p5_labels` are the resolved names of each
five-tuple's populations.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

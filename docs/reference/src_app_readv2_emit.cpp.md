# `readv2_emit.cpp` reference

## 1. Purpose

`src/app/readv2_emit.cpp` is the row serializer for the READv2 command — the three
tiny functions that turn one finished relatedness result into a line of `csv`,
`tsv`, or `json`. It writes the frozen READv2 output schema:

```
sampleA  sampleB  n_windows  n_overlap_sites  P0_mean  P0_norm  degree  z
```

and nothing else. It does no compute, touches no GPU, opens no files, and holds no
state. Given an already-finished `Readv2OutRow` and an output stream, it formats the
eight columns in their fixed order and moves on. Everything upstream of it — reading
genotypes, packing bits, running the mismatch kernel, jackknifing, classifying the
relationship degree — has already happened by the time a row reaches this file.

This is a pure host translation unit. There is no CUDA header here, no `RunConfig`,
no allocation beyond the small strings the number formatters return. That isolation
is deliberate and is enforced by the same build-time grep gate that guards the rest
of the app target: this file must stay free of GPU code.

---

## 2. Where it sits in the READv2 pipeline

Three collaborators bracket this file:

- **`readv2_emit.hpp`** declares the one input struct, `Readv2OutRow` — an
  already-resolved, already-canonicalized row (its `sampleA`/`sampleB` labels are
  real Genetic-IDs, ordered so `sampleA <= sampleB` lexicographically). This file
  never re-orders or re-labels; it trusts the row it's handed.
- **`cmd_readv2.cpp`** builds each `Readv2OutRow` from the kernel's raw
  `Readv2PairRow` (resolving the sample indices `i`/`j` to labels, canonicalizing the
  pair, copying the numeric fields across) and hands it to the writer.
- **`readv2_shard_writer.cpp`** owns the actual output stream and calls these three
  functions: `readv2_emit_header` once at the top of a `csv`/`tsv` file, then
  `readv2_emit_row` or `readv2_emit_json_row` per pair. This file is the formatting
  half of that writer, factored out so the header text and the per-row text live in
  exactly one place.

So this file is a leaf: pure formatting, called by the shard writer, fed by the
command.

---

## 3. Reusing the shared `result_emit` primitives (why the numbers match everywhere)

The single most important design decision here is that this file formats **nothing**
on its own. Every field goes through the same four primitives that every other
steppe command's output uses, declared in `app/result_emit.hpp`:

| Primitive | Job |
|---|---|
| `csv_field(s, sep)` | Conditionally RFC-4180-quotes a string field — only wraps it in quotes (doubling any interior quotes) when it actually contains the separator, a quote, or a newline. |
| `json_quote(s)` | JSON-escapes a string and wraps it in double quotes. |
| `fmt_double(v)` | Formats a double for csv/tsv at 17 significant digits (exact round-trip), rendering NaN as the literal `NA`. |
| `json_double(v)` | The same 17-digit formatting for JSON, but rendering NaN as `null`. |

The payoff is that a `P0_mean` or a `z` printed by READv2 is byte-for-byte the same
number a `z` printed by `f4` or `qpadm` would be — same precision, same NaN spelling,
same quoting rules — because it went through the same code. READv2 gets consistent,
already-tested formatting for free, and there is no second copy of "how steppe writes
a double" to drift out of sync. This mirrors the reuse pattern the rest of the app
follows; READv2 is not a special case.

---

## 4. The three functions

### `readv2_emit_header(os, fmt)`

Writes the one header line for a `csv` or `tsv` file. It picks the separator — a tab
when `fmt == OutputFormat::Tsv`, a comma otherwise — and prints the eight column
names in the frozen order, terminated with `\n`. The column names are plain
identifiers (`sampleA`, `n_windows`, …) with no separator, quote, or newline in
them, so the header writes them as bare literals without going through `csv_field`.
There is no JSON header: a JSON file's structure carries the field names on every
row object, so the shard writer never calls this for `json`.

### `readv2_emit_row(os, fmt, row)`

Writes one `csv`/`tsv` data row. Same separator choice as the header. It runs the three
string fields (`sampleA`, `sampleB`, and `degree`) through `csv_field` so a label or
degree token that ever contained the separator would be correctly quoted, and runs
the four floating fields through `fmt_double`. The two integer counts — `n_windows`
and `n_overlap` — are streamed straight to the stream as integers; they need no
formatting helper. Terminated with `\n`.

### `readv2_emit_json_row(os, row)`

Writes one JSON object `{...}` for the row, with the keys in the same frozen order.
Strings go through `json_quote`, doubles through `json_double`, integers stream
directly. Note the two deliberate spelling choices: the **JSON key** is
`n_overlap_sites` (matching the header column and the schema string), even though the
struct field is named `n_overlap` and the csv/tsv `P0`/`z` values are the same
numbers. There is **no** trailing newline and no surrounding array — the emitter
writes a bare object. That's on purpose: the shard writer is the one that knows how
many rows there are, so it owns the commas between objects and the enclosing
`"rows": [ ... ]` array. This function's job is exactly one object.

---

## 5. Contracts and invariants

- **The schema is frozen and single-sourced.** The eight columns, their order, and
  their names (`sampleA sampleB n_windows n_overlap_sites P0_mean P0_norm degree z`)
  are fixed. The header line, the csv/tsv row, and the JSON keys must all agree, and
  the shard writer's `meta.json` `"schema"` string repeats the same eight tokens.
  Changing a column means changing all four spots in lockstep; that's the point of
  keeping them in one small file.
- **The row arrives finished.** `sampleA`/`sampleB` are already resolved to
  Genetic-ID labels and already canonicalized so `sampleA <= sampleB`
  lexicographically (the command does this to match the concord validator's
  `pair_key`). This file does not sort, dedupe, or relabel — it formats what it's
  given.
- **Number formatting is delegated, not reimplemented.** Every double goes through
  `fmt_double`/`json_double`; every string through `csv_field`/`json_quote`. This
  file introduces no new numeric or quoting behavior.
- **Separator selection is uniform.** Exactly one rule — tab for `Tsv`, comma
  otherwise — used identically by the header and the row so they can never disagree.
- **Statelessness.** These are free functions with no globals and no I/O of their
  own; they only write to the caller's stream. Reordering or repeating calls has no
  hidden effect beyond what lands on the stream.

---

## 6. Edge cases

- **NaN `z` (and any NaN double).** The `z` column is documented to be NaN when a
  z-score can't be computed. Because the value flows through `fmt_double` /
  `json_double`, that NaN renders as the bare token `NA` in csv/tsv and as `null` in
  JSON — never as a platform-specific `nan` spelling, and never as a number. The
  four floating columns all get this treatment, so any of them being NaN is handled
  the same way.
- **A label or degree token containing the separator.** `csv_field` quotes it (and
  doubles interior quotes) only when needed, so a Genetic-ID with a comma in it stays
  a single well-formed field in `csv` and a bare value otherwise. In JSON the
  `json_quote` escaping covers quotes, backslashes, and control characters.
- **No JSON header, no JSON array here.** Asking this file for a JSON header or a
  comma-joined array is a category error — that framing belongs to the shard writer.
  This file only ever emits one header line, one delimited row, or one JSON object.
- **Integer streaming.** `n_windows` (a `long`) and `n_overlap` (an `int64`) are
  written via the stream's own integer formatting, so their range is whatever the
  stream supports; they are counts, never NaN, and need no `NA`/`null` path.

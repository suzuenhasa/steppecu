# `readv2_emit.hpp` reference

## 1. Purpose

`src/app/readv2_emit.hpp` declares the three tiny serializers that turn one
finished READv2 pair result into one line of output — a csv row, a tsv row, or a
json object. It is the READv2 answer to the same job `result_emit` does for the
f-statistic and fit commands: take a result the GPU already computed and write it
out in whichever format the user asked for, with every number formatted exactly
the way every other steppe command formats numbers.

It is a plain host header. There is no CUDA here, no kernel, no compute — just the
shape of one output row (`Readv2OutRow`) and three functions that stream it. The
matching implementation lives in `src/app/readv2_emit.cpp`, and because that `.cpp`
is a direct, one-to-one realization of what this header declares, this one file
documents both.

The load-bearing design choice is in the include list: this header pulls in
`app/result_emit.hpp` and reuses its formatting primitives (`csv_field`,
`json_quote`, `fmt_double`, `json_double`) rather than rolling its own. That reuse
is the whole point — it means a READv2 `P0_mean` and a qpAdm `p` value are printed
by the same `%.17g` round-trip formatter, quoted by the same RFC-4180 rules, and
render a not-computed value with the same sentinel. READv2 does not get its own
number-formatting dialect.

---

## 2. The frozen Phase-0 schema

READv2 emits a fixed, frozen set of eight columns, in this exact order:

```
sampleA  sampleB  n_windows  n_overlap_sites  P0_mean  P0_norm  degree  z
```

That order is the schema. It is frozen — the header comment and the `readv2_emit_header`
line both spell it out — because downstream tooling (the concordance validator, the
committed goldens, anyone parsing the table) keys off exact column names and
positions. The three serializers all agree on this order: the header line, the
per-row csv/tsv writer, and the json object writer emit the same eight fields in
the same sequence, so a csv column and its json key can never drift apart.

What each column carries:

| Column | Field | Meaning |
|---|---|---|
| `sampleA` | `sampleA` | The lexicographically-first Genetic-ID label of the pair. |
| `sampleB` | `sampleB` | The second label. `sampleA < sampleB` always (section 4). |
| `n_windows` | `n_windows` | How many SNP windows the pair was scored over (a `long`). |
| `n_overlap_sites` | `n_overlap` | Non-missing overlapping sites for the pair (an `int64`). |
| `P0_mean` | `p0_mean` | The mean P0 (non-IBD proportion) statistic. |
| `P0_norm` | `p0_norm` | The normalized P0 the relationship call is made from. |
| `degree` | `degree` | The relationship degree label (a string, e.g. the kinship class). |
| `z` | `z` | The z-score; NaN when not computed (section 5). |

Note that the csv/tsv header prints `n_overlap_sites` while the struct field is
named `n_overlap` — the public column name is the frozen one, the field name is
just the C++ shorthand.

---

## 3. `Readv2OutRow` — one already-resolved row

`Readv2OutRow` is a plain value struct: eight fields, sensible zero defaults, no
methods. It is deliberately a *finished* row. By the time one of these reaches a
serializer, all the work is done:

- The two labels are **already resolved** from participant indices to their
  Genetic-ID strings.
- The pair is **already canonicalized** so `sampleA < sampleB` (section 4).
- The statistics are the final per-pair numbers from the READv2 run.

The serializers do not resolve, sort, compute, or validate anything — they format.
This split keeps the emit layer trivial and testable: the caller
(`cmd_readv2.cpp`, via its sink lambda) owns the "what goes in the row" decisions,
and this file owns only "how the row is spelled on disk". The struct is populated
in `cmd_readv2.cpp`'s `make_sink`, where a `Readv2PairRow` from the compute layer
is turned into a `Readv2OutRow` with its labels looked up and ordered.

---

## 4. The canonical `sampleA < sampleB` contract

Every row that reaches a serializer must already have `sampleA` lexicographically
at or before `sampleB`. This isn't enforced here — the serializers trust it — it's
established upstream. In `cmd_readv2.cpp` the sink compares the two resolved labels
(`li <= lj`) and assigns the smaller to `sampleA`, the larger to `sampleB`, before
building the row.

Why it matters: this is the same ordering the concordance validator's `pair_key`
uses. A pair `(X, Y)` and a pair `(Y, X)` are the same unordered pair, and both
sides — the emitter and the validator — have to agree on which label lands in
which column, or a correct result would look like a mismatch. Canonicalizing once,
at the point the row is built, makes the emitted table directly comparable to the
oracle without any re-sorting on either side.

---

## 5. NaN handling: `NA` for tables, `null` for json

The `z` field is documented to carry NaN when the z-score was not computed, and
the two output paths render that absence differently — but they don't decide it
themselves, they inherit it from the shared formatters:

- **csv / tsv** run `z` (and both P0 values) through `fmt_double`, which prints a
  17-digit round-trippable number for a real value and the literal `NA` for NaN.
- **json** runs the same fields through `json_double`, which prints the number or
  the bare literal `null` for NaN.

So a not-computed z appears as `NA` in a csv/tsv cell and as `null` in a json
object — the correct idiom for each format, and identical to how every other
steppe command renders a missing double. Because the serializers delegate this,
there is no NaN branch in `readv2_emit.cpp` at all; the policy lives once, in
`result_emit`.

The two integer counts (`n_windows`, `n_overlap`) are streamed directly as
integers with no sentinel — they are always real counts, never "not computed".

---

## 6. Field quoting and separator safety

String fields (`sampleA`, `sampleB`, `degree`) are never streamed raw:

- **csv / tsv** wrap each string with `csv_field(s, sep)`, which quotes the field
  (doubling any embedded `"`, RFC-4180 style) only when it actually contains the
  active separator, a quote, or a newline — so a clean label stays unquoted and a
  label with a comma survives in a tsv or a tab in a csv. The separator itself is
  chosen per call: a tab for `OutputFormat::Tsv`, a comma otherwise. (`OutputFormat::Json`
  never reaches the csv/tsv path — json rows go through their own function.)
- **json** wraps each string with `json_quote`, which does proper json string
  escaping (`\"`, `\\`, `\n`, `\r`, …).

A Genetic ID is unlikely to contain a comma, but the guarantee is that if one ever
did, the table would still parse. This is a correctness property inherited whole
from `result_emit`, not re-implemented here.

---

## 7. The three entry points and the streaming contract

| Function | Emits | Called |
|---|---|---|
| `readv2_emit_header(os, fmt)` | The single csv/tsv header line. | Once, at the top of a csv/tsv stream. |
| `readv2_emit_row(os, fmt, row)` | One csv/tsv data row, newline-terminated. | Once per pair. |
| `readv2_emit_json_row(os, row)` | One json object `{...}`, **no** trailing newline or comma. | Once per pair. |

There are two deliberate asymmetries here, and both come from how json differs
from a line-oriented format:

1. **No json header.** A csv/tsv stream needs a column-name line; a json array of
   objects does not. So there is a `readv2_emit_header` for the flat formats and
   nothing for json.
2. **The json writer emits only the object.** It writes `{...}` and stops — no
   newline, no comma, no surrounding `[` / `]`. That framing is the *streaming
   writer's* job. `Readv2ShardWriter` (`readv2_shard_writer.cpp`) is what opens the
   rows array, places the commas between objects, and closes it. `readv2_emit_json_row`
   is intentionally a leaf that knows nothing about its neighbors, so the same
   function works whether it's the first object, the last, or one of thousands in a
   shard. The csv/tsv row writer, by contrast, *does* terminate its own line with
   `\n`, because a flat row is self-contained.

The `os` is any `std::ostream`, so the same three functions serve a file, standard
output, or a shard file with no change — the writer picks the stream, these
functions just fill it.

---

## 8. Why this is its own file (host-only, CUDA-free)

READv2's heavy lifting — the bit-matrix pack, the mismatch kernels, the windowed
P0 accumulation — all lives in `src/device/cuda/`. This header is the opposite end
of that pipeline: the last, purely-textual step after the GPU is done. Keeping the
serializers in their own small app-layer translation unit, free of any device
header, means the output schema can be read, tested, and changed without touching
(or recompiling) any CUDA. It also lets the shard writer, the single-file writer,
and the standard-output path in `cmd_readv2.cpp` all share exactly one definition
of what a READv2 row looks like on disk — there is no second place a column could
be added, reordered, or formatted differently.

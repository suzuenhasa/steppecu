# `vcf_record.hpp` reference

## 1. Purpose

`src/io/vcf_record.hpp` is a small header-only kit of VCF text-parse helpers. Its
whole job is to pull the exact byte-substring you asked for out of a single VCF
record line — the value of a named INFO key, the index of a named FORMAT subfield,
the *n*-th colon-delimited piece of a sample column, a non-negative integer — and
nothing more.

It exists so that `vcf_reader.cpp` (the file that actually walks a VCF and turns it
into genotype calls) can stay thin and readable. Rather than re-implement delimiter
scanning inline five times, the reader reaches for `vd::info_field(...)`,
`vd::format_index(...)`, `vd::subfield(...)`, and `vd::parse_int(...)` and gets a
tested, field-exact answer each time. Everything here lives in the
`steppe::io::vcfdetail` namespace and is `inline`, so it links cleanly into the one
translation unit that uses it.

This is a pure host, io-leaf header: standard library only, no CUDA, no allocation
beyond what `split` needs for its result vector, no I/O of its own. It parses text
that someone else has already read off disk.

---

## 2. Why field-exact matching is the whole point

The load-bearing design rule is **match keys exactly, on the correct delimiter.**

A naive substring search would be a bug factory. Searching an INFO string for `DP=`
would happily match inside `MinDP=`, and treating a sample column as one flat string
would let a FORMAT `DP` lookup collide with `AD` or `GT`. The reference tooling
(bcftools' `%INFO/...` and `%FORMAT/...` accessors) gets field-boundary robustness
for free because it tokenizes properly; these helpers reproduce that discipline by
hand so steppe's VCF path agrees with the oracle on which byte belongs to which
field.

Concretely, each extractor knows its delimiter and matches a *whole token*, never a
substring:

- INFO is split on `;`, and within a token the key is everything up to the first
  `=` — matched with `==`, not `find`.
- FORMAT and sample columns are split on `:`, and a key or index selects a whole
  colon-delimited token.

So `MinDP=30;DP=5` returns `30` for key `MinDP` and `5` for key `DP`, never a
cross-contaminated match, and a FORMAT string `GT:AD:DP` gives `DP` index `2`, never
a hit on `AD`.

---

## 3. The four extractors

### `split(s, delim)` — the building block

Splits a `string_view` on a single delimiter into a vector of `string_view`s that
alias the original buffer (no copies of the text itself). A trailing delimiter
yields a final empty piece; an empty input yields a single empty piece — the loop
always pushes exactly one token past the last delimiter. `vcf_reader.cpp` uses this
to break a whole record line into its tab-delimited columns.

### `info_field(info, key)` → `optional<string_view>`

Looks up a key in a `;`-delimited INFO string where each token is either a bare flag
(`KEY`) or a key/value pair (`KEY=VALUE`).

- Returns the **value** for a matching `KEY=VALUE`.
- Returns an **empty `string_view`** for a matching bare flag (present, but no value).
- Returns **`std::nullopt`** when the key is absent, and short-circuits `"."` (the
  VCF "no INFO" sentinel) straight to `nullopt`.

The key is compared exactly against the substring before the first `=`, so `MinDP`
and `DP` are distinct even though one is a substring of the other. Note the
three-state result matters to callers: an empty value ("flag present") and `nullopt`
("key absent") are different answers, and the reader distinguishes them.

### `format_index(format, key)` → `int`

Returns the zero-based position of an exact FORMAT subfield key in a `:`-delimited
FORMAT string, or `-1` if the key isn't there. This index is meant to be fed
straight into `subfield` on the matching sample column — FORMAT declares the layout,
the sample column follows it slot-for-slot.

### `subfield(sample, idx)` → `string_view`

Returns the `idx`-th `:`-delimited piece of a sample column, or an **empty
`string_view`** when `idx` is out of range — including the `idx < 0` case, which it
guards up front. That empty-on-miss contract is what makes the common
`subfield(sample, format_index(format, "DP"))` idiom safe: if FORMAT has no `DP`,
`format_index` returns `-1`, `subfield` returns `""`, and `parse_int("")` returns
`nullopt` — a clean "no value" all the way through with no bounds check at the call
site.

---

## 4. `parse_int` and its deliberately narrow contract

`parse_int(s)` parses a non-negative integer and returns `optional<long long>`. It
returns `nullopt` for the empty string, for the VCF missing-value sentinel `"."`,
and for anything containing a non-digit.

Three properties are intentional and worth stating plainly:

- **Digits only, no sign, no whitespace.** It accepts `0-9` and nothing else — no
  leading `+`/`-`, no spaces, no thousands separators. VCF POS, END, DP, GQ, and a
  stripped autosome number are all non-negative integers, so this narrowness is a
  feature: a stray sign or space is a malformed field and correctly rejects to
  `nullopt` rather than being silently coerced.
- **No overflow guard.** The accumulation `v = v*10 + digit` does not check for
  `long long` overflow. In practice every field it parses (chromosome, position,
  depth, quality) is far inside `long long` range, so this is a conscious
  simplicity-for-the-domain choice, not an oversight — but a genuinely enormous digit
  run would wrap. Callers feed it real VCF numeric fields, where this cannot happen.
- **The `any` guard is belt-and-suspenders.** The loop tracks whether it saw at
  least one digit; combined with the empty/`"."` early-out, the only way to reach the
  end with `any == false` would be an empty string, which is already handled. It
  returns `nullopt` in that case regardless, so an empty scan never reports `0`.

---

## 5. How the reader wires these together

The helpers are designed as a pipeline, and `vcf_reader.cpp` uses them exactly that
way when it processes each record:

1. **CHROM/POS gating.** It strips an optional `chr` prefix and calls `parse_int` on
   the chromosome; a non-numeric result (X, Y, MT, contigs) parses to `nullopt` and
   the record is skipped as out of scope. POS goes through `parse_int` the same way.
2. **Ref-confidence blocks (`ALT == "."`).** It reads the interval end from
   `info_field(info, "END")` (falling back to POS when absent), and takes the block
   depth from `info_field(info, "MinDP")` if present, otherwise from the FORMAT `DP`
   subfield via `format_index` + `subfield` + `parse_int`. The INFO `MinDP` path is
   why exact key matching against `DP` matters.
3. **Explicit variants.** It pulls `GT`, `DP`, and `GQ` out of the sample column by
   looking each key's index up in FORMAT and reading the matching slot — the
   `format_index` → `subfield` → `parse_int` chain, with a missing subfield
   collapsing cleanly to "no value" as described in section 3.

Because every extractor returns a `string_view` aliasing the caller's line buffer
(or a plain `int`/`optional`), the reader does its per-record work with essentially
no allocation beyond `split`'s token vector — the record text is parsed in place.

---

## 6. Contracts and invariants at a glance

- **Views alias the source.** Every returned `string_view` points into the buffer
  passed in. The buffer must outlive the view — which it does, since the reader
  holds the record line for the duration of processing it. Nothing here copies text.
- **`"."` is the missing sentinel.** `info_field` and `parse_int` both treat a lone
  `"."` as "nothing here" rather than as a literal value.
- **Empty vs. absent is preserved where it matters.** `info_field` distinguishes an
  empty-value flag hit from a genuine miss (`string_view{}` vs `nullopt`); `subfield`
  returns empty for both "empty field" and "out of range" because the downstream
  `parse_int` treats both identically.
- **`-1` is the format-miss signal, and `subfield` is safe on it.** `format_index`'s
  `-1` for "not found" feeds `subfield`, whose `idx < 0` guard turns it into an empty
  view — so the composed idiom never indexes out of bounds.
- **No state, no side effects.** Every function is a pure, `[[nodiscard]]`
  transformation of its arguments. There is nothing to initialize, tear down, or get
  wrong across calls.

---

## 7. Edge cases the code handles on purpose

- **Empty input to `split`** yields one empty token, not zero — the loop always
  emits a final piece.
- **Trailing delimiters** (`a;b;`) produce a trailing empty token, matching how a
  VCF field with an empty final subfield should read.
- **A bare INFO flag** (`KEY` with no `=`) returns an empty value, correctly reported
  as "present but valueless."
- **Out-of-range or negative subfield index** returns empty rather than reading past
  the string.
- **Non-digit, empty, or `"."` integer fields** all reject to `nullopt`, so a
  malformed numeric column is a clean skip upstream, never a silent `0`.

# `cmd_ingest_concord.hpp` reference

## 1. Purpose

`src/app/cmd_ingest_concord.hpp` is the thin CLI shim behind
`steppe ingest-concord` — the host-only VCF-ingest concordance validator, the
Stage-1 gVCF block-correctness gate. Its whole job is to be the small seam between
the command line and the arithmetic: it declares the flag bag
(`IngestConcordArgs`) that `cli_parse.cpp` fills in, and it declares the one entry
point (`run_ingest_concord`) that turns that bag into a rendered report and a
process exit code.

The header itself is deliberately tiny — a struct and a function declaration. All
the real work lives one layer down in `ingest_concord.{hpp,cpp}`, which reads the
two report tables, joins them on rsID, and computes every match rate. The shim
never joins a table or counts a mismatch; it parses, calls, renders, and maps the
outcome to an exit code. Its implementation is the twelve-line body in
`cmd_ingest_concord.cpp`, and it mirrors its sibling shim
`cmd_readv2_concord.{hpp,cpp}` line for line.

There is no GPU here. No CUDA header, no `RunConfig`, no device, no precision knob —
this is a plain host tool that reads two text files and compares numbers. That
host-only isolation is the same rule the rest of the app target lives under, and it
is what lets this validator run anywhere, with or without a card in the machine.

---

## 2. The two tables it compares (and which is which)

The validator diffs **two per-site report TSVs** in the Stage-0 oracle schema:

| Flag | Role | Field |
|---|---|---|
| `--a` | **TEST** — steppe's own ingest report, the table under scrutiny. | `a_path` |
| `--b` | **RULER** — the Stage-0 oracle dosage TSV, the trusted reference. | `b_path` |

Both are **required** — `cli_parse.cpp` marks them so, and there is no default for
either. The A-vs-B naming is load-bearing and should not be flipped: A is the thing
being validated, B is the thing it is validated against. The join happens on rsID
over the oracle's valid-`pos38` rows (the reader can't reproduce the orchestrated
`no_lift` / `dup_rsid` lift-stage drops, which carry `pos38 == "None"`), and the
gate asserts exact per-site agreement of the tuple
`{call, dosage, source, drop_reason}`. All of that arithmetic is documented with
`ingest_concord.hpp`; the shim only names the two paths and hands them over.

---

## 3. The `IngestConcordArgs` bag

`IngestConcordArgs` is a plain value-holder — one field per flag, each carrying its
own default, no logic. `cli_parse.cpp` binds every `ingest-concord` option straight
into one of these fields, and `run_ingest_concord` reads them back out. The fields:

| Field | Flag | Default | Meaning |
|---|---|---|---|
| `a_path` | `--a` | — (required) | steppe's ingest report (the TEST table). |
| `b_path` | `--b` | — (required) | the Stage-0 oracle dosage TSV (the RULER table). |
| `overall_match_min` | `--overall-match-min` | `1.0` | PASS floor: overall exact-match fraction. |
| `refblock_match_min` | `--refblock-match-min` | `1.0` | PASS floor: ref-block hom-ref subset match. |
| `variant_match_min` | `--variant-match-min` | `1.0` | PASS floor: explicit-variant subset match. |
| `coverage_min` | `--coverage-min` | `1.0` | PASS floor: oracle-row coverage. |
| `format` | `--format` | `"text"` | Report format: `text` or `json`. |
| `out_path` | `--out` | — (empty) | Where to write; empty means standard output. |

The four `*_min` floors all default to **1.0** — that is the strict default: nothing
short of perfect agreement passes. The shim copies those four floors verbatim into an
`IngestThresholds` before calling the arithmetic, so the gate sees exactly the
numbers the user asked for; loosening a floor is opt-in per flag, never implicit.

---

## 4. What `run_ingest_concord` does, in order

`run_ingest_concord(args)` is the one entry point. It runs a short, strictly-ordered
pipeline, and every stage has its own exit lane (section 5):

1. **Validate `--format` first.** Before touching a file, it rejects any format that
   isn't exactly `"text"` or `"json"`, with a message naming the bad value. This is
   the one piece of input the shim validates itself rather than delegating — a cheap
   guard that fails fast before any I/O.
2. **Copy the floors into `IngestThresholds`.** The four `*_min` fields are moved into
   the threshold struct the arithmetic expects.
3. **Run the arithmetic.** `run_ingest_concordance(a_path, b_path, thr)` reads both
   tables, joins on rsID, computes every metric, and gates PASS/FAIL, returning an
   `IngestConcordResult` — a status plus the filled `IngestReport`.
4. **Branch on the status.** An I/O error or a bad-input error (a missing required
   column, a duplicate rsID) short-circuits to its own exit code before anything is
   rendered (section 5).
5. **Render the report.** On a clean `kOk`, `write_ingest_report` prints the report in
   the chosen format — to standard output when `out_path` is empty, or to the named
   file otherwise. Opening the output file is itself fallible: a `--out` that can't be
   written is an I/O error, reported and exited as such.
6. **Map PASS/FAIL to the exit code.** `report.pass` becomes success; a clean FAIL
   becomes the distinct literal `1` (section 5).

The division of labor is the whole point of the shim: steps 1, 5-partial, and 6 are
its own; steps 3 and 4 are the arithmetic's; and it never reaches inside the
`IngestReport` to recompute anything — it trusts the `pass` flag the gate already set.

---

## 5. The four-way exit-code map

The shim maps its outcome onto four distinct process exit codes, and the spread is
deliberate — a caller (a CI harness, an orchestration script) can tell *why* a run
ended by its code alone, without parsing the report:

| Exit | Constant | When |
|---|---|---|
| `0` | `kExitOk` | **PASS** — every floor met. |
| `1` | literal `1` | Ran cleanly, but concordance **FAIL** — a floor was missed. |
| `2` | `kExitInvalidConfig` | Bad args (`--format` neither text nor json) **or** bad input: a missing required column or a duplicate rsID (`kBadInput`). |
| `4` | `kExitIoError` | A table could not be read, or `--out` could not be written (`kIoError`). |

The one to notice is the split between `0` and `1`. A concordance FAIL is **not** an
error — the tool ran perfectly and simply found a disagreement, so it does not use an
error code. It uses the bare literal `1`, a lane of its own that sits *between*
"passed" (`0`) and "couldn't even run" (`2`/`4`). That keeps three genuinely
different meanings — "good", "measured, and it's bad", and "broke before it could
measure" — on three different codes. The `2` and `4` codes come from
`core/config/exit_code.hpp`; the `1` is written literally in the shim precisely
because it means something the shared enum doesn't have a name for.

---

## 6. Contracts and invariants

- **`--a` and `--b` are required and directional.** Neither has a default; both must
  be supplied. A-is-TEST, B-is-RULER is fixed — swapping them silently inverts the
  meaning of the report.
- **Format is validated before I/O.** An invalid `--format` fails at step 1 with no
  files touched, so a typo never costs a table read.
- **The floors pass through untouched.** Whatever the four `*_min` flags say is exactly
  what the gate enforces; the shim neither clamps nor rewrites them, and the strict
  `1.0` defaults mean an un-flagged run demands perfect agreement.
- **The shim owns no report logic.** Every count, rate, and the `pass` verdict come
  from `run_ingest_concordance`; the shim reads `report.pass` and renders — it never
  re-derives PASS from the individual metrics.
- **`[[nodiscard]]`.** `run_ingest_concord` returns the process exit code and is marked
  `[[nodiscard]]`, so a caller can't drop the outcome on the floor — the whole point of
  the tool is that number.
- **Host-only, forever.** No CUDA, no `RunConfig`, no device. This translation unit
  must stay GPU-free, the same rule its sibling shims live under.

---

## 7. Edge cases

- **`--out` points somewhere unwritable.** The report is fully computed, but the
  `ofstream` fails to open, so the run exits `4` (I/O error) with a `cannot write --out`
  message rather than pretending it succeeded. The concordance result is discarded in
  that case — an un-writable destination is treated as a failure to deliver, not a pass.
- **A clean FAIL.** Both tables read fine, the join succeeds, and a floor is missed. The
  report still renders in full (so the user sees *what* disagreed), and only then does the
  shim return `1`. A FAIL is always reported before it is signalled.
- **Empty `out_path`.** The empty string is the sentinel for "standard output" — not a
  filename — so a run with no `--out` streams the report to stdout.
- **Bad input vs I/O error.** A file that won't open is `kIoError` (`4`); a file that opens
  but is malformed — a missing required column, a duplicate rsID — is `kBadInput` (`2`).
  The shim keeps those two on different codes because they call for different fixes: one is
  a path/permissions problem, the other is a schema problem.

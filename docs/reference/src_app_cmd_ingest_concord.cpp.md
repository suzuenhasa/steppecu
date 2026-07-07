# `cmd_ingest_concord.cpp` reference

## 1. Purpose

`src/app/cmd_ingest_concord.cpp` is the thin command-line shim behind
`steppe ingest-concord`. It does the small, unglamorous job of turning parsed
flags into a call to the real concordance arithmetic, sending the result
somewhere, and turning that result into a process exit code. Nothing else.

`ingest-concord` is the VCF-ingestion Stage-1 correctness gate. It compares two
per-site report tables in the Stage-0 oracle schema: steppe's own ingest report
(the `--a` **TEST** table) against the trusted Stage-0 oracle dosage TSV (the
`--b` **RULER** table). It joins them on rsID and asserts that steppe called the
same thing the oracle did at every site. The actual join, the field-by-field
comparison, the subset match rates, and the PASS/FAIL decision all live next door
in `ingest_concord.cpp` (documented in `ingest_concord.hpp`). This file is only
the front door to that.

Like its sibling `cmd_readv2_concord.cpp`, and by the same deliberate design, this
translation unit touches **no GPU**. There is no CUDA header, no `RunConfig`, no
`build_config` precedence merge. It is host-only by construction — this is a
validator that reads two text tables and diffs them, so there is simply nothing for
a device to do.

---

## 2. What the shim actually does

`run_ingest_concord(args)` runs a fixed four-step sequence:

1. **Validate the one flag it owns** — `--format` must be `text` or `json`
   (section 3).
2. **Repack the PASS floors** — copy the four `--*-match-min` / `--coverage-min`
   values out of `IngestConcordArgs` and into the `IngestThresholds` struct the
   arithmetic expects (section 4).
3. **Call the arithmetic** — `run_ingest_concordance(a_path, b_path, thr)` does the
   real work and hands back an `IngestConcordResult`. The shim inspects its status
   and maps I/O and bad-input failures to exit codes (section 5).
4. **Render and exit** — write the report to `--out` or standard output, then
   return an exit code that encodes PASS vs FAIL (sections 6 and 7).

That is the whole file. It holds no state, launches nothing, and computes no metric
of its own — every number in the report comes from `run_ingest_concordance`.

---

## 3. The `--format` gate

`--format` is validated up front, before anything is read. Only `text` and `json`
are accepted; anything else prints
`--format must be 'text' or 'json' (got '<x>')` to standard error and returns the
invalid-configuration exit (`kExitInvalidConfig`, 2).

This check happens *first*, before the tables are opened, so a typo in `--format`
fails fast and cheap rather than after the (potentially slow) table read. The
validated string is then passed straight through to `write_ingest_report`, which is
the only other place it's consulted.

---

## 4. Repacking the thresholds

`IngestConcordArgs` (the flag-holder the CLI binds into) and `IngestThresholds`
(what the arithmetic consumes) carry the same four PASS floors under the same
names. The shim copies them across one field at a time:

| Arg field | Threshold field | What it floors |
|---|---|---|
| `overall_match_min` | `overall_match_min` | overall exact-match fraction |
| `refblock_match_min` | `refblock_match_min` | ref-block hom-ref subset match |
| `variant_match_min` | `variant_match_min` | explicit-variant subset match |
| `coverage_min` | `coverage_min` | oracle-row coverage |

All four default to `1.0` — the gate demands *perfect* agreement unless the user
deliberately loosens a floor. The reason there are two structs at all is the
host-only isolation rule from section 1: `IngestConcordArgs` is a CLI concern that
also carries `format` and `out_path`, while `IngestThresholds` is the pure-arithmetic
input that the parse-agnostic engine takes. Keeping them separate lets the engine be
tested without any CLI dependency. The same `thr` value is also handed to
`write_ingest_report`, so the rendered report can show each metric against the floor
it was judged by.

---

## 5. Mapping engine failures to exit codes

`run_ingest_concordance` returns a status alongside its report. The shim branches on
it before it renders anything:

- **`kIoError`** — a table file could not be opened or read. The shim prints the
  engine's own `error` string (prefixed `steppe ingest-concord:`) to standard error
  and returns `kExitIoError` (4).
- **`kBadInput`** — the table was readable but malformed: a required column is
  missing, or a duplicate rsID broke the join. Same error print, but returns
  `kExitInvalidConfig` (2), because a bad table is a configuration-class problem, not
  an I/O one.
- **`kOk`** — the diff computed cleanly. Fall through to render; the PASS/FAIL verdict
  now lives in `report.pass` (section 7).

The distinction matters: an unreadable file (4) and a readable-but-wrong file (2) are
different failure classes a caller may want to tell apart, so they get different
codes rather than being collapsed into one "something went wrong".

---

## 6. Where the report goes

Rendering is delegated to `write_ingest_report(os, report, thr, format)`. The shim's
only decision is the destination `os`:

- **No `--out`** — write to `std::cout`.
- **`--out <path>`** — open the file with `std::ios::trunc` (overwrite, don't append)
  and write there. If the file can't be opened, print
  `cannot write --out <path>` to standard error and return `kExitIoError` (4).

The `trunc` is deliberate: a re-run of the gate replaces its previous report rather
than accumulating stale runs on the end of the file. The same `report`, `thr`, and
`format` are passed on both branches, so the destination never changes *what* is
written — only *where*.

---

## 7. The four-way exit code

This command has a richer exit contract than most steppe subcommands because a
"clean run that reports a FAIL" is a distinct, meaningful outcome that a CI harness
needs to tell apart from a crash:

| Exit | Meaning |
|---|---|
| `0` (`kExitOk`) | PASS — every metric met its floor. |
| `1` (literal) | The gate ran cleanly but concordance **FAILED** — at least one floor was missed. |
| `2` (`kExitInvalidConfig`) | Bad args (`--format`) or bad table (missing column / duplicate rsID). |
| `4` (`kExitIoError`) | A table could not be read, or `--out` could not be written. |

The load-bearing subtlety is the final line: `return res.report.pass ? kExitOk : 1;`.
A concordance FAIL is a **literal `1`**, deliberately *not* drawn from
`exit_code.hpp`. That is what carves out its own lane. Exit 1 says "the tool worked,
and the answer is: they don't match" — an expected, well-defined result of running a
gate — whereas 2 and 4 mean "the tool couldn't even get to an answer". A CI job can
therefore treat 1 as a real concordance regression to look at, and 2/4 as a broken
invocation to fix, without conflating the two.

This mirrors `cmd_readv2_concord.cpp` exactly, so the READv2 gate and the VCF-ingest
gate speak the same exit-code language.

---

## 8. Contracts and edge cases

- **`--a` and `--b` are required.** The shim itself does not re-check that they're
  non-empty; a missing or unreadable path surfaces as the engine's `kIoError` (exit
  4) when it tries to open it. The CLI layer is responsible for marking them
  required.
- **Validation order is fixed.** `--format` is checked before the tables are touched,
  so a format typo never pays the cost of a table read.
- **The shim owns no metrics.** Every fraction, count, and the PASS bit come from
  `run_ingest_concordance`. If a number looks wrong, this file is not where it was
  computed — see `ingest_concord.hpp`.
- **Host-only, enforced.** No CUDA, no `RunConfig`, no device. This is guarded by the
  same build-time isolation that keeps the rest of the app's host shims GPU-free.
- **Rendering is total.** On any `kOk` result the report is always written (to stdout
  or `--out`) regardless of PASS or FAIL — a FAIL still produces a full, inspectable
  report, and only *then* returns exit 1.

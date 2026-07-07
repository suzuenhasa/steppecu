# `cmd_readv2_concord.cpp` reference

## 1. Purpose

`src/app/cmd_readv2_concord.cpp` is the thin command-line shim behind
`steppe readv2-concord`. It does the small, unglamorous glue work that sits
between the flags a user typed and the arithmetic that actually diffs two READv2
tables: it validates the one flag it owns (`--format`), copies the parsed flags
into a `ConcordanceThresholds`, calls `run_concordance(...)`, routes the rendered
report to standard output or a file, and turns the outcome into a process exit
code.

That is the whole file. The real work — reading the two tables, keying them on
the unordered sample pair, building the degree confusion matrix, computing the
P0_norm concordance, and gating PASS/FAIL — lives next door in
`readv2_concord.{hpp,cpp}`, which is deliberately parse-agnostic and directly
unit-testable. This shim exists so that module never has to know a thing about
CLI11, `argv`, exit codes, or where its output goes.

It touches no GPU. There is no CUDA header here, no `RunConfig`, no
`build_config` precedence merge — the top comment says as much: "parse flags ->
call the arithmetic -> render -> exit code. No CUDA, no device, no RunConfig."
This is the Phase-0 "ruler": a host-only validator that measures steppe's READv2
output against a reference tool's, and it stays firmly on the host side of the
build's GPU/host split.

The flags themselves are declared over in `cli_parse.cpp` (the `readv2-concord`
subcommand block), which binds them into a `Readv2ConcordArgs` and, on its
callback, calls the single entry point this file provides:
`run_readv2_concord(args)`.

---

## 2. The `Readv2ConcordArgs` contract

The shim's input is a `Readv2ConcordArgs` (declared in the paired
`cmd_readv2_concord.hpp`). It is a plain value-holder — one field per flag, each
with its default baked in as a member initializer so a bare invocation reproduces
the documented defaults:

| Field | Flag | Default | Meaning |
|---|---|---|---|
| `a_path` | `--a` | — (required) | steppe's READv2 output — the **TEST** (A) table. |
| `b_path` | `--b` | — (required) | The reference tool's output — the **ORACLE** (B) table. |
| `p0_atol` | `--p0-atol` | `5e-3` | P0_norm absolute tolerance. |
| `p0_rtol` | `--p0-rtol` | `1e-2` | P0_norm relative tolerance. |
| `degree_agreement_min` | `--degree-agreement-min` | `0.95` | PASS floor: the degree-match fraction. |
| `p0_within_tol_min` | `--p0-within-tol-min` | `0.90` | PASS floor: the P0_norm within-tolerance fraction. |
| `coverage_min` | `--coverage-min` | `1.0` | PASS floor: oracle-pair coverage. |
| `format` | `--format` | `"text"` | Report format, `text` or `json`. |
| `out_path` | `--out` | empty | Destination file; empty means standard output. |

The struct is owned at `run_cli` scope so it outlives CLI11's parse — CLI11 binds
each flag to a field by reference, so the storage has to stay alive through
parsing. `A` (`--a`) is the table under test and `B` (`--b`) is the oracle; that
orientation is load-bearing everywhere downstream (the confusion matrix rows are
oracle truth, the columns are steppe's test), so the shim never swaps them.

The five threshold-ish fields split into two roles that this shim keeps together
in one `ConcordanceThresholds` but that the arithmetic uses differently: the two
tolerances (`p0_atol`, `p0_rtol`) decide whether an individual pair's P0_norm
counts as "within tol", while the three `*_min` floors are the PASS gates the
final verdict is checked against.

---

## 3. The one thing the shim validates itself: `--format`

Almost all validation happens inside the arithmetic (missing columns, bad degree
tokens, duplicate keys, unreadable files). The shim owns exactly one check up
front: `--format` must be `"text"` or `"json"`. Anything else is rejected before
any file is touched, with a message naming the bad value and an
invalid-configuration exit.

The reason this check lives here rather than in the arithmetic is that `format`
is purely a rendering choice — it never reaches `run_concordance`, only
`write_report`. Catching it early means a typo'd `--format xml` fails instantly
and cleanly instead of getting silently treated as one of the two real formats.
(`write_report` itself treats any non-`"json"` string as text, so without this
gate a typo would quietly fall through to the text renderer — the gate is what
turns that silent fallthrough into an explicit error.)

---

## 4. The four-lane exit-code design

This is the most deliberate decision in the file. The tool has **four distinct
process outcomes**, and they are kept on separate exit-code lanes on purpose so
that a caller (a CI script, a `make` gate) can tell them apart from the exit code
alone:

| Exit | Lane | When |
|---|---|---|
| `0` (`kExitOk`) | **PASS** | Ran cleanly and every floor was met (`res.report.pass`). |
| `1` (literal) | **clean concordance FAIL** | Ran cleanly, but a floor was not met. |
| `2` (`kExitInvalidConfig`) | **bad input** | A bad `--format`, or the arithmetic returned `kBadInput` (missing required column, a degree token outside the enum, or a duplicate pair key). |
| `4` (`kExitIoError`) | **I/O error** | The arithmetic returned `kIoError` (a table could not be opened/read), or the `--out` file could not be opened for writing. |

The subtle, important part is lane `1`. A concordance FAIL — steppe's output
genuinely disagreeing with the oracle — is a *successful run that produced a
"no"*, and it must never be confused with a *broken invocation*. So it gets its
own literal `1`, held apart from the `kExitInvalidConfig = 2` bad-input lane.
The in-code comment says it directly: "a clean concordance FAIL is a DISTINCT
lane (literal 1) so a disagreement is never confused with the
kExitInvalidConfig=2 bad-input lane." If FAIL shared an exit code with
"malformed table", a CI job could never distinguish "steppe's numbers drifted"
from "someone handed me a broken file" — and those demand very different
responses.

The mapping from the arithmetic's `ConcordStatus` to an exit code is a small,
explicit switch by hand rather than a shared helper: `kIoError → 4`,
`kBadInput → 2`, and `kOk` falls through to the PASS/FAIL split at the end. Each
error also reprints the arithmetic's own human message to standard error, prefixed
with `steppe readv2-concord:`, so the reason travels with the exit code.

---

## 5. Output routing: standard output vs `--out`

Rendering is a single call to `write_report(os, res.report, thr, args.format)`;
the only question the shim answers is *which `os`*.

- When `--out` is empty, it writes to `std::cout`.
- When `--out` is set, it opens that path with `std::ios::trunc` (a fresh
  overwrite, not an append) and renders into the file stream. If the file can't
  be opened, that is an I/O error (lane `4`) reported before any rendering.

Note the ordering: the report is only rendered **after** `run_concordance`
succeeds. An I/O or bad-input failure from the arithmetic returns early, so a
failed run never emits a half-written report to a file or to standard output. The
`thr` passed to `write_report` is the same `ConcordanceThresholds` used to
compute the verdict, so the "(min …)" annotations the human block prints always
reflect the exact floors that gated this run.

---

## 6. Contracts and invariants

- **Host-only, GPU-free.** No CUDA, no device, no `RunConfig`. This translation
  unit must stay on the host side of the build's grep-enforced GPU/host split,
  the same rule that guards the rest of the app target.
- **Single entry point.** `run_readv2_concord(const Readv2ConcordArgs&)` is the
  whole public surface (declared `[[nodiscard]]`), dispatched straight from the
  `readv2-concord` callback in `cli_parse.cpp`. It takes a plain args struct, not
  a merged config, because there is no precedence chain to merge.
- **A/B orientation is fixed.** `--a` is always the test table and `--b` always
  the oracle; the shim passes them to `run_concordance(a_path, b_path, thr)` in
  that order and never swaps them.
- **The four exit lanes are stable.** `0` / `1` / `2` / `4` mean PASS / clean
  FAIL / bad input / I/O error respectively, and their separation is a promised
  interface, not an accident — the header banner documents the same four values.
- **`--format` is validated here; everything else downstream.** The shim rejects
  a bad format token itself; all table-shape validation is the arithmetic's job.

---

## 7. Edge cases

- **A concordance FAIL is a clean run.** It exits `1` with a fully rendered
  report — not an error message — so a caller reads the verdict from the report
  and the lane from the exit code. This is the case the four-lane design exists
  to protect.
- **Bad `--format`.** Rejected before any file I/O, exit `2`, with a message
  naming the offending value.
- **Unwritable `--out`.** Caught when the `std::ofstream` fails to open, exit
  `4`, before any rendering — the report is never partially emitted.
- **Empty or malformed tables.** Surfaced by the arithmetic as `kIoError`
  (empty/unopenable) or `kBadInput` (missing column, out-of-enum degree token,
  duplicate pair key) and mapped to lanes `4` and `2`; the shim itself does not
  inspect table contents.
- **An empty intersection is never a PASS.** That verdict is decided in the
  arithmetic (degree agreement and within-tol fraction both fall to zero when no
  pairs are common), and the shim simply reports whatever `res.report.pass`
  carries — so a run where the two tables share no pairs exits `1`, not `0`.

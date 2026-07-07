# `cmd_readv2_concord.hpp` reference

## 1. Purpose

`src/app/cmd_readv2_concord.hpp` is the thin command-line shim for
`steppe readv2-concord` — the host-only READv2 concordance validator, the Phase-0
"ruler." Its whole job is small and deliberately so: declare the struct that holds
the flags a user typed, and declare the single entry point that runs the check and
hands back a process exit code.

The actual work — reading two READv2 output tables, keying them on the sample pair,
building the degree confusion matrix, and diffing the `P0_norm` numbers — lives next
door in `readv2_concord.{hpp,cpp}`, which is parse-agnostic and directly
unit-testable. This header is only the seam between CLI11's parsed flags and that
arithmetic. Keeping the two apart is the point: the math can be tested without ever
touching a command line, and the command line can be wired up without knowing how a
confusion matrix is built.

There is no GPU here. No CUDA header, no `RunConfig`, no device, no `build_config`
precedence merge. Unlike the fit and statistic subcommands, `readv2-concord`'s
callback in `cli_parse.cpp` builds no merged configuration at all — it fills a plain
`Readv2ConcordArgs` and calls `run_readv2_concord` directly. The concordance check is
pure on-disk arithmetic over two text tables; there is nothing for a device to do.

---

## 2. `Readv2ConcordArgs` — the flag bundle

`Readv2ConcordArgs` is a plain value-holder, one field per flag, with member
initializers that *are* the documented defaults. It is declared owned at `run_cli`
scope so it outlives CLI11's parse: CLI11 binds each flag to a field of this struct
by reference, so the storage has to stay alive across the whole parse, exactly like
the per-subcommand `CliArgs` structs the rest of the app uses.

| Field | Flag | Default | Meaning |
|---|---|---|---|
| `a_path` | `--a` | — (required) | steppe's own READv2 output — the **TEST** table (the "A" side). |
| `b_path` | `--b` | — (required) | The reference tool's output — the **ORACLE** table (the "B" side, treated as truth). |
| `p0_atol` | `--p0-atol` | `5e-3` | Absolute tolerance for a per-pair `P0_norm` comparison. |
| `p0_rtol` | `--p0-rtol` | `1e-2` | Relative tolerance for a per-pair `P0_norm` comparison. |
| `degree_agreement_min` | `--degree-agreement-min` | `0.95` | PASS floor: the fraction of common pairs whose relatedness `degree` token matches. |
| `p0_within_tol_min` | `--p0-within-tol-min` | `0.90` | PASS floor: the fraction of common pairs whose `P0_norm` lands within tolerance. |
| `coverage_min` | `--coverage-min` | `1.0` | PASS floor: the fraction of oracle pairs steppe also emitted. Default `1.0` means "miss nothing." |
| `format` | `--format` | `"text"` | Report format: `text` or `json`. |
| `out_path` | `--out` | — (empty) | Where to write the report; empty means standard output. |

`--a` and `--b` are the only required flags; everything else has a working default,
so the simplest possible invocation is `steppe readv2-concord --a test.tsv --b
oracle.tsv`. The three floors and two tolerances are exposed so a caller can loosen
or tighten the PASS bar without recompiling — but the defaults are the frozen,
committed acceptance bar, and they live here as the member initializers.

The `A` = test / `B` = oracle convention is load-bearing and shows up again in the
report's confusion matrix, whose rows are the oracle's truth and whose columns are
steppe's guess. Getting `--a` and `--b` backwards doesn't error — it silently
transposes the confusion matrix and swaps which side "coverage" is measured against —
so the mnemonic (A is the thing under test) is worth keeping straight.

---

## 3. `run_readv2_concord` — the entry point and its contract

```cpp
[[nodiscard]] int run_readv2_concord(const Readv2ConcordArgs& args);
```

This is the whole public surface of the shim. It takes the filled arg bundle and
returns the process exit code, and it is `[[nodiscard]]` because that returned code
*is* the result — the CLI11 callback records it as the command's exit status and
throwing it away would silently mask a FAIL.

Its body (in the `.cpp`) is a short, linear pipeline with no branches into GPU code:

1. **Validate `--format`.** Anything other than `text` or `json` is rejected up
   front with a bad-args exit — before any file is opened, so a typo'd format never
   half-runs.
2. **Copy the args into a `ConcordanceThresholds`.** The five tunable numbers
   (`p0_atol`, `p0_rtol`, and the three floors) are lifted straight across into the
   threshold struct the arithmetic layer expects.
3. **Run the check.** `run_concordance(a_path, b_path, thr)` does all the real work
   and returns a `ConcordResult` carrying a status lane plus the report.
4. **Map the status to an exit code** (section 4).
5. **Render** the report to `--out` (or standard output) in the chosen format, then
   return `0` for PASS or `1` for a clean concordance FAIL.

The shim never opens the input tables itself and never parses a row — that is all
inside `run_concordance`. Its only I/O responsibility is the *output* side: opening
`--out` for writing (and failing with an I/O exit if it can't) or falling back to
standard output.

---

## 4. Exit codes — four distinct lanes

The single most important design decision this shim encodes is that **a clean
"the numbers disagree" FAIL is a different outcome from "the input was broken."** A
script driving this validator needs to tell those apart: a FAIL means steppe's
READv2 output drifted from the oracle and someone should look; a bad-input error
means the check never really ran. Collapsing them into one non-zero code would hide
that distinction.

So the shim maps to four separate codes:

| Code | Meaning |
|---|---|
| `0` (`kExitOk`) | **PASS** — the check ran and every floor was met. |
| `1` (literal) | **FAIL** — the check ran cleanly but at least one floor was not met. This is a deliberate literal `1`, chosen so it can't collide with the `kExitInvalidConfig = 2` bad-args lane below. |
| `2` (`kExitInvalidConfig`) | **Bad input** — a bad `--format`, a missing required column, a degree token outside the four-value enum, or a duplicate pair key in a table. The check could not be trusted, so it isn't reported as either PASS or FAIL. |
| `4` (`kExitIoError`) | **I/O error** — a table file could not be opened or read, or `--out` could not be written. |

The mapping is driven by the `ConcordStatus` lane on the returned `ConcordResult`:
`kIoError` → 4, `kBadInput` → 2, and `kOk` defers to `report.pass` to choose between
0 and 1. That "the FAIL lane is literal `1`, distinct from the `2` bad-input lane"
choice is the invariant to preserve — the header banner spells it out precisely
because it's the kind of thing an innocent-looking refactor could quietly break.

---

## 5. What this header does *not* do

A few non-responsibilities are worth stating plainly, because they're where the
boundaries live:

- **It does not classify anything.** The validator only *diffs* the `degree` strings
  and `P0_norm` values the two tables already carry; it never computes a relatedness
  degree itself. That work belongs to the READv2 emit path, not the ruler.
- **It does not know the table schema.** The frozen READv2 column layout
  (`sampleA sampleB n_windows n_overlap_sites P0_mean P0_norm degree z`), the
  four-token degree enum (`identical | first | second | unrelated`), the
  header-driven column lookup, and the unordered-pair keying all live in
  `readv2_concord.hpp`. This shim just forwards two paths and five numbers.
- **It does not merge configuration.** No config file, no environment variables, no
  precedence chain — the arg bundle is exactly what the user typed, nothing layered
  underneath. The concordance check has no device, tier, or precision knobs for a
  merge to resolve.

The collaborator to read next, for anyone tracing how a PASS/FAIL is actually
decided, is `docs/reference/src_app_readv2_concord.hpp.md` (the arithmetic and the
report format). This file is only the doorway.

Review of `src/app/result_emit.cpp`

I read through this carefully. This is **not slop** — it's clearly written by someone who understands the serialization contract and cares about byte-for-byte golden-file compatibility. But a senior C++ developer would flag several things before calling it production-grade.

## What's genuinely good

- **Consistent C++20 idiom throughout.** Everything goes through `std::ostream`, uses `std::span`, `std::string`, and an anonymous namespace for file-local helpers. No `printf`, no `FILE*`, no raw pointers, no CUDA leaking into the app layer — the file header comment even explicitly says "NO CUDA header" (line 4).
- **IEEE-754 exact round-trip precision is documented and correct.** `fmt_double` and `json_double` both set `precision(17)` (lines 28, 38), which is the right value for bit-exact double round-trip. The NaN → `NA` / `null` split between CSV and JSON (lines 26, 36) is thoughtful and matches the schema comments.
- **DRYed-up shared primitives.** The rankdrop CSV body and JSON array emitters were clearly extracted from copy-pasted lambdas, and the file now has one `emit_rankdrop_csv` (line 142), one set of `emit_*_arr` helpers (lines 120-135), and one `label_at` (line 471). That's the kind of cleanup that shows competence.
- **Defensive quoting.** `csv_quote` (line 89) escapes embedded quotes, and `json_quote` (line 100) handles the minimal JSON escape set. For internal genomics labels this is probably sufficient.
- **Clear domain mapping.** `status_str` and `precision_str` use switches (lines 45-55, 61-68), which is the right idiom for enum-to-string and plays nicely with `-Wswitch` if the enum grows.
- **The rotation feasibility fallback is careful.** `rotation_feasible` (line 284) prefers the engine's own popdrop decision and falls back to the canonical weights-in-[0,1] screen, with a comment explaining why (lines 278-283).

## What a senior developer would flag

**Parallel-array length assumptions with no validation.** `emit_rankdrop_csv` iterates over `r.rankdrop_f4rank.size()` (line 144) and blindly indexes into `rankdrop_dof`, `rankdrop_chisq`, `rankdrop_p`, etc. The same pattern appears in `emit_f4_csv` (line 488: `r.est.size()` drives `r.se[k]`, `r.z[k]`, `r.p[k]`), `emit_f3_csv`, and `emit_f4ratio_csv`. If the engine ever produces mismatched vectors, this is UB or silently truncated output. A senior dev would want either a contract assertion at the top of each emitter or a structured result type that enforces equal lengths.

**Stream error handling is absent.** Every `os << ...` chain assumes the stream stays valid. There's no `if (!os) return Status::IoError;` or exception policy. If the disk fills up or the pipe breaks, the user gets a silently truncated file. For a CLI output path this matters.

**The `bool last` comma pattern in `emit_*_arr` is clunky.** Lines 120-135 pass a `last` flag to decide whether to emit `,\n` or `\n`. It works, but it's brittle: reorder fields and you have to update all the `true/false` flags. A small helper that joins a range with commas, or tracking the index, would be more robust.

**Magic numbers and duplicated literals.** `17` appears twice for precision (lines 28, 38). `INT_MIN` is the documented dofdiff sentinel (line 84), but it's still a magic sentinel value. And `"    "` / `"  "` indentation strings are hardcoded in dozens of places — fine for a stable schema, but annoying to change.

**JSON/CSV built by hand.** The JSON emitters manually concatenate braces, colons, and commas (e.g., `emit_json` lines 205-274). With `json_quote` and the array helpers it's *probably* correct, but it's the kind of thing that breaks the moment someone adds a field with an unexpected control character or forgets a comma. For a growing CLI project, this is technical debt. A real JSON library, or at least a tiny RAII object/array builder, would be safer.

**Inconsistent guarding of optional arrays.** In `emit_csv`, `have_se` is defined as `!r.se.empty()` (line 156), but then the code separately checks `i < r.se.size()` and `i < r.z.size()` (lines 166-167). In `emit_qpwave_csv` the per-rank arrays are guarded inline with ternaries (lines 405-407) but `rank_chisq` is assumed valid. The policy is "trust the engine," which is fine, but the inconsistency makes the contract harder to read.

**Long, commit-message-style comments.** Some comments are useful (the rotation `f4rank` naming rationale, lines 303-308 and 368-371), but others just restate the code in more words. Line 565-564 is a 90-character comment explaining that an f4-ratio emitter emits f4-ratio output. A senior dev would trim several of these.

**One-liner loop style.** `join_left` crams its entire loop onto one line (line 293). It works, but it's the only place in the file that does this and it reads like a leftover from a quick refactor.

## The "slop" test

**Not slop.** Slop is unexplained magic numbers, copy-pasted code with stale comments, no error handling at all, and obviously wrong algorithms that happen to pass. This file has none of that. The helpers are shared, the NaN/NA/null semantics are consistent, and the comments — verbose as they are — mostly explain real domain decisions (golden-file compatibility, AT2 schema mapping). The manual JSON/CSV construction is the closest thing to slop-adjacent, but it's at least carefully done and centrally factored.

## What it actually looks like

This looks like **competent serialization-layer code written by a domain expert who values test fidelity over architectural elegance.** The author clearly spent time making the output byte-match reference fixtures, and the DRY cleanup of the rankdrop/array helpers shows they can refactor. At the same time, the hand-rolled JSON/CSV, the missing stream error handling, and the implicit parallel-array contracts give it a research-code-y feel — correct under happy-path assumptions, but not hardened against malformed input or I/O failure.

A senior C++ reviewer would say: "Solid and well-organized, but I want length checks and stream status handling before this goes near user data." A senior architect would say: "Start thinking about a real output abstraction before you add a sixth format."

**Verdict:** B. Ship after adding parallel-array length validation and stream error checks; consider a small JSON/CSV builder before the schema grows.
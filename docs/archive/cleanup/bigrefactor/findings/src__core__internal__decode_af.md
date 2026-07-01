# Review findings — src__core__internal__decode_af

Files: /home/suzunik/steppe/src/core/internal/decode_af.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

- [5.3][MED] decode_af.hpp:58,67-68 — the 2-bit packing convention is single-homed here but DUPLICATED across the core/io seam: `kMissingGenotypeCode = 3` mirrors `io::kMissingCode` (line 53-57) and the MSB-first bit-extract `(3 - (k & 3)) * 2` / `& 0x3u` (line 67-68) mirrors `io::code_in_byte` (line 62-64). The header documents this is deliberate (core must not depend on the io leaf, per architecture.md §4) and that an equivalence test pins them, so it is not a defect — but drift between the two copies of the missing sentinel or the bit order would be a silent decode-correctness bug. Suggested: leave as-is given the layering constraint, but ensure the cross-leaf equivalence test (line 56-57/63-64) is kept green in CI so the two copies cannot drift.
- [5.1][LOW] decode_af.hpp:67 — the bit-extract literals `(3 - (k & 3)) * 2` encode "4 codes per byte (`& 3` = mod 4), 2 bits each, MSB-first" as bare numerics; the `4` (codes/byte) and `2` (bits/code) are implicit. Self-documented by the comment "// 6, 4, 2, 0 for k%4 = 0,1,2,3" so readability is fine. Suggested: optional — name `kCodesPerByte = 4` / `kBitsPerCode = 2` to make the packing geometry explicit and shared with line 68's `0x3u` mask, but the inline comment already covers intent.

## Group 6 — Naming

- [6.3][LOW] decode_af.hpp:88-89,130 — the SAME two accumulators are typed two different ways in adjacent functions in one file: `accumulate_genotype` takes `std::int64_t& ac, std::int64_t& an` while `finalize_af` takes `long ac, long an`. The line-86 comment documents these are equal on the LP64 target (`std::int64_t == long`), so it is not a correctness bug, but mixing the fixed-width spelling and the native-`long` spelling for the same conceptual quantities within one file is an internal-convention inconsistency. Suggested: spell both as `std::int64_t` (or both as `long`) for the AC/AN accumulators across the two functions.

## Group 7 — Duplication

- [7.4][LOW] decode_af.hpp:137-141 — the `else` branch of `finalize_af` re-assigns `r.n = 0.0; r.q = 0.0; r.v = 0.0;`, which are exactly the in-class default member initializers `AfResult` already provides (lines 100-102, `q = 0.0`, `n = 0.0`, `v = 0.0`). For a masked-out cell `r` is already fully-defaulted, so the entire else block is redundant boilerplate folding the same constants twice. Not a correctness issue (the values are identical). Suggested: drop the else branch entirely — `AfResult r;` already yields the `{0,0,0}` masked result, so only the `if (an > 0 && ploidy > 0)` arm needs to assign.

## Group 8 — Comments

- [8.2][LOW] decode_af.hpp:102 — the `AfResult::v` member doc says "validity mask: 1.0 if an > 0, else 0.0", but the actual validity condition implemented in `finalize_af` (line 132: `an > 0 && ploidy > 0`) — and documented elsewhere in this same file (header summary line 28 `V = (AN > 0 && ploidy > 0)`, and the detailed B10/X-11 fail-soft note at lines 116-129) — also requires `ploidy > 0`. The member doc is stale/incomplete relative to the ploidy-fold guard: it omits the non-positive-ploidy masking that is the whole point of that guard, so a reader of the struct alone would think a positive `an` always yields `v == 1.0`. Suggested: update the line-102 doc to "1.0 if an > 0 && ploidy > 0, else 0.0" to match the code and the rest of the file.

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.


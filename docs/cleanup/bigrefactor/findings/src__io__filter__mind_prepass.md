# Review findings — src__io__filter__mind_prepass

Files: /home/suzunik/steppe/src/io/filter/mind_prepass.cpp, /home/suzunik/steppe/src/io/filter/mind_prepass.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.4][LOW] src/io/filter/mind_prepass.cpp:64 — redundant write loop: `out.missing_frac` was already set to 0.0 for all n_ind elements via `assign(n_ind, 0.0)` at line 23, so this loop re-writes the same value and is a pure no-op. The code's own comment (lines 61-63) explicitly states it is "a no-op kept only as an explicit statement of the convention" — i.e. intentional self-documentation, not an accidental dead store. Suggested: optional — drop the loop and let the lead comment carry the convention, or keep as documented intent (lowest priority).

## Group 5 — Hardcoded values / magic numbers

- [5.1][LOW] src/io/filter/mind_prepass.cpp:30 — the `< 1.0` "filter inactive / --mind not requested" sentinel is a bare literal. The same `1.0` boundary is the default in config.hpp:386 (`mind_max_missing = 1.0`), is re-spelled in this TU's header docs (mind_prepass.hpp:4, :56), and the analogous "keep-all" sentinel is duplicated across filter_plan.hpp:51 and filter_decision.hpp:66. It carries real meaning (the max possible missing fraction = "never drop"), so a named constant would document intent and prevent drift across these sites. Suggested: introduce a named constant (e.g. `kMindFilterInactiveThreshold = 1.0`) shared with the config default rather than re-spelling `1.0` per site. Low priority — value is a stable domain boundary, not perf-critical.
- [5.3][LOW] src/io/filter/mind_prepass.cpp:52,64,71 — the `0.0` missing-fraction default appears three times (initial `assign(n_ind, 0.0)` at :23, the no-data write at :64, the no-op `frac` at :71) as a bare literal for the same concept ("zero missing / no-data fail-safe value"). Not a drift risk today (all three want the same numeric zero), but the meaning is implicit. Suggested: optional — a named `kNoMissingFrac`/comment, or just rely on the existing lead comments (lowest priority).
- Note: the packing radix (s/4, s%4) is correctly single-homed in `io::kCodesPerByte` via `kPerByte` (line 44) and the missing sentinel uses the named `io::kMissingCode` (line 48) — NOT magic numbers; no 5.1/5.2 issue there. The `1.0 - nonmissing/n_snp` complement (line 52) is plain fraction math, not a tunable constant. No hardcoded sizes/bounds (5.2), paths/IDs/device ids (5.4), or ambiguous 32 (5.5) — this is a host-pure io TU with no launch dims or shared-mem sizing.

## Group 6 — Naming

- [6.2][LOW] src/io/filter/mind_prepass.cpp:37,64,70 — the loop index over individuals/samples is named `g`, but the entire unit's domain vocabulary is "sample"/"individual" (dims `n_ind`/`n_individuals`, fields `nonmissing`/`missing_frac`/`kept` all indexed per-sample, header §"SAMPLE-GLOBAL invariant"). `g` reads as "genotype" or a locus/SNP index and works against that vocabulary, especially next to the SNP index `s` (which correctly matches `n_snp`). Mildly misleading for the sample axis. Suggested: rename the individual loop index `g` -> `ind` (or `i`) to match the `n_individuals`/sample vocabulary.
- [6.4][LOW] src/io/filter/mind_prepass.cpp:39,48,50 — `nm` is a nonstandard/opaque abbreviation for the per-sample non-missing count; outside the 10-line loop body it gives no hint, and the destination field already has the clear name `out.nonmissing`. Suggested: rename the local `nm` -> `nonmissing_count` (or `n_nonmissing`) to match the `nonmissing` field it feeds.
- Note: other names are clean. `in`/`cfg`/`out` (params/return), `rec`, `frac`, `byte`, `code`, `active`, `kPerByte` are idiomatic and unambiguous (6.1). `s` is a fine tight-loop SNP counter matching `n_snp`. No misleading count-vs-index or list-vs-map names (6.2 otherwise). Conventions are otherwise consistent snake_case (`n_snp`, `bytes_per_record`, `missing_frac`); `nonmissing` as one concept word is acceptable and not worth flagging (6.3). No nonstandard abbreviations beyond `nm` (6.4).

## Group 7 — Duplication

- [7.2][LOW] src/io/filter/mind_prepass.cpp:52 — `static_cast<double>(n_snp)` is a loop-invariant inside the per-individual loop (lines 37-53): `n_snp` is fixed across the whole pass, yet the divisor cast is re-evaluated once per individual (up to ~2500 times). Trivially folded; not a hot path (this is the outer per-sample loop, not the inner per-SNP loop). Suggested: hoist `const double n_snp_d = static_cast<double>(n_snp);` above the loop and divide by it (lowest priority; the compiler likely already CSEs this).
- Note: no copy-pasted blocks differing only by a constant (7.1) — the three `for (g < n_ind)` passes (inner-count loop :37, no-data loop :64, kept-resolution loop :70) each do distinct work over the shared sample axis and are not extractable copy-paste. No repeated sizeof; the packing radix is single-homed in `io::kCodesPerByte`/`kPerByte` (:44) and used once (7.3). No collapsible macro/helper boilerplate (7.4) — this is one short single-function TU with no repeated guard/cleanup scaffolding. The `1.0`/`0.0` literal repetition is a constant-duplication concern already captured under Group 5 (5.1/5.3), not a code-block duplication, so not re-raised here.

## Group 8 — Comments

- [8.2][MED] src/io/filter/mind_prepass.cpp:26-30 — misleading comment on the `active` flag: it says "we still report the missing fractions if we have data, but do not stream when there is nothing to decide." But `active` does NOT gate the streaming pass — the per-SNP count loop at line 32 is gated only on `in.packed != nullptr && n_snp > 0`, independent of `active`. So when `--mind` is inactive (mind_max_missing >= 1.0) but packed data IS present, the code streams and counts ALL SNPs for every sample anyway (the full O(n_ind * n_snp) inner loop runs); `active` is only consulted later at line 71 to decide whether the fraction feeds the drop test. The comment describes a "skip the stream when not active" behavior the code does not have. Not a correctness bug (results are identical; missing_frac is reported either way), but at scale (P~2500, M~584131) it is a real wasted full pass when --mind is off, and the comment hides that. Suggested: either correct the comment to state the stream always runs when data is present (active only gates the drop decision), or actually short-circuit the count loop on `!active` if the missing_frac reporting is not needed in the no-op case.
- Note: 8.1 — no restating-the-obvious comments; every comment carries rationale (bit-order equivalence with the decode path :33-36/:41-43, the no-op fast-path intent :26-29, the no-data keep-all fail-safe and its intentional divergence from snp_filter :55-63). 8.3 — rationale for the non-obvious choices is present and good: the single-homed packing radix (:41-43), the deliberate opposite-of-snp_filter empty-denominator convention (.cpp :55-63 and header :59-64), and the self-documenting no-op loop (:61-63) are all explained. 8.4 — no TODO/FIXME/HACK/XXX markers anywhere in the unit. The line :64 "this loop is a no-op kept only as an explicit statement of the convention" comment is accurate (it re-writes the 0.0 already assigned at :23) and is not stale — the redundancy itself is captured under Group 3 (3.4), so not re-raised here.

## Group 9 — Constants & configuration

- Note (9.1): no should-be-const/constexpr left mutable. Every local that is not subsequently reassigned is already `const` (`n_ind` :19, `n_snp` :20, `active` :30, `rec` :38, `byte` :45, `code` :46, `frac` :71) and the packing radix is `constexpr auto kPerByte` (:44). The only mutable locals — `nm` (:39, `++nm` in the inner loop) and `out` (:21, populated then returned by value) — are correctly mutable. Params `in`/`cfg` are `const&`. No 9.1 issue.
- Note (9.2): the one tunable knob in this TU is the `--mind` activation boundary `cfg.mind_max_missing < 1.0` (:30), and it reads directly off the `FilterConfig` config struct (the knob is surfaced — not buried in a private hardcoded literal); only the comparison-against-`1.0` "inactive" sentinel is a bare literal, which is already captured under Group 5 (5.1, including the drift across config.hpp:386 / header docs / filter_plan / filter_decision). No additional 9.2 issue beyond that 5.1 finding — there are no other buried thresholds, no inline-tuned constants, and no config logic tangled into the streaming loop.
- Note (9.3): no positional-boolean calls anywhere. The three external calls — `code_in_byte(byte, int_pos)` (:47), `sample_passes_mind(frac, cfg.mind_max_missing)` (:72), and the `std::vector` ops `assign`/`reserve`/`push_back` (:22-24,:73) — take no boolean arguments, so there is no `foo(true,false,...)` clarity hazard. No 9.3 issue.

No Group 9 issues found.

## Group 10 — Initialization

- Note (10.1): no late/distant or uninitialized-then-assigned locals. Every local is declared at (or one line above) its first use AND initialized in the same statement: `n_ind`/`n_snp` (:19-20) feed the very next `out` setup; `MindSummary out` (:21) is populated immediately at :22-24; `active` (:30) is initialized from its computing expression; the inner-loop locals `rec` (:38), `nm = 0` (:39), `byte` (:45), `code` (:46) and `frac` (:71) are each declared-with-initializer at point of use. There is no "declare empty, assign later" pattern anywhere in the TU.
- Note (10.2): no unsafe reliance on implicit zero-init. The `out` vectors are EXPLICITLY zeroed via `assign(n_ind, 0)` (:22) and `assign(n_ind, 0.0)` (:23) — not left to default-construction luck — so the else-branch path that leaves `out.nonmissing` unwritten correctly reads the explicit 0 from :22 (the :61 comment "`nonmissing` stays 0 (already assigned)" documents this is intentional, not an implicit-zero assumption). The header default member initializers in `MindPrepassInput` (`= nullptr`, `= 0` at :40-43) are all explicit, and `MindSummary`'s `std::vector` members (:49-51) default-construct to a well-defined empty state with no raw-storage zero-init assumption. No 10.2 issue.

No Group 10 issues found.

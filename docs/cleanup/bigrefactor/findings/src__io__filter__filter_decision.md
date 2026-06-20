# Review findings — src__io__filter__filter_decision

Files: src/io/filter/filter_decision.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

## Group 6 — Naming

No Group 6 issues found.

## Group 7 — Duplication

- [7.1][LOW] src/io/filter/filter_decision.hpp:199,211 — the guard `if (na == '\0' || nb == '\0' || na == nb) return false;` is repeated verbatim in is_transition and is_transversion, and is the negation of the is_multiallelic body (190); a SNP that fails this guard IS multiallelic. Suggested: optionally express the guard once as `if (is_multiallelic(a, b)) return false;` inside is_transition/is_transversion so the "clean biallelic ACGT pair" rule lives in one predicate (consistent with the §8 single-source intent already stated in the header).
- [7.2][LOW] src/io/filter/filter_decision.hpp:208-212 — is_transversion normalizes a/b into na/nb (208-209) only to run the guard, then delegates to is_transition (212) which normalizes a/b a second time; the na/nb computed here are otherwise unused. Redundant double-normalize per call. Suggested: drop the local na/nb here and reuse is_transition's classification (e.g. guard via is_multiallelic then `return !is_transition(a,b);`), so each input is normalized once per logical check.
- [7.4][LOW] src/io/filter/filter_decision.hpp:174-176,187-189,196-198,208-210 — the two-line `const char na = normalize_allele(a); const char nb = normalize_allele(b);` prologue is copy-pasted across four allele-pair predicates. Suggested: leave as-is unless consolidating per 7.1/7.2 — a tiny helper returning a normalized pair would fold it, but the current standalone form is intentionally self-documenting and the cost is nil (inlined). Note only.

## Group 8 — Comments

No Group 8 issues found.

## Group 9 — Constants & configuration

- [9.1][LOW] src/io/filter/filter_decision.hpp:74,83,92,100,123,224 — the pure numeric/threshold predicates (folded_maf, snp_passes_maf, snp_passes_geno, sample_passes_mind, is_monomorphic, is_autosome) are `inline noexcept` but not `constexpr`, though their bodies are constant-expression-eligible (only comparisons/arithmetic, no std::toupper). They are called at runtime on streamed Q/V/N, so this is hygiene, not a correctness gap; the allele-pair predicates (136-213) correctly cannot be constexpr because normalize_allele depends on std::toupper. Suggested: optionally mark the six non-allele predicates `constexpr` to enable compile-time use of FilterConfig thresholds; low priority, no behavior change.

## Group 10 — Initialization

No Group 10 issues found.

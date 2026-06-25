# src__io__filter__snp_filter
Files: /home/suzunik/steppe/src/io/filter/snp_filter.cpp, /home/suzunik/steppe/src/io/filter/snp_filter.hpp
Subsystem: io

## Findings

### G3 dead/unused
- [G3.dead][LOW] snp_filter.cpp:13 — `#include "io/filter/filter_decision.hpp"` is included directly but this TU references no `filter_decision.hpp` symbol of its own; the predicates are used only transitively (snp_keep_decision in the header forwards to keep_decision_pooled, and snp_summary_reduce.hpp already pulls in filter_decision.hpp). The header comment "// the shared predicates (single source)" documents intent, but the include is functionally redundant for this .cpp. Suggested: drop it (snp_summary_reduce.hpp transitively provides the predicates), or keep it only if IWYU-documenting intent is deliberate.

### G8 comments
- [G8.stale][LOW] snp_filter.cpp:84 — comment "incl. the FFMA-immune Σ Q·N (the bit-exact pin)" plus the surrounding block describes the FFMA-immune two-step product, but that logic now lives entirely in snp_summary_reduce.hpp (pooled_ref_fma); this loop only calls `derive_pooled_summary_one`. The rationale is correct but its physical home moved, so the detail risks drifting from the code it sits next to. Suggested: trim to a one-line "shared primitive enforces the bit-exact Σ" and point at snp_summary_reduce.hpp.
- [G8.verbose][LOW] snp_filter.hpp:99-100 / snp_filter.cpp:59-64 — the "`in.v` is intentionally unused" rationale is duplicated near-verbatim in both the header doc and the .cpp body comment. Not wrong (it is load-bearing rationale for why V is absent from the reduction), just stated twice in one unit. Suggested: keep one authoritative copy (header), reference it from the .cpp.

No issues found in groups: G2, G4, G5, G6, G7, G9, G10. (Index math is correct: in DecodedTileSummaryInput M is `long`, the s-loop counter is `long`, and the column base `static_cast<long>(P)*s` is widened before the multiply in snp_summary_reduce.hpp so P~2500 × M~584k stays within long; `total_indiv` is `std::size_t`; the M<0 guards and `static_cast<std::size_t>` conversions are guarded by `M<=0` early-returns. Magic values are config-driven; the {1,2} ploidy literals are named in the throw text. Preconditions are fail-fast with std::invalid_argument per the io-leaf idiom.)

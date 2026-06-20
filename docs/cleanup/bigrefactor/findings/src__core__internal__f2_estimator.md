# Review findings — src__core__internal__f2_estimator

Files: /home/suzunik/steppe/src/core/internal/f2_estimator.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/core/internal/f2_estimator.hpp:35 — `#include "core/internal/launch_config.hpp"` (cdiv/grid_for) is not used by any code in this header; the comment at lines 111-114 states it is included solely to re-export the launch-math helpers to downstream feeder-kernel/test TUs that include this header. This is an intentional umbrella/pass-through include, not accidental dead code, but it couples a host-pure numerics header to launch-config purely for transitive re-export. Suggested: leave as-is if the re-export contract is relied upon, or have those downstream TUs include launch_config.hpp directly and drop the re-export to keep this header's includes self-justifying.

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found. The het-correction floor (AT2's `max(N-1,1)`) is correctly extracted to the named constant `kHetCorrDenomFloor` (steppe/config.hpp:71, used at f2_estimator.hpp:58) — sample size `n` is a per-SNP argument, never hardcoded. The remaining literals are genuine mathematical constants, not magic numbers: the `1.0` in `(1.0 - q)`/`(n - 1.0)` (definition of variance/Bessel term), the `2.0` in `assemble_f2_numerator` (the cross term of `a²−2ab+b²`, explicitly documented as a true math constant at line 93), and the `0.0`/`> 0.0` guards in `het_correction`/`finalize_f2` (the AT2 `Vpair==0 ⇒ 0` and invalid-entry zero-fill conventions, documented at lines 100-108). No hardcoded sizes/bounds, no block/grid dims, no ambiguous `32`/warp-size literal, and no paths/IDs/device ids — this is a host-pure scalar header with no kernel launches.

## Group 6 — Naming

No Group 6 issues found. The single-letter names are domain-standard f-statistics notation, not cryptic opacity: `q` is documented as the reference-allele frequency (line 48), `n` as the non-missing haploid count (lines 48-49), `p_i`/`p_j` as the per-population allele frequencies (lines 73-74, AT2 convention), and `d` is a 3-line-scoped local for the difference `p_i - p_j`, explicitly defined at line 75. The abbreviated names are all expanded in the adjacent doc comments or are AT2/architecture.md symbols: `hc`/`hc_i`/`hc_j` = het correction (lines 44-45, 64-67), `sumsq_i`/`cross`/`hsum_i` map one-to-one to the documented `R_diag`/`G(i,j)`/`H(i,j)` reduced statistics (lines 84-88), and `vpair` is documented as the pairwise-valid-SNP-count (lines 100-108) — names match their contents, none is a count-masquerading-as-index or list-masquerading-as-map. Conventions are consistent within the file: functions and locals are `snake_case` (`het_correction`, `f2_term`, `assemble_f2_numerator`, `finalize_f2`), the constant is `kPascalCase` (`kHetCorrDenomFloor`), and there is no `nElements`-vs-`num_elements`-vs-`n` drift.

## Group 7 — Duplication

- [7.2][LOW] src/core/internal/f2_estimator.hpp:58 — the subexpression `n - 1.0` is computed twice in one statement (once in the comparison `(n - 1.0 > kHetCorrDenomFloor)`, once in the selected branch `(n - 1.0)`). Purely cosmetic — the compiler will CSE it and the result is bit-identical, so there is no parity/perf concern; flagged only as a micro-DRY hygiene note. Suggested: optional — bind `const double nm1 = n - 1.0;` and use `denom = (nm1 > kHetCorrDenomFloor) ? nm1 : kHetCorrDenomFloor;`. Otherwise leave as-is.

## Group 8 — Comments

No Group 8 issues found. No comment restates code mechanically (8.1). No stale comments (8.2): every cited symbol matches current code — `kHetCorrDenomFloor` exists (config.hpp:71, = 1.0) and is used at f2_estimator.hpp:58; `cdiv`/`grid_for` live in launch_config.hpp as the comment at lines 111-114 describes and are included at line 35; `STEPPE_HD` is the shared host/device qualifier from host_device.hpp (also used by decode_af.hpp) as the header doc at lines 25-30 states; the doc comments on `het_correction`/`f2_term`/`assemble_f2_numerator`/`finalize_f2` accurately describe the present formulas. Rationale is present where needed (8.3): the het-correction floor (lines 50-51), the cancellation-free `(p_i−p_j)²` form vs the expanded form (lines 62-72), the `2.0` cross-term constant explicitly justified as a true math constant not a magic number (line 93), the native-FP64 hold of the cancellation site in every precision mode (lines 91-92), and the `vpair > 0.0` vs `>= 1.0` exactness note (lines 104-105) are all documented. No orphan TODO/FIXME/HACK/XXX markers (8.4).

## Group 9 — Constants & configuration

No Group 9 issues found. (9.1) No should-be-const/constexpr left mutable: the only tunable constant `kHetCorrDenomFloor` is `inline constexpr double` (config.hpp:71); both function-local intermediates are already `const` (`const double denom` at f2_estimator.hpp:58, `const double d` at f2_estimator.hpp:75). (9.2) No tangled config: the sole tunable knob (`kHetCorrDenomFloor`) is surfaced in the `steppe/config.hpp` config home and pulled in via the include at line 36 — it is NOT buried in the numerics logic; the remaining literals (`1.0`, `2.0`, `0.0`) are mathematical constants per the Group 5 pass, not configuration. Sample size is a per-SNP `n` argument (lines 56-59), never a hardcoded config value. (9.3) No positional booleans at call sites: this header issues no calls passing boolean literals (`foo(true, false, ...)`); the lone `bool valid` is an incoming function *parameter* on `het_correction` (line 56), documented as the validity bit (lines 44-55), not an ambiguous positional-flag call. The `STEPPE_HD` qualifier macro (host_device.hpp) is a compile-mode portability switch, not a tunable runtime knob.

## Group 10 — Initialization

No Group 10 issues found. (10.1) The only two function-locals are declared at their point of use and initialized in the same statement: `const double denom` (f2_estimator.hpp:58) is computed then read at line 59, and `const double d` (f2_estimator.hpp:75) is computed then read at line 76 — neither is declared far from first use, and neither is the uninitialized-then-assigned-later anti-pattern (both are `const`, so they cannot be). All other identifiers are incoming function parameters, not locals. (10.2) No zero-init assumptions: this header has no structs, arrays, buffers, aggregates, or static/global state that could silently rely on zero-initialization; every function returns a directly-computed value, and the `return 0.0` / `0.0 :` results in `het_correction` (line 57) and `finalize_f2` (line 108) are explicit literal returns encoding the AT2 invalid-entry / `Vpair==0 ⇒ 0` conventions, not implicit zero-init.


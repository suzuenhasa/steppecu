# Review findings — src__core__internal__pchisq

Files: /home/suzunik/steppe/src/core/internal/pchisq.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

- [5.3][LOW] pchisq.hpp:23,39 — `kMaxIter = 1000` is declared independently in both `pchisq_gammp_series` and `pchisq_gammq_cf`; same value duplicated across the two convergence loops, so tuning one tail's iteration cap without the other would silently drift the two tails apart. Suggested: hoist a single namespace-scope `constexpr int kPchisqMaxIter` shared by both helpers.
- [5.3][LOW] pchisq.hpp:24,40 — `kEps = 1e-15` is declared independently in both helper functions (series and continued-fraction); a duplicated convergence-tolerance constant with the same drift risk. Suggested: hoist a single namespace-scope `constexpr double kPchisqEps`.

## Group 6 — Naming

No Group 6 issues found.

## Group 7 — Duplication

- [7.2][LOW] pchisq.hpp:34,58 — the regularized-incomplete-gamma prefactor `std::exp(-x + a * std::log(x) - std::lgamma(a))` is written out identically in both `pchisq_gammp_series` (line 34) and `pchisq_gammq_cf` (line 58); a duplicated multi-op expression that must stay bit-identical for the two tails to compose into a consistent `pchisq_upper`. Suggested: extract a small `inline double pchisq_gamma_prefactor(double a, double x)` helper called by both.
- [7.3][LOW] pchisq.hpp:47 — `static_cast<double>(i)` is computed twice in the same expression `-static_cast<double>(i) * (static_cast<double>(i) - a)`. Suggested: bind `const double di = static_cast<double>(i);` once and reuse.

## Group 8 — Comments

No Group 8 issues found.

## Group 9 — Constants & configuration

- [9.1][LOW] pchisq.hpp:23,24 — `const int kMaxIter = 1000;` and `const double kEps = 1e-15;` are pure literal-initialized fixed knobs but declared `const` (runtime) rather than `constexpr`; they are compile-time constants. Suggested: make them `constexpr`.
- [9.1][LOW] pchisq.hpp:39,40,41 — `const int kMaxIter`, `const double kEps`, `const double kFpMin = 1e-300` are likewise literal compile-time constants declared only `const`. Suggested: make them `constexpr`.
- [9.2][LOW] pchisq.hpp:23,24,39,40,41 — the algorithm tuning knobs (iteration cap, convergence tolerance, floating-point floor) are buried as block-scope locals inside each helper rather than surfaced at namespace/file top as a single set of `constexpr` config values; tuning them requires editing two function bodies and keeping the duplicates in sync (see also 5.3). Suggested: hoist shared `constexpr int kPchisqMaxIter`, `constexpr double kPchisqEps`, `constexpr double kPchisqFpMin` to namespace scope.

## Group 10 — Initialization

No Group 10 issues found.

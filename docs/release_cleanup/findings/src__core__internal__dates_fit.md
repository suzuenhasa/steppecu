# src__core__internal__dates_fit
Files: /home/suzunik/steppe/src/core/internal/dates_fit.hpp
Subsystem: core-stats

## Findings

### G5 hardcoded values/magic numbers
- [G5.src__core__internal__dates_fit][LOW] dates_fit.hpp:107 — the `200` ternary-refine iteration count is an unnamed magic literal, whereas the sibling coarse-grid size is the named `const int coarse = 4000` (line 86). Doc comment (line 73) calls it "200-iter ternary refine", so it is a deliberate tunable. Suggested: hoist to a named `const int kTernaryIters = 200` (or similar) for parity-with `coarse`.

### G7 duplication
- [G7.src__core__internal__dates_fit][LOW] dates_fit.hpp:87-94 vs 96-102 — the coarse 4000-point grid loop is copy-pasted as a fallback, the two copies differing only by the dropped `if (co0 <= 0.0) continue;` positivity filter (line 92). The any-sign fallback (lines 95-104) is an intentional DATES behavior, but the loop body is otherwise identical. Suggested: extract a single grid-scan helper taking a `require_positive` bool (or a predicate), called twice.

### G6 naming
- [G6.src__core__internal__dates_fit][LOW] dates_fit.hpp:110 — the ternary-search out-param names `c1a, c1b, c2a, c2b` are opaque (they hold the `(co0, c)` solve outputs for the two midpoints `m1`/`m2`); the values are written but never read after the loop, so the obscurity has no correctness impact. Suggested: name them e.g. `co0_m1, c_m1, co0_m2, c_m2`, or use a single throwaway pair since they are unused post-loop.

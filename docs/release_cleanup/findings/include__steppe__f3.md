# include__steppe__f3
Files: /home/suzunik/steppe/include/steppe/f3.hpp, /home/suzunik/steppe/src/core/qpadm/f3.cpp
Subsystem: core-qpadm

## Findings

### G4
- [G4.include__steppe__f3][MED] f3.cpp:69 — `const int N = static_cast<int>(triples.size())` truncates a `std::size_t` to `int`. `N` is the f3 batch m-axis (one triple = one column); an all-triples sweep at P~2500 is C(2500,3) ≈ 2.6e9 triples, exceeding INT_MAX, so this silently wraps negative/truncates before the `N<=0` empty-batch guard at line 70 — a real int-index-overflow-at-scale bug, not a theoretical one (cf. the 2.57B-quartet sweep). The same `int m = X.nl * X.nr` (line 94) and the seam's `int nl/nr` are a backend-wide convention, but the truncation point introduced here is the cast at line 69. Suggested: keep the batch count in `std::size_t` (or guard `triples.size() > INT_MAX`) before narrowing.

### G8
- [G8.include__steppe__f3][LOW] f3.hpp:19 — stale comment: the header still describes the SE as `se[k] = sqrt(Q[k + m*k]) (the UNFUDGED diagonal)`, i.e. the dense m×m Q diagonal. The .cpp (lines 108-118) replaced this with `jackknife_diag` (diagonal-only, never forms Q — the OOM fix) and reads `diag.var[ks]` (line 126), not a Q matrix. The header narrative thus describes behavior the code no longer has. Suggested: align the f3.hpp comment with the jackknife_diag path the .cpp documents.
- [G8.include__steppe__f3][LOW] f3.cpp:13 — same staleness in the file-top PIPELINE comment: `se[k] = sqrt(Q[k+m*k]) (the UNFUDGED diagonal)` references the dense Q, while the implementation at line 116-117/126 uses `jackknife_diag`/`diag.var[ks]`. The later block comment (lines 108-115) correctly explains the switch, so the top comment contradicts it. Suggested: reconcile the two comments (point the top one at jackknife_diag).

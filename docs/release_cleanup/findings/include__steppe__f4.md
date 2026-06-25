# include__steppe__f4
Files: /home/suzunik/steppe/include/steppe/f4.hpp, /home/suzunik/steppe/src/core/qpadm/f4.cpp
Subsystem: core-qpadm

## Findings

### G6
- [G6.naming][LOW] f4.cpp:67,95 — `N` (int, from `quartets.size()`) and `m` (`X.nl * X.nr`) are two names for the same logical quantity (the quartet count). The code asserts `m == N` only in a comment (line 95 `// == N`) and never reconciles them: the loop at line 126 iterates `k < N` while indexing `diag.var[ks]`/`X.x_total[ks]` which are sized by the seam's `m`. If a seam ever returned `m != N` (e.g. an upstream filter dropping quartets) the loop would silently read past or under the seam arrays with no guard. Using one variable (or asserting equality) would make the invariant load-bearing rather than commentary. Suggested: `assert(m == N)` or drive the loop on `m`.

### G8
- [G8.comments][LOW] f4.cpp:109 — comment calls jackknife_diag "(the OOM fix)" and references "the deleted dense diagonal" / "jackknife_cov used to compute"; this is process/history narration (refers to code that no longer exists in this TU) rather than describing the current behavior. Mildly stale-leaning; the surrounding rationale (diagonal == diag(Q) by construction) is the valuable part. Suggested: trim the historical "OOM fix / deleted dense" framing, keep the math invariant.

No issues found beyond the above for groups checked: G2, G3, G4, G5, G7, G9, G10. Notes for the record (NOT findings): the `double` literals, `std::erfc`/`std::sqrt`/`std::fabs`, and native-FP64 `est` are intentional per §12/the cancellation carve-out; `static const double kInvSqrt2` (f4.cpp:153) is a C++20 thread-safe magic static in a host-pure function — fine; the `int N` cast at f4.cpp:67 cannot overflow in practice (quartet count, not P*P*n_block), so no G4 index-width concern; `(void)opts` at f4.cpp:61 is a deliberate -Werror acknowledgement of an intentionally-unconsumed param (documented), not dead code.

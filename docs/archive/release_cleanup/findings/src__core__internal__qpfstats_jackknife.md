# src__core__internal__qpfstats_jackknife
Files: /home/suzunik/steppe/src/core/internal/qpfstats_jackknife.hpp
Subsystem: core-stats

## Findings

### G4
- [G4.src__core__internal__qpfstats_jackknife][LOW] qpfstats_jackknife.hpp:90-96 — `f2blocks_pair_est` derives `nb = arr.size()` (line 92) but then indexes the companion `bl` vector at the same indices `b` (lines 96, 104, 113) with no check that `bl.size() == arr.size()`. If `bl` is shorter than `arr`, the `bl[b]` reads are out of bounds. The two are documented as both being length n_block, so this is an undefended precondition rather than a triggered bug. Suggested: add a paired-length assert/contract guard (or iterate to `min(arr.size(), bl.size())`).

### G8
- [G8.src__core__internal__qpfstats_jackknife][LOW] qpfstats_jackknife.hpp:68 — the finiteness guard casts the `long double loo` down to `double` purely to call `std::isfinite(static_cast<double>(loo))`; this discards range/precision and, for a `loo` magnitude that is finite as `long double` but overflows `double`, would spuriously reject it. The intent (drop NaN/Inf loo) reads as if it operated on the long-double value. Suggested: call `std::isfinite(loo)` directly on the `long double` (the `<cmath>` overload covers it), or add a one-line comment explaining the deliberate double-narrowing for AT2 bit-parity.

No other issues (groups checked: G2, G3, G5, G6, G7, G9, G10).

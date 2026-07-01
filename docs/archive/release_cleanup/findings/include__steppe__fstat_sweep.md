# include__steppe__fstat_sweep
Files: /home/suzunik/steppe/include/steppe/fstat_sweep.hpp, /home/suzunik/steppe/src/core/qpadm/fstat_sweep.cpp
Subsystem: core-qpadm

## Findings

### G3
- [G3.core-qpadm][LOW] src/core/qpadm/fstat_sweep.cpp:69-70 — `res.enumerated` is assigned the host `choose_saturating` count, but on the non-early-return (dispatch) path it is unconditionally overwritten at line 110 by `sv.enumerated` (the backend's exact count). The line-69 store to `res.enumerated` is only ever read on the `range < k` (line 72) and `capped` (line 80) early-return paths; the local `enumerated` (line 68) is what the cap compare at line 80 uses. So on the compute path the line-69 assignment is a dead store. Suggested: only set `res.enumerated` in the early-return branches (or document that the host estimate is intentionally placeholder until the backend echoes its exact count).

### G5
- [G5.core-qpadm][LOW] src/core/qpadm/fstat_sweep.cpp:91 — `cfg.filter_mode` is set to a bare `0`/`1` magic literal (`(req.filter == SweepFilter::MinZ) ? 0 : 1`). The meaning lives only in the trailing comment, and the same 0/1 contract must be matched on the backend decode side (drift risk). Suggested: map to a named enum / `static_cast<int>` of a shared enum rather than inline 0/1.

### G8
- [G8.core-qpadm][LOW] include/steppe/fstat_sweep.hpp:14-16 and src/core/qpadm/fstat_sweep.cpp:6 — header/impl comments describe the combine precision as the "native carve-out for the cancellation-sensitive combine", but the impl threads `core::qpadm::default_fit_precision()` (the EmulatedFp64{40} default per the landed fit-precision policy) into `f4_sweep`/`f3_sweep` (lines 61, 108) with no explicit native carve-out here. The carve-out, if any, is entirely inside the backend; the comment reads as if this driver enforces it. Suggested: clarify that precision selection (default + carve-out) is owned by the backend kernels, this driver only forwards `default_fit_precision()`.

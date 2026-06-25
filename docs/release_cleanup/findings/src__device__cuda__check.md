# src__device__cuda__check
Files: /home/suzunik/steppe/src/device/cuda/check.cuh
Subsystem: device-cuda

## Findings

### G7
- [G7.src__device__cuda__check][LOW] check.cuh:54-146 — The three typed exceptions (`CudaError` 54-70, `CublasError` 75-107, `CusolverError` 116-146) share an identical message-construction body (`std::string(loc.file_name()) + ":" + std::to_string(loc.line()) + " (" + loc.function_name() + "): '" + expr + "' -> " + ...` at lines 59-62, 80-82, 121-123) and identical `what()`/`status()` accessors. Only the trailing status-rendering differs (cudaGetErrorName/String vs. a hand-rolled `status_name` switch). The shared prefix could be a single free helper (e.g. `format_call_site(loc, expr)`) to remove the three-way copy and drift risk. Suggested: extract a shared `format_call_site(loc, expr)` helper; keep the per-API status rendering inline.

### G8
- [G8.src__device__cuda__check][LOW] check.cuh:25-27, 165 — The same `TODO(M4.5)` (gate the CAP-1/CAP-2 §12-parity claim in CI by running goldens on P2P-on and P2P-off tiers) is stated at lines 25-27 and again referenced at line 165 ("TODO M4.5"). These are owner-tagged and rationale-bearing (not orphan), but the duplicated TODO text in two comment blocks of one file invites drift — if the task closes, both must be removed. Suggested: keep one canonical TODO(M4.5) and have the other comment cross-reference it rather than restate it.

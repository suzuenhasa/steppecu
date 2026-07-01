# src__core__internal__small_linalg
Files: /home/suzunik/steppe/src/core/internal/small_linalg.hpp
Subsystem: core-stats

## Findings

### G8
- [G8.task][LOW] small_linalg.hpp:215 — `kTol = 1e-15` is the relative per-pair Jacobi skip tolerance (used at line 233 as `kTol * sqrt(alpha*beta)`), but unlike its two siblings `kMaxSweeps` (216-219) and `kOffDiagTol` (220-222) it carries no rationale comment explaining what the magnitude means or whether it is parity-load-bearing. Given `kOffDiagTol` is flagged PARITY-FROZEN, a reader cannot tell if `kTol` is equally frozen. Suggested: add a one-line comment stating its role and frozen/tunable status, mirroring the kOffDiagTol note.

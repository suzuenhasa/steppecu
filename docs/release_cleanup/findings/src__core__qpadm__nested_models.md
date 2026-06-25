# src__core__qpadm__nested_models
Files: /home/suzunik/steppe/src/core/qpadm/nested_models.cpp, /home/suzunik/steppe/src/core/qpadm/nested_models.hpp
Subsystem: core-qpadm

## Findings

### G8
- [G8.stale][MED] nested_models.hpp:5-8 — File-header comment still describes the pre-M7 design: "it re-runs the S6 weight solve (the backend gls_weights virtual) once per LOO block ... then forms the delete-1 jackknife covariance of the weight replicates → se, z." The implementation no longer does this in this file; it makes a single `be.se_from_wmat(...)` call (nested_models.cpp:31) that subsumes the per-block re-fits, scaling, and covariance reduction inside the backend. The .cpp comment (lines 22-30) is accurate and up to date, but the header was not synced. Suggested: rewrite the header comment to describe the single `se_from_wmat` seam call (as the .cpp already does), not the old per-block gls_weights loop.
- [G8.stale][MED] nested_models.hpp:10-13 — Same staleness: "The per-LOO-block re-fits run through the batched-capable backend seam gls_weights_loo_batched (design §2) ... No new backend virtual is added." A new virtual (`se_from_wmat`, backend.hpp:1760) WAS added and is now the seam this function uses; `gls_weights_loo_batched` is no longer called from here. Suggested: point the header at `se_from_wmat` and drop the "no new backend virtual is added" claim.
- [G8.stale][MED] nested_models.hpp:33 — Doxygen on `se_from_loo` says "re-solve the weights via `be.gls_weights_loo_batched`, reusing `cov.Qinv`. Then scale wmat by (numreps-1)/sqrt(numreps) ... and take se = sqrt(diag(cov(wmat)))". None of this scaling/covariance is done here anymore; it is all inside `be.se_from_wmat`. Suggested: update the docstring to "delegates the LOO re-fits + AT2 scale + covariance-diagonal reduction to `be.se_from_wmat`; computes only z = weight/se here."
- [G8.stale][LOW] nested_models.hpp:36 — Docstring ends "Native FP64." but the function takes `const Precision& precision` and forwards it to a backend whose path is precision-policy-selected (EmulatedFp64 default per the landed fit-precision policy; native is the carve-out). The flat "Native FP64" assertion is misleading for a function that is explicitly precision-parameterized. Suggested: drop or qualify ("SE reduction native FP64; underlying re-fits per `precision`").

### G3
- [G3.computed-unread][LOW] nested_models.cpp:14 — `nl` (line 12) and `nb` (line 13) are read, but `nb` is used only in the guard at line 20 (`nb < kMinJackknifeBlocks`); fine. No dead code — noted only to confirm both locals are live. No action.

### G10 / robustness
- [G10.guard][LOW] nested_models.cpp:38 — `weight[static_cast<std::size_t>(i)]` is indexed over `i ∈ [0, nl)` with NO size guard, whereas the adjacent `se[...]` read (lines 33-35) IS guarded against `se.size()`. `weight` is `gw.w`, documented `[nl]` (backend.hpp:147) so it is safe by construction today, but the asymmetric guarding makes the access fragile against a future caller passing a shorter `weight`. Suggested: guard symmetrically (`i < weight.size()`) or assert `weight.size() == static_cast<size_t>(nl)` at entry.

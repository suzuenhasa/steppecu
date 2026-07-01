# GROUP 10 (Initialization) — Review Summary

Tasks: 10.1 Late/distant declarations (declared far from first use, or uninitialized-then-assigned); 10.2 Zero-init assumptions (missing init relying on zero-init that does not hold).

## 1. Coverage

- Units in scope: 61
- Units with a Group 10 section: 61 (100%)
- Clean (no issues): 57
- With findings: 4
- Total findings: 4

Units with findings:
- include__steppe__qpadm (1)
- src__device__cpu__cpu_backend (1)
- src__device__cuda__cuda_backend (1)
- src__device__cuda__qpadm_fit_kernels (2)

## 2. Counts by task + severity

| Task | HIGH | MED | LOW | Total |
|------|------|-----|-----|-------|
| 10.1 Late/distant | 0 | 0 | 2 | 2 |
| 10.2 Zero-init assumptions | 0 | 2 | 0 | 2 |
| **Total** | **0** | **2** | **2** | **4** |

No HIGH findings. No active bug exists on any reachable path today; all four are latent / defensive-contract gaps.

## 3. Top findings

### MED (10.2 — zero-init / contract gaps)

- [10.2][MED] src/device/cpu/cpu_backend.cpp:485-487 (jackknife_cov), consumed at 541 (rank_sweep) / 618 (gls_weights) — when the fudged Q is singular, `core::inverse` (small_linalg.hpp:112) returns early WITHOUT touching the out-param, leaving `out.Qinv` as a default-constructed EMPTY vector while only `out.status = NonSpdCovariance` flags it. Downstream `als_weights`→`opt_A/opt_B/chisq_of` index `qinv[kr + m*kc]` (e.g. :815) → OOB read on an empty vector if m>0. Not live only because the orchestrator gates on `cov.status != Status::Ok` first (qpadm_fit.cpp:71,267); the backend method carries no local guarantee. Suggested: size+zero `out.Qinv` (m*m, 0.0) on the failure path so the "always sized" contract holds unconditionally, OR early-return in rank_sweep/gls_weights on `cov.status != Ok` / `cov.Qinv.empty()`.

- [10.2][MED] src/device/cuda/qpadm_fit_kernels.cu:118-176 (dev_jacobi_svd_V) + 185-204 (dev_seed_ab) — `dev_jacobi_svd_V` writes only the leading `min(r,k)` columns of `Vout` (copy loop :171-175, k=min(m,n)) and never zero-inits the buffer; `dev_seed_ab` then reads ALL r columns (:194-196). If a caller ever passes `r > min(nl,nr)`, columns [k, r) are read UNINITIALIZED into B and propagate into A and chisq. Not live: the rank axis is bounded by `rmax = min(nl,nr)-1` (.cuh:273, kernel :1127/:1160) so every reachable r satisfies r <= k-1 < k. Hazard = silent reliance on the caller invariant with no defensive init/guard. Suggested: zero-init the r*nr region of `Vout` at the top of dev_jacobi_svd_V, OR add a device-side `assert(r <= k)` in dev_seed_ab; do not change the parity math on the in-contract path.

### LOW (10.1 — uninitialized-then-assigned hygiene)

- [10.1][LOW] src/device/cuda/cuda_backend.cu:2354 — `bool survivor;` is the only uninitialized scalar local in the TU. Assigned in all three `switch (opts.jackknife)` arms (None :2357, FeasibleOnly :2362, All/default :2367) before its sole read at :2370; `default` makes the switch exhaustive, so no path reads it uninitialized (correct). Hygiene only: the no-uninitialized-read property is switch-coverage-dependent, not structural. Suggested: `bool survivor = false;` at declaration, or compute as a `const` via IIFE so a future enum case can't open an uninitialized window.

- [10.1][LOW] src/device/cuda/qpadm_fit_kernels.cu:1118,1128,1171,1185 (qpadm_fit_models_kernel) — `chisq`/`cr` declared uninitialized then written by `dev_als_weights`/`fit_reduced` via the `chisq_out` out-pointer before any read (:1124/:1131/:1173/:1188); `dev_als_weights` always writes `*chisq_out` on every return path (r==0 :350, full :380), so the pattern is read-safe. Pure readability nit. Suggested: optional `= 0.0` sentinel at declaration to make the "callee fills it" contract explicit; no functional change.

### LOW (10.2)

- [10.2][LOW] include/steppe/qpadm.hpp:109 — In QpAdmModel, `int target` has NO in-class default initializer, unlike `model_index = -1` (L119) and every scalar member in the three sibling structs (QpAdmOptions L60-99, QpAdmResult L131-161, QpWaveResult L217-220). A value-init omitting target (`QpAdmModel{}` / aggregate-init missing the first member) leaves it indeterminate, and target is then used as an f2_blocks P-axis index. No call site is wrong today (caller always sets target) — defensive/consistency gap. Suggested: `int target = -1;` to match model_index and the value-type convention.

## 4. Cross-cutting patterns

- **Callee-fills-out-param contract, no defensive init at the seam.** Three of the four findings (cpu_backend Qinv, qpadm_fit_kernels Vout, qpadm_fit_kernels chisq/cr) share one shape: a producer leaves a buffer/scalar untouched on some path (early-return on singular, partial column write, or "callee always writes"), and a consumer relies on it being sized/written. Each is safe ONLY because of an upstream guarantee (status gate, rank bound, unconditional out-write) that lives in a different function and is not locally enforced. The robustness fix in all cases is the same: make the producer establish the contract unconditionally (size+zero / full-write / sentinel) so a future caller dropping the upstream guard cannot turn a latent gap into an OOB/uninitialized read.
- **Switch/branch exhaustiveness as the only init guarantee.** cuda_backend survivor and (separately) the qpAdm value-type defaults rely on enumerated coverage rather than declaration-site initialization. Adding an enum case without revisiting these sites reopens the window.
- **Otherwise the codebase is disciplined.** 57/61 units confirmed: scalars declared at first use WITH initializers, API out-locals zero-initialized BEFORE the cuBLAS/cuSOLVER/runtime call, result PODs use explicit in-class default member initializers (not implicit zero-init), and every device-output / result vector is EXPLICITLY value-filled (`.assign(..., 0.0)`) rather than relying on container or VRAM pre-zeroing. No reliance on zero-initialized VRAM was found.

---
HEADLINE: 61 units, 4 total findings, 0 HIGH (2 MED / 2 LOW).

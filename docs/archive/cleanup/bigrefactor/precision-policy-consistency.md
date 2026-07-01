# Precision-Policy Consistency Audit

**Scope:** every matmul / SYRK / GEMM / cuSOLVER-solve / `Precision`-taking site in
`src/` and the fit/search entrypoints. **Method:** 5 independent lenses
(matmul-engage, void-precision, entrypoint-defaults, honorable-fallback,
cpubackend-and-tags), cross-checked and re-verified against source by the lead
synthesizer. Read-only — no edits.

**Policy under audit** (architecture.md §12; fit-engine.md §1.4; commits d6d3cbb
cuSOLVER seam + 8ace724 unified fit policy):
- DEFAULT = `EmulatedFp64{mantissa_bits=40}` (Ozaki fixed-slice) for the
  well-conditioned matmul-heavy stages (f2 3-GEMMs, covariance SYRK/batched GEMM).
- FALLBACK = native FP64, decided by the ONE predicate
  `emulation_honorable(precision)` → `CUBLAS_PEDANTIC_MATH` with a logged tag.
- NATIVE-ALWAYS CARVE-OUTS: cancellation-sensitive elementwise math (f2 numerator,
  f4 4-slab combine, xtau centering), cuSOLVER solves (no CUDA-13 FP64-emulated
  cuSOLVER math mode; `STEPPE_HAVE_CUSOLVER_FP64_EMULATED==0`), and the CpuBackend
  oracle (always native long-double/FP64).

---

## (1) HEADLINE VERDICT

**CONSISTENT. DEVIATIONS FOUND: 0.**

The emulated-FP64-default + honorable-native-fallback policy is applied uniformly
across every matmul-heavy GEMM/SYRK site through the single `emulation_honorable`
predicate and the single `engage_f2_precision` cuBLAS engage. Every native site is
a documented carve-out (cancellation-elementwise, cuSOLVER-solve, base-class stub,
or CpuBackend oracle), each routing the SAME predicate where a seam exists. No
matmul-heavy op is hardcoded native; no site bypasses the honorable predicate; no
matmul-precision entrypoint defaults native; no inconsistent precision tag. Two
non-blocking cleanups (a missing-for-uniformity solver scope and a stale comment)
are noted below — neither changes runtime behavior, neither is a deviation.

---

## (2) THE POLICY MAP

Classification key: **ENGAGES** = correctly engages emulated-default + honorable
fallback · **CARVE-OUT** = legitimately native (cancellation / cuSOLVER-solve /
base-stub / oracle) · **DEVIATION** = a matmul-heavy op hardcoded native, a site
skipping the predicate, an entrypoint defaulting native, or an inconsistent tag.

### Single source of the policy (the ONE predicate / engage / map)

| file:line | what it is | class | note |
|---|---|---|---|
| `f2_block_kernel.cu:197` | `emulation_honorable()` — THE predicate (EmulatedFp64 ∧ `STEPPE_HAVE_EMU_TUNING`) | ENGAGES | single source; verified |
| `f2_block_kernel.cu:252` | `f2_compute_type()` — compute-type map, routes the same predicate (:253) | ENGAGES | C-2 split closed (math mode + compute type cannot disagree) |
| `f2_block_kernel.cu:278` | `engage_f2_precision()` — THE cuBLAS engage; honorable→`FP64_EMULATED_FIXEDPOINT`+EAGER+FIXED, else `PEDANTIC` + one-shot tag (:296-299) | ENGAGES | no unconditional emulated set |
| `handles.hpp:530` | `engage_solver_precision()` — cuSOLVER scope, takes `&emulation_honorable` as fn-ptr | ENGAGES | same predicate; emulated mode gated on `STEPPE_HAVE_CUSOLVER_FP64_EMULATED` (:485), degrades native today (:492-494) |
| `config.hpp:44` | `kDefaultMantissaBits = 40` — THE shared default | ENGAGES | no stray literal `40` |

### f2 GEMMs (3-GEMM groups: G=Q·Qᵀ, Vpair=V·Vᵀ, R=S·Vᵀ)

| file:line | what it computes | class | note |
|---|---|---|---|
| `f2_block_kernel.cu:357` | M0 single-block G = Q·Qᵀ | ENGAGES | `ct=f2_compute_type` (:339), engage at run_f2_gemms (:337) |
| `f2_block_kernel.cu:363` | M0 single-block Vpair = V·Vᵀ | ENGAGES | same ct / same engage |
| `f2_block_kernel.cu:370` | M0 single-block R = S·Vᵀ | ENGAGES | same ct / same engage |
| `f2_blocks_kernel.cu:245` | M4 grouped/batched G (GemmStridedBatchedEx) | ENGAGES | `ct=f2_compute_type` (:236); handle engaged once by caller |
| `f2_blocks_kernel.cu:252` | M4 grouped/batched Vpair | ENGAGES | same ct |
| `f2_blocks_kernel.cu:260` | M4 grouped/batched R | ENGAGES | same ct |
| `cuda_backend.cu:575` | caller engage (resident path) before run_f2_gemms_group (:657) | ENGAGES | `engage_f2_precision(blas_, precision)` |
| `cuda_backend.cu:781` | caller engage (streamed path) before run_f2_gemms_group (:1017) | ENGAGES | same |

### Covariance SYRK / rotation GEMM (Q = xtau·xtauᵀ / nb)

| file:line | what it computes | class | note |
|---|---|---|---|
| `cuda_backend.cu:1411` | single-model SYRK `cublasDsyrk` | ENGAGES | `MathModeScope(PEDANTIC)` (:1409) + `engage_f2_precision` (:1410), restore at :1413 |
| `cuda_backend.cu:2228` | model-batched rotation `cublasDgemmStridedBatched` | ENGAGES | `MathModeScope(PEDANTIC)` (:2226) + `engage_f2_precision` (:2227), scope close :2234 — sibling of the SYRK, engages identically |

### cuSOLVER solves (SPD inverse / SVD / GLS — native carve-out, seam present)

| file:line | what it computes | class | note |
|---|---|---|---|
| `cuda_backend.cu:1447-1461` | SPD Qinv potrf/potri | CARVE-OUT | under `engage_solver_precision(solver_, solve_precision_, &emulation_honorable)` (:1444); `solve_precision_` defaults native → honorable=false → `CUSOLVER_DEFAULT_MATH`; promotable via `set_solve_precision` |
| `cuda_backend.cu:2260-2281` | batched potrfBatched/potrsBatched Qinv | CARVE-OUT | same seam (:2259) |
| `cuda_backend.cu:1504` (`large_svd_V`) | gesvdj/gesvd large-path SVD | CARVE-OUT | native by §1.5/§4 (comment :1503); behavior-correct today. Only solve WITHOUT a `CusolverMathModeScope` — non-blocking uniformity note in §3 |
| `cuda_backend.cu:1648` | `rank_test` (SVD seed + Qinv quadratic-form chisq) | CARVE-OUT | `(void)precision`; solve-grade / ill-conditioned, matches oracle |
| `cuda_backend.cu:1734` | `rank_sweep` (per-r SVD + ALS + chisq + rankdrop) | CARVE-OUT | `(void)precision`; solve-grade |
| `cuda_backend.cu:1873` | `gls_weights` (SVD seed + ALS + constrained weight solve) | CARVE-OUT | `(void)precision`; solve-grade |
| `cuda_backend.cu:1951` | `gls_weights_loo_batched` (per-block SVD + batched ALS/weight solve) | CARVE-OUT | `(void)precision`; bit-identical to serial cuSOLVER path |

### Cancellation-sensitive elementwise carve-outs (native always)

| file:line | what it computes | class | note |
|---|---|---|---|
| `f2_block_kernel.cu:307` | `launch_f2_feeder` (Σp²−2Σpq+Σq² numerator) | CARVE-OUT | cancellation; native kernel, no math mode |
| `f2_block_kernel.cu:375` | `assemble_f2` | CARVE-OUT | cancellation |
| `cuda_backend.cu:1247` | `assemble_f4` 4-slab combine | CARVE-OUT | `(void)precision`; catastrophic-cancellation f-stat diff |
| `cuda_backend.cu:1389` / `:2214` | `launch_f4_xtau` centering (single / batched) | CARVE-OUT | cancellation; native kernel |
| `cuda_backend.cu:1325` | `assemble_f4(F2BlockTensor)` host overload | CARVE-OUT | unreachable throw stub (GPU path uses resident form); `(void)precision` |

### Entrypoint defaults (matmul-precision origin = EmulatedFp64{40})

| file:line | entrypoint | class | note |
|---|---|---|---|
| `qpadm_fit.cpp:55` | `run_impl` | ENGAGES | `EmulatedFp64{kDefaultMantissaBits}` |
| `qpadm_fit.cpp:223` | `run_qpadm(DeviceF2Blocks)` | ENGAGES | same |
| `qpadm_fit.cpp:239` | `run_qpadm(F2BlockTensor)` | ENGAGES | same |
| `qpadm_fit.cpp:258` | `run_qpwave_impl` (both qpWave overloads route here) | ENGAGES | same |
| `model_search.cpp:31` | `fit_one_model_device` | ENGAGES | same |
| `model_search.cpp:86` | `fit_shard` | ENGAGES | same |
| `model_search.cpp:294` | `run_qpadm_search` (host overload) | ENGAGES | same; device overload (:199) delegates to fit_shard, inherits |
| `config.hpp:207,214` | `Precision` struct default = EmulatedFp64{40} | ENGAGES | type-level default matches policy |
| `config.hpp:232` | `DeviceConfig::precision` default-constructed | ENGAGES | inherits emulated{40} |
| `f2_blocks_multigpu.hpp:52,76,123` | `compute_f2_blocks_multigpu*` — precision is a required param (no in-signature default) | CARVE-OUT | cannot self-default native; fed emulated{40} by the callers above |
| `cuda_backend.cu:1189-1191` | capability probe builds EmulatedFp64{40}, tests `emulation_honorable` | ENGAGES | probe matches compute path |
| `cuda_backend.cu:2683` | `solve_precision_{Fp64}` — native default | CARVE-OUT | SEPARATE cuSOLVER-solve axis (set via `set_solve_precision`, :1222); no FP64-emulated cuSOLVER in CUDA 13.0 → native is the correct documented default (comment :2675-2682). NOT a matmul op |

### Precision tags (what-is-tagged == what-ran)

| file:line | tag site | class | note |
|---|---|---|---|
| `qpadm_fit.cpp:64-67` | `run_impl` tag | ENGAGES | `kind==EmulatedFp64 && caps().emulated_fp64_honorable ? EmulatedFp64 : Fp64`; set before all early returns |
| `qpadm_fit.cpp:263-266` | qpWave tag | ENGAGES | identical predicate |
| `cuda_backend.cu:2072` | `fit_models_batched` tag (→ assemble_result :2472) | ENGAGES | identical predicate; one tag per bucket |
| `qpadm.hpp:159` | `precision_tag` field default = Fp64 | CARVE-OUT | always overwritten before return (verified no leak path). Stale comment at :158 — §3 cleanup, not a deviation |

### CpuBackend oracle (always native long-double / FP64, by design)

| file:line | method | class | note |
|---|---|---|---|
| `cpu_backend.cpp:117` | `compute_f2` `(void)precision` | CARVE-OUT | oracle; long-double cancellation-free f2 |
| `cpu_backend.cpp:223` | `compute_f2_blocks` `(void)precision` | CARVE-OUT | oracle |
| `cpu_backend.cpp:379` | `assemble_f4` `(void)precision` | CARVE-OUT | oracle |
| `cpu_backend.cpp:433` | `jackknife_cov` `(void)precision` | CARVE-OUT | oracle |
| `cpu_backend.cpp:497` | `rank_test` `(void)precision` | CARVE-OUT | oracle |
| `cpu_backend.cpp:523` | `rank_sweep` `(void)precision` | CARVE-OUT | oracle |
| `cpu_backend.cpp:615` | `gls_weights` `(void)precision` | CARVE-OUT | oracle |
| (no `capabilities()` override) | inherits base all-false (`backend.hpp:305`) | CARVE-OUT | `emulated_fp64_honorable=false` → every CpuBackend fit tags Fp64 (honest) |
| (no `fit_models_batched` override) | search throws sentinel → per-model `run_impl` | CARVE-OUT | tags Fp64 (honest) |

### Base-class no-op virtual stubs (fail/return until a backend overrides)

| file:line | stub | class |
|---|---|---|
| `backend.hpp:431,455,491,513` | f2 / compute_f2_blocks family default impls | CARVE-OUT |
| `backend.hpp:562` | `set_solve_precision` base no-op (CudaBackend overrides :1221) | CARVE-OUT |
| `backend.hpp:579,593,609,623,646,661,680,722` | assemble_f4 / jackknife_cov / rank_test / rank_sweep / gls_weights / gls_weights_loo_batched / fit_models_batched defaults | CARVE-OUT |

---

## (3) DEVIATIONS

**None.** No matmul-heavy op is hardcoded native; no site re-implements or skips
`emulation_honorable`; no matmul-precision entrypoint defaults native; no
inconsistent tag. The two items below are non-blocking cleanups, NOT deviations —
each is currently behavior-correct.

- **Non-blocking uniformity (not a deviation):** `large_svd_V`
  (`cuda_backend.cu:1504`, the gesvdj/gesvd at :1525/:1535/:1567/:1576) is the one
  cuSOLVER solve that does NOT wrap its call in a `CusolverMathModeScope`. It is
  behavior-correct today: the shared solver handle's last write is
  `CUSOLVER_DEFAULT_MATH` (restored by the preceding Cholesky scope) and
  `solve_precision_` is never honorable, so it always runs native — exactly as the
  §1.5/§4 carve-out comment (:1503) intends. For uniformity-by-construction, it
  *could* wrap its gesvd in
  `engage_solver_precision(solver_.get(), solve_precision_, &emulation_honorable)`
  so a future promotion-or-leak on the shared handle cannot silently reach it.
  Identify-only; do not edit.

- **Non-blocking doc staleness (not a deviation):** the comment at
  `qpadm.hpp:158` ("M(fit-1) is always Fp64") predates the unified
  EmulatedFp64-default policy. The field default at :159 is `Fp64` but is always
  overwritten per-run by the honest tag predicate (verified no return path leaks
  the default), so this is purely a stale comment. Suggested fix: refresh the
  comment to "Which arithmetic produced this — EmulatedFp64 when honorable, else
  native Fp64 (set per-run)." Identify-only; do not edit.

---

## (4) CONFIRMATIONS (checked and consistent)

- **ONE predicate, no re-implementation.** `emulation_honorable`
  (`f2_block_kernel.cu:197`) is the sole honorability decision. Both
  `f2_compute_type` (:253) and `engage_f2_precision` (:279) route through it, so
  the compute type and the math mode can never disagree (C-2 split closed). The
  cuSOLVER seam (`handles.hpp:533`) consults the SAME predicate by function
  pointer. The capability probe (`cuda_backend.cu:1191`) calls the SAME predicate,
  so the reported `emulated_fp64_honorable` and the runtime engage share one
  source. No site open-codes a math mode or compute type bypassing it.

- **All six f2 GEMMs engage** (M0 single-block ×3 + M4 grouped ×3), both via
  `f2_compute_type` + `engage_f2_precision`. No path runs emulated while its
  sibling is hardcoded native.

- **Both covariance accumulations engage identically.** The single-model SYRK
  (`:1411`) and the model-batched rotation GEMM (`:2228`) compute the same math
  (Q = xtau·xtauᵀ/nb) and BOTH use `MathModeScope(PEDANTIC)` + `engage_f2_precision`
  with scoped save/restore — no leak into the native cuSOLVER inverse that follows.

- **cuSOLVER solves are the documented native carve-out, with a live seam.** CUDA
  13.0 exposes no FP64-emulated cuSOLVER math mode
  (`STEPPE_HAVE_CUSOLVER_FP64_EMULATED==0`, `handles.hpp:485`), so the SPD
  inverse (single + batched), rank_test, rank_sweep, gls_weights, and
  gls_weights_loo_batched run native. The single + batched Qinv route through
  `engage_solver_precision` (the d6d3cbb promotion seam, auto-promoting when a
  future toolkit exposes the mode); `solve_precision_` defaults native and is
  promotable via `set_solve_precision`.

- **Cancellation-elementwise carve-outs are native by necessity.** The f2
  numerator (Σp²−2Σpq+Σq²), the f4 4-slab combine, and the xtau centering run as
  native kernels (never touch a math mode) because emulation cannot recover bits a
  prior subtraction annihilated.

- **Every matmul-precision entrypoint defaults EmulatedFp64{40}** from the single
  `config.hpp:44` constant — `run_impl`, both `run_qpadm`, both `run_qpwave`,
  `fit_one_model_device`, `fit_shard`, both `run_qpadm_search`, plus the
  type-level `Precision`/`DeviceConfig` defaults. The f2 precompute takes precision
  as a required parameter (cannot self-default native) and is fed that same
  emulated default. No stray literal `40`.

- **Tags are honest end-to-end.** All three tag sites use the identical predicate
  `kind==EmulatedFp64 && caps().emulated_fp64_honorable`. A degraded run tags
  `Fp64`, never a lying `EmulatedFp64`-tagged-but-ran-native (and never the
  reverse). The CpuBackend inherits all-false caps → always tags `Fp64`, honest
  because it always ran native.

- **CpuBackend is the native oracle throughout** — every method `(void)precision`,
  long-double / FP64, no cuBLAS/cuSOLVER, no `capabilities()` override. Native by
  design.

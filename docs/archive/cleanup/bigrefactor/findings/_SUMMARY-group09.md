# GROUP 9 — Constants & configuration — Rollup

Tasks: 9.1 should-be-const/constexpr left mutable · 9.2 tangled config (knobs buried in logic vs surfaced) · 9.3 positional booleans → named flags/enums.

FP64/§12 context applied: `double` GEMM alpha/beta scalars, the single deterministic statistic stream, and the Ozaki EAGER/FIXED emulation policy enums are intentional parity-load-bearing choices and were NOT flagged as defects.

## 1. Coverage

- Units in scope: 61
- Clean (no Group 9 issue): 47
- With ≥1 finding: 14

Units with findings: f2_blocks_multigpu, pchisq, small_linalg, model_search, cpu_backend, cuda_backend, f2_block_kernel, handles, pinned_buffer, qpadm_fit_kernels, stream, host_ram, filter_decision, block_sink.

## 2. Counts by task + severity

| Task | HIGH | MED | LOW | Total |
|------|------|-----|-----|-------|
| 9.1 should-be-const/constexpr | 0 | 0 | 5 | 5 |
| 9.2 tangled config | 0 | 2 | 9 | 11 |
| 9.3 positional booleans | 0 | 1 | 4 | 5 |
| **Total** | **0** | **3** | **18** | **21** |

(Two of the LOW count once each at multi-line sites; finding *lines* total 21, finding *entries* = 20: 9.1×5, 9.2×11, 9.3×5 minus the dual-line entries collapse — see §3 for the per-entry listing.)

Per-entry total: **20 findings — 0 HIGH / 3 MED / 17 LOW.**

## 3. Findings (HIGH first; none — then MED, then LOW)

### HIGH
None.

### MED (3)

- [9.3][MED] src/device/cuda/cuda_backend.cu:2450-2458 — `assemble_result(...)` passes TWO bare positional booleans (`nonspd = h_info[j] != 0`, `se_computed = se_computed[j] != 0`) at non-adjacent positions in an 18-arg call, interleaved with an `int fit_status` and a long run of `double*`. A caller transposing the two flags compiles silently and mis-classifies the model outcome / SE-presence sentinel. Suggested: named enums (`SpdStatus::NonSpd`, `SeComputed::Yes`) or an `AssembleFlags{bool nonspd; bool se_computed;}` aggregate.
- [9.2][MED] src/device/cuda/cuda_backend.cu:809 — the M5 streamed VRAM split divisor `tile_budget_b = envelope_b / 4u` is a tunable budget-partition knob hardcoded as a bare `/ 4u` mid-function, while every other VRAM lever (`kMaxVramUtilizationFraction`, `kCublasWorkspaceBytes`) is surfaced in config.hpp. Suggested: lift to a named `kStreamTileBudgetFraction` near the other VRAM knobs.
- [9.2][MED] src/core/qpadm/model_search.cpp:69-74 — the small-path dispatch envelope (`nl <= 5 && nr <= 10 && r <= 4`) is bare literals in `model_in_small_path`'s predicate; these MUST stay locked to the kernel's fixed local-array bounds (`kQpMaxNl/Nr/R`, qpadm_fit_kernels.cu:41-43). Same defect Group 5 flags as [5.3][HIGH] (drift → device buffer-overflow); recorded MED here to avoid double-counting the HIGH. Suggested: define `kQpMaxNl/Nr/R` once in a CUDA-free shared header and reference it. **Cross-ref: this same dispatch-envelope drift also appears at cuda_backend.cu:1490 (model_fits_small_path) — see LOW list — and both are the [5.3] HIGH; treat as one fix.**

### LOW (17)

9.1 — should-be-const/constexpr left mutable:
- [9.1][LOW] src/core/internal/pchisq.hpp:23,24 — `const int kMaxIter = 1000` / `const double kEps = 1e-15` are compile-time constants declared only `const`. Suggested: `constexpr`.
- [9.1][LOW] src/core/internal/pchisq.hpp:39,40,41 — `const int kMaxIter`, `const double kEps`, `const double kFpMin = 1e-300` likewise. Suggested: `constexpr`.
- [9.1][LOW] src/core/qpadm/model_search.cpp:31,86,294 — the default `Precision{EmulatedFp64, kDefaultMantissaBits}` is one knob re-constructed as a plain `const` local in three functions. Suggested: name once at TU scope (`inline constexpr Precision kFitDefaultPrecision{...}`). Overlaps Group 7 [7.2].
- [9.1][LOW] src/device/cuda/f2_block_kernel.cu:340-341 — GEMM alpha/beta `const double one/zero`; correct as `const` (must stay `double` + addressable for cublasGemmEx). Suggested: leave, or `static constexpr` if a shared named pair is wanted across the two f2 GEMM TUs. Minor.
- [9.1][LOW] src/device/cuda/pinned_buffer.cuh:162 — `void* p = const_cast<void*>(ptr)` is a write-once single-use local; could be `void* const p`. Pure hygiene.
- [9.1][LOW] src/io/filter/filter_decision.hpp:74,83,92,100,123,224 — six pure numeric/threshold predicates are `inline` but not `constexpr` though their bodies are constant-expression-eligible (allele-pair predicates correctly cannot, they use std::toupper). Suggested: optionally mark the six `constexpr`. Hygiene.

9.2 — tangled config / buried tunable knobs:
- [9.2][LOW] src/core/fstats/f2_blocks_multigpu.cpp:410 — disk-cache default `"./steppe_f2_blocks.cache"` is an inline literal in the Disk switch arm; a genuine configurable default not surfaced at file top. Suggested: file-top `constexpr char kDefaultF2CachePath[]`. Same line as Group 5 [5.4].
- [9.2][LOW] src/core/fstats/f2_blocks_multigpu.cpp:354,409 — env-var keys `"STEPPE_FORCE_TIER"` / `"STEPPE_F2_CACHE_PATH"` are inline literals at their getenv sites; a typo silently disables the override and there is no single place listing the unit's env knobs. Suggested: file-top `kEnvForceTier[]` / `kEnvF2CachePath[]`.
- [9.2][LOW] src/core/internal/pchisq.hpp:23,24,39,40,41 — iteration cap / tolerance / FP floor are block-scope locals duplicated across two helpers rather than namespace-top `constexpr`. Suggested: hoist shared `kPchisqMaxIter/Eps/FpMin`. (See 5.3.)
- [9.2][LOW] src/core/internal/small_linalg.hpp:220 — Jacobi off-diagonal convergence floor `if (off < 1e-30)` is a bare literal in the inner loop while its two siblings (`kTol`, `kMaxSweeps`, lines 179-180) are named `constexpr` at the sweep top. Suggested: hoist `1e-30` to a named `constexpr` (≈ kTol²) beside them. Overlaps Group 5 [5.1] / Group 8 [8.3].
- [9.2][LOW] src/device/cuda/cuda_backend.cu:2150-2151 — fit-path chunk budget `4 GB` free-VRAM fallback and `512 MB` headroom are inline magic sizes in `fit_one_bucket`, tuned separately from the f2 path's centralized VRAM policy. Suggested: `kFitBudgetFreeVramFallbackBytes` / `kFitBudgetHeadroomBytes` near the config VRAM knobs. (Magic-number lens: [5.2].)
- [9.2][LOW] src/device/cuda/cuda_backend.cu:1490 — bit-parity small-path dispatch envelope `nl <= 5 && nr <= 10 && r <= 4` in `model_fits_small_path`, re-typed instead of sourced from kQpMaxNl/Nr/R. Drift → kernel envelope overrun. (Correctness/overflow angle: [5.3].) Same defect as the model_search.cpp:69-74 MED above.
- [9.2][LOW] src/device/cuda/qpadm_fit_kernels.cu:124-125,156 — the one-sided Jacobi SVD convergence config (`kTol=1e-15`, `kMaxSweeps=60`, unnamed `1e-30`) is buried inside `dev_jacobi_svd_V` rather than surfaced with the kQpMax* file-top bounds (:41-45). Parity-load-bearing (tracks small_linalg.hpp:162-267) — locality issue, not magnitudes. Suggested: hoist as `kJacobiTol/MaxSweeps/OffConvergence` near kQpMax*. Overlaps [5.1].
- [9.2][LOW] src/device/cuda/qpadm_fit_kernels.cu (launch wrappers, ~15 sites incl. block=256/128/64, the 65535 gridDim.x clamp, 16×16 symmetrize tile :1445-1447, warp-round 32/31 :1332-1333) — launch-config knobs scattered into each wrapper body; not correctness-load-bearing (all grid-stride/bounds-guarded). Suggested: file-top `constexpr` block (`kBlockElemwise/Reduce/Fanout`, `kMaxGridDimX`, `kSymTile`, `kWarpSize`). Config-surfacing view of [5.1/5.3/5.5].
- [9.2][LOW] src/device/host_ram.cpp:49-51 — env-tier tokens `"resident"`/`"host"`/`"disk"` (the public STEPPE_FORCE_TIER spelling) are inline inside `resolve_output_tier` logic rather than surfaced beside the `OutputTier` enum / `ForceTier` switch they mirror. Suggested: named `constexpr` strings or a `{const char*, OutputTier}` table by the enum. Dovetails §5.1/§7.1.
- [9.2][LOW] src/device/cuda/block_sink.cu:192 — file-permission mode `0644` buried in `::open(..., 0644)` in `DiskSink::begin`, not surfaced beside `kStreamStagingSlots` at the .cuh top. Suggested: `inline constexpr mode_t kCacheFileMode = 0644`. Overlaps [5.1].

9.3 — positional booleans:
- [9.3][LOW] src/device/cpu/cpu_backend.cpp:168,169,280,281 — `core::het_correction(p_i, N.element(i,s), true)` passes a bare positional `bool valid` (4 sites). Suggested: in the SHARED primitive (f2_estimator.hpp) make validity a named enum/tag (`HetCorr::Valid`/`Invalid`) so all callers self-document; do not change from the call side alone.
- [9.3][LOW] src/device/cuda/handles.hpp:450 — `CusolverMathModeScope(handle, bool honorable)` takes a bare positional bool. Mitigated: the public path `engage_solver_precision` derives the flag from a typed `Precision`, and the param is named. Suggested: OPTIONAL `enum class Honorability { Native, Emulated }`.
- [9.3][LOW] src/device/cuda/qpadm_fit_kernels.cu:345,353,936,1120,1130,1163,1217 — `dev_als_weights(..., bool seed)`; all callers already annotate `/*seed=*/true`. Suggested: OPTIONAL scoped enum (`SeedPolicy::FromSvd`/`Preseeded`); existing inline comments are an acceptable mitigation.
- [9.3][LOW] src/device/cuda/stream.hpp:109 — `Event(bool enable_timing = false)` positional bool; well-named + defaulted, single flag. Suggested: OPTIONAL `enum class EventTiming { Disabled, Enabled }`.

## 4. Cross-cutting patterns

- **No HIGH, no real bugs.** Every Group 9 finding is hygiene/locality; the codebase is already disciplined — should-be-const locals are overwhelmingly `const`, the genuine runtime tunables (fudge, alpha, als_iterations, precision, force_tier, disk_cache_path, p2p flags) are surfaced on the `Config`/`QpAdmOptions` structs and threaded as parameters, and CUDA API calls use named enum flags (`CUBLAS_OP_*`, `cudaHostAlloc*`) rather than positional bools.

- **9.2 dominates (11 of 20)** and is one recurring shape: a genuinely configurable VALUE (cache path, env-var key, convergence floor, VRAM divisor, launch block size, file mode, tier token) lives as a bare literal in the middle of the logic that consumes it, while a sibling knob in the same unit is already a named `constexpr` at the top. The fix is uniform — hoist to a named `constexpr`/config-table at the file top / config.hpp — and many of these are the same lines Group 5 (magic numbers) flagged from the other angle, so they resolve together.

- **The dispatch-envelope drift is the only finding with teeth.** `nl<=5 && nr<=10 && r<=4` is hand-copied in THREE places — model_search.cpp:69-74, cuda_backend.cu:1490, and the kernel's own kQpMaxNl/Nr/R bounds (qpadm_fit_kernels.cu:41-43). Group 5 carries the authoritative [5.3][HIGH] (drift → device buffer overflow at scale); both Group 9 entries point at the same single fix: share kQpMaxNl/Nr/R from one CUDA-free header.

- **9.3 is uniformly low-risk** (4 entries): all are SINGLE named/defaulted bool params, never a `foo(true,false,true)` cluster, and several already carry inline `/*name=*/` annotations or a typed-factory entry point. The one with the most leverage is cuda_backend.cu:2450 (MED) — two non-adjacent bools in an 18-arg call, the one place a transposition could silently mis-classify a result.

- **VRAM-budget config is split across two homes** (the f2 path's centralized vram_budget.hpp/kMaxVramUtilizationFraction vs the fit path's inline 4 GB / 512 MB / envelope/4 at cuda_backend.cu:809,2150-2151). Consolidating the fit budgets next to the f2 VRAM knobs would put all VRAM tuning in one place — relevant at the S8/large-model scale where the fit path is the hot envelope.

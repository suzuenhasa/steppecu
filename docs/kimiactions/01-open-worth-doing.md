# steppe — Open / Worth-Doing Action Plan (High + Med)

> Scope: the **genuinely-open, non-hygiene** work for steppe — the GPU/CUDA-13/Blackwell (sm_120)
> reimplementation of ADMIXTOOLS 2 f-statistics + qpAdm, tolerance/bit-validated against AT2 goldens
> on real AADR. This plan is the **High + Med** bucket from `docs/kimireview/ASSESSMENT.md` §3,
> **excluding CI** (CI is its own plan — `docs/kimiactions/02-ci-plan.md`; cross-referenced where the
> sanitizer/compute-sanitizer seam feeds it).
>
> Everything below has been re-verified against the **actual code at HEAD** (branch
> `phase2-fit-engine`), not the Kimi snapshot — every claim cites a real `file:line` and the
> as-built state. Items the assessment **rejected** in §4 (2nd statistic stream, NCCL, a multi-GPU
> parity gate, excising the oracle, `steppe::Index`, a `JsonWriter` reshape, capability
> pure-virtuals, kernel fuzzing, a mock backend) are **out of scope** and untouched here.

## Framing

steppe is a **single-GPU shipping product whose acceptance gate is PARITY** (multi-GPU is parked).
The just-finished HIGH/MED/LOW source-hygiene campaign (`a2f9d64..HEAD`) already closed the
naming/IWYU/dead-code/init items. What remains in this bucket is a different class of work:
**fault-taxonomy honesty, output-correctness, doc-vs-as-built truth, and supply-chain hardening** —
the items that make the product trustworthy at release, not just internally clean.

Two product invariants shape every "optimal" below:

1. **The §12 PARITY LAW is frozen and load-bearing.** Each line is acceptance criteria, so a *false*
   line in it (the cuSOLVER determinism claim, Cluster A) is a genuine integrity gap, not a typo.
2. **Nothing builds locally.** The dev box is an RTX 2070 / CUDA 11.8 with no FP64-emulation; all
   compile/golden verification runs on the ephemeral vast.ai box (`ssh box5090`, single RTX 5090,
   CUDA 13.1). Doc-only and host-only (pyproject regex, CSV-escape unit) work is the only thing
   checkable off-box. Each item's **Verify** says which lane it lands in.

Every item here is **golden-neutral by construction** — they change behavior only on error/edge
paths the small committed goldens (`golden_fit0` / `fit1_NRBIG` / `rot`) never exercise — *except*
Cluster A, whose one bounded parity touch is the loose-tier NRBIG cuSOLVER `gesvd` path and is fully
golden-gated. No item requires regenerating an AT2 golden.

> **Cross-cut additions (from `docs/kimiactions/04-crosscut-vs-kimi.md`).** Three items below
> originate in **our own X1–X7 cross-cutting review** (`docs/release_cleanup/crosscutting/`), *not* the
> Kimi assessment — they are the §3 "shop-window / first-impression" gaps the campaign deferred and the
> `kimiactions` triage chain dropped: **Cluster F** (F1 — relocate six design-doc comment-essays to
> `docs/` behind 1-line `// see docs/…` pointers, cross-cut gap G2; F2 — write the *accepted but undone*
> `cuda_backend.cu` single-seam header note + a DECISIONS record for the parked structural debt,
> cross-cut gap G3) and **D3** (correct the stale `run_qpdstat` "Part B, not yet implemented" docstring
> on the now-shipped genotype-D, cross-cut gap G5). All three are **comment / doc / structure-only and
> golden-neutral by construction** (see each item's Parity-risk). Each carries a
> `(cross-cut gap GN, from docs/kimiactions/04)` provenance tag.

---

## Cluster A — §12 parity-law integrity gap (cuSOLVER determinism) — **HIGH**

### A1. Wire `cusolverDnSetDeterministicMode`, assert it, and correct the §12 claim

**What's needed.** `docs/architecture.md` §12 (the cuSOLVER-determinism bullet, verified present in
the `## 12` block around L788) asserts two things that are **false at HEAD**:

1. "`cusolverDnSetDeterministicMode` **is enabled** on the statistic handle."
2. "**CI asserts** the rank-test routine is one of [the covered set]."

Verification: `grep -rn cusolverDnSetDeterministic src/ include/ bindings/` returns **0 hits**, and no
test reads the mode. This is a false promise *inside the frozen parity law*. The doc also **conflates
two different SVDs**: the same bullet says "the Jacobi SVD used in the rank test *is* covered" — but
the **primary bit-exact 9-pop golden's** rank-test SVD is a **custom on-device fixed-order Jacobi**
(`launch_qpadm_rank_via_jacobi` / `dev_jacobi_svd_V`), **not** cuSOLVER, so
`cusolverDnSetDeterministicMode` is irrelevant to it. Only the **loose-tier large/NRBIG** path
(`large_svd_V`, `cuda_backend.cu:4241`) actually calls cuSOLVER `gesvdj`/`gesvd` — the routines the
mode governs (`gesvdj` unconditionally; `gesvd` only for `m ≥ n`, which `large_svd_V`'s orientation
rule already guarantees, `cuda_backend.cu:4221-4226`). The dispatch predicate is single-sourced in
`gesvdj_applicable` (`cuda_backend.cu:4158`), and `svd_path == 2` (NRBIG) is the only path that hits
`cusolverDnDgesvd` (`cuda_backend.cu:4262`).

**Optimal end state.** Resolution **(a): WIRED + asserted + doc-true** (not doc-only).

- `cusolverDnSetDeterministicMode(h_, CUSOLVER_DETERMINISTIC_MODE)` set **once** in the
  `CusolverDnHandle` ctor, immediately after `cusolverDnCreate(&h_)`
  (`handles.hpp:335-337`), via `CUSOLVER_CHECK`, with a `// §12` citation noting it governs only
  `gesvd`/`gesvdj` (not `potrf`/`potri`/`getrf`). The process holds **one** cuSOLVER dense handle, so
  the ctor is the correct single source (sticky handle state, mirroring the cuBLAS math-mode idiom).
- A `[[nodiscard]] cusolverDeterministicMode_t deterministic_mode() const` getter
  (`cusolverDnGetDeterministicMode`) for the assertion.
- A **device-gated gtest** that asserts the §12 promise literally: post-construction the getter
  returns `CUSOLVER_DETERMINISTIC_MODE`, and the rank-test routine is provably in the covered set
  (`gesvdj` always; `gesvd` because the orientation rule guarantees `m ≥ n`).
- §12 rewritten to the as-built truth: the mode is wired on the cuSOLVER **dense** handle; it covers
  **only** the large/NRBIG `gesvd`/`gesvdj` rank test; the **primary 9-pop bit-parity** rank test is a
  custom on-device fixed-order Jacobi that is deterministic **by construction** (not by this API);
  `potrf`/`potri`/`getrf` stay native FP64 and outside the covered set; the `m ≥ n` precondition is
  met by the orientation rule. The echoing references stay accurate.

Net: the parity law contains no false claim, and the wire **defends against a future cuSOLVER
default-mode change silently de-determinizing the NRBIG `gesvd`**, with a provably bounded blast
radius.

**Steps.**
1. **On box5090**, confirm the cuSOLVER 13.x API in `/usr/local/cuda/include/cusolverDn.h`: exact
   signatures `cusolverDnSetDeterministicMode(cusolverDnHandle_t, cusolverDeterministicMode_t)` /
   `...Get...`, the enum spelling (`CUSOLVER_DETERMINISTIC_MODE` vs `..._NOT_DETERMINISTIC_MODE`), and
   **the DEFAULT mode**. The default decides the parity framing: if default is already deterministic,
   the wire is behaviour-neutral/defensive; if not, it changes NRBIG `gesvd` from possibly-atomic to
   deterministic (the desired direction).
2. Add the wire in the `CusolverDnHandle` ctor (`handles.hpp:337`, right after `cusolverDnCreate`).
3. Add the `deterministic_mode()` getter to `CusolverDnHandle`.
4. Add the device-gated assertion to `tests/reference/test_handles.cu` (already the `handles_unit`
   CUDA TU): mode getter + routine-coverage. Optionally echo near `test_qpadm_parity.cu`'s `svd_path`
   checks.
5. Build Release on box5090; `ctest`: `handles_unit` (new assertion) + `test_qpadm_parity` (9-pop
   byte-identical — structurally immune; NRBIG within tolerance + bit-identical across the two-run
   check the test already prints).
6. Rewrite the §12 bullet to the as-built truth; verify the echoing cells read consistently.
7. Commit: one for wire+getter+test, one for the doc-sync; reference "§12 parity-law integrity".

**Effort.** **S** (XS code, the precision is in the doc rewrite + the API/default check).

**Parity-risk.** **LOW, precisely bounded.** The bit-exact 9-pop golden **cannot move**: its rank
test is the custom on-device Jacobi (`cuda_backend.cu`, `dev_jacobi_svd_V`), which the cuSOLVER mode
does not touch — guarded by the existing byte-identical `test_qpadm_parity` 9-pop gate.
`potrf`/`potri`/`getrf`/`potrfBatched`/`potrsBatched` are **not** in the determinism-affected set, so
Qinv/Cholesky/LU cannot move. The only governed path is the loose-tier NRBIG `gesvd` (and any
`gesvdj`-applicable large model): if the cuSOLVER default is already deterministic (step 1), the wire
is a no-op; if not, forcing determinism moves NRBIG `gesvd` in the **reproducible** direction, bounded
by the NRBIG loose-tier golden gate plus the test's two-run bit-identity check. Worst case the NRBIG
value shifts **within tolerance**, caught by `ctest` before merge. Resolution (b) (doc-only) is
strictly worse if the default is non-deterministic — it leaves NRBIG `gesvd` run-to-run flaky, the
exact failure §12 exists to prevent.

**Verify.** On box5090: (1) `grep -rn cusolverDnSetDeterministicMode src/` returns exactly **1** hit
(the ctor); (2) `ctest handles_unit` passes the getter + routine-coverage assertions; (3) `ctest
test_qpadm_parity`: 9-pop byte-identical, NRBIG within tolerance + bit-identical across the two runs;
(4) the §12 bullet no longer claims an unwired call; (5) the "CI asserts the rank-test routine"
promise is now satisfied by the device test in the GPU reference lane (complementary to the
host-only CI item in `02-ci-plan.md`).

**Files.** `docs/architecture.md` (§12 cuSOLVER bullet, ~L788 + echoing cells L22/L31/L949);
`src/device/cuda/handles.hpp:330-355` (ctor L335-337, getter); `src/device/cuda/cuda_backend.cu`
(`gesvdj_applicable` :4158, `large_svd_V` :4241, `gesvd` :4262, orientation rule :4221-4226, on-device
Jacobi path, single solver handle); `src/device/cuda/check.cuh` (`CUSOLVER_CHECK`);
`tests/reference/test_handles.cu`; `tests/reference/test_qpadm_parity.cu`.

---

## Cluster B — error-handling & exit-code taxonomy — **HIGH / MED**

> Three fault-taxonomy-honesty fixes, all **golden-neutral by construction** (they change behaviour
> only on error paths the goldens never hit). Land in ascending blast-radius: **B1 (torn-write) →
> B2 (OOM exit-code) → B3 (capability-query)**. The first two are app-only and success-path-neutral;
> B3 touches the device seam `backend.hpp` and the frozen fit chain, so it is the most golden-gated.

### B1. Check post-write stream state after every emit; return `kExitIoError` on a torn / short write — **HIGH**

**What's needed.** Every command opens `std::ofstream out(config.out_file(), binary|trunc)`, checks
**only** `if (!out)` (open failure → `kExitIoError`), calls a `void emit_*(out, …)`, and **never
re-checks the stream**. On a full disk, quota, or closed pipe, the failbit/badbit is set silently, the
`ofstream` dtor swallows it, and the process **exits 0 having written a truncated file**. Confirmed
across the emit sites (cmd_f4 / cmd_f3 / cmd_qpadm / cmd_qpwave / cmd_f4ratio / cmd_qpdstat ×2 /
cmd_dates / cmd_rotate / cmd_fstat_sweep ×2, plus the shared `emit_to_destination` in
`cmd_qpgraph.cpp:170`). Only `cmd_fstat_sweep` even calls `out.flush()`, and still does not check
`good()` after. The stdout branch is unchecked too (closed-pipe / `EPIPE` writing to `| head`). For a
product whose deliverable **is files**, a silent truncated output that exits 0 is a real correctness
hole.

**Optimal end state.** One shared emit-to-destination helper that **all** commands route through,
doing **open → write → flush → verify**. `cmd_qpgraph.cpp:170-190`'s `emit_to_destination(config,
prefix, write_fn)` is already the right shape (returns `std::optional<int>`: `nullopt` = ok, else the
exit code). Promote it into a shared app header (`src/app/cmd_emit.hpp` or extend `result_emit.hpp`)
and add, after `write(out, fmt)`: `out.flush(); if (!out.good()) { fprintf(stderr, "steppe <cmd>:
write failed (disk full / short write): <path>\n"); return kExitIoError; }`. For the stdout branch:
`write(std::cout, fmt); std::cout.flush(); if (!std::cout.good()) return kExitIoError;`. Every
`cmd_*.cpp` emit block collapses to one call — removing ~13 hand-rolled open+check blocks (the DRY win
the campaign's dedup did not reach).

**Steps.**
1. Lift `emit_to_destination` (`cmd_qpgraph.cpp:170-190`) into `src/app/cmd_emit.hpp` as a template
   taking `(config, command-prefix, write-lambda)`.
2. Add the post-write verification (`flush()` + `good()` on the file branch; `cout.flush()` + `good()`
   on the stdout branch).
3. Replace each command's inline `if (out_file().empty()) { emit_*(cout,…) } else { ofstream out;
   if(!out) return kExitIoError; emit_*(out,…) }` with a single `if (auto rc =
   emit_to_destination(config, "<cmd>", [&](std::ostream& os, OutputFormat fmt){ emit_*(os, fmt, …);
   })) return *rc;`.
4. Keep the existing format-parse + `kExitInvalidConfig` handling inside the helper (already there).
5. Add a fault-injection test: open `/dev/full` as `--out` (open succeeds, write fails) → assert exit
   `== kExitIoError (4)`; and a normal `--out` → exit 0, byte-identical content.

**Effort.** **M** (~13 call sites + helper lift + test).

**Parity-risk.** **None for goldens.** On the success path the emitted bytes are unchanged (flush is a
content no-op; `good()` is true) so every golden/CLI byte-diff still matches. The check only fires on
the error path. One subtle behaviour change: writing to a closed stdout pipe now yields exit 4 instead
of 0 — but the CLI/golden tests read an open pipe or a real file, so their exit stays 0. Guard = the
existing CLI suite stays green + the new `/dev/full` test.

**Verify.** On box5090: full `ctest` green (content + exit 0 unchanged); new test: `steppe f4 … --out
/dev/full` exits 4 with a "write failed" line, `steppe f4 … --out tmp.csv` exits 0 and byte-matches the
golden; grep confirms no emit site writes a stream without a trailing `good()`/`flush()` check.

**Files.** `src/app/cmd_qpgraph.cpp` (`emit_to_destination` :170, ofstream :181); `src/app/result_emit.hpp`
(or new `cmd_emit.hpp`); `cmd_f4.cpp` `cmd_f3.cpp` `cmd_qpadm.cpp` `cmd_qpwave.cpp` `cmd_f4ratio.cpp`
`cmd_qpdstat.cpp` `cmd_dates.cpp` `cmd_rotate.cpp` `cmd_fstat_sweep.cpp` (all under `src/app/`).

### B2. Map `cudaErrorMemoryAllocation → kExitDeviceOom (3)` at the catch handlers — **MED**

**What's needed.** Every command's `catch (const std::exception& e)` returns `cfg::kExitRuntimeError`
(confirmed: `catch (const std::exception` in all 13 command TUs + `main.cpp`; the
`return …kExitRuntimeError` pattern appears 25× across `src/app/*.cpp`). A real device OOM
(`DeviceBuffer`/`cudaMalloc` throws `CudaError(cudaErrorMemoryAllocation)`) therefore exits **5**, not
the dedicated **3**. The infra is **half-built**: `exit_code.hpp:54-70` already maps
`Status::DeviceOom → kExitDeviceOom (3)`, and `CudaError::status()` (`check.cuh:77`) exposes the
`cudaError_t`. **Critical layering correction to the assessment:** `CudaError` / `cudaError_t` /
`cudaErrorMemoryAllocation` live in `src/device/cuda/check.cuh`, a CUDA `.cuh` **private to
steppe_device**, and `src/app` is **CUDA-free** by the §4 arch-grep gate (grep confirms no `app/*.cpp`
includes `cuda_runtime`/`check.cuh`). So the translator **cannot** live purely in the app — the
assessment's "tiny `exception_to_status()` helper in the app" is wrong on that load-bearing point.

**Optimal end state.** A **CUDA-free declaration + device-layer definition** translator:

- In a CUDA-free device-seam header (`steppe::device`, e.g. extend `src/device/resources.hpp` or a new
  `src/device/fault_status.hpp`):
  `[[nodiscard]] std::optional<steppe::Status> device_fault_status(const std::exception& e) noexcept;`
- Definition in a `steppe_device` TU that *can* include `check.cuh` + `cuda_runtime.h` (new
  `src/device/cuda/fault_status.cu`, or appended to `cuda_backend.cu`):
  `if (auto* ce = dynamic_cast<const CudaError*>(&e)) return ce->status()==cudaErrorMemoryAllocation ?
  DeviceOom : nullopt;` plus `CublasError → CUBLAS_STATUS_ALLOC_FAILED` and `CusolverError` alloc-fail
  → `DeviceOom`; else `std::nullopt`. `dynamic_cast` is safe (single statically-linked executable, one
  typeinfo).
- A thin app helper: `int exit_code_for_caught(const std::exception& e){ if (auto s =
  device::device_fault_status(e)) return cfg::exit_code_for(*s); return cfg::kExitRuntimeError; }`.
  `steppe_app` already links `steppe::device`, so the symbol resolves while the app stays CUDA-free.

Every catch keeps its own command-scoped `fprintf(stderr, "steppe <cmd>: device error: %s", e.what())`
but returns `exit_code_for_caught(e)`. Host `std::bad_alloc` stays 5 (host RAM, not device OOM) —
correct.

**Steps.**
1. Add the CUDA-free decl to a `steppe::device` seam header (include `<optional>` + `steppe/error.hpp`
   only, **no CUDA**).
2. Add the definition in a `steppe_device` TU that includes `check.cuh`; map the three alloc-failed
   statuses to `Status::DeviceOom`, else `nullopt`.
3. Add the app helper `exit_code_for_caught` (a `cmd_common.hpp` included by all `cmd_*.cpp`, or a free
   fn).
4. Mechanically replace `return cfg::kExitRuntimeError;` inside each `catch (const std::exception& e)`
   with `return exit_code_for_caught(e);` (keep the stderr line). Route `main.cpp` through it too.
5. Leave the `resources.gpus.empty()` early-returns as `kExitRuntimeError` — a no-device fault, not
   OOM, and correct.
6. Add a device-side guard test (a `.cu` in `tests/reference`): construct
   `CudaError(cudaErrorMemoryAllocation,…)` → assert `device_fault_status → DeviceOom` and
   `exit_code_for(*) == kExitDeviceOom (3)`; plain `std::runtime_error` → `nullopt → 5`.

**Effort.** **M** (the layering plumbing — CUDA-free decl + device-TU def — makes it M, not XS).

**Parity-risk.** **None for goldens.** Exit codes differ only on the OOM error path, which the small
goldens never hit (they complete, exit 0). The only behaviour change is **5 → 3** on a genuine device
OOM. No statistic, no emitted byte, no §12 reduction touched. Guarded by the new device-side
translation test.

**Verify.** On box5090: golden/CLI suite still all exit 0; new test asserts `DeviceOom → 3`,
`runtime_error → 5`. Optionally force an OOM (over-budget P on real AADR) and confirm `echo $? == 3`
with a clear "device error" line. Grep: no `catch (const std::exception` body hard-returns
`kExitRuntimeError` except the deliberate non-fault early-returns.

**Files.** `src/core/config/exit_code.hpp` (:54-70); `src/device/cuda/check.cuh` (`CudaError` :68,
`status()` :77, `CublasError` :87, `CusolverError` :126); `src/device/resources.hpp` (or new
`fault_status.hpp`); new `src/device/cuda/fault_status.cu` (or append to `cuda_backend.cu`);
`src/app/main.cpp` + all 13 `src/app/cmd_*.cpp`.

### B3. Replace try/catch-as-capability-detection with an explicit backend capability query — **MED**

**What's needed.** Two sites use `catch (const std::runtime_error&)` to detect a missing backend
override (the "not implemented by this backend" sentinel from the non-pure base virtuals in
`backend.hpp`):

- **Site A** (`model_search.cpp:136`): `be.fit_models_batched(…)` in try/catch → fallback to
  `fit_models_batched_default`. This is **LIVE**: `CpuBackend` deliberately does **not** override
  `fit_models_batched` (confirmed: zero `fit_models_batched` overrides in `cpu_backend.cpp`), so the
  sentinel fires on the oracle path every search.
- **Site B** (`qpadm_fit.cpp:187`): `run_rank_sweep` + `run_popdrop` in try/catch → leave
  rankdrop/popdrop columns empty. This catch is now **effectively DEAD**: **both** backends override
  `rank_sweep` (`cpu_backend.cpp:1581`; `cuda_backend.cu`'s `rank_sweep` override) *and* `gls_weights`,
  so the sentinel no longer fires on either real backend. The in-code comment ("the CudaBackend
  implements it in the NEXT phase… until then a backend without the override throws") is **stale** —
  the GPU deliverable landed. The catch now only **masks a genuine `std::runtime_error` thrown from
  inside a real, overridden sweep** — a latent bug-hider that silently blanks rankdrop/popdrop with no
  diagnostic.

**Optimal end state.** Explicit capability-query virtuals on `ComputeBackend`, **reusing the idiom
that already exists** in `backend.hpp` (a non-pure virtual with a default base body: `capabilities()`
at :1840, `batched_dispatch_count()` at :1852):

- `virtual bool provides_batched_fit() const { return false; }` — `CudaBackend` overrides → `true`
  (next to its `fit_models_batched` override).
- `virtual bool provides_rank_sweep() const { return false; }` — **both** backends override → `true`
  (`cpu_backend.cpp:1581`, `cuda_backend.cu`'s `rank_sweep`).

Site A → `if (be.provides_batched_fit()) … else fit_models_batched_default(…);` (no try/catch).
Site B → `if (be.provides_rank_sweep()) { run_rank_sweep…; run_popdrop… }` (no try/catch) — so a
genuine numerical throw inside the sweep now **propagates** as a fault instead of being swallowed.
Delete the stale "NEXT phase" comment.

> The assessment's "`std::optional` override" alternative (make `fit_models_batched` return
> `std::optional`, `nullopt` = not-implemented) is cleaner in theory but **mutates a signature in the
> M(fit-6) frozen contract §2.2** (`backend.hpp` `fit_models_batched` :1810); the **additive
> capability-bool is lighter** and does not touch the frozen ABI. This is **not** the §4-rejected
> "capability pure-virtual" — these are *non-pure* defaults, the sanctioned pattern.

**Steps.**
1. Add the two non-pure `bool` virtuals to `backend.hpp` adjacent to the sentinel virtuals
   (`rank_sweep` :1714, `fit_models_batched` :1810), documented as the explicit replacement for
   sentinel-catch detection.
2. Override `provides_rank_sweep() → true` in both `cpu_backend.cpp` and `cuda_backend.cu`; override
   `provides_batched_fit() → true` in `cuda_backend.cu` only.
3. Rewrite `model_search.cpp:132-142` to branch on `be.provides_batched_fit()` — drop the try/catch.
4. Rewrite `qpadm_fit.cpp:163-190` to branch on `be.provides_rank_sweep()` — drop the try/catch;
   delete/refresh the stale comment.
5. Add a self-consistency unit test: `provides_X()==true ⇒ X()` does not throw the sentinel;
   `==false ⇒` it does (CpuBackend: `provides_rank_sweep` true, `provides_batched_fit` false;
   CudaBackend: both true). This pins the bool to the actual override so they cannot drift.
6. Keep `run_rank_sweep`/`run_popdrop` (`ranktest.hpp`) signatures unchanged — only the call-site
   guard changes.

**Effort.** **M** (touches the device seam header + the frozen fit chain, so golden-gated).

**Parity-risk.** **None on the goldens.** Site A: identical dispatch (CpuBackend still →
`fit_models_batched_default`, CudaBackend still → the batched override); `golden_rot.json` +
`batched_dispatch_count` assertions unchanged. Site B: `rank_sweep` *is* overridden on both backends,
so the branch is taken and rankdrop/popdrop populate exactly as today. The **only** change is that a
genuine `runtime_error` inside an overridden sweep now propagates (a fault) rather than silently
blanking columns — strictly more correct and the intent of the fix.

**Verify.** On box5090: `test_qpadm_parity.cu`, `test_qpadm_rotation.cu`, `test_qpwave_parity.cu` —
rankdrop/popdrop/weights byte-match unchanged; new capability self-consistency unit test passes; grep:
no remaining `catch (const std::runtime_error&)` used for capability detection in `core/qpadm` (the
top-level fault catches in `src/app` are correct and stay).

**Files.** `src/device/backend.hpp` (rank_sweep :1714, fit_models_batched :1810, capabilities :1840,
batched_dispatch_count :1852); `src/device/cpu/cpu_backend.cpp` (rank_sweep :1581);
`src/device/cuda/cuda_backend.cu`; `src/core/qpadm/model_search.cpp:136`;
`src/core/qpadm/qpadm_fit.cpp:187` (+ stale comment :151-162); `src/core/qpadm/ranktest.hpp`.

---

## Cluster C — IO / output correctness + dedup — **MED**

> Three independent, parity-safe-by-construction items. Two findings reshape the naive Kimi framing
> and **must travel with the work** (see C2/C3). The C2 `csv_field`/`json_quote` exposure is the seam
> a future `app/output_sink.hpp` (the deferred §9A "finish the outputter") would build on — a
> down-payment, not throwaway.

### C1. Extract the byte-identical genotype decode front-end into one `core::` helper

**What's needed.** The same ~13-line front-end is copy-pasted in **four** sites: `dstat.cpp:125-138`,
`qpfstats.cpp:236-249`, `dates.cpp:129-142`, and `extract_f2_core.cpp:97-110`. Each opens
`io::GenoReader(geno)`, reads `header().format`, `records_present()`, builds a `PopSelection`, then
`read_ind_partition(fmt, ind, sel, n)` + `read_snp_table(fmt, snp, SIZE_MAX)` +
`M0 = min(header().n_snp, snptab.count)` + `core::read_canonical_tile(reader, part, backend, 0, M0)`.
The dedup campaign deliberately skipped this — `read_canonical_tile.cpp` only dedup'd the format
**switch**, not the reader+ind+snp+M0 **wrapper** around it. It's a real **parity-divergence risk**: a
future ploidy/filter/M0 tweak must be edited in 4 places in lockstep or the genotype tools silently
diverge. The shared boundary is exactly `{tile, snptab, part, fmt, M0}` — each caller **diverges
afterward** (dstat/qpfstats/dates force diploid + autosome-keep; extract does per-sample ploidy +
regime-B filter), so the helper must **stop at the tile/snptab handoff** and not absorb the decode.

**Optimal end state.** A new core TU `src/core/stats/genotype_front_end.{hpp,cpp}` (`steppe::core`):

```cpp
struct GenotypeFrontEnd { io::GenotypeTile tile; io::SnpTable snptab;
                          io::IndPartition part; io::GenoFormat fmt; std::size_t M0; };
GenotypeFrontEnd read_genotype_front_end(const std::string& geno, const std::string& snp,
                                         const std::string& ind, const io::PopSelection& sel,
                                         ComputeBackend& backend);
// + a convenience overload taking std::span<const std::string> pop_labels that builds the
//   Explicit selection (collapsing the 3-line sel-build duplicated in dstat/qpfstats/dates too).
```

It lives beside `read_canonical_tile` (the module already documents itself as "the single place the
four genotype-path tools turn an open `GenoReader` into the canonical tile") and **wraps** it. Callers
keep binding their own `backend` local (reused for decode) and pass it in — the helper does not own the
backend. All four callers shrink to one call; this becomes the single edit point for any read-time
front-end change.

**Steps.**
1. Add `genotype_front_end.hpp`: the POD + two decls; fwd-declare `ComputeBackend` (mirror
   `read_canonical_tile.hpp`); include `io/geno_reader.hpp`, `io/genotype_source.hpp`,
   `io/ind_reader.hpp`, `io/snp_reader.hpp`.
2. Add `genotype_front_end.cpp` as the exact lift: `GenoReader reader(geno)`; `fmt =
   reader.header().format`; `n = reader.records_present()`; `part = read_ind_partition(fmt, ind, sel,
   n)`; `snptab = read_snp_table(fmt, snp, SIZE_MAX)`; `M0 = min(reader.header().n_snp, snptab.count)`;
   `tile = read_canonical_tile(reader, part, backend, 0, M0)`. The label overload builds
   `PopSelection{Explicit, labels}` then delegates.
3. Register `stats/genotype_front_end.cpp` in `src/core/CMakeLists.txt` next to
   `stats/read_canonical_tile.cpp` (:139); **no new link edges** (core already links `steppe::io`
   at :153).
4. Replace `dstat.cpp:125-138`, `qpfstats.cpp:236-249`, `dates.cpp:129-142` with the label-overload
   call (each binds `backend` first, passes it in); preserve every downstream line unchanged (P/M
   extraction, `sample_ploidy`, `decode_af`/`decode_af_compact_autosome`).
5. Replace `extract_f2_core.cpp:97-110` with the `PopSelection`-form call; keep
   `validate_explicit_pops(sel, fe.part)` **in the caller** right after the helper returns (it needs
   `part`).
6. Re-run the four genotype goldens on box5090.

**Effort.** **M** (mechanical, but 4 callers + new TU + CMake + golden re-runs).

**Parity-risk.** **None — behaviour-neutral mechanical lift.** The produced tile/snptab/part bytes are
identical (same reader, same `SIZE_MAX`, same `M0`, same `read_canonical_tile` call), so `decode_af`
is fed identical input and no golden can move. **One ordering nuance:** extract currently calls
`validate_explicit_pops` **between** `read_ind_partition` and `read_snp_table`; folding both reads into
the helper means `.snp` is read before `validate` throws on bad-pops input. This only reorders which
error fires on a malformed-pops input — an error path, never a golden — and is **fully avoided** by
keeping `validate` in the caller after the helper returns `part`.

**Verify.** On box5090: `STEPPE_THOROUGH ctest` green for `qpdstat_geno`, `qpfstats_smooth/fused`,
`dates_parity`, and the extract-f2 → qpadm CLI gate (`test_cli_extract_qpadm`). Grep: the
`GenoReader`+`read_ind_partition`+`read_snp_table`+`M0` quartet now appears **once** (in
`genotype_front_end.cpp`); the four callers each call `read_genotype_front_end`.

**Files.** `src/core/stats/dstat.cpp`, `src/core/stats/qpfstats.cpp`, `src/core/stats/dates.cpp`,
`src/app/extract_f2_core.cpp`, `src/core/stats/read_canonical_tile.{cpp,hpp}`,
`src/io/genotype_source.cpp`, `src/core/CMakeLists.txt` (:139, :153).

### C2. Route the bypassing emitters through a shared CSV-field / JSON-quote layer

**What's needed.** `cmd_dates.cpp` (`emit_dates`), `cmd_qpgraph.cpp` (`emit` / `emit_search`), and
`cmd_fstat_sweep.cpp` (`emit_sweep`) hand-roll their CSV/JSON and emit **raw** labels: e.g.
`cmd_dates.cpp` writes `target` bare into CSV and `"` + target + `"` into JSON with **no** escaping;
`cmd_qpgraph.cpp` writes `admix_from/to` bare; `cmd_fstat_sweep.cpp` writes `resolver.label_at` bare. A
pop name with a comma or quote corrupts the CSV/JSON.

> **Critical correction to the assessment** (verified in-code): `app::csv_quote` (`result_emit.cpp:89`)
> **always** wraps in double-quotes and is byte-gated to the qpadm/f4 goldens (`golden_fit0_f4.csv`
> quotes every string column — confirmed `csv_quote` is the verbatim primitive across `result_emit.cpp`).
> The dates/qpgraph/sweep emitters instead emit **bare** labels and their CLI tests compare bare tokens
> with quote-naive splitters (`test_cli_qpgraph.cpp` `cells[col["from"]] == "pSteppe"`). So **blindly
> applying always-quote `csv_quote` would MOVE these outputs and BREAK `test_cli_qpgraph`.**
> `csv_quote`/`json_quote` are also file-local (`result_emit.cpp:89`/`:100`, anonymous namespace, not
> in any header), so they must be exposed.

**Optimal end state.** `result_emit.hpp` exports two reusable primitives: (1) the existing
`json_quote` (promoted out of the anonymous namespace, unchanged behaviour), and (2) a **new**
`std::string csv_field(const std::string& s, char sep)` doing **RFC-4180 conditional** quoting —
returns the bare string when it contains none of `{sep, '"', '\n', '\r'}`, otherwise wraps + doubles
quotes. The three bypassing emitters route every **label** cell through `csv_field` (CSV) and
`json_quote` (JSON). For all real population names this is **byte-identical** to today's bare output
(no golden, no CLI test moves), while a pathological name now escapes correctly. `csv_quote` stays
exactly as-is (always-quote) for the qpadm/f4 family whose goldens require it. **Number formatting is
left untouched** (out of scope; the §4-rejected `DoubleFormatter`/`JsonWriter` reshape; the
golden-matched 17-digit path in `result_emit` must not move).

**Steps.**
1. In `result_emit.hpp` add decls: `std::string json_quote(const std::string&);` and `std::string
   csv_field(const std::string&, char sep);` (document `csv_field` as RFC-4180 conditional).
2. In `result_emit.cpp` move `json_quote` (:100) to namespace scope; add `csv_field` at namespace
   scope; leave `csv_quote` anonymous/unchanged.
3. `cmd_dates.cpp`: route `target`/`src1`/`src2` through `csv_field(…, sep)` in CSV and `json_quote`
   in JSON; leave the `%.6f` number helper alone.
4. `cmd_qpgraph.cpp` `emit()`: route `admix_from/to`, `type`, `edge_from/to`, `worst_pop2/3` through
   `csv_field` (CSV) and `json_quote` (JSON); `emit_search()`: same for edge-string labels,
   `best.hash`, best-graph from/to labels.
5. `cmd_fstat_sweep.cpp` `emit_sweep()`: route `resolver.label_at(…)` through `csv_field` (CSV) and
   `json_quote` (JSON).
6. Add a unit/CLI test feeding a synthetic pop label with a comma and a quote: assert the CSV field is
   quoted+escaped and the JSON string is escaped, and a normal label stays byte-for-byte bare.

**Effort.** **S**.

**Parity-risk.** **Designed byte-identical for every real pop name.** `csv_field` returns bare unless a
special char is present; `json_quote` on a label with no quote/backslash/control is identical to the
current manual `"` + label + `"`. So no committed golden and no CLI test moves. The dates/qpgraph CLI
tests are the live gate (no committed byte-golden for these emitters; the qpgraph `.csv`/`.rds` goldens
validate result **values**, not emitter bytes). The trap to avoid — and the reason **not** to use
always-quote `csv_quote` — is `test_cli_qpgraph`'s quote-naive splitter; `csv_field`'s conditional
quoting sidesteps it.

**Verify.** On box5090: `test_cli_dates` + `test_cli_qpgraph` stay green; new comma/quote unit test
passes; manual `steppe qpgraph/dates … --format csv|json` byte-matches pre-change output on a normal
pop set; grep confirms `cmd_dates`/`cmd_qpgraph`/`cmd_fstat_sweep` no longer emit a raw label adjacent
to a separator.

**Files.** `src/app/result_emit.hpp`, `src/app/result_emit.cpp` (`csv_quote` :89, `json_quote` :100),
`src/app/cmd_dates.cpp`, `src/app/cmd_qpgraph.cpp`, `src/app/cmd_fstat_sweep.cpp`.

### C3. Unify / document `--out` (file) vs `--out-dir` (directory) for extract-f2 and qpfstats

**What's needed.** Three flag definitions are inconsistent (all verified in `cli_parse.cpp`):
`add_output_flags` (:97) binds `--out` → `out_file` ("Output file (stdout if omitted)") for the 12
file-emitting commands; **extract-f2** (:703) binds `--out` → `out_dir` ("Output f2_blocks dir") — a
**directory** under the same flag name; **qpfstats** (:654) binds `--out-dir` → `out_dir`. So both
extract-f2 and qpfstats write a directory but name the flag differently, and `--out` means "file" in 12
commands but "directory" in extract-f2. The `cli_args` fields are already distinct (`out_dir`,
`out_file`). extract-f2's `--out` is **load-bearing**: `test_cli_extract_qpadm.cpp` passes `--out`, and
`docs/RUN-SHEET.md` + `docs/RESUME.md` document `extract-f2 --out` — so an outright rename is
UX-breaking (the assessment's caveat is correct).

**Optimal end state.** extract-f2's directory flag becomes a CLI11 **multi-name** option
`"--out-dir,--out"` (canonical `--out-dir` shown first; `--out` retained as a back-compat alias bound
to the same `out_dir` setter). Result: `--out` consistently means a **file**, `--out-dir` consistently
means a **directory** (canonical on both extract-f2 and qpfstats), while every existing `extract-f2
--out …`, the CLI test, and the doc scripts keep working unchanged. The file `--out` help is sharpened
to "Output **FILE** (stdout if omitted)"; README/RUN-SHEET examples migrate to `--out-dir`.

**Steps.**
1. In `cli_parse.cpp` at the extract-f2 binder (:703) change the name from `"--out"` to
   `"--out-dir,--out"` and update help to "Output … f2_blocks **DIRECTORY** (f2.bin + pops.txt +
   meta.json)".
2. Sharpen `add_output_flags` `--out` help (:97-98) to "Output **FILE** (stdout if omitted)".
3. (Optional polish) migrate `docs/RUN-SHEET.md` + `docs/RESUME.md` extract-f2 examples to `--out-dir`;
   note the alias in README.
4. Confirm no within-subcommand flag collision (extract-f2 does not call `add_output_flags`; qpfstats
   already owns `--out-dir` — verified).

**Effort.** **S**.

**Parity-risk.** **None.** CLI surface only — no compute, no statistic stream, no golden. The single
gated path (`test_cli_extract_qpadm` uses `--out`) keeps working because `--out` remains an alias; both
names resolve to `out_dir`.

**Verify.** On box5090: `steppe extract-f2 --help` shows `--out-dir` (with `--out` alias); both `--out
DIR` and `--out-dir DIR` produce the f2 dir; `test_cli_extract_qpadm` + `test_cli_qpfstats` stay green;
`steppe qpadm --help` shows `--out` as a FILE.

**Files.** `src/app/cli_parse.cpp` (:97, :654, :703), `src/core/config/cli_args.hpp` (`out_dir`,
`out_file`), `docs/RUN-SHEET.md`, `docs/RESUME.md`.

---

## Cluster E — build / supply-chain hygiene — **MED** (E2 feeds the CI plan)

> Independent of the other clusters. E2 (sanitizer wiring) is the **input** to several lanes in
> `docs/kimiactions/02-ci-plan.md` (asan/ubsan host lane + compute-sanitizer GPU lane), so it has
> outsized leverage; land it before/with the CI work.

### E1. Pin the CPM `EXPECTED_HASH` for fetched deps (or honestly document the deliberate no-pin)

**What's needed.** `cmake/CPM.cmake:16-17` sets `CPM_DOWNLOAD_VERSION 0.42.3` plus a **fake**
`CPM_HASH_SUM` (`97e3f10f…` — verified) that is **never referenced again**; the comment at :28 even
admits "We intentionally do NOT pin EXPECTED_HASH: the placeholder above is illustrative". The actual
`file(DOWNLOAD …CPM.cmake… STATUS)` (:34-37) passes **no `EXPECTED_HASH`**, so the downloaded script —
which is then `include()`'d and **executes arbitrary CMake** — is unverified, while the tree reads as
if it were pinned. The two real fetched deps are **CLI11 2.4.2** (`src/app/CMakeLists.txt:21`,
`CPMAddPackage("gh:CLIUtils/CLI11@2.4.2")` — a movable tag) and **nanobind v2.4.0**
(`bindings/CMakeLists.txt:41`, `GIT_TAG v2.4.0` — also movable). Both sit behind default-OFF options
(`STEPPE_BUILD_CLI`/`STEPPE_BUILD_PYTHON`) and CPM only bootstraps when `find_package` first fails.
GoogleTest is `find_package`-only — no CPM fetch. **Blast radius is exactly the opt-in dev CLI / wheel
build**; the core/device/tests build never reaches CPM.

**Optimal end state.** The CPM bootstrap download is integrity-verified — `file(DOWNLOAD)` carries
`EXPECTED_HASH SHA256=<real hash of the v0.42.3 CPM.cmake release asset>`, and the misleading dead
`CPM_HASH_SUM` placeholder is **gone**. This pin matters most because the bytes are `include()`'d and
run as CMake. The two `CPMAddPackage` deps are pinned to **immutable refs** — full upstream commit SHAs
for CLI11 v2.4.2 and nanobind v2.4.0 — so a re-pointed tag cannot change what builds; **or**, accepting
the dev-only blast radius, an in-file SECURITY note stating tag-pin + `CPM_SOURCE_CACHE` is the
deliberate posture with full-SHA hardening tracked to M(release). Either way the file comment **tells
the truth** (no "illustrative placeholder" language).

**Steps.**
1. On box5090: `curl -fsSL
   https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.42.3/CPM.cmake | sha256sum`.
2. In `cmake/CPM.cmake` delete the dead `CPM_HASH_SUM` (:17); add `EXPECTED_HASH SHA256=<computed>` to
   the `file(DOWNLOAD)` call (:34-37). Note `file(DOWNLOAD)+EXPECTED_HASH` skips the network when a
   cached file already matches and hard-fails on mismatch — keep the `if(NOT EXISTS)` guard but let the
   hash verify a pre-placed `CPM_SOURCE_CACHE` copy too (or add a `file(SHA256 …)` check for that
   branch).
3. Rewrite the comments (:11-14, :27-30) to describe the verified-download posture (drop "illustrative
   placeholder").
4. (Optional, recommended for release) resolve CLI11 v2.4.2 and nanobind v2.4.0 to full commit SHAs and
   pin those in `CPMAddPackage` (`src/app/CMakeLists.txt:21`, `bindings/CMakeLists.txt:41`); or add the
   SECURITY note deferring SHA-pin to M(release).

**Effort.** **S**.

**Parity-risk.** **None.** Build-provenance only — no compute, no CUDA, no golden. The verify-on-fetch
hard-fails a tampered/changed download (fail-fast, not a silent wrong artifact). The only failure mode
is a mistyped hash → an immediate, loud CMake error at configure, caught before any build.

**Verify.** On box5090: a clean `STEPPE_BUILD_CLI=ON` configure with no `CPM_SOURCE_CACHE` downloads
CPM, **verifies the hash**, and proceeds; corrupting one byte of a cached CPM copy now hard-fails
configure (previously silent). `grep -n EXPECTED_HASH cmake/CPM.cmake` returns the real pin; the fake
`CPM_HASH_SUM` is gone. If SHA-pinning the deps: a clean build resolves CLI11/nanobind to the pinned
commits.

**Files.** `cmake/CPM.cmake` (hash :17, download :34-37, comments :11-14/:27-30),
`src/app/CMakeLists.txt:21`, `bindings/CMakeLists.txt:41`.

### E2. Wire `STEPPE_SANITIZER` (declared but never consumed) — **MED, feeds `02-ci-plan.md`**

**What's needed.** `STEPPE_SANITIZER` is **declared but dead**: `cmake/SteppeOptions.cmake:45` sets the
cache var (`"empty | asan;ubsan | compute"`) with the comment "Wired in `cmake/SteppeSanitizers.cmake`
at a later milestone; declared here so the preset cache vars resolve." Verification: `grep -rn
STEPPE_SANITIZER` over `CMakeLists.txt cmake/ src/ tests/ CMakePresets.json` returns **only** that one
declaration — **`cmake/SteppeSanitizers.cmake` does not exist** and nothing consumes the var. So a
preset that sets `STEPPE_SANITIZER=asan;ubsan` (or the `compute` value the CI compute-sanitizer lane
wants) silently does nothing. This is the seam the CI plan's asan/ubsan **host** lane and
compute-sanitizer **GPU** lane both need to exist before they can be turned on.

**Optimal end state.** `cmake/SteppeSanitizers.cmake` exists and is the single place that translates
the `STEPPE_SANITIZER` value into flags, `include()`d from the top-level `CMakeLists.txt`:

- `asan;ubsan` (host TUs only): add `-fsanitize=address,undefined -fno-omit-frame-pointer` to
  **non-CUDA** targets (and the matching link flags). **Must exclude device/`.cu` objects** —
  host-side ASan does not instrument CUDA device code and mixing it with the CUDA runtime is fragile;
  gate via `$<COMPILE_LANGUAGE:CXX>` generator expressions so only C++ TUs get the flags.
- `compute`: a **no-compile-flag** value — it selects the runtime `compute-sanitizer` wrapper used by
  the CI GPU lane (`compute-sanitizer --tool memcheck ctest …`), not a codegen change. Wiring it here
  is just making the value legal + documented so the preset and the CI lane agree on the spelling.
- The Release/CI parity build is **never** sanitized (the perf/golden lane must stay clean FP64); the
  sanitizer is a dev/CI-only debug-build overlay.

**Steps.**
1. Create `cmake/SteppeSanitizers.cmake`: parse `STEPPE_SANITIZER` (`asan;ubsan` → host C++ compile +
   link flags via `$<COMPILE_LANGUAGE:CXX>`; `compute` → set a documented marker the CI lane reads, no
   codegen flags; empty → no-op). Fail loud on an unrecognized value.
2. `include(cmake/SteppeSanitizers.cmake)` from the top-level `CMakeLists.txt` after
   `SteppeOptions.cmake`; update the `SteppeOptions.cmake:45` comment to point at the now-real file.
3. Confirm the dev preset(s) in `CMakePresets.json` can set `STEPPE_SANITIZER` and that the Release/CI
   parity presets leave it empty.
4. Hand off the lane invocations (asan/ubsan host `ctest`; `compute-sanitizer` GPU `ctest`) to
   `02-ci-plan.md` — this item delivers the **CMake seam**; CI delivers the **runner**.

**Effort.** **S** (one CMake module + an `include` + a comment fix).

**Parity-risk.** **None for the shipping build.** Sanitizer flags apply only to **debug** sanitizer
builds, never the Release/CI parity build, so no golden or perf number moves. The `compute` value is a
runtime wrapper, not a codegen change. The one care-point is **not** letting host ASan flags reach
`.cu`/device objects — handled by the `$<COMPILE_LANGUAGE:CXX>` gate; verified by inspecting the
compile DB.

**Verify.** On box5090: configure with `-DSTEPPE_SANITIZER=asan;ubsan` and confirm
`compile_commands.json` shows `-fsanitize=address,undefined` on **C++ TUs only** (no `.cu`); a
deliberately-leaky host unit under that build trips ASan and fails `ctest`. Configure with
`-DSTEPPE_SANITIZER=compute` and confirm it is accepted (no codegen flags) and the documented marker is
set. The default (empty) build is byte-for-byte the current build. Grep: `STEPPE_SANITIZER` is now
consumed in `cmake/SteppeSanitizers.cmake`, not just declared.

**Files.** new `cmake/SteppeSanitizers.cmake`, `cmake/SteppeOptions.cmake:45`, top-level
`CMakeLists.txt` (add the `include`), `CMakePresets.json`. Cross-ref: `docs/kimiactions/02-ci-plan.md`
(asan/ubsan host lane + compute-sanitizer GPU lane consume this seam).

---

## Cluster D — doc-vs-as-built honesty + versioning — **MED (polish)**

> Lowest functional urgency (pure narrative + build-metadata; no compute, no golden), but
> **portfolio-facing**: a senior reviewer reads `architecture.md` first and will hit every lie. Land
> these two together — D2's optional generated `version.hpp` makes one of D1's §16 claims true.

### D1. Doc-sync `architecture.md` to the as-built graph, public surface, and CMake floor

**What's needed.** Three concrete, independently-verified doc-vs-code lies, all corrected in **one
editorial pass** over `docs/architecture.md`:

- **(a) `core → io` edge.** §4 (the "Dependency-direction rule" paragraph, verified at L239) asserts
  allowed edges are exactly `app/bindings → api → core → device` and that "`io` … depends on nothing in
  `core`/`device`. The *app* layer is the only place that wires `io` output into compute." **False as
  built:** `src/core/CMakeLists.txt:153` links `steppe::io` **PRIVATE** into `steppe_core` (the comment
  there even says so), because the genotype-path tools now live **in** core
  (`stats/dstat.cpp` :117, `stats/qpfstats.cpp` :123, `stats/dates.cpp` :130,
  `stats/read_canonical_tile.cpp` :139) and consume the io reader directly. The §4 tree label
  `steppe_core -- pure host C++ (NO CUDA, NO I/O)` is now wrong on the `NO I/O` clause.
- **(b) C-ABI relabel.** §16 / ADR-0008 (verified L906-914) state the installed public boundary **is**
  a C ABI: opaque handles `steppe_f2_blocks_t*`/`steppe_qpadm_result_t*`, functions returning
  `steppe_status_t`, "No `std::` types, no templates, no exceptions cross this boundary." **As built
  there is NO C ABI:** every `include/steppe/*.hpp` is full C++ (`std::`/templates/classes), there is no
  `steppe_status_t`, no `extern "C"`, no `src/c_api/`. The actual surface is exactly the "same-toolchain
  C++ convenience layer" that §16 says *may* wrap the (nonexistent) C ABI. Per the assessment, the C ABI
  is **deferred to M(abi-1)** — the doc must say so. The stale comment at `include/steppe/error.hpp:7`
  promising a `steppe_status_t` C ABI should be corrected too.
- **(c) CMake floor.** Four cells say `≥ 3.30` while the build floor is **3.28**: `architecture.md:20`
  (build-system table), `:84` (§3 pin table), `:302` (§6 `cmake_minimum_required(VERSION 3.30)`), `:892`
  (§15 `cmake.version = ">=3.30"`). As built: `CMakeLists.txt:14` (3.28), `CMakePresets.json` (3.28),
  `pyproject.toml:59` (`>=3.28`). `architecture.md:114` was **already** corrected to 3.28 — so this is
  finishing a partial fix.

**Optimal end state.** `architecture.md` reflects the as-built graph and surface, phrased so the
original design intent survives as *rationale*, not erased:
- (a) §4 allowed-edges **adds** the deliberate `core → io` genotype-path edge (framed as an intentional
  in-core front-end); the §4 tree drops `NO I/O`; §5 S0 notes the genotype path can originate in core.
- (b) §16/ADR-0008 states plainly: as-built, `include/steppe/` is a same-toolchain C++ convenience
  layer (value/RAII, `std::expected` internally), **not** the ADR-0008 C ABI; the true cross-toolchain
  C ABI (`steppe_status_t`, opaque handles) is **deferred to M(abi-1)**. The SemVer paragraph (L914) is
  reworded so today's stability promise tracks the C++ surface, with the C-ABI promise marked future.
- (c) all four cells read `≥ 3.28`, with a one-clause note (mirroring `CMakeLists.txt:15`) that 3.30
  remains the aspirational floor for the future named-Blackwell release matrix.

**Steps.** Edit `architecture.md` L239 (add `core → io`); L~139 tree (drop `NO I/O`); §5 S0 owner note;
§16 ADR-0008 sentence + the C-ABI bullets (L906-912) → "same-toolchain C++ convenience layer, C ABI
deferred to M(abi-1)"; §16 SemVer paragraph (L914); fix `include/steppe/error.hpp:7`; change the four
CMake cells (L20, L84, L302, L892) to `≥ 3.28` + the future-matrix note; proofread the diff; keep the
existing `[STALE - …]` annotation convention.

> **Adjacent staleness to sweep in the same pass** (cheap, same file, keeps the doc honest): §4 tree
> marks `src/app/ # (planned, P3)` but app/access/extract are **built**, and the tree omits
> `src/access`, `src/extract`, `bindings/`; the §6 top-level CMake snippet shows
> `add_subdirectory(third_party)` but there is **no** `third_party/` (CPM lives in `cmake/`) and omits
> the `STEPPE_BUILD_CLI`/access/extract guards; the §15 pyproject snippet lists unsuffixed
> `nvidia-cuda-runtime>=13` + `requires-python >=3.10` + target `steppe._core`, all diverged from the
> as-built `pyproject.toml` (cu13-suffixed extras, `>=3.9`, target `_core`).

**Effort.** **S**.

**Parity-risk.** **None.** Markdown + one comment; no compiled source, no CUDA, no golden. There is no
arch-grep/layering CI gate at HEAD (the §4 "CI architecture test" is itself unbuilt — confirmed no
`tests/*arch*`), so the doc edit cannot move any gate; it simply makes the doc agree with the link that
already compiles green.

**Verify.** No build. Re-grep: `grep -n '3.30' docs/architecture.md` returns only intentional
future-matrix mentions; `grep -rn 'steppe_status_t\|opaque handle' docs/architecture.md include/steppe/`
no longer asserts a present-tense C ABI; the §4 allowed-edges line names `core → io`. Cross-check each
cell against the real value (`CMakeLists.txt:14`, `CMakePresets.json`, `pyproject.toml:59`,
`src/core/CMakeLists.txt:153`). Reviewer diff-read is the gate.

**Files.** `docs/architecture.md` (L20, L84, L114, L139, L208, L239, L251, L302, L316-321, L883-889,
L892, L906-914), `include/steppe/error.hpp:7`, `src/core/CMakeLists.txt:153`.

### D2. Single-source the project version (kill the static `pyproject` literal drift)

**What's needed.** Four `0.1.0` literals; **one pair is a genuine drift hazard**, the rest are guarded
fallbacks. The intended single source is `CMakeLists.txt` `project(... VERSION 0.1.0)`
(`architecture.md:914` already prescribes this):
- `CMakeLists.txt` `project(VERSION 0.1.0)` — the intended single source.
- `pyproject.toml:25` `version = "0.1.0"` — a **static literal, fully independent of CMake**: **this is
  THE drift.** (`architecture.md:879` already documents the intended `dynamic = ["version"]`, so the
  as-built pyproject **regressed** to a static literal — this fix re-converges code to the doc.)
- `src/app/cli_parse.cpp:54` `#define STEPPE_VERSION "0.1.0"` — a **fallback**; the build already
  injects the real value via `src/app/CMakeLists.txt:65`
  (`STEPPE_VERSION="${PROJECT_VERSION}"`). So the CLI/extract paths **are** single-sourced at build
  time; the literal only matters for a standalone TU compile but can masquerade as a real version if it
  drifts.
- `bindings/steppe/__init__.py:27` `__version__ = "0.1.0"` — a **fallback**; the file reads
  `importlib.metadata.version("steppe")` first, so an installed wheel reports the pyproject (soon
  CMake-derived) version; the literal is only the in-tree dev-import path.

**Optimal end state.** `CMakeLists.txt` `project(VERSION …)` is the **sole** authority. The wheel
version is derived from it at build time via scikit-build-core's regex metadata provider
(`[project] dynamic = ["version"]` + `[tool.scikit-build.metadata.version]` `provider =
"scikit_build_core.metadata.regex"`, `input = "CMakeLists.txt"`), so pyproject carries **no** version
literal. CLI/extract `STEPPE_VERSION` continues to flow from `${PROJECT_VERSION}`; the standalone-compile
fallback literals are demoted to an obvious dev sentinel (e.g. `0.0.0+unknown`) so a stale fallback can
never impersonate a release. Python `__version__` resolves from the now-CMake-derived wheel.
**Optionally** a generated `include/steppe/version.hpp.in` (`configure_file` from `PROJECT_VERSION`)
replaces the compile-definition + literals, making `architecture.md:914`'s "`version.hpp` generated
from `project(VERSION)`" **true**. Bumping `CMakeLists.txt` VERSION then moves every surface.

**Steps.**
1. `pyproject.toml`: replace `version = "0.1.0"` (:25) with `dynamic = ["version"]`.
2. Add `[tool.scikit-build.metadata.version]`: `provider = "scikit_build_core.metadata.regex"`,
   `input = "CMakeLists.txt"`, `regex` capturing the `project(... VERSION <x.y.z> ...)` group
   `value`.
3. Keep `CMakeLists.txt` `project(VERSION …)` as the single source.
4. (Optional, doc-conformant) add `include/steppe/version.hpp.in` (`#define STEPPE_VERSION
   "@PROJECT_VERSION@"`) + a top-level `configure_file`; point `cli_parse.cpp` / `cmd_extract_f2.cpp` at
   it and drop their literal fallbacks (or demote to `0.0.0+unknown`), retiring the
   `src/app/CMakeLists.txt:65` / `src/extract/CMakeLists.txt:42` compile-defs.
5. Demote the dev fallbacks (`cli_parse.cpp:54`, `bindings/steppe/__init__.py:27`) to a clearly
   non-release sentinel.
6. Update `architecture.md` §15 pyproject snippet (L883-889 + L892) and confirm §16:914 is now
   accurate.

**Effort.** **S**.

**Parity-risk.** **None.** Build-metadata + version-string plumbing; no compute, no CUDA, no golden.
CLI/extract already derive `STEPPE_VERSION` from `PROJECT_VERSION`, so only the *reported string* can
change, never math. The one failure mode is a mis-written regex → scikit-build-core can't find the
version → a hard, immediate metadata error at build (fail-fast, not a silent wrong value), caught below.

**Verify.** **Host-checkable (no GPU):** `python -c "import re; print(re.search(r'<regex>',
open('CMakeLists.txt').read()).group('value'))"` prints the project VERSION; the scikit-build-core
metadata hook reports the same. On box5090 after a Release wheel build: `python -c "import steppe;
print(steppe.__version__)"` and `steppe --version` both equal `CMakeLists.txt` VERSION. Drift test: bump
to `0.1.1`, rebuild, confirm both surfaces move and **no other file** was edited; revert.

**Files.** `pyproject.toml` (:25, :59), `CMakeLists.txt` (`project(VERSION …)`), optional
`include/steppe/version.hpp.in`, `src/app/cli_parse.cpp:54`, `src/app/cmd_extract_f2.cpp`,
`bindings/steppe/__init__.py:27`, `docs/architecture.md` (L879, L883-889, L892, L914).

### D3. Correct the stale "Part B, not yet implemented" docstring on the SHIPPED genotype-D — **MED** *(cross-cut gap G5, from docs/kimiactions/04)*

**What's needed.** The `run_qpdstat` nanobind docstring ends with a **false promise on a shipped
feature**: `bindings/module.cpp:1107` reads `"The normalized-D magnitude needs a genotype prefix (Part
B, not yet implemented)."`. **Part B ships.** The genotype-path normalized-D is bound right below at
`module.cpp:1109` (`m.def("run_dstat", &run_dstat_py, …)`, C++ impl `run_dstat_py` at
`module.cpp:647`) and is exposed in the public Python facade as `steppe.dstat(...)`
(`bindings/steppe/__init__.py:818`, calling `_core.run_dstat` at `:850`); the CLI also ships it
(`qpdstat --prefix PREFIX.{geno,snp,ind}`). So `help(steppe._core.run_qpdstat)` tells a user the exact
capability they already have is unimplemented — a user-facing doc-vs-as-built lie of the same class as
Cluster D's `architecture.md` items. (Verified at HEAD: the `__init__.py` `qpdstat` wrapper already
points readers at the right place — `__init__.py:803` says "normalized-D, Part B; or the CLI `qpdstat
--prefix …`" — so only the C-extension docstring is stale.)

**Optimal end state.** The trailing clause of the `run_qpdstat` docstring no longer claims "not yet
implemented"; instead it points at the shipped path, mirroring the wording the `__init__.py:803`
wrapper already uses: e.g. `"…The normalized-D magnitude needs the genotype path — call
steppe.dstat(prefix, quadruples) (binding run_dstat below), or the CLI qpdstat --prefix
PREFIX.{geno,snp,ind}."`. Nothing else changes — `run_qpdstat` (Part A, f2-path f4) keeps its current
behaviour and the rest of its docstring verbatim.

**Steps.**
1. Edit the trailing sentence of the `run_qpdstat` docstring literal at `bindings/module.cpp:1101-1107`
   to remove "(Part B, not yet implemented)" and cross-reference `run_dstat` / `steppe.dstat` / the
   `qpdstat --prefix` CLI as the shipped genotype-D path.
2. (Cheap consistency sweep, same pass) confirm no sibling docstring still calls genotype-D
   unimplemented; the `__init__.py:786-816` `qpdstat` wrapper is already correct — leave it.

**Effort.** **XS** (one docstring literal).

**Parity-risk.** **None — golden-neutral.** A help-string edit only; no compute, no CUDA, no emitted
result byte, no §12 path. No statistic, golden, or CLI byte-output moves; only `help()` / docstring
introspection text changes.

**Verify.** On box5090 after a Release wheel build: `python -c "import steppe;
help(steppe._core.run_qpdstat)"` no longer contains "not yet implemented" and names `steppe.dstat` /
`qpdstat --prefix`; `steppe.dstat(...)` runs (already shipping). Grep: `grep -rn "not yet implemented"
bindings/module.cpp` returns 0 hits. No golden re-run (docstring-only).

**Files.** `bindings/module.cpp` (`run_qpdstat` docstring :1101-1107, `run_dstat` binding :1109,
`run_dstat_py` :647), `bindings/steppe/__init__.py` (:786-816 `qpdstat` wrapper, `:818` `dstat`,
`:850` `_core.run_dstat`).

---

## Cluster F — source-comment altitude / first-impression — **LOW (polish, portfolio-facing)**

> **Provenance: this entire cluster is a cross-cutting addition from our own X1–X7 review**
> (`docs/release_cleanup/crosscutting/`, reconciled in `docs/kimiactions/04-crosscut-vs-kimi.md` §3 +
> §5) — the "shop-window professionalism" layer the Kimi `ASSESSMENT §2` over-credited to the campaign
> and `§3` dropped before it reached the action plans. Both items are **comment / doc / structure-only
> and golden-neutral by construction**: they move *prose* out of the source and leave the *code* and its
> load-bearing parity invariants exactly where they are. Functionally the lowest urgency in this plan,
> but the highest first-impression signal — a senior reviewer opens these files first. Sibling cluster
> to D (doc-honesty); land them together. Cross-references the comment-hygiene pass in
> `03-low-polish.md` (G1/G14/G15 ticket-ID + ALL-CAPS scrub) — F1's relocation and that scrub touch the
> same "comment voice" surface and should be sequenced as one editorial sweep.

### F1. Relocate the six design-doc comment-essays to `docs/`, leave a 1-line `// see docs/…` pointer — **LOW** *(cross-cut gap G2, from docs/kimiactions/04)*

**What's needed.** Six source files carry multi-paragraph **design-doc essays** — 30-to-64-line comment
banners explaining *rationale* — wrapped around a few lines of code. All six are present and re-verified
at HEAD:

- `src/core/fstats/f2_blocks_multigpu.cpp` — the **"§4 COMBINE GATE"** banner (~L236-298, a `=====`
  ASCII-boxed essay re-deriving the four-term gate predicate, WHY-it-moved, and the four terms) guarding
  **one** call: `const bool use_p2p = select_p2p_combine(resources, G);` (`:300`). The predicate itself
  is already single-homed in `select_p2p_combine` (`:126`).
- `src/device/p2p_combine.hpp:1-33` — a 33-line file-header essay (CUDA-free-contract + bit-identity
  rationale) above the `#ifndef` (`:34`).
- `src/device/cuda/dstat_kernel.cu:1-49` — a 49-line header essay (statistic derivation, the SNP-tiled
  reuse rationale, the golden-exact carve-out) above `#include <cuda_runtime.h>` (`:50`). Kimi cited this
  one exactly.
- `src/core/internal/decode_af.hpp:1-58` — a 58-line header essay (the full decode convention +
  AT2 pseudo-haploid auto-detection narrative) above the `#ifndef` (`:59`).
- `src/device/cuda/cuda_backend.cu` — the **39-line Stream essay** (~L5560-5586: legacy-default-stream
  overlap, teardown declaration-order reasoning) above the one-line member `Stream stream_{};` (`~:5586`).
- `src/core/qpadm/model_search.cpp:167-193` — the `TODO(multigpu-host-bounce)` `=====`-boxed
  perf-narrative banner (known-problem / root-cause / fix-to-eliminate) above `struct F2Replication`
  (`:194`). Kimi cited this at `09 §3`.

Why: portfolio-facing and parity-neutral. The source reads as a private engineering design journal
rather than shipping code; the *rationale* belongs in `docs/`, the *code* should stand on a 1-line
pointer. **Critical nuance to carry through the move:** several essays state **load-bearing §12 parity
invariants** (the multigpu/p2p "bit-identical to the host-staged combine & single-GPU reference; the
transport only moves bytes", the dstat-kernel "golden-exact cancellation carve-out", the decode-af
"bit-for-bit oracle" convention). Those one-sentence *invariants* must **stay in the code** as a terse
1-line guard-comment so a future editor cannot silently break parity; only the multi-paragraph
*derivation* relocates.

**Optimal end state.** Each essay's prose lives in a `docs/design/` page (one per essay, or folded into
the architecture.md section it already cross-references — the multigpu/p2p two share `§11.4`, decode-af
maps to `§5/§13`), and each source site keeps a **2-to-3-line** stub: a 1-line `// see
docs/design/<page>.md (architecture.md §N)` pointer **plus** the single load-bearing parity-invariant
sentence retained inline. Net: the six files open with code or a terse contract line, the design
narrative is one click away and version-controlled in `docs/`, and the §12 invariants are *more*
prominent (no longer buried in a 60-line wall).

**Steps.**
1. For each of the six, create/append the destination doc page under `docs/design/` (or the named
   architecture.md section), moving the prose verbatim so no rationale is lost; add a back-link to the
   source file.
2. Replace the in-source banner with the 2-3 line stub: `// see docs/design/<page>.md` + the retained
   1-line parity invariant (for the three parity-bearing essays). For the non-parity essays
   (cuda_backend.cu Stream teardown-ordering, model_search.cpp host-bounce TODO) keep a 1-line
   `// see …` plus, for the Stream member, the single load-bearing "declaration order is load-bearing at
   teardown" sentence so the reverse-destruction contract is not lost.
3. Keep `select_p2p_combine`'s single-home status intact — the f2_blocks_multigpu stub points at the doc
   *and* names `select_p2p_combine` as the predicate's single source.
4. For the `model_search.cpp` `TODO(multigpu-host-bounce)`: move the perf narrative to a
   `docs/design/multigpu-host-bounce.md` known-issue page; keep the 1-line `// DEFERRED: multi-GPU
   rotation host-bounce-capped on no-P2P; single-GPU is the supported path — see docs/…` so the
   deferral remains visible at the call site.
5. Build Release on box5090 — comment-only edits are structurally byte-neutral; a clean recompile +
   green `ctest` confirms no code line was disturbed.

**Effort.** **M** (mechanical, but six source files + six destination doc pages + the careful
invariant-retention split).

**Parity-risk.** **None — comment-only relocation.** No statement, expression, kernel, or `<<<>>>` is
touched; the produced bytes are identical, so no golden, CLI byte-output, or §12 path can move. The one
care-point is **not dropping a load-bearing parity invariant in the move** — explicitly mitigated by
retaining each invariant as a 1-line in-code guard-comment (Step 2). Guard = a clean box5090 recompile +
full `ctest` (immune by construction, run as a sanity check).

**Verify.** On box5090: Release build green; full `ctest` unchanged (comment-only). Grep: each of the six
files now opens with code or a ≤3-line stub (`grep -c '^//' ` on each header drops sharply); each stub
contains a `see docs/` pointer; the three parity-bearing sites still contain their 1-line invariant
(`grep -n 'bit-identical\|golden-exact\|bit-for-bit'` still hits the source). No golden re-run.

**Files.** `src/core/fstats/f2_blocks_multigpu.cpp` (:236-298, predicate call :300, `select_p2p_combine`
:126); `src/device/p2p_combine.hpp` (:1-33); `src/device/cuda/dstat_kernel.cu` (:1-49);
`src/core/internal/decode_af.hpp` (:1-58); `src/device/cuda/cuda_backend.cu` (Stream essay ~:5560-5586,
member `Stream stream_{}` ~:5586); `src/core/qpadm/model_search.cpp` (:167-193); new `docs/design/*.md`
pages (+ optional folds into `docs/architecture.md` §11.4 / §5 / §13).

### F2. Write the accepted `cuda_backend.cu` single-seam header note + a DECISIONS record for the parked structural debt — **LOW** *(cross-cut gap G3, from docs/kimiactions/04)*

**What's needed.** `src/device/cuda/cuda_backend.cu` is a **5,679-LOC god-file** (55 `ComputeBackend`
overrides + co-located launch wrappers; LOC re-confirmed at HEAD via `wc -l`). The **cross-TU split is
REJECTED by decision** (`ASSESSMENT §5.5`: a single C++ class cannot be partial across translation
units, and the kernels are co-located deliberately) — that is settled and stays rejected. **But the
*accepted half* is still undone:** the §5.5 decision also called for a **one-line header note** so the
file reads as a deliberate *choice* rather than neglect, and the file's header banner (`:1-30`,
re-verified) carries no such note — grep for `single.seam|single TU|intentionally.*single` returns 0
hits. This item does that accepted down-payment, and **records two parked decisions** so the omission
reads as weighed, not missed.

**Optimal end state.** The `cuda_backend.cu` header banner gains a one-line note —
`// CudaBackend is INTENTIONALLY a single seam TU: a C++ class cannot be partial across translation
units, and the launch wrappers / kernels are co-located here on purpose (architecture.md §4, §7;
ASSESSMENT §5.5). The cross-TU split is rejected by decision.` — so a reviewer opening the largest file
sees the size is a *choice*. A short **DECISIONS** block (in this plan, and optionally mirrored as a
2-line note near the header) records:
- **Rejected (settled):** the Kimi cross-TU class split (`ASSESSMENT §5.5`) — a class can't be partial
  across TUs; kernels co-located deliberately.
- **Parked (never separately evaluated):** X3's own proposed decomposition — a **thin aggregate header +
  per-subsystem shared `.inc` includes**, still **one class** — which *sidesteps the exact "can't be
  partial" objection* that `ASSESSMENT` used to reject the Kimi proposal. It was never weighed on its
  own merits. Recorded as: "the `.inc`-include split (single class, no cross-TU partitioning) is viable
  and was not weighed; parked, not rejected."

> **DECISIONS — `cuda_backend.cu` god-file (cross-cut gap G3).** (1) Cross-TU split → **REJECTED**
> (`ASSESSMENT §5.5`). (2) One-line "intentionally a single seam TU" header note → **ACCEPTED, do now**
> (this item). (3) X3's thin-aggregate-header + per-subsystem `.inc`-include split (single class,
> sidesteps the partial-class objection) → **PARKED, never separately evaluated** — revisit only if the
> file's size becomes a maintenance burden in practice.

**Steps.**
1. Add the one-line "intentionally a single seam TU; cross-TU split rejected by decision (§5.5)" note to
   the `cuda_backend.cu` header banner (`:1-30`, adjacent to the existing §4/§8 framing).
2. Record the DECISIONS block above in this plan (done here) so the rejected + parked structural
   decisions are minuted; optionally mirror the 1-line `.inc`-split-parked note in the header.
3. No code move — this item is explicitly *not* the split; it makes the existing single-TU shape read
   as intentional.

**Effort.** **XS** (one header-comment line + a DECISIONS minute).

**Parity-risk.** **None — comment/doc only.** A header-comment line; no statement, kernel, or §12 path
touched. No golden, CLI byte, or perf number moves.

**Verify.** On box5090: Release build green (comment-only). Grep: `grep -ni "single seam TU"
src/device/cuda/cuda_backend.cu` now returns the header note; the DECISIONS block is present in this
plan. No golden re-run.

**Files.** `src/device/cuda/cuda_backend.cu` (header banner :1-30); this plan (DECISIONS minute);
reference `docs/kimireview/ASSESSMENT.md` §5.5, `docs/kimiactions/04-crosscut-vs-kimi.md` §3 G3.

---

## Recommended sequence

Order is **ascending blast-radius within a priority tier**, and front-loads the items that defend
parity or unblock CI:

1. **A1** (HIGH) — wire + assert cuSOLVER determinism + correct §12. *The frozen parity law should not
   contain a false line; the wire is a cheap defence against a future cuSOLVER default change. Do the
   API/default check first since it sets the framing.*
2. **B1** (HIGH) — post-write stream check. *Silent truncated-output-exits-0 is a real correctness hole
   for a file-producing product; app-only, success-path-neutral.*
3. **B3** (MED) — capability query replacing capability-detection catches. *Removes a latent
   bug-hider (site B masks genuine throws); touches the device seam so do it before C-cluster churn
   settles.*
4. **B2** (MED) — OOM exit-code 5 → 3. *Honest fault taxonomy; the device-TU translator is the only
   plumbing cost.*
5. **C1** (MED) — genotype front-end dedup. *Closes a 4-way parity-divergence risk before any future
   read-time change has to be made in lockstep.*
6. **C2** (MED) — `csv_field`/`json_quote` exposure. *Output-correctness for pathological labels;
   down-payment on the future output sink.*
7. **E2** (MED) — wire `STEPPE_SANITIZER`. *Land before/with `02-ci-plan.md` — it is the CMake seam the
   asan/ubsan + compute-sanitizer CI lanes consume.*
8. **E1** (MED) — CPM `EXPECTED_HASH` pin. *Supply-chain integrity; dev-only blast radius, so safe to
   batch with the build-hygiene work.*
9. **D2** (MED) — single-source the version. *Kills the one real literal drift; host-checkable.*
10. **D1** (MED) — `architecture.md` doc-sync (+ adjacent staleness sweep). *Pure narrative; no build;
    do last, in one editorial pass, alongside D2 so §16:914 lands consistent.*
11. **C3** (MED/polish) — `--out`/`--out-dir` alias unify. *Lowest-stakes UX polish; pick up whenever
    convenient.*
12. **D3** (MED, cross-cut G5) — correct the stale `run_qpdstat` "Part B not yet implemented" docstring.
    *One-string doc-vs-as-built fix; land in the same wheel-build pass as D1/D2.*
13. **F1** (LOW, cross-cut G2) — relocate the six design-doc comment-essays to `docs/` behind 1-line
    pointers. *Highest first-impression signal; sequence as one editorial sweep with `03-low-polish.md`'s
    comment-hygiene scrub (G1/G14/G15) so the "comment voice" surface is touched once.*
14. **F2** (LOW, cross-cut G3) — `cuda_backend.cu` single-seam header note + DECISIONS minute. *XS; pick
    up alongside F1 (same file family) or whenever convenient.*

**Land-together couplings:** A1's two commits (wire+test, doc) ship as a pair. D1 + D2 share the §16:914
claim — pick the version mechanism (generated `version.hpp` *or* the doc describes the
compile-definition path) and keep doc+code consistent in the same PR. E2 ships before the CI plan
turns on its sanitizer lanes.

## One-glance priority table

| # | Item | Cluster | Priority | Effort | Parity-risk | Verify lane |
|---|------|---------|----------|--------|-------------|-------------|
| A1 | Wire `cusolverDnSetDeterministicMode` + assert + §12 doc-fix | A §12 parity-law | **HIGH** | S | LOW (bounded: NRBIG `gesvd` only; 9-pop immune) | box5090 (`handles_unit`, `test_qpadm_parity`) |
| B1 | Post-write `flush`+`good()` → exit 4 on torn write | B error-handling | **HIGH** | M | None (success-path bytes unchanged) | box5090 (`ctest` + `/dev/full` test) |
| B3 | Capability-query replaces capability-detection catch | B error-handling | MED | M | None on goldens (a genuine throw now propagates) | box5090 (qpadm/qpwave/rot parity + unit) |
| B2 | OOM `5 → 3` via CUDA-free decl + device-TU def | B error-handling | MED | M | None (error path only) | box5090 (suite exits 0; OOM→3 test) |
| C1 | `genotype_front_end` dedup (4 callers) | C IO/output | MED | M | None (byte-identical lift) | box5090 (4 genotype goldens) |
| C2 | `csv_field`/`json_quote` shared escaping | C IO/output | MED | S | None (bare for real names) | box5090 (`test_cli_dates/qpgraph` + unit) |
| E2 | Wire `STEPPE_SANITIZER` (`SteppeSanitizers.cmake`) | E build hygiene | MED (→CI) | S | None (debug-only overlay) | box5090 (asan CXX-only; `compute` accepted) |
| E1 | CPM `EXPECTED_HASH` pin (+ optional dep SHA-pin) | E build hygiene | MED | S | None (provenance only) | box5090 (verify-on-fetch; tamper hard-fails) |
| D2 | Single-source version (pyproject `dynamic` regex) | D doc/version | MED | S | None (string only) | host + box5090 wheel |
| D1 | `architecture.md` doc-sync (core→io, C-ABI, 3.28) | D doc/version | MED (polish) | S | None (doc only) | reviewer diff (no build) |
| C3 | `--out`/`--out-dir` alias unify | C IO/output | MED (polish) | S | None (CLI surface only) | box5090 (`--help` + CLI tests) |
| D3 | Correct stale `run_qpdstat` "Part B not yet impl" docstring *(cross-cut G5)* | D doc/version | MED (polish) | XS | None (docstring string only) | box5090 wheel (help text); no golden |
| F1 | Relocate 6 design-doc comment-essays to `docs/` + 1-line pointers *(cross-cut G2)* | F comment-altitude | LOW (polish) | M | None (comment-only; parity invariants kept inline) | box5090 recompile + `ctest` (no golden re-run) |
| F2 | `cuda_backend.cu` single-seam header note + DECISIONS minute *(cross-cut G3)* | F comment-altitude | LOW (polish) | XS | None (comment/doc only) | reviewer diff + box5090 recompile |

**Standing constraints (all items).** Nothing builds locally (RTX 2070 / CUDA 11.8, no FP64-emu) — every
GPU/golden/CLI verification runs on box5090; only D1 (doc-only), D2's regex check, and C2's escape unit
are off-box. No item regenerates an AT2 golden — the existing committed goldens plus a handful of
targeted error/edge tests are the full guard. §4-rejected items (2nd stream, NCCL, multi-GPU gate,
excise-oracle, `steppe::Index`, `JsonWriter` reshape, capability pure-virtuals, kernel fuzzing, mock
backend) stay out of scope.

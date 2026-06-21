# steppe — RESUME (read this first after a compaction / new session)

**The one line that matters:** read `agentscripts/README.md` first — it's the workflow map
(where we are + every workflow's state/retrigger + what's still needed). Everything else is
reachable from there. The rest of this file is the fast re-orientation.

## Read these (don't re-derive)
1. **`agentscripts/README.md`** ← THE WORKFLOW MAP / master index.
2. **`docs/RUN-GUIDE.md`** ← HOW TO RUN EVERYTHING (build, ctest default vs THOROUGH, run a fit / rotation / precompute, the dev loop). Use it for full commands; this card only has the one-liners.
3. `ls handoff-*.md` → read the latest — cold-start handoff.
4. `docs/design/fit-engine.md` ← the Phase-2 fit-engine spec + the **milestone table (M(fit-0)…M(fit-6), all BUILT)**; `docs/design/fit-engine-finish-punchlist.md` ← the backend-finish punch-list (6 FINISH-NOW = DONE, F1–F6; 6 DEFER; 1 NEW-FEATURE).
5. **STEP 2 (CLI/bindings):** `docs/design/cli-bindings.md` ← the CLI + Python-bindings contract + milestone order (M(cli-0..4), M(py-1..2)); `docs/research/pycuda-cuda13-viability.md` (verdict: nanobind + DLPack, NOT PyCUDA) + `docs/research/interop-usecases.md` (what the numpy/CuPy/PyTorch interop is practically for).
5. `docs/cleanup/bigrefactor/` ← the big-refactor record (findings per group, `NAMING-STYLE-STANDARD.md`, `precision-policy-consistency.md`, `group7-provenance.md`); `docs/cleanup/m4.5/{why-d2h,why-multigpu-slow,architecture-audit,parallelism-check}.md` ← the honest perf story.
   (Auto-loaded memories cover the boxes: `[[rtxbox]]`, `[[steppebox5090]]`, `[[steppe-project]]`, `[[steppe-dev-process]]`.)

Verify state: `cd /home/suzunik/steppe && git branch --show-current && git log --oneline -6`

## State (as of this writing)
- **`main`** @ `67dc696` (branch `phase2-fit-engine` may sit a few agentscripts/docs commits ahead — verify with `git rev-parse --short main phase2-fit-engine`). **Phase-1 PRECOMPUTE (S0–S2) COMPLETE THROUGH M5**; **Phase-2 qpAdm FIT ENGINE (S3–S8) BUILT + golden-gated ON THE GPU**; the big-refactor is COMPLETE; **the fit-engine BACKEND is FINISHED** (F1–F6 below). Parity `memcmp` bit-identical / within tolerance on real AADR throughout. **ctest is 44/44.**
- **STEP 2 — PRODUCTIZATION (CLI) IN PROGRESS:** steppe is **runnable from the command line** for the first time.
  - **M(cli-0)** (`62253ab`) — the `steppe` CLI scaffold (CLI11, GPU-only, `app` a plain CXX target that CANNOT include a CUDA header [arch-grep enforced]; the CUDA-free `ConfigBuilder`/`RunConfig` precedence chain; Status→exit-code).
  - **M(cli-1)** (`67dc696`) — `steppe qpadm --f2-dir DIR --target … --left … --right …`: reads an f2 dir (`f2.bin`+`pops.txt`+`meta.json`), resolves pop NAMES, runs on the GPU, emits tidy CSV/JSON — **golden-gated THROUGH the CLI** (reproduces `golden_fit0` + NRBIG). Remaining CLI: M(cli-2) `qpwave`, M(cli-3) `qpadm-rotate`, M(cli-4) `extract-f2` (the heavy genotype→f2-dir writer). Contract: `docs/design/cli-bindings.md`.
  - **Bindings decisions (research committed):** Python = **nanobind**, NOT PyCUDA (`docs/research/pycuda-cuda13-viability.md`), + a **DLPack/`__cuda_array_interface__`** interop seam; interop use-cases in `docs/research/interop-usecases.md` (MUST = results→pandas + a Q/V/N array entry [`afs_to_f2` analog]; flagship = the msprime→steppe→qpAdm power-analysis loop; GPU-resident *input* is low-value; fp64 caveat). **GPU-only — NO CPU runtime**; the CpuBackend is the dev/test parity oracle only (memory `cpu-is-test-only`).
- **BACKEND FINISH — DONE** (step 1 of the backend-first sequence; the 6 FINISH-NOW punch-list items, all git-verifiable + golden-gated on REAL AADR):
  - **F5** (`e8430a2`) — removed the dead public `QpAdmOptions::constrained` field (`include/steppe/qpadm.hpp`); non-negative constrained weights is a deferred step-3 feature.
  - **F3** (`ffdcba2`) — new `Status::ChisqUndefined` (`include/steppe/error.hpp`) + a `dof<=0 ⇒ ChisqUndefined` guard on BOTH the HOST (`qpadm_fit.cpp`) and the CUDA model-batched path (`cuda_backend.cu`); was leaking NaN `p` with `status=Ok`.
  - **F2** (`c8fe397`) — the M(fit-5) domain-outcome acceptance gate, NEW `tests/reference/test_qpadm_domain.cu`: degenerate REAL-AADR models (collinear left ⇒ `RankDeficient`; `fudge=0` singular Q ⇒ `NonSpdCovariance`; over-parameterized `dof<=0` ⇒ `ChisqUndefined`) asserted as STATUS VALUES (no crash/NaN) on BOTH `CpuBackend` + `CudaBackend`.
  - **F6** (`360e386`) — widened the G=1==G=2 determinism `memcmp` in `test_qpadm_rotation.cu` to the FULL `QpAdmResult` (`z`/`dof`/`est_rank`/`rank_*`/`rankdrop_*`/`popdrop_*`).
  - **F4** (`6481dfa`) — pinned a REAL AT2 `qpwave()` golden `golden_qpwave.json` (admixtools 2.0.10 / R 4.3.3, real AADR v66.p1_HO) + NEW `test_qpwave_parity.cu` gating the first-class `run_qpwave` entry on BOTH backends.
  - **F1** (`2496a14`) — missing-block / NA handling (OQ-12). steppe is pairwise-complete (NOT AT2 `maxmiss=0` global-intersection), so a pair-block `Vpair==0` can occur on sparse AADR and was silently imputed `f2=0` (bias-toward-0). Now does AT2 `read_f2(remove_na=TRUE)`: DROP any block with a non-finite / `Vpair==0` pair before the jackknife (not impute-0), via a single shared host/device predicate `core::pair_block_is_missing` (`f2_estimator.hpp`) — CpuBackend oracle + GPU (`f2_block_keep_kernel`; single-model AND S8 model-batched survivor-compaction). Legacy `maxmiss=0` goldens stay BYTE-IDENTICAL (no-drop identity arm). NEW `golden_fitNA.json` + `test_qpadm_missing_block.cu` (real-AADR `maxmiss=0.99`, a sparse right pop ⇒ 1 real dropped block, real `Vpair==0`). NO synthetic data.
  - **New goldens** (all REAL AADR): `golden_qpwave.json`, `golden_fitNA.json`. **New tests**: `test_qpadm_domain.cu`, `test_qpwave_parity.cu`, `test_qpadm_missing_block.cu`. **ctest is now 42 tests** (`qpadm_domain` #19); default + `STEPPE_THOROUGH` green, CPU/GPU consistent, deterministic, WALLCLOCK unchanged.
- **PHASE-2 FIT ENGINE — BUILT, golden-gated on the GPU** (the milestone table in `docs/design/fit-engine.md`; verifiable in git):
  - **M(fit-0)** contract + CpuBackend oracle scaffold (frozen `include/steppe/qpadm.hpp`).
  - **M(fit-1)** f4 (S3) + weighted Q (S4) + the GLS weight fit (S6, `opt_A`/`opt_B` ALS).
  - **M(fit-2)** rank test / qpWave (S5): `rankdrop`/`popdrop` tables; the **nr≤32 9-pop** on-device Jacobi small path **AND** the **nr=39 NRBIG** large path via cuSOLVER `gesvd`.
  - **M(fit-3)** block-jackknife SE (S7) + the opt-in `JackknifePolicy {None, FeasibleOnly, All}` (default `All` ⇒ goldens unchanged; the feature is purely additive).
  - **M(fit-4)** the `CudaBackend` (single-GPU, f2 resident, cuSOLVER, EmulatedFp64{40} on the S4 SYRK).
  - **M(fit-5)** domain outcomes (`RankDeficient` / `NonSpdCovariance` returned as per-model `status` VALUES, not crashes).
  - **M(fit-6)** the **S8 model-space ROTATION** (`run_qpadm_search`): genuinely batched `fit_models_batched` + the multi-GPU shard plumbing.
- **VALIDATION = REAL-AADR AT2 goldens** under `tests/reference/goldens/at2/` (all admixtools 2.0.10 / R 4.3.3 / v66.p1_HO): `golden_fit0.json` (9-pop), `golden_fit1_NRBIG.json` (nr=39 large path), `golden_rot.json` (84-model rotation), **`golden_qpwave.json`** (qpWave entry, F4), **`golden_fitNA.json`** (missing-block drop, F1). The GPU path matches them bit/tolerance; the **CpuBackend is the native oracle** (run under `STEPPE_THOROUGH`).
- **BIG REFACTOR COMPLETE** (`docs/cleanup/bigrefactor/`): **3 HIGH** — `block_sink` silent-corruption fail-fast (`9dbc610`), `kQpMax` single-source (`3beff6d`), `opt_A`/`opt_B` ALS-ridge-solve dedup (`ed6cc44`) — **+ 14 MED groups + 9 LOW groups** (held to `NAMING-STYLE-STANDARD.md`) **+ 2 device dedups** (`block_sink` StagingRing `b1bd620`, qpadm-fit-kernel twin-collapse `25c882a`). ALL golden-gated on real AADR, **bit-identical, WALLCLOCK UNCHANGED** (qpadm_parity 0.86s default / 27.3s thorough; rotation 3.42s; no perf regression). Precision policy is consistent (emulated-FP64 default + native fallback via the ONE `emulation_honorable` predicate; `precision-policy-consistency.md` = 0 deviations). Skipped by decision: the host/device pointer-wrapper tag-type, blanket `const __restrict__`.
- **HONEST multi-GPU story (UNCHANGED — measured on real AADR):** the M(fit-6) rotation is correct + bit-identical G=1==G=2, but on the **consumer 5090s the one-time `f2` replication is a ~8.72 GB / ~3.8 s HOST BOUNCE** (no GeForce↔GeForce P2P), so multi-GPU only reached **~1.21× at 9086 real models — no 1.5× crossover. → RUN THE ROTATION SINGLE-GPU on box5090.** Multi-GPU rotation is DEFERRED (`TODO(multigpu-host-bounce)`, commit `2a0c020`); its real payoff needs P2P hardware (rtxbox / RTX PRO 6000) or per-device precompute. **Do NOT claim a multi-GPU rotation speedup.**

## Boxes
- **box5090** = vast 2× RTX 5090 (sm_120, CUDA 13): `ssh box5090`. Build/test here. **nvcc is NOT on PATH** — prefix the exports (below). **Long jobs (build/ctest/bench) → run DETACHED + poll a `/tmp/*.log`** — the network drops long ssh sessions. **RUN GUIDANCE: run the qpAdm fit / S8 rotation SINGLE-GPU here** (the 5090s have no P2P; multi-GPU rotation host-bounces the f2 → only ~1.21×, deferred). The M5 single-GPU streamed sweep was measured here: full-autosome P=2500 in ~51.5 s on ONE 32 GB 5090, parity bit-identical.
- **rtxbox** = 2× RTX PRO 6000 Blackwell (sm_120, 96 GB ea, stock-driver P2P OK; ephemeral, usually SPUN DOWN): `ssh rtxbox`. The box for AT2 goldens, GDS, the device-resident final-D2H pinning, `ncu`/nsys re-profiling — and the ONLY place a multi-GPU rotation speedup could actually land (P2P).

## Build / test (RELEASE build-rel; nvcc NOT on PATH — note the exports)
```
# from local: edit → sync → build/test on the box
rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr \
  -e ssh /home/suzunik/steppe/ box5090:/workspace/steppe/

ssh box5090 'cd /workspace/steppe && export PATH=/usr/local/cuda/bin:$PATH && \
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && \
  cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release && cmake --build build-rel'
```
- **default ctest** (GPU-vs-AT2-golden, fast ~1 min): `ctest --test-dir build-rel --output-on-failure` (`qpadm_parity` 0.86s, `qpadm_rotation` 3.42s).
- **THOROUGH** (adds the CpuBackend oracle + NRBIG full SE; the CI-without-GPU + localization lane, ~56s): `STEPPE_THOROUGH=1 ctest --test-dir build-rel -R qpadm`.
- **CLI (NEW, build with `-DSTEPPE_BUILD_CLI=ON`):** `build-rel/bin/steppe --help`; `steppe qpadm --f2-dir DIR …` works (GPU). Full `extract-f2` (build an f2 dir from genotypes) is M(cli-4), pending — so to run `steppe qpadm` today you need an f2 dir; the `cli_qpadm` ctest builds one from the committed fixture and exercises it end-to-end. **No Python bindings yet.** Also still available: `build-rel/bin/test_qpadm_parity <goldens-dir>` / `test_qpadm_rotation <goldens-dir>` + the C++ API (`steppe::run_qpadm` / `run_qpadm_search` / `run_qpwave`, `include/steppe/qpadm.hpp`). Full commands: `docs/RUN-GUIDE.md`.
- core dumps: the box accumulates ~404 MB cores on `abort()` — clear with `rm -f /var/lib/vastai_kaalia/data/core-*` (the workflows do this per build).

## Workflows
Retrigger via `Workflow({scriptPath: "agentscripts/<name>.js"})`. Patterns:
- **impl** = Contracts/Design → per-file Implement → build agent → verify agent.
- **fix-pass** = fixer + independent verdict.
- **audit** = per-unit auditor → adversarial critic → holistic capstone (read-only).
- To set up a box: `scripts/box_bringup.sh <alias> [--build]` + `scripts/p2p_probe.cu`.

## Open threads (pick up here)
Phase 1 (precompute, S0–S2 thru M5) and Phase 2 (the qpAdm fit engine, S3–S8, M(fit-0)…M(fit-6)) are **BUILT, golden-gated, the big-refactor is complete, AND the fit-engine BACKEND is now FINISHED** (F1–F6, above; `docs/design/fit-engine-finish-punchlist.md`). The honest "what is left" is the **build-sequence-backend-first** ladder — step 1 (backend finish) is DONE; **the next work is step 2** (see `docs/research/desirable-features-survey.md`):
1. **STEP 2 — PRODUCTIZATION (IN PROGRESS).** The **CLI exists**: M(cli-0) scaffold + M(cli-1) `steppe qpadm` (golden-gated, GPU). **Next:** M(cli-2) `qpwave` → M(cli-3) `qpadm-rotate` → M(cli-4) `extract-f2` (the genotype→f2-dir writer; the heavy one — pulls in `io`/filters/tiering) → **M(py-1) Python bindings** (nanobind + the DLPack interop seam — design in `docs/design/cli-bindings.md` + `docs/research/interop-usecases.md`). GPU-only (no CPU runtime). Contract: `docs/design/cli-bindings.md`.
2. **STEP 3 — standalone f-stats, each WITH its own CLI/bindings (NOT built).** The f4 math is *internal to the fit* — there are no standalone **f4 / f3 / D-stat / f4-ratio / qpDstat** entry points (also covers the deferred non-negative constrained-weights feature from F5). **qpfstats / DATES / qpGraph** are not built either. After step 2.
3. **Precompute polish:** **M6** (multi-dataset merge) and **M7** (on-disk cache) are pending.
4. **Multi-GPU rotation payoff:** DEFERRED — needs P2P hardware (rtxbox) or per-device precompute (`TODO(multigpu-host-bounce)`). Until then **run the rotation SINGLE-GPU.** Not a blocker.

## Rules
Nothing builds locally (rsync to the box). PERF/bench MUST be a Release build (`build-rel`). Commit
only on green (ROADMAP §6 message + the `Co-Authored-By: Claude Opus 4.8 (1M context)
<noreply@anthropic.com>` trailer). VERIFY first, then question the verification. DON'T re-probe a
box that's already set up. Use workflows for substantive work. **No synthetic data for any
accuracy / perf / throughput claim — real AADR only.**

---
*If you need raw session history (rare — the above usually suffices): grep `~/.claude/projects/-home-suzunik-steppe/*.jsonl` (that's how the milestone-build workflows were recovered).*

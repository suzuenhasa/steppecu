# steppe — RESUME (read this first after a compaction / new session)

**The one line that matters:** read `agentscripts/README.md` first — it's the workflow map
(where we are + every workflow's state/retrigger + what's still needed). Everything else is
reachable from there. The rest of this file is the fast re-orientation.

## Read these (don't re-derive)
1. **`agentscripts/README.md`** ← THE WORKFLOW MAP / master index.
2. **`docs/RUN-GUIDE.md`** ← HOW TO RUN EVERYTHING (build, ctest default vs THOROUGH, run a fit / rotation / precompute, the dev loop). Use it for full commands; this card only has the one-liners.
3. `ls handoff-*.md` → read the latest — cold-start handoff.
4. `docs/design/fit-engine.md` ← the Phase-2 fit-engine spec + the **milestone table (M(fit-0)…M(fit-6), all BUILT)**.
5. `docs/cleanup/bigrefactor/` ← the big-refactor record (findings per group, `NAMING-STYLE-STANDARD.md`, `precision-policy-consistency.md`, `group7-provenance.md`); `docs/cleanup/m4.5/{why-d2h,why-multigpu-slow,architecture-audit,parallelism-check}.md` ← the honest perf story.
   (Auto-loaded memories cover the boxes: `[[rtxbox]]`, `[[steppebox5090]]`, `[[steppe-project]]`, `[[steppe-dev-process]]`.)

Verify state: `cd /home/suzunik/steppe && git branch --show-current && git log --oneline -6`

## State (as of this writing)
- **`main`** @ `25c882a` (branch `phase2-fit-engine` tracks it). **Phase-1 PRECOMPUTE (S0–S2) is COMPLETE THROUGH M5** (device-resident output + M5 streaming) and **Phase-2 the qpAdm FIT ENGINE (S3–S8) is BUILT and golden-gated ON THE GPU.** The big-refactor is COMPLETE (below). Parity is `memcmp` bit-identical / within tolerance on real AADR throughout.
- **PHASE-2 FIT ENGINE — BUILT, golden-gated on the GPU** (the milestone table in `docs/design/fit-engine.md`; verifiable in git):
  - **M(fit-0)** contract + CpuBackend oracle scaffold (frozen `include/steppe/qpadm.hpp`).
  - **M(fit-1)** f4 (S3) + weighted Q (S4) + the GLS weight fit (S6, `opt_A`/`opt_B` ALS).
  - **M(fit-2)** rank test / qpWave (S5): `rankdrop`/`popdrop` tables; the **nr≤32 9-pop** on-device Jacobi small path **AND** the **nr=39 NRBIG** large path via cuSOLVER `gesvd`.
  - **M(fit-3)** block-jackknife SE (S7) + the opt-in `JackknifePolicy {None, FeasibleOnly, All}` (default `All` ⇒ goldens unchanged; the feature is purely additive).
  - **M(fit-4)** the `CudaBackend` (single-GPU, f2 resident, cuSOLVER, EmulatedFp64{40} on the S4 SYRK).
  - **M(fit-5)** domain outcomes (`RankDeficient` / `NonSpdCovariance` returned as per-model `status` VALUES, not crashes).
  - **M(fit-6)** the **S8 model-space ROTATION** (`run_qpadm_search`): genuinely batched `fit_models_batched` + the multi-GPU shard plumbing.
- **VALIDATION = REAL-AADR AT2 goldens** under `tests/reference/goldens/at2/` (all admixtools 2.0.10 / R 4.3.3 / v66.p1_HO): `golden_fit0.json` (9-pop), `golden_fit1_NRBIG.json` (nr=39 large path), `golden_rot.json` (84-model rotation). The GPU path matches them bit/tolerance; the **CpuBackend is the native oracle** (run under `STEPPE_THOROUGH`).
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
- run a fit / rank test / rotation directly: `build-rel/bin/test_qpadm_parity <goldens-dir>` and `build-rel/bin/test_qpadm_rotation <goldens-dir>` (they load the real f2 fixtures + run the GPU fit/rotation). The C++ API: `steppe::run_qpadm(...)` + `steppe::run_qpadm_search(...)` in `include/steppe/qpadm.hpp`. **There is NO CLI / Python yet** — running = the test harness + the library API. Full commands: `docs/RUN-GUIDE.md`.
- core dumps: the box accumulates ~404 MB cores on `abort()` — clear with `rm -f /var/lib/vastai_kaalia/data/core-*` (the workflows do this per build).

## Workflows
Retrigger via `Workflow({scriptPath: "agentscripts/<name>.js"})`. Patterns:
- **impl** = Contracts/Design → per-file Implement → build agent → verify agent.
- **fix-pass** = fixer + independent verdict.
- **audit** = per-unit auditor → adversarial critic → holistic capstone (read-only).
- To set up a box: `scripts/box_bringup.sh <alias> [--build]` + `scripts/p2p_probe.cu`.

## Open threads (pick up here)
Phase 1 (precompute, S0–S2 thru M5) and Phase 2 (the qpAdm fit engine, S3–S8, M(fit-0)…M(fit-6)) are **BUILT, golden-gated, and the big-refactor is complete**. The honest "what is left" is **productization + new f-stats features** (see `docs/research/desirable-features-survey.md`):
1. **PRODUCTIZATION — there is no user-facing surface yet.** No **CLI**, no **Python bindings** (the `app/` directory is *planned*). Today "running" steppe = the test harness binaries + the C++ library API (`run_qpadm` / `run_qpadm_search`). Building the CLI/bindings is the next big lever for actual use.
2. **NEW f-stats entry points (not built).** The f4 math is *internal to the fit* — there are no standalone **f4 / f3 / D-stat / f4-ratio / qpDstat** entry points. **qpfstats / DATES / qpGraph** are not built either. These are the obvious next features.
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

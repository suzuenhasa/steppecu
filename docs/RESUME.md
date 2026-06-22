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
- **`main`** @ `66280b7` (branch `phase2-fit-engine` may sit a few commits ahead — verify with `git rev-parse --short main phase2-fit-engine`). **Phase-1 PRECOMPUTE (S0–S2) COMPLETE THROUGH M5**; **Phase-2 qpAdm FIT ENGINE (S3–S8) BUILT + golden-gated**; big-refactor COMPLETE; **fit-engine BACKEND FINISHED**; **FULL AT2 END-TO-END PARITY achieved** (parity-chain block below); **all 5 goldens correct (convertf-PA)**; **CLI productized (extract-f2/qpadm/qpadm-rotate wired) + two studies reproduced + perf characterized on 1240K** (post-parity block below). **ctest 46/46.**
- **🚀 POST-PARITY — PRODUCTIZATION + STUDIES + PERF (2026-06-22).** (a) **CLI complete bar one:** `extract-f2` (streams big-P via the M5 tiered path — `--tier auto|resident|host|disk`; `--hash`/no-hash default; `--ploidy auto`), `qpadm`, and **`qpadm-rotate`** (M(cli-3), `17c4606` — batched `run_qpadm_search`, golden-gated through the CLI) are WIRED; **only `qpwave` CLI is still a scaffold** (M(cli-2), `cli_parse.cpp:189` — engine + `golden_qpwave` done, easy wiring). (b) **Two studies reproduced** on real v66 1240K: Haak 2015 (`docs/studies/haak2015.md`) + Olalde 2018 single-model AND its competing-sources **rotation** (`docs/studies/olalde2018.md`, `olalde2018-rotation.md`) — both vs published, with cross-version pop-mapping documented. (c) **Perf (`docs/perf/1240k-sweep.md`):** `extract-f2` ~5.8 s at 60 pops / streams to 700+ (the ~37 s "wall" was a byte-at-a-time provenance SHA, now `--hash`-gated + overlapped); rotation **~60k models/sec jk0**, and **jk1 (SE) scales to pool_200 single-GPU** after the SE-pass VRAM-budget fix (`66280b7`). Whole-AADR rotation ≈ 36 min (point) / ~1.8 h (with SEs). Run sheet: `docs/RUN-SHEET.md`. **All single-GPU (`--device 0`; multi-gpu PARKED — memory `multi-gpu-parked`).**
- **🎯 AT2 PARITY — COMPLETE (the big arc; 2026-06-21).** steppe reproduces ADMIXTOOLS **end-to-end on real AADR v66** (raw genotypes → decode → f2 → qpAdm), diploid + pseudo-haploid, to the emulated-FP64 floor (~1e-14). **Critical discovery:** the v66 `.geno` is **TGENO** (transposed); **admixtools R v2.0.10 cannot read TGENO and silently misreads it**, so the original committed goldens were CORRUPT (AT2's misread) — **steppe's decode was the correct one**. Proven via `convertf` v8621 (TGENO→PACKEDANCESTRYMAP, which AT2 reads). Then a **4-fix parity chain** (all on main, golden-gated): (1) `--blgsize` = Morgans like AT2 + monomorphic-SNP drop (`d5fcbcc`); (2) `assign_blocks` → AT2 SNP-anchored block convention, element-wise block parity (`a5781dd`); (3) **per-sample pseudo-haploid auto-detection** (`adjust_pseudohaploid`) — steppe hard-coded ploidy=2, EXACT on diploids but **up to 230% wrong on pseudo-haploid f2** (the dominant aDNA type); now per-sample, f2 EXACT vs AT2 (`bc0b773`). **Haak 2015 reproduced** (Corded Ware ~74% Yamnaya, Sardinian mostly-EEF — matches AT2). Writeups: `docs/research/{tgeno-at2-support,block-partition-at2,f2-estimator-at2}.md`, `docs/studies/haak2015.md`; memory `aadr-tgeno-goldens-corrupt`. **The convertf→PA→AT2 pipeline is on the box** (`/workspace/AdmixTools_src/src/convertf`, `/workspace/data/aadr/converted_pa/v66_HO_pa`).
- **STEP 2 — PRODUCTIZATION (CLI):** steppe is **runnable from the command line**.
  - **M(cli-0)** (`62253ab`) — the `steppe` CLI scaffold (CLI11, GPU-only, `app` a plain CXX CUDA-free target; CUDA-free `ConfigBuilder`/`RunConfig`; Status→exit-code).
  - **M(cli-1)** (`67dc696`) — `steppe qpadm --f2-dir DIR …` (reads `f2.bin`+`pops.txt`+`meta.json`, name resolution, GPU, CSV/JSON), golden-gated.
  - **M(cli-4)** (`74c3c71` + streaming/hash) — `steppe extract-f2 --prefix … --pops/--auto-top-k … --out DIR` (genotypes → f2 dir; `--blgsize` Morgans, `--maxmiss`, `--ploidy auto`, `--drop-mono`, **`--tier auto|resident|host|disk`** [streams big-P], **`--hash`/`--no-hash`** [default off]).
  - **M(cli-3)** (`17c4606`) — `steppe qpadm-rotate --target … --pool … --right … --min-sources/--max-sources` (batched `run_qpadm_search`; golden-gated through the CLI vs `golden_rot`).
  - **Remaining CLI:** **M(cli-2) `qpwave`** (the ONE scaffold left — `cli_parse.cpp:189`; engine + `golden_qpwave` done) → then **M(py-1) bindings**. Contract: `docs/design/cli-bindings.md`.
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
- **CLI (build with `-DSTEPPE_BUILD_CLI=ON`):** the full study flow works on the GPU: `steppe extract-f2 --prefix <geno prefix> --pops a,b,… --out DIR --blgsize 0.05 --maxmiss 0` (genotypes → f2 dir; `--blgsize` is MORGANS like AT2; `--ploidy auto` = per-sample pseudo-haploid detection; `--drop-mono`) then `steppe qpadm --f2-dir DIR --target … --left … --right … --format csv|json`. e.g. on the box, England Bell Beaker = Yamnaya+Anatolia from the raw v66 TGENO reproduces AT2. `steppe --help` lists all subcommands. **No Python bindings yet** (M(py-1)). Also: `test_qpadm_parity`/`test_qpadm_rotation` binaries + the C++ API (`run_qpadm`/`run_qpadm_search`/`run_qpwave`). Full commands: `docs/RUN-GUIDE.md`.
- core dumps: the box accumulates ~404 MB cores on `abort()` — clear with `rm -f /var/lib/vastai_kaalia/data/core-*` (the workflows do this per build).

## Workflows
Retrigger via `Workflow({scriptPath: "agentscripts/<name>.js"})`. Patterns:
- **impl** = Contracts/Design → per-file Implement → build agent → verify agent.
- **fix-pass** = fixer + independent verdict.
- **audit** = per-unit auditor → adversarial critic → holistic capstone (read-only).
- To set up a box: `scripts/box_bringup.sh <alias> [--build]` + `scripts/p2p_probe.cu`.

## Open threads (pick up here)
Backend FINISHED + **full AT2 end-to-end parity achieved** (the 4-fix chain, above) + **Haak 2015 reproduced** + **ALL 5 GOLDENS now correct** (regenerated from the convertf-PA). Remaining:
0. **GOLDEN REGEN — DONE.** All 5 goldens (`fit0`, `rot`, `fit1_NRBIG`, `fitNA`, `qpwave`) are now on the convertf-PA (`geno_sha256 e588406…`), replacing the corrupt TGENO-misreads; steppe reproduces every one at tier, THOROUGH ctest 45/45. `rot`/`fit1_NRBIG`/`fitNA` regenerated in `cb4d19c`; `qpwave` provenance corrected in `8f5b3f1` (its numbers were already right but its metadata still claimed the raw TGENO — a stale-provenance trap worth knowing about). Generators: `tests/reference/goldens/at2/scripts/golden_*_generate.R` (all retargeted to `pref=convertf-PA`).
1. **STEP 2 — PRODUCTIZATION (CLI nearly done).** Built: M(cli-0) scaffold, M(cli-1) `qpadm`, M(cli-4) `extract-f2` (+ streaming/hash/tier), **M(cli-3) `qpadm-rotate`** (`17c4606`). **Next:** **M(cli-2) `qpwave`** — the ONE remaining CLI scaffold (`cli_parse.cpp:189`; engine + `golden_qpwave` done — same easy wiring as rotate; route to `run_qpwave`, golden-gate vs `golden_qpwave`) → then **M(py-1) Python bindings** (nanobind + DLPack seam — `docs/design/cli-bindings.md` + `docs/research/interop-usecases.md`). GPU-only (no CPU runtime).
   - **Perf follow-up (optional):** rotation **f2-INPUT streaming** for pools >~1700 (where the resident f2 itself exceeds 32 GB) — the only scaling wall left; jk0 + jk1 both already scale to ~pool_200 resident / ~1000 pops (f2-resident-limited). See `docs/perf/1240k-sweep.md`.
   - Also tracked (`docs/TODO.md`): **older `.GENO`/EIGENSTRAT reader support** (steppe is TGENO-only); a **CI guard** rejecting AT2-on-raw-TGENO goldens; the ~0.3% block-boundary item is RESOLVED (it was the partition convention, now AT2-matched).
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

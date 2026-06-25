# steppe — RELEASE-SCOPE

> Release-readiness audit + the prioritized cleanup/hardening roadmap (blockers vs post-release) + the forward feature roadmap.
> Synthesized from seven per-dimension audits, cross-verified against the live tree at **HEAD `1707460`** (branch `phase2-fit-engine == main`). Every file:line below was read/grepped, not asserted.
> Read-only scoping document — no code was changed.

---

## 1. Verdict (release-readiness)

**steppe is functionally complete and the engineering is release-grade; what blocks a release is hygiene, packaging, and documentation — not the math.** The shipped surface (14 CLI subcommands, the qpAdm/qpWave fit engine + S8 rotation, standalone f4/f3/f4-ratio/qpDstat A+B, the on-device all-quartets sweep, qpfstats, qpGraph fit + 5-pop topology search, DATES, the Python facade + `_core.so`) is built, golden-gated, GPU-bound, host-clean, and perf-passed on real AADR — the four "is the product actually there" dimensions (**correctness/goldens, code-quality, CLI-surface, perf/scale**) come back essentially clean, with perf the single cleanest dimension (zero blockers). **The critical path is a short, mostly-S-effort hygiene pass**: there is **no `LICENSE` file** despite a declared MIT license, **no wheel has ever been built** (the manylinux/auditwheel path is 100% unverified), the **README/RUN-GUIDE actively misinform** (RUN-GUIDE.md:9 says "no CLI and no Python binding yet" — flatly false), **137 tracked `agentscripts/` leak real box IPs** into the source tree, and there is **no CI, no `__version__`, no CUDA-13 build floor**. None of these touch the validated compute; all are addressable in roughly one focused release-eng + docs pass. **Estimate: the blocker set is ~2-3 days of work, dominated by the one genuinely-L item (build + repair + clean-venv-import-verify a wheel).**

---

## 2. RELEASE-BLOCKERS (must-do, grouped)

Eleven blocker items, de-duplicated across the seven audits. Grouped by the work cluster.

### A. Legal / licensing (S each — the hardest *gate*, the cheapest *work*)
1. **No `LICENSE` file** despite `pyproject.toml:28` `license = { text = "MIT" }`, and the spec's own Day-1 checklist lists it. A wheel/sdist claiming MIT with no license text is legally incomplete and trips PyPI/packaging lint. → Add top-level `LICENSE` (MIT text, copyright holder/year); add `license-files = ["LICENSE"]` (or PEP 639 SPDX `license = "MIT"`). **[S]**
2. **MIT-vs-GPL-3 clean-room posture undocumented.** Upstream ADMIXTOOLS 2 is GPL-3; steppe declares MIT. The posture *is* defensible — verified **no AT2/DATES/convertf source is vendored** (the R golden scripts call AT2 as an external oracle; `src/core/stats/dates.cpp` is original) — but the bald MIT claim with no derivation note invites a challenge. → Add `ATTRIBUTION.md`/NOTICE stating independent reimplementation of the published methods, no GPL source copied, AT2 used only as a test oracle; cite the AdmixTools/ADMIXTOOLS 2 + DATES papers. Get a one-line licensing-aware read. **[S]**

### B. Distribution / packaging (mixed)
3. **No wheel has ever been built** (no `.whl` anywhere; no `[tool.cibuildwheel]`; only an ad-hoc `pip wheel .` on the box). The spec's manylinux + `auditwheel repair --exclude` of the CUDA SONAMEs path is 100% unverified — platform tag, manylinux compliance, clean-venv import all unproven. This is the central distributable unknown. → Build inside `nvidia/cuda:13-devel`, `auditwheel repair` with the CUDA-SONAME exclude list, install into a fresh CUDA-13 venv, assert `import steppe` + one real-AADR fit; codify as a cibuildwheel job. **[L]**
4. **No git tag / GitHub release / publish path defined.** `git tag -l` empty. "Release" is operationally undefined. → Decide the channel (a **GitHub release with the prebuilt sm_120 wheel as an asset** is the realistic first step; PyPI is awkward for a single-arch CUDA-13-system-dep wheel and is **POST-RELEASE** until the cu13 redistributables publish a real version — `pyproject.toml:11-16` documents why). Tag `v0.1.0`, write notes, attach wheel + SHA. **[M]**
5. **`find_package(CUDAToolkit REQUIRED)` has no version floor** (`CMakeLists.txt:56`). CUDA 13 is the hard requirement but the build configures against CUDA 12/11 then fails deep in compilation. → `find_package(CUDAToolkit 13.0 REQUIRED)` (or explicit `VERSION_LESS 13` `FATAL_ERROR`). Verify the cusolver SONAME (`.12` vs `.13` under CUDA 13) and align the pyproject note + auditwheel exclude. **[S]**
6. **No single-version-source / no `__version__`.** `0.1.0` is duplicated in `pyproject.toml:24` and `CMakeLists.txt:21`; `bindings/steppe/__init__.py` has **no `__version__`** (`steppe.__version__` → AttributeError); no `include/steppe/version.hpp`. → `configure_file` a `version.hpp` from `project(VERSION)`; expose `steppe.__version__` (importlib.metadata or generated); single-source the literal. **[S]**

### C. CI / automated gating (L)
7. **No CI at all** — no `.github/`, no YAML anywhere — despite architecture.md §14/§16 specifying a full matrix (layering arch-grep gate, warnings-as-errors, sanitizers, parity goldens, clean-venv wheel import). Every quality gate is manual on box5090; a regression ships silently. → Add `.github/workflows/`: (1) a cheap **no-GPU** job (clang-format/clang-tidy + the cross-layer-include arch-grep gate + host-only unit build/tests), (2) a self-hosted Blackwell GPU job running the FAST ctest goldens, (3) nightly sanitizer + THOROUGH. The GPU runner is the hard part. **[L]**

### D. Documentation (front-door, mixed)
8. **`docs/RUN-GUIDE.md` is actively WRONG** — its banner (RUN-GUIDE.md:9) states *"There is no CLI and no Python binding yet"* and §254 *"Where the CLI / Python would land (planned, not built)"*. Every claim is false at HEAD. → **Delete RUN-GUIDE.md** (RUN-SHEET supersedes it) and consolidate to one canonical run doc. **[M]**
9. **README is a 19-line stub** covering only `read_f2`→`qpadm` Python; grep confirms **0 mentions** of CLI/install/pip/extract/f4/qpgraph/dates/qpfstats. It is the GitHub front door *and* the PyPI long-description. → Rewrite: what-it-is, requirements (CUDA 13 GPU, sm_120, single-GPU, TGENO-only), install, a 14-command CLI table, a Python quickstart, links to the run sheet + feature matrix. **[M]**
10. **TGENO-only input limitation is not stated user-facing.** The 19-line README never mentions input formats; the code enforces TGENO-only and throws a good error (`geno_reader.cpp:83`), but discoverability is the gap — most AADR users hold PACKEDANCESTRYMAP/EIGENSTRAT and will perceive a working format as broken. → Add a "Supported inputs" section (README + genotype-path `--help`): TGENO only; transcode via `convertf` first; f2-dir is STPF2BK1 (not AT2 `.rds`-compatible). **[S]** *(doc-only; code already correct)*
11. **RUN-SHEET.md missing 4-6 shipped tools** — verified **0 occurrences** of qpgraph, qpgraph-search, dates, qpfstats, f4-sweep, f3-sweep (pinned to stale `main @ 433b71e`). The canonical command sheet omits the entire qpGraph fit + topology search + DATES + qpfstats. → Add a section per missing tool with a real-AADR invocation + required flags (incl. the `dates` non-obvious gotcha: the `.snp` must carry a real cM genetic map, `cli_parse.cpp:431`). **[M]**

> **Also pre-release (operational, not a code artifact):** the **agentscripts/handoff IP-leak** (item P-1 below) is severity-elevated by one auditor to release-blocker because the 137 tracked files (2.1M) carry literal vast.ai IPs/ports/SSH-key-paths (verified: `agentscripts/fix-pass-phase2.js:13` `root@78.92.24.57`, `handoff-ba37d95.md:16` `root@31.22.104.224`, +6 more). For a public repo this is an info-disclosure smell. **Decision required before making the repo public**, even though the *wheel* itself ships only `bindings/steppe` and is clean. Treat as a **pre-public-repo blocker**; listed under hygiene for the fix shape.

> **Release sign-off gate (process, not artifact):** "ctest green" is near-vacuous without a GPU+AADR runner — **26/34 reference tests and all 21 real-AADR ctest entries SKIP-clean (exit 0)** when GPU/data is absent. Release sign-off MUST be a full ctest on box5090 with `STEPPE_AADR_ROOT` populated AND `STEPPE_THOROUGH=1`. Enforce in the release checklist (and consider a `STEPPE_REQUIRE_GPU=1` mode that converts those SKIPs to failures on the release runner).

---

## 3. Dimension-by-dimension scope

### 3.1 Correctness / goldens / tests — **clean (1 process blocker)**
**State:** 65 ctest entries; 34 reference `.cu` parity gates, 20 host-only unit, ~12 CLI, 11 pytest. All committed goldens git-clean. The one known-defective golden (`golden_qpgraph_toposearch_spotcheck.csv`) is correctly removed + superseded by the 1590-row hash-keyed `at2_scores_5pop.csv` (the remaining references are comment-only, verified `test_qpgraph_search_parity.cu:20`). STEPPE_THOROUGH vs default-FAST split is consistent across all 5 qpAdm-family tests. Coverage is broad — every shipped command has an engine parity gate + CLI golden + (mostly) a pytest gate.
- **[BLOCKER, S]** CI-without-GPU passes trivially (26/34 reference + 21 real-AADR SKIP-clean) → release sign-off must be a full box5090 + AADR + THOROUGH run; consider `STEPPE_REQUIRE_GPU`. *(the §2 sign-off gate)*
- **[POST, M]** qpGraph parity golden VALUES are hardcoded `constexpr` (`test_qpgraph_parity.cu:94` `kGoldenScore`) decoupled from the committed CSVs — silent-drift risk; the CSV goldens are decorative for the engine gate. → Parse the CSV at runtime or add a build-time `constexpr == CSV` assert.
- **[POST, S]** Two orphan R-native `.rds` read by no test (`golden_qpfstats_geno.rds` 251KB, `golden_qpgraph_fit.rds` 49KB) + orphan `golden_qpgraph_edges.csv` (verified zero runtime loaders). → Delete or add a gate (prefer adding an edge-length check — fitted branch lengths are an untested qpGraph surface).
- **[POST, S]** Stale "Part B not yet implemented" docstring in `tests/python/test_py_qpdstat.py` (Part B IS shipped + gated).
- **[POST, M]** Provenance README (`goldens/at2/README.md`) documents only fit0/fitNA/toposearch; qpwave/rot/f4/f3/f4ratio/qpdstat/qpfstats/dates goldens have provenance only in commit messages. Also 6 readf2/qpfstats goldens have **no committed generator script**. → Extend the README table + commit the missing R snippets.
- **[POST, L]** Older-format reader (EIGENSTRAT/GENO/PA) has zero test coverage (TGENO-only) — documented gap, out of release scope.
- **[NICE, S/M]** `verify_bitexact.R` covers only fit0; the DATES external curve files (~130KB) are committed but only the scalar date/SE is diffed.

### 3.2 Code-quality / cleanup — **clean shipping code (1 blocker = doc-drift)**
**State:** the *shipping* `src/` is unusually clean — §4 layering 100% CUDA-free in app/bindings/core/io (kernels confined to `.cu`); zero dead code (no `#if 0`, no DISABLED_ tests, no commented-out blocks); ~10 TODO markers, all documented `TODO(multigpu-host-bounce)`/M4.5 parked-work pointers; consistent naming; `.clang-format`+`.clang-tidy` committed. The cleanup burden is **repo hygiene around the source**, not the source.
- **[BLOCKER, M]** Stale `docs/TODO.md`/`ROADMAP.md`/`RESUME.md`/`architecture.md` assert "NO CLI / NO bindings / NO standalone f-stats" and "topology search DEFERRED" — both false vs shipping code (verified `TODO.md:5` "there is NO CLI, NO Python bindings, NO standalone f-stats yet"). Doc-vs-code drift; owned jointly with docs dimension. → Refresh status sections.
- **[POST, S]** 4 stale "Part B not yet implemented / fail-fast sentinel" comments contradict shipping code (`run_config.hpp:89`, `cli_args.hpp:133/136`, `config_builder.cpp:485/486`) — the genotype-path Part-B IS implemented (`cmd_qpdstat.cpp:107/235`). Comment-only fix.
- **[POST, M]** `wip/fstats-massive-overbuild` branch (178 files, +4459/-61188 vs main) — rolled-back overbuild off an old main; the natural place to mine `read_fstats`/STPFST-shard reader IF on-device-enumeration is picked up. → Explicit keep/mine/delete call.
- **[POST, S]** Obsolete `stash@{0}` ("sweep-cli-FAILED-...OOM") lingering. → `git stash drop`.
- **[POST, S]** Untracked working-tree noise: `tests/tools/stage_qpgraph_f2dir.cpp` (one-off stager, not in CMake), `atlas_results/` (run-output, **not gitignored** — verified). → Commit+wire or delete/gitignore.
- **[POST, M]** 137 tracked `agentscripts/` (2.1M) + stale root `handoff-ba37d95.md` ship in source; clutter + IP leak (see §2 note + P-1). → Exclude from sdist + scrub/relocate.
- **[NICE, M/S]** Stale `docs/cleanup/` backlog (134 files, anchored to long-merged M4.5/big-refactor); root `build_m0.sh` + `experiments/` spikes (M0-era, not in CMake).

### 3.3 CLI surface — **release-grade (1 blocker + 1 doc blocker)**
**State:** single CLI11 app, **14 subcommands** (verified count); uniform flag vocabulary via shared `add_*_flags` helpers; centralized + thorough config validation; documented exit-code taxonomy; `main.cpp` top-level catch. No broken/half-wired subcommand — all 14 dispatch to real GPU compute.
- **[BLOCKER, S]** CLI11 parse-level errors bypass the exit-code taxonomy: unknown-flag/bad-subcommand/missing-required surface through `CLI11_PARSE` (`cli_parse.cpp:747`) as CLI11 codes (verified exit **109**, not `kExitInvalidConfig`=2). Breaks the documented machine-readable contract. → Replace the bare macro with explicit `try/catch` around `app.parse`: CallForHelp→0, ParseError→print + return 2. ~15 lines.
- **[BLOCKER, M]** RUN-SHEET missing 6 of 14 subcommands → *(the §2 docs blocker #11)*.
- **[POST, S]** `kExitDeviceOom`(3) is dead code — a real VRAM OOM exits 5, never 3 (every command catches `std::exception`→`kExitRuntimeError`; verified the `Status::DeviceOom`→3 mapping at `exit_code.hpp:65` is unreachable). → Add a typed OOM catch or drop the documented code. *(same root as 3.6 item 1)*
- **[POST, M]** qpwave advertises qpAdm-fit-only inert flags (`--als-iters/--allow-neg/--jackknife/--p-se-threshold`) it cannot consume. → Split `add_qpadm_option_flags`.
- **[POST, S]** Exit-code category mismatches across commands for equivalent failures (qpfstats write→2 vs IoError; dates device-error→4 vs 5 everywhere else). → Harmonize per the taxonomy.
- **[POST, S]** `--device` help advertises multi-GPU `0,1` while multi-GPU is PARKED; only qpadm-rotate warns. → Trim help or hoist the warning.
- **[NICE, S]** Subcommand listing order not grouped; one terse no-CUDA error string in qpgraph-search.

### 3.4 Python API / packaging — **architecture release-grade (3 blockers)**
**State:** `_core.so` builds + `import steppe` succeeds on box5090; the facade exports 23 names covering all tools; the marshalling-only split is real (zero compute/pandas in `module.cpp`, host-only TU); pandas is a correct lazy soft-dep; `to_numpy` is FP64 F-contiguous with a capsule deleter; 11 pytest modules with a clean no-GPU SKIP. *(blockers 1-3 = §2 #3, #1, #9)*
- **[BLOCKER, L]** No wheel ever built → §2 #3.
- **[BLOCKER, S]** No LICENSE → §2 #1.
- **[BLOCKER, S]** README omits ~10 of 23 public functions → §2 #9.
- **[POST, S]** Missing PyPI metadata: no classifiers, no `[project.urls]`, no authors (verified). → Add.
- **[POST, M]** No `py.typed`/`.pyi` stub — rich annotations invisible downstream; `_core` untyped. → `nanobind_add_stub` + `py.typed`.
- **[POST, M]** **API↔CLI asymmetry**: the on-device sweep (`f4-sweep`/`f3-sweep`/`--all-quartets`) is **absent from Python** (verified no `run_*_sweep` in `module.cpp`); `qpadm_search` requires explicit `models` lists — the pool/min/max rotate sugar is "a future add" (`__init__.py:948`). → Add sweep binding + `pool=/min/max` kwargs.
- **[POST, M / NICE, S]** CUDA-runtime dep model diverges from spec (optional `[cuda]` extra w/ suffixed `nvidia-*-cu13` names that don't resolve, vs spec's base unsuffixed deps) — reasoned but must be a signed-off decision; `requires-python>=3.9`/`cmake>=3.28` skew vs spec's 3.10/3.30.

### 3.5 Build / CI / distribution — **mature build, missing release scaffold (4 blockers)**
**State:** the CMake layering is solid + compiler-enforced (CUDA PRIVATE to `steppe_device`); CMakePresets v6 (dev/release/ci) with the full Turing→Blackwell ship arch list; warnings-as-errors a real hard gate; the CUDA-13/CMake-3.28 sm_75 auto-pin trap correctly handled; no-network-first dep fetch. The release/CI/distribution scaffolding *around* the mature build is the gap. *(blockers = §2 #7, #1, #5, #3)*
- **[BLOCKER]** No CI → §2 #7. No LICENSE → §2 #1. No CUDA-13 floor → §2 #5. No reproducible/manylinux wheel + no version-source → §2 #3+#6.
- **[POST, M]** Dead/inert build options: `STEPPE_SANITIZER` (declared `SteppeOptions.cmake:45`, no `SteppeSanitizers.cmake` exists — verified), `STEPPE_NVTX` (never referenced), device-LTO (named, never wired). → Wire or delete.
- **[POST, S]** No CCCL/CUB pin or assert — architecture.md §3 warns CCCL silently drops to 3.0 on a 13.0 toolkit (the `cuda::execution` determinism API vanishes). Verified **no `find_package(CCCL)` anywhere**. → Add `find_package(CCCL)` + `CCCL_VERSION >= 3.1` assert.
- **[POST, S]** `cmake_minimum_required(3.28)` vs spec's 3.30; the release-preset's full 8-arch list has **no evidence of ever being built** (all box builds are the bare Debug path). → Verified release-preset build on box + reconcile the floor.
- **[POST, S]** No CHANGELOG, no tags → §2 #4. No `SteppeConfig.cmake`/public-header install (only the `_core` .so installs, `bindings/CMakeLists.txt:69`) — decide if a C++ lib is a deliverable or amend the spec to Python-only.
- **[POST, S]** CPM ships a fabricated placeholder `EXPECTED_HASH` and deliberately does NOT verify downloads (`CPM.cmake:17`+`:28`) — supply-chain/reproducibility gap. → Pin real hashes or vendor an offline cache.
- **[POST, M]** The `[cuda]` extra is non-functional (suffixed cu13 names unpublishable) and the base wheel's CUDA-on-loader-path prerequisite is invisible to users. → Document + friendly import-time CUDA-presence check.

### 3.6 Robustness / formats / error-handling — **production-grade (1 doc blocker)**
**State:** the domain-outcome-as-value discipline is correct + pervasive (RankDeficient/NonSpdCovariance/ChisqUndefined returned as `Status` values, never throws; `exit_code_for` maps them to exit 0 — the S8 rotation never aborts on one degenerate model). Input validation is thorough fail-fast (header/magic/stride/size, overflow-guarded parse, heap-overflow gate unit-tested, SNP-axis-desync guards). TGENO-only is enforced at the boundary with a descriptive throw, consistent across all 4 genotype tools. Determinism is engineered (single stat stream, fixed workspace, `CUBLAS_PEDANTIC_MATH`, bit-exact decode).
- **[BLOCKER, S]** TGENO-only not stated user-facing → §2 #10.
- **[POST, S]** `DeviceOom` status/exit-3 is dead in the runtime fault path — a real `cudaErrorMemoryAllocation` is caught as generic `std::exception`→exit 5 (every device command; verified). Contract-vs-reality drift. → Add a typed/classifier OOM catch arm (CUDA-free seam, since `CudaError` is device-private). *(same root as 3.3 item 3)*
- **[POST, L]** Older GENO/EIGENSTRAT/PA readers absent (the format is *detected* at `eigenstrat_format.cpp` but `read_tile` refuses GENO) — the biggest "can I run this on my data" barrier; deferred per standing constraint.
- **[POST, M]** No AT2 f2-stat interop importer — steppe reads only its own STPF2BK1; AT2 users with extracted f2 can't feed the headline `read_f2` path. **The cheapest meaningful unlock for the existing AT2 user base.**
- **[POST, M]** All-NaN/zero-coverage f2 propagation is implicit (surfaces as a correct domain outcome but no "this pop has no coverage" diagnostic; `maxmiss` filter defaults to no-op).
- **[NICE, S]** No left/right/target overlap validation in qpadm model resolution (a typo → confusing domain-outcome row instead of a clear config error).

### 3.7 Release-eng / licensing / deferred-triage / roadmap — **(5 blockers, all hygiene)**
**State:** the wheel build is correct + clean-venv-install-proven (RESUME M(py-2)); the clean-room posture is genuinely clean (no vendored AT2/DATES/convertf source); raw AADR genotypes correctly gitignored (`.gitignore:54-57`), committed goldens are aggregate summary stats not sample-level; golden provenance meticulously documented; no secrets/keys tracked. *(blockers = §2 #1, #2, #4, support-matrix, IP-leak)*
- **[BLOCKER]** No LICENSE (§2 #1), MIT-vs-GPL posture undocumented (§2 #2), no tag/release/publish path (§2 #4), support matrix only implicit (→ add a "Requirements / Support matrix" README section: CUDA 13+/driver, sm_120 prebuilt + how to rebuild, single-GPU, TGENO-only, Linux x86_64), agentscripts IP-leak (P-1).
- **[POST, S/M]** CHANGELOG (§2-adjacent), version single-source (§2 #6), CI (§2 #7), AADR data-use-terms acknowledgment for the committed derived goldens.
- **Deferred-item triage verdict:** **NONE of the 4 named deferrals is a blocker.** L1-L4 are bounded per-model diagnostics off the data-scaling path; the 6-pop topology-search scale-up *extends* a working+gated 5-pop search; constrained-weights qpAdm is an optional non-default AT2 mode (and the dead `QpAdmOptions::constrained` field was already removed in `e8430a2`, so no half-done seam ships); older readers are a documented scope boundary. All are POST-RELEASE **provided the support matrix states the limitations honestly.**

### 3.8 Perf / scale — **the cleanest dimension (ZERO blockers)**
**State:** the host-compute-audit campaign is COMPLETE and code-verified (not just doc-asserted): the jackknife family, decode seam, dates loops, single-model SE are on-device. The f4/f3 O(N²) dense-covariance OOM is **fixed** (`f4.cpp:120`/`f3.cpp:117` call `jackknife_diag`). The all-quartets sweep is fully on-device (unrank kernel, `cuda_backend.cu:2801`, no host enumeration) + bounded top-K reservoir (2.57B quartets in 177s, host RAM bounded 3.1GB). Decode front-end resident seam serves the 3 high-volume genotype tools. M5 large-P streaming makes the precompute footprint M-independent. Every number is real-AADR, single-GPU, Release-build.
- **[POST, S]** Two perf-doc characterizations of the sweep are **stale** — `production-pass.md:68/114` call it "CPU-ENUMERATION-BOUND" with a deferred "on-device enumeration"; the code already does on-device unrank (the residual host cost is the survivor CSV write + chunk driver). → Doc-only correction.
- **[POST, M]** The DATES decode twin uses legacy `decode_af` (full D2H, `dates.cpp:146`) unlike the other 3 tools — but tiny (3-pop, ~86MB, 832MiB peak VRAM); consistency cleanup.
- **[POST, L]** Synchronous single-shot `read_tile` (no IO/decode overlap) — the structural reason genotype stats are decode-bound at small N; walls are seconds, latency-only post-release lever.
- **[POST, S]** Cold-cache decode never measured (all numbers warm); f4-ratio standalone SE jackknife may hit a host loop at large explicit N (verify it routes through the fused M1 seam); the qpGraph fleet-at-scale number is cited-not-rerun.
- **[NICE, S]** L1-L4 bounded host diagnostics.

---

## 4. Recommended ordered sequence to reach release

**Phase 0 — Legal/safety gate (½ day, all S):** add `LICENSE` (#1) + `ATTRIBUTION.md` clean-room note (#2); scrub/relocate `agentscripts/` + delete `handoff-ba37d95.md` (P-1) and verify the sdist excludes them; `git stash drop` the FAILED stash; gitignore `atlas_results/`; decide `tests/tools/stage_qpgraph_f2dir.cpp`.

**Phase 1 — Build correctness (½ day):** add the CUDA-13 `find_package` floor (#5); single-source the version + add `version.hpp` + `steppe.__version__` (#6); add the CCCL>=3.1 assert. *These make a from-source build honest before any wheel.*

**Phase 2 — Wheel + sign-off (1 day, the L item):** build inside `nvidia/cuda:13-devel`, `auditwheel repair` (verify the real cusolver SONAME via `ldd` first), clean-venv `import steppe` + one real-AADR fit (#3). In parallel: a full **box5090 ctest with `STEPPE_AADR_ROOT` + `STEPPE_THOROUGH=1`** as the formal correctness sign-off (the §2 gate). Also verify the release-preset's 8-arch list actually compiles end-to-end.

**Phase 3 — Docs front door (½-1 day):** delete RUN-GUIDE.md (#8); rewrite README (#9) with the support matrix + TGENO-only "Supported inputs" (#10, support-matrix); extend RUN-SHEET with the 6 missing tools (#11); refresh the stale TODO/ROADMAP/RESUME status lines (3.2 blocker); add CHANGELOG. Fix the 4 qpdstat Part-B stale comments + the one stale pytest docstring while in the docs pass.

**Phase 4 — CLI contract + CI (½-1 day):** the CLI11 parse-error→exit-2 fix (15 lines, 3.3 blocker); stand up the cheap **no-GPU CI** job (#7 first leg). Tag `v0.1.0`, cut the GitHub release with the wheel asset (#4).

**Then ship 0.1.0.** Everything in §3 marked POST/NICE follows as a hardening sprint (exit-code harmonization, the DeviceOom typed catch, py.typed stubs, the API↔CLI sweep/rotate parity, the AT2-f2 importer, the GPU CI leg, CPM hash pinning).

---

## 5. Clean bill (already release-ready)

- **Perf/scale** — the cleanest dimension, zero blockers; the host-compute campaign is complete and code-verified, the f4/f3 OOM is fixed, the sweep is on-device + bounded, M5 makes the footprint M-independent. **Ship it.**
- **The validated math** — every shipped command has an engine parity gate + CLI golden + (mostly) a pytest gate; CpuBackend==CudaBackend on every fit path; degenerate/missing-block/INT_MAX/overflow edge cases each have a dedicated gate; the one defective golden is correctly removed + superseded.
- **The shipping `src/`** — §4 layering 100% CUDA-free in app/bindings/core/io; zero dead code; documented TODO hygiene; consistent naming; clang-format/clang-tidy committed.
- **The CLI architecture** — 14 subcommands all dispatching to real GPU compute; uniform flag vocabulary; thorough centralized validation; the record-and-continue domain-outcome contract correct; top-level catch.
- **The bindings architecture** — marshalling-only split, lazy-pandas soft-dep, FP64 capsule-deleter `to_numpy`, clean no-CUDA ValueError, `_core.so` builds + imports on box5090.
- **The CMake build proper** — compiler-enforced layering, correct presets, warnings-as-errors hard gate, the sm_75 auto-pin trap handled, no-network-first deps.
- **Robustness** — domain-outcome-as-value discipline, fail-fast io validation, enforced TGENO boundary, engineered determinism, runtime VRAM/RAM tier probing.
- **Release-eng fundamentals** — clean-room posture genuinely clean (no vendored GPL source), raw AADR gitignored + committed goldens are aggregate-only, meticulous golden provenance, no tracked secrets, the wheel build + clean-venv install already proven once.

---

## 6. Forward roadmap (PLANNING-STAGE — not yet scoped, not yet designed)

> The post-release feature directions the user named — **population creation / simulation, imputation, msprime integration, and other popgen features**. This is a SHAPE and an architecture-fit note, **not a design**: each new modality needs its own GPU-shape design doc (the project's established `docs/research/*` pattern) before any work.

**The three load-bearing seams new features attach to** (already in the codebase):
- **(A) the genotype-stat seam** — the `GenoReader`→`read_tile`→`decode_af`→Q/V/N→reduce front-end, reused verbatim by qpDstat-B / qpfstats / DATES. **Input-side** features attach here.
- **(B) the f2 cache** — the device-resident/tiered `f2_blocks` tensor every fit + f2-path stat reads. **Model/analysis-side** features attach here.
- **(C) the fleet** — the batched-sequential on-device optimizer (one launch, one thread per restart, the whole optimize loop in-kernel) productized for qpGraph topology search. The **reusable on-device-optimizer engine** for any new per-model fit.

**How the named features fit:**
- **msprime integration** — already the documented flagship interop loop (`docs/research/interop-usecases.md`: "msprime/tskit sim → qpAdm power analysis"). Shape: a documented recipe + a DLPack/`__cuda_array_interface__` device-accepting Q/V/N import entry so a simulated TreeSequence's allele-frequency reduction feeds straight into the f2 cache without a host bounce. **LOW new compute** — an annotation on the existing import entry + a doc recipe. **The cleanest first forward feature: high value, small surface, lands on seam (A)/(B).**
- **Population creation / simulation** — largely *satisfied by* the msprime path: steppe consumes msprime/stdpopsim output via the genotype-stat seam rather than owning a coalescent simulator. A native forward/coalescent sim would be a new modality (not f-stat-shaped) — large, low priority vs consuming external sims.
- **Imputation** — surveyed as a high-value but **separate modality** (consumer-chip ~0.7%-coverage gating hosted qpAdm). It is an upstream genotype-completion stage that attaches *before* the genotype-stat seam (a new `io/preprocess` stage emitting the same harmonized tile, like the S-2/S-1 QC pre-pass). The **largest** of the named items; a distinct compute path; correctly characterized as roadmap-fit, not near-term.
- **Other popgen (PCA / ADMIXTURE-validation / LAI)** — each a separate path attaching at seam (A), broadening steppe from an f-stat engine toward a full hosted aDNA pipeline backend.

**Rough recommended forward order** (per the project's backend-first-then-accessible rule, each new modality WITH its own CLI/bindings):
1. **Older GENO/EIGENSTRAT readers** — unblocks non-v66 input; isolated io-leaf change (format-detect-on-magic + SNP-major/EIGENSTRAT dispatch in `io::GenoReader`); the cheapest user-value win.
2. **msprime interop recipe + device Q/V/N import** — the flagship loop; small surface; enables simulation-based power analysis.
3. **The deferred fit features as demanded** — 6-pop topology search scale-up, non-negative constrained-weights qpAdm.
4. **The big new modalities** — imputation, then PCA/ADMIXTURE/LAI — each a separate path with its own GPU-shape design doc.

---

*Methodology: synthesized from seven per-dimension read-only audits, every cross-cutting claim re-verified against HEAD `1707460` (LICENSE/tags/CHANGELOG/CI absence; pyproject metadata; version duplication; `find_package(CUDAToolkit)` floor; 14 CLI subcommands; `CLI11_PARSE` macro; agentscripts IP-leak; the FAILED stash + wip branch; untracked atlas_results/tests-tools; RUN-GUIDE/README/TODO drift; RUN-SHEET tool coverage; qpdstat Part-B stale comments; DeviceOom dead exit-3; TGENO throw; f4/f3 jackknife_diag; on-device sweep unrank; CPM placeholder hash; absent CCCL pin / SteppeSanitizers / version.hpp; orphan .rds/edges goldens; qpgraph constexpr drift). Standing constraints respected, not relitigated: single-GPU-first, EmulatedFp64{40} default, CUDA 13+, real-AADR-only, TGENO-only decode, CpuBackend test-oracle-only.*

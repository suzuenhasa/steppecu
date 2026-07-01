# steppe — RELEASE-READINESS PLAN (v0.1.0)

> **Scope:** the path from the current tree to a clean, credible, public 0.1.0 release of **steppe** — a GPU-only (CUDA 13 / Blackwell sm_120) reimplementation of ADMIXTOOLS 2 (f-statistics / qpAdm / qpWave / qpGraph / DATES), bit/tolerance-validated vs AT2 goldens on real AADR, distributed as a Python wheel (`steppe._core` nanobind extension) + a `steppe` CLI.
> **This document SCOPES and SEQUENCES — it does not execute.** Every claim below is anchored to a real file. It deliberately does **not** re-litigate the validated math, the `src/` layering, or the perf story — those are clean (see `docs/RELEASE-SCOPE.md` §3.8, §5). It complements `docs/RELEASE-SCOPE.md` (the 11-blocker audit) and `docs/ROADMAP.md` (internal build-order) rather than duplicating them; where RELEASE-SCOPE blockers are now **resolved** (LICENSE exists, `__version__` exists, README is a real front door, multi-format readers shipped), this plan says so and moves on.

---

## 1. What "release-ready" means for a GPU-only scientific tool

steppe is not a pure-Python package and not a CPU library. A "release version" here has to satisfy four things at once, and the distribution story is part of the product, not an afterthought:

1. **A versioned, immutable artifact set.** A tagged `v0.1.0` with named, hash-verified binaries (a wheel + a container image), so a researcher can cite *exactly* what they ran. Today there are **zero git tags** (`git tag -l` empty) and no release.
2. **A correctness contract the audience can trust.** The deliverable is believable only if the parity-vs-AT2-on-real-AADR story is front-and-center and reproducible. This is steppe's single biggest credibility asset and it is currently buried in internal docs.
3. **An honest, installable runtime contract.** The wheel `DT_NEEDED`s `libcudart.so.13` / `libcublas.so.13` / `libcusolver.so.12` / `libcufft.so.12` (pyproject.toml:8-9) and resolves them at load — the CUDA-13 toolkit is a **system requirement, deliberately not bundled**. "Release-ready" means every install lane states that contract clearly and at least one lane satisfies it turn-key for a user who cannot install a CUDA toolkit.
4. **A repo a popgen reviewer reads as maintained.** ~440 internal-dev `.md` files (`docs/kimireview/` 117, `docs/cleanup/` 134, `docs/release_cleanup/` 129, plus `RESUME/TODO/ROADMAP/REORIENT/NEXT-STEPS/RELEASE-SCOPE`) currently ship in the sdist and sit on the public front page. That is the "is this maintained or abandoned?" smell. The wheel itself is already clean (`wheel.packages = ["bindings/steppe"]`, pyproject.toml:85) — **the leak is the sdist and the repo presentation, not the wheel.**

**Bottom line:** the math, the engine, and the perf are release-grade. What blocks 0.1.0 is (A) repo hygiene / what-ships, (B) a user-facing doc set + the validation story, (C) the distribution build-out, and (D) the release mechanics (license metadata, version tag, changelog). All four are addressable in a single focused release-engineering pass; the only genuinely large items are the manylinux wheel + the container.

**0.1.0, not 1.0.0.** The version is already `0.1.0` (CMakeLists.txt:20, single-sourced into the wheel via the scikit-build-core regex provider, pyproject.toml:93-96). `0.x` correctly signals a pre-1.0 research tool whose Python facade + CLI flags are still settling (parked multi-GPU, the API↔CLI sweep asymmetry). Reserve `1.0.0` for a frozen public API + format support.

---

## 2. DISTRIBUTION RECOMMENDATION (decisive, researched)

**Audience first.** The users are population-genetics / ancient-DNA researchers. The dominant deployment is an **HPC cluster where they have no root, Docker is banned or unavailable, and Apptainer/Singularity + environment-modules + conda are the norm.** A minority run a single workstation or cloud GPU box. This shapes everything below. The losing move is to ship only a PyPI wheel and assume `pip install` "just works" — it cannot, for a single-arch, system-CUDA, GPU-only package.

### The ranked lanes for 0.1.0

| Rank | Lane | Who it serves | Status today | Verdict |
|---|---|---|---|---|
| **1 — PRIMARY** | **Apptainer/Singularity `.sif`** (prebuilt, SHA256-signed, GitHub-release asset) | HPC / cluster users (no root, no Docker) — the *majority* of the audience | **Missing** (no `.def`, no Containerfile) | **Build it. This is the decisive artifact.** |
| **2 — SECONDARY** | **manylinux_2_28 `abi3` wheel** (sm_120, auditwheel `--exclude` the CUDA SONAMEs), GitHub-release asset | Workstation / cloud-GPU users; anyone with CUDA 13 already on the loader path | Built + clean-venv-imported **once** (`c935b28`); manylinux/auditwheel portability **unproven**; `[cuda]` extra **broken** | **Prove + ship; fix the `[cuda]` extra.** |
| **3 — same source** | **One OCI Containerfile** → feeds both the `.sif` (push to `ghcr.io` → `apptainer pull`) and Podman/Docker users (CDI `--device nvidia.com/gpu=all`) | Workstation users with root who prefer containers | Missing | **Single source of truth for #1 and #3.** |
| **DEFER** | **PyPI** | discovery / `pip install steppe` muscle memory | n/a | Await **Wheel Variants** stabilizing; a single-arch system-CUDA wheel is awkward on PyPI today. |
| **DEFER (post-1.0)** | **conda-forge** (NOT bioconda) | the conda-heavy HPC slice | n/a | conda-forge CI has **no GPU** → cannot run steppe's golden-gated validation; bioconda has no CUDA build support. The audience is already covered by the `.sif`. |

### Why this order — the killer facts (researched, not guessed)

- **Apptainer `apptainer exec --nv` is the ONE thing that runs unprivileged on a locked-down cluster.** `--nv` binds the host NVIDIA driver libs + `/dev/nvidia*` into the container; it needs **no daemon, no nvidia-container-toolkit, no root, no CDI**. The image bakes the CUDA 13 runtime (FROM `nvidia/cuda:13.0.x-runtime`), so the only host requirement is a driver. CUDA 13 needs host driver **>= 580**. This removes the `LD_LIBRARY_PATH` / "is CUDA 13 a cluster module?" friction *entirely* for exactly the users who can't `pip install` a toolkit. Source: Apptainer GPU docs; HPC center guides (apptainer.org, hpc.nih.gov).
- **Building a `.sif` can require `--fakeroot`/root, which cluster users often lack** → ship a **prebuilt** `.sif` as a release asset so they just download + run. The `.def` + Containerfile live in-repo for reproducibility.
- **The `[cuda]` pip extra is a real, shipped BUG.** pyproject.toml:61-66 pins `nvidia-cuda-runtime-cu13>=13,<14` etc. As of 2025-10-31 those **`-cu13`-suffixed names are 0.0.1 deprecation stubs** ("THIS PROJECT IS DEPRECATED, use `nvidia-cuda-runtime`"), so `>=13` is **unsatisfiable → `pip install steppe[cuda]` fails today.** CUDA 13 renamed the redistributables to **unsuffixed**: `nvidia-cuda-runtime` (~13.x), `nvidia-cublas` (13.6.0.2), `nvidia-cusolver` (12.x), `nvidia-cufft` (12.x). The pyproject header comment (lines 12-17) asserting "only 0.0.x placeholders exist" is now **stale on two counts**. Fixable → the extra can finally be made functional.
- **auditwheel `--exclude` is the correct, established pattern** (Qiskit-Aer, CUDA-Quantum ship this way) and matches steppe's own design (`DT_NEEDED`, resolve at load). Excluding the four CUDA SONAMEs keeps the wheel ~single-MB; a naive `auditwheel repair` would **bundle the toolkit and balloon a <2MB extension past ~400MB** while staying manylinux-compliant either way. The CUDA libs stay `DT_NEEDED` and resolve from system CUDA / `LD_LIBRARY_PATH` / the fixed `[cuda]` extra.
- **The exact SONAME exclude list is confirmed two ways** (CUDA 13.0 release-note component versions + the local cu12 wheels on this box, which show cusolver.11/cufft.11 = toolkit-major − 1): cuSOLVER and cuFFT carry **major `.12` under CUDA 13**, cudart/cublas carry `.13`. So:
  `auditwheel repair --exclude libcudart.so.13 --exclude libcublas.so.13 --exclude libcusolver.so.12 --exclude libcufft.so.12`.
  This **answers RELEASE-SCOPE #5's open "verify cusolver .12 vs .13" question: it is `.12`** for both cusolver and cufft. Verify once more with `readelf -d` of the built `_core.so` on the box before locking the CI command.
- **abi3 collapses the build matrix.** nanobind supports `Py_LIMITED_API`/abi3 → one `cp39-abi3-manylinux_2_28_x86_64` wheel instead of one per cp39..cp313. A big maintenance win for a single-arch GPU package.
- **Forward-compat fatbin.** pyproject pins `STEPPE_CUDA_ARCH=120` (sm_120 cubin only, pyproject.toml:82). Switch to `CMAKE_CUDA_ARCHITECTURES="120-real;120-virtual"` (native sm_120 cubin + embedded PTX that JITs on future Blackwell+).
- **PyPI deferral is principled, not lazy.** GPU/CUDA on PyPI is an acknowledged unsolved problem (pypackaging-native key-issues/gpus). The real long-term fix is **NVIDIA Wheel Variants** (`nvidia::cuda_version_lower_bound` / `nvidia::sm_arch::120_real`) — but it is **experimental** in pip/uv/PyTorch 2.8 and the PEP is in draft. **Do not block 0.1.0 on it.**
- **The CLI is not in the wheel — and the `.sif` solves that too.** `STEPPE_BUILD_CLI=OFF` (pyproject.toml:76) + no `[project.scripts]` means `pip install steppe` gives the Python package only — **no `steppe` command on PATH**, even though README advertises 14 subcommands. The `.sif` (built with `STEPPE_BUILD_CLI=ON`) naturally carries *both* the C++ CLI binary and the Python env in one file — a second reason it is the primary lane. (Decision needed: also add a thin `[project.scripts]` argparse shim over the facade? See Open Decisions.)

**Sources:** scikit-build-core config (sdist = git-tracked minus .gitignore; `sdist.exclude` is the lever) https://scikit-build-core.readthedocs.io/en/stable/configuration/ · GPU-wheel SOTA https://pypackaging-native.github.io/key-issues/gpus/ · Wheel Variants https://developer.nvidia.com/blog/streamline-cuda-accelerated-python-install-and-packaging-workflows-with-wheel-variants/ · CUDA 13 wheel layout / metapackages https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/ · deprecation stub https://pypi.org/project/nvidia-cuda-runtime-cu13/ · unsuffixed cuBLAS https://pypi.org/project/nvidia-cublas/ · CUDA 13.0 component versions (SONAMEs) https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/ · auditwheel `--exclude` https://github.com/pypa/auditwheel · cibuildwheel custom CUDA image https://cibuildwheel.pypa.io/en/stable/options/ · real-world CUDA wheel repair https://github.com/Qiskit/qiskit-aer/blob/main/.github/workflows/deploy.yml · Apptainer `--nv` https://apptainer.org/user-docs/3.8/gpu.html · Podman CDI https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html · conda-forge CUDA https://anaconda.org/conda-forge/cuda-version

---

## 3. WORK-STREAM A — Repo hygiene / what-ships

**Core finding:** the **wheel is already clean** (`wheel.packages = ["bindings/steppe"]` + `install(TARGETS _core LIBRARY DESTINATION steppe)` → exactly `steppe/__init__.py` + `steppe/_core*.so` + dist-info). **The entire internal-dev leak is via the SDIST**, because pyproject has **no `[tool.scikit-build.sdist]` table at all** → scikit-build-core's default sdist includes every git-tracked file not matched by `.gitignore`. **MANIFEST.in is the WRONG lever — scikit-build-core ignores it entirely** (it is a setuptools mechanism). The correct lever is **`sdist.exclude`**.

**Archive-vs-delete principle:** ARCHIVE the valuable campaign docs/tooling (institutional provenance, incl. `docs/cleanup/bigrefactor/NAMING-STYLE-STANDARD.md` still cited by recent commit messages); DELETE only regenerable untracked strays. A **history rewrite is already mandated** before the repo goes public (vast.ai IPs live in prior commits, RELEASE-SCOPE §2 P-1) — that is the natural moment to snapshot-then-archive. **Do not keep 6MB+ of campaign chatter in-tree-but-unpackaged long-term** — that is exactly the "abandoned?" front-page smell.

### A. The internal-exclude / disposition list (what leaves the public tree + sdist)

| Path | What it is | Disposition |
|---|---|---|
| `docs/kimireview/` (117 tracked, incl. `kimiwhole/`) | per-TU code-review campaign output | **archive-branch** |
| `docs/cleanup/` (134 tracked, incl. `bigrefactor/`, `m4.5/`, `m5/`) | big-refactor campaign + NAMING-STYLE-STANDARD | **archive-branch** (cited by commits — preserve) |
| `docs/release_cleanup/` (129 tracked + untracked `extra/`) | release-cleanup per-file + X1..X7 findings | **archive-branch**; `extra/` → **delete** |
| `docs/kimiactions/` (8 tracked) | CI plan / low-polish / cuda-split action lists | **archive-branch** |
| `docs/10codereview/` (7, **untracked**) | latest review campaign, regenerable | **delete** |
| `docs/{RESUME,REORIENT-PROMPT,NEXT-STEPS,TODO}.md` | internal handoff/status; TODO.md is **stale** (asserts "no CLI/no bindings", false at HEAD) | **archive-branch** |
| `docs/RELEASE-SCOPE.md` | this readiness audit (internal parent) | **archive-branch** |
| `docs/ROADMAP.md` (36KB) | internal build-order (stale status section) | **archive-branch**; replace with a short public roadmap if any |
| `docs/BOX-RUNBOOK.md` | vast.ai dev-box ops (box-specific IP/path detail) | **exclude-from-sdist** (archive) |
| `docs/RUN-GUIDE.md` | **actively WRONG** ("no CLI / no Python binding yet") | **delete** (RUN-SHEET/new docs supersede) |
| `docs/kimiexample.md` (**untracked** stray) | scratch | **delete** |
| `experiments/` (12 tracked spikes) | M0-era throwaway, not in CMake | **archive-branch** |
| `scripts/box_bringup.sh`, `scripts/p2p_probe.cu` | dev-box bringup + P2P probe (not user runtime) | **exclude-from-sdist** / relocate under gitignored `agentscripts/` |
| `agentscripts/` (~516K, untracked, carries box IPs) | workflow orchestration | **keep gitignored** (line 69 ✓) |
| `aadr/` (3.9G, untracked) | real AADR data, regenerated never committed | **keep gitignored** ✓ |
| `.ruff_cache/`, `.claude/`, `atlas_results/` | lint cache / harness / run-output | **gitignore** (currently **uncovered** — gap) |
| `tests/reference/goldens/at2/{golden_qpfstats_geno.rds, golden_qpgraph_fit.rds, golden_qpgraph_edges.csv}` | orphan goldens — **read by NO test** (verified: only the generator R script `scripts/golden_qpgraph_generate.R` references them) | **delete** (dead weight in the 15M tests/ sdist payload) |

### A — task table

| ID | Title | What | Effort | Deps | Source |
|---|---|---|---|---|---|
| **A1** | Add `[tool.scikit-build.sdist].exclude` | Enumerate `docs/{kimireview,cleanup,release_cleanup,kimiactions,10codereview}`, `docs/{RESUME,REORIENT-PROMPT,NEXT-STEPS,TODO,ROADMAP,RELEASE-SCOPE,BOX-RUNBOOK}.md`, `experiments/`, `scripts/box_bringup.sh`, `scripts/p2p_probe.cu`. **Do NOT add MANIFEST.in** (inert under scikit-build-core). | S | — | H1/D2/P9/R7 |
| **A2** | CI gate: assert sdist + wheel against an allowlist | `python -m build`; `tar tzf` the sdist → fail if any internal-dev path appears; `unzip -l` the wheel → assert it is exactly `steppe/__init__.py` + `steppe/_core*.so` + dist-info. Codify so leaks regress loudly. | S | A1 | H2 |
| **A3** | Archive the internal campaign trail privately, then remove from public tree | Tag/branch the full current tree on the private remote (`git tag internal-snapshot-pre-public` and/or `dev/internal-history`) preserving full provenance; remove the trees above from public `main`. **Pairs with the mandated pre-public history rewrite** (RELEASE-SCOPE P-1). | M | A1 | H3/D2 |
| **A4** | Delete ephemeral strays + close `.gitignore` gaps | Delete `docs/10codereview/`, `docs/kimiexample.md`, `docs/release_cleanup/extra/`, `.ruff_cache/`. Add `.ruff_cache/`, `.claude/`, `atlas_results/` to `.gitignore` (currently only `agentscripts/`+`aadr/*` covered). | S | — | H4 |
| **A5** | Trim dead/heavy tests payload | Delete the 3 orphan goldens (`golden_qpfstats_geno.rds`, `golden_qpgraph_fit.rds`, `golden_qpgraph_edges.csv`) read by no test, **or** wire an edge-length gate for the qpGraph branch lengths (an untested surface — preferred if cheap). Keeps the 15M sdist honest. | S | — | H9 |
| **A6** | Relocate dev-box scripts out of `scripts/` | Move `scripts/box_bringup.sh` + `scripts/p2p_probe.cu` under gitignored `agentscripts/` (or rely on A1's exclude). Leave only `regenerate_goldens.sh` as the product tool-of-record. | S | A1 | H10 |

---

## 4. WORK-STREAM B — Documentation

**Core finding:** ~470 `.md` files in `docs/`, but **~440 are the internal-dev dump.** The actual user-facing surface is tiny. The job is **90% triage/archive + 4 new docs**, not authoring volume. The README is already 80% right (requirements table, install, 14-command CLI table, Python quickstart, MIT note) — a light rewrite, not a redo.

**The single highest-leverage new artifact is `docs/VALIDATION.md`** — the credibility doc. Assemble it from material that **already exists and is high quality**:
- the golden-parity record in `tests/reference/goldens/at2/README.md` (§12 reproduction record: admixtools 2.0.10 / R 4.3.3 / v66.p1_HO, blgsize 0.05, fudge 1e-4, exact weights/chisq/p, rtol tiers) + `docs/feature-matrix.md` (every shipped command, golden-match status, real-AADR wall-clock; every number golden-anchored, zero synthetic);
- the published-study reproductions: `docs/studies/haak2015.md` (Corded Ware steppe 0.743 vs AT2 0.742), `docs/studies/olalde2018.md`, `docs/studies/olalde2018-rotation.md`;
- **the central credibility hook — the TGENO-corruption discovery** (`docs/research/tgeno-at2-support.md` + the goldens README CORRECTION block): **AT2 v2.0.10 silently misreads v66 TGENO; steppe's decode was the correct one; goldens were regenerated via `convertf`→PA.** This is a genuine "we found a bug in the reference and proved we're right" story — **lead with it.**

**The reference docs (`CLI.md`, `PYTHON-API.md`) should be derived, not hand-narrated:** CLI.md from `steppe <cmd> --help` for all **14** subcommands (the source of truth — replaces the stale `RUN-SHEET.md`, pinned to `main @ 433b71e` and missing qpgraph/qpgraph-search/dates/qpfstats); PYTHON-API.md from the `bindings/steppe/__init__.py` docstrings via the existing `STEPPE_BUILD_DOCS` pdoc target (`docs/api/README.md`).

### B. Doc inventory verdict (representative — full list in DOCS audit)

| Class | Paths | Fate |
|---|---|---|
| **user-facing (keep/light-edit)** | `README.md` (rewrite-light), `LICENSE` (keep), `examples/README.md` (keep), `docs/api/README.md` (keep + wire) | front door |
| **new user-facing** | `docs/{INSTALL,QUICKSTART,CLI,PYTHON-API,VALIDATION,CONTAINERS}.md` + `CONTRIBUTING.md` + `CHANGELOG.md` | author |
| **reference (keep, demote from front door)** | `docs/architecture.md` (reconcile [STALE] tags), `docs/design/{fit-engine,cli-bindings,format-readers}`, `docs/research/{tgeno-at2-support,block-partition-at2,f2-estimator-at2}`, `docs/studies/*`, `docs/perf/*` (fix stale "CPU-enumeration-bound" lines), `docs/feature-matrix.md` (merge into VALIDATION) | keep, lightly pruned |
| **reference (archive the spikes)** | `docs/research/{turboquant-l2-*,optimizer-comparison,qpgraph-optimizer-spike,pycuda-cuda13-viability}` | archive |
| **internal-dev (archive)** | all of Work-Stream A's archive set | archive |
| **stale/wrong (rewrite or delete)** | `RUN-GUIDE.md` (delete), `RUN-SHEET.md` (superseded by CLI.md), `NEXT-STEPS.md` (archive) | per A-list |

### B — task table

| ID | Title | What | Effort | Deps |
|---|---|---|---|---|
| **B1** | Stand up the clean user-facing doc skeleton | Create `docs/{INSTALL,QUICKSTART,CLI,PYTHON-API,VALIDATION,CONTAINERS}.md` + `CONTRIBUTING.md` + `CHANGELOG.md` as empty-but-headed stubs so the archive move (A3) doesn't orphan links. | S | — |
| **B2** | Rewrite README as the pure front door | Add a one-line value prop + a credibility line ("bit/tolerance-validated vs ADMIXTOOLS 2 on real AADR; reproduces Haak 2015 & Olalde 2018" → link VALIDATION); replace the 3 prose links with the B1 set; add a **Distribution** section (Apptainer `.sif` primary, wheel + `[cuda]` secondary, build-from-source); **reconcile the exact runtime SONAMEs** (state `libcudart.so.13` / `libcublas.so.13` / `libcusolver.so.12` / `libcufft.so.12`, not generic "libcusolver"); add the f2-dir interop caveat (steppe reads STPF2BK1, **not** AT2 `.rds`); move all build/box/dev content OUT. | M | B1 |
| **B3** | Write `docs/VALIDATION.md` (the credibility doc) | Methodology (real-AADR-only, AT2-as-oracle, tolerance tiers) → per-statistic golden-parity table → the two reproduced studies → the TGENO-corruption discovery (lead) → reproduce-it recipe (`scripts/regenerate_goldens.sh` + the ctest harness). **The single highest-value new artifact.** | L | B1 |
| **B4** | Write `docs/INSTALL.md` | Requirements matrix (CUDA 13 + Blackwell sm_120 + driver ≥580); the three channels; the exact runtime SONAMEs verified by `ldd`/`readelf -d` on the box; `LD_LIBRARY_PATH=/usr/local/cuda/lib64` guidance; the `[cuda]` extra decision (C-stream). Absorbs the README requirements block. | M | B2, C1 |
| **B5** | Generate `docs/CLI.md` from `--help` for all 14 subcommands | Capture `steppe <cmd> --help` per subcommand on the box; add one real-AADR invocation each (mine `RUN-SHEET §1-7`). Replaces stale RUN-SHEET; include the `dates` gotcha (the `.snp` must carry a real cM genetic map, `cli_parse.cpp:431`). | M | B1 |
| **B6** | Generate `docs/PYTHON-API.md` + wire the pdoc target | Wire `STEPPE_BUILD_DOCS` pdoc to emit the facade reference from the 23 `__all__` names; hand-write a short top page. Note the API↔CLI gaps honestly (sweeps absent from Python; `qpadm_search` needs explicit `models`). | M | B1 |
| **B7** | Reconcile `architecture.md`, demote from front door | Strip the self-flagged `[STALE]` tags (lines 18, 46): multi-format readers now shipped (not TGENO-only), LICENSE/`__version__` now exist, CUDA-13 is the real floor. Keep as a design reference linked from CONTRIBUTING, **not** README. Do not delete (canonical spec + portfolio material). | M | A3 |
| **B8** | Write `CONTRIBUTING.md` | The legitimate dev content distilled from RUN-GUIDE + BOX-RUNBOOK + ROADMAP §5-6: build on a CUDA-13 Blackwell box, Release-only for perf, ctest default vs `STEPPE_THOROUGH`, CpuBackend-is-test-oracle-only, layering/RAII gates; point at the archive branch for deep history. Replaces RUN-GUIDE as the dev front door. | M | A3 |
| **B9** | Triage `research/`+`studies/`+`perf/`+`design/` | Keep the VALIDATION-feeders + studies + design; **fix the stale "CPU-enumeration-bound" lines in `perf/production-pass.md:68/114`** (the sweep is on-device unrank now); archive the speculative spikes. Light prune. | S | A3 |
| **B10** | Final link-audit + stale-claim grep | Grep the shipped doc set for residual `TGENO-only` / `not yet built` / `no CLI` / pinned-commit / box-IP strings; verify every internal link resolves post-A3; confirm the 14-subcommand count + 23 Python names match the code. One clean read-through before tagging. | S | B2,B3,B4,B5,B6,B7,B8 |

---

## 5. WORK-STREAM C — Distribution build-out

**Core finding:** the wheel is **FEASIBLE and already built+imported once** (`c935b28`); the gaps are (1) manylinux portability proof, (2) the **broken `[cuda]` extra** (real bug), (3) no container, (4) no CUDA-13 build floor. CpuBackend is **not** in scope here — it is test-only.

### C — task table

| ID | Title | What | Effort | Deps |
|---|---|---|---|---|
| **C1** | Fix the broken `[cuda]` extra (real bug) | pyproject.toml:61-66 pins `nvidia-*-cu13>=13,<14` — **unsatisfiable** (0.0.1 deprecation stubs). Repoint to the unsuffixed names: `nvidia-cuda-runtime>=13,<14`, `nvidia-cublas>=13,<14`, and **`nvidia-cusolver`/`nvidia-cufft` pinned near `~=12.x` to match the `.12` SONAMEs** (or `cuda-toolkit[cublas,cudart,cusolver,cufft]==13.*`). Verify each against the built `_core.so` `ldd`. Update the stale header comment (lines 12-17). | S | — |
| **C2** | CUDA-13 `find_package` floor | CMakeLists.txt:60 `find_package(CUDAToolkit REQUIRED)` → `find_package(CUDAToolkit 13.0 REQUIRED)` so a CUDA 12/11 box fails fast at configure, not deep in nvcc. (RELEASE-SCOPE #5.) | S | — |
| **C3** | `[tool.cibuildwheel]` manylinux CUDA-13 build | Add: `build = "manylinux_2_28"`, `CIBW_MANYLINUX_X86_64_IMAGE` = a CUDA-13 devel image (`nvidia/cuda:13.0.x-devel-rockylinux8`), `CIBW_REPAIR_WHEEL_COMMAND = 'auditwheel repair -w {dest_dir} {wheel} --exclude libcudart.so.13 --exclude libcublas.so.13 --exclude libcusolver.so.12 --exclude libcufft.so.12'`. Confirm SONAMEs with `readelf -d` before locking. | M | C1, C2 |
| **C4** | sm_120-real + PTX-virtual + abi3 single wheel | Change `STEPPE_CUDA_ARCH=120` → `CMAKE_CUDA_ARCHITECTURES="120-real;120-virtual"` (native cubin + forward-compat PTX). Enable nanobind `Py_LIMITED_API`/abi3 → collapse cp39..cp313 to one `cp39-abi3` wheel. | M | C3 |
| **C5** | Import-time CUDA-presence check | On import (or first GPU call), if `libcudart.so.13` is unresolvable, raise a friendly error pointing to `pip install steppe[cuda]` / the `.sif` / `LD_LIBRARY_PATH=/usr/local/cuda/lib64` — not a raw loader failure. | S | C1 |
| **C6** | One OCI Containerfile (the single source of truth) | Multi-stage: **build** FROM `nvidia/cuda:13.0.x-devel-ubuntu24.04` → `pip wheel .` + auditwheel repair (the C3 excludes); **runtime** FROM `nvidia/cuda:13.0.x-runtime-ubuntu24.04` (~2-3GB vs ~8GB devel) → pip-install the repaired wheel + build the CLI (`STEPPE_BUILD_CLI=ON`); `ENTRYPOINT ["steppe"]`. The runtime base carries the four excluded SONAMEs in-image → no system CUDA needed by the end user, only a host driver ≥580. Document Podman CDI (`nvidia-ctk cdi generate`; `--device nvidia.com/gpu=all`) + Docker `--gpus all`. | M | C3 |
| **C7** | Apptainer `steppe.def` + `.sif` (PRIMARY) | Thin `steppe.def` (`Bootstrap: docker`, `From: ghcr.io/<org>/steppe:0.1.0`). Document `apptainer pull`/`build` + the canonical run `apptainer exec --nv -B $PWD steppe.sif steppe qpadm --prefix ...` (bind the data dir; AADR stays on host, never in image). Note `--fakeroot`/root needed to *build* → **ship a prebuilt signed `.sif`** as a release asset. `spython` can auto-convert the Containerfile if a hand-written def is undesirable. | L | C6 |
| **C8** | GitHub Actions: build wheel + image (+ Blackwell smoke) | cibuildwheel builds the abi3 wheel; build+push the OCI image to `ghcr.io`; a **self-hosted Blackwell runner** runs the FAST golden ctest + a clean-venv `import steppe` + one real-AADR fit on the repaired wheel. The GPU runner is the hard part (RELEASE-SCOPE #7). Also add a cheap **no-GPU** leg: clang-format/clang-tidy + the cross-layer arch-grep gate + host-only build/tests + ruff/mypy on the facade + `twine check` on sdist/metadata. | L | C3, C6 |
| **C9** | (post-1.0) `pixi.toml` dev env; revisit conda-forge | Commit a `pixi.toml` pinning the CUDA-13 toolchain (`cuda-nvcc`, `cuda-version=13`, cmake, ninja) for reproducible **contributor** builds (a dev convenience, NOT a user lane). Re-evaluate a conda-forge feedstock only once Wheel Variants stabilize **and** a self-hosted GPU validation path exists. **Not a 0.1 lane.** | M | C8 |

---

## 6. WORK-STREAM D — Release process / metadata / license / version / changelog

**Resolved since RELEASE-SCOPE was written (do NOT re-work):** LICENSE exists (MIT, `Copyright (c) 2026 steppe authors`); `steppe.__version__` exists (`bindings/steppe/__init__.py` via `importlib.metadata`, `0.0.0+unknown` fallback); the pyproject version literal is gone, single-sourced from `CMakeLists.txt project(VERSION 0.1.0)` (line 20) via the regex provider (pyproject.toml:93-96); README is a real front door; multi-format readers shipped; `SteppeSanitizers.cmake` now exists.

**Still open (file:line verified):** no CHANGELOG, no ATTRIBUTION/NOTICE, no `.github/` CI, no git tags, no Containerfile/`.def`; pyproject.toml:37 uses the **legacy** `license = { text = "MIT" }` (no `license-files`); `[project]` has no classifiers/urls/keywords/authors; the `[cuda]` extra is broken (→ C1).

### D — task table

| ID | Title | What | Effort | Deps |
|---|---|---|---|---|
| **D1** | PEP 639 license + bundle LICENSE in the artifact | pyproject.toml:37 `license = { text = "MIT" }` → `license = "MIT"` (SPDX) + `license-files = ["LICENSE"]` so the MIT text lands in wheel/sdist `.dist-info` (currently not guaranteed). Drop the redundant MIT classifier if using the SPDX string. | S | — |
| **D2** | `ATTRIBUTION.md` / NOTICE (clean-room note) | Document the MIT-vs-GPL-3 posture: independent reimplementation of the **published** ADMIXTOOLS/ADMIXTOOLS-2 + DATES methods; **no GPL source vendored** (verified — `src/core/stats/dates.cpp` is original; R golden scripts call AT2 as an external oracle); AT2 used only to generate parity goldens. Cite Patterson 2012 / Maier 2023 / Chintalapati 2022. Acknowledge AADR data-use terms for the committed **aggregate** derived goldens (not sample-level). Get a one-line licensing-aware read. | S | — |
| **D3** | Fill `[project]` PyPI metadata | Add classifiers (`Development Status :: 4 - Beta`, `Environment :: GPU :: NVIDIA CUDA :: 13`, `Operating System :: POSIX :: Linux`, `Programming Language :: Python :: 3.9`..`3.13`, `Programming Language :: C++`, `Intended Audience :: Science/Research`, `Topic :: Scientific/Engineering :: Bio-Informatics`); `[project.urls]` (Homepage/Repository/Documentation/Issues/Changelog); `keywords` (qpadm, qpwave, qpgraph, f-statistics, admixtools, population-genetics, ancient-dna, cuda, gpu, dates); **real authors/maintainers** (replace the `steppe authors` placeholder — also in LICENSE); broaden `description` to name qpGraph/DATES/f-stats (currently qpAdm/qpWave-only). | S | — |
| **D4** | `CHANGELOG.md` (Keep-a-Changelog) seeded 0.1.0 | Document the 0.1.0 surface: f2 precompute, qpAdm/qpWave/qpGraph/DATES, standalone f4/f3/f4-ratio/qpDstat/qpfstats, the on-device sweep, the 14-cmd CLI + the Python facade, multi-format readers. Link from `[project.urls]` and the GitHub release notes. | S | — |
| **D5** | Decide + wire CLI delivery | The wheel ships **no `steppe` binary** (`STEPPE_BUILD_CLI=OFF`, no `[project.scripts]`). Decide: ship the C++ CLI via the `.sif` (recommended; carries CLI + Python env in one file) and/or a GitHub-release tarball, **or** add a thin `[project.scripts] steppe=...` argparse shim over the facade. Update README to match (kill the "pip gives me a `steppe` command" false expectation). | M | — |
| **D6** | Release sign-off gate (process, not artifact) | "ctest green" off-box is **vacuous** — 26/34 reference + 21 real-AADR ctests SKIP-clean without GPU/data. Sign-off MUST be a **full ctest on the Blackwell box with `STEPPE_AADR_ROOT` populated AND `STEPPE_THOROUGH=1`**. Consider a `STEPPE_REQUIRE_GPU=1` mode that converts SKIPs to failures on the release runner. | S | C8 |
| **D7** | Tag `v0.1.0` + GitHub release with assets | Assert `git tag` == `CMakeLists project(VERSION)` (add a checklist assertion to prevent tag/metadata skew). Annotated tag `v0.1.0`; cut the release with **the abi3 sm_120 wheel + the prebuilt signed `steppe_0.1.0.sif` + `SHA256SUMS`** + notes (requirements/support matrix, install/run for both lanes). **PyPI deferred** until C-stream portability is signed off. | M | C4, C7, C8, D6 |

---

## 7. Sequenced execution order (across streams)

The streams interleave; this is the critical path. Items inside a phase are parallelizable.

**Phase 0 — Legal + safety gate (½ day, all S).** `D1` (PEP 639 license) · `D2` (ATTRIBUTION) · `A4` (delete strays + close `.gitignore` gaps) · `A5` (delete orphan goldens) · `D3` (PyPI metadata) · `D4` (CHANGELOG seed). *These are independent and cheap; clear them first.*

**Phase 1 — Build correctness + the `[cuda]` fix (½ day).** `C1` (fix the broken extra) · `C2` (CUDA-13 floor) · `C5` (import-time presence check). *Make a from-source build honest before any wheel.*

**Phase 2 — Repo hygiene + sdist scoping (½-1 day).** `A1` (sdist.exclude) → `A2` (CI allowlist gate) · `A6` (relocate dev scripts) · `A3` (private snapshot + remove campaign trail from public main — **pairs with the mandated history rewrite**). *A3 is the irreversible one; snapshot to the private remote first.*

**Phase 3 — Docs front door (1-1.5 days).** `B1` (skeleton) → then in parallel: `B2` (README) · `B3` (VALIDATION — the headline) · `B5` (CLI.md) · `B6` (PYTHON-API) · `B7` (architecture reconcile) · `B8` (CONTRIBUTING) · `B9` (research/perf triage). `B4` (INSTALL) waits on the C-stream SONAME/extra decisions. Close with `B10` (link + stale-claim audit). `D5` (CLI delivery decision) feeds B2/B4.

**Phase 4 — Distribution build-out (the two L items, ~2 days).** `C3` (cibuildwheel) → `C4` (abi3 + PTX wheel) ; `C6` (Containerfile) → `C7` (Apptainer `.sif`). Then `C8` (GitHub Actions: wheel + image + Blackwell smoke + no-GPU leg). Finish `B4` (INSTALL) once SONAMEs are `readelf`-confirmed in the repaired wheel.

**Phase 5 — Sign-off + ship (½ day).** `D6` (full box5090 ctest with AADR + THOROUGH) → `D7` (tag `v0.1.0` + GitHub release with wheel + `.sif` + SHA256SUMS).

**Then ship 0.1.0.** Post-release hardening sprint (not blockers): PyPI once Wheel Variants stabilize, `py.typed`/stubs, the API↔CLI sweep/rotate parity, the AT2-f2 importer, exit-code harmonization, CPM hash pinning, `C9` (pixi + conda-forge revisit), older GENO/EIGENSTRAT reader test coverage.

---

## 8. OPEN DECISIONS (need the user)

1. **Which lanes to actually build for 0.1.0?**
   **Recommendation: PRIMARY = Apptainer `.sif`, SECONDARY = the abi3 sm_120 wheel, both as GitHub-release assets; one OCI Containerfile as the shared source; PyPI + conda DEFERRED.** Confirm you want both #1 and #2 in 0.1.0 (vs `.sif`-only first). The `.sif` is the higher-value artifact for the cluster audience and also delivers the CLI the wheel lacks.

2. **CLI delivery (D5).** Ship the `steppe` binary **only via the `.sif`/tarball**, **or also** add a thin `[project.scripts] steppe=...` argparse shim over the Python facade so `pip install steppe` puts *something* named `steppe` on PATH? *Recommendation: `.sif` carries the real C++ CLI; optionally add the shim later — but at minimum a one-line README clarification that the wheel is the Python API only.*

3. **Archive-vs-delete the internal docs (A3).** Confirm the **archive-branch** disposition for the campaign trail (`kimireview`/`cleanup`/`release_cleanup`/`kimiactions` + `RESUME/TODO/ROADMAP/REORIENT/NEXT-STEPS/RELEASE-SCOPE` + `experiments/`) — i.e. snapshot to a **private** tag/branch (`internal-snapshot-pre-public`), then remove from public `main`. *Recommendation: archive (preserve provenance — `NAMING-STYLE-STANDARD.md` is still cited by commits), hard-delete only the untracked strays (`10codereview/`, `kimiexample.md`, `release_cleanup/extra/`) and the 3 orphan goldens.* **This is irreversible on public history — needs an explicit yes, and it must ride the same pre-public history rewrite that scrubs the vast.ai IPs.**

4. **First version number.** `0.1.0` (already the project version) vs `0.0.x`/`1.0.0`? *Recommendation: `0.1.0` — signals a usable-but-pre-frozen-API research tool; reserve `1.0.0` for a frozen public API + format support.*

5. **PyPI now or post-release?** *Recommendation: **GitHub release first** (tag `v0.1.0` + `.sif` + wheel + `SHA256SUMS`); PyPI as a fast-follow once the manylinux/auditwheel wheel is portability-proven and the `[cuda]` extra resolves. Revisit a PyPI-native story when NVIDIA Wheel Variants leave experimental.* Confirm you accept PyPI deferral.

6. **Author/maintainer identity (D2/D3).** `LICENSE` and the would-be `[project].authors` currently say `steppe authors` — a placeholder. **Provide the real name + email** to put in LICENSE copyright, `[project].authors`, and ATTRIBUTION before anything goes public.

7. **`[cuda]` extra: keep functional or drop (C1)?** Now that the unsuffixed `nvidia-*` wheels exist, the extra **can** be made to work (self-contained `pip install steppe[cuda]`). *Recommendation: fix it (repoint to unsuffixed names) but keep CUDA-as-system-dep the default contract — do NOT inject an `$ORIGIN/../nvidia/cu13/lib` RPATH unless `[cuda]` becomes a hard dep, which it should not.* Confirm.

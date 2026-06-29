# steppe — Documentation & Project Hygiene Assessment

**Scope:** read-only review of the `steppe` repo's documentation, inline comments, root layout, and licensing/attribution. The goal is a first-principles grading with specific file references and a concrete cleanup plan.

**Files read / sampled:**
- `README.md`, `LICENSE`, `.gitignore`, `pyproject.toml`
- `docs/RUN-SHEET.md`, `docs/RUN-GUIDE.md`, `docs/feature-matrix.md`, `docs/NEXT-STEPS.md`, `docs/TODO.md`, `docs/ROADMAP.md`, `docs/RESUME.md`, `docs/RELEASE-SCOPE.md`
- `docs/architecture.md` (sampled; 958 lines / ~100 KB)
- `docs/design/cli-bindings.md`, `docs/research/tgeno-at2-support.md`, `docs/research/interop-usecases.md`, `docs/studies/haak2015.md`
- Source comment samples: `src/core/qpadm/model_search.cpp`, `src/core/qpadm/f4ratio.cpp`, `src/device/cuda/cuda_backend.cu`, `src/app/cli_parse.cpp`, plus comment-density counts across `src/core/qpadm/*.cpp`, `src/device/cuda/*.cu`, `src/app/*.cpp`
- Python facade: `bindings/steppe/__init__.py`

---

## 1. README quality and newcomer onboarding

### What works
- `README.md` is now a real front door. It states the product (GPU/CUDA-13 qpAdm/qpWave reimplementation), the hard requirements (Blackwell `sm_120`, CUDA 13, single-GPU), install command, a 14-subcommand CLI table, a Python quickstart, and links to the key docs.
- The example flow (`extract-f2` → `qpadm`) is copy-pasteable and uses real AADR population names.
- It correctly documents the GPU-only posture and the `CpuBackend` as test-only.

### Gaps
- **No support matrix.** A newcomer cannot tell at a glance which GPUs, Linux distros, Python versions, and input formats are validated. The requirements table buries the `sm_120` prebuilt assumption.
- **TGENO limitation is not stated.** The README says "TGENO, PACKEDANCESTRYMAP/GENO, EIGENSTRAT, PLINK, ANCESTRYMAP" are auto-detected, but `src/io/geno_reader.cpp` still rejects non-TGENO magic (`geno_reader.cpp:81-85`) and the actual decode path is TGENO-only. `docs/RUN-SHEET.md:159-161` and `docs/RESUME.md:65` correctly note this, but the front door oversells it.
- **Install path is aspirational.** `pip install steppe` assumes a published wheel; there is no PyPI release, no `cibuildwheel` config, and the `[cuda]` extra in `pyproject.toml:50-55` points to redistributable wheels that are not yet published with real `>=13` versions.
- **No attribution / authors section.** The README ends with a one-line MIT note. For a tool that reimplements ADMIXTOOLS 2 + DATES methods, a short "Relation to upstream" paragraph belongs here.
- **No `__version__` / changelog pointer.** Newcomers cannot tell which release the repo represents.

**Verdict:** B+ — functional and honest about GPU-only, but oversells input support and install convenience.

---

## 2. Architecture doc maintainability

### What works
- `docs/architecture.md` is opinionated, load-bearing, and traces every major decision (precompute-once/fit-many, RAII, layering, precision policy) back to a one-line rationale. It is meant to be both human spec and agent scaffolding.
- It correctly flags uncertainty with `[UNCERTAIN]` and stale claims with `[STALE — …]`.

### Gaps
- **It is 958 lines and ~100 KB.** At that length it is no longer a quick reference; it is a monolith. New contributors cannot find the current state without reading the whole file.
- **Stale sections persist despite inline markers.** Examples sampled:
  - `architecture.md:18` still says "GPU-optional at the API level" and the CPU backend "always imports," contradicting the project's GPU-only decision (`README.md:11-12`, `bindings/steppe/__init__.py:10-11`).
  - `architecture.md:43-44` lists "PLINK / EIGENSTRAT / PACKEDANCESTRYMAP readers + merge/harmonize front-end" as planned Phase-2 work, but `README.md` claims these formats are already supported.
  - `architecture.md:208` shows `src/app/` as "(planned, P3) CLI", yet `src/app/` is fully built with 14 subcommands.
  - `architecture.md:227-229` says `docs/ROADMAP.md` is the milestone doc, but `ROADMAP.md` itself is also stale (see below).
- **No "valid as of" header or version.** The doc mixes current design, historical rationale, and future intent without a clear freshness date.
- **Duplication with `ROADMAP.md`, `TODO.md`, `RESUME.md`.** The same milestone narrative (M0–M5, M(fit-0..6), F1–F6) is repeated in four documents with different levels of staleness.

**Verdict:** C+ — rich and rigorous, but oversized and out of sync with the tree. It needs to be split into a short current architecture doc plus per-subsystem design notes.

---

## 3. Inline comment quality

### What works
- Comment density is high and generally useful. Sample ratios:
  - `src/core/qpadm/model_search.cpp`: 131 comments / 193 code lines (~0.68)
  - `src/core/qpadm/f4ratio.cpp`: 69 / 75 (~0.92)
  - `src/device/cuda/cuda_backend.cu`: 1771 / 3572 (~0.50)
  - `src/app/cli_parse.cpp`: 181 / 534 (~0.34)
- File headers explain purpose and link to `architecture.md` sections / `ROADMAP.md` milestones.
- Comments carry load-bearing invariants, e.g. `model_search.cpp:24-36` explains why `assemble_f4` and `run_impl` must be adjacent on the same backend.
- The codebase has very few bare `TODO`/`FIXME` markers: only 28 occurrences across 17 files (`Grep` count).

### Gaps
- **Manifesto headers.** `src/device/cuda/cuda_backend.cu` opens with a 57-line header that restates large chunks of `architecture.md` §4/§7/§11.4. It is useful the first time, but every edit to the architecture doc risks making this header stale. The same pattern appears in `src/app/cli_parse.cpp` (14-line header) and many `src/io/*.cpp` files.
- **Ticket-reference noise.** `Grep` found 433 occurrences of milestone/ticket tokens (`M1`, `M2`, `F1`, `ADR-*`, `[7.4]`, etc.) across 87 files. Many are helpful, but others reference completed work (`M1`, `M2`, `M4.5`) and will never be removed unless someone audits them. They create the impression the code is still under heavy milestone construction.
- **Stale TODOs that look like they are still active:**
  - `src/core/qpadm/model_search.cpp:167` has a full `TODO(multigpu-host-bounce)` banner, yet multi-GPU rotation is a known deferred decision documented elsewhere.
  - `src/device/cuda/check.cuh:25` says `TODO(M4.5): exercise this two-tier parity claim in CI`. M4.5 is merged; this should be a tracked issue or removed.
  - `src/device/cuda/handles.hpp:237` references `TODO M4.5 line 98`.
- **Commented-out / historical references in source.** A few files still talk about "M5 tile loop" (`src/io/geno_reader.cpp:379`, `src/io/geno_reader.hpp:61`) and "M1 requires snp_begin == 0". These are correct but read like construction notes rather than final documentation.

**Verdict:** B — comments are informative, but there is too much duplication with external docs and too many milestone tokens that have aged out.

---

## 4. Run sheets and user docs

### What works
- `docs/feature-matrix.md` is exemplary: every shipped feature, real-AADR wall-clock, GPU util, golden-match status, and Python coverage in one table. It also has a FLAGS section for non-blocking findings and a Notes section explaining data provenance.
- `docs/RUN-SHEET.md` is a practical command sheet with real paths and flags. It explains `--blgsize` in Morgans, `--tier`, `--hash`, and `--jackknife`.
- `docs/RUN-GUIDE.md` has been corrected and now accurately describes the built CLI/Python surface, build commands, and the dev loop.

### Gaps
- **RUN-SHEET is missing shipped subcommands.** It covers `extract-f2`, `qpadm`, `qpadm-rotate`, `qpwave`, `f4`, `f3`, `f4-ratio`, `qpdstat`, `f4 --all-quartets`, `f3 --all-triples`, and the Python wheel. It does **not** have sections for `qpgraph`, `qpgraph-search`, `dates`, or `qpfstats` — four of the 14 commands advertised in `README.md`.
- **No user-facing troubleshooting guide.** A user who hits the TGENO-only error, the CUDA-13 loader error, or VRAM OOM has to infer the fix from scattered notes.
- **No generated API docs.** The Python module has good docstrings (`bindings/steppe/__init__.py`), but there is no Sphinx / mkdocs site. The C++ API is only documented inside `RUN-GUIDE.md`.

**Verdict:** B+ — strong measured matrix and command sheet, but the command sheet is incomplete and there is no consolidated user guide.

---

## 5. Research docs vs engineering docs separation

### What works
- The top-level `docs/` layout has a clear intent:
  - `docs/design/` — contracts and build-order (`cli-bindings.md`, `fit-engine.md`, etc.)
  - `docs/research/` — investigations and comparisons (`tgeno-at2-support.md`, `interop-usecases.md`, etc.)
  - `docs/studies/` — reproductions of published studies (`haak2015.md`, `olalde2018.md`)
  - `docs/perf/` — measured scaling results
- The research docs are well-written and cite sources. `docs/research/tgeno-at2-support.md` is a model forensic write-up.

### Gaps
- **`docs/cleanup/` has 134 files (3.6 MB) and `docs/release_cleanup/` has 161 files (780 KB).** These are internal audit/refactor artifacts, not design or user docs. They drown the actually useful documentation.
- **`docs/kimireview/` has 102 files (852 KB).** This directory is itself a meta-review dump. It is fine as a working area for an AI review workflow, but it should not live in the main `docs/` tree that users and contributors browse.
- **No `docs/README.md` or index.** A new reader landing in `docs/` sees `cleanup/`, `kimireview/`, `release_cleanup/`, and a scattering of top-level `.md` files with no map.

**Verdict:** B- — the taxonomy is sound, but the docs tree is cluttered with internal audit material.

---

## 6. Consistency between docs and code

### What works
- `README.md`, `feature-matrix.md`, and the actual CLI/Python surface are largely aligned on the 14 subcommands and the GPU-only posture.
- `pyproject.toml:24-27` matches `bindings/steppe/__init__.py:21-27` on version (`0.1.0`) and the lazy-pandas design.

### Gaps
- **`docs/NEXT-STEPS.md` is stale.** It lists `M(cli-2) qpwave CLI`, `M(cli-3) qpadm-rotate CLI`, `M(py-1) Python bindings`, and standalone f-stats as not built, but all of these ship in `src/app/` and `bindings/steppe/__init__.py`.
- **`docs/TODO.md`, `docs/ROADMAP.md`, and `docs/RESUME.md` contain stale status lines.** They repeatedly assert "NO CLI / NO bindings / NO standalone f-stats" and "topology search DEFERRED" in passages that have not been refreshed.
- **`docs/RESUME.md:3-8` points to `agentscripts/README.md` as "THE WORKFLOW MAP."** That file is inside `agentscripts/`, which is gitignored and not part of the source tree a clone would receive.
- **`docs/RELEASE-SCOPE.md` is from an older HEAD.** It says there is no `LICENSE` file, but `LICENSE` exists. It also says `README` is a 19-line stub, which has been fixed. The doc needs a re-audit or archival.
- **Architecture/ROADMAP stale claims** noted in §2 contradict the built code.
- **Source comments still reference completed milestones** as if the code were mid-construction.

**Verdict:** C+ — the primary user-facing docs are consistent, but the project-status docs have drifted badly and point to gitignored files.

---

## 7. Project root cleanliness

### What works
- The root has the expected CMake, Python, and license files. `.gitignore` sensibly excludes build trees, Python artifacts, large genotype data, and local agent/handoff files.

### Gaps
- **`aadr/` directory in the repo root contains real genotype data** (`.geno`, `.ind`, `.snp`, `.anno`, and two `.zip` files). While the large binary files are ignored by `.gitignore:54-57`, the directory itself is present and the `.zip` files are also ignored. This is a data directory inside the source tree; it should live outside the repo or be fully ignored at the directory level (`/aadr/`).
- **`agentscripts/` is gitignored but still present in the working tree.** It contains 18 workflow scripts and is referenced by `docs/RESUME.md`. Because it is ignored, a fresh clone will not contain the "workflow map" the project tells people to read first.
- **Hidden dev-tool directories are not ignored:** `.agents/`, `.claude/`, `.codex/` exist at root and are not in `.gitignore`. `.claude/settings.local.json` is a local IDE config that should not be tracked.
- **`experiments/` is a spike directory.** It is appropriate as a throwaway archive, but it should be documented in `README.md` or `docs/` so newcomers know it is not production code.
- **No `CHANGELOG.md`, `NOTICE`, or `ATTRIBUTION.md`.**

**Verdict:** C+ — the root looks tidy, but the presence of ignored data/workflow dirs and untracked tool configs creates a trap for accidental commits and confused newcomers.

---

## 8. License and attribution

### What works
- `LICENSE` is a standard MIT license with copyright line "Copyright (c) 2026 steppe authors."
- `pyproject.toml:32` declares `license = { text = "MIT" }`.
- `README.md:124-128` includes a clean-room note: independent reimplementation, ADMIXTOOLS 2 used only as a test oracle, no GPL source vendored.

### Gaps
- **No `NOTICE` or `ATTRIBUTION.md`.** For a project that reimplements ADMIXTOOLS / ADMIXTOOLS 2 + DATES methods, a standalone attribution file with paper citations and the clean-room statement is stronger than a paragraph in the README.
- **No `license-files` / SPDX metadata in `pyproject.toml`.** PEP 639 `license = "MIT"` would be clearer than `license = { text = "MIT" }`.
- **No per-file copyright headers.** MIT does not require them, but they reduce ambiguity for downstream packagers.
- **No author list.** "steppe authors" is vague.

**Verdict:** B — legally adequate, but attribution could be more explicit and discoverable.

---

## 9. What would make documentation and project hygiene A+

1. **A single, versioned docs status page.** Add `docs/README.md` that maps reader intent (user, contributor, release engineer) to the right doc and states the last-audited commit/date.
2. **Refresh and split `architecture.md`.** Keep a short current architecture doc (~150 lines). Move detailed subsystem specs to `docs/design/<subsystem>.md`. Move historical rationale to `docs/internal/history/` or archive it.
3. **Archive internal audit directories.** Move `docs/cleanup/`, `docs/release_cleanup/`, and `docs/kimireview/` out of the browsable `docs/` tree (e.g., a separate `internal-docs` branch, or `.agents/audit-artifacts/` that is gitignored).
4. **Update `docs/NEXT-STEPS.md`, `docs/TODO.md`, `docs/ROADMAP.md`, `docs/RESUME.md`, and `docs/RELEASE-SCOPE.md`.** Either refresh them to the current HEAD or delete them and replace with the single status page.
5. **Complete `docs/RUN-SHEET.md`.** Add sections for `qpgraph`, `qpgraph-search`, `dates`, and `qpfstats` with real-AADR invocations.
6. **Add a support matrix to `README.md`.** Explicitly state: validated GPU (`sm_120` prebuilt), CUDA 13 + driver requirement, Linux x86_64, Python 3.9+, TGENO-only decode (other formats detected but not yet decoded), single-GPU rotation recommendation.
7. **Scrub code comments.** Remove completed milestone tokens (`M1`, `M2`, `M4.5`) and convert deferred work (`TODO(multigpu-host-bounce)`, M4.5 CI claims) into GitHub issues. Trim manifesto headers that duplicate `architecture.md`.
8. **Clean the root.** Add `/aadr/`, `.agents/`, `.claude/`, `.codex/` to `.gitignore` (or remove the dirs). Relocate `agentscripts/` and `handoff-*.md` out of the repo. Document `experiments/` as spikes.
9. **Add `CHANGELOG.md`, `NOTICE`/`ATTRIBUTION.md`, and a version single-source.** Use `importlib.metadata` as the primary version path; add a `version.hpp` if C++ needs it.
10. **Set up minimal CI + doc generation.** A no-GPU CI job that builds docs and runs `clang-format`/`clang-tidy` and an architecture-grep gate would catch doc-code drift. Generated Sphinx docs from `bindings/steppe/__init__.py` docstrings would expose the Python API.

---

## Concrete cleanup plan

| Priority | Task | Files / commands | Acceptance criteria |
|----------|------|------------------|---------------------|
| **P0 — blocker** | Refresh project-status docs to current HEAD | `docs/NEXT-STEPS.md`, `docs/TODO.md`, `docs/ROADMAP.md`, `docs/RESUME.md`, `docs/RELEASE-SCOPE.md` | No doc asserts "no CLI / no bindings / no standalone f-stats"; `RESUME.md` no longer points to a gitignored file |
| **P0 — blocker** | Complete the command sheet | `docs/RUN-SHEET.md` | Sections for all 14 CLI subcommands, including `qpgraph`, `qpgraph-search`, `dates`, `qpfstats` |
| **P0 — blocker** | Clean the root and `.gitignore` | `.gitignore`, root dirs | `/aadr/`, `.agents/`, `.claude/`, `.codex/` ignored or removed; `agentscripts/` relocated; `experiments/` documented |
| **P1** | Add attribution and release metadata | new `ATTRIBUTION.md` / `NOTICE`, `README.md`, `pyproject.toml` | Citations for ADMIXTOOLS/ADMIXTOOLS 2/DATES; PEP 639 license metadata; `[project.urls]` block |
| **P1** | Split/refresh architecture doc | `docs/architecture.md`, `docs/design/*.md` | Current architecture ≤200 lines; stale `[STALE]` markers removed or moved to `docs/internal/history/` |
| **P1** | Archive internal audit docs | `docs/cleanup/`, `docs/release_cleanup/`, `docs/kimireview/` | Moved out of browsable `docs/` or summarized by a single index |
| **P2** | Scrub milestone tokens and stale TODOs | `src/**/*.cpp`, `src/**/*.cu`, `src/**/*.hpp` | Completed milestone tokens removed; deferred `TODO`s migrated to issue tracker; comment density stays ≥0.25 for complex files |
| **P2** | Add `CHANGELOG.md` and single-source version | `CHANGELOG.md`, `pyproject.toml`, `bindings/steppe/__init__.py`, optional `include/steppe/version.hpp` | `steppe.__version__` works; version literal exists in only one place |
| **P2** | Improve `README.md` support matrix | `README.md` | GPU/driver/OS/Python matrix; honest "TGENO-only decode" line; install caveat about PyPI/wheel status |
| **P3** | Generate API docs | `docs/api/` or ReadTheDocs/MkDocs | Sphinx/mkdocs site from Python docstrings; C++ API extracted from Doxygen or `RUN-GUIDE.md` |
| **P3** | Add no-GPU CI gate | `.github/workflows/` | format/lint/arch-grep pass on every PR; doc-freshness check (e.g., `RUN-SHEET.md` subcommand count matches `steppe --help`) |

---

## Overall grade

- **User-facing docs:** B+ (good README, great feature matrix, incomplete run sheet)
- **Engineering docs:** C+ (architecture monolith is stale, status docs drifted)
- **Inline comments:** B (informative but noisy with milestone tokens and manifestos)
- **Project hygiene:** C+ (root has ignored data/workflow dirs, hidden tool configs untracked, no CHANGELOG/NOTICE)
- **License/attribution:** B (MIT present, attribution too thin)

**Composite: B-** — the docs are substantially better than a typical research-code repo, but the accumulation of stale project-status docs, internal audit material, and root-level clutter is now the main barrier to a release-quality presentation.

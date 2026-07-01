# steppe — Low / Polish Action Plan

> Scope: the genuinely-open **Low / polish** bucket from `docs/kimireview/ASSESSMENT.md` §3
> (docs 06/08), re-verified against the tree at `HEAD` on branch `phase2-fit-engine`.
> Items the assessment **rejected** (§4: 2nd stream, NCCL, multi-GPU gate, excise-oracle,
> `steppe::Index`, JsonWriter reshape, capability pure-virtual, kernel fuzzing,
> mock-backend) are **out of scope** and not planned here. The HIGH/MED/LOW *source-hygiene*
> campaign (`a2f9d64..HEAD`) is already merged; nothing below re-treads it.

## Framing — what "Low/Polish" means against a parity product

steppe is a **single-GPU, parity-gated** product: §12 freezes the AT2/cuBLAS/cuSOLVER math
names, the deterministic fixed-order reductions, and the single statistic stream; emulated
FP64 (Ozaki ~40-bit) is the default with a native-FP64 carve-out; the CpuBackend is the
dev/test parity oracle only. **Acceptance = the committed goldens (`golden_fit0`,
`golden_fit1_NRBIG`, `golden_*rot`) still pass.**

Every item in this document was chosen because it is **behavior-neutral by construction**:
none touch the statistic stream, the cuBLAS/cuSOLVER math, or any reduction order. The
exhaustive grep below confirms there is exactly **one golden-adjacent surface in the whole
bucket** — `regenerate_goldens.sh`, and only when a human deliberately runs it — and it is
guarded by a hard version-preflight plus a mandatory parity re-run. Everything else is XS–S
developer-experience, tooling, profiling-unlock, and robustness polish.

**Build reality that scopes every "Verify" line.** Nothing builds locally (RTX 2070 /
CUDA 11.8, no FP64-emu). CUDA TUs and the bindings build **only on `box5090`** (CUDA 13.1,
single RTX 5090, `ssh box5090`); real AADR `v66.p1_HO` lives there. The committed goldens
ship in-repo. **Two exceptions iterate locally** with no compiler/CUDA: Doxygen (parses
headers as text) and the pure-host C++ TUs (`f2_dir_io`, `config_builder`). Effort ratings
are honest: items tagged **S** are S for *breadth* (74 test sites, a 1010-line facade, 8 R
scripts), not depth.

**Cluster order below is by leverage, not by the assessment's doc order:** build/CI enablers
first (they de-risk the separate §3-HIGH host-only CI lane), then the profiling-unlock +
robustness fixes, then the portfolio DX surface.

**Cross-cut additions (this revision).** The new **Cluster D** item **D1** and the items
**A6–A8 / B6–B9 / C6–C9** below were folded in from our own consolidated cross-cutting review
(`docs/release_cleanup/crosscutting/X1–X7`) via the reconciliation
`docs/kimiactions/04-crosscut-vs-kimi.md` §3/§5 — gaps the X-review caught (several also caught
by Kimi) that the `a2f9d64..HEAD` source-hygiene campaign **neither fixed nor planned**. Every
one is **comment / doc / structure / dead-code and golden-neutral by construction** (the §12
PARITY LAW is untouched); each carries a `(cross-cut gap GN, from docs/kimiactions/04)`
provenance tag. **Cluster D (comment hygiene for an external reader) leads the sequence** — it
is the single highest first-impression signal at near-zero effort, and its regression
**grep-gate lives in `02-ci-plan.md`** (next to the arch-grep A6), because the planned
clang-format pass runs comment-reflow **OFF** and would otherwise *entrench* these tokens rather
than strip them.

---

# Cluster D — Comment hygiene for an external reader (cross-cut; DO FIRST)

> **Leads the sequence despite the letter.** One deliberate comment-only pass over shipping +
> public source, public-surface first. Highest senior-first-impression signal in the whole
> document at the lowest effort; golden-neutral by construction. The regression gate that keeps
> it scrubbed lives in `02-ci-plan.md` (grep-gate sibling of the arch-grep A6).

## D1 — Purge ticket / milestone / "spike" archaeology, public-surface first  ·  **P1 · Effort S (breadth)**

> *(cross-cut gap **G1** — folds **G14** + **G15** — from `docs/kimiactions/04-crosscut-vs-kimi.md`
> §3/§5; X7-F1/F2/F4/F5, X2-#10, X2-#11–14, X7-F8/F9.)*

**What's needed.** Shipping + **public** source carries internal provenance tokens that read as a
private engineering journal. Verified at HEAD: **119** `cleanup [A-Z]-?[0-9]` hits across **38**
source files (the broader ticket-token total ≈252 once milestone codes are counted). The tags
reach the **public seam** `src/device/backend.hpp:56,74,581` (e.g. `:56`
`/// DIAGONAL CONVENTION (pinned; cleanup X-2/B4): …`) and the **public include**
`include/steppe/config.hpp` at **8 sites** (lines 7,10,33,66,91,147,249,312, carrying
ROADMAP/spike/milestone and `cleanup X-8`/`cleanup X-6` tokens). Bare milestone codes are dense
in source: **M4×83, M4.5×63, M5×56** (word-bounded `grep -P`, source-only — the review's
155/69/59 also counted docs). "the spike" archaeology persists in
`src/device/cuda/device_buffer.cuh:9-10`, `stream.hpp:5-6,108`, `handles.hpp:5`, and
`config.hpp:10,33,66`. **Bundle two siblings into the same pass:** **G14** stale self-changelog
one-liners — `model_search.cpp:173-174` ("The doc once said … WRONG"),
`qpgraph_fit_kernels.cu:254` ("unscaled q for now" though scaling completes at `:264`),
`log.hpp:7,15,24,25` ("eventually a spdlog backend … swap later"), `check.cuh:25,227,290`
(`TODO(M4.5)` ×3) — and **G15** ALL-CAPS / charged-vocabulary density (verified source-only:
`NEVER×107`, `MUST×27`, `LAW×8`, `REJECTED×11`).

**Optimal end state.** **Public surface first** — `include/steppe/config.hpp` and
`src/device/backend.hpp` read self-contained with **zero** ROADMAP / milestone / ticket
citations; the comment's *rationale* is kept, only the provenance tag is stripped (e.g.
`/// (pinned; cleanup X-2/B4): f2(i,i) carries the FULL …` → `/// f2(i,i) carries the FULL …`).
Then the private headers/TUs. Milestone codes survive **only** where the sentence is genuinely
explaining that design phase, never as a bare "added in Mx" timestamp. ALL-CAPS is reserved for
true parity/precision invariants (§12). The self-changelog lines are reworded to describe the
code **as it is**, not the history of edits. Campaign Group-8 (`30df7c5`) removed **zero** ticket
tags, so this is net-new work, not a re-tread.

**Steps.**
1. **Public surface FIRST.** Scrub `include/steppe/config.hpp` (the 8 sites) and
   `src/device/backend.hpp:56,74,581`: strip the `cleanup X-N/B-N`, `ROADMAP §…`, milestone, and
   `spike` tokens while preserving every sentence of rationale.
2. Sweep the remaining ~36 source files for `cleanup [A-Z]-?[0-9]` / `group-[0-9]` and delete the
   tag (keep the surrounding explanation).
3. Reword bare milestone codes: keep `M4`/`M4.5`/`M5` only where the sentence explains that design
   phase; drop them where they are a bare "added in Mx" timestamp.
4. **G14**: reword the self-changelog lines (`model_search.cpp:173-174`,
   `qpgraph_fit_kernels.cu:254`, `log.hpp:7,15,24,25`, `check.cuh:25,227,290`) to describe present
   behavior, not a history of edits. (The `qpgraph_fit_kernels.cu:254` "unscaled q for now" is
   simply describing an intermediate that `:264` then scales — reword to say so.)
5. **G15**: down-case the charged vocabulary except on genuine parity/precision invariants (§12).
6. Do this as a **deliberate comment-only edit** — the `02-ci-plan.md` clang-format pass runs
   comment-reflow **OFF** and will *entrench* these tokens, so the scrub must be its own step, not
   a formatter side effect.

**Parity-risk.** **None — comment-only, golden-neutral by construction.** No statement, no math
name, no reduction order, no literal value changes; nothing in the §12 statistic stream is
touched, every golden is byte-identical. The only failure mode is deleting a token that was
load-bearing context (e.g. a constant whose audit trail genuinely needs the "promoted from the
spike's bare `1e-12`" note) — guarded by **keeping the rationale and stripping only the tag**,
never the sentence.

**Verify.** `grep -rIE 'cleanup [A-Z]-?[0-9]|group-[0-9]' include/steppe/config.hpp src/device/backend.hpp`
→ **0**; whole-tree `cleanup`-token count drops from 119 toward 0; the public include + seam read
clean to an external reviewer. The `02-ci-plan.md` grep-gate (the A6 sibling) then locks it
against regression. No build/test behavior change (comment-only; compiles on every lane — including
local).

**Files.** `include/steppe/config.hpp`, `src/device/backend.hpp` (public, first); then the ~36
remaining `cleanup`-tagged TUs across `src/device` + `src/core`; specifically
`src/core/qpadm/model_search.cpp`, `src/device/cuda/qpgraph_fit_kernels.cu`,
`src/core/internal/log.hpp`, `src/device/cuda/check.cuh`,
`src/device/cuda/{device_buffer.cuh,stream.hpp,handles.hpp}`.

---

# Cluster A — Tooling / build polish (CI + iteration enablers)

> The highest-leverage cluster: CTest labels are the literal enabler for the §3-HIGH
> CUDA-free CI lane (`ctest -LE gpu`), and `STEPPE_WERROR`/`ccache`/pytest+ruff+mypy are
> the knobs that future lane will invoke. Doing this cluster first de-risks the CI item.

## A1 — `STEPPE_WERROR=OFF` escape hatch + ccache auto-detection  ·  **P1 · Effort S**

**What's needed.** Warnings-as-errors is hardcoded in `cmake/SteppeWarnings.cmake` on the
`steppe::warnings` INTERFACE target: host `-Wall;-Wextra;-Werror` (line 18), device
`--Werror;all-warnings` (line 24), host-forward `-Xcompiler=-Wall,-Wextra,-Werror`
(line 27). A dev who hits one warning mid-experiment must edit the policy module. Separately
there is **no** compiler-launcher logic anywhere (`grep -n COMPILER_LAUNCHER\|ccache cmake/
CMakeLists.txt` = empty), so heavy `.cu` recompiles get zero caching across the ephemeral-box
rebuilds. Both are option-vocabulary additions; `cmake/SteppeOptions.cmake` is the documented
single home for `option()`/cache vars.

**Optimal end state.** `option(STEPPE_WERROR "..." ON)` gates the three `-Werror` tokens —
default **ON preserves the gate**; `-DSTEPPE_WERROR=OFF` keeps `-Wall -Wextra` but lets a
local build proceed through warnings. `option(STEPPE_CCACHE "..." ON)` + `find_program(ccache)`
sets `CMAKE_CXX_COMPILER_LAUNCHER` / `CMAKE_CUDA_COMPILER_LAUNCHER` when present, a no-op when
absent. CI/release presets pin `STEPPE_WERROR ON` explicitly so a relaxed local build can
never ship.

**Steps.**
1. `cmake/SteppeOptions.cmake`: add `option(STEPPE_WERROR "Treat warnings as errors (default ON; OFF for local iteration)" ON)` and `option(STEPPE_CCACHE "Use ccache as compiler launcher when found" ON)`, with the §6/§8 home-comment style the file uses.
2. `cmake/SteppeWarnings.cmake`: split the `-Werror` tokens out of the always-on `-Wall;-Wextra`. When `STEPPE_WERROR` is OFF: host gets `-Wall;-Wextra`, device drops `--Werror;all-warnings`, host-forward becomes `-Xcompiler=-Wall,-Wextra`. Keep the `COMPILE_LANGUAGE` generator-expression split (nvcc rejects raw host flags) and the `-lineinfo` `RelWithDebInfo` branch untouched.
3. `CMakeLists.txt`, after `include(SteppeOptions)` and before the first `add_subdirectory`: `if(STEPPE_CCACHE) find_program(CCACHE_PROGRAM ccache) if(CCACHE_PROGRAM) set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}") set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}") message(STATUS "steppe: ccache enabled (${CCACHE_PROGRAM})") endif() endif()`.
4. `CMakePresets.json`: set `"STEPPE_WERROR": "ON"` explicitly in the `ci` preset cache vars so the gate is visible in review.
5. Note in the `SteppeWarnings.cmake` header that WERROR is now an option but **ON is the contract** (the M0 "C++20 with warnings-as-errors" invariant) — OFF is a dev-only escape hatch.

**Parity-risk.** **None.** Neither flag is on the compute/precision path. WERROR-off changes
only whether a warning halts the build, never codegen. ccache as a launcher yields byte-identical
objects (it hashes the full command + preprocessed source; nvcc is supported). Guard: default
`STEPPE_WERROR=ON` + CI pins ON.

**Verify.** `cmake --preset ci && grep STEPPE_WERROR build/ci/CMakeCache.txt` → `ON`; clean
build byte-for-byte as before. `cmake -DSTEPPE_WERROR=OFF …` + a deliberate unused var → build
**warns but succeeds**. With ccache installed, second `cmake --build` after `ccache -C` shows
hits (`ccache -s`); absent ccache, configure prints no ccache line and builds normally.

**Files.** `cmake/SteppeOptions.cmake`, `cmake/SteppeWarnings.cmake`, `CMakeLists.txt`,
`CMakePresets.json`.

---

## A2 — CTest `LABELS` (unit / reference / cli / python + gpu/slow) across the 74-test tree  ·  **P1 · Effort S**

**What's needed.** `tests/CMakeLists.txt` registers **74** tests (`grep -c 'add_test(NAME'` = 74)
with **zero** `LABELS` (`grep -c LABELS` = 0 — confirmed). So `ctest -L unit` / `-L reference`
/ `-L python` / `-L slow` all select nothing, and there is no way to run the host-only subset
without a GPU. Tests already split cleanly by source dir: `unit/` (host-only GTest/self-check,
no device), `reference/` (`.cu`/`.cpp` golden+equivalence parity, need GPU), `cli/` (end-to-end
that exec the built `steppe` binary, need GPU), plus the single `py_qpadm` pytest gate
(`tests/CMakeLists.txt:1681`). The reference/cli tests already SKIP cleanly (exit 0) with no
CUDA device, but that is invisible to ctest filtering.

**Optimal end state.** Every test carries a primary lane label (`unit|reference|cli|python`)
plus an orthogonal `gpu` label on device-touching tests and `slow` on long runners. `ctest -L unit`
runs the pure-host lane; **`ctest -LE gpu` is exactly the CUDA-free CI lane** the §3-HIGH item
wants; `ctest -L slow` / `-LE slow` partitions fast-vs-slow. Labels live in **one place** (a thin
wrapper or 4 grouped lists) so a new test inherits a lane by construction.

**Steps.**
1. Taxonomy: primary `{unit,reference,cli,python}`; orthogonal `{gpu,slow}`. Labels are
   multi-valued → a reference test is `reference;gpu`, `qpadm_rotation` is `reference;gpu;slow`.
2. Add a CMake helper `steppe_add_test(NAME … COMMAND … LABELS "l;l")` near the top of
   `tests/CMakeLists.txt` that calls `add_test` then `set_tests_properties(… PROPERTIES LABELS …)`.
   Lower-churn alternative (the ~4-edit XS path): keep existing `add_test` calls and append four
   `set_tests_properties(<name-list> PROPERTIES LABELS …)` blocks at file end, built from name
   lists. **Prefer the wrapper** so future tests are tagged by construction.
3. Tag the `unit/` tests `unit` (no gpu) — **audit each first**: a few `unit/` TUs (e.g.
   `f2_blocks_multigpu_unit`, `f2_combine_unit`, `validate_device_order_unit`) may link
   `steppe::device`/touch a backend; if so they get `gpu`. Confirm by "does the `.cpp` link
   device + run a kernel", not by the dir name.
4. Tag `reference/` `reference;gpu`, `cli/` `cli;gpu`, `py_qpadm` `python;gpu`.
5. Add `slow` to the long runners (`qpadm_rotation`, `fstat_sweep_parity`, the `*_NRBIG`
   sweeps, `f2_multigpu_*`, `cli_extract_qpadm`, `cli_dates`) — confirm by **runtime, not name**.
6. (Paired nicety, not required) `set_tests_properties(<gpu tests> PROPERTIES RESOURCE_LOCK gpu)`
   so a future `ctest -j` cannot oversubscribe the single 5090.
7. Document the taxonomy in a 6-line header block and add a `host` test preset to
   `CMakePresets.json` with `"filter": {"exclude": {"label": "gpu"}}`.

**Parity-risk.** **None** — labels are pure CTest metadata; they change neither assertions nor
golden values. Only failure mode: mislabeling a *device* test as non-gpu (then `-LE gpu` runs it
without a device) — guarded by the per-test host-only audit in step 3.

**Verify.** After configure+build on box5090: `ctest -L unit -N` lists only host tests;
`ctest -LE gpu -N` == the unit set; `ctest -L reference -N` == reference count; `ctest -L python -N`
== `py_qpadm`; total `ctest -N` still **74**. On a CUDA-free host, `ctest -LE gpu` runs green —
the actual CI-lane smoke.

**Files.** `tests/CMakeLists.txt`, `CMakePresets.json`.

---

## A3 — pytest config + ruff + mypy for the pure-Python facade  ·  **P2 · Effort S**

**What's needed.** The 1010-line facade `bindings/steppe/__init__.py` and the
`tests/python/test_py_*.py` + `conftest.py` have **no** lint/type config
(`grep -n 'ruff\|mypy\|tool.pytest' pyproject.toml` = empty; no `ruff.toml`/`mypy.ini`/
`pytest.ini`/`setup.cfg`). The pytest gate is invoked bare (`python -m pytest -v tests/python`,
`tests/CMakeLists.txt:1682`). `pyproject.toml` already has a `test` extra (pytest>=7, pandas,
numpy) but no lint tools. The facade imports the compiled `._core` (`bindings/module.cpp`),
which has no type stub — mypy must excuse that one import.

**Optimal end state.** `pyproject.toml` carries `[tool.pytest.ini_options]` (testpaths, a `gpu`
marker so SKIPs are explicit, a sane timeout), `[tool.ruff]` (house line length, target py39, a
**curated** rule set), and `[tool.mypy]` (py39 + a per-module override for `steppe._core`).
`ruff check` and `mypy bindings/steppe` pass clean and are runnable in the future host-only CI
lane. A `lint` extra provisions the tools.

**Steps.**
1. `[tool.pytest.ini_options]`: `testpaths=["tests/python"]`, `markers=["gpu: requires a CUDA device"]`, `addopts="-ra"`, optional `timeout` (needs `pytest-timeout` in the extra). Keep consistent with the bare invocation so `ctest -R py_qpadm` is unaffected.
2. `[tool.ruff]`/`[tool.ruff.lint]`: `target-version="py39"`, `line-length` **measured from the longest existing line first** to avoid a churn storm, `select=[E,F,I,UP,B]`, `per-file-ignores` for the test dir (asserts/fixtures). Respect the existing `# noqa: WPS433` in `conftest.py`.
3. `[tool.mypy]`: `python_version="3.9"`, `warn_unused_ignores`, `[[tool.mypy.overrides]] module="steppe._core"` `ignore_missing_imports=true`. Keep `disallow_untyped_defs` **OFF** initially (the facade is partially annotated) so this stays XS, not an annotation project.
4. Add a `lint = ["ruff>=0.4", "mypy>=1.8"]` optional-dependency group; optionally fold `pytest-timeout` into `test`.
5. Run `ruff check --fix` once over `bindings/steppe`+`tests/python` (auto-fix import-order/format), hand-resolve real type gaps, **bump line-length / add per-file-ignores rather than mass-rewriting** the 1010-line facade.
6. (Optional) register a `ctest` lint test under `STEPPE_BUILD_PYTHON` gated on `find_program(ruff)` so it joins the labeled `python` lane and SKIPs when tools are absent.

**Parity-risk.** **None** — no compiled code, no golden. Single risk: an over-strict ruleset
forcing churn in the **frozen-shape facade** (result structs deliberately mirror AT2
`$weights/$rankdrop/$popdrop`). Guard: start permissive, never auto-reformat the golden-matched
output-shaping code.

**Verify.** `ruff check bindings/steppe tests/python` → 0; `mypy bindings/steppe` → 0;
`pytest tests/python -m gpu --co` shows markers resolve with no `PytestUnknownMarkWarning`;
`ctest -R py_qpadm` still runs/SKIPs identically (config is additive).

**Files.** `pyproject.toml`, `bindings/steppe/__init__.py`, `tests/python/conftest.py`,
`tests/CMakeLists.txt`.

---

## A4 — f2-dir `meta_schema_version` + `pops.txt` checksum in `meta.json`  ·  **P2 · Effort S**

**What's needed.** The writer `src/app/f2_dir_writer.cpp` hand-emits `meta.json` (≈lines 472–513)
with `format`, `steppe_version`, the filter block, source shas, and `f2_cache_id` = sha256 of
`f2.bin` (computed via `sha256_file`, ≈line 445). It does **not** stamp a schema version for
`meta.json` itself, and it does **not** hash `pops.txt` (≈lines 466–469 just write the labels).
A corrupted/swapped `pops.txt` silently reassigns the name↔index map and changes every result —
undetectably. Critically, `read_f2_dir` (`src/app/f2_dir_io.cpp`) does **not** read `meta.json`
at all (only `f2.bin` + `pops.txt`), and the CUDA-free app layer has **no JSON parser**. The
binary format already has its own version (`kF2DiskVersion=1` in
`src/device/f2_disk_format.hpp`) which is **separate** and must not be conflated with the sidecar
schema version.

**Optimal end state.** `meta.json` carries `"meta_schema_version": 1` (sidecar schema, distinct
from `kF2DiskVersion`) and `"pops_sha256": "sha256:<hex>"` (sha256 of the exact `pops.txt` bytes
the writer just wrote). The writer already owns `sha256_file()`, so this is a two-line emission
from a file it controls. External tooling / any future reader can re-hash `pops.txt` to detect
label corruption. The core load path stays `meta.json`-independent.

**Steps.**
1. `f2_dir_writer.cpp`, after `pops.txt` is written (≈line 469): `const std::string pops_hex = sha256_file(dir / "pops.txt");` (reuse the single SHA home).
2. In the `meta.json` emission block: add `"meta_schema_version": 1,` right after `"format"`, and `"pops_sha256": "sha256:" + pops_hex,` adjacent to `f2_cache_id`. Keep the hand-emitted comma/brace shape correct.
3. `f2_dir_writer.hpp`: add `inline constexpr int kF2MetaSchemaVersion = 1;` and a doc comment stating it is the **meta.json schema version, NOT the f2.bin binary version** (`kF2DiskVersion`).
4. **(Optional, S) advisory reader check** in `read_f2_dir`: after `pops.txt` parses, if `meta.json` exists, grab `pops_sha256` with a minimal single-field substring scan (the writer fully controls the format — avoids pulling a JSON dep into the CUDA-free app layer) and warn-or-fault on mismatch. **Must tolerate** `meta.json` absent or field absent (back-compat: the `conftest.py`-written + cli-test minimal `meta.json` files have no `pops_sha256`).
5. Update the `meta.json` field list in `f2_dir_writer.hpp`'s doc comment and `docs/cli-bindings.md` §4.3 (pairs with the §3-Med architecture.md reproducibility doc-sync).

**Parity-risk.** **None to the statistic stream.** Two additive `meta.json` text fields; `f2.bin`
bytes, `F2DiskHeader`, `kF2DiskVersion`, and every computed f2/qpAdm value are untouched —
no golden moves. `meta.json` is currently never read, so writer-only stamping cannot affect any
test. The **only** risk is the optional reader check: a hard fault on absence/mismatch would
break the minimal-`meta.json` fixtures — guarded by making it **present-only and back-compatible**
(absent field = OK) and **not** bumping `kF2DiskVersion` (which the reader *does* gate on,
`f2_dir_io.cpp:87`).

**Verify.** Run `steppe extract-f2` to write a dir; `grep meta_schema_version meta.json` and
`grep pops_sha256 meta.json` present; `sha256sum pops.txt` matches the stamped value (minus the
`sha256:` prefix). `ctest -R cli_extract_qpadm` unchanged (reader ignores meta.json). If the
optional check lands: corrupt one label → flagged; delete the field → still loads.

**Files.** `src/app/f2_dir_writer.cpp`, `src/app/f2_dir_writer.hpp`, `src/app/f2_dir_io.cpp`,
`src/device/f2_disk_format.hpp`.

---

## A5 — One `regenerate_goldens.sh`; delete drift-prone `build_m0.sh`  ·  **P3 · Effort S**

**What's needed.** Golden generation is 8 loose R scripts under
`tests/reference/goldens/at2/scripts/` (`golden_fit0/fit1/fitNA/rot/qpwave/qpgraph_generate.R`
+ `fit0_fixup.R` + `verify_bitexact.R`) that each hardcode box5090 paths
(`/workspace/data/aadr/…`), `admixtools` 2.0.10 / R 4.3.3, and the convertf-PA dataset; the
DATES goldens under `goldens/dates/` are committed AT2/qpdates outputs with no driver. There is
**no single documented "how to regenerate" entry point** — the recipe lives only in the
`at2/README.md` prose (the convertf v8621 TGENO→PA correction). Separately, `build_m0.sh` (repo
root, **confirmed present**, executable) is an M0-era direct-nvcc fallback that **duplicates** the
CMake include paths + source list; CMake builds fine now, so it is pure drift risk
(`RELEASE-SCOPE.md:64` already tags it NICE-to-remove). It is still referenced at
`architecture.md:115` and `tests/CMakeLists.txt:12`.

**Optimal end state.** One top-level `scripts/regenerate_goldens.sh` is the documented,
version-asserting entry point: preflights the pinned toolchain (admixtools 2.0.10 / R 4.3.3 /
convertf v8621) + AADR presence, runs the convertf TGENO→PA stage then the R generators in
dependency order into `tests/reference/goldens/at2/`, and **fails loudly** (not silently
producing a divergent golden) if any version differs. It is a **manual, opt-in tool of record**,
never a CI auto-overwrite. `build_m0.sh` is deleted and its two doc references scrubbed → a single
source of build truth (CMake) and a single source of golden truth (the script + committed values).

**Steps.**
1. Author `scripts/regenerate_goldens.sh`: `set -euo pipefail`; preflight asserting
   `Rscript -e 'packageVersion("admixtools")'` == 2.0.10, R 4.3.3, `convertf` present (+ version),
   and the AADR prefix exists (env-overridable: `AADR_ROOT`, `OUT` defaulting to the committed
   goldens dir).
2. Encode the convertf TGENO→PACKEDANCESTRYMAP stage first (the README correction: raw v66 TGENO
   is silently misread by AT2 v2.0.10 — must convert with convertf v8621 first), then invoke the
   R scripts in order (`fit0_generate → fit0_fixup → verify_bitexact`, then
   `fit1/fitNA/rot/qpwave/qpgraph`), echoing each command. **Thin orchestrator** — reuse the
   scripts as-is, do not inline/fork their bodies.
3. Add a short DATES section: document `goldens/dates/` as AT2/qpdates outputs reproduced from the
   committed `par.dates` files, with the exact `qpdates` invocation (or mark frozen-by-provenance
   if the binary is not on the box).
4. Print a final reminder that after regeneration the committed parity tests (`qpadm_parity`,
   `qpwave_parity`, `qpgraph_*`, `dates_parity`, the rotation goldens) **must be re-run and must
   still pass** — the script regenerates inputs; the tests are the acceptance gate.
5. `rm build_m0.sh`.
6. Scrub references: `architecture.md:115` (the `build_m0.sh [BUILT]` tree line) and the
   `tests/CMakeLists.txt:12` header comment; mark the `RELEASE-SCOPE.md:64` item closed.
7. `chmod +x` the script; reference it from `tests/reference/goldens/at2/README.md` as the
   canonical regen entry point.

**Parity-risk.** **This is the one golden-touching item** — but **only** when a human deliberately
runs it; it does not run in CI and does not auto-commit. Guard against silent drift = the version
preflight (hard-fail unless admixtools 2.0.10 / R 4.3.3 / convertf v8621) **plus** the mandatory
re-run of the committed parity tests. Deleting `build_m0.sh` has zero parity risk (vestigial;
CMake is authoritative). **Do not** let the script overwrite committed JSON/CSV without an
explicit flag.

**Verify.** `bash -n scripts/regenerate_goldens.sh` parses; on box5090 with the pinned toolchain
it reproduces `golden_fit0.json` (CordedWare 0.8688, Turkey_N 0.1312; 391333 SNPs / 710 blocks —
the README's documented values) bit/tolerance-identical and `git diff tests/reference/goldens` is
empty; a wrong admixtools version aborts at preflight. After `rm build_m0.sh`,
`grep -rn build_m0 .` returns only historical kimireview docs, and `cmake --preset ci && ctest`
is unaffected.

**Files.** `build_m0.sh` (delete), `scripts/regenerate_goldens.sh` (new),
`tests/reference/goldens/at2/scripts/*.R`, `tests/reference/goldens/at2/README.md`,
`docs/architecture.md`, `tests/CMakeLists.txt`, `docs/RELEASE-SCOPE.md`.

---

## A6 — Record the `src/core/qpadm/` directory-misnomer decision (rename to `src/core/fit/`, OR document the grab-bag)  ·  **P2 · Effort S (decision XS; execution S)**

> *(cross-cut gap **G7** — from `docs/kimiactions/04-crosscut-vs-kimi.md` §3/§5; X3-#3. Caught by
> **neither** review's action plans — Kimi's closest hit (`06:60`) is build-target granularity, not
> the dir name — so this exists only to make the misleading name read as a deliberate choice.)*

**What's needed.** The directory named for qpAdm is a grab-bag of **28 files** (verified at HEAD)
that houses far more than qpAdm: the qpGraph family (`qpgraph_enumerate.{cpp,hpp}`,
`qpgraph_fit.cpp`, `qpgraph_model.{cpp,hpp}`, `qpgraph_objective.hpp`, `qpgraph_opt_constants.hpp`,
`qpgraph_search.cpp` — 8 files), the f-statistics (`f3.cpp`, `f3_triples.hpp`, `f4.cpp`,
`f4_matrix.hpp`, `f4_quartets.hpp`, `f4ratio.cpp`), the all-quartets sweep (`fstat_sweep.cpp`), the
model-space search (`model_search*.{cpp,hpp}`, `nested_models.{cpp,hpp}`), plus the genuine qpAdm
fit (`qpadm_fit.{cpp,hpp}`, `qpadm_bounds.hpp`, `ranktest.{cpp,hpp}`, `gls_solve.hpp`,
`jackknife.hpp`). An external reader opening `src/core/qpadm/` to find f4 or qpGraph is misled by
the name. **Caught NOWHERE** — no `ASSESSMENT` decision exists.

**Optimal end state.** Either (a) **rename** `src/core/qpadm/ → src/core/fit/` (it is the
fit/search/stats engine, not just qpAdm), or (b) **split** `qpgraph_* → src/core/qpgraph/` and
`f3/f4/f4ratio/fstat_sweep → src/core/fstats/` — **note `src/core/fstats/` already exists** for the
f2-blocks family (`f2_blocks_multigpu*`, `f2_combine`, `f2_from_blocks`, …), so a split must merge
into it, not collide. **Even if the decision is DEFER**, the outcome is a **recorded decision** (a
one-line note in `architecture.md`'s module map + a `// see architecture.md §… — this dir is the
fit/search/stats engine, not qpAdm-only` header pointer) so the name reads as a weighed choice, not
an accident. The decision must **weigh the git-history cost** (a 28-file `git mv` rewrites blame
across the most-touched core dir) against the readability gain — that trade is the whole content of
this item.

**Steps.**
1. **Decide** (the XS part): rename-to-`fit/`, split-into-`qpgraph/`+`fstats/`, or DEFER-and-document.
   Pairs with Kimi `06:60` (the steppe_core sublibrary split) — coordinate so a rename does not
   thrash a sublibrary boundary that item also moves.
2. If **rename**: `git mv src/core/qpadm src/core/fit`; update `src/core/CMakeLists.txt` source
   paths, every `#include "core/qpadm/…"` (grep the tree), and the `architecture.md` module map.
   One mechanical commit; goldens re-run as the gate.
3. If **split**: move `qpgraph_*` to a new `src/core/qpgraph/`, fold `f3/f4/f4ratio/fstat_sweep`
   into the **existing** `src/core/fstats/`, leave the true qpAdm fit in place (or under `fit/`);
   update includes + CMake source lists accordingly.
4. If **DEFER**: add the recorded-decision note (module-map line + header pointer) and stop. This is
   the XS floor and is acceptable per the cross-cut brief ("record it even if the decision is DEFER").

**Parity-risk.** **None — structure/doc only, golden-neutral.** A rename/move changes no source
text, no math, no reduction order; the goldens are the guard and stay byte-identical. The only
hazard is a missed `#include` path after a `git mv`, which is a **compile error, not a parity
move** — caught immediately by the build. The DEFER path touches only a comment + `architecture.md`.

**Verify.** DEFER: `architecture.md` module map names the dir's true scope and a header pointer
exists. Rename/split: `cmake --preset ci` configures, the full build links on box5090, and the
qpAdm / qpGraph / f4 / fstat-sweep parity goldens are **bit-identical** (the move changed only file
locations). `grep -rn 'core/qpadm/' src include bindings tests` returns zero stale include paths.

**Files.** DEFER: `docs/architecture.md`, plus a one-line header pointer in
`src/core/qpadm/qpadm_fit.hpp`. Rename/split: `src/core/qpadm/` (→ `src/core/fit/` or
`src/core/qpgraph/` + `src/core/fstats/`), `src/core/CMakeLists.txt`, every dependent `#include`,
`docs/architecture.md`.

---

## A7 — Dead `DeviceConfig` knobs: `stream_count` / `search_streams` / `use_mem_pool` (+ `kDefaultSearchStreams`)  ·  **P2 · Effort XS**

> *(cross-cut gap **G8** — from `docs/kimiactions/04-crosscut-vs-kimi.md` §3/§5; X6-1. Same
> "declared-not-wired" pattern this plan already treats for `STEPPE_SANITIZER`/`STEPPE_NVTX` — see
> B1.)*

**What's needed.** Three `DeviceConfig` fields read as live throughput knobs but have **zero
production consumers** (verified — a tree grep finds them only at their own declaration + default):
`stream_count = 1` (`include/steppe/config.hpp:353`), `search_streams = kDefaultSearchStreams`
(`:357`, with `kDefaultSearchStreams = 4` at `:183`), and `use_mem_pool = true` (`:362`). The
device path opens a single statistic stream unconditionally per §12, and `cudaMallocAsync` is used
**unconditionally** — `use_mem_pool` never gates it. The only other references are in a doc-comment
banner at `:440,451` (prose about the parked multi-GPU recompute path), not live code. Kimi
(`08:419`) names `stream_count` only to argue for a `validate()` hook — a *different* angle — and
never flags it dead.

**Optimal end state.** The fields no longer read as live tuning surface. Either **delete** the
three fields + `kDefaultSearchStreams` (and the `:440,451` banner prose that describes them), or —
if any is genuinely forward-reserved for the parked multi-GPU/S8 work — **replace the live-sounding
paragraph with a one-line "forward-reserved for parked multi-GPU/S8; not wired" note** so a reader
isn't misled into thinking turning a knob changes behavior. `stream_count` in particular is a §12
trap: it *looks* tunable but **must** stay 1 on the statistic path.

**Steps.**
1. Confirm dead at HEAD (re-grep `stream_count|search_streams|use_mem_pool|kDefaultSearchStreams`
   across `src include bindings` — only the config.hpp sites + the doc banner should appear).
2. Choose per field: **delete** if truly vestigial, or **demote to a one-line forward-reserved
   note** if parked-multi-GPU genuinely intends to wire it.
3. If deleting `use_mem_pool`: leave the `cudaMallocAsync` call unconditional (it already is) and
   drop only the field + its §7/§11.2 doc paragraph.
4. Keep the existing `:440` banner's accurate sentence that `stream_count` is **forced to 1 on the
   statistic path**; remove only the live-knob framing.

**Parity-risk.** **None — these knobs are not wired to any compute.** Deleting an unconsumed field
cannot move a value; the single stream + unconditional `cudaMallocAsync` behavior is unchanged, so
every golden is byte-identical. Only failure mode: a hidden consumer the grep missed → a **compile
error**, not a parity move; the build catches it. If any field is set by a test fixture, keep it as
the documented forward-reserved no-op rather than deleting.

**Verify.** `grep -rn 'stream_count\|search_streams\|use_mem_pool\|kDefaultSearchStreams' src include bindings tests`
shows only intended sites after the edit; `cmake --preset ci` + full build link on box5090; the
qpAdm/f2 goldens are bit-identical (no compute touched).

**Files.** `include/steppe/config.hpp`.

---

## A8 — `target_compile_features(cxx_std_20)` on `steppe_core` + `steppe_device`  ·  **P3 · Effort XS**

> *(cross-cut gap **G16** — from `docs/kimiactions/04-crosscut-vs-kimi.md` §3/§5; X1-F7.)*

**What's needed.** The C++20 standard is declared per-target with `target_compile_features(…
cxx_std_20)` on **7** targets at HEAD — `steppe_access` (`src/access/CMakeLists.txt:34`),
`steppe_extract` (`src/extract/CMakeLists.txt:38`), `steppe_core_internal` INTERFACE
(`src/core/CMakeLists.txt:34`), `steppe_app` (`src/app/CMakeLists.txt:61`), `steppe_api`
(`include/CMakeLists.txt:17`), `steppe_io` (`src/io/CMakeLists.txt:42`), `_core`
(`bindings/CMakeLists.txt:63`). But the two **STATIC compute libraries** omit it:
`steppe_core` (the STATIC lib at `src/core/CMakeLists.txt:67`) and `steppe_device`
(`src/device/CMakeLists.txt:22`) declare no `target_compile_features`. They compile as C++20 today
only by transitively inheriting it — the standard isn't declared the same way across the tree, an
inconsistency a reader (and a future consumer of the targets) trips on.

**Optimal end state.** All compile targets declare the standard the same way: add
`target_compile_features(steppe_core PUBLIC cxx_std_20)` and
`target_compile_features(steppe_device PUBLIC cxx_std_20)` so the C++20 floor is explicit and
self-documenting on every target, matching the existing 7. A 2-line CMake add.

**Steps.**
1. `src/core/CMakeLists.txt` after the `add_library(steppe_core STATIC …)` block (`:67`): add
   `target_compile_features(steppe_core PUBLIC cxx_std_20)`.
2. `src/device/CMakeLists.txt` after `add_library(steppe_device STATIC …)` (`:22`): add
   `target_compile_features(steppe_device PUBLIC cxx_std_20)`.
3. Use `PUBLIC` to match the access/extract/io STATIC libs (so the requirement propagates the same
   way), not `PRIVATE`.

**Parity-risk.** **None — build-metadata only.** These targets already compile as C++20 by
inheritance; declaring it explicitly changes no flags that reach codegen on a correctly-configured
build (the standard is identical), so objects are byte-identical and no golden moves. Only failure
mode: a stale toolchain that was silently compiling these as a *lower* standard would now fail to
configure — which is the **desired** surfacing, not a regression.

**Verify.** `cmake --preset ci` configures clean; `grep -rn cxx_std_20 src include bindings` now
shows **9** targets (7 + the two added); full build links on box5090; qpAdm/f2 goldens bit-identical.

**Files.** `src/core/CMakeLists.txt`, `src/device/CMakeLists.txt`.

---

# Cluster B — Small code / robustness polish

> NVTX is the highest-leverage single item in the document: it unlocks real Nsight profiling
> of the parity-frozen pipeline and makes a declared-but-dead option honest. The rest are XS
> correctness fixes; the workspace-pool item is honest **hygiene, not a measured perf win**.

## B1 — Wire NVTX ranges behind `STEPPE_NVTX` (option declared, emits zero ranges)  ·  **P1 · Effort S**

**What's needed.** `STEPPE_NVTX` is declared at `cmake/SteppeOptions.cmake:40`
("Emit NVTX ranges … zero-overhead off", OFF) but **consumed nowhere** — a repo grep finds it
only at that line + the doc comment at line 8 (**confirmed**): no `target_compile_definitions`,
no `nvtx3` include, no range calls. CUDA 13 ships header-only NVTX3 (`nvtx3/nvtx3.hpp`), so no
library link is needed. Work = (a) a one-line CMake compile-definition gate, (b) a thin RAII
macro that compiles to nothing when the define is absent, (c) ~6–12 range markers at real phase
boundaries (decode, f2 GEMM batch, jackknife, qpAdm fit) — **explicitly not** inside the per-block
hot inner loops.

**Optimal end state.** With `-DSTEPPE_NVTX=ON`, an Nsight Systems capture of a real-AADR
extract/qpAdm run shows named ranges (`decode`, `f2_gemm`, `jackknife`, `qpadm_fit`) on the
timeline. With the option OFF (shipping default) the build is **byte-identical to today**: the
macro expands to empty, no nvtx symbols link, every golden still passes. The option string's
"zero-overhead off" promise becomes true instead of aspirational.

**Steps.**
1. New CUDA-free wrapper header `src/core/internal/nvtx.hpp`: when `STEPPE_NVTX` is defined,
   `#include <nvtx3/nvtx3.hpp>` and `#define STEPPE_NVTX_RANGE(name) nvtx3::scoped_range _steppe_nvtx_##__LINE__{name}`; else `#define STEPPE_NVTX_RANGE(name)` (empty).
2. In the relevant `CMakeLists` (`src/device`, and any TU that gets a marker):
   `if(STEPPE_NVTX) target_compile_definitions(<tgt> PRIVATE STEPPE_NVTX) endif()` — **mirror
   the existing `STEPPE_HAVE_EMU_TUNING` pattern** at `src/device/CMakeLists.txt:74`.
3. Place ~6–12 `STEPPE_NVTX_RANGE` markers at **coarse phase boundaries only** (decode entry,
   `compute_f2_blocks`, the jackknife kernel launch wrapper, `qpadm_fit`). **Keep them out of
   per-block / per-quartet inner loops** so even ON they don't distort the kernel timeline.
4. Confirm `nvtx3/nvtx3.hpp` resolves under the CUDA 13 toolkit on box5090 (part of the toolkit
   include tree; no `find_package`/link for the header-only v3 C++ API).

**Parity-risk.** **None.** NVTX is host-side annotation only — no device work, no stream creation,
no sync, no kernel/math change. Structurally guarded: default build defines nothing → macro empty
→ object code identical to HEAD. Even ON, ranges are passive markers that never touch the single
statistic stream (§12 untouched). Markers off hot loops so an ON profiling build does not perturb
relative timing.

**Verify.** On box5090: build `-DSTEPPE_NVTX=ON`, run a small real-AADR extract under `nsys`,
confirm the named ranges appear. Build default (OFF) and confirm (a) `grep -r STEPPE_NVTX` now
hits a `target_compile_definitions`, (b) `nm` on the binary shows no nvtx symbols, (c) a qpAdm/f2
golden parity test is byte-identical to pre-change.

**Files.** `cmake/SteppeOptions.cmake`, `src/device/CMakeLists.txt`, `src/core/internal/nvtx.hpp`
(new), `src/device/cuda/cuda_backend.cu`, `src/device/cuda/dates_kernel.cu`,
`src/device/cuda/qpfstats_kernel.cu`.

---

## B2 — Forward `std::error_code::message()` into `f2_dir_io` fault messages  ·  **P2 · Effort XS**

**What's needed.** `read_f2_dir` (`src/app/f2_dir_io.cpp`) threads a `std::error_code ec` into
`fs::exists`/`fs::is_directory` at lines 61–62 but **discards** `ec.message()`, returning only the
generic `"--f2-dir is not a directory: <path>"` (lines 63–65). The two `ifstream` opens —
`read_pops_txt` at line 41 and `f2.bin` at line 71 — capture **no errno at all**, so a
permission-denied (EACCES) or missing-file open is reported with the path but **no OS reason**.
Forward the OS message so a sysadmin can tell "permission denied" from "no such file".
*(Scope correction: the assessment names ENOSPC under this reader, but ENOSPC is a **write** fault
— the reader only sees EACCES/ENOENT. The symmetric write-side fix lives in `f2_dir_writer.cpp`
and overlaps the separate §3-Med "check post-write stream state" item.)*

**Optimal end state.** Every OS-originated fault arm carries the decoded reason, e.g.
`"cannot open f2.bin (Permission denied): <path>"`. The carried `Status` stays `InvalidConfig`
(taxonomy unchanged); only the human-readable string gains OS detail.

**Steps.**
1. Lines 62–65: append `ec.message()` to the directory-check failure string (the `ec` is already
   populated by `fs::exists`/`is_directory`).
2. The `ifstream` opens at line 41 (pops.txt) and line 71 (f2.bin): immediately after a failed
   open, capture the OS reason and append it — read `errno` right then via
   `std::generic_category().message(errno)` / `std::strerror(errno)`, **before any other libc
   call** so it is not clobbered.
3. Keep the `Status::InvalidConfig` classification and existing message prefixes; only append the
   OS detail in parentheses.

**Parity-risk.** **None.** Touches only the human-readable error string on the fault path; the
carried `Status` and all success-path bytes are unchanged, and no golden inspects error text. The
only correctness subtlety is errno-immediacy — capture on the line right after the failing open.

**Verify.** Extend a unit test: point `read_f2_dir` at a `chmod 000` directory and at an
unreadable `f2.bin`; assert the reason string now contains the OS message (e.g. "Permission
denied") and `Status` is still `InvalidConfig`. Confirm existing `read_f2_dir` tests unchanged.
(Host-only TU — compiles/tests on the CUDA-free lane.)

**Files.** `src/app/f2_dir_io.cpp`.

---

## B3 — `geno_hash_thread` try/catch + `exception_ptr` (no `std::terminate` on escape)  ·  **P2 · Effort XS**

**What's needed.** `cmd_extract_f2.cpp:292–294` runs `geno_sha = sha256_file(path)` on a raw
`std::thread` whose lambda has **no try/catch**. An exception escaping a `std::thread`'s top-level
callable calls `std::terminate` (hard crash, no diagnostic). The throw surface is tiny:
`sha256_file` (`f2_dir_writer.cpp:356–371`) returns `""` on open-failure and uses an
exception-disabled `ifstream`, so the only realistic throw is `bad_alloc` on the 8 MiB
`std::vector<char> chunk(8u<<20)`. The codebase already has the idiomatic pattern (per-worker
`std::exception_ptr` captured then rethrown after join) at `f2_blocks_multigpu_core.cpp:74` and
`model_search.cpp:290`. The `ThreadJoiner` RAII guard already exists at lines 298–301.

**Optimal end state.** The hashing worker can never call `std::terminate`. The lambda body is
wrapped in try/catch; on throw it stores a `std::exception_ptr`. At the join site the command
either **rethrows** (top-level catch → real exit code) or **degrades** to empty `geno_sha` +
`source_hash_computed:false` with a stderr warning — the same observable state as today's
open-failure path.

**Steps.**
1. Add a captured `std::exception_ptr geno_hash_err;` alongside `geno_sha` (≈line 288).
2. Wrap the lambda body (lines 292–294): `try { geno_sha = sha256_file(...); } catch(...) { geno_hash_err = std::current_exception(); }`.
3. After the explicit join (and in the `ThreadJoiner` RAII fallback), if `geno_hash_err` is set:
   **choose and document** — rethrow (clean exit via the command catch) **or** degrade to
   `geno_sha=""` + `source_hash_computed:false`. Pick **degrade** (the hash is non-essential UX;
   matches the existing `""` open-failure semantics); rethrow only if a `bad_alloc` should fail
   the run.
4. If degrading, emit a one-line stderr warning so the user knows the hash was skipped.

**Parity-risk.** **None on the success path** — when the hash computes (normal case), `meta.json`
is byte-identical, so any meta.json golden is unaffected. The change only alters the abort/degrade
behavior; `std::terminate` becomes either a clean exit or an empty-sha + `source_hash_computed:false`
meta — **already a representable state today** on open-failure. Keep the degrade output identical
to the existing `""` path so no new meta shape appears.

**Verify.** Code-review that no exception escapes the thread (the `catch(...)` is total). Add a
seam test that the join path tolerates a worker that produced an empty sha (open-failure proxy)
and writes `source_hash_computed:false`. Confirm the `extract_f2` golden (`--hash` on a readable
`.geno`) is unchanged.

**Files.** `src/app/cmd_extract_f2.cpp`.

---

## B4 — Drop the `mutable` on `ConfigBuilder::error_message_` via a non-const `build()`  ·  **P3 · Effort XS**

**What's needed.** `config_builder.hpp:120` declares `mutable std::string error_message_;` **solely**
so the `const`-qualified `build()` (decl `:106`, def `config_builder.cpp:209`) can write it. The
`mutable` is a const-correctness smell: `build()` logically mutates the builder's failure state, so
it should just be non-const. **Verified safe:** the sole production caller (`cli_parse.cpp:61–72`)
holds a non-const `ConfigBuilder`, and every test calls `build()` on a non-const temporary lvalue
via the fluent chain. **No caller holds a `const ConfigBuilder`** (`grep -rn 'const ConfigBuilder'`
clean).

**Optimal end state.** `build()` is a non-const member writing the plain (non-`mutable`)
`error_message_`; `error_message()` stays a `const noexcept` accessor. The const-correctness reads
honestly and the §9 "ConfigBuilder is the only mutable config type" framing is reinforced.

**Steps.**
1. `config_builder.hpp:120` — remove `mutable` from `error_message_`.
2. `config_builder.hpp:106` — `[[nodiscard]] BuildResult<RunConfig> build();` (drop `const`).
3. `config_builder.cpp:209` — `BuildResult<RunConfig> ConfigBuilder::build() {` (drop `const`); the
   fail lambda still captures `[this]` and writes `error_message_` — now legal without `mutable`.
4. Leave `error_message()` `const`. Confirm no `const ConfigBuilder` caller (grep).

**Parity-risk.** **None.** Pure const-qualifier refactor, no runtime behavior change — same error
string, same `BuildResult`. No compute, no golden surface. Only failure mode: a compile error if a
hidden const caller existed; grep confirms none.

**Verify.** Builds clean (host-only TU — **compiles even on the local box**). Run
`test_config_builder.cpp` — all cases pass, including the `error_message()` non-empty assertion.
Grep confirms no remaining `mutable` in `config_builder.hpp` and no `const ConfigBuilder` caller.

**Files.** `src/core/config/config_builder.hpp`, `src/core/config/config_builder.cpp`.

---

## B5 — Pool per-call cuSOLVER potrf/potri/gesvd workspace + cap the pinned staging  ·  **P3 · Effort S (hygiene, not perf)**

**What's needed.** Four cuSOLVER scratch allocations are done **per-call** as RAII
`DeviceBuffer`s, each paying a device-wide `cudaMalloc`+`cudaFree`: potrf `dWork` at
`cuda_backend.cu:2099` and `:2364` (the shared-factor smoothing solves), potrf/potri `dWork` at
`:4044` (reused at `:4055` for the Qinv SPD path), and the multi-buffer gesvd/gesvdj scratch in the
single-shot `large_svd_V` overload (≈`:4200–4221`). **The inner-loop hot scratch was already hoisted**
(§14.2; the LOO sweep calls the scratch-taking overload). Pool the remaining per-call scratch into
persistent per-backend members that grow monotonically to the max `lwork` seen, mirroring the
existing `stage_f2_`/`stage_vpair_` pattern (members at `:5647–5648`, monotonic-grow at `:537–538`).
Separately, that pinned staging grows monotonically and **never shrinks** (`:537–538`); add an
explicit cap/assert so a pathological partial size cannot silently pin an unbounded non-pageable
host buffer.

**Optimal end state.** `CudaBackend` holds a small set of reusable scratch members (e.g.
`solver_work_` for the potrf/potri `lwork` buffer + the gesvd scratch set) sized by
`if (buf.size() < need) buf = DeviceBuffer(need);`. Across a fit/rotation the cuSOLVER
`cudaMalloc`/`cudaFree` count drops to near-zero after warmup; bytes fed to every routine are
unchanged. The pinned staging carries a documented upper bound (= the per-device partial
`P·P·n_block_local`, already VRAM-bounded) with an assert before an over-large pin.

**Steps.**
1. Add persistent members: `DeviceBuffer<double> solver_work_{};` + the gesvd scratch set
   (`dS/dU/dVt/dA2/dInfo`); size each via the monotonic-grow idiom.
2. At `:2099`, `:2364`, `:4044`: replace the local `dWork` with
   `if (solver_work_.size() < lwork) solver_work_ = DeviceBuffer<double>(lwork);` then pass
   `solver_work_.data()`; reuse for the potri at `:4055` (same `lwork_f`).
3. For the single-shot `large_svd_V`: grow the member scratch set to the queried gesvd/gesvdj
   sizes and **delegate to the scratch-taking overload** (which already exists) instead of stack
   `DeviceBuffer`s.
4. Add the cap to the staging at `:537–538`: assert/guard that `total` does not exceed the
   expected partial bound before growing `stage_f2_`/`stage_vpair_`; keep never-shrink.
5. Document the safety invariant: each device owns its own `CudaBackend`
   (`model_search.cpp:292` fans out `resources.gpus[g].backend` per jthread) and §12 pins a single
   stream → pooled scratch is used strictly sequentially, never shared across threads.

**Parity-risk.** **None.** cuSOLVER workspace is uninitialized scratch that potrf/potri/gesvd fully
overwrite; pooling changes only the allocation count, not a byte fed to the routine → bit-identical.
Thread-safety is the only real hazard and is satisfied: one backend per device + single stream §12
⇒ sequential use, no race. **Must** (a) keep monotonic-grow/never-shrink, (b) keep `solver_work_` a
**distinct member** from the f2 D2H staging (`stage_f2_`/`stage_vpair_`) so the two never alias,
(c) not reorder relative to the `engage_solver_precision` math-mode scopes at `:2093`/`:4031`.

**Verify.** On box5090 release build: run the qpAdm fit goldens (`golden_fit0`,
`golden_fit1_NRBIG`) and the rotation golden — **bit-identical to HEAD** (the parity gate is the
guard). Optionally `compute-sanitizer memcheck` over the fit path to confirm no use-after-free on
the pooled buffer, and `nsys` to confirm the per-fit `cudaMalloc` count dropped (value check, not
correctness). **Honest framing:** §14.2 already hoisted the hot inner-loop scratch, so this is
allocation-count tidiness + a defensive cap, **not a measured win** — file as hygiene.

**Files.** `src/device/cuda/cuda_backend.cu`.

---

## B6 — 14 bare `STEPPE_CUDA_CHECK(cudaGetLastError())` → canonical `STEPPE_CUDA_CHECK_KERNEL()`  ·  **P1 · Effort S (breadth) — do before the compute-sanitizer pass**

> *(cross-cut gap **G4** — from `docs/kimiactions/04-crosscut-vs-kimi.md` §3/§5; X1-F1. Framed as a
> prerequisite/multiplier for the `02-ci-plan.md` §3-High one-shot compute-sanitizer pass — do this
> FIRST so that pass keeps launch-site attribution.)*

**What's needed.** Fourteen post-launch error checks call the bare
`STEPPE_CUDA_CHECK(cudaGetLastError())` instead of the canonical `STEPPE_CUDA_CHECK_KERNEL()` macro
(defined at `src/device/cuda/check.cuh:332`). Verified at HEAD — **14** sites across 3 files:
`dates_kernel.cu` (×9), `qpgraph_fit_kernels.cu` (×3), `dstat_kernel.cu` (×2). (A 15th
`STEPPE_CUDA_CHECK(cudaGetLastError())` lives inside `check.cuh` itself — the macro's own body — and
is **not** a site to change.) The bare form only pops the launch-config error; `CHECK_KERNEL` adds
the **debug-only** `cudaDeviceSynchronize` that attributes an *async* kernel fault to the launch
site under compute-sanitizer.

**Optimal end state.** All 14 post-launch sites use `STEPPE_CUDA_CHECK_KERNEL()`, so the
*already-planned* compute-sanitizer pass (`02-ci-plan` §3-High) reports faults in the dates / dstat /
qpgraph kernels **at their launch site** rather than at some later unrelated sync. The canonical
macro becomes the one post-launch check pattern across the device layer. Zero behavior change in a
release build (the extra sync is debug-only inside the macro).

**Steps.**
1. In `dates_kernel.cu` (9 sites), `qpgraph_fit_kernels.cu` (3), `dstat_kernel.cu` (2): replace each
   post-launch `STEPPE_CUDA_CHECK(cudaGetLastError());` with `STEPPE_CUDA_CHECK_KERNEL();`.
2. Confirm each is genuinely a **post-kernel-launch** check (not a generic API-return check) so the
   semantic swap is correct; leave the `check.cuh:332` macro definition and its internal
   `cudaGetLastError` untouched.
3. Re-grep to confirm 0 bare `STEPPE_CUDA_CHECK(cudaGetLastError())` remain outside `check.cuh`.

**Parity-risk.** **None — golden-neutral.** `STEPPE_CUDA_CHECK_KERNEL` checks the *same* launch
error and adds only a **debug-only** synchronize; in the release/parity build the emitted check is
equivalent, no math, no stream, no reduction order changes, every golden byte-identical. Only
failure mode: applying the swap to a non-launch `cudaGetLastError` site (semantically wrong) —
guarded by the per-site "is this a post-launch check" audit in step 2.

**Verify.** `grep -rn 'STEPPE_CUDA_CHECK(cudaGetLastError())' src` returns only `check.cuh`;
release build on box5090 links and the dates/dstat/qpgraph parity goldens are bit-identical; under a
debug `compute-sanitizer` run, an injected fault in one of these kernels now reports the launch
site (the multiplier the planned sanitizer pass wants).

**Files.** `src/device/cuda/dates_kernel.cu`, `src/device/cuda/qpgraph_fit_kernels.cu`,
`src/device/cuda/dstat_kernel.cu`.

---

## B7 — Remove the vestigial `rank_test` virtual (base throws, both backends override, only caller is an uncalled wrapper)  ·  **P2 · Effort XS**

> *(cross-cut gap **G9** — from `docs/kimiactions/04-crosscut-vs-kimi.md` §3/§5; X6-2. A concrete
> down-payment on `ASSESSMENT §5.2` "slim the backend.hpp seam (DEFER)" — trims one virtual without
> the full role-split.)*

**What's needed.** `ComputeBackend::rank_test` is a vestigial virtual: the base declaration
**throws** "not implemented by this backend" (`src/device/backend.hpp:1692`, throw at `:1698`), both
backends override it (`cuda_backend.cu:4386`, `cpu_backend.cpp:1556`), and its **only** "caller" is
a free-function wrapper `rank_test(ComputeBackend&, …)` in `src/core/qpadm/gls_solve.hpp:17` that
**has zero callers itself** (verified — a tree grep for `rank_test` outside the four
declaration/override/wrapper sites is empty). The live rank path uses `rank_sweep` + `gls_weights`,
not `rank_test`. So the virtual + two overrides + wrapper are dead weight on the already-fat
56-virtual `backend.hpp` seam.

**Optimal end state.** The `rank_test` pure-virtual declaration, the two backend overrides, and the
`gls_solve.hpp` wrapper are all deleted. The `ComputeBackend` interface shrinks by one virtual; the
backends drop ~30 lines each of unreachable seed-factor code; `gls_solve.hpp` loses an unused entry
point. Nothing in the live `rank_sweep`/`gls_weights` path changes.

**Steps.**
1. Confirm dead at HEAD: `grep -rn 'rank_test' src bindings tests` shows only `backend.hpp:1692/1698`,
   `cuda_backend.cu:4386`, `cpu_backend.cpp:1556`, `gls_solve.hpp:17,20` — no live caller.
2. Delete the `gls_solve.hpp` free-function wrapper (`:17–20`) first (it is the only named consumer).
3. Delete the `rank_test` override in `cuda_backend.cu:4386…` and in `cpu_backend.cpp:1556…`.
4. Delete the base virtual declaration + throw body in `backend.hpp:1692–1698`; tidy the adjacent
   doc comment at `:1704` that references the `rank_test seed → als refine → chisq` machinery if it
   no longer applies.

**Parity-risk.** **None — dead code, golden-neutral.** The deleted virtual is never invoked on the
live path (the only wrapper that names it has zero callers), so removing it moves no value; every
golden is byte-identical. Only failure mode: a caller the grep missed → a **compile/link error**,
not a parity move; the build catches it immediately.

**Verify.** `grep -rn 'rank_test' src bindings tests` returns **0** after the edit; full build links
on box5090; the qpAdm fit + rank-related parity goldens (`golden_fit0`, `golden_fit1_NRBIG`) are
bit-identical (the live `rank_sweep`/`gls_weights` path is untouched).

**Files.** `src/device/backend.hpp`, `src/device/cuda/cuda_backend.cu`,
`src/device/cpu/cpu_backend.cpp`, `src/core/qpadm/gls_solve.hpp`.

---

## B8 — Device-layer convention re-align (the two lone-deviator files)  ·  **P3 · Effort XS**

> *(cross-cut gap **G11** — from `docs/kimiactions/04-crosscut-vs-kimi.md` §3/§5; X1-F2/F3/F4. One
> tiny cosmetic commit collapsing the few device-TU convention deviations.)*

**What's needed.** A handful of device TUs deviate from the established conventions the other 43
device TUs follow. Verified at HEAD: (a) `qpgraph_fit_kernels.{cu,cuh}` open
`namespace steppe::device::cuda` (`qpgraph_fit_kernels.cuh:12`, `.cu:37`) where 43 other device TUs
use `namespace steppe::device`; (b) `qpgraph_fit_kernels.cuh:55` spells the raw qualifier
`__host__ __device__` (`make_layout`) instead of the project macro `STEPPE_HD`; (c)
`qpadm_fit_kernels.cuh` names device-pointer inputs `f2`/`vpair` with **no `d_` prefix** at the
`launch_assemble_*` signatures (`:37` `launch_assemble_f4_gather(const double* f2, …)`, `:53`
`launch_assemble_f4_quartets_gather`, `:69` `launch_assemble_f3_triples_gather`, `:357`
`launch_assemble_f4_gather_models_batched`) alongside correctly-prefixed `int* d_keep`.

**Optimal end state.** The deviators match the device-layer house style: `namespace steppe::device`
(drop the extra `::cuda`), `STEPPE_HD` for the host/device qualifier, and a `d_` prefix on the
device-pointer parameters so the same convention reads uniformly across all device TUs. One
cosmetic re-align commit.

**Steps.**
1. `qpgraph_fit_kernels.{cu,cuh}`: change `namespace steppe::device::cuda { … }` to
   `namespace steppe::device { … }` (open + close), and update any internal qualified references
   accordingly.
2. `qpgraph_fit_kernels.cuh:55`: replace raw `__host__ __device__` on `make_layout` with `STEPPE_HD`.
3. `qpadm_fit_kernels.cuh` (`:37,:53,:69,:357`) + the matching `.cu` definitions: rename the
   device-pointer params `f2`→`d_f2`, `vpair`→`d_vpair` to match the `d_keep` convention; update
   call sites. **Mechanical rename only** — no logic, no argument reordering.

**Parity-risk.** **None — cosmetic, golden-neutral.** A namespace move, a qualifier-macro swap, and
parameter renames change no statement, no math, no ABI that any golden inspects; the kernels emit
identical code. Only failure mode: a missed reference after the namespace change → a **compile
error**, not a parity move; the build catches it.

**Verify.** Full build links on box5090; `grep -rn 'steppe::device::cuda' src/device` returns 0 (or
only intended sites); `grep -n '__host__ __device__' src/device/cuda/qpgraph_fit_kernels.cuh` returns
0; the qpGraph + qpAdm fit parity goldens are bit-identical.

**Files.** `src/device/cuda/qpgraph_fit_kernels.cu`, `src/device/cuda/qpgraph_fit_kernels.cuh`,
`src/device/cuda/qpadm_fit_kernels.cuh`, `src/device/cuda/qpadm_fit_kernels.cu`.

---

## B9 — Rename the confusable sibling TUs `f2_block_kernel` vs `f2_blocks_kernel` (one `s`)  ·  **P3 · Effort XS**

> *(cross-cut gap **G13** — from `docs/kimiactions/04-crosscut-vs-kimi.md` §3/§5; X7-F6. X7-unique;
> Kimi touches `f2_block_kernel.cu` for magic numbers but never flags the name trap.)*

**What's needed.** Two distinct device TUs differ only by a single `s` and are trivially confused:
`src/device/cuda/f2_block_kernel.{cu,cuh}` vs `src/device/cuda/f2_blocks_kernel.{cu,cuh}` (both
present at HEAD). A reader or grep can't tell at a glance which is the per-block primitive and which
is the batched-blocks path; an `#include` typo silently picks the wrong one.

**Optimal end state.** One of the pair is renamed to a name that states its role unambiguously (e.g.
the batched path → `f2_batched_kernel.{cu,cuh}`, or the singular primitive → `f2_single_block_kernel`),
so the two filenames are no longer one keystroke apart. Includes + CMake source lists updated.

**Steps.**
1. Decide which TU is the per-block primitive and which is the batched-blocks path (read both
   headers); pick the clearer rename (`f2_batched_kernel` is the suggested target for the `…blocks…`
   batched TU).
2. `git mv` the chosen `.cu` + `.cuh`; update `src/device/CMakeLists.txt` source list and every
   `#include "…/f2_block(s)_kernel.cuh"`.
3. Re-grep to confirm no stale include path remains.

**Parity-risk.** **None — file rename only, golden-neutral.** No source text, no math, no kernel
behavior changes; the goldens are the guard and stay byte-identical. Only failure mode: a missed
include after the `git mv` → a **compile error**, not a parity move.

**Verify.** `grep -rn 'f2_block_kernel\|f2_blocks_kernel' src include tests` shows only the
intended (renamed) paths; full build links on box5090; the f2 / qpAdm goldens are bit-identical.

**Files.** `src/device/cuda/f2_block_kernel.{cu,cuh}` and/or
`src/device/cuda/f2_blocks_kernel.{cu,cuh}` (one pair renamed), `src/device/CMakeLists.txt`,
dependent `#include` sites.

---

# Cluster C — API / developer-experience polish (portfolio value)

> Five items. Three are zero-risk XS header/sugar edits that **batch into one commit**
> (C1/C2/C3). The examples + Doxygen are S-effort portfolio surface; write the examples
> **after** the header sugar so they showcase `Precision::emulated_fp64()` + `f2_at`.
> *(Cross-cut additions C6–C9 below extend this cluster; C9 joins the C1/C2/C3 header-sugar batch.)*

## C1 — `Precision` named factories: `fp64()` / `emulated_fp64(bits)` / `tf32()`  ·  **P2 · Effort XS**

**What's needed.** `Precision` (`include/steppe/config.hpp:284–331`) is an aggregate with
`Kind kind = Kind::EmulatedFp64` and `int mantissa_bits = kDefaultMantissaBits` (=40,
`config.hpp:44`). Callers must write the `Precision{Kind::EmulatedFp64, 40}` incantation. Add three
inline static factories returning a populated `Precision`: `fp64()` → `{Fp64, kDefaultMantissaBits}`;
`emulated_fp64(int bits = kDefaultMantissaBits)` → `{EmulatedFp64, bits}`; `tf32()` →
`{Tf32, kDefaultMantissaBits}`. `Precision` lives in the `steppe_api` INTERFACE target
(`include/CMakeLists.txt`), so the functions **must be inline**.

**Optimal end state.** Callers write `Precision::emulated_fp64()` / `::fp64()` / `::tf32()` instead
of brace-init with a magic Kind+bits. The struct **stays a default-constructible aggregate** —
adding static member functions does not break aggregate-ness in C++20 (only user-declared
constructors/virtuals/non-public data members would), so `Precision p{};` and `Precision{Kind::Tf32}`
keep compiling.

**Steps.**
1. Add three `inline static` factories to the `Precision` struct, each returning the same
   `{kind, mantissa_bits}` the aggregate init produces today.
2. `emulated_fp64`'s default arg **must be `kDefaultMantissaBits`** so the no-arg call equals
   today's default (40) exactly.
3. *(Optional, separate commit)* grep every `Precision{…}` / `Kind::` construction site and migrate
   to the factories, **preserving each site's exact mantissa value**.
4. Add a `static_assert` pinning `Precision::emulated_fp64().mantissa_bits == kDefaultMantissaBits`
   && `.kind == EmulatedFp64`.

**Parity-risk.** **Behavior-neutral** — factories emit byte-identical `{kind, mantissa_bits}` to
what the code constructs now; `emulated_fp64()` defaults to the same 40 bits the f2 GEMMs already
use, so no precision mode changes and no golden can move. The **only** way to move a golden is a
migrated call site silently changing a mantissa value — guarded by diffing `{kind,bits}` at every
migrated site and keeping migration in its own commit.

**Verify.** Header compiles in `steppe_api` consumers (core/device/bindings/CLI). `static_assert`
on the factory outputs. Run the qpAdm parity tests on the box (`test_qpadm_parity.cu`) — unchanged.
Grep confirms no construction site changed its `mantissa_bits`.

**Files.** `include/steppe/config.hpp`, `bindings/module.cpp` (optional migration site).

---

## C2 — `F2BlockTensor::f2_at(i,j,b)` accessor  ·  **P2 · Effort XS**

**What's needed.** `F2BlockTensor` (`include/steppe/fstats.hpp:47–78`) exposes
`std::vector<double> f2/vpair` and the documented flat layout `i + P·j + P·P·b`
(`fstats.hpp:35–46`) but forces users to hand-roll the index math. **Confirmed: the struct has only
`size()` today, no `f2_at`.** Add inline accessors:
`[[nodiscard]] double f2_at(int i,int j,int b) const noexcept` + a `double&` overload computing the
flat index; mirror with `vpair_at`; and a `std::span<const double> block(int b) const` returning the
`[P·P]` slab. **Critical:** do the arithmetic in `std::size_t`, **not** `int` — at production scale
(P≈768, n_block≈800) `P·P·n_block ≈ 4.7e8` overflows `int` (the design-for-scale rule; `size()`
already casts at `fstats.hpp:74–76`). Use the same cast pattern as `size()`.

**Optimal end state.** Users index via `t.f2_at(i,j,b)` / `t.vpair_at(i,j,b)` / `t.block(b)`
matching the layout doc, with const + mutable overloads, size_t-safe index math, `noexcept`
unchecked (hot path). The accessor formula is the **single canonical home** for the `i + P·j + P·P·b`
convention, so it cannot drift from the writer (`src/app/f2_dir_writer.cpp`) or the kernel layout.
Requires `#include <span>`.

**Steps.**
1. Add inline `double f2_at(int,int,int) const noexcept` + `double& f2_at(int,int,int)` computing
   `f2[size_t(i) + size_t(P)*j + size_t(P)*size_t(P)*b]`.
2. Add `vpair_at(...)` overloads with the identical index.
3. Add `std::span<const double> block(int b) const` returning
   `{f2.data() + size_t(P)*P*b, size_t(P)*P}`; `#include <span>`.
4. One-line doc each pointing at the §LAYOUT block; *(optional)* a checked `at()` variant for
   debug builds.

**Parity-risk.** **None** — pure read/index helpers; no compute path consumes them unless adopted,
and the formula is identical to the already-used+documented layout. Behavior-neutral. (Adoption
inside compute would be a separate, separately-verified change.)

**Verify.** Unit test: construct a small `F2BlockTensor`, write `f2` via the flat index, assert
`f2_at` returns the same and matches a hand-rolled `i+P·j+P·P·b`; cross-check the convention against
a `read_f2_dir` round-trip. Header compiles in all `steppe_api` consumers.

**Files.** `include/steppe/fstats.hpp`, `src/app/f2_dir_io.hpp` (round-trip cross-check).

---

## C3 — Replace the `f2_to_numpy` element loop with `std::memcpy`  ·  **P2 · Effort XS**

**What's needed.** `f2_to_numpy` (`bindings/module.cpp:1025–1041`) copies element-by-element:
`for (size_t i=0;i<n;++i) buf[i]=src[i];` (**lines 1031–1032, confirmed**). `src` is a contiguous
`std::vector<double>` and `n == h.tensor.size() == P·P·n_block`, and the **same** helper serves both
exports (`_f2_numpy` passes `h.tensor.f2` at `:1059`, `_vpair_numpy` passes `h.tensor.vpair` at
`:1062`), each of length `n`. For P=768, n_block≈800 this is ~3.7 GiB — the scalar loop is real
wasted bandwidth on a hot export path. Replace with `std::memcpy(buf, src.data(), n*sizeof(double))`.

**Optimal end state.** A single `std::memcpy` (add `#include <cstring>`) producing a byte-identical
buffer; optionally `assert(src.size()==n)` to pin the precondition both callers already satisfy.
F-contiguous (P,P,n_block) numpy semantics and the capsule deleter are unchanged.

**Steps.**
1. Add `#include <cstring>` to `bindings/module.cpp`.
2. Replace the for-loop at `:1031–1032` with `std::memcpy(buf, src.data(), n*sizeof(double));`.
3. *(Optional)* `assert(src.size()==n);` documenting the precondition both callers meet.

**Parity-risk.** **None** — bit-identical copy of FP64 doubles; the exported numpy array is
byte-for-byte the same, and this buffer is a **host export**, not part of the statistic stream /
§12 reduction order. No golden involvement.

**Verify.** Build the bindings on the box; in Python `assert np.array_equal` of `f2._f2_numpy()`
and `f2._vpair_numpy()` before vs after (or against a known f2-dir); run `tests/python/test_py_qpadm.py`
and peers — unchanged.

**Files.** `bindings/module.cpp`.

---

## C4 — `examples/` quick-start: minimal C++ **and** Python `read_f2 → qpadm → inspect`  ·  **P3 · Effort S**

**What's needed.** **No `examples/` dir exists** (confirmed: only `docs/`, `src/`, `include/`,
`bindings/`, `tests/`). Create two runnable quick-starts + a README. **Python is trivial** and fully
supported by the facade: `import steppe; f2 = steppe.read_f2(dir, device=0);
res = steppe.qpadm(f2, target=…, left=[…], right=[…]); print(res.weights)` (a pandas DataFrame
`[target,left,weight,se,z]`, `__init__.py:128`); `print(res.p)` (a float, `__init__.py:117`).
The **C++ quick-start must mirror `src/app/cmd_qpadm.cpp` `run_qpadm_command` (lines ~75–122)**
because there is **no header-only/CUDA-free way** to run a real qpAdm: `read_f2_dir` lives in the
app layer (`src/app/f2_dir_io.hpp`, returns the `F2Dir{f2, pop_labels}`), and `run_qpadm` needs
`device::Resources` from `device::build_resources(...)` + an uploaded `device::DeviceF2Blocks`
(`qpadm.hpp`). So the C++ example links the steppe app + `steppe::device` (CUDA) and **only builds
on the box**. Flow: `read_f2_dir(dir)` → resolve target/left/right names to int indices against
`pop_labels` (via `PopResolver`) → `build_resources(config.device())` → `upload_f2_blocks_to_device`
→ `run_qpadm(dev_f2, model, opts, resources)` → print `result.weight[]` and `result.p`. The committed
golden fixtures under `tests/reference/goldens` supply a **real f2-dir** to point both examples at
(honor the real-data-only rule — no synthetic toy tensor).

**Optimal end state.** `examples/` with `README.md` (one screen: what it does, how to get an f2-dir,
the **expected weights/p line for the chosen golden** so a reader can self-check),
`examples/python/quickstart_qpadm.py` (zero-arg-default call reproducing a known golden), and
`examples/cpp/quickstart_qpadm.cpp` compiled by `examples/CMakeLists.txt` behind a new opt-in option
`STEPPE_BUILD_EXAMPLES` (default **OFF**, mirroring `STEPPE_BUILD_CLI` in
`cmake/SteppeOptions.cmake:23`). The C++ target **reuses the same entry points `cmd_qpadm` uses**
(no duplicated compute) and shares its catch→exit-code idiom. Both print a weights table + the tail
`p`, and double as **living API canaries** that fail to compile/run if the public surface drifts.
Should showcase the C1 `Precision` factories and the C2 `f2_at` once those land.

**Steps.**
1. `mkdir examples/{python,cpp}`.
2. `examples/python/quickstart_qpadm.py`: `read_f2` a golden f2-dir, call `steppe.qpadm` with a real
   target/left/right from that golden, `print(res.weights)` and `print(f"p={res.p}")`.
3. `examples/cpp/quickstart_qpadm.cpp` mirroring `run_qpadm_command` (`read_f2_dir` → `PopResolver`
   name→index → `build_resources` → `upload_f2_blocks_to_device` → `run_qpadm` → print
   `result.weight[]`/`result.p`), with the same try/catch the CLI uses.
4. `examples/CMakeLists.txt`: `add_executable(quickstart_qpadm cpp/quickstart_qpadm.cpp)` linking the
   libs `cmd_qpadm` links (steppe app sources + `steppe::device` + `steppe::api`), guarded by
   `if(STEPPE_BUILD_EXAMPLES)`.
5. `option(STEPPE_BUILD_EXAMPLES "Build examples/" OFF)` in `cmake/SteppeOptions.cmake` +
   guarded `add_subdirectory(examples)` in the top `CMakeLists.txt` near the bindings/CLI block
   (`CMakeLists.txt:100–111`).
6. `examples/README.md` with exact run commands, the f2-dir to point at, and the expected weights/p
   so the reader can verify. **State that the local RTX 2070 cannot build the C++ one.**

**Parity-risk.** **None to goldens** — examples invoke the existing frozen `run_qpadm`/`read_f2`
entry points and add no compute. The only failure mode is the example not compiling/running, which
is a **desirable canary**. Guard: pin the example to a committed golden f2-dir and **assert nothing;
just print**.

**Verify.** Python: on the box, `python examples/python/quickstart_qpadm.py` against a golden f2-dir;
eyeball weights/p vs the value pinned in the README. C++: configure `-DSTEPPE_BUILD_EXAMPLES=ON` on
the box, build, run on the same f2-dir, confirm identical weights/p. Local cannot build the C++ one —
expected and stated in the README.

**Files.** `examples/python/quickstart_qpadm.py`, `examples/cpp/quickstart_qpadm.cpp`,
`examples/CMakeLists.txt`, `examples/README.md`, `CMakeLists.txt`, `cmake/SteppeOptions.cmake`,
`src/app/cmd_qpadm.cpp`, `src/app/f2_dir_io.hpp`.

---

## C5 — Generated API reference: Doxygen for the C++ headers + a Python docs surface  ·  **P3 · Effort S**

**What's needed.** **No docs infra exists** (confirmed: no Doxyfile, no `.doxy`, no `docs/api/`).
The headers in `include/steppe/*.hpp` are **already doxygen-style commented** (08 §8), so content
cost is near zero — only toolchain glue is missing. Add (a) a Doxygen config over `include/steppe/`
(optionally the app-public `f2_dir_io.hpp`) emitting HTML, wired as an opt-in CMake target gated on
`find_package(Doxygen)`; and (b) a Python docs surface over `bindings/steppe/__init__.py`. Recommend
**pdoc** for Python (near-zero config, reads existing docstrings) as the minimal; Sphinx+Breathe is
the heavier A+ follow-on, **not required** for this item.

**Optimal end state.** `cmake --build --target docs` runs Doxygen over `include/steppe` →
`docs/api/html/` with cross-referenced `QpAdmResult` / `run_qpadm` / `Precision` / `F2BlockTensor`
pages; a sibling `docs-python` target runs pdoc → `docs/api/python/`. `find_package(Doxygen)` makes
the target a **soft no-op when absent** so it never breaks a normal build. **Key property to
exploit:** Doxygen parses headers as text and needs **no compiler and no CUDA**, so this target
runs on the **local RTX 2070** box (unlike everything else) — call that out so it can be iterated
locally.

**Steps.**
1. `docs/Doxyfile.in`: `INPUT = include/steppe` (+ optionally `src/app/f2_dir_io.hpp`),
   `EXTRACT_ALL=YES`, `GENERATE_HTML=YES`, `GENERATE_LATEX=NO`, `OUTPUT_DIRECTORY` templated to the
   build dir, `PROJECT_NAME`/`PROJECT_NUMBER` from CMake.
2. `cmake/Docs.cmake`: `find_package(Doxygen QUIET)`; if found, `configure_file` the Doxyfile +
   `add_custom_target(docs COMMAND Doxygen::doxygen)`; `include()` it from the top `CMakeLists.txt`
   behind `option(STEPPE_BUILD_DOCS OFF)`.
3. A `docs-python` custom target invoking `pdoc -o docs/api/python bindings/steppe` (or document the
   one-liner) — reads the existing `__init__.py` docstrings; no new content.
4. Single-source `PROJECT_NUMBER` from `project(VERSION)` (ties into the §3-Med "single-source the
   version" item) so the docs version never drifts.
5. A short `docs/api/README.md` (or README section) with the build command + the note that Doxygen
   runs locally without CUDA.

**Parity-risk.** **None** — documentation generation only; touches no source, no compute, no golden.

**Verify.** Run `doxygen` on the generated Doxyfile **locally** (no box needed); open
`docs/api/html/index.html` and confirm `Precision`, `run_qpadm`, `QpAdmResult`, `F2BlockTensor`
pages render with member docs. For Python: `pdoc bindings/steppe` and confirm
`read_f2`/`qpadm`/`QpAdmResult.weights` appear. Optionally add a CI doc-build lane later (pairs with
the §3-Med host-only CI lane).

**Files.** `docs/Doxyfile.in`, `cmake/Docs.cmake`, `CMakeLists.txt`, `cmake/SteppeOptions.cmake`,
`include/steppe/qpadm.hpp`, `bindings/steppe/__init__.py`.

---

## C6 — Unify the precision-token vocabulary across CLI / Python / emit (add `emu32` to Python)  ·  **P2 · Effort XS (sibling of C1)**

> *(cross-cut gap **G6** — from `docs/kimiactions/04-crosscut-vs-kimi.md` §3/§5; X4-2. The missing
> string-token sibling of C1 — C1 fixes the C++ `Precision{Kind,bits}` struct ergonomics; this fixes
> the **user-facing token strings** that don't agree across the three surfaces.)*

**What's needed.** The precision token vocabulary splits three ways. **CLI** (`config_builder.cpp:268–280`)
accepts `emu40|emu | emu32 | fp64 | tf32`. **Python** (`module.cpp:170–178`, `parse_precision`)
accepts `fp64|native | emulated_fp64|emu | tf32` — **no `emu32`, so the 32-bit mantissa is
UNSELECTABLE from Python** (verified at HEAD). The **emit** token is a third spelling (`emu|tf32|fp64`).
So the same precision is named differently depending on the surface, and one mode is reachable from
the CLI but not the facade. C1 does **not** address this — it's about the C++ struct, not the strings.

**Optimal end state.** One accepted token set + one emitted token, consistent across CLI, Python, and
emit. Concretely: **add `emu32`** (and the `emulated_fp64_32` long form if matching the Python style)
to the Python `parse_precision` acceptor so 32-bit emulation is selectable; keep `native` as a
**documented alias** of `fp64`; document the canonical set in `cli-bindings.md` and the facade
docstrings. The emitted token agrees with one of the accepted spellings so a round-trip
(emit → re-parse) is lossless.

**Steps.**
1. `module.cpp:170–178` `parse_precision`: add an `emu32` branch (and the long form if used) →
   `prec.kind = EmulatedFp64; prec.mantissa_bits = 32;` mirroring the CLI's `emu32` at
   `config_builder.cpp:271`. Note the current Python path never sets `mantissa_bits` per-token — the
   `emu32` branch must set it explicitly (the CLI does).
2. Reconcile the three spellings: pick the canonical accepted set (recommend the CLI's
   `emu40|emu32|fp64|tf32` plus the documented aliases `emu`=`emu40`, `native`=`fp64`,
   `emulated_fp64`=`emu40`) and ensure the **emit** token is one of them.
3. Document the canonical token set + aliases in `docs/cli-bindings.md` §4.1 and the relevant facade
   docstrings (`bindings/steppe/__init__.py`).
4. (Optional) a static cross-check table / test asserting every CLI-accepted token has a Python
   equivalent and vice-versa.

**Parity-risk.** **None to existing goldens** — this only *adds* an accepted Python token (`emu32`)
and aligns spellings; every previously-accepted token still maps to the same `{kind,bits}`, so no
existing run changes precision and no golden moves. The **only** way to move a value is mis-mapping a
token to the wrong mantissa — guarded by mirroring the CLI's exact `emu32 → {EmulatedFp64, 32}`
mapping and a round-trip emit/parse check. (Selecting `emu32` from Python is a *new* capability, not a
change to any committed golden, which all pin their own precision.)

**Verify.** From Python, `steppe.qpadm(..., precision="emu32")` now resolves to `{EmulatedFp64, 32}`
(previously raised); `native` still aliases `fp64`; the emitted precision token re-parses to the same
`Precision`. CLI tokens unchanged. The qpAdm goldens (which pin their precision) are bit-identical.

**Files.** `bindings/module.cpp`, `src/core/config/config_builder.cpp` (reconcile only if the emit
token spelling moves), `docs/cli-bindings.md`, `bindings/steppe/__init__.py`.

---

## C7 — `dates()` typed result (`DatesResult` class), not a bare untyped dict  ·  **P3 · Effort XS**

> *(cross-cut gap **G10** — from `docs/kimiactions/04-crosscut-vs-kimi.md` §3/§5; X4-4. The one
> ergonomic outlier in the Python facade — sibling of C1/C2/C3; the C++ `DatesResult` value type
> already exists in `dates.hpp`.)*

**What's needed.** Every other facade entry returns a typed result class (`QpAdmResult`,
`QpWaveResult`, `F4Result`, `F3Result`, `F4RatioResult`, `QpGraphResult`, `QpGraphSearchResult` —
all in `bindings/steppe/__init__.py`), but `dates()` (`__init__.py:855–881`) returns the **bare
dict** straight from `_core.run_dates(...)` with **no return type hint** and no wrapper class. Its
docstring even says *"Returns a dict ``{target, source1, source2, date_gen, se, fit_error_sd,
curve_cm, curve_corr, status}``"*. This is the lone ergonomic outlier; the C++ side already has a
`DatesResult` value type (`dates.hpp`).

**Optimal end state.** A `DatesResult` facade class (matching the sibling result classes' shape:
attribute accessors over the underlying dict, an optional `.table`/DataFrame for the
`curve_cm`/`curve_corr` decay curve, a `__repr__`), and `dates()` annotated `-> DatesResult` (or
`-> DatesResult | dict` if a back-compat dict escape is kept). The keys exposed today
(`date_gen`, `se`, `fit_error_sd`, `curve_cm`, `curve_corr`, `status`, …) become typed attributes so
`help(steppe.dates)` and IDE completion match the other tools.

**Steps.**
1. Add a `class DatesResult:` near the other result classes (`__init__.py`), wrapping the
   `run_dates` dict; expose `date_gen`, `se`, `fit_error_sd`, `status`, `target`/`source1`/`source2`,
   and the `curve_cm`/`curve_corr` arrays (optionally a `.curve` DataFrame), with a `__repr__`.
2. Change `dates()` to `res = DatesResult(_core.run_dates(...)); return res` and annotate
   `-> DatesResult`; update the docstring from "Returns a dict" to "Returns a :class:`DatesResult`".
3. Mirror the existing classes' `as_dataframe`/`.table` convention if dates has a natural tabular
   view (the binned decay curve).
4. Keep the underlying dict reachable (e.g. `res.raw` or the curve accessors) so nothing that
   consumed the dict breaks.

**Parity-risk.** **None — pure Python facade shaping, golden-neutral.** No compiled code, no compute,
no statistic stream; `_core.run_dates` is unchanged, so the numbers are identical — only the Python
return *type* gains structure. The only failure mode is a downstream caller that indexed the dict
(`res["date_gen"]`) — guarded by keeping a dict-style accessor / `.raw` for back-compat and by the
`test_py_*` dates test.

**Verify.** `python -c "import steppe; help(steppe.dates)"` shows `-> DatesResult`; a dates run on
the box yields `res.date_gen`/`res.se` equal to the previous `dict["date_gen"]`/`["se"]`; the
committed dates parity values are unchanged (compute untouched); `tests/python` dates test passes.

**Files.** `bindings/steppe/__init__.py`.

---

## C8 — Split the 1,170-LOC `bindings/module.cpp` single TU into per-tool `register_*` units  ·  **P3 · Effort S**

> *(cross-cut gap **G12** — from `docs/kimiactions/04-crosscut-vs-kimi.md` §3/§5; X3-#6. Untracked
> TU split, placed next to C3 which already edits `module.cpp`. Kimi reviewed `module.cpp` but flagged
> only the f2 loop / exception mapping, not the TU split.)*

**What's needed.** `bindings/module.cpp` is a single **1,170-LOC** translation unit (verified at
HEAD) that binds every tool (qpAdm, qpWave, f3, f4, f4ratio, dstat, qpfstats, qpgraph, dates, the
F2Handle export) in one `PYBIND11_MODULE`. It is the largest, most-edited bindings file and a merge
hotspot; a reader must scroll the whole TU to find one tool's binding.

**Optimal end state.** Per-tool binding TUs (`bind_qpadm.cpp`, `bind_dates.cpp`, `bind_fstats.cpp`,
`bind_qpgraph.cpp`, `bind_f2handle.cpp`, …) each exposing a `void register_<tool>(py::module_& m)`;
`module.cpp` becomes a thin assembler whose `PYBIND11_MODULE` body just calls
`register_qpadm(m); register_dates(m); …`. Shared helpers (`parse_precision`, `f2_to_numpy`, the
result-struct converters) move to a small `bindings/internal/…` header so each TU reuses them.
`bindings/CMakeLists.txt` compiles the new TUs into the `_core` target.

**Steps.**
1. Carve shared helpers (`parse_precision`, `raise_value`, `f2_to_numpy`, the resident-f2 fit
   prologue/epilogue, result converters) into `bindings/internal/bind_common.hpp` (+ `.cpp` if
   needed).
2. Move each tool's binding block into its own `bind_<tool>.cpp` exposing
   `void register_<tool>(py::module_&)`. **Mechanical move** — copy the existing binding code
   verbatim into the function body; do not rewrite signatures or change the exposed API.
3. Reduce `module.cpp` to the `PYBIND11_MODULE` shell that calls each `register_*` in the current
   order (preserve registration order in case any binding depends on a prior type being registered).
4. Update `bindings/CMakeLists.txt` source list to add the new TUs.
5. Coordinate with **C3** (`f2_to_numpy` → `memcpy`): land C3's one-line change either before this
   split or in the moved `bind_common` home, not both.

**Parity-risk.** **None — mechanical code-move, golden-neutral.** The bound surface (function names,
signatures, returned struct shapes) is preserved byte-for-byte; no compute, no math, no statistic
stream. The Python import + every `tests/python` test is the guard. Only failure modes: a changed
registration order that breaks a type dependency (guarded by preserving order in step 3) or a missed
helper (a **compile/link error**, caught by the build).

**Verify.** `_core` builds on box5090; `import steppe` succeeds; `dir(steppe._core)` lists the same
symbols as before; `tests/python/test_py_qpadm.py` + peers pass unchanged; `wc -l bindings/module.cpp`
drops from 1170 to a thin shell.

**Files.** `bindings/module.cpp`, new `bindings/bind_*.cpp`, `bindings/internal/bind_common.hpp`,
`bindings/CMakeLists.txt`.

---

## C9 — `requires std::is_trivially_copyable_v<T>` on the two unconstrained buffer templates  ·  **P3 · Effort XS (optional; joins the C1/C2/C3 header-sugar batch)**

> *(cross-cut gap **G17** — from `docs/kimiactions/04-crosscut-vs-kimi.md` §3/§5; X5-2. Marginal by
> X5's own rating — fold into the header-sugar commit, not a standalone task.)*

**What's needed.** Two device-buffer class templates are unconstrained on `T`:
`DeviceBuffer<T>` (`src/device/cuda/device_buffer.cuh:42` `template <class T>`) and `PinnedBuffer<T>`
(`src/device/cuda/pinned_buffer.cuh:73` `template <class T>`). Both `cudaMemcpy` raw bytes of `T` to
device / page-locked host memory, which is only valid for trivially-copyable `T` — but nothing
*states* that contract, so a non-trivially-copyable `T` would compile and then silently corrupt. Add
a C++20 `requires std::is_trivially_copyable_v<T>` constraint so the contract is compiler-enforced and
self-documenting.

**Optimal end state.** Both templates declare
`template <class T> requires std::is_trivially_copyable_v<T>` (with `#include <type_traits>`), turning
a misuse into a clear compile-time diagnostic at the instantiation site instead of UB. All current
instantiations (`double`, `float`, `int`, the POD layout structs) already satisfy it, so nothing that
compiles today stops compiling.

**Steps.**
1. `device_buffer.cuh`: add `requires std::is_trivially_copyable_v<T>` to the `DeviceBuffer` template
   header (`:42`); `#include <type_traits>` if not already present.
2. `pinned_buffer.cuh`: same on the `PinnedBuffer` template (`:73`).
3. Build to confirm every existing instantiation still satisfies the constraint (it should — all are
   PODs / arithmetic types).

**Parity-risk.** **None — a compile-time constraint only, golden-neutral.** It changes no runtime
code, no allocation, no `cudaMemcpy`, no math; it only *rejects* a misuse that does not exist today.
Every current `T` is trivially-copyable, so codegen is byte-identical and no golden moves. Only
failure mode: an existing `T` that is *not* trivially-copyable would now fail to compile — which is
the **desired** surfacing (it would have been a latent UB), and verified absent by a clean build.

**Verify.** Full build on box5090 links unchanged (every instantiation satisfies the constraint);
a deliberate test instantiation with a non-trivially-copyable type fails to compile with a clear
`requires`-clause diagnostic; the f2/qpAdm goldens are bit-identical.

**Files.** `src/device/cuda/device_buffer.cuh`, `src/device/cuda/pinned_buffer.cuh`.

---

# Recommended sequence

Independence is high; most items are standalone tiny commits. Suggested order maximizes
iteration-unblock and de-risks the separate §3-HIGH CI item first. **The two cross-cut "do-first"
items lead:** D1 (the public-surface comment scrub — highest first-impression signal) and B6 (the
`CHECK_KERNEL` swap — the compute-sanitizer multiplier).

1. **D1 — comment hygiene, public-surface first** *(cross-cut G1)* — highest first-impression signal,
   golden-neutral; do before the `02` clang-format pass entrenches the tokens. Lock with the `02`
   grep-gate.
2. **B6 — `CHECK_KERNEL` swap** *(cross-cut G4)* — do before the `02` §3-High compute-sanitizer pass
   so it keeps launch-site attribution.
3. **A1 — `STEPPE_WERROR`/ccache** (smallest, unblocks faster local box iteration).
4. **A2 — CTest labels** (the literal `ctest -LE gpu` enabler for the HIGH CI lane).
5. **C1 + C2 + C3 + C9 — header sugar batch** (one commit: Precision factories + `f2_at` + memcpy +
   the trivially-copyable constraint; all zero parity risk). *(C9 = cross-cut G17.)*
6. **B2 + B3 + B4 — XS robustness** (each its own tiny commit; B2/B4 build on the local host lane).
7. **B7 + B8 + B9 — dead-virtual removal + device-convention re-align + confusable rename** *(cross-cut
   G9/G11/G13)* — three tiny golden-neutral commits, box-only build.
8. **A7 — dead DeviceConfig knobs** *(cross-cut G8)* and **C7 — `dates()` typed result** *(cross-cut
   G10)* (each its own tiny commit; A7 builds on the host config lane).
9. **C6 — precision-token unification (+`emu32` to Python)** *(cross-cut G6)* — sibling of C1.
10. **B1 — NVTX** (profiling unlock; standalone, box-only build).
11. **A3 — pytest/ruff/mypy** (host-only; feeds the CI lane).
12. **B5 — workspace pool + staging cap** (one commit, box-only; frame as hygiene).
13. **C4 — examples** (after C1/C2 so they showcase the new sugar).
14. **C8 — `module.cpp` per-tool TU split** *(cross-cut G12)* — coordinate with C3 (the `memcpy`
    change lands in the moved `bind_common`, not twice).
15. **C5 — Doxygen/pdoc** (iterate locally; single-source the version with the §3-Med item).
16. **A6 — record the `src/core/qpadm/` misnomer decision** *(cross-cut G7)* — DEFER-and-document is
    the XS floor; a rename/split is a mechanical commit gated by the goldens. **A8 —
    `target_compile_features` on the two STATIC compute libs** *(cross-cut G16)* — 2-line CMake add.
17. **A4 — meta.json fields** + **A5 — `regenerate_goldens.sh` / delete `build_m0.sh`** (do the two
    architecture.md doc-sync edits together; A5 is the only golden-touching item — run it last, by
    hand, with the parity re-run).

**Doc-sync batching:** A4's reproducibility-block edit, A5's `architecture.md:115` scrub, A6's
module-map note (cross-cut G7), and C5's single-sourced version all touch `docs/architecture.md` —
batch the doc edits into one pass with the two §3-Med architecture doc-sync items.

# One-glance priority table

| # | Item | Cluster | Priority | Effort | Parity-risk | Builds where |
|---|------|---------|----------|--------|-------------|--------------|
| **D1** | **Comment hygiene, public-surface first** *(cross-cut G1)* | **Hygiene (do-first)** | **P1** | S (breadth) | None | box + local |
| A1 | `STEPPE_WERROR=OFF` + ccache | Build/CI | **P1** | S | None | box + local |
| A2 | CTest labels (unit/ref/cli/python + gpu/slow) | Build/CI | **P1** | S | None | box |
| B1 | Wire NVTX behind `STEPPE_NVTX` | Robustness | **P1** | S | None | box only |
| **B6** | **14 bare cudaGetLastError → `CHECK_KERNEL`** *(cross-cut G4)* | Robustness | **P1** | S (breadth) | None | box only |
| A3 | pytest + ruff + mypy (facade) | Build/CI | P2 | S | None | host-only |
| A4 | `meta_schema_version` + `pops_sha256` | Build/CI | P2 | S | None | box (app) |
| **A6** | **Record `src/core/qpadm/` misnomer decision** *(cross-cut G7)* | Build/CI | P2 | S (decision XS) | None | box |
| **A7** | **Dead DeviceConfig knobs** *(cross-cut G8)* | Build/CI | P2 | XS | None | host/box |
| B2 | Forward `error_code::message()` in f2_dir_io | Robustness | P2 | XS | None | host-only |
| B3 | geno-hash thread try/catch + exception_ptr | Robustness | P2 | XS | None | box (app) |
| **B7** | **Remove dead `rank_test` virtual** *(cross-cut G9)* | Robustness | P2 | XS | None | box only |
| C1 | `Precision` named factories | API/DX | P2 | XS | None* | box + local |
| C2 | `F2BlockTensor::f2_at` accessor | API/DX | P2 | XS | None | box + local |
| C3 | `f2_to_numpy` → `std::memcpy` | API/DX | P2 | XS | None | box only |
| **C6** | **Precision-token unification (+`emu32`)** *(cross-cut G6)* | API/DX | P2 | XS | None | box (bindings) |
| A5 | `regenerate_goldens.sh` + delete `build_m0.sh` | Build/CI | P3 | S | **Golden (human-run, preflight-guarded)** | box |
| **A8** | **`cxx_std_20` on steppe_core+steppe_device** *(cross-cut G16)* | Build/CI | P3 | XS | None | box + local |
| B4 | Drop `mutable` on `ConfigBuilder::error_message_` | Robustness | P3 | XS | None | host-only |
| B5 | Pool cuSOLVER workspace + cap pinned staging | Robustness | P3 | S (hygiene) | None | box only |
| **B8** | **Device-layer convention re-align** *(cross-cut G11)* | Robustness | P3 | XS | None | box only |
| **B9** | **Rename confusable `f2_block(s)_kernel`** *(cross-cut G13)* | Robustness | P3 | XS | None | box only |
| C4 | `examples/` C++ + Python quick-start | API/DX | P3 | S | None | py:box, cpp:box |
| C5 | Doxygen + pdoc API reference | API/DX | P3 | S | None | **local** |
| **C7** | **`dates()` typed `DatesResult`** *(cross-cut G10)* | API/DX | P3 | XS | None | host (facade) |
| **C8** | **`module.cpp` per-tool TU split** *(cross-cut G12)* | API/DX | P3 | S | None | box only |
| **C9** | **`is_trivially_copyable` on buffer templates** *(cross-cut G17)* | API/DX | P3 | XS | None | box only |

\* C1's only golden vector is a *migrated* construction site silently changing a mantissa — guarded
by keeping migration in its own commit with diff-verified identical `{kind,bits}`.

**Cross-cut provenance:** the **12 bold rows** above (D1, B6, A6, A7, B7, C6, A8, B8, B9, C7, C8, C9)
were folded in from our consolidated cross-cutting review (`docs/release_cleanup/crosscutting/X1–X7`)
via `docs/kimiactions/04-crosscut-vs-kimi.md` §3/§5 — gaps the X-review caught that the
`a2f9d64..HEAD` campaign neither fixed nor planned. All twelve are comment/doc/structure/dead-code and
**golden-neutral by construction**.

---

**Bottom line.** Twenty-eight XS–S items, **twenty-seven behavior-neutral by construction** and one
(`regenerate_goldens.sh`) golden-adjacent only on deliberate human invocation behind a hard
version-preflight + mandatory parity re-run. The §12 PARITY LAW (frozen math names, deterministic
reductions, single statistic stream) is untouched across the entire bucket — the twelve cross-cut
additions (D1, A6–A8, B6–B9, C6–C9) are all comment/doc/structure/dead-code and move no golden. The
frozen goldens (`golden_fit0`/`fit1_NRBIG`/`rot`) remain the guard for the box-only compute-adjacent
items (B1, B5, B6–B9, C8, C9); the host config/io/python unit tests guard the rest. The D1 public
comment scrub is the single highest-signal, lowest-risk item — do it first, and lock it with the
`02-ci-plan.md` grep-gate.

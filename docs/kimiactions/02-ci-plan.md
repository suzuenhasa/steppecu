# steppe — Proper CI/CD Plan (host + GPU, phased)

> Decision-ready action plan. Lead engineer, for pickup later. Cites real `file:line`
> at HEAD (`phase2-fit-engine`), not the Kimi snapshot. Every item carries
> *What's needed / Optimal end state / Steps / Effort / Parity-risk / Verify / Files.*

---

## 0. Framing — why this plan, and what "done" means

steppe is a **parity product**. The acceptance gate is not "tests pass" in the usual
sense; it is "the GPU produces, on **sm_120 / CUDA 13 / driver ≥ 580**, the *same*
numbers AT2 produced, to tolerance, on the committed goldens and real AADR." The
core IP — fixed-slice Ozaki **emulated FP64** (~40-bit, the default per
`STEPPE_HAVE_EMU_TUNING`, `cmake/SteppeOptions.cmake:34`) — **downgrades off
Blackwell** and is meaningless on any other arch. That single fact dominates every CI
decision below.

The source-hygiene campaign (`a2f9d64..HEAD`) is **finished** — formatting, dead code,
IWYU, naming. This plan is the *next* layer: there is **no CI at all today**
(`.github/` does not exist — confirmed), no CTest LABELS (0 of 74 `add_test`
registrations are labelled — confirmed), and a CUDA-free host **cannot even
configure** the project (four hard `find_package`/`LANGUAGES CUDA` requirements in the
top `CMakeLists.txt`). Everything ships through a manual `ssh box5090 + rsync` loop
(`scripts/box_bringup.sh`, `docs/BOX-RUNBOOK.md`). That is the gap.

**The shape of the answer is a HYBRID:**

- **Phase 0 — a free, fast, GPU-free host lane** on `ubuntu-latest` that catches ~90 %
  of regressions (hygiene, seams, host-orchestration logic, and the CpuBackend
  native-FP64 oracle vs golden) in minutes, on every PR, for $0.
- **Phase 1 — an on-demand / nightly GPU parity lane** whose engine is an **ephemeral
  vast.ai JIT self-hosted runner** (the only cost-viable source of sm_120) that boots,
  registers, runs one job, and self-destructs. This is the **only** lane that proves
  production numerics.
- **Phase 2 — sanitizers, nightly AADR tier, release tagging gate, and the §12
  determinism CI assertion** — the proof-hardening layer.

**The parity caveat, stated once and threaded throughout:** a green host lane is a
**smoke/seam gate, not a parity gate**. The CpuBackend is the *native*-FP64 oracle; the
shipping path is *emulated* FP64 on sm_120. Only the GPU lane proves what users run.

> **Cross-cut additions (provenance: `docs/kimiactions/04-crosscut-vs-kimi.md` §3/§5).**
> Two items below originate in **our own X1–X7 cross-cutting review** rather than the Kimi
> assessment, and are folded in here so the cross-cut work is not lost: **A9** — the
> provenance-token regression grep-gate (cross-cut gap **G1**, CI-half) — and the **G4**
> bare-`cudaGetLastError` prerequisite note, folded into the new **B9** compute-sanitizer
> lane item. Both are **comment / structure / robustness only — golden-neutral, parity-risk
> none** — and each is tagged inline where it appears. The one-time *scrub* (G1) and the
> 14-site *conversion* (G4) are the matching `03-low-polish.md` items; the two additions
> here are the **CI lane half** (a regression guard + the lane that consumes the fix).

---

## 1. Constraints that scope "optimal"

| Constraint | Consequence for CI |
|---|---|
| **§12 PARITY LAW** (frozen math names, fixed-order reductions, single stat stream — no 2nd stream / no NCCL) | CI may **assert** the law (Item B5) but every formatter/linter runs **check-only / `--dry-run`** — never `--fix`, never auto-commit. A 17-digit literal, `int P` / `long M` kernel ABI, or `mantissa_bits=40` must never be rewritten by a bot. |
| **Emulated FP64 default + native-FP64 carve-out**; `double` is intentional | Parity can only be validated on sm_120. Hosted/cloud GPU runners (T4/A100/H200) are **useless** for the gate. |
| **CpuBackend = native-FP64 parity ORACLE, dev/test only** | It is a CUDA-built target (`src/device/cpu/cpu_backend.cpp`, `src/device/CMakeLists.txt:43`). It runs GPU-free in **FAST mode** when `CUDA_VISIBLE_DEVICES=""` (`BOX-RUNBOOK.md:76`) — a real no-GPU gate — but is **not** the production numeric path. |
| **Nothing builds locally** (RTX 2070 / CUDA 11.8, no FP64-emu); box is **ephemeral** vast.ai | The GPU lane must dial **out** to GitHub (JIT runner), not chase an ephemeral SSH IP. AADR (~30 GB) lives only on the box and is re-materialized per run. |
| **Single-GPU shipping product** (multi-GPU parked) | The GPU lane is single-GPU. No P2P/NCCL coverage is required for the gate. |
| **Goldens committed; AADR not** | Most parity tests read committed fixtures (`tests/reference/goldens/at2/fixtures/f2_*.bin`) and need **zero** AADR. Only decode / format-reader / f2-blocks tests need the 30 GB. → **data tiering.** |

**Out of scope (assessment §4 rejected):** 2nd stat stream, NCCL, multi-GPU gate,
excise-oracle, `steppe::Index`, JsonWriter reshape, capability pure-virtual, kernel
fuzzing, mock/no-GPU *shippable* backend. The host lane builds **no backend at all** —
it tests CUDA-free *logic*; it does not resurrect a mock backend.

---

## 2. Phased rollout at a glance

| Phase | Lane | Trigger | Runs on | Proves | Cost |
|---|---|---|---|---|---|
| **0** | Host CUDA-free | every PR + push | `ubuntu-latest` (no GPU, no toolkit) | hygiene, seams, §4 layering, **provenance-token regression (A9)**, CUDA-free unit + host-orchestration logic, CpuBackend-vs-golden (toolkit sub-job) | $0 |
| **1** | GPU parity | nightly cron + `workflow_dispatch` + maintainer `gpu-ci` label | ephemeral vast.ai sm_120 (JIT runner) | **emulated-FP64 sm_120 parity** vs committed goldens | ~1 GPU-hr/night |
| **2** | Hardening | nightly / pre-release | host (ASan/UBSan) + GPU (**compute-sanitizer (B9)**, AADR tier, §12 assertion) | memory safety, full-data parity, release readiness | low |

The **honest ceiling**: of 74 ctest registrations, **11 are host-coverable now**, **~21
after the optional refactor (Item A8)**, and the remaining **~53 are nvcc/GPU/box-bound**
(26 `.cu` reference parity tests, 16 `cli_*`, `py_qpadm`, AADR-keyed tests). The host
lane is a *fast structural+unit* gate; **parity acceptance stays the box's job.**

---

# Cluster A — Host-only, CUDA-free PR lane (Phase 0)

> Priority: **HIGH** (assessment §3/§6 step 2). The cheap, honest first gate. Item A1 is
> the hard structural prerequisite for A2/A3/A8. The fast text gates (A4 / A6 / **A9**) are
> independent and land day one.

---

## A1 — `STEPPE_ENABLE_CUDA` option + guard the CUDA-mandatory top CMake  *(HIGH — structural enabler)*

**What's needed.** Today a CUDA-free host **cannot configure**. `CMakeLists.txt`
hard-requires the toolkit at four points (verified):
- `:20-23` `project(steppe … LANGUAGES CXX CUDA)` — compiler detection fails with no nvcc,
- `:41` `include(CUDAArch)`,
- `:56` `find_package(CUDAToolkit REQUIRED)`,
- `:74` `add_subdirectory(src/device)` (the `.cu` target).

Compounding it, `:73` `add_subdirectory(src/core)` builds `steppe_core`, which links
`PRIVATE steppe::device` (`src/core/CMakeLists.txt:152`) — so `steppe_core` is
unconfigurable without the device target existing. `cmake/SteppeOptions.cmake` has
`BUILD_TESTS/PYTHON/CLI/EMU_TUNING/NVTX/SANITIZER/COMPRESS_FATBIN` but **no
ENABLE_CUDA**. Net: zero host build path.

**Optimal end state.** A `STEPPE_ENABLE_CUDA` cache option, **default ON**.
- **ON** ⇒ byte-identical to today; `cmake --preset ci` on the box yields the same
  `build.ninja`, same `sm_120` arch, same targets. Production path untouched.
- **OFF** ⇒ `project()` declares `LANGUAGES CXX` only; CUDA is `enable_language`d
  conditionally; `CUDAArch` / `find_package(CUDAToolkit)` / `CMAKE_CUDA_*` skipped;
  `src/device` + `src/core` not added. Configure resolves with only `steppe::api`
  (INTERFACE), `steppe::io` (CUDA-free leaf), `steppe::core_internal` (INTERFACE),
  `steppe::warnings`. A bare
  `cmake -S . -B build/host -DSTEPPE_ENABLE_CUDA=OFF -DSTEPPE_BUILD_TESTS=ON` succeeds on
  stock `ubuntu-latest` with **no toolkit installed**.

**Steps.**
1. Resolve the option **before** `project()`: add
   `option(STEPPE_ENABLE_CUDA "Build the CUDA device layer + GPU tests" ON)` near the top
   of `CMakeLists.txt` (above `:20`; `option()` works pre-`project()`), and mirror it in
   `SteppeOptions.cmake` for discoverability (note: `SteppeOptions` is currently
   `include()`d at `:40`, *after* `project()` — keep the authoritative `option()` call
   pre-`project()` and let the `SteppeOptions` entry be documentation/`include_guard`-safe).
2. Change `:23` to `LANGUAGES CXX`; immediately after, `if(STEPPE_ENABLE_CUDA)
   enable_language(CUDA) endif()`.
3. Gate the CUDA-only policy: `set(CMAKE_CUDA_STANDARD 20)` / `_REQUIRED` (`:27,:29` —
   harmless if left, cleaner gated), `include(CUDAArch)` (`:41`),
   `find_package(CUDAToolkit REQUIRED)` (`:56`) all behind `if(STEPPE_ENABLE_CUDA)`.
4. Gate `add_subdirectory(src/core)` (`:73`) and `add_subdirectory(src/device)` (`:74`)
   behind `if(STEPPE_ENABLE_CUDA)` for the first cut (host lane = api/io/core_internal).
   Expanding to a host slice of core is **A8**.
5. Leave `Threads` (`:64-65`) unconditional (host-safe). The `STEPPE_BUILD_CLI/PYTHON`
   guards already default OFF, so app/bindings/access/extract are naturally excluded
   host-only (`CMakeLists.txt:80-112`).
6. Pin `STEPPE_ENABLE_CUDA=ON` explicitly in the `ci` and `release` presets so a
   no-CUDA artifact can never ship by misconfiguration.

**Effort.** **S.**

**Parity-risk.** **Behavior-neutral on the box.** Default ON ⇒ production configure takes
the exact same path; the guard is a branch never executed on the GPU build. No kernel, no
golden, no compute touched. Residual risk (someone configures the box OFF and ships a
no-CUDA artifact) is closed by default-ON + the pinned-ON presets.

**Verify.** No-CUDA container: `cmake --preset host && cmake --build build/host` succeeds.
On the box: `cmake --preset ci` regenerates an identical target/arch set — diff a
`-- -n` dry-run, or diff the `CMakeCache.txt` CUDA entries, against a pre-change configure.

**Files.** `CMakeLists.txt`, `cmake/SteppeOptions.cmake`, `cmake/CUDAArch.cmake`,
`src/core/CMakeLists.txt`, `src/device/CMakeLists.txt`.

---

## A2 — Guard `tests/CMakeLists.txt` for the CUDA-free configure + add CTest LABELS  *(HIGH)*

**What's needed.** `tests/CMakeLists.txt` (1692 lines) unconditionally creates 26 `.cu`
test targets (first at `:41` `add_executable(test_f2_equivalence reference/…cu)`) needing
nvcc, plus host `.cpp` tests that link `steppe::core`/`steppe::device` and so transitively
need the CUDA build. With CUDA off, all fail at configure/link. **There are no LABELS**
(confirmed: 0 hits), so there is no clean selector — `ctest -R _unit$` is *unsafe* (it
matches device-linked `shard_plan_unit`, `validate_device_order_unit`,
`config_builder_unit`).

**Optimal end state.** A host-only configure registers **exactly the 11 truly-CUDA-free
tests**, each carrying a LABEL, so `ctest -L host` is stable even inside a full GPU build.
The 11 (verified at the cited `add_executable` lines — each links only
`{steppe::api, core_internal, io, warnings}` + optional GTest, includes only CUDA-free
headers):

| Test | line | What it guards |
|---|---|---|
| `f2_estimator_unit` | 435 | f2 estimator primitive |
| `decode_unit` | 453 | genotype decode primitive |
| `launch_config_unit` | 468 | launch math (+ NDEBUG death cases — see A3) |
| `config_unit` | 487 | RunConfig precedence |
| `backend_factory_unit` | 770 | factory shape |
| `backend_capabilities_unit` | 795 | capability probe shape |
| `f2_partials_validate_unit` | 1026 | partials validator |
| `vram_budget_unit` | 1193 | VRAM budget math |
| `filters_unit` | 1280 | SNP filters |
| `geno_reader_unit` | 1299 | GENO reader |
| `snp_reader_unit` | 1324 | SNP reader |

Note: `device/backend.hpp`, `shard_plan.hpp`, `vram_budget.hpp`, `backend_factory.hpp` are
**CUDA-free seam headers** — *including* them is fine; only `steppe::device`/CUDAToolkit
**linkage** is the disqualifier.

**Steps.**
1. Wrap every `.cu` `add_executable(... reference/*.cu)` + its
   `set_target_properties(... CUDA_SEPARABLE_COMPILATION)` + `add_test` in
   `if(STEPPE_ENABLE_CUDA)` (the 26 `.cu` blocks), and likewise the `steppe::core/device`
   -linked `.cpp` blocks: `config_builder_unit`, `model_search_core_unit`,
   `qpgraph_enumerate`, `f2_combine_unit`, `block_partition_unit`, `block_ranges_unit`,
   `f2_from_blocks_unit`, `shard_plan_unit`, `validate_device_order_unit`,
   `f2_blocks_multigpu_unit`. Keep `STEPPE_AADR_ROOT`/`STEPPE_AADR_SNP` cache vars inside
   the CUDA guard.
2. For the 11 host-pure tests:
   `set_tests_properties(<name> PROPERTIES LABELS "host;unit")`.
3. Label the rest forward-compat: `.cu` → `gpu;reference`; `cli_*` → `cli`; `py_qpadm` →
   `python`; the core/device-linked host-logic `.cpp` → `unit;needs-cuda-build` so A8 can
   pick them up; AADR-keyed → add `aadr`; benches stay `add_executable` (no `add_test`),
   or if registered, `slow`.
4. Optionally factor the 11 into a `tests/host/CMakeLists.txt` always-added subdir so the
   split is structural, not a sea of `if`-guards.

**Effort.** **M.**

**Parity-risk.** **None.** Config-gating + LABELS touch no test body and move no golden;
the 11 run byte-identical assertions. A wrong label is cosmetic.

**Verify.** `cmake --preset host` then `ctest -N -L host` lists **exactly 11**; on the box
`ctest -N` still enumerates all 74 (LABELS are additive). CI asserts
`test $(ctest -N -L host | grep -c 'Test #') -eq 11`.

**Files.** `tests/CMakeLists.txt`.

---

## A3 — Host CMake preset + GitHub Actions workflow on `ubuntu-latest`  *(HIGH)*

**What's needed.** No `.github/` exists. `CMakePresets.json` has only `base/dev/release/ci`
— all CUDA (`CMAKE_CUDA_ARCHITECTURES=120`, `STEPPE_HAVE_EMU_TUNING=ON`,
`CMakePresets.json:12-13`). No host preset, no runner.

**Optimal end state.** A `host` configure+build+test preset
(`STEPPE_ENABLE_CUDA=OFF, STEPPE_BUILD_TESTS=ON, CLI/PYTHON=OFF`, Ninja) and a
`host-ci.yml` on `ubuntu-latest` that configures, builds, and runs `ctest -L host` in a
few minutes with **no GPU and no toolkit**.

> **Build-type nuance (correct the assessment's "Release + ctest"):** use
> `CMAKE_BUILD_TYPE=RelWithDebInfo` (or Debug), **NOT Release**.
> `unit/test_launch_config.cpp` is the only one of the 11 with `STEPPE_ASSERT` /
> NDEBUG-guarded death cases; under Release/NDEBUG those compile out and the **§2
> fail-fast coverage silently vanishes**. Read "Release + ctest" as "non-NDEBUG host build."

**Steps.**
1. Add a `host` configurePreset (do **not** inherit `base` — it carries CUDA cache vars):
   Ninja, `CMAKE_BUILD_TYPE RelWithDebInfo`, `STEPPE_ENABLE_CUDA OFF`,
   `STEPPE_BUILD_TESTS ON`, `CMAKE_EXPORT_COMPILE_COMMANDS ON`; matching build + test
   presets (test preset filters `-L host`).
2. Write `.github/workflows/host-ci.yml`: trigger on `pull_request` + `push`; job
   `runs-on: ubuntu-latest`; steps = `checkout`,
   `apt-get install -y ninja-build clang-tidy clang-format` (+ optional `libgtest-dev`),
   `cmake --preset host`, `cmake --build --preset host`,
   `ctest --preset host --output-on-failure`.
3. Add the A4–A7 + A9 jobs (format / tidy / arch-grep / cmake-lint / provenance-token) as
   **separate, parallel, independently-failing** jobs.
4. **Resolve the `-Werror` × newer-compiler risk before flipping this to required.**

**Effort.** **M.**

**Parity-risk.** Infra only — no source, no golden. **Real operational risk:**
`steppe::warnings` forces `-Wall -Wextra -Werror` on host TUs
(`cmake/SteppeWarnings.cmake:18`). `ubuntu-latest`'s GCC is **newer than the box
compiler** and may emit *new* warnings on box-clean code, red-failing the lane on a false
signal. Mitigate: (a) **pin the runner compiler** (a fixed GCC/Clang version or a
container), and/or (b) add a `STEPPE_WERROR=OFF` escape hatch (the assessment's Low item)
and set it for the host job until the warning delta is triaged. Does **not** affect parity
(warnings ≠ compute). **Do not make this a required check until settled.**

**Verify.** PR opens → workflow green in < 5 min, 11/11 passing on a runner with no nvidia
driver. Manually break a host test → job red.

**Files.** `CMakePresets.json`, `.github/workflows/host-ci.yml`, `cmake/SteppeWarnings.cmake`.

---

## A4 — clang-format diff check job  *(MED — XS effort)*

**What's needed.** A `.clang-format` exists (repo root) and `.clang-tidy` notes format
runs "in pre-commit/CI", but **nothing enforces it**. Drift is unguarded.

**Optimal end state.** A CI step that fails on any unformatted first-party file, in
**check / `--dry-run` mode** (never auto-commit — a formatter must not silently rewrite a
parity-load-bearing literal such as a 17-digit constant or an `int P` / `long M` ABI
spelling). PR-scoped (diff) to stay fast and avoid a giant first-PR reformat.

**Steps.**
1. PR-diff form:
   `git diff --name-only origin/main... | grep -E '\.(cpp|hpp|cu|cuh|h)$' | xargs -r clang-format --dry-run --Werror`
   (FormatStyle `file` picks up `.clang-format`).
2. Full-tree variant for `main` pushes:
   `git ls-files '*.cpp' '*.hpp' '*.cu' '*.cuh' '*.h' | xargs clang-format --dry-run --Werror`.
3. **Pin the clang-format major version** (output drifts across versions) — install a
   fixed version matching the box.
4. Document the developer fixer: `clang-format -i <file>` (never `-i` in CI).

> **Note (interaction with A9):** clang-format here runs with **comment reflow OFF** — it
> reformats whitespace/braces but does **not** rewrite comment *content*, so it will never
> strip internal provenance tokens (`cleanup X-N`, `M4.5`, "the spike"). Stopping those from
> re-entering is a **separate grep step (A9)**, not a clang-format responsibility.

**Effort.** **XS.**

**Parity-risk.** **Behavior-neutral** with `--dry-run --Werror` (no file modified). The
only golden-moving path is auto-fix-and-commit — explicitly forbidden.

**Verify.** Misindent a line in a PR → step fails; format → green. Working tree clean
after the step (CI never writes back).

**Files.** `.clang-format`, `.github/workflows/host-ci.yml`.

---

## A5 — clang-tidy diff check (scoped to the host compile DB)  *(MED)*

**What's needed.** `.clang-tidy` exists (`WarningsAsErrors '*'`,
`HeaderFilterRegex '.*/(include/steppe|src)/.*'`) but is **never run**. It needs
`compile_commands.json`, which only the configured build produces.

**Optimal end state.** A CI step running clang-tidy over the PR's changed files **that
have a host compile entry**, using the host build's `compile_commands.json`. **Honest
scope:** the host configure compiles only the 11 host-pure TUs + the io/api/core_internal
headers they pull. So host-lane tidy covers the CUDA-free TUs and their headers
(`include/steppe`, `src/io`, `src/core/internal`, the CUDA-free `device/*.hpp` seams). TUs
that exist only in the CUDA graph (`src/core/*.cpp` orchestration, `src/device/*.cu`) have
no host compile entry and are **deferred to the GPU/box lane** — a documented coverage
boundary, not a gap to paper over.

**Steps.**
1. After `cmake --preset host` (exports `compile_commands.json` to `build/host`), run
   `clang-tidy -p build/host <changed files present as keys in the DB>`.
2. **Skip-with-note** any changed file absent from the host DB (a `.cu`, a core-only
   `.cpp`) so the step never errors on "no compile command"; emit a "deferred to box lane"
   line.
3. **Pin** the clang-tidy version (checks evolve). `WarningsAsErrors '*'` makes it strict
   by construction.
4. Optional: weekly full-tree host-subset tidy on a schedule for drift detection.

**Effort.** **S.**

**Parity-risk.** **Check-only** — `clang-tidy` without `--fix`. Do **not** pass `--fix` in
CI: an auto-applied modernize/readability fix could rewrite parity-sensitive code.

**Verify.** Add a tidy-flagged construct (e.g. implicit narrowing) in a host TU → step
fails; a changed `.cu` in the same PR is reported "deferred (no host compile entry)" and
does **not** fail the host job.

**Files.** `.clang-tidy`, `.github/workflows/host-ci.yml`.

---

## A6 — Layering / arch-grep gate (the §4 CUDA-isolation law as a fast text check)  *(MED — XS effort)*

**What's needed.** `architecture.md §4` / the `CMakeLists.txt:6-9` header state
core/api/io/tests "physically cannot include a CUDA header" — enforced today **only
structurally** (`CUDA::*` is PRIVATE to `steppe_device`). That structural proof requires a
full CUDA build to trigger. The host lane wants a cheap, GPU-free, millisecond assertion
of the same law that catches a leak in review.

**Optimal end state.** A `scripts/arch_grep_gate.sh` (wired as a CI step) that greps the
layer-isolated trees for any `#include` of a CUDA **toolkit** header
(`cuda_runtime`, `cuda.h`, `cublas*`, `cusolver*`, `cufft*`, `curand*`, `cuda_fp16`) and
fails on a hit. Scopes the public + host-isolated surfaces: `include/`, `src/core/`
(orchestration `.cpp/.hpp` — must reach the GPU only via the `device/backend.hpp` seam),
`src/io/` (the CUDA-free leaf), `tests/unit/`, `tests/cli/`. Excludes `src/device/` (CUDA
allowed) and `reference/*.cu` (allowed). **Must match real `#include` lines only** — the
strings appear in *comments* (e.g. `tests/unit/test_config.cpp:20`), so anchor on
`^\s*#\s*include`.

**Steps.**
1. The script:
   `grep -rEn '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"](cuda_runtime|cuda\.h|cublas|cusolver|cufft|curand|cuda_fp16)' include/ src/core/ src/io/ tests/unit/ tests/cli/ --include='*.hpp' --include='*.cpp' --include='*.h'`
   ; any match → print + `exit 1`.
2. Second assertion: the `device/*.hpp` **seam** headers consumed by core/tests
   (`backend.hpp`, `backend_factory.hpp`, `shard_plan.hpp`, `vram_budget.hpp`) are
   themselves CUDA-free (same grep over those files) — these are what host TUs legitimately
   include.
3. Wire as a standalone fast CI job (no build); also runnable in pre-commit.
4. Document the allowlist (`src/device`, `*.cu`, `*.cuh`) inline so the boundary is explicit.

**Effort.** **XS.**

**Parity-risk.** **None** — read-only grep, no compile, no compute. Encodes an existing
invariant; cannot move a golden.

**Verify.** Add `#include <cublas_v2.h>` to a `src/core/*.cpp` → gate fails with
`file:line`; remove → green. Confirm it does **not** flag the comment occurrences in
`test_config.cpp` (anchored on the include directive).

**Files.** `scripts/arch_grep_gate.sh`, `.github/workflows/host-ci.yml`.

---

## A7 — cmake-lint (`cmake-format --check`) over the build system  *(LOW)*

**What's needed.** No CMake linter config exists (confirmed: no `.cmake-format.yaml`). The
CMake surface is large and hand-maintained (top `CMakeLists.txt`, `cmake/*.cmake`, the
1692-line `tests/CMakeLists.txt`, per-target lists) — exactly what A1/A2 edit.

**Optimal end state.** A `.cmake-format.yaml` tuned to the repo's conventions (2-space
indent; **disable comment reflow** so the heavy `architecture.md`-referencing comment
blocks are not mangled; generous `line_width`) and a CI step `cmake-format --check` (pip
`cmakelang`) over all listfiles. **Start advisory** (`continue-on-error`) to avoid a large
first-PR churn against existing hand-formatted files, then flip to blocking once the tree
is normalized.

**Steps.**
1. `pip install cmakelang` in CI; add `.cmake-format.yaml` (`line_width ~100-120`,
   `tab_size 2`, `dangle_parens` per current style, comment reflow **off**).
2. Step:
   `cmake-format --check CMakeLists.txt cmake/*.cmake include/CMakeLists.txt src/*/CMakeLists.txt tests/CMakeLists.txt bindings/CMakeLists.txt`.
3. Phase 1 advisory (`continue-on-error: true`) while tuning; Phase 2 blocking.
4. Optional: `cmake-lint` (the stricter sibling) for unused-var/missing-parent, advisory only.

**Effort.** **S.**

**Parity-risk.** **None** — checks listfiles, never compiles. `--check` is non-mutating.

**Verify.** `cmake-format --check` clean on the tuned tree; a misindented `add_executable`
in a PR fails the (eventually-blocking) step.

**Files.** `.cmake-format.yaml`, `.github/workflows/host-ci.yml`, `tests/CMakeLists.txt`.

---

## A8 — Coverage-expansion: host-buildable steppe_core logic + CUDA-free `src/device` `.cpp`  *(LOW — L effort, optional)*

**What's needed.** Ten more tests are pure host **logic** (compile with no nvcc) but are
pinned out of the host lane because they link a CUDA-built target:
`config_builder_unit`, `model_search_core_unit`, `f2_combine_unit`, `block_partition_unit`,
`block_ranges_unit`, `f2_from_blocks_unit`, `qpgraph_enumerate` (all → `steppe::core`,
which links `PRIVATE steppe::device` at `src/core/CMakeLists.txt:152`), plus
`shard_plan_unit`, `validate_device_order_unit`, `f2_blocks_multigpu_unit` (→
`steppe::device`, resolving the **CUDA-free plain-`.cpp`** symbols
`plan_block_shards`/`validate_device_order`/`build_resources` that live *inside* the CUDA
target — `src/device/CMakeLists.txt:44-46`, i.e. `host_ram.cpp` + `resources.cpp` +
`shard_plan.cpp`, all documented device-toolkit-free at `:49-55`). The CUDA-free logic
they exercise (config precedence/`build()`, block partition + inverse ranges, host-staged
f2 combine, the model-shard planner, the multi-GPU host orchestrator core, the
device-ordinal validator) is **real, parity-load-bearing host logic** that today only the
box can compile.

**Optimal end state.** With `STEPPE_ENABLE_CUDA=OFF`, a **host-pure slice of steppe_core**
compiles (its `.cpp`s that reach the GPU only via the `ComputeBackend` seam —
`f2_from_blocks.cpp`, `f2_combine.cpp`, `f2_blocks_multigpu_core.cpp`,
`block_partition_rule.cpp`, `model_search_core.cpp`, `config/config_builder.cpp`, the
f4/f3/qpgraph host drivers) against a CUDA-free stand-in for the device dependency; and the
three plain CUDA-free `.cpp`s in `src/device` (`shard_plan.cpp`, `resources.cpp`,
`host_ram.cpp`) build into a tiny host lib. Result: **~21 host tests**, covering the
config/precedence, partition, combine, planner, and orchestrator-core logic — the
**highest parity-divergence-risk host code**. The CUDA-graph-only `.cpp`s (`qpadm_fit.cpp`,
dstat/qpfstats/dates, the multigpu *entry* point that names `combine_f2_partials_p2p`) stay
GPU-lane.

**Steps.**
1. Move the three CUDA-free device `.cpp`s into a `steppe_device_host` plain-CXX static lib
   that builds in **both** modes; `steppe_device` depends on it when CUDA is on.
2. Host-only config of `steppe_core`: either (a) split into `steppe_core_host` (seam-only
   `.cpp`s) + `steppe_core_cuda` (device-symbol entry points), or (b) keep one target with
   the `steppe::device` link **conditional**, substituting `steppe_device_host` + a
   **CUDA-free asserting stub** for `make_cuda_backend`/`combine_f2_partials_p2p` when CUDA
   is OFF (host test executables resolve; any accidental GPU call **fail-fasts**).
3. In `tests/CMakeLists.txt`, move the 10 tests out of the A2 `if(STEPPE_ENABLE_CUDA)`
   guard and re-label `host;unit` once their link targets exist host-side.
4. Keep `f2_blocks_multigpu_unit` honest: it drives the **host** orchestrator core
   (`f2_blocks_multigpu_core.cpp`) against a fake backend, resolves `plan_block_shards`
   from `steppe_device_host`, and does **not** need the RDC P2P symbol (the comment near
   `tests/CMakeLists.txt:1132-1140` confirms) — it joins the host lane cleanly.

**Effort.** **L.**

**Parity-risk.** Higher-touch but still **behavior-neutral if done as pure re-homing**:
moving already-CUDA-free `.cpp`s between targets changes no code and no golden; the stub
path is reached only when CUDA is OFF (host tests), never on the box. **The one trap:** the
stub for `make_cuda_backend`/p2p-combine must **fail-fast (throw)**, never silently no-op,
so a mis-built host artifact can't masquerade as a GPU run. Guard with a death-test;
default ON. No §12 reduction order, no kernel, no precision path touched.

**Verify.** `STEPPE_ENABLE_CUDA=OFF` → `ctest -L host` runs ~21 (incl. `config_builder_unit`,
`block_ranges_unit`, `f2_combine_unit`, `shard_plan_unit`, `f2_blocks_multigpu_unit`); on
the box the same tests still build via the CUDA path and pass byte-identically (re-homing
is structural). **Diff the box golden suite before/after = unchanged.**

**Files.** `src/core/CMakeLists.txt`, `src/device/CMakeLists.txt`, `CMakeLists.txt`,
`tests/CMakeLists.txt`, `src/core/fstats/f2_blocks_multigpu_core.cpp`,
`src/device/shard_plan.cpp`, `src/device/resources.cpp`, `src/device/host_ram.cpp`.

---

## A9 — Provenance-token regression grep-gate (no new ticket / milestone / "spike" token in shipping comments)  *(HIGH — XS effort)*  ·  *(cross-cut gap G1 CI-half, from `docs/kimiactions/04`)*

**What's needed.** The shipping + **public** source carries internal-provenance archaeology
*inside comments* — `cleanup X-N/B-N` ticket codes, `group-N` campaign labels,
`M4`/`M4.5`/`M5` milestone codes, and "the spike" references — and it reaches the **public
seam** and **public include**. Verified at HEAD:
- `cleanup X-N/B-N` on the public seam `src/device/backend.hpp:56,74,581` and the public
  include `include/steppe/config.hpp:249,312,399,431`;
- "the spike" on the public include `include/steppe/config.hpp:10,33,66` (and many more),
  plus `src/device/cuda/device_buffer.cuh:9`, `stream.hpp:5`, `handles.hpp:5`;
- aggregate density: **~119** `cleanup X-N` hits + **~57** `group-N` hits across
  `src/`+`include/`, and milestone tokens **M4×146 / M4.5×63 / M5×56**.

This is the single worst senior-first-impression signal — a public header reading as a
private engineering journal. **The one-time SCRUB of these is a *separate* action**
(`03-low-polish.md` G1 "Comment hygiene for an external reader", public-surface-first).
**This item is only the regression guard** that stops new tokens from re-entering after
(and during) the scrub.

> **Critical caveat (why this cannot be a clang-format job).** The planned A4 clang-format
> step runs with **comment reflow OFF**, so it reformats whitespace but **never rewrites
> comment content** — it will not strip these tokens; it *entrenches* them. The regression
> guard MUST therefore be a **separate grep step**, a sibling of the A6 arch-grep gate, not
> a formatter responsibility.

**Optimal end state.** A `scripts/provenance_token_gate.sh` wired as a fast standalone CI
job next to A6, greping first-party comments for any NEW internal-provenance token and
FAILing CI on a hit:
- `cleanup [A-Z]-?[0-9]` (e.g. `cleanup X-8/B16`, `cleanup C-1`),
- `group-[0-9]` (campaign group labels),
- `M[0-9]\.?[0-9]*` milestone codes (`M4`, `M4.5`, `M5`) — word/comment-anchored to avoid
  false hits on legitimate identifiers,
- `the spike` archaeology.
**PR-diff-scoped** on PRs (only NEW/changed `+` lines gate, so the pre-scrub backlog does
not red every PR before `03-G1` lands), **full-tree** on `main` push once the scrub is in.

**Steps.**
1. Script restricts to first-party source comments (`*.hpp/*.cuh/*.cu/*.cpp/*.h`),
   excluding `docs/`, `experiments/`, `tests/reference/` fixtures; match comment context
   (`//`, `///`, `/*`, ` * `) so a token inside a string literal isn't required to trip it
   but a comment one does.
2. PR-diff mode:
   `git diff -U0 origin/main... -- '*.hpp' '*.cuh' '*.cu' '*.cpp' '*.h' | grep '^+' | grep -E '<token-alternation>'`
   → any hit prints file + the offending added line + `exit 1`.
3. Full-tree mode for `main` push (post-scrub): grep the trees; expected zero.
4. Dev-side message: "internal provenance tokens (cleanup / group / M-milestone / spike)
   must not appear in shipping comments — see `docs/kimiactions/03` G1."
5. **Sequence:** land the gate in **PR-diff mode alongside A6 immediately** (guards all new
   code); flip to **full-tree blocking only after `03-G1`'s scrub** clears the ~119+ backlog
   (otherwise the existing hits red every push).

**Effort.** **XS.**

**Parity-risk.** **None** — read-only grep over comments; no compile, no compute,
**golden-neutral** (comment-only by construction; cannot move a golden).

**Verify.** Add `// cleanup Z-9` (or `// M4.5 perf note`) to a changed `.cu` in a PR → gate
fails with `file:line`; remove → green. Confirm full-tree mode is **not** switched to
blocking until `03-G1` lands. Confirm a legitimate identifier (e.g. a variable named `m5`)
is not falsely flagged (comment-anchored).

**Files.** `scripts/provenance_token_gate.sh`, `.github/workflows/host-ci.yml`; *(guards,
not edited here)* `src/device/backend.hpp`, `include/steppe/config.hpp`.

---

# Cluster B — GPU parity CI (Phase 1 + 2)

> Priority: **HIGH** for the lane that proves production numerics (B2/B3/B5); the
> option-evaluations (B6 Option A defer, B7 Option C reject) are recorded decisions. **B9**
> (compute-sanitizer lane) is the Phase-2 hardening item that itemizes the sanitizer pass
> the §0 framing/table reference.

## B.0 — GPU-runner option comparison (the decision, on the record)

| Option | Hardware | sm_120? | Cost | Data residency | Verdict |
|---|---|---|---|---|---|
| **A. Persistent self-hosted box** | one owned/reserved Blackwell, 24/7 | ✅ | pays for **idle GPU** | AADR resident once, warm ccache | **DEFER** — best reliability, worst cost; revisit if GPU-CI volume > ~1 GPU-hr/day or perf-gating wanted (**B6**) |
| **B. Ephemeral vast.ai JIT runner** | sm_120 booted per job, self-destructs | ✅ | ~1 GPU-hr/night + ad-hoc | re-materialize AADR per box | **RECOMMENDED engine** (**B2**) |
| **C. Hosted/cloud GPU CI** (GH-hosted, RunsOn/Cirun/Namespace/Depot) | GH-hosted = **T4/sm_75**; AWS-backed tops out **H200**, no Blackwell | ❌ | would relocate 30 GB AADR to 3rd-party cloud | **REJECT** for the parity gate — wrong arch, not price (**B7**) |
| **D. Hybrid** (host PR lane + B as GPU engine) | — | ✅ (GPU half) | lowest viable | tiered | **RECOMMENDED topology** (**B3**) |

**Why C is dead for the gate:** GitHub-hosted GPU runners are **Tesla T4 / sm_75 Turing**
(GA 2024) — they cannot execute sm_120 kernels and the Ozaki emulation downgrades
off-Blackwell, so they validate *nothing*. AWS-backed services expose **no consumer
Blackwell (RTX 5090 / sm_120)** and not even datacenter **B200 / sm_100**. A green light on
T4/A100/H200 says nothing about the emulated-FP64 sm_120 numerics users run. (Only re-open
if a provider lists sm_120, or steppe adds a *validated* sm_100 lane with its own goldens —
and even then never as the sm_120 acceptance gate.)

---

## B1 — Host-only CUDA-free PR lane (foundation)

This **is** Cluster A (Phase 0). Listed here only to make the dependency explicit: the
hybrid topology and the GPU dispatcher both sit on top of the host lane. The runbook
already documents the **no-GPU CpuBackend FAST gate** (`CUDA_VISIBLE_DEVICES=""` →
CpuBackend native-FP64 oracle auto-runs, e.g. `qpadm_parity` fit vs `golden_fit0`,
`BOX-RUNBOOK.md:76`) — a real gate needing zero GPU and zero AADR. An optional **Job B**
on `ubuntu-latest` can `apt-get install` the **CUDA 13 toolkit (no driver/GPU)**, do
`cmake --preset ci` (full nvcc build needs the *toolkit*, not a *GPU*), then run
`CUDA_VISIBLE_DEVICES='' ctest -L host` so GPU-needing tests SKIP-exit-0 and the CpuBackend
gate runs. **Still a smoke gate, not parity** (native vs emulated FP64). See Cluster A for
the full build-out; effort **M**, parity-risk **none**.

---

## B2 — Ephemeral vast.ai instance as a JIT self-hosted runner  *(HIGH — the GPU engine)*

**What's needed.** The parity validation must run on the **same hardware the goldens were
validated on** — RTX 5090 / sm_120 / CUDA 13.0 U2+ / driver ≥ 580 (`BOX-RUNBOOK.md:37`).
vast.ai is the only cost-viable source. Replace the manual `ssh box5090 + rsync` loop
(`scripts/box_bringup.sh`) with a **self-registering ephemeral runner**: a dispatcher mints
a GitHub **JIT** runner config (`POST /repos/{owner}/{repo}/actions/runners/generate-jitconfig`),
creates a vast instance whose `--onstart-cmd` downloads the Actions runner and runs
`./run.sh --jitconfig <token>`; the runner dials **out** to GitHub (no inbound SSH, no
ephemeral-IP alias to chase), pulls exactly one queued GPU job, runs build + parity ctest,
auto-deregisters, and the instance self-destructs to stop billing. vast supports this:
`--onstart-cmd` (16 KB; gzip+base64 for longer), `/root/onstart.sh`, account/instance env
injection, and the predefined `CONTAINER_API_KEY`/`CONTAINER_ID` vars.

**Optimal end state.** A `runs-on: [self-hosted, gpu, sm120]` job serviced by a single-use
vast box the pipeline boots and tears down per run.
- `scripts/runner_onstart.sh` = the `box_bringup.sh` REJECT preflight (`cc==12.0`,
  `CUDA≥13.0`, `driver≥580`, `disk≥80G` — `box_bringup.sh:18` already encodes these as a
  REJECT *header*; **turn the REJECT into a non-zero exit that fails the JOB loudly**, never
  silent-skip) + register-JIT + run-one-job + self-destruct (`trap` →
  `vastai destroy instance $CONTAINER_ID`). Carry over `ulimit -c 0`
  (`box_bringup.sh:11` / `BOX-RUNBOOK.md:50-52` — an `abort()` can dump a ~404 MB core and
  wedge the box).
- `scripts/vast_ci_launch.sh` =
  `vastai search offers 'gpu_name=RTX_5090 cuda_vers>=13.0 disk_space>80 inet_down>100' --on-demand`
  then `vastai create instance` with the onstart.
- Runner is **always ephemeral + single-job** (GitHub-recommended posture) and **only**
  triggered by trusted events (push to own branches, `workflow_dispatch`, schedule) —
  **never fork-PR** — so a self-hosted runner cannot execute untrusted code.
- The GPU job runs parity ctests in a **Release build** (`cmake --preset ci`; debug's
  per-kernel `cudaDeviceSynchronize` voids timing — the documented footgun).

**Steps.**
1. Generalize `scripts/box_bringup.sh` → `scripts/runner_onstart.sh`: keep the §1-2
   hardware REJECT gate but make REJECT a non-zero exit; drop the rsync (the runner checks
   out via `actions/checkout`); add: `curl` the actions-runner tarball, `./run.sh
   --jitconfig $JIT`, and an exit `trap` that destroys the instance.
2. Add `scripts/vast_ci_launch.sh` (dispatcher helper): query the GH API for a repo-level
   JIT config, base64 the onstart, `vastai create instance --on-demand
   --image nvidia/cuda:13.x-devel --onstart-cmd …` passing the JIT token + repo via env.
3. Author `.github/workflows/gpu-parity.yml`, two jobs:
   - `dispatch` (ubuntu-hosted) mints JIT + launches the vast box;
   - `gpu` (`runs-on: [self-hosted, gpu, sm120]`) does `cmake --preset ci` +
     `ctest -L parity` — the **committed-fixture emulated-FP64 lane**: `qpadm_parity`
     (`tests/CMakeLists.txt:539`, golden `golden_fit0`), `qpadm_rotation` (`:672`, golden
     `golden_rot`/`golden_fit1_NRBIG`), `qpgraph_parity`, `qpfstats_fused_parity`,
     `fstat_sweep_parity` — which need **no AADR** (fixtures `f2_*.bin` are committed under
     `tests/reference/goldens/at2/fixtures/`).
4. Pin **on-demand** (non-interruptible) instances, **not spot** — a mid-run reclaim flaps
   the parity gate; add a job `timeout-minutes` and a **retry-once on infra (not test)**
   failure.
5. Gate the workflow to push / `workflow_dispatch` / schedule only (no fork PRs).

**Effort.** **L.**

**Parity-risk.** Infra-only; changes no math. The real risk is a **false green/red from the
wrong host or flaky timing**: (1) marketplace hands a non-5090 / older driver — guarded by
the REJECT preflight failing the job loudly; (2) spot reclaim mid-ctest — guarded by
on-demand + infra-retry; (3) shared/cold-GPU timing jitter — guarded by running **only
parity/correctness** ctests here (deterministic, golden-gated) and keeping timing benches
out (**B8**). This is the **one lane that proves emulated-FP64 sm_120 parity** — its
integrity is the whole point.

**Verify.** `workflow_dispatch`: a vast instance boots, a JIT runner appears under
*Settings → Actions → Runners*, the `gpu` job runs `ctest -L parity` green, the runner
deregisters, `vastai show instances` shows it destroyed. Inject a 1-ulp golden
perturbation on a branch → the lane goes red on sm_120.

**Files.** `scripts/box_bringup.sh`, `scripts/runner_onstart.sh`,
`scripts/vast_ci_launch.sh`, `.github/workflows/gpu-parity.yml`, `docs/BOX-RUNBOOK.md`,
`CMakePresets.json`.

---

## B3 — Hybrid wiring: PR = host lane, nightly + on-demand = GPU parity lane  *(HIGH — the topology)*

**What's needed.** Tie the host lane (B1) and the vast JIT engine (B2) into one coherent
policy: never pay for a GPU on routine PRs, but always re-validate emulated-FP64 sm_120
parity before a merge to default / on a nightly cron / on explicit request. Matches the
assessment's own staging (host lane low; GPU runner harder; full GPU runner a follow-on).

**Optimal end state.** PRs get only the free host lane. The GPU parity lane fires on:
(a) **schedule** (nightly cron) — full fixture-parity + a rotating AADR-tier slice;
(b) **`workflow_dispatch`** — on-demand pre-merge validation;
(c) optional **`gpu-ci` PR label** (maintainer-applied only, so fork code never reaches the
runner). A `concurrency` group ensures **at most one vast box alive at a time** (cost
ceiling). Branch protection requires the host lane on every PR and the latest GPU parity
run green before release tagging. Total spend ≈ one GPU-hour/night + ad-hoc dispatches, vs.
a 24/7 box.

**Steps.**
1. In `gpu-parity.yml`: `on: { schedule: nightly, workflow_dispatch, label: gpu-ci }` and
   `concurrency: { group: gpu-ci, cancel-in-progress: false }` to serialize boxes.
2. Split into a **light nightly tier** (`ctest -L parity`, no AADR, ~minutes) and a
   **heavier weekly/dispatch tier** that provisions AADR (`ctest -L 'aadr|parity'`) — see
   **B4** for provisioning.
3. Branch protection: host lane required on all PRs; a manual "gpu-parity validated" gate
   (the dispatch run) required before a release tag.
4. Document the policy in `BOX-RUNBOOK.md` / a `CONTRIBUTING` note with the explicit
   "green host lane ≠ parity" caveat.
5. Add a status badge / run summary so a maintainer sees at a glance when parity was last
   re-proven on sm_120.

**Effort.** **M.**

**Parity-risk.** Behavior-neutral policy wiring. The only parity-relevant judgement is
**cadence**: if the GPU lane runs too rarely an emulated-FP64 regression could land on a
branch unnoticed — mitigated by requiring a dispatch GPU run before any release tag + the
nightly cron. No golden moves.

**Verify.** Open a PR → only the host lane runs (no vast spend). Apply `gpu-ci` as
maintainer → the GPU lane boots a box. Two simultaneous triggers do **not** start two boxes
(concurrency holds). Nightly cron produces a green parity run with a dated summary.

**Files.** `.github/workflows/gpu-parity.yml`, `.github/workflows/host-ci.yml`,
`docs/BOX-RUNBOOK.md`.

---

## B4 — Secrets, data residency (goldens vs 30 GB AADR), cost-safety  *(MED)*

**What's needed.** The cross-cutting plumbing the GPU lane needs to be safe and affordable.
Three concerns: (1) **secrets** — a fine-grained GitHub PAT (repo *Self-hosted runners*
RW + *Contents* read) to mint JIT configs, plus a `VAST_API_KEY`, both as GitHub Actions
secrets consumed only by the ubuntu dispatcher; (2) **data residency** — goldens travel
with `actions/checkout` (committed under `tests/reference/goldens/at2/`: `golden_fit0.json`,
`golden_fit1_NRBIG.json`, `fixtures/f2_*.bin` — all confirmed present), but the **~30 GB
AADR does not live in git** and must be re-materialized per ephemeral box; (3)
**cost-safety** — an ephemeral GPU box that fails to self-destruct silently bleeds money.

**Optimal end state.** Secrets live only in GitHub Actions secrets and pass to vast as
**short-lived per-instance env** (not account-global, not on disk). The JIT token is
single-use and expires — no long-lived registration token ever exists. **Data is tiered:**
- The **light/nightly parity tier** runs entirely off committed fixtures (qpAdm / qpWave /
  qpGraph / qpfstats / rotation read `fixtures/f2_*.bin` from the checkout — **zero AADR**,
  confirmed in `tests/CMakeLists.txt`), so most GPU runs provision nothing heavy.
- The **AADR-keyed tests** (decode / transpose / pa / eigenstrat / plink / ancestrymap
  equivalence, `f2_blocks`, `f2_determinism` at `tests/CMakeLists.txt:218`, real-data CLI
  smoke — all keyed on `STEPPE_AADR_ROOT`) run only in the **heavier tier**, which provisions
  AADR by pulling the raw triple from durable object storage (S3/R2) and **regenerating
  derived tensors on the box**.

> **Correction to the snapshot:** the matrix-build script is **not** `aadr/build_tgeno_matrix.py`
> (does not exist). The real script is **`experiments/aadr/02_build_matrix.py`**, and the raw
> AADR triple lives at `aadr/v66.p1_HO.aadr.patch.PUB.{geno,snp,ind}` (~3.8 GB). The heavy-tier
> onstart pulls the raw triple from object storage and runs `experiments/aadr/02_build_matrix.py`
> to regenerate the derived tensors (the runbook's documented, faster-than-download path).

A self-destruct trap + a sweeper cron guarantee no orphaned billing.

**Steps.**
1. Create a fine-grained PAT scoped to the repo: *Administration / Self-hosted runners:
   read+write* + *Contents: read*; store as secret `GH_RUNNER_PAT`. Store `VAST_API_KEY`.
   Both referenced **only** in the ubuntu `dispatch` job.
2. Pass the JIT token + repo to vast via `--onstart-cmd` env at create time (per-instance,
   short-lived), **not** via vast account-level env injection.
3. Put the AADR raw triple in S3/R2; the heavy-tier onstart pulls raw + runs
   `experiments/aadr/02_build_matrix.py` to regenerate the derived tensors. Keep the light
   parity tier AADR-free.
4. Add `scripts/gpu_ci_sweeper.sh` run by a **separate scheduled workflow**:
   `vastai show instances` and destroy any steppe-CI instance older than the max job
   timeout (kills orphans from crashed runs).
5. Set `ulimit -c 0` in onstart (the runbook's core-dump-flood guard, `BOX-RUNBOOK.md:50-52`).
6. Optional: cache derived tensors in object storage keyed by raw checksum to skip
   regeneration on the heavy tier.

**Effort.** **M.**

**Parity-risk.** **None to the math.** Residency only affects *which* tests run on a box,
not their results. The guardrail: if AADR provisioning fails, AADR-labelled tests must
**SKIP-exit-0 loudly-logged** (their existing behavior) rather than silently shrinking the
parity surface — surface "AADR tier skipped" in the run summary so a green light is never
mistaken for full coverage.

**Verify.** Light tier with no object-storage access → parity tests still pass (proves AADR
independence). Heavy tier → derived tensors regenerate and decode/f2-blocks equivalence
pass. Kill a job mid-run → `gpu_ci_sweeper` destroys the orphan within one sweep interval
(`vastai show instances` empty).

**Files.** `scripts/vast_ci_launch.sh`, `scripts/gpu_ci_sweeper.sh`,
`scripts/runner_onstart.sh`, `experiments/aadr/02_build_matrix.py`, `tests/CMakeLists.txt`,
`.github/workflows/gpu-parity.yml`.

---

## B5 — Wire the §12-promised `cusolverDnSetDeterministicMode` CI assertion into the GPU lane  *(HIGH — parity-law integrity)*

**What's needed.** `architecture.md:788` states *"CI asserts the rank-test routine is one
of [the deterministic-mode-covered cuSOLVER routines]"* — but **there is no CI** and (per
the assessment's #1 High item) the mode is **not even wired** (zero call sites). §12 is
frozen acceptance criteria and currently contains a **false claim**. The GPU lane is the
only home for the promised assertion. This is the **CI half**; the wire-or-doc-fix decision
is the separate High parity item — but the assertion must land here so the doc stops lying.

**Optimal end state.** Once the determinism-mode decision is made (wire it on the statistic
handle covering the `large_svd_V` `gesvdj`/`gesvd` rankdrop path, **OR** correct the doc),
the GPU parity lane runs a small ctest that asserts the rank-test SVD path uses a
deterministic-mode-covered routine (with the `gesvd` `m>n` orientation precondition
checked) **and** that EmulatedFp64 rank-test results are bit-identical run-to-run (extends
the existing `test_f2_determinism` pattern, `tests/CMakeLists.txt:207-218`, to the rank
test). The §12 sentence becomes **true**: an actual CI gate backs it.

**Steps.**
1. Coordinate with the determinism-mode High item: if **wired**, add a reference test
   asserting the statistic handle has deterministic mode set and the rank-test calls a
   covered routine (`gesvdj`, or `gesvd` with `m>n`); if **doc-corrected**, instead assert
   the documented Jacobi-path determinism.
2. Add a **bit-identity rank-test ctest** (run the rankdrop SVD twice under
   `EmulatedFp64{40}`, assert byte-equal) labelled `parity`, in the light GPU tier.
3. Update `architecture.md §12` (~`:788`) so the "CI asserts…" clause names the actual test.
4. Run in `gpu-parity.yml` `ctest -L parity`; require it before release tagging.

**Effort.** **S** (the CI half).

**Parity-risk.** The CI-assertion half is **behavior-neutral**. The **coupled wiring half**
(actually calling `cusolverDnSetDeterministicMode`) **CAN change cuSOLVER algorithm
selection and therefore COULD move the qpadm rankdrop golden** — it **MUST** be guarded by
the existing `qpadm_parity` / `qpadm_rotation` golden gates (`golden_fit0` /
`golden_fit1_NRBIG` / `golden_rot`) on sm_120 before merge. If a golden moves, that is the
decision point: validate the new value against AT2, or back out. The assertion test itself
moves nothing.

**Verify.** On the GPU lane the new rank-test determinism/coverage test passes; flip
determinism mode off in a scratch branch → test red. Confirm `qpadm_parity`/`qpadm_rotation`
goldens are **unchanged** by the wiring (no golden diff).

**Files.** `docs/architecture.md`, `tests/CMakeLists.txt`, `src/device/cuda/cuda_backend.cu`,
`.github/workflows/gpu-parity.yml`.

---

## B6 — Option A (EVALUATED, DEFER): persistent self-hosted GPU box  *(decision)*

**What's needed.** The reliability-maximizing alternative: rent/own one Blackwell box
(sm_120) 24/7, install the Actions runner as a service, keep AADR + derived tensors +
ccache warm on local disk. Evaluated so the decision is on record; **recommendation =
DEFER** unless budget/throughput justifies it.

**Optimal end state (if adopted later).** A single always-on RTX 5090 / RTX PRO 6000 box,
runner as a systemd service (still `--ephemeral` per job for hygiene), AADR resident once
(no per-run 30 GB provisioning), warm ccache/CPM cache (big win for heavy `.cu` builds),
stable timing baselines (the **one** place perf benches could become ctest-gated, since
timing is no longer on a cold shared box). Best reliability + lowest data friction; worst
cost (pays for idle GPU). Adopt **only** if GPU-CI volume > ~1 GPU-hr/day **or**
perf-regression gating becomes a priority.

**Steps (if pursued).** Provision one persistent Blackwell box (vast on-demand reserved, or
owned); install via `config.sh` with a repo registration; run as systemd with `--ephemeral`
(clean workspace per job). Provision AADR once to local disk; enable ccache
(`CMAKE_CXX/CUDA_COMPILER_LAUNCHER`). **Reuse the same `gpu-parity.yml`
`runs-on: [self-hosted, gpu, sm120]` job** — the workflow is identical to Option B; only the
runner lifecycle differs. Lock down: private repo / trusted triggers only, no fork-PR
execution, egress controls. Re-evaluate cost monthly vs. the ephemeral lane's actual spend.

**Effort.** **M.**

**Parity-risk.** Same as Option B (infra-only, golden-gated). Added subtlety: a warm
always-on box could mask a from-cold-build failure or a fresh-driver regression —
periodically run a clean-checkout, cache-cleared build to keep it honest. No golden moves.

**Verify.** Runner shows Online persistently; a dispatched job reuses warm caches and
finishes faster than the ephemeral lane; `ctest -L parity` green; a deliberate cache-clear
run still builds clean under `-Werror`.

**Files.** `.github/workflows/gpu-parity.yml`, `docs/BOX-RUNBOOK.md`, `CMakePresets.json`.

---

## B7 — Option C (EVALUATED, REJECT for the parity lane): hosted/cloud GPU CI  *(decision)*

**What's needed.** A documented rejection so the team does not re-litigate "just use a
hosted GPU runner." The blocker is **hardware + data, not price.**

**Optimal end state (recorded decision).** GitHub-hosted GPU runners are **Tesla T4 only
(sm_75 Turing)** — cannot execute sm_120 kernels; the Ozaki emulation downgrades
off-Blackwell, so they validate parity **not at all**. AWS-backed services (RunsOn supports
T4/A10G/L4/L40S/V100/A100/H100/H200; Cirun/Namespace/Depot run in AWS/GCP/Azure/Oracle)
expose **no consumer Blackwell (RTX 5090/sm_120)** and not even datacenter **B200/sm_100**
— the public clouds do not rent the validated arch. Secondary blocker: **data residency** —
these require moving the 30 GB AADR (and the convertf/derived regeneration) into a
third-party cloud account. **Conclusion: hosted GPU CI is unusable for the sm_120 parity
gate today.** (If/when steppe adds a validated datacenter-Blackwell sm_100 lane with its own
goldens, RunsOn-on-AWS B200 *could* host a **secondary arch-coverage** lane — but never the
sm_120 acceptance gate, and never with goldens validated on a different arch.)

**Steps.** Record the rationale in a docs CI-design note with the concrete facts (T4/sm_75
hosted runners; RunsOn/AWS tops at H200, no Blackwell/sm_120; data-residency cost). Leave a
re-open trigger: revisit only if (a) a provider lists RTX 5090/sm_120, or (b) steppe adds a
validated sm_100 lane with its own goldens. Build **no** pipeline against these for the
parity gate.

**Effort.** **S.**

**Parity-risk.** N/A — a decision *not* to use hardware that cannot reproduce the parity
arch. The risk it **avoids** is real: validating on T4/sm_75 or A100/sm_80 gives a green
light that says nothing about the sm_120 emulated-FP64 numerics the product ships.

**Verify.** Decision recorded; no hosted-GPU workflow exists. Sanity: a quick
`vastai search` vs. any AWS GPU catalog confirms sm_120 absent from clouds, present on vast.

**Files.** `docs/BOX-RUNBOOK.md`.

---

## B8 — Keep timing benches OFF the GPU CI gate  *(MED)*

**What's needed.** Single-shared/ephemeral-GPU timing is **flaky and Release-only** (memory
+ assessment). The four `bench_*.cu` are `add_executable`, **not** `add_test` (intentional).
The GPU CI gate must gate on **correctness/parity only**; perf must not gate on an ephemeral
box where cold-cache + neighbor-noise + cold-clocks make wall-time non-reproducible.

**Optimal end state.** `gpu-parity.yml` runs only deterministic golden/parity ctests.
Benches stay manual, **or** run in a separate nightly `slow` tier with **generous
tolerances** against a stored baseline captured on a **known** box (the Med item: register
`bench_*.cu` as `add_test` with wide tolerances; single-GPU timing is flaky; needs a
dedicated-box baseline). A perf regression posts a **warning/comment, never blocks a
merge**. Release builds enforced for any timing (debug per-kernel
`cudaDeviceSynchronize` voids it).

**Steps.**
1. Confirm `gpu-parity.yml` `ctest -L parity` **excludes** benches (the `slow` label keeps
   them out).
2. If perf tracking is wanted: a nightly `slow` tier runs benches in **Release** with wide
   tolerances vs a committed baseline JSON, emitting a **non-blocking** summary.
3. Document that perf numbers are meaningful only on Release and ideally on the persistent
   box (Option A) if it ever exists; the ephemeral lane is for **parity, not throughput**.

**Effort.** **S.**

**Parity-risk.** **None** — keeps non-deterministic timing out of the acceptance gate,
protecting the parity signal from flaky-timing false reds. No golden interaction.

**Verify.** Run the GPU lane 3× on fresh vast boxes: parity ctests byte-stable green every
time (no timing flakiness in the gate). Any bench tier produces advisory output only, never
a gate failure.

**Files.** `tests/CMakeLists.txt`, `.github/workflows/gpu-parity.yml`, `docs/BOX-RUNBOOK.md`.

---

## B9 — GPU compute-sanitizer lane (Phase 2), with the bare-`cudaGetLastError` post-launch prerequisite  *(MED)*  ·  *(cross-cut gap G4 dependency note, from `docs/kimiactions/04`)*

**What's needed.** The §0 framing + the §2 phase table both name a **compute-sanitizer**
pass as part of Phase-2 hardening, but the plan never itemizes it — and the `04` cross-cut
doc references it as "the `02-ci-plan` one-shot compute-sanitizer pass." This item gives the
lane a home **and records the dependency that makes it actually useful.**

**The dependency (cross-cut gap G4).** There are **14 bare `STEPPE_CUDA_CHECK(cudaGetLastError())`
post-launch sites** — verified at HEAD across **3 files** (the `04` doc says "4 files" but
lists 3): `src/device/cuda/dates_kernel.cu` ×9 (`:341,:351,:360,:368,:377,:385,:395,:405,:417`),
`src/device/cuda/dstat_kernel.cu` ×2 (`:281,:292`), `src/device/cuda/qpgraph_fit_kernels.cu`
×3 (`:492,:511,:538`). The bare form checks only **synchronous launch-config** errors. The
canonical `STEPPE_CUDA_CHECK_KERNEL()` (defined `src/device/cuda/check.cuh:332`) additionally
performs the debug-only `cudaDeviceSynchronize` that gives compute-sanitizer proper
**launch-site + async-fault attribution** on exactly these dates/dstat/qpgraph kernels.
Converting the 14 → `CHECK_KERNEL` is **a PREREQUISITE / multiplier** for this lane: without
it, the sanitizer pass loses launch-site attribution precisely where these kernels fault.
**The 14-site conversion itself is `03-low-polish.md` G4 (the actual fix); this item records
the dependency and consumes the result.**

**Optimal end state.** A Phase-2 `slow`/nightly tier in `gpu-parity.yml` (same vast sm_120
JIT engine as B2, the only place compute-sanitizer can run) executes
`compute-sanitizer --tool memcheck` (plus `racecheck`/`synccheck`/`initcheck` variants) over
`ctest -L parity` **after** the `03-G4` conversion lands — so any async kernel fault is
attributed to its launch site rather than a downstream sync. Advisory/non-blocking at first
(sanitizer overhead + first-run triage), promotable to a pre-release gate once clean. Kept
**off** the per-PR path (sanitizer is slow + GPU-only); lives next to B4's AADR tier and
B5's determinism assertion.

**Steps.**
1. **Sequence `03-G4` FIRST:** convert the 14 bare post-launch checks to
   `STEPPE_CUDA_CHECK_KERNEL()`. Its `cudaDeviceSynchronize` is **compiled out under
   Release/NDEBUG** (zero production cost; the attribution benefit is the sanitizer/debug
   build). Without it, this lane loses launch-site attribution on the dates/dstat/qpgraph
   kernels — the note belongs here so the dependency is explicit and ordered.
2. Add a Phase-2 job to `.github/workflows/gpu-parity.yml` (schedule/dispatch only, same
   vast JIT engine as B2) that builds a sanitizer-friendly config and runs
   `compute-sanitizer … ctest -L parity`.
3. Start `continue-on-error: true` (advisory) while the surface is triaged; promote to a
   pre-release gate once clean.
4. Keep it out of the per-PR path and out of the timing tier (B8) — sanitizer overhead
   makes any wall-time meaningless.

**Effort.** **S** (the CI lane). The 14-site prerequisite is **XS** and lives in `03-G4`.

**Parity-risk.** **None** — `compute-sanitizer` is an observation tool; the
`CHECK_KERNEL` swap only adds a **debug-only** `cudaDeviceSynchronize` (compiled out under
NDEBUG/Release, the production path), touching no kernel math, no reduction order, no
golden. **Golden-neutral.**

**Verify.** After `03-G4` lands, inject a deliberate out-of-bounds in a dates kernel on a
scratch branch → compute-sanitizer reports it **at the launch site** (not a later sync);
with the old bare check the report would land at a downstream sync. Confirm the Release
build is unaffected (the added sync is NDEBUG-gated).

**Files.** `.github/workflows/gpu-parity.yml`, `src/device/cuda/check.cuh`,
`src/device/cuda/dates_kernel.cu`, `src/device/cuda/dstat_kernel.cu`,
`src/device/cuda/qpgraph_fit_kernels.cu`, `docs/BOX-RUNBOOK.md`.

---

# 3. Recommended sequence

**Two independent tracks; do the cheap text gates and the structural enabler first.**

1. **A6 arch-grep** + **A4 clang-format** + **A9 provenance-token gate** — XS, no build, no
   dependency. Land day one as standalone fast jobs; immediate §4-layering + style +
   provenance-hygiene signal. A9 lands in **PR-diff mode** (guards new code now; flip to
   full-tree blocking only after the `03-G1` scrub clears the ~119+ backlog).
2. **A1 `STEPPE_ENABLE_CUDA`** — the hard prerequisite. Nothing host-only configures until
   this lands. Verify byte-identical box configure.
3. **A2 LABELS + test guards** → **A3 host preset + `host-ci.yml`** — the 11-test host lane
   goes green. Add **A5 clang-tidy** (needs the A1 host configure) and **A7 cmake-lint**
   (advisory). *Resolve the `-Werror` × newer-GCC delta before marking host-ci required.*
4. **B2 vast JIT engine** (`runner_onstart.sh` + `vast_ci_launch.sh` + `gpu-parity.yml`) —
   the only lane that proves parity. **B3 hybrid wiring** + **B4 secrets/data tiering** in
   parallel.
5. **B5 §12 determinism CI assertion** — *after* the separate High wire/doc decision;
   golden-gated on sm_120. **B9 compute-sanitizer lane** — *after* `03-G4` converts the 14
   bare post-launch checks (its multiplier); advisory first, then a pre-release gate.
6. **B8 benches-off-gate** — confirm/lock; **B6 (Option A)** and **B7 (Option C)** are
   recorded decisions, near-zero build cost.
7. **A8** — optional refactor lifting the host lane 11 → ~21; do it only when the
   host-orchestration logic churns enough to justify L effort.

**Build order in one line:**
`A6/A4/A9 → A1 → A2 → A3 (+A5,A7) → B2 → B3+B4 → B5 → B9(after 03-G4)/B8/B6/B7 → A8`.

---

# 4. One-glance priority table

| ID | Item | Phase | Priority | Effort | Parity-risk | Depends on |
|---|---|---|---|---|---|---|
| **A1** | `STEPPE_ENABLE_CUDA` + guard CMake | 0 | **HIGH** (enabler) | S | none (default ON) | — |
| **A2** | tests guard + CTest LABELS | 0 | **HIGH** | M | none | A1 |
| **A3** | host preset + `host-ci.yml` | 0 | **HIGH** | M | none (op risk: `-Werror`) | A1, A2 |
| **A4** | clang-format `--dry-run` | 0 | MED | XS | none | — |
| **A5** | clang-tidy diff (host DB) | 0 | MED | S | none | A1 |
| **A6** | arch-grep §4 gate | 0 | MED | XS | none | — |
| **A7** | cmake-lint `--check` | 0 | LOW | S | none | — |
| **A8** | host-buildable core slice (11→~21) | 0 | LOW (opt) | L | none if pure re-home (stub must throw) | A1, A2 |
| **A9** | provenance-token regression grep-gate *(cross-cut G1)* | 0 | **HIGH** | XS | none (comment-only, golden-neutral) | — (scrub = `03` G1) |
| **B2** | vast.ai ephemeral JIT GPU runner | 1 | **HIGH** | L | infra; REJECT-preflight + on-demand guard | A2 (labels) |
| **B3** | hybrid topology wiring | 1 | **HIGH** | M | none (cadence) | B1, B2 |
| **B4** | secrets / AADR tiering / sweeper | 1–2 | MED | M | none to math | B2 |
| **B5** | §12 determinism CI assertion | 2 | **HIGH** | S (CI half) | **wiring half can move a golden — golden-gated** | B2, wire/doc decision |
| **B6** | Option A persistent box | — | DEFER | M | same as B2 | — |
| **B7** | Option C hosted/cloud GPU | — | REJECT | S | N/A (risk avoided) | — |
| **B8** | benches off the gate | 2 | MED | S | none | B2 |
| **B9** | GPU compute-sanitizer lane *(cross-cut G4 dep)* | 2 | MED | S (CI half) | none (golden-neutral; sync is NDEBUG-gated) | B2, `03` G4 |

**Net posture:** every item is build/CI infra and **behavior-neutral by construction**, held
that way by two rules — (1) all formatters/linters run **check-only**, never `--fix` /
auto-commit, so no parity-load-bearing literal is rewritten; (2) `STEPPE_ENABLE_CUDA`
**defaults ON** so the box/production configure is byte-identical and host branches are dead
code on the GPU build. The two **cross-cut additions** folded in from our X1–X7 review
(`docs/kimiactions/04`) — **A9** (a comment-only regression grep) and **B9** (a sanitizer
*observation* lane whose only source touch, the `03-G4` `CHECK_KERNEL` swap, adds an
NDEBUG-gated sync compiled out in production) — are likewise **golden-neutral**. The
**single** exception that can move a golden remains B5's coupled wiring half, and it is
golden-gated on sm_120 before merge. The host lane is a fast **smoke/seam** gate; **parity
acceptance remains the box's job.**

# Build System & Developer Tooling Assessment

This is a first-principles review of the `steppe` build system and developer tooling, based on the files that actually exist in the repository today.  Everything below is read-only: no source files were edited.

## Files examined

- `/home/suzunik/steppe/CMakeLists.txt`
- `/home/suzunik/steppe/CMakePresets.json`
- `/home/suzunik/steppe/cmake/SteppeOptions.cmake`
- `/home/suzunik/steppe/cmake/CUDAArch.cmake`
- `/home/suzunik/steppe/cmake/SteppeWarnings.cmake`
- `/home/suzunik/steppe/cmake/CPM.cmake`
- `/home/suzunik/steppe/pyproject.toml`
- `/home/suzunik/steppe/.clang-format`
- `/home/suzunik/steppe/.clang-tidy`
- `/home/suzunik/steppe/build_m0.sh`
- `/home/suzunik/steppe/include/CMakeLists.txt`
- `/home/suzunik/steppe/src/io/CMakeLists.txt`
- `/home/suzunik/steppe/src/core/CMakeLists.txt`
- `/home/suzunik/steppe/src/device/CMakeLists.txt`
- `/home/suzunik/steppe/src/app/CMakeLists.txt`
- `/home/suzunik/steppe/src/access/CMakeLists.txt`
- `/home/suzunik/steppe/src/extract/CMakeLists.txt`
- `/home/suzunik/steppe/bindings/CMakeLists.txt`
- `/home/suzunik/steppe/tests/CMakeLists.txt` (1,692 lines)
- `/home/suzunik/steppe/tests/python/conftest.py`
- `/home/suzunik/steppe/.gitignore`

---

## 1. CMake structure and target organization

### What is there

The top-level `CMakeLists.txt` is intentionally minimal.  It declares the project, sets C++20 / CUDA C++20, enables `CMAKE_EXPORT_COMPILE_COMMANDS`, includes three modules (`SteppeOptions`, `CUDAArch`, `SteppeWarnings`), finds `CUDAToolkit` and `Threads`, then adds subdirectories in dependency order.  This matches the documented intent in `architecture.md` §4 and §6: one target per source directory and layering enforced by CMake link visibility.

Targets (from leaves to products):

| Target | Type | Directory | Role |
|--------|------|-----------|------|
| `steppe_api` | `INTERFACE` | `include/` | Public, CUDA-free headers |
| `steppe_io` | `STATIC` | `src/io/` | Genotype readers / filters |
| `steppe_core_internal` | `INTERFACE` | `src/core/` | Shared host/device views and estimator primitive |
| `steppe_core` | `STATIC` | `src/core/` | Host orchestration (f2, qpadm, stats, config) |
| `steppe_device` | `STATIC` | `src/device/` | CUDA kernels and backends |
| `steppe_access` | `STATIC` | `src/access/` | f2-dir reader + pop resolver, shared by CLI and Python |
| `steppe_extract` | `STATIC` | `src/extract/` | Genotype-to-f2 extract chain + dir writer |
| `steppe_app` | `EXECUTABLE` | `src/app/` | CLI (`steppe`) |
| `_core` | `MODULE` | `bindings/` | nanobind Python extension |

The layering is mostly sound:

- `steppe_device` keeps CUDA toolkits `PRIVATE` (`CUDA::cublas`, `CUDA::cudart`, `CUDA::cusolver`, `CUDA::cufft`).
- `steppe_core` links `steppe::device` `PRIVATE` and sets `CUDA_RESOLVE_DEVICE_SYMBOLS OFF` to avoid duplicate device-link objects.
- `steppe_extract` mirrors that `CUDA_RESOLVE_DEVICE_SYMBOLS OFF` property, which is correct because it also links `steppe::device` privately.
- `steppe_app` and `_core` are plain C++ targets (no `LANGUAGES CUDA`, no `.cu` files), which structurally enforces that no CUDA header leaks into the CLI / Python surface.

### What is weak

- **`steppe_core` is very large.**  It currently compiles ~45 `.cpp` files spanning f2 assembly, qpAdm fitting, qpGraph search, qpDstat, qpfstats, DATES, f4/f3/f4ratio, model search, and config.  This is not a problem for correctness, but it is a compile-time and incremental-build bottleneck.  There is no finer-grained static library split (e.g., `steppe_qpadm`, `steppe_stats`) for products that only need a subset.
- **Source files live outside their owning target directory.**  `steppe_access` and `steppe_extract` compile files that physically live in `src/app/` (`f2_dir_io.cpp`, `pop_resolver.cpp`, `extract_f2_core.cpp`, `f2_dir_writer.cpp`).  This works but is confusing; the DRY rationale is documented, yet the directory layout does not match the CMake target ownership.
- **No `install()` rules for the C++ library surface.**  Only the Python extension and the implicit executable are produced.  There is no `install(TARGETS steppe_core ...)` or exported CMake package config, so `steppe` cannot be consumed as a C++ dependency by another project.
- **No `OBJECT` library consideration for the huge `steppe_core`.**  Because it is linked into both the CLI and the Python module (via `steppe_extract`), its object code is archived twice.  Using an `OBJECT` library for the common host code would reduce duplication and build time.

---

## 2. CUDA compilation setup and arch handling

### What is there

- `project(... LANGUAGES CXX CUDA)` at the top level.
- `cmake_minimum_required(VERSION 3.28)`.
- `CMAKE_CUDA_STANDARD 20` with `_REQUIRED ON` and extensions off.
- `CUDAArch.cmake` implements a project default of `sm_120` via a custom cache variable `STEPPE_CUDA_ARCH` and handles the CMake 3.28 / CUDA 13 auto-default wrinkle (`CMAKE_CUDA_ARCHITECTURES` gets pre-populated to `75`).
- `steppe_device` enables `CUDA_SEPARABLE_COMPILATION ON` and `POSITION_INDEPENDENT_CODE ON` (required because it is linked into a shared Python module).
- `STEPPE_COMPRESS_FATBIN` optionally adds `--compress-mode=size` for Release.
- The `release` preset carries the full shipping arch list: `75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual`.

### What is weak

- **`sm_120` default is Blackwell-only and not portable.**  That is a deliberate M0 decision (the validation box is an RTX 5090 / CUDA 13), but a developer on an older GPU cannot `cmake --preset dev` out of the box.  Presets for `sm_86`, `sm_89`, `sm_90`, and a CPU-only test mode are missing.
- **No `nvcc` host-compiler compatibility pin.**  The project requires C++20, but there is no detection that the host compiler (g++ / clang++) supports the needed features.  On CUDA 11.8 boxes the default host compiler may be too old for the full C++20 surface used (e.g., `std::source_location`).
- **No `cuda.std` / libcudacxx handling.**  The code uses `std::source_location` inside CUDA headers.  On CUDA 11.8 this may require `cuda::std::source_location` instead.  There is no CMake-level feature probe.
- **No `-G` / `-lineinfo` matrix.**  `SteppeWarnings.cmake` only adds `-lineinfo` for `RelWithDebInfo`.  A dedicated `profile` preset with `-lineinfo` and NVTX would be useful.
- **Device LTO is mentioned but not wired.**  `architecture.md` §6 discusses multi-arch device LTO; the CMake side only has the placeholder `STEPPE_COMPRESS_FATBIN`.
- **`CUDA_RESOLVE_DEVICE_SYMBOLS OFF` is repeated by rote.**  Both `steppe_core` and `steppe_extract` set it.  This is correct, but easy to forget for any future target that links `steppe::device` privately.  A helper function (e.g., `steppe_add_host_consumer`) would eliminate the repetition.

---

## 3. Build presets and developer experience

### What is there

`CMakePresets.json` defines:

| Preset | Build type | Tests | Archs | Notes |
|--------|-----------|-------|-------|-------|
| `base` | hidden | — | `120` | `Ninja`, `STEPPE_HAVE_EMU_TUNING=ON` |
| `dev` | `RelWithDebInfo` | ON | `120` | Default dev preset |
| `release` | `Release` | OFF | Full list | Ship build |
| `ci` | `Release` | ON | Full list | Inherits `release` |

There are matching `buildPresets` and `testPresets` for `dev`, `release`, and `ci`.

### What is weak

- **Only four presets.**  Missing:
  - `debug` (Debug, `STEPPE_HAVE_EMU_TUNING=ON`, fast compile, no fatbin compression).
  - `asan` / `ubsan` / `compute-sanitizer` presets (the option `STEPPE_SANITIZER` exists but is not wired).
  - A `python` preset that builds the wheel target with `STEPPE_BUILD_PYTHON=ON`.
  - A `no-gpu` or `mock` preset that compiles and runs host-only unit tests on a machine without CUDA.
  - A `profile` preset (Release + `-lineinfo` + NVTX).
- **No `CMakeUserPresets.json` template.**  Local box-specific arch overrides are left to the developer to remember.
- **`STEPPE_HAVE_EMU_TUNING` is hard-coded to `ON` in `base`.**  This is fine for the validation box, but a local Turing/Ampere dev box may not have the cuBLAS symbols.  A `legacy` preset with it off would help.
- **No `ccache` integration.**  For a project with heavy `.cu` compilation, `ccache` should be recommended / auto-enabled.  There is no `CMAKE_CXX_COMPILER_LAUNCHER` / `CMAKE_CUDA_COMPILER_LAUNCHER` logic.
- **`build_m0.sh` is a fallback script that bypasses CMake.**  It is documented as a stop-gap, but it duplicates the include paths, source list, and flags.  It is likely to drift out of sync with the real CMake build.

---

## 4. Warning flags and static analysis setup

### What is there

- `SteppeWarnings.cmake` defines `steppe::warnings` with `-Wall -Werror` for host C++ and `--Werror all-warnings` + `-Xcompiler=-Wall,-Wextra,-Werror` for device TUs.  Warnings-as-errors is intentionally strict.
- `.clang-format` is present, based on LLVM style, 4-space indent, 100-column limit, attached braces, left pointer alignment, C++20.
- `.clang-tidy` is present with a broad check list (`bugprone-*`, `cppcoreguidelines-*`, `performance-*`, `modernize-*`, `readability-*`, `portability-*`, `misc-*`), naming-convention rules, and `WarningsAsErrors: '*'`.  It scopes to `include/steppe` and `src/`.

### What is weak

- **No CI job actually runs clang-format, clang-tidy, or build warning checks.**  The configuration files exist, but there is no enforcement mechanism.
- **`.clang-tidy` disables a large number of checks** (`readability-magic-numbers`, `cppcoreguidelines-pro-bounds-pointer-arithmetic`, `cppcoreguidelines-avoid-c-arrays`, etc.).  Some of these are reasonable for a numerical kernel, but the blanket disablement is not periodically reviewed.
- **No `clang-tidy` CI diff mode.**  Running clang-tidy over the entire 1,692-line test CMake tree and all CUDA TUs on every commit is slow.  A script that checks only changed lines is needed.
- **No include-what-you-use (IWYU).**  The project has layered headers and an explicit seam discipline; IWYU would catch leaked includes.
- **No `cppcheck`, `clang-static-analyzer`, or `nvdisasm` / `nvcc` device-side lint pass.**
- **No compiler warning matrix.**  The project is only tested with one host compiler / one nvcc version on one box.  GCC vs Clang differences are unexplored.
- **Warning flags are not configurable.**  A developer who needs to temporarily relax a warning for an experiment has to edit the module.  A cache variable `STEPPE_WERROR=OFF` (default ON) would help local iteration without editing source.

---

## 5. Python packaging (wheel, scikit-build-core, nanobind)

### What is there

- `pyproject.toml` uses `scikit-build-core>=0.10` with `nanobind>=2`.
- Build backend is `scikit_build_core.build`.
- The wheel is configured to build only `_core`, install the pure-Python facade from `bindings/steppe`, and use `cmake.build-type = "Release"`.
- `cmake.args` pins `STEPPE_BUILD_PYTHON=ON`, `STEPPE_BUILD_CLI=OFF`, `STEPPE_BUILD_TESTS=OFF`, and `STEPPE_CUDA_ARCH=120`.
- Runtime deps are minimal: only `numpy>=1.22`.  `pandas` and test deps are optional extras.
- CUDA 13 runtime is intentionally **not** bundled; the header comment documents the rationale.

### What is weak

- **No CI wheel build.**  There is no GitHub Actions / GitLab CI job that builds the wheel on a CUDA runner and uploads it as an artifact.
- **No wheel matrix.**  The project ships only an `sm_120`-compiled `_core.so`.  There is no multi-arch wheel, no `manylinux` build, no `auditwheel` repair, and no RPATH injection for the optional `nvidia-*-cu13` wheels.
- **No version-source-of-truth sync.**  The version is duplicated between `CMakeLists.txt` (`project(steppe VERSION 0.1.0)`) and `pyproject.toml` (`version = "0.1.0"`).  This will drift.
- **No `python -m build` / `cibuildwheel` configuration.**  The project relies on manual `pip wheel .` with config settings.
- **No mypy / pyright / ruff for the Python facade.**  The pure-Python package in `bindings/steppe/` has no linting or type-checking configured.
- **No Python test coverage reporting.**  Pytest runs, but there is no coverage gate.
- **No `pip install -e .` editable workflow documented.**  Developers must manually point `PYTHONPATH` at the build tree.

---

## 6. Missing tooling (CI, formatting checks, linting, IWYU, pre-commit)

This is the largest gap.  The repository currently has **no CI at all**.

Missing:

- `.github/workflows/` or equivalent CI definitions.
- Pre-commit hooks (`.pre-commit-config.yaml`).
- A `justfile`, `Makefile`, or `taskfile` for common dev commands.
- Scripts to run `clang-format --dry-run`, `clang-tidy`, `iwyu`, and `cmake-format` / `cmake-lint`.
- A reproducible dev container / Dockerfile for CUDA 13 + Blackwell.
- `dependabot` / `renovate` configuration for `pyproject.toml` and CPM pins.
- A release-drafter or changelog workflow.

The existing `agentscripts/` directory contains ad-hoc local scripts, but it is gitignored and not a substitute for CI.

---

## 7. Dependency management (CPM.cmake usage)

### What is there

- `cmake/CPM.cmake` is a downloader bootstrap that fetches CPM.cmake v0.42.3 on demand.
- CPM is only included from `src/app/CMakeLists.txt` (for CLI11 2.4.2) and `bindings/CMakeLists.txt` (for nanobind 2.4.0 fallback).  This keeps CLI11 and nanobind out of the core/device build.
- `find_package(... CONFIG QUIET)` is attempted first for CLI11 and nanobind, so offline / cached builds are supported.

### What is weak

- **No hash verification.**  The `CPM_HASH_SUM` in `cmake/CPM.cmake` is a placeholder (`"97e3f10b5b0..."`) and explicitly noted as such.  A supply-chain attacker who controls the GitHub download could inject code into the CLI.
- **No lockfile / pinned manifest.**  CPM packages are pinned by version tag, but there is no single `dependencies.cmake` or `vcpkg.json` / `conanfile.py` that lists all third-party deps with hashes.
- **GoogleTest is not fetched; it is only used if already installed.**  The tests gracefully fall back to self-checking executables, but there is no CPM path for GTest.  That makes the GTest-based unit tests non-hermetic.
- **No CPM_SOURCE_CACHE documentation.**  Developers may accidentally re-download on every clean build.
- **No SBOM / license audit.**  A shipped wheel bundles nanobind runtime statically; the license implications are not tracked.

---

## 8. Test integration and benchmarking

### What is there

- `tests/CMakeLists.txt` is comprehensive: ~1,692 lines defining reference tests, unit tests, CLI tests, and the Python pytest gate.
- Tests use CTest.  Many CUDA reference tests are self-checking executables that `exit 0` on skip (no GPU / no data) and non-zero on failure.
- `STEPPE_HAVE_GTEST` is detected with `find_package(GTest CONFIG QUIET)`.  Host-only unit tests link GTest when available and fall back to self-checking main otherwise.
- The Python gate uses `pytest` via `add_test(... COMMAND "${Python_EXECUTABLE}" -m pytest -v ...)` with `STEPPE_BINDINGS_DIR` and `STEPPE_AADR_ROOT` passed through the environment.
- Tests are organized under `tests/reference/`, `tests/unit/`, `tests/cli/`, and `tests/python/`.

### What is weak

- **No benchmark target.**  There is no `benchmark/` directory, no Google Benchmark integration, and no CMake `add_subdirectory(benchmarks)`.  Performance claims are validated manually on the box.
- **Tests are not labeled / grouped for CTest.**  There is no `LABELS` usage, so `ctest -L unit`, `ctest -L reference`, `ctest -L python`, and `ctest -L slow` cannot be run independently.
- **No test fixture provisioning.**  Many tests rely on `/workspace/data/aadr` or committed goldens under `tests/reference/goldens/`.  If those are absent, the tests skip.  There is no CMake / CI step that downloads or validates the fixtures.
- **No coverage reporting.**  Neither C++ (`gcov` / `llvm-cov`) nor Python coverage is collected.
- **No sanitizer CI job.**  `STEPPE_SANITIZER` exists as a cache string but is never wired to compile flags.  There is no `compute-sanitizer` / `cuda-memcheck` / `valgrind` pass.
- **No flaky-test retry policy.**  GPU tests can be flaky due to driver / memory state; CTest has no `COST` or retry configuration.
- **`tests/CMakeLists.txt` is monolithic.**  At 1,692 lines it is larger than many source files.  Splitting it into `tests/reference/CMakeLists.txt`, `tests/unit/CMakeLists.txt`, etc., would improve maintainability.
- **No `pytest.ini` / `pyproject.toml` pytest config.**  Test discovery, markers, and timeout settings are not centralized.

---

## Plan to make the build / CI / tooling A+

The following is ordered by impact and prerequisites.

### Phase 0: Immediate hygiene (no new infrastructure)

1. **Fix the duplicated version.**  Drive `pyproject.toml` version from `CMakeLists.txt` via a small CMake configure step, or use `scikit-build-core`'s `metadata.version.provider` / `setuptools-scm`.  Do not keep two literals.
2. **Replace the placeholder CPM hash.**  Compute the real SHA-256 of CPM.cmake v0.42.3 and set `EXPECTED_HASH` in the `file(DOWNLOAD ...)` call.  Same for any CPM-fetched package.
3. **Add `STEPPE_WERROR` option.**  Default ON; allow local dev to turn it OFF without editing `SteppeWarnings.cmake`.
4. **Document the exact host-compiler / CUDA versions that work.**  Add a `docs/dev_environment.md` or section in `README.md`.

### Phase 1: Presets and local developer velocity

5. **Expand `CMakePresets.json`:**
   - `debug` (Debug, sm_120, tests on, no fatbin compression).
   - `python` (`STEPPE_BUILD_PYTHON=ON`, `STEPPE_BUILD_CLI=OFF`, tests off, Release).
   - `sm86`, `sm89`, `sm90` dev presets for non-Blackwell boxes.
   - `profile` (Release + `-lineinfo` + `STEPPE_NVTX=ON`).
   - `host-only` (tests on, only targets that compile without a GPU; or a mock backend mode).
6. **Add `ccache` auto-detection.**  In `CMakeLists.txt`, set `CMAKE_CXX_COMPILER_LAUNCHER` / `CMAKE_CUDA_COMPILER_LAUNCHER` to `ccache` when found.
7. **Add a `steppe_add_host_consumer(target)` helper.**  Encapsulates the `target_link_libraries(... steppe::device)` + `CUDA_RESOLVE_DEVICE_SYMBOLS OFF` pattern so future consumers cannot forget it.
8. **Delete or freeze `build_m0.sh`.**  Either remove it now that CMake works, or add a CI check that verifies its source list matches the CMake target sources.

### Phase 2: Static analysis and formatting enforcement

9. **Add a formatting check script** (`scripts/check_format.py` or shell) that runs:
   - `clang-format --dry-run --Werror` on all C/C++/CUDA files.
   - `cmake-format --check` on all `CMakeLists.txt` and `.cmake` files.
10. **Add a clang-tidy CI script** that runs only on changed files for speed, but with the full `.clang-tidy` config.
11. **Introduce IWYU** on a best-effort basis for the host-only targets first (`steppe_io`, `steppe_access`, `steppe_core` when mock-backed).
12. **Add `cmake-lint` / `cmake-format`** to keep the growing CMake files readable.

### Phase 3: CI pipeline

13. **Create `.github/workflows/ci.yml`** (or equivalent) with jobs for:
    - **Configure & build host-only** (no GPU) on Ubuntu + latest GCC/Clang: builds `steppe_io`, `steppe_access`, and host-only unit tests.
    - **Format & lint check** (clang-format, cmake-format, clang-tidy diff).
    - **CUDA build & test** on a self-hosted or cloud GPU runner with CUDA 13: runs `cmake --preset ci && cmake --build --preset ci && ctest --preset ci`.
    - **Python wheel build & test**: `pip install -e .[test]` then `pytest tests/python`.
    - **Sanitizer build**: at minimum ASan/UBSan host-only; compute-sanitizer on GPU tests when available.
14. **Add a nightly / on-demand benchmark job** that runs micro-benchmarks (to be written) and posts results as a PR comment or to a metrics file.
15. **Add artifact upload** for the wheel and the `steppe` CLI binary.

### Phase 4: Python packaging hardening

16. **Add `cibuildwheel` configuration** in `pyproject.toml` for Linux CUDA wheels (start with one CUDA-13 / manylinux image).
17. **Add the optional `cuda` extra** only when the PyPI redistributable wheels are real; until then, document the manual `LD_LIBRARY_PATH` requirement clearly.
18. **Add Python linting and type checking:** `ruff`, `mypy`, and `pytest` config in `pyproject.toml`.
19. **Add an editable install workflow** and document `pip install -e . --no-build-isolation` for local iteration.

### Phase 5: Installable C++ product

20. **Add `install(TARGETS ...)` and `install(EXPORT steppe-targets ...)` rules** so the C++ libraries and CLI can be installed.
21. **Generate a CMake package config file** (`steppeConfig.cmake`) from a template so downstream projects can `find_package(steppe)`.
22. **Split `steppe_core` into smaller static libraries** if the compile time becomes painful: `steppe_fstats`, `steppe_qpadm`, `steppe_stats`, `steppe_config`, leaving `steppe_core` as a thin umbrella.

### Phase 6: Benchmarking and observability

23. **Create a `benchmarks/` directory** with Google Benchmark for the hot paths: f2 single-block, f2-blocks, decode AF, qpadm fit iteration.
24. **Wire NVTX ranges** behind `STEPPE_NVTX` and add a profiling guide.
25. **Add a `STEPPE_BENCHMARK_DATA` CMake cache variable** and CI fixture provisioning so benchmarks can run reproducibly.

---

## Summary verdict

The CMake structure is **solid at the architectural level**: target layering is enforced by link visibility, CUDA is kept private to `steppe_device`, and the Python wheel is configured correctly for a GPU-only product.  The project also has good warning and static-analysis configuration files.

However, the tooling is **not yet production-grade** because:

- There is **no CI** of any kind.
- Formatting, linting, and clang-tidy are **configured but not enforced**.
- The **wheel build is manual** and single-arch.
- **Benchmarking is absent**.
- **Dependency hashes are placeholders**.
- The **test CMake file is a monolith** and lacks labels/fixture management.

Following the phased plan above would move the project from "works on the validation box" to an A+ build/CI/tooling state.

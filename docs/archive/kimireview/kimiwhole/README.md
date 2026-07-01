# steppe — Whole-Project Engineering Assessment: Path to 10/10

## Executive summary

`steppe` is a credible, well-architected GPU population-genetics compute engine. The CUDA-free seam is real and compiler-enforced: no CUDA header leaks out of `src/device/` (`01_architecture_and_layering.md`, §2), RAII resource ownership is strong, AT2 parity testing is excellent, and the CLI/Python surface covers fourteen subcommands. The code is technically competent and numerically careful. At its current state the project sits at a solid **B+/A-**: it works, it is layered, and it is unusually honest about its limitations.

But it is not yet **10/10**. What separates B+/A- from A+/10 is not more features—it is disciplined finishing. The public API does not deliver the C ABI promised in `architecture.md:908-914` (`01_architecture_and_layering.md`, §3). The `ComputeBackend` seam has become a 1,857-line god interface that forces everything to recompile on every change (`01_architecture_and_layering.md`, §2; `04_cuda_engineering.md`, §4). `core` and `io` are tangled in a way the architecture doc denies (`src/core/CMakeLists.txt:153`, `01_architecture_and_layering.md`, §1). There is **no CI at all** (`06_build_system_and_tooling.md`, §6). The output layer is half-centralized: `result_emit.cpp` does the right thing for six commands while `dates`, `qpgraph`, sweeps, `extract-f2`, and `qpfstats` hand-build JSON and emit inconsistent number formatting (`05_io_cli_and_output_ux.md`, §4). Project-status docs are stale, the root directory contains ignored data and workflow dirs, and `docs/architecture.md` contradicts the built tree in multiple places (`09_documentation_and_project_hygiene.md`, §2, §6, §7).

A **10/10 `steppe`** means: a stable C ABI with opaque handles and `steppe_status_t` accessors; a slimmed, role-segregated device seam; CI enforcing every commit; a single `OutputSink`/`CsvRowWriter`/`JsonWriter` used by every command; atomic writes and checked stream state; `cusolverDnSetDeterministicMode` wired and documented; multi-GPU rotation finished with per-device precompute; and docs that match the code. The roadmap below orders the work by impact, not by ease.

## Dimension summary

| Dimension | Grade | Top 1-line issue | Section |
|---|---|---|---|
| Architecture & Layering | B+ | `ComputeBackend` is a god interface and `core` secretly depends on `io`. | [01_architecture_and_layering.md](01_architecture_and_layering.md) |
| Code Conventions & Quality | B+ | Enormous comment manifestos, duplicated boilerplate, and inconsistent index/span contracts. | [02_code_conventions_and_quality.md](02_code_conventions_and_quality.md) |
| Error Handling & Robustness | B+ | No central `Error` type; `Status` taxonomy incomplete; I/O and OOM exit codes wrong. | [03_error_handling_and_robustness.md](03_error_handling_and_robustness.md) |
| CUDA Engineering | B / B+ | Single-stream serialization, serialized P2P combine, missing cuSOLVER determinism. | [04_cuda_engineering.md](04_cuda_engineering.md) |
| I/O, CLI & Output UX | B | Commands bypass the shared emitter; `--out` semantics diverge; silent overwrites. | [05_io_cli_and_output_ux.md](05_io_cli_and_output_ux.md) |
| Build System & Tooling | B- | No CI, formatting/linting configured but not enforced, placeholder dependency hashes. | [06_build_system_and_tooling.md](06_build_system_and_tooling.md) |
| Testing & Performance | B+ / A- | No CI, no coverage, no perf regression gates, manual golden regeneration. | [07_testing_and_performance.md](07_testing_and_performance.md) |
| Public API & Library Design | B+ | Result structs are flat, Python returns polymorphic types, no stable C ABI. | [08_public_api_and_library_design.md](08_public_api_and_library_design.md) |
| Documentation & Project Hygiene | B- / C+ | Stale status docs, architecture monolith, root clutter, inline milestone archaeology. | [09_documentation_and_project_hygiene.md](09_documentation_and_project_hygiene.md) |

## Top 10 cross-cutting actions

Ordered by impact on engineering credibility and maintainability. Each action references the section that owns the details.

1. **Implement the C ABI layer the architecture doc already promises.** Add `include/steppe/steppe_c.h` with opaque handles (`steppe_f2_blocks_t*`, `steppe_result_t*`) and `steppe_status_t` accessors. Keep the current `include/steppe/*.hpp` headers as an explicitly same-toolchain C++ convenience wrapper. This closes the biggest gap between the documented boundary and the installed surface (`01_architecture_and_layering.md`, §3; `08_public_api_and_library_design.md`, §2, §9).

2. **Slim the `ComputeBackend` seam.** Split the 1,857-line `src/device/backend.hpp` into role-specific interfaces (`PrecomputeBackend`, `FitBackend`, `DecodeBackend`, `SweepBackend`, `DatesBackend`, `QpfstatsBackend`) plus a thin facade. Move the upward references to `core::qpadm` helpers out of a device header (`backend.hpp:624-651`). This is the single highest-impact decoupling in the codebase (`01_architecture_and_layering.md`, §2; `04_cuda_engineering.md`, §1.3; `08_public_api_and_library_design.md`, §4).

3. **Decouple `core` from `io` and fix directory ownership.** Either move genotype-path orchestration (`run_dstat`, `run_dates`, `run_qpfstats`, `run_extract_f2`, `read_canonical_tile`) into `steppe_extract`, or update `architecture.md` §4 to honestly state that `core` depends on `io`. Relocate `f2_dir_io.cpp`, `pop_resolver.cpp`, `extract_f2_core.cpp`, and `f2_dir_writer.cpp` from `src/app/` into `src/access/` and `src/extract/` (`01_architecture_and_layering.md`, §1, §3, §4).

4. **Add CI and make formatting/linting/blocking.** Create `.github/workflows/ci.yml` with jobs for host-only build, CUDA build+test, Python wheel build+test, format/lint, and sanitizer runs. Until CI exists, `.clang-format`, `.clang-tidy`, and the warning matrix are decorations (`06_build_system_and_tooling.md`, §4, §6; `07_testing_and_performance.md`, §1, §8).

5. **Build one shared output layer and force every command through it.** Create `app/output_sink.hpp` with `OutputSink`, `CsvRowWriter`, `JsonWriter`, and `DoubleFormatter`. Retrofit `cmd_dates.cpp`, `cmd_qpgraph.cpp`, `cmd_fstat_sweep.cpp`, `cmd_extract_f2.cpp`, and `cmd_qpfstats.cpp` to use it. Today the same double renders as `0.12345678901234568`, `0.123457`, `0.123456`, or `0.1235` depending on the command (`05_io_cli_and_output_ux.md`, §4, §9.A).

6. **Fix the error contract end-to-end.** Introduce a central `struct Error { Status status; std::string message; }`, add `Status::IoError` and `Status::CudaRuntime`, implement `exception_to_status()`, and make every CLI command return the correct exit code (`kExitDeviceOom` for `cudaErrorMemoryAllocation`, `kExitIoError` for disk-full writes). Stop using `std::runtime_error` for normal control flow (`03_error_handling_and_robustness.md`, §1, §2, §5; `05_io_cli_and_output_ux.md`, §5).

7. **Wire CUDA determinism and add a second compute stream.** Implement `cusolverDnSetDeterministicMode` as documented in `architecture.md` §12. Add a second compute stream with explicit `Event` dependencies so decode/upload/GEMM can overlap with reductions. This is the highest-impact CUDA performance/determinism fix (`04_cuda_engineering.md`, §3.3, §4.3, §9.2, Roadmap P0).

8. **Refactor public result structs and Python ergonomics.** Nest `WeightsTable`, `RankDropTable`, `PopDropTable` inside `QpAdmResult`; do the same for `QpGraphResult` and `DatesResult`. Add `F2BlockTensor::f2_at(i,j,b)`. Remove polymorphic Python returns (`qpadm_search(..., as_dataframe=True)` returning either a list or a DataFrame). Add named `Precision` factories (`fp64()`, `emulated_fp64()`, `tf32()`) (`08_public_api_and_library_design.md`, §1, §3, §6, §7, §9).

9. **Refresh documentation and project hygiene.** Archive `docs/cleanup/`, `docs/release_cleanup/`, and `docs/kimireview/` out of the browsable tree. Split `docs/architecture.md` into a short current spec plus per-subsystem design notes. Update `NEXT-STEPS.md`, `TODO.md`, `ROADMAP.md`, `RESUME.md`, and `RELEASE-SCOPE.md` to current HEAD or delete them. Add `/aadr/`, `.agents/`, `.claude/`, `.codex/` to `.gitignore`. Add `NOTICE`/`ATTRIBUTION.md` (`09_documentation_and_project_hygiene.md`, §2, §5, §6, §7, §9).

10. **Finish multi-GPU fit/rotation.** Implement the per-device precompute path described in the `TODO(multigpu-host-bounce)` at `src/core/qpadm/model_search.cpp:167-191` so each GPU builds its own `f2_blocks` and eliminates `replicate_f2`. Replace static contiguous model sharding with a dynamic atomic work queue. Encapsulate peer topology in a `PeerTopology` object. This turns the honestly-deferred S8 story into a real speedup story (`01_architecture_and_layering.md`, §6; `04_cuda_engineering.md`, §8; `07_testing_and_performance.md`, §3).

## Phased roadmap

### Phase 0 — Foundational hygiene (1–2 weeks)

- Refresh project-status docs (`NEXT-STEPS.md`, `TODO.md`, `ROADMAP.md`, `RESUME.md`, `RELEASE-SCOPE.md`) to current HEAD or archive them.
- Complete `docs/RUN-SHEET.md` with sections for all 14 subcommands, including `qpgraph`, `qpgraph-search`, `dates`, and `qpfstats`.
- Clean the root: add `/aadr/`, `.agents/`, `.claude/`, `.codex/` to `.gitignore` (or remove the dirs); relocate `agentscripts/`; document `experiments/` as spikes.
- Add `STEPPE_WERROR` option, fix the duplicated version literal between `CMakeLists.txt` and `pyproject.toml`, and replace the placeholder CPM hash.
- Add a minimal formatting/lint check script (`clang-format --dry-run --Werror`, `cmake-format --check`) as a stop-gap until CI lands.

### Phase 1 — Structural fixes (2–4 weeks)

- Split `ComputeBackend` into role-specific interfaces and remove the `core::qpadm` upward references from `src/device/backend.hpp`.
- Decouple `core` from `io`: move genotype-path orchestration into `steppe_extract` or update `architecture.md` honestly; relocate shared files out of `src/app/`.
- Centralize `kPrimaryGpu` / `primary_backend()` in `device/resources.hpp` and extract the shared genotype decode front-end used by `run_dstat` and `run_qpfstats`.
- Introduce the central `Error` struct, complete the `Status` taxonomy (`IoError`, `CudaRuntime`), and implement `exception_to_status()` with correct CLI exit codes.
- Begin the unified output layer: land `OutputSink`, `CsvRowWriter`, `JsonWriter`, and `DoubleFormatter`; migrate the highest-leakage commands first (`dates`, `qpgraph`, sweeps).

### Phase 2 — Polish and infrastructure (1–2 months)

- Implement the C ABI shim (`src/c_api/`) and mark the C++ headers as same-toolchain-only.
- Refactor public result structs into composed tables; add `Precision` factories and `F2BlockTensor` index helpers; remove polymorphic Python returns.
- Enforce atomic writes, parent-directory creation, stream-state checks, and `--force`/`--no-clobber` semantics across all commands.
- Stand up `.github/workflows/ci.yml` with host-only build, CUDA build+test, Python wheel+pytest, format/lint, and sanitizer jobs.
- Wire `cusolverDnSetDeterministicMode`, add the second compute stream, and add NVTX ranges behind a `STEPPE_NVTX` option.
- Add compute-sanitizer memcheck/racecheck/synccheck to CI, move binary fixtures to Git LFS, and add a shared golden loader/schema.

### Phase 3 — Ecosystem (ongoing)

- Finish multi-GPU rotation: per-device precompute, dynamic model queue, `PeerTopology`, and NCCL `AllReduce`/tree combine.
- Add `cibuildwheel` for Linux CUDA wheels and publish the Python wheel pipeline.
- Build a perf regression harness (Google Benchmark or nvbench) with stored baselines and nightly runs.
- Generate API docs (Sphinx for Python, Doxygen/Breathe for C++) and publish a quick-start example in `examples/`.
- Add an analytical/small-model oracle test suite as a third correctness signal independent of AT2.

## Closing note: what to highlight vs. what to fix first

**Highlight with confidence** when showing the project to senior engineers: the compiler-enforced CUDA-free seam, the strong RAII ownership model, the two-oracle correctness strategy (CUDA vs. CPU oracle and vs. AT2 goldens), the real-AADR performance discipline, and the broad CLI/Python surface. These are genuinely unusual strengths for a research HPC codebase.

**Fix before you ask senior engineers to endorse a release or a public repo:** the missing CI, the stale project-status docs, the `core`→`io` layering lie, the 1,857-line `ComputeBackend` god interface, the half-centralized output layer, silent file overwrites, the unimplemented C ABI, and the root-level clutter. Those are the items that will dominate a code review and that signal “not production yet” louder than any algorithmic detail.

# steppe — First-Principles Testing & Performance Assessment

**Scope:** read-only assessment of the `tests/` tree, reference goldens, performance docs, and supporting scripts. The goal is to judge the project against an A+ testing/performance strategy from first principles, point out concrete gaps, and propose a roadmap.

**Key files reviewed**

- `tests/CMakeLists.txt` (the test manifest; ~1,700 lines of dense commentary)
- `tests/unit/*.cpp` — samples: `test_f2.cpp`, `test_geno_reader.cpp`, `test_backend_factory.cpp`, `test_f2_partials_validate.cpp`, `test_f2_combine.cpp`, `test_shard_plan.cpp`, `test_f2_blocks_multigpu.cpp`
- `tests/cli/*.cpp` — samples: `test_cli_f4.cpp`, the full CLI list
- `tests/reference/*.cu` — samples: `test_f2_equivalence.cu`, `test_qpadm_parity.cu`, `bench_f2_multigpu.cu`
- `tests/reference/goldens/at2/README.md` and golden layout
- `docs/perf/1240k-sweep.md`, `docs/perf/fstats-sweep.md`, `docs/perf/production-pass.md`
- `docs/feature-matrix.md`, `docs/NEXT-STEPS.md`
- `scripts/box_bringup.sh`, `pyproject.toml`, top-level `CMakeLists.txt`

---

## 1. Test pyramid

### What exists

| Layer | Count | What it covers | File references |
|---|---|---|---|
| **Unit** | ~20 `.cpp` TUs | Host-only, GPU-free seams: f2 estimator primitives, decode primitives, launch math, config defaults, block partition/ranges, shard planner, f2 combine validator, device-order validation, VRAM budget, filters, geno/snp readers, model-search shard planner, backend factory signatures | `tests/unit/test_f2.cpp`, `test_decode.cpp`, `test_launch_config.cpp`, `test_config.cpp`, `test_block_partition.cpp`, `test_block_ranges.cpp`, `test_shard_plan.cpp`, `test_f2_partials_validate.cpp`, `test_f2_combine.cpp`, `test_f2_blocks_multigpu.cpp`, `test_validate_device_order.cpp`, `test_vram_budget.cpp`, `test_filters.cpp`, `test_geno_reader.cpp`, `test_snp_reader.cpp`, `test_model_search_core.cpp`, `test_backend_factory.cpp`, `test_backend_capabilities.cpp`, `test_config_builder.cpp`, `test_f2_from_blocks.cpp` |
| **Reference / integration** | ~39 `.cu`/`.cpp` TUs | GPU-vs-CPU equivalence on real AADR; f2, decode, format readers (TGENO/PA/EIGENSTRAT/PLINK/ANCESTRYMAP), filters, f2 blocks, multi-GPU parity, determinism, backend capabilities/resources, qpAdm/qpWave/qpGraph/qpfstats/DATES parity, f-stat sweep parity | `tests/reference/test_f2_equivalence.cu`, `test_decode_equivalence.cu`, `test_f2_blocks_equivalence.cu`, `test_f2_multigpu_parity.cu`, `test_qpadm_parity.cu`, `test_qpwave_parity.cu`, `test_qpgraph_parity.cu`, `test_qpgraph_search_parity.cu`, `test_qpfstats_fused_parity.cu`, `test_dates_parity.cu`, `test_fstat_sweep_parity.cu`, `test_f2_determinism.cu` |
| **CLI end-to-end** | 16 `.cpp` TUs | Spawn the built `steppe` binary and diff stdout/CSV/JSON against goldens for `extract-f2`, `qpadm`, `qpwave`, `qpadm-rotate`, `f4`, `f3`, `f4-ratio`, `qpdstat`, `qpfstats`, `qpgraph`, `dates`, and format-prefix auto-cover gates | `tests/cli/test_cli_*.cpp` |
| **Python bindings** | 11 pytest files | Real-AADR smoke through the nanobind module | `tests/python/test_py_*.py` |
| **Manual benchmarks** | 4 `.cu` executables | Not ctest gates; used for perf docs | `tests/reference/bench_f2_multigpu.cu`, `bench_fstats_1240k.cu`, `bench_rotation_1240k.cu`, `bench_optimizers.cu` |

The manifest (`tests/CMakeLists.txt`) is well-structured and extensively commented. Each `add_test` states the acceptance criterion, the architectural contract it pins, and the data requirement. Many tests are **self-checking executables** (return non-zero on failure) rather than GoogleTest TUs; this is a deliberate choice so the M0/M1 gates build before GTest is fetched. GoogleTest is auto-detected and used for unit tests when present.

### Strengths

- Strong **layering discipline**: unit tests deliberately avoid CUDA headers to prove the core→device seam is CUDA-free (`test_backend_factory.cpp`, `test_f2_from_blocks.cpp`, `test_f2_blocks_multigpu.cpp`).
- GPU tests usually compare **steppe against steppe** (CUDA backend vs. CPU long-double oracle) and **steppe against AT2** goldens, giving two independent correctness signals.
- Real-AADR-only policy for precision/throughput claims is documented and enforced in test comments.

### Weaknesses

- **No fuzz/property-based layer.** Inputs are hand-built or drawn from one real dataset. Edge cases like malformed headers are covered, but combinatorial or randomized stress is absent.
- **No mutation testing** and no coverage tooling integration (gcov, nvcc `-lineinfo` coverage, etc.).
- The self-checking executable pattern is fine, but it means no rich test fixtures/parametrization; some test code is verbose and repeated across `.cu` files.

### Verdict

The pyramid is **solid B+/A- for a research HPC project**: broad reference coverage, good unit coverage of seams, CLI smoke, Python smoke. It is missing the fuzz/stress apex and coverage instrumentation that would make it A+.

---

## 2. Golden-file testing strategy and maintainability

### What exists

Goldens live under `tests/reference/goldens/`:

- `at2/` — ADMIXTOOLS 2 reference for qpAdm/qpWave/qpGraph/f-stats/qpfstats.
- `dates/` — DATES reference outputs.

`tests/reference/goldens/at2/README.md` is exemplary: it documents the AT2 version (2.0.10), R version (4.3.3), RNG seed, dataset prefix, `extract_f2` parameters, `qpadm` parameters, the model, why it was chosen, the convertf-vs-TGENO correction, and the regeneration/fixup scripts. It separates the **directory-path canonical golden** from the **f2-object-path fixture** and explains the ~1e-5 caveat.

Committed artifacts include:

- `golden_fit0.json`, `golden_fitNA.json`, `golden_rot.json`, `golden_qpwave.json`
- CSV tables: weights, rankdrop, popdrop, f4, X, Q, dstat, f3, f4-ratio, qpfstats
- Binary fixtures: `f2_fit0_9pop.bin`, `f2_fit1_NRBIG.bin`, `f2_fitNA.bin`, `f2_qpgraph_9pop.bin`, `f2_rot.bin`
- R scripts: `golden_fit0_generate.R`, `golden_fit0_fixup.R`, `verify_bitexact.R`, etc.

### Strengths

- **Metadata-rich.** Every golden carries provenance in JSON/CSV/README.
- **Regeneration scripts are committed**, not just the outputs.
- **Independent verification script** (`verify_bitexact.R`) re-runs the canonical invocation.
- **Format-reader goldens are transitive:** PA==TGENO==goldens, EIGENSTRAT==PA, PLINK==PA, ANCESTRYMAP==PA. This avoids maintaining a separate statistical oracle for each input format.

### Weaknesses / risks

- **Binary fixtures in git.** The `.bin` files are tens of MB; they will bloat history if regenerated often. Git LFS is not mentioned.
- **Golden regeneration requires a specific box setup:** R 4.3.3, AT2 2.0.10, convertf v8621, the real AADR files, and the convertf-PA workaround. The README documents it, but there is no automated "re-golden" script that a CI job can run.
- **No golden schema validation.** JSON/CSV structure is asserted by hand in each test; a shared golden loader + schema would reduce duplication.
- **TGENO-vs-PA trap is documented but not automatically prevented.** NEXT-STEPS mentions a CI guard to reject TGENO+AT2-R goldens; this is not yet implemented.

### Verdict

Golden strategy is **industry-good for a small team** but not yet "self-healing." To reach A+, add Git LFS, a single `regenerate_goldens.sh`, JSON schema checks, and the TGENO+AT2-R guard.

---

## 3. Coverage of CUDA code paths

### What exists

CUDA coverage is unusually thorough for a project of this size:

- **Precision lanes:** most GPU tests run both native FP64 and EmulatedFp64{40} (`test_f2_equivalence.cu`, `test_f2_blocks_equivalence.cu`).
- **Multi-GPU lanes:** single-GPU, 2-GPU host-staged, and 2-GPU P2P device-resident combine are all asserted bit-identical (`test_f2_multigpu_parity.cu`).
- **Capability probes:** `test_backend_capabilities_probe.cu`, `test_resources_build.cu`, `test_cuda_check.cu`, `test_handles.cu`, `test_device_buffer.cu`.
- **Guard/death tests:** degenerate extents (`test_f2_empty_guard.cu`), `M > INT_MAX` narrowing (`test_f2_int_max_guard.cu`), DeviceBuffer size wrap (`test_device_buffer.cu`), grouped-batch empty guards (`test_f2_blocks_group.cu`).
- **Determinism:** `test_f2_determinism.cu` asserts bit-identical EmulatedFp64 run-to-run.
- **Format decode on GPU:** five format-reader equivalence tests exercise the GPU decode path.
- **Backend honorability:** `test_emu_honorability.cu` pins the fixed-slice Ozaki vs. native downgrade predicate.

### Strengths

- The tests exercise **production seams**, not inline re-implementations. This is explicitly called out in `test_f2_equivalence.cu` as a post-fix improvement.
- Multi-GPU parity is not just "it runs"; it asserts **bit-identical results** across combine tiers, which is the strongest possible contract.

### Weaknesses / gaps

- **No compute-sanitizer / memcheck / racecheck integration.** CUDA-specific memory errors are caught by parity and guard tests, but not by tooling.
- **No line/branch coverage for `.cu` files.** There is no indication that `gcov`/`nvcc -coverage` or `ncu`/`nsys` are run in any automated way.
- **No synthetic adversarial GPU kernels.** All GPU inputs are real or small hand-built; randomized malformed Q/V/N, extreme aspect ratios, or NaN/Inf payloads are not systematically probed.

### Verdict

CUDA functional coverage is **A-.** To reach A+, add compute-sanitizer runs in CI, instrument coverage for device code, and add adversarial GPU stress tests.

---

## 4. Performance measurement and regression prevention

### What exists

Performance documentation is rich and honest:

- `docs/perf/1240k-sweep.md` — extract-f2 + qpAdm + rotation scaling on real AADR 1240K, with corrections and root causes.
- `docs/perf/fstats-sweep.md` — f4/f3/f4-ratio/qpDstat scaling, including the OOM finding and the proposed diagonal-only fix.
- `docs/perf/production-pass.md` — golden-parity roll-up + production-scale table for every tool.
- `docs/feature-matrix.md` — real-AADR smoke + wall-clock matrix.

Measurement discipline is good:

- Release build enforced (`CMAKE_BUILD_TYPE=Release`, `pyproject.toml` pins it).
- `/usr/bin/time -v` for wall + MaxRSS, `nvidia-smi` for GPU util/VRAM.
- Median-of-3 after warm-up.
- Every number is anchored to a real dataset and a committed golden.

### Weaknesses

- **No automated perf regression gates.** The benchmark executables are `add_executable` only; they are not `add_test`, so `ctest` will not fail if performance regresses.
- **No perf database / history.** Results live in markdown files; there is no JSON/CSV timeseries or plotting.
- **No micro-benchmark framework.** Manual timing loops in `.cu` files are acceptable but do not provide stable statistics, outlier rejection, or fixture parameterization.
- **No automated Release-only build check.** The docs repeatedly warn that Debug builds void timing, but nothing in CMake or CI rejects a Debug perf run.

### Verdict

Measurement practice is **A for honesty and documentation, C for automation.** To reach A+, make benchmarks ctest-gated (with generous tolerances), store historical results, and adopt a framework like Google Benchmark or nvbench.

---

## 5. Benchmarking infrastructure

### What exists

Four manual benchmark TUs:

1. `bench_f2_multigpu.cu` — OOM-tolerant ascending P sweep for single/multi-GPU f2 precompute; includes tiered (host/disk) mode.
2. `bench_rotation_1240k.cu` — batched qpAdm rotation throughput on real 1240K f2.
3. `bench_fstats_1240k.cu` — f4/f3/f4-ratio/qpDstat scaling.
4. `bench_optimizers.cu` — qpGraph GPU optimizer spike (IDEA1 vs IDEA2).

These are built by CMake but are **not registered as tests**.

### Strengths

- Benchmarks use real AADR data and report scaling, not just single points.
- `bench_f2_multigpu.cu` is OOM-tolerant and catches failures per cell, which is appropriate for an envelope-finding sweep.

### Weaknesses

- No benchmark framework; each bench re-implements warm-up, median, timing loops.
- No automatic comparison against a baseline.
- No separation of "micro" (kernel-level) vs. "macro" (end-to-end) benchmarks.
- No `nsys`/`ncu` profiling integration.

### Verdict

**B.** Functional but artisanal. A+ would use a framework, baseline comparisons, and CI-triggered profiling.

---

## 6. Correctness validation against ADMIXTOOLS 2

### What exists

This is a project strength. AT2 parity is pursued end-to-end:

- `tests/reference/goldens/at2/` generated from AT2 2.0.10 / R 4.3.3 on real AADR.
- Parity tests exist for: qpAdm (`test_qpadm_parity.cu`), qpWave (`test_qpwave_parity.cu`), qpGraph single-graph (`test_qpgraph_parity.cu`) and topology search (`test_qpgraph_search_parity.cu`), f4/f3/f4-ratio/qpdstat (CLI tests), qpfstats (`test_qpfstats_fused_parity.cu`, `test_cli_qpfstats`), DATES (`test_cli_dates`).
- Tolerances are documented and tiered (tight rtol 1e-6, loose rtol 1e-3, exact for dof/rank).
- The README honestly records the earlier TGENO-misread golden corruption and the convertf fix.

### Strengths

- **Two-oracle design:** steppe CUDA vs. steppe CPU oracle, and steppe vs. AT2 golden. This localizes bugs quickly.
- **Real-data-only policy** avoids synthetic precision traps.
- **Regeneration scripts + verification script** give a reproducible path back to AT2.

### Weaknesses

- **AT2 is treated as ground truth, but AT2 itself has documented quirks** (TGENO misread, directory-path vs. f2-object-path differences). The project handles this well, but a third independent oracle (e.g., analytical small cases) is absent.
- **No continuous re-validation against AT2.** If AT2 releases a new version, the committed goldens could drift. There is no nightly "regenerate and diff" job.

### Verdict

**A.** Among the strongest AT2-parity testing setups one is likely to see in a research codebase. A+ would add a nightly regeneration job and a small analytical oracle.

---

## 7. Determinism / reproducibility testing

### What exists

- `test_f2_determinism.cu` runs EmulatedFp64 compute twice and asserts bit-identical results.
- `test_f2_multigpu_parity.cu` asserts bit-identical results across G=1/2 and across combine tiers.
- `test_qpgraph_search.cu` / `test_qpgraph_search_parity.cu` assert deterministic global-best argmin vs. AT2.
- `test_qpadm_rotation.cu` asserts G=1 vs G=2 bit-identical + identically-ordered determinism.
- cuBLAS handle/workspace determinism is tested indirectly through these parity tests.

### Strengths

- Determinism is treated as a first-class property, not assumed.
- The emulated-FP64 path (the production default) is specifically targeted because it is the precision lane most sensitive to workspace ordering.

### Weaknesses

- **No standalone seed/RNG determinism test** for `qpGraph` optimizer stochasticity (the optimizer spike uses curand but is not a ctest gate).
- **No cross-run determinism test for the full CLI** (each CLI test spawns one process; inter-process determinism is not asserted).
- **No checksum regression test** for output files (e.g., sha256 of `f2.bin` produced by `extract-f2`).

### Verdict

**A-.** Strong in-process determinism coverage. A+ would add CLI cross-run checksums and RNG seed tests.

---

## 8. What testing/perf infrastructure is needed for an A+ project

An A+ project in this domain would have:

1. **Continuous integration with a GPU runner** (or a "skip on no GPU" matrix) that builds Release and runs `ctest`.
2. **Automated golden regeneration** — a single script + CI job that re-runs AT2 on the canonical dataset and produces a diff report.
3. **Code coverage** — host coverage via `gcov` + `.gcno`/`.gcda`, and device coverage via `nvcc -lineinfo` + `ncu` or `cuda-memcheck --tool coverage` where available.
4. **CUDA sanitizers in CI** — `compute-sanitizer --tool memcheck`, `--tool racecheck`, `--tool synccheck` on a representative subset of tests.
5. **Performance regression suite** — convert the four manual benches to `add_test` with stored baselines and fail on statistically significant regressions.
6. **Benchmark framework** — adopt Google Benchmark or nvbench; replace hand-rolled timing loops.
7. **Fuzz / property tests** — especially for `io::GenoReader`, format detection, block partition, and the f2 combine validator.
8. **CLI reproducibility checksums** — run `extract-f2` and selected CLI commands twice and compare output file digests.
9. **Wheel/integration tests for Python** — install the built wheel in a clean venv and run the pytest suite against it.
10. **Nightly/stress tests** — large-P multi-GPU f2 precompute, full rotation pool sweeps, large f-stat sweeps.
11. **Schema validation for goldens** — JSON schema + CSV column contract checks, shared across tests.
12. **Documentation tests** — assert every CLI subcommand has `--help`, no stale "not yet implemented" strings, and JSON output matches a schema.

---

## Roadmap to A+ testing/perf

### Immediate (next 1–2 weeks)

| # | Action | Owner/files | Impact |
|---|---|---|---|
| 1 | Add a top-level `regenerate_goldens.sh` that documents the exact R/AT2/convertf invocation and writes to `tests/reference/goldens/at2/` | New script + README update | Self-healing goldens |
| 2 | Move binary fixtures to Git LFS | `.gitattributes` | Prevents repo bloat |
| 3 | Add a golden schema/loader shared by all reference tests | `tests/reference/golden_loader.hpp/.cpp` | Reduces test duplication |
| 4 | Convert the four manual benches to `add_test` with generous pass/fail tolerances | `tests/CMakeLists.txt` | Perf regression detection |
| 5 | Add a Release-build assertion in CMake when benchmarks run | `cmake/SteppeOptions.cmake` | Stops invalid Debug perf runs |

### Short-term (next month)

| # | Action | Owner/files | Impact |
|---|---|---|---|
| 6 | Set up CI (GitHub Actions or self-hosted) with a GPU runner running `ctest` Release | `.github/workflows/` or equivalent | Prevents regressions |
| 7 | Add host code coverage (`-fprofile-arcs -ftest-coverage`) to CI | CMake + CI | Visibility into untested seams |
| 8 | Add `compute-sanitizer` memcheck/racecheck/synccheck to CI on a subset of CUDA tests | CI script | Catches device memory/race bugs |
| 9 | Add a nightly/stress `ctest -L stress` label for large-P multi-GPU and rotation pool200 | `tests/CMakeLists.txt` + CI | Catches scale regressions |
| 10 | Implement the TGENO+AT2-R golden guard mentioned in NEXT-STEPS | Golden generation + CI | Prevents golden corruption |
| 11 | Add checksum determinism tests for CLI outputs | New `tests/cli/test_cli_determinism.cpp` | Catches non-determinism in packaging |

### Medium-term (next quarter)

| # | Action | Owner/files | Impact |
|---|---|---|---|
| 12 | Adopt a C++/CUDA benchmark framework (Google Benchmark or nvbench) | New `tests/bench/` directory | Stable, comparable perf numbers |
| 13 | Build a perf database + dashboard (even a simple JSON file + plot script) | `docs/perf/` or `benchmarks/` | Historical regression tracking |
| 14 | Add property-based / fuzz tests for format readers and filters | New `tests/fuzz/` | Finds edge-case bugs |
| 15 | Add Python wheel integration tests in CI | `tests/python/` + CI | Validates packaging |
| 16 | Add analytical/small-model oracle tests (e.g., 2-pop f2 by hand) | New reference tests | Third correctness signal |
| 17 | Add `nsys`/`ncu`-driven profiling gates for hot kernels | CI + docs | Catches unexpected kernel regressions |

---

## Summary

The `steppe` project has a **strong, thoughtfully layered test suite** with excellent AT2 parity, real-data discipline, multi-GPU bit-identity checks, and honest performance documentation. Its biggest gaps are **automation**: no CI, no coverage, no automated perf regression, no compute-sanitizer runs, and golden regeneration is still manual. Closing those gaps with the roadmap above would raise the project from a solid B+/A- to a true A+ testing and performance posture.

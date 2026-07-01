# Architecture & Layering Assessment — `steppe`

**Scope:** first-principles read-only review of the C++/CUDA codebase, focused on module layering, the CUDA-free seam, public API surface, dependency direction, config/resources injection, and multi-GPU maturity.

**Sources inspected:**
- `docs/architecture.md` (the canonical spec)
- `CMakeLists.txt`, `include/CMakeLists.txt`, `src/device/CMakeLists.txt`, `src/core/CMakeLists.txt`, `src/io/CMakeLists.txt`, `src/app/CMakeLists.txt`, `bindings/CMakeLists.txt`, `src/access/CMakeLists.txt`, `src/extract/CMakeLists.txt`
- `src/device/backend.hpp`, `src/device/resources.hpp`, `src/device/shard_plan.hpp`, `src/device/tier_select.hpp`, `src/device/vram_budget.hpp`, `src/device/backend_factory.hpp`
- `src/core/config/run_config.hpp`
- All `include/steppe/*.hpp` public headers
- Spot checks of `src/core/stats/dstat.cpp`, `src/core/stats/read_canonical_tile.cpp`, `src/core/qpadm/model_search.cpp`, `bindings/module.cpp`

**Overall grade: B+.**

The project has a credible, compiler-enforced layered architecture, a genuine CUDA-free seam, RAII resource ownership, and no global mutable state. It is production-credible for a single-node GPU qpAdm pipeline. What keeps it from an A+ is (1) the `ComputeBackend` seam has become a “god interface,” (2) `core` and `io` are more tangled than the architecture doc claims, (3) the public API does not match the documented C ABI promise, and (4) multi-GPU fit/rotation is honestly deferred rather than fully realized.

---

## 1. Separation of concerns

### What is strong

The directory/target map is clean on paper:

| Layer | Role | CMake target |
|---|---|---|
| `include/steppe/` | Public, CUDA-free C++ API | `steppe_api` (INTERFACE) |
| `src/io/` | Genotype decode + QC leaf | `steppe_io` |
| `src/core/internal/` + `src/core/domain/` | Shared host/device helpers, single-source domain rules | `steppe_core_internal` (INTERFACE) |
| `src/core/` | Host orchestration of compute | `steppe_core` |
| `src/device/` | CUDA backends + CPU reference | `steppe_device` |
| `src/access/` | Shared f2-dir reader + pop resolver | `steppe_access` |
| `src/extract/` | Shared genotype→f2 chain + dir writer | `steppe_extract` |
| `src/app/` | CLI | `steppe_app` |
| `bindings/` | Python module | `_core` |

CUDA is `PRIVATE` to `steppe_device` (`src/device/CMakeLists.txt:63-65`), which is the structural guarantee that the rest of the tree cannot accidentally compile a CUDA header. The `io` layer is explicitly designed as a leaf and does not link `core` or `device` (`src/io/CMakeLists.txt:38-40`).

### Concrete problems

1. **`core` knows about `io`, contrary to the “app wires io” rule.**
   - `src/core/CMakeLists.txt:153` links `steppe::io` PRIVATE because `src/core/stats/dstat.cpp`, `src/core/stats/qpfstats.cpp`, `src/core/stats/dates.cpp`, and `src/core/stats/read_canonical_tile.cpp` all directly include `io/` headers and orchestrate genotype I/O.
   - Example: `src/core/stats/dstat.cpp:47-52` includes `io/eigenstrat_format.hpp`, `io/geno_reader.hpp`, `io/genotype_source.hpp`, etc.
   - The architecture doc (`architecture.md:239`) says: *“The `app` layer is the only place that wires `io` output into compute.”* That statement is no longer true. The as-built design moved the genotype-path entry points (`run_dstat`, `run_dates`, `run_qpfstats`, `run_extract_f2`) into `core`/`extract`, so `core` is no longer “pure host compute with no I/O.”

2. **`src/app/` still owns source files that are compiled by other libraries.**
   - `src/access/CMakeLists.txt:24-25` compiles `../app/f2_dir_io.cpp` and `../app/pop_resolver.cpp`.
   - `src/extract/CMakeLists.txt:25-26` compiles `../app/extract_f2_core.cpp` and `../app/f2_dir_writer.cpp`.
   - This works, but it is a directory-ownership smell. The shared code should live in `src/access/` and `src/extract/`, and `app` should consume it via the libraries, not vice versa.

3. **`read_canonical_tile.cpp` is a cross-layer bridge that lives in `core`.**
   - `src/core/stats/read_canonical_tile.cpp` takes an `io::GenoReader`, calls `backend.transpose_to_canonical()`, and returns an `io::GenotypeTile`. It is literally the wiring point between `io` and `device` that the architecture doc says should be in `app`.

### Recommendations

- Either move all genotype-I/O orchestration (`run_dstat`, `run_dates`, `run_qpfstats`, `run_extract_f2`, `read_canonical_tile`) into `steppe_extract` and keep `steppe_core` purely f2/fit, **or** update `architecture.md` §4 to honestly state that `core` depends on `io` for the genotype-path commands.
- Relocate `f2_dir_io.cpp`, `pop_resolver.cpp`, `extract_f2_core.cpp`, and `f2_dir_writer.cpp` into `src/access/` and `src/extract/` directories; remove the `../app` source references from those CMake targets.

---

## 2. CUDA-free seam design — enforced, but bloated

### What is strong

The seam is real and is actually enforced by the compiler:

- `src/device/backend.hpp` is CUDA-free. It uses only standard types, `core/internal/views.hpp`, and the opaque device handles (`DeviceF2Blocks`, `DevicePartial`, `DeviceDecodeResult`, `StreamTarget`).
- A grep for `#include <cuda_*`, `<cublas_*`, `<cusolver_*`, `<cufft.h>`, `<cub/...>` across `src/core/`, `src/app/`, `include/`, `src/access/`, `src/extract/`, and `bindings/` (excluding tests) returns **zero hits**. All CUDA headers are confined to `src/device/cuda/` and `scripts/`/`experiments/`.
- The opaque handles use PIMPL (`struct Impl`) so the CUDA allocations stay in `.cu` files; the headers are CUDA-free (`device_f2_blocks.hpp`, `device_partial.hpp`, `device_decode_result.hpp`).
- `core/CMakeLists.txt:157-164` sets `CUDA_RESOLVE_DEVICE_SYMBOLS OFF` so `core` does not archive device symbols; the CUDA dependency is purely through the `ComputeBackend` vtable.

### Concrete problems

1. **`ComputeBackend` is a god interface.**
   - The header is 1,857 lines long (`src/device/backend.hpp`). The abstract class spans lines 670–1853 and declares roughly 40 pure/semi-pure virtuals covering:
     - precompute (`compute_f2`, `compute_f2_blocks`, `compute_f2_blocks_device`, `compute_f2_blocks_resident`, `compute_f2_blocks_into`, `compute_f2_blocks_streamed`)
     - decode/transpose (`decode_af`, `detect_sample_ploidy_device`, `transpose_to_canonical`, `decode_af_compact_autosome`, `decode_af_compact_filter`)
     - fit engine (`assemble_f4`, `jackknife_cov`, `jackknife_diag`, `rank_test`, `rank_sweep`, `gls_weights`, `gls_weights_loo_batched`, `se_from_wmat`, `fit_models_batched`)
     - standalone stats (`assemble_f4_quartets`, `assemble_f3_triples`, `ratio_block_jackknife`, `f4ratio_blocks_jackknife`, `dstat_blocks_jackknife`, `f4_sweep`, `f3_sweep`)
     - qpGraph (`qpgraph_fit_fleet`, `qpgraph_fit_fleet_batch`)
     - DATES (`dates_curve`, `dates_repack`, `dates_fit`)
     - qpfstats (`qpfstats_smooth`, `qpfstats_blocks_smooth`)
     - capabilities/probing (`capabilities`, `batched_dispatch_count`)
   - Every new GPU feature extends this one seam. That forces `core`, `device`, `app`, `bindings`, and tests to recompile whenever the interface changes, and it makes alternative backends (e.g., a future ROCm port or a mock backend for unit tests) expensive to write.

2. **The seam header declares `core::qpadm` helpers — an upward reference.**
   - `src/device/backend.hpp:624-651` declares `core::qpadm::fit_models_batched_default` and `core::qpadm::model_in_small_path` so the base `fit_models_batched` virtual can throw a sentinel. A device-layer header referencing core-layer symbols is a layering inversion.
   - The base implementation of `fit_models_batched` (`backend.hpp:1810-1819`) explicitly throws rather than delegating, because delegating would pull `core` into every TU that instantiates the vtable.

3. **Runtime errors for “not supported by this backend” instead of compile-time interface segregation.**
   - Many base virtuals throw `std::runtime_error` (e.g., `compute_f2_blocks_device` at `backend.hpp:747-754`, `decode_af_compact_autosome` at `backend.hpp:1015-1022`). This is fine for guarding CUDA-only paths, but it means a test fake or new backend can silently fail at runtime if it forgets to override a method.

### Recommendations

- Split `ComputeBackend` into role-specific interfaces: `PrecomputeBackend`, `FitBackend`, `DecodeBackend`, `SweepBackend`, `DatesBackend`, `QpfstatsBackend`. `PerGpuResources` can hold a small struct that aggregates these interfaces, or keep a thin `ComputeBackend` facade for existing callers.
- Move `fit_models_batched_default` / `model_in_small_path` declarations into `src/core/qpadm/model_search.hpp` and pass a callback/delegate into the backend, rather than declaring core symbols inside `device/backend.hpp`.
- Where a method is truly backend-specific (e.g., device-resident output), prefer a pure virtual in a dedicated sub-interface so missing overrides are compile errors.

---

## 3. Public API surface vs internal implementation

### What is strong

- All public headers in `include/steppe/` are CUDA-free and forward-declare `device::DeviceF2Blocks` / `device::Resources`.
- The public types are value-oriented (`QpAdmModel`, `QpAdmResult`, `F2BlockTensor`, `SweepRequest`, etc.) and use indices instead of strings at the compute seam, which is exactly right for GPU-first code.
- `Precision`, `DeviceConfig`, `FilterConfig`, and the named constants live in one public place (`include/steppe/config.hpp`).

### Concrete problems

1. **The public API is C++, not the documented C ABI.**
   - `architecture.md:908-914` states: *“The installed/versioned boundary is a C ABI. `include/steppe/` exposes opaque handles … functions returning `steppe_status_t` … No `std::` types, no templates, and no exceptions cross this boundary.”*
   - The actual `include/steppe/` headers expose `std::vector`, `std::string`, `std::span`, `std::array`, and C++ structs/classes. For example:
     - `QpAdmResult` carries `std::vector<std::string> popdrop_pat` (`include/steppe/qpadm.hpp:149`).
     - `F2BlockTensor` is a plain struct with `std::vector<double>` (`include/steppe/fstats.hpp:47-78`).
     - `run_qpadm` takes `device::Resources&` and returns a C++ struct with a `Status` field.
   - There is no C ABI shim (`steppe_f2_blocks_t*`, `steppe_status_t` accessors) in the repo. The current public surface is a same-toolchain C++ convenience layer, not a stable ABI.

2. **`RunConfig` is an app config object living in `core/config`.**
   - `src/core/config/run_config.hpp:40-179` is a large value object with 40+ fields, including CLI-specific strings (`out_file`, `format`, `geno`, `snp`, `ind`, `graph_file`, `shard_dir`, etc.) alongside the real compute config (`DeviceConfig`, `QpAdmOptions`, `FilterConfig`, `PopSelection`).
   - `core` should not know about output filenames or CLI column flags. These belong in `app`/`bindings`.

3. **`PopSelection` is defined in `src/io/ind_reader.hpp` and pulled into `core/config/run_config.hpp`.**
   - `run_config.hpp:32` includes `io/ind_reader.hpp`. This is another sign that `core` and `io` are entangled.

### Recommendations

- Implement the promised C ABI layer under `src/c_api/` (or similar) with opaque handles and `steppe_status_t` accessors. Keep the current `include/steppe/` headers as an optional, same-toolchain C++ convenience wrapper, explicitly documented as not ABI-stable.
- Split `RunConfig` into a small, library-facing `ComputeConfig` (`DeviceConfig` + `QpAdmOptions` + `FilterConfig`) and an app-facing `CliConfig` for filenames/CLI-specific fields.
- Move `PopSelection` to a neutral header (e.g., `include/steppe/io_types.hpp`) or keep it in `io` and stop including it from `core/config`.

---

## 4. Dependency direction and link visibility

### What is strong

- The top-level `CMakeLists.txt:67-112` orders subdirectories correctly and gates `app`/bindings/tests behind options.
- `steppe_device` exports only CUDA-free headers publicly (`src/device/CMakeLists.txt:60-65`).
- `steppe_core` links `steppe::device` PRIVATE (`src/core/CMakeLists.txt:152`).
- `steppe_app` and `_core` are plain CXX targets with no CUDA language and no `.cu` sources, yet they can call GPU code through the CUDA-free seam.

### Concrete problems

1. **`core` → `io` is a downward dependency that the architecture doc does not acknowledge.**
   - `src/core/CMakeLists.txt:153` links `steppe::io` PRIVATE. As noted above, this is real and intentional for the genotype-path commands, but it contradicts the documented layering.

2. **`app` and `bindings` include headers from `src/app/`.**
   - `bindings/module.cpp:40-42` includes `app/f2_dir_io.hpp`, `app/f2_dir_writer.hpp`, and `app/pop_resolver.hpp`. These files are compiled into `steppe_access` and `steppe_extract`, but their headers still live in `app/`. This makes the Python binding logically dependent on the CLI app’s source tree.

3. **`extract` must duplicate the `CUDA_RESOLVE_DEVICE_SYMBOLS OFF` workaround.**
   - `src/extract/CMakeLists.txt:44-52` copies the same property logic from `core/CMakeLists.txt:157-164`. This is a brittle duplication; if one changes, the other can silently break with `__fatbinwrap_…` double-registration errors.

### Recommendations

- Document the actual dependency graph in `architecture.md` §4, or refactor genotype-path orchestration out of `core` to restore the documented graph.
- Move `f2_dir_io.hpp`, `pop_resolver.hpp`, `f2_dir_writer.hpp`, and `extract_f2_core.hpp` into `src/access/` and `src/extract/` headers. App and bindings should include `access/...` and `extract/...`, not `app/...`.
- Extract the `CUDA_RESOLVE_DEVICE_SYMBOLS OFF` setting into a reusable CMake function or a target property on `steppe::device` so consumers do not have to copy it.

---

## 5. Config/resources injection vs hidden state

### What is strong

- No singletons or global mutable state were found. A grep for `getInstance`, `std::call_once`, `std::once_flag`, or `extern … =` in `src/` returned nothing concerning (only a few function-local `static const` math constants and a one-time log flag).
- `RunConfig` is documented as immutable and const-after-build; accessors are `const noexcept` (`src/core/config/run_config.hpp:46-124`).
- `Resources` is injected into every compute entry point (e.g., `run_qpadm(f2, model, opts, resources)` in `include/steppe/qpadm.hpp:170-173`).
- `PerGpuResources` is move-only and owns the backend via `std::unique_ptr<ComputeBackend>` (`src/device/resources.hpp:131-145`).
- `build_resources` validates the device order, probes capabilities, and constructs backends in one place (`src/device/resources.hpp:200-229`).

### Concrete problems

1. **`Resources` carries mutable observability state.**
   - `src/device/resources.hpp:167` declares `CombinePath last_combine_path`.
   - `src/device/resources.hpp:174` declares `MultiGpuTimings last_multigpu_timings`.
   - These are updated by `compute_f2_blocks_multigpu`. This is reasonable runtime telemetry, but a purer design would return a `MultiGpuResult` metadata object and leave `Resources` strictly as a resource bundle.

2. **`ComputeBackend::capabilities()` has a non-pure default that returns zeros.**
   - `src/device/backend.hpp:1840-1842` returns `BackendCapabilities{}`. A CPU backend or a misconfigured backend silently reports “unknown” capability. Making this pure virtual would force every backend to be explicit about what it supports.

3. **`RunConfig` has a public default constructor.**
   - `src/core/config/run_config.hpp:42` provides `RunConfig() = default`, even though the comment says *“ConfigBuilder is the ONLY constructor of a validated RunConfig.”* The public default undermines the invariant.

4. **Python `F2Handle` caches a `Resources` inside a Python object.**
   - `bindings/module.cpp:76-86` holds `std::unique_ptr<sd::Resources> resources` inside `F2Handle`. This is localized state and acceptable, but it means resource lifetime is tied to a Python object rather than explicitly managed by the caller.

### Recommendations

- Return combine-path and timing metadata from `compute_f2_blocks_multigpu` as a struct, not by mutating `Resources`.
- Make `ComputeBackend::capabilities()` pure virtual; `CpuBackend` should return a capabilities object that explicitly says `device_count == 0`, `can_access_peer == false`, etc.
- Hide `RunConfig`’s default constructor, or document it as test-only.

---

## 6. Multi-GPU abstractions and their maturity

### What is strong

- The single-node multi-GPU model is well documented and honest:
  - `DeviceConfig::devices` pins both the GPU set and the fixed combine order (`include/steppe/config.hpp:337-344`).
  - `build_resources` validates duplicate/out-of-range ordinals (`src/device/resources.hpp:180-200`).
  - `shard_plan.hpp` provides a deterministic, block-aligned SNP shard plan (`plan_block_shards`, line 98).
  - `tier_select.hpp` selects Resident/HostRam/Disk tiers at runtime from free VRAM/RAM probes (`select_output_tier`, lines 136-149).
  - The precompute combine is fixed-order host-staged or device-resident P2P; NCCL is used only for broadcast, never for the parity-critical reduction (`architecture.md:760-770`).

### Concrete problems

1. **S8 model-space search multi-GPU is deferred and bounce-capped.**
   - `src/core/qpadm/model_search.cpp:167-191` contains a large `TODO(multigpu-host-bounce)` explaining that on no-P2P cards (consumer RTX 5090s), replicating `f2_blocks` to all GPUs costs ~8.72 GB / ~3.8 s through host, capping G2/G1 at ~1.21× for 9,086 real models.
   - The same TODO appears in `src/device/device_f2_blocks.hpp:92-97`.
   - The fix — per-device precompute so each GPU builds its own f2 — is described but not implemented.

2. **Model-space sharding is static contiguous partitioning.**
   - `src/core/qpadm/model_search.cpp:287-288` uses `plan_model_shards(n, G)` and assigns contiguous model sub-spans to each GPU. There is no dynamic work stealing and no sub-chunking for heterogeneous per-model cost. For a rotation where all models have the same shape this is fine, but it will load-imbalance once per-model SVD/workspace costs vary.

3. **`f2_blocks` is gathered to one device during precompute, then re-broadcast for rotation.**
   - As the `model_search.cpp` TODO notes, this gather-then-scatter round-trip is the root cause of the host-bounce. A mature multi-GPU design would avoid producing a single resident copy on one device only to replicate it.

### Recommendations

- Implement the per-device precompute path for S8: each GPU builds its own full `f2_blocks` from the genotype stream (or reads a cached f2-dir), eliminating `replicate_f2` entirely.
- Add a dynamic atomic work queue over model indices for the rotation, with deterministic result ordering via pre-sized result slots (the existing `scatter_into_slots` pattern is good; just make the work assignment dynamic).
- Encapsulate peer-topology discovery in a small `PeerTopology` object rather than scattering `can_access_peer` checks across `cuda_backend.cu` and `p2p_combine.cu`.

---

## 7. What would make this architecture A+

1. **Honor the documented C ABI.** Either implement the opaque-handle C boundary promised in `architecture.md:908-914`, or stop claiming the public surface is C ABI-stable.
2. **Slim the `ComputeBackend` seam.** Split it into focused sub-interfaces so the seam does not grow monolithically with every new tool.
3. **Decouple `core` from `io`.** Move genotype-path orchestration into `extract` or a new `genotype_path` library, leaving `core` as pure f2/fit orchestration.
4. **Finish multi-GPU fit/rotation.** Per-device precompute + dynamic model queue would turn the honestly-deferred S8 multi-GPU story into a real speedup story.
5. **Make `capabilities()` pure virtual** and return explicit “no CUDA” metadata from the CPU backend.
6. **Return multi-GPU metadata** from the entry point instead of mutating `Resources`.
7. **Clean up cross-directory source ownership.** Move shared files out of `src/app/` into their owning libraries.
8. **Align the public `Status` enum** with the richer taxonomy in `architecture.md:682-689` (add IO/CUDA-runtime categories or document why they are intentionally absent).
9. **Reduce header/comment bloat.** `backend.hpp` and `config.hpp` are loaded with internal ticket references and line-number citations that will go stale. A+ code lets the architecture doc carry the prose and keeps headers focused on contracts.

---

## Appendix: selected line references

- `CMakeLists.txt:14` — CMake minimum is 3.28, while the architecture doc targets 3.30.
- `CMakeLists.txt:67-112` — top-level target wiring.
- `src/device/CMakeLists.txt:22-89` — `steppe_device` target; CUDA kept PRIVATE.
- `src/core/CMakeLists.txt:148-164` — `steppe_core` links `steppe::device` PRIVATE and `steppe::io` PRIVATE.
- `src/io/CMakeLists.txt:38-40` — `steppe_io` links only `steppe::api` + warnings.
- `src/app/CMakeLists.txt:31-61` — CLI target compiles many command files and links shared libs.
- `src/access/CMakeLists.txt:24-25` — compiles `../app/f2_dir_io.cpp` and `../app/pop_resolver.cpp`.
- `src/extract/CMakeLists.txt:25-26` — compiles `../app/extract_f2_core.cpp` and `../app/f2_dir_writer.cpp`.
- `src/extract/CMakeLists.txt:44-52` — duplicated `CUDA_RESOLVE_DEVICE_SYMBOLS OFF` logic.
- `src/device/backend.hpp:670-1853` — the `ComputeBackend` abstract class (god interface).
- `src/device/backend.hpp:624-651` — declarations of `core::qpadm` helpers inside a device header.
- `src/device/backend.hpp:747-754` — example base virtual that throws for CUDA-only path.
- `src/device/resources.hpp:131-145` — `PerGpuResources`.
- `src/device/resources.hpp:151-178` — `Resources` with mutable `last_combine_path` / `last_multigpu_timings`.
- `src/device/resources.hpp:200-229` — `build_resources`.
- `src/device/shard_plan.hpp:38-100` — block-aligned sharding.
- `src/device/tier_select.hpp:33-149` — output-tier selection policy.
- `src/core/config/run_config.hpp:40-179` — large `RunConfig` value object.
- `include/steppe/config.hpp:337-473` — `DeviceConfig` and `FilterConfig`.
- `include/steppe/qpadm.hpp:121-163` — `QpAdmResult` with `std::vector<std::string>`.
- `include/steppe/fstats.hpp:47-78` — `F2BlockTensor` public C++ struct.
- `src/core/stats/dstat.cpp:47-52` — direct `io/` includes inside `core`.
- `src/core/qpadm/model_search.cpp:167-191` — honest `TODO(multigpu-host-bounce)`.
- `src/core/qpadm/model_search.cpp:282-288` — static model sharding + `replicate_f2`.
- `bindings/module.cpp:40-42`, `76-86` — binding depends on `app/` headers and caches `Resources`.
- `architecture.md:239` — claim that only `app` wires `io` into compute.
- `architecture.md:908-914` — C ABI promise that is not reflected in `include/steppe/`.

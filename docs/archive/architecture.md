# `steppe` — GPU/CUDA qpAdm: Architecture & Engineering Best Practices

## 0. Purpose & how to use this document

This is the canonical engineering reference for **`steppe`**: a CUDA Toolkit 13 reimplementation of the f-statistic / qpAdm machinery from [ADMIXTOOLS 2](https://uqrmaie1.github.io/admixtools/). It is an **as-built** document — the architecture, conventions, and quality bar the code actually follows — and the file paths and symbol names cited below are meant to resolve against the current tree. (It began as a build spec / scaffolding prompt; §17 is the original ordered scaffold manifest, now largely historical. The style rules are enforced by the repo's lint config — `.clang-format`, `.clang-tidy` — and the correctness rules by the GoogleTest/CTest + pytest suites, not by a hosted CI runner, which is not yet stood up, §14.)

The document is **opinionated by design**: every choice carries a one-line justification, and we do not present menus. Where a load-bearing CUDA-13 / CCCL fact is genuinely unverifiable from primary sources it is marked **[UNCERTAIN]** rather than asserted.

The single load-bearing domain insight that shapes everything: ADMIXTOOLS 2 is **precompute-once / fit-many**. One expensive, bandwidth-bound genotype pass produces a cacheable `f2_blocks` tensor (`[P × P × n_block]`); thereafter every model fit is small dense linear algebra over that resident tensor, batched across a block (jackknife) axis and a model axis. The architecture mirrors this split into a **precompute engine** (S0–S2) and a **fit engine** (S3–S8). **`f2_blocks` is scale-dependent, not flatly "tiny": it is `O(P²·n_block)`** — MB-scale at AT2-typical population counts (tens of pops), where it stays GPU-resident for the whole run, but tens of GB at thousands of pops (≈76 GB at `P=2500`, ≈220 GB at the `P=4266` top end, §11.4). Two corollaries the rest of the document keeps returning to: (a) the *input* genotype matrix routinely does **not** fit VRAM, so precompute is an out-of-core streaming computation (§11.1, **BUILT** — M5 SNP-tile streaming); and (b) the *output* itself does not fit VRAM past a few thousand pops, so it is **adaptively tiered** resident → host RAM → disk (§11.1/§11.2, **BUILT** — M5). At AT2-typical scale the input streams and the output stays resident; the at-scale machinery is exactly what makes thousands of pops tractable (a 75.7 GB result computed on a single 32 GB GPU, §11.1).

### Assumptions & chosen defaults

| Area | Default | One-line rationale |
|---|---|---|
| CUDA Toolkit | **13.x** (built green on box toolkit 13.0.88; `find_package(CUDAToolkit REQUIRED)` sets no version floor, `CMakeLists.txt:60`) | Minimum arch is Turing sm_75; full Blackwell (sm_120). The honest floor is **CUDA 13** — the 13.1 / CCCL-3.1 determinism-API floor was a design target, never enforced by the build. The default fixed-slice FP64 emulation on the f2 GEMMs needs the CUDA 13.0 Update 2 emulation API and degrades to native FP64 without it (§5, §9). |
| C++ standard | **C++20** (CUDA + host) | `std::source_location`, concepts, `std::span` parity; above CCCL's C++17 floor. |
| Build system | **CMake ≥ 3.28 + Ninja** | 3.28 is the as-built floor (`CMakeLists.txt:14`); 3.30 stays the aspirational floor for the named-Blackwell release matrix. Ninja gives correct CUDA dep tracking + `compile_commands.json`. |
| Device std lib | **CCCL ≥ 3.1** (Thrust / CUB / libcudacxx), pinned | One source for `cuda::std::span`/`mdspan`, device primitives, and the `cuda::execution` determinism controls. |
| Dense linalg | **cuBLAS + cuSOLVER** (cuSOLVERMp deferred) | Maps directly onto GEMM/SYRK/Cholesky/SVD/batched-LAPACK with *documented, scoped* determinism (§12). |
| RNG | **cuRAND Philox** (counter-based) | Reproducible per-replicate substreams independent of launch geometry. |
| Python bindings | **nanobind + scikit-build-core** | Fast builds, small binaries; CMake-native wheel backend; GPU-optional. |
| Unit/golden tests | **GoogleTest + CTest** (+ pytest for bindings) | Mature CUDA test patterns; `gtest_discover_tests`. |
| Benchmarks | **nvbench** (CUDA) + google-benchmark (host) | Purpose-built CUDA benchmarking with throughput reporting + JSON for regression gates. |
| Lint / format | **clang-format + clang-tidy + IWYU** | Single source of style and static analysis over `compile_commands.json`. |
| Precision | **Emulated-FP64 (Ozaki) default for matmul-heavy work; native FP64 oracle; TF32 opt-in** | GEMM/SYRK covariance & f4 assembly run in cuBLAS fixed-point FP64 emulation (accuracy ≈ native FP64, tensor-core throughput); native FP64 is the validation oracle/fallback and the path for the small cancellation-prone numerator/divide; the **f2 GEMMs themselves use fixed-slice Ozaki (`EmulatedFp64{mantissa_bits=40}` default) — measured 8–17× over native FP64 at native-grade accuracy on real AADR** (§5 S2, §12); dynamic-mantissa Ozaki is rejected as the parity trap; TF32 is screening-only (§9, §12). |
| Multi-GPU | **Single-node multi-GPU, day one** (single process, N devices) | The objective is parallelism + speedup **with parity** across the GPUs in one workstation. Single-process-multi-GPU (one host thread per device, per-device streams, NCCL for broadcast only — the parity reduce is host-side fixed-order, §11.4); multi-*node* remains deferred (§9, §11.4, §12). **Honest role (MEASURED, M4.5/M5):** on the *precompute* multi-GPU is a **modest throughput layer** — the at-scale enabler is M5 device-resident output + streaming, not sharding (§11.1, §11.4). Multi-GPU's decisive payoff was *expected* in the **fit/rotation phase (S8)**; but **MEASURED (real AADR, P=600): the S8 rotation is host-bounce-capped on the consumer 5090s** — the one-time `f2` replication to each GPU is ~8.72 GB / ~3.8 s through host (no GeForce P2P), so G2/G1 only ~1.21× at 9086 real models. **Run the fit/rotation single-GPU on the 5090s;** multi-GPU rotation is a genuine deferred item (`TODO(multigpu-host-bounce)`, `src/device/device_f2_blocks.hpp:92`, `src/core/qpadm/model_search.cpp:264`, `src/app/cmd_rotate.cpp:157`) — its payoff needs P2P hardware or per-device precompute (§5, §11.4). |
| Dense LA framework | **Native cuBLAS/cuSOLVER — no array framework (no JAX, no CuPy)** | The hot path is hand-issued GEMM/SYRK/Cholesky/SVD through `ComputeBackend` (`src/device/backend.hpp`); the statistics are *reformulated* into dense tensor ops (§2, §5 S2), not transliterated from an array DSL. |
| Determinism | **single-stream statistic path + CCCL `run_to_run` reductions + scoped deterministic cuSOLVER; emulated-FP64 needs an explicit `cublasSetWorkspace` workspace** | Bit-stable on a given GPU under the documented constraints (§12). Native FP64 and TF32 GEMM are run-to-run bit-reproducible single-stream; **fixed-point emulation voids that guarantee unless an adequate workspace is supplied** (§12) → non-flaky regression gates vs ADMIXTOOLS 2 goldens. |

---

## 1. Project vision & non-goals

**Vision.** `steppe` computes f2/f3/f4 statistics and fits qpWave / qpAdm models on the GPU, producing results numerically equivalent (within a stated tolerance) to ADMIXTOOLS 2, at a throughput that makes large model-space searches (rank tests across thousands of left/right configurations, graph-topology screening) practical on a single multi-GPU workstation. It on-the-fly filters and harmonizes its input dataset during the streaming pass (§5), reformulates the f-statistics into dense tensor ops to run at tensor-core throughput where the conditioning allows it (§2, §5 S2, §12), and shards the precompute across all GPUs in the box **with parity** (§11.4, §12); the model-space search parallelizes across GPUs where the hardware allows, but on no-P2P consumer GPUs the S8 rotation runs single-GPU and multi-GPU rotation is a deferred `TODO(multigpu-host-bounce)` (see below). It exposes a clean C++ library, a CLI (`src/app/`), and a Python package (`bindings/`, nanobind); `f2_blocks` is the cacheable, ADMIXTOOLS-compatible interchange artifact. **BUILT (M5):** the precompute returns a **device-resident** `f2_blocks` handle (`DeviceF2Blocks`, `src/device/device_f2_blocks.hpp`, kept in VRAM for the downstream fit), with host materialization as an **opt-in `.to_host()`** (the ADMIXTOOLS-compatible host `F2BlockTensor`), and adaptively tiers the result to host RAM or disk when it does not fit VRAM (`F2BlocksOut` / `OutputTier::{Resident,HostRam,Disk}`, `src/device/f2_blocks_out.hpp`; §11.1, §11.2).

**It IS:**
- A faithful, validated reimplementation of the f2 → f3/f4 → block-jackknife → qpWave/qpAdm GLS+SVD pipeline. The matmul-heavy covariance/f4 assembly defaults to **Ozaki-scheme emulated FP64** (accuracy ≈ native FP64) while the cancellation-sensitive elementwise math stays in **native FP64**; native FP64 remains the gold reference every other mode is validated against (§9, §12).
- A *precompute-once / fit-many* engine: one **out-of-core** streaming genotype pass (BUILT — M5 SNP-tile input streaming, GPU footprint `O(P·tile + P²·n_block)`, independent of the SNP count `M`), then batched small dense linear algebra over a resident tensor. When the `f2_blocks` *output* is itself larger than VRAM it streams to the fastest tier it fits (resident → host RAM → disk), so the engine is bounded by neither the input nor the output size (§11.1).
- GPU-optional at the API level: a CPU reference backend (`CpuBackend`, `src/device/cpu/cpu_backend.cpp`) always exists and always imports — the obviously-correct scalar oracle the GPU is continuously diffed against.

**It is NOT:**
- A general-purpose popgen suite (no PCA, ADMIXTURE, IBD, phasing, association testing).
- A multi-dataset merge/harmonize tool — **yet**. **BUILT:** genotype QC / data-munging as a *single-dataset* streaming front-end. The decoder auto-detects and reads **five formats** — packed TGENO, packed GENO (PACKEDANCESTRYMAP), ASCII EIGENSTRAT, PLINK `.bed`+`.bim`/`.fam`, and unpacked ANCESTRYMAP — in the `GenoReader` ctor (`src/io/geno_reader.cpp`), normalizing every non-TGENO format onto the canonical SNP-major source + on-device transpose; the five arms are dispatched by `read_canonical_tile` (`src/core/stats/read_canonical_tile.cpp`) and reached from the real precompute path. Per-tile filtering/harmonization is applied during the stream (`src/io/filter/`): the default missing-data policy is **pairwise-complete** (validity-mask + per-SNP/per-pop sample size `N`) for AT2 parity, with **imputation optional and off by default** (screening/PCA only, never an AT2-golden-compared statistic). The cheap filters are MAF/geno/include-exclude and `--mind`; **monomorphic-drop, autosome-only, and ts/tv are additions beyond bare "filter," gated behind explicit flags and tagged** (`include/steppe/config.hpp`). We do *not* **infer strand or resolve A/T·C/G ambiguity heuristically** (the `Flip` strand mode is a documented not-yet-implemented token; ambiguous or multiallelic sites are dropped, never flipped by frequency guesswork; the default `Drop` mode drops palindromic A/T·C/G SNPs), do *not* **compute LD/pruning ourselves** (we accept only an externally supplied `prune.in`), and do *not* **rewrite datasets to disk**. **[PLANNED]** Cross-dataset *merge/harmonize* over multiple sources is not built — no `src/io/merge`, no `src/io/impute`; the intended design is an in-memory *plan* over the sources with harmonization + cheap filters applied per tile during the stream.
- A multi-*node* HPC framework. **Single-node multi-GPU is in scope and central** — parallelism + speedup across the N devices of one workstation, *with* numerical parity, is a primary objective (§9, §11.4, §12). It is single-*process*-multi-GPU (one host thread per device, per-device streams, NCCL for the broadcast only; the parity reduce is host-side fixed-order, §11.4). Out-of-core genotype streaming is a v1 requirement (§11.1) and shards SNP tiles across the GPUs. **Honest role (MEASURED, M4.5/M5):** on the precompute the multi-GPU sharding is a *modest* throughput layer — the at-scale win came from getting the result off the CPU (device-resident output, M5) and streaming, not from sharding (§11.1, §11.4). Multi-GPU's payoff was *expected* in the model-space search (S8) — per-fit there is **zero** inter-GPU traffic — BUT **MEASURED (real AADR, P=600): the one-time `f2` replication to each GPU is a ~8.72 GB / ~3.8 s host bounce on the no-P2P 5090s, capping the S8 rotation at ~1.21× at 9086 real models — so run it single-GPU on the 5090s.** Multi-GPU rotation is therefore **deferred** (`TODO(multigpu-host-bounce)`, `src/device/device_f2_blocks.hpp:92`, `src/core/qpadm/model_search.cpp:264`, `src/app/cmd_rotate.cpp:157`); its payoff needs P2P hardware or per-device precompute (§5, §11.4). The deferred boundary is **multi-node** (multiple processes / MPI / a launcher). **cuSOLVERMp stays deferred** — not for lack of multi-GPU ambition, but because qpAdm's `Q`/`X` are tiny: the right parallelism is *across many independent small fits* (S8), not *within one distributed factorization* (§11.4).
- A drop-in R package replacement. We target *numerical* and *file-format* compatibility, not R-API compatibility.
- A research sandbox for low-precision tricks. TF32/FP16 never produce a reported statistic without re-validation; TF32 is a user-selectable mode for *model-space screening only*, and its results carry a precision tag and a looser tolerance tier — never bit-compared to AT2 goldens (§9, §12). Emulated FP64 is *not* a low-precision trick: it targets native-FP64 accuracy and is gated against the native-FP64 oracle before it ships.

## 2. Engineering principles

**DRY / single source of truth.** Every cross-cutting concern — CUDA error checking, launch-config math, precision policy, span/mdspan views, **and every domain rule shared between layers** — is implemented exactly once and consumed through one target. Concretely: one `STEPPE_CUDA_CHECK` (`src/device/cuda/check.cuh`), one `cdiv()` (`src/core/internal/launch_config.hpp`), one **`Precision` policy value** (`include/steppe/config.hpp` — a *runtime* `struct Precision { enum class Kind { Fp64, EmulatedFp64, Tf32 }; int mantissa_bits; }`, not a compile-time `real_t<P>` template), one `assign_blocks()` SNP→block rule (`src/core/domain/block_partition_rule.hpp`). The single most important DRY consequence in this codebase: the **SNP→block assignment rule** is a domain decision that `io`, the device kernels, and the jackknife all depend on bit-for-bit, so it lives host-pure in `core` (`assign_blocks`) and is consumed everywhere — never re-derived in `io` (§5, §8).

The allocation family is **confined to RAII wrappers**. All direct allocation calls live in exactly **two** translation units: `src/device/cuda/device_buffer.cuh` (`cudaMalloc`/`cudaFree`) and `src/device/cuda/pinned_buffer.cuh` (`cudaHostAlloc`/`cudaFreeHost`). No other code touches the allocation family, and the async/managed variants (`cudaMallocAsync`/`cudaMallocManaged`/`cudaFreeAsync`) are not used at all. This confinement is currently enforced by **review and by the wrapper-header comments**, not by an automated check — an automated grep gate with a checked-in TU allowlist is **planned, not built** (no CI configuration exists in-repo).

**Strict separation of concerns.** The codebase is layered `app/bindings → core → device`, with `io` an isolated leaf, and the direction is enforced by the *compiler* via CMake link visibility (§4), not by convention: `CUDA::cublas` / `CUDA::cudart` are `PRIVATE` to `steppe_device`, so CUDA headers never compile into `core`, the CLI, or tests. **`core` is pure host C++20 and issues every device operation through the `ComputeBackend` interface** (`src/device/backend.hpp`, verified CUDA-free) — it never includes a CUDA header and never calls cuBLAS/cuSOLVER directly (this is what makes the f4/GLS/SVD stages in §5 layering-legal). `ComputeBackend` has two implementations, `CudaBackend` (`src/device/cuda/cuda_backend.cuh`) and `CpuBackend` (`src/device/cpu/cpu_backend.cpp`). The `f2_blocks` tensor is the seam between the GPU hot loop and the small-LA derivations.

**RAII everywhere.** No raw allocation, `cudaStreamCreate`, `cublasCreate`, `cusolverDnCreate`, or library workspace lives outside an owning wrapper: `DeviceBuffer<T>` (`device_buffer.cuh`), `Stream` / `Event` (`stream.hpp`), `CublasHandle` / `CusolverDnHandle` (`handles.hpp`), `PinnedBuffer<T>` (`pinned_buffer.cuh`). Move-only owning types with full move-construct **and** move-assign (§7); destructors never throw — but they are not silent: in debug builds a nonzero destroy status is routed to `STEPPE_LOG_WARN` so "fail-fast" does not become "fail-silent at teardown" (§7, §10).

**No global mutable state.** Configuration is an immutable `RunConfig` value object (`include/steppe/config.hpp`). Device resources are bundled per-device in a `Resources` struct (`src/device/resources.hpp`): one `PerGpuResources` per device, in fixed `g=0..G-1` combine order, each owning its `ComputeBackend` (which in turn owns the device's stream, handles, and buffers). `Resources` is injected into every compute entry point. No singletons, no hidden statics — the precondition for both thread-safety and unit-testability.

**Reformulate statistics into dense tensor ops; fuse the elementwise feeders.** A statistic is implemented by *re-deriving it as matrix algebra* — outer products, masked reductions, and contractions expressed as cuBLAS GEMM/SYRK batched over the block and model axes (native cuBLAS/cuSOLVER, **no array framework — no JAX, no CuPy**) — not by transliterating the scalar CPU loop onto the GPU. Per-SNP missingness becomes masking-by-multiply (`@ Vᵀ`); per-pair sums become matmuls (§5 S2). The elementwise *feeders* of those matmuls (zero-filled `Q`, validity mask `V`, `Q²`, het correction) are produced in a **single fused sweep** over the decoded tile, never materializing the `[SNP × pop × pop]` intermediate (§11.1); the elementwise *consumers* (the catastrophic-cancellation numerator and the masked divide) are likewise fused, on the small reduced matrices, in native FP64. Precision follows the **conditioning of the operation, not its shape** (§12): the reformulation does not *abolish* the f2 cancellation, it *localizes* it — the `Σp²−2Σpq+Σq²` difference still lands on the small `O(n_pop²)` reduced matrix, where keeping it (and the GEMMs that feed it) in native FP64 is essentially free, so emulation is reserved for the genuinely well-conditioned matmul-heavy stages (covariance SYRK / f4 assembly, §5 S4/S3), not the f2 reduction (§12).

**Fail-fast.** Every CUDA API return is checked at the call site (`STEPPE_CUDA_CHECK`); every kernel launch is followed by `cudaGetLastError()` and, in debug, a forced sync (`STEPPE_CUDA_CHECK_KERNEL`, `check.cuh`). Configuration is **statically** validated once at `ConfigBuilder::build()` and frozen — invalid arch lists and absent/duplicate device ordinals surface immediately with file/line context (`src/core/config/config_builder.cpp`). The `build()` layer is deliberately CUDA-free and does **not** probe the GPU: the **live device-memory / VRAM budget check runs later, in the device layer at resource construction** (`build_resources`; `src/device/vram_budget.hpp`, `tier_select.hpp`, `decode_budget.hpp`, `resources.cpp`), where a config that exceeds VRAM is rejected before the tiered output path commits (§9, §11.2) — not as silent corruption or an OOM three stages later.

**Testability.** Per-element numerics and the reference backend live in `__host__ __device__` pure functions that compile and unit-test on the CPU with no GPU. `ComputeBackend` has two implementations (`CudaBackend`, `CpuBackend`) so the entire pipeline is exercisable GPU-free and the GPU is continuously diffed against an obviously-correct scalar reference. **The CPU reference is an *oracle that validates results, not a structural template the GPU mimics.*** The GPU production path may have an entirely different control structure (three batched GEMMs + two fused elementwise kernels) from the reference's scalar triple loop, as long as the two agree at the `f2_blocks`/`Q`/`X` seams within the tolerance tiers (§12, §13). Thin `__host__ __device__` scalar functions survive only as (a) the reference implementation and (b) per-element primitives invoked *inside* the fused kernels — **never as the structure of the production hot path** (§5 S2, §7).

**Numerical reproducibility / determinism.** Native FP64 for the cancellation-sensitive elementwise math, Ozaki-emulated FP64 for the matmul-heavy covariance/f4 assembly (validated against the native-FP64 oracle), TF32 opt-in for screening only (§9, §12); CCCL `run_to_run` deterministic reductions on the **single statistic stream**; scoped deterministic cuSOLVER; counter-based (Philox) RNG with seeds threaded through the API and recorded in golden metadata. The determinism guarantee is **specific and constrained** (single stream, given GPU/arch, enumerated routines, and an explicit `cublasSetWorkspace` workspace for the emulated-FP64 path) — see §12, which states exactly where bit-stability holds and where it does not, rather than claiming it globally.

**Correctness before speed.** The CPU reference and ADMIXTOOLS 2 goldens gate every change — they validate the *result* the reformulated GEMM path produces, not its internal structure (the reference walks the exact AT2 pairwise-complete path; the GPU walks three GEMMs + two fused kernels; they must agree at the seams, §12, §13). TF32 fast paths are opt-in screening tools that never produce a reported `est`/`se`/`z`/`p` without re-validation, while emulated FP64 is the default for the matmul-heavy path and *does* produce reported numbers — but only after passing the native-FP64 oracle gate (§12). We optimize only the kernel the profiler (Nsight Systems → Compute) proves dominant, never on speculation.

## 3. Tech stack & pinned versions

There is **no** `third_party/` directory and no single central `FetchContent`/CPM pin file. The CPM bootstrap lives at `cmake/CPM.cmake` (a SHA256-verified downloader shim, CPM 0.42.3) and is `include()`d **only** from the two CUDA-free leaf subtrees that need a fetched dependency: `src/app` (behind `STEPPE_BUILD_CLI`, for CLI11) and `bindings` (behind `STEPPE_BUILD_PYTHON`, for nanobind). Every external pin is therefore **distributed per-subtree** via a `find_package`-first / CPM-fallback pattern, scoped `PRIVATE` to its subtree; the core/device build pulls only toolkit libraries. GoogleTest is resolved by `find_package` with a self-checking-executable fallback and is never CPM-fetched.

| Component | Pin | One-line rationale |
|---|---|---|
| CUDA Toolkit | **13.x** | Turing→Blackwell. As-built floor is CUDA 13 (dev box green on 13.0.88). The fixed-point FP64 emulation math mode requires **CUDA 13.0 Update 2+**. |
| CMake | ≥ 3.28 | As-built floor: `CMakeLists.txt:14` (`cmake_minimum_required(3.28)`), `CMakePresets.json` (`cmakeMinimumRequired` 3.28), `pyproject.toml:81` (`cmake.version=">=3.28"`). 3.30 stays a planned target for the named-Blackwell release matrix (§6). |
| Ninja | ≥ 1.11 | Recommended CUDA generator; emits `compile_commands.json` (`CMAKE_EXPORT_COMPILE_COMMANDS ON`, `CMakeLists.txt:31`). |
| C++ standard | C++20 | `CMakeLists.txt:26-29` set `CMAKE_{CXX,CUDA}_STANDARD 20` + `_REQUIRED`. `source_location`, concepts. |
| **CCCL** | toolkit-bundled | Thrust/CUB/libcudacxx, used **only** as shipped with the toolkit: `<cub/...>`/`<thrust/...>` in the device TUs, `cuda::std` in `src/core/internal/views.hpp`. No `find_package(CCCL)`, no `CCCL_VERSION` guard, no CPM fetch; the `cuda::execution::require(determinism::…)` API is **not** used. |
| cuBLAS / cuSOLVER | from toolkit | GEMM/SYRK + Cholesky/SVD; linked `PRIVATE` to `steppe_device` (`src/device/CMakeLists.txt:71`, `CUDA::cublas CUDA::cusolver`). Supplies the three precision modes: native FP64, **fixed-point FP64 emulation (Ozaki)** for the matmul-heavy path, and TF32 for screening (§9, §12). |
| FP64 emulation (Ozaki) | cuBLAS/cuSOLVER 13.x | **Handle-level** (not per-matmul): `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` / `CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH` (`src/device/cuda/handles.hpp:65-88,241`). The tuning knobs (`cublasSetEmulationStrategy` / mantissa control) are gated behind `STEPPE_HAVE_EMU_TUNING` (default ON), which the measured FIXED-40-bit default requires. |
| cuFFT | from toolkit | Autocorrelation/LD engine for the DATES admixture-timing path; linked `PRIVATE` to `steppe_device` (`src/device/CMakeLists.txt:71`, `CUDA::cufft`). |
| CLI11 | 2.4.2 | CLI parsing → `ConfigBuilder`. `find_package(CLI11 2.4 CONFIG)` first, else CPM-fetch tag v2.4.2 (`src/app/CMakeLists.txt:17-30`); behind `STEPPE_BUILD_CLI`, `PRIVATE` to `src/app`. |
| nanobind | ≥ 2 (v2.4.0 fallback) | Fast-building, small Python bindings → the `steppe._core` module. `find_package` first (with pip-config discovery), else CPM-fetch tag v2.4.0 (`bindings/CMakeLists.txt:30-51`); behind `STEPPE_BUILD_PYTHON`. |
| scikit-build-core | ≥ 0.10 | CMake-driven PEP 517 wheel backend with CUDA support (`pyproject.toml:20`, `requires = ["scikit-build-core>=0.10", "nanobind>=2"]`). |
| GoogleTest | not pinned | Unit/golden/regression harness when discoverable: `find_package(GTest CONFIG QUIET)` with a self-checking-executable fallback that returns non-zero on failure (`tests/CMakeLists.txt:14-21`). Never CPM-fetched. |
| ccache | when found | Compiler launcher wired inline behind `STEPPE_CCACHE` (default ON): `find_program(ccache)` sets `CMAKE_{CXX,CUDA}_COMPILER_LAUNCHER` (`CMakeLists.txt:71-85`, `cmake/SteppeOptions.cmake:56-61`). nvcc-aware, byte-identical objects, a no-op when absent. |
| clang-format / clang-tidy | clang-based | `.clang-format` and `.clang-tidy` at the repo root are the single sources of formatting and static analysis; `compile_commands.json` is exported (`CMakeLists.txt:31`) for clang-tidy / IWYU / clangd to consume. |

### Not pinned dependencies — hand-rolled internals

Several capabilities the earlier draft listed as external pins are implemented in-tree, not linked:

- **Host small dense LA** — `src/core/internal/small_linalg.hpp` (header-only, standard-library-only LU-with-partial-pivoting `solve`, one-sided Jacobi SVD, explicit inverse), the CpuBackend reference oracle's solvers. No Eigen anywhere in the tree.
- **Logging facade** — `src/core/internal/log.hpp` is a printf-style `std::fprintf(stderr, …)` warn sink: one `[steppe][warn]` line in debug, removed entirely under `NDEBUG`. A structured backend (e.g. spdlog) is a reserved future seam (§10), not a wired dependency.
- **Microbenchmarks** — hand-rolled `<chrono>` timers under `tests/reference/bench_*.cu` (e.g. `bench_f2_multigpu.cu`); nvbench is not used or pinned.

### Deliberately absent

- **NCCL / cuRAND** — neither is linked. The single-node multi-GPU f2 combine uses CUDA-runtime **P2P** (`cudaMemcpyPeerAsync`, with a host-staged fixed-order fallback) so the reduction stays bit-identical to the single-GPU sum; it is **never** an NCCL AllReduce (`src/device/p2p_combine.hpp`, `src/device/cuda/p2p_combine.cu`; §11.4, §12). The block jackknife is deterministic leave-one-out, so no RNG library is needed.

### Planned (not yet built)

- **CMake 3.30 floor** for the named-Blackwell release matrix (current floor is 3.28; §6).
- **Format/lint runner** — the `.clang-format`/`.clang-tidy` configs describe a pre-commit/CI harness that is not yet wired (no `.github/` or `.pre-commit-config.yaml` in the repo).

---

## 4. Repository layout

```text
# Legend:  [BUILT] = exists at HEAD;  (planned) = genuine future work, not yet in-tree.
# STATE: The system is BUILT end-to-end on the GPU. Phase-1 precompute (S0-S2) and the
#        Phase-2 qpAdm/qpWave FIT ENGINE (S3-S8) are built and golden-gated (CPU ref vs GPU)
#        on real AADR. The public surface now also ships standalone f-stats (f3/f4/f4ratio/
#        D-stat/sweep), qpGraph fit + topology search, qpfstats, and DATES; a CLI (src/app,
#        behind STEPPE_BUILD_CLI) and nanobind Python bindings (bindings/, behind
#        STEPPE_BUILD_PYTHON), each with C++ / CLI / Python golden tests.
#        Still deferred/planned: multi-GPU rotation throughput (G>=2 host-bounce TODO, below);
#        io/merge, io/impute, and an on-disk precomputed-f2 cache.
steppe/                                  # repo root.  git: HEAD on branch main.
├── CMakeLists.txt                       # [BUILT] top-level: project policy + add_subdirectory
│                                        #   (include, src/io, src/core, src/device, then opt-in
│                                        #   src/access, src/extract, src/app, bindings, docs/examples, tests)
├── CMakePresets.json                    # [BUILT] named configs; CMake >= 3.28 + Ninja
├── .clang-format / .clang-tidy          # [BUILT] single source of style + static-analysis policy
├── .gitignore                           # [BUILT] excludes build trees, binaries, AND the real genotype DATA (aadr/)
├── pyproject.toml / install.sh          # [BUILT] Python packaging metadata + one-shot build/install helper
├── LICENSE / README.md                  # [BUILT]
│
├── cmake/                               # [BUILT] reusable build logic (defines NO targets)
│   ├── CPM.cmake                        #   CPM.cmake bootstrap shim -- the single dependency-fetch home (CLI11, GoogleTest, nanobind)
│   ├── CUDAArch.cmake                   #   CMAKE_CUDA_ARCHITECTURES policy (Blackwell sm_120 under CUDA 13)
│   ├── SteppeOptions.cmake              #   ALL option()/cache vars in one place (STEPPE_BUILD_{CLI,PYTHON,TESTS,EXAMPLES,DOCS},
│   │                                    #     STEPPE_HAVE_EMU_TUNING, STEPPE_WERROR, STEPPE_NVTX, STEPPE_SANITIZER, STEPPE_CCACHE)
│   ├── SteppeSanitizers.cmake           #   translates the STEPPE_SANITIZER cache var into compile/link flags
│   ├── SteppeWarnings.cmake             #   warnings-as-errors INTERFACE target (host + nvcc)
│   └── Docs.cmake                       #   opt-in GENERATED API reference (Doxygen + pdoc; STEPPE_BUILD_DOCS)
│
├── include/                             # ----- PUBLIC API (installed surface); steppe_api INTERFACE target -----
│   └── steppe/                          # 14 public, CUDA-FREE headers:
│       ├── config.hpp                   #   Precision{kind, mantissa_bits=40}, DeviceConfig, FilterConfig +
│       │                                #     named constants (kCdivBlock, kRelFloor, kDefaultBlockSizeCm...) -- no magic numbers
│       ├── error.hpp                    #   Status enum (Ok/DeviceOom/RankDeficient/NonSpdCovariance/ChisqUndefined/
│       │                                #     InvalidConfig...); the THREE domain outcomes = RankDeficient/NonSpdCovariance/ChisqUndefined
│       ├── fstats.hpp                   #   public host F2BlockTensor (opt-in .to_host() materialization; device handle lives in src/device/)
│       ├── qpadm.hpp                    #   run_qpadm / run_qpwave / run_qpadm_search (+ QpAdmModel by-index, QpAdmResult, JackknifePolicy)
│       ├── f3.hpp / f4.hpp / f4ratio.hpp #  standalone f-stat entry points: run_f3 / run_f4 / run_f4ratio (read the f2 cache)
│       ├── dstat.hpp                    #   run_dstat -- normalized D (qpDstat Part B)
│       ├── fstat_sweep.hpp             #   run_f4_sweep / run_f3_sweep -- GPU all-combinations f-stat sweep
│       ├── qpgraph.hpp / qpgraph_search.hpp # qpGraph single-graph fit + bounded topology-search entry points
│       ├── qpfstats.hpp                 #   run_qpfstats -- genotype-path joint f-stat smoother
│       ├── dates.hpp                    #   run_dates -- admixture-dating (ALDER/DATES); reads genotypes directly
│       └── extract.hpp                  #   run_extract_f2 -- library-level genotype->f2 counterpart of the CLI extract-f2 command
│
├── src/
│   ├── core/                            # steppe_core -- host C++ orchestration (NO CUDA; links steppe::io PRIVATE for the genotype front-end)
│   │   ├── internal/                    #   DRY shared helpers: views.hpp (MatView column-major [P x M] contract),
│   │   │                                #     f2_estimator.hpp + decode_af.hpp (shared __host__ __device__ primitives), host_device.hpp,
│   │   │                                #     launch_config.hpp, pchisq.hpp, small_linalg.hpp, qpfstats_jackknife.hpp, dates_fit.hpp, log/nvtx
│   │   ├── config/                      #   layered config: cli_args + config_builder{.hpp,.cpp} (defaults < TOML < env < CLI, validate-once) ->
│   │   │                                #     immutable RunConfig; exit_code, build_result
│   │   ├── domain/
│   │   │   └── block_partition_rule.{hpp,cpp} # host-pure SNP->block rule: block_of/assign_blocks + inverse block_ranges (both backends call it)
│   │   ├── fstats/                       #   host orchestration of the f2 precompute (via the ComputeBackend seam):
│   │   │   ├── f2_from_blocks.{hpp,cpp}  #     drives the f2 compute
│   │   │   ├── f2_combine.{hpp,cpp}      #     fixed-order combine of per-device partials (parity reduce)
│   │   │   ├── f2_blocks_multigpu{,_core}.{hpp,cpp} # multi-GPU precompute orchestration (block sharding)
│   │   │   └── f2_partials_validate.hpp  #     per-device partial validation
│   │   ├── qpadm/                       #   the FIT / SEARCH / STATS engine (S3-S8) -- host-pure orchestration.
│   │   │   ├── qpadm_fit.{hpp,cpp}       #     run_impl: S3 assemble_f4 -> S4 jackknife_cov -> S6 GLS -> S7 chisq/p -> result
│   │   │   ├── f4_matrix.hpp / f4_quartets.hpp / f3_triples.hpp # S3 f4/f3 contraction drivers (cancellation-sensitive, native FP64)
│   │   │   ├── jackknife.hpp            #     S4 block-jackknife -> covariance Q + LOO SE
│   │   │   ├── gls_solve.hpp            #     S6 constrained GLS weight-solve
│   │   │   ├── ranktest.{hpp,cpp}       #     S5 rank sweep / qpWave (rankdrop/popdrop)
│   │   │   ├── nested_models.{hpp,cpp}  #     S7 nested chisq/dof/p + rank-decision
│   │   │   ├── qpadm_bounds.hpp         #     feasibility/bounds predicates on fitted weights
│   │   │   ├── model_search{,_core}.{hpp,cpp} # S8 rotation: run_qpadm_search orchestrator + host-pure plan_model_shards
│   │   │   ├── f3.cpp / f4.cpp / f4ratio.cpp / fstat_sweep.cpp # standalone f-stat implementations (the public f3/f4/... headers)
│   │   │   └── qpgraph_{enumerate,fit,model,objective,opt_constants,search}.{hpp,cpp} # the qpGraph fit + topology-search family
│   │   │       # NAMING NOTE: the dir is named qpadm/ but is the FIT/SEARCH/STATS engine, NOT qpAdm-only (it also
│   │   │       #   houses the qpGraph family and the standalone f-stats). A rename/split was weighed and DEFERRED
│   │   │       #   (a large `git mv` rewrites blame across the most-touched core dir for a readability-only gain).
│   │   └── stats/                       #   genotype-path host orchestration (the core->io PRIVATE front-end edge):
│   │       ├── dstat.cpp / qpfstats.cpp / dates.cpp # run_dstat / run_qpfstats / run_dates over decoded genotypes
│   │       ├── read_canonical_tile.{hpp,cpp}       # open io::GenoReader -> canonical decoded tile
│   │       └── genotype_front_end.{hpp,cpp}        # shared decode/QC wiring these tools reuse
│   │
│   ├── device/                          # steppe_device -- the backend layer (CUDA isolated here)
│   │   ├── backend.hpp                  # [BUILT] ComputeBackend interface (CUDA-FREE) -- the DI seam
│   │   ├── backend_factory.hpp          # [BUILT] CUDA-free factory (selects CPU vs CUDA backend)
│   │   ├── resources.{hpp,cpp} / shard_plan.{hpp,cpp} / vram_budget.hpp # device enumeration, per-block shard plan, VRAM sizing
│   │   ├── device_partial.hpp / f2_blocks_out.hpp / decode_budget.hpp / device_decode_result.hpp / p2p_combine.hpp # multi-GPU plumbing
│   │   ├── device_f2_blocks.hpp         # [BUILT] DeviceF2Blocks VRAM handle (move-only device-resident precompute return)
│   │   ├── tier_select.hpp              # [BUILT] CUDA-FREE OutputTier{Resident,HostRam,Disk} + select_output_tier() (host-pure, unit-testable)
│   │   ├── stream_f2_blocks.hpp         # [BUILT] streamed-loop entry (block-axis output + SNP-tile input streaming)
│   │   ├── f2_disk_format.hpp           # [BUILT] STPF2BK1 on-disk f2_blocks cache layout (Disk tier)
│   │   ├── host_ram.cpp                 # [BUILT] HostRam-tier block-by-block host spill
│   │   ├── cpu/cpu_backend.cpp          # [BUILT] REFERENCE backend: long-double cancellation-free f2 (the correctness oracle)
│   │   └── cuda/                        # [BUILT] CUDA implementation -- split across TUs:
│   │       ├── cuda_backend.cu / .cuh    #   the ComputeBackend GPU impl, plus split TUs cuda_backend_{decode,f2_blocks,
│   │       │                            #     qpadm_fit,fstats_assemble,dstat,qpfstats,qpgraph,dates}.cu (S3-S8 overrides live here, not in cuda_backend.cu)
│   │       ├── device_buffer.cuh        #   DeviceBuffer<T> move-only RAII -- the ONLY place cudaMalloc/cudaFree live
│   │       ├── stream.hpp / handles.hpp / check.cuh / device_guard.cuh / pinned_buffer.cuh # Stream/Event/Cublas RAII + error checks
│   │       ├── f2_block_kernel.{cu,cuh} / f2_batched_kernel.{cu,cuh} # fused pre-pass (Q,V,Qsq,Hc) + 3-GEMM (fixed-slice Ozaki / native FP64)
│   │       ├── decode_af_kernel + decode_compact_kernel + detect_ploidy_kernel + transpose_canonical_kernel # decode / layout kernels
│   │       ├── qpadm_fit_kernels.{cu,cuh} + qpgraph_fit_kernels + dstat_kernel + dates_kernel + qpfstats_kernel
│   │       │                            #   + qpfstats_jackknife_kernel + ratio_block_jackknife_kernel # the fit/stats device kernels
│   │       ├── device_f2_blocks.{cu,cuh,_impl} # resident f2/Vpair compute + opt-in D2H + no-peer assembly transport
│   │       ├── block_sink.{cu,cuh}      #   streamed-tier sink (persistent pinned ring + background writer)
│   │       └── device_decode_result.cu / device_partial.cu / f2_blocks_out.cu / p2p_combine.cu # multi-GPU transfer/combine impls
│   │
│   ├── io/                              # [BUILT] steppe_io -- genotype decode + QC front-end (ISOLATED leaf, host-pure)
│   │   ├── eigenstrat_format.{hpp,cpp}  #   (T)GENO header parse + format constants
│   │   ├── {geno,snp,ind}_reader.{hpp,cpp} # .geno (tiled raw bytes) / .snp (chrom,genpos,alleles) / .ind (pops)
│   │   ├── plink_reader.{hpp,cpp}       #   PLINK .bed/.bim/.fam decode
│   │   ├── genotype_source.{hpp,cpp}    #   format-agnostic source dispatch (EIGENSTRAT / ANCESTRYMAP / PLINK / packed)
│   │   ├── ploidy_detect.{hpp,cpp}      #   ploidy auto-detection
│   │   ├── genotype_tile.hpp / snp_major_tile.hpp # plain decoded-tile structs (the leaf's output)
│   │   └── filter/                      #   filter_decision (shared predicates) + snp_filter + mind_prepass +
│   │                                    #     include_exclude + filter_plan + snp_summary_reduce (MAF/geno/mind/autosomes; drop-not-flip)
│   │   # (planned) io/merge (multi-dataset merge-plan), io/impute (optional imputation), precomputed_f2 cache -- NOT yet in-tree
│   │
│   ├── access/                          # [BUILT] steppe_access -- CUDA-FREE f2-dir reader + pop-name->index resolver.
│   │                                    #   Dir holds ONLY CMakeLists.txt; the sources live in src/app/ (f2_dir_io.cpp, pop_resolver.cpp)
│   ├── extract/                         # [BUILT] steppe_extract -- run_extract_f2 + STPF2BK1 dir writer.
│   │                                    #   Dir holds ONLY CMakeLists.txt; sources live in src/app/ (extract_f2_core.cpp, f2_dir_writer.cpp)
│   └── app/                             # [BUILT, STEPPE_BUILD_CLI] the `steppe` CLI (steppe_app; CLI11):
│       ├── main.cpp / cli_parse.{hpp,cpp} #   dispatch + arg parsing
│       ├── cmd_*.cpp                    #   extract-f2, qpadm, qpwave, qpdstat, qpfstats, qpgraph, f3, f4, f4ratio, fstat-sweep, dates, rotate, emit
│       ├── result_emit.{hpp,cpp} / precision_label.hpp / exit_code_for_caught.hpp # output + exit-code helpers
│       └── f2_dir_io.cpp / pop_resolver.cpp / extract_f2_core.cpp / f2_dir_writer.cpp # the access/extract sublib sources (shared with bindings)
│
├── bindings/                            # [BUILT, STEPPE_BUILD_PYTHON] nanobind module steppe._core + the `steppe` Python facade:
│   ├── module.cpp + bind_{qpadm,fstats,qpgraph,dates,f2handle}.cpp + internal/bind_common.hpp
│   └── steppe/__init__.py + steppe/_rds.py  # Python package + ADMIXTOOLS-compatible .rds export
│
├── tests/
│   ├── reference/*.cu                   # [BUILT] golden-gated GPU-vs-CPU parity + decode-equivalence on real AADR:
│   │                                    #   f2 / decode (eigenstrat/ancestrymap/plink/pa) equivalence, qpadm_parity, qpadm_rotation,
│   │                                    #   qpadm_domain, qpwave_parity, qpadm_missing_block, qpgraph/qpfstats/fstat_sweep/dates parity,
│   │                                    #   plus bench_*.cu; goldens/ holds the ADMIXTOOLS-2 fixtures + expected JSON/CSV
│   ├── unit/*.cpp                       # [BUILT] host unit tests (f2, block_partition, block_ranges, filters, config, shard_plan, vram_budget...)
│   ├── cli/test_cli_*.cpp               # [BUILT] CLI integration tests (per command + IO-fault + prefix-detection)
│   ├── python/test_py_*.py              # [BUILT] pytest coverage of the Python facade
│   └── r/verify_export_rds.R            # [BUILT] R-side verification of the .rds export
│
├── docs/
│   ├── archive/architecture.md          # [BUILT] this document (archived location)
│   ├── archive/ROADMAP.md               # [BUILT] milestones + magic-number->config inventory
│   └── examples/                        # [BUILT, STEPPE_BUILD_EXAMPLES] cpp/ + python/ quick-starts (read_f2 -> qpadm) + f2_9pop sample
│
└── aadr/                                # real AADR genotype DATA -- GITIGNORED, never committed (~GB scale)
```

**Dependency-direction rule.** Allowed edges only: `app/bindings → api → core → {device, io}`; `io` is a leaf that produces plain data structs (genotype tiles + per-SNP genetic positions) and depends on nothing in `core`/`device`. Two layers consume `io`: `app` wires `io` output into compute for the file-emitting commands, and `core` itself links `steppe::io` **PRIVATE** (`src/core/CMakeLists.txt`) for the shipped in-core genotype-path stats tools (`stats/dstat.cpp`, `stats/qpfstats.cpp`, `stats/dates.cpp`, `stats/read_canonical_tile.cpp`, `stats/genotype_front_end.cpp`) — a deliberate `core → io` front-end edge that turns an open `io::GenoReader` into the canonical tile, then dispatches decode through `ComputeBackend`. **Nothing depends upward; no cycles** (`io` stays a leaf below `core`). The one shared domain rule (`domain/block_partition_rule.hpp`) lives in `core` and is the single exception that both `io` consumers and device kernels read — it is host-pure and CUDA-free, so it does not break the layering.

**How CMake enforces it.** Link visibility is the mechanism. A `PUBLIC`/`INTERFACE` dependency propagates include dirs to consumers; a `PRIVATE` one does not. CUDA is `PRIVATE` to `steppe_device`, so `core`/`app` *physically cannot* `#include` a CUDA header — it won't compile; `core` reaches the GPU only through `ComputeBackend` (a CUDA-free header). The `steppe_access` and `steppe_extract` sublibs are declared as plain-C++20 (`CXX`) targets — their CMakeLists cite this section ("PLAIN C++20 CXX target (architecture.md §4)") precisely so they cannot pull in CUDA. `.clang-tidy` plus the exported `compile_commands.json` (`CMAKE_EXPORT_COMPILE_COMMANDS`) back local static analysis. See §7 for the target wiring.

**Planned / deferred.** Multi-GPU *rotation* throughput is deferred: on no-P2P consumer cards (e.g. RTX 5090) the G≥2 path replicates the f2 tensor via a host bounce (`src/core/qpadm/model_search.cpp`, `TODO(multigpu-host-bounce)`); the replication is correct and deterministic (G1==G2 bit-identical) but throughput-suboptimal, so the `G==1` single-GPU fast path is preferred until it lands. Also planned but not yet in-tree: `io/merge`, `io/impute`, and an on-disk precomputed-f2 cache.

---

## 5. Layered architecture mapped onto the qpAdm pipeline

The numerics define the layers. Each stage names its **owner** (where the orchestration code lives) and, where device work is involved, makes explicit that the device call is **dispatched through `ComputeBackend`** — `core` never issues a GEMM/SVD/Cholesky itself. **S0–S2 = precompute engine** (out-of-core streaming, bandwidth-bound, run once); **S3–S8 = fit engine** (batched small dense LA over the resident `f2_blocks`, with `n_block` and model index as two batch axes).

| Stage | Owner (orchestration) | Device call via | Input → output | Parallelism | Bound by |
|---|---|---|---|---|---|
| **S0 Format decode** | `io` (read, tiled) → `device` kernel | `ComputeBackend::decode_af` | packed `.bed` tile → dosage `[SNP_tile × sample]` | data-parallel over SNP rows; 2-bit unpack per warp (00→0,10→1,11→2,01→NA) | disk I/O / mem bandwidth |
| **S1 Allele-freq reduction** | `device` (`decode_af_kernel.cu`) | in-kernel | dosages → `afmat`,`countmat` `[SNP_tile × pop]` | segmented reduction over samples within pop partition | mem bandwidth |
| **S2 f2 + block accumulate** | `device/cuda/f2_block_kernel.cu`; assembled by `core/fstats/f2_from_blocks.cpp` | `ComputeBackend::accumulate_f2` | afs → device-resident `f2_blocks [n_pop × n_pop × n_block]` + per-block pairwise valid-SNP count `Vpair`, bias-corrected, via the **3-GEMM reformulation** below, batched over blocks | three batched **native-FP64** GEMMs + a fused elementwise pre-pass and a tiny native-FP64 cancellation step, reduction into block bins via the shared `block_partition_rule` | compute + bandwidth, **streamed, runs once**; returns a device-resident `DeviceF2Blocks` (output adaptively tiered when it exceeds VRAM, §11.1/§11.2) |
| **S3 f3/f4 contraction** — **BUILT** (M(fit-1)) | `core/qpadm/f4_matrix.hpp` driver, orchestrated by `core/qpadm/qpadm_fit.cpp`; `CudaBackend::assemble_f4` override (`device/cuda/cuda_backend_fstats_assemble.cu` + `qpadm_fit_kernels.cu` gather kernels) | `ComputeBackend::assemble_f4` | `f2_blocks` → f4 matrix `X [(n_L−1)·(n_R−1) × n_block]` | element-wise linear combos, batched over block axis; the GPU path gathers from the **resident** f2 (zero D2H) | trivial / compute |
| **S4 Block jackknife → Q** — **BUILT** (M(fit-1)/M(fit-3)) | `core/qpadm/jackknife.hpp` driver, orchestrated by `qpadm_fit.cpp`; `CudaBackend::jackknife_cov` override (`device/cuda/cuda_backend_qpadm_fit.cu` + `qpadm_fit_kernels.cu`) | `ComputeBackend::jackknife_cov` | per-block f4 → covariance `Q [m×m]`, SEs | LOO replicate refits batched over the block axis; covariance SYRK in `EmulatedFp64{40}` (S4 only) | compute |
| **S5 Rank test (SVD)** — **BUILT** (M(fit-2)) | `core/qpadm/ranktest.cpp` (**orchestrates**, backend-agnostic); `CudaBackend::rank_sweep` override (`device/cuda/cuda_backend_qpadm_fit.cu`) | `ComputeBackend::rank_sweep` | `X` → rank sweep (`χ²`,`dof`,`p`) + AT2 `rankdrop`/`popdrop` tables | small-path on-device Jacobi (nl,nr≤32); cuSOLVER `gesvdj`/`gesvd` LARGE path for the `nr>32` tail (see note); `svd_path` reports the executed routine | batched/per-model linalg |
| **S6 qpAdm GLS fit** — **BUILT** (M(fit-1)) | `core/qpadm/gls_solve.hpp` driver, orchestrated by `qpadm_fit.cpp`; `CudaBackend::gls_weights` override (`device/cuda/cuda_backend_qpadm_fit.cu` + `qpadm_fit_kernels.cu`) | `ComputeBackend::gls_weights` | `X,Q` → weights `w`, χ² | one constrained GLS (`opt_A`/`opt_B` ALS) solve per model (see note) | batched dense LA |
| **S7 p-value / nested test** — **BUILT** (M(fit-2)) | `core/qpadm/nested_models.cpp` + `qpadm_fit.cpp` result assembly | host-only | χ², dof → p; rank-decision `f4rank` | embarrassingly parallel over models | trivial |
| **S8 Model-space search** — **BUILT** (M(fit-6)) | `core/qpadm/model_search.{hpp,cpp}` + host-pure `model_search_core.{hpp,cpp}`; public `run_qpadm_search` (`include/steppe/qpadm.hpp`); `CudaBackend::fit_models_batched` override (`device/cuda/cuda_backend_qpadm_fit.cu` + `qpadm_fit_kernels.cu`) | reuses backend (genuinely batched, NOT a per-model loop) | resident `f2_blocks` across many models | massive task-parallel; same-shape buckets dispatched batched (`batched_dispatch_count ≪ #models`); G≥2 shard deferred (run single-GPU, see note + §11.4) | throughput / scheduling |

**Note on the genotype-path stats tools (the `core → io` edge).** Beyond the precompute pipeline above, the shipped standalone genotype tools (`dstat`/`qpfstats`/`dates` in `core/stats/`, plus `extract_f2`) read the `io` genotype front-end **directly from `core`** (`src/core/CMakeLists.txt` links `steppe::io` PRIVATE) and dispatch decode through `ComputeBackend` — so for these tools the S0 read originates in `core`, not `app`. This is the deliberate `core → io` edge in §4; `io` remains a leaf (it still depends on nothing upward).

**Note on S2's 3-GEMM reformulation (the production hot path).** Per SNP block (`s` SNPs, `P = n_pop`), the fused pre-pass emits four dense `P×s` column-major matrices, where `p` is the per-pop *allele frequency* (not a dosage sum) and `N` is the per-pop *non-missing allele count* at that SNP (`2·#non-missing diploids` — alleles, not individuals; the AT2 bias-correction convention, pinned to a golden): `Q` (zero-filled allele frequencies — the zero is what makes the masked GEMM correct), `V` (validity mask, 1 valid / 0 missing), `Qsq = Q⊙Q`, and `Hc = Q⊙(1−Q)/max(N−1,1)⊙V` (per-SNP het bias correction). Stack `S = [Qsq ; Hc]` (`2P×s`). Three GEMMs then yield the reduced statistics — replacing the old "outer product over pop axis per SNP" with library dense LA:

```
G[P×P]     = Q @ Qᵀ            # cross-products over shared SNPs
Vpair[P×P] = V @ Vᵀ            # shared-valid SNP count per pop pair (RETAINED as the jackknife weight)
R[2P×P]    = S @ Vᵀ            # ONE GEMM yields BOTH masked reductions:
                               #   R_diag = Qsq@Vᵀ (rows 0..P-1), H = Hc@Vᵀ (rows P..2P-1)
f2_num[i,j] = R_diag[i,j] + R_diag[j,i] − 2·G[i,j] − H[i,j] − H[j,i]
f2[i,j]     = (Vpair[i,j] > 0) ? f2_num[i,j] / Vpair[i,j] : 0
```

Per SNP valid in both pops this is `(p_i−p_j)² − p_i(1−p_i)/(N_i−1) − p_j(1−p_j)/(N_j−1)` — the AT2 unbiased f2 estimator. Masking-by-multiply (`@ Vᵀ`) turns "sum over the SNPs valid in the *other* population of the pair" into a matmul, so per-SNP missingness needs no per-pair scalar loop — this is the **pairwise-complete** valid-mask path that gives AT2 parity (§5 S0′, §12). **Precision (load-bearing, §12 — MEASURED on real AADR).** `f2_num = R_diag + R_diagᵀ − 2G − H − Hᵀ` is a difference of large like-magnitude sums (`Σp_i² + Σp_j² − 2Σp_i p_j`) — catastrophic cancellation. But emulation survives it with **few fixed slices**: a capped **32-bit mantissa** gives 8.6e-9 worst-case f2 error, **40-bit** gives 2.2e-11 (≈ native FP64), while running **8–17× faster than native FP64** on the 5090 (the lead grows with population count). So the **f2 GEMMs default to `EmulatedFp64{mantissa_bits=40}`**, not native. The trap to avoid is *dynamic*-mantissa Ozaki: on real data's wide dynamic range it overshoots to ~60 bits → ~parity with native (no win). The small `O(n_pop²)` numerator/divide step stays native FP64; native FP64 is the validation oracle and the fallback. The covariance SYRK / f4 assembly (§5 S3/S4) likewise use fixed-slice emulation. Batched over all `n_block` blocks via `cublasDgemmStridedBatched` (uniform shapes; pad `s` to a common stride), accumulating into the resident `f2_blocks` and `Vpair`. Two custom kernels only: the **fused elementwise pre-pass** (`decode_af`→`Q,V,Qsq,Hc` in one tile sweep, native FP64) and the **fused numerator+divide** (native FP64, `O(P²·n_block)`). **AT2-parity caveats, each pinned to a golden:** (a) `Vpair` (the per-block, per-pair pairwise-complete SNP count) is **carried forward alongside `f2_blocks`** and is the weighted-block-jackknife weight at S4 (AT2 weights blocks by their pairwise non-missing count); the per-block divide here and the S4 weighting must compose to AT2's `f2_blocks` definition exactly, not be applied twice; (b) the `Vpair==0 ⇒ 0` branch and the `q(1−q)/max(N−1,1)` denominator + the allele-count `N` convention must match AT2 — all of which live in the shared `__host__ __device__` feeder primitive so the oracle and the GEMM path cannot diverge on the formula (§13).

**Multi-GPU parallelism for S0–S2 and S8 (planned; run single-GPU today).** In single-node multi-GPU runs the `io` streamer shards SNP tiles across the G devices (round-robin / static range / host queue, §9, §11.4); each device accumulates its own partial `f2_blocks` + `Vpair` from the tiles it owns, then the G partials are combined once — the parity path sums them **host-side in fixed device order** (not via NCCL AllReduce — §11.4, §12) and broadcasts the result. After the (broadcast) `f2_blocks` is replicated on every GPU, **S8's model-space search shards across the GPUs with zero inter-GPU communication** — each fit is wholly on one device (its internal reductions stay single-GPU and inherit single-GPU parity), and results are re-sorted by model index on the host so ordering is deterministic regardless of which GPU produced which model. On the no-P2P consumer 5090s the one-time `f2` replication host bounce makes G≥2 not worth it at AT2 scale, so this shard is deferred behind `TODO(multigpu-host-bounce)` (§11.4) — the shipped path runs single-GPU.

**Note on S5 batched-SVD limits.** `gesvdjBatched` is practical only for small matrices (per NVIDIA's batched-Jacobi guidance, roughly `m,n ≤ 32`). qpAdm's rank-test matrix is `(n_L−1)×(n_R−1)`; with the "thousands of right configurations" this tool targets, **`n_R−1 > 32` is common**, which drops the model off the batched path. The contract is explicit: when either dimension exceeds the batched limit, S5 falls back to a **per-model `gesvd`** (or a QR-based rank estimate) issued in a loop over a small number of streams. This fallback is launch-bound (§11) — exactly the regime to watch on Nsight Systems — and the model-space scheduler (S8) must keep enough independent solves in flight to hide the per-launch latency. `svd_path` reports which path each run took.

**Note on S6 GLS.** qpAdm's fit is a **single generalized-least-squares solve** given `Q`: Cholesky-factor `Q` (`potrf`), then solve the weighted normal equations for the admixture weights `w` and the residual χ². It is **not** an iterative sweep. The only iteration in the vicinity is the *outer model-space search* (S8), which re-runs this one-shot solve per candidate model. (Parity note: the as-built weight solve matches AT2's constrained `opt_A`/`opt_B` ALS, validated against the `qpadm()` golden.)

**Note on the fit engine (S3–S8): BUILT, golden-gated on the GPU, and the backend is finished.** As of the M(fit-0..6) milestones the entire fit chain exists and the production CUDA-backend path is validated against **real-AADR AT2 goldens**: `tests/reference/goldens/at2/golden_fit0.json` (9-pop, `nr≤32` small path), `golden_fit1_NRBIG.json` (`nr=39` cuSOLVER `gesvd` LARGE path), `golden_rot.json` (84-model rotation), `golden_qpwave.json` (a real `qpwave()` run), and `golden_fitNA.json` (real-AADR `maxmiss=0.99`, one real dropped block). The `CpuBackend` is the native-FP64 oracle the GPU is diffed against (run under `STEPPE_THOROUGH`). The orchestration lives in `core/qpadm/` (host-pure, layering-legal); the device overrides live in `src/device/cuda/cuda_backend_qpadm_fit.cu` (`jackknife_cov`/`rank_sweep`/`gls_weights`/`fit_models_batched`) and `src/device/cuda/cuda_backend_fstats_assemble.cu` (`assemble_f4`), with kernels in `src/device/cuda/qpadm_fit_kernels.cu`. The public C++ entry points are `steppe::run_qpadm` / `steppe::run_qpwave` / `steppe::run_qpadm_search` in `include/steppe/qpadm.hpp` (a model references populations by **index** into the resident `f2_blocks` P axis — the GPU-first compute seam, no strings); `run_qpwave` is a tested first-class entry — its no-target-prepend / `left[0]`-is-reference semantic is gated on both backends against `golden_qpwave` (`tests/reference/test_qpwave_parity.cu`). **Per-model domain failures are returned as `status` values, not crashes — M(fit-5) ships as THREE outcomes (`RankDeficient`, `NonSpdCovariance`, `ChisqUndefined`; `include/steppe/error.hpp`)**: `ChisqUndefined` closes a `dof≤0` model that would otherwise silently leak a NaN `p` with `status==Ok`, guarded on both the host and the CUDA model-batched path; `tests/reference/test_qpadm_domain.cu` asserts all three on degenerate REAL-AADR models on both `CpuBackend` and `CudaBackend`. **NA / missing-block handling (F1):** steppe's resident f2 is **pairwise-complete** (not the AT2 `maxmiss=0` global intersection), so a per-pair `Vpair==0` block CAN occur on sparse AADR and was being silently imputed `f2=0` (bias toward 0); it now follows AT2 `read_f2(remove_na=TRUE)` — **DROP** any partially-covered block before the LOO/jackknife — via the single shared host/device predicate `core::pair_block_is_missing` (`src/core/internal/f2_estimator.hpp`), on `CpuBackend` and the GPU (`f2_block_keep_kernel` in `qpadm_fit_kernels.cu`, single-model and S8 survivor-compaction). The legacy `maxmiss=0` goldens stay BYTE-IDENTICAL (no-drop identity arm). The opt-in jackknife-SE policy (`JackknifePolicy{None,FeasibleOnly,All}`, default `All`; `include/steppe/qpadm.hpp`) lets the rotation pay the expensive LOO SE only for the survivors worth reporting (M(fit-3 SE-policy)). The dead public `QpAdmOptions::constrained` field was removed (F5, API hygiene — non-negative constrained weights is a deferred step-3 feature, not shipped). **Productization (step-2) is now BUILT.** The earlier "no CLI / no Python bindings / no standalone f3/f4/D-stat entry points" status is stale: a CLI11 command-line app ships under `src/app/` (`main.cpp` dispatching `cmd_qpadm`/`cmd_qpwave`/`cmd_rotate`/`cmd_qpdstat`/`cmd_f3`/`cmd_f4`/`cmd_extract_f2`/…; CLI11 fetched PRIVATE to the app subtree via CPM in `src/app/CMakeLists.txt`), a nanobind Python module `steppe._core` ships under `bindings/` (`nanobind_add_module(_core …)` in `bindings/CMakeLists.txt`; `bind_qpadm`/`bind_fstats`/`bind_dates`/`bind_qpgraph`/`bind_f2handle`), and the standalone stat entry points are public headers `include/steppe/{f3,f4,dstat,extract}.hpp` with cores in `src/core/qpadm/f3.cpp`/`f4.cpp` and `src/core/stats/dstat.cpp`. **Multi-GPU caveat (deferred):** the S8 rotation is genuinely batched and shards across GPUs, but on the no-P2P consumer 5090s the one-time `f2` replication is a ~8.72 GB / ~3.8 s host bounce — only ~1.21× at 9086 real models — so **run the rotation single-GPU**; G≥2 is deferred (`TODO(multigpu-host-bounce)`, §11.4).

The decisive consequences: keep `f2_blocks` and `Q` GPU-resident; stream and fuse S0–S2 so the full `[SNP × pop × pop]` array is never materialized; exploit `n_block` + model index as batch dimensions; and route every device primitive through `ComputeBackend` so the same orchestration code runs against `CpuBackend` for the reference seam. **BUILT (M5):** the S2 precompute satisfies the "keep `f2_blocks` GPU-resident" corollary directly — it returns a device-resident `DeviceF2Blocks` handle (the result stays in VRAM for the downstream fit; host materialization is opt-in `.to_host()`), rather than the old forced host round-trip (~4.3× at `P=512`). When the output is itself larger than VRAM (thousands of pops) it falls back to the fastest tier it fits — host RAM, then disk — selected at runtime (§11.1, §11.2); at AT2-typical scale it stays resident.

**Preprocessing stages (S−2, S−1, S0′) before S0–S2.** Genotype QC / data-munging (§1) slots in *upstream of the kernel boundary* as `io`-owned, host-orchestrated, out-of-core stages emitting only plain data — exactly the property that keeps `io` a leaf (§4). They are additive: they produce the same harmonized tile the existing S0 already needed (dosage + validity mask `V` + per-SNP/per-pop sample sizes `N` + block ids from the shared `block_partition_rule`), so S1/S2 are unchanged in shape — S2's masked-GEMM reformulation already consumes `V`/`N`.

| Stage | Owner (orchestration) | Pass type | Input → output | Layering note |
|---|---|---|---|---|
| **S−2 Source schema + merge plan** | `io/merge` (host) | metadata-only (no genotype read) | per-source `{.bim/.fam}`/`{.snp/.ind}` → harmonized SNP set (intersection default / union optional), per-source allele-polarity map from *declared* alleles only (ref/alt swap → `2−dosage`; A/T·C/G ambiguous and multiallelic sites **dropped, never strand-flipped by frequency guesswork** — §1), sample/pop column maps, a `.missnp`-equivalent dropped-SNP list | host-pure leaf; reads `core::block_partition_rule` only as a consumer (§8) so the SNP→block map is bit-identical to the single-dataset path. No CUDA, no upward dep. Merge is a *plan*, not an on-disk rewrite (§1). |
| **S−1 QC pre-pass (conditional)** | `io/filter/prepass` (host, streams tiles) | one light streaming pass, **only if** `--mind` requested (or an external `prune.in` supplied) | dosages → per-sample non-missing counts (`--mind`); an externally-supplied `prune.in` is read, not computed → resolved include/exclude sets folded back into the plan | streams via the same tiler; emits plain sets. We do **not** compute LD ourselves (§1). Skipped entirely when no aggregate filter is requested. |
| **S0′ Harmonized+filtered tile produce** | `io` decode (host) → `app` wires the decoded tile into `ComputeBackend::decode_af` | the existing out-of-core tile loop (§11.1) | packed tile → harmonized tile in reference polarity + `V` + `N` + block ids, applying **cheap in-tile filters** (MAF/geno/include-exclude, plus flag-gated monomorphic/autosome-only/ts-tv) and the missing-data policy | this is S0 with harmonization + cheap-filter folded into the `io`-side decode; **`io` does not depend on `device`/`ComputeBackend`** — `app` wires `io` output into compute for this preprocessing stage, and (for the shipped genotype-path stats tools) so does `core` via the deliberate `core → io` edge (§4). Same `block_partition_rule` → S2 block bins unchanged. |

Cheap filters decidable from one tile (or from `.bim/.fam` metadata) are applied *in-tile* before S2 accumulation — a dropped SNP simply contributes nothing to its block, so jackknife block identity is unchanged (the §8 DRY invariant holds). The **default missing-data handling is pairwise-complete** (emit `V`/`N`; this *is* the parity path, no new math); **imputation (mean-fill `2p`) is optional** and tagged so its outputs are never bit-compared to AT2 goldens (§12). Heavy-missingness smoothing (`qpfstats`-style) is a fit-engine mode at S3+, not a preprocessing stage.

---

## 6. Build system

Target-based, modern CMake. The top-level `CMakeLists.txt` sets project policy and calls `add_subdirectory`; each `src/` subdirectory owns exactly one target. The reusable logic lives in `cmake/` modules (which define no targets): `SteppeOptions`, `CUDAArch`, `SteppeWarnings`, `SteppeSanitizers`, plus `CPM` (opt-in leaves only) and `Docs` (doc-only). There is **no** `third_party/` tree and no top-level dependency fetch.

```cmake
# CMakeLists.txt (top level) — project policy + add_subdirectory ONLY.
cmake_minimum_required(VERSION 3.28)          # the sm_120 build box ships cmake 3.28
project(steppe VERSION 0.1.0
        DESCRIPTION "GPU/CUDA qpAdm: f-statistics & model fitting"
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)                    # do NOT trust nvcc's host-derived default
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)          # clang-tidy/IWYU/clangd (Ninja/Make only)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SteppeOptions)      # all option()/cache vars in one place
include(CUDAArch)           # CMAKE_CUDA_ARCHITECTURES policy
include(SteppeWarnings)     # defines INTERFACE target steppe::warnings
include(SteppeSanitizers)   # consumes STEPPE_SANITIZER; MUST precede add_subdirectory()

find_package(CUDAToolkit REQUIRED)             # cuBLAS/cuSOLVER/cuFFT for steppe_device
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)                 # SPMG per-device host fan-out

if(STEPPE_BUILD_PYTHON)                         # nanobind _core is a SHARED module linking
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)       #   the first-party STATIC libs → all PIC
endif()

if(STEPPE_CCACHE)                               # opt-in ccache launcher (default ON); a no-op
  find_program(CCACHE_PROGRAM ccache)           #   when ccache is absent (stock box unaffected)
  if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER  "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  endif()
endif()

add_subdirectory(include)                      # steppe_api            (INTERFACE public headers)
add_subdirectory(src/io)                       # steppe_io            (LEAF genotype readers)
add_subdirectory(src/core)                     # steppe_core_internal (INTERFACE) + steppe_core
add_subdirectory(src/device)                   # steppe_device        (CUDA, PRIVATE cuBLAS…)
if(STEPPE_BUILD_CLI OR STEPPE_BUILD_PYTHON)    # shared CUDA-free host helpers
  add_subdirectory(src/access)                 #   steppe_access  (f2-dir reader + pop resolver)
  add_subdirectory(src/extract)                #   steppe_extract (run_extract_f2 + STPF2BK1 writer)
endif()
if(STEPPE_BUILD_CLI)      add_subdirectory(src/app)      endif()  # steppe CLI (CLI11)
if(STEPPE_BUILD_PYTHON)   add_subdirectory(bindings)     endif()  # steppe._core wheel (nanobind)
if(STEPPE_BUILD_EXAMPLES) add_subdirectory(docs/examples) endif() # read_f2→qpadm quick-starts
if(STEPPE_BUILD_TESTS)    enable_testing(); add_subdirectory(tests) endif()
if(STEPPE_BUILD_DOCS)     include(Docs)         endif()  # Doxygen + pdoc (doc-only; §16)
```

All build switches live in `cmake/SteppeOptions.cmake`: `STEPPE_BUILD_TESTS` (ON), `STEPPE_BUILD_CLI` / `STEPPE_BUILD_PYTHON` / `STEPPE_BUILD_EXAMPLES` / `STEPPE_BUILD_DOCS` (OFF), `STEPPE_WERROR` (ON), `STEPPE_CCACHE` (ON), `STEPPE_HAVE_EMU_TUNING` (ON), `STEPPE_NVTX` (OFF), `STEPPE_SANITIZER` (""), `STEPPE_COMPRESS_FATBIN` (OFF).

**`CMAKE_CUDA_ARCHITECTURES` policy** (`cmake/CUDAArch.cmake`). M0 dev pins a single concrete arch — sm_120 (the RTX 5090 validation box) — through the `STEPPE_CUDA_ARCH` cache var (default `"120"`), **not** `native`. Under CUDA 13 + CMake 3.28, `project(... LANGUAGES CUDA)` auto-populates `CMAKE_CUDA_ARCHITECTURES` to the toolkit baseline (sm_75) during compiler detection *before* this module runs, so a plain `if(NOT DEFINED …)` cannot fix it; the module treats the bare toolkit default (`""`, `"75"`, or `"OFF"`) as "unset by the user" and replaces it with `STEPPE_CUDA_ARCH`, while an explicit `-DCMAKE_CUDA_ARCHITECTURES=…` (or a preset) still wins. The shippable base list is held separately in `STEPPE_CUDA_ARCH_RELEASE` and is **not** applied by default — the `release` preset selects it:

```cmake
set(STEPPE_CUDA_ARCH "120" CACHE STRING "M0 target arch (consumer/workstation Blackwell sm_120)")
set(STEPPE_CUDA_ARCH_RELEASE
    "75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
    CACHE STRING "Shippable base arch list (Turing→Blackwell + one PTX fallback)")
```

The release list ships **base** architectures only. `100` (datacenter Blackwell) is **not** cubin-compatible with `120` despite both being "Blackwell"; `103`/`110`/`121` and the `a`/`f` accelerated family targets are deliberately omitted (narrow SKUs, not generic-fatbin-friendly). If a kernel ever needs an accelerated/family path, set `CUDA_ARCHITECTURES OFF` on *that one target* and pass explicit `-gencode arch=compute_100a,code=sm_100a` via `target_compile_options`, version-tested against the CMake in use.

**Separable compilation (RDC).** `steppe_device` sets `CUDA_SEPARABLE_COMPILATION ON` (backends launch kernels from a separate TU via narrow `launch_*` wrappers, §7) and `POSITION_INDEPENDENT_CODE ON` (the latter also required for the Python `_core` shared module):

```cmake
# src/device/CMakeLists.txt
set_target_properties(steppe_device PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON)
```

> **Planned (deferred).** Multi-arch **device LTO** — `-dlto` at both compile and device-link with per-arch `lto_<arch>` intermediates — is not wired at M0; it lands with the release matrix (see the comment in `src/device/CMakeLists.txt`). When added it must be driven explicitly per-arch: CMake's `INTERPROCEDURAL_OPTIMIZATION` conflicts with a pinned `-gencode` list (it would either mean host-only LTO or fight the list). Verify with `cuobjdump --dump-elf` that the final binary carries real SASS for each `-real` arch and that LTO actually fired.

**Flag policy** (`cmake/SteppeWarnings.cmake`). One INTERFACE target `steppe_warnings` (alias `steppe::warnings`), linked PRIVATE by every first-party target so flags never propagate to consumers, with host vs device flags split by generator expression (nvcc rejects raw GCC/Clang flags). Warnings-as-errors is gated by `STEPPE_WERROR` (default ON; CI/release pin it ON — OFF is a dev escape hatch that keeps `-Wall -Wextra` but drops the `-Werror` tokens):

```cmake
target_compile_options(steppe_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>              # host C++ TUs
  $<$<COMPILE_LANGUAGE:CUDA>:--Werror;all-warnings>            # nvcc warning surface (TWO tokens)
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror> # host-forward through nvcc
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>)  # Nsight source map
# all-warnings is the argument to --Werror, so it is a SEPARATE token (a single
# "--Werror=all-warnings" string is rejected). -G (full device debug) is Debug-only and
# never combined with -lineinfo — -G makes kernels many× slower and timing unrepresentative.
```

**Sanitizers** (`cmake/SteppeSanitizers.cmake`). `STEPPE_SANITIZER` (empty default) is translated here, included *before* any `add_subdirectory` so its directory-scoped flags propagate. Empty is a no-op (the Release/CI parity build stays byte-identical); `asan;ubsan` adds `-fsanitize=address,undefined -fno-omit-frame-pointer` on **host C++ TUs only** (`$<COMPILE_LANGUAGE:CXX>`, so `.cu`/device objects are excluded) with matching `$<LINK_LANGUAGE:CXX>` link flags; `compute` sets **no** codegen flags — just a marker (`STEPPE_SANITIZER_COMPUTE`) the CI compute-sanitizer GPU lane reads to wrap `ctest`. Any other value is a hard configure error.

**Optional fatbin compression.** CUDA 13's nvcc supports `--compress-mode` to shrink fatbins. Exposed as an opt-in cache var, Release-only, on `steppe_device`:

```cmake
# src/device/CMakeLists.txt
if(STEPPE_COMPRESS_FATBIN)
  target_compile_options(steppe_device PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--compress-mode=size>)
endif()
```

**Dependency fetching (CPM, opt-in leaves only).** `cmake/CPM.cmake` is a pinned bootstrap shim (CPM v0.42.3, downloaded and SHA256-verified into the build/cache tree, no vendoring). It is `include()`d **only** from the two CUDA-free leaf subtrees that need a fetched dependency, each behind its own option, and neither the CPM machinery nor its deps reach core/device (§4 layering):

- `src/app` (behind `STEPPE_BUILD_CLI`) — **CLI11**: `find_package(CLI11 2.4 CONFIG)` first, CPM fetch of v2.4.2 as the offline fallback.
- `bindings` (behind `STEPPE_BUILD_PYTHON`) — **nanobind**: `find_package(nanobind CONFIG)` / the pip-installed `nanobind.cmake_dir()` hint first, CPM fetch of v2.4.0 as the fallback.

**GoogleTest is not CPM-fetched.** `tests/CMakeLists.txt` uses `find_package(GTest CONFIG)` and, when GTest is absent, falls back to a self-checking executable that returns non-zero on failure (all CTest needs to gate). There is no CCCL/Eigen/spdlog dependency in the build.

**`CMakePresets.json`** — presets `base` (hidden), `dev`, `dev-asan`, `dev-compute`, `release`, `ci`. `base` pins arch `120` and turns on emulation tuning; it sets **no** compiler launcher (ccache is wired inline in `CMakeLists.txt` behind `STEPPE_CCACHE`):

```json
{
  "version": 6,
  "cmakeMinimumRequired": { "major": 3, "minor": 28, "patch": 0 },
  "configurePresets": [
    { "name": "base", "hidden": true, "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/${presetName}",
      "cacheVariables": {
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "CMAKE_CUDA_ARCHITECTURES": "120",
        "STEPPE_HAVE_EMU_TUNING": "ON" } },
    { "name": "dev",   "inherits": "base",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "RelWithDebInfo", "STEPPE_BUILD_TESTS": "ON" } },
    { "name": "dev-asan",    "inherits": "dev", "cacheVariables": { "STEPPE_SANITIZER": "asan;ubsan" } },
    { "name": "dev-compute", "inherits": "dev", "cacheVariables": { "STEPPE_SANITIZER": "compute" } },
    { "name": "release", "inherits": "base",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Release", "STEPPE_NVTX": "OFF",
        "STEPPE_WERROR": "ON", "STEPPE_BUILD_TESTS": "OFF",
        "CMAKE_CUDA_ARCHITECTURES": "75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual" } },
    { "name": "ci", "inherits": "release", "cacheVariables": { "STEPPE_WERROR": "ON", "STEPPE_BUILD_TESTS": "ON" } }
  ],
  "buildPresets": [
    { "name": "dev", "configurePreset": "dev" },
    { "name": "dev-asan", "configurePreset": "dev-asan" },
    { "name": "dev-compute", "configurePreset": "dev-compute" },
    { "name": "release", "configurePreset": "release" },
    { "name": "ci", "configurePreset": "ci" }
  ],
  "testPresets": [
    { "name": "dev", "configurePreset": "dev", "output": { "outputOnFailure": true } },
    { "name": "ci",  "configurePreset": "ci",  "output": { "outputOnFailure": true } },
    { "name": "host", "inherits": "ci", "filter": { "exclude": { "label": "gpu" } } }
  ]
}
```

The `host` test preset excludes the `gpu` label — exactly the CUDA-free CI lane (`ctest -LE gpu`). `compile_commands.json` is produced by Ninja and consumed by clang-tidy/IWYU/clangd. ccache (opt-in, `STEPPE_CCACHE` default ON) is nvcc-aware and hashes the full compile command plus preprocessed source, so cached objects are byte-identical — zero codegen/parity impact, purely a rebuild-latency win on the heavy `.cu` recompiles.

---

## 7. CUDA coding standards & idioms

**RAII wrappers — the only place CUDA resources are owned.** Application memory is allocated *only* inside `DeviceBuffer<T>` (device, `src/device/cuda/device_buffer.cuh`) and `PinnedBuffer<T>` (page-locked host, `src/device/cuda/pinned_buffer.cuh`) — the two translation units allowlisted to call the allocation family (`cudaMalloc`/`cudaFree`, `cudaHostAlloc`/`cudaFreeHost`) directly. Both are **fully move-only: move-construct AND move-assign** (a deleted move-assign would silently make `buf = std::move(other)` ill-formed). Everything else takes a non-owning raw device pointer + explicit extent — an owning type hands out a bare pointer via `data()`, never a copy or a view type.

```cpp
// src/device/cuda/device_buffer.cuh — move-only owning device allocation.
template <class T>
    requires std::is_trivially_copyable_v<T>   // raw-byte cudaMemcpy contract
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) : size_(n) {
        if (n) {
            if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))     // fail-fast:
                throw CudaError(cudaErrorMemoryAllocation,                    // unsigned wrap
                                "n*sizeof(T) overflows size_t",              // would under-alloc
                                std::source_location::current());
            STEPPE_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));             // ALLOWLISTED TU
        }
    }
    DeviceBuffer(DeviceBuffer&& o) noexcept
        : ptr_(std::exchange(o.ptr_, nullptr)), size_(std::exchange(o.size_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { reset(); ptr_ = std::exchange(o.ptr_, nullptr);
                          size_ = std::exchange(o.size_, 0); }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { reset(); }

    T*          data()  noexcept { return ptr_; }        // raw ptr for cuBLAS/kernel args
    std::size_t size()  const noexcept { return size_; }
    std::size_t bytes() const noexcept { return size_ * sizeof(T); }  // exact §11.2 budget term
private:
    void reset() noexcept {
        if (ptr_) { cudaError_t e = cudaFree(ptr_);                  // dtor never throws;
                    STEPPE_DEBUG_ONLY(if (e) STEPPE_LOG_WARN(        // BARE free — no
                        "cudaFree at teardown: {}", cudaGetErrorString(e))); }  // cudaSetDevice
        ptr_ = nullptr; size_ = 0;
    }
    T* ptr_ = nullptr;  std::size_t size_ = 0;
};
```

Two load-bearing details in that owner: the ctor is **fail-fast on an `n*sizeof(T)` overflow** (unsigned wrap would under-allocate and hand every downstream kernel a too-small buffer), which is what makes `bytes()` exact for the §11.2 VRAM budget; and `reset()` issues a **bare `cudaFree` with no `cudaSetDevice`** on purpose — `cudaFree` is pointer-device-aware, so a buffer may be freed under a different ambient device than the one it was allocated on (the multi-GPU escape seam, §11.4).

Library handles follow the **same fully-movable shape**; create them **once** at startup (`cublasDestroy`/`cusolverDnDestroy` implicitly synchronize, so never per-iteration), and route destroy-time errors to a debug log rather than swallowing them:

```cpp
// src/device/cuda/handles.hpp
class CublasHandle {
public:
    explicit CublasHandle(cudaStream_t s = nullptr) {
        CUBLAS_CHECK(cublasCreate(&h_)); if (s) CUBLAS_CHECK(cublasSetStream(h_, s));
    }
    CublasHandle(CublasHandle&& o) noexcept : h_(std::exchange(o.h_, nullptr)) {}
    CublasHandle& operator=(CublasHandle&& o) noexcept {
        if (this != &o) { destroy(); h_ = std::exchange(o.h_, nullptr); }
        return *this;
    }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    ~CublasHandle() { destroy(); }
    cublasHandle_t get() const noexcept { return h_; }
private:
    void destroy() noexcept {
        if (h_) { cublasStatus_t s = cublasDestroy(h_);
                  STEPPE_DEBUG_ONLY(if (s != CUBLAS_STATUS_SUCCESS)
                      STEPPE_LOG_WARN("cublasDestroy at teardown: status {}", int(s))); }
        h_ = nullptr;
    }
    cublasHandle_t h_ = nullptr;
};
// Stream, Event (stream.hpp), CusolverDnHandle (handles.hpp), PinnedBuffer<T>
// (pinned_buffer.cuh) all follow this exact shape: move-construct + move-assign +
// dtor->destroy()-with-debug-log. No exception escapes a destructor.
```

The snippet above is the **RAII shape** reference. The production `CublasHandle` additionally **owns the (stream, workspace) invariant** the §12 emulated-FP64 determinism contract depends on: it holds the non-owning `(ptr, bytes)` workspace and exposes `set_workspace(ptr, bytes)` + `set_stream(stream)`, where `set_stream` **re-applies** the pinned workspace after `cublasSetStream`. This is required because `cublasSetStream` "unconditionally resets the cuBLAS library workspace back to the default workspace pool" (cuBLAS §2.4.7), so any stream change after the workspace is bound would otherwise silently discard the §12 reproducibility workspace. The backend binds both **once** at construction and the GEMM routines never call raw `cublasSetStream` — so the workspace survives for every GEMM batch on both the block and grouped/strided-batched paths (the batched path otherwise reset it per chunk). `CusolverDnHandle` has no such hazard (`cusolverDnSetStream` does not reset workspace) and instead pins the cuSOLVER deterministic-results mode. The single statistic stream rule (§12) is unchanged.

**`STEPPE_CUDA_CHECK` + post-launch checks.** One macro (`src/device/cuda/check.cuh`) captures file/line/function via `std::source_location` and throws a typed `CudaError`; sibling `CUBLAS_CHECK`/`CUSOLVER_CHECK` throw `CublasError`/`CusolverError` off those status enums (they do not share `cudaGetErrorString`). Every API call is checked; every launch is followed by a synchronous *and* (debug-only) async check:

```cpp
#define STEPPE_CUDA_CHECK(expr) /* cuda_check(expr, loc) — throws CudaError w/ source_location */

#define STEPPE_CUDA_CHECK_KERNEL()                 \
  do { STEPPE_CUDA_CHECK(cudaGetLastError());      /* bad launch config, synchronous */ \
       STEPPE_DEBUG_ONLY(STEPPE_CUDA_CHECK(cudaDeviceSynchronize())); /* async fault */ \
  } while (0)
```

The debug sync makes compute-sanitizer attribute faults to the exact kernel; release relies on the next runtime call to surface the sticky error. Throwing (not `exit()`) lets tests `catch (const CudaError&)`.

**Launch-config helpers** live once in `src/core/internal/launch_config.hpp`: `cdiv(n, block)` (int and `long` overloads — the SNP count `M` can exceed 2^31), `grid_for(n, block, max_grid)`, `grid_z_extent(n)`, the grid-dimension limits `kMaxGridX`/`kMaxGridY`/`kMaxGridZ`, and the decode kernel's named block dims `kDecodeBlockX`/`kDecodeBlockY` (hard-coded constants, not occupancy-derived). Kernel files never recompute grid math. A CUDA grid is capped per axis at `(x, y, z) = (2^31−1, 65 535, 65 535)` on every compute capability incl. Blackwell sm_120 (`kMaxGridZ` is single-sourced from `kMaxGridY`), so **the large (SNP/`M`-scale) extent always rides `gridDim.x`** — putting it on `y`/`z` is a latent launch failure at `M > ~1.05M` SNPs. `grid_for` clamp-asserts its extent against `max_grid`; the strided-batched gather/scatter set `gridDim.z = n_in_group` **directly** (bypassing `grid_for`), so they route through the dedicated `grid_z_extent(n)` guard (asserts `1 ≤ n ≤ kMaxGridZ`) and the backend tiles the batch into ≤ `kMaxGridZ`-block chunks so the z extent never exceeds the limit — the `kMaxGridZ` cap is folded into `core/internal/vram_budget.hpp`'s `max_blocks_per_chunk`, the same chunk loop that already tiles for VRAM.

**CCCL usage policy.** Use the highest abstraction that fits; drop down only for control. Thrust for whole-container ops; **CUB** when you need explicit temp storage, a specific stream, or fused operators. The CUB calls in the codebase all use the **classic two-call temp-storage query idiom** (`d_temp_storage == nullptr` to size, allocate once, then call) and all sit on **off-statistic-path** sweep/decode work:

- `cub::DeviceSelect::Flagged` — kept-column / keep-mask compaction (`cuda_backend_decode.cu`).
- `cub::DeviceScan::ExclusiveSum` — the compacted-column index / decode seam (`cuda_backend_decode.cu`).
- `cub::DeviceRadixSort::SortPairsDescending` — the sweep top-K reservoir sort (`cuda_backend_fstats_assemble.cu`).

**Statistic-path reductions do not use CUB device reductions at all** — the reproducible arithmetic runs through cuBLAS GEMM (§5 S2), so bit-stability is a cuBLAS-workspace + single-stream property (§12), not a CUB concern. This keeps the two worlds separate: the throughput-only CUB idiom above needs no determinism guarantee, and the statistic path never mixes a pre-sized temp buffer with a determinism request on the same call.

**Interface currency at the host↔device boundary.** Device kernels are reached *only* through narrow `void launch_*(...)` wrappers declared in the `*_kernel.cuh` headers; the `<<<>>>` and kernel bodies live only in the `.cu` files, so host code never includes a kernel body. The wrapper ABI is **raw device pointers + explicit extents + stream**, e.g.:

```cpp
// src/device/cuda/f2_block_kernel.cuh
void launch_f2_feeder(const double* dQ_raw, const double* dV_raw, const double* dN_raw,
                      double* dQ_masked, double* dV_out, double* dS,
                      int P, long M, cudaStream_t stream);
```

Host-side backend orchestration entry points take `std::span<const T>` for input *host* arrays (e.g. the decode/D-stat drivers in `cuda_backend_decode.cu` / `cuda_backend_dstat.cu`), and a tiny host-pure `MatView` (`src/core/internal/views.hpp`) is shared by the CPU oracle and GPU feeder so both index identically. *(Planned)* Generalizing the device ABI to `cuda::std::span`/`mdspan` view types is aspirational — `views.hpp` notes the intended `span_view.hpp` generalization, but it is not built; kernels currently take raw pointers.

> **CUDA-13 note (verified):** cooperative groups can **no longer** be used for multi-device synchronization — the multi-device launch APIs (`cudaLaunchCooperativeKernelMultiDevice`, `multi_grid_group`) were removed. Single-device grid/warp collectives still work; prefer `cg::reduce` over hand-rolled `__shfl` *within a kernel*.

**Allocation is synchronous.** *(Planned, not built.)* The device path uses only synchronous `cudaMalloc`/`cudaHostAlloc` through the two owners above. Stream-ordered pool allocation (`cudaMallocAsync`/`cudaFreeAsync`, high `cudaMemPoolAttrReleaseThreshold`) is **not** implemented — there is no separate pool allocator TU. The `use_mem_pool` field in `include/steppe/config.hpp` is forward-reserved and gates no compute (it wires to nothing today).

**Streams.** One `Stream` per independent lane; express cross-stream deps with `Event`, not device-wide syncs. **Caveat that propagates to §12:** the statistic-bearing GEMM/SYRK/SVD/Cholesky must run on a *single* stream when bit-stable goldens are required, because cuBLAS reproducibility does not hold across concurrent streams (§12). Independent *throughput-only* work may use multiple streams. *(Planned)* The fit engine replays the same kernel sequence across the model space, so capturing the per-model fit loop into a CUDA graph is an intended optimization — but graph capture (`cudaStreamBeginCapture`/`cudaGraphInstantiate`/`cudaGraphLaunch`) is not yet implemented; today the fit loop is issued eagerly.

**Host/device separation.** The **production hot path is library dense LA** (§5 S2): the heavy arithmetic is hand-issued cuBLAS/cuSOLVER GEMM/SYRK/Cholesky/SVD (native cuBLAS, **no array framework — no JAX/CuPy**, §0). Custom kernels are the *feeders and consumers* around those GEMMs — e.g. the S2 fused pre-pass (`Q,V,Qsq,Hc`) and the fused numerator+divide. Such a kernel is a thin `__global__` shell doing index math + bounds, calling a `__host__ __device__` pure per-element primitive that holds the per-element numerics — CPU-unit-testable and **shared with the reference backend so oracle and GPU can't diverge on a formula** (§13). The thin-shell-around-a-scalar-function model therefore describes the *reference backend and the per-element primitives*, **not** the structure of the hot path, which is reformulated tensor algebra (§2). Anything touching the CUDA runtime stays `__host__`-only in the platform layer; device functions are `noexcept`; shared headers use `cuda::std::` types.

---

## 8. DRY & shared utilities

All cross-cutting helpers live once in `src/core/internal/` (host-pure) or `src/device/cuda/` (CUDA), exposed through the `steppe::core_internal` INTERFACE target. The link wiring below makes the layering *compiler-enforced*:

```cmake
# src/device/CMakeLists.txt
add_library(steppe_device STATIC
    cuda/f2_block_kernel.cu cuda/decode_af_kernel.cu cuda/cuda_backend.cu
    cuda/qpfstats_jackknife_kernel.cu cuda/ratio_block_jackknife_kernel.cu
    cpu/cpu_backend.cpp ...)               # (abbreviated; ~30 TUs)
add_library(steppe::device ALIAS steppe_device)
target_link_libraries(steppe_device
    PUBLIC  steppe::api steppe::core_internal                  # CUDA-free seam + DRY helpers
    PRIVATE steppe::warnings CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cufft)
# CUDA libs are PRIVATE → core/cli cannot see a CUDA header. backend.hpp is CUDA-free.
# cufft is PRIVATE too — it exists only for the DATES autocorrelation LD engine.

# src/core/CMakeLists.txt
add_library(steppe_core ...)
add_library(steppe::core ALIAS steppe_core)
target_link_libraries(steppe_core
    PUBLIC  steppe::api steppe::core_internal
    PRIVATE steppe::device steppe::warnings steppe::io Threads::Threads)  # GPU only via ComputeBackend
```

| Concern | Single home | Contract |
|---|---|---|
| CUDA error check | `device/cuda/check.cuh` | `STEPPE_CUDA_CHECK(expr)` → throw `CudaError{file:line, cudaGetErrorName/String}`; `STEPPE_CUDA_CHECK_KERNEL()` adds post-launch `cudaGetLastError` + a debug forced-sync |
| cuBLAS / cuSOLVER / cuFFT check | `device/cuda/check.cuh` | `CUBLAS_CHECK` / `CUSOLVER_CHECK` / `CUFFT_CHECK` translate the respective status enums (cuFFT is the DATES engine) |
| Logging | `core/internal/log.hpp` | `STEPPE_LOG_WARN(fmt, …)` — the one realized level, a printf-style facade; it is the §7 teardown-warning sink the move-only RAII wrappers route a nonzero destroy status to. Debug-only (compiled out under `NDEBUG`); never `printf`/`cout` in lib code. *Planned:* fuller `STEPPE_LOG_INFO/ERROR` levels for a structured-logging backend (reserved, not built) |
| NVTX ranges | `core/internal/nvtx.hpp` | `STEPPE_NVTX_RANGE("name")` — RAII `nvtx3::scoped_range`, a single string-literal arg, **no color argument and no color palette**. Empty unless `STEPPE_NVTX` is defined (default OFF), so shipping object code is byte-identical |
| Launch math | `core/internal/launch_config.hpp` | `cdiv` (int + long), `grid_for(n, block, max_grid)` (debug-asserts the y/z 65 535 cap), the per-kernel block dims (`kDecodeBlockX/Y`), the hardware grid limits `kMaxGridX/Y/Z`, and `grid_z_extent(n)` for the M4 batch axis (which sets `gridDim.z` directly, bypassing `grid_for`). All `STEPPE_HD constexpr` |
| Host/device + debug | `core/internal/host_device.hpp` | `STEPPE_HD` (the one host/device qualifier), `STEPPE_DEBUG_ONLY`, `STEPPE_ASSERT` |
| Views | `core/internal/views.hpp` | `struct MatView` — non-owning, column-major `[P × M]` `double` view (element (i, s) at `data[i + P·s]`), the Q/V/N contract seam the CPU reference and the GPU feeder both index against. Plain pointers, host-pure, no CUDA. *Planned:* generalize to a `span_view.hpp` over `cuda::std::span`/`mdspan` |
| **SNP→block rule** | `core/domain/block_partition_rule.hpp` | `assign_blocks` (the AT2 SNP-anchored cumulative walk → per-SNP `block_id[]`), the inverse `block_ranges(block_id, M, n_block)` (per-block `[begin,end)` ranges, validated once), and the single cM→Morgan site `block_size_cm_to_morgans`; host-pure, consumed by `io` *and* both device backends (§5). `block_of` remains a valid per-SNP primitive but is **no longer on the assignment path** — the AT2 walk does not bin to a fixed grid |
| **Reference backend** | `device/cpu/cpu_backend.cpp` | `CpuBackend`, same `ComputeBackend` interface, scalar host implementation |

**The CPU reference backend is both DRY and the correctness anchor.** `ComputeBackend` (in `device/backend.hpp`, CUDA-free) is one interface with two implementations: `CudaBackend` (`device/cuda/cuda_backend.cuh` + the split `cuda_backend_*.cu` TUs) and `CpuBackend` (`device/cpu/cpu_backend.cpp`). The compute layer is written once against the interface and never branches on GPU-vs-CPU. The deliberately-naive scalar `CpuBackend` is what the GPU kernels are continuously diffed against in `tests/reference/` (e.g. `test_f2_equivalence.cu`, `test_qpadm_parity.cu`), and both are validated against ADMIXTOOLS 2 through the golden generators/verifier under `tests/reference/goldens/at2/scripts/` (`golden_*_generate.R`, `verify_bitexact.R`) and `tests/r/`. It also guarantees `import steppe` works GPU-free.

---

## 9. Configuration management

**Typed, immutable, layered, injected.** Resolution order (lowest precedence first): `compiled defaults < TOML file < env (STEPPE_*) < CLI`. A mutable `ConfigBuilder` accumulates the layers; `build()` validates once — fail-fast — and freezes into an immutable `RunConfig`. After construction, config is `const`. `build()` does **static, CUDA-free** validation only (token legality, ranges, flag conflicts); the device-dependent checks (VRAM budget, device presence, precision honorability) live one layer down and run at resource-build time — see the validation split below.

```cpp
// include/steppe/config.hpp (public)
struct Precision {                        // a struct, not a bare enum — carries kind + mantissa_bits
    enum class Kind {
        Fp64,           // Native FP64. The validation oracle / gold reference and the fallback. Used for
                        //   the cancellation-prone f2 numerator/divide and the ill-conditioned GLS/SVD,
                        //   regardless of the selected matmul precision (§12).
        EmulatedFp64,   // DEFAULT for the matmul-heavy stages incl. the f2 GEMMs (§5 S2, §12). Fixed-slice
                        //   Ozaki emulation with a FIXED mantissa-bit count (Precision::mantissa_bits,
                        //   default 40 ⇒ ≈ native FP64; 32 ⇒ 8.6e-9, faster). MEASURED 7–17× over native
                        //   FP64 on real AADR. FIXED slices only — dynamic mantissa control overshoots to
                        //   ~60 bits on real data and collapses to parity (the trap, §12). Accuracy-
                        //   approximate, not bit-identical; not IEEE-754 on specials.
        Tf32            // Opt-in fast/approximate. Model-space screening / ranking ONLY. Results carry a
                        //   precision tag, land in the loose tolerance tier, and are never emitted as a
                        //   reported est/se/z/p without recomputation in EmulatedFp64/Fp64 (§12).
    };
    Kind kind          = Kind::EmulatedFp64;
    int  mantissa_bits = kDefaultMantissaBits;   // 40; meaningful only for EmulatedFp64. FIXED cap, never dynamic.
    // Named factories (aggregate-preserving): Precision::fp64(), ::emulated_fp64(bits=40), ::tf32().
};

struct DeviceConfig {                     // resources injected, not globally discovered
    std::vector<int> devices;             // SINGLE source of truth for which/how many GPUs. empty ⇒ auto-
                                          //   enumerate all visible CUDA devices in enumeration order; a
                                          //   non-empty list PINS both the set AND the ordering, which is the
                                          //   fixed f2_blocks combine order (§11.4, §12). Size 1 ⇒ single-GPU.
                                          //   (No separate count field — count == devices.size().)
    Precision   precision;                // default EmulatedFp64{40}; mantissa_bits lives ON Precision, not here.
    bool        enable_peer_access = true;// MAY-WE knob: permit cudaDeviceEnablePeerAccess (§11.4).
    bool        prefer_p2p_combine = true;// WHICH-PATH knob: device-resident P2P combine when peer access is
                                          //   available, else the host-staged fixed-order baseline (§11.4).
    bool        deterministic      = true;// §12 bit-stability INTENT the reproducibility contract is phrased
                                          //   against (scoped-deterministic cuSOLVER + single statistic stream +
                                          //   fixed host-order multi-GPU combine).
    DeviceConfig::ForceTier force_tier = ForceTier::Auto;   // Auto = select_output_tier; else pin Resident/HostRam/Disk.
    std::string disk_cache_path;          // Disk-tier f2_blocks cache; empty ⇒ STEPPE_F2_CACHE_PATH, else the frozen default.
    // FORWARD-RESERVED (present but not wired; gate no compute): stream_count (pinned 1, §12 trap),
    //   search_streams (parked S8 rotation lanes), use_mem_pool.
};

// include/steppe/qpadm.hpp
enum class JackknifePolicy : int {        // the S8-rotation SE policy — NOT a delete-1-vs-bootstrap choice
    None         = 0,   // no SE for any model (fastest screen)
    FeasibleOnly = 1,   // SE only for models passing the feasibility criterion
    All          = 2    // SE for every model — THE DEFAULT
};                       // The SE itself is ALWAYS the delete-1/LOO block jackknife; there is no bootstrap mode.
```

The immutable value object and its builder live in `src/core/config/`:

```cpp
// src/core/config/run_config.hpp — immutable; const accessors only; ConfigBuilder is its only friend ctor.
class RunConfig {                         // holds the four resolved real structs + carried I/O paths
public:
    Command                 command()       const noexcept;
    const DeviceConfig&     device()        const noexcept;   // empty devices ⇒ auto-enumerate (§9)
    const QpAdmOptions&     qpadm_options() const noexcept;   // fudge/als/rank/rank_alpha/jackknife/...
    const FilterConfig&     filter()        const noexcept;
    const io::PopSelection& pop_selection() const noexcept;
    double                  blgsize_cm()    const noexcept;   // cM-valued; see unit note below
    // + carried I/O strings (geno/snp/ind, f2_dir, out_file, format, ...).
};

// src/core/config/config_builder.hpp — the ONLY mutable config type.
class ConfigBuilder {
public:
    ConfigBuilder& with_defaults();                                 // compiled struct defaults (lowest layer)
    ConfigBuilder& merge_file(const std::filesystem::path&);        // TOML, below env+CLI
    ConfigBuilder& merge_env();                                     // STEPPE_* prefix
    ConfigBuilder& merge_cli(const CliArgs&);                       // highest precedence
    [[nodiscard]] BuildResult<RunConfig> build();                  // validates HERE, fail-fast
    [[nodiscard]] const std::string& error_message() const noexcept;// human-readable reason for the app's stderr
};
```

`build()` returns `BuildResult<RunConfig>` — a header-only C++20 stand-in for `std::expected<RunConfig, Status>` (the toolchain compiles `-std=c++20`, where `std::expected` is unavailable; `src/core/config/build_result.hpp`). Its failure arm is `unexpected(Status::InvalidConfig)`, with the detailed reason exposed via `error_message()` so `printf` stays out of the library (§10). There is **no** `Error` type and no `resampling()/bootstrap_reps()/seed()` accessor; the fit entry points consume the four resolved structs directly rather than an abstract `GenotypeDataset`.

**Why `Precision` has three named modes (and what each is *for*).** The qpAdm hot path is matmul-heavy (covariance SYRK/GEMM over jackknife blocks, f4 assembly), and CUDA 13.x ships **Ozaki-scheme fixed-point FP64 emulation** in cuBLAS/cuSOLVER that targets native-FP64 accuracy at tensor-core throughput. So the default is **`EmulatedFp64`** for that matmul work; **`Fp64`** (native) is the oracle/reference and the path for the cancellation-prone elementwise math; **`Tf32`** is an opt-in screening mode. There is deliberately **no `Fp32`** and **no dynamic-mantissa option**. The load-bearing scope rule: **emulation governs only the well-conditioned matmuls.** Allele-frequency accumulation and the catastrophic-cancellation-sensitive f2/f4 subtraction `(a−b)(c−d)` stay in **native FP64 regardless of `precision`** — emulation faithfully computes the *matrix product* but cannot recover significant bits annihilated by a prior subtraction (§12). `EmulatedFp64` must be validated against native FP64 before it is trusted, and `Tf32` results are typed as candidate-rankings with the final number always recomputed in `EmulatedFp64`/`Fp64` (§12).

**Block-size unit (verified against ADMIXTOOLS 2 semantics).** AT2's `blgsize` default is `0.05` **Morgans** = **5 cM**. The user-facing `--blgsize` CLI flag (and the Python `blgsize=` kwarg) speaks **Morgans**, the AT2 convention — a bare `--blgsize 0.05` reproduces AT2's block partition. `ConfigBuilder::build()` converts Morgans → cM (× `kCentimorgansPerMorgan`, `config_builder.cpp`) into the cM-valued `RunConfig::blgsize_cm()` field; the block math then converts cM → Morgans at the single `block_size_cm_to_morgans` site in `block_partition_rule.hpp` before calling `assign_blocks` (which speaks Morgans). Two conversion sites, both single-homed; the only unit the *user* sees is Morgans.

**Dependency injection of resources.** `RunConfig` says *what*; a `Resources` struct supplies concrete handles, injected into every compute call. The backend is **per-device** — there is no single top-level backend, and **NCCL is intentionally absent** (the multi-GPU combine is `cudaMemcpyPeer` or host-staged, never an AllReduce; §11.4):

```cpp
// src/device/resources.hpp — built once by build_resources(config); move-only.
struct PerGpuResources {                              // one per device in DeviceConfig::devices
    int                             device_id;        // physical CUDA ordinal (== devices[g])
    std::unique_ptr<ComputeBackend> backend;          // per-device backend; OWNS this device's single statistic
                                                      //   stream + cuBLAS handle + cuSOLVER handle + emulated-FP64
                                                      //   determinism workspace one layer down (CudaBackend, §7/§8)
    BackendCapabilities             caps;             // probed: compute cap, free/total VRAM, can_access_peer,
                                                      //   emulated_fp64_honorable — recorded here, never on F2BlockTensor
};
struct Resources {
    std::vector<PerGpuResources> gpus;                // g = 0..G-1 = the FIXED combine order (§11.4); gpus[0] = root
    DeviceConfig                 config;              // the frozen device policy
    CombinePath                  last_combine_path;   // which transport the last multi-GPU run took (discovered state)
    MultiGpuTimings              last_multigpu_timings;// out-of-band phase timings (observability only)
};

// include/steppe/qpadm.hpp — fit entries take PRECOMPUTED f2 blocks + QpAdmOptions + Resources (no GenotypeDataset).
QpAdmResult run_qpadm(const device::DeviceF2Blocks&, const QpAdmModel&,
                      const QpAdmOptions&, device::Resources&);   // GPU-resident f2 (primary)
QpAdmResult run_qpadm(const F2BlockTensor&, const QpAdmModel&,
                      const QpAdmOptions&, device::Resources&);   // host-oracle / parity overload
```

**Validation split — where each check lives.** `ConfigBuilder::build()` is **CUDA-free by contract**: it parses + range-checks the raw string knobs into the typed structs and rejects, as `Status::InvalidConfig` with a reason in `error_message()`:
- `--device`: a non-numeric, negative, or **duplicate** ordinal, and `--device cpu` (GPU-only — `CpuBackend` is a test-only path, never a shipped backend);
- `--precision`: an unknown token (`emu40`/`emu32` ⇒ `EmulatedFp64{40|32}`, `fp64` ⇒ `Fp64`, `tf32` ⇒ `Tf32`);
- `--jackknife`: out of `0|1|2`;
- `--format`: not `csv|tsv|json`;
- numeric ranges: `fudge ≥ 0`, `als_iterations ≥ 1`, `rank_alpha ∈ (0,1)`, `blgsize > 0`, filter fractions `∈ [0,1]`, `min_sources ≥ 1`;
- a non-empty TOML path (no TOML parser is compiled in this build).

`build()` does **not** touch a device: the **VRAM budget** (`device/vram_budget.hpp`), **device presence / ordinal-in-range** (`build_resources` + the pure `validate_device_order` predicate, `device/resources.{hpp,cpp}`), and **precision honorability** (the `emulation_honorable` / `engage_f2_precision` predicate in `f2_block_kernel.cu`, which downgrades `EmulatedFp64` to native `Fp64` with a logged capability tag when the fixed-slice tuning API is unavailable — e.g. built with `STEPPE_HAVE_EMU_TUNING` off) all run at resource-build time and, for allocation, mid-run. An over-budget or failed allocation surfaces as `Status::DeviceOom` (a mid-run mapping of `cudaErrorMemoryAllocation`, `cuda_backend.cu`), not an up-front `build()` reject. Injected allocators/streams (not singletons) are what make compute unit-testable, and the CUDA-free config layer is unit-tested with no GPU.

> **Planned.** `DeviceConfig::stream_count`, `search_streams`, and `use_mem_pool` are present but **not wired** (they gate no compute; `stream_count` is a §12 trap that must stay 1). The multi-GPU S8 model-space *rotation* (the `search_streams` lanes) is likewise deferred to a single-GPU path pending the multi-GPU host-bounce work in §11.

## 10. Error handling, logging & observability

**Error taxonomy (the categories, not just the mechanism).** Callers must distinguish *recoverable domain outcomes* from *faults*. As built, the public boundary is the C++ `Status` enum (`include/steppe/error.hpp`) returned directly across the same-toolchain C++ surface; the full cross-toolchain C ABI — a richer C enum `steppe_status_t` plus an internal `Error` carrying a `category` and message (§16) — is **deferred to M(abi-1)** and not yet built, so `Status` is the whole public status type today. The as-built taxonomy has exactly six values:

| `Status` value | Category | Recoverable? | Meaning |
|---|---|---|---|
| `Ok` | — | — | success |
| `InvalidConfig` | config (fault) | caller fixes input | failed `ConfigBuilder::build()` static validation (bad arch list, conflicting flags, unhonorable precision) — fail-fast |
| `DeviceOom` | resource | maybe (smaller chunk/budget) | a device allocation failed (§11.2) |
| `NonSpdCovariance` | **domain outcome** | **yes** | `Q` not SPD; Cholesky failed. A *statistical* result (degenerate/collinear model), not a bug |
| `RankDeficient` | **domain outcome** | **yes** | the rank test / GLS hit a rank-deficient design `X`; the model is unidentifiable |
| `ChisqUndefined` | **domain outcome** | **yes** | dof ≤ 0 or χ² not computable for this model; `p` is left at its NaN sentinel |

The three **domain outcomes** are *expected* results of fitting some models in a large search; the API surfaces them as ordinary per-model statuses (the search records them and moves on), **not** as exceptions or process aborts. Faults are fail-fast. Two failure classes have **no enum value yet** (planned with the M(abi-1) C ABI): IO-format failures (malformed `.bed`/`.geno`/`.snp`/`.ind`, bad magic bytes) surface instead as a thrown `std::runtime_error` from the io leaf (`src/io/geno_reader.hpp`), and CUDA runtime faults surface as typed exceptions mapped at the app boundary (below) rather than a `CudaRuntime` status.

**Errors mechanism.** The config layer returns `BuildResult<T>` — a header-only C++20 stand-in for `std::expected<T, Status>` (`src/core/config/build_result.hpp`; the toolchain is `-std=c++20`, and `std::expected` is C++23). There is no `STEPPE_TRY` and no `Error` type. CUDA / cuBLAS / cuSOLVER / cuFFT *faults* throw typed exceptions — `CudaError` / `CublasError` / `CusolverError` / `CufftError` (`src/device/cuda/check.cuh`) — each carrying the call site (`file:line:function` via `std::source_location`), the failing expression, and the API status name/string, through the single macros `STEPPE_CUDA_CHECK` / `CUBLAS_CHECK` / `CUSOLVER_CHECK` / `CUFFT_CHECK`. `STEPPE_CUDA_WARN` is the non-throwing sibling for *recoverable* statuses (capability probes / pollers — e.g. P2P `canAccessPeer`="no", `cudaErrorPeerAccessAlreadyEnabled`): it logs one warn line and yields the `cudaError_t` so the caller can degrade instead of faulting. At the app boundary `device::device_fault_status` (defined in `src/device/cuda/cuda_backend.cu`, declared in `src/device/resources.hpp`) maps an *allocation-failure* exception (`cudaErrorMemoryAllocation` / `CUBLAS_STATUS_ALLOC_FAILED` / `CUSOLVER_STATUS_ALLOC_FAILED`) to `Status::DeviceOom` and every other typed exception to the catch-all runtime exit. cuSOLVER's per-call `int* devInfo > 0` (singular / not-SPD) is a **domain outcome** mapped to a `Status` value, never thrown. Fail-fast: static validation at `ConfigBuilder::build()` (CUDA-free, returns `InvalidConfig` only); post-launch `cudaGetLastError()` always plus a debug-only forced `cudaDeviceSynchronize` via `STEPPE_CUDA_CHECK_KERNEL()`. RAII destructors never throw but log a warning on nonzero destroy status in debug (§7).

**Logging.** Never `printf`/`std::cout` in library code. `src/core/internal/log.hpp` is the single logging facade. As built it realizes exactly **one** level — `STEPPE_LOG_WARN` — a printf-style `std::fprintf(stderr, "[steppe][warn] " …)` sink that emits one line in debug builds and is compiled out entirely under `NDEBUG` (release is silent at teardown, and, like `assert`, does not evaluate its arguments). It is the §7 teardown-warning sink the move-only RAII wrappers (`DeviceBuffer`, `Stream`, `Event`, cuBLAS/cuSOLVER/cuFFT handles) route a nonzero destroy status to, and the sink behind `STEPPE_CUDA_WARN`. **Planned:** the fuller `STEPPE_LOG_INFO`/`ERROR`/… levels and a swappable structured-logging backend (e.g. spdlog with `RunConfig`-driven sinks/levels/async and a Python-`logging` forwarder) are reserved but **not built**; the printf format string is the single seam such a backend would swap.

**Observability (NVTX).** `src/core/internal/nvtx.hpp` provides `STEPPE_NVTX_RANGE(name)` — an RAII `nvtx3::scoped_range` opened at a coarse phase boundary and closed at end of scope, taking a single string-literal label (no color, no palette). It is gated by the `STEPPE_NVTX` CMake option (set PRIVATE on `steppe_device` only by the `-DSTEPPE_NVTX=ON` profiling build); the shipping default (OFF) expands to nothing and includes no NVTX header, so release object code is byte-identical (`nm` shows no nvtx symbols) — zero-overhead-off is structural. It is a passive host-side timeline annotation only: no device work, no stream, no sync, so it never touches the single statistic stream (§12 parity is untouched). Markers sit only at coarse phase boundaries, never in per-block / per-quartet inner loops, so even an ON build does not perturb the relative kernel timeline.

```cpp
CudaBackend::ResidentBlocks CudaBackend::run_f2_blocks_resident(/* ... */) {
    STEPPE_NVTX_RANGE("f2_gemm");                          // no-op unless -DSTEPPE_NVTX=ON
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(/* ... */, stream));
    f2_block_kernel<<<grid, block, 0, stream>>>(/* ... */);
    STEPPE_CUDA_CHECK_KERNEL();
}
```

The eight ranges actually in the code are: `decode`, `f2_gemm`, `jackknife`, `qpfstats`, `qpfstats_smooth_prep`, `qpadm_fit`, `dates_curve`, `dates_fit_curves`. They name the pipeline phases on the Nsight Systems timeline and can trigger capture ranges (§11.3).

---

## 11. Memory & performance strategy

### 11.1 Out-of-core genotype streaming (the part that can actually fall over)

The precompute pass is bandwidth-bound *and out-of-core*: real EIGENSTRAT/PLINK datasets are 1M+ SNPs × thousands of samples (tens of GB of packed genotypes), far exceeding any single GPU's VRAM. The `f2_blocks` output is **scale-dependent** (`O(P²·n_block)`): MB-scale at AT2-typical population counts (where it stays resident, point 3 below), but tens of GB at thousands of pops (≈76 GB at `P=2500`). We therefore **never load the genotype matrix whole, and we do not assume the output is always resident either.** The mechanism (**BUILT — M5**):

1. **Tile over SNPs.** `io` streams the dataset as contiguous SNP tiles (rows). For each tile: decode (S0) → allele-freq reduce (S1) → f2 partial + block-bin accumulate (S2), then discard the tile. Only `afmat`/`countmat` for the *current tile* and the running `f2_blocks` accumulator are resident.
2. **Double-buffered, pinned, overlapped I/O.** Two `PinnedBuffer` staging slots per tile feed `cudaMemcpyAsync` on a copy stream while the compute stream processes the previous tile (classic two-buffer pipeline). Pinned memory is required for true H2D overlap; pin *only* the staging slots.
3. **Block accumulator, and its tiering when large.** `f2_blocks` is `[P × P × n_block]`. At AT2-typical `P` it is MB-scale and **stays on the device** across the whole stream and the entire downstream model search (the device-resident `DeviceF2Blocks` handle, `src/device/device_f2_blocks.hpp`); tiles only add their contribution via the shared `block_partition_rule`. **When the result is larger than free VRAM** (thousands of pops — e.g. 76 GB at `P=2500`), M5 spills it to the **fastest tier it fits**: *Resident* (VRAM, the unchanged resident path) → *HostRam* (block-by-block host spill into a host `F2BlockTensor`) → *Disk* (a `STPF2BK1` cache file — magic `kF2DiskMagic` in `src/device/f2_disk_format.hpp` — via a persistent pinned staging buffer). The tier is selected **at runtime** from a `cudaMemGetInfo` free-VRAM probe (`caps.free_vram_bytes`) + a `sysinfo` free-host-RAM probe (`free_host_ram_bytes()`), by the CUDA-free `OutputTier` + `select_output_tier()` in `src/device/tier_select.hpp` (`resolve_output_tier()` layers the `STEPPE_FORCE_TIER` / `config.force_tier` test override on top). Small `P` keeps the resident fast path with no penalty; large `P` streams out-of-core. The streamed tiers reuse the resident path's per-block gather/GEMM/assemble verbatim; only *where* a slab lands changes, never its bits (parity by construction, §12).
4. **`afmat`/`countmat` sizing, and SNP-tile input streaming.** These are per-tile `[T × n_pop]`, *not* `[SNP × n_pop]` — they scale with the tile, never the dataset. The **streamed** (HostRam/Disk) tiers additionally decode **only the current SNP-column tile** of the host inputs (the full inputs stay in host RAM), so their GPU per-chunk footprint is **`O(P·tile + P²·n_block)`, independent of the SNP count `M`** — replacing the all-`M` feeder envelope (`7·P·M` doubles resident, `resident_working_set_bytes` in `tier_select.hpp`) that the **Resident** tier still runs and that used to cap full-autosome runs at `P≲768` on a 32 GB card. This is what lets full-autosome `P=2500` (M=584131, n_block=757) complete on a single 32 GB GPU, with the streamed GPU peak bounded by `streamed_working_set_bytes` (`tier_select.hpp`).

**Per-block chunk sizing (not a global SNP tile).** There is no single global SNP-tile width `T`, no `bytes_per_tile`/`T_max` symbol. The M5 path chunks each jackknife-block bucket: `max_blocks_per_chunk()` (`src/device/vram_budget.hpp`) picks the most blocks whose one strided-batched chunk's transient slabs (`per_block_chunk_bytes`) fit `chunk_budget_bytes() = kMaxVramUtilizationFraction · (free_vram − resident tensors − cuBLAS workspace)`, clamped to the bucket size and the hardware grid-z limit `kMaxGridZ` (grid-z tiling), and **floored at 1** — a single block is always attempted, so an over-budget bucket OOMs cleanly mid-chunk (`Status::DeviceOom`, fail-fast) rather than being rejected up front.

`extract-f2 --dry-run` (`src/app/cmd_extract_f2.cpp`) prints the plan so users size jobs before launching: geno/snp/ind paths, `P`, precision, `blgsize → n_block`, filters, free VRAM/total and free host RAM, the f2 slab bytes/block and total result MiB, and the **resolved `OutputTier` verdict** (resident/host/disk) via the same `resolve_output_tier()` the real run calls. It does *not* print a `T_max` or tile count.

### 11.2 Workspace & VRAM footprint

This is the resident/streamed footprint the device path (`src/device/vram_budget.hpp`, `tier_select.hpp`) actually accounts for. **There is no `build()`-time VRAM gate** — `ConfigBuilder::build()` is CUDA-free by contract (§9; `src/core/config/config_builder.hpp`) and does static validation only (see below). With `n_pop = P`, blocks `n_block = B`, fit-matrix dim `m = (n_L−1)·(n_R−1)`, batch count `K` (models × replicates in flight):

| Resident object | Bytes | Notes |
|---|---|---|
| `f2_blocks` | `P² · B · 8` | FP64 storage, resident for the whole run (storage is FP64 in every precision mode — emulation/TF32 are operation modes, not storage types) |
| `Vpair` | `P² · B · 8` | the retained per-block pairwise-valid count (the S4 jackknife weight, §5 S2 (a)) — an EQUAL-sized FP64 tensor held co-resident with `f2_blocks` (device path: `dF2_all` + `dVpair_all`), so the resident pair is `2 · P² · B · 8` (`resident_tensor_bytes`, `kResidentTensorCount = 2`, `src/device/vram_budget.hpp`; cleanup X-13/B26) |
| `Q` (covariance) | `m² · 8` | per active model, in the fit path |
| f4 matrix `X` | `m · B · 8` | per active model |
| cuSOLVER `potrf`/`gesvd` workspace | `ws_solve(m) · 8` | **query the largest `_bufferSize` once** over the batch and reuse; size grows with `m` |
| cuBLAS SYRK/GEMM workspace | `kCublasWorkspaceBytes` | one workspace **per stream**; on the statistic path that is exactly one (§12) |
| Genotype working set (Resident tier) | `resident_working_set_bytes(P,M) = 7·P·M·8 + ws_blas` | the all-`M` feeder envelope; dominant during resident-tier precompute (`tier_select.hpp`) |
| Genotype working set (streamed tiers) | `streamed_working_set_bytes ≈ (P·tile + P²·nb)·8` | per-block-tile decode; **no `P·M` term** (`tier_select.hpp`) |
| Pinned staging | `2 · T · n_sample · sizeof(dosage)` | double-buffered |

**No up-front budget rejection, no memory pool.** `ConfigBuilder::build()` includes no device header and makes no CUDA call: it does static validation (token legality, ranges, GPU-only, flag conflicts) and returns `Status::InvalidConfig` — there is no summed `total_vram(P,B,m,K,T)` function and no `cudaMemGetInfo` gate here. VRAM fitting is decided **at run time** by the device path: `select_output_tier()` chooses the tier the result fits (§11.1), and a genuine over-commit surfaces as `Status::DeviceOom` mapped from `cudaErrorMemoryAllocation` mid-run (`src/device/cuda/cuda_backend.cu`), fail-fast. The sole device allocator is **synchronous `cudaMalloc`/`cudaFree`** (`src/device/cuda/device_buffer.cuh`, the only allowlisted TU); there is no CUDA memory pool / `cudaMallocAsync` / release-threshold knob (that remains a documented future switch). Per-chunk allocation churn is instead avoided by **pre-sizing the chunk slabs once** and reusing the freed feeder VRAM.

**Output-does-not-fit → adaptive tiering (BUILT, M5).** The `2·P²·B·8` resident pair is the *Resident* tier; it is `O(P²·B)` and at thousands of pops itself exceeds VRAM (≈76 GB at `P=2500`, ≈220 GB at `P=4266` — §11.4). When the result plus working set does not fit free VRAM, the run-time selector does **not** reject: `select_output_tier()` picks *HostRam* (block-by-block host spill) if it fits free host RAM, else *Disk* (`STPF2BK1` cache). The streamed tiers budget against the **real** streamed footprint `O(P·tile + P²·B)` (independent of `M`), **not** the resident all-`M` feeder — those terms are scoped to the *Resident* tier. So `Status::DeviceOom` is the verdict only when even a single streamed chunk cannot fit, not when the whole result cannot.

### 11.3 Performance & profiling

S0–S2 are bandwidth-bound: maximize coalescing on the `.bed` unpack, fuse decode→reduce→**feeders**(`Q,V,Qsq,Hc`)→3-GEMM→cancellation-step→block-bin so the `[SNP × pop × pop]` intermediate is never materialized (§5 S2, §11.1). S3–S8 are launch-/throughput-bound across many tiny solves: batch over `n_block` and model index (strided-batched GEMM/SYRK for homogeneous shapes, pointer-array batched only for non-uniform shapes), capture the per-model fit into a CUDA graph, and keep enough independent solves in flight to hide per-launch latency on the per-model SVD fallback (§5). On a multi-GPU box, S8 is additionally sharded across devices (§11.4).

**Profiling workflow — system-level before kernel-level.**
1. **Nsight Systems first** (where does time go?): `nsys profile --trace cuda,nvtx,osrt --cuda-memory-usage true -o run ./steppe ...`. Read `cudaapisum` (excessive `cudaMalloc`/launch overhead) and `gpukernsum` (dominant kernel). Look for GPU idle gaps (launch-bound) and un-overlapped copies (the tile pipeline of §11.1 should show overlapping copy/compute lanes). NVTX ranges name the phases.
2. **Nsight Compute second**, scoped to the kernel `gpukernsum` named: `ncu --set full --kernel-name regex:f2_block_kernel --launch-count 1 -o k ./steppe ...`. Read Speed-Of-Light + Roofline (f-stat reductions land memory-bound), Occupancy (and its limiter), then Memory Workload (DRAM throughput, L2 hit, coalescing). Never run `--set full` over a whole app; ncu replays kernels and is far slower than nsys.

### 11.4 Multi-GPU execution (single node)

Single-node multi-GPU is a day-one objective (§0, §1): parallelism across the G devices of one workstation **with parity** (§12). The model is **single-process-multi-GPU (SPMG)** — one host process driving all G devices, one host thread + per-device CUDA streams per device, `cudaSetDevice` to switch. One address space keeps `f2_blocks` and the model-space results in-process (no MPI, no launcher), and direct peer access (`cudaDeviceCanAccessPeer`/`cudaDeviceEnablePeerAccess`, §9 `enable_peer_access`) is enabled opportunistically. **NCCL is intentionally absent** (`src/device/resources.hpp`): steppe never puts the f2 reduction on NCCL AllReduce, whose order varies with GPU count and would break parity (§12). CUDA 13 removed the multi-device cooperative-launch APIs (`cudaLaunchCooperativeKernelMultiDevice`, `multi_grid_group`, §7), which is fine — steppe never spans one grid across devices; all cross-GPU coordination is host-level.

The two shardable phases, both additive on the existing tiling and per-stream design (§11.1, §9):

1. **Precompute (S0–S2): tile sharding + a fixed-order combine.** The streamer partitions SNP tiles across the G devices; each device accumulates its own partial `f2_blocks` (+ `Vpair`), with each block's accumulation order fixed so a block's partial is identical regardless of *which* physical GPU computed it. The G partials are then combined **in fixed device order** (`g = 0..G−1`). Two transports, both bit-exact and parity-safe:
   - **Device-resident P2P combine** (`src/device/cuda/p2p_combine.cu`), an opt-in `cudaDeviceCanAccessPeer`-gated fast path (RTX PRO 6000 stock driver; full-host 5090s via the aikitoria open-kernel-module P2P patch, dev-only): the root device places its own partial via D2D and pulls each peer's partial via **`cudaMemcpyPeerAsync`** (a byte-exact DMA) into disjoint result slices in the same fixed `g = 0..G−1` order, assembling a `DeviceF2Blocks` with no final D2H.
   - **Host-staged fixed-order combine**, the portable parity baseline and the only path on no-peer consumer cards: a host bounce is used purely as the cross-card assembly transport, then `upload_f2_blocks_to_device` re-uploads so the return is still a device-resident handle.

   Both are **bit-identical** to each other and to the single-GPU reference (the transport only moves bytes; software fixes the order). Because this cross-GPU traffic is kB–MB and off the critical path, P2P is architectural cleanliness, not a throughput lever — steppe is deliberately P2P/NVLink-insensitive. **Honest role (MEASURED, M4.5/M5):** tile sharding is a *modest* throughput layer, not the at-scale enabler — it was measured *slower* than single-GPU until the host data-bounce was removed. What actually unlocked scale on the precompute was getting the result off the CPU (device-resident output) and the M5 input/output streaming above (§11.1), not sharding. Multi-GPU's decisive payoff would be phase 2 (S8), next.

2. **Model-space search (S8): embarrassingly parallel per fit — but the one-time `f2` replication is NOT free on no-P2P cards.** With `f2_blocks` replicated (`replicate_f2` in `src/core/qpadm/model_search.cpp`), models are partitioned across devices via contiguous shards; each fit runs wholly on one device (single-GPU parity), and results are re-sorted by model index so the final ordering is deterministic regardless of which GPU produced which model. **PLANNED / DEFERRED (`TODO(multigpu-host-bounce)`).** MEASURED on real AADR (`P=600`): getting `f2_blocks` onto each GPU costs ~8.72 GB / ~3.8 s through host on the consumer 5090s (no GeForce P2P), which caps the rotation at ~1.21× (no 1.5× crossover) — so **the rotation runs single-GPU-preferred** on the 5090s (`qpadm-rotate` warns and recommends `--device 0` for `G≥2`). The intended fix is a **per-device precompute** (each GPU builds its own f2, zero cross-GPU transfer) or P2P hardware; until then multi-GPU rotation stays deferred.

Because the `f2_blocks` combine is off the bandwidth critical path (kB–MB at AT2-typical P; larger at thousands of pops, where the result is anyway tiered to host/disk per §11.1/§11.2 rather than replicated whole) and each S8 fit has zero inter-GPU traffic, steppe is largely insensitive to NVLink-vs-PCIe — the streaming build is bound by host→device ingest, disk, and the output spill, not inter-GPU links. **cuSOLVERMp stays deferred** (§1): it distributes one *large* dense factorization across GPUs, but qpAdm's `Q`/`X` are low-double-digit — the right parallelism is across many independent small fits (above), not within one distributed solve. **Multi-node** (multiple processes / MPI) is the remaining deferred boundary.

---

## 12. Numerical correctness & reproducibility

**Precision policy.** Three named modes (`Precision`, §9), assigned by the **conditioning of the operation, not by whether it is a matmul**. **The cancellation-sensitive elementwise math is always native FP64.** Allele-frequency accumulation and the f2/f4 difference-of-products `(a−b)(c−d)` are differences of large nearly-equal quantities — catastrophic cancellation, where the leading digits annihilate; if the operands enter at less than FP64 the rounding error already baked in becomes the dominant surviving content, and no downstream GEMM can reconstruct it. These stay native FP64 **regardless of `precision`** (the allele-freq / f2-feeder pass is bandwidth-bound anyway, so reduced precision buys no throughput). This fused elementwise pre-pass is `launch_f2_feeder` / `f2_feeder_kernel` in `src/device/cuda/f2_block_kernel.cu`. **The matmul-heavy covariance `Q` / SYRK / f4 assembly defaults to `EmulatedFp64`** — cuBLAS fixed-point (Ozaki-scheme) FP64 emulation on tensor cores; this is the well-conditioned accumulate-many-similar-terms regime where emulation earns its place.

**The f2 GEMMs use FIXED-slice Ozaki emulation.** The numerator `Σp_i² + Σp_j² − 2Σp_i p_j` (`= Σ(p_i−p_j)²`) is catastrophic cancellation, but emulation survives it with a *small, fixed* mantissa: on real AADR a **fixed 32-bit mantissa** yields ≈8.6e-9 worst-case f2 error (40-bit → ≈2.2e-11 ≈ native; 48-bit → ≈1e-12), well ahead of native FP64. So the f2 GEMMs default to **`EmulatedFp64{mantissa_bits=40}`** (drop to 32 when 8.6e-9 suffices). The trap: **dynamic** mantissa control auto-selects ~60 bits on real data's wide dynamic range — far more slices than f2 needs — collapsing to parity with native. Hence *fixed* slices, never dynamic.

**The FIXED pin is enforced by one predicate.** `emulation_honorable(const Precision&)` (declared `src/device/cuda/f2_block_kernel.cuh:53`, defined in `f2_block_kernel.cu`) is true only for `EmulatedFp64` on a build carrying the fixed-slice tuning API (`STEPPE_HAVE_EMU_TUNING`, gated in `src/device/CMakeLists.txt`). It is the **single** honorability source: the math-mode engagement (`engage_f2_precision`, cuBLAS `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH`), the cuBLAS compute-type mapping (`f2_compute_type`), and the solver-side mode (`engage_solver_precision`, taking `&emulation_honorable` as a function pointer) all consult it, so they can never disagree. Without the tuning API `cublasSetFixedPointEmulationMantissaControl(FIXED)` cannot be called and cuBLAS would default to dynamic; instead the path **downgrades to native `Fp64`** (PEDANTIC math + `CUBLAS_COMPUTE_64F`) and the capability probe reports `emulated_fp64_honorable = emulation_honorable(...)` (`src/device/cuda/cuda_backend.cu:238`) — never silently running dynamic under the `EmulatedFp64` tag.

**The GLS Cholesky and the rank-test SVD stay native `Fp64`**: small, so emulation buys no throughput, and oracle-grade arithmetic is what you want near rank-deficiency. **`Tf32` and FP16 never produce a reported statistic** — `Tf32` is permitted only as a fast model-space screen, with survivors recomputed in `EmulatedFp64`/`Fp64` before any `est`/`se`/`z`/`p` is emitted. **Mandatory gate:** `EmulatedFp64` is accuracy-approximate, not bit-identical to native FP64 (and not IEEE-754-compliant on special values), so every reported quantity is spot-validated against a native-FP64 oracle — recompute the covariance for a sample of jackknife blocks in native `Fp64` (PEDANTIC math) and require the downstream `est`/`se`/`z`/`p` to match; if it fails, raise the mantissa bits or fall back to native `Fp64`. The oracle is `CpuBackend` (`src/device/cpu/cpu_backend.cpp`), the reference lane of the `ComputeBackend` seam (§7), which walks the exact AT2 pairwise-complete path; the shared `__host__ __device__` f2 feeder primitive means oracle and GEMM path cannot diverge on the formula (§13). [UNCERTAIN] native-FP64 GEMM accumulation order is itself implementation-defined; bit-stability rests on single-stream execution + the oracle diff, not on assuming a source-order reduction.

**The determinism guarantee, stated precisely (and its boundaries).** The thesis "bit-stable on a given GPU → non-flaky regression gates" holds only under the following constraints. Some are enforced in code today; where the enforcement is currently *documented intent* rather than a coded gate, that is called out.

- **Reductions: fixed combination order, single stream.** The statistic-path reductions are **hand-rolled per-block `cub::BlockReduce`** (e.g. `qpfstats_jackknife_kernel.cu`) plus `cub::DeviceScan::ExclusiveSum`, `cub::DeviceSelect::Flagged`, and `cub::DeviceRadixSort`; a fixed tree/combination order under a single stream is what makes them reproducible. **[PLANNED]** the CCCL 3.1 single-phase determinism API (`cuda::execution::require(determinism::run_to_run)` on `cub::DeviceReduce`, and `gpu_to_gpu` for cross-GPU FP accumulation) is the intended upgrade — it is referenced as intent in `include/steppe/config.hpp` but is **not** wired into any reduction call yet.
- **cuBLAS: single stream on the statistic path.** cuBLAS's bitwise-reproducibility guarantee does not hold across concurrent streams (the library may pick different internal implementations regardless of per-stream workspaces). The statistic-bearing GEMM/SYRK/Cholesky/triangular-solve are therefore meant to run on **one** stream. `Config::stream_count` defaults to `1` (`include/steppe/config.hpp:413`) and the `deterministic` intent field documents the single-stream rule (`config.hpp:496-514`). **[GAP]** `ConfigBuilder::build()` does **not** currently read or force `stream_count`; determinism holds today because the default is 1, not because build() rejects `stream_count > 1`. (`CUBLAS_WORKSPACE_CONFIG=:4096:8` is the documented belt-and-suspenders for the single-stream path.)
- **cuSOLVER: deterministic mode, with its real scope.** `cusolverDnSetDeterministicMode(handle, CUSOLVER_DETERMINISTIC_RESULTS)` is pinned **once** in the `CusolverDnHandle` ctor (`src/device/cuda/handles.hpp:337-358`), right after `cusolverDnCreate`. The CUDA 13.x default is already `CUSOLVER_DETERMINISTIC_RESULTS`, so the pin is **defensive** — it forecloses a future toolkit default flip, with no behaviour change today. Per the cuSOLVER docs the mode covers the `geqrf`/`syevd`/`gesvd` (m ≥ n)/`gesvdj`/`gesvdr`/`gesvdp` family and their `X` variants, and **explicitly does NOT** cover `potrf`/`potri`/`getrf`/`getrs` (the SPD `Qinv` Cholesky and the ALS/weight/LOO LU solves), which stay native-FP64 outside the deterministic set. **Scope, precisely.** The mode governs only the loose-tier large/NRBIG rank-test SVD, `CudaBackend::large_svd_V` (`src/device/cuda/cuda_backend_qpadm_fit.cu:342`): `gesvdj` when `gesvdj_applicable(nl, nr)` — both dims ≤ `kGesvdjMaxDim` (32) (`cuda_backend_qpadm_fit.cu:311`) — else `gesvd`, for which the orientation rule `rows = max(nl,nr) ≥ cols = min(nl,nr)` (same file, `large_svd_scratch_sizes` / `large_svd_V`) satisfies the `m ≥ n` precondition `gesvd` needs. The **primary bit-exact 9-pop golden's** rank-test SVD is **not** cuSOLVER at all: it is a **custom on-device fixed-order Jacobi** (`launch_qpadm_rank_via_jacobi` / `dev_jacobi_svd_V`, `src/device/cuda/qpadm_fit_kernels.cuh`), deterministic **by construction** and untouched by this API. The reference-lane `handles_unit` test (`tests/reference/test_handles.cu`) asserts `CusolverDnHandle::deterministic_mode()` returns `CUSOLVER_DETERMINISTIC_RESULTS`.
- **Cross-GPU determinism.** Floating-point addition is non-associative, so a multi-GPU reduction's bits depend on reduction *order*, and a collective AllReduce's order changes with GPU count and buffer size — fatal for parity to ADMIXTOOLS 2. **NCCL is intentionally absent** from the parity path (`src/device/resources.hpp:122`, `src/core/fstats/f2_combine.cpp`, `src/device/cuda/p2p_combine.cu:41`): the `f2_blocks` (+ `Vpair`) combine gathers the per-device partials and places them in **fixed device order** `g = 0..G−1` (the `DeviceConfig::devices` order) into disjoint result slices — provably configuration-independent (same arithmetic order whether G = 1, 2, or 8), via `cudaMemcpyPeerAsync` peer copies, **never** a collective reduce. The S8 model search introduces no cross-GPU reduction (each fit is single-GPU; results re-sorted by model index on the host), so it inherits single-GPU parity directly. **[PLANNED]** multi-GPU *rotation* itself is deferred: the G ≥ 2 engine path is correct but throughput-capped by the host bounce on no-P2P consumer cards, so it runs single-GPU-preferred today — `TODO(multigpu-host-bounce)` in `src/core/qpadm/model_search.cpp:264`, `src/device/device_f2_blocks.hpp:92`, and `src/app/cmd_rotate.cpp:157`.

**Block/tree reductions: two distinct properties, not one.** A pairwise/tree reduction gives an *accuracy* bound of `O(ε log n)` versus `O(ε n)` for a naive sequential or `atomicAdd` chain — a bounded worst-case rounding improvement, not "free." Separately, a reduction is *deterministic* only if its combination order is fixed. The dominant reason to prefer a fixed-order `cub::BlockReduce` tree over long `atomicAdd` chains on the statistic path is that **`atomicAdd` accumulation order is itself the dominant nondeterminism source** we are eliminating; the accuracy bound is a secondary benefit. Treat the two arguments separately.

**Tolerance policy.** Never one absolute epsilon. Combined form `|a−b| ≤ atol + rtol·|b|`. Concretely: point estimates `est` ~`1e-9`–`1e-6` relative (near bit-stable under single-stream + native-FP64 elementwise), and `EmulatedFp64` covariance entries are held to this same tight tier against the native-FP64 oracle; jackknife `se`/`z` ~`1e-3` relative; the **rank decision** (and any value whose cuSOLVER routine is outside the deterministic set, or any TF32-screened intermediate) sits in the loose tier and is documented as such. TF32-screened results carry a precision tag and are **never** bit-compared to AT2 goldens — they are provisional shortlisting signals, promoted to `EmulatedFp64`/`Fp64` before any reported number. The `se` tolerance is **derived, not asserted**: the jackknife SE is a variance over `B` block-delete replicates, and the relative variability from reduction-order/resampling differences is empirically `O(1/√B)` for the `B≈700` blocks of a typical genome — landing near `1e-3`. The golden harness records the measured spread per fixture and sets `rtol` from the observed distribution plus margin.

**Resampling and block assignment.** The built statistic-resampling path is the **seedless delete-1 block jackknife** (`src/device/cuda/qpfstats_jackknife_kernel.cu`, `ratio_block_jackknife_kernel.cu`), moved on-device in native FP64 in the fixed row/block order of the host long-double reference — no RNG, so no seed to control. SNP→block assignment is the deterministic pure function of genetic position in `src/core/domain/block_partition_rule.hpp` (the `block_of` floor primitive over the `blgsize` block size, default 0.05 Morgans = 5 cM), reproducing AT2's `dis >= blgsize` boundary convention verbatim. **[PLANNED]** bootstrap (RNG) resampling is not yet implemented; when added it will use a counter-based generator so a draw is a pure function of `(seed, replicate, offset)` independent of launch geometry, with the seed recorded in golden metadata. No cuRAND/Philox generator exists in production code today.

**Validation against ADMIXTOOLS 2.** ADMIXTOOLS 2 has **no `seed` argument** — its resampling rides R's RNG state — so "bit-reproducible regeneration" of goldens is fragile across versions and is treated that way. R's stream depends on the R version, the `RNGkind`, and AT2's internal call order, so pinning `set.seed(N)` alone is insufficient. The regression goldens live under **`tests/reference/goldens/at2/`** (`golden_fit0.json`, `golden_fit1_NRBIG.json`, `golden_fitNA.json`, `golden_qpwave.json`, `golden_rot.json`, plus the qpgraph/qpfstats fixtures; see that directory's `README.md`). Their metadata records R version, `RNGkind`, ADMIXTOOLS 2 version, `blgsize` (=0.05 Morgans), and `boot` setting (`FALSE` for delete-1 jackknife, or integer `N`), alongside the dumped `qpdstat` `est/se/z/p`. Regeneration must reproduce that exact environment; cross-version drift is expected and is why `se`/`z` use the loose tier above. The diff uses the two-tier tolerance.

## 13. Testing strategy

The CPU reference backend is the oracle. `make_cpu_backend()` (long-double host path) and `make_cuda_backend()` (the GPU path) implement the same CUDA-free `ComputeBackend` seam, so every numerically subtle GPU result is diffed against the obviously-correct host computation rather than an inline re-implementation, and the shared per-element numerics are additionally pinned GPU-free.

- **Harness & lanes.** GoogleTest is the project harness (`find_package(GTest CONFIG QUIET)`); when it is not discoverable each TU falls back to a self-checking `main()` that returns non-zero on the first failure — all CTest needs to gate is the exit code. `tests/CMakeLists.txt` tags every test via the `steppe_add_test` helper with exactly one lane label `{unit|reference|cli|python}` plus orthogonal `{gpu}`/`{slow}` labels; `gpu` tests take `RESOURCE_LOCK "gpu"` so `ctest -j` never oversubscribes the single GPU, and `ctest -LE gpu` is exactly the CUDA-free host lane. (There is no `CudaTest`/`TearDown` device-reset fixture — that was never built.)
- **Host unit tests (`tests/unit/`, no GPU).** Pure C++ TUs over the shared `__host__ __device__` primitives: the f2 estimator (`test_f2.cpp` → `core/internal/f2_estimator.hpp`) and the SNP→block partition rule (`test_block_partition.cpp` → `core/domain/block_partition_rule.{hpp,cpp}`: `block_size_cm_to_morgans`, `assign_blocks`), alongside config-builder, backend-factory, decode, filter, shard-plan and VRAM-budget tests. f4 contraction and jackknife weighting live in `.cu` kernels and are covered by the reference/parity suite below, not a host `.cpp` unit test. The CUDA control-flow test `tests/reference/test_cuda_check.cu` is itself a self-checking `main()` asserting `STEPPE_CUDA_CHECK` faults while `STEPPE_CUDA_WARN` logs-and-continues on the same recoverable status.
- **Reference-equivalence (`tests/reference/`, GPU).** Identical inputs through `CpuBackend` and `CudaBackend`, asserted equal within the tight tolerance tier — the central trust seam. `test_f2_equivalence.cu` diffs the full `P × P` f2 matrix including the diagonal; `test_f2_blocks_equivalence.cu`, `test_qpadm_parity.cu`, `test_qpwave_parity.cu`, `test_dates_parity.cu`, the `qpfstats`/`qpgraph` parity tests, and `test_f2_determinism.cu` extend the seam to per-block f2, `Q`, `X`, `w`, χ², and the fit outputs.
- **Regression goldens (`tests/reference/goldens/at2/`).** ADMIXTOOLS 2 `est/se/z/p` on small committed fixtures (`golden_fit0.json`, `golden_fit1_NRBIG.json`, `golden_fitNA.json`, `golden_qpwave.json`, `golden_rot.json`, plus the qpgraph/qpfstats goldens) with the **full pinned environment** recorded per golden and in the directory README (R 4.3.3, `admixtools` 2.0.10, `RNGkind` Mersenne-Twister + seed 42, `blgsize=0.05`, `boot=FALSE`, `fudge=1e-4` — §12). Tight `rtol` on `est`, loose on `se`/`z`/rank. DATES goldens live alongside at `tests/reference/goldens/dates/`.
- **Domain-outcome tests (BUILT — F2 `c8fe397`, `tests/reference/test_qpadm_domain.cu`).** The M(fit-5) acceptance gate: degenerate **REAL-AADR** models (collinear left → `RankDeficient`; `fudge=0` singular `Q` → `NonSpdCovariance`; over-parameterized `dof≤0` → `ChisqUndefined`) assert all **three** domain outcomes (§10) are returned as a *value* with no crash/NaN leak, identical on `CpuBackend` and `CudaBackend`.
- **NA / missing-block tests (BUILT — F1 `2496a14`, `tests/reference/test_qpadm_missing_block.cu`).** Real-AADR `maxmiss=0.99` with a sparse right pop → one real `Vpair==0` block dropped, gated against `golden_fitNA`; plus a `maxmiss=0` no-drop arm asserting the legacy goldens stay byte-identical. Both backends.
- **qpWave parity (BUILT — F4 `6481dfa`, `tests/reference/test_qpwave_parity.cu`).** `run_qpwave` called directly against a real AT2 `qpwave()` golden on both backends — the first-class entry's distinguishing no-target / `left[0]`-reference semantic.
- **CLI & Python suites.** `tests/cli/*.cpp` are host TUs (no CUDA header) that spawn the built `steppe` binary, parse stdout, and reproduce the AT2 goldens *through the CLI*; they self-check and skip on the child's "no CUDA device" fault. `tests/python/*.py` drive the `_core` extension under pytest and compare against the committed AT2 goldens (`rtol` 1e-6 / 1e-12), skipping cleanly via `maybe_skip_no_gpu` when no device is visible.
- **compute-sanitizer.** The built seam is `cmake/SteppeSanitizers.cmake`: `STEPPE_SANITIZER=compute` sets the `STEPPE_SANITIZER_COMPUTE` marker for a GPU lane that wraps the normally-built binaries with `compute-sanitizer --tool memcheck ctest …` (**memcheck only**). A separate `STEPPE_SANITIZER=asan;ubsan` overlay adds `-fsanitize=address,undefined` to **host** C++ TUs only (gated `$<COMPILE_LANGUAGE:CXX>` so device/`.cu` objects are excluded). *Planned:* the full four-tool sweep (racecheck/initcheck/synccheck + `--track-stream-ordered-races all` for `cudaMallocAsync`, §7) and the CI runner that invokes any of these — there is no `.github/` workflow in the tree yet.

---

## 14. CI/CD & tooling

**No CI runner is committed.** There is no `.github/` directory and no workflow `*.yml`/`*.yaml` anywhere in the tree; the 5-stage pipeline, self-hosted GPU-runner matrix, sccache/GHA cache, pre-commit hooks, `CCCL_VERSION >= 3.1` assertion, nvbench regression gate, and cibuildwheel/auditwheel steps that earlier drafts described do **not** exist. The runner is *planned* (deferred to `docs/archive/kimiactions/02-ci-plan.md`). What is actually built is the local build/test/tooling **seam** a future runner would drive: preset lanes, a ccache launcher, the sanitizer CMake seam, an exported compile DB, and bare lint config.

### Preset lanes

`CMakePresets.json` defines the configure/build/test lanes:

- **`release`** — full ship arch list (`75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual`), `STEPPE_WERROR=ON`, `STEPPE_NVTX=OFF`, tests off.
- **`ci`** — inherits `release`, `STEPPE_BUILD_TESTS=ON` (the GPU lane).
- **`host`** — test preset inheriting `ci` with `filter.exclude.label = gpu` → the CUDA-free `ctest -LE gpu` subset that runs on a GPU-less box. GPU tests carry the `gpu` ctest label; `host` runs everything except them.
- **`dev` / `dev-asan` / `dev-compute`** — dev lanes; `dev-asan` sets `STEPPE_SANITIZER=asan;ubsan`, `dev-compute` sets `STEPPE_SANITIZER=compute`.

### ccache (build-iteration only)

`STEPPE_CCACHE` (`cmake/SteppeOptions.cmake`, default **ON**) sets `CMAKE_CXX_COMPILER_LAUNCHER` / `CMAKE_CUDA_COMPILER_LAUNCHER` to `ccache` (`CMakeLists.txt`, before any `add_subdirectory()`). It is a no-op when ccache is absent; ccache hashes the full compile command + preprocessed source and is nvcc-aware, so cached objects are byte-identical — a rebuild-latency win with **zero codegen/parity impact**. There is no sccache / remote cache.

### Sanitizer seam

`cmake/SteppeSanitizers.cmake` is the single place that translates the `STEPPE_SANITIZER` cache var (declared in `SteppeOptions.cmake`, default `""`) into behaviour; it defines **no targets**:

- **`""`** — no-op; the Release/CI parity build is never sanitized (byte-for-byte the stock build).
- **`asan;ubsan`** — `-fsanitize=address,undefined -fno-omit-frame-pointer` on **host C++ TUs only**, gated by `$<COMPILE_LANGUAGE:CXX>` (compile) and `$<LINK_LANGUAGE:CXX>` (link) so `.cu`/device objects and CUDA-device-linked targets are excluded (`dev-asan`).
- **`compute`** — no codegen flags; sets the `STEPPE_SANITIZER_COMPUTE` marker and selects the runtime GPU lane `compute-sanitizer --tool memcheck ctest …` on the normally-built binaries (`dev-compute`). Only **memcheck** is wired — racecheck/initcheck/synccheck are not.
- **anything else** — hard configure error.

_Planned:_ the lane **invocations** themselves (host asan/ubsan ctest; compute-sanitizer memcheck ctest) belong to the not-yet-committed runner; this module delivers only the CMake seam.

### Lint / static-analysis config

`CMAKE_EXPORT_COMPILE_COMMANDS` is `ON` (`CMakeLists.txt`; also in the base preset) so `compile_commands.json` feeds external tooling (clang-tidy, IWYU, clangd). Root **`.clang-format`** and **`.clang-tidy`** exist as bare config files; there is **no** pre-commit wiring (`.pre-commit-config.yaml` is absent) and no wired format/tidy/IWYU gate. A host-only `[lint]` extra (`ruff` + `mypy` over `bindings/steppe`) is provisioned in `pyproject.toml` for a future CUDA-free lint lane; nothing in-repo invokes it.

### CUDA arch

`cmake/CUDAArch.cmake` defaults `STEPPE_CUDA_ARCH=120` (the Blackwell sm_120 validation box). The `release` preset ships the full fatbinary list (`75/80/86/89/90/100/120-real` + `120-virtual`, == `STEPPE_CUDA_ARCH_RELEASE`), covering **sm_75–sm_120** (README). CUDA is verified on **13.0 and 13.1**.

### Wheels / packaging (see §15)

One **GPU-only** wheel via scikit-build-core + nanobind. The **CUDA-13 runtime is a system requirement resolved at load — not bundled, not a pip dep**. `_core.so` carries `DT_NEEDED` for `libcudart.so.13` / `libcublas.so.13` / `libcusolver.so.12` / `libcufft.so.12`, matching the device link set `CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cufft` (`src/device/CMakeLists.txt`; cuFFT is the DATES autocorrelation LD engine — **curand is not linked**). `numpy` is the only hard runtime dep. An optional `cuda` extra pins the **suffixed** `nvidia-*-cu13` redistributable wheels but is **deferred** until those publish a real `>=13` version (pinning the current `0.0.x` placeholders would make the wheel pip-uninstallable).

_Planned:_ cibuildwheel-over-scikit-build-core + `auditwheel repair` packaging (RPATH-injecting the pip runtime) is referenced only as a future step — there is no `CIBW_*` / auditwheel wiring in the repo.

---

## 15. Python bindings & packaging

**nanobind over pybind11**: substantially faster compiles, smaller binaries, and lower per-instance and call overhead; runtime is effectively identical for the large-NumPy-array workloads here, so we win on build/size and pay nothing at runtime. `nanobind_add_module(_core NB_STATIC STABLE_ABI ...)` (`bindings/CMakeLists.txt:64`) links nanobind's runtime statically into one self-contained `_core.abi3.so` and builds against CPython's limited API (`Py_LIMITED_API`), so a single wheel tagged `cp312-abi3` runs on Python 3.12/3.13/3.14+ with no per-version rebuild. nanobind is obtained find_package-first (pip-installed config) with a CPM fetch of the pinned v2.4.0 commit as the offline fallback.

The extension is `steppe._core`; users import the pure-Python facade `steppe` (`bindings/steppe/__init__.py`) layered over it. `bindings/module.cpp` is a **thin** `NB_MODULE` assembler — its body just calls `register_f2handle` / `register_qpadm` / `register_qpgraph` / `register_fstats` / `register_dates` in registration order (`register_f2handle` first, so the `F2Handle` type exists before any fit entry takes it). The per-tool binding code lives in the `bind_<tool>.cpp` TUs (`bind_f2handle.cpp`, `bind_qpadm.cpp`, `bind_qpgraph.cpp`, `bind_fstats.cpp`, `bind_dates.cpp`); the shared marshalling helpers live in `bindings/internal/bind_common.hpp`. Marshalling only: NumPy↔`std::span` conversion, name→index resolution against the f2-dir's `pops.txt`, `QpAdmOptions`/`DeviceConfig` fill from kwargs, and the result-struct → flat-Python-dict emitters (`result_to_dict`, `f4_to_dict`, …). No compute logic and no pandas link — the pandas/DataFrame shaping stays in the pure-Python facade. `_core` is a **plain C++20 host target** (no `LANGUAGES CUDA`, no `.cu` sources); it reaches the GPU only through the CUDA-free seams and links `steppe::{core,io,api,device,access,extract,warnings}`.

**GPU-only, fail-fast.** There is no `CpuBackend` in the bindings. When a fit runs on a box with no CUDA device, `ensure_resources()` (`bindings/internal/bind_common.hpp:100`) raises via `raise_no_device()` — `"no CUDA device available (steppe is a GPU product; a CUDA-capable GPU is required)"`. The Python acceptance tests (`tests/python/`) do **not** compare GPU vs CPU: each stages a committed AT2-derived golden f2 fixture, runs it through the binding seams on the GPU, and asserts the output against committed AT2 CSV/binary goldens (e.g. `test_py_f4.py`, est/se/z/p at `rtol 1e-6`). They **skip cleanly** when no CUDA device is visible via `maybe_skip_no_gpu` (`conftest.py:161`).

**scikit-build-core** is the PEP 517 backend driving the same CMake. There is **one GPU-only wheel — no CPU-only variant** (the CPU backend is test-only and never shipped as a wheel flavor). `cmake.build-type = "Release"` is mandatory (a Debug/unspecified build voids the precision/perf contract), the Blackwell arch is pinned (`-DSTEPPE_CUDA_ARCH=120`), and `wheel.py-api = "cp312"` selects the abi3 stable-ABI packaging. The CUDA-13 runtime is a **system requirement resolved at load** — `_core.so` carries `DT_NEEDED` for `libcudart.so.13` / `libcublas.so.13` / `libcusolver.so.12` / `libcufft.so.12` (cuFFT drives the DATES ancestry-covariance LD engine) and resolves them dynamically from the user's CUDA 13 install; the toolkit is deliberately **not** bundled and **not** a base pip dep. `numpy` is the ONLY hard runtime dep; `import steppe` works without a GPU, and any GPU call then raises the clear no-device fault.

```toml
# pyproject.toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2"]
build-backend = "scikit_build_core.build"

[project]
name = "steppe"
dynamic = ["version"]              # single-sourced from CMakeLists project(VERSION ...)
requires-python = ">=3.9"
dependencies = ["numpy>=1.22"]     # the ONLY hard runtime dep

# The CUDA-13 runtime is a SYSTEM requirement resolved at load (DT_NEEDED on _core.so),
# NOT bundled and NOT a base pip dep. Optional extras pull soft / redistributable deps.
[project.optional-dependencies]
pandas = ["pandas>=1.5"]           # soft/lazy DataFrame accessors (import steppe works without)
rds = ["pyreadr>=0.5"]             # optional AT2 .rds read path (steppe.import_f2_rds)
cuda = [                           # not yet on PyPI with a real >=13 version
  "nvidia-cuda-runtime-cu13>=13,<14",
  "nvidia-cublas-cu13>=13,<14",
  "nvidia-cusolver-cu13>=13,<14",
  "nvidia-cufft-cu13>=13,<14",     # cuFFT drives the DATES ancestry-covariance LD engine
]

[project.scripts]
steppe-rds = "steppe._rds:main"    # GPU-free f2 .rds converter CLI

[tool.scikit-build]
cmake.version = ">=3.28"
cmake.build-type = "Release"       # mandatory: a non-Release build voids the precision/perf contract
cmake.args = [
  "-DSTEPPE_BUILD_PYTHON=ON",
  "-DSTEPPE_BUILD_CLI=OFF",
  "-DSTEPPE_BUILD_TESTS=OFF",
  "-DSTEPPE_CUDA_ARCH=120",         # Blackwell sm_120; override via --config-settings for another GPU
]
build.targets = ["_core"]
wheel.packages = ["bindings/steppe"]
wheel.py-api = "cp312"              # stable ABI (abi3): ONE wheel tagged cp312-abi3 runs on 3.12+

# dynamic = ["version"] resolves from CMakeLists.txt project(VERSION ...) — CMake is the
# single version source (D2); the regex is project-anchored 3-part (NOT cmake_minimum_required).
[tool.scikit-build.metadata.version]
provider = "scikit_build_core.metadata.regex"
input = "CMakeLists.txt"
regex = 'project\([^)]*?VERSION\s+(?P<value>\d+\.\d+\.\d+)'
```

> The `cuda` extra is optional because the `-cu13`-suffixed redistributable wheels are not yet on PyPI with a real (>=13) version — pinning them as base deps would make the wheel pip-uninstallable. The base wheel documents CUDA 13 as a system requirement; the extra (and a future auditwheel `$ORIGIN` RPATH) is the path to a self-contained install once those wheels ship. **Planned:** multi-GPU is parked in the binding (`ensure_resources` fills a single-GPU `DeviceConfig`; `bind_common.hpp:103`) — the device upload happens per fit call so no `DeviceF2Blocks` crosses into Python, deferring any DLPack / `__cuda_array_interface__` zero-copy export.

---

## 16. Documentation & ABI/versioning

**ABI (as built, HEAD): a same-toolchain C++ convenience layer — there is NO C ABI yet.** `include/steppe/` is ordinary C++ (`error.hpp` exposes `enum class Status`, plus `config.hpp` and `fstats.hpp` whose `struct F2BlockTensor` holds `std::vector`/`std::span`; value/RAII throughout). There is no `steppe_status_t`, no `extern "C"`, no `src/c_api/`. This C++ layer **is** the current public surface; the binding (§15) links it in-process, same toolchain, and carries no cross-toolchain promise.

> **Planned (deferred to M(abi-1)).** The installed/versioned boundary is *designed* as a true **C ABI**: opaque handles (`steppe_f2_blocks_t*`, `steppe_qpadm_result_t*`), functions returning `steppe_status_t` (the §10 status enum), and accessor functions for results — no `std::` types, no templates, no exceptions across the line. That is what would make `find_package(steppe)` + a prebuilt library usable across toolchains, and what an ABI/SemVer promise would then cover; when it lands the C++ convenience layer wraps it. The reason for a C boundary rather than `std::expected` returns: `std::expected<T,Error>` is a `std::` type with library-defined layout, so exporting it (especially holding `std::string`/`std::vector`) reintroduces exactly the libstdc++/libc++/MSVC coupling a stable ABI must avoid.

**Internal error propagation uses `BuildResult<T>`** (`src/core/config/build_result.hpp`) — a minimal, CUDA-free, **C++20** stand-in for `std::expected<T, Status>` (the toolchain is `-std=c++20`; `std::expected` is C++23), exposing the slice the config/access layers use (`has_value`/`operator bool`, `value`/`operator*`/`operator->`, `error()`) plus an `unexpected(Status)` free function. It is internal only and crosses no public boundary, and retires for `std::expected` at C++23 with no call-site churn. There is no `STEPPE_TRY` macro and no `expected.hpp`.

**API docs (opt-in via `STEPPE_BUILD_DOCS`, `cmake/Docs.cmake`):** two independent custom targets, each a soft no-op when its tool is absent — `docs` runs **Doxygen** over `include/steppe/` (parses headers as text: no compiler, no CUDA, so it builds on the local non-Blackwell box), and `docs-python` runs **pdoc** over `bindings/steppe/`. Neither Sphinx nor Breathe is used.

**SemVer + stability:** the stability promise tracks the **C++ surface** in `include/steppe/`; nothing in `src/` is stable. Additive API ⇒ MINOR, breaking ⇒ MAJOR; the CUDA major version is encoded in the wheel tag and in the SONAME the wheel depends on (§14). The cross-toolchain **C-ABI** contract is the M(abi-1) end state described above.

**Version single-sourced on `project(VERSION ...)` in the top-level `CMakeLists.txt`.** The wheel version is derived from it by scikit-build-core's regex metadata provider (`pyproject.toml` `[tool.scikit-build.metadata.version]`, `dynamic = ["version"]`, `provider = scikit_build_core.metadata.regex` over `CMakeLists.txt` with a project-anchored 3-part regex). The `STEPPE_VERSION` macro is injected as a PRIVATE compile definition from `${PROJECT_VERSION}` (`src/app/CMakeLists.txt:72`, `src/extract/CMakeLists.txt:42`) and consumed by the CLI `--version` flag (`src/app/cli_parse.cpp`, falling back to `"0.0.0+unknown"` only if undefined) and stamped into extract `meta.json` (`src/app/cmd_extract_f2.cpp`). Macro and wheel version therefore never drift — bumping `project(VERSION)` moves every surface. *(Planned: a generated `version.hpp` from `project(VERSION)` is the deferred M(abi-1) end state; today the compile-definition + regex-metadata path carries the same single-source guarantee, and no `version.hpp`/`version.hpp.in` exists in the tree.)*

**Design decisions are recorded inline in this architecture document** (the numbered sections), not in a separate ADR log — there is no `docs/adr/`, and no `CHANGELOG.md`. The public C++ targets `steppe::api` and `steppe::core` are in-tree CMake `ALIAS` targets (`include/CMakeLists.txt`, `src/core/CMakeLists.txt`); they are **not** installed or exported — there is no `SteppeConfig.cmake` and no `install(EXPORT)`/`install(TARGETS)` for the C++ libraries. The only `install()` in the tree ships the Python `_core` extension into the wheel (`bindings/CMakeLists.txt:91`).

---

## 17. Scaffold manifest (ordered)

This is the as-built layer order, bottom-up: each step's targets depend only on those above it. An engineer or agent reading top-to-bottom sees how the tree is assembled into a building, testable whole. Two code comments cite step numbers here (`cmake/CPM.cmake:1` → step 2; `include/CMakeLists.txt:3` → step 5), so those numbers are load-bearing.

1. **Repo + tooling roots:** `.gitignore`, `.clang-format`, `.clang-tidy`, `LICENSE`, `README.md`. (No `.editorconfig`, `.pre-commit-config.yaml`, or `CHANGELOG.md` — not scaffolded.)
2. **CMake policy:** top-level `CMakeLists.txt` (§6) + `cmake/{CPM,CUDAArch,Docs,SteppeOptions,SteppeSanitizers,SteppeWarnings}.cmake`. `SteppeWarnings.cmake` defines the `steppe::warnings` INTERFACE target (linked `PRIVATE` by every first-party target); `SteppeOptions.cmake` holds the option knobs; `CPM.cmake` is the network-fallback dependency-fetch shim (§6). The compiler-launcher (ccache) wiring is inline in `CMakeLists.txt` + `SteppeOptions.cmake`, not a separate module.
3. **Presets:** `CMakePresets.json` — configure presets `base`/`dev`/`dev-asan`/`dev-compute`/`release`/`ci`, matching build presets, and a `host` test preset.
4. **Dependencies (per-target, no central pin file):** there is no `third_party/`; each dependency is declared at the narrowest scope that needs it. `find_package(CUDAToolkit REQUIRED)` (`CMakeLists.txt:60`) supplies the CCCL headers and `CUDA::{cudart,cublas,cusolver,cufft}` (cuFFT is the DATES autocorrelation/LD engine); `Threads` at `CMakeLists.txt:69`. CLI11 is obtained find-package-first, CPM-fallback, confined to `src/app/CMakeLists.txt`; GoogleTest the same way (optional; a self-checking `main()` fallback when absent) in `tests/CMakeLists.txt`; nanobind likewise in `bindings/CMakeLists.txt`. NCCL is intentionally **not** linked (the multi-GPU combine is P2P/host-order, never AllReduce — step 8, §11.4). Eigen, spdlog, and nvbench are not dependencies.
5. **Public API headers (`include/steppe/`):** `config.hpp` (§9), `error.hpp` (`enum class Status`, §10), and the per-command surfaces `fstats.hpp`, `f3.hpp`, `f4.hpp`, `f4ratio.hpp`, `fstat_sweep.hpp`, `dstat.hpp`, `qpadm.hpp`, `qpfstats.hpp`, `qpgraph.hpp`, `qpgraph_search.hpp`, `dates.hpp`, `extract.hpp`. `include/CMakeLists.txt` defines the header-only `steppe_api` INTERFACE (alias `steppe::api`), CUDA-free, `cxx_std_20`. These are ordinary C++ headers (results are structs holding `std::vector`/`std::span`, e.g. `F2BlockTensor` in `fstats.hpp`); the opaque-handle **C ABI** (`steppe_status_t`, `version.hpp`, an umbrella `steppe.hpp`) is **deferred to M(abi-1)** and not present (§16).
6. **DRY internals (`src/core/internal/` + `src/core/domain/`):** `host_device.hpp` (`STEPPE_HD`, `STEPPE_DEBUG_ONLY`, `STEPPE_ASSERT`), `log.hpp` (header-only `STEPPE_LOG_WARN` teardown-warning sink over `std::fprintf`; no `log.cpp`), `nvtx.hpp`, `launch_config.hpp`, `views.hpp` (a minimal `MatView` — the reduced stand-in for the fuller `span_view.hpp` it references), plus `f2_estimator.hpp` (step 8), `decode_af.hpp`, `pchisq.hpp`, `small_linalg.hpp`, `dates_fit.hpp`, `qpfstats_jackknife.hpp`; and `domain/block_partition_rule.{hpp,cpp}` (`block_of` + `assign_blocks` + inverse `block_ranges`). `src/core/CMakeLists.txt` defines the `steppe_core_internal` INTERFACE (alias `steppe::core_internal`). There is no `expected.hpp`/`STEPPE_TRY` and no `type_traits.hpp`/`real_t` — those internal primitives were never built.
7. **Device RAII + checks (`src/device/cuda/`):** `check.cuh` (`STEPPE_CUDA_CHECK`, `STEPPE_CUDA_CHECK_KERNEL`, `CUBLAS_CHECK`, `CUSOLVER_CHECK`), `device_buffer.cuh`, `pinned_buffer.cuh`, `stream.hpp`, `device_guard.cuh`, and `handles.hpp` (full move-assign + debug teardown logging, single-stream cuBLAS with `cublasSetWorkspace`, §7). These plus the inline `cudaMallocAsync` in the `cuda_backend*.cu` TUs are the **allowlisted** allocation sites (§2); there is no separate pool `allocator.hpp`.
8. **Backend seam:** `src/device/backend.hpp` (the `ComputeBackend` class, CUDA-free) + `backend_factory.hpp`. `cpu/cpu_backend.cpp` (scalar reference) and `cuda/cuda_backend.cu` (with the per-domain partials `cuda_backend_{f2_blocks,fstats_assemble,qpadm_fit,qpfstats,qpgraph,dstat,dates,decode}.cu`). Device CMake keeps `CUDA::*` `PRIVATE` so the layering grep holds. Multi-GPU is **not** a single `multi_gpu.*` file: SNP-tile sharding is `src/device/shard_plan.{hpp,cpp}`, per-device partials are `src/device/cuda/device_partial.cu`, the fixed-order combine is `src/device/cuda/p2p_combine.cu` (`cudaMemcpyPeerAsync` in device order `g=0..G-1`, **never** an NCCL AllReduce — §11.4, §12), and the host-side orchestration is `src/core/fstats/f2_blocks_multigpu{,_core}.cpp`.
9. **First real kernel + reference:** the f2 estimator is the shared `__host__ __device__` per-element primitive in `core/internal/f2_estimator.hpp` (`STEPPE_HD`; the single source of the `q(1−q)/max(N−1,1)` bias-corrected formula), consumed by both `cuda/f2_block_kernel.{cu,cuh}` and the matching `CpuBackend` path — the reference-equivalence seam.
10. **IO leaf (`src/io/`, no dependency on `device`):** readers `geno_reader`, `snp_reader` (parses SNP/genetic positions; the block rule itself lives in `core/domain`), `ind_reader`, `plink_reader`, `genotype_source`, `eigenstrat_format`, `ploidy_detect`, plus the tile views `genotype_tile.hpp`/`snp_major_tile.hpp`; `filter/{include_exclude,mind_prepass,snp_filter}.{hpp,cpp}` with `filter_decision.hpp`/`filter_plan.hpp`/`snp_summary_reduce.hpp` (on-the-fly + `--mind` pre-pass, §5). Isolated `src/io/CMakeLists.txt`. The precomputed-f2 on-disk format is not in `io`: it is `src/device/f2_disk_format.hpp` + `src/app/f2_dir_io.cpp`/`f2_dir_writer.cpp`. The `merge/` and `impute/` preprocessing subdirs described in earlier drafts were never built (no merge-plan/allele-harmonize/missing-policy code exists).
11. **Core compute + CLI:** `core/fstats/{f2_from_blocks,f2_combine,f2_blocks_multigpu*}.cpp`; `core/qpadm/{ranktest,nested_models,model_search,qpadm_fit,f3,f4,f4ratio,fstat_sweep,qpgraph_*}.cpp` with the header-only solvers `gls_solve.hpp`, `f4_matrix.hpp`, `jackknife.hpp` (all orchestration-only, GPU reached via `ComputeBackend`). `src/app/main.cpp` + `cli_parse.cpp` (the arg struct is `src/core/config/cli_args.hpp`) drive flat `cmd_*.cpp` handlers (no `commands/` subdir); `cmd_extract_f2.cpp` carries the `--dry-run` budget print (§11.1).
12. **Tests (`tests/`):** `tests/CMakeLists.txt` with `gtest_discover_tests`; host unit tests under `tests/unit/` (`test_f2.cpp`, `test_f2_combine.cpp`, …) and GPU-vs-CPU / AT2-golden parity tests under `tests/reference/` (`test_f2_equivalence.cu`, `test_f2_multigpu_parity.cu`, the domain-outcome `test_qpadm_domain.cu`, plus per-command `test_*_parity.cu` and CLI tests under `tests/cli/`).
13. **Bindings + packaging:** `bindings/{CMakeLists.txt,module.cpp,bind_*.cpp}` (the nanobind `steppe._core` module) + `bindings/steppe/{__init__.py,_rds.py}`, and top-level `pyproject.toml` (scikit-build-core; §15).
14. **Benchmarks:** self-contained `std::chrono` benches under `tests/reference/` — `bench_fstats_1240k.cu`, `bench_rotation_1240k.cu`, `bench_f2_multigpu.cu`, `bench_optimizers.cu`. There is no top-level `benchmarks/` dir and no nvbench dependency.

**Not yet scaffolded:** there is no `.github/` CI (lint/build-test/sanitizer/nightly are not automated gates) and no `docs/adr/` ADR set; this document lives only at `docs/archive/architecture.md` (no `docs/architecture.md` mirror).

**Planned (deferred):** multi-GPU **rotation** is host-bounce-capped on no-P2P consumer cards (the `replicate_f2` ~3.8 s / 8.72 GB host bounce) and is deferred — prefer `G==1` until the fix lands; see the `TODO(multigpu-host-bounce)` at `src/core/qpadm/model_search.cpp:168-266`.

---

## 18. Definition of Done / quality bar

A change is Done when **all** hold:

- **Builds clean** on the target arch with `-Werror`/`--Werror all-warnings` (`cmake/SteppeWarnings.cmake`: `-Wall;-Wextra;-Werror` on host TUs, `--Werror;all-warnings` on nvcc TUs, gated by `STEPPE_WERROR`); no new clang-tidy/IWYU diagnostics. The M0 default compute arch is **sm_120** (consumer Blackwell / RTX 5090 validation box: `CMakePresets.json` `base` preset `CMAKE_CUDA_ARCHITECTURES="120"`, pinned by `cmake/CUDAArch.cmake` `STEPPE_CUDA_ARCH`); the `release`/`ci` presets build the full ship list `75/80/86/89/90/100/120`. An explicit `-DCMAKE_CUDA_ARCHITECTURES` still wins.
- **Layering intact:** no upward dependency; CUDA headers compile only into `steppe_device`; `core` reaches the GPU only via `ComputeBackend`. The device-allocation allowlist (only the three wrapper TUs `src/device/cuda/{device_buffer,pinned_buffer,block_sink}.cuh` may call the raw `cudaMalloc*`/`cudaHostAlloc`/`cudaFree*` family) is enforced **by convention and review**, not by an automated grep/CI gate (there is no `.github/` and no CI in the repo).
- **RAII complete:** no allocation from the `cudaMalloc*`/`cudaHostAlloc`/`cudaFree*` family outside the allowlisted wrappers; all owning types fully move-only (construct **and** assign); destructors log-on-error in debug, never throw; no new global mutable state.
- **Correctness anchored:** every new statistic-bearing kernel has a `CpuBackend` reference path and a reference-equivalence test; affected ADMIXTOOLS 2 goldens pass within the tier tolerance (`est` tight; `se`/`z`/rank loose, with the loose tolerance derived from observed spread, not a magic number).
- **Numerics policy honored:** native FP64 on the small cancellation-sensitive numerator/divide step; **fixed-slice Ozaki (default 40-bit mantissa, `kDefaultMantissaBits`, `include/steppe/config.hpp:44`) on the matmul-heavy path including the S2 f2 GEMMs** (measured 8–17× over native at native-grade accuracy on real AADR; dynamic-mantissa rejected as the parity trap), gated against the native-FP64 oracle with an explicit `cublasSetWorkspace` workspace (`src/device/cuda/handles.hpp` `set_workspace`) under `deterministic`. Run-to-run determinism is achieved by confining the statistic-bearing cuBLAS to a **single stream** with the pinned workspace re-applied on every `cublasSetStream` (`handles.hpp`), plus cuSOLVER deterministic-results mode (`cusolverDnSetDeterministicMode(h_, CUSOLVER_DETERMINISTIC_RESULTS)`, `handles.hpp:358`) — the `run_to_run`-deterministic property `config.hpp:500` phrases the reductions against, not a named CCCL execution-policy API (none is called). Philox seeds for the qpGraph optimizer are threaded and recorded; no TF32/FP16 in any reported value (rejected in `config_builder`).
- **GEMM-reformulation reference-equivalent:** any statistic implemented as dense tensor ops (the S2 3-GEMM f2, §5) is diffed against the slow native-FP64 `CpuBackend` oracle that walks the exact AT2 pairwise-complete path — agreement at the `f2_blocks` seam within the tight tier, including the `Vpair==0 ⇒ 0` branch (`src/core/internal/f2_estimator.hpp:118`), the `q(1−q)/max(N−1,1)` het-correction denominator + haploid-count `N` convention (`het_correction`, `f2_estimator.hpp:71`), and that the per-block `Vpair` carried to the S4 jackknife weight composes to AT2's `f2_blocks` definition (no double-normalization). The shared `__host__ __device__` primitives (`het_correction` / `f2_term` / `assemble_f2_numerator`, `f2_estimator.hpp`) are the single source of the per-element formula. No array framework (no JAX/CuPy) introduced.
- **Multi-GPU parity (precompute):** the single-node multi-GPU (SPMG) f2 precompute is **bit-identical** across the devices in `DeviceConfig::devices` and to the single-GPU reference — the `f2_blocks` combine is a fixed host-order placement of disjoint block-aligned shards (raw D2D/`cudaMemcpyPeer`, `src/device/p2p_combine.hpp`), **never** an NCCL AllReduce. Gated by an exact `memcmp`/`==` parity test, not a tolerance (`tests/reference/test_f2_multigpu_parity.cu`, `test_f2_multigpu_gate.cu`, `tests/unit/test_f2_blocks_multigpu.cpp`).
  - **Planned:** multi-GPU *rotation* for the qpAdm search (`run_qpadm_search`, G≥2) is **DEFERRED** — on no-P2P consumer cards (RTX 5090: `caps.can_access_peer == false`) it is host-bounce-capped (`TODO(multigpu-host-bounce)`, `src/core/qpadm/model_search.cpp:168`, `src/device/device_f2_blocks.hpp:92`, `src/app/cmd_rotate.cpp`). The G≥2 replication path is correct and deterministic (G1==G2 bit-identical) but throughput-suboptimal; the single-GPU fast path is the supported one.
- **Deterministic re-sort:** search results are scattered into pre-sized result slots by `model_index` (each result echoes the caller's stable index; `src/core/qpadm/model_search.cpp:179`), fail-fast on an out-of-range index — the pre-sized write **is** the deterministic re-sort.
- **Preprocessing in scope & layering-legal:** merge/filter/impute live in the `io` leaf as plain-data producers reading only `core::block_partition_rule` downward and **not** depending on `device`/`ComputeBackend` (decode is wired into compute by `app`); merge is an in-memory plan (no on-disk rewrite, no strand inference, no self-computed LD); the pairwise-complete default reproduces AT2 and the SNP→block map is unchanged from the single-dataset path; imputation outputs are tagged and never golden-compared.
- **Memory budgeted:** new resident allocations accounted in the §11.2 budget; `build()` still rejects over-VRAM configs; out-of-core paths keep the genotype matrix tiled (no whole-matrix load).
- **Errors classified:** new failure modes map to a `steppe::Status` category (`enum class Status`, `include/steppe/error.hpp:23`); the three domain outcomes (`RankDeficient`, `NonSpdCovariance`, `ChisqUndefined`) are returned **as values** and covered by `tests/reference/test_qpadm_domain.cu`, not surfaced as faults. (The richer C ABI `steppe_status_t` / `extern "C"` seam is **deferred** to M(abi-1), `error.hpp:7`; `Status` is the as-built public status type.)
- **Sanitizers green:** compute-sanitizer memcheck + racecheck clean on the relevant tests (`--error-exitcode 1`), driven manually via the `dev-compute` preset lane marker (`CMakePresets.json`) wrapping `ctest`; new async-alloc paths covered by `--track-stream-ordered-races all`. Host TUs additionally build under ASan/UBSan (`dev-asan` preset).
- **Observable & documented:** new pipeline phases carry NVTX ranges (`src/core/internal/nvtx.hpp`); public header/behavior changes update the affected headers and their Doxygen comments; the public compute seam stays **CUDA-free** (no CUDA types in `include/steppe/`), which is the enforced boundary as built. (There is no `CHANGELOG.md` and no `docs/adr/` in the repo; do not treat those as gates.)
- **No perf regression** on the affected kernel, checked against the custom scaling-sweep benchmarks under `tests/reference/bench_*.cu` (e.g. `bench_f2_multigpu.cu`, `bench_fstats_1240k.cu`, `bench_rotation_1240k.cu`) — these are hand-rolled sweeps, not an nvbench framework and not an automated 10%-budget CI gate; a speed change carries a manual Nsight Compute roofline/occupancy note (clocks locked, GPU model pinned).
- **Python parity:** `import steppe` and CPU smoke tests pass GPU-free; the GPU wheel imports in a clean venv resolving the CUDA runtime via the declared (unsuffixed CUDA-13) deps, without bundling CUDA libs.

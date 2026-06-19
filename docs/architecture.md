# `steppe` — GPU/CUDA qpAdm: Architecture & Engineering Best Practices

## 0. Purpose & how to use this document

This is the canonical engineering specification for **`steppe`** (working name): a CUDA Toolkit 13+ reimplementation and reimagining of the f-statistic / qpAdm machinery from [ADMIXTOOLS 2](https://uqrmaie1.github.io/admixtools/). It serves two audiences at once:

1. **A spec for the human team** — the architecture, conventions, and quality bar everyone follows.
2. **A scaffolding prompt for an AI coding agent** — concrete enough that an agent handed §17 can lay down a correct repository skeleton on day one. Where a section gives a code snippet, that snippet is meant to be (nearly) pasteable; where it gives a rule, the rule is testable in CI.

The document is **opinionated by design**. Every choice carries a one-line justification. We do not present menus. Where a load-bearing CUDA-13/CCCL fact is genuinely unverifiable from primary sources, it is marked **[UNCERTAIN]** rather than asserted; every other CUDA-13 claim below is traceable to the vendor docs cited in the footer.

The single load-bearing domain insight that shapes everything: ADMIXTOOLS 2 is **precompute-once / fit-many**. One expensive, bandwidth-bound genotype pass produces a cacheable `f2_blocks` tensor (`[n_pop × n_pop × n_block]`); thereafter every model fit is small dense linear algebra over that resident tensor, batched across a block (jackknife) axis and a model axis. The architecture mirrors this split: a **precompute engine** (S0–S2) and a **fit engine** (S3–S8). **`f2_blocks` is scale-dependent, not flatly "tiny": it is `O(P²·n_block)`** — MB-scale at typical AT2 population counts (tens of pops), where it stays GPU-resident across the whole run, but tens of GB at thousands of pops (≈76 GB at `P=2500`, ≈220 GB at the `P=4266` top end of §11.4). The crucial corollaries the rest of the document keeps returning to: (a) the *input* (the genotype matrix) routinely does **not** fit in VRAM, so the precompute engine is an out-of-core streaming computation, not a load-it-all kernel (§11.1, **BUILT** via M5 SNP-tile streaming); and (b) the *output* itself does not fit VRAM past a few thousand pops, so it is **adaptively tiered** (resident → host RAM → disk, §11.1/§11.2, **BUILT** via M5). At AT2-typical scale both the input streams and the output stays resident; the at-scale machinery is exactly what makes thousands of pops tractable (a 75.7 GB result computed on a single 32 GB GPU, §11.1).

### Assumptions & chosen defaults

| Area | Default | One-line rationale |
|---|---|---|
| CUDA Toolkit | **13.1+** (build on latest 13.x; 13.0 is the lowest arch baseline) | Minimum arch is Turing sm_75; full Blackwell support. **13.1+ is required** because the determinism API ships in CCCL 3.1, which CUDA 13.0 does not bundle (§3). |
| C++ standard | **C++20** (CUDA + host) | `std::source_location`, concepts, `std::span` parity; above CCCL's C++17 floor. |
| Build system | **CMake ≥ 3.30 + Ninja** | 3.30 understands Blackwell arch numbers; Ninja gives correct CUDA dep tracking + `compile_commands.json`. |
| Device std lib | **CCCL ≥ 3.1** (Thrust / CUB / libcudacxx), pinned | One source for `cuda::std::span`/`mdspan`, device primitives, and the `cuda::execution` determinism controls. |
| Dense linalg | **cuBLAS + cuSOLVER** (cuSOLVERMp deferred) | Maps directly onto GEMM/SYRK/Cholesky/SVD/batched-LAPACK with *documented, scoped* determinism (§12). |
| RNG | **cuRAND Philox** (counter-based) | Reproducible per-replicate substreams independent of launch geometry. |
| Python bindings | **nanobind + scikit-build-core** | Fast builds, small binaries; CMake-native wheel backend; GPU-optional. |
| Unit/golden tests | **GoogleTest + CTest** (+ pytest for bindings) | Mature CUDA test patterns; `gtest_discover_tests`. |
| Benchmarks | **nvbench** (CUDA) + google-benchmark (host) | Purpose-built CUDA benchmarking with throughput reporting + JSON for regression gates. |
| Lint / format | **clang-format + clang-tidy + IWYU** | Single source of style and static analysis over `compile_commands.json`. |
| Precision | **Emulated-FP64 (Ozaki) default for matmul-heavy work; native FP64 oracle; TF32 opt-in** | GEMM/SYRK covariance & f4 assembly run in cuBLAS fixed-point FP64 emulation (accuracy ≈ native FP64, tensor-core throughput); native FP64 is the validation oracle/fallback and the path for the small cancellation-prone numerator/divide; the **f2 GEMMs themselves use fixed-slice Ozaki (~40-bit mantissa) — measured 8–17× over native FP64 at native-grade accuracy on real AADR** (§5 S2, §12); dynamic-mantissa Ozaki is rejected as the parity trap; TF32 is screening-only (§9, §12). |
| Multi-GPU | **Single-node multi-GPU, day one** (single process, N devices) | The objective is parallelism + speedup **with parity** across the GPUs in one workstation. Single-process-multi-GPU (one host thread per device, per-device streams, NCCL for broadcast only — the parity reduce is host-side fixed-order, §11.4); multi-*node* remains deferred (§9, §11.4, §12, ADR-0009). **Honest role (MEASURED, M4.5/M5):** on the *precompute* multi-GPU is a **modest throughput layer** — the at-scale enabler is M5 device-resident output + streaming, not sharding (§11.1, §11.4). Multi-GPU's decisive payoff is the **fit/rotation phase (S8)** — thousands of independent models, zero inter-GPU traffic (§5, §11.4). |
| Dense LA framework | **Native cuBLAS/cuSOLVER — no array framework (no JAX, no CuPy)** | The hot path is hand-issued GEMM/SYRK/Cholesky/SVD through `ComputeBackend`; the statistics are *reformulated* into dense tensor ops (§2, §5 S2), not transliterated from an array DSL. |
| Determinism | **single-stream statistic path + CCCL `run_to_run` reductions + scoped deterministic cuSOLVER; emulated-FP64 needs an explicit `cublasSetWorkspace` workspace** | Bit-stable on a given GPU under the documented constraints (§12). Native FP64 and TF32 GEMM are run-to-run bit-reproducible single-stream; **fixed-point emulation voids that guarantee unless an adequate workspace is supplied** (§12) → non-flaky regression gates vs ADMIXTOOLS 2 goldens. |

---

## 1. Project vision & non-goals

**Vision.** `steppe` computes f2/f3/f4 statistics and fits qpWave / qpAdm models on the GPU, producing results numerically equivalent (within a stated tolerance) to ADMIXTOOLS 2, at a throughput that makes large model-space searches (rank tests across thousands of left/right configurations, graph-topology screening) practical on a single multi-GPU workstation. It harmonizes/merges and on-the-fly filters its input datasets during the streaming pass (§5), reformulates the f-statistics into dense tensor ops to run at tensor-core throughput where the conditioning allows it (§2, §5 S2, §12), and shards both the precompute and the model-space search across all GPUs in the box **with parity** (§11.4, §12). It exposes a clean C++ library, a CLI, and a Python package; `f2_blocks` is the cacheable, ADMIXTOOLS-compatible interchange artifact. **BUILT (M5):** the precompute returns a **device-resident** `f2_blocks` handle (`DeviceF2Blocks`, kept in VRAM for the downstream fit), with host materialization as an **opt-in `.to_host()`** (the ADMIXTOOLS-compatible host `F2BlockTensor`), and adaptively tiers the result to host RAM or disk when it does not fit VRAM (§11.1, §11.2).

**It IS:**
- A faithful, validated reimplementation of the f2 → f3/f4 → block-jackknife → qpWave/qpAdm GLS+SVD pipeline. The matmul-heavy covariance/f4 assembly defaults to **Ozaki-scheme emulated FP64** (accuracy ≈ native FP64) while the cancellation-sensitive elementwise math stays in **native FP64**; native FP64 remains the gold reference every other mode is validated against (§9, §12).
- A *precompute-once / fit-many* engine: one **out-of-core** streaming genotype pass (BUILT — M5 SNP-tile input streaming, GPU footprint `O(P·tile + P²·n_block)`, independent of the SNP count `M`), then batched small dense linear algebra over a resident tensor. When the `f2_blocks` *output* is itself larger than VRAM it streams to the fastest tier it fits (resident → host RAM → disk), so the engine is bounded by neither the input nor the output size (§11.1).
- GPU-optional at the API level (a CPU reference backend always exists and always imports).

**It is NOT:**
- A general-purpose popgen suite (no PCA, ADMIXTURE, IBD, phasing, association testing).
- **In scope: genotype QC / data-munging as a streaming front-end.** We read PLINK/EIGENSTRAT/PACKEDANCESTRYMAP and additionally **merge/harmonize multiple datasets, filter on the fly, and handle missing data** (§5, ADR-0011). The default missing-data policy is **pairwise-complete** (validity-mask + per-SNP/per-pop sample size `N`) for AT2 parity; **imputation is optional and off by default** (screening/PCA only, never an AT2-golden-compared statistic). The cheap filters are MAF/geno/include-exclude and `--mind`; **monomorphic-drop, autosome-only, and ts/tv are additions beyond bare "filter," gated behind explicit flags and tagged.** What we still do *not* do: **infer strand or resolve A/T·C/G ambiguity heuristically** (harmonization uses only user-supplied/declared allele polarity; ambiguous or multiallelic sites are dropped, never flipped by frequency guesswork), **compute LD/pruning ourselves** (we accept only an externally supplied `prune.in`), or **rewrite datasets to disk** — merge is an in-memory *plan* over the sources, harmonization and cheap filters are applied per tile during the stream, and only genuinely-aggregate filters (`--mind`, an external `prune.in`) use a light pre-pass producing small include/exclude sets.
- A multi-*node* HPC framework. **Single-node multi-GPU is in scope and central, day one** — parallelism + speedup across the N devices of one workstation, *with* numerical parity, is a primary objective (§9, §11.4, §12). It is single-*process*-multi-GPU (one host thread per device, per-device streams, NCCL for the broadcast). Out-of-core genotype streaming remains a v1 requirement (§11.1) and now shards SNP tiles across the GPUs. **Honest role (MEASURED, M4.5/M5):** on the precompute the multi-GPU sharding is a *modest* throughput layer — it was in fact measured slower than single-GPU until the host data-bounce was removed, and even then shows only a partial overlap with a serial host tail; the at-scale precompute win came instead from getting the result off the CPU (device-resident output, M5) and streaming, not from sharding (§11.1, §11.4). Multi-GPU's *decisive* payoff is the model-space search (S8) — thousands of independent fits, **zero** inter-GPU traffic — which is its proper home (§5, §11.4). The deferred boundary is **multi-node** (multiple processes / MPI / a launcher). **cuSOLVERMp stays deferred** — not for lack of multi-GPU ambition, but because qpAdm's `Q`/`X` are tiny: the right parallelism is *across many independent small fits* (S8), not *within one distributed factorization*; a distributed solve of a low-double-digit matrix is pure overhead and worse for determinism (§11.4).
- A drop-in R package replacement. We target *numerical* and *file-format* compatibility, not R-API compatibility.
- A research sandbox for low-precision tricks. TF32/FP16 never produce a reported statistic without re-validation; TF32 is a user-selectable mode for *model-space screening only*, and its results carry a precision tag and a looser tolerance tier — never bit-compared to AT2 goldens (§9, §12). Emulated FP64 is *not* a low-precision trick: it targets native-FP64 accuracy and is gated against the native-FP64 oracle before it ships.

---

## 2. Engineering principles

**DRY / single source of truth.** Every cross-cutting concern — CUDA error checking, error propagation, logging, NVTX, launch-config math, precision/type traits, span/mdspan views, **and every domain rule shared between layers** — is implemented exactly once and consumed through one `INTERFACE` target. Concretely: one `STEPPE_CUDA_CHECK`, one `cdiv()`, one `real_t<P>` precision switch, one `block_assignment()` rule (§5, §11.1). The single most important DRY consequence in this codebase: the **SNP→block assignment rule** is a domain decision that `io`, the device kernels, and the jackknife all depend on bit-for-bit, so it lives in `core` (host-pure) and is consumed everywhere — never re-derived in `io` (see §5, fix to the old "block rule lives in io" smell).

The DRY grep gate (enforced in CI) is precise, not aspirational. It flags any call to the **whole allocation family** — `cudaMalloc`, `cudaMallocAsync`, `cudaMallocManaged`, `cudaMallocHost`, `cudaHostAlloc`, and the matching `cudaFree`/`cudaFreeAsync`/`cudaFreeHost` — that appears **outside the explicit wrapper allowlist** (`device_buffer.cuh`, `allocator.cu`, `pinned_buffer.cuh`). The allowlist is a checked-in list of translation units, not a pattern; adding a TU to it requires review.

**Strict separation of concerns.** The codebase is layered `app/bindings → api → core → device`, with `io` an isolated leaf, and the direction is enforced by the *compiler* via CMake link visibility (§4), not by convention. CUDA headers are `PRIVATE` to `steppe_device` and never compile into `core` or the CLI. **`core` is pure host C++20 and issues every device operation through the `ComputeBackend` interface — it never includes a CUDA header and never calls cuBLAS/cuSOLVER directly** (this is what makes the f4/GLS/SVD stages in §5 layering-legal). The `f2_blocks` tensor is the seam between the GPU hot loop and the small-LA derivations.

**RAII everywhere.** No raw allocation, `cudaStreamCreate`, `cublasCreate`, `cusolverDnCreate`, or library workspace lives outside an owning wrapper (`DeviceBuffer<T>`, `Stream`, `Event`, `CublasHandle`, `CusolverDnHandle`, `PinnedBuffer<T>`). Move-only owning types with full move-construct **and** move-assign (§7); destructors never throw — but they are not silent: in debug builds a nonzero destroy status is routed to `STEPPE_LOG_WARN` so "fail-fast" does not become "fail-silent at teardown" (§7, §10).

**No global mutable state.** Configuration is an immutable `RunConfig` value object; device resources (`backend`, `streams`, `allocator`) are bundled in a `Resources` struct and injected into every compute entry point. No singletons, no hidden statics — the precondition for both thread-safety and unit-testability.

**Reformulate statistics into dense tensor ops; fuse the elementwise feeders.** A statistic is implemented by *re-deriving it as matrix algebra* — outer products, masked reductions, and contractions expressed as cuBLAS GEMM/SYRK batched over the block and model axes (native cuBLAS/cuSOLVER, **no array framework — no JAX, no CuPy**) — not by transliterating the scalar CPU loop onto the GPU. Per-SNP missingness becomes masking-by-multiply (`@ Vᵀ`); per-pair sums become matmuls (§5 S2). The elementwise *feeders* of those matmuls (zero-filled `Q`, validity mask `V`, `Q²`, het correction) are produced in a **single fused sweep** over the decoded tile, never materializing the `[SNP × pop × pop]` intermediate (§11.1); the elementwise *consumers* (the catastrophic-cancellation numerator and the masked divide) are likewise fused, on the small reduced matrices, in native FP64. Precision follows the **conditioning of the operation, not its shape** (§12): the reformulation does not *abolish* the f2 cancellation, it *localizes* it — the `Σp²−2Σpq+Σq²` difference still lands on the small `O(n_pop²)` reduced matrix, where keeping it (and the GEMMs that feed it) in native FP64 is essentially free, so emulation is reserved for the genuinely well-conditioned matmul-heavy stages (covariance SYRK / f4 assembly, §5 S4/S3), not the f2 reduction (§12).

**Fail-fast.** Every CUDA API return is checked at the call site; every kernel launch is followed by `cudaGetLastError()` (and a forced sync in debug). Config is validated once at `ConfigBuilder::build()` and frozen — including a **device-memory budget check** (§9, §11.2): invalid arch lists, missing devices, non-SPD covariance, *and configs that exceed VRAM* surface immediately with file/line context, not as silent corruption or an OOM three stages later.

**Testability.** Per-element numerics and the reference backend live in `__host__ __device__` pure functions that compile and unit-test on the CPU with no GPU. `ComputeBackend` has two implementations (CUDA, CPU reference) so the entire pipeline is exercisable GPU-free and the GPU is continuously diffed against an obviously-correct scalar reference. **The CPU reference is an *oracle that validates results, not a structural template the GPU mimics.*** The GPU production path may have an entirely different control structure (three batched GEMMs + two fused elementwise kernels) from the reference's scalar triple loop, as long as the two agree at the `f2_blocks`/`Q`/`X` seams within the tolerance tiers (§12, §13). Thin `__host__ __device__` scalar functions survive only as (a) the reference implementation and (b) per-element primitives invoked *inside* the fused kernels — **never as the structure of the production hot path** (§5 S2, §7).

**Numerical reproducibility / determinism.** Native FP64 for the cancellation-sensitive elementwise math, Ozaki-emulated FP64 for the matmul-heavy covariance/f4 assembly (validated against the native-FP64 oracle), TF32 opt-in for screening only (§9, §12); CCCL `run_to_run` deterministic reductions on the **single statistic stream**; scoped deterministic cuSOLVER; counter-based (Philox) RNG with seeds threaded through the API and recorded in golden metadata. The determinism guarantee is **specific and constrained** (single stream, given GPU/arch, enumerated routines, and an explicit `cublasSetWorkspace` workspace for the emulated-FP64 path) — see §12, which states exactly where bit-stability holds and where it does not, rather than claiming it globally.

**Correctness before speed.** The CPU reference and ADMIXTOOLS 2 goldens gate every change — they validate the *result* the reformulated GEMM path produces, not its internal structure (the reference walks the exact AT2 pairwise-complete path; the GPU walks three GEMMs + two fused kernels; they must agree at the seams, §12, §13). TF32 fast paths are opt-in screening tools that never produce a reported `est`/`se`/`z`/`p` without re-validation, while emulated FP64 is the default for the matmul-heavy path and *does* produce reported numbers — but only after passing the native-FP64 oracle gate (§12). We optimize only the kernel the profiler (Nsight Systems → Compute) proves dominant, never on speculation.

---

## 3. Tech stack & pinned versions

Pin everything; `third_party/CMakeLists.txt` is the single place `FetchContent_Declare`/CPM pins live, fetched by tag or hash.

| Component | Pin | One-line rationale |
|---|---|---|
| CUDA Toolkit | **13.1+** (13.0 = lowest arch baseline) | Turing→Blackwell; **13.1 bundles CCCL 3.1**, which carries the determinism API. CUDA 13.0 bundles CCCL 3.0.x and lacks it. |
| CMake | ≥ 3.30 | Understands Blackwell arch numbers; CUDA-13 nvcc-default handling. |
| Ninja | ≥ 1.11 | Recommended CUDA generator; emits `compile_commands.json`. |
| C++ standard | C++20 | `source_location`, concepts; above CCCL's C++17 floor. |
| **CCCL** | **≥ 3.1.0** | Thrust/CUB/libcudacxx; `cuda::std::span`/`mdspan`; **`cuda::execution::require(determinism::…)`**. **Pin explicitly:** `find_package(CCCL)` against a 13.1+ toolkit yields 3.1; against a 13.0 toolkit it silently yields 3.0 and the determinism API vanishes — so CI asserts `CCCL_VERSION >= 3.1` and CPM-fetches 3.1 if the toolkit copy is older (§6). |
| cuBLAS / cuSOLVER | from toolkit | GEMM/SYRK + Cholesky/SVD/batched-LAPACK with the *scoped* reproducibility controls in §12. Supplies the three precision modes: native FP64, **fixed-point FP64 emulation (Ozaki scheme)** for the matmul-heavy path, and TF32 for screening (§9, §12). |
| FP64 emulation (Ozaki) | cuBLAS/cuSOLVER 13.x | Handle-level fixed-point emulation: cuBLAS via `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` (compute type) or `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` (`cublasSetMathMode`); cuSOLVER via `cusolverDnSetMathMode(…, CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH)`. Introduced in CUDA 13.0 Update 2; supported on CC 8.x/9.0/10.0/11.0/12.x, **demonstrated speedups are Blackwell-only** [UNCERTAIN: pre-Blackwell is *supported* but not shown faster than native]. No per-matmul descriptor attribute is confirmed — configuration is **handle-level** (math mode / strategy / mantissa-control) [UNCERTAIN]. |
| cuRAND | from toolkit | Philox counter-based RNG for reproducible resampling. |
| NCCL | from toolkit | Single-node multi-GPU **broadcast** of the small `f2_blocks` after the host-side fixed-order combine (§11.4) — and the opt-in `determinism="fast"` reduce only. **[UNCERTAIN]** default NCCL AllReduce is reproducible run-to-run at a *fixed* configuration but **not** bit-identical to a single-GPU sum and **not** invariant to GPU count / buffer size, so the parity path does **not** put the f2 reduction on NCCL — it sums the per-device partials host-side in fixed device order and uses NCCL/`cudaMemcpy` only for the (order-independent) broadcast (§12). |
| Eigen | 3.4.x | Host-only small dense LA in the CPU reference backend. |
| spdlog | 1.14.x | Logging backend behind our facade. |
| CLI11 | 2.4.x | CLI parsing → `ConfigBuilder`. |
| nanobind | ≥ 2.x | Fast-building, small Python bindings. |
| scikit-build-core | ≥ 0.10 | CMake-driven PEP 517 wheel backend with CUDA support. |
| GoogleTest | 1.15.x | Unit/golden/regression harness. |
| nvbench | pinned commit | CUDA microbenchmark + regression JSON. |
| sccache | ≥ 0.8 | Caches host compilation across CI; **best-effort for nvcc** — caching of separable-compilation / `-dlto` device objects is partial and not guaranteed (§6, §14). |
| clang-format / clang-tidy / IWYU | clang 18+ | Format + static analysis + include hygiene. |

---

## 4. Repository layout

```text
# Legend:  [BUILT] = exists as of M0 (the f2 vertical slice);  (planned, Mn/Pn) = future milestone/phase.
steppe/                                  # repo root.  git: main = docs; branch m0-f2-scaffold = this slice.
├── CMakeLists.txt                       # [BUILT] top-level: options + add_subdirectory(include, src, tests)
├── CMakePresets.json                    # [BUILT] named configs (dev/release/ci); CMake >= 3.28 + Ninja
├── build_m0.sh                          # [BUILT] fallback direct-nvcc build of the equivalence test (sm_120)
├── .clang-format / .clang-tidy          # [BUILT] single source of style + static analysis
├── .gitignore                           # [BUILT] excludes build trees, binaries, AND the real genotype DATA
│
├── cmake/                               # [BUILT] reusable build logic (defines NO targets)
│   ├── CUDAArch.cmake                   #   forces CMAKE_CUDA_ARCHITECTURES=120 (Blackwell) under CUDA 13
│   ├── SteppeOptions.cmake              #   option() vars incl. STEPPE_HAVE_EMU_TUNING (default ON)
│   └── SteppeWarnings.cmake             #   warnings-as-errors INTERFACE target (host + nvcc)
│
├── include/                             # ----- PUBLIC API (the installed surface) -----
│   ├── CMakeLists.txt                   # [BUILT] defines the steppe_api INTERFACE target
│   └── steppe/
│       ├── config.hpp                   # [BUILT] Precision{kind, mantissa_bits=40}, DeviceConfig, FilterConfig +
│       │                                #   named constants (kCdivBlock, kRelFloor, kDefaultBlockSizeCm...) -- no magic numbers
│       ├── error.hpp                    # [BUILT] Status enum (Ok/DeviceOom/RankDeficient/NonSpdCovariance/InvalidConfig)
│       ├── fstats.hpp                   # [BUILT] public host F2BlockTensor (the opt-in .to_host() materialization);
│       │                                #   f3/f4 entry points still planned (P2). Device-resident handle + tiering BUILT in src/device/.
│       └── qpadm.hpp                    # (planned, P2) qpWave/qpAdm public API
│
├── src/
│   ├── core/                            # steppe_core -- pure host C++ (NO CUDA, NO I/O)
│   │   ├── CMakeLists.txt               # [BUILT] steppe_core target
│   │   ├── internal/                    #   DRY shared helpers (the "kernel" of the codebase)
│   │   │   ├── views.hpp                # [BUILT] MatView -- the Q/V/N column-major [P x M] contract (element i + P*s)
│   │   │   ├── f2_estimator.hpp         # [BUILT] shared __host__ __device__ f2 primitive (bias-corrected estimator,
│   │   │   │                            #   het-correction, numerator/divide, cdiv/grid_for) -- CPU ref & GPU cannot diverge
│   │   │   └── decode_af.hpp            # [BUILT, M1] shared __host__ __device__ decode primitive (2-bit unpack, raw-value
│   │   │                                #   0/1/2=copies, 3=missing; AC/AN -> Q/V/N; ploidy param) -- one source, no divergence
│   │   ├── domain/
│   │   │   └── block_partition_rule.hpp # [BUILT, M3/M4] host-pure SNP->block rule (+ .cpp): block_of + assign_blocks (per-chrom
│   │   │                                #   reset, dense-renumber occupied bins, cM<->Morgan) + the inverse block_ranges
│   │   │                                #   (per-block [begin,end), validated once; both backends call it); single source
│   │   └── fstats/
│   │       ├── f2_from_blocks.hpp/.cpp  # [BUILT] host orchestration: drives the f2 compute via the ComputeBackend seam
│   │       ├── f4_matrix.cpp            # (planned, P2) f3/f4 contraction
│   │       └── jackknife.cpp            # (planned, P2) block jackknife -> covariance/SEs
│   │
│   ├── device/                          # steppe_device -- the backend layer (CUDA isolated here)
│   │   ├── CMakeLists.txt               # [BUILT] steppe_device; CUDA PRIVATE; links CUDA::cublas
│   │   ├── backend.hpp                  # [BUILT] ComputeBackend interface (CUDA-FREE) -- the DI seam; compute_f2() + decode_af() [M1]
│   │   ├── cpu/
│   │   │   └── cpu_backend.cpp          # [BUILT] REFERENCE backend: long-double cancellation-free f2 (the correctness oracle)
│   │   └── cuda/
│   │       ├── check.cuh                # [BUILT] STEPPE_CUDA_CHECK / CUBLAS_CHECK / post-launch kernel check
│   │       ├── device_buffer.cuh        # [BUILT] DeviceBuffer<T> move-only RAII -- the ONLY place cudaMalloc/cudaFree live
│   │       ├── stream.hpp               # [BUILT] Stream / Event RAII
│   │       ├── handles.hpp              # [BUILT] CublasHandle RAII (created once)
│   │       ├── f2_block_kernel.cuh/.cu  # [BUILT] fused pre-pass (Q,V,Qsq,Hc) + 3-GEMM (fixed-slice Ozaki / native FP64)
│   │       │                            #   + assemble_f2 kernel; all constants from config (no magic numbers)
│   │       ├── decode_af_kernel.cuh/.cu # [BUILT, M1] 2-bit unpack + segmented reduction over individuals -> Q/V/N
│   │       ├── cuda_backend.cu          # [BUILT] implements ComputeBackend on the GPU (decode_af + f2; DeviceBuffer + handle)
│   │       ├── device_f2_blocks.{cu,cuh/_impl} # [BUILT, M5] DeviceF2Blocks impl: resident f2/Vpair compute, opt-in to_host D2H,
│   │       │                            #   no-peer upload_f2_blocks_to_device assembly transport (1f80c0c)
│   │       └── block_sink.{cu,cuh}      # [BUILT, M5] streamed-tier sink: persistent pinned ring + background writer (176a07d)
│   │   ├── device_f2_blocks.hpp         # [BUILT, M5] DeviceF2Blocks VRAM handle (move-only; the device-resident precompute return)
│   │   ├── tier_select.hpp              # [BUILT, M5] CUDA-FREE OutputTier{Resident,HostRam,Disk} + select_output_tier()
│   │   │                                #   (host-pure; cudaMemGetInfo + sysinfo probes; STEPPE_FORCE_TIER pin; unit-testable, 176a07d)
│   │   ├── stream_f2_blocks.hpp         # [BUILT, M5] streamed-loop entry (block-axis output + SNP-tile input streaming, c65179f)
│   │   ├── f2_disk_format.hpp           # [BUILT, M5] STPF2BK1 on-disk f2_blocks cache layout (Disk tier; per-block byte-exact accessor)
│   │   └── host_ram.cpp                 # [BUILT, M5] HostRam-tier block-by-block host spill
│   │
│   ├── io/                              # [BUILT, M1/M2] steppe_io -- genotype decode + QC front-end (ISOLATED leaf, host-pure)
│   │   ├── eigenstrat_format.{hpp,cpp}  # [BUILT, M1] TGENO/GENO header parse + format constants (no magic numbers)
│   │   ├── {geno,snp,ind}_reader.{hpp,cpp} # [BUILT, M1] .geno (tiled raw bytes) / .snp (chrom,genpos,alleles) / .ind (pops)
│   │   ├── genotype_tile.hpp            # [BUILT, M1] plain decoded-tile struct (the leaf's output to app)
│   │   ├── filter/                      # [BUILT, M2] filter_decision (shared predicates) + snp_filter + mind_prepass +
│   │   │                                #   include_exclude + filter_plan (MAF/geno/mind/autosomes/ts-tv; drop-not-flip)
│   │   ├── merge/ impute/               # (planned, M6) multi-dataset merge-plan + optional imputation
│   │   └── precomputed_f2.cpp           # (planned, M7) on-disk f2_blocks cache (ADMIXTOOLS-compatible)
│   └── app/                             # (planned, P3) CLI (extract-f2, qpadm, qpdstat)
│
├── tests/
│   ├── CMakeLists.txt                   # [BUILT] CTest wiring (gtest if present, else self-checking harness)
│   ├── reference/test_f2_equivalence.cu # [BUILT] GPU (EmuFp64{40} & Fp64) vs long-double CPU ref on real AADR -- the trust seam
│   ├── reference/test_decode_equivalence.cu # [BUILT, M1] GPU/CPU decode vs numpy Q/V/N oracle (bit-for-bit) on real AADR
│   ├── reference/test_filter_oracle.cu # [BUILT, M2] no-op-when-default + drop-equals-mask + exact-mask vs scalar oracle
│   ├── unit/test_f2.cpp                 # [BUILT] host unit test of the shared f2_estimator primitive
│   ├── unit/test_block_partition.cpp   # [BUILT, M3] assign_blocks logic (synthetic layouts) + real-AADR consistency
│   ├── unit/test_block_ranges.cpp      # [BUILT, M4] block_ranges inverse: valid layouts + fail-fast on malformed (X-3/B3)
│   └── unit/test_filters.cpp           # [BUILT, M2] host unit test of the filter_decision predicates
│
├── docs/
│   ├── architecture.md                  # [BUILT] this document
│   └── ROADMAP.md                       # [BUILT] milestones, magic-number->config inventory, commit discipline
│
├── experiments/                         # THROWAWAY spike -- validated the kernel + precision; NOT production, NOT layered
│   ├── f2_emu_spike/{f2_emu_spike,f2_timing,f2_prec_acc}.cu  # the precision/throughput experiments
│   └── aadr/{00_setup,01_download,02_build_matrix,03_run,...} # AADR fetch + per-pop Q/V/N matrix builder
│
├── aadr/                                # real AADR genotype DATA -- GITIGNORED, never committed (~4 GB)
└── (root strays: build_run.sh, f2_emu_spike.cu -- leftover spike dupes, slated for removal)
```

**Dependency-direction rule.** Allowed edges only: `app/bindings → api → core → device`; `io` is a sibling leaf that produces plain data structs (genotype tiles + per-SNP genetic positions) and depends on nothing in `core`/`device`. The *app* layer is the only place that wires `io` output into compute. **Nothing depends upward; no cycles.** The one shared domain rule (`block_partition_rule.hpp`) lives in `core` and is the single exception that both `io` consumers and device kernels read — it is host-pure and CUDA-free, so it does not break the layering.

**How CMake enforces it.** Link visibility is the mechanism. A `PUBLIC`/`INTERFACE` dependency propagates include dirs to consumers; a `PRIVATE` one does not. We make CUDA `PRIVATE` to `steppe_device`, so `core`/`cli` *physically cannot* `#include` a CUDA header — it won't compile. `core` reaches the GPU only through `ComputeBackend` (a CUDA-free header). See §7 for the target wiring. An additional CI "architecture test" greps for cross-layer includes and asserts the link graph; IWYU prevents transitive leakage — compile-time guarantee plus verification.

---

## 5. Layered architecture mapped onto the qpAdm pipeline

The numerics define the layers. Each stage names its **owner** (where the orchestration code lives) and, where device work is involved, makes explicit that the device call is **dispatched through `ComputeBackend`** — `core` never issues a GEMM/SVD/Cholesky itself. **S0–S2 = precompute engine** (out-of-core streaming, bandwidth-bound, run once); **S3–S8 = fit engine** (batched small dense LA over the resident `f2_blocks`, with `n_block` and model index as two batch axes).

| Stage | Owner (orchestration) | Device call via | Input → output | Parallelism | Bound by |
|---|---|---|---|---|---|
| **S0 Format decode** | `io` (read, tiled) → `device` kernel | `ComputeBackend::decode` | packed `.bed` tile → dosage `[SNP_tile × sample]` | data-parallel over SNP rows; 2-bit unpack per warp (00→0,10→1,11→2,01→NA) | disk I/O / mem bandwidth |
| **S1 Allele-freq reduction** | `device` (`decode_af_kernel.cu`) | in-kernel | dosages → `afmat`,`countmat` `[SNP_tile × pop]` | segmented reduction over samples within pop partition | mem bandwidth |
| **S2 f2 + block accumulate** | `device/cuda/f2_block_kernel.cu`; assembled by `core/fstats/f2_from_blocks.cpp` | `ComputeBackend::accumulate_f2` | afs → device-resident `f2_blocks [n_pop × n_pop × n_block]` + per-block pairwise valid-SNP count `Vpair`, bias-corrected, via the **3-GEMM reformulation** below, batched over blocks | three batched **native-FP64** GEMMs + a fused elementwise pre-pass and a tiny native-FP64 cancellation step, reduction into block bins via the shared `block_partition_rule` | compute + bandwidth, **streamed, runs once**; returns a device-resident `DeviceF2Blocks` (output adaptively tiered when it exceeds VRAM, §11.1/§11.2) |
| **S3 f3/f4 contraction** | `core/fstats/f4_matrix.cpp` (**orchestrates**; GEMM dispatched via `ComputeBackend::gemm`) | `ComputeBackend::gemm` | `f2_blocks` → f4 matrix `X [(n_L−1)·(n_R−1) × n_block]` | element-wise linear combos, batched over block axis | trivial / compute |
| **S4 Block jackknife → Q** | `core/fstats/jackknife.cpp` (**orchestrates**) + `device/cuda/jackknife_kernel.cu` | `ComputeBackend::jackknife_cov` | per-block f4 → covariance `Q [m×m]`, SEs | fused kernel: total−block_b, center, then `Dsyrk` over centered replicates | compute |
| **S5 Rank test (SVD)** | `core/qpadm/ranktest.cpp` (**orchestrates**) | `ComputeBackend::svd` | `X` → `U,S,V`, rank | batched small SVD when dims permit; per-model fallback otherwise (see note) | batched/per-model linalg |
| **S6 qpAdm GLS fit** | `core/qpadm/gls_solve.cpp` (**orchestrates**) | `ComputeBackend::{potrf,trsm,gemm}` | `X,Q,A,B` → weights `w`, χ² | one weighted GLS solve per model (see note) | batched dense LA |
| **S7 p-value / nested test** | `core/qpadm/nested_models.cpp` | host-only | χ², dof → p | embarrassingly parallel over models | trivial |
| **S8 Model-space search** | `core/qpadm/` orchestration + `app` | reuses backend | resident `f2_blocks` across many models | massive task-parallel; feeds the fit engine | throughput / scheduling |

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

**Multi-GPU parallelism for S0–S2 and S8.** In single-node multi-GPU runs the `io` streamer shards SNP tiles across the G devices (round-robin / static range / host queue, §9, §11.4); each device accumulates its own partial `f2_blocks` + `Vpair` from the tiles it owns, then the G partials are combined once — the parity path sums them **host-side in fixed device order** (not via NCCL AllReduce — §11.4, §12) and broadcasts the result. After the (broadcast) `f2_blocks` is replicated on every GPU, **S8's model-space search shards across the GPUs with zero inter-GPU communication** — each fit is wholly on one device (its internal reductions stay single-GPU and inherit single-GPU parity), and results are re-sorted by model index on the host so ordering is deterministic regardless of which GPU produced which model.

**Note on S5 batched-SVD limits.** `gesvdjBatched` is practical only for small matrices (per NVIDIA's batched-Jacobi guidance, roughly `m,n ≤ 32`). qpAdm's rank-test matrix is `(n_L−1)×(n_R−1)`; with the "thousands of right configurations" this tool targets, **`n_R−1 > 32` is common**, which drops the model off the batched path. The contract is explicit: when either dimension exceeds the batched limit, S5 falls back to a **per-model `gesvd`** (or a QR-based rank estimate) issued in a loop over a small number of streams. This fallback is launch-bound (§11) — exactly the regime to watch on Nsight Systems — and the model-space scheduler (S8) must keep enough independent solves in flight to hide the per-launch latency. Tooling reports which path each run took.

**Note on S6 GLS.** qpAdm's fit is a **single generalized-least-squares solve** given `Q`: Cholesky-factor `Q` (`potrf`), then solve the weighted normal equations for the admixture weights `w` and the residual χ². It is **not** an iterative sweep; any earlier "~20 sweeps" language was wrong and is removed. The only iteration in the vicinity is the *outer model-space search* (S8), which re-runs this one-shot solve per candidate model.

The decisive consequences: keep `f2_blocks` and `Q` GPU-resident; stream and fuse S0–S2 so the full `[SNP × pop × pop]` array is never materialized; exploit `n_block` + model index as batch dimensions; and route every device primitive through `ComputeBackend` so the same orchestration code runs against `CpuBackend` for the reference seam. **BUILT (M5):** the S2 precompute now satisfies the "keep `f2_blocks` GPU-resident" corollary directly — it returns a device-resident `DeviceF2Blocks` handle (the result stays in VRAM for the downstream fit; host materialization is opt-in `.to_host()`), rather than the old forced host round-trip (~4.3× at `P=512`). When the output is itself larger than VRAM (thousands of pops) it falls back to the fastest tier it fits — host RAM, then disk — selected at runtime (§11.1, §11.2); at AT2-typical scale it stays resident.

**Preprocessing stages (S−2, S−1, S0′) before S0–S2.** Genotype QC / data-munging (§1) slots in *upstream of the kernel boundary* as `io`-owned, host-orchestrated, out-of-core stages emitting only plain data — exactly the property that keeps `io` a leaf (§4). They are additive: they produce the same harmonized tile the existing S0 already needed (dosage + validity mask `V` + per-SNP/per-pop sample sizes `N` + block ids from the shared `block_partition_rule`), so S1/S2 are unchanged in shape — S2's masked-GEMM reformulation already consumes `V`/`N`.

| Stage | Owner (orchestration) | Pass type | Input → output | Layering note |
|---|---|---|---|---|
| **S−2 Source schema + merge plan** | `io/merge` (host) | metadata-only (no genotype read) | per-source `{.bim/.fam}`/`{.snp/.ind}` → harmonized SNP set (intersection default / union optional), per-source allele-polarity map from *declared* alleles only (ref/alt swap → `2−dosage`; A/T·C/G ambiguous and multiallelic sites **dropped, never strand-flipped by frequency guesswork** — §1), sample/pop column maps, a `.missnp`-equivalent dropped-SNP list | host-pure leaf; reads `core::block_partition_rule` only as a consumer (§8) so the SNP→block map is bit-identical to the single-dataset path. No CUDA, no upward dep. Merge is a *plan*, not an on-disk rewrite (§1). |
| **S−1 QC pre-pass (conditional)** | `io/filter/prepass` (host, streams tiles) | one light streaming pass, **only if** `--mind` requested (or an external `prune.in` supplied) | dosages → per-sample non-missing counts (`--mind`); an externally-supplied `prune.in` is read, not computed → resolved include/exclude sets folded back into the plan | streams via the same tiler; emits plain sets. We do **not** compute LD ourselves (§1). Skipped entirely when no aggregate filter is requested. |
| **S0′ Harmonized+filtered tile produce** | `io` decode (host) → `app` wires the decoded tile into `ComputeBackend::decode` | the existing out-of-core tile loop (§11.1) | packed tile → harmonized tile in reference polarity + `V` + `N` + block ids, applying **cheap in-tile filters** (MAF/geno/include-exclude, plus flag-gated monomorphic/autosome-only/ts-tv) and the missing-data policy | this is S0 with harmonization + cheap-filter folded into the `io`-side decode; **`io` does not depend on `device`/`ComputeBackend`** — `app` is the only layer that wires `io` output into compute (§4). Same `block_partition_rule` → S2 block bins unchanged. |

Cheap filters decidable from one tile (or from `.bim/.fam` metadata) are applied *in-tile* before S2 accumulation — a dropped SNP simply contributes nothing to its block, so jackknife block identity is unchanged (the §8 DRY invariant holds). The **default missing-data handling is pairwise-complete** (emit `V`/`N`; this *is* the parity path, no new math); **imputation (mean-fill `2p`) is optional** and tagged so its outputs are never bit-compared to AT2 goldens (§12). Heavy-missingness smoothing (`qpfstats`-style) is a fit-engine mode at S3+, not a preprocessing stage.

---

## 6. Build system

Target-based, modern CMake. The top-level `CMakeLists.txt` only sets project policy and calls `add_subdirectory`; each `src/` subdirectory owns exactly one target.

```cmake
# CMakeLists.txt (top level)
cmake_minimum_required(VERSION 3.30)          # 3.30 knows Blackwell arch numbers
project(steppe VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)                   # do NOT trust nvcc's host-derived default
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)         # clang-tidy/IWYU/clangd (Ninja/Make only)

include(cmake/SteppeOptions.cmake)
include(cmake/CUDAArch.cmake)
include(cmake/CompilerLauncher.cmake)
include(cmake/SteppeWarnings.cmake)           # defines INTERFACE target steppe::warnings
add_subdirectory(third_party)
add_subdirectory(include)                     # steppe_api (INTERFACE)
add_subdirectory(src/io)
add_subdirectory(src/device)
add_subdirectory(src/core)
add_subdirectory(src/app)
if(STEPPE_BUILD_PYTHON)  add_subdirectory(bindings) endif()
if(STEPPE_BUILD_TESTS)   enable_testing(); add_subdirectory(tests) endif()
```

**`CMAKE_CUDA_ARCHITECTURES` policy** (`cmake/CUDAArch.cmake`). `native` for dev (clamps to toolkit max); an explicit list ending in one `-virtual` PTX entry for shippable artifacts. The list below targets **base** architectures only — it deliberately ships no `103`/`110`/`121` and no `a`/`f` accelerated variants by default, because those are narrow datacenter/Jetson/DGX-Spark SKUs and the `f`/`a` family targets are not generic-fatbin-friendly:

```cmake
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
# Base Turing→Blackwell + one PTX fallback. 120 == consumer/workstation Blackwell;
# 100 == datacenter Blackwell (NOT cubin-compatible with 120 despite both being "Blackwell").
set(STEPPE_CUDA_ARCH_RELEASE
    "75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
    CACHE STRING "Shippable base arch list (Turing→Blackwell + PTX fallback)")
```

> **CUDA-13 arch facts (verified).** CUDA 13.0 renumbered **Jetson Thor from `sm_101` (CUDA 12.8/12.9) to `sm_110`** — a copied 12.x gencode list containing `sm_101` *will fail* under CUDA 13. **`sm_103`** (B300 / GB300) is a **family target (`compute_103f`)**, not a base `sm_103` you drop in a generic fatbin; **`sm_121`** (DGX Spark) is **experimental**. We therefore omit all three from the default list. If you must ship one, add it deliberately. If a kernel ever needs an accelerated/family path (e.g. a CUTLASS-backed tcgen05/wgmma kernel on `sm_100a`/`sm_103f`), set `CUDA_ARCHITECTURES OFF` on *that one target* and pass explicit `-gencode arch=compute_100a,code=sm_100a` via `target_compile_options($<$<COMPILE_LANGUAGE:CUDA>:...>)`, version-tested against the CMake in use.

**Separable compilation + device LTO** (Release only). Device LTO requires `-dlto` at **both** compile and device-link. CMake's `INTERPROCEDURAL_OPTIMIZATION` is NVIDIA's documented switch for this *for the simple multi-arch case*, **but it conflicts with an explicit `-gencode` list**: when you pin architectures you must request the LTO intermediate per-arch via `code=lto_<arch>`, not a bare `-dlto`. We therefore drive it explicitly and keep `IPO` off for CUDA to avoid silently getting host-only LTO:

```cmake
set_target_properties(steppe_device PROPERTIES CUDA_SEPARABLE_COMPILATION ON)  # RDC
# Device LTO, explicit and arch-correct (NOT via INTERPROCEDURAL_OPTIMIZATION, which would
# either mean host LTO or fight the -gencode list). Emit lto_<arch> intermediates + final dlto.
target_compile_options(steppe_device PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-gencode=arch=compute_90,code=lto_90 -dlto>)
target_link_options(steppe_device PRIVATE $<$<CONFIG:Release>:-dlto>)
# Generate the lto_<arch> list from STEPPE_CUDA_ARCH_RELEASE in CUDAArch.cmake; the line above is illustrative.
```

> **[UNCERTAIN]** The exact, least-fragile CMake incantation for "device LTO across a *pinned multi-arch* list" varies by CMake version; CMake's own `INTERPROCEDURAL_OPTIMIZATION` handling for CUDA has changed across releases. Treat the snippet as the intent (compile+link both `-dlto`, per-arch `lto_<arch>`), and pin/verify it against the CMake actually in CI. Confirm with `cuobjdump --dump-elf` that the final binary contains real SASS for each `-real` arch and that LTO actually fired.

**Flag policy** — host vs device flags separated with generator expressions or nvcc rejects raw GCC/Clang flags:

```cmake
# cmake/SteppeWarnings.cmake — one source of truth, linked PRIVATE everywhere.
add_library(steppe_warnings INTERFACE)
add_library(steppe::warnings ALIAS steppe_warnings)
target_compile_options(steppe_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:--Werror;all-warnings>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>)  # Nsight source map
# NB: -G (full device debug) only in Debug; never combine with -lineinfo. -G makes kernels
# many× slower and timing/occupancy unrepresentative.
```

**Optional fatbin compression.** CUDA 13's nvcc supports `--compress-mode` (incl. higher-ratio modes) to shrink fatbins. We expose it as an opt-in cache var rather than asserting it as a free win, and only apply it on Release:

```cmake
option(STEPPE_COMPRESS_FATBIN "Use nvcc --compress-mode to shrink device fatbins" OFF)
if(STEPPE_COMPRESS_FATBIN)
  target_compile_options(steppe_device PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--compress-mode=size>)
endif()
```

**Dependency fetching (CPM)** — prefer the toolkit's bundled CCCL, but **enforce the 3.1 floor** because that is where the determinism API lives:

```cmake
# third_party/CMakeLists.txt — ALL pins here.
include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)
find_package(CCCL CONFIG)                          # toolkit copy: 3.1 on CUDA 13.1+, 3.0 on 13.0
if(NOT CCCL_FOUND OR CCCL_VERSION VERSION_LESS 3.1.0)
  message(STATUS "Toolkit CCCL missing or < 3.1; fetching CCCL 3.1 for determinism API")
  CPMAddPackage(NAME CCCL GITHUB_REPOSITORY NVIDIA/cccl GIT_TAG v3.1.0)
endif()
CPMAddPackage(NAME Eigen3 GITLAB_REPOSITORY libeigen/eigen GIT_TAG 3.4.0)
CPMAddPackage("gh:gabime/spdlog@1.14.1")
CPMAddPackage("gh:CLIUtils/CLI11@2.4.2")
CPMAddPackage("gh:google/googletest@1.15.2")
```

> **CCCL 13 header relocation.** Under CUDA 13 the bundled CCCL headers moved (`${CTK_ROOT}/include/cccl/`). Link `CCCL::CCCL` explicitly on every consuming target so the moved headers resolve; relying on `CUDA::cudart` to pull them in is not reliable. This is the consensus mitigation; the `find_package(CCCL)` + explicit `target_link_libraries(... CCCL::CCCL)` pattern below (§8) is what we use.

**`CMakePresets.json`** — `dev`, `debug`, `release`, `cuda-debug`, `asan`, `ci`:

```json
{
  "version": 6,
  "cmakeMinimumRequired": { "major": 3, "minor": 30, "patch": 0 },
  "configurePresets": [
    { "name": "base", "hidden": true, "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/${presetName}",
      "cacheVariables": {
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "CMAKE_CUDA_ARCHITECTURES": "native",
        "CMAKE_CUDA_COMPILER_LAUNCHER": "sccache",
        "CMAKE_CXX_COMPILER_LAUNCHER": "sccache" } },
    { "name": "dev",   "inherits": "base", "cacheVariables": { "CMAKE_BUILD_TYPE": "RelWithDebInfo" } },
    { "name": "debug", "inherits": "base", "cacheVariables": { "CMAKE_BUILD_TYPE": "Debug" } },
    { "name": "cuda-debug", "inherits": "debug",
      "cacheVariables": { "STEPPE_NVTX": "ON", "STEPPE_SANITIZER": "compute" } },
    { "name": "asan", "inherits": "dev", "cacheVariables": { "STEPPE_SANITIZER": "asan;ubsan" } },
    { "name": "release", "inherits": "base",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Release", "STEPPE_NVTX": "OFF",
        "CMAKE_CUDA_ARCHITECTURES": "75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual" } },
    { "name": "ci", "inherits": "release", "cacheVariables": { "STEPPE_BUILD_TESTS": "ON" } }
  ],
  "buildPresets": [
    { "name": "dev", "configurePreset": "dev" },
    { "name": "release", "configurePreset": "release" },
    { "name": "ci", "configurePreset": "ci" }
  ]
}
```

`compile_commands.json` is produced by Ninja and consumed by clang-tidy/IWYU/clangd. sccache caches host TUs reliably; **nvcc caching is best-effort** — separable-compilation and `-dlto` device objects may miss — so treat `sccache --show-stats` device-object hit rate as informational, not a gate (a flag like `-lineinfo` changing busts the key).

---

## 7. CUDA coding standards & idioms

**RAII wrappers — the only place CUDA resources are owned.** Application memory is allocated *only* inside `DeviceBuffer<T>` (device) / `PinnedBuffer<T>` (pinned host) / `allocator.cu` (pool); everything else takes non-owning `cuda::std::span`/`mdspan`. All owning types are **fully move-only: move-construct AND move-assign** (the old draft deleted move-assign on the handles, which silently made `h = std::move(other)` ill-formed — fixed below).

```cpp
// device/cuda/device_buffer.cuh — move-only owning device allocation.
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) : size_(n) {
        if (n) STEPPE_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));   // ALLOWLISTED TU
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

    cuda::std::span<T>       view()       noexcept { return {ptr_, size_}; }
    cuda::std::span<const T> view() const noexcept { return {ptr_, size_}; }
    T* data() noexcept { return ptr_; }  std::size_t size() const noexcept { return size_; }
private:
    void reset() noexcept {
        if (ptr_) { cudaError_t e = cudaFree(ptr_);                  // dtor never throws
                    STEPPE_DEBUG_ONLY(if (e) STEPPE_LOG_WARN("cudaFree at teardown: {}",
                                                             cudaGetErrorString(e))); }
        ptr_ = nullptr; size_ = 0;
    }
    T* ptr_ = nullptr;  std::size_t size_ = 0;
};
```

Library handles follow the **same fully-movable shape**; create them **once** at startup (`cublasDestroy`/`cusolverDnDestroy` implicitly synchronize, so never per-iteration), and route destroy-time errors to a debug log rather than swallowing them:

```cpp
class CublasHandle {
public:
    explicit CublasHandle(cudaStream_t s = nullptr) {
        CUBLAS_CHECK(cublasCreate(&h_)); if (s) CUBLAS_CHECK(cublasSetStream(h_, s));
    }
    CublasHandle(CublasHandle&& o) noexcept : h_(std::exchange(o.h_, nullptr)) {}
    CublasHandle& operator=(CublasHandle&& o) noexcept {           // move-ASSIGN now defined
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
// Stream, Event, CusolverDnHandle, CurandGenerator, PinnedBuffer<T> all follow this exact
// shape: move-construct + move-assign + dtor->destroy()-with-debug-log. No exceptions.
```

The snippet above is the **RAII shape** reference. The production `CublasHandle` additionally **owns the (stream, workspace) invariant** the §12 emulated-FP64 determinism contract depends on: it holds the non-owning workspace span and exposes `set_workspace(ptr, bytes)` + `set_stream(stream)`, where `set_stream` **re-applies** the pinned workspace after `cublasSetStream`. This is required because `cublasSetStream` "unconditionally resets the cuBLAS library workspace back to the default workspace pool" (cuBLAS §2.4.7, CUDA 13.x), so any stream change after the workspace is bound would otherwise silently discard the §12 reproducibility workspace. The backend binds both **once** at construction and the GEMM routines never call raw `cublasSetStream` — so the workspace survives for every GEMM batch on both the M0 and M4 paths (the M4 grouped path otherwise reset it per chunk). The single statistic stream rule (§12) is unchanged.

**`STEPPE_CUDA_CHECK` + post-launch checks.** One macro (`device/cuda/check.cuh`) captures file/line via `std::source_location` and throws a typed `CudaError`; sibling `CUBLAS_CHECK`/`CUSOLVER_CHECK` map those status enums (they do not share `cudaGetErrorString`). Every API call is checked; every launch is followed by a synchronous *and* (debug-only) async check:

```cpp
#define STEPPE_CUDA_CHECK(expr) ::steppe::detail::cuda_check((expr))   // throws CudaError w/ loc

#define STEPPE_CUDA_CHECK_KERNEL()                 \
  do { STEPPE_CUDA_CHECK(cudaGetLastError());      /* bad launch config */ \
       STEPPE_DEBUG_ONLY(STEPPE_CUDA_CHECK(cudaDeviceSynchronize())); /* async fault */ \
  } while (0)
```

The debug sync makes compute-sanitizer attribute faults to the exact kernel; release relies on the next runtime call to surface the sticky error. Throwing (not `exit()`) lets tests `catch (const CudaError&)`.

**Launch-config helpers** live once in `core/internal/launch_config.hpp`: `cdiv(n,b)`, `grid_for(n, block, max_grid)`, occupancy-aware block sizing. Kernel files never recompute grid math. A CUDA grid is capped per axis at `(x, y, z) = (2^31−1, 65 535, 65 535)` on every compute capability incl. Blackwell sm_120, so **the large (SNP/`M`-scale) extent always rides `gridDim.x`** (the decode kernel, the f2 feeder) — putting it on `y`/`z` is a latent launch failure at `M > ~1.05M` SNPs. `grid_for` debug-asserts its extent fits `max_grid` (default the y/z cap `kMaxGridZ`); the M4 strided-batched gather/scatter set `gridDim.z = n_in_group` directly (bypassing `grid_for`), so they route through the dedicated `grid_z_extent(n)` guard and the backend tiles the batch into ≤ `kMaxGridZ`-block chunks so the z extent never exceeds the limit (this is the chunk loop that already tiles for VRAM; the z cap is folded into `vram_budget.hpp::max_blocks_per_chunk`).

**CCCL usage policy.** Use the highest abstraction that fits; drop down only for control. Thrust for whole-container ops; **CUB** when you need explicit temp storage, a specific stream, or fused operators. **Two important constraints, reconciled here:**

1. The CCCL **determinism execution environment is a single-phase-API feature only.** A call that must be deterministic (`run_to_run`/`gpu_to_gpu`) uses the single-phase overload that takes the `env` and manages its own temporary storage:
   ```cpp
   namespace det = cuda::execution::determinism;
   auto env = cuda::execution::require(det::run_to_run);   // default for statistic path
   cub::DeviceReduce::Sum(d_in, d_out, n, env);            // single-phase; env-managed temp
   ```
2. The classic **two-call temp-storage query idiom** (`d_temp_storage == nullptr` to size, then allocate once, then call) does **not** accept an execution environment. Use it only for reductions that are **not** on the statistic path (throughput-only, `not_guaranteed`). **You cannot pre-allocate temp storage and request determinism on the same call** — pick one per reduction by whether the result feeds an assertion/golden.

`cuda::std::span`/`mdspan` are the interface currency — owning types hand out views; kernels accept only views, never owners or bare pointers:

```cpp
using F2Tensor = cuda::std::mdspan<double, cuda::std::dextents<int, 3>>;  // [pop × pop × block]
__global__ void f4_contract(F2Tensor f2, /* index lists */, cuda::std::span<double> out) { /* ... */ }
```

> **CUDA-13 note (verified):** cooperative groups can **no longer** be used for multi-device synchronization — the multi-device launch APIs (`cudaLaunchCooperativeKernelMultiDevice`, `multi_grid_group`) were removed. Single-device grid/warp collectives still work; prefer `cg::reduce` over hand-rolled `__shfl` *within a kernel*.

**Async memory pools.** A pool-backed allocator variant takes a `cudaStream_t` and uses `cudaMallocAsync`/`cudaFreeAsync` (both inside `allocator.cu`, allowlisted); set the pool release threshold high (`cudaMemPoolAttrReleaseThreshold = UINT64_MAX`) so per-iteration allocations hit the cache, not the OS. Respect stream-ordering: an allocation may only be touched by work ordered after the `cudaMallocAsync`; cross-stream use requires an event dependency. (This is why §13 runs compute-sanitizer with `--track-stream-ordered-races all`.)

**Streams & graphs.** One `Stream` per independent lane; express cross-stream deps with `Event`, not device-wide syncs. **Caveat that propagates to §12:** the statistic-bearing GEMM/SYRK/SVD/Cholesky must run on a *single* stream when bit-stable goldens are required, because cuBLAS reproducibility does not hold across concurrent streams (§12). Independent *throughput-only* work may use multiple streams. The fit engine replays the same kernel sequence thousands of times across the model space → capture the per-model fit into a CUDA graph (`cudaStreamBeginCapture` → `cudaGraphInstantiate` → `cudaGraphLaunch`), using `cudaGraphExecUpdate` when only parameters change. Keep *what runs* (a `record_work` lambda of tiny kernels) separate from *how it's orchestrated* (a `GraphPipeline`), so the same work runs eagerly under compute-sanitizer or captured in production.

**Host/device separation.** The **production hot path is library dense LA** (§5 S2): the heavy arithmetic is hand-issued cuBLAS/cuSOLVER GEMM/SYRK/Cholesky/SVD (native cuBLAS, **no array framework — no JAX/CuPy**, §0). Custom kernels are the *feeders and consumers* around those GEMMs — e.g. the S2 fused pre-pass (`Q,V,Qsq,Hc`) and the fused numerator+divide. Such a kernel is a thin `__global__` shell doing index math + bounds, calling a `__host__ __device__` pure per-element primitive that holds the per-element numerics — CPU-unit-testable and **shared with the reference backend so oracle and GPU can't diverge on a formula** (§13). The thin-shell-around-a-scalar-function model therefore describes the *reference backend and the per-element primitives*, **not** the structure of the hot path, which is reformulated tensor algebra (§2). Anything touching the CUDA runtime stays `__host__`-only in the platform layer; device functions are `noexcept`; shared headers use `cuda::std::` types. Kernels are exposed via narrow `void launch_xxx(...)` wrappers so host code never includes kernel bodies or `<<<>>>`.

---

## 8. DRY & shared utilities

All cross-cutting helpers live once in `src/core/internal/` (host-pure) or `src/device/cuda/` (CUDA), exposed through `INTERFACE` targets. The link wiring below makes the layering *compiler-enforced*:

```cmake
# src/device/CMakeLists.txt
add_library(steppe_device)
add_library(steppe::device ALIAS steppe_device)
target_sources(steppe_device PRIVATE cuda/cuda_backend.cu cuda/f2_block_kernel.cu
               cuda/decode_af_kernel.cu cuda/jackknife_kernel.cu cpu/cpu_backend.cpp)
target_link_libraries(steppe_device
    PUBLIC  steppe::core_internal CCCL::CCCL          # DRY helpers + CCCL (explicit, §6)
    PRIVATE steppe::warnings CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::curand)
# CUDA libs are PRIVATE → core/cli cannot see a CUDA header. backend.hpp is CUDA-free.

# src/core/CMakeLists.txt
add_library(steppe_core)
add_library(steppe::core ALIAS steppe_core)
target_link_libraries(steppe_core
    PUBLIC  steppe::api
    PRIVATE steppe::device steppe::warnings Eigen3::Eigen)   # reaches GPU only via ComputeBackend
```

| Concern | Single home | Contract |
|---|---|---|
| CUDA error check | `device/cuda/check.cuh` | `STEPPE_CUDA_CHECK(expr)` → `CudaError{file:line, cudaGetErrorName/String}` |
| cuBLAS/cuSOLVER check | `device/cuda/check.cuh` | `CUBLAS_CHECK` / `CUSOLVER_CHECK` switch on status enums |
| Error propagation (internal) | `internal/expected.hpp` | `STEPPE_TRY(expr)` over `std::expected<T,Error>` — **internal only; never crosses the ABI** (§16) |
| Logging | `core/internal/log.hpp` | `STEPPE_LOG_*` facade (a spdlog backend lands with §10); never `printf`/`cout` in lib code; also the teardown-warning sink `STEPPE_LOG_WARN` (§7) |
| NVTX + colors | `internal/nvtx.hpp` | `STEPPE_NVTX_SCOPE("name", color)` RAII; **the color palette (`nvtx::Color`) is defined here, once** |
| Launch math | `core/internal/launch_config.hpp` | `cdiv` (int + long), `grid_for(n, block, max_grid)` (debug-asserts the y/z 65 535 cap), the per-kernel block dims (`kDecodeBlockX/Y`), the hardware grid limits `kMaxGridX/Y/Z`, and `grid_z_extent(n)` for the M4 batch axis (which sets `gridDim.z` directly, bypassing `grid_for`); occupancy sizing later |
| Host/device + debug | `core/internal/host_device.hpp` | `STEPPE_HD` (the one host/device qualifier), `STEPPE_DEBUG_ONLY`, `STEPPE_ASSERT` |
| Precision / traits | `internal/type_traits.hpp` | `real_t<P>`, `is_fp64`, index aliases — the precision switch lives here |
| Views | `internal/span_view.hpp` | non-owning matrix/tensor views over `cuda::std::span`/`mdspan` |
| **SNP→block rule** | `core/domain/block_partition_rule.hpp` | the forward `int block_of(genetic_pos, blgsize)` + `assign_blocks` (per-SNP `block_id[]`) AND the inverse `block_ranges(block_id, M, n_block)` (per-block `[begin,end)` ranges, validated once) — host-pure, consumed by `io` *and* both device backends (§5; the inverse was previously hand-duplicated per backend, cleanup X-3/B3) |
| **Reference backend** | `device/cpu/cpu_backend.cpp` | same `ComputeBackend` interface, scalar host implementation |

**The CPU reference backend is both DRY and the correctness anchor.** `ComputeBackend` (in `device/backend.hpp`, CUDA-free) is one interface with two implementations (`CudaBackend`, `CpuBackend`). The compute layer is written once against the interface and never branches on GPU-vs-CPU. The deliberately-naive scalar `CpuBackend` is what the GPU f2/jackknife kernels are continuously diffed against in `tests/reference/`, and both are validated against ADMIXTOOLS 2 via `tools/compare_against_admixtools2.R`. It also guarantees `import steppe` works GPU-free.

---

## 9. Configuration management

**Typed, immutable, layered, injected.** Resolution order (lowest precedence first): `compiled defaults < TOML file < env (STEPPE_*) < CLI`. A mutable `ConfigBuilder` accumulates layers; `.build()` validates once — **including the device-memory budget (§11.2)** — and freezes into an immutable `RunConfig`. After construction, config is `const`.

```cpp
// include/steppe/config.hpp (public)
enum class Precision {
    EmulatedFp64,   // DEFAULT for ALL matmul-heavy stages incl. the f2 GEMMs (§5 S2, §12). Ozaki
                    //   emulation with a FIXED mantissa-bit count (DeviceConfig::mantissa_bits, default
                    //   40 ⇒ ≈ native FP64; 32 ⇒ 8.6e-9, faster). MEASURED 8–17× over native FP64 on
                    //   real AADR. Do NOT use dynamic mantissa control — it overshoots to ~60 bits on
                    //   real data and collapses to parity (the trap, §12). Accuracy-approximate, not
                    //   bit-identical; not IEEE-754 on specials. The tiny f2 numerator/divide and the
                    //   ill-conditioned GLS/SVD stay native Fp64.
    Fp64,           // Native FP64. The validation oracle / gold reference. Used for the
                    //   cancellation-prone elementwise math, the small ill-conditioned GLS/SVD,
                    //   and as the reference every other mode is validated against (§12).
    Tf32            // Opt-in fast/approximate. Model-space screening / ranking ONLY. Results carry a
                    //   precision tag and land in the loose tolerance tier; NEVER bit-compared to AT2
                    //   goldens and never emitted as a reported est/se/z/p without re-validation (§12).
};
enum class JackknifeMode { Delete1, Bootstrap };

struct DeviceConfig {                         // resources injected, not globally discovered
    std::vector<int> devices;                              // SINGLE source of truth for which/how many GPUs.
                                                           //   empty ⇒ auto-enumerate all visible CUDA devices in
                                                           //   enumeration order; a non-empty list PINS both the set
                                                           //   AND the ordering, which is the fixed f2_blocks combine
                                                           //   order (§11.4, §12). Size 1 ⇒ single-GPU; CPU backend
                                                           //   ignores it. (No separate count field — count == size.)
    Precision    precision      = Precision::EmulatedFp64;  // default for ALL matmul-heavy stages incl. f2 GEMMs (§12)
    int          mantissa_bits  = 40;                       // fixed-slice Ozaki bit count: 40 ≈ native FP64; 32 ⇒
                                                            //   8.6e-9 worst-case (faster). FIXED, never dynamic (trap,
                                                            //   §12). Fp64 is the oracle/fallback; Tf32 screening-only.
    std::size_t  stream_count   = 1;                        // 1 statistic stream PER GPU (cuBLAS reproducibility, §12)
    std::size_t  search_streams = 4;                        // throughput-only lanes for the model-space search (S8)
    bool         use_mem_pool   = true;
    bool         enable_peer_access = true;                 // cudaDeviceEnablePeerAccess when canAccessPeer (§11.4)
    bool         deterministic  = true;                     // run_to_run reductions + scoped det cuSOLVER + explicit
                                                            //   cublasSetWorkspace for emulated FP64 + the parity
                                                            //   multi-GPU combine (fixed host-order, §11.4) (§12)
};

class RunConfig {                             // immutable value object — const accessors only
public:
    [[nodiscard]] const DeviceConfig& device() const noexcept;
    [[nodiscard]] double              block_size_cm() const noexcept;   // see unit note below
    [[nodiscard]] JackknifeMode       resampling() const noexcept;
    [[nodiscard]] int                 bootstrap_reps() const noexcept;  // 0 ⇒ delete-1 jackknife
    [[nodiscard]] std::uint64_t       seed() const noexcept;            // recorded in golden meta
};

class ConfigBuilder {                         // the ONLY mutable config type
public:
    ConfigBuilder& with_defaults();
    ConfigBuilder& merge_file(const std::filesystem::path&);
    ConfigBuilder& merge_env();                                   // STEPPE_* prefix
    ConfigBuilder& merge_cli(const CliArgs&);
    [[nodiscard]] std::expected<RunConfig, Error> build() const;  // validates HERE, fail-fast
};
```

**Why `Precision` has three named modes (and what each is *for*).** Earlier drafts exposed only `Fp64` (after an even earlier `Fp32` foot-gun — an ABI-stable enum value constructible but illegal everywhere). That was too blunt: the qpAdm hot path is matmul-heavy (covariance SYRK/GEMM over jackknife blocks, f4 assembly), and CUDA 13.x ships **Ozaki-scheme fixed-point FP64 emulation** in cuBLAS/cuSOLVER that targets native-FP64 accuracy at tensor-core throughput. So the default is now **`EmulatedFp64`** for that matmul work; **`Fp64`** (native) is the oracle/reference and the path for the cancellation-prone elementwise math; **`Tf32`** is an opt-in screening mode. The load-bearing scope rule: **emulation governs only the well-conditioned matmuls.** Allele-frequency accumulation and the catastrophic-cancellation-sensitive f2/f4 subtraction `(a−b)(c−d)` stay in **native FP64 arithmetic regardless of `precision`** — emulation faithfully computes the *matrix product*, it cannot recover significant bits annihilated in a prior subtraction, so a low-precision subtraction is unrecoverable downstream (§12). `EmulatedFp64` must be validated against pedantic native FP64 before it is trusted, and `Tf32` results are typed as candidate-rankings (never `QpAdmResult`) with the final number always recomputed in `EmulatedFp64`/`Fp64` (§12). Emulated FP64 is accuracy-approximate (≈ or better than native FP64 under dynamic mantissa control), **not bit-identical to native FP64** and not IEEE-754 compliant on special values — which is exactly why native `Fp64` remains the oracle.

**Block-size unit (verified against ADMIXTOOLS 2 semantics).** ADMIXTOOLS 2's `blgsize` default is `0.05` **Morgans** = **5 cM**. `block_size_cm()` returns cM (default `5.0`); internally we store Morgans to match upstream block math. The accessor name says cM; the stored value is Morgans — the conversion lives in one place next to `block_partition_rule.hpp`. Do not conflate the two.

**Dependency injection of resources.** `RunConfig` says *what*; a `Resources` struct supplies concrete handles, injected into every compute call:

```cpp
struct PerGpuResources {                              // one per device in DeviceConfig::devices; all RAII-owned
    int                  device_id;                   // the physical CUDA device
    StreamPool           streams;                     // 1 statistic stream + N search streams, on THIS device
    DeviceAllocator*     allocator;                   // injected (non-owning) pool / cudaMalloc, on THIS device
    CublasHandle         blas;                         // RAII per-device handle (created once, §7)
    CusolverDnHandle     solver;                       // RAII per-device handle (created once, §7)
    NcclComm             comm;                          // RAII-owned NCCL communicator (broadcast; §11.4) — empty on 1-GPU/CPU
};
struct Resources {                                    // all RAII-owned
    std::unique_ptr<ComputeBackend> backend;          // CUDA (multi-GPU aware) or CPU, chosen at build()
    std::vector<PerGpuResources>    gpus;             // one entry per device; size 1 on single-GPU / CPU
};
QpAdmResult run_qpadm(const GenotypeDataset&, const QpAdmModel&,
                      const RunConfig&, Resources&);  // no hidden globals
```

**Validation at `build()`** rejects: an unknown/unbuilt arch; any device id in `DeviceConfig::devices` that is absent or duplicated (empty ⇒ auto-enumerate, never invalid); `bootstrap_reps > 0` with `Delete1`; `deterministic == true` with `stream_count > 1` on the statistic path (forces 1, or errors if explicitly set higher); a `precision` the selected backend cannot honor (e.g. `EmulatedFp64`/`Tf32` requested with no CUDA device or on a toolkit/arch where the emulation math mode is unavailable — fall back to native `Fp64` or error); **`deterministic == true` with `precision == EmulatedFp64` unless an explicit `cublasSetWorkspace` workspace is configured**, since fixed-point emulation voids the run-to-run bit-wise guarantee without an adequate workspace (§12); and **any config whose estimated peak VRAM exceeds device free memory** (§11.2). At `build()`, `DeviceConfig` also chooses the backend: no CUDA device or `--device cpu` ⇒ `CpuBackend`. Injected allocators/streams (not singletons) are what make compute unit-testable.

---

## 10. Error handling, logging & observability

**Error taxonomy (the categories, not just the mechanism).** Callers must distinguish *recoverable domain outcomes* from *faults*. The public boundary returns a C enum `steppe_status_t` (§16); internal code carries the richer `Error` with a matching `category` and a message. The taxonomy:

| Code | Category | Recoverable? | Meaning |
|---|---|---|---|
| `STEPPE_OK` | — | — | success |
| `STEPPE_ERR_INVALID_CONFIG` | config | caller fixes input | failed `ConfigBuilder::build()` validation (bad arch, conflicting flags) |
| `STEPPE_ERR_IO_FORMAT` | io | caller fixes input | malformed/unsupported `.bed`/`.geno`/`.snp`/`.ind`; bad magic bytes |
| `STEPPE_ERR_DEVICE_OOM` | resource | maybe (smaller chunk/budget) | device allocation or budget check failed (§11.2) |
| `STEPPE_ERR_CUDA_RUNTIME` | fault | no | a `CudaError` (sticky/async fault, bad launch) — a bug or environment failure |
| `STEPPE_ERR_NON_SPD_COVARIANCE` | **domain outcome** | **yes** | `Q` not SPD; Cholesky failed. A *statistical* result (degenerate/collinear model), not a bug |
| `STEPPE_ERR_RANK_DEFICIENT` | **domain outcome** | **yes** | rank test / GLS hit a rank-deficient `X`; the model is unidentifiable |
| `STEPPE_ERR_CHISQ_UNDEFINED` | **domain outcome** | **yes** | dof ≤ 0 or χ² not computable for this model |

The three **domain outcomes** are *expected* results of fitting some models in a large search; the API surfaces them as ordinary per-model statuses (the search records them and moves on), **not** as exceptions or process aborts. Faults (`CUDA_RUNTIME`, and `INVALID_CONFIG` at build time) are fail-fast.

**Errors mechanism.** Internal code propagates `std::expected<T, Error>` via `STEPPE_TRY`. CUDA failures throw a typed `CudaError` (file/line/name/string) at the call site through `STEPPE_CUDA_CHECK`; `CUBLAS_CHECK`/`CUSOLVER_CHECK` translate status enums. The public API converts all of this to `steppe_status_t` and **never lets exceptions or `std::expected` cross the ABI** (§16). Fail-fast: validation at `build()`, post-launch `cudaGetLastError()` always, forced sync in debug. RAII destructors never throw but log a warning on nonzero destroy status in debug (§7).

**Logging.** Never `printf`/`std::cout` in library code. `internal/log.hpp` wraps spdlog behind `STEPPE_LOG_*` so sinks/levels/async are swappable; levels and sinks come from `RunConfig`, not globals. The Python binding installs a sink forwarding to Python's `logging`.

**Observability (NVTX).** `internal/nvtx.hpp` provides an RAII scope emitting named, colored ranges, gated by the `STEPPE_NVTX` CMake option (zero overhead in release). The color palette (`nvtx::Color`) is defined **once** in that header — no per-call-site color literals. Annotate every pipeline phase so they appear named on the Nsight Systems timeline and can trigger capture ranges:

```cpp
void CudaBackend::compute_f2_blocks(/* ... */) {
    STEPPE_NVTX_SCOPE("f2_block_kernel", nvtx::Color::Green);     // no-op if STEPPE_NVTX=OFF
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(/* ... */, stream));
    f2_block_kernel<<<grid, block, 0, stream>>>(/* ... */);
    STEPPE_CUDA_CHECK_KERNEL();
}
```

Standard ranges: `read_genotypes`, `decode_af`, `f2_blocks`, `f4_contract`, `jackknife_cov`, `rank_test`, `gls_fit`. These line up directly with the profiling workflow (§11.3).

---

## 11. Memory & performance strategy

### 11.1 Out-of-core genotype streaming (the part that can actually fall over)

The precompute pass is bandwidth-bound *and out-of-core*: real EIGENSTRAT/PLINK datasets are 1M+ SNPs × thousands of samples (tens of GB of packed genotypes), far exceeding any single GPU's VRAM. The `f2_blocks` output is **scale-dependent** (`O(P²·n_block)`): MB-scale at AT2-typical population counts (where it stays resident, point 3 below), but tens of GB at thousands of pops (≈76 GB at `P=2500`). We therefore **never load the genotype matrix whole, and we do not assume the output is always resident either.** The mechanism (**BUILT — M5**):

1. **Tile over SNPs.** `io` streams the dataset as contiguous SNP tiles of `T` SNPs (rows). For each tile: decode (S0) → allele-freq reduce (S1) → f2 partial + block-bin accumulate (S2), then discard the tile. Only `afmat`/`countmat` for the *current tile* and the running `f2_blocks` accumulator are resident.
2. **Double-buffered, pinned, overlapped I/O.** Two `PinnedBuffer` staging slots per tile feed `cudaMemcpyAsync` on a copy stream while the compute stream processes the previous tile (classic two-buffer pipeline). Pinned memory is required for true H2D overlap; pin *only* the staging slots.
3. **Block accumulator, and its tiering when large.** `f2_blocks` is `[n_pop × n_pop × n_block]`. At AT2-typical `P` it is MB-scale and **stays on the device** across the whole stream and the entire downstream model search (the device-resident `DeviceF2Blocks` handle, M5 `1f80c0c`); tiles never touch it except to add their contribution via the shared `block_partition_rule`. **When the result is larger than free VRAM** (thousands of pops — e.g. 76 GB at `P=2500`), M5 spills it to the **fastest tier it fits**: *Resident* (VRAM, the unchanged resident path) → *HostRam* (block-by-block host spill) → *Disk* (a `STPF2BK1` cache file via a persistent pinned staging ring + background writer that overlaps the GPU compute of the next chunk). The tier is selected **at runtime** from `cudaMemGetInfo` + `sysinfo` probes (the CUDA-free `OutputTier` + `select_output_tier()` in `src/device/tier_select.hpp`), so small `P` keeps the resident fast path with no penalty and large `P` streams out-of-core (M5 `176a07d`). The streamed tiers reuse the resident path's per-block gather/GEMM/assemble verbatim; only *where* a slab lands changes, never its bits (parity by construction, §12).
4. **`afmat`/`countmat` sizing, and SNP-tile input streaming.** These are per-tile `[T × n_pop]`, *not* `[SNP × n_pop]` — they scale with the tile, never the dataset. The streamed path likewise decodes **only the current SNP-column tile** `[s_lo, s_hi)` of the host `[P × M]` inputs (the full inputs stay in host RAM), so the GPU per-chunk footprint is **`O(P·tile + P²·n_block)`, independent of the SNP count `M`** — the old all-`M` feeder wall (`7·P·M` doubles resident) is removed (M5 `c65179f`). This is what lets full-autosome `P=2500` (M=584131, n_block=757) complete on a single 32 GB GPU with GPU peak bounded ≈26 GB.

**Chunk-size derivation.** Choose the largest tile `T` such that the resident working set fits a target fraction (say 60–70%) of free VRAM, leaving headroom for cuSOLVER/cuBLAS workspaces (§11.2) and the f2 accumulator:

```
bytes_per_tile(T) ≈ 2 * [ T * n_sample * sizeof(dosage)        # 2 pinned-staged + device dosage buffers
                        + T * n_pop * sizeof(double) * 2 ]      # afmat + countmat, double-buffered
T_max = floor( (VRAM_budget − bytes(f2_blocks) − bytes(workspaces)) / bytes_per_tile(1) )
```

`extract-f2 --dry-run` prints `T_max`, the number of tiles, and the **largest dataset (in SNPs × samples) that fits a given VRAM** at the chosen budget, so users size jobs before launching. If `T_max < 1` (a single SNP row's working set exceeds budget — pathological sample counts), `build()` returns `STEPPE_ERR_DEVICE_OOM` with the shortfall.

### 11.2 Workspace & VRAM budget (quantified, validated at `build()`)

Generic "query bufferSize once and reuse" is correct but unquantified; here is the budget the allocator actually checks. With `n_pop = P`, blocks `n_block = B`, fit-matrix dim `m = (n_L−1)·(n_R−1)`, batch count `K` (models × replicates in flight):

| Resident object | Bytes | Notes |
|---|---|---|
| `f2_blocks` | `P² · B · 8` | FP64 storage, resident for the whole run (storage is FP64 in every precision mode — emulation/TF32 are operation modes, not storage types) |
| `Vpair` | `P² · B · 8` | the retained per-block pairwise-valid count (the S4 jackknife weight, §5 S2 (a)) — an EQUAL-sized FP64 tensor held co-resident with `f2_blocks` (device path: `dF2_all` + `dVpair_all`), so the resident pair is `2 · P² · B · 8`. `build()` MUST reserve for both, else it under-reserves by `P²·B·8` and OOMs mid-stream (the M4 device budget reserves both — `src/device/vram_budget.hpp`; cleanup X-13/B26) |
| `Q` (covariance) | `m² · 8` | per active model |
| f4 matrix `X` | `m · B · 8` | per active model |
| cuSOLVER `potrf`/`gesvd` workspace | `ws_solve(m) · 8` | **query the largest `_bufferSize` once** over the batch and reuse; size grows with `m` |
| cuBLAS SYRK/GEMM workspace | `ws_blas · 8` | one workspace **per stream**; on the statistic path that is exactly one (§12) |
| Genotype tile working set | `bytes_per_tile(T)` | §11.1; dominant term during precompute |
| Pinned staging | `2 · T · n_sample · sizeof(dosage)` | double-buffered |

`ConfigBuilder::build()` sums the resident terms for the configured `K`/`T`, compares against `cudaMemGetInfo` free memory at the chosen budget fraction, and **rejects over-budget configs up front** (`STEPPE_ERR_DEVICE_OOM`) rather than failing mid-stream. The single expression validated is `total_vram(P, B, m, K, T) ≤ budget · free`. Set the pool release threshold to `UINT64_MAX` so variable-size per-replicate allocations recycle from cache instead of round-tripping the OS.

**Output-does-not-fit → adaptive tiering (BUILT, M5).** The `2·P²·B·8` resident-pair row above is the *Resident* tier; it is `O(P²·B)` and at thousands of pops it itself exceeds VRAM (≈76 GB at `P=2500`, ≈220 GB at `P=4266` — §11.4). When the result (plus working set) does not fit free VRAM, `build()` does **not** simply reject: M5 selects a streamed output tier (`OutputTier` / `select_output_tier()`, `src/device/tier_select.hpp`) — *HostRam* (block-by-block host spill) if it fits free host RAM, else *Disk* (`STPF2BK1` cache). The streamed tiers budget against the **real** streamed footprint `(4·P·s_pad + 8·P²)·n_block + tile feeder` (`O(P·tile + P²·n_block)`, independent of `M`), **not** the resident budget that reserved the full result tensor and the `7·P·M` all-`M` feeder — those terms are scoped to the *Resident* tier only. So `STEPPE_ERR_DEVICE_OOM` is the verdict when even a single streamed chunk cannot fit, not when the whole result cannot.

### 11.3 Performance & profiling

S0–S2 are bandwidth-bound: maximize coalescing on the `.bed` unpack, fuse decode→reduce→**feeders**(`Q,V,Qsq,Hc`)→3-GEMM→cancellation-step→block-bin so the `[SNP × pop × pop]` intermediate is never materialized (§5 S2, §11.1). S3–S8 are launch-/throughput-bound across many tiny solves: batch over `n_block` and model index (strided-batched GEMM/SYRK for homogeneous shapes, pointer-array batched only for non-uniform shapes), capture the per-model fit into a CUDA graph, and keep enough independent solves in flight to hide per-launch latency on the per-model SVD fallback (§5). On a multi-GPU box, S8 is additionally sharded across devices (§11.4).

**Profiling workflow — system-level before kernel-level.**
1. **Nsight Systems first** (where does time go?): `nsys profile --trace cuda,nvtx,osrt --cuda-memory-usage true -o run ./steppe ...`. Read `cudaapisum` (excessive `cudaMalloc`/launch overhead) and `gpukernsum` (dominant kernel). Look for GPU idle gaps (launch-bound) and un-overlapped copies (the tile pipeline of §11.1 should show overlapping copy/compute lanes). NVTX ranges name the phases.
2. **Nsight Compute second**, scoped to the kernel `gpukernsum` named: `ncu --set full --kernel-name regex:f2_block_kernel --launch-count 1 -o k ./steppe ...`. Read Speed-Of-Light + Roofline (f-stat reductions land memory-bound), Occupancy (and its limiter), then Memory Workload (DRAM throughput, L2 hit, coalescing). Never run `--set full` over a whole app; ncu replays kernels and is far slower than nsys.

### 11.4 Multi-GPU execution (single node)

Single-node multi-GPU is a day-one objective (§0, §1): parallelism + speedup across the G devices of one workstation **with parity** (§12). The model is **single-process-multi-GPU (SPMG)** — one host process driving all G devices, one host thread + per-device CUDA streams per device, `cudaSetDevice` to switch, NCCL for broadcast only. One address space keeps `f2_blocks` and the model-space results in-process (no MPI, no launcher), and direct peer access (`cudaDeviceCanAccessPeer`/`cudaDeviceEnablePeerAccess`, §9 `enable_peer_access`) is enabled opportunistically. **[UNCERTAIN]** CUDA 13 removed the multi-device cooperative-launch APIs (`cudaLaunchCooperativeKernelMultiDevice`, `multi_grid_group`, §7), which is fine — steppe never spans one grid across devices; all cross-GPU coordination is host-level.

The two shardable phases, both additive on the existing tiling and per-stream design (§11.1, §9):

1. **Precompute (S0–S2): tile sharding + a host-side fixed-order combine.** The streamer partitions SNP tiles across the G devices; each device accumulates its own partial `f2_blocks` (+ `Vpair`) — and within a device each block's accumulation order is fixed (CCCL `gpu_to_gpu` partials make a block's partial identical regardless of *which* physical GPU computed it). The G partials are then combined **once, host-side, summed in fixed device order** (`g = 0..G−1`, the `DeviceConfig::devices` order). Because the combined artifact (`f2_blocks` + the co-resident equal-sized `Vpair`, `2·P²·B` doubles — kB–low-MB at AT2-typical P, but `O(P²·B)` and large at scale: it reaches ≈76 GB at `P=2500` and ≈220 GB at the §0 top end `P=4266`/`B=757`, and is itself VRAM-budgeted / adaptively tiered per §11.2) is off the bandwidth critical path, this combine is essentially free; **the result is broadcast back** (NCCL Broadcast or `cudaMemcpy`, both order-independent and therefore parity-safe) so every GPU holds the full `f2_blocks`. steppe does **not** put this reduction on default NCCL AllReduce — see §12 for why (AllReduce order varies with GPU count, breaking parity). **Capability-tiered combine (re-verified, workflow `wxz1fiiln`).** On hardware with peer access (RTX PRO 6000 stock driver; full-host 5090s via the aikitoria open-kernel-module P2P patch — dev-only) the combine may run **device-resident**: GPU 0 pulls each peer's partial via `cudaMemcpyPeer` (a byte-exact DMA copy) and sums them in the same fixed `g = 0..G−1` order on-device — **bit-identical** to the host-staged combine and the single-GPU reference (the transport only moves bytes; software fixes the order; still never NCCL AllReduce). The **host-staged fixed-order combine remains the portable parity baseline** and is the only path on the budget consumer box; the P2P combine is an opt-in, `cudaDeviceCanAccessPeer`-gated fast-path with an explicit logged fallback (the device-resident combine assembles a `DeviceF2Blocks` with no final D2H; the no-peer path uses a host bounce purely as the cross-card assembly transport and re-uploads, so the return is still a device-resident handle — M5 `1f80c0c`). Because this cross-GPU traffic is kB–MB and off the critical path, it is architectural cleanliness, not a throughput lever — steppe is deliberately P2P/NVLink-insensitive. **Honest role on the precompute (MEASURED, M4.5/M5).** Tile sharding is a *modest* throughput layer here, **not the at-scale enabler**: it was measured *slower* than single-GPU until the host data-bounce was removed, and nsys still shows only ≈22–74% compute overlap with a serial D2H/host tail (`docs/cleanup/m4.5/{why-multigpu-slow,parallelism-check,architecture-audit}.md`). What actually unlocked scale on the precompute was getting the result off the CPU (device-resident output) and the M5 input/output streaming above (§11.1) — not sharding. Multi-GPU's decisive payoff is phase 2 (S8), next.

2. **Model-space search (S8): embarrassingly parallel, zero cross-GPU traffic.** With `f2_blocks` replicated, models are partitioned across devices via a dynamic atomic work-queue over model indices (load-balances uneven per-model cost). Each fit runs wholly on one device, so its internal reductions stay single-GPU and inherit single-GPU parity; results are returned to the host and **re-sorted by model index**, so the final ordering is deterministic regardless of which GPU produced which model.

Because the `f2_blocks` broadcast is off the bandwidth critical path (kB–MB at AT2-typical P; larger at thousands of pops, where the result is anyway tiered to host/disk per §11.1/§11.2 rather than broadcast whole) and the search phase has **zero** inter-GPU traffic, steppe is largely insensitive to NVLink-vs-PCIe — it is viable on commodity multi-GPU workstations; the streaming build is bound by host→device ingest, disk, and the output spill, not inter-GPU links. **cuSOLVERMp stays deferred** (§1): it distributes one *large* dense factorization across GPUs, but qpAdm's `Q`/`X` are low-double-digit — the right parallelism is across many independent small fits (above), not within one distributed solve, which would be pure overhead and worse for determinism. **Multi-node** (multiple processes / MPI) is the remaining deferred boundary.

---

## 12. Numerical correctness & reproducibility

**Precision policy.** Three named modes (`Precision`, §9), assigned by the **conditioning of the operation, not by whether it is a matmul**. **The cancellation-sensitive elementwise math is always native FP64.** Allele-frequency accumulation and the f2/f4 difference-of-products `(a−b)(c−d)` are differences of large nearly-equal quantities — catastrophic cancellation, where the leading digits annihilate; if the operands enter at less than FP64 the rounding error already baked in becomes the dominant surviving content, and no downstream GEMM can reconstruct it. These stay native FP64 **regardless of `precision`** (the allele-freq pass is bandwidth-bound anyway, so reduced precision buys no throughput). **The matmul-heavy covariance `Q` / SYRK / f4 assembly defaults to `EmulatedFp64`** — cuBLAS fixed-point (Ozaki-scheme) FP64 emulation, which targets accuracy ≈ native FP64 under dynamic mantissa control on tensor cores; this is the well-conditioned accumulate-many-similar-terms regime where emulation earns its place.

**The S2 f2 GEMMs use FIXED-slice Ozaki emulation (MEASURED on real AADR — supersedes the earlier native-FP64 assumption).** The numerator `Σp_i² + Σp_j² − 2Σp_i p_j` (`= Σ(p_i−p_j)²`) is catastrophic cancellation — but emulation *does* survive it, and the number of slices needed is **small**: on real AADR (P up to 4,266, 100k SNPs) a **fixed 32-bit mantissa** yields 8.6e-9 worst-case f2 error (40-bit → 2.2e-11 ≈ native; 48-bit → 1e-12), at **8.5–17.5× the speed of native FP64** (the lead grows with population count, arithmetic intensity ≈ P/8). So the f2 GEMMs default to **`EmulatedFp64{mantissa_bits=40}`** (drop to 32 when 8.6e-9 suffices). The decisive trap, also measured: **dynamic** mantissa control auto-selects ~60 bits on real data's wide dynamic range — far more slices than f2 needs — collapsing to **parity with native (no win)**. Hence *fixed* slices, never dynamic. **The FIXED pin is what makes `EmulatedFp64` *honorable*, and the device layer enforces this as a single predicate**: `emulation_honorable(precision)` (device `f2_block_kernel.cu`) is true only for `EmulatedFp64` on a build carrying the fixed-slice tuning API (`STEPPE_HAVE_EMU_TUNING`, default ON); **both** the math-mode engagement (`engage_f2_precision`) and the cuBLAS compute-type mapping (`f2_compute_type`) consult it, so they can never disagree. On a toolkit/build without the tuning, `cublasSetFixedPointEmulationMantissaControl(FIXED)` cannot be called and cuBLAS would otherwise default to `CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC` (the trap); instead the path **downgrades to native `Fp64`** (PEDANTIC math + `CUBLAS_COMPUTE_64F`) and emits a one-shot capability-tagged log line — never silently running dynamic while reporting the `EmulatedFp64` tag (§9 build() "fall back to native `Fp64` or error"; cleanup X-6/B2). The tiny `O(n_pop²)` numerator/divide step stays native FP64. **Cautionary tale:** synthetic uniform data falsely showed dynamic Ozaki at 8×; real data showed parity — precision/throughput is benchmarked on real data only. **Parity caveat:** the masked-GEMM path reproduces AT2's pairwise-complete NaN-mean *result*, not its loop structure, so a slow native-FP64 validation oracle walking the exact AT2 pairwise-complete path (`CpuBackend`) is diffed against the fast GEMM path on the goldens — in particular the `Vpair==0` branch, the `q(1−q)/max(N−1,1)` denominator, and the allele-count `N` convention must match AT2, and the per-block `Vpair` carried to the S4 jackknife weighting must compose to AT2's `f2_blocks` definition (not double-normalize). All of this lives in the shared `__host__ __device__` feeder primitive so oracle and GEMM path can't diverge on the formula (§13). **[UNCERTAIN]** native-FP64 GEMM accumulation order is itself implementation-defined; we rely on `run_to_run`/single-stream + the oracle diff for bit-stability, not on assuming `Σp_i²` reduces in source order. **The small, dense, potentially ill-conditioned GLS Cholesky + SVD rank test stays native `Fp64`**: tiny, so emulation buys no throughput, and oracle-grade arithmetic is what you want near rank-deficiency. **`Tf32` (19 bits: 1 sign + 8 exponent + 10 mantissa) and FP16 never produce a reported statistic** — `Tf32` is permitted *only* as a fast model-space screen (ranking/feasibility), with survivors recomputed in `EmulatedFp64`/`Fp64` before any `est`/`se`/`z`/`p` is emitted. Enable emulation at the handle level — `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` / `cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH)`, and `cusolverDnSetMathMode(…, CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH)` for the covered factorizations; no per-matmul descriptor attribute is confirmed [UNCERTAIN]. Mandatory gate: **`EmulatedFp64` is accuracy-approximate, not bit-identical to native FP64** (and not IEEE-754-compliant on special values), so every reported quantity is spot-validated against the native-FP64 oracle per release — recompute the covariance for a sample of jackknife blocks in native `Fp64` and require the downstream `est`/`se`/`z`/`p` to match the oracle to all reported digits; if it fails, raise the mantissa-bit/slice target or fall back to native `Fp64` — do not ship. For the oracle pass use `cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)`. [UNCERTAIN] cuSOLVER `potrf` (Cholesky) and plain `gesv`/`gels` are absent from the documented emulation-affected routine list, so we treat them as native-precision; the emulation-affected dense set is `geqrf`/`syevd`/`syevdx`/`gesvd` (m>n)/`gesvdj`/`gesvdr`/`gesvdp` and their `X` variants. (The separate IRS iterative-refinement solvers — low-precision factorization including `CUSOLVER_R_TF32` → FP64 refinement — are a distinct path we do not rely on for reported numbers.)

**The determinism guarantee, stated precisely (and its boundaries).** The thesis "bit-stable on a given GPU → non-flaky regression gates" holds **only under all of the following**, and we encode each as a hard constraint, not a hope:

- **Reductions: CCCL `run_to_run`, single-phase API.** CCCL 3.1's `cuda::execution::require(determinism::run_to_run)` selects the existing two-pass reduction (status quo, low cost) and guarantees *same bits across runs on the same GPU* — exactly what kills flaky gates. `gpu_to_gpu` (the new reproducible-FP-accumulation path) is bitwise-identical *across GPUs* but slower; we expose it behind a flag for cross-machine reproducibility of *published* numbers. `not_guaranteed` (new single-pass atomic) is allowed only on throughput paths whose output never feeds an assertion. As established in §7, determinism requires the single-phase overload; you cannot also pre-allocate temp storage on that call.
  ```cpp
  namespace det = cuda::execution::determinism;
  cub::DeviceReduce::Sum(d_in, d_out, n, cuda::execution::require(det::run_to_run));  // default
  ```
- **cuBLAS: single stream on the statistic path.** cuBLAS's bitwise-reproducibility guarantee **explicitly does not hold when multiple CUDA streams are active** — under concurrency the library may pick different internal implementations regardless of per-stream workspaces. Per-stream `cublasSetWorkspace` prevents *corruption*, not *nondeterministic algorithm selection*. Therefore the statistic-bearing GEMM/SYRK/Cholesky/triangular-solve run on **one** stream (`stream_count = 1`, enforced by `build()` when `deterministic`); the model-space search may still use multiple *throughput* streams for work that is recomputed in FP64 before any reported number is emitted. (Setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` is the documented belt-and-suspenders for the single-stream path.)
- **cuSOLVER: deterministic mode, with its real scope.** `cusolverDnSetDeterministicMode` is enabled on the statistic handle. Per the cuSOLVER docs it covers `geqrf`, `syevd`/`syevdx`, `gesvd` (for `m > n`), **`gesvdj`/`gesvdjBatched`**, `Xgeqrf`, `Xsyevd`/`Xsyevdx`, `Xgesvd` (`m > n`), `Xgesvdr`, and `Xgesvdp`. So the Jacobi SVD used in the rank test *is* covered — but note the **`m > n` constraint on the `gesvd` family**: orient the rank-test matrix so the deterministic precondition holds, or use the covered Jacobi path. We do **not** claim blanket "deterministic cuSOLVER"; we claim the enumerated set, and CI asserts the rank-test routine is one of them.

- **Cross-GPU determinism: where parity is and is NOT achievable.** Floating-point addition is non-associative, so a multi-GPU reduction's bit pattern depends on the reduction *order*. **[UNCERTAIN, per NVIDIA NCCL maintainer guidance]** default NCCL AllReduce is *reproducible run-to-run at a fixed configuration* (fixed buffer size + rank→GPU mapping ⇒ fixed order), but it is **not** bit-identical to a single-GPU/sequential sum and **its order changes with the GPU count and buffer size** (NCCL auto-switches Ring/Tree/NVLS by a size/topology-driven threshold). Pinning `NCCL_ALGO`/`NCCL_PROTO`/a topology file makes it *stable at a fixed G* but still does not match the single-GPU sum nor survive a change in G. That is fatal for "parity to ADMIXTOOLS 2," so the parity path **keeps every parity-critical cross-GPU reduction off NCCL collective reduction entirely**: (1) the `f2_blocks` (+ `Vpair`) combine gathers the G partials to the host and sums them in **fixed device order** (`g = 0..G−1`, the `DeviceConfig::devices` order) in reference precision — provably **configuration-independent** (same arithmetic order whether G=1, 2, or 8) and essentially free because the combine is off the bandwidth critical path (regardless of `f2_blocks` size — at scale the result is tiered to host/disk per §11.1/§11.2, not the parity concern here); **NCCL is used only for the subsequent broadcast** (or a `cudaMemcpy`), which is order-independent and so parity-safe; (2) any *on-device* reduction that must match across physical GPUs uses CCCL **`gpu_to_gpu`** determinism (the reproducible-FP-accumulator path, §7) so a block's *partial* is identical regardless of which GPU computed it — this is a complement to, not a substitute for, the host-side fixed-order combine across devices. The S8 search introduces **no** cross-GPU reduction (each fit is single-GPU; results re-sorted by model index on the host), so it inherits single-GPU parity directly. **Honest boundary:** under this path steppe's results are bit-identical *across G and to the single-GPU reference*; we explicitly do **not** put a parity-critical sum on NCCL AllReduce. A `determinism.mode = "fast"` escape hatch may use default NCCL AllReduce for exploratory runs — run-to-run stable at a fixed G, but **not** parity and **not** stable across different G — and its outputs are never bit-compared to goldens.

**Block/tree reductions: two distinct properties, not one.** A pairwise/tree reduction gives an **error bound** of `O(ε log n)` versus `O(ε n)` for naive sequential or `atomicAdd` chains — that is an *accuracy* argument, and it is not "free," it is a bounded improvement in worst-case rounding error. Separately, a reduction is **deterministic** only if its *combination order is fixed*: CUB's `run_to_run` guarantees that fixed order; a hand-rolled block reduction does not unless you fix the tree shape yourself. The real reason to prefer tree reductions over long `atomicAdd` chains on the statistic path is that **`atomicAdd` accumulation order is itself the dominant nondeterminism source** we are eliminating — the accuracy bound is a secondary benefit. Treat the two arguments separately; do not assume a block reduction is automatically reproducible.

**Tolerance policy.** Never one absolute epsilon. Combined form `|a−b| ≤ atol + rtol·|b|`. Concretely: point estimates `est` ~`1e-9`–`1e-6` relative (near bit-stable with `run_to_run` + single stream + native-FP64 elementwise) — and `EmulatedFp64` covariance entries are held to this same tight tier against the native-FP64 oracle (relative error within a few ULP of FP64), since emulation under dynamic mantissa control targets native-FP64 accuracy; jackknife `se`/`z` ~`1e-3` relative; the **rank decision** (and any value whose cuSOLVER routine is outside the deterministic set, **or any TF32-screened intermediate**) sits in the loose tier and is documented as such. TF32-screened results additionally carry a precision tag and are **never** bit-compared to AT2 goldens — they are provisional shortlisting signals, promoted to `EmulatedFp64`/`Fp64` before any reported number. The `se` tolerance is **derived, not asserted**: the jackknife SE is a variance over `B` block-delete replicates, and the relative variability of a variance estimator from reduction-order and resampling differences is empirically `O(1/√B)` for the `B≈700` blocks of a typical genome — which lands near `1e-3`. The golden harness records the measured spread per fixture and we set `rtol` from the observed distribution plus margin, not from a magic constant.

**Seed control.** Use cuRAND **Philox** (counter-based) so draw `f(seed, replicate, offset)` is a pure function independent of launch geometry/scheduling — the single most important decision for reproducible GPU resampling. Each bootstrap replicate is an independent substream via `subsequence = rep`. SNP→block assignment is the deterministic pure function of genetic position from `core/domain/block_partition_rule.hpp` (the `blgsize` rule). The seed is threaded through the API and **recorded in the golden file metadata**.

```cpp
curandCreateGenerator(&g, CURAND_RNG_PSEUDO_PHILOX4_32_10);
curandSetPseudoRandomGeneratorSeed(g, cfg.seed());
curandSetGeneratorOffset(g, replicate_id * stride);          // independent, reproducible substream
```

**Validation against ADMIXTOOLS 2.** ADMIXTOOLS 2 has **no `seed` argument** — its resampling rides R's RNG state — so the "bit-reproducible regeneration" of goldens is **fragile across versions** and we treat it that way. R's stream depends on the **R version, the `RNGkind`, and AT2's internal call order**, so pinning `set.seed(N)` alone is insufficient. The golden metadata therefore records **R version, `RNGkind`, ADMIXTOOLS 2 version, `blgsize` (=0.05 Morgans), and `boot` setting** (`FALSE` for delete-1 jackknife, or integer `N`), alongside the seed and the dumped `qpdstat` `est/se/z/p`. Regeneration must reproduce that exact environment; cross-version drift is expected and is why `se`/`z` use the loose tier above. The diff uses the two-tier tolerance.

---

## 13. Testing strategy

The CPU reference backend underpins everything: the whole pipeline is exercisable GPU-free, and every numerically subtle GPU kernel is diffed against an obviously-correct scalar host loop.

- **Unit tests (GoogleTest, fast).** Pure `__host__ __device__` numerics (f2 estimator, f4 contraction, jackknife weighting, `block_of`) tested in plain `.cpp` on the host — no GPU. Thin kernels get a launch-and-compare test. A `CudaTest` fixture resets the device per suite and asserts a clean error state in `TearDown` so a leaked sticky error can't cascade.
- **Golden / reference-equivalence (`tests/reference/`).** Identical inputs through `CpuBackend` and `CudaBackend`, asserted equal within tolerance — the central trust seam at `f2_blocks`, `Q`, `X`, `w`, χ².
- **Regression (`tests/golden/`).** ADMIXTOOLS 2 `est/se/z/p` on small committed fixtures with the **full pinned environment** (R version, `RNGkind`, AT2 version, `blgsize`, `boot`, seed — §12). Tight `rtol` on `est`, loose on `se`/`z`/rank.
- **Property tests.** Identities that hold regardless of values: `f4(A,B;C,D) == ½(f2(A,D)+f2(B,C)−f2(A,C)−f2(B,D))`, `f3(A;B,C) == f4(A,B;A,C)`, `Q` symmetric PSD, antisymmetry of f4 under index swaps.
- **Domain-outcome tests.** Construct a deliberately collinear/rank-deficient model and assert the API returns `STEPPE_ERR_RANK_DEFICIENT`/`STEPPE_ERR_NON_SPD_COVARIANCE` as a *value*, not a crash (§10).
- **compute-sanitizer in CI.** All four tools against the unit-test binary (built `-lineinfo`) with `--error-exitcode 1`: `memcheck --leak-check full`, `racecheck --racecheck-report analysis`, `initcheck --track-unused-memory`, `synccheck`. Add `--track-stream-ordered-races all` because we use `cudaMallocAsync` (§7). Scope to a fast subset per-PR (sanitizers are 10×+ slower); full sweep nightly. Commit a vetted suppressions file for unavoidable driver/cuBLAS-internal noise — never blanket-disable a tool.

---

## 14. CI/CD & tooling

**Pipeline order.**
1. **PR, no GPU (`ubuntu-latest`):** `pre-commit run --all-files` (clang-format/clang-tidy/IWYU), CPU-side unit tests, build matrix with sccache. CI asserts `CCCL_VERSION >= 3.1` (§3).
2. **PR, GPU runner:** reference-equivalence + ADMIXTOOLS 2 regression goldens (`run_to_run`, single statistic stream, Philox seeds ⇒ bit-stable; tight `rtol` on `est`, loose on `se`/rank).
3. **PR, GPU, scoped:** compute-sanitizer memcheck + racecheck on a fast subset, `--error-exitcode 1`.
4. **Nightly, GPU:** full four-tool sanitizer sweep; nvbench with clocks locked vs committed baseline (10% regression gate); `nsys --stats` smoke check.
5. **On demand:** Nsight Systems→Compute deep dive; build/publish wheels.

**GPU runner matrix (CUDA × arch).** GitHub-hosted runners have no GPU → self-hosted, labeled by arch so perf gates pin one GPU model (cross-arch numbers aren't comparable):

```yaml
strategy:
  fail-fast: false
  matrix:
    cuda: ["13.1", "13.3"]          # 13.1+ for CCCL 3.1 determinism API
    include:
      - { arch: "80", runner_label: sm80 }   # Ampere
      - { arch: "90", runner_label: sm90 }   # Hopper
runs-on: [self-hosted, linux, gpu, "${{ matrix.runner_label }}"]
container: { image: "nvidia/cuda:${{ matrix.cuda }}.0-devel-ubuntu24.04", options: --gpus all }
```

`fail-fast: false` so one combo failing doesn't cancel the rest; an `sm_90` build's tests run on a Hopper card. sccache uses the GHA backend (`SCCACHE_GHA_ENABLED=true`); device-object hit rate is informational (§6).

**Gates.** Format/tidy blocking (need `compile_commands.json`); IWYU starts non-blocking, promoted once clean, scoped to first-party targets. compute-sanitizer per §13.

**Pre-commit:** `mirrors-clang-format` (`types_or: [c++, c, cuda]`) and `cmake-pre-commit-hooks` clang-tidy (configures CMake first to generate the compile DB). Lint runs CPU-only.

**Wheels — CUDA-13 redistributable model (verified).** `cibuildwheel` over scikit-build-core. The manylinux image lacks CUDA → build inside `nvidia/cuda:*-devel` via `CIBW_MANYLINUX_X86_64_IMAGE`. **Do not bundle CUDA runtime libs**; exclude them in `auditwheel repair` and declare a runtime dependency on NVIDIA's pip packages so a clean environment can actually import the wheel:

```bash
auditwheel repair --exclude libcudart.so.13 --exclude libcublas.so.13 \
                  --exclude libcusolver.so.13 --exclude libcurand.so.13 wheel.whl
```

> **Two CUDA-13 specifics that the exclude alone does not handle:**
> 1. **SONAME major coupling.** `libcublas.so.13` hardcodes SONAME major `13`; the exclude list and the declared deps must move together when the CUDA major bumps. Encode the CUDA major in the wheel tag (multi-wheel-per-CUDA-major).
> 2. **The dependency must be declared, or the wheel is broken at import.** A wheel that excludes `libcublas.so.13` but declares nothing providing it fails on `import`. CUDA 13 also changed the pip package naming: the suffixed `nvidia-*-cu13` packages are being **deprecated in favor of unsuffixed `nvidia-cuda-runtime` / `nvidia-cublas` / `nvidia-cusolver` / `nvidia-curand`**, and CUDA-13 wheels consolidate headers/libs under `site-packages/nvidia/cu13/`. Pin the **unsuffixed** runtime packages in `pyproject.toml` `[project.dependencies]` (§15) and verify import in a clean venv in CI. Never let `-cu12` and `-cu13`/unsuffixed-cu13 coexist.

---

## 15. Python bindings & packaging

**nanobind over pybind11** (ADR-0002): substantially faster compiles and smaller binaries, lower per-instance and call overhead; runtime is effectively identical for the large-NumPy-array workloads we have, so we pay nothing at runtime and win on build/size. `nanobind_add_module()` handles linking. The binding (`bindings/module.cpp`) is **thin** — it converts NumPy arrays to/from `span_view`s and forwards into the public API; no compute logic, keeping C++ and Python on one implementation.

**scikit-build-core** is the PEP 517 backend driving the same CMake. Because the backend is injected and `CpuBackend` always exists, `import steppe` and smoke tests pass on GPU-free CI runners and laptops; `pytest` asserts GPU-vs-CPU equivalence wherever a device is present.

```toml
# pyproject.toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2"]
build-backend = "scikit_build_core.build"

[project]
name = "steppe"
dynamic = ["version"]
requires-python = ">=3.10"
# Runtime CUDA libs are NOT bundled (auditwheel --exclude, §14); declare them here so a
# clean environment can import. Unsuffixed CUDA-13 packages per the 13.x redistributable model.
dependencies = [
  "numpy>=1.23",
  "nvidia-cuda-runtime>=13,<14",
  "nvidia-cublas>=13,<14",
  "nvidia-cusolver>=13,<14",
  "nvidia-curand>=13,<14",
]

[tool.scikit-build]
cmake.version = ">=3.30"
build.targets = ["steppe._core"]
wheel.packages = ["bindings/steppe"]
[tool.scikit-build.cmake.define]
STEPPE_BUILD_PYTHON = "ON"
STEPPE_BUILD_CLI = "OFF"
```

> The CUDA runtime deps apply to GPU-enabled wheels; a CPU-only wheel variant omits them (the loader resolves the CUDA libs lazily and `CpuBackend` needs none). Pin/verify the exact package names against PyPI at release time — NVIDIA is mid-transition from `-cu13`-suffixed to unsuffixed names (§14).

---

## 16. Documentation & ABI/versioning

**ADRs** in `docs/adr/`, Michael Nygard format (*Title, Status, Context, Decision, Consequences*), one decision per immutable record; a reversal gets a new superseding ADR. Initial set: 0000 (record ADRs), 0001 (layered architecture, compiler-enforced), 0002 (nanobind over pybind11), 0003 (CPU reference backend as correctness anchor — an *oracle*, not a structural template, §2), 0004 (precision policy: Ozaki emulated-FP64 default for the well-conditioned matmul-heavy path, native-FP64 oracle AND f2 reduction, TF32 screening; single-stream determinism), 0005 (precompute-once/fit-many split & the `f2_blocks` seam), 0006 (Philox + recorded-environment reproducibility), 0007 (out-of-core SNP tiling), 0008 (**C ABI at the public boundary** — see below), 0009 (**single-node multi-GPU, day one**: SPMG + per-device streams + NCCL; SNP-tile sharding for S0–S2 and model-space sharding for S8; parity via fixed host-order `f2_blocks` combine + broadcast + CCCL `gpu_to_gpu`, *not* NCCL AllReduce; multi-node and cuSOLVERMp deferred — §11.4, §12), 0010 (**GEMM-reformulated f-statistics + fused feeders**: native cuBLAS, no array framework; the 3-GEMM bias-corrected f2 with `Vpair` carried as the jackknife weight; the f2 GEMMs use FIXED-slice Ozaki (~40-bit mantissa; measured 8–17× over native FP64 at native-grade accuracy on real AADR), dynamic-mantissa rejected as the parity trap, native FP64 the oracle/fallback — §2, §5 S2, §12), 0011 (**genotype QC / preprocessing in scope**: in-memory merge plan / on-the-fly filter / pairwise-complete default with optional imputation; no strand inference, no self-computed LD; additive `io`-leaf S−2/S−1/S0′ stages, no on-disk rewrite — §1, §5).

**ABI: a true C boundary, not `std::expected` across the line.** The earlier claim that the public ABI "uses `std::expected` returns" while being ABI-stable was self-contradictory: `std::expected<T,Error>` is a `std::` type with library-defined layout, and if `T`/`Error` hold `std::string`/`std::vector` you reintroduce exactly the libstdc++/libc++/MSVC coupling you meant to avoid. We resolve it by picking one model explicitly:

- **The installed/versioned boundary is a C ABI.** `include/steppe/` exposes **opaque handles** (`steppe_f2_blocks_t*`, `steppe_qpadm_result_t*`), functions returning **`steppe_status_t`** (the §10 enum), and accessor functions for results. **No `std::` types, no templates, and no exceptions cross this boundary.** This is what makes `find_package(steppe)` + a prebuilt library usable across toolchains and what the SemVer/ABI promise actually covers.
- **`std::expected`/`STEPPE_TRY` are internal only.** They live in `core` and `src/`; the C API is a thin shim that converts `std::expected<T,Error>` to `steppe_status_t` + out-params at the boundary.
- A header-only, same-toolchain C++ convenience layer *may* wrap the C ABI with `std::expected`/RAII for in-tree and Python-binding use, but it is **not** the ABI contract and carries no cross-toolchain promise. The binding (§15) links the C++ layer in-process, so it is unaffected.

**API docs** via Doxygen + Sphinx/Breathe from `include/steppe/`. **SemVer + stability:** the contract is exactly the C surface in `include/steppe/`; nothing in `src/` is stable. Additive C API ⇒ MINOR, breaking ⇒ MAJOR; the CUDA major version is encoded in the wheel tag and in the SONAME the wheel depends on (§14). `CHANGELOG.md` in Keep-a-Changelog; `version.hpp` generated from `project(VERSION ...)` so macro and wheel version never drift. Installed `SteppeConfig.cmake` exports `steppe::api`/`steppe::core`. This living document is mirrored at `docs/architecture.md`.

---

## 17. Scaffold manifest (ordered)

An AI agent or engineer can execute this top-to-bottom to reach a building, testable skeleton.

1. **Repo + tooling roots:** `git init`; `.gitignore`, `.editorconfig`, `.clang-format`, `.clang-tidy`, `.pre-commit-config.yaml`, `LICENSE`, `README.md`, `CHANGELOG.md`.
2. **CMake policy:** top-level `CMakeLists.txt` (§6) + `cmake/{SteppeOptions,CUDAArch,CompilerLauncher,SteppeWarnings,SteppeSanitizers,CPM}.cmake`. Define `steppe::warnings`; wire device-LTO and `--compress-mode` opt-ins (§6).
3. **Presets:** `CMakePresets.json` (`dev/debug/release/cuda-debug/asan/ci`).
4. **Dependency pins:** `third_party/CMakeLists.txt` with **CCCL ≥ 3.1 (toolkit-first, 3.1 floor enforced, CPM fallback)**, **NCCL (from toolkit, single-node multi-GPU broadcast, §3/§11.4)**, Eigen, spdlog, CLI11, GoogleTest, nvbench, nanobind.
5. **Public C-ABI headers (`include/steppe/`):** `version.hpp.in`, `error.hpp` (`steppe_status_t` taxonomy, §10), `config.hpp` (§9), `io/genotype.hpp`, `fstats.hpp`, `qpadm.hpp` (opaque handles + status returns), `steppe.hpp`. Add `include/CMakeLists.txt` defining `steppe_api` INTERFACE.
6. **DRY internals (`src/core/internal/` + `src/core/domain/`):** `host_device.hpp` (`STEPPE_HD`, `STEPPE_DEBUG_ONLY`, `STEPPE_ASSERT`), `expected.hpp` (`STEPPE_TRY`, internal-only), `log.hpp/.cpp` (`STEPPE_LOG_*` incl. the teardown-warning sink), `nvtx.hpp` (with the one color palette), `type_traits.hpp` (`real_t<P>`), `launch_config.hpp`, `span_view.hpp`, and `domain/block_partition_rule.hpp` (`block_of` + `assign_blocks` + the inverse `block_ranges`). Define `steppe::core_internal` INTERFACE.
7. **Device RAII + checks (`src/device/cuda/`):** `check.cuh` (`STEPPE_CUDA_CHECK`, `STEPPE_CUDA_CHECK_KERNEL`, `CUBLAS_CHECK`, `CUSOLVER_CHECK`), `device_buffer.cuh`, `pinned_buffer.cuh`, `stream.hpp`, `handles.hpp` (full move-assign + debug teardown logging, §7), `allocator.hpp` (pool). These are the **allowlisted** allocation TUs (§2).
8. **Backend seam:** `src/device/backend.hpp` (`ComputeBackend`, CUDA-free). Stub `cpu/cpu_backend.cpp` (scalar) and `cuda/cuda_backend.cu` first as compiling no-ops, plus `cuda/multi_gpu.cu/.hpp` (device enumeration, per-device streams/handles, NCCL comm setup, SNP-tile sharding, fixed-order host-side `f2_blocks` combine + broadcast — §11.4) and `src/device/CMakeLists.txt` wiring CUDA `PRIVATE` + `CCCL::CCCL` + **`CUDA::nccl`** (§8).
9. **First real kernel + reference:** f2 estimator as a `__host__ __device__` pure function (`core/fstats/`), the `f2_block_kernel.cu` shell, and the matching CPU-backend path. Minimum slice proving the architecture and the reference seam.
10. **IO leaf (`src/io/`):** `reader.hpp` (streams harmonized/filtered tiles + `V` + `N`), `plink_bed.cpp` (magic-byte + 2-bit decode, polarity-aware), `merge/{merge_plan,allele_harmonize}.cpp` (in-memory merge plan, no strand inference, §5 S−2), `filter/{snp_filter,sample_filter,prepass}.cpp` (on-the-fly + `--mind`/external-`prune.in` pre-pass, §5 S−1/S0′), `impute/missing_policy.cpp` (pairwise-complete default / opt-in tagged mean-impute), `genetic_positions.cpp` (positions only — block rule lives in `core`), `precomputed_f2.cpp`; isolated `CMakeLists.txt` (no dependency on `device`).
11. **Core compute + CLI:** `core/fstats/{f2_from_blocks,f4_matrix,jackknife}.cpp`, `core/qpadm/{ranktest,gls_solve,nested_models}.cpp` (all orchestration-only, device via `ComputeBackend`); `src/app/{main,cli_args}.cpp` + `commands/` (incl. `extract-f2 --dry-run` budget print, §11.1).
12. **Tests:** `tests/CMakeLists.txt` with `gtest_discover_tests`; `tests/unit/test_f2.cpp` (host), `tests/reference/test_f2_equivalence.cu` (GPU-vs-CPU), a domain-outcome test (§13); one tiny fixture and one ADMIXTOOLS 2 golden CSV with full env metadata (§12).
13. **Bindings + packaging:** `bindings/{CMakeLists.txt,module.cpp,steppe/__init__.py}`, `pyproject.toml` with CUDA runtime deps (§15).
14. **Benchmarks:** `benchmarks/bench_f2_kernel.cu` (nvbench) with a throughput axis.
15. **CI:** `.github/workflows/` — lint (CPU, asserts CCCL≥3.1), build-test matrix (GPU, sccache), sanitizer (scoped), nightly (full sanitizer + nvbench gate + wheels with clean-venv import check).
16. **ADRs:** seed `docs/adr/0000`–`0011` (§16); copy this document to `docs/architecture.md`.

---

## 18. Definition of Done / quality bar

A change is Done when **all** hold:

- **Builds clean** on the matrix (CUDA 13.1+ × {sm_80, sm_90}) with `-Werror`/`--Werror all-warnings`; no new clang-tidy/IWYU diagnostics; CI confirms CCCL ≥ 3.1.
- **Layering intact:** no upward dependency; CUDA headers compile only into `steppe_device`; `core` reaches the GPU only via `ComputeBackend`; the architecture grep test and the allocation-allowlist grep both pass.
- **RAII complete:** no allocation from the `cudaMalloc*`/`cudaHostAlloc`/`cudaFree*` family outside the allowlisted wrappers; all owning types fully move-only (construct **and** assign); destructors log-on-error in debug, never throw; no new global mutable state.
- **Correctness anchored:** every new statistic-bearing kernel has a `CpuBackend` reference path and a reference-equivalence test; affected ADMIXTOOLS 2 goldens pass within the tier tolerance (`est` tight; `se`/`z`/rank loose, with the loose tolerance derived from observed spread, not a magic number).
- **Numerics policy honored:** native FP64 on the small cancellation-sensitive numerator/divide step; **fixed-slice Ozaki (default 40-bit mantissa) on the matmul-heavy path including the S2 f2 GEMMs** (measured 8–17× over native at native-grade accuracy on real AADR; dynamic-mantissa rejected as the parity trap), gated against the native-FP64 oracle with an explicit `cublasSetWorkspace` workspace under `deterministic`; `run_to_run` reductions via the single-phase API; **statistic-bearing cuBLAS confined to one stream**; cuSOLVER deterministic mode with the routine confirmed in-scope; Philox seeds threaded and recorded with full R/AT2 environment; no TF32/FP16 in any reported value without re-validation.
- **GEMM-reformulation reference-equivalent:** any statistic implemented as dense tensor ops (the S2 3-GEMM f2, §5) is diffed against the slow native-FP64 `CpuBackend` oracle that walks the exact AT2 pairwise-complete path — agreement at the `f2_blocks` seam within the tight tier, including the `Vpair==0` branch, the `q(1−q)/max(N−1,1)` denominator + allele-count `N` convention, and that the per-block `Vpair` carried to the S4 jackknife weight composes to AT2's `f2_blocks` definition (no double-normalization); the shared `__host__ __device__` feeder primitive is the single source of the per-element formula. No array framework (no JAX/CuPy) introduced.
- **Multi-GPU parity:** on a single-node multi-GPU box, results are **bit-identical across the number of devices in `DeviceConfig::devices` and to the single-GPU reference** — the `f2_blocks` combine is a fixed host-order sum (not NCCL AllReduce) followed by an order-independent broadcast, on-device partials use CCCL `gpu_to_gpu`, and S8 results are re-sorted by model index; a 1-GPU-vs-G-GPU equivalence test passes within the tight tier (§11.4, §12).
- **Preprocessing in scope & layering-legal:** merge/filter/impute live in the `io` leaf as plain-data producers reading only `core::block_partition_rule` downward and **not** depending on `device`/`ComputeBackend` (decode is wired into compute by `app`); merge is an in-memory plan (no on-disk rewrite, no strand inference, no self-computed LD); the pairwise-complete default reproduces AT2 and the SNP→block map is unchanged from the single-dataset path; imputation outputs are tagged and never golden-compared.
- **Memory budgeted:** new resident allocations accounted in the §11.2 budget; `build()` still rejects over-VRAM configs; out-of-core paths keep the genotype matrix tiled (no whole-matrix load).
- **Errors classified:** new failure modes map to a `steppe_status_t` category; domain outcomes (non-SPD, rank-deficient, χ²-undefined) are returned as values and covered by a test, not surfaced as faults.
- **Sanitizers green:** compute-sanitizer memcheck + racecheck clean on the relevant tests (`--error-exitcode 1`); new async-alloc paths covered by `--track-stream-ordered-races all`.
- **Observable & documented:** new pipeline phases carry NVTX ranges; public C-API/behavior changes update headers, Doxygen, `CHANGELOG.md`, and an ADR if a load-bearing decision changed; the ABI stays a C boundary (no `std::` across it).
- **No perf regression** beyond the 10% nvbench budget on the affected kernel (clocks locked, GPU model pinned); a speed change to a kernel carries an Nsight Compute roofline/occupancy note.
- **Python parity:** `import steppe` and CPU smoke tests pass GPU-free; the GPU wheel imports in a clean venv resolving CUDA runtime via the declared (unsuffixed CUDA-13) deps, without bundling CUDA libs.

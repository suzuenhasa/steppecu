# CUDA Engineering Assessment

## Scope

A first-principles review of the GPU layer in `steppe`, concentrated on
`src/device/cuda/*.cu` / `src/device/cuda/*.cuh`, plus the supporting host-side
abstractions in `src/device/` and `src/core/internal/launch_config.hpp` and the
architecture contract in `docs/architecture.md` §7, §11, §12.

The codebase is a population-genetics compute backend whose hot path is a
dense f2/GEMM algebra coupled to streaming genotype decode, jackknife
resampling, qpAdm/qpGraph optimization, and DATES date-estimation. The CUDA
layer is therefore both a heavy cuBLAS consumer and a producer of custom
kernels. The assessment below evaluates whether the implementation is correct,
performant, maintainable, and reproducible at scale.

## Summary Grade

**B / B+**. The foundations are solid: RAII handles, one error-check macro, a
single non-blocking stream, grid-stride patterns in the right places, explicit
FP64/emulation policy seams, and a clear multi-tier out-of-core design. The
major issues are not crashes but **performance and determinism gaps**: a single
stream leaves overlapping opportunities on the table, several kernels are
memory-bound in avoidable ways, peer combine is not all-reduced, small qpAdm
kernels are serial, and the documented cuSOLVER determinism mode is not wired
in. These are fixable and are enumerated in the roadmap at the end.

---

## 1. Kernel design

### 1.1 Grid-stride loops and launch bounds

The project has a dedicated, well-factored launch-math header at
`src/core/internal/launch_config.hpp` (lines 17-73). It exposes:

- `cdiv` overloads for `int` and `long long`;
- `grid_for` helpers that clamp `x`/`y` extents against `kMaxGridX`/`kMaxGridY`;
- `grid_z_extent` that clamps `z` to `kMaxGridZ = 65535`.

Most custom kernels use grid-stride loops when iterating over an arbitrary
number of independent work items:

- `ratio_block_jackknife` and `qpfstats_jackknife` in
  `src/device/cuda/cuda_backend.cu` (lines ~3010-3140) stride over blocks of
  statistics and recompute global sums per iteration.
- `qpgraph_fused_log_likelihood`, `qpgraph_weighted_lr`, and the qpGraph
  optimizers (lines ~4890-5310) stride over restarts / parameter sets and
  dispatch serial work inside each thread.
- The small qpAdm helper kernels `qpadm_fit_models_kernel` and
  `qpadm_loo_models_kernel` in `src/device/cuda/qpadm_fit_kernels.cu`
  (lines ~1569-1698) launch one thread per model and perform serial linear
  algebra in registers/shared memory. The launch wrappers are
  `launch_qpadm_fit_models_batched` / `launch_qpadm_loo_models_batched`
  (lines ~1861-1887).

`__launch_bounds__` is applied where it matters:

- `decode_af_kernel` declares `__launch_bounds__(kDecodeBlockX * kDecodeBlockY)`
  in `src/device/cuda/decode_af_kernel.cu` (line ~99).
- The grouped f2 feeder `gather_group_kernel` in
  `src/device/cuda/f2_blocks_kernel.cu` (lines ~73, ~155) uses
  `__launch_bounds__(kCdivBlock * kCdivBlock)` (256 threads).

### 1.2 Block sizing

Block shapes are chosen with the data layout in mind:

- **Decode**: `dim3(32, 8)` (`src/core/internal/launch_config.hpp:38`); the
  SNP-major axis is `x`, so consecutive threads read contiguous 32-bit packed
  genotype words (`src/device/cuda/decode_af_kernel.cu`, lines ~110-180). The kernel then
  writes allele frequencies into a transposed shared-memory tile so the output
  store back to global memory is coalesced.
- **f2 assemble**: `dim3(16, 16)` matching `kCdivBlock`
  (`src/core/internal/launch_config.hpp:69`) for a 2-D tile over the `[P × P]`
  f2 output.
- **D-stat**: `dstat_kernel` uses a 2-D block of `16 × 16` and a 32KB
  shared-memory tile (`src/device/cuda/dstat.cu`, lines ~80-120). The tile is
  sized explicitly to leave enough registers/shared for two concurrent blocks
  per SM.
- **DATES**: 1-D `256`-thread blocks (`src/device/cuda/dates.cu`, lines ~60,
  ~160), which is a conservative occupancy choice for short-range correlation
  kernels.

### 1.3 Occupancy and grid sizing concerns

- **qpadm small path**: One CUDA thread per model is correct for model-level
  parallelism, but the serial LA inside each thread means small models never
  exploit warp execution. For very small `M` (number of sources) this is fine,
  but at the boundary between the "small" and "large" paths the device may be
  dramatically under-utilized.
- **qpGraph fleet**: Each thread executes a full L-BFGS-like optimization.
  This is a classic "embarrassingly parallel across restarts" pattern, but the
  kernels allocate scratch in global memory per thread (`theta_dev[16]`
  registers plus device arrays indexed by `restart_idx`). If the optimizer
  requires more registers than the launch bounds permit, spilling will be
  severe. There is no dynamic-parallelism or CG/OMP offloading alternative.
- **f2 grouped kernel**: `f2_blocks_kernel.cu` groups many SNP blocks into a
  single strided-batched GEMM. This is the right shape for cuBLAS, but the
  grouped feeder kernel that packs the input data is not grid-strided in `z`
  (it relies on `blockIdx.z` being within `kMaxGridZ`).
  `src/device/vram_budget.hpp` correctly clamps chunk size to `kMaxGridZ`
  (`grid_z_extent`), so this is safe but not robust to future changes in grid
  limits.
- **Decode duplication**: Two decode front-ends exist in
  `src/device/cuda/cuda_backend.cu` (the legacy single-block path and the new
  grouped path). They repeat block-size, pitch, and transpose logic. This is a
  maintainability hazard noted in the conventions review and worth fixing here
  because it directly affects kernel launch tuning.

### 1.4 Memory-bound kernels

Several kernels could benefit from explicit shared memory and vectorized loads:

- `ratio_block_jackknife` and `qpfstats_jackknife` read the same f2 matrices
  multiple times (once per jackknife block). With `B` blocks this is an
  `O(B)` memory amplification over the compute. A chunked reduction in shared
  memory, or reusing partial sums across consecutive blocks, would cut device
  memory traffic significantly.
- The qpGraph gradient kernels compute `f2_model - f2_data` per restart; this
  read is strided across restarts rather than coalesced within a warp.
  Restructuring so restarts are the leading dimension for the data array
  would improve coalescing.

---

## 2. Memory management

### 2.1 RAII ownership

The memory layer is one of the strongest parts of the project.

- `DeviceBuffer<T>` (`src/device/cuda/device_buffer.cuh`) is move-only,
  overflow-guards its byte count in construction, and owns a single device
  pointer. It is deliberately device-agnostic (no stored device ordinal), with
  a comment noting that `cudaMallocAsync` would re-impose
  record/restore semantics (line ~40).
- `PinnedBuffer<T>` (`src/device/cuda/pinned_buffer.cuh`) and
  `RegisteredHostRegion` provide host-pinned ownership and CUDA host
  registration caches.
- `Stream` and `Event` (`src/device/cuda/handles.hpp`, lines ~220-310) are
  move-only and non-copyable.

### 2.2 Allocator strategy

- `CudaBackend` pre-allocates a 64 MiB cuBLAS workspace once
  (`src/device/cuda/cuda_backend.cu:5596` and `:277`), sized by
  `steppe::kCublasWorkspaceBytes` (`include/steppe/config.hpp:118`).
- The backend also maintains an L4b "slab" of reusable `DeviceBuffer`s for
  scratch tensors (lines ~550-650). Slab reuse avoids `cudaMalloc`/`cudaFree`
  churn on the hot path.
- For streamed out-of-core work, the resident ring uses only
  `kStreamDeviceChunks = 2` chunks (`include/steppe/config.hpp:177`), which is
  the minimal double-buffer needed to overlap compute with transfer.

### 2.3 Pinned memory and staging

- `BlockSink` (`src/device/cuda/block_sink.cuh` / `.cu`) implements a
  triple-buffered pinned staging ring plus a background writer thread. The
  shared `StagingRing` template is reused for `HostRamSink` and `DiskSink`.
- `PinnedRegistryCache` amortizes `cudaHostRegister` costs when caller memory
  is reused across calls.

### 2.4 Concerns

- **No pool allocator**: All device allocations go through `cudaMalloc`. The
  L4b slab mitigates churn for fixed scratch, but per-call temporaries still
  hit the driver allocator. Moving to `cudaMallocAsync` from a memory pool is
  explicitly noted as future work; doing so would require storing/restoring
  the current device in `DeviceBuffer` because pools are per-device.
- **Pinned cache does not track aliasing**: `pinned_in_` caches host
  registrations of caller-provided pointers. If the caller frees or reallocates
  that memory while the cache still holds a registration, subsequent
  `cudaHostUnregister` is undefined. A generation counter or size check should
  be added.
- **Monotonic staging growth**: The pinned staging areas for f2/vpair outputs
  (`stage_f2_`, `stage_vpair_`) are sized to the maximum encountered problem
  and never shrink. For a long-lived backend object serving mixed problem
  sizes, this can pin many gigabytes indefinitely.
- **VRAM budget math is host-side and conservative**: `vram_budget.hpp`
  computes `resident_tensor_bytes()` from stack counts (`kChunkInputStacks`,
  `kChunkOutputStacks`) and applies `kMaxVramUtilizationFraction = 0.80`
  (`include/steppe/config.hpp:206`). This is good, but it does not query
  actual free memory at runtime, so transient fragmentation from other
  processes or cuBLAS workspace growth can still cause OOM.

---

## 3. Stream usage and async patterns

### 3.1 Single-stream discipline

Every `CudaBackend` owns one non-blocking stream (`stream_`), declared in
`src/device/cuda/cuda_backend.cu:5585` via the `Stream` RAII wrapper. cuBLAS,
cuSOLVER, cuFFT, and all custom kernels are launched on this stream. The
constructor binds it once:

```cpp
blas_.set_stream(stream_.get());
solver_.set_stream(stream_.get());
```

This is the correct default for a deterministic statistics pipeline: one
stream means implicit serialization and no subtle race bugs.

### 3.2 Async transfers

- H2D/D2H copies are asynchronous on `stream_` and paired with `Event`
  objects where needed (`BlockSink` uses per-slot events).
- The out-of-core path (`streamed_tier_*`) overlaps chunk upload, compute,
  and result download using the two device chunks and pinned staging ring.

### 3.3 Concerns

- **Everything is serialized on one stream**. f2 GEMMs, decode kernels,
  cuSOLVER Cholesky, DATES FFT, and jackknife reductions cannot overlap even
  when they have no data dependency. For throughput-bound workloads, adding a
  second compute stream (e.g., one for GEMM/decode and one for reductions) or
  per-model streams for qpAdm/qpGraph would improve utilization with modest
  complexity.
- **cuFFT stream binding**: `dates.cu` creates FFT plans bound to `stream_`.
  Plan creation is synchronous and not cheap; creating/destroying plans per
  call would be a bottleneck. The current code appears to cache plans, but
  verify that the plan lifetime matches the backend lifetime.
- **No CUDA graphs**: The decode → f2 GEMM → reduction pipeline is very
  regular across chunks and is an obvious graph candidate. Architecture
  §11/§12 mention graphs as optional; none are implemented yet.

---

## 4. cuBLAS / cuSOLVER integration and workspace handling

### 4.1 cuBLAS

`CublasHandle` (`src/device/cuda/handles.hpp`, lines ~40-210) is a strong
wrapper:

- Records the CUDA device ordinal on construction and asserts it on every
  operation.
- Binds workspace with `cublasSetWorkspace` once.
- Re-applies workspace after `set_stream()` because `cublasSetStream` can
  invalidate the workspace binding (line ~160). This is exactly the kind of
  subtle cuBLAS behavior that is easy to miss.
- `MathModeScope` captures and restores `cublasMath_t`, so scoped FP64/Ozaki
  changes do not leak.

### 4.2 cuSOLVER

`CusolverDnHandle` (`src/device/cuda/handles.hpp`, lines ~560-720) similarly
wraps the dense solver handle. It provides `CusolverMathModeScope` for solver
math mode. The backend uses:

- `cusolverDnDpotrf` / `cusolverDnDpotri` for qpAdm Cholesky/inverse
  (`cuda_backend.cu:2097`, `:2362`, `:4042`, `:4055`).
- `cusolverDnDgesvdj` for the rank-1 SVD path (`cuda_backend.cu:4207`,
  `:4256`, `:4286`).
- `cusolverDnDpotrfBatched` for batched small Cholesky (`cuda_backend.cu:5167`).

### 4.3 Concerns

- **Determinism mode mismatch with architecture doc**: `docs/architecture.md`
  §12 states that `cusolverDnSetDeterministicMode` is used for
  reproducibility. The actual code does **not** call it; only
  `cusolverDnSetMathMode` is wrapped. This is a documentation/code mismatch
  and a real determinism gap for QR/eigen-solver paths.
- **Per-call workspace allocation for cuSOLVER**: Each `cusolverDnDpotrf` and
  `cusolverDnDgesvdj` call queries `bufferSize`, allocates a temporary
  `DeviceBuffer`, and frees it. For qpGraph and qpAdm this can happen hundreds
  of times per run. A pooled or pre-sized solver workspace (similar to the
  cuBLAS workspace) would reduce allocator pressure.
- **cuSOLVER batched path**: The small qpAdm Cholesky uses batched cuSOLVER
  with one matrix per model. This is correct, but if models vary in size the
  batch is homogeneous; heterogeneous sizes fall back to the serial or
  large-model paths.

---

## 5. Precision / emulation policy implementation

### 5.1 Policy seams

The precision design is explicit and centralized:

- `emulation_honorable()` (`f2_block_kernel.cu`, lines ~60-90) is the single
  predicate that decides whether Ozaki fixed-slice FP64 emulation is used or
  native FP64 is requested.
- `engage_f2_precision()` / `f2_compute_type()` map the policy to cuBLAS math
  mode and compute type (`f2_block_kernel.cu` / `f2_blocks_kernel.cu`).
- `MathModeScope` ensures the handle is restored after each GEMM.

### 5.2 Native FP64 carve-outs

The architecture correctly identifies that **elementwise assembly of f2
matrices is cancellation-sensitive and must stay native FP64**. The code
honors this:

- `assemble_f2_kernel` and `assemble_vpair_kernel` use `double` arithmetic.
- qpAdm Cholesky and inverse call cuSOLVER in native FP64 mode (the
  `CusolverMathModeScope` is constructed with `honorable = false`).

### 5.3 Concerns

- **Emulation predicate is a single global switch**: There is no per-call
  knob to force native FP64 for numerical debugging. A debug mode that
  bypasses `emulation_honorable()` would make it easier to attribute
  bit-level differences.
- **No runtime check that Ozaki slices are sufficient**: The fixed-slice count
  is derived from `mantissa_bits` (`include/steppe/config.hpp:44` defaults to
  40). If problem conditioning requires more slices, the failure mode is
  silent accuracy loss rather than an explicit fallback.

---

## 6. CUDA error checking and debuggability

### 6.1 Check macro design

`src/device/cuda/check.cuh` is the single source of truth for error handling:

- `STEPPE_CUDA_CHECK` throws `CudaError` with `std::source_location`
  (lines ~60-90).
- `STEPPE_CUDA_WARN` is non-throwing and used for capability probes
  (lines ~110-130).
- `STEPPE_CUDA_CHECK_KERNEL()` inserts a post-launch check and debug-only
  synchronization (lines ~150-180).

This is a clean, one-stop shop and makes it easy to enable synchronous
launch/debug modes.

### 6.2 Device-guard assertions

Every public entry point calls `guard_device()` (or equivalent) and the
handle wrappers assert the current device matches the handle's device. This
makes multi-GPU misuse fail loudly rather than corrupt silently.

### 6.3 Concerns

- **Kernel launch errors are checked, but not all kernels use the macro**: In
  performance-critical loops some kernels are launched with bare `<<< >>>`
  followed by a later `STEPPE_CUDA_CHECK_KERNEL()`. This is acceptable for
  non-debug builds, but adding a `__CUDA_ARCH__`-free launch wrapper would
  make it impossible to forget.
- **No NVTX ranges**: The backend does not emit NVTX markers. For a backend
  with many phases (decode, f2, qpfstats, qpAdm, qpGraph, DATES), NVTX would
  make Nsight Systems/Compute profiling dramatically easier.
- **No persistent CUPTI/perf counter integration**: Profiling is left to
  external tools. A lightweight, compile-time optional metrics hook would help
  validate occupancy and memory bandwidth claims.

---

## 7. Custom kernel quality vs. library usage

### 7.1 Good library choices

- f2 GEMM → cuBLAS (strided-batched `cublasDgemmStridedBatched` / grouped).
- qpAdm large LA → cuSOLVER Cholesky/inverse/SVD.
- DATES correlation → cuFFT.

These are the right calls.

### 7.2 Good custom kernels

- `decode_af_kernel` uses shared-memory transpose to maintain coalescing in
  both read and write directions.
- `dstat_kernel` has a tiled shared-memory accumulator and a fallback path
  for very large P.
- `ratio_block_jackknife` / `qpfstats_jackknife` are grid-strided and avoid
  global atomics by writing per-thread partials.

### 7.3 Custom kernels that should be revisited

- **qpGraph optimizers**: Running full L-BFGS in a single CUDA thread per
  restart is simple but leaves SM resources idle. Consider:
  - One-warp-per-restart model where inner linear algebra is cooperative.
  - Or keeping thread-per-restart but moving to a solver library (cuSOLVER or
    a custom CG) for the inner linear solve.
- **qpadm small path**: `qpadm_fit_models_kernel` / `qpadm_loo_models_kernel`
  run serial `double` loops inside one thread per model. This is fine for
  thousands of tiny models but becomes the bottleneck at the small/large
  boundary. A warp-cooperative batched GEMM/inverse would raise the crossover
  point.
- **Jackknife reductions**: Re-reading full f2 blocks for every jackknife
  block is algorithmically simple but memory-inefficient. A shared-memory
  blocked reduction over multiple jackknife blocks would reduce global reads.

---

## 8. Multi-GPU and peer-access patterns

### 8.1 Current design

`p2p_combine.cu` implements the device-resident multi-GPU reduction:

- Each GPU computes a disjoint slice of the result matrix.
- Slices are combined using `cudaMemcpyPeerAsync` into a single GPU's buffer
  in fixed order `g = 0..G-1` (`p2p_combine.cu`, lines ~80-120).
- A host-staged fallback exists for systems without P2P.

### 8.2 Why it is correct but not optimal

Fixed-order peer copies guarantee deterministic parity and avoid NCCL
complexity. However, for the final reduction across GPUs the code does not use
NCCL `ncclAllReduce` or a tree reduction. This means:

- The reduce-scatter/gather step is serialized GPU-by-GPU.
- P2P copy bandwidth is under-utilized because only one transfer is active at
  a time per direction.
- The host-staged path copies device → host → device, which is much slower
  than a P2P or NCCL tree.

### 8.3 Concerns

- **No NVLink topology awareness**: The combine loop does not prefer
  peer-accessible pairs or use `cudaDeviceCanAccessPeer` to build a ring.
- **No overlapping compute and P2P**: The combine happens after all GPUs have
  finished; during compute the P2P links are idle.
- **Device ordinal in backend**: `CudaBackend` is constructed with a single
  `device_id`. Multi-GPU fan-out is handled at a higher layer; there is no
  single backend object that owns multiple GPUs. This is fine for SPMD-style
  MPI/forking but makes multi-GPU scheduling within one process more complex.

---

## 9. Determinism and reproducibility

### 9.1 Mechanisms in place

- One stream per backend eliminates cross-stream reordering.
- cuBLAS workspace is bound once to a fixed 64 MiB allocation.
- `CublasHandle::set_stream()` re-applies workspace to avoid silent workspace
  changes.
- `MathModeScope` and `CusolverMathModeScope` restore math mode.
- Fixed-order peer combine and host-side final summation.
- Architecture §12 documents Philox seeds for stochastic paths.

### 9.2 Gaps

- **`cusolverDnSetDeterministicMode` is documented but not implemented**. For
  eigenvalue/SVD paths this is the most important missing piece.
- **cuFFT plan flags**: Verify that FFT plans are created with deterministic
  flags and that auto-tuning (`CUFFT_PATIENT`) is not used in reproducibility
  mode.
- **Ozaki slice count**: The fixed slice count derived from `mantissa_bits`
  should be validated against problem condition number; otherwise
  reproducibility may be bitwise but not accurate.
- **No `CUDA_DEVICE_ORDER` / `CUDA_VISIBLE_DEVICES` guard**: Multi-GPU runs
  rely on caller environment; the backend does not enforce PCI_BUS_ID ordering.

---

## Performance concerns ranked by impact

1. **Single-stream serialization** (high). All phases run back-to-back on
   `stream_`. Independent work (decode vs. GEMM, qpGraph restarts, jackknife
   reductions) cannot overlap.
2. **P2P combine is serialized peer copies** (high for multi-GPU). Should use
   NCCL `AllReduce` or at least a tree/ring over peer copies.
3. **cuSOLVER per-call allocations** (medium-high). Every Cholesky/SVD
   allocates a scratch buffer. Pooled workspace would help.
4. **Jackknife memory amplification** (medium). Re-reading f2 blocks `B`
   times is a large traffic increase over an incremental reduction.
5. **qpGraph thread-serial optimizer** (medium). Each restart is one thread;
   inner LA could be cooperative.
6. **No CUDA graphs** (medium). The chunked pipeline is regular and would
   benefit from graph capture.
7. **Pinned staging never shrinks** (low-medium). Long-lived backend pins more
   host memory than necessary for the current problem.

---

## Roadmap to A+ GPU engineering

| Priority | Item | Files / notes |
|----------|------|---------------|
| P0 | Add a second compute stream and explicit dependencies via `Event` for decode/upload/GEMM overlap. | `cuda_backend.cu`, `handles.hpp` |
| P0 | Wire `cusolverDnSetDeterministicMode(handle, CUSOLVER_DETERMINISTIC_MODE)` and document the exact reproducibility contract. | `handles.hpp`, `architecture.md` §12 |
| P1 | Replace serialized P2P copies with NCCL `ncclAllReduce` (or a peer tree when NCCL is unavailable). | `p2p_combine.cu` |
| P1 | Pool cuSOLVER workspace per handle/size rather than allocating per call. | `cuda_backend.cu`, `handles.hpp` |
| P1 | Capture the decode → f2 → reduction chunk pipeline in a CUDA graph for repeated execution. | `cuda_backend.cu` |
| P2 | Revisit jackknife kernels to fuse block reductions and reduce global reads. | `cuda_backend.cu` |
| P2 | Add NVTX ranges around phases and top-level API calls. | `cuda_backend.cu`, `handles.hpp` |
| P2 | Evaluate warp-cooperative qpGraph optimizer or move inner solve to a library/cg. | `cuda_backend.cu` |
| P2 | Add `cudaMallocAsync`/memory-pool option behind a config flag, with per-device lifetime tracking. | `device_buffer.cuh` |
| P3 | Shrink pinned staging when problem size decreases (or cap to a config maximum). | `cuda_backend.cu` |
| P3 | Add lightweight CUPTI/Nsight Compute counter validation in CI. | tests, scripts |

---

## Conclusion

The `steppe` CUDA layer is well-architected and safe: RAII ownership, a
single error macro, non-blocking streams, explicit precision seams, and a
thoughtful out-of-core tiering design. Its weaknesses are classic
"next-level" GPU engineering issues: more streams, better multi-GPU
reductions, pooled solver workspace, CUDA graphs, and a fully wired
reproducibility contract. Addressing the P0/P1 items above would move the
implementation from a solid B/B+ to an A+ GPU backend.

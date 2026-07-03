# `src/device/CMakeLists.txt` reference

## 1. Purpose

`src/device/CMakeLists.txt` defines the build for `steppe_device`, the static
library that holds all of steppe's CUDA code. Everything that touches the GPU —
the kernels, the GPU implementation of the compute backend, the RAII wrappers
around CUDA resources — is compiled here and nowhere else.

The library is built as a single static archive and also given the alias
`steppe::device`, so the rest of the build refers to it by that friendlier name.

Two ideas run through the whole file:

1. **This is the one place CUDA lives.** The kernel sources, the GPU backend, and
   the small helper headers that own CUDA handles and device memory are all
   collected here. Nothing outside this target is expected to include a CUDA
   header.
2. **CUDA is kept sealed inside.** The build is arranged so that the code which
   consumes this library — the core library, the tests, the command-line tool —
   can call into the GPU work through a plain C++ seam without themselves being
   dragged into compiling against the CUDA toolkit. The CMake link and include
   rules are what enforce that; they are not just convenience settings. This is
   explained in section 3.

---

## 2. The source files

The library compiles a mix of CUDA sources (`.cu`) and plain C++ sources
(`.cpp`). They fall into a few groups.

### Kernel sources (the GPU compute)

Each of these `.cu` files defines one family of GPU kernels.

| File | What it computes |
|---|---|
| `cuda/f2_block_kernel.cu` | The f2 statistic for a single genome block. It reformulates the computation as three matrix multiplications, runs them in the fixed-slice emulated double-precision math, and assembles the block result. |
| `cuda/f2_batched_kernel.cu` | The same f2 work but for many blocks at once. Blocks are grouped by size and run as strided-batched matrix multiplies (gather the inputs, do the GEMMs, assemble the outputs). |
| `cuda/qpadm_fit_kernels.cu` | The qpAdm model-fit kernels: gathering f4 values, the leave-one-out and total passes, and the small on-device linear algebra (singular-value decomposition, alternating least squares, weight solve, chi-square), all done in batch. |
| `cuda/qpgraph_fit_kernels.cu` | The qpGraph solver run entirely on the GPU, one thread per random restart: filling the path-weight table, evaluating a generalized-least-squares objective, and solving edge weights with non-negative least squares. |
| `cuda/decode_af_kernel.cu` | Unpacks 2-bit-packed genotypes and does a segmented allele-frequency reduction into per-population sums, variances, and counts. |
| `cuda/detect_ploidy_kernel.cu` | An on-device pre-pass that detects each sample's ploidy, one thread per individual. This replaced work that used to be done on the host. |
| `cuda/transpose_canonical_kernel.cu` | Transposes SNP-major input into steppe's canonical individual-major layout, with gather and re-encoding, one thread per output byte. This is the engine behind reading foreign genotype file formats. |
| `cuda/decode_compact_kernel.cu` | Builds an autosome keep-mask and gathers the kept columns into a compact population-by-marker matrix directly on the device. |
| `cuda/dstat_kernel.cu` | The second stage of the D-statistic: a per-block, per-quadruple normalized-D reduction. |
| `cuda/dates_kernel.cu` | The kernels for the DATES admixture-dating engine, which measures autocorrelation with FFTs: weighting, residuals, scatter, magnitude and cross-power spectra, and lag re-binning. |
| `cuda/qpfstats_kernel.cu` | The preparation kernels for the qpfstats smoothing solve: zeroing out NaN entries, adding a ridge term to the diagonal, and downdating NaN rows. |
| `cuda/qpfstats_jackknife_kernel.cu` | The on-device per-combination block jackknife for qpfstats. Runs in native double precision (a deliberate carve-out, see section 6). |
| `cuda/ratio_block_jackknife_kernel.cu` | A shared on-device block jackknife for ratio statistics, used by both f4-ratio and the D-statistic. Also native double precision. |

### The GPU backend, split across translation units

`cuda/cuda_backend.cu` implements the compute-backend interface on the GPU. That
implementation is large, so it is split into several separate translation units,
each owning one subsystem. Splitting it this way keeps each file compilable on its
own and keeps build times manageable.

| File | The subsystem it implements |
|---|---|
| `cuda/cuda_backend.cu` | The core backend that implements the compute-backend interface. |
| `cuda/cuda_backend_qpgraph.cu` | The qpGraph on-device solver. |
| `cuda/cuda_backend_dates.cu` | The DATES FFT-based autocorrelation dating engine (native double precision in its delicate steps). |
| `cuda/cuda_backend_decode.cu` | The decode and transpose engine that reads foreign genotype formats. |
| `cuda/cuda_backend_dstat.cu` | The D-statistic block reduction plus the shared ratio block jackknife. |
| `cuda/cuda_backend_qpfstats.cu` | The qpfstats smoothing solve (Gram matrix, Cholesky factorization, multi-right-hand-side solve). |
| `cuda/cuda_backend_fstats_assemble.cu` | The f-statistic sweep and the f4/f3 assembly. |
| `cuda/cuda_backend_f2_blocks.cu` | The `compute_f2` entry point and the f2 block tensors, both the fully-resident and the streamed variants. |
| `cuda/cuda_backend_qpadm_fit.cu` | The full qpAdm fit engine: jackknife, the large singular-value decomposition, the rank test, the generalized-least-squares solve, the standard errors, and the batched fit. |

These files call across to each other through narrow declarations rather than
duplicating logic, so a routine like the block-survivor computation has exactly
one definition even though several backend files use it.

### Device-resident result handles

These files hold the special member functions (constructors, destructors, moves)
for the handle objects that keep results living in GPU memory instead of copying
them back to the host.

| File | The handle it defines |
|---|---|
| `cuda/device_partial.cu` | A partial-result handle whose underlying resident data is kept behind an opaque implementation pointer. |
| `cuda/device_f2_blocks.cu` | The device-resident full f2-block result, including the routines to upload it and copy it back to the host. |
| `cuda/device_decode_result.cu` | The device-resident decoded result: the autosome-compacted allele frequencies and variances, plus accessors. |
| `cuda/p2p_combine.cu` | Combines per-GPU partial results while they stay on the devices, using direct device-to-device copies with disjoint placement. |

### Spilling results out of GPU memory

When results are too large to stay in GPU memory, they spill to host RAM or disk.
These two files handle that.

| File | What it does |
|---|---|
| `cuda/block_sink.cu` | The host-RAM sink and the disk sink: a triple-buffered, pinned-memory writer that streams blocks out to the chosen tier. |
| `cuda/f2_blocks_out.cu` | A tier-agnostic reader that copies results back from whichever tier they landed in, plus the disk-backed result's members. |

### The CPU reference backend

| File | What it does |
|---|---|
| `cpu/cpu_backend.cpp` | A plain, scalar, host-only implementation of the same compute backend. It exists to be the trusted reference that the GPU results are checked against, not a runtime users would choose. |

### Plain C++ host files

The last three `.cpp` files contain no CUDA at all. They are covered in section 5.

| File | What it does |
|---|---|
| `host_ram.cpp` | Reports how much host RAM is free and decides which memory tier to use. The tier policy itself is CUDA-free. |
| `resources.cpp` | Builds the per-device resource bundle: one backend plus one capability probe for each GPU. |
| `shard_plan.cpp` | Plans how SNP work is split into block-aligned shards across devices. Pure host arithmetic. |

---

## 3. The CUDA-free layering mechanism

The single most important thing this build file does is keep CUDA sealed inside
`steppe_device` so that the rest of steppe never has to compile against the CUDA
toolkit. This is achieved with CMake's `PUBLIC` versus `PRIVATE` link and include
rules rather than by convention or discipline.

Three rules combine to make it work.

1. **The CUDA libraries are linked `PRIVATE`.** `CUDA::cudart`, `CUDA::cublas`,
   `CUDA::cusolver`, and `CUDA::cufft` are all private dependencies. That means a
   target linking against `steppe_device` inherits none of them and cannot see a
   CUDA header through this library. The core library, the public API, and the
   tests stay CUDA-free.
2. **The source root is included `PUBLIC`.** The `src/` directory is exposed as a
   public include path, so a consumer can include the plain-C++ seam header
   `device/backend.hpp` to reach the GPU work. That header is deliberately written
   with no CUDA in it — it is the interface everything talks to, while the CUDA
   implementation stays hidden behind it.
3. **The shared internal helpers are linked `PUBLIC`.** `steppe::core_internal` is
   a public dependency because both the GPU kernels *and* the CPU reference backend
   need the same shared f2 estimator primitive and the same data views. Exposing it
   publicly keeps that single definition of the formula available to both, so the
   two implementations can never drift into computing the statistic differently.

The net effect: callers get a clean, CUDA-free way to invoke GPU work, and the
compiler enforces that they cannot accidentally start depending on CUDA headers.

---

## 4. Include path and link libraries

This section lists the concrete settings that put section 3 into effect.

**Include directory.** The library publicly adds its parent directory (the `src/`
root) to the include path. That is what lets a consumer write
`#include "device/backend.hpp"` for the CUDA-free seam and
`#include "core/internal/..."` for the shared estimator and views, both resolving
from the same root.

**Link libraries.**

| Dependency | Visibility | Why |
|---|---|---|
| `steppe::api` | `PUBLIC` | The public, CUDA-free interface types, shared with consumers. |
| `steppe::core_internal` | `PUBLIC` | The shared f2 estimator and data views used by both the GPU and CPU backends. |
| `steppe::warnings` | `PRIVATE` | The project's warning flags, an internal build detail. |
| `CUDA::cudart` | `PRIVATE` | The CUDA runtime. Private so it never leaks to consumers. |
| `CUDA::cublas` | `PRIVATE` | The dense matrix-multiply library used by the f2 GEMMs. |
| `CUDA::cusolver` | `PRIVATE` | The dense linear-algebra library used for the solves and decompositions. |
| `CUDA::cufft` | `PRIVATE` | The FFT library used by the DATES autocorrelation engine. |

The library also requests the C++20 language standard `PUBLIC`, so the same
standard applies to code that consumes it.

---

## 5. The plain-C++ (non-CUDA) source files

`resources.cpp` and `shard_plan.cpp` are ordinary C++ files, not CUDA files, even
though they live in the device library. This is intentional and worth
understanding.

- `resources.cpp` builds one backend and one capability probe per GPU. It reaches
  the GPU only indirectly, through a CUDA-free factory function that manufactures a
  backend and through a CUDA-free capabilities probe. It never includes a CUDA
  header itself.
- `shard_plan.cpp` plans how the SNP work is divided into block-aligned shards
  across devices. This is pure host arithmetic over block ranges — there is no GPU
  code in it at all.

Both files are compiled into `steppe_device` so that the backend factory resolves
and so the multi-GPU orchestrator in the core library can reach the shard planner
through this target. But because they are themselves free of any CUDA toolkit
dependency, keeping them here does not violate the layering: they are host code
that happens to live next to the GPU code they help drive.

---

## 6. Compile-time feature switches

Two optional preprocessor definitions are applied to the device sources, each
gated behind a build option and each kept `PRIVATE` so it never leaks to
consumers.

### `STEPPE_HAVE_EMU_TUNING`

The fixed-slice emulated double-precision math is only actually engaged when the
GPU code can call a specific set of matrix-multiply library functions that set the
emulation strategy and cap the mantissa bit count. Those calls are guarded behind
this macro.

The measured precision policy — a fixed 40-bit mantissa as the default — *requires*
those calls, so the production build defines this macro (the underlying option is
on by default). If it were compiled out, the emulated mode could not be honored and
the code would fall back to native double precision. The macro is applied
privately because engaging the emulation is a detail of how the device code is
generated, not something a consumer needs to know.

### `STEPPE_NVTX`

This gate turns on NVTX phase-boundary annotations, which mark named ranges in a
GPU profiler's timeline so you can see where each phase of the computation begins
and ends. The underlying build option defaults to off.

When it is on, the macro is defined on the device translation units, and the
project's range macro expands into a scoped NVTX range that pulls in the
header-only NVTX C++ API (which ships inside the CUDA toolkit, so it needs no extra
link step). When it is off, that macro expands to nothing, no NVTX header is
included, and the compiled object code is byte-for-byte identical to a build
without profiling — so turning profiling on or off can never change a reported
result. Like the emulation-tuning macro, it is applied privately: it is a
device-side observability detail that consumers never see.

---

## 7. CUDA build properties

The remaining settings control how the CUDA code is compiled.

### Separable compilation and position-independent code

The library is built with both separable compilation and position-independent code
turned on.

Separable compilation is needed because the backend translation unit launches a
kernel that is defined in a *different* translation unit, through a narrow launch
wrapper. That cross-file kernel launch, and the link-time device optimization used
in release builds, both require this mode. Position-independent code lets the
resulting objects be linked into shared libraries.

### Optional fatbin compression

There is a release-only, opt-in setting that compresses the compiled GPU code
(the "fatbin") to reduce its size. It is guarded so that it only applies to CUDA
compilation in release configurations, and only when the corresponding build option
is turned on. It is a size optimization with no effect on results.

# `cuda_backend.cu` reference

## 1. Purpose

`src/device/cuda/cuda_backend.cu` is where the GPU implementation of steppe's
compute backend lives. It is the one place a host caller actually meets the GPU
code path. The rest of the library only ever talks to an abstract, CUDA-free
interface (`ComputeBackend`); this file provides the concrete class, `CudaBackend`,
that fulfills that interface using CUDA, cuBLAS, cuSOLVER, and cuFFT.

The backend's core job is the "f2" computation — steppe's central population-genetics
statistic — expressed on the GPU as three matrix multiplications with a fused
feeder stage before them and a fused numerator/divide stage after. The full compute
methods that do that work are grouped into companion source files; this particular
file holds the parts that stand apart from the heavy math:

- constructing the backend and binding it to a GPU,
- reporting what the bound GPU can do (`capabilities()`),
- two small helpers that create the backend and count visible GPUs,
- and the single function that translates a low-level device error into a
  program exit code.

It also carries the design rationale for how the whole backend is organized: why it
is a single translation unit, how one backend instance binds to exactly one GPU,
why its memory can be freed safely no matter which GPU is currently selected, and
which numerical guarantees it upholds.

---

## 2. One translation unit, one class

The entire `CudaBackend` class is deliberately kept in a single translation unit
(one compiled source file) rather than split across several. The reason is a hard
C++ rule: a single class cannot be spread across multiple translation units. Because
the launch wrappers and GPU kernels belong together with the methods that call them,
they are co-located here on purpose. Splitting the class across files was considered
and rejected.

A lighter alternative — keeping it one class but pulling per-subsystem bodies into
separate include-fragments — has been noted as a possibility but not pursued. It
would only be revisited if the file's size ever became a real maintenance burden.

This file is CUDA code and is private to the GPU layer. The core of the library
reaches it only through the CUDA-free interface, so nothing outside the GPU layer
ever includes a CUDA header or names the concrete `CudaBackend` type.

---

## 3. One backend, one GPU (the multi-GPU binding)

Each `CudaBackend` instance is bound to exactly one physical GPU. This is the
foundation that lets steppe scale across several GPUs: the layer above
(`Resources`) creates one backend per GPU and orchestrates the work between them.
That cross-GPU coordination — splitting the SNPs across GPUs and combining the
partial results in a fixed order — happens above this file and is not implemented
here. What this file provides is the per-GPU building block.

The binding works like this:

- The constructor takes an `int device_id`. It defaults to `0`, so a plain
  single-GPU run is unchanged and every existing zero-argument call site stays
  bound to GPU 0.
- Constructing the backend first selects that GPU, so that the cuBLAS context and
  all allocations are created on it.
- Every method that does GPU compute re-selects the bound GPU on entry (via an
  internal `guard_device()` step), so its allocations always land on the right
  device even if some other GPU was current when the call arrived.

Every knob this multi-GPU design exposes only changes *where* data lives or *how*
it moves between GPUs — never a number steppe reports. So results are identical
whether a computation ran on one GPU or several.

---

## 4. Construction: binding the workspace and stream

The constructor sets up three things in a specific order, and the order matters.

1. **Select the GPU first.** The member that stores the device id is initialized by
   a helper that both selects the GPU and returns its id. Because that runs before
   anything else, the cuBLAS handle and every later allocation are created on the
   correct GPU.

2. **Give cuBLAS an explicit fixed-size workspace.** The emulated double-precision
   matrix multiplies only produce the exact same result every run if cuBLAS is
   handed a fixed workspace buffer. The constructor binds that workspace *before*
   binding the stream.

3. **Bind the single statistics stream.** All GPU work runs on one stream, which is
   what makes results reproducible run to run. Binding the stream to cuBLAS resets
   its workspace, so the workspace is deliberately set first — the stream-binding
   step re-applies it afterward. After this one-time setup the matrix-multiply
   routines never touch the stream again, so the workspace survives for every later
   batch of matrix multiplies.

cuSOLVER (used for the linear-algebra steps) shares that same single stream, and
binding a stream to it has no workspace-reapply hazard, so it needs no special
ordering.

All device memory and library handles here are owned by RAII wrappers — buffers,
cuBLAS handle, cuSOLVER handle, streams, and events all free themselves. There is
no raw allocation in this file.

---

## 5. `capabilities()`: probing the bound GPU

`capabilities()` inspects the GPU this backend is bound to and returns a plain
struct describing it. It is a read-only query: it saves whichever GPU was current
on entry and restores it before returning, so calling it leaves no side effect.

It reports these things:

| What it probes | How | What it means |
|---|---|---|
| **Number of visible GPUs** | `cudaGetDeviceCount` | The upper bound on how many GPUs the multi-GPU combine could fan out over. Failing to enumerate GPUs is treated as a real fault — a process running a GPU backend must be able to list its devices. |
| **Compute capability** | `cudaGetDeviceProperties` | The GPU's hardware generation (for example, `{12, 0}` on the Blackwell test box). This is informational only. One build serves both test boxes, so this is never used to pick a code path. |
| **Total and free VRAM** | `cudaMemGetInfo` | Both numbers are kept. Total memory feeds the memory-budget logic that decides how large a problem fits; free memory is the live headroom. |
| **Peer-to-peer reachability** | `cudaDeviceCanAccessPeer` | Whether the bound GPU can directly read another GPU's memory. Reported true if it can reach *at least one* other GPU (the combine step needs one peer to pull from). A single GPU, or GPUs that cannot reach each other, report false. |
| **Emulated double-precision honorability** | the single shared predicate | Whether this build actually has the emulated double-precision math compiled in. See section 7 for what "not honorable" leads to. |

Two details are worth calling out:

- **The peer-access probe never throws.** On consumer GPUs, direct GPU-to-GPU
  access is commonly turned off by the driver. That "no" is *expected*, not an
  error: it simply means steppe uses the slower host-staged path to combine
  results instead of the direct GPU-to-GPU fast path. So a failed or negative
  peer-access probe is logged as a warning and treated as a capability downgrade,
  never a crash. It also skips checking a GPU against itself, since that query is
  only defined for two different GPUs.

- **The emulated-precision probe asks exactly the question the compute path asks.**
  It runs the same predicate the matrix-multiply path uses, with a default emulated
  precision request. That guarantees the capability report and the actual compute
  path can never disagree about whether emulated precision is honored.

---

## 6. Freeing GPU memory under any current GPU

There is a subtle multi-GPU safety property this file guarantees, and it is worth
understanding because it looks wrong at first glance.

Every compute method re-selects the bound GPU before allocating, so allocations land
on the right device. But *freeing* memory is different. Some buffers this backend
produces (the resident f2 result and its paired-variance buffer) are moved out of
the backend and freed much later by the host-side combine step — possibly while a
completely different GPU is the current one. None of the RAII wrappers re-select the
bound GPU when they free.

This is correct by design. The CUDA calls that release resources — freeing device
memory, destroying a stream, an event, a cuBLAS handle, or a cuSOLVER handle — are
all independent of which GPU is currently selected. Each pointer or handle already
carries its own device association, so the free works no matter which GPU is current.
Freeing device memory in particular neither requires nor forbids that its owning GPU
be the current one.

Because of that, the RAII wrappers intentionally do **not** save-and-restore the
current GPU around a free. Doing so would hide global device-selection state inside
the object that owns the memory, and the design keeps that responsibility with the
caller (the coordination layer), not the owner. This is the backend-side statement
of a rule that also lives with the buffer wrapper itself, so that the multi-GPU
teardown path is self-documenting from either side.

---

## 7. The numerical standards the backend upholds

The backend is written to a few fixed guarantees so that GPU results stay
trustworthy and can never quietly drift from the reference.

- **Precision is a typed setting that is passed straight through.** The chosen
  precision (see the config reference) is forwarded unchanged into the matrix-multiply
  path, which engages either emulated double precision or native double precision
  based on one honorability check.

- **An unhonorable emulated request degrades safely.** If this build was compiled
  without the emulated double-precision math, an emulated request does *not* fall
  back to a rejected automatic-mantissa mode. Instead it downgrades to native double
  precision and logs a capability note. The dangerous automatic mode is never run.

- **The delicate arithmetic always stays native double precision.** The small,
  cancellation-prone numerator-and-divide step at the end of the f2 computation is
  always done in native double precision, regardless of the selected mode, inside
  the assemble stage.

- **The f2 formula exists in exactly one place.** Both the CPU reference oracle and
  the GPU path share the same estimator definition, so the two implementations
  cannot diverge in what they compute.

- **Every capability lever is result-neutral.** All the multi-GPU and
  memory-movement knobs only affect where data lives, how it moves, or what gets
  observed — never a reported number. So the reproducibility guarantees hold
  identically whether steppe runs on a capable multi-GPU box or a single budget GPU.

---

## 8. Factory functions

Two free functions let the CUDA-free part of the codebase create and count GPU
backends without ever naming the concrete type or touching a CUDA header.

- **`make_cuda_backend(int device_id)`** builds a `CudaBackend` and returns it as
  the abstract interface. The `device_id` binds the new backend to one physical GPU
  (the coordination layer passes the id of each GPU it wants to use). The default of
  `0` keeps the single-GPU path and every existing zero-argument caller bound to
  GPU 0, with no behavior change.

- **`visible_device_count()`** returns the number of visible GPUs with a single
  lightweight call. It deliberately does *not* create a cuBLAS context, allocate a
  workspace, or change the current GPU, so it is the cheap process-wide count query
  that auto-detection and device-id validation need. It returns the same number that
  `capabilities()` reports for the device count. A failure to enumerate GPUs is a
  real fault and throws; a genuine count of zero is returned as `0` (deciding what
  to do with no GPUs is the caller's job).

---

## 9. Translating device faults to exit codes

`device_fault_status()` is the one function that lets the CUDA-free application layer
react to a device-specific error without being able to see the CUDA-specific
exception types (which are private to the GPU layer). The application catches a plain
`std::exception`, hands it to this function, and gets back either a specific status
or "nothing special."

Its only job is to single out genuine device *out-of-memory* failures, because those
deserve their own exit code. It inspects the exception's real type and its embedded
status code:

| Exception type | Status it carries | Result |
|---|---|---|
| CUDA runtime error | the memory-allocation error code | out-of-memory status → the device-out-of-memory exit code (3) |
| cuBLAS error | the allocation-failed status | out-of-memory status → the device-out-of-memory exit code (3) |
| cuSOLVER error | the allocation-failed status | out-of-memory status → the device-out-of-memory exit code (3) |
| anything else | — | nothing special → the general runtime-error exit code (5) |

Anything that is not one of these three device-memory failures — including a
non-allocation CUDA/cuBLAS/cuSOLVER error, or a host-side out-of-memory (which is
system RAM, not GPU memory) — returns "nothing special," so the application falls
back to its catch-all runtime-error exit code.

The function is safe to call with no risk of throwing: it only does runtime type
checks and integer comparisons, with no allocation of its own. The type checks are
well-defined because steppe links as a single executable, so each type has exactly
one identity across the program.

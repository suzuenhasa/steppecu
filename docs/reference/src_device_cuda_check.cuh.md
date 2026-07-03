# `check.cuh` reference

## 1. Purpose

`src/device/cuda/check.cuh` is the single home for CUDA and CUDA-library
error checking. Every call into the CUDA runtime, cuBLAS, cuSOLVER, or cuFFT
whose failure is a genuine fault routes through one of the macros defined here.
There is exactly one CUDA check, one cuBLAS check, one cuSOLVER check, and one
cuFFT check in the whole codebase, and they all live in this file. It replaced
three separately-copied `CUDA_CHECK`/`CUBLAS_CHECK` macros that had drifted, so
consolidating them here removes that duplication and its drift risk.

The file gives you five things:

1. **Four typed exception classes** — `CudaError`, `CublasError`,
   `CusolverError`, `CufftError` — one per library, each carrying the exact call
   site and a human-readable status.
2. **A shared call-site formatter** (`format_call_site`) used by all four.
3. **Throwing check helpers and their macros** — the "something failed and we
   cannot continue" path.
4. **A non-throwing warn helper and macro** (`STEPPE_CUDA_WARN`) — the
   "this call can legitimately fail, log it and let me decide" path.
5. **A post-kernel-launch check macro** (`STEPPE_CUDA_CHECK_KERNEL`).

This is a CUDA header (`.cuh`): it includes `<cuda_runtime.h>`, `<cublas_v2.h>`,
`<cufft.h>`, and `<cusolverDn.h>`, so it is private to the GPU device layer. The
core library, the public API, and the command-line tool never include it — only
device-layer code does.

---

## 2. Throw a typed exception, never exit

The central design decision is that a failed call raises a C++ exception rather
than calling `exit()`. Calling `std::exit(EXIT_FAILURE)` on failure is fine for a
throwaway script but fatal for a library: it would kill the caller's whole
process. Because these helpers throw instead, two things become possible:

- **Tests can catch the failure.** A test can wrap a call in
  `try { ... } catch (const CudaError&) { ... }` and assert on it.
- **The public API can translate the failure into a status code.** Instead of
  the process dying, the API layer catches the typed exception and maps it to the
  library's own error enum (for example, a CUDA runtime fault becomes a
  "CUDA runtime error" status, and an out-of-memory fault specifically —
  `cudaErrorMemoryAllocation` — becomes a distinct "device out of memory" status,
  distinguishable through the exception's `status()` accessor).

Each CUDA library reports failures with its own status type and its own way of
turning that status into text. The CUDA runtime has `cudaGetErrorName` and
`cudaGetErrorString`; cuBLAS, cuSOLVER, and cuFFT do **not**. That is why there
are four sibling exception classes and four sibling checks rather than one: the
three library classes each carry a small hand-written switch that maps their
status enum to its symbolic name, since there is no runtime function to do it for
them.

---

## 3. The typed exception classes

All four classes follow the same shape. Each derives from `std::exception`,
stores the raw status value and a fully-formatted message string, exposes the
message through the standard `what()`, and exposes the raw status through a
`status()` accessor so callers can branch on the specific failure. The message is
built once in the constructor.

### The shared call-site prefix (`format_call_site`)

Every exception message starts with the same prefix:
`file:line (function): 'expr' -> `, where `expr` is the exact source text of the
call that failed (for example, the literal string `cudaMemcpyAsync(...)`). That
prefix is built by the free `format_call_site` function so the four classes do
not each re-implement it. It takes a `std::source_location` (the file, line, and
function of the call site) and the stringized expression, and returns the prefix
string. Only the trailing part — the per-library status rendering — differs
between the four classes, so it is appended by each constructor after the shared
prefix. The function is `inline` (this is a header) and does pure host-side string
formatting, so it touches no GPU work and no reported result.

### `CudaError`

Thrown when a CUDA runtime call returns anything other than `cudaSuccess`. Its
message appends the runtime's own `cudaGetErrorName` and `cudaGetErrorString` to
the shared prefix. Its `status()` returns the `cudaError_t`, which is what lets
the API layer separate a plain runtime error from an out-of-memory error.

### `CublasError`

Thrown when a cuBLAS call returns anything other than `CUBLAS_STATUS_SUCCESS`.
Because cuBLAS has no `cudaGetErrorString` equivalent, the class carries a static
`status_name(cublasStatus_t)` that switches over the cuBLAS status enum and
returns its symbolic name (for example `CUBLAS_STATUS_ALLOC_FAILED`). Any value
not in the switch renders as `CUBLAS_STATUS_UNKNOWN`.

### `CusolverError` — and why `devInfo` is not routed here

Thrown when a cuSOLVER dense API call returns anything other than
`CUSOLVER_STATUS_SUCCESS`. Like cuBLAS, it carries its own static
`status_name(cusolverStatus_t)` switch, defaulting to `CUSOLVER_STATUS_UNKNOWN`.

There is an important distinction here. cuSOLVER routines report **two** kinds of
outcome. One is the API status returned by the call itself (bad handle, allocation
failure, architecture mismatch) — that is a fault, and it throws. The other is a
per-call `int* devInfo` output that reports the numerical outcome of the
factorization or solve: a value greater than zero means the matrix was singular
or not positive-definite. That second kind is **not** routed through this
exception. A singular or not-positive-definite matrix is a legitimate domain
outcome — for the model fit it is a meaningful result, not a crash — so the fit
maps `devInfo` to one of its own status values by hand. Only the API status is
thrown. This split is a frozen contract: do not start throwing on `devInfo`.

### `CufftError`

Thrown when a cuFFT call returns anything other than `CUFFT_SUCCESS`. cuFFT again
has no string function, so it carries a static `status_name(cufftResult)` switch,
defaulting to `CUFFT_STATUS_UNKNOWN`. This one is used by the FFT-based
autocorrelation path (the `dates_curve` computation) for its plan create, execute,
and set-stream calls, and by the RAII plan owner's teardown, so plan destruction
is no longer an unchecked call.

The set of enumerators in this switch is deliberately kept in exact sync with the
`cufftResult` enum as it exists in the CUDA 13.x `cufft.h` header (verified
against the installed header). Compared to older CUDA versions, three legacy
enumerators were removed and four new ones were added. The switch names only the
enumerators that actually exist in the current header — naming a removed one, or
omitting a present one, would trip the project's warnings-as-errors build.

---

## 4. The fault-check macros

The throwing checks are exposed as macros so the failing expression can be
captured as text and the call site can be captured automatically.

| Macro | Wraps | Throws on failure |
|---|---|---|
| `STEPPE_CUDA_CHECK(expr)` | any CUDA runtime call | `CudaError` |
| `CUBLAS_CHECK(expr)` | any cuBLAS call | `CublasError` |
| `CUSOLVER_CHECK(expr)` | any cuSOLVER dense API call | `CusolverError` |
| `CUFFT_CHECK(expr)` | any cuFFT call | `CufftError` |

Each macro forwards the call's return status and the stringized expression
(via `#expr`) to a small `inline` helper in the `detail` namespace
(`cuda_check`, `cublas_check`, `cusolver_check`, `cufft_check`). The helper
compares the status against that library's success value and throws the matching
typed exception if they differ; otherwise it returns and the program continues.

**How the call site is captured with no `__FILE__`/`__LINE__` plumbing.** Each
helper takes a third parameter, a `std::source_location`, that **defaults** to
`std::source_location::current()`. Because a default argument is evaluated at the
point where the caller expands it, the source location that gets filled in is the
caller's file, line, and function — automatically, without the macro passing any
file or line tokens. So the macros only ever pass the stringized expression; the
location rides in for free through the default argument.

Use these for **fault** calls only — calls whose failure means the run cannot
sensibly continue (a memory copy, a GEMM, a solve, an FFT plan). Calls whose
"failure" is an expected, recoverable condition must use the warn path in the
next section instead.

The `CUSOLVER_CHECK` macro checks only the API status, consistent with the
`devInfo` distinction described above.

---

## 5. `STEPPE_CUDA_WARN` — the non-throwing path for recoverable statuses

Some CUDA calls can return a non-success status as a normal, expected outcome
rather than a fault. The two motivating cases are capability probes and pollers:

- Asking whether one GPU can directly access another's memory
  (`cudaDeviceCanAccessPeer`) can legitimately answer "no" on consumer-grade
  hardware where that peer-to-peer capability is disabled.
- Turning on peer access (`cudaDeviceEnablePeerAccess`) can return
  `cudaErrorPeerAccessAlreadyEnabled`, which just means it was already on.
- Non-blocking pollers (`cudaStreamQuery`, `cudaEventQuery`) can return a
  "not ready yet" status that is the entire point of polling.

Routing any of these through the throwing check would turn a graceful,
expected degrade into a hard crash. So `STEPPE_CUDA_WARN(expr)` is the
non-throwing sibling. On a non-`cudaSuccess` status it emits exactly **one**
warning line — in the same `file:line (function): 'expr' -> name: string` shape
as a `CudaError` message — and then **returns the `cudaError_t`** so the caller
can branch on it and tag the degrade. It never throws.

A few details worth knowing:

- **The return value is always available; only the log line is build-gated.**
  The warning line is emitted through the project's one warning sink, which is
  silent in release builds. Under a release build the log macro compiles down to
  a no-op and its arguments are not even evaluated — but the status is still
  returned regardless of build mode, so a release-build caller still sees the
  status it must branch on. The behavior a caller depends on (getting the status
  back) is therefore independent of whether logging is compiled in.
- **`[[maybe_unused]]` on the message arguments.** Because the expression string
  and the source location are consumed only by that log line — which disappears
  in release — they would otherwise be unused parameters in a release build and
  trip the project's `-Wunused-parameter` under warnings-as-errors. Marking them
  `[[maybe_unused]]` keeps the release build clean while only `status` is
  actually referenced.
- **The helper is `[[nodiscard]]`.** The whole point is that the caller inspects
  the returned status, so ignoring it is flagged.

Typical use is `if (STEPPE_CUDA_WARN(cudaDeviceEnablePeerAccess(peer, 0)) !=
cudaSuccess) { degrade(); }`, or capturing the status into a variable to branch
on.

Every knob that this path guards is a capability lever that only affects *how*
data moves between GPUs, never a reported statistic. Because of that, the warn
path never appears on the actual statistics computation — that path uses only the
throwing checks — so the numbers a run reports are identical whether the machine
has the peer-to-peer capability or not.

---

## 6. `STEPPE_CUDA_CHECK_KERNEL` — the post-launch kernel check

Place this immediately after every kernel launch (`kernel<<<...>>>(...)`). It
does two things:

1. **Always** calls `cudaGetLastError()` through `STEPPE_CUDA_CHECK`. This
   synchronously surfaces a bad launch configuration — for example an invalid
   grid or block shape, or too much shared memory — right at the launch site.
2. **In debug builds only**, additionally calls `cudaDeviceSynchronize()` through
   `STEPPE_CUDA_CHECK`. Kernel execution is asynchronous, so a fault that happens
   *inside* a kernel would otherwise not surface until some later, unrelated CUDA
   call — making it hard to attribute. Forcing a synchronize right after the
   launch makes an in-kernel fault attribute to *this* launch, which is what tools
   like compute-sanitizer need to pinpoint the culprit.

The debug-only synchronize is gated through the project's single debug-only
facility, not a hand-written `#if defined(NDEBUG)` at each call site, so the gate
lives in one place. Release builds deliberately skip the forced synchronize:
paying for a full device synchronize after every kernel would wreck hot-path
performance, and in release the runtime's next call will surface the sticky error
anyway.

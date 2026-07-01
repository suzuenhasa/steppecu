# Review — `src/device/cuda/check.cuh` (unit: device-cuda-check)

Adversarial second pass over the first-pass review (8.5/10, 18 findings). Every
first-pass finding was re-verified against the actual source **and** the official
NVIDIA Runtime-API / cuBLAS docs and the ISO C++ / nvcc record (each load-bearing
device-behavior claim is cited inline). False positives are demoted to *Considered
& rejected* with the reason; real ones are confirmed (several strengthened); two
genuinely-new micro-findings were added on this pass (S-1 non-variadic macro
robustness, R-5 `final`/special-member hygiene). Standards judged against:
`docs/architecture.md` §2/§4/§7/§8/§10/§12/§13/§16/§17, `docs/ROADMAP.md` §4/§5/§6,
`docs/TODO.md` (cleanup backlog §A/§B + the ⚡/`wxz1fiiln` capability-tier section).

The file is a 148-line CUDA header (`.cuh`) that is the project's single home for
CUDA/cuBLAS error checking: two typed exception classes (`CudaError`,
`CublasError`), two `inline detail::` checker functions, and three macros
(`STEPPE_CUDA_CHECK`, `CUBLAS_CHECK`, `STEPPE_CUDA_CHECK_KERNEL`).

**Re-verified facts (grep, this pass):**
- It is the most-reused header in `steppe_device`: included by `device_buffer.cuh`,
  `stream.hpp`, `handles.hpp`, `cuda_backend.cu`, `f2_block_kernel.cu`,
  `f2_blocks_kernel.cu`, `decode_af_kernel.cu` (confirmed).
- `CublasError::status_name` has a **real, non-test consumer**: `handles.hpp:75`
  calls it from a `noexcept destroy()` teardown path (confirmed) — this constrains
  A-1's fix to stay `noexcept` + NULL-guarded.
- Every `STEPPE_CUDA_CHECK_KERNEL()` (in `f2_block_kernel.cu:234,279`,
  `f2_blocks_kernel.cu:182,236`, `decode_af_kernel.cu:112`) sits **immediately after
  a `kernel<<<>>>` inside a narrow `launch_xxx`/`run_xxx` wrapper**, with all prior
  cuBLAS/runtime calls in the same wrapper routed through `CUBLAS_CHECK`/
  `STEPPE_CUDA_CHECK` (confirmed by reading each site) — the C-1 invariant holds
  everywhere today.
- The test TU **shadows** the production macros: `test_f2_equivalence.cu:104,115`
  `#define`s its own `STEPPE_CUDA_CHECK`/`CUBLAS_CHECK` (confirmed; its own comment
  at 98–102 says so), so this header's throw path is exercised by **nothing** in
  `tests/` (T-1).
- `STEPPE_DEBUG_ONLY` / `STEPPE_ASSERT` and `internal/log.hpp` **do not exist** in
  the tree (grep: tokens appear only in `architecture.md` prose and the cleanup
  docs). `check.cuh` open-codes `#if defined(NDEBUG)`; `stream.hpp`, `handles.hpp`,
  `device_buffer.cuh` each open-code their own `STEPPE_*_WARN_ON_TEARDOWN` (R-4).
- The current public error enum is `steppe::Status` (`include/steppe/error.hpp:21`,
  member `DeviceOom` at :27); the `STEPPE_ERR_*` C names the header's comments cite
  exist **nowhere** in the tree yet (R-1).

## Role & layering

`check.cuh` is the DRY single-home for CUDA/cuBLAS error checking mandated by
architecture §2 ("one `STEPPE_CUDA_CHECK`"), §7 ("`STEPPE_CUDA_CHECK` + post-launch
checks"), and the §8 single-home table (rows "CUDA error check" and
"cuBLAS/cuSOLVER check"). It correctly lives under `src/device/cuda/` as a CUDA
header (`#include <cuda_runtime.h>`, `#include <cublas_v2.h>`), which makes it
`PRIVATE` to `steppe_device` per the §4 layering rule — `core`/`api`/the CLI
physically cannot include it (enforced by CMake link visibility). The header guard,
namespace placement (`steppe::device`, checkers tucked in `detail`), and the
global-scope macros are idiomatic and layering-legal. The header allocates no device
memory, so it is correctly absent from the §2 allocation allowlist.

Two layering observations:
- **`CUSOLVER_CHECK`, promised by §7/§8/§17, is absent** (Finding L-1). Defensible at
  M4 (cuSOLVER is a Phase-2 dependency), but it is a documented single-home that does
  not exist yet.
- The macro/namespace shape diverges from the §7 spec snippet in a way the code
  resolved *more* correctly than the doc (the spec writes
  `::steppe::detail::cuda_check`; the code writes
  `::steppe::device::detail::cuda_check`). Doc-staleness, not a defect (see
  *Considered & rejected*).

## Score: 8.5/10 — strong, idiomatic, near-complete; held back by an OOM-path allocation hazard (constructor *and* the implicit copy ctor), a reinvention of official cuBLAS status APIs that drops the human-readable description, a release-mode post-launch-check contract that overstates what the code delivers, a device-wide debug sync that fights the §7 one-stream-per-lane idiom, and forward-promises the tree has not kept

The score is unchanged from the first pass, and the second-pass verification *upholds*
the first pass's headline composition: B-1 stays confirmed and strengthened (the
implicit copy constructor is a second OOM-path allocation hazard); A-1/P-1 are
confirmed against the cuBLAS reference (both `cublasGetStatusName` and
`cublasGetStatusString` exist, return `const char*`, and are *undocumented* for
unknown codes — so the NULL guard the code already has must survive the refactor),
with the constraint that the fix stays `noexcept` for the `handles.hpp:75` teardown
caller; C-1's *contract* defect is confirmed and its real-world severity correctly
sits at MED because every current call site maintains the
"every-runtime-call-is-checked" invariant; C-2 (device-wide debug sync) is confirmed
against the Runtime-API docs and latent today (default-stream-only backend). The
prior pass's two new findings (A-4 nvcc `source_location` fragility — corroborated by
NVIDIA bug #4173735; R-4 `STEPPE_DEBUG_ONLY` single-source divergence — corroborated
by grep showing four open-coded debug gates) are upheld. This pass adds two small
findings (S-1, R-5) and does not change the number — they are LOW polish. It remains
disciplined, well-cited code clearly written by someone who read §7; it falls short
of 9.5–10 on the substantive items below.

---

## Findings

### (1) Correctness & bugs

**C-1 [MED, effort S, before-M4.5: yes] — Release-mode `STEPPE_CUDA_CHECK_KERNEL` overstates its attribution contract; `cudaGetLastError()` reports the first un-retrieved error since the last clear, not "this launch."**
Location: macro `STEPPE_CUDA_CHECK_KERNEL`, lines 129–146 (release branch 135–139).

*Verified against the docs.* NVIDIA documents `cudaGetLastError()` as: *"Returns the
last error that has been produced by any of the runtime calls in the same instance
of the CUDA Runtime library in the host thread and resets it to cudaSuccess,"* and
*"Note that this function may also return error codes from previous, asynchronous
launches"* (CUDA Runtime API, `group__CUDART__ERROR`, fetched 2026-06). So the
comment's claim (lines 129–134) that the macro surfaces "a bad launch configuration"
attributable to *this* `kernel<<<>>>` is true **only if the error state was clean
immediately before the launch** — i.e. only under the invariant "every prior runtime
call is checked." When true, the thrown `CudaError` carries `expr ==
"cudaGetLastError()"` and a `source_location` at the macro-expansion line (= the
launch line) — correct attribution. When a prior async fault went unretrieved, the
macro misattributes it to the wrong launch.

*Severity MED (not HIGH), confirmed,* because I re-read every current call site: in
`f2_block_kernel.cu` (234, 279), `f2_blocks_kernel.cu` (182, 236), and
`decode_af_kernel.cu` (112) each `STEPPE_CUDA_CHECK_KERNEL()` immediately follows the
launch inside a narrow `launch_xxx`/`run_xxx` wrapper, and all prior cuBLAS/runtime
calls in those wrappers route through `CUBLAS_CHECK`/`STEPPE_CUDA_CHECK`, so the error
state is provably clean. The defect is in the *comment's contract*, not current
behavior — but the comment states a stronger guarantee than the code delivers, which
will mislead the next kernel author who does *not* maintain the invariant. This is the
§7 "fail-fast, not fail-silent" contract written more strongly than honored.

Concrete fix: reword the comment to state that `cudaGetLastError()` reports the first
un-retrieved error since the last clear, attributable to *this* launch **only under
the maintained invariant that every runtime call is checked** (which the codebase
does maintain) — and note the §13 `CudaTest`-fixture practice of asserting a clean
error state in `TearDown` is the backstop. No release-branch code change is required;
the wording is the fix. (A `cudaPeekAtLastError` variant is *not* a fix: peeking would
not clear the sticky state and the next check would re-report it; the invariant is the
real mechanism.)

**C-2 [MED, effort S, before-M4.5: no] — Debug-mode `cudaDeviceSynchronize()` is device-wide, not stream-scoped; over-synchronizes and fights the §7 one-stream-per-lane idiom.**
Location: `STEPPE_CUDA_CHECK_KERNEL` debug branch, line 144.

*Verified against the docs.* `cudaDeviceSynchronize()` is documented as *"Wait for
compute device to finish"* — it blocks the host until *all* work on *all* streams of
the current device completes (CUDA Runtime API, `group__CUDART__DEVICE`, fetched
2026-06); `cudaStreamSynchronize(stream)` blocks only on the named stream.
Architecture §7 explicitly says "express cross-stream deps with `Event`, not
device-wide syncs," and §12 pins the statistic path to a single stream for cuBLAS
reproducibility. Today the backend uses the default stream (`nullptr`) only (TODO §A
"Default-stream debt": *"`cuda_backend.cu` uses `cudaStream_t stream_ = nullptr` …
every `cudaMemcpyAsync` is host-synchronous"*), so the behavior is currently
identical to a stream sync — **latent**, hence MED/before-M4.5:no. Once TODO §A wires
an owning `Stream` and M5 adds the copy lane, a debug-only device-wide sync after
every launch serializes the copy lane behind the compute lane and distorts the debug
timeline the profiler reads (§11.3 already warns `-G`/debug timing is
unrepresentative).

Concrete fix: give the macro an optional stream argument
(`STEPPE_CUDA_CHECK_KERNEL(stream)` defaulting to `0`/legacy default stream) and call
`cudaStreamSynchronize(stream)` in the debug branch. This narrows the sync to the
launch's lane (still gives compute-sanitizer correct fault attribution to THIS
launch — the kernel ran on that stream) and matches §7. Keep `cudaGetLastError()` in
both branches for the synchronous launch-config error.

**C-3 [LOW, effort S, before-M4.5: no] — `STEPPE_CUDA_CHECK_KERNEL` discards the kernel identity from the thrown message.**
*Confirmed (corollary of C-1, not a separate bug).* The macro checks
`cudaGetLastError()`/`cudaDeviceSynchronize()`, so the thrown `CudaError.expr` reads
`"cudaGetLastError()"` / `"cudaDeviceSynchronize()"` rather than the kernel name. The
`source_location` still localizes to the macro-expansion line, which is the launch
line, so the file:line is correct — but the `expr` string is unhelpful. Worth a
one-line comment that the kernel name is not captured (a `__func__`-style capture is
not worth the macro complexity). Low.

### (2) Edge cases & failure modes

**B-1 [HIGH, effort M, before-M4.5: yes] — The exception path allocates `std::string`, which can throw `std::bad_alloc` on the exact OOM path it reports — and the implicit copy constructor repeats the hazard.**
Location: `CudaError` ctor lines 42–49; same shape in `CublasError` ctor 63–69;
**plus the implicitly-declared copy constructors of both classes.**

*Confirmed and strengthened.* The header's own comment (lines 37–39) states the point
of carrying `status()` is to "distinguish `cudaErrorMemoryAllocation` →
STEPPE_ERR_DEVICE_OOM" (a §10 *recoverable* `DeviceOom` outcome). But the message is
assembled by concatenating `std::string` temporaries in the ctor body — a series of
heap allocations. Under host-memory pressure (a fragmented host heap under the
§11.1/§11.2 pinned-staging working set) those allocations can throw `std::bad_alloc`,
which propagates *instead of* the `CudaError`, destroying the typed device-OOM signal.
Device OOM (`cudaMalloc` failing) does not itself exhaust *host* memory, so the common
case is safe — but the failure mode is real precisely when the system is under host
memory stress.

**The first-pass strengthening, re-verified:** both classes hold a `std::string msg_`
and declare no copy/move members, so the compiler generates a copy constructor that
copies `msg_` — *another* heap allocation. `throw CudaError(...)` copy-initializes the
exception object (the copy may be elided, but elision is not guaranteed in all
rethrow / catch-by-value paths), and any `catch (CudaError e)` *by value* (or a future
translation layer that copies the exception to stash it) re-allocates on the same OOM
path. So fixing only the ctor body is insufficient: a robust fix must make the *whole
object* allocation-free on copy.

Concrete fix (preferred): store `status_` plus a fixed-size `char buf_[N]` filled by
`std::snprintf` (no heap), and have `what()` return `buf_`. `cudaGetErrorName`/
`cudaGetErrorString` return short, **non-null, bounded** strings (NVIDIA documents both
return a `char*` to a NULL-terminated string, *"'unrecognized error code' is returned"*
for unknown codes — verified, `group__CUDART__ERROR`, fetched 2026-06), the file and
expr are bounded, and `snprintf` truncates safely on overflow. **Note for the fix:**
`loc.function_name()` can be long (mangled/templated signatures run to hundreds of
chars), so size `N` generously — a named `constexpr kErrorMessageBufferBytes` (e.g.
512) per §4 — and accept that pathologically long signatures truncate (still safe).
This removes the ctor allocation *and* makes the implicit copy trivial/allocation-free
(a `char` array copies without heap), is faster on the error path, and keeps
`status()` intact. Add `<cstdio>`/`<cstddef>`, drop `<string>` (R-3). The alternative
(`try/catch(...)` around the string build, falling back to a static message) does
*not* fix the copy-constructor hazard, so the fixed-buffer design is the correct one.

**B-2 [LOW, effort S, before-M4.5: no] — `expr` is a borrowed `const char*` with no documented lifetime; the ctors are `public` and directly constructible.**
*Confirmed.* `#expr` is a string literal (static storage), so the borrow is always
valid via the two macros — the only sanctioned construction path. But the ctors are
`public` and reachable directly. They consume `expr` immediately into `msg_` (it is not
retained), so even a temporary `c_str()` would be safe today. Fix: a one-line invariant
comment ("`expr` is consumed in the constructor; not retained"), or make the ctors
`private` with `detail::cuda_check`/`cublas_check` as `friend`s (only they need to
construct — see R-5). Under the B-1 fixed-buffer rewrite the same holds — `expr` is
copied into `buf_` by `snprintf` and not retained.

**B-3 [LOW, effort S, before-M4.5: no — flagged for M5] — No distinction for `cudaErrorNotReady` (the query-API non-error status).**
*Confirmed latent.* `detail::cuda_check` throws on *any* `status != cudaSuccess`.
`cudaErrorNotReady` is a legitimate non-error return from `cudaStreamQuery`/
`cudaEventQuery`. Grep confirms no current call site uses query APIs (all are
`cudaMalloc`/`cudaMemcpy(Async)`/`cudaStream/EventCreate`/`cudaStreamSynchronize`/
`cudaEventRecord`/`cudaEventElapsedTime`/`cudaGetLastError`/`cudaDeviceSynchronize`),
so this is not active. M5's double-buffered pipeline (§11.1, TODO ⚡ item 2) polls
overlap with `Event`/`cudaEventQuery`, and routing that through `STEPPE_CUDA_CHECK`
would throw on "not ready yet." Worth a comment that `STEPPE_CUDA_CHECK` is for
fail-fast calls, not pollers; query APIs need a separate non-throwing path (this
dovetails with CAP-1/CAP-2 — the recoverable/non-fatal status path).

### (3) Numerical / precision (§12)

**N/A** — the unit performs no arithmetic; no accumulation order, no FP64-vs-Ozaki
concern, no determinism surface of its own. The only adjacency: the release branch
correctly omits any sync, so the macro never forces a host stall the bit-reproducible
hot path lacks; and the C-2 debug→stream-sync narrowing is determinism-neutral (it
moves no reduction). Explicitly N/A, verified against §12.

### (4) CUDA idioms / RAII / stream & async semantics / launch config / occupancy (§7)

**A-1 [MED, effort S, before-M4.5: no] — `CublasError::status_name` reinvents the official `cublasGetStatusName` and ignores `cublasGetStatusString`; the comment's premise is stale.**
Location: `CublasError::status_name`, lines 73–88.

*Confirmed against the docs.* The comment (lines 58–60, 73) and architecture §8
justify the hand-rolled switch by "cuBLAS has no `cudaGetErrorString` equivalent."
That is **false for the pinned toolchain.** cuBLAS provides
`const char* cublasGetStatusName(cublasStatus_t)` ("The string representation of the
status") and `const char* cublasGetStatusString(cublasStatus_t)` ("The description of
the status") — both verified in the current cuBLAS reference
(`docs.nvidia.com/cuda/cublas/index.html`, fetched 2026-06) and present back to at
least cuBLAS 11.4.4. steppe pins CUDA 13.1+ (§3), well within range. The hand-rolled
switch is a second source of truth (a §2 DRY violation) that must be hand-updated if
NVIDIA adds an enumerator, and it drops the human-readable description (P-1).

*Adversarial check — is the local switch justified?* Two legitimate reasons to keep a
local *adapter* (not a reimplementation): (a) `cublasGetStatusName` is **not documented
`noexcept`**, and `status_name` is `noexcept` *and is called from a `noexcept
destroy()` in `handles.hpp:75`* (re-verified by grep), so the replacement must stay
`noexcept`. `cublasGetStatusName` is a plain C function (no throwing contract), so
wrapping it in a `noexcept` function is safe in practice, but the docs do not *promise*
it — keeping the one-line adapter (rather than calling the API at every site)
localizes the `noexcept` assumption and the NULL guard. (b) NULL-for-unknown is
**undocumented** for `cublasGetStatusName` (the cuBLAS docs do not state the
unknown-code behavior — confirmed this pass; contrast `cudaGetErrorName`, which *is*
documented to return "unrecognized error code"), so the existing NULL guard is prudent
and must survive. Neither reason justifies *reimplementing the table*.

Concrete fix: replace the switch body with a delegating one-liner that preserves
`noexcept`, the NULL guard, and the call-site contract:
```cpp
const char* n = cublasGetStatusName(s);
return n ? n : "CUBLAS_STATUS_UNKNOWN";
```
Update the comment to cite the two official functions. This *strengthens* the DRY claim
(NVIDIA becomes the single source) and removes the hand-maintained table, while keeping
`handles.hpp`'s teardown caller compiling unchanged.

**A-2 [LOW, effort S, before-M4.5: no] — The checker's failure branch is not `[[unlikely]]`-hinted.**
*Confirmed as a nice-to-have, not a gate violation.* `detail::cuda_check`/`cublas_check`
are inlined on every CUDA/cuBLAS call in the device layer; the `throw` is essentially
never taken. `.clang-tidy` enables `performance-*` but does not force branch hints, so
this is optional. Marking `if (status != ...)` `[[unlikely]]` (C++20, portable)
documents intent and can keep the success path in icache. Low; cite §11.3 (optimize
what the profiler proves dominant — not proven dominant here, so optional).

**A-3 [LOW, before-M4.5: no — no action] — No `cudaPeekAtLastError` non-clearing variant.**
*Confirmed not a defect.* The throw-and-clear design is correct and simpler; a peek
variant is only an idiom affordance for inspect-without-consume, which nothing needs.
`cudaPeekAtLastError` exists (verified) but adding it would be speculative. No action.

**A-4 [LOW, effort S, before-M4.5: no] — `std::source_location::current()` in a default argument is a documented nvcc fragility; the whole device layer routes through this seam.**
Location: `detail::cuda_check`/`cublas_check` default args, lines 102–103, 108–109.

*Confirmed.* `std::source_location::current()` used as a default argument has a history
of nvcc trouble: **NVIDIA bug #4173735** (reported June 2023, CUDA 12.1 era) documents
nvcc failing to resolve `__builtin_source_location` in CUDA TUs — *"cannot call
non-constexpr function '__builtin_source_location'"* / *"identifier
'__builtin_source_location' is undefined"* — with a `consteval`→`constexpr`-style
workaround (NVIDIA Developer Forums thread "C++20's source_location compilation error
when using NVCC 12.1", fetched 2026-06). The *semantics* are correct (P1208R6: a
`current()` default argument captures the invocation point — see *Considered &
rejected*), and the pinned CUDA 13.1+ toolchain evidently compiles this (M0–M4 green on
the box). So this is **not a bug today** — it is a portability/fragility note: this is
the most-included header in `steppe_device`, leaning on a C++20 feature with a
documented nvcc gap. Recommendation: a one-line comment recording that
`std::source_location` in a default argument requires CUDA ≥ 13.1 (the project floor)
and was historically nvcc-fragile, so a toolchain bump must re-verify it; and the §13
unit test (T-1) should include a source_location-captures-caller assertion so a
regression is caught in CI rather than at a far-away call site. Low.

### (5) Magic numbers & hardcoded values (§4 / ROADMAP §4)

**Clean.** ROADMAP §4: "No literal may survive M0 except true mathematical constants."
This file has none: the `switch` cases are *named enum constants*, the strings are
diagnostic literals (not tunables), and `EXIT_FAILURE` correctly does not appear (the
file replaced the spike's `std::exit(EXIT_FAILURE)`, lines 11–15). The
`"CUBLAS_STATUS_UNKNOWN"` fallback (line 86) is a diagnostic string, not config —
acceptable; under A-1 it consolidates to one NULL-guard site. The B-1 fixed-buffer
size (`N`) would be the *one* new literal — it must be a named `constexpr` (e.g.
`kErrorMessageBufferBytes`) per §4, not a bare `512`. **No current magic-number
findings.**

### (6) Decomposition / single-responsibility / function size (§2)

**Good.** Each entity has one job: `CudaError`/`CublasError` carry a typed status +
message; `detail::cuda_check`/`cublas_check` test-and-throw; the three macros are thin
stringizing shells. No function exceeds ~10 lines; `status_name` is the longest (a pure
lookup, shrinking to one line under A-1). The split between public exception types and
`detail::` checkers is the right separation (the macros are the sanctioned construction
path; the checkers own the comparison; the exceptions own the message). The B-1
fixed-buffer rewrite keeps responsibilities clean (ctor captures
`status_`/`expr`/`loc` into the buffer; `what()` returns it). No single-responsibility
violations.

### (7) Readability, naming, const-correctness, [[nodiscard]]/noexcept, comment density

Strengths (verified):
- `[[nodiscard]]` on every accessor (`what`, `status`, `status_name` — lines 50–51,
  70–71, 74) and `noexcept` on all of them and on the `what()` override (which matches
  `std::exception::what`'s `noexcept`). `.clang-tidy`'s `bugprone-exception-escape`
  would flag a throwing `noexcept` function — `what()` only calls `c_str()` (noexcept,
  no alloc) and `status()` returns a scalar, so it is clean.
- Comment density matches the surrounding device headers (dense, arch-citing) and is
  genuinely informative.
- Macro naming is `UPPER_CASE`; matches the §8 table verbatim.

Findings:

**R-1 [MED, effort S, before-M4.5: yes] — Header comments make forward promises the tree does not keep.**
*Confirmed by grep.* Lines 36–39 / 57–60 reference `STEPPE_ERR_CUDA_RUNTIME` /
`STEPPE_ERR_DEVICE_OOM` (the C-ABI `steppe_status_t` names of §10/§16). The codebase's
*current* enum is `steppe::Status` (`include/steppe/error.hpp:21`, member `DeviceOom`
at :27) — the `STEPPE_ERR_*` C names exist **nowhere** in the tree yet (the C ABI is
Phase-3 work, §16). The `status()`→`Status` translator the comment implies **does not
exist** (grep for exception-`status()` consumers returns only the `handles.hpp`
`status_name` call, not a `status()`-on-exception mapping). Acceptable for M4, but the
comment states it as present fact. Fix: soften to "intended to be mapped by the public
C API (Phase 3, §10/§16) to `Status::DeviceOom` via `status() ==
cudaErrorMemoryAllocation`." Likewise the `catch (const CudaError&)` use (lines 13–15)
is real-but-unexercised: no `catch` site for either type exists in `src/` or `tests/`
(the production `STEPPE_CUDA_CHECK` is even shadowed by a local copy in
`test_f2_equivalence.cu:104`, so the test does not exercise this file's throw path).
Add "(no catch sites yet; API translation is Phase 3)" so the §13 testability claim is
not read as already-tested.

**R-2 [LOW, effort S, before-M4.5: no] — The spike file:line citations (lines 6–8) will rot.**
*Confirmed.* Lines 6–8 cite `f2_emu_spike.cu:182/193, f2_prec_acc.cu:24,
f2_timing.cu:25` — `experiments/` spike files slated for deletion (§4 "root strays …
slated for removal"; ROADMAP §1). Once removed, the line numbers dangle. Keep the
narrative ("replaces three duplicated macros in the spike"); drop the `:line` numbers.

**R-3 [LOW, effort S, before-M4.5: no] — Include set is IWYU-clean now but should be revisited with B-1.**
*Confirmed.* `<exception>` (base class), `<source_location>`, `<string>`
(`std::string`/`std::to_string`) are each justified for the current body; IWYU-clean as
written. Under the B-1 fixed-buffer rewrite, add `<cstdio>` (`std::snprintf`) and
`<cstddef>`, and `<string>` can be dropped. Flagged only so the include set is updated
alongside B-1.

**R-4 [LOW, effort S, before-M4.5: no] — the debug gating open-codes `#if defined(NDEBUG)` instead of the §7 single-source `STEPPE_DEBUG_ONLY`, and the device layer now has four independent copies of the idiom.**
Location: `STEPPE_CUDA_CHECK_KERNEL` gating, lines 135–146.

*Confirmed by grep.* Architecture §7 (the spec snippet at `architecture.md:460`)
defines the debug-gated post-launch sync via a single `STEPPE_DEBUG_ONLY(...)` macro:
`STEPPE_DEBUG_ONLY(STEPPE_CUDA_CHECK(cudaDeviceSynchronize()))`. `check.cuh` instead
open-codes `#if defined(NDEBUG) … #else … #endif`. Separately, `stream.hpp` (38–44) and
`handles.hpp` (29–35) each open-code their *own* `#if defined(NDEBUG)` teardown-warning
macro (`STEPPE_STREAM_WARN_ON_TEARDOWN`, `STEPPE_HANDLES_WARN_ON_TEARDOWN`), and
`device_buffer.cuh` (33–38) open-codes a third (`STEPPE_DEVICE_BUFFER_WARN_ON_TEARDOWN`).
`STEPPE_DEBUG_ONLY`/`STEPPE_ASSERT` and `internal/log.hpp` **do not exist** (grep: tokens
appear only in `architecture.md` prose and the cleanup docs). So the "is this a debug
build?" decision is currently re-expressed in **four** places across the device layer
with no single home — a §2/§8 DRY smell. TODO §B lists "triplicated teardown-warning
macros → one shared," but it does **not** capture that `check.cuh`'s `NDEBUG` gate is the
*same* single-source concern, nor that the §7 spec already names the canonical macro
(`STEPPE_DEBUG_ONLY`). Fix: when `internal/log.hpp` lands (it owns the teardown-warning
sink per §8/§10), introduce one `STEPPE_DEBUG_ONLY` (and the warning facade) in a single
device-visible header and have `check.cuh`, `stream.hpp`, `handles.hpp`,
`device_buffer.cuh` all consume it. Until then this file's local `#if` is consistent
with its siblings, so it is LOW — but track it as part of the same consolidation, not
left implicit.

**R-5 [LOW, effort S, before-M4.5: no] — NEW: the exception classes are not `final` and rely entirely on implicit special members, including the implicit copy ctor folded into B-1.**
Location: `CudaError` (40–56), `CublasError` (61–93).

*New this pass.* Neither class is `final`; neither is designed for further derivation
(they are concrete typed exceptions), so marking both `final` documents intent and lets
the compiler devirtualize `what()`/`status()` at known-type call sites — a zero-cost
clarity win. More substantively, the *only* user-declared member of each class is the
converting constructor; the destructor, copy/move ctors, and copy/move assignment are
all implicit. That is intentional and `cppcoreguidelines-special-member-functions` does
**not** fire (no user-declared dtor/copy/move — confirmed against the rule), so this is
**not** a rule-of-five violation (see *Considered & rejected*). The one implicit member
that *matters* is the copy constructor, whose `std::string`-copy allocation is the
second OOM-path hazard already captured in B-1. Recommendation: mark both classes
`final`; under the B-1 fixed-buffer rewrite the implicit copy becomes trivial, at which
point the special-member story is clean and self-documenting. Low; pairs with B-1.

### (8) Performance

**Largely N/A, and what exists is correct.** The success path is the only hot path:
`if (status != cudaSuccess) return;` inlined at every call site — minimal. The failure
path's string cost (B-1) is irrelevant to throughput (it runs once). The genuine perf
notes are A-2 (`[[unlikely]]` on the failure branch) and the B-1 fixed-buffer rewrite
(a small error-path win — no heap). The release `STEPPE_CUDA_CHECK_KERNEL` correctly
omits the sync (no host stall on the hot loop, §7) — the single most important perf
decision here, and it is right. The device-wide debug sync (C-2) is a debug-only perf
concern. No further performance findings.

### (9) Layering / API / ABI (§4 / §16)

- **Correct:** CUDA-private header, never crosses into `core`/`api`/CLI (verified by the
  include graph — only device-layer TUs include it).
- **No exceptions across the ABI:** the typed exceptions are internal; §10/§16 require
  exceptions never cross the public C boundary. This file is the *source* of those
  exceptions, consumed internally; the Phase-3 translator that catches and maps them to
  `steppe_status_t` is R-1's referenced future work. Consistent.
- **L-1 [MED, effort M, before-M4.5: no] — `CUSOLVER_CHECK` promised by §7/§8/§17 is
  absent.** *Confirmed by grep* (no `cusolver` include, no `CusolverError`, zero
  `CUSOLVER_CHECK` in the tree). Fine at M4 — cuSOLVER first appears at S5/S6 (Phase 2).
  When it lands it must mirror this file: a `CusolverError` with `status()` and a
  status-to-string. **Doc note for that future work:** unlike cuBLAS, cuSOLVER
  historically has *no* official status-to-string function, so the sibling will
  legitimately need a hand-rolled switch (or a cuSOLVER-provided helper if one has since
  shipped) — verify against the then-current cuSOLVER docs before assuming the cuBLAS
  delegating-adapter pattern transfers. Layering/contract gap, not a defect.

### (10) Testability (§13)

**T-1 [MED, effort M, before-M4.5: no] — The error-checking unit has no direct unit test, and the one .cu test shadows the production macro.**
*Confirmed.* §13 wants failure paths tested as values, not crashes, and "obviously
correct" seams pinned. The throwing behavior and message format are untested. Worse,
`test_f2_equivalence.cu:104,115` *redefines* `STEPPE_CUDA_CHECK`/`CUBLAS_CHECK` locally
(its comment at 98–102 says so), so the production header's throw path is exercised by
**nothing** in `tests/`. Most of this is trivially CPU-testable (construct
`CudaError`/`CublasError` with a chosen status, assert `status()` round-trips,
`status_name`/(post-A-1) the official-name passthrough maps each enumerator, `what()`
contains the expr and a `file:line`). The source_location-captures-caller property (and
the A-4 nvcc-fragility) is worth one host assertion via a tiny wrapper calling
`detail::cuda_check(cudaSuccess, ...)` and an error status. Fix: add
`tests/unit/test_cuda_check.cpp` (host, no GPU); it guards A-1's
switch→`cublasGetStatusName` swap and B-1's fixed-buffer rewrite against regressions.

**T-2 [LOW — no action] — `status_name` is `static` + pure → trivially testable; a point in favor.** This is why T-1 is MED not HIGH.

### (11) Capability tiers (TODO ⚡ / `wxz1fiiln`)

The unit is **capability-tier-agnostic by nature** — error checking is identical on the
PRO-6000 capable path and the budget-5090 fallback (§ TODO `wxz1fiiln`: "Parity:
bit-identical on both paths … levers are data-movement/observability only"). But two
hooks belong here because the capability story (TODO: "a capability probe +
capability-tagged results … every run records which path it took + why it degraded")
needs a *non-fatal, tagged* diagnostics path distinct from the fatal throwing checks
here — and routing a graceful-degradation probe through the throwing checker would
convert a documented degrade point into a hard failure.

**CAP-1 [MED, effort M, before-M4.5: yes] — No `STEPPE_LOG_*`/non-throwing warn path; every capability "explicit logged tag" the tiers mandate has nowhere to go.**
*Confirmed by grep* (only doc-comments reference `internal/log.hpp`; the three RAII
sibling headers each roll a private `STEPPE_*_WARN_ON_TEARDOWN` `fprintf` stub "until
log.hpp lands" — see R-4). The TODO capability table requires explicit logged degrade
reasons (verbatim: *"P2P combine unavailable (no peer access) → host-staged fixed-order
combine"*; *"GDS unavailable (GeForce GPU-class / OverlayFS) → POSIX pread into pinned
double-buffer"*), routed via `STEPPE_LOG_WARN` (§7/§10) — which does not exist yet. Not
a defect *of check.cuh*, but check.cuh is the closest existing "diagnostics" home, and
the capability work (M4.5) needs a `STEPPE_CUDA_WARN(expr)`-style non-throwing variant
for *recoverable* statuses. Recommend this variant land next to the throwing checks when
`internal/log.hpp` arrives.

**CAP-2 [LOW, effort S, before-M4.5: yes] — Document that capability probes must not route through the throwing checker.**
*Confirmed against the TODO.* M4.5's combine adds *"the optional `canAccessPeer`-gated
P2P device-combine (host-staged stays the baseline)"* (TODO "Per-milestone design
changes"): `cudaDeviceCanAccessPeer` returning "cannot" and a subsequent
`cudaDeviceEnablePeerAccess` returning `cudaErrorPeerAccessAlreadyEnabled` are *expected*
outcomes on the budget box. If M4.5 wraps them in `STEPPE_CUDA_CHECK`, the budget-5090
path throws where it should log-and-degrade. A two-line comment now ("`STEPPE_CUDA_CHECK`
is for *fault* calls only; capability probes use the CAP-1 non-throwing path") prevents a
foot-gun in the milestone the TODO flags as calcifying current patterns. Overlaps B-3
(query-API statuses) — both want a sanctioned non-fatal status path.

---

## Considered & rejected

- **"`std::source_location` default-argument usage is wrong / won't capture the
  caller."** Rejected. The checkers take `const std::source_location& loc =
  std::source_location::current()` and the macros call `cuda_check((expr), #expr)`
  *without* supplying `loc`, so the default is evaluated at the macro-expansion = user's
  call site. P1208R6 specifies that `current()` in a default argument captures the
  *invocation* point (the caller), which is the intent (the header comment lines 97–99
  describe this correctly). No bug. *(The nvcc toolchain fragility of this pattern is a
  real, separate note — Finding A-4.)*
- **"Passing `std::source_location` by `const&` default-argument dangles."** Rejected.
  The default-constructed temporary lives for the full call expression; `loc` is consumed
  entirely within the ctor body before the temporary dies. No lifetime issue.
- **"`cudaGetErrorName`/`cudaGetErrorString` could return NULL and crash the
  `std::string` concatenation."** Rejected. NVIDIA documents both return a non-null
  NULL-terminated string (*"'unrecognized error code' is returned"* for unknown codes —
  verified, `group__CUDART__ERROR`, fetched 2026-06). The `CudaError` message build is
  NULL-safe. (Contrast `cublasGetStatusName` under A-1, where unknown-code behavior is
  *undocumented* — hence the NULL guard there.)
- **"The `switch` in `status_name` is missing enumerators."** Rejected for correctness
  today: the switch lists the 10 cuBLAS status enumerators and the `default` returns
  `"CUBLAS_STATUS_UNKNOWN"` (no UB) for any future addition. The objection is maintenance
  (A-1), not correctness.
- **"`noexcept` on `what()` is unsafe because `msg_.c_str()` could throw."** Rejected.
  `std::string::c_str()` is `noexcept` and does not allocate; returning it from a
  `noexcept what()` is standard-conforming (`bugprone-exception-escape` would otherwise
  flag it — it does not).
- **"`what()` could dangle after a move of the exception object."** Rejected. The
  implicit move ctor moves `msg_` into the new exception object; `what()` returns
  `msg_.c_str()` of the *live* object, which owns its buffer. No dangling on the
  throw/catch path. (Under the B-1 fixed-buffer rewrite the buffer is an inline array,
  so the question disappears entirely.)
- **"Throwing from an `inline` function in a header violates no-exceptions-across-ABI."**
  Rejected. The throw is internal to `steppe_device`; the ABI rule (§10/§16) governs the
  *public C boundary*, which translates these to `steppe_status_t`. Inline throwing here
  is the documented design.
- **"`CUBLAS_CHECK` is missing the `STEPPE_` prefix (inconsistent with
  `STEPPE_CUDA_CHECK`)."** Rejected. §8's table, §7, and ROADMAP §5 all spell it
  `CUBLAS_CHECK` (no prefix). The unprefixed name matches the spec verbatim; diverging
  would be the inconsistency. Intentional.
- **"The expression-checker macros should `do { } while(0)`-wrap like the kernel
  macro."** Rejected. `STEPPE_CUDA_CHECK(expr)` expands to a single function-call
  expression — already a valid statement, safe in braceless `if`/`else`. The `do/while`
  wrap is needed only for the multi-statement kernel macro, which uses it.
- **"`detail::cuda_check` should be `[[nodiscard]]`."** Rejected. It returns `void`; its
  effect is the throw.
- **"The checkers should be in `::steppe::detail`, not `::steppe::device::detail`, to
  match the §7 spec snippet (`architecture.md:456`)."** Rejected (doc is stale, code is
  correct). The §7 snippet writes `::steppe::detail::cuda_check`, but the file lives in
  the device layer and is `PRIVATE` to `steppe_device` (§4), so `steppe::device::detail`
  is the *more* layering-faithful placement, and the macros reference it consistently.
  The spec snippet is illustrative pseudocode (it also omits the `#expr` argument the
  real macro passes). No change to code; if anything, the `architecture.md` snippet
  should be reconciled to the implementation.
- **"`CudaError`/`CublasError` need a `virtual` destructor / rule-of-five."** Rejected.
  They inherit `std::exception`'s virtual destructor; declaring only a constructor + a
  `what()` override does not trip `cppcoreguidelines-special-member-functions` (no
  user-declared dtor/copy/move). The implicit copy ctor is the only special member of
  concern, and that is a substantive finding folded into B-1 (allocation on copy), with
  the `final`/intent angle in R-5 — not a rule-of-five style nit.
- **"`STEPPE_CUDA_CHECK(expr)` will break on a function call containing commas."**
  Rejected as a *bug*, downgraded to a polish note (S-1). The preprocessor treats commas
  inside the function-call parentheses as part of the single `expr` argument, so
  `STEPPE_CUDA_CHECK(cudaMemcpy(a, b, c, d))` is one macro argument — safe. Only a *top
  level* comma (which never occurs for a single function call) would break. The variadic
  hardening is captured as S-1, not a correctness defect.

---

## What it takes to reach 10/10

In rough priority (the first four are the substantive ones):

1. **Make the whole exception object OOM-safe (B-1)** — store `status_` + a
   named-`constexpr`-sized `char buf_[kErrorMessageBufferBytes]` filled by
   `std::snprintf`; `what()` returns the buffer. This removes the ctor allocation *and*
   makes the implicit copy constructor allocation-free, so the device-OOM `status()`
   signal can never be masked by a `std::bad_alloc` on construction *or* copy. Size the
   buffer for long mangled `function_name()`s; truncation is safe. Keep `status()`
   intact.
2. **Replace `status_name`'s hand-rolled switch with `cublasGetStatusName` and add
   `cublasGetStatusString` to the message (A-1, P-1)** — one genuine source of truth,
   richer diagnostics (name + description, parity with the `CudaError` path), no
   hand-maintained table; preserve `noexcept` + the NULL guard for the `handles.hpp:75`
   teardown caller.
3. **Fix the release `STEPPE_CUDA_CHECK_KERNEL` comment/contract (C-1)** — state
   accurately that `cudaGetLastError()` reports the first un-retrieved error since the
   last clear, attributable to *this* launch only under the every-runtime-call-is-checked
   invariant the codebase maintains (and the §13 `TearDown` clean-state backstop).
4. **Stream-scope the debug sync (C-2)** — give the kernel macro an optional stream arg,
   use `cudaStreamSynchronize` instead of device-wide `cudaDeviceSynchronize`, before M5
   wires the copy lane.
5. **Add `tests/unit/test_cuda_check.cpp` (T-1)** — host-only assertions on `status()`,
   the cuBLAS name mapping, `what()` content, and the source_location-captures-caller
   property (also a regression guard for the A-4 nvcc fragility and the A-1/B-1
   rewrites). The .cu test currently *shadows* the production macro, so this header's
   throw path is otherwise untested.
6. **Correct the forward-looking comments (R-1, R-2)** — soften the
   `STEPPE_ERR_*`/`status()`-mapping and `catch`-site language to "intended (Phase 3)";
   drop the rotting spike `:line` numbers.
7. **Consolidate the debug gate to one `STEPPE_DEBUG_ONLY` (R-4)** — when
   `internal/log.hpp` lands, replace this file's `#if defined(NDEBUG)` (and the four
   sibling teardown macros) with the §7 single-source macro + warning facade.
8. **Document the fault-only contract + add a non-throwing capability-probe/query path
   (CAP-1, CAP-2, B-3)** — capability probes (peer access, GDS) and pollers
   (`cudaEventQuery`) must NOT route through `STEPPE_CUDA_CHECK`; pair with the
   `STEPPE_LOG_*` facade when it lands.
9. **Land `CUSOLVER_CHECK` (L-1)** at Phase 2, mirroring this file's shape (verify the
   then-current cuSOLVER status-to-string availability first — historically absent).
10. **Minor polish:** `[[unlikely]]` on the failure branch (A-2); mark both exception
    classes `final` and document the implicit-special-member story (R-5); harden the
    expression-checker macros to variadic `(...)`/`__VA_ARGS__` (S-1); document the
    `expr` borrow / consider making the ctors `private` with the checkers as `friend`
    (B-2); add the A-4 nvcc/source_location toolchain-floor comment; note the kernel name
    is not captured by the kernel macro (C-3).

### Category-(7) findings folded into the above

**P-1 [MED, effort S, before-M4.5: no] — `CublasError` drops the human-readable
description.** *Confirmed.* The `CudaError` message ends with both name
(`cudaGetErrorName`) and description (`cudaGetErrorString`); `CublasError` ends with
only the symbolic name. cuBLAS *does* provide a description via `cublasGetStatusString`
("The description of the status" — verified). Appending it gives `CublasError::what()`
parity with `CudaError::what()` (`… -> name : description`). Folds into A-1's refactor.

**S-1 [LOW, effort S, before-M4.5: no] — NEW: the expression-checker macros take a fixed
single `expr` parameter rather than variadic `(...)`.** *Not a bug today* (commas inside
a function-call's parens are one macro argument; only a top-level comma would break, and
none occurs). But making `STEPPE_CUDA_CHECK(...)` / `CUBLAS_CHECK(...)` variadic
(forwarding `__VA_ARGS__`, stringizing the whole `#__VA_ARGS__`) future-proofs them
against any expression with a top-level comma and costs nothing. Low; bundle with the
A-1/B-1 edits.

---

## Good patterns to keep

- **Typed exceptions carrying `status()`, not `exit()`** (lines 11–15, 40–56, 61–93) —
  the single most important upgrade over the spike's `std::exit`; enables the §10
  recoverable-vs-fault taxonomy (`cudaErrorMemoryAllocation` → `DeviceOom`) and
  `catch`-based tests (§13). Preserve.
- **`std::source_location` default-argument instead of `__FILE__`/`__LINE__` plumbing**
  (lines 97–111) — clean, correct, macro-free location capture; the macros pass only
  `#expr`. Textbook C++20 (P1208R6-correct). Preserve and replicate in the future
  `CUSOLVER_CHECK` (with the A-4 toolchain-floor note).
- **`detail::` namespace for the checkers + public exception types** — the macros are
  the sanctioned construction path; the comparison logic is one place; the exceptions own
  their message. Clean §2 separation.
- **`NDEBUG`-gated debug-only async sync in `STEPPE_CUDA_CHECK_KERNEL`** (lines 135–146)
  — release omits the host-stalling sync on the hot loop (§7/§11.3); debug forces fault
  attribution under compute-sanitizer (§13). The gating *decision* is exactly right; only
  the *scope* of the sync (C-2) needs narrowing and the gate's *single-sourcing* (R-4)
  needs consolidating.
- **The post-launch macro surfaces a bad launch config synchronously** — `cudaGetLastError()`
  in both branches catches `cudaErrorInvalidConfiguration` (e.g. the deferred `grid.z >
  65535` clamp TODO §B lists) at the launch site, the correct safety net for the
  per-block batched launches in `f2_blocks_kernel.cu`. Keep.
- **`[[nodiscard]]` + `noexcept` discipline** on every accessor — consistent and correct,
  including the `what()` override (clean under `bugprone-exception-escape`).
- **Dense, architecture-citing comments** that explain *why* (DRY single-home, why cuBLAS
  gets a sibling, why throw-not-exit) — matches the surrounding device headers and the
  project's documentation culture. Keep (just keep them *accurate* per R-1/R-2 and
  *current* per A-1's stale "no cuBLAS equivalent" premise).
- **CUDA-private placement** (`.cuh`, includes `<cuda_runtime.h>`/`<cublas_v2.h>`)
  enforcing §4 layering by construction. Keep.

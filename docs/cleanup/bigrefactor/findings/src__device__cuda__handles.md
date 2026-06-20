# Review findings — src__device__cuda__handles

Files: /home/suzunik/steppe/src/device/cuda/handles.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/device/cuda/handles.hpp:476 — `CusolverMathModeScope::promoted()` accessor (and the `promoted_` member it returns, line 517) is read by nothing in src/ or tests/ (grep finds no call site; doc says "for tests/logging"). It is the observable-state surface of the §6 fit-solve promotion seam deliberately landed ahead of its consumer, so this is scaffold-not-wired-yet rather than truly dead. Suggested: leave (intentional seam API), or add the test/log read that justifies it; flag only if the seam consumer is dropped.
- [3.3][LOW] src/device/cuda/handles.hpp:339 — `CusolverDnHandle::device_id()` is never read (the test at tests/reference/test_handles.cu:185 exercises only `CublasHandle::device_id()`); it mirrors the cuBLAS accessor as M4.5 multi-GPU scaffold. Suggested: keep for symmetry with the documented §11.4 multi-GPU use, or add the matching cuSOLVER device-ordinal assertion in test_handles.cu.

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

## Group 6 — Naming

No Group 6 issues found.

## Group 7 — Duplication

- [7.1][LOW] src/device/cuda/handles.hpp:179-191, 348-358 — `assert_on_creation_device()` is copy-pasted verbatim between `CublasHandle` and `CusolverDnHandle`: identical `STEPPE_DEBUG_ONLY(cudaGetDevice + two STEPPE_ASSERTs against device_id_)` body differing only by the two assertion message strings ("CublasHandle"/"cuBLAS §2.1.2" vs "CusolverDnHandle"/"cuSOLVER context binds..."). Suggested: hoist a `detail::assert_current_device(int created_on, const char* what)` free helper both call, or a tiny CRTP/base mixin; behavior-preserving (debug-only, parity-neutral §12).
- [7.4][LOW] src/device/cuda/handles.hpp:193-209, 278-291, 359-368, 501-513 — the teardown-warning idiom (`if (h_) { const ...Status_t s = <destroy/restore call>; if (s != ..._SUCCESS) STEPPE_LOG_WARN(<sink>, status_name(s)); } h_ = nullptr;`) is repeated 4x across `CublasHandle::destroy`, `MathModeScope::restore`, `CusolverDnHandle::destroy`, `CusolverMathModeScope::restore`, differing only by the API call, the success enum, the message text, and the `*Error::status_name` mapper. Suggested: a `detail::warn_on_nonzero(status, success_value, msg, name_fn)` helper (or a small macro) folds the §7 "fail-fast not fail-silent" teardown shape into one place; the four sites then differ only by their one call expression.
- [7.4][LOW] src/device/cuda/handles.hpp:114-129, 260-270, 318-327, 455-465 — move-ctor + move-assign boilerplate (`std::exchange(o.h_, nullptr)` plus copying the trivial trailing members, and the `if (this != &o) { destroy()/restore(); ...; }` assign body) is duplicated across all four move-only classes. This is the standard move-only-handle pattern; folding it would need a CRTP/base or a generic owning-handle template. Suggested: OPTIONAL — extract a `detail::MoveOnlyHandle<HandleT, Deleter>` base (or leave as-is, since the four classes carry slightly different extra members ws_/ws_bytes_/device_id_/prev_/promoted_ and the explicitness is intentional); low priority.
- [7.1][LOW] src/device/cuda/handles.hpp:250-295, 443-518 — `MathModeScope` (cuBLAS) and `CusolverMathModeScope` (cuSOLVER) are structurally the same RAII get/apply/restore-math-mode scope, differing by the handle type, the `cublas*`/`cusolver*` get/set calls, the math-mode enum type, and the extra honorability/`promoted_` logic on the cuSOLVER side. The cuBLAS-vs-cuSOLVER API split (separate vendor enums + functions) genuinely prevents a clean merge, and the cuSOLVER scope carries promotion-seam state the cuBLAS one lacks. Suggested: leave as two types (the divergence is real and the duplication is bounded); a template parameterized on a traits struct is possible but would obscure more than it saves at only two instantiations.

## Group 8 — Comments

No Group 8 issues found.

## Group 9 — Constants & configuration

- [9.3][LOW] src/device/cuda/handles.hpp:450 — `CusolverMathModeScope(cusolverDnHandle_t, bool honorable)` takes a bare positional `bool`; a direct `CusolverMathModeScope(h, true)` reads opaquely at the call site (which `true`?). Mitigated in practice: the public construction path is `engage_solver_precision` (line 530) which derives the flag from a typed `Precision` via the `emulation_honorable` predicate, and the param is at least named `honorable`. No multi-bool `foo(true,false)` cluster exists. Suggested: OPTIONAL — a 2-value enum (e.g. `enum class Honorability { Native, Emulated }`) would make any future direct construction self-documenting; low priority since the typed factory is the intended entry point.

## Group 10 — Initialization

No Group 10 issues found.

## Group 13 — Error handling

No Group 13 issues found.

(13.1) Every CUDA/cuBLAS/cuSOLVER API return in this header is checked: the fault-path calls route through the throwing CHECK macros — `cudaGetDevice`+`cublasCreate` (handles.hpp:110-111), `cublasSetWorkspace` (146), `cublasSetStream`+re-apply (158-159), `cublasGetMathMode`/`cublasSetMathMode` (256-257), `cudaGetDevice`+`cusolverDnCreate` (315-316), `cusolverDnSetStream` (336), `cusolverDnGetMathMode`/`cusolverDnSetMathMode` (451-452). The teardown/restore calls (`cublasDestroy` 198, `cublasSetMathMode` restore 284, `cusolverDnDestroy` 361, `cusolverDnSetMathMode` restore 506) deliberately do NOT throw — they inspect the status and route a nonzero value to STEPPE_LOG_WARN (the documented architecture.md §7 "dtor never throws / fail-fast not fail-silent" teardown discipline), so the error is reported, not swallowed. The debug-only `cudaGetDevice` in `assert_on_creation_device` (184, 351) is guarded by `STEPPE_ASSERT(e == cudaSuccess, ...)` inside `STEPPE_DEBUG_ONLY` — appropriate for a debug-only probe.
(13.2) No kernel `<<<...>>>` launches in this header — N/A.
(13.3) Checking is consistent — all fault paths CHECK, all teardown paths warn; no mixed guarded/unguarded sibling calls.
(13.4) No error-swallowing macro: the CHECK macros (check.cuh) throw typed source-located exceptions and STEPPE_CUDA_WARN logs-and-yields the status; neither hides errors. The teardown warn idiom reports the failing status name.

## Group 14 — Memory: allocation & lifetime

- [14.4][LOW] src/device/cuda/handles.hpp:142-147, 206-208 — `set_workspace(ptr,bytes)` binds a NON-owning workspace span via `cublasSetWorkspace`, and `destroy()` only clears `ws_`/`ws_bytes_` (never frees). cuBLAS GEMMs issued on the single statistic stream use that VRAM asynchronously, so freeing the backing `DeviceBuffer` before those GEMMs complete is a use-after-free the handle CANNOT detect — correctness rests entirely on the documented "backend declares the handle BEFORE the workspace buffer so reverse-order destruction frees the buffer after the handle is destroyed" contract (lines 139-141) plus `cublasDestroy`'s implicit sync (line 7). This is an unenforceable lifetime invariant living in a comment, not a bug in this file. Suggested: leave (correct as documented), but audit the actual backend declaration order at the call site to confirm the handle is declared before its workspace `DeviceBuffer`; consider a debug-only note/assert tying the two together if the seam is ever reordered.

(14.1) No alloc/free mismatch: the four classes own cuBLAS/cuSOLVER *contexts* (cublasCreate/cublasDestroy 111/198, cusolverDnCreate/cusolverDnDestroy 316/361) and *math-mode* state (sticky handle state, get/set/restore — not an allocation), all correctly paired. No cudaMalloc/new[]/malloc anywhere in the header; the only buffer (`ws_`) is explicitly non-owning and never freed here (206-208).
(14.2) No stream-ordered alloc concern: handles are created ONCE at startup and reused (header comment line 5-7); no cudaMalloc/cudaMallocAsync on any hot path in this file. N/A.
(14.3) No cudaMallocAsync/cudaFreeAsync in this header — async/sync free pairing N/A.
(14.5) No missing frees on error paths: in both ctors (CublasHandle 109-112, CusolverDnHandle 314-317) the only throwing call after a successful create is none — `cudaGetDevice` precedes the create, and if `cublas/cusolverDnCreate` itself throws via the CHECK macro no handle was acquired (h_ stays nullptr), so the half-constructed object's dtor `destroy()` is a guarded no-op. The two MathModeScope ctors (255-258, 450-453) acquire no allocation (math mode is restorable sticky state, captured into `prev_` BEFORE the set); a throwing set leaves nothing to release. No acquire-then-leak on any throw path.

## Group 15 — Memory: transfers

No Group 15 issues found.

(15.1) No `cudaMemcpy`/`cudaMemcpyAsync` anywhere in this header — the four classes own cuBLAS/cuSOLVER *contexts* and *math-mode* sticky state; there are no host<->device data transfers, in a loop or otherwise. The handles are created ONCE at startup and reused (header comment lines 5-7), and the only device pointer touched is the non-owning workspace `ws_` bound via `cublasSetWorkspace` (handles.hpp:146, 159) — a pointer hand-off, not a copy. N/A.
(15.2) No transfers ⇒ no `cudaMemcpyKind` direction enum to mismatch. N/A.
(15.3) No host allocations and no frequent transfers in this header — nothing is pinnable/transferred. The only host-side state (`ws_bytes_`, `device_id_`, `prev_`, `promoted_`) are scalar members consumed on the host, never DMA'd. N/A.

## Group 16 — RAII: ownership & wrapper hygiene

No Group 16 issues found.

(16.1) Every library resource in this header is wrapped. The two owning context handles are RAII: `CublasHandle` pairs `cublasCreate` (handles.hpp:111) with `cublasDestroy` (198); `CusolverDnHandle` pairs `cusolverDnCreate` (316) with `cusolverDnDestroy` (361). The two sticky-state resources are RAII scopes: `MathModeScope` captures via `cublasGetMathMode` (256) and restores via `cublasSetMathMode` (284); `CusolverMathModeScope` captures via `cusolverDnGetMathMode` (451) and restores via `cusolverDnSetMathMode` (506). Streams/events/buffers are NOT owned here — they live in `Stream`/`DeviceBuffer` wrappers declared in the backend (cuda_backend.cu:2661,2672); the only device pointer this header touches, the cuBLAS workspace `ws_` (211-212), is DELIBERATELY non-owning (documented 139-141; cleared not freed in destroy() 206-208) and is owned by the backend's `workspace_` `DeviceBuffer`. No unwrapped *Create/*Destroy, pool, array, texture, or pinned-host resource appears in this file.
(16.2) All four classes are move-only with null-on-move. Copy ctor + copy assign are `= delete` on every class (CublasHandle 131-132, MathModeScope 272-273, CusolverDnHandle 328-329, CusolverMathModeScope 467-468). Every move ctor / move assign nulls the moved-from handle via `std::exchange(o.h_, nullptr)` (114-118, 120-129, 260-261, 263-270, 318-319, 320-327, 455-456, 458-465), and move-assign self-guards (`if (this != &o)`) and frees/restores the existing resource first (destroy() 122 / restore() 265,460). A moved-from object has `h_ == nullptr`, so its destroy()/restore() is a guarded no-op — no double-free, no double-restore.
(16.3) Rule of five satisfied on every freeing/restoring type: dtor (134,275,330,470) + move-ctor + move-assign + deleted copy-ctor + deleted copy-assign, all five present on each of the four classes.
(16.4) Single, clear ownership. The owning handles are held BY VALUE as backend members (`CublasHandle blas_` cuda_backend.cu:2662, `CusolverDnHandle solver_` 2671) — never passed by value into the wrappers. The math-mode scopes take the RAW handle type (`cublasHandle_t`/`cusolverDnHandle_t`, handles.hpp:255,450,531) as documented non-owning views (293,515) so they compose with `engage_f2_precision`/`engage_solver_precision` without aliasing ownership; their restore() touches only sticky state, never destroys the context. Raw `get()` (162,338) returns a non-owning view handed only to compute entry points. No owning wrapper is copied or passed by value.
(16.5) The wrapper is genuinely needed, not reinvented: cuBLAS/cuSOLVER context handles and sticky math-mode state have no `thrust`/`unique_ptr`+deleter stdlib equivalent (they are opaque library handles, not device allocations), and the type additionally owns load-bearing invariants a generic deleter could not — the (stream,workspace) re-apply (156-160) and the device-ordinal record-and-assert (179-191,348-358). A `unique_ptr` with a custom deleter would lose those.

## Group 17 — RAII: lifetime & deleter pitfalls (CUDA-specific)

- [17.5][MED] src/device/cuda/handles.hpp:193-209, 359-368 — `CublasHandle::destroy()`/`CusolverDnHandle::destroy()` call `cublasDestroy(h_)`/`cusolverDnDestroy(h_)` WITHOUT making `device_id_` current, and the class deliberately forbids the wrapper from ever calling `cudaSetDevice` (lines 35-37, 176, 347). The `assert_on_creation_device()` device guard is invoked from `set_workspace`/`set_stream` (143,157,335) but NOT from `destroy()` — so the destroy path is entirely unguarded as to the current device. The backend that owns these handles has no explicit destructor and does not `guard_device()` at teardown (cuda_backend.cu:2599 `guard_device` is called only on compute entries, never on destruction), so under M4.5 multi-GPU the per-device backend's `blas_`/`solver_` are destroyed with whatever device is current — possibly a DIFFERENT GPU than `device_id_`. cuBLAS/cuSOLVER `Destroy` carries its own context device, so on the box today this is tolerated; the documented 17.5 record-and-assert discipline (current==device_id_ on every cuBLAS-context mutation) is nonetheless silently violated by the teardown call, with no debug assert to catch a future driver/toolkit that does mind the current device. Suggested: at minimum add the debug-only `assert_on_creation_device()` to `destroy()` (consistent with set_workspace/set_stream); or, since the design rule bars `cudaSetDevice` in the wrapper, document that backend teardown MUST run with `device_id_` current and add a debug assert in the backend dtor — keep the wrapper `cudaSetDevice`-free (parity-neutral, §12 observability-only).
- [17.1][LOW] src/device/cuda/handles.hpp:193-209, 278-291, 359-368, 501-513 — all four dtors route through `noexcept` `destroy()`/`restore()` that swallow a nonzero cuBLAS/cuSOLVER status to `STEPPE_LOG_WARN` (never throw during unwinding) — correct per the 17.1 rule. One residual teardown-order note: `MathModeScope`/`CusolverMathModeScope::restore()` (284,506) call `cublasSetMathMode`/`cusolverDnSetMathMode` on a NON-owning handle (`h_` is a raw `cublasHandle_t`/`cusolverDnHandle_t`, 293,515). These scopes are stack-local and constructed strictly inside a live owning handle's lifetime, so they always destruct before the owning `CublasHandle`/`CusolverDnHandle` (no use-after-destroy in the intended usage). If a scope's lifetime were ever extended past its owning handle (e.g. stored as a class member outliving the handle, or moved out and held), `restore()` would touch a destroyed context; correctly swallowed to a warn, but still wrong-context. Suggested: leave (correct under documented stack-scoped usage); if a scope is ever promoted to a long-lived member, re-audit that the owning handle outlives it.

(17.2) No allocator/deleter mismatch in this header: the only resources owned are cuBLAS/cuSOLVER *contexts* (`cublasCreate`/`cublasDestroy` 111/198, `cusolverDnCreate`/`cusolverDnDestroy` 316/361) and sticky *math-mode* state (get/set/restore — not an allocation). There is NO `cudaMalloc`/`cudaMallocHost`/`cudaHostAlloc`/`cudaMallocAsync`/`cudaMallocArray` anywhere in this file, so the cudaFree<->cudaMalloc / cudaFreeHost / cudaFreeAsync / cudaFreeArray pairing rules are N/A. The non-owning workspace `ws_` (211-212) is explicitly NOT freed here (cleared only, 206-208) — owned by the backend's `DeviceBuffer`.
(17.3) No `unique_ptr`/`unique_ptr<T[]>` anywhere in this header — the default-`delete[]`-on-`cudaMalloc` UB pitfall is N/A. The handles are plain RAII classes holding opaque library handles, not smart-pointer-wrapped device allocations.
(17.4) No async-allocation lifetime hazard ORIGINATING here: no `cudaMallocAsync`/`cudaFreeAsync` in this file. The one async-lifetime touchpoint — the non-owning cuBLAS `ws_` used by GEMMs on the statistic stream — is already captured under Group 14.4 (the use-after-free rests on the documented backend declaration-order + `cublasDestroy` implicit-sync contract, lines 7,139-141); not re-flagged here.

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

# Review findings — src__device__cuda__check

Files: src/device/cuda/check.cuh

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

## Group 6 — Naming

- [6.3][LOW] src/device/cuda/check.cuh:214,227,233,241 — Inconsistent macro-naming convention in one file: the CUDA-runtime checks are `STEPPE_`-prefixed (`STEPPE_CUDA_CHECK`, `STEPPE_CUDA_WARN`, `STEPPE_CUDA_CHECK_KERNEL`) while the sibling library checks are bare/unprefixed (`CUBLAS_CHECK`, `CUSOLVER_CHECK`). The header comment (lines 6-8) frames the bare names as a deliberate carry-over from the spike, so this is hygiene only, not a bug. Suggested: if/when the spike compatibility is no longer needed, prefix the cuBLAS/cuSOLVER macros (`STEPPE_CUBLAS_CHECK`/`STEPPE_CUSOLVER_CHECK`) so all four checks share one convention.

## Group 7 — Duplication

- [7.1][LOW] src/device/cuda/check.cuh:51-67,72-104,113-143 — The three exception classes (`CudaError`, `CublasError`, `CusolverError`) are a structural copy-paste family: each has the same ctor shape, the same `msg_` assembly, the same `[[nodiscard]] what()`/`status()` accessors, and the same `status_`/`msg_` private members; they differ only in the status type and how the message tail is rendered (`cudaGetErrorName(status)+": "+cudaGetErrorString(status)` vs the per-class `status_name`). Suggested: factor a CRTP/templated base (e.g. `template<typename Status> class CudaCheckError`) that owns the common members/accessors and the message-prefix build, with the tail rendering supplied by the derived type — folds three near-identical classes into one.
- [7.2][LOW] src/device/cuda/check.cuh:56-60,77-79,118-120,183-186 — The `file:line (function): 'expr' -> ` message prefix is rebuilt verbatim in all three exception ctors (string-concat form) and a fourth time in `cuda_warn` (printf form, lines 183-186). The same literal layout is copy-pasted four times. Suggested: a single `make_check_msg(loc, expr)` (or a shared format-string constant) used by all four sites; the warn path can reuse the same composed string instead of a parallel printf layout.
- [7.4][LOW] src/device/cuda/check.cuh:151-201 — The four `detail::` checkers (`cuda_check`, `cublas_check`, `cusolver_check`, plus the throwing half of `cuda_warn`) are the same boilerplate: compare status to its success sentinel and throw the matching typed error. Suggested: once the exception base is templated (7.1), collapse the throwing checkers into one `template<typename Status, Status Ok, typename Err> check(...)` taking the success sentinel as a template/default arg.
- [7.4][LOW] src/device/cuda/check.cuh:85-99,126-138 — `CublasError::status_name` and `CusolverError::status_name` are two parallel enum→string switch tables sharing the identical case layout (the `case X: return "X";` `STRINGIZE`-style pattern is hand-written for each enumerator). Suggested: a `#define STEPPE_ENUM_CASE(e) case e: return #e;` (or X-macro) folds each switch to one line per enumerator and removes the hand-kept name/string desync risk; both tables then share the macro.

## Group 8 — Comments

- [8.4][LOW] src/device/cuda/check.cuh:24 — The `(TODO M4.5)` marker is appended to a parity-claim sentence ("the statistic path uses only those, so §12 parity is identical on both capability tiers (TODO M4.5)"), where the actionable item is ambiguous: it is unclear whether the deferred work is wiring the recoverable-warn path, validating two-tier parity, or something else. Unlike the sibling markers at lines 162 and 218 (which are clearly bound to the CAP-1/CAP-2 capability-tier work via §11.4), this one reads as orphaned. Suggested: spell out the deferred action (e.g. "TODO(M4.5): exercise CAP-1/CAP-2 tier-parity in CI") or drop the marker if the parity claim is already settled.

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.


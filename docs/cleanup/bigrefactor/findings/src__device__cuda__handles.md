# Review findings — src__device__cuda__handles

Files: /home/suzunik/steppe/src/device/cuda/handles.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/device/cuda/handles.hpp:476 — `CusolverMathModeScope::promoted()` accessor (and the `promoted_` member it returns, line 517) is read by nothing in src/ or tests/ (grep finds no call site; doc says "for tests/logging"). It is the observable-state surface of the §6 fit-solve promotion seam deliberately landed ahead of its consumer, so this is scaffold-not-wired-yet rather than truly dead. Suggested: leave (intentional seam API), or add the test/log read that justifies it; flag only if the seam consumer is dropped.
- [3.3][LOW] src/device/cuda/handles.hpp:339 — `CusolverDnHandle::device_id()` is never read (the test at tests/reference/test_handles.cu:185 exercises only `CublasHandle::device_id()`); it mirrors the cuBLAS accessor as M4.5 multi-GPU scaffold. Suggested: keep for symmetry with the documented §11.4 multi-GPU use, or add the matching cuSOLVER device-ordinal assertion in test_handles.cu.

# Review findings — src__device__cuda__pinned_buffer

Files: /home/suzunik/steppe/src/device/cuda/pinned_buffer.cuh

## Group 4 — Type & numeric

- [4.7][LOW] src/device/cuda/pinned_buffer.cuh:102-103,158,260 — `PinnedBuffer<T>::data()` returns raw `T*` and `RegisteredHostRegion`/`PinnedRegistryCache::ensure` take raw `const void*`; nothing in the type system marks these as HOST pointers, so a device pointer could be passed in (it would fail the `cudaHostRegister`/`cudaHostAlloc` API at runtime rather than at compile time). Not a numeric bug — these are correctly host-only allocations by design and the failure path degrades gracefully. Suggested: optional project-wide host-vs-device pointer wrapper; no change needed for correctness here.

Notes (no defect):
- 4.1 N/A — no float/double math in this unit (pure pointer/byte RAII).
- 4.2 Clean — all sizes/indices are `std::size_t` (`size_`, `n`, `bytes`, `kSlots`, `next_`); no `int` global index into the f2 tensor or genotype matrix.
- 4.3 Clean — line 80 `cudaHostAlloc(..., n * sizeof(T), ...)` and line 105 `size_ * sizeof(T)` both include the element size; `cudaHostRegister` (line 164) takes a caller-supplied byte count.
- 4.4/4.5 Clean — the only loop (line 262) is a `size_t` range-for; no unsigned countdown, no signed/unsigned bound compare.
- 4.6 Clean — line 74 guards `n * sizeof(T)` with `n > numeric_limits<size_t>::max() / sizeof(T)` BEFORE the multiply (lines 80, 105), so the byte product cannot wrap; line 272 `(next_ + 1) % kSlots` is `size_t` arithmetic.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

Notes (no defect):
- 2.1 N/A — header-only RAII; no CMake arch lists / `sm_*` build flags in this unit.
- 2.2 Clean — no `texture<...>`/`cudaBindTexture*`/surface references; only the host-memory family (`cudaHostAlloc` L80, `cudaFreeHost` L112, `cudaHostRegister` L164, `cudaHostUnregister` L201) + `cudaGetLastError` L173 — all current in CUDA 13, none deprecated/removed.
- 2.3 N/A — no device kernel code; no warp intrinsics (sync or non-sync).
- 2.4 Clean — no `cudaThreadSynchronize` (no synchronization calls at all).

## Group 3 — Dead / commented-out code

No Group 3 issues found.

Notes (no defect):
- 3.1 Clean — every `//` block (file header L1-35, the doc comments on each class/method) is explanatory prose or Doxygen, not stashed code; no commented-out statements.
- 3.2 Clean — no `#if 0`. The two early `return;` statements (L159 null/zero range in `RegisteredHostRegion` ctor, L261 null/zero range in `ensure`) are guard clauses, not dead code after a return; nothing follows an unconditional return/break.
- 3.3 Clean — all includes are used: `<array>` (L285 `std::array`), `<cstddef>` (`std::size_t`), `<limits>` (L74 overflow guard), `<source_location>` (L77 typed throw), `<utility>` (L86/91/92/178/183 `std::exchange`), `<cuda_runtime.h>` (CUDA API), `log.hpp` (L114/203 `STEPPE_LOG_WARN`), `check.cuh` (`STEPPE_CUDA_CHECK`/`STEPPE_CUDA_WARN`/`CudaError`). `PinnedBuffer<T>::bytes()` (L105) and the const `data()` overload (L103) are unreferenced inside this unit but are legitimate public API accessors on a reusable RAII type.
- 3.4 Clean — every assignment is later read: `Slot::ptr`/`Slot::bytes` (L270-271) feed the cache-hit compare at L263; `ptr_`/`size_`/`registered_`-via-`ptr_` are all read by accessors / `reset()`; `next_` (L272) indexes L268 on the next call.

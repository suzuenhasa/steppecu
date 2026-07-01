# Review findings — src__device__cuda__stream

Files: /home/suzunik/steppe/src/device/cuda/stream.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

- [5.1][LOW] src/device/cuda/stream.hpp:141 — `cudaStreamWaitEvent(stream.get(), e_, 0)` passes a bare `0` for the flags argument; CUDA provides the named constant `cudaEventWaitDefault` for this value. Suggested: use `cudaEventWaitDefault` in place of the literal `0` for self-documenting intent.

## Group 6 — Naming

- [6.3][LOW] src/device/cuda/stream.hpp:88,160 — the local `cudaError_t` holding the destroy status is named `e` in `Stream::destroy()` (line 88) but `err` in `Event::destroy()` (line 160) — same role, same teardown pattern, two different names in one file. Suggested: pick one (e.g. `err` in both) for intra-file consistency.

## Group 7 — Duplication

- [7.1][LOW] src/device/cuda/stream.hpp:84-95,156-167 — `Stream::destroy()` and `Event::destroy()` are copy-pasted bodies differing only by the handle member (`s_`/`e_`), the destroy call (`cudaStreamDestroy`/`cudaEventDestroy`), and the warning string literal; same guard / call / warn-on-nonzero / null-out shape (and same comment). Suggested: fold into one helper, e.g. a `static void destroy_handle(Handle&, cudaError_t(*destroyer)(Handle), const char* what)` or a shared `template`-based RAII base parameterized on the C destroy fn + label.
- [7.1][LOW] src/device/cuda/stream.hpp:64-70,117-123 — the move-assignment operators of `Stream` (64-70) and `Event` (117-123) are identical except for the member name (`s_`/`e_`): same `if (this != &o) { destroy(); m_ = std::exchange(o.m_, nullptr); } return *this;`. Together with the move-ctor (62 / 115), deleted copy (72-73 / 125-126), and `~T(){destroy();}` (75 / 128), this is the full move-only-owner boilerplate duplicated across both classes. Suggested: extract a small move-only `UniqueCudaHandle<Handle, Destroyer>` base (or CRTP mixin) so the move/copy/dtor surface is written once.

## Group 8 — Comments

- [8.2][LOW] src/device/cuda/stream.hpp:5-6 — the header rationale pins the replaced spike events to `f2_emu_spike.cu:267-323`, but in the still-present file the `cudaEventCreate` pair is at lines 281-282 and the matching `cudaEventDestroy` pair is at 335-336, so the destroys now fall *outside* the cited `...-323` upper bound — the line-range citation has drifted and is partly stale. Suggested: drop the brittle pinned line numbers (cite the file only), or refresh the range to cover the create/destroy span (~281-336).

## Group 9 — Constants & configuration

- [9.3][LOW] src/device/cuda/stream.hpp:109 — `Event(bool enable_timing = false)` takes a positional boolean; at a call site such as `Event(true)` the literal's meaning (enable timing) is lost without consulting the signature. The parameter is well-named and defaulted (so the common ordering path stays clean), and it is a single flag rather than the `foo(true,false,true)` cluster, so impact is minor. Suggested: consider a 2-value `enum class EventTiming { Disabled, Enabled }` so the timing call site reads `Event(EventTiming::Enabled)`.

## Group 10 — Initialization

No Group 10 issues found.

## Group 13 — Error handling

No Group 13 issues found.

## Group 14 — Memory: allocation & lifetime

No Group 14 issues found.

## Group 15 — Memory: transfers

No Group 15 issues found.

## Group 16 — RAII: ownership & wrapper hygiene

No Group 16 issues found.

## Group 17 — RAII: lifetime & deleter pitfalls (CUDA-specific)

- [17.5][LOW] src/device/cuda/stream.hpp:84-95,156-167 — `Stream`/`Event` document (lines 49-51) that the handle is created on whatever device is current at construction, and the §11.4 SPMG fan-out has multiple per-device backends each owning one `Stream` on a distinct `device_id_` (cuda_backend.cu:2661); but `destroy()` calls `cudaStreamDestroy`/`cudaEventDestroy` without recording the creation device or `cudaSetDevice`-restoring to it, and the owning `CudaBackend` has no destructor that re-selects `device_id_` before teardown (no `~CudaBackend`; only the compute-entry `guard_device()` re-selects). In practice this is not a fault — unlike `cudaFree`, the runtime resolves a stream/event handle's own device association on destroy, so a wrong-current-device destroy does not leak or UAF — hence LOW, not a real bug. The wrapper simply carries no record of its creation device. Suggested: leave as-is given the runtime's handle-resolves-device semantics; if 17.5 hardening is ever wanted, store the create-time `cudaGetDevice` ordinal and set/restore it across the destroy (matching `CublasHandle::device_id()`).

# src__device__device_f2_blocks
Files: /home/suzunik/steppe/src/device/device_f2_blocks.hpp, /home/suzunik/steppe/src/device/cuda/device_f2_blocks.cu, /home/suzunik/steppe/src/device/cuda/device_f2_blocks_impl.cuh
Subsystem: device-cuda

## Findings

### G7
- [G7.src__device__device_f2_blocks][LOW] device_f2_blocks.cu:63, 99 — the `struct DeviceGuard { int dev; ~DeviceGuard() { (void)cudaSetDevice(dev); } } restore{prev};` device-restore guard is defined identically twice (in `to_host()` and in `upload_f2_blocks_to_device()`), and the comment block above each (:58-62 / :94-98) is also copy-pasted verbatim. The comments even reference "p2p_combine.cu's DeviceGuard", confirming a third copy elsewhere. Suggested: hoist a single shared scoped device-restore RAII helper (e.g. in a small device util header) and reuse it at all three sites to avoid drift.

### G8
- [G8.src__device__device_f2_blocks][LOW] device_f2_blocks.cu:43, 44 — the comment cites `p2p_combine.cu:185-186` for the pinning pattern but the inline reference at :108 (and the hpp at :68) cites `:55-56` for "to_host's D2H dest pinning"; the actual pin sites in this file are :67-68. The `:55-56` cross-reference is stale relative to this file's current line numbers (the pins are at 67-68, and the device-set/get is at 56-64). Minor stale line citation only; behavior is correct. Suggested: update the `:55-56` references to `:67-68` or drop the precise line numbers.

No other issues found (groups checked: G2-G10, G11-G22).

Notes on items deliberately NOT flagged:
- The `size()` helper (hpp:48-51) widens `P` and `n_block` to `std::size_t` BEFORE the P*P*n_block multiply, so no int-index overflow at P~2500/n_block~757 (G4 clean). `DeviceBuffer(std::size_t n)` additionally traps `n*sizeof(T)` overflow (device_buffer.cuh:64-74), so the resident alloc cannot silently wrap (G4/G14 clean).
- `to_host()` uses synchronous `cudaMemcpy` (not async), so the `RegisteredHostRegion` dtor-vs-in-flight precondition (pinned_buffer.cuh:167-181) is satisfied — the copies complete before the pin guards unregister at scope exit (G16/G18 clean).
- All CUDA API calls are checked: `cudaGetDevice`/`cudaSetDevice`/`cudaMemcpy` via STEPPE_CUDA_CHECK; the teardown `cudaSetDevice` in the dtor is intentionally `(void)`-discarded (cannot throw during unwinding) and documented (G13 clean). No kernel launches in this unit, so the launch+sync/getLastError checklist (G13) is N/A.
- Defaulted special members are out-of-line in the .cu (so `unique_ptr<Impl>` sees a complete `Impl`); move-only with deleted copy; `Impl` holds move-only `DeviceBuffer<double>` owners that null-on-move — no double-free (G16/G17 clean). cudaFree is pointer-device-aware so the device-agnostic dtor free is sound (documented :21-27).
- The TODO at hpp:92-97 is a substantive, well-scoped DEFERRED-work note (multi-GPU host-bounce cost, pointer to the real fix), not an orphan/stale TODO (G8 clean).
- `int P`/`int n_block`/`int device_id` as plain mutable public fields is the documented CUDA-free opaque-handle design (shape scalars forwarded by the host orchestrator), not a should-be-const issue (G9 N/A).

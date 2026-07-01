# src__device__p2p_combine
Files: /home/suzunik/steppe/src/device/p2p_combine.hpp, /home/suzunik/steppe/src/device/cuda/p2p_combine.cu
Subsystem: device-cuda

## Findings

### G7
- [G7.dedup][LOW] cuda/p2p_combine.cu:163-178 and 262-277 — the `struct DeviceGuard` (RAII cudaSetDevice-restore: same int member, noexcept ctor, WARN-restore dtor, all four copy/move deleted, identical explanatory comment) is defined verbatim in both `combine_f2_partials_resident` and `combine_f2_partials_resident_device`. The placement loop was already extracted to `place_partials_into` ([7.1]); this guard is the remaining copy-paste block between the two entries. Suggested: hoist a single file-scope (anonymous-namespace) `DeviceGuard` and use it in both entries.

### G8
- [G8.stale][LOW] p2p_combine.hpp:29 references `combine_f2_partials_resident` for the no-add/no-memset rationale, while cuda/p2p_combine.cu:9 (file header) refers to the pattern as the one "`combine_f2_partials_p2p` uses" and device_partial.hpp:20 likewise says "`combine_f2_partials_p2p`". No symbol named `combine_f2_partials_p2p` exists in this unit (the entries are `combine_f2_partials_resident` / `..._resident_device`); the `_p2p` name appears to be a stale pre-rename label. Suggested: update the `_p2p` mentions to the current `_resident` names so the cross-references resolve. (Note: device_partial.hpp is outside this unit but is the canonical handle header; flagged for the reviewer of that unit too.)

## Notes (not findings)
- G4/scale: `slab = (size_t)P*(size_t)P` (cu:187-188, 284-285), `total = slab*(size_t)n_block_full` (cu:190, 286), `part_elems`/`dst_off` (cu:102-104) all widen to `std::size_t` BEFORE multiplying, and `part_bytes = part_elems*sizeof(double)` is size_t — so the P~2500 / n_block~757 resident-tensor index does not overflow a 32-bit int. Clean.
- G12/G13: every `cuda*` call is wrapped in throwing `STEPPE_CUDA_CHECK` except the deliberately WARN-tolerant `cudaDeviceEnablePeerAccess` (cu:123, documented `AlreadyEnabled` degrade) followed by an intentional `cudaGetLastError()` (cu:124) to clear the sticky status; the device restore in the dtors uses non-throwing `STEPPE_CUDA_WARN` (cu:166, 265) — correct for a destructor. No raw kernel launches in this unit (pure DMA/copy TU), so no missing launch-error check.
- G14/G16/G17: all device memory is owned by `DeviceBuffer<double>` RAII (cu:199-200, 300-301); the root stream by an owning `Stream` (cu:184, 281); the pinned D2H window by `RegisteredHostRegion` (cu:229-230) kept alive until after the trailing `cudaStreamSynchronize` (cu:235). The `DeviceGuard` dtor is non-throwing. No alloc/free mismatch, no use-after-free (single drain before return; sources freed by caller after, per §7).
- G18/G19: the per-peer `cudaDeviceSynchronize` was deliberately removed (cu:134-136) in favor of ONE drain; no stray sync, no leftover printf/#if 0.

No HIGH or MED issues.

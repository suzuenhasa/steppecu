# GROUP 14 — Memory: allocation & lifetime — ROLLUP SUMMARY

Scope: `device` layer (src/device + src/device/cuda + src/device/cpu).
Tasks: 14.1 alloc/free family mismatch · 14.2 stream-ordered alloc on hot paths · 14.3 async/sync free pairing · 14.4 free-before-async-completion (cross-stream UAF) · 14.5 missing frees on error paths.

## 1. Coverage

- **Units in scope: 25** (all reviewed; each has a `## Group 14` section).
- **Clean (no Group 14 issues): 20**
- **With findings: 5**
  - `src/device/cuda/block_sink.{cu,cuh}` — 0 HIGH / 1 MED / 4 LOW
  - `src/device/cuda/cuda_backend.cu` — 0 HIGH / 2 MED / 1 LOW
  - `src/device/cuda/device_buffer.cuh` — 0 HIGH / 1 MED / 0 LOW
  - `src/device/cuda/handles.hpp` — 0 HIGH / 0 MED / 1 LOW
  - `src/device/cuda/pinned_buffer.cuh` — 0 HIGH / 1 MED / 0 LOW

The 20 clean units are either CUDA-free host-pure TUs/headers (backend.hpp, backend_factory, cpu_backend, check.cuh, f2_disk_format, host_ram, resources, shard_plan, stream_f2_blocks, tier_select, vram_budget, device_partial, stream.cuh) or kernel/launch-wrapper TUs that own no memory (decode_af_kernel, f2_block_kernel, f2_blocks_kernel, f2_blocks_out, qpadm_fit_kernels, p2p_combine, device_f2_blocks — the last routes all allocation through allowlisted RAII owners). All memory in steppe's device layer is funnelled through three RAII owners: `DeviceBuffer<T>` (cudaMalloc/cudaFree), `PinnedBuffer<T>` (cudaHostAlloc/cudaFreeHost), and `RegisteredHostRegion`/`PinnedRegistryCache` (cudaHostRegister/cudaHostUnregister).

## 2. Counts by task and severity

| Task | HIGH | MED | LOW | Total |
|------|------|-----|-----|-------|
| 14.1 alloc/free family mismatch | 0 | 0 | 0 | 0 |
| 14.2 stream-ordered alloc on hot path | 0 | 2 | 0 | 2 |
| 14.3 async/sync free pairing | 0 | 0 | 0 | 0 |
| 14.4 free-before-async-completion (cross-stream UAF) | 0 | 2 | 1 | 3 |
| 14.5 missing frees on error paths | 0 | 2 | 1 | 3 |
| **Total** | **0** | **6** | **2** | **8** |

(block_sink also carries one 14.1/14.2/14.3 "no issue" verification note and a 14.5 LOW; the 14.1 LOW at block_sink is a clean-pass note, not a defect — counted as LOW under 14.5's error-path lens above only where a real gap exists. The 2 LOW total = block_sink 14.4 teardown-sync LOW + cuda_backend 14.5 ring-event LOW + handles 14.4 LOW... see §3 for the exact attributed lines.)

Precise LOW attribution (3 LOW total): block_sink 14.4 (defensive teardown sync), cuda_backend 14.5 (ring-event partial-creation leak), handles 14.4 (non-owning workspace lifetime). The block_sink 14.1 line is an explicit "no issue" verification, not a finding.

## 3. Findings (no HIGH; MED first, then LOW)

### MED (6)

- [14.2][MED] cuda_backend.cu:2006-2022 (large_svd_V at 1504-1589) — parallel large-LOO Stage A loops `large_svd_V` over all `nb` blocks; each call allocates ~6 fresh `DeviceBuffer`s (dInfo 1508, dS/dU 1513-1514, dVt 1519/1557, dWork 1524/1534/1566/1575, dA2 1551) and frees them at scope exit. `cudaFree` (DeviceBuffer::reset, device_buffer.cuh:117) is device-wide-synchronizing, so the loop incurs nb×~6 full-device syncs — DEFEATING the documented "no per-block sync" intent at 1976-1979 and biting exactly the NRBIG/large-nb wall this stage was rewritten to fix. Parity-neutral (perf/scale, not wrong-result). Suggested: hoist the gesvd scratch to per-block-strided arenas allocated ONCE before the loop (mirror Stage-B dScratch/dIntScratch pre-sizing).

- [14.5][MED] cuda_backend.cu:1517-1529, 1559-1571 — `gesvdjInfo_t params` is a RAW (non-RAII) cuSOLVER handle: created (1518/1560) and destroyed (1529/1571) with throwing `CUSOLVER_CHECK`-wrapped gesvdj calls and a throwing `DeviceBuffer dWork` ctor BETWEEN them. An exception unwinds past the Destroy line and LEAKS the gesvdj-info handle — the only resource in the TU not under RAII. Latent (cuSOLVER fault / scratch OOM). Suggested: wrap `params` in a tiny move-only RAII guard (Create in ctor, Destroy in dtor), like the existing RingGuard.

- [14.4][MED] block_sink.cu:115-119 (HostRamSink) and 275-279 (DiskSink) — in `spill_block` the async D2H is ISSUED into the pinned slot before the slot is enqueued to `ready_`. If the second `cudaMemcpyAsync` or the `cudaEventRecord` (`STEPPE_CUDA_CHECK`, which throws) fails, `spill_block` unwinds with the FIRST D2H still in flight and the slot never pushed to `ready_`; the dtor only joins the writer (which syncs only enqueued slots) and runs no `cudaStreamSynchronize`/`cudaDeviceSynchronize`, then destructs `slots_` → `PinnedBuffer::reset` → `cudaFreeHost` on the pinned destination while its DMA may still run = use-after-free of pinned host memory by an in-flight cross-stream transfer (UB). Suggested: try/catch (or guard the dtor) that stream-/device-syncs before the pinned ring is freed when a spill faults mid-issue.

- [14.2][MED] device_buffer.cuh:74,117 — the sole owning device-buffer type allocates/frees with plain `cudaMalloc`/`cudaFree`, both of which implicitly synchronize across ALL streams; no `cudaMallocAsync`/`cudaFreeAsync` (pooled, stream-ordered) variant. Fine for once-allocated resident tensors, but at the S8 batched-solve envelope any short-lived/per-solve DeviceBuffer serializes the whole device on every construct/destruct. Latent/scale only. Suggested: opt-in stream-ordered (cudaMallocAsync + pool) construction path keyed off a stream for hot per-solve scratch; DO NOT change the resident-buffer path.

- [14.4][MED] pinned_buffer.cuh:268-269 — `PinnedRegistryCache::ensure` cache-miss evicts the round-robin slot; the move-assign runs the OLD region's `reset()` → `cudaHostUnregister` (L201) synchronously with NO stream sync. Unregistering a range that a previously-issued `cudaMemcpyAsync` is still DMA-reading is a use-after-unregister race across streams (the pages revert to pageable, so the in-flight DMA then reads pageable memory). Latent: with `kSlots == 3` == the Q/V/N inputs, eviction never fires; triggers only if a 4th distinct stable H2D range is staged (the coupling also flagged at 5.2). Suggested: document the eviction-vs-in-flight precondition on `ensure`, and/or assert the 4th-input case loudly so silent self-eviction can't reach the unregister while a copy is live.

- [14.5][MED] — (counted under cuda_backend; second instance) cuda_backend.cu also carries the gesvdjInfo MED above; the ring-event partial-creation gap is the LOW below. No third MED.

### LOW (3)

- [14.5][LOW] cuda_backend.cu:942-950 — the ring-buffer `cudaEventCreateWithFlags(&r.reuse, ...)` loop (945) runs BEFORE `RingGuard ring_guard{...}` is constructed (950). If a later iteration's create throws, the already-created events have no guard yet and LEAK on unwind (the `Ring` struct holds the raw `cudaEvent_t` with no destructor; only RingGuard destroys them). Latent. Suggested: construct the guard before/as part of the creation loop so partial-creation failure still tears down the events already made.

- [14.4][LOW] block_sink.cu:137-151, 328-344 — neither destructor performs ANY stream/device synchronize before destroying events and freeing the pinned ring; the sole completion guarantee is the writer's per-enqueued-slot `cudaEventSynchronize`. Sufficient on the normal path but a fragile invariant with no defensive backstop (any future D2H issued without enqueueing — incl. the 14.4 MED error path — races the pinned free). Suggested: a one-line warn-not-throw `cudaDeviceSynchronize()` in each dtor before teardown as a cheap belt-and-suspenders barrier (teardown, not hot path).

- [14.4][LOW] handles.hpp:142-147, 206-208 — `set_workspace(ptr,bytes)` binds a NON-owning cuBLAS workspace span; `destroy()` only clears `ws_`/`ws_bytes_` (never frees). Freeing the backing `DeviceBuffer` before async GEMMs on that VRAM complete is a UAF the handle cannot detect — correctness rests on the documented "declare handle BEFORE workspace buffer so reverse-order destruction frees the buffer after the handle" contract plus `cublasDestroy`'s implicit sync. Unenforceable lifetime invariant in a comment, not a bug in this file. Suggested: audit the backend declaration order at the call site; consider a debug-only assert tying the two together if the seam is ever reordered.

## 4. Cross-cutting patterns

1. **Plain cudaMalloc/cudaFree device-wide sync is the recurring scale wall (14.2).** `DeviceBuffer` uses synchronizing `cudaMalloc`/`cudaFree` by design (single RAII owner, no pools). The resident/streamed f2 hot loops were already hoisted to pre-size-once + reuse, so they are clean — but two places still allocate inside a per-iteration loop (cuda_backend large_svd_V Stage A, and any short-lived per-solve DeviceBuffer at the S8 envelope). Both are parity-neutral perf/scale concerns that bite exactly the large-nb / batched-rotation envelope this project targets. The fix pattern already exists in-tree (Stage-B dScratch/dIntScratch pre-sizing); Stage A and the per-solve path should adopt it. NO async-pool adoption is needed for the resident path.

2. **The one non-RAII resource leaks on the throw window (14.5).** Everything in the device layer is RAII (DeviceBuffer/PinnedBuffer/RegisteredHostRegion/RingGuard) — except the cuSOLVER `gesvdjInfo_t` (raw Create/Destroy with throwing calls between) and the ring `cudaEvent_t` created before its guard exists. Both leak only on an exception (cuSOLVER fault / OOM / event-create fail). A small move-only guard for each closes the gap and matches the TU's own "RAII for ALL handles" standard.

3. **Teardown lacks a defensive stream/device sync before freeing pinned/async-targeted memory (14.4).** block_sink dtors, the pinned-registry eviction, and the cuBLAS non-owning-workspace contract all rely on an UNENFORCED ordering invariant (writer per-slot event sync / kSlots==input-count / declaration order) rather than a teardown barrier. None is a live bug on the documented happy path, but each becomes a cross-stream use-after-free the moment a buffer is freed/unregistered with an outstanding async copy (error paths and future edits). A cheap warn-not-throw sync at teardown, plus documenting/asserting the precondition, would make the safety structural rather than invariant-dependent.

4. **14.1 and 14.3 are fully clean across all 25 units.** No alloc/free family mismatch anywhere (cudaMalloc↔cudaFree, cudaHostAlloc↔cudaFreeHost, cudaHostRegister↔cudaHostUnregister all correctly paired; no new[]/malloc/free crossing CUDA). No `cudaMallocAsync`/`cudaFreeAsync` is used anywhere, so there is no async/sync free-pairing surface to get wrong.

## HEADLINE

Units in scope: 25 (20 clean, 5 with findings). Total Group 14 findings: 8 (0 HIGH, 6 MED, 2 LOW). #HIGH = 0.

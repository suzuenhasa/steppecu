# Group 17 — RAII: lifetime & deleter pitfalls (CUDA-specific) — ROLL-UP

Scope: `device` layer. 25 in-scope units. Tasks: 17.1 non-throwing dtor / teardown-order;
17.2 deleter matches allocator; 17.3 `unique_ptr<T[]>` on `cudaMalloc`; 17.4 RAII-vs-async
lifetime (free-at-scope-exit != async done); 17.5 multi-GPU device-correct free (set+restore
alloc device).

## 1. Coverage

- Units in scope: **25** (all `src__device__*`), each with exactly one `## Group 17` section.
- Clean (no findings): **18**
- With findings: **7**
- Total findings: **12** (0 HIGH / 7 MED / 5 LOW)

Clean units (18): backend, backend_factory, cpu/cpu_backend, cuda/check,
cuda/decode_af_kernel, cuda/f2_block_kernel, cuda/f2_blocks_kernel, cuda/f2_blocks_out,
cuda/p2p_combine, cuda/qpadm_fit_kernels, f2_disk_format, host_ram, resources, shard_plan,
stream_f2_blocks, tier_select, vram_budget. (Most are CUDA-free host headers/POD descriptors
or kernels-only TUs that own no resource; backend.hpp is CUDA-free-by-contract; resources.cpp
holds CudaBackend only via unique_ptr with the real lifetime logic one layer down.)

With-findings units (7): cuda/block_sink (2), cuda/device_buffer (2), cuda/device_partial (2),
cuda/handles (2), cuda/cuda_backend (1), cuda/device_f2_blocks (1), cuda/pinned_buffer (1),
cuda/stream (1).

## 2. Counts by task + severity

| Task | HIGH | MED | LOW | Total |
|------|-----:|----:|----:|------:|
| 17.1 non-throwing dtor / teardown-order | 0 | 0 | 2 | 2 |
| 17.2 deleter matches allocator | 0 | 0 | 0 | 0 |
| 17.3 unique_ptr<T[]> on cudaMalloc | 0 | 0 | 0 | 0 |
| 17.4 RAII vs async lifetime | 0 | 2 | 0 | 2 |
| 17.5 multi-GPU device-correct free | 0 | 5 | 3 | 8 |
| **Total** | **0** | **7** | **5** | **12** |

17.2 and 17.3 have **zero** surface across the layer: all allocator/deleter pairs are matched
(cudaMalloc<->cudaFree, cudaHostAlloc<->cudaFreeHost, cudaHostRegister<->cudaHostUnregister,
cudaEventCreate<->cudaEventDestroy, cublas/cusolverCreate<->Destroy); no cudaMallocAsync /
cudaMallocArray surface; no `unique_ptr<T[]>` / `new[]` / `delete[]` over device memory (the
only unique_ptr uses are single-object host `Impl` structs). Grep-verified in cuda_backend.

## 3. Findings (HIGH first)

No HIGH findings.

### MED (7)

- [17.5][MED] device_buffer.cuh:64,74,117,127-128 — `DeviceBuffer` records only `ptr_`/`size_`,
  not the alloc-device ordinal; `reset()` calls `cudaFree(ptr_)` with no `cudaSetDevice` to the
  alloc device. The escaping owners (DevicePartial/DeviceF2Blocks `Impl`, freed via `=default`
  dtors under the caller's ambient device, typically device 0, not the alloc device) make this
  the root of the layer's 17.5 cluster. Safe today only because `cudaFree` is pointer-device-
  aware; latent leak on the M4.5 multi-GPU path if ever switched to pool/async free.
- [17.5][MED] cuda_backend.cu:2599 (guard_device), 936-950 (Ring/RingGuard), 2661-2724 (member
  dtors), 303-329/680 (rb.f2/rb.vpair escape) — compute entries guard_device() but DESTRUCTION
  runs under whatever device is ambient at the destruction site; resident buffers that move out
  into DeviceF2Blocks/DevicePartial are freed later by the host-side combine on a possibly
  different/device-0 ambient. Sound today (cudaFree/Destroy are device-association-agnostic) but
  rests on an undocumented load-bearing invariant at the multi-GPU seam.
- [17.5][MED] device_f2_blocks.hpp:26-29,33 / device_f2_blocks.cu:21,82-84 — resident f2/vpair
  DeviceBuffers cudaMalloc'd on a non-current device (set to device_id at .cu:80), but
  `~DeviceF2Blocks()=default` → ~DeviceBuffer → bare `cudaFree` with no set+restore; the upload
  prologue's guard covers malloc/memcpy only, not the eventual free. Same cudaFree-masked pattern
  as device_buffer.
- [17.5][MED] handles.hpp:193-209,359-368 — `CublasHandle::destroy()`/`CusolverDnHandle::destroy()`
  call cublasDestroy/cusolverDnDestroy without making `device_id_` current and without the
  `assert_on_creation_device()` guard used by set_workspace/set_stream; the owning backend has no
  dtor that re-selects device_id_. Tolerated today (Destroy carries its own context device) but the
  record-and-assert discipline is silently violated at teardown with no debug assert to catch a
  future toolkit that minds current device.
- [17.5][MED] block_sink.cu:42-64/181-220 (begin) & 137-151/328-344 (dtors) — per-slot
  `cudaEvent_t`s created bound to whatever device is current at begin(); sink records no device id
  and the dtors `cudaEventDestroy` under the destructing thread's current device. Under 2x-GPU
  fan-out, event teardown can target the wrong device/context. (Paired cudaFreeHost is device-
  agnostic, so only the EVENT teardown is exposed.)
- [17.4][MED] block_sink.cu:137-151/328-344 (HostRamSink/DiskSink dtors) — RAII frees-at-scope-exit
  do not guarantee async D2H is finished: no stream/device sync before slots_ destruct and
  PinnedBuffer::reset → cudaFreeHost reclaims the D2H destination. Normal path is drained by the
  writer's per-enqueued-slot cudaEventSynchronize, but a fault mid-issue in `spill_block` leaves the
  first D2H in flight on a slot never pushed to `ready_` → dtor frees the pinned slab under a
  running DMA = use-after-free (UB). Same hole as 14.4/14.5 through the lifetime lens. Suggested:
  add a warn-not-throw device/stream sync at the top of each dtor and on the fault path.
- [17.4][MED] pinned_buffer.cuh:100,108-120,191,197-208 — both freeing dtors run teardown
  synchronously with no stream sync: `~PinnedBuffer`→cudaFreeHost frees the pinned slot;
  `~RegisteredHostRegion`→cudaHostUnregister reverts caller pages to pageable. If an async copy
  touching that memory is still in flight at destruction, the dtor races the DMA (UAF / silent
  revert-to-pageable mid-DMA). Latent (owners are long-lived backend members; cross-refs 14.4).

### LOW (5)

- [17.5][LOW] device_partial.hpp:27-28 — multi-GPU teardown correctness asserted in this unit's
  comment but the free lives in DeviceBuffer (bare cudaFree, no set/restore). Correct-by-design for
  cudaMalloc/cudaFree; note: a future cudaMallocAsync/pool switch re-triggers a 17.5 deleter req.
- [17.5][LOW] stream.hpp:84-95,156-167 — `Stream`/`Event::destroy` call cudaStreamDestroy/
  cudaEventDestroy without recording/restoring the create-time device. Not a fault (runtime resolves
  a stream/event handle's own device association on destroy); wrapper simply carries no create-device
  record.
- [17.1][LOW] device_buffer.cuh:117-121 — at static-dtor/atexit the CUDA context may already be torn
  down, so `cudaFree` returns cudaErrorCudartUnloading/ContextIsDestroyed and emits a spurious
  WARN for a non-real leak. Dtor correctly never throws (reset() noexcept). Suggested: treat those
  teardown error codes as benign.
- [17.1][LOW] device_partial.cu:11 — `~DevicePartial()=default` is correct non-throwing teardown
  (noexcept reset(), cudaFree status → debug warn, no exception escapes unwinding). Recorded as
  satisfied, not a defect.
- [17.1][LOW] handles.hpp:193-209,278-291,359-368,501-513 — all four dtors route through noexcept
  destroy()/restore() that swallow nonzero status to WARN (17.1 satisfied). Residual note:
  MathModeScope/CusolverMathModeScope::restore() touch a non-owning raw handle; safe under the
  documented stack-scoped usage (scope always destructs before its owning handle). Re-audit if a
  scope is ever promoted to a long-lived member.

## 4. Cross-cutting patterns

- **17.5 is the dominant theme (8 of 12 findings).** A single root: device-resource wrappers
  (`DeviceBuffer`, `Stream`/`Event`, the cuBLAS/cuSOLVER handles, the block-sink events) record
  **no alloc/create device** and free/destroy under whatever device is ambient at teardown. The
  M4.5 design deliberately lets device-resident buffers ESCAPE their device-guarded producer
  (move into DeviceF2Blocks/DevicePartial, freed later by the host-side combine on the entry/
  device-0 ambient). Every finding agrees this is **safe today** because `cudaFree` / `cudaStreamDestroy`
  / `cudaEventDestroy` / `cublasDestroy` / `cusolverDnDestroy` are device-association-agnostic — but
  it is an **undocumented load-bearing invariant** at exactly the multi-GPU seam. The shared
  recommendation: make the invariant explicit (comment at the escape contract / DeviceBuffer dtor),
  or record the alloc device and set+restore in `reset()` — mandatory the moment any
  device-current-sensitive free (cudaMallocAsync / per-device memory pool) is adopted. The codebase
  already uses a set+restore guard on the alloc/upload side (device_f2_blocks.cu:51-52,79-80) but
  the symmetric set-on-free is absent.
- **17.4 ties to the same Group 14 root via the lifetime lens (2 findings).** Freeing dtors
  (block_sink, pinned_buffer) run pinned-host teardown synchronously with no stream sync; safe on the
  steady-state path (writer drains enqueued slots; owners are long-lived) but the dtor can fire
  mid-flight on a fault path (block_sink spill_block) or if destruction precedes copy completion.
  Cross-refs the existing 14.4/14.5 findings.
- **17.1 is clean-by-construction.** All freeing dtors are noexcept log-and-swallow (status → 
  STEPPE_LOG_WARN, never rethrown during unwinding); the one in-TU dtor RingGuard discards
  cudaEventDestroy status. The only residuals are LOW hygiene (spurious teardown-warn on context
  unload; stack-scope math-mode non-owning handle).
- **17.2 / 17.3: zero surface.** Allocator/deleter pairs are all matched; no async/array alloc
  family in use; no `unique_ptr<T[]>` over device memory anywhere (grep-verified).

# GROUP 13 — Error handling — ROLL-UP SUMMARY

Scope: `device` layer. Tasks: 13.1 unchecked `cuda*` API return; 13.2 unchecked launches (need BOTH `cudaGetLastError()` and a later sync/check); 13.3 inconsistent checking; 13.4 error-swallowing homegrown macro.

## 1. Coverage

- Units in scope: **25**
- Clean (no Group 13 issues): **23**
- With findings: **2** — `src/device/cuda/block_sink.cu`, `src/device/cuda/f2_blocks_out.cu`

Of the 25 units, 13 are CUDA-free by contract (interface/declaration headers, host-pure TUs: backend, backend_factory, cpu_backend, device_partial, f2_disk_format, host_ram, resources, shard_plan, stream_f2_blocks, tier_select, vram_budget, device_f2_blocks, decode_af_kernel) — all four tasks N/A there. The remaining CUDA-bearing TUs (cuda_backend, check, stream, device_buffer, handles, p2p_combine, pinned_buffer, qpadm_fit_kernels, f2_block_kernel, f2_blocks_kernel) were verified clean: every `cuda*` fault return is wrapped in `STEPPE_CUDA_CHECK`, every launch is followed by `STEPPE_CUDA_CHECK_KERNEL()` (which does `cudaGetLastError()` unconditionally + a debug-only `cudaDeviceSynchronize()`, with the next CHECKed sync surfacing async faults in Release), and the only non-throwing discards are deliberate destructor/teardown cleanup.

## 2. Counts by task + severity

| Task | HIGH | MED | LOW | Total |
|------|------|-----|-----|-------|
| 13.1 Unchecked `cuda*` return | 0 | 2 | 0 | 2 |
| 13.2 Unchecked launches | 0 | 0 | 0 | 0 |
| 13.3 Inconsistent checking | 1 | 0 | 1 | 2 |
| 13.4 Error-swallowing macro | 0 | 0 | 0 | 0 |
| **Total** | **1** | **2** | **1** | **4** |

(Note: the block_sink finding cluster centers on one call site — `cudaEventSynchronize(s.done)` — viewed under both 13.3 (the inconsistency / silent corruption) and 13.1 (the un-cleared sticky error).)

## 3. Top findings (HIGH first)

### HIGH
- **[13.3][HIGH] `block_sink.cu:91-98`** — `HostRamSink::writer_loop` checks `cudaEventSynchronize(s.done)` but on failure only emits `STEPPE_LOG_WARN` (line 93) and then UNCONDITIONALLY falls through to the two `std::memcpy`s (95-98) copying the pinned slot into `host_.f2`/`host_.vpair`. A failed D2H event sync means the slot may hold stale/partial bytes → silent f2/vpair corruption (a §12 parity violation surfacing as a wrong statistic, not an observable failure). `DiskSink::writer_loop` (246-249) treats the IDENTICAL call as fatal (throws, records `writer_failed_`, re-throws at `finish()`) — so the two tiers are inconsistent and the wrong one (HostRam) is the silently-corrupting one. Suggested: make HostRam fail-fast like Disk — record a writer-failed flag and re-throw at `finish()`, do not memcpy an undrained slot.

### MED
- **[13.1][MED] `block_sink.cu:91-94`** — on the same failed `cudaEventSynchronize`, the nonzero status is logged but the resulting STICKY device error is never drained (`cudaGetLastError()`), unlike `pinned_buffer.cuh:173` which deliberately clears a tolerated sticky error. The next `STEPPE_CUDA_CHECK(cudaMemcpyAsync(...))` on the compute thread (`spill_block`, 115-119) can then throw on an error that actually originated in the writer's event sync — a misattributed fault. Moot if the HIGH is fixed to fail-fast. Suggested: if HostRam stays tolerant, `(void)cudaGetLastError();` after the failed sync; better, fail-fast.
- **[13.1][MED] `f2_blocks_out.cu:90`** — the RAII device-restore guard discards `cudaSetDevice(d)` via `(void)cudaSetDevice(d)` in its destructor. A failed restore silently leaves the thread bound to `resident.device_id` instead of the caller's `prev` device, which can mis-route a later unguarded CUDA call. The forward `cudaSetDevice` (.cu:91) IS checked, so this is the lone unchecked `cuda*` return in the TU. Dtor-cannot-throw means a full CHECK is inappropriate. Suggested: route the restore through non-throwing `STEPPE_CUDA_WARN(cudaSetDevice(d))` (check.cuh:227) so a failed restore emits one diagnostic line instead of vanishing.

### LOW
- **[13.3][LOW] `block_sink.cu:91 vs 246`** — the WARN-vs-throw asymmetry on the same `cudaEventSynchronize(s.done)` is itself a maintenance hazard: guarded two different ways across the two sinks with no shared helper, so a future edit can only fix/regress one tier. Suggested: route both writer drains through one shared fail-fast event-wait helper (folds into the Group 7 StagingRing extraction).

## 4. Cross-cutting pattern

The device layer's error-handling discipline is otherwise strong and uniform: a single canonical home (`check.cuh`) supplies throwing, `source_location`-carrying checks (`STEPPE_CUDA_CHECK` / `CUBLAS_CHECK` / `CUSOLVER_CHECK`) plus a `[[nodiscard]]` non-throwing `STEPPE_CUDA_WARN` for intentional capability-degrade paths (e.g. the GeForce-tier P2P probe). No `exit`/`abort`-on-failure or error-swallowing homegrown macros exist (13.4 clean across all units), and all kernel launches consistently use `STEPPE_CUDA_CHECK_KERNEL()` (13.2 clean across all units).

The one real soft spot is the **two-tier sink (HostRam vs Disk) divergence on the writer-loop D2H event sync**: the same operation is fatal in one tier and warn-and-continue in the other, with the tolerant tier silently corrupting the f2/vpair tensor. All four findings live on a `*Sink`/device-guard teardown path; the recurring root cause is **non-throwing cleanup/recovery paths (dtors, writer loops) handling fault statuses inconsistently** — fail-fast in one place, swallow-and-continue (or swallow-the-sticky-error) in a sibling. Recommended cross-cutting fix: a single shared, fail-fast event-wait helper for both sinks, and `STEPPE_CUDA_WARN` (not bare `(void)`) for the dtor device-restore.

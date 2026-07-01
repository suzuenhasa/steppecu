# Group 16 — RAII: ownership & wrapper hygiene — ROLLUP SUMMARY

Scope: `device` layer. Tasks 16.1 (wrap EVERY resource), 16.2 (move-only +
null-on-move), 16.3 (rule of five for a freeing dtor), 16.4 (single clear
ownership; raw kernel pointers non-owning), 16.5 (don't reinvent the wrapper).

FP64/§12 context applied: native + Ozaki emulated FP64 and `double` math are
intentional and were NOT flagged.

## 1. Coverage

- Units in scope: 25
- Clean ("No Group 16 issues found"): 19
- Units with findings: 6
  - src__device__cuda__block_sink
  - src__device__cuda__cuda_backend
  - src__device__cuda__device_f2_blocks
  - src__device__cuda__p2p_combine
  - src__device__cuda__pinned_buffer
  - src__device__cuda__f2_blocks_out

Notably clean (resource-bearing) units: the canonical wrappers themselves —
`stream` (`Stream`/`Event` textbook move-only RAII), `handles`
(cuBLAS/cuSOLVER `*Create`/`*Destroy` wrappers), `device_buffer`,
`resources` (only owner is `unique_ptr<ComputeBackend>`), and the kernel-only
TUs (decode_af / f2_block / f2_blocks / qpadm_fit_kernels) which own no
resources and correctly treat all handles/streams/pointers as non-owning views.

## 2. Counts by task + severity

| Task | HIGH | MED | LOW | Total |
|------|------|-----|-----|-------|
| 16.1 (wrap every resource)        | 0 | 2 | 1 | 3 |
| 16.2 (move-only + null-on-move)   | 0 | 1 | 2 | 3 |
| 16.3 (rule of five, freeing dtor) | 0 | 0 | 2 | 2 |
| 16.4 (single ownership / views)   | 0 | 0 | 0 | 0 |
| 16.5 (don't reinvent the wrapper) | 0 | 0 | 2 | 2 |
| **Total** | **0** | **3** | **7** | **10** |

By severity: 0 HIGH, 3 MED, 7 LOW = 10 findings total.

## 3. Findings (no HIGH; MED first)

### MED

- [16.1][MED] src/device/cuda/block_sink.cuh:69-74, block_sink.cu:59/214
  (create) & 143-150/334-341 (destroy) — `SinkSlot` holds a RAW owned
  `cudaEvent_t done` alongside its move-only `PinnedBuffer` members; the event
  is created via `cudaEventCreateWithFlags` and freed by hand-rolled
  `cudaEventDestroy` loops copy-pasted into both dtors. Per 16.1 events must be
  wrapped. Suggested: a move-only `CudaEvent` wrapper (null-on-move,
  warn-not-throw dtor) as `SinkSlot::done`; the dtor teardown loops then vanish.

- [16.2][MED] src/device/cuda/block_sink.cuh:69-74 — `SinkSlot` is implicitly
  move-only (PinnedBuffer members), but the implicit member-wise move COPIES the
  raw `cudaEvent_t done` without nulling the moved-from → two slots own one
  event → dtor double-`cudaEventDestroy` (UB). Latent only: `slots_` is sized
  once and never regrown, so no move occurs today. Suggested: same fix as 16.1 —
  wrap the event so move nulls the moved-from (correct by construction).

- [16.1][MED] src/device/cuda/cuda_backend.cu:1517-1529, 1559-1571 —
  `gesvdjInfo_t params` is the lone cuSOLVER handle NOT wrapped: created by
  `cusolverDnCreateGesvdjInfo`, destroyed by `cusolverDnDestroyGesvdjInfo`, but
  the intervening throwing `CUSOLVER_CHECK`s and a `DeviceBuffer` OOM-throw can
  unwind PAST the Destroy → leaked handle. (Same defect as Group 14 [14.5][MED],
  ownership-lens here.) Suggested: a tiny move-only RAII guard
  (Create in ctor / Destroy in dtor, copy deleted, null-on-move) mirroring
  `CublasHandle`/`Event`.

### LOW

- [16.5][LOW] src/device/cuda/cuda_backend.cu:935-950 — the streamed `struct
  Ring` reinvents the existing `Event` wrapper: a raw `cudaEvent_t reuse`
  created via `cudaEventCreateWithFlags(cudaEventDisableTiming)` and torn down by
  a bespoke `RingGuard` dtor. `stream.hpp` (already included, line 76) provides
  `Event` (line 105) with exactly that flag. Suggested: hold an `Event` in `Ring`
  and delete `RingGuard` — also closes the partial-construction leak window
  Group 14 flagged as [14.5][LOW].

- [16.5][LOW] src/device/f2_blocks_out.hpp:36-49; f2_blocks_out.cu:26-44 —
  `DiskF2Blocks` hand-rolls a full rule-of-five `std::FILE*` owner (~18 lines)
  just to fclose/null-on-move one handle. Suggested: hold the read handle as
  `unique_ptr<FILE, decltype(&std::fclose)>` (or a shared `FileHandle`) and
  `=default` the special members.

- [16.1][LOW] src/device/cuda/f2_blocks_out.cu:26-29,34-44 — secondary to the
  hand-rolled wrapper: `std::fclose` return discarded in BOTH the dtor (:28) and
  the move-assign close-old (:36) with no diagnostic, despite the file's own
  `// STEPPE_LOG_WARN (teardown)` include comment (dead include, also [3.3]/[8.2]).
  Suggested: fold close into a single wrapper deleter and emit the warn once.

- [16.3][LOW] src/device/cuda/block_sink.cuh:80-106 (HostRamSink) & 117-156
  (DiskSink) — both have a user-declared freeing dtor (rule-of-five trigger) but
  neither EXPLICITLY declares/deletes copy/move; they are non-copyable only
  IMPLICITLY via their `std::mutex`/`condition_variable` members. Safe today,
  but ownership posture is incidental. Suggested: explicitly `= delete` copy and
  move on both sinks.

- [16.3][LOW] src/device/cuda/p2p_combine.cu:82-85,217-220 — local `struct
  DeviceGuard` has a side-effecting (`cudaSetDevice`) dtor but declares no
  copy/move, so it stays implicitly COPYABLE → a copy would re-fire the restore.
  Not triggered (aggregate-init in place, never copied); it is a scope-restore
  guard not a freeing owner, so a "double" is a redundant rebind, not a
  double-free. Suggested: `= delete` copy/move, or lift to one file-scope
  move-only scope guard (also resolves the Group 7.2 dup).

- [16.2][LOW] src/device/device_f2_blocks.hpp:75,53,48-51 — `=default` move
  (device_f2_blocks.cu:22-23) nulls `impl` but leaves moved-from shape scalars
  `P`/`n_block`/`device_id` stale → `empty()`/`size()` on a moved-from husk can
  contradict the documented "impl null iff empty()" invariant. Accessors stay
  null-safe and no caller queries a moved-from husk → latent doc blemish, not a
  free bug. Suggested: reset shape scalars on a user-written move, or weaken the
  doc to "impl null ⇒ no resident buffers (husk may report stale shape)".

- [16.2][LOW] src/device/cuda/pinned_buffer.cuh:250-251,280-286 —
  `PinnedRegistryCache` `=default` move correctly nulls each `Slot::reg`
  (`RegisteredHostRegion`) but COPIES the trivial `Slot::ptr`/`bytes`
  bookkeeping, leaving stale `(ptr,bytes)` against a now-null `reg`. BENIGN: the
  hit-scan gates on `s.reg.registered()` (false on moved-from), so no false
  cache hit; not a double-unregister (resource lives in `reg`). Suggested: leave
  as-is; optionally note `Slot::ptr/bytes` are valid only when `reg.registered()`.

## 4. Cross-cutting patterns

1. Unwrapped CUDA events are the dominant real risk. The two MED 16.1 findings
   plus the 16.2 MED and the 16.5 Ring finding all reduce to the same root: a
   raw `cudaEvent_t` (block_sink `SinkSlot::done`, cuda_backend `Ring::reuse`)
   that the project's existing `Event` wrapper (stream.hpp) already solves. A
   single fix — adopt/expose `Event` at these two sites — closes both the
   leak-on-throw (16.1) and the double-free-on-move (16.2) latent hazards, and
   removes the bespoke teardown/RingGuard scaffolding (16.5).

2. Implicit-only ownership posture. Several freeing-dtor types are
   non-copyable/non-movable only INCIDENTALLY — via a `std::mutex`/`cond_var`
   member (block_sink sinks) or by aggregate-init-and-never-copy (p2p_combine
   `DeviceGuard`). They are safe now but robust only by accident; the 16.3 LOWs
   ask for explicit `= delete`.

3. Moved-from invariant drift on `=default` moves. Where a wrapper defaults its
   move, the owning handle is nulled correctly but adjacent trivial scalars
   (shape in device_f2_blocks, `ptr/bytes` in PinnedRegistryCache) are not —
   benign/latent doc-vs-state mismatches, never a double-free.

4. The canonical wrappers are sound. `DeviceBuffer`, `PinnedBuffer`,
   `RegisteredHostRegion`, `Stream`, `Event`, `CublasHandle`, `CusolverDnHandle`
   all pass 16.1-16.5 (copy deleted, `std::exchange` null-on-move, full
   rule-of-five, noexcept teardown). 16.4 is clean across ALL units: raw kernel
   `double*` are consistently borrowed views, owners cross by `const&`/span and
   only ever transfer by RETURN-by-move. The remaining defects are at hand-rolled
   sites that should reuse these existing types — i.e. wrapper-hygiene drift, not
   a broken ownership model.

## Headline

Units in scope: 25 (19 clean / 6 with findings). Total findings: 10
(0 HIGH, 3 MED, 7 LOW). #HIGH = 0.

# Review — `src/device/cuda/p2p_combine.cu` (+ `src/device/p2p_combine.hpp`) (unit: device-cuda-p2p_combine)

Reviewer: ADVERSARIAL second pass (perf-first) against `docs/architecture.md` (§2, §4, §7, §8, §9, §11.2, §11.4, §12, §13), `docs/ROADMAP.md` (§4/§5/§6), and the M0–M4 per-unit reviews in `docs/cleanup/*.md`.
Scope: **`combine_f2_partials_p2p`** — the OPT-IN device-resident `cudaMemcpyPeer` fixed-order f2 combine (the M4.5 capability-tiered fast-path), its CUDA-free declaration `p2p_combine.hpp`, the `place_add_f2_kernel`, and the `validate_partials` guard. Read line-by-line alongside its directly-related context, all re-read this pass: the host-staged sibling `core::combine_f2_partials_host` (`f2_combine.{hpp,cpp}`), the sole caller `core::compute_f2_blocks_multigpu` (`f2_blocks_multigpu.cpp`), `F2BlockTensor` (`include/steppe/fstats.hpp`), `DeviceShard`/`plan_block_shards` (`shard_plan.hpp`), `Resources`/`CombinePath`/`PerGpuResources` (`resources.hpp`), `enable_peer_access`/`prefer_p2p_combine` (`config.hpp`), the RAII owners `DeviceBuffer<T>` (`device_buffer.cuh`), `Stream`/`Event` (`stream.hpp`), the check macros (`check.cuh`), the launch-math home (`launch_config.hpp`), `STEPPE_ASSERT`/`STEPPE_DEBUG_ONLY` (`host_device.hpp`), the per-device producer `cuda_backend.cu`, and the parity gate `tests/reference/test_f2_multigpu_parity.cu`.

Method note: every CUDA-behavior claim was re-checked against the official NVIDIA docs and is cited inline. Each pre-existing finding was re-verified against the actual code; false positives are moved to "Considered & rejected"; the auditor's draft (7.5/10, 27 findings, 11 perf) was treated as a hypothesis, not ground truth. This pass adds **four findings the first pass missed** (N1–N4), corrects one over-stated claim (P5 "silently synchronous" → "partially synchronous"), and re-frames P4/C2 with a fact the first pass did not establish (cudaMemcpyPeer works *without* peer access enabled). Parity is LOCKED and PROVEN (the `memcmp` gate passes on EmulatedFp64{40} across G and to single-GPU, both tiers); the job is to make this unit better-optimized and architecturally cleaner **without** regressing the §12 bit-identity contract.

Re-verified device-behavior sources (all quotes below are from these):
- [CUDA Runtime API — API synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html): *"For transfers from device memory to device memory, no host-side synchronization is performed"*; *"For transfers from pageable host memory to device memory, a stream sync is performed before the copy is initiated. The function will return once the pageable buffer has been copied to the staging memory for DMA transfer to device memory, but the DMA to final destination may not have completed."*
- [CUDA Runtime API — Peer Device Memory Access](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html): cudaDeviceEnablePeerAccess access is *"unidirectional"*; *"all allocations from peerDevice will immediately be accessible by the current device"* (prior AND future allocations); enabled until disabled or device reset (process-persistent); `cudaErrorPeerAccessAlreadyEnabled` if already active.
- [CUDA C++ Programming Guide — Programming Systems with Multiple GPUs](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html): cudaMemcpyPeer does **not** require peer access enabled — *"peer-to-peer memory copies between these two devices no longer need to be staged through the host and are therefore faster"* (i.e. peer access is a speed optimization, not a precondition); NULL-stream cross-device copy *"does not start until all commands previously issued to either device have completed and runs to completion before any commands issued after the copy to either device can start."*
- [CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/): grid-stride *"support[s] any problem size even if it exceeds the largest grid size"*; *"thread reuse amortizes thread creation and destruction cost"*; stride = blockDim.x·gridDim.x keeps *"addressing within warps … unit-stride, so we get maximum memory coalescing."*

---

## Role & layering

This TU is the device-resident counterpart of `core::combine_f2_partials_host`: GPU 0 (the combine root) pulls each peer device's compact `[P × P × nb_local]` partial across the fabric and sums it on-device, in the fixed `g = 0..G-1` order, into a zero-initialized full accumulator, then copies the result back to host as an `F2BlockTensor` (arch §11.4 "capability-tiered combine," §12 PARITY LAW). The CUDA-free/CUDA split is **exemplary** and matches the project's `backend_factory.hpp`/`cuda_backend.cu` precedent (verified): `p2p_combine.hpp` includes only `<span>`, `fstats.hpp`, `shard_plan.hpp` — no `<cuda_runtime.h>` — so the CUDA-free core entry point `compute_f2_blocks_multigpu` (`steppe::core`) reaches the device combine without dragging the CUDA toolkit into `steppe_core` (arch §4). The `.cu` includes only device-private CUDA headers (`check.cuh`, `device_buffer.cuh`) plus CUDA-free headers. It owns no raw `cudaMalloc`/`cudaFree` — all device memory is `DeviceBuffer<double>` (arch §2 RAII, §7).

The combine **policy** sits in the right layer: the §4 gate `prefer_p2p_combine && gpus[0].caps.can_access_peer` lives in `compute_f2_blocks_multigpu` (verified `f2_blocks_multigpu.cpp:170-191`; the `G>=2` half of the gate is structural — `G==1` returns at line 88 before any combine), not here. This routine is "the chosen path," not the chooser. The which-path tag is off the numeric payload (`Resources::last_combine_path`, set by the caller at `f2_blocks_multigpu.cpp:183`).

The problems concentrate on the **performance** axis the workflow flagged, plus a cluster of **idiom/contract** gaps the first pass missed: the unit (a) does a documented host→peer→root **double-bounce** for data that is already host-resident, (b) hard-`cudaDeviceSynchronize`s per peer partial, (c) re-enables peer access on every loop iteration (and ignores the `enable_peer_access` knob entirely — N2), (d) launches a non-grid-stride monolithic kernel **directly inline** rather than behind the project-wide narrow `void launch_xxx` wrapper (N1, §7), and (e) carries a comment that factually misstates `cudaMemcpyPeer`'s host-sync behavior (N3). None break parity. The honest counterweight — and the reason this is not lower — is that arch §11.4 itself classifies this combine as *"architectural cleanliness, not a throughput lever — steppe is deliberately P2P/NVLink-insensitive,"* and the only P2P-capable tier (PRO 6000) is currently spun down, so this fast-path is **dark on the live 5090 box**. But a senior engineer will not sign off at 9+ on the file literally named "the prime perf target" while it violates the §7 streams/events, grid-stride, narrow-launch-wrapper, and async-pinned idioms this directly, and leaves a documented config knob unwired.

---

## Score: 7.5/10 — parity-correct and layering-exemplary, but a serialized host-round-trip double-bounce with per-partial device sync, an inline non-grid-stride kernel that can silently under-cover in release, a per-iteration peer-enable, an unwired `enable_peer_access` knob, a doc comment that contradicts the CUDA API, and a `long`/`size_t` cast tangle

I confirm the first pass's 7.5. The verification *raised* my confidence in the perf cluster (every CUDA claim it leaned on checks out against the docs) but also surfaced four substantive misses (N1–N4) and one over-statement (P5), which net out at the same score: the new idiom/contract findings are real but mostly LOW–MED, and they are offset by the architecture's own explicit de-prioritization of this combine's throughput. The correctness and the parity argument are genuinely strong and airtight. The gap to 9.5–10 is the dense stack of §7 idiom violations (streams/events, grid-stride, narrow launch wrapper, async-pinned, no-hot-path-alloc) in the named perf exemplar, the host→peer→root double-bounce, and the unwired knob — all parity-safe to fix.

---

## Findings

### (A) Performance (first-class this pass)

**[P1 — HIGH, PARITY-SAFE: yes] The peer path does a needless host→peer→root DOUBLE-BOUNCE for data that is already host-resident; the M4.5-local optimum is a single host→root H2D.** *(CONFIRMED)*
Location: peer branch, lines 252–319 (esp. 284–300). For a peer-owned partial: (1) `cudaSetDevice(owning_device)`, (2) allocate `dPeer_f2`/`dPeer_vp` on the peer, (3) `cudaMemcpy` H2D host→peer, (4) `cudaSetDevice(root)`, (5) `cudaMemcpyPeer` peer→root into `dStage_*`. That is two device copies of the same bytes plus a peer allocation, when the source `part.f2`/`part.vpair` are host `std::vector<double>` — verified: the producer `cuda_backend.cu` allocates `dF2`/`dVpair` as `DeviceBuffer`s in `compute_f2_blocks`, copies them to the host `out` (lines 202–209), and frees them at scope-exit `return out` (line 210). So by the time the combine runs, every partial is host-resident. The staging buffer on the root could be filled by a **single** `cudaMemcpy(dStage_*, part.*.data(), …, HostToDevice)` — exactly what the root's own branch (lines 248–251) already does.
Why it matters: arch §11.4 "data bouncing." This moves `2·P²·nb_local·8` bytes twice plus a peer malloc/free on the hot loop, where one H2D would do. The `.cu` header (lines 18–24) is candid that "the H2D pre-stage is a PERFORMANCE wart" but frames `cudaMemcpyPeer` as load-bearing for "exercising the credited P2P transport" — it is load-bearing for the *demo of P2P*, not for *this combine's result*.
Adversarial check / the tension: the single-H2D shortcut makes the routine **not exercise `cudaMemcpyPeer` at all**, which is the literal point of the "P2P device-resident" tier and what the PRO-tier parity assertion (3) + the `last_combine_path == P2pDeviceResident` tag (test lines 487–488) verify. So P1 is correct *as a perf observation* but collides with the unit's stated mission. The clean resolution is P2 (keep the partial device-resident so `cudaMemcpyPeer` is a *genuine* pull). The header flags a device-resident per-device compute as "OUT OF SCOPE for M4.5" (lines 22–24). Mark P1 as the correct critique whose proper resolution is P2, not a blind deletion.
Severity: HIGH. Effort: S for the deletion / L for the P2 resolution. Parity-safe: yes — both variants move identical bytes and add in identical fixed order.

**[P2 — HIGH, PARITY-SAFE: yes-if-careful] The architecturally-correct fix is a device-resident per-device partial so `cudaMemcpyPeer` becomes a true peer→root pull — but it touches `compute_f2_blocks`, so it is a scoped M5 follow-up, NOT an M4.5-local edit.** *(CONFIRMED)*
Location: design-level; producer `cuda_backend.cu:202-210` (frees device buffers after the D2H), consumer `f2_blocks_multigpu.cpp:152` (takes the host `F2BlockTensor`).
The double-bounce exists only because the partial is host-resident at combine time. If the per-device backend retained its `[P × P × nb_local]` f2/vpair on its device (or the orchestrator passed device handles), the peer branch would be a *single* `cudaMemcpyPeer(dStage, root, dPartial, owning, bytes)` — the genuinely fast P2P shape, and the only configuration where the credited 55.6 GB/s DMA beats host-staging (the doc: peer copies "no longer need to be staged through the host and are therefore faster"). This makes P1's tension vanish: real P2P *and* no bounce.
Adversarial check: this is the only finding that touches the parity-critical hot path. Parity-safe **only if** the retained device bytes are bit-identical to today's host bytes (they are — the D2H is byte-exact) and the per-device GEMM batch shape is not perturbed (that is the documented native-FP64 sensitivity cell, test lines 177–214). Hence "yes-if-careful," explicitly deferred to M5.
Severity: HIGH. Effort: L. Parity-safe: yes-if-careful.

**[P3 — HIGH, PARITY-SAFE: yes] A full `cudaDeviceSynchronize()` after every peer pull serializes the whole combine; it should be an event/stream fence (or be eliminated with P1/P2). The hazard the sync guards is REAL.** *(CONFIRMED; the doc verifies BOTH that a fence is needed AND that the sync is the wrong fence)*
Location: line 316, inside the peer branch, once per peer partial.
The comment (302–315) correctly identifies the hazard: `dPeer_*` (the peer-resident source) is freed at iteration end (RAII dtor), and freeing the source buffer races the cross-device DMA. I verified the doc: *"For transfers from device memory to device memory, no host-side synchronization is performed"* — so `cudaMemcpyPeer` does **not** block the host until the DMA completes, and *some* fence IS required before `dPeer_*` is reclaimed. The code is right that a fence is needed; `cudaDeviceSynchronize()` is just the heaviest possible one (it blocks the host until *all* root-device work is done), which (a) prevents overlap of the next H2D with the current place-add, and (b) is far stronger than the actual dependency.
Why it matters: arch §7 "express cross-stream dependencies with an Event, never a device-wide sync." With G devices this is G full device drains back-to-back.
Concrete fix: with P1/P2 the `dPeer_*` lifetime problem disappears (no peer buffer) and the sync is *deleted* — the place-add reads `dStage_*`, filled by an H2D on the same stream, so stream ordering suffices. If the peer form is kept: issue `cudaMemcpyPeerAsync` on a root `Stream`, `Event::record` after it, and `Event::wait` it on both the place-add stream and before the `dPeer_*` free (`stream.hpp` already provides `record`/`wait`).
Severity: HIGH. Effort: S (delete under P1) / M (event fence). Parity-safe: yes.

**[P4 — MED, PARITY-SAFE: yes] `cudaDeviceEnablePeerAccess` is called inside the per-partial loop every iteration; it should be hoisted to once-per-(root,peer) before the loop — and is not even required for correctness.** *(CONFIRMED + STRENGTHENED)*
Location: line 265, inside `for g`, peer branch.
Verified from the docs: peer access *"will remain accessible until access is explicitly disabled … or either device is reset"* (process-persistent) and covers *"all allocations from peerDevice"* (prior + future). So one enable per (root, peer) pair suffices; calling it every iteration means every call after the first returns `cudaErrorPeerAccessAlreadyEnabled`, which the code then launders via `STEPPE_CUDA_WARN` + a `cudaGetLastError()` scrub (266–278 — see C2). **New fact the first pass missed:** `cudaMemcpyPeer` does **not** require peer access at all — without it the copy still works, just staged through host. So the enable is *purely a perf optimization*; the per-iteration call is doubly wasteful (a redundant driver round-trip for an optimization that is already in effect after the first call).
Concrete fix: enable once per distinct non-root `owning_device` before the loop. Better, per arch §11.4 ("enable opportunistically"), do the enable once in `build_resources`, gated on `config.enable_peer_access` (see N2), so the combine assumes it and only `STEPPE_CUDA_CHECK`s the DMA.
Severity: MED. Effort: S. Parity-safe: yes (transport setup, parity-neutral).

**[P5 — MED, PARITY-SAFE: yes] The H2D uploads are synchronous `cudaMemcpy` on pageable host memory; with pinned staging + `cudaMemcpyAsync` on a stream they could overlap the place-add — but the pageable copy is *partially* synchronous, not fully blocking (first-pass over-stated this).** *(CONFIRMED with correction)*
Location: lines 248–251 (root H2D), 287–290 (peer H2D). All copy from `part.f2.data()`/`part.vpair.data()` — pageable `std::vector<double>`.
Correction to the first pass: it called the pageable `cudaMemcpy` "silently synchronous." The doc is more precise — *"For transfers from pageable host memory to device memory, a stream sync is performed before the copy is initiated. The function will return once the pageable buffer has been copied to the staging memory for DMA transfer … but the DMA to final destination may not have completed."* So the call is **partially** synchronous: it stalls on the prior stream work and on the host→staging copy, then returns while the DMA may still be in flight. It is not a full host stall, but it cannot overlap subsequent host-issued work the way a pinned async copy can, and the leading stream-sync defeats pipelining.
Why it matters: arch §7 "async pools, STREAMS/EVENTS for overlap … missing pinned memory for async copies." To pipeline (P6) the source must be pinned and the copy async on a real stream.
Concrete fix: stage through a `PinnedBuffer<double>` (allowlisted but **not yet implemented** — verified `find` returns nothing; `device_buffer.cuh:5` only *anticipates* it) and `cudaMemcpyAsync` onto a root `Stream`. M4.5-local form: pin a *scratch* host buffer and memcpy into it before the async H2D — only worth it if the combine is ever on the critical path.
Adversarial check / ROID: arch §11.4 says the combine is off the critical path; on the live box (no P2P) this routine never runs. So P5 is correct-by-the-idiom but low-ROI; it matters because the unit is the *named* perf exemplar and currently models the anti-pattern. Keep MED, ROI flagged.
Severity: MED. Effort: M (needs the pinned wrapper). Parity-safe: yes.

**[P6 — MED, PARITY-SAFE: yes-if-careful] The G partials are pulled and added strictly sequentially with zero copy/compute overlap; the COPIES can pipeline via a ring of staging buffers + streams + events, while the ADDS stay serialized in fixed g order.** *(CONFIRMED; strengthened by N4)*
Location: the whole `for g` loop, 223–338. Each iteration fully completes (stage → DMA → sync → place-add ×2) before the next begins.
The combine has a true serial dependency only on the *accumulator add order* (§12), NOT on the *transport*. A clean shape: a ring of K staging buffers + K streams; issue partial g's H2D/DMA on stream `g mod K`, record an event; the place-add for g runs on a *single accumulator stream* and `cudaStreamWaitEvent`s on g's copy event — because all place-adds share one stream, they execute in issue order = fixed g order. Overlaps transport with compute, preserves the exact add order.
Adversarial check — the parity trap (and N4): the place-add must remain in fixed g order *and* see the fully-arrived `src`. With disjoint block-aligned shards each global slab is written by exactly one device (lines 37–41), so the adds are to **disjoint** accumulator regions and would be order-independent *for the bytes* — BUT §12 and the parity test pin the *literal* fixed-order sum, and the comment is explicit the `+=` "keeps this the literal §12 fixed-order sum." Therefore overlap the *copies* freely, keep the place-adds serialized on one accumulator stream in g order. **N4 corollary:** a *single* reused `dStage_*` (today) cannot be the ring — partial g+1's copy would overwrite `dStage_*` while g's place-add still reads it (a Write-After-Read hazard currently masked only by NULL-stream serialization + the per-partial device sync, see N4). So P6 *requires* the K-deep staging ring, not just streams.
Severity: MED (off critical path; the win is real only on the PRO tier with many G). Effort: M–L. Parity-safe: yes-if-careful — overlap transport, never reorder the accumulator adds.

**[P7 — MED, PARITY-SAFE: yes] `place_add_f2_kernel` is a monolithic one-element-per-thread kernel with no grid-stride loop; in a RELEASE build it can silently under-cover a pathological count. Make it grid-stride.** *(CONFIRMED + correctness-elevated)*
Location: kernel lines 82–92; launch 327–337.
The kernel computes one global index and a single `if (k < count)` add — the "monolithic kernel [that] assumes that the thread grid is large enough to cover the entire data array" the NVIDIA Pro Tip contrasts against grid-stride. A grid-stride loop gives *"support [for] any problem size even if it exceeds the largest grid size,"* *"thread reuse [that] amortizes thread creation,"* and preserves coalescing (unit-stride within a warp). **Correctness angle the first pass under-weighted:** the code guards the grid extent with `STEPPE_ASSERT` (329–330), which I verified is *debug-only* (`host_device.hpp:63-67`: compiled out under NDEBUG). So in **release**, an over-`kMaxGridX` count (the comment's own P=768/nb=757 example is far below, but a pathological shape — e.g. the §0 top end P=4266 with a large single shard — could exceed it) would launch a grid covering only `kMaxGridX·block` elements and **silently under-cover** the rest — exactly the "silent under-cover" the comment claims to be preventing. The grid-stride form makes a *fixed-size* grid cover any count, removing the launch trap by construction.
Concrete fix:
```cuda
__global__ void place_add_f2_kernel(double* __restrict__ acc, const double* __restrict__ src,
                                    long acc_base, long count) {
    for (long k = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         k < count; k += static_cast<long>(blockDim.x) * gridDim.x)
        acc[acc_base + k] += src[k];
}
```
and size the grid by occupancy (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`) or a generous SM-count multiple, capped at `kMaxGridX`.
Adversarial check — parity: a grid-stride loop changes which thread touches which element, but each element is written exactly once with the same `+=` onto the same `acc[acc_base+k]`; no cross-element reduction, no reassociation. Byte-identical.
Severity: MED (now correctness-adjacent in release, not just perf). Effort: S. Parity-safe: yes.

**[P8 — LOW, PARITY-SAFE: yes] The two place-add launches (f2, then vpair) are independent but serialized on the NULL stream; fuse them into one kernel pass (matches the host baseline's single interleaved loop).** *(CONFIRMED)*
Location: 332–337 — two launches, both on the default stream, each with `STEPPE_CUDA_CHECK_KERNEL()`.
The f2 and vpair adds are fully independent (disjoint buffers). On the NULL stream they serialize. Cleanest: **fuse** into one kernel adding both `f2` and `vpair` for the same `k` (one launch, one index computation). The fused form mirrors the host baseline, which interleaves the two adds in one loop (verified `f2_combine.cpp:124-125`: `out.f2[...] += ...; out.vpair[...] += ...;`). This also halves the launch+check overhead and the index math.
Adversarial check — parity: fusing two disjoint placement-adds changes neither operand nor add order. Byte-identical.
Severity: LOW. Effort: S. Parity-safe: yes.

**[P9 — LOW, PARITY-SAFE: yes] The whole routine runs on the default NULL stream, which cross-device-serializes; explicit streams enable P3/P5/P6 — but note the NULL stream currently provides FREE correctness that the refactor must replace with events.** *(CONFIRMED + nuance)*
Location: every CUDA call here uses the implicit NULL stream.
Verified: a NULL-stream cross-device copy *"does not start until all commands previously issued to either device have completed and runs to completion before any commands issued after the copy to either device can start"* — a global cross-device barrier per partial, compounding P3. **Nuance the refactor must respect:** that same barrier is what currently makes the peer branch's H2D-to-`dPeer` (on the owning device) safely complete before the `cudaMemcpyPeer` reads `dPeer` (on the root), and makes the place-add of g safely finish reading `dStage_*` before g+1's H2D overwrites it (N4). Moving to explicit streams (P9) removes these implicit guarantees, so P9 is a *prerequisite* for P3/P6 but only correct when paired with the explicit `Event` fences they introduce.
Concrete fix: a root accumulator `Stream` + per-pull staging streams; every copy/kernel on explicit streams; the H2D→peercopy and place-add→reuse orderings expressed via `Event`.
Severity: LOW (subsumed by P3/P6). Effort: S as a prerequisite. Parity-safe: yes (with the events).

**[P10 — LOW, PARITY-SAFE: yes] The host `out.f2`/`out.vpair` are zero-filled `assign(total, 0.0)` then ENTIRELY overwritten by the D2H copy; `resize(total)` suffices.** *(CONFIRMED)*
Location: device zero-init 184–187 (needed); host zero-init 212–213; D2H overwrite 341–346.
The device accumulators must be zeroed (the placement-onto-0.0 argument). But the host `out.f2`/`out.vpair` are zero-filled and then fully overwritten by the D2H of the complete device accumulator. So `assign(total, 0.0)` does `total` redundant value-init stores; `resize(total)` (no value-init) suffices since every element is overwritten. (The host baseline `f2_combine.cpp` genuinely needs the 0.0 init because *it* accumulates host-side; this device path does not.) Note `block_sizes` still needs its `assign(.., 0)` (placed sparsely, not fully overwritten).
Adversarial check — parity: `resize` leaves f2/vpair uninitialized, but every element is then written by the D2H of a fully-populated device accumulator. The `covered == n_block_full` invariant (validated, 135–140) guarantees full tiling, so the device accumulator has no holes → host result byte-identical. Keep the validation.
Severity: LOW. Effort: S. Parity-safe: yes.

**[P11 — LOW, PARITY-SAFE: yes] Per-partial peer staging buffers `dPeer_f2`/`dPeer_vp` are allocated and freed inside the loop — allocations on the hot path.** *(CONFIRMED)*
Location: 285–286, freed at iteration end. With P1/P2 these vanish; if the peer form is kept, size them to `stage_elems` (the max) and hoist out of the loop like `dStage_*` already are (199–202), reused across pulls, to avoid a `cudaMalloc`/`cudaFree` per peer partial (arch §7 "no allocations on the hot path").
Severity: LOW. Effort: S. Parity-safe: yes.

### (B) Idioms / contract gaps the first pass MISSED (new this pass)

**[N1 — MED, PARITY-SAFE: yes] §7 narrow-launch-wrapper idiom violated: `place_add_f2_kernel<<<>>>` is issued INLINE in the orchestration function; every other kernel in the codebase is behind a `void launch_xxx(...)` wrapper.** *(NEW — first pass missed)*
Location: launch sites 332–337, inside `combine_f2_partials_p2p`.
Arch §7 is explicit: *"Kernels are exposed via narrow `void launch_xxx(...)` wrappers so host code never includes kernel bodies or `<<<>>>`."* I verified this is a **pervasive, enforced convention** elsewhere: `launch_decode_af` (`decode_af_kernel.cu:90`), `launch_f2_feeder`/`launch_assemble_f2` (`f2_block_kernel.cu:307,375`), `launch_gather_group`/`launch_assemble_blocks_group` (`f2_blocks_kernel.cu:171,267`) — and `f2_block_kernel.cuh:5-6` quotes the rule verbatim: *"Host orchestration (cuda_backend.cu) calls these `void launch_*` … the kernel bodies and `<<<>>>` live only in the .cu."* `p2p_combine.cu` is the **only** TU in `src/device/cuda/` that issues a `<<<>>>` from inside its host orchestration rather than from a narrow wrapper.
Adversarial check — is the rule even applicable here? The §7 rationale ("host code never includes kernel bodies") is weaker in *this* file because the host orchestration and the kernel live in the *same* TU (unlike the others where orchestration sits in `cuda_backend.cu`). So this is not the latent-recompile/leak hazard §7 primarily targets. BUT (a) the launch-config/grid-extent-assert math is currently smeared into the orchestration body (327–331), which a `void launch_place_add(double* acc, const double* src, long acc_base, long count, cudaStream_t stream)` wrapper would single-home alongside the kernel (and is exactly where the P7 grid-stride/occupancy sizing belongs), and (b) codebase consistency is itself a §2/§8 value. Real, but MED-not-HIGH given the same-TU mitigant.
Concrete fix: extract a `void launch_place_add(...)` wrapper in this `.cu` (taking a stream, ready for P9) holding the grid math + `STEPPE_CUDA_CHECK_KERNEL`; the orchestration loop calls it twice (or once, fused, under P8).
Severity: MED. Effort: S. Parity-safe: yes.

**[N2 — MED, PARITY-SAFE: yes] The `enable_peer_access` config knob is UNWIRED: this routine calls `cudaDeviceEnablePeerAccess` unconditionally, and the documented "MAY-WE" gate is read NOWHERE in the codebase.** *(NEW — first pass missed)*
Location: line 265 (unconditional enable); the knob is `config.hpp:260`.
`config.hpp:255-260` documents `enable_peer_access` as *"the MAY-WE knob: whether the backend is permitted to call cudaDeviceEnablePeerAccess at all. DISTINCT from `prefer_p2p_combine`."* I grepped the entire `src/` tree: the **only** occurrences of `enable_peer_access` are doc comments (`backend.hpp:168`, `resources.hpp:101`) — **no code ever reads it.** Meanwhile this routine calls `cudaDeviceEnablePeerAccess(owning_device, 0)` unconditionally (line 265), and the capability probe (`cuda_backend.cu` `capabilities()`) only calls `cudaDeviceCanAccessPeer` (a query), never gating the *enable* on the knob. So a user who sets `enable_peer_access=false` (documented to mean "do NOT call cudaDeviceEnablePeerAccess at all") still has it called inside the combine. The knob is dead.
Why it matters: arch §9 typed-immutable-config / §2 fail-fast-not-fail-silent; a config field that silently does nothing is worse than absent (it implies a guarantee it does not deliver). This is a contract-coherence bug, partly out-of-unit (the cleanest home for honoring it is the probe/`build_resources`, per arch §11.4 "enable opportunistically"), but the unconditional call lives **here**.
Concrete fix: honor the knob. Either (a) do the enable once in `build_resources` gated on `config.enable_peer_access` and have this routine assume it (best — folds into P4); or (b) thread `enable_peer_access` to this routine and skip the enable when false (relying on the doc fact that `cudaMemcpyPeer` still works without it, just host-staged). Mark the knob unwired until then.
Adversarial check — is it harmless because `prefer_p2p_combine` already gates the path? No: the two knobs are documented as *orthogonal* (`config.hpp:264-267`); `enable_peer_access=false, prefer_p2p_combine=true` is a legal, meaningful combination ("I want the P2P combine but never let you flip the driver's peer-access state") that the current code silently ignores.
Severity: MED. Effort: M (touches probe/config wiring). Parity-safe: yes (peer-enable is parity-neutral).

**[N3 — LOW, PARITY-SAFE: yes] The sync comment (lines 305–307) factually MISSTATES `cudaMemcpyPeer` as "host-blocking for the DMA itself"; the official API contradicts it, and the comment is internally self-contradictory.** *(NEW — first pass missed; first pass cited the doc correctly in P3 but did not flag the wrong source comment)*
Location: comment lines 304–307: *"cudaMemcpyPeer enqueues onto the destination device's NULL stream and is host-blocking for the DMA itself, but … freeing the SOURCE peer buffers … races the DMA's cross-device completion unless we fence here."*
The doc states the opposite for device→device: *"For transfers from device memory to device memory, no host-side synchronization is performed."* So `cudaMemcpyPeer` does **not** host-block for the DMA. The comment is also internally inconsistent — if it truly blocked the host for the DMA, there would be *no* race with the subsequent free, and *no* fence would be needed; the very next clause asserts the race that the doc (not the comment's premise) actually justifies. The *conclusion* (a fence is required) is correct; the *stated reason* is wrong. A load-bearing comment in a parity-/correctness-critical unit must not assert a CUDA semantic the docs refute.
Concrete fix: replace with the accurate statement: "`cudaMemcpyPeer` does device→device work with no host-side synchronization (CUDA API sync behavior), so the host returns before the DMA completes; freeing the peer source races the DMA unless we fence — hence the sync (to be replaced by an Event, P3)."
Severity: LOW (comment-only; behavior is correct). Effort: S. Parity-safe: yes.

**[N4 — LOW→MED, PARITY-SAFE: yes-if-careful] The single reused `dStage_*` buffer carries a Write-After-Read hazard across iterations that is currently masked ONLY by NULL-stream serialization + the per-partial device sync; any P6/P9 refactor must replace that masking with explicit fences (and a staging ring).** *(NEW — first pass's P6 mentioned a ring but did not name this as the load-bearing reason)*
Location: `dStage_f2`/`dStage_vp` allocated once (201–202), written by the H2D/peer-copy each iteration (248–251 / 297–300), read by the place-adds (332–337).
Iteration g's place-add reads `dStage_*`; iteration g+1's H2D (root branch) or `cudaMemcpyPeer` (peer branch) overwrites the *same* buffer. Today this is safe: in the root branch both the place-add and the next H2D are on the root NULL stream (serialized); in the peer branch the `cudaDeviceSynchronize` (316) drains everything first. So the reuse is correct *by accident of the NULL stream + the heavy sync*. This is the concrete reason P6's overlap needs a **K-deep staging ring** (not just streams): a single reused `dStage_*` cannot be filled for g+1 while g's place-add still reads it. Flag so the perf refactor does not introduce a silent data race (which `compute-sanitizer --track-stream-ordered-races`, arch §13, would catch — but better designed-out).
Severity: LOW today (no bug), MED as a refactor-safety constraint. Effort: folded into P6. Parity-safe: yes-if-careful.

### (C) Type-casting noise / width

**[C1 — MED, PARITY-SAFE: yes] `device_ids` / `root_device_id` (and the peer-vs-root distinction) are pure transport plumbing the RESULT does not depend on; they become dead once P1/P2 land.** *(CONFIRMED)*
Location: signature `int root_device_id`, `std::span<const int> device_ids` (`.cu:148-149`); used only for the root-vs-peer branch (243) and as `cudaMemcpyPeer`/`cudaSetDevice` args. The header (60–62) is explicit the *result* is "identical regardless of which physical ordinal stages each partial." The caller builds `device_ids` afresh each call (verified `f2_blocks_multigpu.cpp:179-181`, a small host alloc). Under P1's single-H2D shortcut, `device_ids` and the peer distinction are entirely unused (the combine only needs *which device the accumulator lives on* = `root_device_id`, a single int). Under P2, both are replaced by device-buffer handles. Until then the param is justified (it IS needed to issue `cudaMemcpyPeer`), so this is contingent.
Severity: MED (contingent). Effort: S (under P1). Parity-safe: yes.

**[C2 — MED, PARITY-SAFE: yes] The `(void)cudaGetLastError()` scrub (266–278) is a 13-line-commented workaround for the per-iteration peer-enable WARN; it can mask a genuine sticky error and disappears with P4.** *(CONFIRMED)*
Location: line 278 + 13-line comment.
The reasoning is correct (`cudaGetLastError` reads-and-resets the sticky error), but the construct is brittle: a blanket `cudaGetLastError()` discards **any** outstanding sticky error, not just the tolerated `cudaErrorPeerAccessAlreadyEnabled`. The robust pattern branches on the WARN's returned status and scrubs only the tolerated one:
```cuda
cudaError_t pe = STEPPE_CUDA_WARN(cudaDeviceEnablePeerAccess(owning_device, 0));
if (pe == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();  // scrub only the tolerated one
else if (pe != cudaSuccess) /* genuine enable failure: let the DMA below surface it */;
```
With P4 (enable once, ideally in `build_resources`), the `AlreadyEnabled` status never appears on the hot loop and the scrub vanishes.
Adversarial check: is the blanket scrub dangerous today? The preceding calls (`cudaSetDevice`/`cudaMalloc`/`cudaMemcpy`) are checked via the *throwing* `STEPPE_CUDA_CHECK`, so a real error would already have thrown before the scrub — so the swallow risk is mostly theoretical *given the current sequence*. But it is a latent trap if the sequence changes. MED.
Severity: MED. Effort: S. Parity-safe: yes.

**[C3 — LOW, PARITY-SAFE: yes] `long` vs `std::size_t` vs `int` for the same element-count quantity is mixed within one function, forcing a re-cast.** *(CONFIRMED)*
Location: `slab`/`total`/`stage_elems`/`part_elems`/`part_bytes` are `std::size_t` (169–202, 237–238); `acc_base`/`cnt`/`grid_l`/`covered` are `long` (115, 239–241, 328); `P`/`n_block_full`/`max_local_nblock` are `int`. The *same* element count `P²·nb_local` is `std::size_t part_elems` (237) then re-cast to `long cnt` (241) for the kernel; `acc_base = long(slab)*long(b0)` while `slab` is `size_t`.
Why it matters: arch §2 "redundant/scattered int↔long↔size_t casts … a cleaner type/contract would remove." The kernel correctly uses signed `long` indices (good for grid-stride arithmetic). Cleanest contract: compute kernel-feeding counts/indices as `long` throughout, keep `std::size_t` only for `bytes()`/alloc sizes; then `cnt` is not a re-cast of `part_elems`.
Adversarial check — width safety: at the §0 top end (P=4266, nb=757), `slab ≈ 1.82e7`, `acc_base` up to `slab·757 ≈ 1.38e10` — exceeds `INT_MAX` but well within `long`/`size_t`, so `long` is correctly chosen over `int` (good). The casts are noisy, not wrong. LOW.
Severity: LOW. Effort: S. Parity-safe: yes.

**[C4 — LOW, PARITY-SAFE: yes] The `n_block_full < 0 ? 0 : n_block_full` clamp is repeated (172, 215) after `validate_partials` already rejects negative `n_block_full`.** *(CONFIRMED)*
Location: 172, 215; `validate_partials` throws on `P < 0 || n_block_full < 0` (111–114) *before* any of these. So `n_block_full >= 0` is guaranteed and the ternaries are dead defensive branches (mirrored from the host baseline `f2_combine.cpp:101,103`, which has the same redundancy). The kernel-side `cnt`/`acc_base` math has no such clamp, so the clamps are not even uniformly applied — pure noise.
Concrete fix: drop the ternaries (rely on the validated invariant) or assert once; keep in lock-step with the host baseline's choice (C5).
Severity: LOW. Effort: S. Parity-safe: yes.

### (D) Decomposition / DRY / single-responsibility

**[C5 — MED, PARITY-SAFE: yes] `validate_partials` is duplicated almost verbatim between this `.cu` and `f2_combine.cpp`; the contract has two homes kept in lock-step by comment.** *(CONFIRMED)*
Location: `p2p_combine.cu:100-141` vs `f2_combine.cpp:29-74`. I diffed them: identical logic (sizes equal, P non-negative, each partial's `n_block == shard span`, P agreement, full tiling), near-identical messages; the only delta is the P2P version also checks `device_ids.size()`. The `.cu` comment (96–99) says "Kept in lock-step with the host combine's guard so the two tiers reject identically" — a self-admission of the DRY violation. A drift (one tightens a bound) would make the tiers reject *differently*, which the parity-neutrality story forbids.
Concrete fix: hoist the shared partial/shard/P/tiling validation into a CUDA-free helper in `core` (both combines are CUDA-free); the P2P side adds only the `device_ids.size()` check. Messages can take a `const char* who` prefix or use `std::source_location`.
Severity: MED. Effort: M. Parity-safe: yes (pre-compute, no bytes touched).

**[C6 — LOW, PARITY-SAFE: yes] `combine_f2_partials_p2p` is a ~200-line function doing five jobs; the per-partial transport+add step is the natural extract.** *(CONFIRMED)*
Location: function body 145–349 — validation, device binding, allocation, host-result construction, the per-partial loop, copy-back. The loop body (223–338) is a self-contained `place_partial(g, …)` (or, post-P1, a trivial `stage_and_add`). The prior `cuda_backend.cu` review flagged a comparable monolith at the same bar.
Severity: LOW. Effort: M. Parity-safe: yes.

### (E) CUDA idioms / RAII / launch config (§7)

**[I1 — LOW, POSITIVE → minor] RAII is correct; the `DeviceGuard` restore-on-exit is a good pattern.** The `struct DeviceGuard` (163–166) restores the caller's device on every exit (including throw), via the non-throwing `STEPPE_CUDA_WARN` in its destructor (correct — dtors must not throw, arch §7). Worth keeping. Minor DRY nit: it duplicates the `Stream`/`Event`/`DeviceBuffer` teardown-WARN pattern; a tiny reusable `ScopedDevice` in `device/cuda/` would single-home it (arch §8), but the local struct is acceptable.

**[I2 — LOW, POSITIVE] `STEPPE_CUDA_CHECK_KERNEL()` is correctly placed after each launch** (334, 337) — the §7 post-launch idiom (verified `check.cuh:185-197`: `cudaGetLastError` for bad config + debug-only async sync). Good. (Under P8's fuse, it appears once.)

**[I3 — LOW→MED, PARITY-SAFE: yes] The grid-extent `STEPPE_ASSERT` (329–330) is debug-only; in RELEASE an over-`kMaxGridX` count silently under-covers.** *(CONFIRMED; elevated — see P7)* Verified `STEPPE_ASSERT` is removed under NDEBUG. So the assert does NOT protect a release build against the very "silent under-cover" its sibling comment warns of. The grid-stride form (P7) removes the failure mode entirely (a fixed grid covers any count) and makes the assert unnecessary. Severity raised from LOW to LOW→MED because it is a latent release-only correctness gap, not just a style point.

**[I4 — LOW, POSITIVE] `kPlaceAddBlockX = 256` is named, justified, routed through the shared launch math** (`cdiv`, `kMaxGridX`) — arch §4/§7/§8 compliant. With P7+occupancy sizing the block stays 256, the grid becomes occupancy-derived.

### (F) Correctness & edge cases

**[E1 — LOW, POSITIVE] Empty-shard / all-empty / zero-size paths are handled correctly.** `part.n_block <= 0` short-circuits (235) after placing `block_sizes` (a no-op loop for an empty shard); `total == 0` guards the `cudaMemset` (184) and the D2H (341); `DeviceBuffer(0)` is a documented no-op (verified `device_buffer.cuh` ctor `if (n)` guard). `stage_elems == 0` when all shards empty → null `dStage_*`, never dereferenced (loop early-continues). The single-GPU case never reaches here (caller special-cases `G==1`, `f2_blocks_multigpu.cpp:88`). Solid.

**[E2 — LOW, PARITY-SAFE: yes] `total` overflow is delegated to `DeviceBuffer`'s ctor guard; the host `out.f2.assign(total, 0.0)` is overflow-safe ONLY because the device alloc precedes it.** `total = slab * n_block_full` is a `std::size_t` product (171). `DeviceBuffer<double>(total)` rejects `total > SIZE_MAX/sizeof(double)` (verified `device_buffer.cuh:64-73`), and that alloc (182) runs *before* the host `assign` (212), so an overflowing request throws first. A reorder would expose the unguarded host `assign`. With P10 (`resize`) the same dependency holds. `slab = P*P` as `size_t` cannot overflow for `int P`. LOW (ordering-protected).

**[E3 — N/A] Misaligned inputs: N/A** — partials are dense `std::vector<double>`, device buffers are `cudaMalloc`'d (≥256-byte aligned), placement-add has no alignment hazard; block-aligned shards guarantee disjoint contiguous offsets (`acc_base = slab*b0`), validated to tile `[0,n_block_full)`.

### (G) Readability / naming / const-correctness / comments

**[R1 — POSITIVE, with one carve-out] Comment density and parity rationale are excellent — except the workaround comments (C2's 13-liner, N3's wrong claim).** The file documents *why* it is bit-identical (byte-exact transport + fixed-order placement-onto-0.0), cites the architecture, and is honest about the H2D pre-stage wart. That is the right load-bearing documentation for a §12-critical unit. The exceptions: the 13-line `cudaGetLastError` comment (C2) is a smell pointing at P4, and the sync comment's "host-blocking for the DMA itself" is factually wrong (N3) — those are not documentation to keep.

**[R2 — LOW, POSITIVE] `[[nodiscard]]` present on the public combine** (`p2p_combine.hpp:104`); `validate_partials` returns void (fine); the function is correctly not `noexcept` (it throws). Const-correctness is clean (`const F2BlockTensor& part`, `const DeviceShard& sh`, `const long acc_base`). No issue.

**[R3 — LOW] Naming acceptable.** `dStage_*`/`dPeer_*`/`dAcc_*` clear; `cnt`/`grid_l` terse but local. `max_local_nblock` as `int` is fine (block counts fit int). No action.

### (H) Layering / API / ABI (§4)

**[L1 — POSITIVE] The CUDA-free decl / CUDA def split is a model of arch §4** (verified: `p2p_combine.hpp` includes only `<span>`/`fstats.hpp`/`shard_plan.hpp`; the `.cu` is the only CUDA TU; `compute_f2_blocks_multigpu` calls it through the CUDA-free decl). Combine *policy* in the caller, combine *mechanism* here. Correct dependency direction.

**[L2 — LOW] `device_ids`/`root_device_id` leak a transport detail through the API** (C1) — the only layering nit, contingent on P1/P2. The CUDA-free seam cannot surface device handles, so threading ordinals is the current honest workaround (header 55–62 explains it). Acceptable until P2.

**[L3 — MED] `enable_peer_access` (N2) is a layering/contract gap:** the "MAY-WE" policy knob is documented at the public-config layer (`config.hpp`) but honored at no layer. The cleanest home is the probe/`build_resources` (device layer), not this routine — but this routine is where the unconditional `cudaDeviceEnablePeerAccess` actually fires, so it is co-responsible. See N2.

### (I) Testability (§13) & capability-tier coherence

**[T1 — POSITIVE] Capability-tier coherence is clean.** The which-path tag (`CombinePath::P2pDeviceResident`) is recorded out-of-band on `Resources::last_combine_path` by the caller (verified `f2_blocks_multigpu.cpp:183`; `resources.hpp:46-59,104-110`), NEVER on the numeric `F2BlockTensor` (arch §12). Probe at `build_resources`, gate in caller, "chosen path" here. The parity test reads the tag to confirm P2P *actually ran* on the PRO tier (test lines 486–488) and *degraded* on no-peer (495–496) — good observability.

**[T2 — MED, PARITY-SAFE: yes] No DIRECT unit test exercises the P2P path / `place_add_f2_kernel` in isolation; the parity test's P2P assertions are PRO-tier-gated, so on the live 5090 box (P2P disabled) this routine is NEVER exercised.** *(CONFIRMED)* I grepped `tests/`: the only reference to `combine_f2_partials_p2p`/`place_add` is the parity test, whose P2P arm runs only `if (resG2p.gpus[0].caps.can_access_peer)` (test lines 401, 428, 486–488). On the no-peer tier the routine degrades to host-staged in the caller, so the *transport* (cudaMemcpyPeer, peer-enable, double-bounce, sync) is exercised only on PRO hardware that is currently spun down. The *combine arithmetic* is covered (host-staged sibling is byte-identical), but a transport-independent unit test of `place_add_f2_kernel` + the fixed-order loop would let the combine logic be tested on any box. The workflow notes the tier-assertions being PRO-specific is "a separate concern," so flagged not heavily scored.
Severity: MED. Effort: M. Parity-safe: yes (test-only).

---

## Considered & rejected (incl. rejected-for-parity)

- **Reorder/parallelize the place-add kernels across all partials for max overlap — REJECTED-FOR-PARITY.** Disjoint shards make the *arithmetic* identical in any add order (no reassociation), and it would pass `memcmp` today. BUT §12 and the parity test pin the *literal* fixed `g=0..G-1` sum, the host baseline (`f2_combine.cpp:111-127`) sums in that order, and the `.cu` comment is explicit the `+=` "keeps this the literal §12 fixed-order sum." Reordering the *adds* (not the copies) diverges from the contract's letter. P6 overlaps only the *copies* and keeps the adds serialized — the parity-safe form. Do not reorder the accumulator adds.

- **Replace the fixed-order combine with NCCL AllReduce / a tree reduce — REJECTED-FOR-PARITY.** AllReduce order varies with G and buffer size and is not bit-identical to a single-GPU sum (arch §12, §11.4 "NEVER an NCCL AllReduce"). The explicit parity line. Not a candidate.

- **`atomicAdd` the partials directly into a shared accumulator — REJECTED-FOR-PARITY (and pointless).** atomicAdd order is the dominant nondeterminism the project eliminates (§12, §11.4). With disjoint shards there is no contention anyway, so it buys nothing and breaks the contract's letter.

- **Drop the per-partial `cudaDeviceSynchronize` without an alternative fence — REJECTED (correctness).** Verified against the doc: device→device copies do "no host-side synchronization," so freeing the peer source races the DMA. *Some* fence is required while the peer-copy form exists. The fix (P3) is a lighter event fence or eliminating the peer buffer (P1), never deleting the sync outright — that would reintroduce the documented intermittent `cudaErrorLaunchFailure`.

- **Vectorize the place-add (double2 loads) — considered, low ROI.** The kernel is bandwidth-bound (one load + add + store per element); a double2 form is a sub-1% win on an off-critical-path combine. Subsumed by P7 (grid-stride) + P8 (fuse). Not a separate finding.

- **Skip the `cudaMemset` and instead `cudaMemcpy` the root's own partial straight into `dAcc` at its offset (no place-add for the root) — REJECTED (parity letter + marginal).** Tempting: the root's blocks land on a fresh 0.0 region, so a direct copy-into-offset would equal `+= 0.0`. But it special-cases g==root out of the uniform fixed-order `+=` loop the §12 contract describes, for a negligible saving (one place-add of the root's slab). Keep the uniform `+=` for contract clarity. (Note: this is *arithmetically* parity-safe but contract-letter-divergent, same class as the add-reorder rejection.)

- **Pin the source `F2BlockTensor` vectors for async H2D — partially rejected for M4.5.** True async H2D needs pinned host memory (P5), but the sources are pageable (produced by `compute_f2_blocks`); pinning them is a producer-side change (P2-adjacent) and the combine is off the critical path, so M4.5-local ROI is low. Recorded under P5 with the ROI caveat.

- **Make `device_ids`/`root_device_id` const-ref spans or by value — rejected as noise.** `std::span` is already a cheap value type. The real fix is C1 (remove under P1/P2), not micro-tuning the passing convention.

- **Hoist the device zero-init via `cudaMemsetAsync` — folded into P9/P10.** Worth doing once streams exist; standalone it is two negligible memsets. Not separate.

- **First-pass claim "the pageable `cudaMemcpy` H2D is silently synchronous" — DOWNGRADED to "partially synchronous" (P5).** The doc says pageable H2D does a leading stream-sync then returns after the host→staging copy, with the DMA possibly still in flight — not a full host stall. The first pass over-stated it; the corrected wording is in P5. The *conclusion* (it cannot overlap and a pinned async copy is the idiom) stands.

---

## What it takes to reach 10/10

Perf (the first-class axis):
1. **Eliminate the host→peer→root double-bounce.** M4.5-local: a single `cudaMemcpy` host→root per partial (P1). Architecturally-correct: keep each device's partial device-resident so `cudaMemcpyPeer` is a genuine peer→root pull (P2, an M5 follow-up touching `compute_f2_blocks` — yes-if-careful).
2. **Replace the per-partial `cudaDeviceSynchronize` with an `Event` fence (or delete it under P1)** so copies and place-adds pipeline (P3).
3. **Pipeline the G pulls** — overlap each partial's H2D/DMA with the previous place-add via a K-deep staging ring + per-pull streams + events, keeping the place-adds serialized on one accumulator stream in fixed g order (P6 + the N4 ring constraint).
4. **Hoist `cudaDeviceEnablePeerAccess` to once-per-pair (ideally into `build_resources`)** and delete the `cudaGetLastError` scrub (P4 + C2).
5. **Make the place-add grid-stride** (removes the release-only silent-under-cover trap, I3/P7) and **fuse the f2 and vpair adds into one launch** (P8), behind a **narrow `void launch_place_add(...)` wrapper** (N1).
6. **Run everything on explicit non-default streams** (P9, with the N4 fences); **pin the staging buffer** once a `PinnedBuffer` wrapper lands (P5); **reuse hoisted peer buffers** if the peer form survives (P11); **`resize` the host result instead of `assign(0.0)`** (P10).

Cleanliness / contract:
7. **Wire the `enable_peer_access` knob** (honor it at the probe/`build_resources`, or skip the enable here when false) — it is currently dead (N2/L3).
8. **Fix the N3 comment** so it matches the CUDA API (cudaMemcpyPeer does no host-side D2D sync; that is *why* the fence is needed).
9. **Single-home the partials/shards validation** shared with `f2_combine.cpp` (C5).
10. **Settle one width per concept** (`long` for kernel-feeding counts/indices, `size_t` only for byte/alloc sizes); remove the `part_elems`→`cnt` re-cast and the dead `n_block_full < 0 ? 0 :` clamps (C3, C4).
11. **Extract the per-partial transport+add step** into a helper to cut the ~200-line function (C6); add a `ScopedDevice` RAII type to single-home the device-restore (I1).
12. **Drop `device_ids`/`root_device_id` from the API** once P1/P2 land (C1).

Testability:
13. **Add a transport-independent unit test of the fixed-order place-add loop** so the combine arithmetic is covered on any box, and make the P2P-transport assertions runnable (or explicitly skipped-with-a-reason) on the no-peer tier (T2).

If items 1–5 land (double-bounce, sync, pipeline, grid-stride, fuse + narrow wrapper), the knob is wired (7), the N3 comment is fixed (8), and the validation is single-homed (9), this is a 9.5–10: a device-resident, overlapped, grid-stride, idiom-clean combine that is byte-identical to the host baseline by construction.

## Good patterns to keep

- **The CUDA-free decl / CUDA def split** (`p2p_combine.hpp` names no CUDA type; the `.cu` is the only CUDA TU) — a model of arch §4, mirroring `backend_factory.hpp`/`cuda_backend.cu` (L1).
- **The parity rationale comments** — the byte-exact-transport + fixed-order-placement-onto-0.0 argument, with architecture citations — exactly the right load-bearing documentation for a §12-critical unit (R1), *minus* the two workaround comments (C2, N3).
- **`DeviceGuard` restore-on-exit via a non-throwing destructor** — correct RAII for `cudaSetDevice` state, exception-safe on every path (I1).
- **All device memory through `DeviceBuffer<double>`** (no raw `cudaMalloc`), the overflow-guarded ctor, and the post-launch `STEPPE_CUDA_CHECK_KERNEL()` after every launch — arch §2/§7 (I2, I4).
- **The named, justified, shared-launch-math block constant** (`kPlaceAddBlockX = 256` via `cdiv`/`kMaxGridX`) — no magic numbers, single-home launch math (I4).
- **Out-of-band capability tagging** (`CombinePath` on `Resources`, never on `F2BlockTensor`) and the probe-once/gate-in-caller/chosen-path-here split — clean capability-tier coherence (T1).
- **Up-front fail-fast `validate_partials`** before any device work, with contextual messages — arch §2 (its only flaw is being duplicated, C5; the pattern is right).
- **Correct, exhaustive edge-case guards** (empty shard, `total==0`, null staging, ordering-protected overflow) — E1/E2.

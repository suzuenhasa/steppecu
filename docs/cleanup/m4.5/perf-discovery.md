# M4.5 Multi-GPU PERF DISCOVERY — the parity-safe speedup plan

> **RESOLVED / OUTCOME (post-M5) — this whole plan is HISTORICAL.** This doc hunts for the multi-GPU
> precompute speedup; the honest finding is that **multi-GPU per se was never the precompute
> speedup.** The data-bounce was fixed (`867a4bf`, the local 1.10× below), but the REAL wins came from
> getting the result OFF the CPU: **device-resident OUTPUT (commit `1f80c0c`)** — the result stays in
> VRAM, host `F2BlockTensor` is opt-in — measured **P=512 ~673 ms resident vs ~2879 ms bulk-to-host =
> ~4.3×** (the precompute was HOST-RESULT-BOUND; ~80% of the old wall was the host copy). Then **M5
> out-of-core** (`176a07d` adaptive tiered output + `c65179f` SNP-tile input streaming) bounded the
> GPU footprint to O(P·tile + P²) and let P=2500 full-autosome complete on a single 32 GB 5090 in
> ~51.5 s, parity bit-identical. Multi-GPU on the precompute is a modest throughput layer at best; its
> real home is the embarrassingly-parallel FIT / ROTATION phase (Phase 2, unbuilt). The "speedup plan"
> below is kept for the build-type / overlap / pinned-staging findings, but treat the multi-GPU
> framing as historical.
>
> ---
>
> **PRIOR STATUS: SUPERSEDED on its central premise (P1) — see `why-multigpu-slow.md`.** This doc's #1
> claim — that the dominant cost is **~1440 ms/run of redundant combine host-zeroing** (`assign(0.0)`),
> and that **P1 (`resize` not `assign`) is THE speedup lever** — was **REFUTED on-box** by a later nsys
> trace. The accumulator zeroing measures **5 ms** (a `cudaMemset`), not 1440 ms; the host
> `assign(0.0)` first-touch is real but is merely **relocated into the D2H** (removing it just moves the
> same page-fault into the copy), so P1 is **not a separate ~1440 ms bucket and not the cure.** P1 was
> reverted / never the fix. The ACTUAL root cause was the **data-bounce wart**: a redundant SECOND full
> 7.14 GB Device→Host copy. The ACTUAL fix is the **device-resident combine (commit `867a4bf`)**, after
> which **multi-GPU is FASTER** (rtxbox, Release, EmuFp64{40}, median of 10: P=768 G2 = 2125 ms vs
> G1 = 2342 ms = **1.10×**; P=400 = **1.22×**), with `memcmp` parity preserved. The "multi-GPU is
> slower (0.70× / 0.72–0.85×)" tables below are the **PRE-FIX** state and are now FALSE. Read
> `docs/cleanup/m4.5/why-multigpu-slow.md` (nsys root cause @165f655) for the corrected analysis; the
> rest of this doc is kept for the build-type / overlap findings that remain valid.

Branch `m4.5-multigpu`. Box `rtxbox` (2×RTX PRO 6000 Blackwell, sm_120, CUDA 13, 96 GB ea,
REAL P2P `can_access_peer=true`). All numbers below are MEASURED on rtxbox unless tagged
"expected". Every fix is tagged against the §12 `memcmp` bit-identity law and the
`00-overview.md` audit backlog (W*/B*/L*/X*).

This document supersedes the agentscripts README's pre-rtxbox hypothesis. The README guessed
"L4 (per-call alloc) + B2 (P2P combine) are the speedup levers, to be measured on rtxbox."
We measured. The verdict below corrects two of those guesses (B2 is NOT a speed lever; the
debug-sync was never the lever either) and adds a net-new #1 cause (W9 redundant host-zeroing)
and a net-new root-cause precondition (F1, the NULL legacy default stream) that no prior
audit item named.

---

## (1) THE HEADLINE

> **[PRE-FIX — now FALSE]** As of `867a4bf` multi-GPU is **FASTER** (P=768 = 1.10×, P=400 = 1.22×).
> The "genuinely slower" finding below was the pre-fix state; it correctly ruled out the debug-build
> artifact, but its attribution to host-zeroing (P1) was wrong — see the banner at the top of this doc
> and `why-multigpu-slow.md`.

**[PRE-FIX] Multi-GPU was genuinely slower than single-GPU on rtxbox — and it was NOT a debug-build
artifact.** The leading prior hypothesis (the per-kernel `cudaDeviceSynchronize` in
`check.cuh:201`, active because the bench built with no `CMAKE_BUILD_TYPE`) is **REFUTED**.

A true Release build (`-DCMAKE_BUILD_TYPE=Release` ⇒ `NDEBUG` ⇒ the debug post-launch
`cudaDeviceSynchronize` compiled out, verified) moved the G1-vs-G2 ratio by essentially
**nothing** — within 1–3 % run-to-run noise at every P:

| P | DEFAULT G1 | DEFAULT G2 | DEFAULT ratio | Release G1 | Release G2 | Release ratio |
|---|---|---|---|---|---|---|
| 200 | 401 ms | 398 ms | 1.01× | 396.9 ms | 380.5 ms | **1.04×** |
| 400 | 947 ms | 1140 ms | 0.83× | 939.4 ms | 1107.5 ms | **0.85×** |
| 600 | 1713 ms | 2232 ms | 0.77× | 1675.5 ms | 2176.1 ms | **0.77×** |
| 768 | 2494 ms | 3467 ms | 0.72× | 2473.8 ms | 3521.3 ms | **0.70×** |
(EmulatedFp64{40}, M=584131, n_block=757, min-of-2. Release: built clean 120/120; parity PASS
both datasets, P2P actually ran `last_combine_path=P2pDeviceResident`.)

**How much of the original finding was the debug per-kernel sync: ~0 % of the GAP.** The debug
sync was symmetric — it slowed G1 and G2 roughly equally (both run kernels), so removing it sped
both equally and left the ratio unchanged. In the DEBUG nsys it dominated CUDA-API time at
**42.1 % / 2.53 s / 186 calls**; in the Release nsys `cudaDeviceSynchronize` collapsed to
**2.8 % / 0.22 s / 15 calls**, and those 15 are the *explicit* P2P combine fence
(`p2p_combine.cu:283`), NOT the debug macro. So the 42 % was real but it inflated BOTH bars; the
slowdown ratio is a different defect entirely. **[PRE-FIX]** Multi-GPU was slower in Release (1.04× → 0.70×
over P=200 → 768), the same monotone-with-P curve as DEFAULT — this is the pre-`867a4bf` state and is
now FALSE (post-fix: P=768 = 1.10×, P=400 = 1.22×).

PARITY: bit-identity holds in BOTH build types (parity test PASS RC=0 in Release on
`derived_acc` P=50 and `derived_full` P=768; P2P device-resident AND host-staged each
`memcmp`-identical to single-GPU). **Bit-identity does NOT depend on build type** — the parity
law is upheld.

---

## (2) THE REAL BOTTLENECK RANKING (Release, P=768, nsys + microbench)

Release nsys run reproduced the finding (G1=2463 ms, G2=3420 ms, 0.72×; report at
rtxbox:/tmp/nsys768.nsys-rep). CUDA-API time share, whole run:

| API | % | Time | Calls | what it is |
|---|---|---|---|---|
| `cudaStreamSynchronize` | 35.3 % | 2.73 s | 93 | per-chunk feeder syncs (`cuda_backend.cu:209/333/386/397`) |
| `cudaMemcpyAsync` | 34.8 % | 2.69 s | 363 | **PAGEABLE** H2D/D2H ⇒ effectively blocking |
| `cudaMemcpy` | 14.1 % | 1.09 s | 18 | synchronous; P2P combine D2H + peer staging |
| `cudaFree` | 7.1 % | 0.55 s | 648 | per-chunk DeviceBuffer churn |
| `cudaMallocAsync` | 3.0 % | 0.23 s | 225 | cuBLAS workspace |
| **`cudaDeviceSynchronize`** | **2.8 %** | 0.22 s | 15 | **the combine fence — was 42 % in DEBUG, the macro is GONE** |
| `cudaMalloc` | 2.6 % | 0.20 s | 645 | per-chunk churn |
| `cudaMemcpyPeer` | ~0.0 % | 112 µs | 6 | the credited P2P DMA — 12 GB at ~36 ms/op, **OFF the critical path** |

GPU kernels: dgemm 59.7 %, emu max_scale_pack 23.6 %, feeder/gather/assemble the rest;
place_add (the combine) only 1.7 %. GPU MemOps: H2D 53.5 % (pageable), D2H 40.7 %
(the 3.57 GB result tensors, ~88 ms each), Peer-to-Peer 5.4 %.

### Device-overlap verdict (the key question): both devices run, but they barely overlap, AND the GPUs are starved

Per clean G2 timed iteration:
- **Fan-out compute window ~488 ms; dev0 kernels 199 ms, dev1 kernels 205 ms — kernel-overlap
  only 36 ms = 18 % concurrency.** Including copies, overlap is 40 %. The `std::jthread` fan-out
  IS present (`f2_blocks_multigpu_core.cpp:109–146`, one worker per device, joined before the
  combine) — the threads ARE spawned concurrently — but their GEMMs serialize instead of
  overlapping.
- **GPU starvation:** over the whole G2 segment the GPUs are kernel-busy only ~210 ms each inside
  a ~2.6 s GPU-activity span. The wall is dominated by host-side stalls, not compute.

### Why G2 was ~1000 ms slower than G1 — [the host-zeroing attribution below is REFUTED]

> **REFUTED — see banner + `why-multigpu-slow.md`.** The real dominant cost is the **redundant SECOND
> full 7.14 GB D2H** (the data-bounce wart), not host-zeroing. The `cudaMemset` accumulator zero is
> **5 ms**; the host `assign(0.0)` first-touch below is real CPU work but it is **relocated into the D2H
> write** when removed, so it is not a separable ~1440 ms bucket. The microbench timed the page-faults
> in isolation; under the real run that cost overlaps/relocates into the copy. The text below is kept as
> the (incorrect) original hypothesis.

A `[2·P²·n_block]` host result tensor is **3.57 GB**; zeroing both `out.f2.assign(total,0.0)` +
`out.vpair.assign(total,0.0)` costs a **measured ~1440 ms of pure host CPU** (microbenched on
the box; GPU idle throughout, verified 0 ms GPU-busy in the stall).

- **G1 pays this full-tensor zeroing ONCE** (`cuda_backend.cu:236-237`). nsys: exactly one
  ~1550 ms GPU-idle host stall per run.
- **G2 pays it TWICE:** (1) the two workers each zero their compact partials (concurrent
  ~795 ms), THEN (2) the combine does its OWN full 3.57 GB `out.f2/vpair.assign(total,0.0)`
  (`p2p_combine.cu:180-181`) — a SECOND ~1440 ms GPU-idle host stall, strictly serial after the
  join, overlapping nothing.

That redundant ~1440 ms full-tensor host-zeroing in the combine, serialized after the fan-out
join, is the dominant cause of 0.72×. It dwarfs everything: the fan-out window is only ~488 ms
and the P2P DMA is ~36 ms.

### Ranked causes (by measured impact on the G1↔G2 gap)

> **CORRECTION — the real #1 cause is the data-bounce (second full D2H), fixed @867a4bf.** The ranking
> below put combine host-zeroing first; that was refuted on-box (see banner). The true ranking is:
> #1 the redundant SECOND full 7.14 GB D2H (the data-bounce wart) — eliminated by the device-resident
> combine; host-zeroing is a 5 ms `cudaMemset` plus a page-fault that relocates into the D2H. Item 2's
> overlap finding (18% kernel concurrency) was ALSO later re-measured at **74%** — see
> `why-multigpu-slow.md`; the L4 allocator lock is at most a few percent, not the serializer.

1. **[REFUTED as #1] Redundant combine host-zeroing — was claimed ~1440 ms/run.** Real cost is a 5 ms
   `cudaMemset`; the host `assign(total,0.0)` page-fault relocates into the D2H, so it is not a separable
   lever. Superseded by the data-bounce diagnosis. (audit **W9**, X4/§12 note.)
2. **Fan-out cannot overlap (18 % kernel concurrency).** Compounds from three layers:
   - **F1 (root cause):** the per-device backend runs on the **NULL legacy default stream**
     (`cuda_backend.cu:596 stream_ = nullptr`) and the build is NOT `--default-stream
     per-thread`, so the two workers' streams implicitly serialize against each other — the
     "per-device default streams overlap" claim in the `cuda_backend.cu:80-83` comment is FALSE
     for this compile mode. (net-new finding, precondition under B1/W1.)
   - **L4:** 645 `cudaMalloc` + 648 `cudaFree` of per-chunk `DeviceBuffer`s take the global
     driver lock and device-synchronize, so the two host threads serialize on the allocator.
   - **L2/W9 pinned:** all H2D/D2H is PAGEABLE, so `cudaMemcpyAsync` is effectively blocking
     (drives the 35.3 % `cudaStreamSynchronize`), and copies cannot overlap compute or each other.
3. **G2 moves ~2× the bytes (33.3 GB vs 16.7 GB at P=768).** The fan-out *inputs* are correctly
   partitioned (zero-copy column sub-views, `f2_blocks_multigpu_core.cpp:123-127` — NOT
   duplicated); the extra ~13.3 GB of blocking pageable host PCIe is the combine's
   D→H→D→H round-trip (re-upload of partials + D2H of the accumulator). (audit **X3/W2/L1**.)
4. **Per-partial `cudaDeviceSynchronize` in the combine (W3) — 2.8 %, small at G=2.** Off the
   critical path; matters only at G>2.

**NOT levers (confirmed/extended):** the DEBUG per-kernel `cudaDeviceSynchronize`
(42 % → 2.8 % in Release, fully compiled out, those 2.8 % are the explicit fence not the macro);
`cudaMemcpyPeer` (~0.0 % / 36 ms-per-op, exactly as architecture.md §11.4 predicts). **B2 (the
P2P rework) is NOT the speedup lever.**

---

## (3) PRIORITIZED, PARITY-SAFE FIX PLAN

> **OUTCOME (post-fix):** The plan below was the pre-fix proposal. In reality **P1 was REFUTED** (not a
> lever) and the cure was **P6 in disguise** — the **device-resident combine (commit `867a4bf`)**, which
> made multi-GPU **1.10× faster @ P=768**. P0/P2/P3 landed as preconditions (`970fa42` / `9fdc946` /
> `a41d67a`). Treat P1 below as historical; do not action it as a speedup. The L4 allocator lock was
> later re-measured as at most a few percent (overlap is 74%, not 18% — see `why-multigpu-slow.md`), so
> P3 is optional cleanup, not a blocker.

Fix order is chosen so each lever's payoff is *visible*: P1 removes the serial tail the fan-out's
`max` is added to; P2 gives the workers a stream that CAN overlap; P3 stops the alloc-lock
serialization; P4 makes the blocking copies actually async. P1 alone closes most of the gap; P2+P3+P4
together recover the ~2× fan-out. **[The P1 premise was refuted; the real cure was P6 / the
device-resident combine `867a4bf` — see the OUTCOME note above.]**

| # | Fix | Where | Measured / expected impact on the G1↔G2 gap | Effort | Audit map | Parity |
|---|---|---|---|---|---|---|
| **P1 — REFUTED** | **`resize(total)` not `assign(total,0.0)`** for the host result of the P2P combine AND the per-shard partial. | `p2p_combine.cu:180-181`; `cuda_backend.cu:236-237` | **REFUTED — not a lever.** The "~1440 ms/run" was wrong: the accumulator zero is a 5 ms `cudaMemset`; the `assign(0.0)` page-fault relocates into the D2H. The real cure was the device-resident combine (`867a4bf`, see P6). The swap is at most cosmetic. | **S** | **W9** | **PARITY-SAFE** (but no perf payoff) |
| **P2** | **Give each worker a real per-device `Stream`** — replace `stream_ = nullptr` with an owning RAII `Stream stream_{}` (the wrapper in `stream.hpp` already exists, never instantiated), constructed after `device_id_` is set; pass `stream_.get()` everywhere. | `cuda_backend.cu:596` (+ all launch/copy/`set_stream` sites) | **Precondition** — alone moves the wall little, but without it P3/P4 cannot yield overlap. With P3+P4 it turns the ~488 ms fan-out window toward ~205–250 ms (max not sum). | **S** | new (precondition under **B1/W1**); subsumes part of L4 | **PARITY-SAFE** (stream is pure scheduling; cuBLAS workspace re-applied via `set_stream`) |
| **P3** | **Stop the per-chunk alloc churn (L4):** pre-allocate the per-bucket slab buffers (`dIds/dQg/dVg/dSg/dGg/dVpairg/dRg`) ONCE at max bucket width and reuse across chunks (L4b, simpler, higher ROI on 96 GB), OR a per-device `cudaMallocAsync` pool with `cudaMemPoolAttrReleaseThreshold=MAX` (L4a). | `cuda_backend.cu:355-388`, `:306-309`, `:320`; `device_buffer.cuh:74,117` | **Expected ~150–240 ms/run** — lifts fan-out kernel overlap from 18 % toward 50–80 %. Bigger on the 32 GB budget tier (many chunks) than on 96 GB rtxbox (few chunks). | **S** (L4b) / **M** (L4a) | **L4** | **PARITY-SAFE** (allocation moves no bits; every buffer fully written before read; pool mem not zeroed but nothing relies on it) |
| **P4** | **Pinned staging + async H2D/D2H (L2/W9):** add a `PinnedBuffer` RAII (`cudaHostAlloc`/`cudaFreeHost`; none exists in `src/` today) and stage Q/V/N and the partial/result transfers through pinned host memory. | `cuda_backend.cu` H2D/D2H sites; `p2p_combine.cu` staging; `device_buffer.cuh` (new sibling) | **Expected: unblocks the ~44 % pageable copy serialization.** Makes `cudaMemcpyAsync` truly async so device A's H2D overlaps device B's compute. Needs P2 first; little payoff alone. | **M** | **L2 / W9** | **PARITY-SAFE** (pinned vs pageable moves identical bytes) |
| **P5** | **Build/perf config:** make the bench + perf builds Release (`-DCMAKE_BUILD_TYPE=Release`). | CMake invocation | **~0 on the gap** (symmetric) but ~2× absolute on BOTH G1 and G2 by removing the debug per-kernel sync. Already validated. | **S** | build-config (not numbered) | **PARITY-SAFE** (bit-identity proven in Release) |
| **P6 — LANDED @867a4bf (THE actual cure)** | **Device-resident partial return + combine (L1 + W3):** the per-device compute leaves `dF2_all/dVpair_all` RESIDENT (returns an opaque move-only `DevicePartial`); the combine `cudaMemcpyPeer`s each peer slab straight into its disjoint slice, D2D-copies the root partial, then does ONE final D2H. Deletes the re-upload H2D, the accumulator `cudaMemset`, the place-add kernel, the staging buffers, and the per-peer `cudaDeviceSynchronize` (one fence before the final D2H). | `cuda_backend.cu` + seam; `p2p_combine.cu`, staging | **THE WIN.** Removes the data-bounce wart (the redundant SECOND full 7.14 GB D2H). **Measured: P=768 G2 = 2125 ms vs G1 = 2342 ms = 1.10×; P=400 = 1.22×** (rtxbox, Release, EmuFp64{40}, median of 10), bit-identical parity preserved. This — not P1 — was the cure. | **L** | **L1 / W3 / B2(W2)** | **SAFE — proven** (resident-to-resident DMA byte-exact; disjoint placement, fixed g order; memcmp parity passed) |

**Parity argument for P1 (proposed as the #1 lever — later REFUTED; parity-safe but no perf payoff), in full:** the host `out.f2`/`out.vpair` are pure D2H
landing buffers. The device accumulator is `cudaMemset(0)`'d and every element is written by the
fixed-order place-adds, then copied back wholesale (`total > 0` branch overwrites all `total`
elements unconditionally). `std::vector<double>::resize(n)` value-initializes to +0.0 anyway, so
even the bit pattern is identical — the win is eliminating the *redundant second pass*, not the
zero. The §12 `memcmp` gate compares the final host tensor, which is fully overwritten; the
GEMM bits and the fixed g=0..G-1 combine order are untouched. **PARITY-SAFE.**

**Rejected-for-parity (do not let these slip in under "optimize the combine"):**
- **NCCL / tree-reduce / any non-`g=0..G-1` fan-in** for the combine — reduction order varies
  with G, not bit-identical (§12). The explicit parity line.
- **`atomicAdd` of partials into a shared device accumulator** — atomicAdd order is the dominant
  nondeterminism §12 eliminates.
- **Parallelizing/reordering the place-adds across streams** — only the COPIES may overlap; the
  place-adds MUST stay serialized on one accumulator stream in fixed g order (§12 sum order).
- **`std::transform`/parallel reduce for the host accumulate** — re-introduces the `+=` onto +0.0
  and the −0.0 flip the current `std::copy_n` avoids.
- **Tolerance for the EmulatedFp64 parity asserts** — §12 is bit-IDENTITY (`memcmp`), not tolerance.
- **`resize` for the HOST-STAGED tier `f2_combine.cpp:64-65` — yes-if-careful only:** that tier uses
  `std::copy_n` placement and the +0.0 init is the defensive floor for a genuinely-unowned slab.
  On the real disjoint-tiling path every slab is owned so `resize` is bit-identical, but it changes
  behavior on a degenerate unowned-tail input the contract proves cannot occur. Guard by the
  tiling invariant, or leave that one site as-is. (The P2P tier and the backend partial are
  unconditionally safe.)

---

## (4) IS B2 (THE P2P COMBINE REWORK) STILL WORTH DOING?

**For SPEED on the 2-GPU case: NO.** B2 is NOT a speedup lever and the measurements prove it
three ways: (a) `cudaMemcpyPeer` is ~0.0 % of CUDA-API time (112 µs / 36 ms-per-op), exactly off
the critical path as architecture.md §11.4 predicts; (b) the combine's place-add kernel is only
1.7 % of GPU kernel time; (c) the combine's actual cost is the host-side `assign(0.0)` (W9/P1) and
the host PCIe re-upload (X3) — neither is the `cudaMemcpyPeer` transport B2 reworks. At G=2 the
B2 event-fence + ring (W3) replaces a 2.8 % device-sync that only fires once per run.

**For CLEANLINESS and FUTURE G>2: YES, but de-prioritized.** B2 makes the named §7 perf exemplar
idiomatic (W4 grid-stride kernel, W5 fused launch, W6 hoisted peer-enable, W3 event-fence + ring).
The X3 double-bounce elimination IS a real ~6.65 GB host-PCIe win at P=768 — but its M5/device-
resident form (P6/L1) is the part with the byte-count payoff, not the `cudaMemcpyPeer` rework
itself. The W3 event-fence + ring matters for G>2 (more peer pulls to pipeline), which no current
box can test. So: **B2 is a cleanliness + G>2-readiness item, NOT a 2-GPU speed item.**

> **UPDATE (post-`867a4bf`):** P6 (the device-resident combine) has LANDED and already delivered the
> X3 / data-bounce win (the redundant second full D2H is gone; the per-peer device-sync collapsed to one
> fence). B2's original motivation — the slow / bouncing combine — is therefore largely **subsumed**.
> Mark B2 as **REASSESS / likely-subsumed**, NOT a pending speedup. (`agentscripts/m4.5-b2-p2p-fix-pass.js`.)

---

## (5) RECOMMENDED NEXT WORKFLOW + BUILD-TYPE DEFAULT

**Set Release as the bench/perf default: YES, do it now.** Make `-DCMAKE_BUILD_TYPE=Release`
(⇒ NDEBUG ⇒ the `check.cuh:201` debug sync compiled out) the default for the bench and any perf
build. It is a ~2× absolute win on both bars, costs nothing, and bit-identity is proven in
Release (parity does NOT depend on build type). Keep a DEBUG build for the parity/dev loop. Note:
a Release build surfaces three benign `-Werror` warnings (unused-parameter in
`launch_config.hpp:114`; unused `child_aborts` in two death-test TUs) because the consuming
asserts/death-tests vanish under NDEBUG — demote those two warning kinds via cache flags
(`-Wno-error=unused-parameter -Wno-error=unused-function` + nvcc `-diag-suppress 177`); do NOT
edit source. Their appearance is itself confirmation NDEBUG took effect.

> **HISTORICAL — this fix-pass plan is now complete, with a different outcome than proposed.** The
> `m4.5-perf-fix-pass.js` ran: P0/P2/P3 landed as preconditions (`970fa42` / `9fdc946` / `a41d67a`);
> **P1 was REFUTED** (a sham/no-op fixer report, and not a lever anyway). The actual cure was **P6 — the
> device-resident combine, `867a4bf`** — which flipped G2 from 0.70× to **1.10× (faster) @ P=768** and
> **1.22× @ P=400**, bit-identical parity preserved. The original ROI-ordered plan below is kept for the
> record; do not re-action P1.

**Author a fix-pass workflow for the top levers (the proven pattern: STRICTLY SEQUENTIAL, 2
agents per item — independent fixer + adversarial verdict; template `fix-pass-phase2.js`).**
Suggested name `m4.5-perf-fix-pass.js`, run on rtxbox (P1/P6 touch the P2P path; only rtxbox has
real P2P). Item order (as originally proposed — see the HISTORICAL note above for what actually happened):
- **P1 — W9 `resize` not `assign(0.0)`** — **[REFUTED, not a lever, do not action]**. Gate:
  re-run `bench_f2_multigpu` G2 vs G1 + the locked `memcmp` parity test.
- **P2 — real per-worker `Stream`** (S, the overlap precondition) — **landed `9fdc946`**.
- **P3 — L4 buffer reuse / pool** (S–M) — **landed `a41d67a`** (optional cleanup, not the lever).
- **P4 — pinned staging** (M, needs the new `PinnedBuffer`) — *was the remaining optional lever* (pin
  the final pageable result D2H to push past 1.10×). **SUPERSEDED post-M5:** device-resident OUTPUT
  (`1f80c0c`) removes that final D2H entirely for the in-VRAM case (~4.3× @ P=512), and M5 streaming
  (`176a07d`/`c65179f`) handles the beyond-VRAM spill in overlapped block-tiles; an M5 pinned
  double-buffer is the spill mechanism, not a multi-GPU lever.
- **P6 — device-resident combine — the local cure that flipped G2 to 1.10× @ P=768 (`867a4bf`);
  later superseded by device-resident OUTPUT (`1f80c0c`) + M5 streaming as the architectural fix.**

**Recommended measurement:** the post-fix bench (rtxbox, Release, EmuFp64{40}, median of 10) recorded
P=768 G2 = 2125 ms vs G1 = 2342 ms = 1.10×, P=400 = 1.22×, via `tests/reference/bench_f2_multigpu.cu`
(now ITERS=10, median + p10/p90, out-of-band timers).

Every item above is re-gated by the locked §12 `memcmp` (`test_f2_multigpu_parity`) on both
datasets; commit-green / revert per the fix-pass convention. None of P1–P6 perturbs a single
arithmetic byte — bit-identity does not depend on build type, stream choice, allocator strategy,
pinning, or transport.

---

## KEY FILES (all absolute)

- `/home/suzunik/steppe/src/device/cuda/p2p_combine.cu:180-181` — the full-tensor
  `assign(total,0.0)` (was claimed the ~1440 ms #1 bottleneck under P1/W9; **REFUTED** — the zero is a
  5 ms `cudaMemset`); `:283` per-partial `cudaDeviceSynchronize` (collapsed to one fence by P6/W3 @867a4bf);
  `:215-218,254-257` partial re-upload bounce (X3 — **deleted** by the device-resident combine @867a4bf);
  `:309-312` full D2H (now the SINGLE D2H, no second bounce).
- `/home/suzunik/steppe/src/device/cuda/cuda_backend.cu:236-237` — per-shard `assign(0.0)`
  (P1/W9); `:596 stream_ = nullptr` (the NULL legacy default stream, P2/F1); `:80-83` the FALSE
  "per-device default streams overlap" comment; `:107` cuBLAS `set_stream`;
  `:209,333,386,397` per-chunk `cudaStreamSynchronize`; `:355-388,306-309,320` per-chunk + resident
  alloc churn (P3/L4); `:311-397` pageable copies (P4/L2).
- `/home/suzunik/steppe/src/device/cuda/check.cuh:197-202` — the now-confirmed-irrelevant debug
  per-kernel `cudaDeviceSynchronize` (P5); `/home/suzunik/steppe/src/core/internal/host_device.hpp:63-67`
  — the NDEBUG gate.
- `/home/suzunik/steppe/src/core/fstats/f2_blocks_multigpu_core.cpp:109-146` — the `std::jthread`
  fan-out (present; overlaps only 18 %); `:123-127` the zero-copy column sub-views (inputs NOT
  duplicated).
- `/home/suzunik/steppe/src/device/cuda/device_buffer.cuh:74,117` — the only raw
  `cudaMalloc`/`cudaFree` (the L4 fix lands here; no `PinnedBuffer` sibling exists yet — P4).
- `/home/suzunik/steppe/src/device/cuda/stream.hpp` — the RAII `Stream`/`Event` (exist, unused by
  the backend/combine; the P2 and P6 fix primitives).
- `/home/suzunik/steppe/src/device/vram_budget.hpp:140` — `max_blocks_per_chunk` (few chunks on
  96 GB / many on 32 GB; sizing source for P3/L4b).
- `/home/suzunik/steppe/src/core/fstats/f2_combine.cpp:64-65,103-104` — host-staged zeroing
  (P1 yes-if-careful: the −0.0/unowned-slab caveat).
- `/home/suzunik/steppe/docs/cleanup/m4.5/00-overview.md` — the audit backlog (W1/W3/W9, B1/B2,
  L1/L2/L4, X2/X3/X4); this doc tags every claim to it.
- On the box: `/tmp/nsys768.nsys-rep` (+ `.sqlite`), `/tmp/relbench.log`, `/tmp/relparity.log`,
  `/tmp/relbuild3.log`; build tree `/workspace/steppe/build-rel` (the default `build` untouched).

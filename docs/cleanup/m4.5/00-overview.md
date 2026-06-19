# M4.5 Multi-GPU — WHOLE-MILESTONE HOLISTIC REVIEW

> **STATUS UPDATE (post-M5).** This review was written at the close of M4.5 with multi-GPU
> framed as "the headline speedup." That framing has since been **corrected by the measured
> record**: on the *precompute*, multi-GPU is a **modest throughput layer**, NOT the headline.
> The real wins came AFTER M4.5, from getting the result **off the CPU** and **streaming** it:
> **(1) device-resident output** (`1f80c0c`) — the precompute returns a VRAM-resident handle
> (`DeviceF2Blocks`), host `F2BlockTensor` is an opt-in `.to_host()`; **measured P=512 ~673 ms
> resident vs ~2879 ms bulk-to-host = ~4.3×** (the precompute was HOST-RESULT-BOUND — ~80% of
> the old wall was the host copy). **(2) M5 out-of-core streaming** — **adaptive tiered output**
> (`176a07d`, result goes to the fastest tier it FITS: VRAM -> host RAM -> disk) + **SNP-tile
> input streaming** (`c65179f`, GPU footprint O(P·tile + P²), independent of M). **M5 is DONE:**
> full-autosome (M=584131, n_block=757) **P=2500 COMPLETES on a single 32 GB RTX 5090 in
> ~51.5 s** (76 GB result streamed, GPU peak ~26 GB), parity `memcmp` bit-identical. The
> per-combine 1.10×/1.22× figures below are real M4.5-internal measurements and stand; the
> *interpretation* changes — multi-GPU's proper home is the **Phase-2 FIT/ROTATION** (thousands
> of independent qpAdm models, no combine — embarrassingly parallel), not the precompute. The
> real next work is **Phase-2, the qpAdm FIT ENGINE** (S3-S8, still unbuilt). See
> `architecture-audit.md`, `parallelism-check.md`, `why-d2h.md`, `why-multigpu-slow.md`.

Scope: the entire single-node multi-GPU (SPMG) addition on branch `m4.5-multigpu` —
the shard plan, per-device compute fan-out, the two combine tiers (host-staged baseline
+ P2P device-resident fast-path), the DI bundle, the capability probe, the config knob,
and the parity gate. Judged against `docs/architecture.md` §2/§4/§7/§9/§11.4/§12/§13 at a
demanding-senior 9.5–10 bar.

**The parity GATE is met and is not in question.** `tests/reference/test_f2_multigpu_parity.cu`
proves the production `EmulatedFp64{40}` path is `memcmp`-identical to single-GPU across G
for BOTH the host-staged fixed-order combine and the `cudaMemcpyPeer` P2P device-combine, on
real AADR (`derived_acc` P=50, `derived_full` P=768). This review's mandate is to make the
code better-optimized and architecturally cleaner **without** regressing that bit-identity.
Everything proposed below is tagged PARITY-SAFE / yes-if-careful / rejected-for-parity, and
every parity-critical change is gated by re-running that locked `memcmp`.

This document holds the CROSS-FILE findings the eleven per-unit reviews cannot individually
see — the end-to-end data flow, the DRY duplications and contract drift that span files, and
a deduped, prioritized master backlog. The per-unit reviews (read in full and folded in
below) remain authoritative for their own internal findings.

Per-unit final scores folded in:

| unit | score | perf-findings |
|---|---|---|
| `device/cuda/p2p_combine.{cu,hpp}` | 8.5/10 | 11 (the headline data-bounce/sync set RESOLVED @867a4bf) |
| `core/fstats/f2_blocks_multigpu` | 8.0/10 | 7 |
| `device/shard_plan` | 9.0/10 | 4 |
| `device/resources` | 8.5/10 | 4 |
| `core/fstats/f2_combine` | 9.0/10 | 5 |
| `tests/.../test_f2_multigpu_parity` | 8.5/10 | 5 |
| `device/cuda/cuda_backend` (M4.5 delta) | 9.0/10 | 6 |
| `device/cuda/handles` (M4.5 delta) | 8.5/10 | 4 |
| `device/cuda/check` (M4.5 delta) | 8.0/10 | 1 |
| `device/backend.hpp` (M4.5 delta) | 9.5/10 | 2 |
| `include/steppe/config.hpp` (M4.5 delta) | 9.5/10 | 2 |

---

## OVERALL M4.5 SCORE: 8.7 / 10

The milestone is **correct, parity-locked, faster than single-GPU, and exemplary on layering** —
the CUDA-free seam (`backend.hpp` POD probe, the `p2p_combine.hpp` / `shard_plan.hpp` CUDA-free
decls, `resources` reaching the GPU only through `make_cuda_backend`+`capabilities()`) is the
cleanest realization of arch §4 in the codebase, and the out-of-band `CombinePath` tagging
discipline (§12 / cleanup §(2).2) is held without exception. The two highest-rated units
(`backend.hpp`, `config.hpp`, both 9.5) earn it.

**The within-M4.5 combine slowdown was fixed (but multi-GPU is a modest layer, not the headline).**
As of the device-resident combine fix (`867a4bf`), the 2-GPU path is **faster** than single-GPU on
the rtxbox (2× RTX PRO 6000 Blackwell sm_120, Release, EmuFp64{40}, median of 10): P=768
G1=2342ms vs G2=2125ms = **1.10×**; P=400 = **1.22×**. This *replaces* the prior "multi-GPU is
slower (0.70× / 0.75–0.97×)" finding — that was the pre-fix state and is now FALSE. **Post-M5
correction (per `architecture-audit.md` / `why-d2h.md` / `parallelism-check.md`):** this ~1.1× is a
MODEST throughput layer, NOT the headline — the precompute was HOST-RESULT-BOUND, and the real
speedups came after M4.5 from **device-resident output** (`1f80c0c`, P=512 ~4.3×) and **M5
streaming** (`176a07d`/`c65179f`), i.e. getting OFF the CPU, not multi-GPU per se (nsys measured only
~22–74% overlap on the precompute combine). Multi-GPU's proper home is the Phase-2 FIT/ROTATION
(thousands of independent models, no combine). The nsys root-cause diagnosis (`165f655`,
`docs/cleanup/m4.5/why-multigpu-slow.md`) pinned the slowdown on a redundant SECOND full ~7.14 GB
Device→Host copy (the DATA-BOUNCE wart): `compute_f2_blocks` D2H-copied each partial to host and
freed its device buffers, forcing the P2P combine to re-upload (H2D) + place-add + a 2nd D2H. The
device-resident combine deletes that bounce — per-device compute leaves its partial RESIDENT (no
D2H / no free, returns a move-only `DevicePartial` handle), the combine allocates ONE root result,
D2D-copies the root partial + `cudaMemcpyPeer`s each peer partial straight into its disjoint block
slice, then does ONE final D2H.

It is held to **8.7** by a smaller cluster of issues that only the whole-milestone view makes
visible:

1. **A small constellation of DRY duplications and host-test gaps** — `validate_partials`
   duplicated byte-for-byte across the two combine tiers, the §11.4 gate predicate restated in
   5–6 homes, the degrade WARN compiled out under NDEBUG, and the host-pure planner / orchestrator
   / combine all exercised only through one GPU `.cu`.

2. **A dead config knob + a documented-but-false gate.** `enable_peer_access` is read by NO
   code (confirmed by grep — only doc comments) yet the combine path unconditionally calls the
   very `cudaDeviceEnablePeerAccess` it is documented to gate; and the P2P gate is documented as
   3-term (`… && G >= 2`) in FIVE files while the shipped gate is 2-term. These are contract lies
   on the seam the whole tier keys off.

**Remaining OPTIONAL levers (NOT blockers):** the "pin the final pageable result D2H" lever is now
largely moot — **device-resident output (`1f80c0c`) removed the eager whole-tensor D2H** for the
in-VRAM case (the result stays a `DeviceF2Blocks` handle; host copy is an opt-in `.to_host()`),
which is where the ~4.3× came from; the only remaining D2H is the streamed block-tile spill of the
beyond-VRAM case (`176a07d`). The bench byte-traffic columns are observability-only
(currently print `0.00`). The L4 pool-allocator and the host-zeroing premise of the old
`perf-discovery.md` P1 plan are NOT the speedup lever (the nsys trace measured **74% GPU overlap**,
not the ~18% the pre-fix hypothesis assumed; the accumulator memset is ~5ms, P1 was reverted) —
demote to optional cleanup. The B2 P2P transport rework (`agentscripts/m4.5-b2-p2p-fix-pass.js`)
is **largely subsumed** by the resident combine — its original motivation (the slow/bouncing
combine) is now addressed; mark it REASSESS, not a pending speedup.

**Concrete gap to 9.5+:** land the remaining BEFORE-M5 backlog below. The decisive items are now
single-homing `validate_partials`, wiring/fixing the `enable_peer_access`+`G>=2` gate, and the
GPU-free host tests for the planner/combine/orchestrator. With those, no unit sits below 9 and the
cross-file story is coherent — a 9.5+ milestone. The perf rabbit-hole is CLOSED — and the subsequent
record proved the right lever was never multi-GPU: it was device-resident output (`1f80c0c`) + M5
streaming (`176a07d`/`c65179f`), now both DONE (full-autosome P=2500 on a single 32 GB 5090 in
~51.5 s).

---

## (1) CROSS-FILE ISSUES (no single-unit review sees these)

### END-TO-END DATA FLOW: shard → per-device compute → combine → result

The pipeline (confirmed by reading the source, not just the reviews):

```
compute_f2_blocks_multigpu  (core, f2_blocks_multigpu.cpp)
  ├─ block_ranges(partition)                              [host, O(M)]
  ├─ block_sizes[b] = ranges[b].size()                    [host, O(n_block) — REDUNDANT, see X1]
  ├─ plan_block_shards(block_sizes, ranges, G)            [host, returns G DeviceShards]
  ├─ for g in 0..G-1:
  │     build Qg/Vg/Ng zero-copy sub-views               [host, pointer offset — good]
  │     build block_id_local (per-device subtract)        [host alloc, O(M_local)]
  │     partials[g] = gpus[g].backend->compute_f2_blocks(...)
  │         └─ CudaBackend: cudaSetDevice → H2D upload Q/V/N (pageable) → feeder
  │            → grouped strided-batched GEMM
  └─ combine (the §4 fork):
        host-staged:  combine_f2_partials_host(partials, shards, P, n_block)
          └─ zero-init full [P²·n_block] f2+vpair, then += each compact partial   [host]
        P2P device-resident (867a4bf):  combine_f2_partials_p2p(...)
          └─ each per-device compute leaves its partial RESIDENT (move-only DevicePartial)
             alloc ONE root result [P²·n_block]
             for g: D2D-copy root partial / cudaMemcpyPeer each peer partial straight
                    into its DISJOINT block slice  ◄── no H2D re-upload, no place-add
             ONE fence, then ONE final D2H → HOST F2BlockTensor
```

**X1 [MED, PARITY-SAFE: yes] — `block_sizes` is a redundant host materialization derived from
`ranges`, forcing a parallel-array contract and a `long→int` narrowing in the caller.**
Files: `f2_blocks_multigpu.cpp:104–108` (builds it), `shard_plan.{hpp,cpp}` (consumes it).
The planner is handed BOTH `ranges` and a `block_sizes` that is literally `ranges[b].size()`
narrowed to `int` (confirmed: `f2_blocks_multigpu.cpp:106-107`). `plan_block_shards` then
re-validates `block_sizes.size() == ranges.size()` — a check that can only fire on the
orchestrator's own three-line bug. **Drop the `block_sizes` parameter; derive sizes from
`ranges[b].size()` (already `long`) inside the planner.** One move collapses: the caller's
redundant vector + loop + `long→int` narrowing (removing a latent truncation if a block ever
exceeded INT_MAX columns), the planner's unchecked parallel-array contract (shard_plan C-1),
its sign hole (E-1), and half its cast scatter (T-1). The plan is byte-identical. This is the
single best cleanliness move in the milestone and it spans two units, so neither single review
fully owns it. (shard_plan D-1; f2_blocks_multigpu P3.)

**X2 [RESOLVED @867a4bf — was: G devices run STRICTLY SEQUENTIALLY, speedup unrealized].** The
old finding held that the blocking per-device host loop forced wall-clock `Σ_g time(g)` and the
milestone shipped a parity *scaffold* with no speedup. The nsys root-cause diagnosis (`165f655`,
`why-multigpu-slow.md`) **refuted** that as the dominant cost: the trace measured **74% GPU
overlap**, and the real slowdown was the DATA-BOUNCE wart (X3), not the host-loop serialization.
With the device-resident combine (`867a4bf`) the 2-GPU path is now **faster** than single-GPU
(P=768 = 1.10×, P=400 = 1.22×, rtxbox, Release, EmuFp64{40}, median of 10). **REMAINING OPTIONAL
(not a blocker):** fanning the G `compute_f2_blocks` calls out across G host threads (one per
device, write own `partials[g]`, join before the combine, `exception_ptr` rethrow; would need
`Threads::Threads` on `steppe_core`) is a further overlap lever, parity-safe, but it is no longer
the headline gap. (f2_blocks_multigpu P1.)

**X3 [RESOLVED @867a4bf — was: P2P combine host→peer→root DOUBLE-BOUNCE].** This was the diagnosed
ROOT CAUSE of the pre-fix slowdown (`why-multigpu-slow.md`). The old return contract had
`compute_f2_blocks` D2H-copy each partial to a HOST tensor and free its device buffers, so the P2P
combine had to re-upload (H2D to peer) + `cudaMemcpyPeer` peer→root — a redundant SECOND full
~7.14 GB Device→Host copy plus a re-upload, when the bytes were freshly on-device. **Fixed by the
device-resident combine (`867a4bf`):** per-device compute now leaves its partial RESIDENT (no D2H,
no free, returns a move-only `DevicePartial` handle); the combine allocates ONE root result,
D2D-copies the root partial, and `cudaMemcpyPeer`s each peer partial straight into its DISJOINT
block slice — a genuine peer→root pull with NO host bounce. Deleted: the re-upload H2D, the
accumulator `cudaMemset`, the `place_add_f2_kernel`, the staging buffers. The host-staged tier's
D2H gather remains inherent to the portable no-peer baseline (arch §12 sanctions it as the design).
(p2p_combine P1/P2; f2_blocks_multigpu P7.)

**X4 [RESOLVED @867a4bf — was: pipeline sequential where streams/events would overlap].** The old
P2P combine fenced with a full `cudaDeviceSynchronize()` after EVERY peer pull (needed because the
peer source buffer was freed at iteration end and raced the cross-device DMA) and ran everything on
the NULL stream. The device-resident combine (`867a4bf`) removes both: the partials stay resident
(no freed-buffer race), the per-peer `cudaDeviceSynchronize` is gone, and there is now **one fence
before the final D2H** instead of a per-partial drain. The combine still has a true serial
dependency ONLY on the accumulator add ORDER (§12) — preserved because each peer partial lands in
its own disjoint block slice. **REMAINING OPTIONAL (not a blocker), now largely moot post-M5:** the
eager whole-tensor final D2H this once pointed at is **gone** — device-resident output (`1f80c0c`)
returns the result as a VRAM-resident `DeviceF2Blocks` handle (host copy opt-in), and the
beyond-VRAM case streams block-tiles (`176a07d`); only the host-staged baseline / spill path keeps a
D2H worth pinning. (p2p_combine P3/P6/P9/N4; f2_blocks_multigpu P7.)

**Where data bounces H↔D (the inventory, post-867a4bf):**
- P2P peer partials: **RESOLVED** — partials stay device-resident, `cudaMemcpyPeer` is a genuine
  peer→root pull, no H2D re-upload (X3). The redundant 2nd ~7.14 GB D2H is deleted.
- Host-staged D2H gather: inherent to the portable baseline (X3 note). **Not removable.**
- `resolve_device_order` builds a THROWAWAY device-0 backend (cuBLAS create + 64 MiB workspace
  alloc + full peer-scan probe) just to read `device_count`, then probes device 0 AGAIN as
  `gpus[0]` (`resources.cpp:52-53,92`). **Removable** with a CUDA-free `visible_device_count()`
  factory query (resources P1/P5). Not a statistic bounce, but gratuitous cold-start waste +
  a leaked `cudaSetDevice(0)` ambient side effect (resources E5).
- The combine zero-init then full overwrite (`f2_combine.cpp` `assign(total,0.0)` then `+=`):
  a redundant `2·P²·n_block` write, real only at the §0 220 GB top end; off the critical path
  per §11.4. **Deliberate** (the += onto 0.0 is the §12 fixed-order sum); the real fix needs a
  no-init allocator (f2_combine P1) — low ROI.

### LAYERING — clean, with two contract-coherence gaps

The §4 CUDA-free seam is upheld everywhere (verified against every include + the CMake PRIVATE
links): `core` reaches the GPU only through `ComputeBackend` + two CUDA-free combine decls;
`config.hpp`/`backend.hpp`/`resources.hpp`/`shard_plan.hpp`/`p2p_combine.hpp` name no CUDA type;
`steppe_core` archives no nvcc dlink object. No layering VIOLATION exists. Two coherence gaps:

**X5 [MED, PARITY-SAFE: yes] — The `enable_peer_access` knob is documented across the public
config layer but honored at NO layer, and the path it gates ACTIVELY VIOLATES it.** Confirmed by
grep: `enable_peer_access` is read by no code (only doc comments in `resources.hpp:101`,
`backend.hpp:168`). The combine gate (`f2_blocks_multigpu.cpp:171-172`) checks only
`prefer_p2p_combine && can_access_peer`; `p2p_combine.cu:265` calls `cudaDeviceEnablePeerAccess`
unconditionally — the exact operation `config.hpp:255-260` documents `enable_peer_access=false`
to forbid ("the MAY-WE knob: whether the backend is permitted to call cudaDeviceEnablePeerAccess
at all"). A user setting `enable_peer_access=false, prefer_p2p_combine=true` (a legal, documented
combination) is silently ignored AND violated. This single defect surfaces in FOUR per-unit
reviews (config C-1/K-2, resources K2, p2p_combine N2/L3, f2_blocks_multigpu CT2) — a textbook
cross-file finding. **Fix:** widen the gate to `prefer_p2p_combine && enable_peer_access &&
can_access_peer` so the doc becomes true and the unconditional enable is reached only with
permission.

**X6 [MED, PARITY-SAFE: yes] — The P2P gate is documented as 3-term (`… && G >= 2`) in FIVE
files but the shipped gate is 2-term.** Confirmed: `resources.hpp:41,57`, `p2p_combine.hpp:66`,
`f2_blocks_multigpu.cpp:27,158` all state `… && G >= 2`; the code (`f2_blocks_multigpu.cpp:172`)
has NO `G` term — `G>=2` is enforced structurally by the `if (G==1) return` at `:88`. Benign
today (the term is dead-true at the gate), but a latent refactor hazard: lifting the gate into a
reusable `select_combine_path(resources)` would carry the documented 3-term contract in its name
and the actual 2-term logic, entering the P2P combine at G==1 (untested). **Fix:** add the
dead-true `&& G >= 2` to the gate so it matches its five-times-documented contract (changes no
reached path), and collapse the 5 doc restatements to one authoritative home + cross-refs.
(backend.hpp 11.5; config C-2/DRY-1; this is also the canonical example of the next item.)

### DRY DUPLICATION ACROSS THE NEW FILES

**X7 [MED, PARITY-SAFE: yes] — `validate_partials` is duplicated byte-for-byte between the two
combine tiers.** Confirmed: `f2_combine.cpp:29` and `p2p_combine.cu:100` are the same contract
(sizes equal, P non-negative, each partial's n_block == shard span, P agreement, full tiling),
differing only in namespaced messages + the P2P side's extra `device_ids.size()` check. The
`.cu` comment even admits the lock-step ("so the two tiers reject identically") — a self-described
DRY violation that a drift would turn into the two tiers rejecting DIFFERENTLY (forbidden by the
parity-neutrality story). Both combines are CUDA-free in the parts that matter. **Fix:** hoist one
CUDA-free `validate_f2_partials(partials, shards, P, n_block_full)` into a shared `core` header
(it names only `F2BlockTensor`/`DeviceShard`); the P2P side adds only the `device_ids.size()`
check. Closing it in one place ALSO fixes the shared short-partial OOB gap (no validator checks
`part.f2.size() == P²·n_block` — f2_combine C1, p2p_combine inherits). (f2_combine CL1; p2p_combine C5.)

**X8 [LOW, PARITY-SAFE: yes] — The "CUDA-FREE BY CONTRACT" paragraph and the §11.4 gate condition
are each restated in 4–6 files.** The CUDA-free-seam WHY paragraph is re-spelled in full in
`resources.{hpp,cpp}`, `shard_plan.hpp`, `p2p_combine.hpp`, `backend.hpp`; the gate predicate in
5–6 (X6). Per §8 single-home each belongs once with cross-refs. (resources Cmt1; cuda_backend Read-1.)
Also: the dead `n_block_full < 0 ? 0 :` ternaries are copy-pasted between `f2_combine.cpp:96/100`
and `p2p_combine.cu:172/215` after `validate_partials` already rejects negatives (f2_combine C2;
p2p_combine C4) — falls out of X7.

### NAMING / CONTRACT INCONSISTENCIES

- **X9 [MED] — `backend.hpp` per-device-instance doc (lines 200-202) is STALE:** says the SNP-shard
  + combine "is the next workflow, not implemented here" — it IS implemented
  (`compute_f2_blocks_multigpu` + both combines); and says "full-shape partials" where the shipped
  design returns COMPACT `[P×P×(b1-b0)]` partials. A reader trusting the seam header thinks the
  combine is unwritten and full-shape. (backend.hpp 6.1.)
- **X10 [MED] — `CombinePath::None` doc claims it covers the G==1 fast path, but no run ever SETS
  `None`** — it is only the value-init default (the G==1 early-return at `f2_blocks_multigpu.cpp:88`
  returns without touching the field). After a G==2 run then a G==1 run on the same `Resources`,
  the tag stale-reads `P2pDeviceResident`. (resources E3.) Fix: set `None` on the G==1 return, or
  correct the doc.
- **X11 [LOW] — `can_access_peer` is an any-peer / root-only boolean** named as a clean per-device
  capability; exactly sufficient for the shipped G==2 case, but for G>2 on a partial fabric it
  cannot prove root-reaches-EVERY-peer (the gate would admit a P2P combine that fails-fast at the
  DMA). The header doc does not state the invariant. (backend.hpp 11.1; cuda_backend Cap-2.)
- **X12 [LOW] — `STEPPE_LOG_WARN` degrade is a NDEBUG no-op**, so the §11.4-mandated "explicit
  logged fallback" emits NOTHING in release — the exact build that ships to the budget box where
  the degrade ALWAYS fires. (f2_blocks_multigpu W1.) The `last_combine_path` tag is the only
  release-visible signal; route the degrade through a release-surviving log level when logging lands.

---

## (2) THE OPTIMIZE-WITHIN-PARITY STORY (top performance wins)

Ordered by impact. Every item re-gated by the locked `memcmp`. **The within-M4.5 combine win has
landed:** the device-resident combine (`867a4bf`) makes the 2-GPU path faster than single-GPU
(P=768 = 1.10×, P=400 = 1.22×, rtxbox/Release/EmuFp64{40}/median of 10), by deleting the redundant
2nd ~7.14 GB D2H + the H2D re-upload + the place-add (X3/X4). W2/W3/W4/W5 below are SUBSUMED by that
fix; the remaining rows are optional cleanups, not blockers. **(Post-M5 perspective: this ~1.1× is a
modest throughput layer; the real precompute speedups came from device-resident output `1f80c0c`
(~4.3×) + M5 streaming `176a07d`/`c65179f` — getting OFF the CPU — not multi-GPU.)**

| # | Win | Files | Parity | Effort |
|---|-----|-------|--------|--------|
| **W1** | **Per-device GEMM fan-out across G host threads** (a further overlap lever, NOT the speedup — the nsys trace already measured 74% overlap; X2) | `f2_blocks_multigpu.cpp` (+ `Threads::Threads` link) | **PARITY-SAFE: yes** (concurrency only; join then combine in fixed g order) | M |
| **W2** | ✅ **LANDED @867a4bf — Eliminate the P2P double-bounce** (X3): partials kept device-resident, `cudaMemcpyPeer` is a genuine peer→root pull, no H2D re-upload | `p2p_combine.cu`, `cuda_backend.cu` + seam | **yes** (proven `memcmp`-identical both tiers, both datasets) | DONE |
| **W3** | ✅ **SUBSUMED @867a4bf — per-partial `cudaDeviceSynchronize` + NULL-stream serialization removed** (X4): partials resident ⇒ no freed-buffer race, ONE fence before the final D2H | `p2p_combine.cu` | **yes** | DONE |
| **W4** | ✅ **SUBSUMED @867a4bf — `place_add_f2_kernel` deleted** (the resident combine places each partial into its disjoint slice via `cudaMemcpyPeer`, no add kernel) | `p2p_combine.cu` | **yes** | DONE |
| **W5** | ✅ **SUBSUMED @867a4bf — f2+vpair place-adds gone** (no place-add launch remains to fuse) | `p2p_combine.cu` | **yes** | DONE |
| **W6** | **Hoist `cudaDeviceEnablePeerAccess` to once-per-(root,peer)** (ideally into `build_resources`, gated on `enable_peer_access`) and delete the `cudaGetLastError` sticky-scrub | `p2p_combine.cu`, `resources.cpp` | **yes** (transport setup, parity-neutral) | S–M |
| **W7** | **Collapse the host combine's scalar `+=` triple loop into `std::copy_n`** of the contiguous owned runs (`memcpy`-grade; ALSO removes the latent −0.0 bit-flip — see note) | `f2_combine.cpp` | **yes** — `std::copy` is *strictly more* faithful to single-GPU than `+=` on −0.0 | S |
| **W8** | **Drop the throwaway device-0 backend in `resolve_device_order`** (64 MiB alloc/free + cuBLAS create/destroy + a discarded full probe) for a CUDA-free `visible_device_count()` query | `resources.cpp`, `backend_factory.hpp`, `cuda_backend.cu` | **yes** | S |
| **W9** | **Pin any remaining host-staged/spill D2H** (the eager whole-tensor D2H is gone post-M5 — device-resident output `1f80c0c` + tiered spill `176a07d`; this lever now only touches the host-staged baseline path); **`resize` not `assign(0.0)`** for the host result of the device combine | `p2p_combine.cu` | **yes** | S / S |
| — | Casting cleanup: settle one width per concept (`long` for kernel-feeding counts/indices, `size_t` for byte/alloc sizes); drop dead `?:` clamps; centralize the `DeviceShard` narrowing casts in one `make_shard`; use `core::cdiv` for the planner's ceiling-div | all combine + planner units | **yes** | S | 

**REJECTED-FOR-PARITY (do not let these slip in under "optimize the combine"):**
- NCCL AllReduce / tree-reduce / any non-`g=0..G-1` fan-in for the combine — order varies with G,
  not bit-identical to single-GPU (§12). The explicit parity line.
- `atomicAdd` partials into a shared accumulator — atomicAdd order is the dominant nondeterminism
  §12 eliminates; disjoint shards make it pointless anyway.
- Reordering the *accumulator place-adds* (only the COPIES may overlap) — disjoint shards make it
  arithmetically safe but §12 + the parity test pin the *literal* fixed-order sum.
- `std::transform(…, std::plus<>{})` for the host combine — vectorizes but KEEPS the −0.0 flip;
  only `std::copy` removes it (W7).
- Tolerance for the EmulatedFp64 parity asserts — §12 is bit-IDENTITY (`memcmp`), not tolerance.
- Skipping the combine zero-init / pure-placement that diverges the two tiers' arithmetic SHAPE.

**Latent numerical note (W7, f2_combine N2):** the host combine's `+=` onto a `+0.0`-init accumulator
flips any `−0.0` partial element to `+0.0` (IEEE: `(+0.0)+(−0.0)=+0.0`, a different bit pattern),
while the single-GPU reference computes the slab directly. DORMANT on AADR (f2 is a sum of squares,
no −0.0 observed), and the doc's "`x + 0.0 == x` for all finite x" is wrong on `x=−0.0`. `std::copy`
(W7) removes the hazard unconditionally — making W7 both a perf win and a parity-hardening.

---

## (3) PRIORITIZED MASTER BACKLOG (deduped across all units)

### BEFORE-M5 → now M4.5 RESIDUAL DEBT (M5 streaming `176a07d`/`c65179f` has since landed on top; B3–B9 are still-open debt, not blockers)

| # | Item | File(s) | Fix | Sev | Parity |
|---|------|---------|-----|-----|--------|
| **B1** | ✅ DONE @867a4bf — multi-GPU now FASTER than single-GPU (P=768 1.10×, P=400 1.22×). Achieved via the device-resident combine (B2), NOT the fan-out. Per-device host-thread fan-out remains an OPTIONAL further-overlap lever (not a blocker; nsys already measured 74% overlap). | `f2_blocks_multigpu.cpp` (+ `Threads::Threads` if pursued) | G host threads, write own `partials[g]`, join before combine, `exception_ptr` rethrow | OPTIONAL | yes |
| **B2** | ✅ DONE @867a4bf — device-resident combine: deletes the redundant 2nd ~7.14 GB D2H + the H2D re-upload + the `cudaMemset` + the place-add + per-peer `cudaDeviceSynchronize`. Partials stay resident; root D2D-copies its own + `cudaMemcpyPeer`s each peer straight into its disjoint slice; ONE final D2H. Proven `memcmp`-identical both tiers, both datasets. The B2 P2P-transport fix-pass (`agentscripts/m4.5-b2-p2p-fix-pass.js`) is REASSESS / largely subsumed. | `p2p_combine.cu`, `cuda_backend.cu` + seam | landed | DONE | yes |
| **B3** | Wire+fix `enable_peer_access` gate (dead knob actively violated) | `f2_blocks_multigpu.cpp`, `p2p_combine.cu`, config doc | Gate `prefer_p2p_combine && enable_peer_access && can_access_peer`; reach the enable only with permission | MED | yes |
| **B4** | Make gate match its 3-term doc + single-home the predicate | `f2_blocks_multigpu.cpp` + 5 doc sites | Add dead-true `&& G>=2`; collapse 5 restatements to one home + cross-refs | MED | yes |
| **B5** | Single-home `validate_partials` (+ close the shared short-partial OOB gap) | new `core` header; `f2_combine.cpp`, `p2p_combine.cu` | One CUDA-free `validate_f2_partials`; P2P adds only `device_ids.size()`; add `f2.size()==P²·nb` check | MED | yes |
| **B6** | Drop redundant `block_sizes`; derive from `ranges` | `shard_plan.{hpp,cpp}`, `f2_blocks_multigpu.cpp` | Remove the param, the parallel-array contract, the caller `long→int` narrowing, the sign hole, half the cast scatter | MED | yes |
| **B7** | Host combine: `std::copy_n` placement (perf + removes the −0.0 flip) | `f2_combine.cpp` | Replace scalar `+=` triple loop; correct the `x+0.0` doc | MED | yes (strictly safer) |
| **B8** | Drop the throwaway device-0 backend + add §9 ordinal validation | `resources.cpp`, `backend_factory.hpp`, `cuda_backend.cu` | CUDA-free `visible_device_count()`; reject duplicate/out-of-range ordinals (one count query serves both); removes the leaked `cudaSetDevice(0)` | MED (HIGH perf) | yes |
| **B9** | GPU-free host unit tests for the host-pure logic | `tests/unit/test_shard_plan.cpp`, `test_f2_combine.cpp`, `test_f2_blocks_multigpu.cpp` | Planner tiling/skew/edges; combine placement + fixed-order + every validate throw + the −0.0 case; orchestrator gate predicate + sub-view/local-id + empty/`n_block<G` (needs a fake `ComputeBackend` + D1 extraction) | MED | yes (tests) |

Rationale for the (now-historical) BEFORE-M5 cut: B1+B2 are **DONE** @867a4bf — the within-M4.5
combine slowdown is fixed (2-GPU ~1.1×) AND the surface M5 streaming builds on was in place, with
parity freshly proven and locked by the `memcmp` gate. **M5 has since landed on top** (device-resident
output `1f80c0c` + adaptive tiered output `176a07d` + SNP-tile input streaming `c65179f`; full-autosome
P=2500 on a single 32 GB 5090 in ~51.5 s) — so the framing here is no longer "pay before streaming" but
"residual M4.5 debt to retire when the multi-GPU paths are next touched (most relevant on the Phase-2
FIT/ROTATION, multi-GPU's proper home)." The remaining items are debt, not speedup:
B3–B4 are contract lies that calcify under any refactor. B5–B6 are DRY/contract debt that a
streaming refactor would otherwise duplicate further. B7 is a small perf win that also hardens a
latent parity hazard. B8 is a cold-start cleanup + a §9 fail-fast gap. B9 is the fast inner-loop
gate that makes B5–B7 safe to land without the slow GPU parity run each time.

### LATER (post-M5, or hardware-gated, or doc-only nice-to-haves)

| # | Item | File(s) | Sev | Parity |
|---|------|---------|-----|--------|
| L1 | ✅ DONE @867a4bf (combine partial) + generalized post-M5 — Device-resident `compute_f2_blocks` partial (real P2P pull, no bounce) was pulled forward into the combine; the broader **device-resident OUTPUT** then shipped as `1f80c0c` (`DeviceF2Blocks` handle, host copy opt-in `.to_host()`, P=512 ~4.3×) — the precompute is no longer host-result-bound. | `cuda_backend.cu` + seam | DONE | yes |
| L2 | ✅ MOSTLY DONE post-M5 — the eager whole-tensor result D2H is gone (device-resident output `1f80c0c`); the beyond-VRAM case streams block-tiles to the fastest tier (`176a07d`). Residual optional: a `PinnedBuffer` wrapper to pin any remaining host-staged/spill H2D/D2H. | `p2p_combine.cu`, `device_buffer.cuh`, new `pinned_buffer.cuh` | LOW (optional) | yes |
| L3 | Balanced contiguous shard partition (idle-GPU on skewed inputs) + bucketed cost model | `shard_plan.cpp` | MED-HIGH (narrow-input) | yes-if-careful |
| L4 | `MathModeScope` is built but UNWIRED — the §12 oracle math-mode leak is still open in `engage_f2_precision`; wire it (return a scope) | `handles.hpp`, `f2_block_kernel.cu` | MED | yes |
| L5 | `MathModeScope` lacks the device-ordinal guard the same delta added to `CublasHandle` (delta-internal asymmetry) | `handles.hpp` | MED | yes |
| L6 | NVTX phase ranges for SPMG shard-compute/combine (facade must be BUILT first — does not exist) | new `core/internal/nvtx.hpp`, `f2_blocks_multigpu.cpp` | LOW | yes |
| L7 | `check.cuh`: document the sticky-last-error caveat of `STEPPE_CUDA_WARN`; DRY the message format; honest "log once per invocation" wording | `check.cuh` | MED | yes |
| L8 | Parity-gate hardening: assert EmulatedFp64 actually ENGAGED (silent-native-fallback blind spot, HIGH for that unit); bound `report_first_diff` vpair loop; distinguish absent vs malformed dataset; capability-biconditional instead of "PRO" name match | `test_f2_multigpu_parity.cu` | MED–HIGH (test) | yes |
| L9 | Make the degrade WARN survive release (NDEBUG no-op today) | `f2_blocks_multigpu.cpp` (+ logging milestone) | MED | yes |
| L10 | `device_id()` accessor on `CublasHandle` is unreachable from its documented consumer (abstract backend) — trim doc or drop | `handles.hpp` | MED | yes |
| L11 | Stale/half-specified seam docs: `backend.hpp` "next workflow"/"full-shape" (X9); `CombinePath::None` G==1 lie (X10); `can_access_peer` G>2 invariant (X11); `capabilities()` exception + device-neutrality contract; APPEND-ONLY ABI banner on `BackendCapabilities`; `prefer_p2p_combine` in the arch §9 listing | `backend.hpp`, `resources.hpp`, `config.hpp`, arch | LOW each | yes |
| L12 | `cuda_backend` probe: narrow the `cudaSetDevice` bracket to `cudaMemGetInfo` only; two `cudaDeviceGetAttribute` instead of the full `cudaDeviceProp`; fix the Corr-2 same-device-peer comment; per-host-thread caveat | `cuda_backend.cu` | LOW | yes |
| L13 | Decompose `compute_f2_blocks_multigpu` (~150 lines, 6 jobs) and `run_dataset` (~220 lines) and `combine_f2_partials_p2p` (~200 lines) into named steps (enables B9's host tests) | `f2_blocks_multigpu.cpp`, test, `p2p_combine.cu` | MED | yes |
| L14 | Balance-quality / G>2 / partial-fabric coverage and the all-peers reachability tier — hardware-gated (no >2-GPU box exists) | `shard_plan.cpp`, `backend.hpp`, tests | LOW–MED | yes |

---

## (4) PER-AREA SCORES

| Area | Score | Note |
|------|-------|------|
| **End-to-end data flow / overlap** | **9.0** | The data-bounce wart is GONE (X3/X4 RESOLVED @867a4bf): the device-resident combine deletes the redundant 2nd ~7.14 GB D2H + the H2D re-upload, and multi-GPU is now FASTER than single-GPU (1.10× P=768, 1.22× P=400, nsys-measured 74% overlap). Held off 10 only by the optional per-device host-thread fan-out (X2). Post-M5, the "pin the final result D2H" lever is largely moot — device-resident output (`1f80c0c`) removed the eager whole-tensor D2H and M5 streaming (`176a07d`/`c65179f`) covers the beyond-VRAM case. |
| **Layering / CUDA-free seam** | **9.5** | Exemplary and verified end-to-end; the cleanest §4 realization in the codebase. Only the two contract-coherence gaps (X5 dead knob, X6 gate doc) keep it off 10. |
| **DRY / single-home** | **8.0** | `validate_partials` duplicated (X7), the gate predicate 5–6 homes (X6/X8), the CUDA-free paragraph 4–6 homes, dead ternaries copy-pasted. All §8 misses, none a bug. |
| **Capability-tier coherence** | **8.5** | The probe-once / out-of-band tag / non-throwing degrade discipline is textbook; held by the dead `enable_peer_access` (X5), the `None` G==1 lie (X10), the release-silent WARN (X12), the any-peer/G>2 gap (X11). |
| **Parity correctness / §12** | **9.5** | The bit-identity GATE is met and proven both tiers across G. Latent −0.0 flip (dormant, W7), the gate's silent-emulation-fallback blind spot (L8). The fixed-order law is held without exception. |
| **Testability / §13** | **7.5** | The locked GPU parity gate is strong; but the host-pure planner, combine, and orchestrator are exercised ONLY through that one `.cu` — no GPU-free unit coverage of the cheapest-to-test units (B9). |
| **Naming / contract precision** | **8.5** | Mostly precise; the stale seam docs (X9/X10/X11) and the 3-term gate doc (X6) are the drift. |
| **CUDA idioms / §7** | **8.5** | RAII/record-and-assert/post-launch-checks are clean; the device-resident combine (`867a4bf`) replaced the NULL-stream/per-partial-`cudaDeviceSynchronize`/inline-add anti-pattern with a resident D2D + `cudaMemcpyPeer` + ONE fence (X4 resolved). Remaining: `MathModeScope` is unwired (L4) without its own device guard (L5); pinned async H2D is an optional lever (W9). |

---

## CONCLUSION

M4.5 is **correct, parity-locked, beautifully-layered, AND faster than single-GPU** — the milestone
is COMPLETE and on `main`. The device-resident combine (`867a4bf`) deleted the data-bounce wart that
the nsys diagnosis (`165f655`, `why-multigpu-slow.md`) pinned as the root cause of the old slowdown,
and the 2-GPU path now beats single-GPU (P=768 1.10×, P=400 1.22×, rtxbox/Release/EmuFp64{40}).
**Post-M5 verdict (the honest perf story):** that ~1.1× is a *modest throughput layer*, not the
headline — the precompute was HOST-RESULT-BOUND, and the real speedups came after M4.5 from
**device-resident output** (`1f80c0c`, P=512 ~4.3×) and **M5 out-of-core streaming**
(`176a07d` adaptive tiered output + `c65179f` SNP-tile input streaming), i.e. getting OFF the CPU.
**M5 is now DONE:** full-autosome (M=584131, n_block=757) P=2500 completes on a single 32 GB RTX 5090
in ~51.5 s (76 GB result streamed, GPU peak ~26 GB), parity bit-identical. None of the M4.5 gaps
threatens the proven bit-identity. The remaining M4.5 work is debt, not speedup: wiring the
`enable_peer_access` gate (B3), making the gate match its 3-term doc (B4), single-homing
`validate_partials` (B5), dropping the redundant `block_sizes` (B6), the host-combine `std::copy`
(B7), the cold-start probe fix + §9 validation (B8), and GPU-free host tests (B9). Optional further
levers: per-device host-thread fan-out (B1). The B2 P2P-transport fix-pass is REASSESS / largely
subsumed by the resident combine. **The real next work is the qpAdm FIT ENGINE (Phase 2, S3-S8,
still unbuilt)** — multi-GPU's proper home (thousands of independent qpAdm models, no combine —
embarrassingly parallel); then AT2 goldens (the validation gate), M6 merge. The perf
rabbit-hole is CLOSED.

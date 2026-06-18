# M4.5 Multi-GPU — WHOLE-MILESTONE HOLISTIC REVIEW

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
| `device/cuda/p2p_combine.{cu,hpp}` | 7.5/10 | 11 |
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

## OVERALL M4.5 SCORE: 8.4 / 10

The milestone is **correct, parity-locked, and exemplary on layering** — the CUDA-free seam
(`backend.hpp` POD probe, the `p2p_combine.hpp` / `shard_plan.hpp` CUDA-free decls, `resources`
reaching the GPU only through `make_cuda_backend`+`capabilities()`) is the cleanest realization
of arch §4 in the codebase, and the out-of-band `CombinePath` tagging discipline (§12 / cleanup
§(2).2) is held without exception. The two highest-rated units (`backend.hpp`, `config.hpp`,
both 9.5) earn it.

It is held to **8.4** by a coherent cluster of issues that only the whole-milestone view makes
visible:

1. **The headline multi-GPU SPEEDUP is unrealized.** The per-device GEMM work runs strictly
   serially (`f2_blocks_multigpu.cpp:130–154` — each `compute_f2_blocks` blocks on its own
   trailing `cudaStreamSynchronize` before the next device is even *issued*). On the 2-GPU box
   wall-clock is `Σ_g time(g)`, not `max_g time(g)`. The milestone ships a parity *scaffold*,
   not the ~2× the milestone promises. This is the single biggest gap, and it is the thing M5
   streaming will be built ON TOP OF — so it must be paid first.

2. **The named "prime perf target" (`p2p_combine.cu`, 7.5) violates the §7 idiom stack it is
   supposed to exemplify** — a host→peer→root DOUBLE-BOUNCE for already-host-resident data, a
   per-partial full `cudaDeviceSynchronize`, a per-iteration `cudaDeviceEnablePeerAccess`, an
   inline non-grid-stride kernel, all on the NULL stream. None breaks parity; all are
   parity-safe to fix.

3. **A dead config knob + a documented-but-false gate.** `enable_peer_access` is read by NO
   code (confirmed by grep — only doc comments) yet `p2p_combine.cu:265` unconditionally calls
   the very `cudaDeviceEnablePeerAccess` it is documented to gate; and the P2P gate is
   documented as 3-term (`… && G >= 2`) in FIVE files while the shipped gate is 2-term. These
   are contract lies on the seam the whole tier keys off.

4. **A small constellation of DRY duplications and host-test gaps** — `validate_partials`
   duplicated byte-for-byte across the two combine tiers, the §11.4 gate predicate restated in
   5–6 homes, the degrade WARN compiled out under NDEBUG, and the host-pure planner / orchestrator
   / combine all exercised only through one GPU `.cu`.

**Concrete gap to 9.5+:** land the BEFORE-M5 backlog below (items 1–9). The decisive items are
the per-device fan-out (#1, the actual speedup), the P2P transport rework (#2–#4, the named perf
target made idiomatic), wiring/fixing the `enable_peer_access`+`G>=2` gate (#5), single-homing
`validate_partials` (#6), and the GPU-free host tests for the planner/combine/orchestrator (#9).
With those, no unit sits below 9 and the cross-file story is coherent — a 9.5+ milestone.

---

## (1) CROSS-FILE ISSUES (no single-unit review sees these)

### END-TO-END DATA FLOW: shard → per-device compute → combine → result

The pipeline (confirmed by reading the source, not just the reviews):

```
compute_f2_blocks_multigpu  (core, f2_blocks_multigpu.cpp)
  ├─ block_ranges(partition)                              [host, O(M)]
  ├─ block_sizes[b] = ranges[b].size()                    [host, O(n_block) — REDUNDANT, see X1]
  ├─ plan_block_shards(block_sizes, ranges, G)            [host, returns G DeviceShards]
  ├─ for g in 0..G-1:                  ◄── STRICTLY SEQUENTIAL (X2 — the headline gap)
  │     build Qg/Vg/Ng zero-copy sub-views               [host, pointer offset — good]
  │     build block_id_local (per-device subtract)        [host alloc, O(M_local)]
  │     partials[g] = gpus[g].backend->compute_f2_blocks(...)
  │         └─ CudaBackend: cudaSetDevice → H2D upload Q/V/N (pageable) → feeder
  │            → grouped strided-batched GEMM → D2H copy partial to HOST vector
  │            → cudaStreamSynchronize  ◄── BLOCKS before g+1 is issued
  └─ combine (the §4 fork):
        host-staged:  combine_f2_partials_host(partials, shards, P, n_block)
          └─ zero-init full [P²·n_block] f2+vpair, then += each compact partial   [host]
        P2P:          combine_f2_partials_p2p(partials, shards, device_ids, P, n_block, root)
          └─ cudaMemset(0) accumulators on root
             for g: host partial → H2D to PEER → cudaMemcpyPeer PEER→ROOT  ◄── DOUBLE-BOUNCE (X3)
                    → cudaDeviceSynchronize  ◄── per-partial full device drain (X4)
                    → place_add_f2_kernel (inline, non-grid-stride, NULL stream)
             D2H combined accumulator → HOST F2BlockTensor
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

**X2 [HIGH, PARITY-SAFE: yes] — The G devices run STRICTLY SEQUENTIALLY; the multi-GPU speedup
is unrealized.** Files: `f2_blocks_multigpu.cpp:130–154`; blocker is the `compute_f2_blocks`
seam (`cuda_backend.cu:397` trailing `cudaStreamSynchronize`, returns a HOST tensor). The
hardware can run device g and g+1 concurrently (per-device default streams, CUDA Programming
Guide §3.4); the blocking host loop forfeits it. **Fix:** fan the G `compute_f2_blocks` calls
out across G host threads (one per device), each writing its own pre-sized `partials[g]` slot,
join before the combine, keep the combine fixed g=0..G-1. No shared mutable state (distinct
slots; each backend `guard_device`s its own device; `cudaSetDevice` is per-host-thread). Capture
each worker's exception via `std::exception_ptr` and rethrow the first on join (a thread that
lets an exception escape calls `std::terminate`). **Requires** linking `Threads::Threads` on
`steppe_core` (it does not today — f2_blocks_multigpu W2). Parity-safe: the combine reads
`partials[g]` in fixed g order AFTER the join barrier; GEMM bits are independent of wall-clock
slot. This is the milestone's whole point and the foundation M5 streaming sits on. (f2_blocks_multigpu P1.)

**X3 [HIGH, PARITY-SAFE: yes (M4.5-local) / yes-if-careful (the real fix)] — The P2P combine
does a host→peer→root DOUBLE-BOUNCE for data that is already host-resident.** Files:
`p2p_combine.cu:284–300`; root cause is the `compute_f2_blocks` return contract (HOST tensor,
device buffers freed — `cuda_backend.cu:202-210`). For a peer-owned partial the code (1)
`cudaSetDevice(peer)`, (2) allocates `dPeer_*` on the peer, (3) H2D host→peer, (4)
`cudaSetDevice(root)`, (5) `cudaMemcpyPeer` peer→root — two device copies of the same bytes plus
a per-iteration peer malloc/free, when the source is a host `std::vector<double>`. The
M4.5-local optimum is a single `cudaMemcpy(dStage, host, …, H2D)` — exactly what the root branch
already does — but that makes the routine no longer exercise `cudaMemcpyPeer`, the literal point
of the P2P tier and what the PRO-tier parity assertion verifies. The **architecturally correct**
resolution (M5 follow-up, touches `compute_f2_blocks`, yes-if-careful): keep each device's partial
device-resident so `cudaMemcpyPeer` is a genuine peer→root pull — real P2P AND no bounce.
NOTE: this is the ONE cross-file data-bounce that is genuinely removable; the host-staged tier's
D2H gather is inherent to the portable baseline (on a no-peer box you cannot keep partial g on
device 0 without a host bounce), and arch §12 line 741 sanctions it as the design. (p2p_combine P1/P2; f2_blocks_multigpu P7.)

**X4 [HIGH, PARITY-SAFE: yes] — The whole pipeline is sequential where streams/events would
overlap.** Three serialization points compound:
- the per-device fan-out (X2);
- inside the P2P combine, a full `cudaDeviceSynchronize()` after EVERY peer pull (`p2p_combine.cu:316`)
  — needed today (the peer source buffer is freed at iteration end and races the cross-device DMA)
  but the heaviest possible fence; an `Event` fence (or eliminating the peer buffer under X3) replaces it;
- everything on the NULL stream, which cross-device-serializes per partial.
The combine has a true serial dependency ONLY on the accumulator add ORDER (§12), never on the
transport. The parity-safe shape: a K-deep staging RING + per-pull streams + events overlapping
the COPIES, with the place-adds serialized on one accumulator stream in fixed g order. A single
reused `dStage_*` (today) cannot be the ring — partial g+1's copy would overwrite it while g's
place-add still reads it (a WAR hazard currently masked only by NULL-stream + the heavy sync).
(p2p_combine P3/P6/P9/N4; f2_blocks_multigpu P7.)

**Where data bounces H↔D unnecessarily (the full inventory):**
- P2P peer partials: H2D→peer then peer→root, removable to one H2D now / device-resident later (X3). **Real, removable.**
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

Ordered by impact. Every item re-gated by the locked `memcmp`.

| # | Win | Files | Parity | Effort |
|---|-----|-------|--------|--------|
| **W1** | **Per-device GEMM fan-out across G host threads** (the actual ~2× speedup; X2) | `f2_blocks_multigpu.cpp` (+ `Threads::Threads` link) | **PARITY-SAFE: yes** (concurrency only; join then combine in fixed g order) | M |
| **W2** | **Eliminate the P2P host→peer→root double-bounce** (X3): single H2D now / device-resident `compute_f2_blocks` partial later | `p2p_combine.cu` (now); `cuda_backend.cu` + seam (later) | **yes** (M4.5-local) / **yes-if-careful** (the device-resident M5 variant — byte-exact retained partial, unperturbed GEMM batch shape) | S / L |
| **W3** | **Streamed P2P: K-deep staging ring + per-pull streams + events** replacing the per-partial `cudaDeviceSynchronize` + NULL stream; overlap COPIES, keep place-adds serialized on one accumulator stream in fixed g order (X4) | `p2p_combine.cu` | **yes-if-careful** (overlap transport, NEVER reorder the accumulator adds — §12) | M–L |
| **W4** | **Grid-stride `place_add_f2_kernel`** — removes the RELEASE-only silent-under-cover trap (the grid-extent `STEPPE_ASSERT` is debug-only); a fixed grid covers any count | `p2p_combine.cu` | **yes** (each element written once with same `+=`, no reassociation) | S |
| **W5** | **Fuse the f2+vpair place-adds into one launch** behind a narrow `void launch_place_add(...)` wrapper (§7 idiom; matches the host baseline's single interleaved loop) | `p2p_combine.cu` | **yes** (disjoint buffers, same operands/order) | S |
| **W6** | **Hoist `cudaDeviceEnablePeerAccess` to once-per-(root,peer)** (ideally into `build_resources`, gated on `enable_peer_access`) and delete the `cudaGetLastError` sticky-scrub | `p2p_combine.cu`, `resources.cpp` | **yes** (transport setup, parity-neutral) | S–M |
| **W7** | **Collapse the host combine's scalar `+=` triple loop into `std::copy_n`** of the contiguous owned runs (`memcpy`-grade; ALSO removes the latent −0.0 bit-flip — see note) | `f2_combine.cpp` | **yes** — `std::copy` is *strictly more* faithful to single-GPU than `+=` on −0.0 | S |
| **W8** | **Drop the throwaway device-0 backend in `resolve_device_order`** (64 MiB alloc/free + cuBLAS create/destroy + a discarded full probe) for a CUDA-free `visible_device_count()` query | `resources.cpp`, `backend_factory.hpp`, `cuda_backend.cu` | **yes** | S |
| **W9** | **Pinned staging + async H2D** for the P2P uploads (enables W3 pipelining); **`resize` not `assign(0.0)`** for the host result of the device combine | `p2p_combine.cu` | **yes** | M / S |
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

### BEFORE-M5 (pay before building streaming on top)

| # | Item | File(s) | Fix | Sev | Parity |
|---|------|---------|-----|-----|--------|
| **B1** | Per-device GEMM fan-out (the speedup) | `f2_blocks_multigpu.cpp` + `core/CMakeLists.txt` | G host threads, one per device, write own `partials[g]`, join before combine, `exception_ptr` rethrow; link `Threads::Threads` | HIGH | yes |
| **B2** | P2P streamed + double-bounce-eliminated combine | `p2p_combine.cu` | Single H2D (W2); event fence not `cudaDeviceSynchronize` (W3); K-deep staging ring + streams (W3); grid-stride kernel behind `launch_place_add` (W4); fuse f2+vpair (W5); hoist peer-enable (W6) | HIGH | yes / yes-if-careful (W3) |
| **B3** | Wire+fix `enable_peer_access` gate (dead knob actively violated) | `f2_blocks_multigpu.cpp`, `p2p_combine.cu`, config doc | Gate `prefer_p2p_combine && enable_peer_access && can_access_peer`; reach the enable only with permission | MED | yes |
| **B4** | Make gate match its 3-term doc + single-home the predicate | `f2_blocks_multigpu.cpp` + 5 doc sites | Add dead-true `&& G>=2`; collapse 5 restatements to one home + cross-refs | MED | yes |
| **B5** | Single-home `validate_partials` (+ close the shared short-partial OOB gap) | new `core` header; `f2_combine.cpp`, `p2p_combine.cu` | One CUDA-free `validate_f2_partials`; P2P adds only `device_ids.size()`; add `f2.size()==P²·nb` check | MED | yes |
| **B6** | Drop redundant `block_sizes`; derive from `ranges` | `shard_plan.{hpp,cpp}`, `f2_blocks_multigpu.cpp` | Remove the param, the parallel-array contract, the caller `long→int` narrowing, the sign hole, half the cast scatter | MED | yes |
| **B7** | Host combine: `std::copy_n` placement (perf + removes the −0.0 flip) | `f2_combine.cpp` | Replace scalar `+=` triple loop; correct the `x+0.0` doc | MED | yes (strictly safer) |
| **B8** | Drop the throwaway device-0 backend + add §9 ordinal validation | `resources.cpp`, `backend_factory.hpp`, `cuda_backend.cu` | CUDA-free `visible_device_count()`; reject duplicate/out-of-range ordinals (one count query serves both); removes the leaked `cudaSetDevice(0)` | MED (HIGH perf) | yes |
| **B9** | GPU-free host unit tests for the host-pure logic | `tests/unit/test_shard_plan.cpp`, `test_f2_combine.cpp`, `test_f2_blocks_multigpu.cpp` | Planner tiling/skew/edges; combine placement + fixed-order + every validate throw + the −0.0 case; orchestrator gate predicate + sub-view/local-id + empty/`n_block<G` (needs a fake `ComputeBackend` + D1 extraction) | MED | yes (tests) |

Rationale for the BEFORE-M5 cut: B1+B2 are the speedup the milestone promised and the surface M5
streaming builds on — pay them while parity is freshly proven and the locked `memcmp` is the gate.
B3–B4 are contract lies that calcify under any refactor. B5–B6 are DRY/contract debt that a
streaming refactor would otherwise duplicate further. B7 is a perf win that also hardens a latent
parity hazard. B8 is a HIGH-perf cold-start win + a §9 fail-fast gap. B9 is the fast inner-loop
gate that makes B1–B7 safe to land without the slow GPU parity run each time.

### LATER (post-M5, or hardware-gated, or doc-only nice-to-haves)

| # | Item | File(s) | Sev | Parity |
|---|------|---------|-----|--------|
| L1 | Device-resident `compute_f2_blocks` partial (real P2P pull, no bounce) — the M5 form of W2 | `cuda_backend.cu` + seam | HIGH | yes-if-careful |
| L2 | Pinned staging + async H2D for the P2P uploads (needs a `PinnedBuffer` wrapper, not yet implemented) | `p2p_combine.cu`, `device_buffer.cuh` | MED | yes |
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
| **End-to-end data flow / overlap** | **7.0** | The serial fan-out (X2) + the P2P double-bounce/sync (X3/X4) are the milestone's biggest gap; all parity-safe to fix, but the speedup is unrealized and the named perf target models the anti-pattern. |
| **Layering / CUDA-free seam** | **9.5** | Exemplary and verified end-to-end; the cleanest §4 realization in the codebase. Only the two contract-coherence gaps (X5 dead knob, X6 gate doc) keep it off 10. |
| **DRY / single-home** | **8.0** | `validate_partials` duplicated (X7), the gate predicate 5–6 homes (X6/X8), the CUDA-free paragraph 4–6 homes, dead ternaries copy-pasted. All §8 misses, none a bug. |
| **Capability-tier coherence** | **8.5** | The probe-once / out-of-band tag / non-throwing degrade discipline is textbook; held by the dead `enable_peer_access` (X5), the `None` G==1 lie (X10), the release-silent WARN (X12), the any-peer/G>2 gap (X11). |
| **Parity correctness / §12** | **9.5** | The bit-identity GATE is met and proven both tiers across G. Latent −0.0 flip (dormant, W7), the gate's silent-emulation-fallback blind spot (L8). The fixed-order law is held without exception. |
| **Testability / §13** | **7.5** | The locked GPU parity gate is strong; but the host-pure planner, combine, and orchestrator are exercised ONLY through that one `.cu` — no GPU-free unit coverage of the cheapest-to-test units (B9). |
| **Naming / contract precision** | **8.5** | Mostly precise; the stale seam docs (X9/X10/X11) and the 3-term gate doc (X6) are the drift. |
| **CUDA idioms / §7** | **8.0** | RAII/record-and-assert/post-launch-checks are clean; the P2P combine breaks the streams/events/grid-stride/narrow-wrapper/async-pinned stack (X4/W3–W5), and `MathModeScope` is unwired (L4) without its own device guard (L5). |

---

## CONCLUSION

M4.5 is a **correct, parity-locked, beautifully-layered scaffold** that has not yet realized the
multi-GPU speedup it exists to deliver, and whose named perf exemplar (`p2p_combine.cu`) models the
very anti-patterns it should showcase. None of the gaps threatens the proven bit-identity; all are
parity-safe to fix. The decisive BEFORE-M5 work is the per-device fan-out (B1 — the speedup) and the
P2P transport rework (B2 — the idiomatic streamed combine), backed by wiring the `enable_peer_access`
gate (B3), single-homing the duplicated contracts (B4–B6), the host-combine `std::copy` (B7), the
cold-start probe fix + §9 validation (B8), and GPU-free host tests (B9). Landing those moves every
unit to ≥9 and makes the cross-file story coherent — a 9.5+ milestone and a sound foundation for M5
streaming.

# M4.5 multi-GPU architecture — adversarial verdict

Lead-architect cross-examination of four lens audits (data-layout-memory, parallelism-scaling,
algorithm-formulation, system-fit) against the measured scaling sweep
(`docs/cleanup/m4.5/scaling-sweep.md`, rtxbox 2× RTX PRO 6000, 95.6 GB/GPU, 169 GB host, Release,
`EmulatedFp64{40}`, P=2500/M=584131/n_block=757). **Where a lens and the numbers conflict, the
numbers win.** READ-ONLY on code; every claim is tagged to a file:line or a measured number.

---

## (1) THE VERDICT

**IDEAL? = no.** The 3-GEMM + fixed-slice-Ozaki + native-FP64-assemble *kernel formulation* is sound
and measured, and the device-resident combine (the `867a4bf` fix) genuinely killed the old double-D2H
bounce. But the **shape of the deliverable is wrong**, and that is what produces every suspicious
signal in the sweep. Three load-bearing reasons: **(i)** the result is a full-DENSE FP64
`[P²·n_block]` tensor for a provably SYMMETRIC quantity (`fstats.hpp:40` says so explicitly) — a flat
**2× waste** that alone moves ceilings (P=2500 root: 113.6 GB → 56.7 GB, *fits*; verified below);
**(ii)** the single-GPU OOM at P=1536 is **not** the 28.6 GB result — it is a **78.8 GB feeder-phase
peak** from holding the whole result *plus* the un-tiled full-width inputs/feeder-outputs co-resident
(`cuda_backend.cu:410,414-415,428`), in direct violation of the spec's own out-of-core mandate
(`architecture.md:12`, §11.1); **(iii)** the multi-GPU speedup shrinks 1.29×→1.07× because the entire
result is funnelled through ONE serial, **pageable** D2H on the root that grows as P²
(`p2p_combine.cu:178-181`) — pure Amdahl on a tail that 2 GPUs cannot touch. Multi-GPU was built
*before* the out-of-core (M5) work that the same sweep proves is the actual ceiling-breaker; it opens
only a narrow P1536–2000 window at a *shrinking* sub-1.3× speedup. **Multi-GPU is a real but
secondary throughput lever bolted onto a deliverable whose memory shape is the binding constraint —
it papers over a memory-shape problem it cannot solve (the P2P root still OOMs at 2500).**

---

## (2) RANKED architectural flaws

Each: what / quantified cost (which sweep signal it explains) / file:line / PARITY-safety of the fix.

### FLAW 1 — Full-DENSE storage of a SYMMETRIC tensor: a flat 2× waste on the binding axis (explains signal c, and half of a)

**What.** `F2BlockTensor.f2` and `.vpair` are full column-major `[P×P]` slabs (`fstats.hpp:47-60`),
and the header *itself* states "Each [P×P] slab is SYMMETRIC" (`fstats.hpp:40`) with the diagonal
carried but "Downstream f3/f4 read off-diagonal f2 only" (`fstats.hpp:45`). Both kernels write the
full P² (`f2_block_kernel.cu:158,174`). The budget reserves the full pair (`vram_budget.hpp:62-68`),
the chunk slabs `dGg/dVpairg/dRg` are full P² (`cuda_backend.cu:520`), and the combine D2Hs the full
P² (`p2p_combine.cu:178-181`). Storing the upper triangle incl. diagonal (`P(P+1)/2`) is exactly half.

**Quantified.** Resident-pair waste: P=1536 **14.3 GB**, P=2000 **24.2 GB**, P=2500 **37.8 GB**.
Decisive at the ceiling: the device-resident combine root OOMs at P=2500 because it holds the full
result + its own resident partial ≈ 75.7 + 37.9 ≈ **113.6 GB ≫ 95.6 GB** (`scaling-sweep.md:35`).
Triangular halves both terms → **56.7 GB, which FITS on one 95.6 GB card.** This single change moves
the device-resident ceiling from P≈2000 past P≈2500. It also halves the P²-growing D2H tail (Flaw 3)
and the GEMM-output buffers. **This is the single highest-leverage change for the OOM ceilings.**

**PARITY-safety.** Packed-triangle write + a fit-side symmetric accessor; the stored doubles are
bit-identical (the lower triangle is a redundant copy, never independently consumed). SAFE if the
diagonal convention (`fstats.hpp:40-46`) is preserved in the packed layout. The optional `cublasDsyrk`
for G=Q·Qᵀ and Vpair=V·Vᵀ is a *separate* change with FP-reassociation risk against the strided-batched
`GemmEx` (`f2_blocks_kernel.cu:245,252`) — defer it; the storage change alone needs no GEMM touch.

### FLAW 2 — Un-tiled full-width inputs + result all co-resident at feeder time: the TRUE single-GPU OOM (explains signal a)

**What.** `compute_f2_blocks` "does NOT tile — it uploads all M" (`cuda_backend.cu:165`). At feeder
time three things are co-resident: the resident result pair `rb.f2/rb.vpair` (2·P²·n_block, allocated
`cuda_backend.cu:414-415`), the feeder outputs `dQ/dV/dS` (4·P·M, `:410`, persist the whole loop), and
the raw inputs `dQ_raw/dV_raw/dN_raw` (3·P·M, `:428`, in an inner scope freed at `:464` *after* the
result is already allocated).

**Quantified — recomputed from the actual allocations (M=584131, n_block=757):**

| P | result (2·P²·nb) | feeder out (4·P·M) | raw in (3·P·M) | **feeder-phase peak** | sweep |
|---:|---:|---:|---:|---:|:--|
| 1024 | 12.70 | 19.14 | 14.36 | **46.20 GB** | fits |
| 1536 | 28.58 | 28.71 | 21.53 | **78.82 GB** | **OOM** |
| 2000 | 48.45 | 37.38 | 28.04 | **113.87 GB** | (OOM regardless) |

This **exactly explains the single-GPU 1024→1536 cliff**: P=1024 peaks at 46.2 GB (fits), P=1536 at
78.8 GB + cuBLAS workspace (`kCublasWorkspaceBytes`) + one transient chunk slab tips past 95.6 GB. The
user's "result 28.6 + inputs 21 ≈ 50 GB on a 95.6 GB card" is right as a naive sum but **wrong as the
peak** — the **28.7 GB of feeder outputs (as large as the result, scaling with M not the result)** is
the missing third term. The raw + feeder = **50.2 GB of transient, streamable working set** that
`architecture.md:12` ("out-of-core streaming computation, not a load-it-all kernel") and §11.1 ("scale
with the tile, never the dataset") say must be tiled. The code violates its own spec by ~50 GB.

**PARITY-safety.** SNP-axis tiling (the M5 generalization the code already calls a superset,
`cuda_backend.cu:527-529`) accumulates into the resident block tensor over `[P×T]` tiles. The GEMM
contraction is a sum over SNPs; tiling re-associates that sum → **NOT bit-identical** unless the tile
boundaries match the block boundaries (which they can, since the result is per-block anyway). Tile on
block boundaries and parity holds. Sub-flaw 2b (below) is the parity-neutral cheap part.

### FLAW 3 — The result D2H is serial, root-only, and PAGEABLE — the entire speedup shrink (explains signal b)

**What.** The whole `[P²·n_block]` result is gathered onto the root and drained in ONE D2H
(`p2p_combine.cu:176→182`) into a fresh `std::vector` that is `resize`d, never pinned
(`p2p_combine.cu:112-117`); the deliberate-pageable rationale is at `cuda_backend.cu:233-237`. Both
GPUs sit idle for the entire tail (`:176` sync, `:178` D2H, `:182` sync), after the join barrier
(`f2_blocks_multigpu_core.cpp:152`).

**Quantified.** Measured final D2H = **1720 ms for 7.14 GB = 4.15 GB/s** (`why-multigpu-slow.md:58`) —
pageable-staging speed, ~12× below a pinned PCIe5 D2H (~50 GB/s). The result is `2·P²·n_block·8`, so
the tail grows as **P²** while the parallelizable compute (~P²·M, M fixed) is split across 2 GPUs.
The D2H fraction of the multi-GPU wall climbs **54% (P=256) → 81% (P=768) → 86% (P=1024) → 88%
(P=1536)** — both the parallelism lens's Amdahl fit (serial fraction 0.55→0.88, tracking measured T2
within 5-9%) and the algorithm lens's D2H-fraction table independently land here. The peer copy is
*not* the cap (72 ms vs the 1720 ms tail at P=768, `why-multigpu-slow.md:55`). **This is the whole
1.29×→1.07× shrink, and it is NOT load imbalance or poor overlap (both refuted by the 74.1% measured
overlap, `why-multigpu-slow.md:64`).** Pinning alone: P=768 tail 1720→~143 ms; sharding the D2H so
each GPU drains its own disjoint half to its own pinned host region concurrently makes the tail itself
parallel — `(result/G)/50GB/s` — turning a serial P² tail into a per-GPU one.

**PARITY-safety.** Pinning moves the identical bytes (`cuda_backend.cu:445-446` argues this). Sharded
output D2H places the same disjoint slabs at the same offsets `slab·b0` (`p2p_combine.cu:134`). Both
**SAFE / bit-identical** — timing-only changes.

### FLAW 4 — Holding the ENTIRE tensor resident is the wrong memory model for the consumer (explains signal d, and is the meta-cause of 1-3)

**What.** The deliverable is materialized whole (`rb.f2/rb.vpair`, `cuda_backend.cu:414-415`) then
copied whole. But the spec's load-bearing premise is `f2_blocks` is "MB-scale, tiny, cacheable"
(`architecture.md:12`) — true at AT2's typical P (tens of pops: P=100 ⇒ 0.12 GB), catastrophically
false at the swept P (P=2500 ⇒ 75.7 GB) because the tensor is O(P²·n_block). The fit engine (Phase 2,
unbuilt) does leave-one-block-out jackknife: it contracts over the `n_block` axis and reads a small
*pair-subset* (the pops of the current qpAdm model, low-tens), **never needing all P²·n_block FP64
co-resident**. The architecture optimizes residency of an artifact the consumer reads
block-by-block / subset-by-subset. The spec even concedes the full-resident model doesn't scale
(`architecture.md:717`: at P=4266 the pair "≈220 GB and is itself VRAM-budgeted") yet ships it.

**Quantified.** This is the meta-cause: it forces Flaw 2's feeder peak, Flaw 3's serial P² tail, and
the OOM ceilings the sweep maps. `scaling-sweep.md:52-56` reaches the same conclusion: "the wall is the
resident full result… to go past ~2000 pops you must stop holding the entire tensor resident — i.e.
M5 out-of-core." Block-axis streaming makes device footprint `O(P²·tile_blocks)` not `O(P²·n_block)`,
which **dissolves the OOM wall entirely** and overlaps the D2H with the next chunk's GEMM.

**PARITY-safety.** Block-axis tiling is parity-neutral by construction (the result is already
per-block; emitting block-tiles changes only *when* a slab is copied out, not its bytes). SAFE.

### FLAW 5 (secondary) — The VRAM budget omits the 28.7 GB feeder-output term

**What.** `cudaMemGetInfo(&free_b)` is queried at `cuda_backend.cu:398`, **before** `dQ/dV/dS` are
allocated at `:410`. `chunk_budget_bytes` (`vram_budget.hpp:103-108`) subtracts only
`resident_tensor_bytes` + `kCublasWorkspaceBytes` from that early figure — the **4·P·M feeder outputs
are never reserved.** At P=1536 the slab budget over-commits by up to 0.80·28.7 ≈ **23 GB**.

**Cross-examination caveat (numbers win).** The data-layout lens called this "partially explains the
OOM at P=1536." The numbers say **it does NOT cause the P=1536 single-GPU OOM** — that OOM is the
*feeder-phase* peak (Flaw 2, 78.8 GB), which happens *before* the bucket loop and is independent of
the chunk budget. The budget bug is a *separate, real* latent over-commit that bites in the bucket
loop at sizes where the feeder phase survives. Genuine flaw, lower rank, correctly demoted here.

**PARITY-safety.** Query `cudaMemGetInfo` after committing `dQ/dV/dS`, or subtract `4·P·M·8`. Budget
arithmetic is byte-neutral. SAFE.

### What is SOUND (do NOT touch)

- **The 3-GEMM kernel formulation** (`f2_block_kernel.cu:327-373`): cancellation localized to the
  native-FP64 assemble, emulated-FP64 on the GEMMs — measured, well-motivated, parity-pinned (§12).
- **The device-resident combine** (`p2p_combine.cu:95-185`): disjoint *placement*, not accumulation —
  no memset, no place-add, no re-upload (the `why-multigpu-slow.md` bounce is genuinely gone; that
  doc's `:215-218/:299-304/:308-313` line cites are PRE-FIX and no longer exist). `-0.0`-faithful.
  The double-D2H regression is fixed. The combine *algorithm* is correct; its *input* is the problem.
- **Block-aligned SNP sharding** (`shard_plan.cpp:64-106`): SNP-count balance also balances the
  O(P²·s_block) GEMM FLOPs; each device owns a disjoint output region. Load balance is NOT a flaw.
- **The chunk-reuse / slab pre-sizing** (`cuda_backend.cu:471-520`): correctly kills per-chunk
  malloc/free lock contention, parity-safe.

### Cross-lens conflicts resolved by the numbers

- **Pinning priority.** algorithm-formulation ranks "pin the D2H" #1 by wall-clock; data-layout and
  system-fit rank triangular storage #1 by ceilings. **Both are right on their own axis** — pinning is
  the biggest *wall-clock* lever (50-88% of the multi-GPU wall), triangular is the biggest *ceiling*
  lever (moves P=2500 onto the card). They are orthogonal; ship both. Resolved in §5 ordering.
- **"syrk halves the GEMM."** algorithm-formulation correctly notes the GEMMs are memory-bound at
  scale and the wall is D2H-dominated, so syrk is a minor lever; data-layout/parallelism over-weight
  it. Numbers agree with algorithm-formulation: syrk is last.
- **Feeder-term budget bug as OOM cause.** Refuted as the P=1536 cause (see Flaw 5). The numbers put
  the feeder-*phase peak* (Flaw 2), not the budget bug, as the OOM driver.

---

## (3) THE IDEAL ARCHITECTURE

Concretely, the deliverable should be a **packed-symmetric, block-tile-streamed** tensor, with
single-GPU out-of-core as the primary scaling axis and multi-GPU as a throughput multiplier on top:

1. **Packed upper-triangular (incl. diagonal) storage** for `f2` and `vpair` — `P(P+1)/2 · n_block`
   each. Flat 2× cut to every resident/transient/D2H/peer byte. **Effect on ceilings:** device-resident
   root P=2500 113.6 → **56.7 GB (fits)**; single-GPU result-only ceiling roughly doubles.
2. **SNP-axis tiling within each device** (Flaw 2 / M5) on block boundaries (parity-preserving):
   the 7·P·M (50-160 GB) transient collapses to one tile + the resident accumulator. **Effect:**
   single-GPU stops OOMing for *input* reasons at P=1536/2000; the only residual wall is the result.
3. **Block-axis result streaming** (Flaw 4 / M5): produce block-tiles, spill each to host-pinned (or
   GDS) as it completes, overlapping the D2H with the next tile's GEMM. **Effect:** device footprint
   becomes `O(P²·tile_blocks)` — the OOM wall **dissolves**; P=2500 (and beyond) runs on one GPU.
4. **Pin + shard the output D2H** (Flaw 3): each GPU drains its own disjoint half to its own pinned
   host region concurrently — no root gather, two PCIe links. **Effect:** the serial P² tail becomes a
   per-GPU `(result/G)/~50 GB/s`; combined with overlap, the speedup goes from 1.07-1.29× *shrinking*
   to **near-2× and holding (or growing) with P** because BOTH compute and tail now scale with G.
5. **(Research, gated) FP32 storage of the assembled result.** The GEMMs stay emulated-FP64 and the
   cancellation-prone assemble stays native FP64; only the *already-assembled, stored* f2 could be
   FP32 — another 2×. Needs a fit-engine accuracy gate before it ships. Not a default.
6. **(Last) `cublasDsyrk` for G/Vpair** — ~25% GEMM MACs, after the memory walls are gone; has
   FP-reassociation parity risk, so it is the lowest-priority lever.

Net: triangular + SNP-tiling + block-streaming takes P=2500 from OOM-everywhere to **comfortably
single-GPU**; pinning + sharded-output-D2H turns the multi-GPU 1.07-1.29× shrink into a **near-2×**
that holds across P.

---

## (4) Was multi-GPU even the right thing to build FIRST?

**No — M5 out-of-core should have come first, and it is the prerequisite multi-GPU also needs.** The
sweep is the proof: multi-GPU's *enabling* window is just **P1536–2000** (single-GPU OOMs, 2-GPU runs,
then 2-GPU OOMs at 2500, `scaling-sweep.md:17-19`), and across the range where it works at all the
speedup is **1.29×→1.07× and shrinking** — a sub-1.3× return on a 2× resource for an O(P²·M) compute.
The "value" cells (P1536/2000) are *enabling-by-accident*: they fit on 2 GPUs only because 2 cards
hold ~191 GB of the same un-tiled full-dense footprint, not because the architecture scales. And the
ceiling it cannot break (P=2500, the actual dataset size) needs M5 *regardless* — the P2P root still
OOMs at 113.6 GB (`scaling-sweep.md:35`). So multi-GPU spent its complexity (sharding, P2P combine,
fan-out, the resident-handle seam) buying a narrow window that a triangular + block-streamed
single-GPU path would have covered *and exceeded*, on one card, with no combine at all. The honest
read: **multi-GPU is a throughput optimization that was prioritized over the correctness-of-shape
work (M5) that the same sweep proves is the real ceiling.** The fan-out and combine *code* is good
engineering; it was aimed at the wrong axis first.

That said — multi-GPU is not wasted. Once the deliverable is packed + streamed, a *sharded output
D2H* across GPUs (Flaw 3 fix) gives a genuine near-2× on the now-streamed artifact. Multi-GPU should
**ride on top of** M5, not substitute for it.

---

## (5) PRAGMATIC RECOMMENDATION — priority order

**Do, in this order:**

1. **Pin the result D2H** (Flaw 3, parity-neutral, ~1 line). Biggest *wall-clock* win for zero risk:
   reclaims 50-88% of the multi-GPU wall (P=768 tail 1720→~143 ms). Do today.
2. **Packed-triangular storage** (Flaw 1, parity-safe with a fit-side accessor). Biggest *ceiling*
   win: P=2500 root 113.6 → 56.7 GB (fits), every memory term halves. The one change that puts the
   real dataset on these cards.
3. **SNP-axis tiling on block boundaries** (Flaw 2, parity-safe if tiled on block edges). Removes the
   78.8 GB feeder-phase peak; unblocks single-GPU past P=1024 independent of result size.
4. **Block-axis result streaming to pinned host / GDS** (Flaw 4 = M5). Dissolves the OOM wall;
   parity-neutral. This is the strategic centerpiece — schedule it as the next milestone, ahead of
   any further multi-GPU perf work.
5. **Fix the feeder-output budget reservation** (Flaw 5, parity-neutral). Cheap correctness fix for
   the latent 23 GB over-commit.

**Then, on top of the streamed artifact:** sharded + pinned output D2H across GPUs (the near-2×).

**Leave alone:** the 3-GEMM formulation, the device-resident combine algorithm, the block-aligned
shard plan, the chunk-reuse mechanics. Defer `cublasDsyrk` (parity risk, minor MAC win) and FP32
storage (needs a fit-engine accuracy gate) until the memory walls are down.

---

## Key evidence anchors

`include/steppe/fstats.hpp:40-60` (symmetric, full-dense, off-diag-only consumer) ·
`src/device/cuda/cuda_backend.cu:165` (no SNP tiling), `:398` (memGetInfo before feeder alloc),
`:410,414-415,428,464` (feeder/result/raw co-residency + free ordering), `:233-237` (deliberate
pageable D2H), `:471-520` (chunk-reuse) · `src/device/cuda/p2p_combine.cu:104-117` (root full-result
alloc + pageable host vectors), `:134` (disjoint slice), `:143-167` (D2D/peer placement, no
accumulate), `:176-182` (one serial final D2H) · `src/device/vram_budget.hpp:62-68` (full-pair
reserve), `:103-108` (feeder term omitted) · `src/device/shard_plan.cpp:64-106` (SNP-balanced
block-aligned) · `src/core/fstats/f2_blocks_multigpu_core.cpp:144,152` (fan-out + join barrier) ·
`src/device/cuda/f2_block_kernel.cu:158,174,327-373` (full-P² write, 3-GEMM) · `docs/architecture.md:12`
(out-of-core mandate), `:717` (full-resident concession) · `docs/cleanup/m4.5/scaling-sweep.md:17-19`
(OOM ceilings), `:35` (113 GB root), `:44-56` (shrink + M5) · `docs/cleanup/m4.5/why-multigpu-slow.md:58,64`
(1720 ms / 4.15 GB/s pageable D2H, 74.1% overlap — note this doc's p2p line cites are PRE-FIX).

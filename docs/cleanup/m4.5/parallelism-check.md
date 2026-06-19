# M4.5 multi-GPU: is it TRUE parallel, or built wrong?

> **RESOLVED / OUTCOME (post-M5).** The diagnosis here held — the precompute wall was HOST-bound,
> not GPU-bound — and the fix was NOT "make multi-GPU truly parallel" but **get the result off the
> CPU**. §2/§5 found the ~570 ms serial tail was per-call D2H host pin/unpin of the ~3 GB result
> slice, and §3 framed the deeper problem ("multi-GPU-vs-D2H is the wrong axis; the wall is host
> work, not GPU time"). Both were answered by **device-resident output (commit `1f80c0c`)**: the
> precompute returns a VRAM-resident handle, the host copy becomes opt-in, and the per-call pin tail
> is simply not executed in the in-VRAM case. **Measured P=512 ~673 ms resident vs ~2879 ms
> bulk-to-host = ~4.3×.** Scale beyond VRAM is handled by **M5 streaming** (`176a07d` tiered output +
> `c65179f` SNP-tile input streaming): footprint O(P·tile + P²), P=2500 full-autosome completes on
> one 32 GB 5090 in ~51.5 s, parity bit-identical. **The honest takeaway:** multi-GPU per se was NOT
> the precompute speedup — getting off the CPU was. Multi-GPU's real home is the embarrassingly
> parallel FIT / ROTATION phase (Phase 2, unbuilt), not the precompute. The analysis below is
> preserved as the record of how the host-bound nature was proven.

Box: box5090 (2x RTX 5090 sm_120, 32 GB ea, CONSUMER no-P2P, CUDA 13.0.88), live.
Method: code read (read-only) reconciled against an nsys CUDA/NVTX/OSRT timeline at P=512
(`bench_f2_multigpu /workspace/data/aadr 512`), SQLite export of CUPTI KERNEL+MEMCPY+RUNTIME.
**Where the code read and the measurement disagree, the MEASUREMENT wins.** All numbers below
are from the nsys ground-truth blob; all structural claims are tagged to file:line.

---

## 1. THE VERDICT — blunt

**The two GPUs DO run concurrently. The work IS split ~50/50. But it is NOT true parallel
multi-GPU.** Only **22.7% of the GPU-union wall overlaps**; ~63% of the wall is a **serial
D2H tail** in which the two devices run one-after-another. A 2-GPU run that should cost ~50% of
the 1-GPU wall instead costs **~1.8x the 1-GPU GPU-union** (G1 510 ms vs G2 909 ms), i.e. the
second GPU makes it *slower*, not faster.

| metric (P=512, median over 5 steady G2 calls) | value | source |
|---|---|---|
| G1 GPU-union wall | 510 ms | nsys |
| G2 GPU-union wall | 909 ms | nsys |
| dev0 busy / dev1 busy | 305.6 / 323.7 ms | nsys |
| work split ratio | 0.94 : 1 (≈50/50) | nsys |
| compute+H2D overlap | 205.9 ms = **22.7% of union** | nsys |
| serial portion | ~703 ms (≈ 77% of union) | nsys |

- **Concurrent?** YES at the thread + compute level. Two distinct worker tids, one launching
  kernels only on dev0, one only on dev1; during the compute+H2D phase the kernel windows
  genuinely overlap (~206 ms). The fan-out and shard plan are correct.
- **Split 50/50?** YES. dev0 305.6 ms vs dev1 323.7 ms busy = 0.94:1. The shard plan is NOT
  the problem.
- **True parallel overall?** **NO.** 22.7% overlap is not parallelism; it is two GPUs taking
  turns. The hypothesis ("partly sequential / built at the wrong place") is **CONFIRMED** — but
  the bottleneck is NOT where the code-read construction blamed it (see §4).

---

## 2. THE EXACT SERIAL BOTTLENECK (measured ms + file:line)

The G2 wall splits into two phases. Phase 1 is healthy; Phase 2 is the bug.

**PHASE 1 — compute + H2D (+0 .. +333 ms): GENUINELY PARALLEL.**
Both devices upload (~131 ms/dev H2D) and run their GEMMs concurrently; overlap here is
205.9 ms. dev0 kernel window [+135.6..+328.7], dev1 [+132.5..+333.5]. The non-blocking
per-device stream (`cuda_backend.cu:844`, `Stream stream_{}` = `cudaStreamNonBlocking`) and the
amortized H2D pin cache (`pinned_buffer.cuh:246` `PinnedRegistryCache`, used at
`cuda_backend.cu:493-495`) are real and working: the two H2Ds run as concurrent pinned DMAs,
not contending pageable copies. **This phase is the multi-GPU done right.**

**PHASE 2 — D2H tail (+333 .. +904 ms = ~570 ms ≈ 63% of the wall, ZERO device overlap): THE BUG.**
The actual D2H byte-copies are cheap and could overlap (dev0 50.2 ms + dev1 64.2 ms). They do
NOT overlap — dev0 D2H window [+541..+592], dev1 D2H window [+840..+904], separated by ~248 ms
of idle GPU. The tail is host-side and serialized **on the driver-wide lock across the two
worker threads**, dominated by per-call host-page pin/unpin of the D2H destination slices:

- **ROOT CAUSE — per-call `cudaHostRegister` / `cudaHostUnregister` on the D2H destination,
  with NO amortization:** `compute_f2_blocks_into` pins each device's output slice every call
  via two `RegisteredHostRegion` objects:
  - `src/device/cuda/cuda_backend.cu:352-353` — `RegisteredHostRegion pin_f2(f2_slice, bytes);`
    / `pin_vp(vpair_slice, bytes);`
  - which call `cudaHostRegister` at `src/device/cuda/pinned_buffer.cuh:164` and
    `cudaHostUnregister` at `pinned_buffer.cuh:201` (the dtor, at scope exit of `:359-360`).
  - Measured per steady G2 iter (CUPTI_RUNTIME, two worker tids): `cudaHostRegister`
    282.6 + 189.3 ms, `cudaHostUnregister` 218.1 ms. Concretely in the tail: dev0
    `cudaHostRegister` 97.6 + 91.8 ms then `cudaHostUnregister` 104.7 + 113.4 ms (~407 ms),
    dev1 `cudaHostRegister` 169.6 + 113.0 ms (~282 ms). dev1's D2H **cannot even start until
    +839 ms** because dev0 is holding the device-wide lock pinning ~3 GB of result slice.
  - This is registering a multi-GB range *every call* — `pinned_buffer.cuh:216` itself measures
    that at ~50–360 ms — and unlike the H2D inputs there is **no cache**: the comment at
    `pinned_buffer.cuh:236-240` and `cuda_backend.cu:232-237` explicitly says the D2H result is
    a fresh `std::vector` base pointer each call so caching "never hits." So the design pins it
    raw each call. That raw per-call register/unregister is the serializer.

- **SECONDARY — per-call `cudaMalloc`/`cudaFree`:** ~270 ms across the compute phase
  (`cudaFree` 198.5 + 65.4 ms in the window). The remaining per-call device buffers in
  `run_f2_blocks_resident` (`cuda_backend.cu:455` dQ/dV/dS, `:459-462` rb.f2/rb.vpair/dOffsets/
  dSizes, `:473` raw inputs, `:562-565` the reused slabs) allocate/free via
  `device_buffer.cuh:74,117` (`cudaMalloc`/`cudaFree`, device-wide-synchronizing, global lock).
  These **mostly hide** behind Phase-1 compute and are NOT dominant — most of them overlap.

**Why the bottleneck is the D2H pin, NOT what the code-read claimed.** The construction read
fingered per-call `cudaMalloc`/`cudaFree`/`cudaMemGetInfo` (its findings #1/#2) as "the dominant
cross-device serializer." The timeline does not support that: the per-chunk slab alloc churn was
already fixed (slabs are pre-sized ONCE outside the loop, `cuda_backend.cu:516-565`, "L4b"), so
only a fixed handful of allocs remain and they overlap inside Phase 1. The ~570 ms serial tail is
register/unregister of the result, not malloc. **MEASUREMENT WINS: the dominant serializer is
the un-amortized D2H `cudaHostRegister`/`cudaHostUnregister` at `cuda_backend.cu:352-353`.**

- **NOT the cause (cleared):** work imbalance (0.94:1, balanced); the jthread fan-out
  (`f2_blocks_multigpu_core.cpp:117-151` launches all G workers before any join — genuinely
  concurrent); cuBLAS handle (per-backend, created once, `cuda_backend.cu:845`); stream
  (per-device non-blocking, `:844`); shared mutable state (each worker writes a disjoint slab,
  `f2_blocks_multigpu_core.cpp:258-261`); PCIe bandwidth (the real copies are only 50+64 ms);
  the shard plan (`shard_plan.cpp:52-106`, SNP-balanced greedy block-aligned).

---

## 3. IS THE PARALLELISM IN THE WRONG PLACE? — what the construction SHOULD be

**Partly wrong place.** The compute fan-out is in the right place and works (Phase 1 overlaps).
The mistake is structural in the D2H stage: parallelism is bolted onto a per-call pin/unpin that
*requires* the driver-wide lock, so two threads cannot pin concurrently — the "parallel" D2H is
forced serial by its own host-side setup. The corrected construction:

1. **Amortize the D2H destination pin OUT of the hot path** (the direct analogue of the H2D
   `PinnedRegistryCache` win). Either (a) D2H into a **persistently-pinned staging buffer** owned
   per backend (allocate once with `cudaHostAlloc`, reuse every call — `PinnedBuffer` at
   `pinned_buffer.cuh:59` already exists for exactly this), then a cheap host copy into the
   caller's vector; or (b) have the orchestrator pre-allocate the result as ONE persistently-pinned
   region (registered once, not per call) and pass stable slices so the per-call
   `RegisteredHostRegion` at `cuda_backend.cu:352-353` is deleted. Goal: the two devices'
   D2Hs run as concurrent DMAs (they are only 50+64 ms) instead of serializing 570 ms on the lock.

2. **Move the remaining per-call `cudaMalloc`/`cudaFree` off the hot path too** — pre-allocate
   the run-long device tensors (`cuda_backend.cu:455,459-462`) once per backend and reuse, same
   as the slab fix at `:516-565` already did for the chunk loop. This recovers the ~270 ms that
   currently only *mostly* hides.

3. **The deeper framing problem (per `why-d2h`): multi-GPU-vs-D2H is the wrong axis here.** The
   end-to-end bench median (~2653 ms) is dominated by per-call HOST work — repack of 768×584k +
   ~3.18 GB result alloc + combine, with inter-call GPU-idle gaps of 1700–2140 ms in the trace —
   NOT GPU time. Splitting the GPU work across two GPUs cannot speed up a workload whose wall is
   host-bound; the GPU-union (510 vs 909 ms) is the only honest parallelism metric, and even that
   regresses. Fixing the D2H pin recovers parallelism *within* the GPU-union; making multi-GPU
   actually pay off end-to-end additionally requires overlapping/eliminating the host repack+alloc
   that bracket each call.

---

## 4. RECONCILE: why-multigpu-slow.md (rtxbox, 74% overlap) vs box5090 (22.7%)

They do not contradict — **different boxes, different combine path, and the rtxbox number is the
COMPUTE-PHASE overlap, not the whole wall.**

- **rtxbox = PRO 6000, P2P-CAPABLE.** `can_access_peer == true`, so the §4 gate
  (`f2_blocks_multigpu.cpp:155-157`) selects the **device-resident P2P** path
  (`compute_multigpu_partials_resident` -> `combine_f2_partials_resident`). That path leaves
  partials resident, does ONE final D2H after a peer pull — so it **never executes the per-call
  `cudaHostRegister`/`cudaHostUnregister` on a 3 GB result slice on both threads.** The 74%
  figure is the H2D+compute overlap, which the 5090 *also* achieves in Phase 1 (~206 ms is
  comparable as a fraction of the compute phase).
- **box5090 = consumer, NO P2P.** `can_access_peer == false`, so the gate **forces the
  host-staged path** (`f2_blocks_multigpu.cpp:192-216`, `compute_f2_blocks_into`). That path is
  the ONLY one that pins the D2H destination per call (`cuda_backend.cu:352-353`). So the serial
  tail is path-specific to the host-staged combine, which box5090 is forced onto.
- Hence: rtxbox's 74% (P2P, no D2H-pin tail) and box5090's 22.7% (host-staged, D2H-pin tail) are
  measuring the same compute concurrency plus a tail the rtxbox path structurally avoids. The
  74% does NOT hold on the 5090, exactly as the prompt suspected — because the 5090 cannot take
  the P2P path.
- Note: the prompt's pathological older numbers (G2host 1.41x slower, P=512 3733 ms) are from an
  older build; the live box5090 now measures G2host/G1 ≈ 0.98–1.0x end-to-end (flat, no speedup)
  and the 1.8x GPU-union regression above. Still not parallel; just no longer catastrophic.

**Bench-attribution caveat (load-bearing):** `compute_f2_blocks_multigpu`
(`f2_blocks_multigpu.cpp`) never writes `resources.last_multigpu_timings` — there is no writer in
the orchestrator. Any prior claim that blamed "the combine" or "the D2H tail" using the bench's
printed `compute_wall_ms`/`combine_wall_ms`/byte phase numbers is **unsupported by data** (the
bench prints zeros). The nsys timeline above is the real attribution.

---

## 5. HONEST BOTTOM LINE

Multi-GPU was set up CORRECTLY in three of four places: the jthread fan-out is genuinely
concurrent (`f2_blocks_multigpu_core.cpp:117-151`), the work is split ~50/50
(`shard_plan.cpp:52-106`, measured 0.94:1), per-device backend/handle/stream/workspace are built
ONCE and amortized (`resources.cpp:123-129`, `cuda_backend.cu:98-112`), and the compute+H2D phase
genuinely overlaps (22.7% of union = ~206 ms concurrent). **It was set up WRONG in the D2H stage:**
the host-staged path pins/unpins each device's ~3 GB result slice on EVERY call
(`cuda_backend.cu:352-353` -> `pinned_buffer.cuh:164,201`) with no amortization, and those
register/unregister calls serialize on the device-wide driver lock across the two worker threads,
producing a ~570 ms serial tail (63% of the wall) in which the GPUs take turns. That is why two
GPUs give ~1.0x, and on the GPU-union ~1.8x slower than one. **Corrected construction:** amortize
the D2H destination pin (persistent pinned staging buffer or a once-registered result region, like
the H2D `PinnedRegistryCache` already does for inputs), move the remaining per-call device allocs
out of the hot path, and overlap/eliminate the host repack+alloc that brackets each call — then
the host-staged 2-GPU path can approach the ~0.5x the P2P path nearly reaches via its single final
D2H.

---

### EXECUTIVE SUMMARY (verbatim answer)

The M4.5 multi-GPU is NOT true parallel: measured GPU-union overlap is only 22.7% (G1 510 ms vs
G2 909 ms — two GPUs run ~1.8x SLOWER on the union than one). The two GPUs DO run concurrently and
the work IS split ~50/50 (dev0 305.6 ms vs dev1 323.7 ms = 0.94:1, balanced), and the compute+H2D
phase genuinely overlaps (~206 ms). The defect is a ~570 ms SERIAL D2H tail (63% of the wall, zero
device overlap) — NOT PCIe and NOT the D2H bytes (only 50+64 ms of real copy). Root cause:
`compute_f2_blocks_into` pins each device's ~3 GB result slice EVERY call via `RegisteredHostRegion`
(`cuda_backend.cu:352-353` -> `cudaHostRegister`/`cudaHostUnregister` at `pinned_buffer.cuh:164,201`)
with NO amortization, and those calls serialize on the device-wide driver lock across the two worker
threads (dev0 holds the lock ~407 ms, so dev1's D2H can't start until +839 ms). The code-read blamed
per-call `cudaMalloc`/`cudaFree` — the timeline says no: the slab churn was already fixed
(`cuda_backend.cu:516-565`) and the remaining allocs mostly hide in Phase 1; the dominant serializer
is the D2H pin. The 74%-overlap from why-multigpu-slow.md is the rtxbox P2P path (device-resident,
ONE final D2H, never pins the result per call); box5090 has no P2P so the gate forces host-staged
(`f2_blocks_multigpu.cpp:155-216`), which is the only path that pays the per-call pin tail — so 74%
does NOT hold here. Bench phase numbers are dead (`last_multigpu_timings` is never written), so the
nsys timeline is the only valid attribution. Corrected construction: amortize the D2H destination pin
(persistent pinned staging or a once-registered result region, like the H2D `PinnedRegistryCache`),
move remaining per-call device allocs out of the hot path, and overlap the host repack+alloc that
brackets each call.

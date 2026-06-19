# Why M4.5 multi-GPU was slower than single-GPU — the definitive answer (RESOLVED)

> **STATUS: RESOLVED (@867a4bf).** The slowdown described below is the **PRE-FIX** state and is now
> historical. The data-bounce wart this doc diagnoses (the redundant second full 7.14 GB D2H) was
> eliminated by the **device-resident combine** (commit `867a4bf`; the nsys root-cause that drove it is
> documented here and at commit `165f655`). After the fix, **multi-GPU is FASTER than single-GPU**:
> measured on rtxbox (2× RTX PRO 6000 Blackwell sm_120, Release, EmulatedFp64{40}, median of 10) at
> **P=768 G1 = 2342 ms, G2 = 2125 ms = 1.10×**, and **P=400 = 1.22×**, with bit-identical (`memcmp`)
> parity preserved on both the host-staged and the device-resident P2P combine paths, both datasets
> (`derived_acc` P=50, `derived_full` P=768). The "Item 1 (THE cure)" design in §4 is the fix that
> landed. The full diagnosis is kept below as the record of how this was found and fixed.
>
> *Remaining optional lever (not a blocker):* pin the final pageable result D2H — the next available
> speedup, could push past 1.10×.

Branch: `m4.5-multigpu`. Box: rtxbox (2× RTX PRO 6000 Blackwell sm_120, CUDA 13, 96 GB, real P2P).
Bench: `bench_f2_multigpu` @ P=768, M=584,131, n_block=757. Result is bit-identical between G1 and G2
(parity proven). **[PRE-FIX] G2 was 0.70×** — measured G1 = 2429.5 ms, G2 = 3446.5 ms (**+1017 ms**).
This is the state the diagnosis below explains; it was cured by the device-resident combine (`867a4bf`),
after which G2 is **1.10× FASTER** (P=768: G1 2342 ms, G2 2125 ms).

Method of record: nsys (Nsight Systems 2025.3.2) GPU-timeline trace, per-op start/end/device/bytes pulled
from the `.sqlite` (CUPTI_ACTIVITY_KIND_KERNEL / MEMCPY / MEMSET). GPU-union wall matches the bench's
host-synchronous wall within ~30 ms (G1-i2 union 2395 ms vs bench 2429 ms; G2-i1 union 3363 ms vs bench 3446 ms).
**The measured buckets are ground truth. Where a reading-phase hypothesis disagrees with a bucket, the bucket wins.**

---

## (1) THE HEADLINE — one sentence, one bucket

**The +1017 ms is a SECOND full-tensor Device→Host copy: G2 does TWO 7.14 GB D2H passes (a per-device
partial D2H to host + a final full-accumulator D2H) where G1 does ONE — measured D2H is 1714 ms in G1 vs
3565 ms in G2 (Δ ≈ +1851 ms of D2H, of which the combine's final full D2H alone is 1720 ms), and that extra
copy exists only because `compute_f2_blocks` D2H-copies each device's partial to host and frees its device
buffers (`cuda_backend.cu:483-489`), forcing `combine_f2_partials_p2p` to re-materialize and re-copy what
was already resident (`p2p_combine.cu:18-25` admits the wart, `:308-313` does the second D2H).**

This is the data-bounce wart the user named. It is an architectural choice, not memory-bound physics.

---

## (2) Measured per-phase G1-vs-G2 table (ground truth, ms)

All numbers are nsys GPU-active durations per device per kind, summed per run; "wall" = GPU-union span.

| Phase                                   | G1 (1 GPU) | G2 (2 GPU)                          | Notes |
|-----------------------------------------|-----------:|-------------------------------------|-------|
| H2D inputs (Q/V/N)                       | 187        | 188 (split across 2 dev)            | pinned; ~10.77 GB G1 / ~halved per dev G2 |
| feeder                                   | 22         | 11 each (concurrent)                | overlaps |
| GEMM                                     | 245        | 119 (dev0) + 128 (dev1), concurrent | halved + overlapped — G2 WINS here |
| gather                                   | 24         | overlapped                          | |
| assemble                                 | 14         | overlapped                          | |
| **D2H partial → host** (compute phase)   | —          | **dev0 770 + dev1 977 = 1747**      | `cuda_backend.cu:483-489` — G2-ONLY first D2H |
| combine: H2D re-upload                    | —          | 184 (`p2p_combine.cu:215-218,254-257`) | G2-ONLY |
| combine: cudaMemcpyPeer dev1→dev0         | —          | 72 (6 ops, 2.005 GB ea, 55.6 GB/s)  | G2-ONLY, **cheap** |
| combine: place_add kernel                 | —          | 14 (`:299-304`)                     | G2-ONLY, cheap |
| combine: acc-zero cudaMemset              | —          | 5 (`:152-155`)                      | G2-ONLY, cheap |
| **D2H final full accumulator**            | **1714**   | **1720** (`p2p_combine.cu:308-313`) | G1's ONLY D2H; G2's SECOND D2H |
| **D2H total**                             | **1714**   | **3565** (14.29 GB vs 7.14 GB)      | **the regression lives here** |
| compute-phase wall (both devices)         | —          | 2023                                | |
| combine-phase wall (dev0-only)            | 0          | 1595                                | the serial tail |
| **wall (GPU union)**                      | **2395**   | **3363**                            | bench: 2429 vs 3446 |

**Stream overlap: 74.1%** (G2-i1: dev0_busy 2954 ms + dev1_busy 1380 ms, union 3311 ms ⇒ 1023 ms concurrent
= 74.1% of dev1 busy; the two feeders start within 0.1 µs of each other, dev0 t=14.1337 / dev1 t=14.1336).

Bus traffic per run (measured): G1 — D2H 7.14 GB / H2D 10.77 GB / PtP 0 / memset 0. G2 — D2H **14.29 GB**
/ H2D 17.91 GB / PtP 4.01 GB / memset 7.14 GB.

### Hypotheses the data REFUTES

- **REFUTED — "~18% fan-out overlap / cudaMalloc-lock serialization is the bug."** At HEAD the P2/F1
  non-blocking per-device stream fix has landed (`cuda_backend.cu` owns a `cudaStreamNonBlocking` stream).
  Measured overlap is **74.1%**, not 18%. The two devices' GEMM+gather+assemble run concurrently in the
  same wall window. **Overlap is not the problem.** The cudaMalloc/cudaFree lever is at most a few percent
  and is dwarfed by the D2H buckets.
- **REFUTED — "the redundant host zeroing (P1 / `assign(total,0.0)`) is the dominant ~1440 ms cost."** The
  acc-zero `cudaMemset` measures **5 ms** (`p2p_combine.cu:152-155`); the host `out.f2/.vpair.assign(0.0)`
  first-touch (`:180-181`, `cuda_backend.cu:255-256`) is real but is overlapped/relocated into the D2H write
  — the prior on-box refutation already showed removing the assign just moves the same page-fault cost into
  the D2H copy. It is **not** a separate +1440 ms bucket. The reading-phase "P1 is the #1 lever" story is
  wrong against the trace; P1 is a cleanup, not the cure.
- **REFUTED — "the bench is unfair / not apples-to-apples."** The warm-up run (`bench_f2_multigpu.cu:59`)
  correctly amortizes cuBLAS create, the 64 MB workspace, `cudaHostRegister` pinning, and
  `cudaDeviceEnablePeerAccess`; `Resources` are built outside `time_run`. The combine is (correctly) inside
  the timed region (`:62-64`), which is exactly what exposes the wart. The 0.70× is steady-state and honest.
  (Two real bench *defects* — `ITERS=2` + min-of-2, and no phase instrumentation — make the number noisy and
  un-attributable, but they do not manufacture the regression. See §4 item 4.)

### Hypotheses the data CONFIRMS

- **CONFIRMED — the triple host-bounce.** Trace shows, per G2 run: per-device partial D2H to host
  (dev0 1.567 GB×2 = 386+384 ms, dev1 2.005 GB×2 = 488+489 ms = 1747 ms), then H2D re-upload (dev0 root own
  partial `:215-218`, dev1 peer pre-stage `:254-257` = 184 ms), then cudaMemcpyPeer dev1→dev0 (72 ms), then
  place-add (14 ms), then the **second** full D2H of the accumulator (dev0 3.572 GB×2 = 859+861 ms = 1720 ms,
  `:308-313`). G1 does the single 3.572 GB×2 = 1710 ms D2H and stops.

---

## (3) What was architected wrong — the diagnosis, cited to file:line

The seam is the **per-device-instance return type**: `ComputeBackend::compute_f2_blocks` returns a HOST
`F2BlockTensor` whose `f2`/`vpair` are `std::vector<double>` (`include/steppe/fstats.hpp:52,60`). So the
per-device worker, at the end of its GEMM, has its partial **resident on the device** in `dF2_all`/`dVpair_all`
— and then throws that residency away:

- `cuda_backend.cu:483-489` — `cudaMemcpyAsync(... DeviceToHost)` of the full partial into a freshly-allocated
  pageable host vector, then `cudaStreamSynchronize`, then `return out;` — at which point the RAII
  `DeviceBuffer` dtors **free** `dF2_all`/`dVpair_all`. The device copy is gone.
- `p2p_combine.cu:18-25` — the code comment **admits** this: "The current per-device compute_f2_blocks
  returns a HOST partial (it frees its device buffers before returning), so this routine stages each partial
  onto its OWNING device first (a byte-exact H2D upload) ... a device-resident per-device compute that skips
  it ... is OUT OF SCOPE for M4.5." **That punt is the architectural flaw.**
- Consequence in the combine (all measured G2-only): re-upload H2D (`:215-218` root, `:251-257` peer),
  cudaMemcpyPeer (`:264-267`), place-add into a zeroed full accumulator (`:152-155` memset, `:299-304` add),
  and the **second** full D2H (`:308-313`).

So the "device-resident P2P combine" is device-resident in name only: it first **de-residents** every partial
to host and then **re-residents** it. The combine is also strictly serial after the join barrier
(`f2_blocks_multigpu_core.cpp:151`) and runs dev0-only by design (combine-phase wall 1595 ms, all on dev0),
so the 74% compute overlap is followed by a 1595 ms serial tail dominated by the second D2H.

Two structural facts make the fix clean and the current design wasteful:

1. **The shard is block-aligned and DISJOINT** — each global slab is written by exactly one device
   (`p2p_combine.cu:35-41`, the parity argument). The combine is therefore a **PLACEMENT, not a sum**: the
   `cudaMemset` zero-init, the full accumulator `dAcc`, and the `place_add_f2_kernel` (a `+=` onto a known
   0.0) are all solving a problem — overlapping contributions — that disjoint sharding guarantees does not
   exist.
2. The partials are **already resident** at `cuda_backend.cu:326` (`dF2_all`/`dVpair_all`) at the moment they
   are copied out and freed.

Net: G2 moves **14.29 GB D2H** for a result G1 delivers with **7.14 GB D2H**. That doubled D2H is the regression.

---

## (4) The corrected design — ordered, parity-safe work items

Each item lists the seam, the expected ms recovered (from the measured buckets), and the §12 bit-identity risk.

### Item 1 (THE cure) — device-resident per-device compute; D2H the result ONCE
**What:** Add a backend path that computes the partial and **leaves `dF2_all`/`dVpair_all` resident** instead
of D2H-ing to host and freeing. The combine then pulls each peer's partial **straight from its resident
device buffer** via `cudaMemcpyPeer` (root's own partial needs no copy) directly into one device result
buffer at its disjoint block offset `slab·b0`, then does the **ONE** final D2H.

**Seam:** a new virtual on `ComputeBackend` (e.g. `compute_f2_blocks_resident(...)` returning a device handle —
a `DeviceBuffer<double>` pair + shape), consumed by the combine fork at `f2_blocks_multigpu.cpp:184-188`.
Delete the D2H + free for this path (`cuda_backend.cu:483-489` + keep the buffers alive past return).
Rewrite `combine_f2_partials_p2p` (`p2p_combine.cu:190-313`) to consume device pointers: drop the re-upload
H2D (`:215-218,251-257`), and — because the shard is disjoint — drop `dAcc`+`cudaMemset` (`:150-155`),
`dStage` (`:167-170`), and `place_add_f2_kernel` (`:299-304`) entirely; each device's resident slab-stack is
`cudaMemcpyPeer`'d (peer) or D2D-copied (root) straight into the disjoint slice `[slab·b0, slab·b1)` of one
device result, then ONE D2H.

**Expected payoff:** removes the **1747 ms** partial-D2H bucket and the **184 ms** re-upload H2D; leaves the
combine at 72 ms PtP + ~14 ms placement + **one** 1720 ms final D2H. G2 then does the SAME single D2H as G1
plus a cheap 72 ms peer hop, and the 74% compute overlap finally shows as a real win. Projected G2 wall:
~2023 ms compute-wall + ~72 ms PtP + ~1720 ms single D2H − (the partial-D2H that was inside compute-phase) →
on the order of **~1.8–2.0 s**, i.e. **G2 beats G1 (~2.4 s)**, roughly 1.2–1.3× and scaling toward `max_g` as G grows.

> **LANDED @867a4bf — projection CONFIRMED.** This is the fix that shipped: per-device compute leaves its
> partial RESIDENT (no D2H / no free, returns an opaque move-only `DevicePartial` handle); the combine
> allocates ONE root result, D2D-copies the root partial and `cudaMemcpyPeer`s each peer partial straight
> into its DISJOINT block slice, then does ONE final D2H. Deleted: the re-upload H2D, the accumulator
> `cudaMemset`, the place-add kernel, the staging buffers, and the per-peer `cudaDeviceSynchronize`
> (now one fence before the final D2H). Measured result (rtxbox, Release, EmuFp64{40}, median of 10):
> **P=768 G2 = 2125 ms vs G1 = 2342 ms = 1.10×; P=400 = 1.22×** — bit-identical parity held. The
> ~1.8–2.0 s projection was correct.

**§12 bit-identity:** SAFE and memcmp-identical. The bytes are unchanged — a resident-to-resident DMA moves
the exact same doubles the host bounce moved (`p2p_combine.cu:31-41` argument holds verbatim). Disjoint
placement means each slab is written once by its owner at the same offset the host/p2p combine used
(`f2_combine.cpp:103-104`); dropping the `+= 0.0` onto a zeroed accumulator is exact (x written directly ==
x + 0.0). Keep the fixed g=0..G-1 order for `block_sizes` placement. No recompute, so no FP reassociation.

### Item 2 — kill the per-peer `cudaDeviceSynchronize` in the combine loop
**What:** Replace the full-device sync fired once per peer partial (`p2p_combine.cu:283`) with an `Event`
fence (or a single `cudaStreamSynchronize` before the final D2H). It exists only to keep `dPeer_*` alive past
the cross-device DMA; under Item 1 the peer buffers stay resident (nothing to free mid-loop), so the fence
collapses to one sync before the final D2H.

**Seam:** `p2p_combine.cu:283` (and the per-peer staging lifetime). The `Event` wrapper already exists.

**Expected payoff:** small at G=2 (one extra drain, the peer copies are already only 72 ms) but it removes a
hard serialization point that grows with G; it lets peer g+1's copy pipeline behind peer g's placement.

**§12 bit-identity:** SAFE. A fence change alters timing only, not bytes or order.

### Item 3 — persistent per-device buffers / `cudaMallocAsync` pool (deferred — not the bug, but clean it)
**What:** Reuse the large resident `DeviceBuffer`s across `compute_f2_blocks` calls (or use a per-device
`cudaMallocAsync` pool with `cudaMemPoolAttrReleaseThreshold=MAX`) so the per-call `cudaMalloc`/`cudaFree`
(`device_buffer.cuh` raw alloc/free) stop taking the driver global allocator lock that the two worker threads
contend on.

**Seam:** `CudaBackend` member buffers reused across calls (`cuda_backend.cu:323-328`).

**Expected payoff:** marginal at HEAD — measured overlap is already 74% and the allocator lock is no longer
the serializer (the ~18% lead was REFUTED). Do it for cleanliness/headroom, not for the +1017 ms. Low priority.

**§12 bit-identity:** SAFE. Allocation strategy does not touch the computed bytes.

### Item 4 — make the bench trustworthy (do alongside, so the post-fix number is honest)
**What:** (a) raise `ITERS` from 2 (`bench_f2_multigpu.cu:112`) to ≥10 and report **median + p10/p90**
instead of min-of-2 (`:67,:69`) — the G2 path has structurally higher per-iter variance, so min-of-2 hides
the steady state. (b) Phase-instrument `time_run` (`:55-74`) — break out {compute-wall, partial-D2H,
re-upload, peer, final-D2H} via an out-of-band debug timing struct on `Resources` (mirroring
`last_combine_path`), never on the numeric `F2BlockTensor`. This is what would have killed the wrong "1.4 s
zeroing" hypothesis in one run. (c) Optionally print measured H2D/D2H byte totals per G so the table
self-documents the 2× bus traffic.

**Seam:** `tests/reference/bench_f2_multigpu.cu` only; a debug-only field on `device::Resources`.

**Expected payoff:** none on the kernel — makes the regression attributable and the post-fix speedup robust.

**§12 bit-identity:** N/A (bench harness; the numeric result is untouched — keep timing out-of-band).

---

## (5) §12 bit-identity risk summary

| Item | Touches bytes? | Touches order? | Risk | Mitigation |
|------|----------------|----------------|------|------------|
| 1 — device-resident combine | No (DMA moves same doubles) | No (disjoint placement, fixed g-order kept) | **None** | Resident-to-resident DMA is byte-exact (`p2p_combine.cu:31-41`); placement offset `slab·b0` identical to host baseline (`f2_combine.cpp:103-104`); drop of `+=0.0` is exact |
| 2 — fence change | No | No | None | Event/single-sync alters timing only |
| 3 — buffer pool | No | No | None | Allocation strategy is byte-neutral |
| 4 — bench | N/A | N/A | None | Timing out-of-band, numeric result untouched |

All four are parity-neutral: the only operations that change bytes are the GEMM/feeder/assemble kernels, and
none of the items touch them. The combine remains a byte-exact placement in fixed g=0..G-1 order. Verify with
the existing memcmp parity test after Item 1.

---

## Key file:line anchors

- `cuda_backend.cu:255-256` — host result `assign(total,0.0)` (overlapped, NOT the +1017 ms; REFUTED as dominant).
- `cuda_backend.cu:326` — partials resident here (`dF2_all`/`dVpair_all`).
- `cuda_backend.cu:483-489` — the per-device partial **D2H to host + stream-sync** (then RAII free at `return`). FLAW origin.
- `p2p_combine.cu:18-25` — the comment that admits the wart and punts the fix as "OUT OF SCOPE for M4.5".
- `p2p_combine.cu:35-41` — disjoint block-aligned shard ⇒ combine is a placement, not a sum (parity argument).
- `p2p_combine.cu:150-155` — full accumulator + `cudaMemset` (measured 5 ms; removable under Item 1).
- `p2p_combine.cu:167-170` — staging buffers (removable under Item 1).
- `p2p_combine.cu:215-218` — root own-partial H2D re-upload (combine_h2d_reupload, 184 ms G2-only).
- `p2p_combine.cu:251-267` — peer pre-stage H2D + `cudaMemcpyPeer` (72 ms PtP).
- `p2p_combine.cu:283` — per-peer full `cudaDeviceSynchronize` (Item 2).
- `p2p_combine.cu:299-304` — `place_add_f2_kernel` (14 ms; removable under Item 1).
- `p2p_combine.cu:308-313` — the **SECOND** full-tensor D2H (1720 ms). THE regression bucket.
- `f2_blocks_multigpu.cpp:87-90` — G1 fast path (one D2H, no combine — the baseline).
- `f2_blocks_multigpu.cpp:171-189` — §4 combine gate ⇒ `P2pDeviceResident` at P=768.
- `f2_blocks_multigpu_core.cpp:114-151` — jthread fan-out + join barrier (74% overlap; correct, NOT the bug).
- `include/steppe/fstats.hpp:52,60` — `F2BlockTensor.f2/.vpair = std::vector<double>` (the host-return seam).
- `tests/reference/bench_f2_multigpu.cu:59,62-64,112` — warm-up, timed region (combine inside), `ITERS=2`.

> **SUPERSEDED — historical (pre-M5).** This sweep concluded that **P≥2000 needs multi-GPU and
> P=2500 OOMs on every path** (75.7 GB resident result vs 95.6 GB/GPU). That conclusion was correct
> *before M5*, and is now **FALSE**. M5 SNP-tile **input streaming** (commit `c65179f`) + **adaptive
> tiered output** (commit `176a07d`) decouple the GPU footprint from the result size: full-autosome
> (M=584131, n_block=757) **P=2500 now COMPLETES on a single 32 GB RTX 5090 in ~51.5 s**, with the
> 76 GB result streamed out and GPU peak bounded at ~26 GB (parity `memcmp` bit-identical). The
> "wall is the resident full result" / "M5 becomes mandatory at ≥2000 pops" framing below is the
> *map of why M5 was built* — M5 then removed the wall. The honest perf story also shifted: on the
> precompute, multi-GPU is a modest throughput layer, not the headline; the real wins came from
> getting the result **off the CPU** (device-resident output `1f80c0c`) and **streaming** it
> (`176a07d` + `c65179f`). See **`docs/cleanup/m5/00-results.md`** for the current results. Kept
> verbatim below as the historical record — **do not act on its conclusions.**

# M4.5 multi-GPU — at-scale scaling sweep (where 2 GPUs top out)

Box: rtxbox (2× RTX PRO 6000 Blackwell sm_120, **95.6 GB/GPU**, **169 GB host RAM**), CUDA 13, Release,
production `EmulatedFp64{40}`. Data: AADR `derived_2500` (`--auto-top 2500`, **P=2500, M=584131, n_block=757**).
Harness: `tests/reference/bench_f2_multigpu.cu` (OOM-tolerant 3-path sweep, median of 3 + warm-up per cell).
Run via `agentscripts/m4.5-scaling-sweep.js`. The result tensor is `f2 + Vpair`, each `[P² · n_block]` FP64 ⇒
`result_GB = 2 · P² · n_block · 8 / 1e9`.

## The table (measured)

| P | n_block | result | single-GPU (G1) | G2 device-resident | G2 host-staged | G2res/G1 | G2host/G1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256  | 757 | 0.79 GB | 458.2 ms | **354.9 ms** | 463.2 ms | 1.29× | 0.99× |
| 512  | 757 | 3.18 GB | 1213.3 ms | **1045.2 ms** | 1478.9 ms | 1.16× | 0.82× |
| 768  | 757 | 7.14 GB | 2334.2 ms | **2117.0 ms** | 3086.9 ms | 1.10× | 0.76× |
| 1024 | 757 | 12.70 GB | 3802.4 ms | **3569.7 ms** | 5297.9 ms | 1.07× | 0.72× |
| 1536 | 757 | 28.58 GB | **OOM** | 7843.2 ms | 11746.2 ms | *enabling* | — |
| 2000 | 757 | 48.45 GB | **OOM** | 15099.5 ms | 21828.6 ms | *enabling* | — |
| 2500 | 757 | 75.70 GB | **OOM** | **OOM** | **OOM** | — | — |

(All OOMs are `cudaErrorMemoryAllocation` in `device_buffer.cuh:74` — a device-side `cudaMalloc` of a
`DeviceBuffer<double>`. The bench catches each per-cell and continues.)

## Headline — can we do 2500 pops on 2 GPUs?

> **SUPERSEDED:** the "No" below was true only for the *fully-resident* pre-M5 paths. With M5
> streaming (`c65179f` + `176a07d`) P=2500 completes on a single **32 GB** card (~51.5 s) — see the
> banner at the top and `docs/cleanup/m5/00-results.md`.

**No** (pre-M5, resident result). At P=2500 the result tensor alone is **75.7 GB**, and *every*
fully-resident path OOMs on a 95.6 GB GPU. **The ceiling on these two GPUs is P≈2000** (both
multi-GPU paths reach it; single-GPU does not). M5 lifts this ceiling by never holding the full
`P²·n_block` result on-device.

## Where each path hits its wall (with the memory budget)

- **Single-GPU** caps **~1024** (works at 1024 = 12.7 GB result; **OOM at 1536**). One device must hold the
  full result (`dF2_all`+`dVpair_all`) + the full-width inputs + GEMM working buffers; ~28.6 GB result + inputs
  + workspace already exceeds 95.6 GB at P=1536.
- **Multi-GPU device-resident** (the fast P2P combine) reaches **2000** (48.5 GB result, 15.1 s); **OOM at 2500**.
  The combine root holds the *full* result resident **plus** its own resident partial — at 2500 that is
  ~75.7 GB + ~37.9 GB ≈ 114 GB ≫ 95.6 GB.
- **Multi-GPU host-staged** also reaches **2000** (21.8 s) and **also OOMs at 2500** — on a *device* alloc, not
  host. Even though the full result lives in host RAM here, each device still needs its ~37.9 GB partial +
  inputs + workspace, which tips over 95.6 GB at 2500. So in this configuration host-staging did **not** extend
  the P ceiling beyond device-resident.

## The scaling read

1. **The device-resident speedup is real but shrinks with P:** 1.29× (256) → 1.16× (512) → 1.10× (768) →
   1.07× (1024). The combine bounce is gone (per `why-multigpu-slow.md`), but the **one final D2H of the result
   grows with P²** and is a serial tail after the (overlapped) compute — so the relative win erodes as the
   result dominates. It stays >1× at every P that fits, and **beats host-staged at every single P**.
2. **From P≈1536 up, multi-GPU is *enabling*, not just faster:** single-GPU OOMs, multi-GPU runs (to 2000).
   *(SUPERSEDED nuance:* this was the pre-M5 value — a second GPU bought P headroom one resident GPU could not
   hold. M5 streaming made that moot: a *single* 32 GB card now reaches P=2500 by never holding the full result.
   On the precompute, multi-GPU is therefore a **modest throughput layer**, not the at-scale enabler — and it was
   measured *slower* than single-GPU until the data-bounce was fixed (nsys ~22–74% overlap, a serial D2H/host
   tail; see `why-multigpu-slow.md`). Multi-GPU's proper home is the **fit/rotation phase** — thousands of
   independent qpAdm models, no combine — not the precompute.)*
3. **Host-staged is the portable fallback, not a lever:** slower than device-resident everywhere (0.99×→0.72×
   vs G1), and here it bought no extra P headroom. Its purpose remains correctness on no-P2P (consumer) boxes.
4. **The wall is the resident full result** (`2·P²·n_block·8`), not compute or the combine. To go past ~2000
   pops you must stop holding the entire `f2_blocks` tensor resident — i.e. **M5 out-of-core**, **which has
   since been built** (input streaming `c65179f` + adaptive tiered output `176a07d`). M5 streams the SNP-tiles in
   (per-block decode; GPU footprint `O(P·tile + P²)`, *independent of M* — no `7·P·M` feeder wall) and routes the
   result to the fastest tier it FITS (VRAM-resident → host RAM → disk, auto-selected from runtime free
   VRAM/RAM), so neither the device nor a single buffer holds all `P²·n_block`. With M5, full-autosome
   (n_block=757) P=2500 runs on a single **32 GB** card (~51.5 s; GPU peak ~26 GB). This sweep is the precise
   map of *why* M5 was needed (≥~2000 pops at full-autosome would otherwise OOM). Fewer blocks (e.g. a single
   chromosome) or smaller P stay comfortably on-GPU and keep the device-resident fast path (`1f80c0c`).

## Caveats

- The bench's per-cell `bytes: H2D/D2H/peer` columns print `0.00 GiB` — the orchestrator-side byte arithmetic
  is not filled across the CUDA-free seam (observability only; the wall-clock numbers are unaffected).
- Median of 3 (+ warm-up) for sweep wall-clock sanity; the P=768 numbers reproduce the 1.10× from
  `why-multigpu-slow.md` (committed 867a4bf), confirming the harness measures the same thing.

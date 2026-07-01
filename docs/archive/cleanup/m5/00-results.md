# M5 results — out-of-core f2 precompute (device-resident output + adaptive tiering + SNP-tile input streaming)

This is the repo-verified record of M5: the milestone that takes the Phase-1 f2 precompute
from "fits-in-VRAM only, OOMs past P≈2000" to **full-autosome P=2500 completing on a single
32 GB consumer 5090 in ~51.5 s**. It supersedes the pre-M5 ceiling that
`docs/cleanup/m4.5/scaling-sweep.md` mapped (that doc is now **historical** — it correctly
reported that *every* path OOM'd at P=2500 *before M5*; M5 makes that false).

Three commits, in dependency order:

- **`1f80c0c`** — device-resident output (get the result OFF the CPU).
- **`176a07d`** — adaptive tiered output (Resident / HostRam / Disk, auto-selected).
- **`c65179f`** — SNP-tile INPUT streaming (remove the all-M feeder wall; the other half of M5).

All parity gates per `architecture.md §12` hold: every tier reads back `memcmp` **bit-identical**
to the `EmulatedFp64{40}` single-GPU reference; block-axis and SNP-tile streaming are exact
**by construction** (see §5).

---

## TL;DR — the headline

- **The real perf win was getting the result OFF the CPU, not multi-GPU.** At P=512 (3.18 GB,
  full-autosome M=584131, n_block=757) the device-resident compute is **~666 ms vs ~2873 ms** for
  the old bulk-to-host return — **~4.3×**, single-GPU. The old precompute was *host-result-bound*:
  ~80% of the wall was copying the 6.36 GB+ result into CPU RAM to satisfy a host return type that
  **nothing consumes** (the fit engine is unbuilt). (`1f80c0c`)
- **The output is adaptively tiered, not mandatorily streamed.** The result lands in the fastest
  tier it *fits*: **Resident** (VRAM, keeps the 4.3×) → **HostRam** → **Disk**, auto-selected from
  the runtime free-VRAM / free-host-RAM probes. Small P stays resident; large P spills, with no
  penalty to the small-P path. (`176a07d`)
- **The GPU footprint is now independent of M.** SNP-tile input streaming makes the per-chunk
  device footprint **O(P·tile + P²·n_block)** instead of the old **7·P·M** all-M feeder wall.
  (`c65179f`)
- **The headline number is now repo-verified:** full-autosome (M=584131, n_block=757) **P=2500
  COMPLETES on a single 32 GB RTX 5090 in ~51.5 s**, streaming a **75.7 GB** result out-of-core
  with GPU peak bounded **~26 GB**, parity bit-identical.

---

## 1. Device-resident output — the actual lever (`1f80c0c`)

The pre-M5 precompute computed `f2_blocks[P×P×n_block]` + `Vpair` on the GPU and was then **forced
by its host `F2BlockTensor` return type** to dump the entire result into CPU RAM (a host
`resize(total)` zero-fill + the D2H + staging). At P=512 that host round-trip was **~2150 ms of the
~2900 ms wall** — and it existed **only to satisfy the host return**; nothing downstream consumes
the host result (the Phase-2 fit engine does not exist yet — see `docs/cleanup/m4.5/why-d2h.md`).

The cure: the precompute now produces a **DEVICE-RESIDENT result** and returns a VRAM handle
(`DeviceF2Blocks`). No forced D2H, no host alloc/zero on the hot path. The host `F2BlockTensor`
becomes an **opt-in `.to_host()` materialization** — the *sole* D2H + host alloc — used only by the
parity test, the future M7 disk cache, and explicit host/CLI callers.

- **G==1** (the headline, box5090, no P2P): the result stays on the one GPU after the GEMM; the
  handle is returned. **No D2H at all.**
- **G≥2 P2P box:** per-device `DevicePartial`s stay resident, assembled device-resident on the root
  (`combine_f2_partials_resident_device`, the no-final-D2H variant). No host bounce.
- **G≥2 no-peer box:** documented `§11.4` limitation — a single tensor across two non-P2P cards
  needs a host bounce as the *assembly transport*; the partials stay resident and the result is
  re-uploaded so the primary return is still a `DeviceF2Blocks`.

**MEASURED (box5090, RTX 5090, Release, P=512, full-autosome, median of 3):**

| path | wall |
|---|---|
| G1 host-returning (incl. `to_host` tail) | 2876–2922 ms |
| G1 device-resident (no materialization) | 670–736 ms |

⇒ the ~2150 ms CPU round-trip is gone; **~4.3×** faster; the resident wall sits near GPU compute
time. **Parity:** `test_f2_multigpu_parity` drives the device-resident primary and materializes once
(`.to_host()`) for the `memcmp` — **PASS, bit-identical** to the single-GPU reference.

**THE KEY LESSON (corrects the older M4.5 framing):** the precompute was *host-result-bound*, not
compute-bound or multi-GPU-bound. Getting the result off the CPU was the real win — **not** the
multi-GPU sharding M4.5 built. See §6.

---

## 2. Adaptive tiered output — opt-in by need (`176a07d`)

The `[P×P×n_block]` f2/Vpair result now lives in the **fastest tier it FITS**, selected
**automatically** from the runtime free-VRAM (`cudaMemGetInfo`) + free-host-RAM (`sysinfo`) probes
— never hardcoded — so large P streams out-of-core **without** penalizing the small-P
device-resident path. Streaming is **opt-in-by-need**, not mandatory.

The policy is the CUDA-free `OutputTier` enum + `select_output_tier()` in
`src/device/tier_select.hpp` (host-pure, unit-testable with no GPU; `STEPPE_FORCE_TIER` /
`config.force_tier` pins a tier for tests):

| tier | when | what happens |
|---|---|---|
| **0 Resident** | result + working set fit free VRAM | the **existing** device-resident path UNCHANGED (`compute_f2_blocks_device`, no sink, no streaming) — keeps the 4.3× |
| **1 HostRam** | does not fit VRAM, fits free host RAM | block-by-block spill into a host tensor |
| **2 Disk** | fits neither (laptop) | block-by-block spill to a `STPF2BK1` cache file via a small persistent pinned staging ring — the precompute-once / fit-many on-disk artifact |

**Mechanism (tiers 1+2): block-axis streaming.** The streamed loop reuses the resident path's
per-block gather/GEMM/assemble verbatim; only the assemble's destination offset is chunk-local while
the spill places each slab under its **global** block id (value bits unchanged). The sink uses a
**persistent pinned ring** (pinned once in `begin()`, reused — not per-call `cudaHostRegister`) + a
**background writer thread** + a double-buffered device chunk-ring, so the slow host-copy / `pwrite`
**overlaps the GPU compute of the next chunk** (triple-buffer).

**EVIDENCE (single-GPU, EmuFp64{40}, derived_full M=584131 n_block=757):**

- **P=512 (3.18 GB) AUTO-selects Resident** — ResCompute 666.5 ms vs ToHost 2873.1 ms (the ~4.3×
  device-resident win INTACT; streaming never engages). **P=768 (7.14 GB) AUTO-selects HostRam**
  (result + working set exceeds the resident-tier VRAM fraction of free VRAM) — opt-in-by-need.
- **HostRam ≈ ToHost (1.01–1.04×):** block-streaming adds **no penalty** over bulk host
  materialization — the spill fully hides behind compute + D2H.
- **Disk overlaps compute:** measured fs bandwidth ~852 MB/s → a 3.18 GB serial write would be
  ~3734 ms; the Disk wall (4566 ms) is only +1693 ms over ToHost (≪ 3734 ms), so ~2000 ms of the
  write was hidden by the triple-buffer (a fully-serialized impl would be ~6600 ms). The residual is
  genuine disk bandwidth, not serialization.

**Parity:** Forced **Resident** AND **HostRam** AND **Disk** all read back (`to_host`) `memcmp`
bit-identical to the EmuFp64{40} single-GPU reference; the Disk per-block accessor
`read_block_to_host(b)` is byte-exact for every `b` (`§4` pread offsets); Auto at small P picks
Resident.

---

## 3. SNP-tile INPUT streaming — remove the all-M feeder wall (`c65179f`)

Output tiering block-streamed the **result** to host/disk, but the streamed path still reused the
resident prologue's **all-M feeder**: it uploaded + decoded ALL M SNPs at once
(3·P·M raw + 4·P·M persisted feeder outputs = **7·P·M** doubles resident). At full-autosome
M=584131 that feeder alone needs ~25 GB @P=768 / ~32.7 GB @P=1000, so it **OOM'd at
`device_buffer.cuh:74` (`cudaMalloc`) at P≥768 on a 32 GB card BEFORE the result tier ever
mattered** — output streaming could not help because the **input feeder died first**.

This commit moves **SNP-tile INPUT streaming** into `stream_f2_blocks_impl`: each block-stream chunk
now decodes **only its own SNP-column tile** `[s_lo, s_hi)` — a single contiguous
`cudaMemcpyAsync` per matrix of the host `[P×M]` Q/V/N (column-major ⇒ the slice starts at
`P·s_lo`), the **same** feeder over `tile` columns, then a gather via rebased local offsets
(`block_offsets[gid] - s_lo`). The full `[P×M]` inputs stay in **host RAM** (the caller's
`MatView`). The GPU per-chunk footprint becomes **O(P·tile + P²·n_block)**, **independent of M**.
The streamed VRAM budget is rebuilt against this real footprint
(`(4·P·s_pad + 8·P²)·nb + tile feeder`), not the resident budget that reserved the phantom result
tensor; `tier_select.hpp` scopes the **7·P·M** envelope to the **Resident** tier only and adds a
`streamed_working_set_bytes` with no `P·M` term.

**Verified (box5090, 2× RTX 5090 32 GB consumer, CUDA 13, Release, single-GPU):**

- `ctest` **36/36 green** incl. `f2_multigpu_parity`.
- Forced-tier `memcmp` gate **bit-identical**: streamed HostRam/Disk `to_host()` == single-GPU
  reference; per-block accessor byte-exact.
- Full-autosome `derived_2500` (M=584131, n_block=757), HostRam streamed tier:
  - **before:** P≥768 ALL OOM (the 7·P·M feeder wall at `device_buffer.cuh:74`).
  - **after:** COMPLETES at P=1000 / 1500 / 2000 / 2500; GPU peak bounded **~26 GB**, independent
    of M. **Max-P completing: 768 → 2500.**

The unchanged all-M **Resident** path (`ResCompute`/`ToHost`) still OOMs at P≥1000 (expected — that
is the path the streaming tiers replace at scale).

---

## 4. THE HEADLINE — measured full-autosome sweep (one 5090, streamed)

box5090 (one **32 GB** RTX 5090, CONSUMER, CUDA 13, Release, `EmulatedFp64{40}`). Data: AADR
full-autosome **M=584131, n_block=757**. Streamed HostRam tier (SNP-tile input + block-axis output).
`result_GB = 2 · P² · n_block · 8 / 1e9`.

| P | result | wall | notes |
|---:|---:|---:|---|
| 512  | 3.18 GB  | **~3.6 s**  | (Auto would pick Resident at this P; sweep forces the streamed tier) |
| 1000 | 12.12 GB | **~10.4 s** | |
| 1500 | 27.26 GB | **~20.2 s** | |
| 2000 | 48.45 GB | **~34.0 s** | |
| 2500 | **75.70 GB** | **~51.5 s** | **completes**; GPU peak bounded **~26 GB**, independent of M |

**This is the number that makes the 51.5 s repo-verified.** Before M5 (see
`docs/cleanup/m4.5/scaling-sweep.md`), P=2500 OOM'd on *every* path — even on the 96 GB RTX PRO 6000
box, because the 75.7 GB result + a resident partial exceeded VRAM. M5 streams the 75.7 GB result
out-of-core, the GPU only ever holds the bounded **~26 GB** working set, and the run completes on a
**single consumer 32 GB card**.

---

## 5. Parity — bit-identical, by construction

Both streaming axes are **exact by construction**, not approximate:

- **Block-axis output streaming** (`176a07d`): each block is computed identically and independently;
  the sink changes only *when/where* a slab lands, never its bits.
- **SNP-tile input streaming** (`c65179f`): the feeder is per-column elementwise, so feeding
  `[s_lo, s_hi)` in isolation is bit-identical to feeding it inside the all-M sweep; the gather reads
  the **same** host columns; only *when* columns upload moves, never values.

Proven both ways: a forced-tier `memcmp` gate shows Resident / HostRam / Disk `to_host()` ==
the `EmulatedFp64{40}` single-GPU reference (full pipeline, including at the small P=50 parity dataset
the multi-GPU gate uses), and the Disk per-block accessor `read_block_to_host(b)` is byte-exact for
every block. `ctest` is 36/36 green including `f2_multigpu_parity`.

---

## 6. The honest multi-GPU story (corrects the older M4.5 docs)

M4.5 built single-node multi-GPU (block-aligned shard, host-staged + device-resident P2P combine,
bit-identical), and it is correct and parity-locked. But the M5 measurements settle which lever
actually mattered:

- **The real perf wins came from getting OFF the CPU** — device-resident output (`1f80c0c`) and
  streaming (`176a07d`, `c65179f`) — **not multi-GPU per se.** The precompute was host-result-bound.
- On the precompute, multi-GPU is a **modest throughput layer**, and was in fact measured *slower*
  than single-GPU until the data-bounce was fixed; nsys showed only ~22–74% overlap with a serial
  D2H / host tail (`docs/cleanup/m4.5/parallelism-check.md`, `why-multigpu-slow.md`, `why-d2h.md`).
- **Multi-GPU genuinely shines on the FIT / ROTATION phase** — thousands of *independent* qpAdm
  models, no combine — which is its proper home, not the precompute
  (`docs/cleanup/m4.5/architecture-audit.md`).

---

## 7. vs ADMIXTOOLS 2 (estimate, not a measured ratio)

Per `docs/research/at2-timing-comparison.md`: the f2 precompute is plausibly **~2–3 orders of
magnitude** faster than AT2 `extract_f2` at thousands-of-pops scale. **This is an ESTIMATE, not a
measured ratio** — AT2 published no `extract_f2`-vs-P wall-clock, so the AT2 side is a
complexity-based estimate (O(P²·M), anchored to the one "~a day → ~20 min" anecdote), set against
steppe's **measured** anchors (P=2000 = 15.1 s on 2× RTX PRO 6000; P=2500 = 51.5 s on one 5090). Do
**not** quote a single clean "N×" as fact.

---

## 8. Caveats

- **Disk-tier ENOSPC on a small overlay.** On the box5090's small overlay filesystem the Disk tier
  fails at P≥2000 on disk space (`pwrite` ENOSPC) — that is a *capacity* limit of the test overlay,
  not the streaming mechanism. The HostRam tier (the sweep above) completes through P=2500. The Disk
  tier is correctness-proven at the sizes that fit (forced-tier `memcmp` bit-identical).
- **Parity proven at P=50 + by construction.** The locked `memcmp` gate runs at the small parity
  dataset (P=50); at scale, bit-identity rests on the by-construction argument of §5 (per-block,
  per-column exactness) plus the forced-tier `to_host` / per-block-accessor byte-exact checks.
- **The streamed per-P numbers are box5090 (32 GB consumer 5090) wall-clocks**, single-GPU, Release,
  EmuFp64{40}. The earlier on-VRAM ceiling sweep (`docs/cleanup/m4.5/scaling-sweep.md`) was on the
  96 GB RTX PRO 6000 box and is **pre-M5 / superseded** by the results here for the P=2500 claim.

---

## 9. What's next — the Phase-2 qpAdm fit engine (does NOT exist yet)

M5 finishes Phase-1 precompute through the scale story. The next phase is the **qpAdm FIT engine
(Phase 2, S3–S8)**, which **does not exist yet**. It reads `f2_blocks` (device-resident for the
in-VRAM case per `why-d2h.md`; streamed tiles for large P), runs the GLS solve + rank test + the
**model rotation** (many independent qpAdm models = the embarrassingly-parallel, multi-GPU-friendly
phase). AT2 goldens (R + admixtools, pinned `extract_f2` / `qpadm` goldens) are the validation gate.

Deferred / optional: multi-GPU block-sharding on the stream (throughput); pinning the device-resident
final D2H (rtxbox); the TurboQuant-L2 rotation screen — **after** the fit exists
(`docs/research/turboquant-l2-experiment.md`).

---

### Cited anchors

`1f80c0c` (device-resident output) · `176a07d` (adaptive tiered output) · `c65179f` (SNP-tile input
streaming) · `src/device/tier_select.hpp` (`OutputTier`, `select_output_tier`, the host-pure policy)
· `src/device/stream_f2_blocks.hpp` + `src/device/cuda/block_sink.{cu,cuh}` (the streamed loop +
persistent-pinned sink) · `src/device/f2_disk_format.hpp:21` (`STPF2BK1` magic) ·
`src/device/cuda/device_buffer.cuh:74` (the `cudaMalloc` that OOM'd on the old 7·P·M feeder) ·
`docs/cleanup/m4.5/scaling-sweep.md` (**SUPERSEDED** pre-M5 ceiling; P=2500 OOM was true *before* M5)
· `docs/cleanup/m4.5/why-d2h.md` (device-resident handoff = the design contract) ·
`docs/cleanup/m4.5/parallelism-check.md`, `why-multigpu-slow.md`, `architecture-audit.md` (the honest
multi-GPU story) · `docs/research/at2-timing-comparison.md` (the AT2 ESTIMATE) ·
`tests/reference/test_f2_multigpu_parity.cu` (the locked `memcmp` gate).

# Why is there a D2H of `f2_blocks` at all?

> **RESOLVED / OUTCOME (post-M5).** This doc's recommendation SHIPPED. The precompute now
> returns a **device-resident handle** (`DeviceF2Blocks`; the result stays in VRAM) and the host
> `F2BlockTensor` is an **opt-in `.to_host()`** — exactly the §5 recommendation below (commit
> `1f80c0c`). **Measured:** P=512 device-resident ~673 ms vs ~2879 ms bulk-to-host = **~4.3×** —
> confirming the key lesson that the precompute was HOST-RESULT-BOUND (~80% of the old wall was
> copying the multi-GB result to CPU; getting it OFF the CPU was the real win). The §2(c) /
> §3 / §4 scale-beyond-VRAM case is also DONE via **M5 out-of-core streaming**: ADAPTIVE TIERED
> output (commit `176a07d`) sends the result to the fastest tier it FITS (VRAM-resident -> host
> RAM -> disk, auto-selected from runtime free VRAM/RAM), and **SNP-tile input streaming** (commit
> `c65179f`) bounds the device footprint to O(P·tile + P²), independent of M. Full-autosome
> (M=584131, n_block=757) P=2500 now **COMPLETES on a single 32 GB RTX 5090 in ~51.5 s** (76 GB
> result streamed, GPU peak ~26 GB), parity bit-identical. So the unconditional eager whole-tensor
> D2H this doc argues against no longer exists: the in-VRAM case has no D2H, and the
> beyond-VRAM case spills in block-tiles per the design. The analysis below is preserved as the
> design rationale that drove `1f80c0c` + `176a07d` + `c65179f`.

Lead-architect answer, from the design + code. Read-only audit; all claims cited to
`file:line` or doc section. The three review lenses (deliverable-contract,
consumer-dataflow, eliminate-or-fuse) converged on the same verdict; this doc
states it plainly and quantifies it.

---

## Executive summary (read this)

The precompute ends with a Device→Host copy of the whole `f2_blocks` (+ `Vpair`)
tensor **because the function's return type is a host-resident `F2BlockTensor`
(`std::vector<double> f2/vpair`, `include/steppe/fstats.hpp:52,60`), and the only
thing on the far side of that seam today is the test/bench suite — there is no fit
engine, no on-disk cache, no CLI/Python consumer in the tree.** So the copy is
**incidental, not fundamental**: it is the phase boundary (precompute is Phase 1 /
done; the fit engine is Phase 2 / "does not exist yet", `ROADMAP.md:77`) leaking
through a host-typed return value into a consumer that isn't built. The spec
mandates the *opposite* of an eager host copy: `f2_blocks` "**stays on the device
across the whole stream and the entire downstream model search**"
(`architecture.md:673`); keep it "**GPU-resident**" (`§5:235`); after the multi-GPU
combine "**every GPU holds the full `f2_blocks`**" so each fit runs on-device
(`§11.4:717`). A **fused on-GPU precompute→fit** that keeps the tensor in VRAM and
lets the fit read small pair-subsets block-by-block would have **no D2H at all** in
the common (fits-in-VRAM) case — eliminating the measured wall (**1720 ms / 7.14 GB
at P=768, 4.15 GB/s pageable; 54%→88% of the multi-GPU wall as P grows**,
`why-multigpu-slow.md:58`, `architecture-audit.md:99`). A host/disk copy IS real,
but only as **opt-in terminal steps** the design names: the M7 on-disk
AT2-compatible cache and the Phase-3 public C-ABI/Python egress — never on every
precompute's critical path. **Recommendation:** design the Phase-2 fit engine to
consume `f2_blocks` **device-resident** for the in-VRAM case (the device handle —
`DevicePartial` — already exists internally), and treat host/disk as the
persistence path, not the default return. This is audit **Flaw 4** restated.

---

## 1. The direct answer — why the D2H exists today

The terminal act of every precompute path is a D2H of the whole tensor into freshly
allocated host vectors:

- single-block / grouped: `cuda_backend.cu:238-244` and `:281-290`
  (`out.f2.resize(...)`, `cudaMemcpyAsync(... cudaMemcpyDeviceToHost ...)`),
- the host-staged-direct multi-GPU seam: `compute_f2_blocks_into` →
  `cuda_backend.cu:354-357`,
- and even the **device-resident** P2P combine ends in "the ONE final D2H"
  (`p2p_combine.cu:18,176,187-190`).

It exists for exactly one reason: **the return type is a host `F2BlockTensor`.** The
public f-stats handle is `struct F2BlockTensor { std::vector<double> f2; vpair; ... }`
(`fstats.hpp:47-60`); every `ComputeBackend::compute_f2_blocks` override returns that
host type (`backend.hpp:261-263`), so the copy is mandated by the *signature*, not by
any downstream read. And **nothing downstream reads it for fitting**: a whole-tree
grep for the named consumers (`qpadm`, `fit`, `f3`, `f4`) finds no such code —
`qpadm` appears only in the CMake project *description* (`CMakeLists.txt:3,22`);
`src/app`, `src/api`, `src/core/qpadm` **do not exist**; the only readers of
`.f2/.vpair` are the producer itself, the combine/parity validators, the CPU oracle,
and tests. The D2H feeds a return value that flows into tests and the bench, then is
freed.

In one sentence: **the deliverable is a host-resident `F2BlockTensor` because the
precompute function is typed to return host vectors across the CUDA-free seam, and
the Phase-2 fit engine that the architecture says should consume the tensor
*device-resident* has not been built yet — so the host copy is the only way to
"finish" precompute and prove it correct in tests.**

---

## 2. Fundamental or incidental? — split it cleanly

### Incidental (the part that is the measured wall)

The **intra-run, fit-feeding D2H** — the one perf work hit — is incidental, and it
directly *contradicts* the spec. Two contingencies, neither a design mandate, force
it:

1. **The phase boundary is currently a process boundary.** Precompute returns a value
   and exits because there is no fused precompute→fit driver — the fit engine is
   "**← next real work (does not exist yet)**" (`ROADMAP.md:77`; `TODO.md:5` "The fit
   engine (Phase 2) does not exist yet"). The spec's residency contract is the
   opposite of a host copy:
   - "`f2_blocks` … **stays on the device across the whole stream and the entire
     downstream model search**" (`architecture.md:673`, §11.1),
   - "keep `f2_blocks` and `Q` **GPU-resident**" (`§5:235`),
   - after the multi-GPU combine "the result is **broadcast back** … so **every GPU
     holds the full `f2_blocks`**" and "each fit runs wholly on one device"
     (`§11.4:717`) — an **on-device broadcast** handoff, not a D2H to a host vector.
2. **The host typing is a layering/ABI artifact, not a data-flow mandate.**
   `F2BlockTensor` lives in `include/` and is host-typed so it can cross the
   CUDA-free seam to "`core`, the CLI, and the Python bindings without dragging in the
   device toolkit" (`fstats.hpp:10-15`; ADR-0008 C-ABI). But a device-resident handle
   already exists internally — `DevicePartial` (`device_partial.hpp`;
   `compute_f2_blocks_resident`, `cuda_backend.cu:300`) — and the P2P combine builds
   the full result *device-resident* then collapses it back to host purely to satisfy
   the return type (`p2p_combine.cu:176-190`). The D2H is glued to the **return
   type**, not to a consumer's requirement.

### Fundamental — but conditional, and only at boundaries the design names

A host (or disk) copy is genuinely unavoidable in three cases, all **opt-in /
terminal**, never "every precompute":

- **(a) Persistence — the M7 on-disk, AT2-compatible cache.** `f2_blocks` is "the
  cacheable, ADMIXTOOLS-compatible interchange artifact" (`architecture.md:37`, §0:12;
  ADR-0005). "Precompute-once / fit-many" *across sessions/datasets* means the artifact
  must outlive the GPU context. The store is `io/precomputed_f2.cpp` **"(planned,
  M7)"** (`architecture.md:169`; `ROADMAP.md:74`) and **does not exist yet**. This is a
  mandate to **persist when asked**, not to D2H on every run.
- **(b) Public boundary — the C-ABI / Python egress.** Host-side callers cannot see
  CUDA, so returning results to them needs host data (`fstats.hpp:10-15`; ADR-0008).
  But that surface consumes *fit outputs* (`est/se/z/p`), not the raw tensor; only the
  M7 cache export and a CPU-backend fit need the raw host `f2_blocks`. Phase 3
  (`ROADMAP.md:80`), unbuilt.
- **(c) Scale beyond VRAM.** The spec itself concedes the full-resident model breaks
  at the top end: at `P=4266 / B=757` the `f2 + Vpair` pair "**reaches ≈220 GB and is
  itself VRAM-budgeted per §11.2**" (`architecture.md:717`). The pre-M5 scaling sweep
  reported every full-resident path OOMing past P≈2000 on 96 GB cards
  (`scaling-sweep.md:52-56`, now SUPERSEDED — that doc is pre-M5). **M5 fixed exactly this:**
  block-tile spill to host RAM or disk, overlapped (`176a07d`), plus SNP-tile input streaming
  (`c65179f`), so P=2500 full-autosome now completes on a single 32 GB 5090. Here the bytes
  *do* spill — to **pinned host or disk in block-tiles, overlapped** (M5), not a single serial
  whole-tensor copy.

The honest meta-point (audit **Flaw 4**, `architecture-audit.md:112-122`): the
deliverable is "materialized whole … then copied whole," but the consumer (the
leave-one-block-out jackknife fit) "contracts over the `n_block` axis and reads a
small *pair-subset* … **never needing all P²·n_block FP64 co-resident**. The
architecture optimizes residency of an artifact the consumer reads block-by-block /
subset-by-subset." And the "tiny / MB-scale / cacheable, so the copy is free"
premise (`architecture.md:12`) that *would* excuse an eager D2H is **false at this
project's own target P** — true at P=100 (0.12 GB) but 75.7 GB at P=2500
(`architecture-audit.md:116-120`). So the contract neither mandates nor excuses the
current eager whole-tensor D2H.

---

## 3. The fused alternative — no D2H at all (the common case)

When `f2_blocks` fits VRAM (AT2's typical P of tens-to-hundreds, and up to P≈2000 at
full-autosome `n_block` on a 96 GB card), the entire D2H disappears:

1. **Precompute leaves `f2_blocks` + `Vpair` device-resident.** The resident path
   already exists: `compute_f2_blocks_resident` → `DevicePartial`
   (`cuda_backend.cu:300`); the device-resident combine places the disjoint slices in
   VRAM (`p2p_combine.cu:109-174`). The only thing to cut is the final D2H
   (`p2p_combine.cu:187-190`); the precompute returns a **device handle** instead of a
   host `F2BlockTensor`.
2. **The fit engine (S3–S8) runs directly on the resident tensor in VRAM** — exactly
   `architecture.md:673,235,717`. The fit is small dense LA: qpAdm's matrix is
   `m = (n_L−1)·(n_R−1)`, "low-double-digit" (`§11.4:721`); the GLS is one Cholesky +
   one SVD per model, **not** an iterative sweep (`§5:233`). Each fit gathers a
   `[m × n_block]` pair-subset from VRAM, contracts over the jackknife `n_block` axis,
   and the only bytes crossing PCIe are the tiny `est/se/z/p` outputs. For multi-GPU,
   the cross-phase transport is the spec's on-device broadcast (`§11.4:717`), not a
   D2H.

**What it saves:** the measured wall. The single mandatory final D2H is **1720 ms for
7.14 GB at 4.15 GB/s** (pageable) at P=768 (`why-multigpu-slow.md:58`), and the D2H
share of the multi-GPU wall climbs **54% (P=256) → 81% (P=768) → 86% (P=1024) → 88%**
(`architecture-audit.md:99`). It is "**an architectural choice, not memory-bound
physics**" (`why-multigpu-slow.md:38`). Fusing eliminates it entirely for the in-VRAM
regime — and building the Phase-2 fit engine *is* the act that removes it.

---

## 4. When the D2H is genuinely unavoidable

Only the three conditional cases of §2, each opt-in and overlapped — never the
current unconditional, serial, pageable whole-tensor copy:

- **Persist / cache (M7):** stream block-tiles to the on-disk AT2-compatible store,
  overlapping each spill with the next tile's GEMM; **pinned**, ideally **GDS
  Device→disk skipping host RAM**. Honest caveat from `TODO.md:147,156,164`: true GDS
  is **dead on the budget consumer 5090 box** (GeForce GSP-firmware lock + OverlayFS),
  where the pinned double-buffer is the correct fallback; GDS is a **real bet only on
  the RTX PRO 6000** (Quadro-class, full-host) for the large `.f64` cache. Off the fit
  critical path either way.
- **CPU/host consumer or public ABI:** materialize a host `F2BlockTensor` only when the
  CLI/Python caller actually asks for the host artifact, or when a CPU-backend fit runs
  (`fstats.hpp:10-15`; ADR-0008). API egress, not the internal handoff.
- **Scale beyond VRAM (M5, P > ~2000):** block-axis streaming makes device footprint
  `O(P²·tile_blocks)` not `O(P²·n_block)` (`architecture-audit.md:127-128`), dissolving
  the OOM wall; completed block-tiles spill to pinned host / GDS, overlapped, and the
  fit streams them back in tiles for the jackknife contraction — never re-materializing
  the whole tensor host-side.

In all three, when a host copy *is* taken it should be **pinned + block-sharded**
(audit Flaw 3, ~12× win: P=768 tail 1720→~143 ms, `architecture-audit.md:104,233`) and
block-axis overlapped — not the current `~4.15 GB/s` pageable serial tail.

---

## 5. Recommendation — does this change the roadmap?

Yes, but as a *design constraint on Phase 2*, not a reordering:

1. **Make the precompute→fit handoff device-resident. [DONE — commit `1f80c0c`.]** The
   device-resident result handle (now `DeviceF2Blocks`) IS the **primary** output of the
   precompute; the host `F2BlockTensor` is an **opt-in `.to_host()`** materialization, not the
   unconditional return — exactly as recommended here. This is what `architecture.md:673,235,717`
   prescribe; the code now implements it. **Measured: P=512 device-resident ~673 ms vs ~2879 ms
   bulk-to-host = ~4.3×.**
2. **Design the Phase-2 fit engine to read `f2_blocks` from VRAM, block-by-block /
   pair-subset** (Flaw 4; `scaling-sweep.md:52-56`), so it never needs the whole
   `2·P²·B` co-resident and never round-trips through host. Building this is the act
   that removes the measured wall for the common case.
3. **Treat host/disk as the persistence path, not the default:** the M7 cache spill
   (opt-in, block-tiled, pinned/GDS) and the Phase-3 public egress are the only places
   a host copy belongs. **M5 out-of-core (DONE) handles the P-beyond-VRAM spill** with the
   block-tiled, overlapped discipline this doc prescribes: adaptive tiered output (`176a07d`)
   spills to host RAM or disk only when the result does not fit VRAM, and SNP-tile input
   streaming (`c65179f`) keeps the device footprint at O(P·tile + P²). P=2500 full-autosome now
   completes on one 32 GB 5090 in ~51.5 s (parity bit-identical), so the beyond-VRAM case no
   longer OOMs.

### Where the design is explicit vs silent

- **Explicit:** `f2_blocks` stays device-resident across the stream *and the downstream
  model search* (`§11.1:673`, `§5:235`, `§11.2:692`); multi-GPU handoff is an on-device
  broadcast (`§11.4:717`); `f2_blocks` is the cacheable AT2-compatible artifact destined
  for an M7 on-disk store (`§37`, `:169`); the C-ABI public seam is host-typed
  (ADR-0008); the full-resident model is conceded to break past ≈220 GB (`§11.4:717`).
- **Silent:** the spec never names a single `compute_f2_blocks(...) -> F2BlockTensor`
  return type, nor that the precompute must D2H to *feed the fit* — that is purely the
  current backend signature (`backend.hpp:261-263`). It does not specify the exact
  device-handle type the fit consumes, nor an explicit "fused precompute+fit" driver;
  those are implied by the residency mandate but not spelled out. It is silent on
  whether a single in-process run *must* write the M7 cache (the cache is for
  cross-run/cross-session reuse, not an obligation per run).

**Bottom line:** the contract never says the result must leave the GPU **to be
fitted** — it says the opposite. The D2H that perf work hit is the *fit-feeding* one,
and it is incidental: the host return type + the absent fit engine. The only
fundamental host copies are the M7 cache and the public-API egress, both opt-in and
off the critical path. Build the fit engine to consume `f2_blocks` device-resident,
and the wall is gone.

---

### Cited anchors

`include/steppe/fstats.hpp:10-15,47-60` (host-vector deliverable = the seam) ·
`src/device/backend.hpp:261-263` (return type forces the D2H) ·
`src/device/cuda/cuda_backend.cu:238-244,281-290,300,354-357` (D2H sites; resident
path exists) · `src/device/cuda/p2p_combine.cu:18,176,187-190` (even the resident
combine ends in the one final D2H) · `src/device/device_partial.hpp` (the
device-resident handle) · no consumer: `CMakeLists.txt:3,22` (qpadm = description
only); `src/app|src/api|src/core/qpadm` absent ·
`docs/ROADMAP.md:65,74,77,80` (Phase 1 done / Phase 2 fit "does not exist yet" / M7
cache / Phase 3 bindings) · `docs/architecture.md:12,37,169,235,673,692,717`
(MB-scale/cacheable; device-resident through the fit; on-device broadcast; ≈220 GB
concession) · `docs/TODO.md:5,147,156,164` (fit engine absent; GDS dead on 5090, live
on PRO 6000) · `docs/cleanup/m4.5/architecture-audit.md:96,99,104,112-128,233`
(Flaw 4; D2H = 54%→88% of wall; pinning/sharding ~12×) ·
`docs/cleanup/m4.5/why-multigpu-slow.md:38,58` (the measured wall; "architectural
choice, not memory-bound physics") · `docs/cleanup/m4.5/scaling-sweep.md:52-56`
(resident full result is the ceiling past P≈2000).

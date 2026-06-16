# `steppe` — TODO / task tracker

Living, checkable companion to [`ROADMAP.md`](ROADMAP.md) (the **order & rationale**) and [`architecture.md`](architecture.md) (the **design & standards**). This is the **granular next-steps checklist for future-us** — keep it current as milestones land. Update the checkboxes; move finished items to "Done".

**Big picture (2026-06-16):** the **precompute half exists** — M0–M4 done (validated 3-GEMM f2 kernel → data front-end → filters → per-block `f2_blocks` tensor), all green on the box. The **fit engine (Phase 2) does not exist yet**, and the precompute still has scale/robustness milestones (M4.5–M7). Everything below sits on branch `m4-perblock-f2` until M4 is merged to `main`.

---

## ✅ Done (Phase 1 precompute, M0–M4)
- [x] **M0** — structure lift: 3-GEMM f2 kernel + fixed-slice Ozaki into the layered architecture (`a7a27ec`)
- [x] **M3** — SNP→block rule `assign_blocks` (per-chrom reset, dense renumber); 757/719 block counts (`f7f31c6`)
- [x] **M1** — GPU TGENO decode → Q/V/N (raw-value 2-bit; bit-for-bit oracle match) (`150bfb3`)
- [x] **M2** — missingness + filters (host-pure predicates; drop-equals-mask) (`1bbbad4`)
- [x] **M4** — per-block `f2_blocks[P×P×n_block]` via grouped strided-batched GEMMs (`75c6c10`)

---

## ▶ Next: M4.5 — Single-node multi-GPU precompute
*Depends: M4. Composes with M5. Refs: architecture §11.4 (the design), §9 (Resources/PerGpuResources), §12 (parity/determinism).*
- [ ] `Resources` / `PerGpuResources` structs (§9): per-device id, StreamPool, allocator, `CublasHandle`, optional `NcclComm` — one per `DeviceConfig::devices`.
- [ ] SPMG orchestration: one host thread + per-device stream per GPU, `cudaSetDevice` to switch, opportunistic `cudaDeviceEnablePeerAccess` (gated on `enable_peer_access` + `canAccessPeer`).
- [ ] SNP-range shard across G devices (static range to start; tile round-robin later with M5).
- [ ] Each device computes a **full-shape partial** `f2_blocks` + `Vpair` over its shard.
- [ ] Host-side combine: sum the G partials in **fixed device order** (`g=0..G-1`), reference precision — **NOT NCCL AllReduce** (§12 parity).
- [ ] Broadcast result to all devices (NCCL Broadcast or `cudaMemcpy` — order-independent).
- [ ] **Parity test:** result bit-identical across G (1 vs 2 GPUs) AND to the single-GPU reference, on real AADR.
- [ ] Confirm NCCL present on the box (else `cudaMemcpy` broadcast).

## M5 — Out-of-core SNP-tile streaming
*Depends: M1 + M4. Refs: architecture §11.1 (mechanism), §11.2 (VRAM budget), §7 (streams/pools).*
- [ ] `PinnedBuffer<T>` RAII (page-locked staging).
- [ ] Dedicated copy `Stream` + compute stream; double/triple-buffered overlap. **Pays the default-stream debt** (see Cleanup).
- [ ] Tile-size `T` derivation vs VRAM budget (largest T fitting ~60–70% free VRAM).
- [ ] Resident `f2_blocks` accumulator; per tile: decode → AF → per-block f2 partial → accumulate → discard.
- [ ] `extract-f2 --dry-run` prints `T_max`, #tiles, largest dataset fitting VRAM.
- [ ] **Fold in the GPU-dominant perf research** (workflow `wqd0a9o0l`): pinned async transfer (definite); GDS/cuFile + nvCOMP (feasibility-gated).

## M6 — QC / data-munging front-end (multi-dataset)
*Depends: M1 + M2. Refs: architecture §5 (S-2/S-1/S0′ table), §1 (scope).*
- [ ] S-2 merge plan: intersection (default) / union; declared-allele polarity; **drop** ambiguous/multiallelic (never strand-flip).
- [ ] S-1 conditional pre-pass (`--mind` / external `prune.in`, read not computed).
- [ ] S0′ harmonized + filtered tile produce; transversions-only option. No on-disk rewrite, no strand inference, no LD compute.

## M7 — On-disk cache + FST + AT2 parity
*Depends: M4. Refs: architecture §5 (FST), §12/§13 (parity).*
- [ ] On-disk `f2_blocks` cache, ADMIXTOOLS-compatible read/write.
- [ ] **FST** as a cheap add-on output of the same pass.
- [ ] **Install R + admixtools on the box → generate pinned AT2 goldens** (record R version / `RNGkind` / AT2 version / `blgsize` / `boot` / seed, §12) → **wire the AT2-parity gate** (the deferred acceptance criterion for M1–M7; ROADMAP §6 note).

---

## Phase 2 — Fit engine (operates on cached `f2_blocks`)
*Refs: architecture §5 (S3–S8), §11.4 (S8 multi-GPU), §12 (determinism near rank-deficiency).*
- [ ] S3 — f3/f4 contraction from `f2_blocks` (identity-based derivation).
- [ ] S4 — block jackknife → covariance `Q` + SEs (weighted by `Vpair`).
- [ ] S5 — qpWave rank test (SVD; batched `gesvdj` where dims allow, per-model `gesvd` fallback).
- [ ] S6 — qpAdm GLS fit (`potrf`/`trsm`/`gemm`) → weights, χ².
- [ ] S7 — p-values / nested-model test.
- [ ] S8 — model-space search (multi-GPU, embarrassingly parallel; **CUDA-graph capture per fit** — this stage is launch/host-bound, see perf research).
- ⚠️ Phase 2 is far more **host/launch-sensitive** than the precompute → lean on CUDA graphs + keep-resident (perf research `wqd0a9o0l`).

## Phase 3 — Interfaces
- [ ] CLI (`extract-f2`, `qpadm`, `qpdstat`).
- [ ] nanobind Python bindings.
- [ ] scikit-build-core wheels.

---

## 🧹 Cleanup / tech-debt backlog
*Set a good standard on M0–M4 before building M4.5. Items marked **[pre-M4.5]** block starting M4.5.*

Confirmed (from review):
- [ ] **[pre-M4.5] Default-stream debt** — `cuda_backend.cu` uses `stream_ = nullptr`; `cudaMemcpyAsync` on the default stream is effectively **synchronous** wrt the host. Introduce a dedicated RAII `Stream` (we already have `stream.hpp`) before the M5 overlap pipeline. (§7, §11.1)
- [ ] **[pre-M4.5] `0.80` VRAM-budget magic number** → named `constexpr kMaxVramUtilizationFraction` in `config.hpp`. (§11.2; ROADMAP §4 — "no literal may survive")
- [ ] **[pre-M4.5] `compute_f2_blocks` monolith (>100 lines)** → extract testable private helpers: bucket/"spike-rule" grouping, VRAM-footprint estimation, chunked launch. (§2 separation; §13 testability)
- [ ] **Casting noise** — repeated inline `static_cast<std::size_t>(...)`; sanitize + cast inputs **once** into named `const` locals at function top, use thereafter.
- [ ] **(pending)** more from the cleanup-review fan-out (workflow in flight) — folder org, further magic numbers, DRY, layering conformance.

---

## 🔧 Cross-cutting tracked tasks
- [ ] **Merge M4 → `main`** (after the pre-M4.5 cleanup lands).
- [ ] **Install R + admixtools** on the box → AT2 goldens (gates M7 parity).
- [ ] **GPU-dominant perf research** (workflow `wqd0a9o0l`) → fold into M5 (transfer) + Phase 2 (CUDA graphs).
- [ ] **Nsight Systems profiling baseline** (§11.3) — empirically confirm host-vs-device bottlenecks before optimizing.
- [ ] Prune merged feature branches (`m0-f2-scaffold`, `m1-decode-af`, `m2-filters`, `m3-block-partition`) once merged to `main`.

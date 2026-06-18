# `steppe` — TODO / task tracker

Living, checkable companion to [`ROADMAP.md`](ROADMAP.md) (the **order & rationale**) and [`architecture.md`](architecture.md) (the **design & standards**). This is the **granular next-steps checklist for future-us** — keep it current as milestones land. Update the checkboxes; move finished items to "Done".

**Big picture (2026-06-17):** the **precompute half exists** — M0–M4 done (validated 3-GEMM f2 kernel → data front-end → filters → per-block `f2_blocks` tensor) **+ the entire before-M4.5 cleanup (B1–B27)**, all merged to `main` @ `1fbb417`, clean from-scratch build 24/24 ctest green, codebase ~9.5/10 on the fixed axes. The **fit engine (Phase 2) does not exist yet**, and the precompute still has scale/robustness milestones (M4.5–M7). **Next = M4.5** (single-node multi-GPU) — the moment to spin up the **RTX PRO 6000**.

---

## ✅ Done (Phase 1 precompute, M0–M4)
- [x] **M0** — structure lift: 3-GEMM f2 kernel + fixed-slice Ozaki into the layered architecture (`a7a27ec`)
- [x] **M3** — SNP→block rule `assign_blocks` (per-chrom reset, dense renumber); 757/719 block counts (`f7f31c6`)
- [x] **M1** — GPU TGENO decode → Q/V/N (raw-value 2-bit; bit-for-bit oracle match) (`150bfb3`)
- [x] **M2** — missingness + filters (host-pure predicates; drop-equals-mask) (`1bbbad4`)
- [x] **M4** — per-block `f2_blocks[P×P×n_block]` via grouped strided-batched GEMMs (`75c6c10`)

---

## ✅ M4.5 — Single-node multi-GPU precompute — DONE (2026-06-17, branch `m4.5-multigpu`)

> **COMPLETE & bit-identity-proven on rtxbox (2× RTX PRO 6000).** Built via two sequenced per-file workflows (`agentscripts/m4.5-scaffold.js` → `agentscripts/m4.5-multigpu.js`), 8 commits `d81d2a1`→`18b7801`, clean from-scratch build, **30/30 ctest green**. Scaffold: `STEPPE_CUDA_WARN` (non-throwing tagged degrade), `BackendCapabilities`+probe, `prefer_p2p_combine`, device-ordinal+`MathModeScope`, `CudaBackend(device_id)`+`cudaSetDevice`. Algorithm: `Resources`/`PerGpuResources`+builder, **block-aligned shard** (whole contiguous block ranges → each block computed on one device), **host-staged fixed-order combine** (portable baseline), **opt-in `cudaMemcpyPeer` P2P device-combine** (real stock-driver P2P, ran on rtxbox). **Parity (test_f2_multigpu_parity):** for the production `EmulatedFp64{40}` path, G==2 host-staged AND P2P are **bit-identical** to single-GPU on `derived_acc`+`derived_full`; native `Fp64` at G≥2 matches to oracle tol (rel ≤2.87e-15 = §12 `[UNCERTAIN]` batched-GEMM batchCount-sensitivity, correctly scoped). NEVER NCCL AllReduce. **NOT yet done: the consumer-5090 graceful-degrade validation** (`can_access_peer==false` → host-staged + tag) = the "5090 after" job; the parity test's tier-assertions are PRO-specific and need a tier branch to run on the vast box. **Pending: merge `m4.5-multigpu` → `main`.**

*Refs: architecture §11.4 (design), §9 (Resources/PerGpuResources), §12 (parity); the audit `docs/cleanup/00-overview.md` §(2) "CAPABILITY-TIER COHERENCE — the ONE unified design"; the ⚡ section's capability-tier table + box-role split.*

**Goal.** Shard SNP work across the G devices; each computes a full-shape **partial** `f2_blocks`+`Vpair`; combine the partials **once, host-side, in fixed device order** (`g=0..G-1` = `DeviceConfig::devices` order) in reference precision; broadcast back. **Bit-identical across G AND to the single-GPU reference (§12).** This is *the* milestone that introduces the **capability-tier machinery** — it's the first capable-vs-budget fork (P2P device-combine vs host-staged). **Build the host-staged baseline first; P2P is an opt-in fast-path.**

*Resources / device threading (Theme-1 debt M4.5 calcifies — these land here):*
- [ ] **`Resources` / `PerGpuResources`** (§9), one per `DeviceConfig::devices`, all RAII: `{int device_id; Stream stream; CublasHandle blas; DeviceAllocator* allocator; (NcclComm comm);}`. Each binds `cudaSetDevice(device_id)`; **replicate B1's (stream,workspace)-owning `CublasHandle` per device.**
- [ ] **`CudaBackend` device_id threading** — thread `device_id` into the ctor + `cudaSetDevice` (audit F19/F20: cleanup left the backend single-current-device). The SPMG prerequisite.

*Capability probe = the §(2) unified design, lands HERE:*
- [ ] **`BackendCapabilities` probe** at `Resources`/`build()` assembly, per device: compute capability + free/**total** VRAM (`cuda_backend` already calls `cudaMemGetInfo` and **discards `total_b`** — capture it), `cudaDeviceCanAccessPeer` (P2P), emulated-FP64-honorable state (B2's predicate). Result = a small value in `Resources`.
- [ ] **Non-throwing, tagged-degrade path DISTINCT from `STEPPE_CUDA_CHECK`** (the `device-cuda-check` CAP-1/CAP-2 finding) — `canAccessPeer="no"` / `cudaErrorPeerAccessAlreadyEnabled` are EXPECTED on the budget box ⇒ `STEPPE_LOG_WARN`-and-degrade, **NEVER throw**. Add a `STEPPE_CUDA_WARN`-style variant alongside `check.cuh` (uses the `log.hpp` from B7).

*Shard → partials → combine:*
- [ ] **SNP-range shard** across G devices (static range to start; tile round-robin later with M5); each device runs `compute_f2_blocks` on its shard → its own **full-shape partial** `f2_blocks`+`Vpair`.
- [ ] **Host-side fixed-order combine = the PORTABLE PARITY BASELINE** (works on the budget 5090): gather G partials to host, sum in fixed device order, reference precision — **NOT NCCL AllReduce** (its order varies with G, breaks §12). Broadcast back (NCCL Broadcast or `cudaMemcpy`, order-independent).
- [ ] **Optional capable-path P2P device-combine** (gated on `prefer_p2p_combine` + `cudaDeviceCanAccessPeer`): device 0 pulls each peer's partial via `cudaMemcpyPeer` (byte-exact DMA) and sums in the SAME fixed `g=0..G-1` order **on-device — bit-identical** to host-staged (transport only moves bytes; never NCCL AllReduce). **Tagged fallback** on the budget box: log *"P2P combine unavailable (no peer access) → host-staged fixed-order combine"*. Clean stock-driver on the **PRO 6000** (IOMMU/ACS off, no NVLink this SKU); aikitoria-patch-only on 2× 5090 (dev-only, NEVER production). steppe is P2P/NVLink-insensitive by design (combine is kB–MB, off critical path).
- [ ] Confirm NCCL present on the box (else `cudaMemcpy` broadcast).

*Knob separation (audit §(2).3) + the gate:*
- [ ] **Two knob types:** override-intent → `DeviceConfig` (`enable_peer_access` exists; **add `prefer_p2p_combine`**); discovered-capability + which-path tag → `Resources`/result metadata (**NEVER on `F2BlockTensor`** — keep it pure numeric).
- [ ] **Parity test (the GATE):** multi-GPU combine bit-identical across G (1 vs 2 GPUs) AND to the single-GPU reference, on real AADR (`derived_full` P=768). Extend the `f2_determinism` pattern to the multi-device combine.

**Files that change (audit §(2)):** `config.hpp` (override-knob banner + `prefer_p2p_combine`; `deterministic` landed in B9) · `backend.hpp` (`BackendCapabilities probe()` + per-device contract) · `cuda_backend.cu` (probe + capture `total_b` + `cudaSetDevice`/`device_id` + EmulatedFp64-not-honorable tagged degrade) · `f2_from_blocks.cpp` (thread `Resources`, host the `canAccessPeer`-gated combine *policy*) · `handles.hpp` (record device ordinal) · `log.hpp` (tagged-warn sink, exists from B7) · `device/cuda/check.cuh` (the non-throwing `STEPPE_CUDA_WARN` probe variant).

**VRAM note (B26):** `f2_blocks`+`Vpair` = `2·P²·n_block·8`. Multi-GPU **REPLICATES the full-shape partial per device** — it shards the **input (SNP) work** for throughput + the combine, it does NOT reduce the per-device **output** footprint. Per-device `f2_blocks` ceiling stays P-bound (~P=1360 @70% on a 5090, ~P=2356 on the 96 GB PRO 6000). Large-P input streaming is M5.

**Where it's testable:** the **host-staged combine + parity test works on the budget 2× 5090** (do the bulk there). The **P2P device-combine path needs the PRO 6000** (clean stock-driver P2P). Start with a short design step (Resources + capability-probe + the combine seam) before code.

## M5 — Out-of-core SNP-tile streaming
*Depends: M1 + M4. Refs: architecture §11.1 (mechanism), §11.2 (VRAM budget), §7 (streams/pools).*
- [ ] `PinnedBuffer<T>` RAII (page-locked staging).
- [ ] Dedicated copy `Stream` + compute stream; double/triple-buffered overlap. **Pays the default-stream debt** (see Cleanup).
- [ ] Tile-size `T` derivation vs VRAM budget (largest T fitting ~60–70% free VRAM).
- [ ] Resident `f2_blocks` accumulator; per tile: decode → AF → per-block f2 partial → accumulate → discard.
- [ ] `extract-f2 --dry-run` prints `T_max`, #tiles, largest dataset fitting VRAM.
- [ ] **CPU-light ingest is the M5 acceptance criterion** — PinnedBuffer + dedicated copy stream + double-buffer overlap (see ⚡ section below). Pin only small slots (RLIMIT_MEMLOCK cap); `cudaMallocAsync` pool; O_DIRECT/fadvise; resident accumulator. (GDS ruled out on vast; nvCOMP deferred to M7.)

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

> **✅ ALL before-M4.5 cleanup COMPLETE (2026-06-17).** B1–B27 done — **26 fix commits** (`659a8e5`→`8246800`; B26 folded into B5/`faeb1f4`), each independently verdict-gated on the box. **Clean from-scratch build green, 24/24 `ctest`** (+11 tests since Phase 1: determinism, emu-honorability, empty/int-max guards, device_buffer, decode, config, backend_factory, f2_from_blocks, geno_reader, snp_reader, gather). The four gap-themes are closed: §12 determinism *holds*, fail-fast replaces fail-silent at every seam, the single-source homes are real, the VRAM budget counts both tensors. Codebase ~8.0 → ~9.5 on the fixed axes. **Deferred (by design):** L1–L19 (M5/M6/Phase-2-timed or cosmetic); B5/B6 *at-scale* validation + AT2 goldens → the PRO-6000 session. Per-phase detail below.

> **Phase 1 ✅ COMPLETE (2026-06-16)** — B7, B1, B2, B3, B4, B5, B6 all fixed, build + `ctest` green (**13/13**, +5 new tests), committed `659a8e5`→`23bb873` on `m4-perblock-f2`, each gated by an independent verdict re-verified on the box. Highlights: **§12 determinism now *actually holds*** (B1 — `cublasSetStream` no longer discards the workspace; new `f2_determinism` test asserts bit-identical `f2`+`vpair` run-to-run on M0 *and* M4 at `EmulatedFp64{40}`); **EmulatedFp64 dynamic-mantissa trap closed** (B2 — observable downgrade-with-tag, verified both build lanes); **VRAM budget counts both `f2`+`vpair`+workspace** (B5/B26); **M0 diagonal fixed + mutation-tested** (B4); `block_ranges` single-homed+validated (B3); grid-dim clamps + feeder re-orient (B6); real `launch_config.hpp`/`host_device.hpp`/`log.hpp` homes, duplicate `cdiv` gone (B7). **B5/B6 at-scale validation deferred to a PRO-6000 session.** Authoritative master backlog: `docs/cleanup/00-overview.md` (B1–B27 / L1–L19). **Next: Phase 2 = B8–B27** (MED fail-fast / parser-robustness; incl. B17 HIGH heap-overflow).
*From the `wn01sl1wz` review (5 lenses + synthesis). Verdict: M0–M4 is **unusually disciplined** (layering compiler-enforced, RAII textbook, shared `__host__ __device__` primitives prevent oracle/GPU divergence, config.hpp exemplary); debt is **concentrated, not pervasive** and clusters into two themes M4.5 would calcify. **No folder reorg needed.** Full findings: workflow `wn01sl1wz` result.*

### A. Before M4.5 (pay first — sets the standard; M4.5 replicates/calcifies these)
*Theme 1 — resource-ownership debt M4.5 hits immediately (all touch the `CudaBackend` ctor — do together):*
- [ ] **[high] Default-stream debt** — `cuda_backend.cu` uses `cudaStream_t stream_ = nullptr` (RAII `Stream` exists, unused) ⇒ every `cudaMemcpyAsync` is host-synchronous + no lane to per-device-ify. Fix: owning `Stream stream_;` member (declared before `blas_`), wire `blas_{stream_.get()}` + launches + memcpys + `stream_.synchronize()`; keep §12 single-stream sync points.
- [ ] **[med] No `device_id` / `cudaSetDevice`** — backend ignores `DeviceConfig`, runs on the current device. Fix: thread `device_id` into the ctor + `cudaSetDevice`. The SPMG prerequisite.
- [ ] **[med] Shared cuBLAS handle math-mode mutated unscoped** — determinism hazard once M4.5 Fp64 parity-recompute coexists with EmulatedFp64 on one handle. Fix: engage once per compute call; drop the redundant re-engage in `run_f2_gemms`; add an RAII `MathModeScope` that restores prior mode.

*Theme 2 — the `compute_f2_blocks` monolith + duplicated block-range scan:*
- [ ] **[high] `compute_f2_blocks` ~170-line monolith** (6 concerns; 3 are pure host int-math buried in a CUDA TU). Fix: extract host-pure, CPU-testable helpers — `bucket_blocks_by_padded_size`, `max_blocks_per_chunk` (consumes the VRAM constant), block-range scan (→ core). Method becomes alloc→ranges→feeder→per-bucket launch (~40 lines). Unit-test the helpers.
- [ ] **[high] Block-range scan duplicated in BOTH backends** (`cuda_backend.cu:136-147` ≡ `cpu_backend.cpp:214-226`) — re-derives the partition layout twice (DRY/layering, §2/§8). Fix: `block_ranges(block_id, M, n_block) → vector<BlockRange>` in `core/domain/block_partition_rule.{hpp,cpp}` (host-pure, sanctioned home); both backends call it; unit-test it.

*High-severity correctness/precision:*
- [ ] **[high] `0.80` VRAM literal → `kMaxVramUtilizationFraction`** (config.hpp) — AND the budget is subtly unsound: it doesn't subtract the 64 MiB cuBLAS workspace (`kCublasWorkspaceBytes`) before applying the fraction. Fix both.
- [ ] **[high] EmulatedFp64 silently runs the REJECTED dynamic-mantissa trap when `STEPPE_HAVE_EMU_TUNING=OFF`** — `cublasSetMathMode(…FIXEDPOINT)` engages emulation but the FIXED-slice pin is compiled out ⇒ cuBLAS defaults to ~60-bit dynamic (no speedup, voids §12 bit-stability) while still reporting `EmulatedFp64`. Fix: when tuning off + EmulatedFp64 requested, do NOT silently engage — `#error` / fall back to native Fp64 + logged warning / throw `INVALID_CONFIG` (§9). Foot-gun closed observably.

*Conformance (small but real):*
- [ ] **[med] `launch_config.hpp` is a phantom** — config.hpp/f2_estimator.hpp name it the single home for `cdiv`/`grid_for`, but it doesn't exist ⇒ decode_af_kernel.cu re-rolled its own `cdiv`. Fix: create `core/internal/launch_config.hpp`, move `cdiv`+`grid_for` there, route the decode kernel through `core::cdiv`.
- [ ] **[med] `CpuBackend` in `steppe::core` but compiles into `steppe_device`** — namespace/layer mismatch (tests forward-declare with mismatched namespaces). Fix: move `CpuBackend`/`make_cpu_backend` to `steppe::device`; update the 3 test forward-decls.
- [ ] **[med] `STEPPE_HD` macro doubly-defined** (f2_estimator.hpp + decode_af.hpp, no shared home, no `#undef`). Fix: one `core/internal/host_device.hpp` both include.

### B. Later (real but cosmetic / M5-timed)
- [ ] casting sanitize-once (folds into the block_ranges extraction) · `/4`,`&3` packing literals in 3 TUs → `code_byte_index`/`kCodesPerByte` · delete duplicate `cdiv_l/cdiv_i` · fix stale "16×16" comment (it's 32×8) · bare chrom codes 23/24/90 → named constants · triplicated teardown-warning macros → one shared · restore `DeviceBuffer::view()` + new M4.5 wrappers span-first · clamp `grid.z` to 65535 · standardize device flat-index on `size_t` · extract duplicated CPU reference f2-pair body · split `read_ind` (parse/select/order) · single-source the Tf32→native mapping · factor `per_block_chunk_bytes()` (kill bare `4u`).

### C. Good patterns to PRESERVE (the standard for every future kernel)
Shared `__host__ __device__` per-element primitives (oracle≡GPU) · every constant doc-commented w/ arch+ROADMAP cite + measured rationale · textbook move-only RAII (replicate for M4.5 `NcclComm`/`StreamPool`/`PinnedBuffer`) · narrow `launch_*` wrappers + post-launch checks + allocation allowlist · "engage precision once, shared by all paths" · every deferred/duplicated thing carries a why+what-replaces-it comment · thin CUDA-free orchestration seam (`f2_from_blocks.cpp`) · predicate-as-shared-primitive in the io leaf.

---

## ⚡ Keeping it GPU-dominant (perf research `wqd0a9o0l`)
*Answer to "a GPU thing shouldn't be punished by a bad CPU." Host-CPU sensitivity has 3 surfaces: ingest (#1), launch overhead (#2), orchestration (#3). Verdict: **#1 and #3 are M5/M4.5 problems solvable now with definitely-works-on-vast primitives; #2 is a Phase-2 problem for CUDA Graphs.** Guiding principle: **the CPU becomes a sequencer + metadata janitor — never a data mover or a per-SNP reducer.** Full synthesis: workflow `wqd0a9o0l`.*

**Environments:** routine build/test/dev runs on the cheap **vast box** (CUDA 13, consumer 5090) — the design must stay fast *there* (pinned double-buffer baseline, **no GDS**). For the "for sure need to" moments — the **measure-first Nsight/`ncu` profile, GDS experiments, and final perf validation** — use the **RTX 6000 full-host box** (professional card ⇒ no GeForce GDS lock; root/host ⇒ `nvidia-fs` + a GDS-qualified FS are controllable; full perf counters). Run it sparingly (expensive). ⇒ **GDS upgrades from "not worth it" to a real research bet on that box** (target: the large `.f64`/M7 cache), while the portable pinned pipeline stays the baseline everywhere. *(Which RTX 6000 — Blackwell sm_120 vs Ada sm_89 — TBD; matters for perf-representativeness vs the 5090.)*

**Ranked plan:**
1. **[adopt now] Measure-first Nsight gate** — `nsys profile --trace cuda,nvtx,osrt` on the real end-to-end path; classify ingest-bound (un-overlapped copy) vs launch-bound (GPU-idle gaps) vs orchestration-bound (host gaps). Free, container-safe (the CUDA+NVTX trace needs no SYS_ADMIN), and it orders everything below. **Don't optimize against a guess.**
2. **[M5] Pinned double-buffered ingest = the CPU-insensitive spine** — replace `geno_reader`'s pageable `ifstream` per-individual gather with large sequential reads into `PinnedBuffer` ring slots + `cudaMemcpyAsync` on a **dedicated copy stream**, overlapped with the ~0.3 s GEMM via `Event`; end-to-end → `max(read, gpu)` not their sum. Copy-engine DMA moves the data; the host just sequences. **Pin only small slots (tens–low-hundreds MB), never the 4 GB .geno** (RLIMIT_MEMLOCK cap ~0.5–1 GB in containers); fallback ladder `cudaMallocHost → malloc+cudaHostRegister → pageable+warn`. Plus `cudaMallocAsync` pool (release threshold MAX), hoist the per-tile block-scan/bucketing out of the loop, keep `f2_blocks`+handle resident, `O_DIRECT`/`fadvise(DONTNEED)` cold reads, `--ulimit memlock=-1` launch template + probe at `build()`.
3. **[M4.5] Fuse the cheap in-tile filter into the decode device pass** — make `filter_decision.hpp` predicates `__host__ __device__`, apply the keep-mask as a validity-mask multiply inside `decode_af`; deletes a D2H + an O(P·M) host reduction + an O(M) host predicate loop per tile (the clearest host liability in `snp_filter.cpp` today). Gate on bit-for-bit parity vs `test_filter_oracle`.
4. **[Phase 2] CUDA Graphs for the fit engine** — capture the per-model ~6–10 tiny-kernel fit once, re-launch via `cudaGraphExecUpdate` across thousands of models (NVIDIA 1.5–3× + jitter collapse — exactly the many-tiny-launch regime the ~1.6× host inflation lives in). Prereqs: ZERO `cudaMalloc` during capture (pre-allocate all buffers + cuSOLVER/cuBLAS workspaces + devInfo), topology-class graph cache, RAII graph wrappers. Gated on the Nsight profile proving launch-bound.
5. **[M7 research bet] nvCOMP field-split cache compression** — aggressive on the V mask (~46×) and N counts (~4.9×), near-raw on the FP64 f2 payload (~1.06×); GPU SM-decompress (userspace `.so`, works on vast). Needs a bit-exact round-trip golden. Only once M7's cache exists.

**Decided against (do NOT re-litigate / don't engineer for the vast 5090):**
- **True GPUDirect Storage / cuFile** — three independent disqualifiers on vast consumer 5090 (GeForce = compat-mode-only by NVIDIA policy; `nvidia-fs`/NVMe-P2PDMA need host-kernel access we don't have in a rented container; instance data is on **OverlayFS**, not a GDS-qualified FS). cuFile silently degrades to POSIX `pread`+copy = identical to today. The pinned double-buffer **is** the correct fallback (already §11.1). Reconsider only on a datacenter (Quadro/Tesla) box for the large `.f64` cache.
- Graphing the **precompute** (bandwidth-bound, ≤1.4×, and the current code is capture-hostile) · moving **block-binning to GPU** (one-time host metadata scan, saves µs) · "optimizing" the `f2_from_blocks` 2-line forwarder · the **Blackwell HW decompression engine** (datacenter-only; nvCOMP SM-decompress is the path) · compressing the **FP64 payload** itself (~1.06×) · a transposed `.geno` layout (no ratio gain) · cuSOLVERMp distributed solve (qpAdm matrices are tiny).

**Open questions (the Nsight gate + a tiny spike answer most):** is the ~1.6× actually ingest/launch/orchestration-bound? real RLIMIT_MEMLOCK in the vast container? vast volume read bandwidth + does O_DIRECT succeed on overlay? does `cudaMemcpyAsync` overlap materialize on the 5090? can cuSOLVER `gesvdj`/`potrf` + Ozaki-emulated GEMM be stream-captured on this 13.0 box?

### Re-verified adversarially (`wxz1fiiln`) — capability tiers + box roles
*"Officially unsupported ≠ impossible" — **confirmed**. The aikitoria fork (`open-gpu-kernel-modules @610.43.02-p2p`; geohot→nimlgen→valdemardi lineage) enables REAL BAR1 PCIe P2P on full-host 5090s (measured **~55 GB/s**). **But NCCL #1637 ≠ aikitoria:** the NCCL 2.26 fix merely makes NCCL FUNCTION over slow SHM on 5090s — it does NOT enable P2P. And the decisive nuance: **steppe is P2P/NVLink-INSENSITIVE by design** (§11.4 keeps cross-GPU traffic to a kB–MB `f2_blocks` broadcast + the parity reduction host-side, fixed-order) → P2P is architectural cleanliness, not throughput.*

**Verdicts (full-host RTX PRO 6000 = primary; vast 5090 = budget fallback):**
- **FLIP on the PRO 6000** (driver-policy/container blocks, not silicon): true **GDS/cuFile** (Quadro-class CC 12.0) · full **`ncu --set full`** counters (`NVreg_RestrictProfilingToAdminUsers=0`) · **O_DIRECT** · **RLIMIT_MEMLOCK** (memlock=unlimited).
- **STAND** (genuine): consumer-5090 **GDS dead** (GeForce GSP-firmware lock survives full host + NVMe-P2PDMA + every patch; no community GeForce-GDS exists — the P2P patch is *orthogonal* to storage RDMA) · **HW Decompression Engine** absent on GB202 / both boxes (missing silicon → nvCOMP SM-decompress) · **cuSOLVERMp** wrong workload shape (breaks §12).
- **CONDITIONAL — P2P:** clean **stock-driver** on the PRO 6000 (no NVLink on this SKU — just IOMMU/ACS off); aikitoria-patched on full-host 5090s (out-of-tree, rebuild per driver — **dev convenience only, NEVER the production parity path**).

**Capability tiers** (build for capable, degrade with an explicit logged tag — every lever is **parity-neutral**: data-movement/observability only):
| Lever | Capable path (PRO 6000 / full-host) | Budget-5090 fallback + logged reason |
|---|---|---|
| **M4.5 combine** | P2P device-resident: GPU0 pulls peer partials via `cudaMemcpyPeer` (byte-exact DMA), sums fixed `g=0..G-1` on-device | host-staged fixed-order combine — *"P2P combine unavailable (no peer access) → host-staged fixed-order combine"* |
| **M5/M7 ingest** | true **GDS** cuFile DMA NVMe→VRAM (probe `DMA_BUF_SUPPORTED`/`cuFileBufRegister`, **not** `gdscheck`) | pinned double-buffer POSIX-pread — *"GDS unavailable (GeForce GPU-class / OverlayFS) → POSIX pread into pinned double-buffer"* |
| **Profiling** | `ncu --set full` SoL/Roofline | nsys CUDA+NVTX only — *"ncu counters unavailable (no SYS_ADMIN) → nsys trace only; deep dive deferred to PRO 6000"* |
| **M7 decompress** | nvCOMP SM-decompress | identical (box-agnostic — HW DE absent on both; no tier difference) |

**Parity:** bit-identical on both paths (combine sums the same fixed order; `cudaMemcpyPeer` is byte-exact; GDS only changes *how* bytes reach VRAM). §12 holds; the AT2-golden + native-FP64 oracle gate both boxes so a capable-path lever can never silently change a reported number.

**Box role-split:** PRIMARY = **RTX PRO 6000 Blackwell** (sm_120 ⇒ perf-representative, *resolves the TBD*; 96 GB ⇒ single-GPU P_max ~3,331@70% vs ~1,923 on a 5090; owns the perf gate + GDS + official P2P). BUDGET = **vast 2× 5090** (cheap parallel dev + the graceful-degradation TEST TARGET that exercises every fallback tag). One `sm_120` build + one §12 contract serve both; the Ozaki ratio transfers. **M5 input-streaming is NEVER deferred** (genotype input ≫ 96 GB); 96 GB only defers OUTPUT-accumulator sharding for P≤~3,300 — M4.5 sharding stays needed for P>~3,300 + throughput.

**Per-milestone design changes:** M4.5 — add the optional `canAccessPeer`-gated P2P device-combine (host-staged stays the baseline) · M5 — runtime-probed GDS lane (NOT `gdscheck`) + `build()` RLIMIT_MEMLOCK probe · M7 — GDS for the `.f64` cache + nvCOMP SM-decompress · **cross-cutting — a capability probe + capability-tagged results** (every run records which path it took + why it degraded), slotting into `DeviceConfig`/`Resources`; `--dry-run` reports per-box P_max.

## 🔧 Cross-cutting tracked tasks
- [x] **Merge M4 → `main`** — DONE @ `1fbb417` (M0–M4 + before-M4.5 cleanup B1–B27, 24/24 ctest green).
- [ ] **Install R + admixtools** on the box → AT2 goldens (gates M7 parity) → the PRO-6000 session.
- [x] **GPU-dominant perf research** (`wqd0a9o0l`) — DONE; results folded into the ⚡ section (M5 pinned ingest, M4.5 filter-fusion, Phase 2 graphs, M7 nvCOMP; GDS ruled out on vast).
- [ ] **[adopt now] Nsight Systems measure-first gate** (§11.3) — classify ingest/launch/orchestration-bound before building any perf lever (see ⚡ section). The prerequisite that orders the perf work.
- [ ] Prune the merged feature branches (`m0-f2-scaffold`, `m1-decode-af`, `m2-filters`, `m3-block-partition`, `m4-perblock-f2`) — all merged to `main`.

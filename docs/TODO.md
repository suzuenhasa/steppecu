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

## 🔧 Cross-cutting tracked tasks
- [ ] **Merge M4 → `main`** (after the pre-M4.5 cleanup lands).
- [ ] **Install R + admixtools** on the box → AT2 goldens (gates M7 parity).
- [x] **GPU-dominant perf research** (`wqd0a9o0l`) — DONE; results folded into the ⚡ section (M5 pinned ingest, M4.5 filter-fusion, Phase 2 graphs, M7 nvCOMP; GDS ruled out on vast).
- [ ] **[adopt now] Nsight Systems measure-first gate** (§11.3) — classify ingest/launch/orchestration-bound before building any perf lever (see ⚡ section). The prerequisite that orders the perf work.
- [ ] Prune merged feature branches (`m0-f2-scaffold`, `m1-decode-af`, `m2-filters`, `m3-block-partition`) once merged to `main`.

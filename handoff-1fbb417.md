# steppe — cold-start handoff @ `main` `1fbb417`

Everything a fresh session needs to resume. Supersedes `handoff-ff81498.md`. Canonical spec: `docs/architecture.md`; build order + status: `docs/ROADMAP.md`; living task list: `docs/TODO.md`; the per-file audit + master cleanup backlog: `docs/cleanup/00-overview.md`. **Read those before writing code.**

---

## 0. TL;DR
`steppe` = GPU/CUDA-13/Blackwell (sm_120) reimplementation of ADMIXTOOLS 2 f-statistics (precompute-once / fit-many), senior-engineer quality bar. **Status: Phase 1 (the `f2_blocks` precompute engine) M0–M4 DONE + the entire before-M4.5 cleanup (B1–B27) DONE, all merged to `main` @ `1fbb417`. Clean from-scratch build, 24/24 ctest green. Codebase ~9.5/10 on the fixed axes.** **NEXT = M4.5 (single-node multi-GPU precompute)** — and it's the moment to spin up the **RTX PRO 6000**.

---

## 1. Boxes
- **Budget/daily box (vast.ai):** `ssh -i ~/.ssh/id_vastai -p 43215 root@78.92.24.57` — 2× **RTX 5090** (sm_120, 32 GB), **CUDA 13.0.88**, driver 580, 100 GB disk. Repo rsync'd to `/workspace/steppe` (NOT git on the box); data at `/workspace/data/aadr/{derived_acc P=50/M=100k, derived_full P=768/M=584131, derived_big, derived_big2 P=2416, derived_all P=4266, raw=the v66 TGENO triple}`. **Ephemeral — update this when it changes** (see [[steppebox5090]] memory for the box-selection checklist: CUDA≥13/driver≥580, disk≥80 GB, rsync may ship as a 0-byte stub).
- **Capable box (spin up for M4.5): RTX PRO 6000 Blackwell (sm_120, 96 GB, full host).** Same arch ⇒ perf-representative. Unlocks (all DEAD on the budget 5090): true GDS, full `ncu`/`nsys`, official stock-driver P2P (IOMMU/ACS off), 96 GB. **Four threads converge here:** M4.5 P2P combine · B5/B6 at-scale · the `ncu`/`nsys` profile · installing **R + admixtools** for the **AT2 goldens** (the real parity gate, deferred until now).
- **DISK GOTCHA:** fail-fast/assert tests `abort()` → kernel dumps a ~404 MB core PER abort into `/var/lib/vastai_kaalia/data/`; 313 filled the disk and broke a clean build with bogus "No space left on device". **Run tests with `ulimit -c 0`**; reclaim with `rm -f /var/lib/vastai_kaalia/data/core-*`. Always do a periodic clean-build stamp (`rm -rf build`) — incremental builds mask such issues.

## 2. Working rules (hard)
NOTHING builds/runs locally — develop locally → rsync to `/workspace/steppe` → build/test on the box. NO synthetic data for any precision/accuracy claim (real AADR only). git identity `suzuenhasa <suzu@enhasa.co>`. Commit only on green, per-milestone, ROADMAP §6 message format. Don't reuse the old MVP at `/home/suzunik/vahaduo/qpadm/steppe`.

## 3. Build & run (on the box)
```
ssh -i ~/.ssh/id_vastai -p 43215 root@78.92.24.57
cd /workspace/steppe
ulimit -c 0                                   # disable core dumps (disk gotcha)
cmake -S . -B build -GNinja && cmake --build build
ctest --test-dir build --output-on-failure
```
Dev loop: edit locally → `rsync -az --delete-after --exclude .git --exclude build --exclude aadr -e "ssh -i ~/.ssh/id_vastai -p 43215" /home/suzunik/steppe/ root@78.92.24.57:/workspace/steppe/` → build/test on box → commit locally on green. **24/24 ctest green at HEAD** (~156 s; `filter_oracle` ~90 s = the CPU long-double parity oracle, test-only).

## 4. State (main @ 1fbb417)
- **Done:** M0 (3-GEMM f2 kernel + fixed-slice Ozaki) · M1 (GPU TGENO decode → Q/V/N, bit-for-bit oracle) · M2 (filters) · M3 (SNP→block `assign_blocks`) · M4 (per-block `f2_blocks[P×P×n_block]` via grouped strided-batched GEMMs) · **before-M4.5 cleanup B1–B27** (26 commits + B26-in-B5).
- **Cleanup highlights:** §12 determinism now *holds* (B1 — `CublasHandle` owns (stream,workspace); `f2_determinism` test asserts bit-identical run-to-run) · EmulatedFp64 dynamic-trap closed (B2) · VRAM budget counts both `f2`+`vpair` (B5/B26) · `block_ranges` single-homed+validated (B3) · M0 diagonal fixed (B4) · grid-dim clamps (B6) · real `core/internal/{launch_config,host_device,log}.hpp` homes (B7) · parser fail-fast (B14/B15/B17/B18) · etc.
- **The science (validated, real AADR, 2× 5090):** f2 = 3-GEMM (`G=Q·Qᵀ`, `Vpair=V·Vᵀ`, `R=[Q²;Hc]·Vᵀ`); **fixed-slice Ozaki** is the f2-GEMM precision — 40-bit ≈ native (2.2e-11) at 7–13×, 32-bit (8.6e-9) at 8.5–17.5×; **dynamic mantissa is the rejected parity trap**; native FP64 = oracle + numerator/divide. Benchmark precision on REAL data only.
- **The audit:** `docs/cleanup/` = 28 per-file deep reviews + `00-overview.md` (master backlog B1–B27 done; **L1–L19 deferred**). Overall was 8.0/10 → ~9.5 on the fixed axes.
- **agentscripts/** = the 15 reusable Workflow orchestration scripts (research/review = parallel read-only fan-out; code-fix = strictly-sequential fixer+independent-verdict, build+ctest-gated, commit-green/revert, `ulimit -c 0`). `fix-pass-phase1.js`/`fix-pass-phase2.js` are the fix-pass pattern.

---

## 5. ▶ NEXT MILESTONE — M4.5: single-node multi-GPU precompute
*Refs: architecture.md §11.4 (the design), §9 (Resources/PerGpuResources), §12 (parity); the audit's `00-overview.md` §(2) "CAPABILITY-TIER COHERENCE — the ONE unified design"; the [[steppebox5090]] capability map + the aikitoria/P2P verdict.*

**Goal.** Shard SNP work across the G devices; each computes a full-shape **partial** `f2_blocks`+`Vpair`; combine the partials **once, host-side, in fixed device order** (`g=0..G-1` = `DeviceConfig::devices` order) in reference precision; broadcast back. **Result must be bit-identical across G AND to the single-GPU reference (§12).** This is *the* milestone that introduces the **capability-tier machinery** (the audit §(2) design) because it's the first capable-vs-budget fork (P2P device-combine vs host-staged).

**Plan / sub-tasks (build host-staged baseline first; P2P is an opt-in fast-path):**
1. **`Resources` / `PerGpuResources` (architecture §9).** One per `DeviceConfig::devices`, all RAII: `{int device_id; Stream stream; CublasHandle blas; DeviceAllocator* allocator; (NcclComm comm);}`. Each binds `cudaSetDevice(device_id)`; replicate B1's (stream,workspace)-owning `CublasHandle` per device.
2. **`CudaBackend` device_id threading.** Thread `device_id` into the ctor + `cudaSetDevice` (the audit F19/F20 — the cleanup left the backend single-current-device; M4.5 fixes it).
3. **Capability probe (the §(2) unified design — lands HERE).** At `Resources`/`build()` assembly, probe per device: compute capability + free/**total** VRAM (`cuda_backend` already calls `cudaMemGetInfo` and *discards* `total_b` — capture it), `cudaDeviceCanAccessPeer` (P2P), the emulated-FP64-honorable state (B2's predicate). Result = a small `BackendCapabilities` value in `Resources`. **Critical:** a **non-throwing, tagged-degrade path distinct from `STEPPE_CUDA_CHECK`** (the `device-cuda-check` CAP-1/CAP-2 finding) — `canAccessPeer`="no" / `cudaErrorPeerAccessAlreadyEnabled` are EXPECTED on the budget box ⇒ `STEPPE_LOG_WARN`-and-degrade, NEVER throw (uses the `log.hpp` from B7 + a `STEPPE_CUDA_WARN`-style variant).
4. **SNP-range shard + per-device partials.** Streamer partitions SNP tiles/ranges across G devices (round-robin / static range); each device runs `compute_f2_blocks` on its shard → its own full-shape partial `f2_blocks`+`Vpair`.
5. **Host-side fixed-order combine = the PORTABLE PARITY BASELINE (works on the budget 5090).** Gather G partials to host, sum in fixed device order, reference precision — **NOT NCCL AllReduce** (its order varies with G, breaks parity, §12). Broadcast back (NCCL Broadcast or `cudaMemcpy`, order-independent).
6. **Optional capable-path P2P device-combine** (gated on `enable_peer_access` + `cudaDeviceCanAccessPeer`): device 0 pulls each peer's partial via `cudaMemcpyPeer` (byte-exact DMA) and sums in the SAME fixed `g=0..G-1` order ON-DEVICE — **bit-identical** to host-staged (the transport only moves bytes; never NCCL AllReduce). Tagged fallback on the budget box: log "P2P combine unavailable (no peer access) → host-staged fixed-order combine". On 2× consumer 5090 P2P needs the **aikitoria patched open-gpu-kernel-modules** (full-host, dev-only, NEVER production); on the **RTX PRO 6000 it's clean stock-driver** (IOMMU/ACS off, no NVLink on this SKU). **steppe is P2P/NVLink-insensitive by design** — this is architectural cleanliness, not throughput (the combine is kB–MB, off the critical path).
7. **Two knob types (audit §(2).3):** override-intent → `DeviceConfig` (`enable_peer_access` exists; add `prefer_p2p_combine`); discovered-capability + which-path tag → `Resources`/result metadata (NEVER on `F2BlockTensor` — keep it pure numeric).
8. **Parity test (the GATE):** multi-GPU combine bit-identical across G (1 vs 2 GPUs) AND to the single-GPU reference, on real AADR (`derived_full` P=768). Extend the `f2_determinism` pattern to the multi-device combine.

**Files that change (audit §(2)):** `config.hpp` (override-knob banner + `prefer_p2p_combine` + the `deterministic` field landed in B9), `backend.hpp` (a `BackendCapabilities probe()` + per-device contract), `cuda_backend.cu` (the probe + capture `total_b` + `cudaSetDevice`/`device_id` + the EmulatedFp64-not-honorable tagged degrade), `f2_from_blocks.cpp` (thread `Resources`, host the `canAccessPeer`-gated combine *policy*), `handles.hpp` (record device ordinal), `log.hpp` (the tagged-warn sink, exists from B7), `device-cuda/check.cuh` (the non-throwing `STEPPE_CUDA_WARN` probe variant).

**VRAM note (B26):** `f2_blocks`+`Vpair` = `2·P²·n_block·8`. Multi-GPU REPLICATES the full-shape partial per device — it shards the **input (SNP) work** for throughput + the combine, it does NOT reduce the per-device **output** footprint. The per-device `f2_blocks` ceiling stays P-bound (~P=1360 @70% on a 5090, ~P=2356 on the 96 GB PRO 6000). Large-P input streaming is M5.

**Where it's testable:** the **host-staged multi-GPU combine + parity test works on the budget 2× 5090** (do the bulk there). The **P2P device-combine path needs the PRO 6000** (clean stock-driver P2P). Start with a short design step (Resources + capability-probe + the combine seam) before code.

---

## 6. Deferred (do NOT lose these)
- **L1–L19** (`docs/cleanup/00-overview.md`) — M5/M6/Phase-2-timed or cosmetic (e.g. L3 `Stream` non-blocking for the M5 overlap; L4 pool allocator + pinned staging; L8 `geno_reader`/`mind_prepass` streaming reshape; L16 more host unit tests; L19 `ind_reader` semantic bug).
- **B5/B6 at-scale validation** (large-P near the VRAM ceiling / >1.05M-SNP grid trigger) → PRO-6000 session.
- **AT2 goldens** — install R + admixtools on a box, generate pinned `extract_f2` goldens (record R/RNGkind/AT2-version/blgsize/boot/seed per §12); the hard parity gate for M7. AT2 autosome parity = chr 1–22 → 719 blocks.
- **Rest of Phase 1:** M5 (out-of-core SNP-tile streaming — the CPU-light pinned double-buffer ingest, the biggest CPU-offload win), M6 (multi-dataset merge), M7 (on-disk cache + FST + the AT2 goldens).
- **Phase 2 (fit engine):** S3 f3/f4, S4 jackknife→SE/cov, S5 qpWave SVD, S6 qpAdm GLS, S7 p-values, S8 model-space search (multi-GPU). Host/launch-sensitive → CUDA graphs (`agentscripts/research-gpu-dominant-pipeline.js` findings).
- **Tiny:** make the assert/death tests core-free (so they stop dumping 404 MB cores).

## 7. Resume / git
- `main` @ `1fbb417` = everything (M0–M4 + cleanup + audit + agentscripts). Branches `m0-f2-scaffold`/`m1-decode-af`/`m2-filters`/`m3-block-partition`/`m4-perblock-f2` are all merged → prunable.
- This handoff (`handoff-1fbb417.md`) is committed to `main` for durability; the superseded `handoff-ff81498.md` was deleted. Untracked strays (leave): `aadr/`, `build_run.sh`, `f2_emu_spike.cu`.
- Memory: `~/.claude/projects/-home-suzunik-steppe/memory/` ([[steppe-project]], [[steppebox5090]], [[steppe-dev-process]], [[steppe-old-mvp-deprecated]]).
- Workflow pattern for fixes: see `agentscripts/fix-pass-phase2.js` — strictly sequential, fixer + independent verdict, build+`ctest`-gated, commit-green / revert, `ulimit -c 0`, retry-once-on-transient-500.

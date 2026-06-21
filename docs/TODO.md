# `steppe` — TODO / task tracker

Living, checkable companion to [`ROADMAP.md`](ROADMAP.md) (the **order & rationale**) and [`architecture.md`](architecture.md) (the **design & standards**). This is the **granular next-steps checklist for future-us** — keep it current as milestones land. Update the checkboxes; move finished items to "Done".

**Big picture (2026-06-21):** **the BACKEND is now FINISHED.** Both halves are built and golden-gated. (1) The **precompute half is COMPLETE through M5** — M0–M4 (validated 3-GEMM f2 kernel → data front-end → filters → per-block `f2_blocks` tensor) **+ the entire before-M4.5 cleanup (B1–B27)** **+ M4.5 single-node multi-GPU (correct, bit-identical)** **+ device-resident output (`1f80c0c`)** **+ M5 out-of-core streaming (tiered output `176a07d` + SNP-tile input streaming `c65179f`)**; full-autosome (M=584131, n_block=748 under the AT2 walk) P=2500 completes on one 32 GB RTX 5090 in ~51.5 s, parity bit-identical. (2) The **qpAdm FIT ENGINE (Phase 2 / S3–S8) is BUILT on the GPU and golden-gated** — the f3/f4 contraction, weighted block-jackknife covariance, the qpWave/qpAdm rank sweep (small-Jacobi + NRBIG cuSOLVER-`gesvd`), the GLS ALS + Σw=1 constrained weight solve, χ²/p, block-jackknife SE, and the S8 batched model rotation, on both `CpuBackend` (parity oracle) and `CudaBackend`. (3) The **BACKEND-FINISH** (step 1 of the backend-first sequence) is COMPLETE: the 6 FINISH-NOW items from [`fit-engine-finish-punchlist.md`](design/fit-engine-finish-punchlist.md) all landed (`e8430a2`→`2496a14`), closing the domain-outcome taxonomy, missing-block (NA) handling, the `run_qpwave` golden, and API/test hygiene. **All fit/qpwave goldens are REAL AADR AT2 goldens** (`golden_fit0`/`fit1_NRBIG`/`rot`/`qpwave`/`fitNA`; admixtools 2.0.10, R 4.3.3, real AADR v66.p1_HO) — no synthetic data; **42 ctest tests** green (default + `STEPPE_THOROUGH`), CPU/GPU consistent, deterministic, wallclock unchanged. `main == phase2-fit-engine`. **THE HONEST NEXT = step 2 productization: there is NO CLI, NO Python bindings, NO standalone f-stats yet** (no `app/` or `bindings/` dir; `STEPPE_BUILD_PYTHON` is an OFF stub; no production `main()`). Remaining precompute milestones M6 (multi-dataset QC front-end) + M7 (on-disk cache / FST). **THE KEY PRECOMPUTE LESSON (kept): the precompute was HOST-RESULT-BOUND** — ~80% of the old wall was copying the result to CPU; the win was getting it OFF the CPU (device-resident output `1f80c0c`, ~4.3× at P=512), **NOT multi-GPU**. Multi-GPU's proper home is the fit's model rotation (thousands of independent models) — that remains DEFERRED (run single-GPU; `TODO(multigpu-host-bounce)`), do NOT claim a multi-GPU speedup.

---

## ✅ Done (Phase 1 precompute, M0–M4)
- [x] **M0** — structure lift: 3-GEMM f2 kernel + fixed-slice Ozaki into the layered architecture (`a7a27ec`)
- [x] **M3** — SNP→block rule `assign_blocks` (AT2 SNP-anchored `setblocks` walk, per-chrom reset, re-anchored cumulative cut); full-v66 748 blocks (autosome 711); AT2 cache parity target 709 (`f7f31c6`; reconciliation `docs/research/block-partition-at2.md`)
- [x] **M1** — GPU TGENO decode → Q/V/N (raw-value 2-bit; bit-for-bit oracle match) (`150bfb3`)
- [x] **M2** — missingness + filters (host-pure predicates; drop-equals-mask) (`1bbbad4`)
- [x] **M4** — per-block `f2_blocks[P×P×n_block]` via grouped strided-batched GEMMs (`75c6c10`)

---

## ✅ M4.5 — Single-node multi-GPU precompute — DONE (2026-06-17, on `main`)

> **COMPLETE, bit-identity-proven.** Built via two sequenced per-file workflows (`agentscripts/m4.5-scaffold.js` → `agentscripts/m4.5-multigpu.js`), then a perf fix-pass culminating in the **device-resident combine** (`867a4bf`); root-cause diagnosed by nsys (`165f655`, `docs/cleanup/m4.5/why-multigpu-slow.md`). Clean from-scratch build, **30/30 ctest green**. Scaffold: `STEPPE_CUDA_WARN` (non-throwing tagged degrade), `BackendCapabilities`+probe, `prefer_p2p_combine`, device-ordinal+`MathModeScope`, `CudaBackend(device_id)`+`cudaSetDevice`. Algorithm: `Resources`/`PerGpuResources`+builder, **block-aligned shard** (whole contiguous block ranges → each block computed on one device), **host-staged fixed-order combine** (portable baseline), **device-resident `cudaMemcpyPeer` P2P device-combine** (real stock-driver P2P, ran on rtxbox). **Parity (test_f2_multigpu_parity):** for the production `EmulatedFp64{40}` path, G==2 host-staged AND the device-resident P2P combine are **bit-identical** (memcmp parity) to single-GPU on `derived_acc` (P=50) + `derived_full` (P=768); native `Fp64` at G≥2 matches to oracle tol (rel ≤2.87e-15 = §12 `[UNCERTAIN]` batched-GEMM batchCount-sensitivity, correctly scoped). NEVER NCCL AllReduce. **The honest multi-GPU verdict (correct the older "faster" framing):** on the *precompute*, multi-GPU is a **modest throughput layer, NOT the perf win** — it was measured *slower* than single-GPU until the data-bounce was fixed (nsys showed ~22–74% overlap, a serial D2H/host tail), and even after the device-resident combine the gain is marginal. The real perf wins came from getting the result **off the CPU** — device-resident output (`1f80c0c`) + M5 streaming (`176a07d`/`c65179f`). **Multi-GPU genuinely SHINES on the FIT/ROTATION phase** (thousands of INDEPENDENT qpAdm models, no combine) — that is its proper home, not the precompute. See `docs/cleanup/m4.5/{architecture-audit,parallelism-check,why-d2h,why-multigpu-slow}.md`. **Consumer-5090 graceful-degrade validation** (`can_access_peer==false` → host-staged + tag) is an optional tier-branch in the parity test. M4.5 is **merged to `main`.**

*Refs: architecture §11.4 (design), §9 (Resources/PerGpuResources), §12 (parity); the audit `docs/cleanup/00-overview.md` §(2) "CAPABILITY-TIER COHERENCE — the ONE unified design"; the ⚡ section's capability-tier table + box-role split.*

### M4.5 cleanup (audit `docs/cleanup/m4.5/` — overall 8.4/10, all parity-safe)
- **5090 fix-pass** (`agentscripts/m4.5-fix-pass.js`) — the 5090-validatable before-M5 backlog, fixer+verdict gated by the locked `f2_multigpu_parity` bit-identity: **T1** parity-test VRAM-gate (skip `derived_full` gracefully on 32 GB) + **B1** per-device fan-out (the unrealized speedup) + **B3** wire the dead `enable_peer_access` gate + **B4** 3-term gate doc + **B5** single-home `validate_partials` + **B6** drop redundant `block_sizes` + **B7** host-combine `std::copy_n` (perf + −0.0 fix) + **B8** drop throwaway probe backend + **B9** GPU-free host tests.
- **B2 — P2P transport rework — REASSESS / largely SUBSUMED by the device-resident combine (`867a4bf`).** Workflow `agentscripts/m4.5-b2-p2p-fix-pass.js`'s original motivation was the slow/bouncing combine; its core item — P3, kill the host→peer→root double-bounce via device-resident partials — is now **DONE** (the resident combine: per-device compute leaves its partial resident, no D2H/no free; the combine D2D-copies the root partial + `cudaMemcpyPeer`s each peer straight into its disjoint block slice, then ONE final D2H — the redundant second 7.14 GB D2H + re-upload H2D + place-add + accumulator memset are all deleted). Mark B2 as **likely-subsumed**, not a pending speedup. Remaining optional B2-flavored lever: streamed P2P / pinning the final pageable result D2H (see "remaining optional levers" below).

**Goal.** Shard SNP work across the G devices; each computes a full-shape **partial** `f2_blocks`+`Vpair`; combine the partials **once, host-side, in fixed device order** (`g=0..G-1` = `DeviceConfig::devices` order) in reference precision; broadcast back. **Bit-identical across G AND to the single-GPU reference (§12).** This is *the* milestone that introduces the **capability-tier machinery** — it's the first capable-vs-budget fork (P2P device-combine vs host-staged). **Build the host-staged baseline first; P2P is an opt-in fast-path.**

*Resources / device threading (Theme-1 debt M4.5 calcifies — these land here):*
- [x] **`Resources` / `PerGpuResources`** (§9), one per `DeviceConfig::devices`, all RAII: `{int device_id; Stream stream; CublasHandle blas; DeviceAllocator* allocator; (NcclComm comm);}`. Each binds `cudaSetDevice(device_id)`; **replicate B1's (stream,workspace)-owning `CublasHandle` per device.**
- [x] **`CudaBackend` device_id threading** — thread `device_id` into the ctor + `cudaSetDevice` (audit F19/F20: cleanup left the backend single-current-device). The SPMG prerequisite.

*Capability probe = the §(2) unified design, lands HERE:*
- [x] **`BackendCapabilities` probe** at `Resources`/`build()` assembly, per device: compute capability + free/**total** VRAM (`cuda_backend` already calls `cudaMemGetInfo` and **discards `total_b`** — capture it), `cudaDeviceCanAccessPeer` (P2P), emulated-FP64-honorable state (B2's predicate). Result = a small value in `Resources`.
- [x] **Non-throwing, tagged-degrade path DISTINCT from `STEPPE_CUDA_CHECK`** (the `device-cuda-check` CAP-1/CAP-2 finding) — `canAccessPeer="no"` / `cudaErrorPeerAccessAlreadyEnabled` are EXPECTED on the budget box ⇒ `STEPPE_LOG_WARN`-and-degrade, **NEVER throw**. Add a `STEPPE_CUDA_WARN`-style variant alongside `check.cuh` (uses the `log.hpp` from B7).

*Shard → partials → combine:*
- [x] **SNP-range shard** across G devices (block-aligned static range; tile round-robin later with M5); each device runs `compute_f2_blocks` on its shard → its own **full-shape partial** `f2_blocks`+`Vpair`.
- [x] **Host-side fixed-order combine = the PORTABLE PARITY BASELINE** (works on the budget 5090): gather G partials to host, sum in fixed device order, reference precision — **NOT NCCL AllReduce** (its order varies with G, breaks §12). Broadcast back (NCCL Broadcast or `cudaMemcpy`, order-independent).
- [x] **Capable-path P2P device-combine** (gated on `prefer_p2p_combine` + `cudaDeviceCanAccessPeer`) — now **device-resident** (`867a4bf`): the combine allocates ONE root result, D2D-copies the root partial + `cudaMemcpyPeer`s each peer partial straight into its DISJOINT block slice (byte-exact DMA), then ONE final D2H and a single fence — **bit-identical** to host-staged (transport only moves bytes; never NCCL AllReduce). **Tagged fallback** on the budget box: log *"P2P combine unavailable (no peer access) → host-staged fixed-order combine"*. Clean stock-driver on the **PRO 6000** (IOMMU/ACS off, no NVLink this SKU); aikitoria-patch-only on 2× 5090 (dev-only, NEVER production). steppe is P2P/NVLink-insensitive by design (combine is off the critical path).
- [x] Confirm NCCL present on the box (else `cudaMemcpy` broadcast).

*Knob separation (audit §(2).3) + the gate:*
- [x] **Two knob types:** override-intent → `DeviceConfig` (`enable_peer_access` + `prefer_p2p_combine`); discovered-capability + which-path tag → `Resources`/result metadata (**NEVER on `F2BlockTensor`** — kept pure numeric).
- [x] **Parity test (the GATE) — PASSING:** multi-GPU combine bit-identical (memcmp) across G (1 vs 2 GPUs) AND to the single-GPU reference, on real AADR (`derived_acc` P=50 + `derived_full` P=768), on BOTH the host-staged AND device-resident P2P combine paths. Extends the `f2_determinism` pattern to the multi-device combine.

**Files that change (audit §(2)):** `config.hpp` (override-knob banner + `prefer_p2p_combine`; `deterministic` landed in B9) · `backend.hpp` (`BackendCapabilities probe()` + per-device contract) · `cuda_backend.cu` (probe + capture `total_b` + `cudaSetDevice`/`device_id` + EmulatedFp64-not-honorable tagged degrade) · `f2_from_blocks.cpp` (thread `Resources`, host the `canAccessPeer`-gated combine *policy*) · `handles.hpp` (record device ordinal) · `log.hpp` (tagged-warn sink, exists from B7) · `device/cuda/check.cuh` (the non-throwing `STEPPE_CUDA_WARN` probe variant).

**VRAM note (B26):** `f2_blocks`+`Vpair` = `2·P²·n_block·8`. Multi-GPU **REPLICATES the full-shape partial per device** — it shards the **input (SNP) work** for throughput + the combine, it does NOT reduce the per-device **output** footprint. Per-device `f2_blocks` ceiling stays P-bound (~P=1360 @70% on a 5090, ~P=2356 on the 96 GB PRO 6000). Large-P input streaming is M5.

**Where it's testable:** the **host-staged combine + parity test works on the budget 2× 5090** (do the bulk there). The **P2P device-combine path needs the PRO 6000** (clean stock-driver P2P).

**Perf — RESOLVED, the rabbit-hole is CLOSED.** Root cause (nsys, `docs/cleanup/m4.5/why-multigpu-slow.md`, `165f655`) was the **data-bounce wart**: `compute_f2_blocks` D2H-copied each partial to host and freed its device buffers, forcing the P2P combine to re-upload (H2D) + place-add + a 2nd D2H of the full 7.14 GB. The **device-resident combine** (`867a4bf`) deleted the redundant bounce. **But the real lesson outran the multi-GPU framing** (`docs/cleanup/m4.5/why-d2h.md`): the precompute was **HOST-RESULT-BOUND** — ~80% of the wall was the final bulk D2H of the 6.36 GB+ result. **Device-resident output (`1f80c0c`)** returns a `DeviceF2Blocks` handle that stays in VRAM (the host `F2BlockTensor` is an opt-in `.to_host()`), measured **~4.3× at P=512 (~673 ms vs ~2879 ms bulk-to-host).** Multi-GPU on the precompute is a marginal throughput layer, never the speedup.
- ~~Refuted (do NOT carry as open problems):~~ the "per-call `cudaMalloc`/`cudaFree` driver-lock serializes the device threads / only ~18% overlap" hypothesis — nsys measured up to ~74% overlap; the L4 pool-allocator is at most a few % and is **NOT** a speedup lever (demote to OPTIONAL cleanup). The "~1.4 s combine host-zeroing (P1 default-init allocator) dominates" hypothesis — the accumulator memset is **5 ms**; P1 was REVERTED and is not the cure. `docs/cleanup/m4.5/perf-discovery.md`'s P1 premise is superseded by `why-multigpu-slow.md` / `why-d2h.md`.
- [ ] **[optional / deferred]** Pin the device-resident final-D2H (the opt-in `.to_host()` path; rtxbox) — observability/throughput only, not on the in-VRAM critical path. Not a blocker.
- [ ] **[optional]** Bench byte-traffic columns are observability-only (currently print 0.00).
- [ ] **[optional / reassess]** B2 P2P transport rework (`agentscripts/m4.5-b2-p2p-fix-pass.js`) — subsumed by the resident combine; **not a pending speedup.**

## ✅ M5 — Out-of-core streaming — DONE (2026-06-19, on `main`; `176a07d` + `c65179f`)

> **COMPLETE, parity bit-identical.** Two pieces: **adaptive tiered OUTPUT (`176a07d`)** + **SNP-tile INPUT streaming (`c65179f`)**. **(a) Tiered output is ADAPTIVE, not mandatory** — the result goes to the fastest tier it FITS, auto-selected from runtime free VRAM/RAM: VRAM-resident (small P — keeps the device-resident 4.3×) → host RAM (big box) → disk (laptop). **(b) SNP-tile input streaming** — per-block decode makes the GPU footprint `O(P·tile + P²)` **INDEPENDENT of M** (kills the old `7·P·M` feeder wall). **RESULT: full-autosome (M=584131, n_block=748 under the AT2 walk) P=2500 COMPLETES on a single 32 GB RTX 5090 in ~51.5 s** (76 GB result streamed, GPU peak ~26 GB bounded), parity memcmp **bit-identical** to the in-core path. **Measured sweep (one 5090, streamed):** P=512 ~3.6 s, P=1000 ~10.4 s, P=1500 ~20.2 s, P=2000 ~34.0 s, P=2500 ~51.5 s. This **supersedes** the pre-M5 scaling sweep (`docs/cleanup/m4.5/scaling-sweep.md`, which claimed P=2500 OOMs on every path — now FALSE; that doc is historical/SUPERSEDED). **Optional / deferred:** multi-GPU block-sharding on the stream (throughput); GDS lane on the PRO-6000 (`.f64`/M7 cache).
- [x] `PinnedBuffer<T>` RAII (page-locked staging) + dedicated copy `Stream` (pays the default-stream debt).
- [x] Tile-size derivation vs VRAM budget; resident accumulator OR tiered host/disk spill, runtime-selected from free VRAM/RAM.
- [x] Per-block streaming decode → AF → per-block f2 partial → accumulate/spill → discard; GPU footprint `O(P·tile + P²)`, independent of M.
- [x] Parity gate: streamed P=2500 full-autosome memcmp bit-identical to the in-core reference.

## ⏭️ M6 — QC / data-munging front-end (multi-dataset) — **REMAINING precompute**
*Depends: M1 + M2. Refs: architecture §5 (S-2/S-1/S0′ table), §1 (scope).*
- [ ] S-2 merge plan: intersection (default) / union; declared-allele polarity; **drop** ambiguous/multiallelic (never strand-flip).
- [ ] S-1 conditional pre-pass (`--mind` / external `prune.in`, read not computed).
- [ ] S0′ harmonized + filtered tile produce; transversions-only option. No on-disk rewrite, no strand inference, no LD compute.

## ⏭️ M7 — On-disk cache + FST — **REMAINING precompute**
*Depends: M4. Refs: architecture §5 (FST), §12/§13 (parity). NOTE: R + admixtools is already installed and the REAL AADR AT2 goldens are PINNED (used by the Phase-2 fit gate — `golden_fit0`/`fit1_NRBIG`/`rot`/`qpwave`/`fitNA`); M7 only needs the on-disk-cache + FST parity arms.*
- [ ] On-disk `f2_blocks` cache, ADMIXTOOLS-compatible read/write.
- [ ] **FST** as a cheap add-on output of the same pass.
- [ ] Wire an `extract_f2` AT2-parity arm for the on-disk cache round-trip (the goldens already exist; §12 metadata recorded).

---

## ✅ Phase 2 — qpAdm FIT ENGINE (S3–S8) — DONE (2026-06-21, on `main` == `phase2-fit-engine`)
*Refs: [`docs/design/fit-engine.md`](design/fit-engine.md), architecture §5 (S3–S8), §11.4 (S8 multi-GPU), §12 (determinism near rank-deficiency). The fit operates on `f2_blocks`.*

> **BUILT + golden-gated on the GPU.** Input contract: reads `f2_blocks` **DEVICE-RESIDENT** for the in-VRAM case (small P — never bounce through the host) and **streamed tiles** for large P (the M5 path). Built on BOTH `CpuBackend` (the parity oracle) and the production `CudaBackend`. The **S8 MODEL ROTATION is the embarrassingly-parallel phase** — thousands of INDEPENDENT models, no combine — multi-GPU's proper home, but **multi-GPU is DEFERRED** (`TODO(multigpu-host-bounce)`; run single-GPU — do NOT claim a multi-GPU speedup).
- [x] **S3** — f3/f4 contraction from `f2_blocks` (identity-based derivation; `assemble_f4`).
- [x] **S4** — weighted block-jackknife → covariance `Q` + SEs.
- [x] **S5** — qpWave/qpAdm rank sweep (SVD: small on-device Jacobi + the NRBIG cuSOLVER-`gesvd` path).
- [x] **S6** — qpAdm GLS fit (ALS opt_A/opt_B + the Σw=1 constrained weight solve) → weights, χ².
- [x] **S7** — p-values / nested-model test (dof-aware χ² tail).
- [x] **S8** — model-space search / rotation (batched, single-GPU; multi-GPU deferred).
- [x] **AT2 goldens = the FIT validation gate (REAL AADR, no synthetic data)** — `tests/reference/goldens/at2/golden_fit0.json`, `golden_fit1_NRBIG.json`, `golden_rot.json` (admixtools 2.0.10 / R 4.3.3, AADR v66.p1_HO); gated by `qpadm_parity` (#17) + `qpadm_rotation` (#18).
- **Deferred/optional (do NOT re-litigate as blockers):** S8 multi-GPU block-sharding/rotation (`TODO(multigpu-host-bounce)`, run single-GPU); the device-resident final-D2H pinning (rtxbox); the **TurboQuant-L2 rotation screen** (`docs/research/turboquant-l2-experiment.md`); GPU SVD nr>32 sweep is currently CPU-oracle-validated only. Phase-2 is far more **host/launch-sensitive** than the precompute → CUDA-graph capture per fit remains a future perf lever (perf research `wqd0a9o0l`).

## ✅ Backend FINISH (step 1 of backend-first) — DONE (2026-06-21, `e8430a2`→`2496a14` on `main`)
*The 6 FINISH-NOW items from [`docs/design/fit-engine-finish-punchlist.md`](design/fit-engine-finish-punchlist.md) — contract closure (NOT new math). After these, the BACKEND is honestly done; the NEXT is step 2 (CLI + bindings) below.*
- [x] **F5** (`e8430a2`) — REMOVE the dead public `QpAdmOptions::constrained` field (`include/steppe/qpadm.hpp`); non-negative constrained-weights is a deferred step-3 feature (API hygiene).
- [x] **F3** (`ffdcba2`) — add `Status::ChisqUndefined` (`include/steppe/error.hpp:44`) + a `dof<=0 ⇒ ChisqUndefined` guard on the HOST (`qpadm_fit.cpp`) AND the CUDA model-batched path (`cuda_backend.cu`); was leaking NaN p with `status=Ok`.
- [x] **F2** (`c8fe397`) — the M(fit-5) domain-outcome acceptance gate, NEW `tests/reference/test_qpadm_domain.cu` (ctest `qpadm_domain`, #19): degenerate REAL-AADR models asserted as STATUS VALUES (no crash/NaN) on BOTH backends — collinear left ⇒ `RankDeficient`; fudge=0 singular Q ⇒ `NonSpdCovariance`; over-parameterized dof<=0 ⇒ `ChisqUndefined`.
- [x] **F6** (`360e386`) — widen the G1==G2 determinism memcmp in `test_qpadm_rotation.cu` to the FULL `QpAdmResult` (z/dof/est_rank/rank_*/rankdrop_*/popdrop_*).
- [x] **F4** (`6481dfa`) — pin a REAL AT2 `qpwave()` golden `tests/reference/goldens/at2/golden_qpwave.json` + NEW `tests/reference/test_qpwave_parity.cu` (ctest `qpwave_parity`, #21) gating the first-class `run_qpwave` entry on BOTH backends.
- [x] **F1** (`2496a14`) — missing-block / NA handling (OQ-12). steppe is pairwise-complete (NOT AT2 maxmiss=0 global-intersection), so a pair-block `Vpair==0` CAN occur on sparse AADR and was silently imputed `f2=0`. Now implements AT2 `read_f2(remove_na=TRUE)`: DROP any block with a non-finite/`Vpair==0` pair before the LOO/jackknife, via the single shared host/device predicate `core::pair_block_is_missing` (`f2_estimator.hpp`) on the CpuBackend oracle + GPU (`f2_block_keep_kernel`; single-model AND S8 survivor-compaction). Legacy maxmiss=0 goldens stay BYTE-IDENTICAL (no-drop identity arm). NEW `golden_fitNA.json` + `tests/reference/test_qpadm_missing_block.cu` (ctest `qpadm_missing_block`, #20; real-AADR maxmiss=0.99, a real `Vpair==0` dropped block). NO synthetic data.

> **Gate status:** 42 ctest tests green (default + `STEPPE_THOROUGH` oracle), CPU/GPU consistent, deterministic, wallclock unchanged. Legacy maxmiss=0 goldens byte-identical.

---

## 🚨 GOLDEN INTEGRITY + I/O FORMATS — **BLOCKER (do FIRST; PROVEN 2026-06-21)**
*Memory `aadr-tgeno-goldens-corrupt`; `docs/research/tgeno-at2-support.md`. PROVEN on box5090.*
The AADR v66 `.geno` is **TGENO** (transposed/individual-major). **admixtools R v2.0.10 does NOT support TGENO — it silently MISREADS it** (the AADR README warns this). So **every committed golden + f2 fixture is CORRUPT** (all built by `extract_f2` on the raw v66 TGENO): `golden_fit0`/`rot`/`fit1_NRBIG`/`fitNA`/`qpwave` + `f2_*.bin`. **steppe's TGENO decode is CORRECT** — proven: `convertf` v8621 (DReichLab/AdmixTools) → PACKEDANCESTRYMAP → AT2 `extract_f2` gives **391,333 SNPs, [CW 0.869, Turkey_N 0.131]**, matching steppe to the SNP digit, vs the corrupt golden's 500,848 / [0.559,0.441]. The engine is sound; the GOLDEN REFERENCE was wrong. (The M(cli-4) "decode bug" verdict was wrong — it inverted cause/effect.)
- [ ] **Regenerate ALL goldens** (unblocks M(cli-4) + re-validation + any real study): `convertf >=v8.0.0` TGENO→PACKEDANCESTRYMAP, then AT2 `extract_f2` on the converted prefix. Reproducible artifacts on the box: `/workspace/data/aadr/convertf_tgeno_to_pa.par`, `at2_on_converted.R`, convertf at `/workspace/AdmixTools_src/src/convertf`, converted prefix `/workspace/data/aadr/converted_pa/v66_HO_pa`. Regenerate the 5 goldens + the `f2_*.bin` fixtures; re-run the parity suite (steppe should MATCH the corrected goldens; the old corrupt ones will differ).
- [ ] **Older `.GENO` reader support (USER ASK)** — steppe is currently **TGENO-ONLY** (`io::GenoReader::read_tile` requires TGENO). Add **classic PACKEDANCESTRYMAP ("GENO" magic, SNP-major) + EIGENSTRAT** readers so older AADR releases / other datasets work. v66 TGENO is priority but older GENO matters too (format-detect on the magic; dispatch).
- [ ] **CI guard** — reject any golden built by AT2-R directly on a raw TGENO `.geno` (record the format in the golden metadata; fail if TGENO+AT2-R).
- [ ] **Re-gate M(cli-4) `extract-f2`** against the corrected golden (steppe's 391,333 / [0.869,0.131] is the CORRECT answer — it now passes). Then re-run the studies (the earlier Yamnaya/Bell-Beaker numbers were on corrupt f2 — INVALID; redo on correct f2).

## ⏭️ Step 2 — Productization: CLI + Python bindings
*Refs: [`docs/design/cli-bindings.md`](design/cli-bindings.md) (the contract); `desirable-features-survey.md`; memory `build-sequence-backend-first`.*

**CLI** (`src/app/`, CLI11 via CPM, GPU-only, app is a plain-CXX CUDA-free target):
- [x] **M(cli-0)** scaffold + the CUDA-free `ConfigBuilder`/`RunConfig` precedence + Status→exit-code (`62253ab`).
- [x] **M(cli-1)** `steppe qpadm` over an f2 dir (`f2.bin`+`pops.txt`+`meta.json` reader, name→index, GPU, tidy CSV/JSON) — golden-gated through the CLI (`67dc696`).
- [ ] **M(cli-4)** `steppe extract-f2` (genotypes → f2 dir; STPF2BK1 writer w/ real vpair) — **BUILT but BLOCKED**: its golden gate fails only because the golden is corrupt (above). Re-gate after golden regen. *(Mechanically complete; verdict reverted it pending the corrected golden.)*
- [ ] **M(cli-2)** `steppe qpwave` (the `run_qpwave` entry).
- [ ] **M(cli-3)** `steppe qpadm-rotate` (S8 rotation; `--jackknife 0|1|2`).
- [ ] CLI arg/IO contract: pop lists by name, `blgsize`/`maxmiss`/precision, CSV/JSON, `--dry-run` per-box P_max.

**Python bindings** (new `bindings/`; flip `STEPPE_BUILD_PYTHON`) — *decisions made: nanobind (NOT PyCUDA, `docs/research/pycuda-cuda13-viability.md`) + a DLPack/CAI interop seam; use-cases in `docs/research/interop-usecases.md` (MUST = results→pandas + f2→numpy; the msprime power-analysis loop; GPU-only, fp64-enforced):*
- [ ] **M(py-1)** nanobind module: `qpadm`/`qpwave`/`qpadm_rotate` from a dir → pandas; results→DataFrame + f2→numpy; status enum + NA sentinels.
- [ ] **M(py-2)** `extract_f2` from Python; scikit-build-core wheel (GPU-only, one wheel).

## ⏭️ Step 3 — Standalone f-stats (each WITH its own CLI/bindings)
*After step 2. Refs: architecture.md (the "(planned)" standalone tools); `desirable-features-survey.md`.*
- [ ] Standalone **f3** / **f4** statistics (direct from `f2_blocks`) + CLI + bindings.
- [ ] **D-statistic / qpDstat** + CLI + bindings.
- [ ] Non-negative constrained-weights qpAdm (the deferred F5 feature).

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
- [x] **Install R + admixtools → pin AT2 goldens** — DONE for the **Phase-2 FIT validation gate** (admixtools 2.0.10 / R 4.3.3, real AADR v66.p1_HO): `golden_fit0`/`fit1_NRBIG`/`rot`/`qpwave`/`fitNA` under `tests/reference/goldens/at2/`. Remaining use: the M7 on-disk-cache parity arm.
- [x] **GPU-dominant perf research** (`wqd0a9o0l`) — DONE; results folded into the ⚡ section (M5 pinned ingest, M4.5 filter-fusion, Phase 2 graphs, M7 nvCOMP; GDS ruled out on vast).
- [ ] **[adopt now] Nsight Systems measure-first gate** (§11.3) — classify ingest/launch/orchestration-bound before building any perf lever (see ⚡ section). **Already applied for M4.5** (nsys diagnosed the data-bounce wart → `867a4bf`, `why-multigpu-slow.md`) **and M5** (the host-result-bound diagnosis → device-resident output `1f80c0c`, `why-d2h.md`); now the gate for Phase-2 fit levers.
- [ ] Prune the merged feature branches (`m0-f2-scaffold`, `m1-decode-af`, `m2-filters`, `m3-block-partition`, `m4-perblock-f2`, `m4.5-multigpu`, `m5-input-streaming`) — all merged to `main`. (`phase2-fit-engine` IS `main` — keep until step 2 branches off it.)

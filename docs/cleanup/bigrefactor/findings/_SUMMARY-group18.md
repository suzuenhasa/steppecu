# GROUP 18 — Correctness traps (wrong numbers, not crashes) — ROLLUP

Scope = kernel. Tasks 18.1–18.7 (divergent/missing `__syncthreads()`, warp-synchronous
assumptions, non-`_sync` warp intrinsics, missing bounds guards, cross-thread reads
without a barrier/atomic, order-dependent float reductions).

## (1) Coverage

13 in-scope units (all `src/device/cuda/`). 13 reviewed, **13 clean, 0 with findings**.

| Unit | Result | Kind |
|---|---|---|
| src__device__cuda__block_sink | clean | host-side pinned-ring orchestration (no kernels) |
| src__device__cuda__check | clean | host-only error-check header |
| src__device__cuda__cuda_backend | clean | pure host orchestration (no `__global__`/launches) |
| src__device__cuda__decode_af_kernel | clean | **kernel** — 1-thread/(pop,SNP) map, integer accumulate |
| src__device__cuda__device_buffer | clean | header-only RAII device-mem owner |
| src__device__cuda__device_f2_blocks | clean | host-side transport (cudaMemcpy only) |
| src__device__cuda__device_partial | clean | declaration-only pimpl handle |
| src__device__cuda__f2_block_kernel | clean | **2 kernels** — elementwise feeder + assemble |
| src__device__cuda__f2_blocks_kernel | clean | **2 kernels** — gather + assemble (map/scatter) |
| src__device__cuda__f2_blocks_out | clean | host/file transport (no kernels) |
| src__device__cuda__p2p_combine | clean | host-only multi-GPU placement (no kernels) |
| src__device__cuda__pinned_buffer | clean | host-only page-locked RAII |
| src__device__cuda__qpadm_fit_kernels | clean | **many kernels** — fit chain (f4/LA/SE/SVD) |

Kernel-bearing units (the substantive surface for this group): decode_af_kernel,
f2_block_kernel, f2_blocks_kernel, qpadm_fit_kernels. All four reviewed in full and clean.

## (2) Counts by task + severity

| Task | HIGH | MED | LOW |
|---|---|---|---|
| 18.1 divergent `__syncthreads()` | 0 | 0 | 0 |
| 18.2 missing `__syncthreads()` (RAW/WAR) | 0 | 0 | 0 |
| 18.3 warp-synchronous assumptions | 0 | 0 | 0 |
| 18.4 non-`_sync` warp intrinsics | 0 | 0 | 0 |
| 18.5 missing bounds guard (OOB) | 0 | 0 | 0 |
| 18.6 cross-thread read w/o barrier/atomic | 0 | 0 | 0 |
| 18.7 order-dependent float reduction | 0 | 0 | 0 |
| **TOTAL** | **0** | **0** | **0** |

## (3) Top findings

None. No HIGH/MED/LOW findings in Group 18 across all 13 units.

## (4) Cross-cutting patterns (why clean — auditable, not padding)

The clean result is structural, and the agents verified it against the actual kernel
bodies (real line numbers), not by assertion:

- **Map/scatter kernel shape, no inter-thread cooperation.** Every f-stat-path kernel
  is one-element-per-thread with disjoint output cells and no `__shared__` memory:
  decode_af (decode_af_kernel.cu:56-86), f2 feeder/assemble (f2_block_kernel.cu:102-134,
  151-175), gather/assemble-blocks (f2_blocks_kernel.cu:69-112, 129-163). With no shared
  buffer and no lane-to-lane exchange, 18.1/18.2/18.3/18.4/18.6 are vacuous by
  construction — confirmed by grep showing zero `__syncthreads`/`__syncwarp`/`__shfl`/
  `__ballot`/`__any`/`__all`/`__shared__` in those kernels (the only `atomic` is a
  HOST-side `std::atomic_flag` one-shot guard, f2_block_kernel.cu:61,218,222).

- **The one block barrier in the codebase is correct.** `add_fudge_diag_models_kernel`
  (qpadm_fit_kernels.cu:1058-1064) is the sole `__syncthreads()`. Its only preceding
  early-return is block-uniform (`model = blockIdx.x`, :1053-1054), so all-or-none of a
  block returns — never the "some-threads-only" 18.1 UB; and the grid `<<<n_models,block>>>`
  means the guard is never even taken. The shared `s_tr` is written by thread 0 (:1058-1061),
  barrier-published (:1063), then read by all threads (:1064) — RAW barrier present and
  ordered, no WAR (write-once-before-publish). Clean on both 18.1 and 18.2.

- **Bounds guards are present on every kernel.** Grid-stride kernels test `idx/gid < total`
  in the loop header; per-element kernels guard each axis before any load/store
  (e.g. f2 `if (i>=P || s>=M) return;` f2_block_kernel.cu:114; gather
  `if (i>=P || c>=s_pad || k>=n_in_group) return;` f2_blocks_kernel.cu:82; the full
  enumerated fit-kernel guard list at qpadm_fit_kernels.cu:395/426/469/485/495/501/510/519/
  545/702/717/742/775/843/960/990/1033/1054/1076/1105/1204/1232/1267). The grid-z batch
  axis is backed by the `1 ≤ n ≤ kMaxGridZ` assert (launch_config.hpp:144). No 18.5 OOB.

- **Reductions are §12 fixed-order, not order-assuming (18.7 correctly distinguished).**
  Two legitimate forms appear and both are the intended deterministic kind, not the
  anti-pattern: (a) the heavy contractions are cuBLAS GEMMs whose summation order is fixed
  by the once-bound §12 precision-policy workspace (f2_block_kernel.cu:357-370,
  f2_blocks_kernel.cu:245-264); (b) the fit reductions (f4 4-slab, jackknife/loo sums, the
  SE mean/var at qpadm_fit_kernels.cu:1238-1246) are executed by a SINGLE thread in a fixed
  ascending loop — documented "FIXED reduction order ⇒ no atomics ⇒ G=1==G=2 bit-identical"
  (qpadm_fit_kernels.cu:1224-1227). No hand-rolled multi-thread reduction assumes an
  unenforced thread order. Per the FP64/§12 context, these were correctly NOT flagged.

- **Host-side TUs are out of scope and correctly so.** 9 of 13 units are host orchestration
  / RAII / transport with zero kernels; their cross-thread state (e.g. block_sink's
  producer/consumer ring) is mutex-guarded + CUDA-event-ordered, and their transports are
  verbatim byte copies with no FP recompute — so even the host analogues of 18.6/18.7 hold.

## HEADLINE

Units: 13 in scope (13 clean, 0 with findings). Total findings: 0. HIGH: 0.

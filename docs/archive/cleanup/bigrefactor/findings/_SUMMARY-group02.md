# GROUP 2 roll-up — Deprecated / removed APIs & platform support

Tasks in scope:
- 2.1 Dropped archs (Maxwell/Pascal/Volta removed in CUDA 13.0; `sm_50/60/70` build flags or CMake arch lists; min sm_75).
- 2.2 Texture/surface REFERENCES (`texture<...>`, `cudaBindTexture*`) removed in CUDA 12 (hard error) — port to texture objects.
- 2.3 Non-`_sync` warp intrinsics (`__shfl`/`__ballot`/`__any`/`__all`/`__activemask`; also Group 18).
- 2.4 `cudaThreadSynchronize` -> `cudaDeviceSynchronize`.

## 1. Coverage

- Units in scope: **61** (scope = all).
- Units reviewed with a `## Group 2` section: **61 / 61** (100%).
- Clean (no findings): **61**.
- With findings: **0**.

Every unit returned an explicit "No Group 2 issues found."

## 2. Counts by task + severity

| Task | HIGH | MED | LOW | Total |
|------|------|-----|-----|-------|
| 2.1 Dropped archs        | 0 | 0 | 0 | 0 |
| 2.2 Texture/surface refs | 0 | 0 | 0 | 0 |
| 2.3 Non-`_sync` warp     | 0 | 0 | 0 | 0 |
| 2.4 `cudaThreadSynchronize` | 0 | 0 | 0 | 0 |
| **All Group 2** | **0** | **0** | **0** | **0** |

Total findings: **0**. #HIGH: **0**.

## 3. Top findings

None. No HIGH/MED/LOW findings exist in any in-scope unit for Group 2.

## 4. Cross-cutting pattern (why the whole group is clean)

The clean result is structural, not lucky — it falls out of steppe's layering:

1. **Most units are CUDA-free by contract.** ~50 of the 61 units are host-pure (`include/steppe` public headers, all of `src/core`, the `backend.hpp`/`backend_factory` seam, `cpu_backend`, all `src/io`, and the host-side tier/budget/shard/resource headers). They include no CUDA toolkit symbols at all, so 2.1–2.4 are N/A by construction. CUDA-adjacent identifiers that appear (`DeviceF2Blocks`, `ComputeBackend`, `cudaMemcpyPeer*` in `f2_blocks_multigpu`) are CUDA-free seam abstractions or descriptive comments, not API calls.

2. **The actual `.cu` units already use the current CUDA 13 / Blackwell sm_120 surface** — none touch any of the four deprecated/removed patterns:
   - 2.1: Arch selection lives in CMake, not in any reviewed TU; no `sm_50/60/70` or `compute_50/60/70` anywhere. Compute capability is read only as observability (`cudaGetDeviceProperties` in `cuda_backend.cu:1139-1142`) and explicitly NOT used as a dispatch key. Grid-limit constants (`kMaxGridX/Y/Z`, `launch_config`/`vram_budget`) are capability-independent and valid on all CUDA-13 archs.
   - 2.2: No texture/surface references exist anywhere. Every kernel reads via plain `const double* __restrict__` global loads (`f2_blocks_kernel.cu:102-105,149-154`; `qpadm_fit_kernels` f4 4-slab `at()` lambda). There is no texture-reference path to port to texture objects.
   - 2.3: There are **no warp intrinsics at all** in the codebase. All kernel reductions are single-thread sequential loops — this is the §12 deterministic-order parity requirement, not a perf oversight — so there is nothing to migrate to `_sync` variants. The only synchronization primitive used is the non-deprecated `__syncthreads()` (`qpadm_fit_kernels.cu:1063`, `add_fudge_diag_models_kernel`).
   - 2.4: No `cudaThreadSynchronize`/`cudaThreadExit` (or any removed `cudaThread*` alias). Synchronization uses the current `cudaStreamSynchronize` / `cudaEventSynchronize`. Kernel launches post-check via `STEPPE_CUDA_CHECK_KERNEL()`. All cuBLAS/cuSOLVER routines used (`cublasGemmEx`, `cublasDsyrk`, `cublasDgemmStridedBatched`, `cublasGemmStridedBatchedEx`, `cusolverDnDpotrf/Dpotri/DgesvdJ/Dgesvd`, batched `Dpotrf`/`Dpotrs`) and the host-memory family (`cudaHostAlloc`/`cudaHostRegister`, `pinned_buffer`) are all current, non-deprecated CUDA 13 entry points.

**Audit note / residual risk:** The reviewed scope is source/header TUs only. Per the 2.1 notes that recur across the `.cu` units, GPU arch flags / `CUDA_ARCHITECTURES` lists live in CMake, which is **outside the per-unit review scope**. If a separate CMake/build-flag pass has not been run, that is the one place a dropped-arch (`sm_50/60/70`) regression could still hide for Group 2 — the source code itself is clean.

## Headline

61 units in scope, all reviewed; 0 total findings; 0 HIGH.

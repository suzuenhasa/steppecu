# src__device__cuda__qpadm_fit_kernels
Files: /home/suzunik/steppe/src/device/cuda/qpadm_fit_kernels.cu, /home/suzunik/steppe/src/device/cuda/qpadm_fit_kernels.cuh
Subsystem: device-cuda

## Findings

### G5
- [G5.5.1][MED] qpadm_fit_kernels.cu:160-161 and qpadm_fit_kernels.cu:1034-1035 — the one-sided-Jacobi convergence constants `kTol = 1e-15` and `kMaxSweeps = 60` are duplicated as separate local literals in `dev_jacobi_svd_V` and again in `rank_via_jacobi_kernel`. The two sweeps are documented as bit-identical (parity-localizer: gpu rank_Q == cpu rank_Q), so these must move together; `kOffConvergence` (line 76) was already extracted to a file-local named constant but `kTol`/`kMaxSweeps` were left inline in two places — a drift hazard (a tweak to one sweep silently diverges the other). Suggested: hoist `kTol`/`kMaxSweeps` to file-local `constexpr` next to `kOffConvergence` and reference both sweeps from there.

### G6
- [G6][LOW] qpadm_fit_kernels.cu:497, qpadm_fit_kernels.cu:1130, qpadm_fit_kernels.cu:1264, qpadm_fit_kernels.cu:1370 — the RankDeficient status is emitted as the bare magic literal `6` (and `return 6;` in `dev_als_weights` line 497) instead of `core::qpadm::kQpStatusRankDeficient`, the single-source symbol that exists precisely for these emit sites. This is explicitly documented as a deferred group-7 leftover (qpadm_bounds.hpp:76-79), so it is a known/tracked drift, not a fresh one. Suggested: adopt `kQpStatusRankDeficient` at the kernel emit sites when group-7 runs (already planned).

### G8
- [G8][LOW] qpadm_fit_kernels.cu:1014 — the comment cites `core::jacobi_svd` as `small_linalg.hpp:204-286`, but the sibling transliteration comment at qpadm_fit_kernels.cu:151 cites the SAME function as `small_linalg.hpp:162-267`. The two line-range citations for one function disagree, so at least one is stale. Suggested: drop the line numbers (they rot) or reconcile both to the current span.

## Notes (checked, NOT issues)
- G4/G18 int-index width: every resident-f2 / dX / dLoo / dXtau global index that can exceed 2^31 is widened to `long` before the multiply (e.g. `slab = (long)P*P*b` lines 522/629/659; `idx`/`total` as `long` in all grid-stride kernels; `model*m*m`, `model*nb*nl` etc. cast to `long` at lines 1314-1315, 1355, 1451, 1486, 1564-1565, 1679-1680, 1706). The small-path per-thread `int` indices (xmat[i+nl*j], m=nl*nr) are bounded by kQpMaxM=50 so cannot overflow. Clean at P~2500 / nb~757 scale.
- G12 grid-stride: the SNP/element-scale kernels (gather, xtau, loo_total models, transpose, fill_identity, loo_large/loo_models, se_from_wmat, gather_loo_qinv) all use a `long` grid-stride loop with grid clamped via `launch_grid_stride` to `kMaxGridDimX`. The non-grid-stride sweep/keep/iota kernels (`sweep_*`, `f2_block_keep_kernel`, `qpadm_fit_models_kernel`) size grid directly from the count (`(C+block-1)/block`, `(nb+block-1)/block`, `(n_models+block-1)/block`) with no cap, so the grid always covers the input — correct (no missing-stride break).
- G13 launch checks: every `<<<>>>` is followed by `STEPPE_CUDA_CHECK_KERNEL()` (the intentional NDEBUG-gated launch-error + sync check). Consistent across all 30 launch wrappers.
- G11 const __restrict__: all read-only kernel pointer params carry `const ... __restrict__`; outputs carry `__restrict__`. Consistent.
- G18 syncthreads: the only `__syncthreads()` (add_fudge_diag_models_kernel line 1494) is at uniform block scope (all threads reach it; the divergent work is the per-thread `if model>=n_models` early-return at line 1485 which returns the WHOLE block uniformly since model==blockIdx.x). The shared `s_tr` write→read is correctly fenced. No divergent/missing barrier.
- G5 sweep_choose double range (line 683) and sweep_unrank boundary scan (line 707) are caller-guarded and documented (binomials within 2^53; host guards c0+t < C(range,k)); not flagged.
- G19 no stray cudaDeviceSynchronize / printf / #if 0 in kernels. `__launch_bounds__` pins (64/64/128) are documented forward-compat occupancy guards, never under-launched.
- G3 `(void)dLoo; (void)d_se;` at line 1566 is an intentional suppress for params kept on the FROZEN-CONTRACT signature while the SE moved to a separate kernel; documented inline. Not dead-code noise.

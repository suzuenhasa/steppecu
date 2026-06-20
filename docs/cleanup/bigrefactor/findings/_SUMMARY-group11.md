# GROUP 11 — Qualifiers & const-correctness — ROLLUP

Scope = kernel units (`src/device/cuda/*`). 13 in-scope units reviewed.

## Headline
- Units in scope: 13
- Units clean: 12
- Units with findings: 1 (`src__device__cuda__qpadm_fit_kernels`)
- Total findings: 2
- By severity: HIGH 0 / MED 0 / LOW 2

## 1. Coverage

| Unit | Result |
|------|--------|
| src__device__cuda__block_sink | clean (host-only orchestration; no kernels/device code) |
| src__device__cuda__check | clean (header; no `__global__`, no kernel params) |
| src__device__cuda__cuda_backend | clean (pure host orchestration; all kernels reach device via `launch_*` wrappers in other TUs; grep-verified zero CUDA qualifiers) |
| src__device__cuda__decode_af_kernel | clean (kernel ptrs already `const __restrict__`; helpers shared `STEPPE_HD`) |
| src__device__cuda__device_buffer | clean (host-only RAII; no kernels) |
| src__device__cuda__device_f2_blocks | clean (CUDA-free seam; accessors const; `F2BlockTensor` passed `const&`) |
| src__device__cuda__device_partial | clean (declaration-only; no kernels) |
| src__device__cuda__f2_block_kernel | clean (both kernels fully `const __restrict__`; helpers `STEPPE_HD`) |
| src__device__cuda__f2_blocks_kernel | clean (both kernels fully `const __restrict__`; helpers `STEPPE_HD`) |
| src__device__cuda__f2_blocks_out | clean (no kernels; host/file I/O only) |
| src__device__cuda__p2p_combine | clean (host-only DMA orchestration; no kernels) |
| src__device__cuda__pinned_buffer | clean (host-only page-locked RAII; no kernels) |
| **src__device__cuda__qpadm_fit_kernels** | **2 LOW (11.1, 11.4)** |

## 2. Counts by task + severity

| Task | HIGH | MED | LOW | Total |
|------|------|-----|-----|-------|
| 11.1 const `__restrict__` on read-only kernel ptrs | 0 | 0 | 1 | 1 |
| 11.2 inconsistent `__host__`/`__device__`/`__global__` | 0 | 0 | 0 | 0 |
| 11.3 host/device helper duplicated vs `__host__ __device__` | 0 | 0 | 0 | 0 |
| 11.4 large by-value structs as kernel params | 0 | 0 | 1 | 1 |
| **Total** | **0** | **0** | **2** | **2** |

Note 11.4's single entry is a confirm-only LOW (recorded to show the task was checked; no remediation needed).

## 3. Findings (no HIGH; both LOW, both in qpadm_fit_kernels)

- [11.1][LOW] src/device/cuda/qpadm_fit_kernels.cu:66,95-96,118-120,186-187,213-215,264-266,315-317,342-345,590-595,635-640,679-681 — The `__device__ inline`/`__device__ __noinline__` small-LA helpers (dev_lu_factor, dev_solve, dev_jacobi_svd_V, dev_seed_ab, dev_opt_A, dev_opt_B, dev_chisq_of, dev_als_weights, and `*_large` variants) take read-only inputs as plain `const double*`/`const int*` WITHOUT `__restrict__`. The `__global__` kernels do mark their pointers `const … * __restrict__`, so on inline the restrict propagates — but the `__noinline__` dev_opt_A/dev_opt_B (:213,:264) and the non-inlined large-path helpers lose that guarantee across the call boundary, so the compiler must assume aliasing (reloads, no vectorization) inside the hottest GEMM/solve loops. Not a correctness issue (deterministic single-statistic stream intact). Suggested: add `__restrict__` to the read-only pointer params (and distinct out-params) of dev_solve/dev_opt_A/dev_opt_B/dev_chisq_of and the `*_large` siblings; pure missed-opt, no math change.
- [11.4][LOW] src/device/cuda/qpadm_fit_kernels.cu:1091-1103 (qpadm_fit_models_kernel) — Confirm-only. The model-batched full-fit kernel takes ~18 by-value params (12 scalars + 6 pointers), all small scalars/pointers, well under the 4 KB constant-param-bank limit — no `__grid_constant__`/by-pointer needed. Across all 27 kernels in the unit, no struct/class is ever passed by value; the param-space limit is never approached. Suggested: none.

## 4. Cross-cutting patterns

- **f2/decode kernels are already exemplary on 11.1.** Every read-only pointer in the f2_block, f2_blocks, and decode_af kernels is already `const double* __restrict__` and every output is `double* __restrict__` (disjoint buffers, so the no-alias promise is sound). The only `__restrict__` gap in the whole group is the qpadm_fit small-LA device helpers — the inner-loop LA routines, not the kernel entry points.
- **11.3 is the inverse of a defect across the group (single-source `STEPPE_HD`).** The catastrophic-cancellation primitives (`het_correction`, `assemble_f2_numerator`, `finalize_f2`) and launch-math (`grid_for`/`cdiv`) are declared `STEPPE_HD`/`constexpr` once in core/internal headers and shared host↔device — the §13 anti-divergence pattern. No host/device fork exists to merge. The one genuine host↔device "twin" (qpadm_fit_kernels' dev_* LA helpers vs the CpuBackend oracle) is correctly NOT merged: the CPU side accumulates in `long double` (parity oracle), the GPU side is native FP64 (production), so a `__host__ __device__` merge would force one precision and break the §12 oracle/production split.
- **Most kernel-scoped TUs hold no device code (11.1/11.2/11.4 vacuous).** 8 of 13 units (block_sink, check, cuda_backend, device_buffer, device_f2_blocks, device_partial, f2_blocks_out, p2p_combine, pinned_buffer) are host-side orchestration / RAII / CUDA-free seams — kernels are defined in the 4 `*_kernel.cu` TUs. The CUDA-free `.hpp` seam (architecture.md §4) deliberately carries no execution-space qualifiers; this is correct-by-contract, not 11.2 drift.
- **No HIGH, no MED, no correctness/aliasing-UB issue group-wide.** Both findings are LOW missed-opt in a single TU; one is confirm-only. Group 11 is essentially clean for steppe's kernel layer.

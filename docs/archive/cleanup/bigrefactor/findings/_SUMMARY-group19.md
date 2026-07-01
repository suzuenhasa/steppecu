# Group 19 — Performance: debug leftovers — ROLLUP SUMMARY

Scope = `kernel` units. Tasks: 19.1 stray `cudaDeviceSynchronize()` left from debugging; 19.2 leftover `printf`/`#if 0` in kernels; 19.3 redundant `__syncthreads()` that no longer guards anything.

Exemption applied (per brief): `STEPPE_CUDA_CHECK_KERNEL`'s `STEPPE_DEBUG_ONLY`/NDEBUG-gated `cudaDeviceSynchronize()` (check.cuh:251-256) is the intentional per-launch async-fault-attribution sync — NOT a leftover.

## 1. Coverage

13 in-scope units (all `src/device/cuda/*`), 13 reviewed, 13 clean, 0 with findings.

| Unit | Result |
|---|---|
| src__device__cuda__block_sink | clean |
| src__device__cuda__check | clean |
| src__device__cuda__cuda_backend | clean |
| src__device__cuda__decode_af_kernel | clean |
| src__device__cuda__device_buffer | clean |
| src__device__cuda__device_f2_blocks | clean |
| src__device__cuda__device_partial | clean |
| src__device__cuda__f2_block_kernel | clean |
| src__device__cuda__f2_blocks_kernel | clean |
| src__device__cuda__f2_blocks_out | clean |
| src__device__cuda__p2p_combine | clean |
| src__device__cuda__pinned_buffer | clean |
| src__device__cuda__qpadm_fit_kernels | clean |

## 2. Counts by task + severity

| Task | HIGH | MED | LOW | Total |
|---|---|---|---|---|
| 19.1 stray cudaDeviceSynchronize | 0 | 0 | 0 | 0 |
| 19.2 leftover printf / #if 0 | 0 | 0 | 0 | 0 |
| 19.3 redundant __syncthreads | 0 | 0 | 0 | 0 |
| **TOTAL** | **0** | **0** | **0** | **0** |

## 3. Top findings

None. No HIGH / MED / LOW findings across all 13 units.

## 4. Cross-cutting observations (verified-clean, not findings)

- **No stray full-device serialize anywhere.** The only `cudaDeviceSynchronize()` reachable from any kernel-launching TU is the NDEBUG-gated one inside `STEPPE_CUDA_CHECK_KERNEL()` (check.cuh:251-256) — the brief's explicit exemption. Host-orchestration TUs synchronize only at the targeted, load-bearing granularity: `cudaStreamSynchronize` on the per-device statistic stream (cuda_backend.cu — e.g. 260/1095/1313/2285, the latter a MEASURED batched-potrs-lane hazard fix), `cudaStreamSynchronize(root_stream)` minimal combine drains (p2p_combine.cu:183/192/297), and per-slot `cudaEventSynchronize` writer drains (block_sink.cu:91/246). Several hot paths actively DOCUMENT having removed per-block/per-chunk syncs (cuda_backend.cu:1029-1030 "NO cudaStreamSynchronize here"; p2p_combine.cu:171-174/288-290 "NO per-peer cudaDeviceSynchronize here") — i.e. debug-sync removal already happened and is annotated.
- **No leftover prints / dead `#if 0`.** Zero `printf`/`fprintf`/`std::cout`/`std::cerr` in any kernel body; diagnostics route exclusively through the structured `STEPPE_LOG_WARN`/`STEPPE_CUDA_WARN` sinks (the §7/§10 "no printf in library code" policy). One unit even documents that an interim `std::fprintf(stderr,…)` debug sink was already removed (f2_block_kernel.cu:216). The only preprocessor conditionals are include guards and the load-bearing `STEPPE_HAVE_EMU_TUNING` precision feature-gates (f2_block_kernel.cu:78-80/199/206/281/290) — feature gates, not `#if 0` dead fences.
- **No stale barriers.** Most CUDA kernels in scope are one-element-per-thread map/scatter/reduce with no `__shared__` memory and thus no `__syncthreads()` at all. The single barrier in the entire group — `add_fudge_diag_models_kernel` (qpadm_fit_kernels.cu:1063) — is load-bearing: it publishes thread-0's `s_tr` trace before all threads read it (genuine RAW). No `__syncwarp()` anywhere.
- **Structural N/A.** Roughly half the in-scope units (block_sink, cuda_backend, device_buffer, device_f2_blocks, device_partial, f2_blocks_out, p2p_combine, pinned_buffer) are pure host-side orchestration / RAII / I/O with no `__global__` kernels, so 19.2-kernel and 19.3 are structurally inapplicable; their 19.1 surface was checked and is clean.

## HEADLINE

Units in scope: 13. Total findings: 0. HIGH: 0. (All 13 clean; Group 19 = no debug leftovers.)

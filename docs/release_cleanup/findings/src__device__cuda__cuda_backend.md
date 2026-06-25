# src__device__cuda__cuda_backend
Files: /home/suzunik/steppe/src/device/cuda/cuda_backend.cu
Subsystem: device-cuda

## Findings

### G5
- [G5.device-cuda-cuda_backend][LOW] cuda_backend.cu:2729,2764,2800 — the INT-clamp cap `0x40000000` is an unnamed literal duplicated at three sites (K_budget clamp, chunk_sz clamp, K_sz clamp). All three mean "clamp to <=2^30 for the CUB num_items / int-kernel ceiling"; a drift between them would mis-size one path. Suggested: hoist a single named `constexpr std::size_t kFstatIntClampMax = 0x40000000;` and reference it at all three.
- [G5.device-cuda-cuda_backend][LOW] cuda_backend.cu:2735 — the per-slot reservoir byte estimate `160` is an unnamed magic literal whose own adjacent comment computes `9·8 + 10·4 = 112 B/slot` then silently uses 160. The 160 is "112 rounded up generously" but nothing names or derives it. Suggested: a named constant (e.g. `kFstatReservoirBytesPerSlot`) with the 112→160 padding rationale in one place.

### G7
- [G7.device-cuda-cuda_backend][MED] cuda_backend.cu:2727-2731 vs 2798-2803 — the bounded-top-K target is computed TWICE through byte-identical clamp steps: `K_budget`/`cap_budget` (2727-2731, used only to size `reservoir_bytes`) and `K_sz`/`CAP_sz`/`K`/`CAP` (2798-2805, the live reservoir state). Same `(cfg.top_k>0)?:kFstatDefaultSweepTopK`, same `>enumerated`, `>0x40000000`, `<1` clamps, same `*2`. A change to the clamp policy must be made in two places or the reservoir sizing and the reservoir state drift. Suggested: compute K/CAP once near the top and reuse for both the byte estimate and the state.

### G8
- [G8.device-cuda-cuda_backend][LOW] cuda_backend.cu:2733-2734 — the comment states `≈ 9·8 + 10·4 = 112 B/slot` but the code on the next line multiplies by 160 (G5 above); the comment understates the constant actually used, so a reader sizing VRAM from the comment is misled. Suggested: reconcile the comment arithmetic with the 160 actually used (state it is 112 padded to 160).

### G13
- [G13.device-cuda-cuda_backend][MED] cuda_backend.cu:1404,1409,1426,1429,1435,1439 (decode_af_compact_autosome); 1557,1562,1573,1576,1582,1586 (decode_af_compact_filter); 2825,2829,2924-2946 (run_fstat_sweep_device) — every `cub::DeviceScan::ExclusiveSum`, `cub::DeviceSelect::Flagged`, and `cub::DeviceRadixSort::SortPairsDescending` returns a `cudaError_t` that is NEVER captured or checked, while every adjacent `cudaMemcpyAsync` in the same blocks IS wrapped in STEPPE_CUDA_CHECK. A failing CUB temp-size query or compaction (e.g. an internal launch/config error) is swallowed and only surfaces — if at all — at the next checked sync, defeating the fail-fast intent. Inconsistent checking within the same routine. Suggested: wrap the CUB calls in STEPPE_CUDA_CHECK (they already return cudaError_t).

### G16
- [G16.device-cuda-cuda_backend][HIGH] cuda_backend.cu:1777-1922 (dates_curve) — `cufftHandle plan_fwd/plan_inv` are RAW (non-RAII) resources created by `cufftPlanMany` (1779-1782) and destroyed only at the function tail (1921-1922). Every `cufft_ok` between them THROWS on error: the two `cufftSetStream` (1783-1784) and, more reachably, the per-sample-loop `cufftExecD2Z`/`cufftExecZ2D` (via the `fft_fwd`/`fft_inv` lambdas, 1833/1837) called n_target times in the 1842-1898 loop. Any throw there leaks BOTH plans (cufftDestroy never reached). This is the one place in the TU a CUDA resource is not RAII-wrapped (contrast the GesvdjInfo RAII at 4173/4224/4254 added for exactly this hazard). Suggested: wrap the cuFFT plans in a move-only RAII type with a non-throwing cufftDestroy dtor (mirror GesvdjInfo / CublasHandle).

(groups checked: G2-G10, G11-G22; no issues found for G2, G3, G4, G6, G9, G10, G11, G12, G14, G15, G17, G18, G19, G20, G21, G22 — this TU is an orchestration/host layer: all device kernels live behind `launch_*` wrappers in other .cu files, device memory/handles/streams/events are RAII (DeviceBuffer/Stream/Event/Cublas/CusolverDn handles), launch/index/syncthreads/coalescing concerns belong to the kernel TUs, the `double`/FP64 usage is intentional per §12, and the int-narrowing at M0 scale is guarded by the explicit B22 M>INT_MAX fail-fast at :346.)

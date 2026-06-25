# src__device__cuda__dstat_kernel
Files: /home/suzunik/steppe/src/device/cuda/dstat_kernel.cu, /home/suzunik/steppe/src/device/cuda/dstat_kernel.cuh
Subsystem: device-cuda

## Findings

### G8
- [G8.comment][LOW] dstat_kernel.cu:61 — Comment cites the host pair_index at `qpfstats.cpp:80`, but the function actually lives at `src/core/stats/qpfstats.cpp:82` (wrong line number, and the path omits the `stats/` subdir). The formula itself matches `pair_index` exactly (verified). Stale citation. Suggested: update to `qpfstats.cpp:82` (stats/).

### G12
- [G12.launch][LOW] dstat_kernel.cu:269-271 — `tilesY = ceil(N/kThreads)` is assigned directly to `grid.y` with no check against the CUDA `gridDim.y` limit (65535). The tiled path is the hot qpfstats/qpDstat path with arbitrary, potentially very large combo counts N; for N > 65535·256 ≈ 16.7M combos the launch would silently fail (`cudaGetLastError()` at :275 catches it, but only as an opaque invalid-config error). qpfstats ~305k is safe today, but the kernel is explicitly "design-for-scale, batched over the N axis." Suggested: document/assert the grid.y bound, or fold extra combo-tiles into a grid-stride over blockIdx.y.

### G18
- [G18.correctness][LOW] dstat_kernel.cu:218,221 — den uses `het[pair_index_lo_hi(min(p1,p2), max(p1,p2), P)]`. `min`/`max` resolve to the CUDA device `int` builtins here, which is correct, but it relies on ADL/builtin visibility rather than an explicit `::min`/`umin`-style call; if a host `<algorithm>` style `std::min` were ever pulled into device scope this would not compile/behave. Not a current bug (builds clean for device int). Suggested: optionally qualify or note the device-builtin reliance. (Informational; no action strictly required.)

Note: barriers verified correct — the three `__syncthreads()` (dstat_kernel.cu:182, :199, :227) are all reached unconditionally by every thread in the grid-block (the `active` guard at :158-163 and :202-226 wraps only private quad reads, the per-combo accumulate, and the final write, never a barrier), so there is no divergent-barrier UB even for OOB threads in the last combo-tile. Index widths checked: SNP column offset `col = (long)P*s` (:175, :98), output index `out = k*(long)n_block + b` (:231, :113), and `cell`/`total` (:79-80) are all `long`, so no int-index overflow at P~2500/M~584k/N-combo scale. `npairs` is `int` in the tiled kernel (:144) but the tiled path only runs when smem ≤ 99 KB → P ≲ 112 → npairs ≲ 6216, no overflow; the wrapper computes it as `long` (:251). Launches are checked (cudaGetLastError at :275, :286) and the opt-in cudaFuncSetAttribute is STEPPE_CUDA_CHECK-wrapped (:264). Native-FP64 accumulation is the intentional §12 cancellation carve-out (not flagged). const __restrict__ present on all read-only kernel pointers.

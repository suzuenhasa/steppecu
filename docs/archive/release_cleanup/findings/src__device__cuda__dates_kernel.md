# src__device__cuda__dates_kernel
Files: /home/suzunik/steppe/src/device/cuda/dates_kernel.cu, /home/suzunik/steppe/src/device/cuda/dates_kernel.cuh
Subsystem: device-cuda

## Findings

### G5
- [G5.hardcoded][LOW] dates_kernel.cu:278 — `coarse = 4000` is the v-grid resolution baked into the kernel body and also drives the refine bracket `1.0/coarse` at lines 303-304; it is a parity-load-bearing DATES constant but is a magic literal local to the function. Suggested: hoist to a named `constexpr int kCoarseGridSize = 4000;` so the grid count and the bracket width can never drift.
- [G5.hardcoded][LOW] dates_kernel.cu:305 — ternary-refine iteration count `200` is an unnamed literal. Suggested: name it (e.g. `kTernaryIters`) alongside the coarse-grid constant.

### G8
- [G8.comments][LOW] dates_kernel.cuh:73 — comment says "s12 += dd11[d], s11 += dd02[d]"; the kernel (dates_kernel.cu:187-188) does map `s12 += dd11[cell]` and `s11 += dd02[cell]` (the s11/s12 naming is deliberately crossed relative to the dd index), so the comment is accurate but the cross-mapping is non-obvious. The rationale for `s11<-dd02` / `s12<-dd11` is not stated. Suggested: add a one-line note on why the output-stat index does not match the dd-moment index.

### G18
- [G18.correctness][LOW] dates_kernel.cu:70-76 — the block reduction `for (int off = blockDim.x/2; off>0; off>>=1)` is correct ONLY because `blockDim.x` is a power of two; the sole launch site passes `kBlock=256` (line 334), so this holds today. It is a latent trap if a future launch uses a non-power-of-two block. Suggested: note the power-of-two precondition at the loop, or guard with the odd-remainder handling.

## Notes (no defect)
- G4/scale: all element/cell indices that can exceed 2^31 (M~584k, n_chrom*n_fft, n_target*dst_bpr) are computed in `long` before use (e.g. cu:50, 113-117, 134-135, 144-145, 155-160, 174-179, 205-208); the narrowing casts to `int kc`/`int j`/`int lag`/`int d` are bounded by n_chrom (~23) and n_fft/diffmax (per-chrom map grid), which are int-sized by construction. No int-index overflow at scale.
- G12: every kernel that is one-elem-per-thread uses `if (idx >= total) return;` bounds guards; the repack kernel (cu:207-208) uses a proper grid-stride loop; grids are derived via `grid_for` = (n+kBlock-1)/kBlock (cu:326), not hardcoded. kBlock=256 is %32.
- G13: every launch is followed by `STEPPE_CUDA_CHECK(cudaGetLastError())`; the NDEBUG-gated kernel-sync is the intentional STEPPE_CUDA_CHECK_KERNEL path (not present here, not required).
- G11: all read-only kernel pointers are `const __restrict__`; kernel params are scalars/pointers (no large by-value structs).
- G14/G16/G17: no allocations or owned resources in this TU — kernels operate on caller-owned device buffers (non-owning views); ownership/RAII lives in the backend.
- G18: `__syncthreads()` in regress_dots_kernel (cu:69,75) is reached uniformly by all threads (no divergent early return before it — the missing/valid branch only zeroes the partials, threads still fall through). No missing/divergent sync.

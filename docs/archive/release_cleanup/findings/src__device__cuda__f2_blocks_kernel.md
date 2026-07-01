# src__device__cuda__f2_blocks_kernel
Files: /home/suzunik/steppe/src/device/cuda/f2_blocks_kernel.cu, /home/suzunik/steppe/src/device/cuda/f2_blocks_kernel.cuh
Subsystem: device-cuda

## Findings

### G6
- [G6.naming][LOW] f2_blocks_kernel.cu:267 — `int twoP = kF2StackedBlocks * P` shadows the `long twoP` used inside both kernels (lines 90, 169) with the SAME name but a different type and a different role (here it is a cuBLAS `m`/`lda` extent; in the kernels it is an index scale factor). One file, one identifier, two types — mildly confusing when cross-reading the GEMM strides against the kernel index math. Suggested: rename the host-side one (e.g. `twoP_dim`) to signal it is the GEMM row extent, not the index scale.

## Notes (verified clean, not findings)
- G4/scale: gather kernel index math (lines 89-106) is fully widened to `long` (Pl, twoP, Psp, twoPsp) before the `c`/`k` products; assemble kernel (lines 168-190) uses `size_t` throughout so the resident `P*P*n_block` (>2^31 at scale) cannot int-overflow. GEMM strides passed as `long`/`static_cast<long>(P)*P` match cuBLAS `long long` stride params on LP64. `int twoP = 2*P` (line 267) is 5000 at P=2500, safely within int for the cuBLAS `m`/`lda` int params.
- G12: bounded-grid kernels with explicit `if (i>=P || c>=s_pad || k>=n_in_group) return;` guards; no grid-stride loop is needed because the batch z-extent is tiled to `kMaxGridZ` by the backend and asserted via `grid_z_extent`. Grid derived via `grid_for`/`grid_z_extent`, not hardcoded; block is `dim3(kCdivBlock,kCdivBlock)` (16x16=256, %32 ok).
- G11: all read-only kernel pointers carry `const ... __restrict__`; `__launch_bounds__(kCdivBlock*kCdivBlock)` present and documented.
- G13: both launches are followed by `STEPPE_CUDA_CHECK_KERNEL()`; cuBLAS calls wrapped in `CUBLAS_CHECK`.
- G18: no `__syncthreads`/warp intrinsics/shared memory in these kernels (pure elementwise gather/scatter), so no sync-divergence or RAW/WAR hazard surface.
- G20: the documented uncoalesced transpose read in assemble (lines 134-150) is the previously-reviewed 20.1 ACCEPTED-COST branch (mathematically-required symmetric f2 read off the GEMM-dominated critical path) — not re-flagged.

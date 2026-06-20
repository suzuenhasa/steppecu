# Review findings — src__device__cuda__f2_blocks_kernel

Files: /home/suzunik/steppe/src/device/cuda/f2_blocks_kernel.cu, /home/suzunik/steppe/src/device/cuda/f2_blocks_kernel.cuh

## Group 4 — Type & numeric

No Group 4 issues found.

Notes (verified clean, not findings):
- 4.2/4.6 (index width / overflow before widening): every global index into the [P×M]/[2P×M] feeder and the [P×s_pad×n] slabs in gather_group_kernel is computed in `long` with operands widened first — Pl/twoP/Psp/twoPsp are `long` (cu:84-87), and the source index `twoP * src` (cu:101, src derived from the `long` block_offsets, cu:98) is long×long, so the 2P·M ≈ 2.9e9 product cannot overflow int. The resident-tensor scatter in assemble_blocks_group_kernel uses `size_t` throughout (Pp/twoP/slab bases cu:141-147,160), so the P·P·n_block ≈ 4.7e9 index `Pp*Pp*id` is 64-bit. The GEMM strides (cu:240-241,248,255,263) all widen with `static_cast<long>(...)` before the multiply.
- 4.1 (float-vs-double): all literals/math are `double` (1.0/0.0/0.5-class, cu:107-110,237-238) and FP64-by-design per the parity law — intentional, no narrowing.
- 4.3 (allocation sizing): this TU performs no cudaMalloc/new; all buffers are passed in.
- 4.4/4.5 (unsigned countdown / signed-unsigned compares): kernels use forward `int` thread indices guarded by `>=` bounds (cu:82,139); the grid extents are bounded by grid_z_extent's `1 <= n <= kMaxGridZ` assert, so the `static_cast<unsigned>` of n_in_group is safe. No reverse/unsigned loops.
- 4.7 (host/device pointer typing): raw `double*`/`const int*`/`const long*` carry no host-vs-device space distinction, but this is consistent with the codebase-wide convention (d-prefixed device params, documented in the .cuh) and is a pervasive project-level seam choice, not a defect localized to this unit.
- Latent (LP64-only, not a finding): the `long` strides passed to cublasGemmStridedBatchedEx (a `long long` param) are value-preserving on the Linux x86-64 target; would narrow on an LLP64 platform, which steppe does not target.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

Notes (verified clean, not findings):
- 2.1 (dropped Maxwell/Pascal/Volta archs): no per-TU arch/compute flags or CMake arch lists in either file; nothing names sm_50/60/70 or compute_50/60/70. Arch selection is a build-system concern, out of this unit's scope.
- 2.2 (texture/surface references removed in CUDA 12): no `texture<...>`, `surface<...>`, `cudaBindTexture*`/`cudaUnbindTexture*`, or `tex1D/2D/3D`/`surf*` anywhere. Both kernels read via plain `const double* __restrict__` global loads (cu:102-105, 149-154) — no texture/surface path to port.
- 2.3 (non-`_sync` warp intrinsics): no warp intrinsics at all — no `__shfl*`, `__ballot`, `__any`, `__all`, `__activemask`. The kernels are independent per-thread map/scatter with no warp-collective communication.
- 2.4 (cudaThreadSynchronize -> cudaDeviceSynchronize): no `cudaThreadSynchronize` (or any host-side device sync) in this TU; the launch wrappers use only `STEPPE_CUDA_CHECK_KERNEL()` post-launch error checks (cu:203, 285).
- API surface is current CUDA 13: cuBLAS v2 (`cublasGemmStridedBatchedEx`, `cublasHandle_t`, `cublasComputeType_t`; cu:236, 245-264) and runtime `cudaStream_t` (cu:176, 271) — none deprecated/removed.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

Notes (verified clean, not findings):
- 3.1 (commented-out blocks): every comment in both files is explanatory/design-rationale (spike verdict, grid-math invariants, slab-layout maps, precondition rationale; e.g. cu:1-36, 57-68, 177-195, 227-235) — no disabled `// foo();`-style executable code retained "just in case".
- 3.2 (unreachable code): no `#if 0`, and no statements after `return`/`break`. The kernel early-return guards (cu:82, cu:139) gate the kernel body in the normal CUDA out-of-range pattern, not dead-after-return code.
- 3.3 (unused symbols): all 9 includes are live — `<cublas_v2.h>`/`<cuda_runtime.h>`/`<library_types.h>` (cublasHandle_t, cublasGemmStridedBatchedEx, cudaStream_t, CUDA_R_64F; cu:236,245-264,176), f2_estimator.hpp (assemble_f2_numerator/finalize_f2, cu:157,161), host_device.hpp (STEPPE_ASSERT, cu:185,221,224), launch_config.hpp (grid_for/grid_z_extent, cu:197-199,280-282), check.cuh (CUBLAS_CHECK/STEPPE_CUDA_CHECK_KERNEL, cu:245,203,285), f2_block_kernel.cuh (f2_compute_type, cu:236), config.hpp (kCdivBlock, cu:196,279). No unused params: every wrapper/kernel parameter is read. Locals one/zero/ct/twoP/Psp/twoPsp in run_f2_gemms_group are all consumed (cu:245-264).
- 3.4 (computed but unread): every computed index/value is read — slab bases and per-element offsets in both kernels feed loads/stores (cu:84-111, 141-162); no assigned-never-read locals.
- Comment-accuracy nit (not a finding): the include comment at cu:47 lists `engage_f2_precision` among the SHARED policy symbols, but only `f2_compute_type` is called in this TU — `engage_f2_precision` is engaged by the caller before the group loop (cu:233-234, cuh:71). The include itself is still required for `f2_compute_type`, so it is not removable; the comment accurately documents the shared-policy header's surface rather than this TU's call set, so this is documentation, not a dead symbol.

# Review findings — src__core__fstats__f2_from_blocks

Files: /home/suzunik/steppe/src/core/fstats/f2_from_blocks.cpp, /home/suzunik/steppe/src/core/fstats/f2_from_blocks.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

This unit is a thin host-side orchestration/dispatch layer: it validates the Q/V/N + BlockPartition contracts and dispatches through the ComputeBackend seam. It performs no FP math, no device allocation, and computes no global indices/offsets into the f2 tensor or genotype matrix (that arithmetic lives in the backend, not here).
- 4.1: No floating-point math; only dispatch + integer comparisons. N/A.
- 4.2: The only loop (f2_from_blocks.cpp:80-87) uses `long s` against `long M` and subscripts via `static_cast<std::size_t>(s)` (line 83). M (<=~584131) fits in `long`; no 32-bit `int` index into a >2^31 array is built here. Clean.
- 4.3: No cudaMalloc/new/DeviceBuffer in this unit. N/A.
- 4.4: Only loop (line 82) is `for (long s = 0; s < M; ++s)` — signed, ascending; no unsigned countdown. Clean.
- 4.5: All loop/bound comparisons are same-signedness — `id >= n_block` (int/int, line 84), `block_id.size() == static_cast<std::size_t>(...)` (line 108), `static_cast<long>(n_block) <= M` (lines 116/118), `s < M` (long/long, line 82). Clean.
- 4.6: No multiplicative index arithmetic (no `i*P+j` style); line 83 is a single subscript. No int overflow before widening. Clean.
- 4.7: `partition.block_id.data()` -> `const int*` across the seam (line 150) is the defined CUDA-free backend contract for host memory, not a host/device-pointer confusion within this unit. Clean.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

This unit is host-pure C++20 (layer=core, is_cuda=false): it includes only the CUDA-free seam (device/backend.hpp), the shared views, the block rule, the public config/fstats headers, and `<cstddef>` (f2_from_blocks.cpp:37-46; f2_from_blocks.hpp:11-15) — no `<cuda_runtime.h>`/`<cuda.h>`. A grep over both files for `sm_*`/`compute_*` arch flags, `texture`/`surface`/`cudaBindTexture*`, non-`_sync` warp intrinsics (`__shfl`/`__ballot`/`__any`/`__all`/`__activemask`), `cudaThreadSynchronize`/`cudaDeviceSynchronize`, kernel-launch `<<<>>>`, and `__global__`/`__device__` returned no matches.
- 2.1 (dropped Maxwell/Pascal/Volta archs): no architecture flags or CMake arch lists in this unit; it is not a translation unit compiled by nvcc. N/A.
- 2.2 (removed texture/surface references): no `texture<...>`/`surface<...>` or `cudaBindTexture*` — no device memory access of any kind here. N/A.
- 2.3 (non-`_sync` warp intrinsics): no warp intrinsics; the only loop (f2_from_blocks.cpp:82-87) is plain host iteration. N/A.
- 2.4 (`cudaThreadSynchronize` -> `cudaDeviceSynchronize`): no CUDA runtime sync calls; `core` reaches the GPU only via the ComputeBackend seam and issues no device call directly. N/A.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

This unit is a thin host-side orchestration/dispatch layer with no dead or commented-out code.
- 3.1: All comments in both files (f2_from_blocks.cpp:1-36, 52-61, 72-78, 91-102, 127-131, 139-147; f2_from_blocks.hpp:1-7, 19-31, 36-51) are documentation/rationale (layering, B11 fail-fast contract, IWYU notes) — no commented-out statements or expressions kept "just in case". Clean.
- 3.2: No `#if 0` and no code after `return`/`break`. The `#ifndef NDEBUG`/`#endif` block (f2_from_blocks.cpp:79-89) is intentional conditional compilation, not unreachable dead code: `block_ids_dense_nondecreasing` is referenced from the debug STEPPE_ASSERT at line 118 (same `#ifndef NDEBUG` regime). Clean.
- 3.3: All includes are used — `<cstddef>` (cpp:46) for `std::size_t` (lines 83, 108); `device/backend.hpp` (`ComputeBackend`/`F2Result`); `views.hpp` (`MatView`); `host_device.hpp` (`STEPPE_ASSERT`); `block_partition_rule.hpp` (`BlockPartition`); `fstats.hpp` (`F2BlockTensor`); `config.hpp` (`Precision`). Helpers `validate_qvn` (cpp:62), `validate_partition` (cpp:103), `block_ids_dense_nondecreasing` (cpp:80) are all called (lines 132, 148, 149, 118). The `[[maybe_unused]]` params (cpp:62-63, 103-104) are intentionally so for the NDEBUG build where STEPPE_ASSERT collapses to a no-op — not unused symbols to remove. Clean.
- 3.4: No computed-but-unread values. `prev` (cpp:81) is written and read across loop iterations (lines 84-85); the `[[nodiscard]]` result of `block_ids_dense_nondecreasing` is consumed by the STEPPE_ASSERT at line 118. Clean.


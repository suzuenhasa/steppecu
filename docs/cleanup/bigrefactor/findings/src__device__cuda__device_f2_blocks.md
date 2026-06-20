# Review findings — src__device__cuda__device_f2_blocks

Files: src/device/cuda/device_f2_blocks.cu, src/device/cuda/device_f2_blocks_impl.cuh, src/device/device_f2_blocks.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Notes (why clean), so the result is auditable:
- 4.2/4.6 index width (the real risk at scale): size() in device_f2_blocks.hpp:48-51
  widens EACH int operand to std::size_t BEFORE multiplying
  (static_cast<std::size_t>(P) * static_cast<std::size_t>(P) * static_cast<std::size_t>(n_block)).
  P*P*n_block reaches ~10^10 (> 2^31), but the product is formed in size_t, so no
  int overflow. Every consumer of the flat count (.cu:44 `total = size()`, .cu:74
  `total = out.size()`) takes that size_t value. No int global index is computed.
- 4.3 allocation sizing: bytes computed as `total * sizeof(double)` (.cu:54, .cu:85)
  with total already size_t — byte count, not element count, passed to cudaMemcpy.
  DeviceBuffer<double>(total) (.cu:83-84) takes an element count by design and the
  owner itself guards the n*sizeof(T) byte product (device_buffer.cuh ctor). No
  missing `* sizeof` and no element/byte confusion.
- 4.4/4.5 loops: none in any of the three files — no unsigned countdown, no
  signed/unsigned loop-bound compare. N/A.
- 4.1 float/double: all buffers/scalars are double (FP64 by design, §12 parity).
  No narrowing temp. N/A.
- 4.7 host/device pointer typing: D2H/H2D directions are explicit and correct
  (to_host = DeviceToHost; upload = HostToDevice); device pointers come from
  impl->f2.data() and host from std::vector::data(). The raw `double*` accessors
  lack a compile-time space tag, but that is the project-wide DeviceBuffer::data()
  convention (not introduced here) and not a numeric defect — not flagged.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Notes (why clean), so the result is auditable:
- 2.1 Dropped archs: these three files are TU/header source, not CMake. No sm_/
  compute_ arch flags, no __CUDA_ARCH__ guards, no compute-capability literals
  anywhere in the unit. Arch lists live in CMake, not here. N/A.
- 2.2 Texture/surface REFERENCES: no `texture<...>`, no `surface<...>`, no
  cudaBindTexture*/cudaBindSurface*/cudaCreateTextureObject in any file. The unit
  touches no textures/surfaces at all. N/A.
- 2.3 Non-_sync warp intrinsics: there are NO kernels in this unit (no __global__,
  no __device__ code). No __shfl*/__ballot/__any/__all/__syncwarp usage — these are
  host-side TUs (special members, to_host D2H, upload H2D) plus the Impl struct. N/A.
- 2.4 cudaThreadSynchronize: absent. The only CUDA runtime calls are cudaGetDevice,
  cudaSetDevice, cudaMemcpy (with cudaMemcpyDeviceToHost/HostToDevice), and cudaFree
  (via DeviceBuffer dtor) — all current in CUDA 13, none deprecated/removed.
  device_f2_blocks.cu:50,52,57,59,78,80,86,88. No deprecated sync API.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Notes (why clean), so the result is auditable:
- 3.1 Commented-out blocks: none. All comments in the unit are documentation —
  file-header banners (.cu:1-5, _impl.cuh:1-5, .hpp:1-6), Doxygen doc comments,
  the multi-line TODO(multigpu-host-bounce) design note (.hpp:86-91), and member
  banners. No statement/expression is commented out "just in case". The existing
  <!-- --> blocks in this findings file are reviewer notes, not source.
- 3.2 Unreachable code: no `#if 0`/`#if 0`-style guards anywhere. The two early
  returns are GUARDED and reachable — to_host() `if (total == 0 || !impl) return out;`
  (.cu:47) and upload_..._to_device() `if (total == 0) return out;` (.cu:75). No code
  follows an unconditional return/break/continue. N/A.
- 3.3 Unused symbols: every include is consumed — .cu: <cuda_runtime.h> (cudaMemcpy/
  cudaGetDevice/cudaSetDevice), <cstddef> (std::size_t), <memory> (std::make_unique
  .cu:82), check.cuh (STEPPE_CUDA_CHECK), device_buffer.cuh (DeviceBuffer),
  pinned_buffer.cuh (RegisteredHostRegion .cu:55-56), fstats.hpp (F2BlockTensor).
  _impl.cuh: device_f2_blocks.hpp + device_buffer.cuh both used in the Impl struct.
  .hpp: <cstddef>/<memory>/<vector>/fstats.hpp all used. No unused params (the two
  RAII guard structs read `prev`/`device_id`); no unused locals or static helpers.
- 3.4 Computed but unread: none. RegisteredHostRegion pin_f2/pin_vp (.cu:55-56) and
  the `G restore{prev}` guards (.cu:51, .cu:79) are RAII objects held for their dtor
  side effects (un-pin / restore device), which is a use — not dead stores. Every
  assigned scalar (`total`, `bytes`, `prev`, out.* shape fields) is subsequently read.
-->


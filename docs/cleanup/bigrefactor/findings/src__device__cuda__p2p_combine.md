# Review findings — src__device__cuda__p2p_combine

Files: /home/suzunik/steppe/src/device/cuda/p2p_combine.cu, /home/suzunik/steppe/src/device/p2p_combine.hpp

## Group 4 — Type & numeric

- [4.7][LOW] src/device/cuda/p2p_combine.cu:138-141,188-191,264-267 — Raw `double*` are used for both host (`out.f2.data()`) and device (`dResult_f2.data()`/peer `src_f2`) pointers with no host-vs-device type distinction; a swapped space relies on the explicit `cudaMemcpy*` direction enum / peer device ids to fault rather than the type system. Consistent with the project-wide backend seam (DeviceBuffer is the owner). Suggested: optional — if the codebase later adopts a thin DevicePtr/HostPtr wrapper, route these through it; no functional change needed now.

Note (4.2/4.6 — the scale-critical path, verified CLEAN): every global index into the up-to-~10^10-element f2/vpair tensor is widened to `std::size_t` BEFORE the multiply — `slab = (size_t)P * (size_t)P` (cu:94-95, 227-228), `total = slab * (size_t)n_block_full` (cu:97, 229), `part_elems = slab * (size_t)part.n_block_local` (cu:134, 260), `dst_off = slab * (size_t)part.b0` (cu:136, 262). No `int` product overflows before widening. The only in-`int` arithmetic, `part.b0 + lb` (cu:129, 255), is bounded by n_block_full (~757), far below 2^31.
Note (4.3 — CLEAN): `DeviceBuffer<double>(total)` / `.resize(total)` / `.assign(n_block_full,0)` are element-count APIs; raw `cudaMemcpy*` byte counts correctly carry `* sizeof(double)` (cu:135, 185, 261).
Note (4.1/4.4/4.5 — CLEAN/N-A): all math is FP64 by design (no wrong narrowing); loops are ascending `size_t<size_t` (cu:122,249) and `int<int` (cu:128,254) — no unsigned countdown, no signed/unsigned mismatch.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

Note (2.1 — N/A): this unit is a host-side translation unit only — no `__global__`/`__device__` kernels, no `<<<>>>` launches, and no per-arch code; it contains no `sm_*` flags or CMake arch lists (those live in the build files, not here).
Note (2.2 — CLEAN): no texture/surface usage. No `texture<...>`, `surface<...>`, `cudaBindTexture*`, `tex1Dfetch`, or any CUDA-12-removed texture-reference API appears; the unit moves bytes only via `cudaMemcpyAsync` / `cudaMemcpyPeerAsync` (cu:145-148, 166-169, 188-191, 271-273, 283-286).
Note (2.3 — N/A): no warp intrinsics at all (no kernels). No `__shfl*`/`__ballot`/`__any`/`__all`/`__activemask`, deprecated non-`_sync` or otherwise.
Note (2.4 — CLEAN): synchronization uses the current `cudaStreamSynchronize` (cu:183, 192, 297) and `cudaGetDevice`/`cudaSetDevice` (cu:81, 86, 216, 221); no `cudaThreadSynchronize` (or any other `cudaThread*` removed alias). The comments at cu:171-174, 288-290 reference `cudaDeviceSynchronize` only in prose explaining its deliberate absence — no removed-API call. The .hpp (p2p_combine.hpp) is CUDA-free by contract and contains no CUDA API at all.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

Note (3.1 — CLEAN): the large comment blocks (cu:1-41, cu:65-105, cu:197-203; the doc-comments in p2p_combine.hpp:1-110) are design/parity rationale (architecture.md §11.4/§12 references), NOT commented-out code "kept just in case" — no disabled statements exist.
Note (3.2 — CLEAN): no `#if 0`, no `#ifdef`-disabled regions, no code after `return`/`break`. The two `if (part.empty()) continue;` guards (cu:132, 258) skip only the empty-shard case; the following copy code is reachable for non-empty shards.
Note (3.3 — CLEAN): every parameter is read — `shards` (cu:63, 206) feeds `validate_resident_partials` (cu:70-71, 209-210); `partials`, `P`, `n_block_full`, `root_device_id` all consumed. Every `#include` is used: `<cstddef>`/`<span>` (std::size_t, std::span); check.cuh (STEPPE_CUDA_CHECK/WARN); device_buffer.cuh (DeviceBuffer); device_partial_impl.cuh (`part.impl->f2/vpair`); device_f2_blocks_impl.cuh (`DeviceF2Blocks::Impl`, cu:242); pinned_buffer.cuh (RegisteredHostRegion, cu:186-187); stream.hpp (Stream, cu:91, 224); steppe/fstats.hpp (F2BlockTensor); shard_plan.hpp (DeviceShard); f2_partials_validate.hpp (validate_resident_partials). No unused local/helper.
Note (3.4 — CLEAN): every assigned local is read — `prev_device` (cu:80) → `restore{prev_device}` (cu:85, 220); `part_elems` (cu:134, 260) → `part_bytes` (cu:135, 261); `slab`/`total`/`dst_off`/`bytes` and the dst/src pointers all consumed by the copies/D2H. Nothing computed-but-unread.

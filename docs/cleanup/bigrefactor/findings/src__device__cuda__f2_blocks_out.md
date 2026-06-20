# Review findings — src__device__cuda__f2_blocks_out

Files: /home/suzunik/steppe/src/device/cuda/f2_blocks_out.cu, /home/suzunik/steppe/src/device/f2_blocks_out.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

Notes (clean, but verified at scale — not findings):
- 4.2/4.6 The block-major index into the resident/host tensor (the up-to-~10^10-element f2/vpair buffers) is computed 64-bit: `slab = (size_t)P * (size_t)P` then `off = slab * (size_t)b` (.cu:79,92,100). `slab` is already `size_t`, so the `* b` multiply is 64-bit — no `int` overflow before widening.
- 4.6 The Disk-tier region/total sizes widen every operand before multiplying: `region = (uint64_t)P * (uint64_t)P * (uint64_t)n_block * sizeof(double)` (.cu:68-70) and `total = (size_t)P * (size_t)P * (size_t)n_block` (.cu:144-146). The offset helpers in f2_disk_format.hpp (f2_block_offset/vpair_block_offset) also widen to uint64_t before the `* b` multiply.
- 4.3 Allocation sizing correct: `vector<double>::assign(total, 0.0)` is element-count (.cu:147-148); `pread_all(..., total * sizeof(double), ...)` and `bytes = slab * sizeof(double)` are byte-counts (.cu:80,153,155). No element/byte confusion.
- 4.1 All math is `double` (FP64-by-design); no float narrowing in a parity-critical path.
- 4.4/4.5 No loops in either file — no unsigned countdown / signed-unsigned bound risk.
- 4.7 The D2H accessor takes raw `double*` host out-pointers and copies from `const double*` device pointers (.cu:78,85-96); this is the established backend-seam idiom and not a unit-local defect. The 64-bit file-offset assumption is pinned by `static_assert(sizeof(long) >= 8)` (.cu:53).

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

Notes (verified, not findings):
- 2.1 No SM-arch flags or CMake arch lists in either unit file (build flags live in CMake, out of unit scope); no sm_50/60/70-gated code.
- 2.2 No texture/surface references (no `texture<...>`, `cudaBindTexture*`, `surface<...>`); this TU has no kernels — it is host-side D2H/memcpy/file I/O only.
- 2.3 No warp intrinsics of any kind (no `__shfl*`/`__ballot`/`__any`/`__all` legacy or `_sync` variants); no `__syncthreads`. No kernels in the TU.
- 2.4 No `cudaThreadSynchronize`. The only CUDA runtime calls are current/supported: `cudaGetDevice`/`cudaSetDevice` (.cu:89,90,91), `cudaMemcpy` with `cudaMemcpyDeviceToHost` (.cu:93-96). `cudaMemcpy` is implicitly synchronizing here (no explicit device-sync call needed/used).

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/device/cuda/f2_blocks_out.cu:19 — `#include "device/cuda/pinned_buffer.cuh"` is unused; no `RegisteredHostRegion`/`PinnedBuffer` symbol is referenced anywhere in the TU (the Resident D2H at .cu:93-96 is a plain `cudaMemcpy` into an unpinned `double*`). The "pin the D2H; graceful degrade" intent in the comment was never realized. Suggested: drop the include (and its stale comment), or actually pin the host out-buffer if the perf intent stands.
- [3.3][LOW] src/device/cuda/f2_blocks_out.cu:21 — `#include "core/internal/log.hpp"` is unused; no `STEPPE_LOG_WARN`/`STEPPE_LOG*` macro is invoked. The dtor/move-assign `std::fclose` (.cu:28,36) ignore the return without logging, so the "STEPPE_LOG_WARN (teardown)" comment describes a never-written log call. Suggested: drop the include, or add the intended teardown warn-on-close-failure log.
- [3.3][LOW] src/device/cuda/f2_blocks_out.cu:17 — `#include "device/cuda/device_buffer.cuh"` is a redundant direct include; `DeviceBuffer` is not named in this TU and is transitively provided by `device_f2_blocks_impl.cuh` (the include's own comment says "via device_f2_blocks_impl"). Suggested: rely on the transitive include and remove the direct one (or keep only if an IWYU policy mandates direct includes).
- [3.2][LOW] src/device/cuda/f2_blocks_out.cu:160 — `return F2BlockTensor{};` after the `switch(tier)` is unreachable (all three `OutputTier` enumerators return in their `case`, as the inline comment notes). It is the intentional fall-through to silence the "control reaches end of non-void function" warning given the switch has no `default`. Not a defect — defensive; noted for completeness only. Suggested: leave as-is.

# Review findings — src__core__fstats__f2_partials_validate

Files: /home/suzunik/steppe/src/core/fstats/f2_partials_validate.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Verification notes (not findings):
- 4.1 (float/double): no floating math in this header — pure integer/index extent
  checks plus std::to_string for error strings. N/A.
- 4.2 / 4.6 (index width / overflow before widening): the one place that could
  overflow is the P*P*n_block extent (P up to ~2500, n_block up to ~757 ⇒ ~4.7e9
  > 2^31). It is computed correctly in size_t: `slab = (size_t)P * (size_t)P`
  (lines 92-93) then `want_slabs = slab * (size_t)part.n_block` (lines 122-123),
  matching F2BlockTensor::size()'s own size_t widening (fstats.hpp:74-76). No int
  intermediate. `span_blocks = sh.b1 - sh.b0` (lines 102, 198) is a block COUNT
  (≤ ~757), safe in int; `covered` is `long` and compared as `(long)n_block_full`
  (lines 98, 140-142, 194, 227-229) — no truncation.
- 4.3 (allocation sizing): no cudaMalloc/new/DeviceBuffer here — validation-only,
  CUDA-free header. N/A.
- 4.4 (unsigned countdown): both loops count UP `for (size_t g=0; g<size(); ++g)`
  (lines 99, 195) — no decrementing-unsigned wrap. N/A.
- 4.5 (signed/unsigned compare): loop counter `g` is size_t compared against
  `partials.size()` (size_t) — same signedness (lines 99, 195). Other comparisons
  (`part.n_block != span_blocks`, `part.P != P`) are int-vs-int. Clean.
- 4.7 (host/device pointer typing): no raw pointers; operates on std::span of
  CUDA-free value types (F2BlockTensor / DevicePartial / DeviceShard). N/A.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Verification notes (not findings):
- 2.1 (dropped archs): no CMake / arch lists / sm_* build flags in this header
  (CUDA-free, header-only INLINE per lines 15-22). N/A.
- 2.2 (texture/surface references): no texture<...>, surface<...>, or
  cudaBindTexture* — header is CUDA-free by contract (includes only <cstddef>,
  <span>, <stdexcept>, <string> + CUDA-free steppe headers, lines 31-38). N/A.
- 2.3 (non-_sync warp intrinsics): no __shfl/__ballot/__any/__all/warp intrinsics;
  pure host integer/index validation code. N/A.
- 2.4 (cudaThreadSynchronize): no CUDA runtime API calls of any kind. N/A.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Verification notes (not findings):
- 3.1 (commented-out blocks "just in case"): no commented-out code anywhere — the
  comments (lines 1-27, 42-73, 150-177, inline 91-97, 102, 111-112, 119-125, 194,
  198) are all doc/contract prose explaining the §8 single-source guard, not stashed
  code. Clean.
- 3.2 (unreachable code): no `#if 0`, no code after return/break. Each `throw`
  (lines 82, 87, 105, 114, 129, 143, 186, 191, 201, 208, 214, 221, 230) is on a
  reachable conditional branch; the loops fall through to the final tiling check.
  No early return before a tail. Clean.
- 3.3 (unused symbols/includes/params): all 4 std includes are used — <cstddef>
  (std::size_t, lines 92, 124, 220), <span> (std::span params, lines 76-77,
  180-181), <stdexcept> (std::runtime_error throws), <string> (std::string/
  std::to_string, line 79 onward). All 3 steppe includes name a type actually used:
  steppe/fstats.hpp → F2BlockTensor (line 76, 100), device/shard_plan.hpp →
  DeviceShard (lines 77, 101, 181, 197), device/device_partial.hpp → DevicePartial
  (lines 180, 196). Every param of both functions (who/partials/shards/P/
  n_block_full) is read. Clean.
- 3.4 (computed but unread): `slab` (lines 92-93) is read at lines 122-123;
  `covered` accumulated and read at 142 / 229; `span_blocks` read at 104/140 and
  200/227; `prefix`, `want_slabs`, `want_counts` all consumed in throw strings. No
  dead assignment. Clean.
-->


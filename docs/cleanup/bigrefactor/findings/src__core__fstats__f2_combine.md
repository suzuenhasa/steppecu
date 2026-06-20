# Review findings — src__core__fstats__f2_combine

Files: /home/suzunik/steppe/src/core/fstats/f2_combine.cpp, /home/suzunik/steppe/src/core/fstats/f2_combine.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Verified against the SCALE context (P up to ~2500, n_block up to ~757, total up to ~10^10 > 2^31):
- 4.1: FP64 by design; the only literals are 0.0 (+0.0 init, cpp:64-65) — intentional, no wrong narrowing.
- 4.2/4.6: every global index/offset is widened to std::size_t BEFORE the multiply.
  slab = (size_t)P * (size_t)P (cpp:59-60); total = slab * (size_t)n_block_full (cpp:63, slab already size_t);
  part_slabs = slab * (size_t)part.n_block (cpp:100-101); destination offsets slab*b0 (cpp:103-104) and b0 (cpp:108-110)
  all carry a size_t operand, so the arithmetic promotes to size_t — no int*int product can overflow before widening.
- 4.3: no cudaMalloc/new; std::vector::assign(count,value) (cpp:64-66) takes element counts — sizeof(T) N/A.
- 4.4: only ascending loop `for (std::size_t g=0; g<partials.size(); ++g)` (cpp:96) — terminates.
- 4.5: g and partials.size() are both size_t (cpp:96); part.n_block<=0 is int-vs-int (cpp:98) — no signed/unsigned mismatch.
- 4.7: host-pure unit; all pointers are host double*/int* from std::vector::data() — no device pointers.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Host-pure, CUDA-FREE unit (is_cuda=false, layer=core; hpp:13-14 "CUDA-FREE, host-pure, in steppe::core").
- 2.1 (dropped archs sm_50/60/70): no .cu, no CMake/nvcc arch flags in this unit; both files are C++ compiled into steppe_core without the device toolkit (hpp:14-16). N/A.
- 2.2 (texture/surface references removed in CUDA 12): no texture<>/surface<>/cudaBindTexture* — includes are only <algorithm>,<cstddef>,<span> + project headers (cpp:16-22). N/A.
- 2.3 (non-_sync warp intrinsics): no __shfl*/__ballot/__any/__all or any warp intrinsic — host code only. N/A.
- 2.4 (cudaThreadSynchronize -> cudaDeviceSynchronize): no cuda* runtime calls at all (combine is std::copy_n over host vectors, cpp:103-110). N/A.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
- 3.1: No commented-out CODE kept "just in case". All comments are narrative design/rationale
  (B5/B7/N2/P2). References to removed code describe deletions, not retained dead code:
  "the old `< 0 ? 0 :` ternary was dead" (cpp:62) and "the prior scalar `+=`" (cpp:92) document
  code that was REMOVED, not commented out and kept.
- 3.2: No `#if 0`, no code after return/break. Sole `return out;` at cpp:113 (function end);
  `continue;` at cpp:98 is reachable (empty-shard skip).
- 3.3: All includes used — cpp: <algorithm>→std::copy_n (103-110), <cstddef>→std::size_t,
  <span>→std::span, steppe/fstats.hpp→F2BlockTensor, device/shard_plan.hpp→DeviceShard,
  f2_partials_validate.hpp→validate_f2_partials (36); hpp: <span>/fstats.hpp/shard_plan.hpp all
  used in the decl (76-79). All locals read: slab (63,101,103,104), total (64,65), part (98,100,
  103,104,108), b0 (103,104,110), part_slabs (103,104). All 4 params (partials/shards/P/n_block_full) used.
- 3.4: No computed-but-unread values; every assignment (out.P, out.n_block, out.f2/vpair/block_sizes,
  slab, total, b0, part_slabs) feeds a later read or the returned `out`.
-->


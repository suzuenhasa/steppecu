# Review findings — src__device__vram_budget

Files: src/device/vram_budget.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Notes (why clean), for the record:
- 4.1: only FP64 used is the intentional utilization-fraction multiply at line 107
  (`kMaxVramUtilizationFraction * static_cast<double>(net)`); no wrong narrowing.
- 4.2/4.6: at scale (P≤2500, n_block≤757) `2·P²·n_block` ≈ 9.5e9 overflows 32-bit
  int, but lines 64-65 cast P/n_block to std::size_t BEFORE multiplying (line 67),
  and per_block_chunk_bytes (lines 81-84) and the min/clamp (lines 150-155) are all
  done in std::size_t before the final narrowing. The `2u`/`4u` literals promote to
  size_t in the mixed expression. The final `static_cast<int>` at line 156 is
  provably ≤ kMaxGridZ (65535), so it cannot overflow.
- 4.3: no allocation here (pure budget arithmetic); `* sizeof(double)` present at
  lines 67 and 84 where byte counts are produced.
- 4.4/4.5: no loops.
- 4.7: no pointers — all params are scalar std::size_t / int.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Notes (why clean), for the record:
- This header is host-pure / CUDA-FREE by design (lines 13-18, "No CUDA header here";
  only includes <algorithm>, <cstddef>, launch_config.hpp, steppe/config.hpp).
- 2.1: no arch/sm_xx build flags or CMake arch lists here (build-system concern, not
  this header). kMaxGridZ (65 535, line 152) is a capability-independent grid-z limit,
  valid on every CUDA-13 arch (min sm_75); not a dropped-arch reference.
- 2.2: no texture<...> / surface<...> / cudaBindTexture* — no texture/surface API at all.
- 2.3: no warp intrinsics (no __shfl/__ballot/__any/__all, _sync or otherwise).
- 2.4: no cudaThreadSynchronize / cudaDeviceSynchronize — no CUDA runtime call exists
  in this TU.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Notes (why clean), for the record:
- 3.1: no commented-out code blocks kept "just in case"; all comments (lines 1-28,
  the /// doc-comments, the inline rationale at 66, 74-75, 81-82, 92-97, 113-122,
  125-132, 148-149, 153-154) are explanatory prose, not disabled code.
- 3.2: no #if 0, no #ifdef-disabled regions; the only return-before-code is the
  early `return 0` at line 142 (nb_total <= 0 guard) and line 63 (P/n_block <= 0) —
  both are guard clauses on a separate path, the code after them is reachable on the
  other branch. No code after an unconditional return/break.
- 3.3: every include is used — <algorithm> (std::min line 151, std::max line 155),
  <cstddef> (std::size_t throughout), launch_config.hpp (core::kMaxGridZ line 152),
  steppe/config.hpp (kMaxVramUtilizationFraction lines 45/107, kCublasWorkspaceBytes
  line 105). All 4 functions (resident_tensor_bytes, per_block_chunk_bytes,
  chunk_budget_bytes, max_blocks_per_chunk) are inline header API consumed by the
  backend and host tests (header purpose, lines 8-12), not dead symbols. No params
  go unread: every parameter feeds the return value in each function.
- 3.4: no computed-but-unread locals — p, nb (62-67); p, sp, slab (81-84); reserved,
  net (105-107); budget, per_block, fit, capped, clamped (143-156) each flow into the
  returned expression. The static_assert (45-47) is consumed at compile time.
-->


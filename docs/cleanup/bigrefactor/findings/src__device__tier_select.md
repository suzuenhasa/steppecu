# Review findings — src__device__tier_select

Files: /home/suzunik/steppe/src/device/tier_select.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Verification notes (not findings):
- 4.2/4.6 (index/overflow at scale): All sizing arithmetic widens to std::size_t
  BEFORE multiplying. resident_working_set_bytes (line 58) does
  static_cast<std::size_t>(P) * static_cast<std::size_t>(M) first; streamed_working_set_bytes
  (lines 85-91) widens P/max_tile/max_nb/max_s_pad to size_t before every product; the
  reused resident_tensor_bytes (vram_budget.hpp:64-67) likewise widens P/n_block first.
  No int global index/offset, no i*P+j in int. Clean at P~2500, M~584131, n_block~757.
- 4.1 (float/double): only float math is the budget fractions (lines 125, 128),
  kResidentTierVramFraction/kHostTierRamFraction * static_cast<double>(free_vram/host_ram).
  FP64 is intentional and the result is only a parity-neutral budget threshold. Not a finding.
- 4.3 (alloc sizing): no cudaMalloc/new/DeviceBuffer here; byte computations correctly
  include sizeof(double) (lines 59, 92). N/A.
- 4.4/4.5 (unsigned countdown / signed-unsigned compares): no loops; guards (P<=0, M<=0,
  max_tile<0) are signed-vs-literal. Clean.
- 4.7 (host/device pointer typing): only pointer is const char* env_value (host env string,
  line 138); no device pointer. N/A.
- Note: 7u/4u/2u/3u unsigned-int literals multiplied against std::size_t operands promote
  the literal to size_t (LP64), so no 32-bit truncation. Correct as written.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Verification notes (not findings): host-pure, CUDA-FREE header (only #include <cstddef>;
no CUDA header per lines 12, 22). 2.1 no CMake/arch flags or sm_* lists here. 2.2 no
texture<>/surface/cudaBindTexture* — pure std::size_t budget arithmetic. 2.3 no warp
intrinsics (no __shfl/__ballot/__any/__all/__activemask). 2.4 no cudaThreadSynchronize /
no CUDA runtime calls at all. All four Group 2 tasks N/A.
-->

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/device/tier_select.hpp:82 — `streamed_working_set_bytes` is an unreferenced inline helper: a codebase-wide grep finds zero call sites (only this definition). Its own doc comment (lines 73-75) admits it is "NOT used by select_output_tier" and is merely "exposed for the bench's high-P feasibility narrative and any future select" — but no bench or other caller invokes it. A documented-intent dead helper kept "just in case". Suggested: drop it (and its `max_tile`/`max_nb`/`max_s_pad` params) until a real caller needs it, or wire it into the bench/select that the comment promises.

<!--
Verification notes (not findings):
- 3.1 (commented-out blocks): none. All non-code lines are explanatory doc comments
  describing live policy (lines 1-18, 29-32, 42-55, 62-76, 95-117, 133-136); no
  commented-out statements kept "just in case".
- 3.2 (unreachable / #if 0 / code after return): no #if 0; the early `return`s in
  resident_working_set_bytes (57), streamed_working_set_bytes (84), select_output_tier
  (121), are guard returns with reachable code after only on the other branch — all
  reachable. No dead branches.
- 3.3 (unused includes/symbols): includes both used — vram_budget.hpp (24) for
  resident_tensor_bytes (lines 15,122) + DeviceF2Blocks footprint reuse; config.hpp (25)
  for kCublasWorkspaceBytes (59,92), kResidentTierVramFraction (125), kHostTierRamFraction
  (128), DeviceConfig::ForceTier (138). <cstddef> for std::size_t. All other declared
  symbols ARE referenced elsewhere: OutputTier (f2_blocks_out.hpp:63, stream_f2_blocks.hpp:36,
  host_ram.cpp, cuda_backend.cu:405/411, f2_blocks_multigpu.cpp), resident_working_set_bytes
  (used by select_output_tier line 123), free_host_ram_bytes (host_ram.cpp:17,53; tests;
  f2_blocks_multigpu.cpp:352), select_output_tier (host_ram.cpp:53; tests; bench),
  resolve_output_tier (f2_blocks_multigpu.cpp:353). Only streamed_working_set_bytes is
  unreferenced — see the [3.3] finding above.
- 3.4 (computed but unread): no local that is assigned and never read; every local in
  select_output_tier (result_bytes, resident_need, vram_budget, host_budget) feeds a
  return/compare. The intentionally-unused `M` param of streamed_working_set_bytes is
  marked /*M*/ (line 83) with a doc note (line 78) — deliberate call-site symmetry, not a
  finding.
-->

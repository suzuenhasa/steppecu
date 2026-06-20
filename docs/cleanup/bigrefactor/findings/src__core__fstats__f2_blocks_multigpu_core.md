# Review findings — src__core__fstats__f2_blocks_multigpu_core

Files: src/core/fstats/f2_blocks_multigpu_core.cpp, src/core/fstats/f2_blocks_multigpu_core.hpp

## Group 4 — Type & numeric

- [4.7][LOW] src/core/fstats/f2_blocks_multigpu_core.cpp:246 — compute_multigpu_partials_into takes the result buffers as raw `double* dst_f2, double* dst_vpair, int* block_sizes_dst` with no host-vs-device space typing; they are genuinely HOST (pinned) D2H destinations here (comment :261 "into one host buffer"), so there is no wrong-space hazard in this TU, but the bare pointers can't statically prevent a device pointer being passed by a future caller. Suggested: optionally a thin host-pointer wrapper at the seam; not load-bearing, file is correct as-is.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.


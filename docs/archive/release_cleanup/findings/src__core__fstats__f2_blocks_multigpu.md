# src__core__fstats__f2_blocks_multigpu
Files: /home/suzunik/steppe/src/core/fstats/f2_blocks_multigpu.cpp, /home/suzunik/steppe/src/core/fstats/f2_blocks_multigpu.hpp
Subsystem: core-stats

## Findings

### G3
- [G3.src__core__fstats__f2_blocks_multigpu][LOW] f2_blocks_multigpu.cpp:194-206 — `finish_streamed_tier` declares template parameter `TierHandle handle` but only reads `handle.P` and `handle.block_sizes`; `handle.n_block` is deliberately NOT mirrored here (each switch arm applies its own n_block rule afterward at lines 518/541). This is correct-by-design but means the helper's name ("finisher") over-promises — a reader may expect it to also set `out.n_block`. Not a bug; flagging only as a mild readability trap. Suggested: a one-line comment at the helper noting n_block is intentionally left to the caller arm.

### G6
- [G6.src__core__fstats__f2_blocks_multigpu][LOW] f2_blocks_multigpu.cpp:202-206 — in `finish_streamed_tier` the parameter `out` (the `F2BlocksOut&` written into) and the parameter `handle` (the tier-specific POD read from) are both passed, but `handle` is in practice always `out.host` or `out.disk` (a sub-object of `out`) — see call sites lines 514-515 and 538-539. Passing both an aggregate and its own member as separate references is correct here (no aliasing hazard since the writes target distinct members) but is easy to misread as two independent objects. Suggested: leave as-is or rename `handle` to `tier_handle` to signal it is a view into `out`.

### G8
- [G8.src__core__fstats__f2_blocks_multigpu][LOW] f2_blocks_multigpu.hpp:32-47 and .cpp:427-439 — the doc-comment block for `compute_f2_blocks_multigpu_tiered` is extensive and partly duplicated between the header declaration and the .cpp definition (tier descriptions, "3.9x win", "single-GPU first / drives gpus[0] regardless of G"). Two near-identical prose copies risk drift on edit. Not a code defect. Suggested: keep the authoritative tier description in one location (header) and have the .cpp comment cross-reference it rather than restate the tiers.

No type/numeric, launch, or memory bugs found. The scale-sensitive index arithmetic is correctly widened: `slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P)` (line 337) casts the `int` P operands to `std::size_t` BEFORE the multiply, and `total = slab * nonneg_count(n_block)` (line 338) is a `std::size_t * std::size_t` product, so the P~2500 / n_block~757 product (≈4.7e9 elements) cannot overflow a 32-bit index. The `nonneg_count` overloads correctly route `M` (long, line 166/485) and `n_block` (int) to their matching defensive-clamp widener. The block loop `for (int b = 0; b < n_block; ++b)` (line 489) iterates only over block count (~757), well within `int`. Groups checked: G2-G10 (G11-G22 N/A — unit is host-pure CUDA-free, is_cuda=false).

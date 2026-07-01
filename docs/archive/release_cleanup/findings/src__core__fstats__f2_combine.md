# src__core__fstats__f2_combine
Files: /home/suzunik/steppe/src/core/fstats/f2_combine.cpp, /home/suzunik/steppe/src/core/fstats/f2_combine.hpp
Subsystem: core-stats

## Findings
No issues found (groups checked: G2-G10).

Notes (NOT defects — confirmed clean during review):
- G4 (int-index overflow at scale): the placement math widens every operand to
  `std::size_t` BEFORE multiplying — `slab = (size_t)P * (size_t)P` (.cpp:59-60),
  `total = slab * (size_t)n_block_full` (.cpp:63), `part_elems = slab * (size_t)part.n_block`
  (.cpp:100-101), `out_base = slab * b0` (.cpp:105) with `b0` already a `size_t` (.cpp:99).
  At P~2500 / n_block~757 the full tensor exceeds 2^31 elements, but no intermediate is a
  32-bit `int`, so there is no overflow-before-widening. Correct.
- G3 (dead/computed-but-unread): the prior `< 0 ? 0 :` clamp and the recomputed `slab*b0`
  were already removed (.cpp:61-62, :102-105 comments document the prior cleanups B5/C2 and
  [7.2]); no current dead code, unused includes, or unread values. `<cstddef>`/`<algorithm>`/
  `<span>` are all used.
- G7 (duplication): the f2/vpair copies were already folded into the single `place` lambda
  invoked per member (.cpp:111-115); block_sizes legitimately differs in count/offset and
  is correctly left as its own copy_n (.cpp:119-121).
- G8/G10: comments are accurate (the −0.0/+0.0 IEEE rationale at .cpp:47-55 is load-bearing
  and correct), `out` is value-init then immediately populated, no stale/orphan TODO.

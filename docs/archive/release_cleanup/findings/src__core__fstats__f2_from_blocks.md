# src__core__fstats__f2_from_blocks
Files: /home/suzunik/steppe/src/core/fstats/f2_from_blocks.cpp, /home/suzunik/steppe/src/core/fstats/f2_from_blocks.hpp
Subsystem: core-stats

## Findings
No issues found (groups checked: G2-G10).

<!--
Reviewed in full. Notes on why each group is clean (not findings, just rationale):
- G3: every include is used (cstddef -> std::size_t at .cpp:83,108; all headers IWYU-justified). No dead/commented-out code; [[maybe_unused]] params are intentional (debug-only STEPPE_ASSERT bodies compile out under NDEBUG). block_ids_dense_nondecreasing is #ifndef NDEBUG-guarded and referenced only from the debug assert at .cpp:118 — not dead.
- G4 (scale/sign, the load-bearing group at P~2500/M~584k): types are consistent across the int/long seam. MatView::P is int, MatView::M is long (views.hpp:57,60); BlockPartition::block_id is vector<int>, n_block is int (block_partition_rule.hpp:109,112). validate_partition widens n_block to long before comparing to M (.cpp:116: static_cast<long>(partition.n_block) <= M). The size compare at .cpp:108 guards negative M (M<0?0:M) before the size_t cast. The O(M) scan (.cpp:80-88) uses long s/long M, int id vs int n_block — no truncation, no unsigned countdown, no index-width overflow. P*P*n_block resident-tensor overflow is the backend's concern, not this host orchestration (it only forwards a pointer + count).
- G5: no magic numbers; only -1 sentinel (.cpp:81) and 0 bounds, all self-documenting.
- G8: comments are rationale-heavy and accurate to the current code (block_id.data()/n_block seam, the size()-erasure argument, the NDEBUG no-op behavior). No stale/orphan TODO/FIXME.
- G9/G10: backend taken by ref, views/partition/precision by const ref; locals (prev, id) declared at first use and initialized.
-->

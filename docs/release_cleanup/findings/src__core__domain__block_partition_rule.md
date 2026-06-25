# src__core__domain__block_partition_rule
Files: /home/suzunik/steppe/src/core/domain/block_partition_rule.cpp, /home/suzunik/steppe/src/core/domain/block_partition_rule.hpp
Subsystem: core-stats

## Findings
No issues found (groups checked: G2-G10).

### Review notes (not findings)
- G3 `block_of` (hpp:80) is uncalled by `assign_blocks` but is INTENTIONALLY retained: it is a documented public `__host__`-shareable per-SNP primitive (hpp:20-23 explicitly records this status), not accidental dead code. Not flagged.
- G4 numeric/width at scale is sound: `block_id` stays `std::vector<int>` and `n_block`/`global` cast `long -> int` (cpp:91,94) only carry block COUNTS, which are O(1e3) even at M~584k worst case (every SNP its own block ~= 584k, still well within int); per-SNP loop index and span extents are `long` (cpp:42,69), and the widening to `std::size_t` is single-homed through `idx()` (hpp:49) with documented non-negativity. No int-index-overflow exposure.
- G4 the `!(block_size_morgans > 0.0)` guard (cpp:35) correctly rejects 0 / negative / NaN in one test; the float->int UB path through `block_of` is closed before any such call. Length-mismatch is clamped to the shorter span (cpp:42-43) rather than read OOB.
- G5/G8 magic constants are named or rationale-documented: the `-1.0e20` / `-1` sentinels (cpp:63-67) carry inline rationale; the 100.0 / 5.0 factors live single-homed in config.hpp (`kCentimorgansPerMorgan`, `kDefaultBlockSizeCm`) and are applied at exactly one site (hpp:95-97).
- G6 naming is consistent (`fpos`, `pos`, `prev_chrom`, `global`, `s`/`e`/`b`); the loop counters are tight-scope. `BlockRange::size()`/`begin`/`end` are unambiguous.
- G7 duplication is actively removed, not present: `idx()` and the `fail` prefix lambda (hpp:237-239) single-home the repeated cast and message prefix; the per-block scan that was hand-duplicated across backends is centralized in `block_ranges`.

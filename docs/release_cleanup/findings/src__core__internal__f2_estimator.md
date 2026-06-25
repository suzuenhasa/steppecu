# src__core__internal__f2_estimator
Files: /home/suzunik/steppe/src/core/internal/f2_estimator.hpp
Subsystem: core-stats

## Findings
No issues found (groups checked: G2-G10).

Notes (no defect, recorded for the reviewer):
- The header is pure `STEPPE_HD` scalar numerics (`het_correction`, `f2_term`,
  `assemble_f2_numerator`, `finalize_f2`, `pair_block_is_missing`) plus one
  structural constant (`kF2StackedBlocks`). No state, no allocation, no launches.
- G4 (numeric/scale): all arithmetic is `double`; counts (`n`, `vpair`) are carried
  as `double` by intentional FP64 design — no int-index-overflow surface here. The
  `(vpair > 0.0)` / `!(vpair > 0.0)` guards in `finalize_f2`:131 and
  `pair_block_is_missing`:145 correctly diverge on NaN (the NB at :126-129 and the
  single-source rationale at :134-143 document this on purpose).
- G5 (magic numbers): the `2.0` in `assemble_f2_numerator`:115 is the algebraic
  `a²−2ab+b²` constant (documented :111); `kF2StackedBlocks = 2` (:53) and the named
  `kHetCorrDenomFloor` (config.hpp:89 == 1.0) single-home the only other literals.
  No duplicated/drift-prone constants.
- G3/G8: includes are all used (`STEPPE_HD`, `cdiv`/`grid_for` re-export verified,
  `kHetCorrDenomFloor`); comments are accurate to current behavior, no stale text,
  no orphan TODO/FIXME/HACK.

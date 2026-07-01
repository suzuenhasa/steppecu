# include__steppe__qpadm
Files: /home/suzunik/steppe/include/steppe/qpadm.hpp
Subsystem: public-api

## Findings

### G7
- [G7.duplication][LOW] qpadm.hpp:145-147,212-217 — The nested rankdrop field cluster (`rankdrop_f4rank/dof/dofdiff` + `rankdrop_chisq/p/chisqdiff/p_nested`, plus `rank_chisq`/`rank_dof`) is repeated verbatim across `QpAdmResult` (145-147) and `QpWaveResult` (212-217). A shared `struct RankSweepTable` (the comment at :145 already names "RankSweep.rd_*") embedded in both would prevent the two copies drifting if a column is added later. Cosmetic only — both are frozen value shapes. Suggested: factor the shared rankdrop columns into one nested struct reused by both result types.

(Groups checked G2-G10: G2/G11-G22 N/A — CUDA-free header, no device code; G3 all includes/symbols used; G4 indices are `int`, no arithmetic/index-width surface at P~2500; G5 all literals are named members with AT2-parity rationale, no bare/duplicated constants; G6 naming clear and consistent; G8 comments are accurate rationale, no stale/orphan TODO; G9 enum mapping frozen, defaults surfaced; G10 every member has a default-member-initializer or is a default-constructed container.)

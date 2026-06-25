# src__io__ind_reader
Files: /home/suzunik/steppe/src/io/ind_reader.cpp, /home/suzunik/steppe/src/io/ind_reader.hpp
Subsystem: io

## Findings

### G4
- [G4.ind_reader][LOW] ind_reader.cpp:64-67,79 — when `row >= n_records_present` the loop still does `++row` and `continue`s, so `row` (and therefore `part.n_individuals_total = row`) keeps counting rows BEYOND the present-genotype cap. The hpp contract (hpp:71,80-81) defines `n_individuals_total` as "total .ind rows == TGENO records" and says rows past the cap are "ignored"; counting capped-out rows means `n_individuals_total` can exceed `n_records_present`, which contradicts the "== TGENO records" promise when the .ind is longer than the genotype file. Not a crash and the row INDICES used for grouping are correctly bounded; purely a counter-semantics ambiguity. Suggested: either stop incrementing/counting past the cap, or clarify the doc to say it is the raw .ind row count (un-capped).

### G8
- [G8.ind_reader][LOW] ind_reader.cpp:79 — comment "total .ind rows seen (the individual axis)" is slightly stale w.r.t. the cap behavior at lines 64-67: under a partial-file cap this value also includes rows past `n_records_present`, so it is not strictly "the individual axis" of the decoded matrix. Suggested: tighten the comment to match whichever semantics are chosen for the G4 item above.

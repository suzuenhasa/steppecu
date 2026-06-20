# Review findings — src__io__filter__mind_prepass

Files: /home/suzunik/steppe/src/io/filter/mind_prepass.cpp, /home/suzunik/steppe/src/io/filter/mind_prepass.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.4][LOW] src/io/filter/mind_prepass.cpp:64 — redundant write loop: `out.missing_frac` was already set to 0.0 for all n_ind elements via `assign(n_ind, 0.0)` at line 23, so this loop re-writes the same value and is a pure no-op. The code's own comment (lines 61-63) explicitly states it is "a no-op kept only as an explicit statement of the convention" — i.e. intentional self-documentation, not an accidental dead store. Suggested: optional — drop the loop and let the lead comment carry the convention, or keep as documented intent (lowest priority).

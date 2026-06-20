# Review findings — src__core__qpadm__nested_models

Files: /home/suzunik/steppe/src/core/qpadm/nested_models.cpp, /home/suzunik/steppe/src/core/qpadm/nested_models.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.1][LOW] nested_models.hpp:10-12 — Stale doc comment: header still describes the impl as "an n_block host loop over gls_weights (correctness first) ... the batched device S7 is M(fit-3)", but se_from_loo now calls the batched seam gls_weights_loo_batched (nested_models.cpp:57, documented at .cpp:52-56). The described host-loop code no longer exists. Suggested: update the header comment to reflect the batched-seam implementation (not dead code, stale narrative).


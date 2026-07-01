# Review findings — src__core__qpadm__f4_matrix

Files: /home/suzunik/steppe/src/core/qpadm/f4_matrix.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

## Group 6 — Naming

No Group 6 issues found.

## Group 7 — Duplication

- [7.1][LOW] src/core/qpadm/f4_matrix.hpp:23-37 — The two `assemble_f4` overloads have textually identical bodies (`return be.assemble_f4(f2, left_idx, right_idx, precision);`), differing only in the `f2` parameter type (`DeviceF2Blocks` vs `F2BlockTensor`). Both forward unchanged to a backend entry point. Suggested: a single `template <class F2T>` forwarding function would fold both — but only if the intent is to keep one external dispatch surface; the explicit device vs host split is a documented design seam, so keeping two overloads is defensible. Low priority.

## Group 8 — Comments

No Group 8 issues found.

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.


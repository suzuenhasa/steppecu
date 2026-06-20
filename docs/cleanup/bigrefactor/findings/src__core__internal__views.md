# Review findings — src__core__internal__views

Files: /home/suzunik/steppe/src/core/internal/views.hpp

## Group 4 — Type & numeric

- [4.7][LOW] views.hpp:54 — `MatView::data` is a raw `const double*` with no host-vs-device space distinction; this same struct is documented as the seam consumed by both the CPU reference and the GPU feeder (lines 63-64), so nothing prevents a device pointer being stored where host is expected (or vice versa). Suggested: acceptable for this minimal host-side anchor, but if it is ever bound to device storage, route through the device layer's `span_view.hpp` typed-space wrapper rather than the raw `const double*`.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

## Group 6 — Naming

No Group 6 issues found.

## Group 7 — Duplication

- [7.3][LOW] views.hpp:67 — the flat-index expression casts both operands: `static_cast<long>(i) + static_cast<long>(P) * s`. Since `s` is already `long`, the `static_cast<long>(P)` is redundant for overflow safety (`P * s` already promotes to `long`); the doubled cast is mild duplication of the widening pattern. Suggested: keep one explicit cast for clarity (`static_cast<long>(i) + static_cast<long>(P) * s` is fine as documentation of intent), or drop the `P` cast as `static_cast<long>(i) + P * s`; not load-bearing either way.

## Group 8 — Comments

No Group 8 issues found.

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.


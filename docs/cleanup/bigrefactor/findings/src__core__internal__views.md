# Review findings — src__core__internal__views

Files: /home/suzunik/steppe/src/core/internal/views.hpp

## Group 4 — Type & numeric

- [4.7][LOW] views.hpp:54 — `MatView::data` is a raw `const double*` with no host-vs-device space distinction; this same struct is documented as the seam consumed by both the CPU reference and the GPU feeder (lines 63-64), so nothing prevents a device pointer being stored where host is expected (or vice versa). Suggested: acceptable for this minimal host-side anchor, but if it is ever bound to device storage, route through the device layer's `span_view.hpp` typed-space wrapper rather than the raw `const double*`.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.


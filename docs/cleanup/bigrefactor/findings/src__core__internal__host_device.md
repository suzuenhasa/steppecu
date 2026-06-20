# Review findings — src__core__internal__host_device

Files: src/core/internal/host_device.hpp

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

No Group 7 issues found.

## Group 8 — Comments

- [8.2][LOW] src/core/internal/host_device.hpp:61 — Comment says "The trailing `(void)0` keeps the macro usable as a statement", but `(void)0` appears only in the NDEBUG branch (line 64 `((void)0)`); the debug branch (line 66) expands to bare `__VA_ARGS__` with no trailing `(void)0`. The phrasing implies the `(void)0` is always present. Suggested: reword to clarify the `(void)0` is the release-build expansion (the debug expansion is the argument itself), both terminated by the call-site `;`.

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.


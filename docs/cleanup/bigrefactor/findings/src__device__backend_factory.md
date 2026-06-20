# Review findings — src__device__backend_factory

Files: /home/suzunik/steppe/src/device/backend_factory.hpp

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

- [8.2][MED] /home/suzunik/steppe/src/device/backend_factory.hpp:50-51 — Stale comment. The doc-comment on `make_cuda_backend` states the SNP-sharding + host-side fixed-order combine is "NOT here — that combine algorithm is the next workflow", but per docs/ROADMAP.md §72 that combine (shard + fixed-order device-order sum, plus the device-resident P2P combine) is already DONE (`867a4bf`, bit-identical). The "next workflow" framing no longer describes the current state. Suggested: drop "the next workflow" and point to the implemented `Resources` combine (M4.5 DONE) so the rationale tracks reality.

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.

## Group 13 — Error handling

No Group 13 issues found.

## Group 14 — Memory: allocation & lifetime

No Group 14 issues found.

## Group 15 — Memory: transfers

No Group 15 issues found.

## Group 16 — RAII: ownership & wrapper hygiene

No Group 16 issues found.

## Group 17 — RAII: lifetime & deleter pitfalls (CUDA-specific)

No Group 17 issues found.


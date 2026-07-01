# Review findings — src__device__resources

Files: /home/suzunik/steppe/src/device/resources.cpp, /home/suzunik/steppe/src/device/resources.hpp

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

- [7.2][LOW] resources.cpp:54,112 — `visible_device_count()` is queried twice per `build_resources` on the auto-enumerate path (once inside `resolve_device_order` at line 54 to size the order, then again at line 112 to validate it), yet the comment at lines 107-109 asserts "One CUDA-free count query (visible_device_count()) serves both the auto-enumerate sizing above and this validation". The code does not match the comment: the count is read twice (and on the explicit-list path the resolve call is skipped, so the count is read once). Low cost (host-only `cudaGetDeviceCount`) but a repeated loop-invariant query whose own comment claims it is computed once. Suggested: compute `const int visible = visible_device_count();` once in `build_resources`, pass it into both `resolve_device_order` and `validate_device_order`, so the "one count query" comment becomes true.

## Group 8 — Comments

- [8.2][MED] resources.hpp:14-15 — stale cross-file line-number citation: the comment cites the per-device-instance contract as "backend.hpp:193-202", but backend.hpp lines 193-202 now describe `DecodeTileView` (a genotype tile view). The actual "PER-DEVICE-INSTANCE CONTRACT" block is at backend.hpp:341. The cited line range has drifted and now points at unrelated code, mis-directing any reader who follows it. Suggested: drop the `:193-202` line numbers (the bare "per-device-instance contract, backend.hpp" form used at resources.hpp:137 and resources.cpp:13 does not rot), or re-anchor to the §9/§11.4 spec section instead of a line range.
- [8.2][LOW] resources.cpp:107-109 — stale/inaccurate comment: "One CUDA-free count query (visible_device_count()) serves both the auto-enumerate sizing above and this validation" describes a single shared query, but the code reads the count twice (line 54 inside `resolve_device_order`, line 112 here) on the auto path. The comment asserts behavior the code does not have. (Same root cause as the 7.2 duplication finding above; flagged here as a comment-accuracy defect.) Suggested: either make the code match (hoist one `visible` and pass it in) or rewrite the comment to state the count is read per-call.

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

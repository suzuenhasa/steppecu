# Review findings — src__core__domain__block_partition_rule

Files: /home/suzunik/steppe/src/core/domain/block_partition_rule.cpp, /home/suzunik/steppe/src/core/domain/block_partition_rule.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/core/domain/block_partition_rule.hpp:31 — `#include <cstdint>` appears unused: no fixed-width integer type (`int8_t`/`uint64_t`/`intptr_t`/etc.) is named anywhere in either file (`block_id` is `std::vector<int>`, indices are `long`/`std::size_t`, positions are `double`). The conversion constant comes from `steppe/config.hpp` (line 37), `std::size_t` from `<cstddef>`. Suggested: drop `<cstdint>` (or confirm it is a deliberate transitive-include guarantee for downstream consumers before removing).

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

## Group 6 — Naming

- [6.3][LOW] src/core/domain/block_partition_rule.cpp:42 — `assign_blocks` names the SNP count lowercase `m`, but the sibling `block_ranges` in the same unit's header (block_partition_rule.hpp:205) names the identical quantity uppercase `M` (and its doc consistently calls it "M"/`MatView::M`). Same concept, two conventions across the two functions of one unit. Suggested: rename the local `m` in `assign_blocks` to `M` (or `num_snps`) to match `block_ranges`/`MatView::M`.

## Group 7 — Duplication

- [7.3][LOW] src/core/domain/block_partition_rule.cpp:57,58,70 — `static_cast<std::size_t>(s)` is recomputed three times inside the per-SNP loop for the same loop index `s`; the same `block_id[static_cast<std::size_t>(...)]` index-cast idiom also repeats across `block_ranges` (block_partition_rule.hpp:236,250,251). Repeated `long → size_t` cast boilerplate. Suggested: hoist a single `const auto us = static_cast<std::size_t>(s);` per iteration (or a small `inline std::size_t idx(long)` helper used by both functions) to compute the cast once.
- [7.4][LOW] src/core/domain/block_partition_rule.hpp:222-225,238-240,243-246 — the three fail-fast paths in `block_ranges` repeat the same construction shape: `"core::block_ranges: " + ... + std::to_string(x) + ...` string-concatenation boilerplate, each differing only by message text and the interpolated value. Suggested: optional — a tiny local lambda/helper that prefixes `"core::block_ranges: "` and throws would fold the three near-identical `throw std::runtime_error(...)` sites; low value given they are genuinely distinct messages.

## Group 8 — Comments

No Group 8 issues found.

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.


# Review findings — src__core__domain__block_partition_rule

Files: /home/suzunik/steppe/src/core/domain/block_partition_rule.cpp, /home/suzunik/steppe/src/core/domain/block_partition_rule.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/core/domain/block_partition_rule.hpp:31 — `#include <cstdint>` appears unused: no fixed-width integer type (`int8_t`/`uint64_t`/`intptr_t`/etc.) is named anywhere in either file (`block_id` is `std::vector<int>`, indices are `long`/`std::size_t`, positions are `double`). The conversion constant comes from `steppe/config.hpp` (line 37), `std::size_t` from `<cstddef>`. Suggested: drop `<cstdint>` (or confirm it is a deliberate transitive-include guarantee for downstream consumers before removing).


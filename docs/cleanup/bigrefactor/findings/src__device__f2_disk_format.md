# Review findings — src__device__f2_disk_format

Files: /home/suzunik/steppe/src/device/f2_disk_format.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/device/f2_disk_format.hpp:16 — `#include <cstddef>` is unused: no `std::size_t`/`std::ptrdiff_t`/`std::byte`/`offsetof` symbol appears in the file (verified by grep), and `sizeof(double)` on lines 42/46 is a built-in operator needing no include. `<cstdint>` (line 17) IS required (uint32_t/int32_t/uint64_t/uint8_t). Suggested: drop the `<cstddef>` include.

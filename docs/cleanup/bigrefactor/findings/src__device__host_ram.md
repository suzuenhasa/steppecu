# Review findings — src__device__host_ram

Files: src/device/host_ram.cpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/device/host_ram.cpp:12 — `#include <cstring>` is unused: no `str*`/`mem*` function from `<cstring>` is called anywhere in the file (the only `<cstring>`-adjacent symbol, `std::size_t`, comes from `<cstddef>` already pulled in via `<cctype>`/`<sys/sysinfo.h>`/`tier_select.hpp`); `iequals` uses a hand-rolled loop rather than `strcmp`/`strncasecmp`. Suggested: drop the `<cstring>` include.

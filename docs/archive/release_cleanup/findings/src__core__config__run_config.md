# src__core__config__run_config
Files: /home/suzunik/steppe/src/core/config/run_config.hpp
Subsystem: core-stats

## Findings

### G3
- [G3.src__core__config__run_config][LOW] run_config.hpp:27,68 — `std::vector` is used directly (e.g. accessors at lines 68-85, fields at 149-154) but only `<string>` is `#include`d; `<vector>` is pulled in transitively via steppe/config.hpp, steppe/qpadm.hpp, io/ind_reader.hpp, and core/config/cli_args.hpp. IWYU hygiene: a direct dependency on `std::vector` should `#include <vector>` directly rather than rely on transitive inclusion (brittle if any of those headers later drops it). Suggested: add `#include <vector>` next to `#include <string>`.

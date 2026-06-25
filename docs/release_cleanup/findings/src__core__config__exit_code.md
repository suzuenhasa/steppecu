# src__core__config__exit_code
Files: /home/suzunik/steppe/src/core/config/exit_code.hpp
Subsystem: core-stats

## Findings
No issues found (groups checked: G2-G10).

Notes (verified, not defects):
- The switch in `exit_code_for` (lines 55-67) covers all 6 members of `enum class Status` (Ok, DeviceOom, RankDeficient, NonSpdCovariance, ChisqUndefined, InvalidConfig per include/steppe/error.hpp); no `default:` is intentional so `-Wswitch` flags a future status, and the trailing `return kExitRuntimeError` (line 70) is the documented defensive fallthrough.
- `kExitIoError` (4) and `kExitRuntimeError` (5) are never produced by `exit_code_for` (no matching Status), but they are NOT dead: app commands return them directly from I/O/catch paths (e.g. src/app/cmd_qpadm.cpp:84, src/app/main.cpp:21). Matches the documented design.
- Includes are both used: `<cstdlib>` (EXIT_SUCCESS, line 32) and `steppe/error.hpp` (Status). `constexpr`/`noexcept`/`[[nodiscard]]` and `enum : int` are all correct; literals 2-5 are named, documented enum constants (not magic numbers).

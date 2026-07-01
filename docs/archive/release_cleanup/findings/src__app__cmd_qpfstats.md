# src__app__cmd_qpfstats
Files: /home/suzunik/steppe/src/app/cmd_qpfstats.cpp, /home/suzunik/steppe/src/app/cmd_qpfstats.hpp
Subsystem: app-cli

## Findings

### G7
- [G7.src__app__cmd_qpfstats][LOW] cmd_qpfstats.cpp:30-37 — `precision_label(const Precision&)` is copy-pasted verbatim from cmd_extract_f2.cpp:62-69 (identical switch + fallthrough `return "fp64"`); the comment at line 29 even says "mirrors cmd_extract_f2". Two copies of the emu/tf32/fp64 vocabulary can drift when a `Precision::Kind` is added/renamed. Suggested: hoist one shared `precision_label` into a common app/precision header (or core/config) and have both commands call it.

### G8
- [G8.src__app__cmd_qpfstats][LOW] cmd_qpfstats.hpp:31 — doc comment says the return is `steppe::config::CliExitCode`, but the function is declared `int` and the .cpp returns `cfg::kExit*` constants from core/config/exit_code.hpp; "CliExitCode" appears to be a stale type name (no such type is referenced). Suggested: update the comment to name the actual exit-code source (exit_code.hpp `kExit*`) or the real type.

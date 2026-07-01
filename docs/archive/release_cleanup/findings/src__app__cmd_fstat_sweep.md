# src__app__cmd_fstat_sweep
Files: /home/suzunik/steppe/src/app/cmd_fstat_sweep.cpp, /home/suzunik/steppe/src/app/cmd_fstat_sweep.hpp
Subsystem: app-cli

## Findings

### G6
- [G6.naming][LOW] cmd_fstat_sweep.hpp:25-26, cmd_fstat_sweep.cpp:78 — the third parameter of `run_fstat_sweep` is named `prog` in the header declaration but `cmd` in the .cpp definition (and the header doc-comment at lines 23-24 also talks about `prog`). Inconsistent name for the same parameter across declaration/definition; a reader cross-referencing the two sees two different identifiers. Suggested: pick one name (the impl consistently uses `cmd` in all the fprintf calls, so rename the header parameter + doc to `cmd`).

### G8
- [G8.comment][LOW] cmd_fstat_sweep.hpp:24 — doc-comment says `prog` "is the program-name string ... ('f4' / 'f3' / 'qpdstat')", but the two public wrappers actually pass "f4-sweep"/"f3-sweep" (cmd_fstat_sweep.cpp:214,217); the listed values describe only the standalone-command routing case, not the dedicated subcommands. Mildly misleading about what strings reach the parameter. Suggested: note both call sites ("f4-sweep"/"f3-sweep" from the dedicated subcommands, "f4"/"f3"/"qpdstat" from the standalone routing).

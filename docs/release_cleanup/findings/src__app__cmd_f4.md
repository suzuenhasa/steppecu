# src__app__cmd_f4
Files: /home/suzunik/steppe/src/app/cmd_f4.cpp, /home/suzunik/steppe/src/app/cmd_f4.hpp
Subsystem: app-cli

## Findings

### G8
- [G8.task][LOW] cmd_f4.cpp:160 — Comment claims "opts here is the struct default", but line 161 binds `opts` to `config.qpadm_options()` (the parsed/frozen config struct, not a default-constructed one). The fudge value is whatever the config carries, not guaranteed the struct default. The comment is misleading/stale relative to the code. Suggested: reword to "opts comes from the frozen config; fudge defaults to 0 for a bare f4 SE unless overridden".

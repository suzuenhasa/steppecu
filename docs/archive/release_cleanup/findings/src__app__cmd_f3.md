# src__app__cmd_f3
Files: /home/suzunik/steppe/src/app/cmd_f3.cpp, /home/suzunik/steppe/src/app/cmd_f3.hpp
Subsystem: app-cli

## Findings

### G3
- [G3.src__app__cmd_f3][LOW] cmd_f3.cpp:26-27 — both `<iostream>` and `<ostream>` are included; `std::cout` (used at line 190) comes from `<iostream>`, and `<ostream>` adds nothing beyond what `<iostream>`/`<fstream>` already pull in for the `std::ostream&` parameter of `emit_f3_result`. Minor redundant include. Suggested: drop `<ostream>`.

### G8
- [G8.src__app__cmd_f3][LOW] cmd_f3.cpp:99-101 — the sweep-mode comment describes on-device behavior ("on-device unrank+compute+|z|filter+CUB-compact, survivors only") that lives entirely in `cmd_fstat_sweep.cpp`; it restates the callee's internals rather than this site's contract (route to the sweep). Not stale, but risks drift if the sweep implementation changes. Suggested: trim to the dispatch rationale, leave the algorithm description with the sweep.

(No issues in groups G2, G4, G5, G6, G7, G9, G10. The file is a thin CLI orchestrator: literals are named exit codes (`cfg::kExit*`) and small loop bounds (3 = triple width, self-documented by the C,A,B comments); index widths use `std::size_t`/`int` consistently with the resolver/`std::array<int,3>` API and never index the P*P*n_block resident f2 (no overflow surface here); `device_id = resources.gpus.front().device_id` is the intentional single-GPU selection, not a bug; all resources (`Resources`, `DeviceF2Blocks`, `std::ofstream`) are RAII-owned and freed on the throwing/error paths.)

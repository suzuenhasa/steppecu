# src__app__cmd_rotate
Files: /home/suzunik/steppe/src/app/cmd_rotate.cpp, /home/suzunik/steppe/src/app/cmd_rotate.hpp
Subsystem: app-cli

## Findings

### G8
- [G8.task][LOW] cmd_rotate.cpp:165-166 — The literal `TODO(multigpu-host-bounce):` token is embedded inside the user-facing stderr warning string (the `fprintf` format), so end users see a `TODO(...)` marker in shipped CLI output. The deferral itself is legitimate (multi-GPU is parked), but a TODO token leaking into user output is poor release polish. Suggested: keep the TODO as a `//` source comment and phrase the user-facing warning without the `TODO(...)` token.
- [G8.task][LOW] cmd_rotate.cpp:159 — Comment cites a doc location by raw line span (`§379-382`) alongside the section ref; raw line numbers go stale on any doc edit. Suggested: cite the stable section anchor only (`cli-bindings.md §4.5`), drop the line span.

(No issues found for G2, G3, G4, G5, G6, G7, G9, G10. Includes are all used; every function parameter is read; the `-1` max-sources sentinel and `right_n = right.size()-1` convention are documented and consistent with the config layer; `int`/`std::size_t` index casts are explicit and correct at pool scale; the `hi < lo` and empty-enumeration edges are both guarded; `model_index`/`counter` being `int` matches the public `QpAdmModel::model_index` field type.)

# src__app__cmd_qpdstat
Files: /home/suzunik/steppe/src/app/cmd_qpdstat.cpp, /home/suzunik/steppe/src/app/cmd_qpdstat.hpp
Subsystem: app-cli

## Findings

### G6
- [G6.src__app__cmd_qpdstat][LOW] cmd_qpdstat.cpp:150,212 — In `run_qpdstat_prefix` the name `fmt` is bound twice to two unrelated types in one function: `const io::GenoFormat fmt` (line 150, scoped inside the try-block) and `OutputFormat fmt` (line 212, function scope). No actual shadow conflict (disjoint scopes), but reusing one cryptic name for two distinct concepts in a single function hurts readability. Suggested: rename the geno-format local to `geno_fmt`.

### G7
- [G7.src__app__cmd_qpdstat][LOW] cmd_qpdstat.cpp:169-187,282-302 — The "resolve each name quad to an index quad and carry resolved labels into l1..l4" block is copy-pasted between `run_qpdstat_prefix` (169-187) and `run_qpdstat_command` (282-302); the two copies differ only in that the f2-dir copy pre-reserves l1..l4 (285-286) and the prefix copy does not (l1..l4 declared bare at 171). Suggested: extract a small helper `resolve_quartets(resolver, quartet_names, quadruples, l1..l4, err)` shared by both paths.

### G10
- [G10.src__app__cmd_qpdstat][LOW] cmd_qpdstat.cpp:171 — In the prefix path `l1,l2,l3,l4` are declared without `reserve`, whereas the structurally identical f2-dir path reserves them (285-286). Functional parity is fine; only a missing-reserve inconsistency between the two sibling loops. Suggested: mirror the `reserve(quartet_names.size())` calls (folds away if the G7 helper is extracted).

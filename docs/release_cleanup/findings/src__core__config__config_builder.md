# src__core__config__config_builder
Files: /home/suzunik/steppe/src/core/config/config_builder.cpp, /home/suzunik/steppe/src/core/config/config_builder.hpp
Subsystem: core-stats

## Findings

### G3 dead/computed-but-unread
- [G3.dead][LOW] config_builder.cpp:136 — `take(merged_.config_path, args.config_path)` writes `merged_.config_path`, but `build()` never reads it: the TOML path is tracked exclusively via `toml_path_`/`toml_requested_`, and the CLI config request is consumed directly from `args.config_path` at lines 197-200, not from `merged_.config_path`. So this merge is computed-but-unread (the field is dead in the merged-state path). Suggested: drop the `take(merged_.config_path, ...)` line (the dedicated `args.config_path` handling at 197-200 already covers it), or document that `merged_.config_path` is intentionally unused.

### G4 type & numeric
- [G4.range][LOW] config_builder.cpp:512,515 — `qpgraph_numstart` and `qpgraph_max_nadmix` are carried through with NO range check, unlike every neighboring numeric knob (fudge >=0, als_iterations >=1, rank, jackknife 0..2, min_sources >=1, etc.). A negative or zero `--numstart` (multistart restart count, default 10) or a negative `--max-nadmix` would be accepted as-is and reach the qpgraph engine. Not a scale/overflow bug, but inconsistent with the file's documented fail-fast input-gate contract (hpp:84-95). Suggested: add `if (*merged_.numstart < 1) return fail("--numstart must be >= 1");` and a `>= 0` (or {0,1}-bounded, per run_config.hpp:177 "v1 {0,1}") check for `--max-nadmix`, matching the surrounding pattern.

### G8 comments
- [G8.stale][LOW] config_builder.cpp:31 — the header comment "it is unit-tested with NO GPU (the test_config pattern)" describes the test harness, not this file's behavior; harmless but is documentation-of-tests living in the implementation header. Acceptable rationale comment; flagging only as low-value/borderline. Suggested: none required (informational).

# src__app__cmd_dates
Files: /home/suzunik/steppe/src/app/cmd_dates.cpp, /home/suzunik/steppe/src/app/cmd_dates.hpp
Subsystem: app-cli

## Findings
No issues found (groups checked: G2-G10).

Notes (verified clean, not defects):
- G3: every include is used (`<cstdio>` snprintf/fprintf, `<cmath>` isnan, `<fstream>` ofstream, `<vector>`/`<string>` config refs); `DatesOptions opts` is forwarded to run_dates at cpp:117 (not dead).
- G7: emit_dates is called twice (cpp:132 stdout, cpp:140 ofstream) but is already a shared helper — no copy-paste duplication.
- G8: the "date-neutral" source-order comment (cpp:87-89) accurately matches behavior; src1/src2 are still threaded through for labeling.
- G9/G10: `result` (cpp:108) is default-constructed and only read after the successful assignment at cpp:117 (the throw path returns before any read); `fmt` defaults to Csv (cpp:124).
- The JSON NaN double-guard (cpp:61-63: ternary -> "null" while `d()` -> "NA") is intentional, not redundant: JSON requires literal `null`, the CSV/TSV path requires the "NA" token.
- The parse_output_format-after-run_dates ordering (cpp:117 then cpp:125) is the established sibling pattern in cmd_qpdstat.cpp (run_qpdstat_prefix: build_resources/run at :194, parse_output_format at :213) — consistent, not a wart.

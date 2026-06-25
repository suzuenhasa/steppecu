# src__io__filter__include_exclude
Files: /home/suzunik/steppe/src/io/filter/include_exclude.cpp, /home/suzunik/steppe/src/io/filter/include_exclude.hpp
Subsystem: io

## Findings
No issues found (groups checked: G2-G10).

Notes (sub-threshold, NOT defects):
- G7: The three insert loops in `SnpMembership::SnpMembership` (include_exclude.cpp:49-51, 55-57, 62-64) are structurally similar but each reads a different source container into a different member set (and the middle one move-inserts owned strings); extracting a helper would not meaningfully reduce code and would obscure the union-vs-drop intent. Left as-is intentionally.
- G3: `<string>` in include_exclude.cpp:10 is transitively available via the unit header, but the TU directly names `std::string`/`std::runtime_error` arguments, so the explicit include is defensible self-documentation, not dead.

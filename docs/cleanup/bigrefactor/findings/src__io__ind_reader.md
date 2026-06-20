# Review findings — src__io__ind_reader

Files: /home/suzunik/steppe/src/io/ind_reader.cpp, /home/suzunik/steppe/src/io/ind_reader.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

## Group 6 — Naming

- [6.4][LOW] src/io/ind_reader.cpp:54 — `std::istringstream ls(line)`: `ls` is a nonstandard/opaque abbreviation for the line stream (reads as "list" or "left-shift"). Scope is short but the name carries no meaning. Suggested: rename to `line_stream` (or `fields`).

## Group 7 — Duplication

- [7.1][LOW] src/io/ind_reader.cpp:84-88,107-111 — The Explicit and MinN selection cases are the same `for (const auto& g : groups) { if (<pred>) selected.push_back(&g); }` filter loop differing only by the predicate (`want.count(g.label)` vs `g.rows.size() >= sel.min_n`). Suggested: factor a small `filter_into(selected, groups, pred)` helper (or inline lambda) taking the predicate; low priority given the bodies are 1-line each.
- [7.4][LOW] src/io/ind_reader.cpp:42,75,116 — The three `throw std::runtime_error("io::read_ind: " + ... + path)` sites repeat the `"io::read_ind: "` prefix and `+ path` suffix boilerplate. Suggested: a local `[&](msg){ throw std::runtime_error("io::read_ind: " + msg + path); }` helper would fold the prefix/suffix; minor, the messages are otherwise distinct and clear.

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.

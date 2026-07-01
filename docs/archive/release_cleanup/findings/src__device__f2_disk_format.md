# src__device__f2_disk_format
Files: /home/suzunik/steppe/src/device/f2_disk_format.hpp
Subsystem: backend

## Findings

### G8
- [G8.comment][LOW] f2_disk_format.hpp:50 — comment cites internal cleanup-tracking tags ("cleanup group-7 7.1 dup + 7.3 repeated widening cast chain") that document the refactor process rather than the code's rationale; same pattern at line 22 ("group-5 5.3"). These are orphan process annotations that will read as noise/stale once the cleanup campaign closes. Suggested: drop the parenthetical group-N tags, keep the substantive rationale ("single home for the slab arithmetic so the two region accessors cannot drift").

### G9
- [G9.config][LOW] f2_disk_format.hpp:34-35 — the header's `vpair_offset`/`block_sizes_offset` semantics ("== 64 + P²·n_block·8", "== vpair_offset + P²·n_block·8") are documented only in trailing comments; the actual computation lives in the (not-in-this-unit) writer with no shared helper, so the writer and these field invariants can drift. The `f2_block_offset`/`vpair_block_offset` accessors here only consume `h.f2_offset`/`h.vpair_offset` and never recompute the region starts, so this is informational, but the region-start formula has no single home the way the per-slab stride does (`detail::slab_offset`). Suggested: surface a `vpair_offset(P, n_block)`/`block_sizes_offset(P, n_block)` constexpr helper in this header for the writer to use, mirroring the `slab_offset` single-home pattern.

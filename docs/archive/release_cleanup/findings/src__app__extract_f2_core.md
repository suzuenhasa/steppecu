# src__app__extract_f2_core
Files: /home/suzunik/steppe/src/app/extract_f2_core.cpp
Subsystem: app-cli

## Findings

### G6
- [G6.naming][LOW] extract_f2_core.cpp:196 — `const bool resident = (backend.capabilities().device_count > 0);` is named `resident` but it tests device PRESENCE (count > 0), not an output-residency tier; it gates the on-device decode/compact path vs the CPU oracle. The name reads as a residency-tier flag (cf. `OutputTier::Resident` used later at line 322), which is misleading in context. Suggested: rename to `on_device` / `has_gpu`.

### G8
- [G8.comments][LOW] extract_f2_core.cpp:92,131,293,303,310 — the section-number comments are non-contiguous: `1.` (line 92), `5.` (line 131), `7.` (line 293), `8.` (line 303), `8b.` (line 310); sections 2/3/4/6 are absent. This is a stale numbering artifact carried over from the verbatim lift of cmd_extract_f2.cpp:157-498 (the dropped steps live in the CLI's pre/post-amble). A first-time reader will assume missing code. Suggested: renumber the steps contiguously for this function, or drop the numeric prefixes.

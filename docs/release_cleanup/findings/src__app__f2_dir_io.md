# src__app__f2_dir_io
Files: /home/suzunik/steppe/src/app/f2_dir_io.cpp, /home/suzunik/steppe/src/app/f2_dir_io.hpp
Subsystem: app-cli

## Findings

### G8
- [G8.comments][LOW] f2_dir_io.cpp:36-38 — The read_pops_txt header comment contains a self-cancelling parenthetical ("a leading/trailing space is NOT trimmed inside a label (pop names can legitimately... they cannot, but we keep the label verbatim...)"). The "(pop names can legitimately... they cannot," fragment is an unfinished/contradictory aside that adds noise rather than rationale. Suggested: replace with a single clear statement, e.g. "labels are kept verbatim minus the line terminator so the name↔index map is byte-exact."

### G3
- [G3.dead-code][LOW] f2_dir_io.cpp:16 — The include comment advertises `kF2DiskHeaderSize` as a reason for reaching f2_disk_format.hpp, but the reader never references `kF2DiskHeaderSize` (it uses `sizeof(hdr)` at line 77 and the authoritative `hdr.*_offset` fields for seeks). Listing it as a used import is misleading. Suggested: drop `kF2DiskHeaderSize` from the include's used-symbol comment (the constant is genuinely unused here).

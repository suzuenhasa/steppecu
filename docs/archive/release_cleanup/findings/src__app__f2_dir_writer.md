# src__app__f2_dir_writer
Files: /home/suzunik/steppe/src/app/f2_dir_writer.cpp, /home/suzunik/steppe/src/app/f2_dir_writer.hpp
Subsystem: app-cli

## Findings

### G8
- [G8.src__app__f2_dir_writer][LOW] f2_dir_writer.cpp:482 — `blgsize_cm` (double) is serialized into meta.json via a default `std::ostringstream`, which formats doubles at the default precision (~6 significant digits). For a provenance/reproducibility record (architecture.md §12 block) a value could be silently rounded on round-trip. Not parity-load-bearing (provenance only), but the default precision is an undocumented implicit choice. Suggested: set `js << std::setprecision(...)` (e.g. `max_digits10`) or document that blgsize_cm provenance is intentionally coarse.

### G10
- [G10.src__app__f2_dir_writer][LOW] f2_dir_writer.cpp:457 — `F2DirMeta m = meta;` is an unconditional full-struct copy of the const param made solely so the (default-OFF) `hash_source_files` branch can fill the three small snp/ind/geno sha strings. When `hash_source_files == false` (the documented default), the copy is pure overhead and `m` is otherwise read-only. Minor; the copy is cheap (small struct). Suggested: copy only inside the `if (m.hash_source_files)` branch, or read `meta` directly and keep three local sha strings.

(No issues found for groups G2, G3, G4, G5, G6, G7, G9. Notes verified clean: the `slab_bytes` widening at line 414 and the int32-trailer size at line 438 both multiply through `std::uint64_t` factors — no int-index overflow at P~2500/n_block~757 scale, matching the F2BlockTensor::size() and f2_disk_format slab_offset widening conventions; the `static_cast<int>(pop_labels.size())` compares at lines 385/398 are against `f2.P` which is `int` by design and bounded by P~2500; magic numbers 64/56 are SHA-256 block-geometry constants, the single magic/version home is f2_disk_format.hpp, and the `0xB1/0x1B/0xF0` SHA-NI shuffle masks are documented Intel-reference flow; the duplicated SHA round-constants comment is explained as a deliberate single-home shared by scalar+SHA-NI; warp/`32`-related items are N/A — non-CUDA TU.)

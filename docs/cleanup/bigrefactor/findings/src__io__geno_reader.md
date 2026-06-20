# Review findings — src__io__geno_reader

Files: /home/suzunik/steppe/src/io/geno_reader.cpp, /home/suzunik/steppe/src/io/geno_reader.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Verification notes (not findings; explains why each 4.x is clean):
- 4.1 float/double: no floating-point literals or math anywhere in this unit. N/A.
- 4.2 index width: every global index/offset is std::size_t (GenoHeader.n_snp/
  n_records/bytes_per_record/header_bytes, PopGroup::rows[i] as `row`, out_ind,
  n_ind) or std::streamoff (the record offset). No 32-bit int global index.
  geno_reader.cpp:30-31 statically asserts sizeof(std::streamoff) >= 8.
- 4.3 allocation sizing: tile.packed is std::vector<std::uint8_t>, so
  resize(n_ind * bytes_per_rec) (geno_reader.cpp:164) is element==byte count; no
  `* sizeof(T)` needed. in.read writes bytes_per_rec bytes into a char* (line 198).
- 4.4 unsigned countdown: no decrementing loops; all loops range-based/ascending.
- 4.5 signed/unsigned compare: all loop/guard compares are size_t vs size_t
  (row >= records_present_ line 119; n_ind > MAX/bytes_per_rec line 140);
  gcount() compares cast to std::streamsize (lines 41, 199).
- 4.6 int overflow before widening: the record offset multiply (lines 192-195)
  casts BOTH row and bytes_per_record to std::streamoff BEFORE multiplying, so the
  product is 64-bit; the n_ind*bytes_per_rec product is explicitly overflow-checked
  (lines 140-145) before resize; pointer arithmetic is size_t throughout.
- 4.7 host/device pointer typing: host-only io-leaf TU (no CUDA); the only raw
  pointers are host file-buffer char* — no device address space to confuse. N/A.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Verification notes (not findings; explains why each 2.x is N/A):
- This is a pure host C++20 `io`-leaf TU (geno_reader.hpp:16, geno_reader.cpp:8-9
  declare "no CUDA, no core/device dependency"). No .cu, no CUDA runtime/driver
  calls, no device code anywhere in the unit.
- 2.1 dropped archs (Maxwell/Pascal/Volta, sm_50/60/70): no arch flags, no CMake
  arch lists, no device compilation in this TU. N/A.
- 2.2 texture/surface references (texture<...>, cudaBindTexture*): none present;
  grep for texture/surface/cudaBindTexture over both files = no matches. N/A.
- 2.3 non-_sync warp intrinsics (__shfl/__ballot/__any/__all without _sync): no
  warp intrinsics, no device code. N/A.
- 2.4 cudaThreadSynchronize -> cudaDeviceSynchronize: no CUDA sync calls of any
  kind. N/A.
- The only tokens matching "cuda" are the substring inside "std::runtime_error"
  (hpp:17,72; cpp:158) — false positives, not API usage.
-->

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/io/geno_reader.cpp:14 — `#include <cstdint>` is unused: the TU names no `<cstdint>`-defined type (no `std::uint8_t`/`int32_t`/etc. token anywhere in the file). The `tile.packed` member is `std::vector<std::uint8_t>` but its element type is declared in genotype_tile.hpp; this TU only calls `.resize()`/`.data()` on it. Borderline (one could argue it documents the byte-buffer intent), so LOW. Suggested: drop the include, or keep it deliberately as an IWYU marker for the byte-buffer member.

<!--
Verification notes (not findings; explains why each 3.x is otherwise clean):
- 3.1 commented-out blocks: every comment in both files is documentation/rationale
  prose (header contract, fail-fast rationale, cleanup-item cites) — NO commented-out
  statements or expressions kept "just in case". Clean.
- 3.2 unreachable code: no `#if 0` / `#ifdef` dead branches; every `throw` is a
  reachable guard; the sole `return tile` (cpp:208) is the last statement of
  read_tile with no code after it. Clean.
- 3.3 unused symbols (besides the <cstdint> LOW above): all other .cpp includes are
  used — <array> (head buffer cpp:39), <cstddef> (std::size_t), <fstream>
  (std::ifstream cpp:34,179), <ios> (std::ios::*, std::streamoff/streamsize),
  <limits> (std::numeric_limits cpp:140), <stdexcept> (std::runtime_error),
  <string> (std::string/to_string). hpp includes all used (GenoHeader/GenoFormat,
  GenotypeTile, IndPartition). All members read (path_, header_, records_present_);
  all params used (geno_path; part/snp_begin/snp_end). No unused local helper.
- 3.4 computed but unread: every assigned local is subsequently read — data_bytes
  (cpp:67), fsize (cpp:62,66), records_present_ (cpp:67,72-73 + getter), tile_snps
  (cpp:108,149), bytes_per_rec (cpp:140,148,164,197,198), n_ind (cpp:140,150,164),
  out_ind (cpp:197,204,206), off (cpp:196), dst (cpp:198). Clean.
-->


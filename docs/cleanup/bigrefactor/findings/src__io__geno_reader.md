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

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

<!--
Verification notes (not findings; explains why each 5.x is clean):
- 5.1 unnamed literals: the unit uses NAMED constants and helpers from
  eigenstrat_format.hpp, not raw magic numbers. The 48-byte header is the named
  kGenoHeaderBytes (cpp:39,40,41,43-44; never the literal 48). The 4-SNPs/byte
  packing factor is encapsulated in packed_bytes(tile_snps) (cpp:108) — no raw `/4`
  or `>>2` here (kCodesPerByte lives in eigenstrat_format.hpp:54). Remaining
  literals are not "magic": 0 (snp_begin sentinel guard cpp:84; pop_offsets origin
  push_back(0) cpp:188; degenerate-header check cpp:52) and the `+ 1` reserve slack
  for the offsets fencepost (cpp:176, n_groups+1 segment bounds) are self-evident
  structural constants, not tunables.
- 5.2 hardcoded sizes/bounds that should be params/derived: none. Every size is
  DERIVED — bytes_per_rec from packed_bytes(snp range) (cpp:108), the data region
  from header_.header_bytes / n_records / bytes_per_record (cpp:62,65-67), the
  record offset from header_bytes + row*bytes_per_record (cpp:192-195), the tile
  buffer from n_ind*bytes_per_rec (cpp:164). SNP range comes in as params
  (snp_begin/snp_end); all bounds (records_present_, n_snp, n_records) come from the
  parsed header. No baked-in cap or buffer size.
- 5.3 duplicated constants / drift: NONE — single source of truth. The record
  stride exists only as header_.bytes_per_record (derived once in eigenstrat_format)
  and is reused for both file-size validation (cpp:67) and offset math (cpp:195);
  bytes_per_rec for the gather is computed once (cpp:108) and reused for the
  overflow check, resize, pointer step, and read length (cpp:140,164,197,198). No
  literal repeated in two places that could drift apart.
- 5.4 hardcoded paths / IDs / device ids: none. The only path is the constructor
  argument geno_path stored in path_ (cpp:33,90) and reused for reopen (cpp:179) and
  error messages. Host-only io-leaf TU — no device ids, stream ids, or GPU indices.
- 5.5 ambiguous 32 (warp size vs other): N/A — the literal 32 does not appear
  anywhere in the unit, and this is a pure host TU with no warp/thread concept. The
  `sizeof(std::streamoff) >= 8` static_assert (cpp:30) uses 8 = bytes-for-64-bits
  and is self-documented by its own diagnostic string ("a >=64-bit std::streamoff")
  — an unambiguous platform-width floor, not a magic tunable.
-->

## Group 6 — Naming

- [6.3][LOW] src/io/geno_reader.cpp:108,148 — abbreviation drift in one function: the local per-record byte count is `bytes_per_rec` (cpp:108,140,144,164,168,173,197,198,199) yet it is assigned straight into the full-name member `tile.bytes_per_record` (cpp:148) and parallels the header field `header_.bytes_per_record` (cpp:52,67,195) — the same concept spelled `_rec` (local) vs `_record` (member/field) within `read_tile`. The error string at cpp:143 even prints `bytes_per_record=` while interpolating the `bytes_per_rec` variable, underscoring the inconsistency. Suggested: rename the local to `bytes_per_record` to match the member/field convention used everywhere else in the unit.
- [6.3][LOW] src/io/geno_reader.cpp:116,150 — same abbreviation drift for the individual count: the local is `n_ind` (cpp:116,126,140,150,164,168,173) but is assigned into the full-name member `tile.n_individuals` (cpp:150); `out_ind` (cpp:187,197,204,206) shares the `_ind` abbreviation. `ind` is domain-conventional (ind_reader), so unambiguous, hence LOW. Suggested: keep `ind` consistently or align the local to the `n_individuals`/`out_individual` spelling of the member — pick one convention per concept.

<!--
Verification notes (not findings; explains why each 6.x is otherwise clean):
- 6.1 cryptic names: `g` (cpp:117,189) is a tight range-based loop counter over
  part.groups — explicitly permitted. `dst` (cpp:197), `off` (cpp:192), `in`
  (cpp:34,179), `head` (cpp:39), `row` (cpp:118,191), `fsize`/`data_bytes`
  (cpp:61,65) are standard, descriptive, and well understood in context. No
  tmp/data2/arr/flag anywhere.
- 6.2 misleading names: every name matches its content — `records_present_`
  counts complete records, `pop_offsets`/`pop_labels` hold offsets/labels,
  `n_ind`/`out_ind`/`n_snp`/`tile_snps` are counts/indices that are exactly what
  they say. No count-that-is-an-index or list-that-is-a-map mismatch.
- 6.4 nonstandard abbreviations: `rec` (record), `ind` (individual), `snp` are
  domain-standard in this codebase and the eigenstrat/AT2 vocabulary; not
  nonstandard. Captured the only real drift under 6.3 above.
-->

## Group 7 — Duplication

- [7.1][LOW] src/io/geno_reader.cpp:165-175 — the two resize() catch handlers are near-copy-pasted blocks that differ only by the caught exception type and a single prose phrase ("out of memory allocating tile" vs "tile too large for the allocator"); both build the same `... (<n_ind> individuals * <bytes_per_rec> bytes/record...) for <path_>` message. Suggested: build the shared message body once (a local string with the common `n_ind`/`bytes_per_rec`/`path_` interpolation) and have each catch prepend its distinguishing phrase, or factor a tiny `throw_alloc_failure(reason)` helper.
- [7.2][LOW] src/io/geno_reader.cpp:142-144,167-169,172-174 — the `std::to_string(n_ind) + " ... " + std::to_string(bytes_per_rec) + " bytes/record ... " + path_` interpolation is repeated three times (the overflow throw and both catch throws), all reporting the same `n_ind * bytes_per_rec` size for `path_`. Suggested: compute the shared `n_ind`/`bytes_per_rec`/`path_` size-description substring once and reuse it across the three throws.
- [7.2][LOW] src/io/geno_reader.cpp:40-41 — `static_cast<std::streamsize>(kGenoHeaderBytes)` is computed twice on adjacent lines (the read length and the gcount comparand), both loop-invariant for `kGenoHeaderBytes`. Suggested: hoist into one `const auto header_size = static_cast<std::streamsize>(kGenoHeaderBytes);` and reuse.
- [7.2][LOW] src/io/geno_reader.cpp:198-199 — `static_cast<std::streamsize>(bytes_per_rec)` is computed twice inside the gather loop (the read length and the gcount comparand); it is loop-invariant. Suggested: hoist a `const std::streamsize rec_bytes = static_cast<std::streamsize>(bytes_per_rec);` once before the gather loop and reuse for both read and compare.

<!--
Verification notes (not findings; explains why the rest of Group 7 is clean):
- 7.1 copy-pasted blocks: the only true near-duplicate pair is the bad_alloc/
  length_error catch handlers (flagged above). The many `throw std::runtime_error(
  "io::GenoReader...: <msg> ... " + path_)` sites (cpp:36,42-44,49-50,53,63,69,
  80-82,85-87,91-93,103-104,120-123,141-144,166-174,181,200-202) share only the
  idiomatic `"prefix: " + ... + path_` shape with DIFFERENT messages/operands — that
  is conventional error construction, not a copy-pasted block differing by a
  constant. Not flagged.
- 7.2 repeated expressions: the two-pass walk over part.groups (validation+count at
  cpp:116-127, then gather at cpp:189-207) re-traverses the partition, but the first
  pass is the deliberate fail-fast-BEFORE-resize guard (architecture.md §2) and the
  second is the post-allocation gather; fusing them would resize before validating,
  defeating the design — intentional, not flagged. Captured the genuine
  loop-invariant recompute cases (the duplicated streamsize casts) above. The record
  offset (cpp:192-195) is computed once per row and used once.
- 7.3 repeated sizeof/casts: no `sizeof` anywhere (tile.packed is a byte vector, so
  element==byte; see Group 4 4.3). The streamsize casts that genuinely repeat are
  flagged under 7.2 above. The per-row `static_cast<std::streamoff>` casts (cpp:193-
  195) wrap DIFFERENT operands (header_bytes, row, bytes_per_record) and are not a
  redundant recompute. No cast worth templating.
- 7.4 collapsible boilerplate: a `throw_io_error(msg)` helper could fold the ~12
  `throw std::runtime_error("io::GenoReader...: " + ... + path_)` sites, but each
  carries a distinct message and operands so a macro/helper buys little beyond the
  shared "...: " + path_ tail — borderline-subjective hygiene, not a clear win;
  noted, not flagged as a finding. The catch-pair fold is the one concrete
  collapsible case and is captured under 7.1.
-->

## Group 8 — Comments

No Group 8 issues found.

<!--
Verification notes (not findings; explains why each 8.x is clean):
- 8.1 restating code: every comment is rationale/contract prose, not a restatement
  of the statement it sits on. The fail-fast block (cpp:96-105), the checked-multiply
  block (cpp:129-139), the exception-translation block (cpp:151-162), and the gather
  description (cpp:184-186) all explain WHY (downstream silent-empty-tile, size_t wrap
  -> heap overflow, length_error/bad_alloc not deriving from runtime_error, pop-
  contiguous layout), never paraphrase the code. cpp:108 `// ceil(tile_snps/4)` and
  cpp:71-73 (records_present cap) annotate the EFFECT/intent of a call, not echo it.
  No `i++; // increment i` pattern anywhere. Clean.
- 8.2 stale comments: every comment still matches the code. cpp:108
  `// ceil(tile_snps/4)` matches packed_bytes(tile_snps); the streamoff rationale
  (cpp:25-29) matches the dual streamoff casts at cpp:192-195; the oracle formula
  `n_records = (fsize - hdr) // bpi` (cpp:59) matches records_present_ = data_bytes /
  header_.bytes_per_record (cpp:65-67); hpp:55-57's `bytes_per_record ==
  ceil((snp_end-snp_begin)/4)` matches packed_bytes at cpp:108; the size-multiply-wrap
  comment (cpp:113-115, 129-139) matches the row<records_present_ guard (cpp:119) and
  the n_ind>MAX/bytes_per_rec check (cpp:140). The M1/M5 framing (cpp:84-88, hpp:9-14,
  59-61) is forward-looking contract, not behavior the code lacks. No stale comment.
- 8.3 missing rationale: every non-obvious choice carries its WHY and an owned cite.
  static_assert width floor (cpp:25-31, "fails to BUILD rather than silently
  truncating"); empty-partition reject (cpp:96-105, cleanup 2.2); checked multiply
  (cpp:129-139, cleanup 1.5, incl. the bytes_per_rec!=0 proof for the divide at
  cpp:140); exception translation (cpp:151-162, cleanup 2.1, names [vector.capacity]/
  [container.alloc]); records_present_ cap (cpp:71-73). No naked constant/workaround.
- 8.4 orphan TODO/FIXME/HACK: NONE present in either file. No TODO/FIXME/HACK/XXX
  token anywhere (both files read in full). The "cleanup geno_reader N.N" tags
  (cpp:27,99-100,115,129,138,151; hpp:69) are owned review-cite anchors with full
  inline context, not ownerless markers. Clean.
-->

## Group 9 — Constants & configuration

No Group 9 issues found.

<!--
Verification notes (not findings; explains why each 9.x is clean):
- 9.1 should-be-const/constexpr left mutable: every invariant local in read_tile is
  already `const` — tile_snps (cpp:107), bytes_per_rec (cpp:108), n_ind after the
  count loop is read-only by use (declared mutable at cpp:116 because it accumulates
  in the loop, then never reassigned), off (cpp:192), dst (cpp:197), data_bytes
  (cpp:65), fsize (cpp:61). The genuinely-mutable locals MUST be mutable: `head`
  (cpp:39) is the read target buffer, `tile` (cpp:147) is the accumulating builder,
  `out_ind` (cpp:187) increments per gathered row, `in` (cpp:34,179) is the stream.
  The members path_/header_/records_present_ (hpp:90-92) cannot be `const`: the class
  is move-only with defaulted move-assignment (hpp:85-86), which requires non-const
  members. records_present_ is reassigned during construction (cpp:67,73). No
  should-be-const symbol left mutable.
- 9.2 tangled config: ALL tunable knobs are surfaced at the top of
  eigenstrat_format.hpp as `constexpr` (kGenoHeaderBytes=48 line 50, kCodesPerByte=4
  line 54, kBitsPerCode=2 line 57) and consumed here only via named symbols/helpers
  (kGenoHeaderBytes at cpp:39-44; packed_bytes() at cpp:108) — never inlined as a
  buried literal in read_tile/ctor logic (cross-ref Group 5). The sole literal config
  in this TU, the `sizeof(std::streamoff) >= 8` static_assert (cpp:30), is a
  self-documented platform-width floor at file top, not a buried logic knob. No
  tunable hidden inside a function body.
- 9.3 positional booleans: NO function in this unit takes a bool parameter
  (GenoReader(const std::string&); read_tile(const IndPartition&, size_t, size_t);
  header()/records_present() are nullary). No `foo(true, false, ...)` call site.
  The std::ios flags passed to ifstream/seekg (std::ios::binary cpp:34,179;
  std::ios::end cpp:60; std::ios::beg cpp:196) are NAMED enum bitmask values, not
  positional bool args. Clean.
-->

## Group 10 — Initialization

No Group 10 issues found.

<!--
Verification notes (not findings; explains why each 10.x is clean):
- 10.1 late/distant or uninitialized-then-assigned: every local in both functions
  is declared AT first use AND initialized at the point of declaration. ctor: head{}
  (cpp:39, value-init), fsize (cpp:61 const), data_bytes (cpp:65 const). read_tile:
  tile_snps (cpp:107 const), bytes_per_rec (cpp:108 const), n_ind = 0 (cpp:116),
  tile (cpp:147, declared just before its field assigns), in (cpp:179, just before
  use), out_ind = 0 (cpp:187), off (cpp:192 const, per-row), dst (cpp:197 const).
  No `T x;` followed later by `x = ...` (no uninitialized-then-assigned), no local
  hoisted far from first use. Clean.
- 10.2 zero-init assumptions that do not hold: nothing relies on implicit zero-init.
  The two accumulators are EXPLICITLY zeroed — n_ind = 0 (cpp:116), out_ind = 0
  (cpp:187) — before the count/gather loops increment them. head is value-init `{}`
  (cpp:39), not left indeterminate. The `tile` builder (cpp:147) relies on
  GenotypeTile's own default member initializers, which are all present and sound
  (genotype_tile.hpp:53,57,61 bytes_per_record/n_snp/n_individuals = 0; packed/
  pop_offsets/pop_labels default-construct empty); and this TU additionally assigns
  bytes_per_record/n_snp/n_individuals explicitly (cpp:148-150) before any reader
  observes them, so no zero-init reliance for the scalar fields it uses. The member
  records_present_ has an explicit default-member-initializer = 0 (hpp:92) and is
  set during construction (cpp:67,73) before the getter is reachable. The member
  header_ (hpp:91) is default-constructed with GenoHeader's full default member
  initializers (eigenstrat_format.hpp:124-129: format=Unknown, all size_t=0,
  header_bytes=kGenoHeaderBytes) and then UNCONDITIONALLY overwritten by
  header_ = parse_geno_header(head) (cpp:47) before any field is read — no
  read-before-assign and no dependence on the default state. Clean.
-->


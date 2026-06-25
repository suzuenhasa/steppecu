# src__io__geno_reader
Files: /home/suzunik/steppe/src/io/geno_reader.cpp, /home/suzunik/steppe/src/io/geno_reader.hpp
Subsystem: io

## Findings

### G3
- [G3.geno_reader][LOW] geno_reader.cpp:546-548 — `n_records_to_read = min(records_present_, tile_snps)` is computed only to be compared `< tile_snps` in the very next line; the gather loop (line 623) iterates `tile_snps`, never `n_records_to_read`. The min/variable is a dead intermediary — once the `< tile_snps` throw is taken, the only surviving value is `tile_snps`. Suggested: replace with a direct `if (records_present_ < tile_snps) throw ...` (matching the EIGENSTRAT/PLINK/ANCESTRYMAP readers, which already use the plain `tile_snps > records_present_` form at lines 671, 824, 968).

### G4
- [G4.geno_reader][LOW] geno_reader.cpp:1050 — `const std::size_t n_lines = tile_snps * n_ind;` is an UNCHECKED size_t multiply, unlike every other tile-size product in this file (which all get a `> SIZE_MAX/b` guard). The inline comment argues it cannot wrap because `tile_snps*src_bpr` was checked and `n_ind <= 4*src_bpr`, but that bound gives `n_lines <= 4*(tile_snps*src_bpr)`, and `tile_snps*src_bpr` is only guaranteed `<= SIZE_MAX` — so `4x` it CAN wrap. `n_lines` is only a loop bound / context-string value (buffer indexing uses `s*src_bpr` with `s<tile_snps`, which is bounded), so a wrap is a wrong-result/early-EOF logic bug at extreme ANCESTRYMAP scale, not a heap overflow. The comment's "no wrap in practice" reasoning is the load-bearing claim and it is not airtight. Suggested: apply the same `tile_snps > SIZE_MAX/n_ind` checked-multiply guard used for the buffer products, or derive the loop without materializing the full product.

### G7
- [G7.geno_reader][LOW] geno_reader.cpp:565-586, 686-707, 839-860, 983-1004 — the SELECTION + pop-contiguous gather-list build (reserve pop_offsets/pop_labels, push 0, per-group loop validating `row >= header_.n_ind` then `sel_rows.push_back(row)`, finalize `n_individuals = out_ind`) is copy-pasted verbatim across all four SNP-major readers, differing only by the reader name in the throw string. The four checked-multiply + try/resize blocks (597-612, 715-733, 868-886, 1012-1030) are likewise near-identical. The comments at each site explicitly acknowledge "byte-for-byte the same construction as read_snp_major_tile". Suggested: extract a private helper (e.g. `build_selection(part, n_ind, const char* reader_name, SnpMajorTile&)` and a `checked_alloc_snp_major(...)`) so the four readers share the one validated implementation; this is the dominant duplication in the file.
- [G7.geno_reader][LOW] geno_reader.cpp:638-650, 787-799, 934-947 — the "wrong reader" dispatch error builds a format-name string via a nested ternary chain that grows by one branch per reader (EIGENSTRAT lists 3 cases, PLINK 4, ANCESTRYMAP 5). The chains are copy-paste with one added arm each. Suggested: a single `geno_format_name(GenoFormat)` (or "which reader to use" map) shared by all the mismatch throws.

## Notes (verified clean, not findings)
- read_tile's `bytes_per_record = packed_bytes(tile_snps)` (line 401) differs from the seek stride `header_.bytes_per_record` (line 496); this is CORRECT for the documented `snp_begin==0` byte-aligned-prefix case (hpp:55-58) — the prefix bytes start at record offset 0, so reading the prefix width from each record's start is right.
- All record-offset multiplies (lines 493-496, 624-626, 907-909) are widened to `std::streamoff` before the multiply, with a `static_assert(sizeof(std::streamoff) >= 8)` (line 76) guarding the AADR-scale (~4e9) seek target — no int-index overflow.
- The four buffer-size products (n_individuals*bytes_per_record, tile_snps*src_bpr x3) each get a checked-multiply `> SIZE_MAX/b` guard with a proven-nonzero divisor and a bad_alloc/length_error -> runtime_error translation matching the documented exception contract — sound.
- Pack-loop byte index `i/4 < src_bpr` holds since `i < n_ind` and `src_bpr == packed_bytes(n_ind) == ceil(n_ind/4)`; no OOB write.
- No magic numbers: shifts/masks/strides all route through the single-homed eigenstrat_format.hpp constants (kCodesPerByte, kBitsPerCode, kBedMagic*, packed_bytes, bed_code_in_byte, kBedToCanon). Note the "32" mentioned in the checklist is warp-specific and does not appear here (host-only TU).

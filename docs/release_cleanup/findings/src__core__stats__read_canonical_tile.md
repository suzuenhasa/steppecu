# src__core__stats__read_canonical_tile
Files: /home/suzunik/steppe/src/core/stats/read_canonical_tile.cpp, /home/suzunik/steppe/src/core/stats/read_canonical_tile.hpp
Subsystem: core-stats

## Findings
No issues found (groups checked: G2-G10).

Notes (no action; reviewed and confirmed clean, not flagged):
- read_canonical_tile.cpp:36 `static_cast<int>(src.n_pop())` narrows std::size_t -> int to fill `SnpMajorTileView::n_pop` (declared `int` in device/backend.hpp:486). Correct and necessary; P~2500 is far inside int range, no overflow. (G4 considered.)
- read_canonical_tile.cpp:72/80/91/102 the five dispatch arms share the shape `transpose_snp_major(reader.read_*_snp_major_tile(...), backend)`, differing only by the reader method name — this is idiomatic format dispatch, not extractable duplication. (G7 considered.)
- Comments verified against current structs: io::SnpMajorTile (src/io/snp_major_tile.hpp:43-76), SnpMajorTileView/CanonicalTile/TileEncoding (src/device/backend.hpp:446-506), GenoFormat enumerators (src/io/eigenstrat_format.hpp:192), and all five reader methods (src/io/geno_reader.hpp) all match the code and the prose; no stale comments. Switch is exhaustive over all six GenoFormat enumerators; the `default` arm is genuinely defensive. (G8 considered.)

# src__io__eigenstrat_format
Files: /home/suzunik/steppe/src/io/eigenstrat_format.cpp, /home/suzunik/steppe/src/io/eigenstrat_format.hpp
Subsystem: io

## Findings

### G8
- [G8.comments][LOW] eigenstrat_format.hpp:342-345 — the parse_geno_header doc says "n_ind, n_snp for TGENO; n_ind, n_snp for GENO" — the same pair is restated verbatim for both formats, which reads as a copy-paste slip even though the intent is "both store the same two numbers". The clause "both store the same two numbers, the difference is the record axis" already conveys this, so the duplicated enumeration adds confusion not information. Suggested: drop the second "n_ind, n_snp for GENO" and keep only the "both store the same two numbers" clause.

No further issues found (groups checked: G2-G10). The unit is a host-pure io leaf: G2 (no CUDA APIs), G11-G22 (no device code) are N/A; the overflow guard at eigenstrat_format.cpp:87-96 correctly handles unsigned modular wrap (G4 clean), constants are single-homed and derived not duplicated (G5/G7 clean), all includes are used (G3 clean), and packed_bytes/code_in_byte use std::size_t / guarded int arithmetic with no index-overflow risk at P~2500/M~584k scale (G4 clean).

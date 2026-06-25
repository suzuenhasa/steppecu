# src__io__genotype_source
Files: /home/suzunik/steppe/src/io/genotype_source.cpp, /home/suzunik/steppe/src/io/genotype_source.hpp
Subsystem: io

## Findings
No issues found (groups checked: G2-G10).

Notes (not findings):
- genotype_source.cpp:20-22 — the shared `std::error_code ec` is the non-throwing-overload sink for both `exists` probes; its value is intentionally never inspected (a probe I/O error is correctly treated as "not present"). The first `exists` result is overwritten before being read, but that is the documented non-throwing-`exists` idiom, not a computed-but-unread bug.
- Includes are all present and used: the .cpp gets `read_snp`/`read_ind` declarations transitively via genotype_source.hpp (snp_reader.hpp / ind_reader.hpp), and includes plink_reader.hpp directly for `read_bim`/`read_fam`.

# src__io__snp_reader
Files: /home/suzunik/steppe/src/io/snp_reader.cpp, /home/suzunik/steppe/src/io/snp_reader.hpp
Subsystem: io

## Findings

### G8
- [G8.cleanup snp_reader][LOW] snp_reader.cpp:30-38, 47-67, 96-101, 110-124 — Several block comments are unusually long and partly restate the implementation / re-justify already-landed cleanup decisions (e.g. the `parse_full` comment re-narrates the copy-paste fold it performed; the `chrom_code` comment re-argues the from_chars-vs-stoi choice across ~15 lines). These are rationale-bearing (not pure restatement) so this is borderline, but the volume reads as historical changelog ("cleanup snp_reader 7.1", "B14", "B15", "F12/B16") baked into source comments rather than current-state documentation. Suggested: trim the changelog/ticket-id narration, keep the load-bearing rationale (why from_chars not stoi; why explicit isfinite guard).

No other issues found (groups checked: G2-G10).

Notes (clean, verified, NOT findings):
- G4 (numeric/scale): `count`/`line_no`/`max_snps` are `std::size_t`; SNP-count indexing is element-based, no byte/element confusion and no int-index overflow at M~584k (all loop/index vars are size_t or string indices). `next_other` decrements from -1 (`kFirstOtherChromCode`) — bounded by distinct labels, not a realistic underflow. `chrom_code` returns `int` only after `parse_full(tok,value)` succeeds for in-range tokens; overflow falls to the sentinel path (snp_reader.cpp:80-83), correctly avoiding UB.
- G5 (magic numbers): all column indices, field counts, chrom codes, and the missing-allele char are named `constexpr` single-homed in eigenstrat_format.hpp (kMinSnpFields=3, kFullSnpFields=6, kRefAlleleCol=4, kAltAlleleCol=5, kMissingAllele='N', kChromCodeX/Y/Mt, kFirstOtherChromCode=-1) and referenced by name (snp_reader.cpp:86-88, 148, 174-192). No bare literals in logic.
- G9 (constants/config): `parse_full`/`parse_genpos`/`chrom_code` are `[[nodiscard]]` where appropriate; no positional-boolean calls; the SNP cap is a surfaced parameter (`max_snps`), not buried.
- G3 (dead/unused): no dead code; all includes are used (<charconv> from_chars, <cmath> isfinite, <map> other_codes, <sstream> istringstream, <fstream> ifstream, <cctype> isdigit). `static_cast<unsigned char>` on the isdigit arg (snp_reader.cpp:72) is the correct defensive cast, not noise.
- G10 (init): `value`/`out` are zero-initialized before from_chars writes; `table`, `other_codes` default-constructed; no uninitialized reads.
- G6/G7: naming consistent (snake_case, k-prefixed constants); the ref/alt extraction is folded into one `allele` lambda (snp_reader.cpp:191-193), no copy-paste duplication remaining.

# include__steppe__dstat
Files: /home/suzunik/steppe/include/steppe/dstat.hpp, /home/suzunik/steppe/src/core/stats/dstat.cpp
Subsystem: core-stats

## Findings

### G3 dead/unused
- [G3.include__steppe__dstat][LOW] src/core/stats/dstat.cpp:39 — `<stdexcept>` is included but no exception type or `throw` is used in this TU (the only `throw` mentions are in comments at lines 31/108/110; domain outcomes use the NaN sentinel, io faults propagate from the io leaf). Suggested: drop the `<stdexcept>` include.
- [G3.include__steppe__dstat][LOW] src/core/stats/dstat.cpp:48,52,53 — `io/eigenstrat_format.hpp`, `io/ind_reader.hpp`, `io/snp_reader.hpp` are already re-exported transitively by `io/genotype_source.hpp` (line 50, which `#include`s all three), and this TU references none of their symbols beyond what genotype_source surfaces (GenoFormat/PopSelection/IndPartition/SnpTable). Defensible as IWYU-explicit, but redundant. Suggested: optional — rely on genotype_source.hpp or keep as documented direct deps; low priority.

### G5 hardcoded values / duplicated constants
- [G5.include__steppe__dstat][MED] src/core/stats/dstat.cpp:171 vs 182 — the autosome keep bound is expressed two ways inside the same function: the resident path passes the named `kAutosomeChromMin`/`kAutosomeChromMax` (config.hpp = 1/22) to `decode_af_compact_autosome`, while the CpuBackend fallback hardcodes the literal predicate `chr < 1 || chr > 22` (line 182). The two branches MUST stay byte-identical for parity (comment at 158-161 asserts "BOTH produce the IDENTICAL kept SET"); the duplicated literal is a drift bug waiting to happen if the named constants ever change. Suggested: use `kAutosomeChromMin`/`kAutosomeChromMax` in the host-fallback predicate at line 182 too.

### G7 duplication
- [G7.include__steppe__dstat][LOW] src/core/stats/dstat.cpp:129-133, 194-198, 208-212 — the degenerate-result fill (`res.est/se/z/p.assign(N, std::nan("")); res.status = Status::Ok;`) is copy-pasted verbatim at three early-return sites (and a fourth degenerate path is the empty-N guard at line 86). Suggested: extract a small `fill_nan(res, N)` (or a `DstatResult make_degenerate(N)`) helper.

### G8 comments
- [G8.include__steppe__dstat][LOW] src/core/stats/dstat.cpp:6-7 (and the header phrasing at 20-27) — the top-of-file block calls the num/den block-jackknife "host-pure here" / describes it as accumulated in this TU, but the code (lines 224-230) now delegates entirely to the `dstat_blocks_jackknife` backend virtual; lines 67-73 explicitly correct this ("NO LONGER a host loop here … it is the SHARED on-device ratio_block_jackknife backend virtual"). The two comment blocks contradict each other; line 6-7 is stale. Suggested: reword line 6-7 to match the backend-virtual reality stated at 67-73.

No issues found in groups G2, G4, G6, G9, G10. (G11-G22 not applicable: is_cuda=false.) Scale/overflow checked (G4): all `P*M`, `N*4`, and `P*s` products are `static_cast<std::size_t>`-widened before multiply (lines 94, 176-177, 183); the `M`-length spans over `snptab.chrom`/`genpos_morgans` (lines 169-170, 180-190) are in-bounds since `M = tile.n_snp = M0 = min(header.n_snp, snptab.count) <= snptab.count`.

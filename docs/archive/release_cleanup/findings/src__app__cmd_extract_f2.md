# src__app__cmd_extract_f2
Files: /home/suzunik/steppe/src/app/cmd_extract_f2.cpp, /home/suzunik/steppe/src/app/cmd_extract_f2.hpp
Subsystem: app-cli

## Findings

### G3 (dead/unread)
- [G3.dead-code][MED] cmd_extract_f2.cpp:353,362-363 — `extracted.precision_tag` (documented in include/steppe/extract.hpp:67 as "the ENGAGED precision") is never read; instead `const Precision engaged = precision;` aliases the *requested* config precision (line 220) and that is what is written to `meta.precision_tag` (line 362) and the stdout summary (line 404). The library's engaged-precision result field is computed-but-unread, so meta.json records the requested precision even on an internal downgrade. Suggested: drive `engaged`/the meta tag from `extracted.precision_tag`, or drop the field if a downgrade is impossible.
- [G3.unused-include][LOW] cmd_extract_f2.cpp:50 — `#include "io/genotype_tile.hpp"` (GenotypeTile) is unused: `GenotypeTile`/`read_tile` appear only in the stale header comment (line 14), no tile is read here (confirmed by the line-185 comment "no tile is read here"), and `geno_reader.hpp` (line 48) already transitively includes it. Suggested: drop the direct include.
- [G3.unused-include][LOW] cmd_extract_f2.cpp:47,51,52 — `io/eigenstrat_format.hpp`, `io/ind_reader.hpp`, `io/snp_reader.hpp` are all re-included by `io/genotype_source.hpp` (line 49), which is the dispatcher this file actually calls (`read_ind_partition`/`read_snp_table`). The named types (GenoFormat, IndPartition/PopSelection/PopGroup, SnpTable) arrive transitively through it. Borderline under include-what-you-use (the symbols ARE named directly), but the three are redundant given the dispatcher header. Suggested: optionally rely on `genotype_source.hpp` and drop the three; or keep them deliberately under IWYU.

### G6 (naming)
- [G6.misleading][MED] cmd_extract_f2.cpp:353 — the name `engaged` (and the comments at 218-219 "honors it or downgrades to native" / 353 "the lib honors/downgrades internally") asserts this holds the precision actually engaged after any downgrade, but it is a verbatim copy of the requested `precision`. The name promises observability it does not provide. Suggested: rename to e.g. `requested_precision`, or make it actually carry `extracted.precision_tag`.

### G8 (comments)
- [G8.stale][LOW] cmd_extract_f2.cpp:354 — `const Precision engaged = precision; // recorded tag (the lib honors/downgrades internally).` The comment claims the value reflects the library's honor/downgrade decision; it does not (see G3/G6). Stale/misleading. Suggested: correct the comment or wire it to `extracted.precision_tag`.
- [G8.stale][LOW] cmd_extract_f2.cpp:7-9,14 — the header-comment "THE CHAIN" lists step 4 `reader.read_tile(part, 0, M) -> GenotypeTile`, but no `read_tile` call exists in this file anymore (the up-front read was reduced to sizing/validation; the real read moved into `steppe::run_extract_f2`, per the line-181-185 and line-312-321 comments). The step-4 line describes behavior the code no longer has. Suggested: update the chain comment to reflect that the tile read now lives in the library entry.
- [G8.stale][LOW] cmd_extract_f2.hpp:8 — the file-doc says the command wires "...-> `compute_f2_blocks_multigpu_device(/_tiered)` -> the STPF2BK1 dir WRITER", but the cpp now delegates the decode->filter->assign_blocks->tiered-compute->to_host chain to `steppe::run_extract_f2` (cpp lines 312-338) rather than calling `compute_f2_blocks_multigpu_*` directly. The header narrative predates that refactor. Suggested: update to reference `run_extract_f2`.

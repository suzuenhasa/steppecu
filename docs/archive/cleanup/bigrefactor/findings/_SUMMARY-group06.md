# GROUP 6 (Naming) — Roll-up Summary

Tasks: 6.1 cryptic names · 6.2 misleading names · 6.3 inconsistent conventions in one file · 6.4 nonstandard abbreviations.

## 1. Coverage

- Units in scope: **61** (scope = all).
- Units with findings: **31**.
- Clean ("No Group 6 issues found."): **30**.
- Total findings: **51** — **0 HIGH / 0 MED / 51 LOW**.

The high clean rate is structural, not a gap in the pass: the per-unit reviews consistently and correctly excluded the project's parity-load-bearing vocabulary from the cryptic-name net — the AT2/linear-algebra single letters (`P`, `M`, `Q`/`V`/`N`, `A`/`B`, `r`, `nl`/`nr`, `f2`, `Vpair`), the documented SPMG device index `g`/`G`, the device/host pointer prefixes (`d*`/`h_*`), and tight-loop counters. Renaming those would HARM the §12 parity-against-oracle diffability, so they were (rightly) left alone.

## 2. Counts by task + severity

| Task | HIGH | MED | LOW | Total |
|------|------|-----|-----|-------|
| 6.1 Cryptic names | 0 | 0 | 18 | 18 |
| 6.2 Misleading names | 0 | 0 | 6 | 6 |
| 6.3 Inconsistent conventions in one file | 0 | 0 | 17 | 17 |
| 6.4 Nonstandard abbreviations | 0 | 0 | 10 | 10 |
| **Total** | **0** | **0** | **51** | **51** |

## 3. Findings (all LOW — no HIGH/MED to lead with)

No HIGH or MED findings exist. The highest-leverage LOW findings (clearest wins / most cross-cutting) first:

### Most actionable — `vp` vs `vpair` and `_rec` vs `_record` abbreviation drift (mechanical renames)
- [6.4][LOW] src/device/cuda/p2p_combine.cu:107,139,141,147-148,168-169,187,190-191,246,265,267,273,285-286 — pervasive local `vp` vs the public struct field `vpair`.
- [6.3][LOW] src/device/cuda/f2_blocks_out.cu:84-85 — lone `vp_dev` breaks the file's own `vpair` convention (`f2_dev` keeps the full stat name).
- [6.4][LOW] src/device/cuda/decode_af_kernel.cu:69 — `byte_in_rec` (`rec`) vs `bytes_per_record` (full) in one TU.
- [6.3][LOW] src/io/geno_reader.cpp:108,148 — local `bytes_per_rec` assigned into member `tile.bytes_per_record`; the error string at :143 even prints `bytes_per_record=` while interpolating `bytes_per_rec`.
- [6.3][LOW] src/io/geno_reader.cpp:116,150 — local `n_ind` assigned into member `tile.n_individuals`.

### Same letter, two meanings within one TU (the recurring `M`/`m` and `s`/`t` collisions)
- [6.3][LOW] src/device/cuda/cuda_backend.cu:150,441,1053 vs 1359,1364,1877,1953,2120,2125,2173 — `M` = SNP count on the f2 path but = widened f4-matrix dim (nl·nr) in the fit virtuals.
- [6.3][LOW] src/device/cpu/cpu_backend.cpp:435,449,634 — same `m` (model dim) vs `M` (SNP count) dual meaning across methods.
- [6.1][LOW] src/core/internal/small_linalg.hpp:239 — swap temp `t` collides with the Jacobi rotation tangent `t` (line 200) in the same SVD function.
- [6.3][LOW] src/core/internal/small_linalg.hpp:235-242 — sort indices `a`/`b` break the file's `i`/`j`/`k`/`p`/`q` index convention and shadow value temporaries `a`/`b` in the same function.

### Opaque single-letter statistical coefficient `s` (the jackknife SE scale) — appears twice
- [6.1][LOW] src/device/cuda/cuda_backend.cu:2381 — `const double s = (nb-1)/sqrt(nb)`, read 60+ lines away at the launch sites, no comment naming it.
- [6.1][LOW] src/device/cuda/qpadm_fit_kernels.cu:1201,1219,1369,1377 — the matching kernel/launcher param `double s` (same scale), opaque in a multi-arg signature.

### count/index/extent named like a verb or boolean
- [6.2][LOW] src/device/vram_budget.hpp:147 — `fit` is an int block-count named like a predicate.
- [6.2][LOW] src/device/cuda/block_sink.cuh:104,149 + .cu:55,69-74,101,210,225-230,262 — `free_` is a per-slot flag VECTOR but reads as a single bool/action; odd one out vs `slots_`/`ready_`.
- [6.2][LOW] src/device/cuda/block_sink.cuh:140-141 + .cu:189-190 — members `f2_region_`/`vpair_region_` hold byte OFFSETS (header field for the same datum is correctly `f2_offset`/`vpair_offset`).

### misleading axis / role
- [6.2][LOW] src/io/filter/mind_prepass.cpp:37,64,70 — sample-axis loop index named `g` (reads as genotype/SNP) against the unit's sample/individual vocabulary.
- [6.2][LOW] src/core/fstats/f2_combine.cpp:100-101 — `part_slabs` is named a slab count but holds the element (double) count `slab*part.n_block`.
- [6.2][LOW] src/device/cuda/qpadm_fit_kernels.cu:364-376,559-570,790-800,890-917 — `RHS`=coefficient matrix / `LHS`=constant vector reads inverted vs A·x=b, but is a faithful CPU↔GPU transliteration of cpu_backend.cpp (parity-traceable; rename in lock-step only).

### remaining LOW (cryptic locals / params, convention drift, one-off abbreviations)
- [6.4][LOW] src/core/qpadm/model_search.cpp:103,115 — `sr`/`lr` (small/large results) next to clear `small`/`large` partition locals.
- [6.1][LOW] src/core/qpadm/nested_models.cpp:16-17 — matrix param `w` (a single letter for the row-major data matrix).
- [6.1][LOW] src/core/qpadm/ranktest.cpp:90 — `xr`/`cr` (reduced F4Blocks / JackknifeCov) next to descriptive `nl_red`/`m_red`.
- [6.1][LOW] src/core/qpadm/ranktest.cpp:22-23 — accumulator flag `any` for "a surviving weight was seen".
- [6.1][LOW] src/core/internal/launch_config.hpp:82,89 — `cdiv`'s divisor param `b` vs the sibling `grid_for`'s `block` for the same concept.
- [6.3][LOW] src/core/internal/decode_af.hpp:88-89,130 — `std::int64_t ac/an` vs `long ac/an` for the same accumulators in adjacent functions.
- [6.3][LOW] src/core/domain/block_partition_rule.cpp:42 — local `m` (SNP count) vs sibling `block_ranges`/`MatView::M` uppercase `M`.
- [6.3][LOW] src/core/fstats/f2_blocks_multigpu.cpp:351-352 — `free_vram` (carries memory-class suffix) vs `free_host` (drops it) on the same two lines.
- [6.3][LOW] src/device/backend.hpp:73 — field `vpair` (lowercase) vs its own docs `Vpair` vs neighbor field `P`.
- [6.3][LOW] src/device/backend.hpp:228-232 vs 380-383 — `q`/`v`/`n` fields vs `Q`/`V`/`N` params (documented intentional split; flagged for completeness).
- [6.1][LOW] src/device/cpu/cpu_backend.cpp:454,705-710,721-724 — `sh`/`wln`/`wld`/`wbn` jackknife/loo scratch accumulators.
- [6.1][LOW] src/device/cpu/cpu_backend.cpp:561-566 — `dd`/`cd` (dof-diff/chisq-diff) vs the `rd_dofdiff`/`rd_chisqdiff` fields they feed.
- [6.3][LOW] src/device/cuda/check.cuh:214,227,233,241 — `STEPPE_CUDA_*` prefixed vs bare `CUBLAS_CHECK`/`CUSOLVER_CHECK` (documented spike carry-over).
- [6.4][LOW] src/device/cuda/cuda_backend.cu:1601,1605 — `wc` (weight/chisq scratch size) vs the descriptive `als`/`als_*` siblings.
- [6.1][LOW] src/device/cuda/cuda_backend.cu:1598,1608,1616-1621 — scratch-layout dims `t`/`a`/`bb`/`rr` (borderline; tightly commented).
- [6.1][LOW] src/device/cuda/device_buffer.cuh:78,81 — move-source param `o` (header prose already calls it `other`).
- [6.3][LOW] src/device/cuda/device_buffer.cuh:12 vs 78,81 — doc says `other`, code says `o` (covered by the rename above).
- [6.1][LOW] src/device/cuda/device_f2_blocks.cu:51,79 — scope-restore guard `struct G { int d; }` (single-letter type/member; mirrors p2p_combine.cu).
- [6.1][LOW] src/device/cuda/f2_blocks_out.cu:90 — same `struct G { int d; }` device-restore guard.
- [6.1][LOW] src/device/cuda/f2_blocks_out.cu:48-49 — `pread_all`'s diagnostic label param `const char* what`.
- [6.4][LOW] src/device/cuda/f2_block_kernel.cu:116 (`Pl`) vs :160 (`Pp`) — two ad-hoc P-widening suffix conventions in one TU.
- [6.4][LOW] src/device/cuda/f2_block_kernel.cu:118 (`sidx`) vs :162-163 (`si`/`sj`) — `s`-prefix means "stacked index" then "size_t index".
- [6.4][LOW] src/device/cuda/f2_block_kernel.cu:339 (`ct`), :352 (`Mi`), :166 (`vp`) — short tight-scope locals (acceptable; no action).
- [6.1][LOW] src/device/cuda/f2_block_kernel.cu:102-108,151-155 — spec-aligned `Q`/`V`/`N`/`S`/`G`/`R`/`P`/`M`/`hc` (intentional; no action).
- [6.4][LOW] src/device/cuda/f2_blocks_kernel.cu:94-95,100-101 — `dSqsq`/`dShc`/`sSqsq`/`sShc` packed abbreviations vs the spelled-out sibling `dQidx`.
- [6.3][LOW] src/device/cuda/f2_blocks_kernel.cu:84-95 vs 141-160 — `Pl`/`Psp` family vs `Pp`/`gSlab`/`rSlab` family across the two kernels.
- [6.3][LOW] src/device/cuda/f2_blocks_kernel.cu:69-78 vs 171-176 — kernel params drop the `d`-prefix the launch wrappers carry (borderline host/device boundary convention).
- [6.3][LOW] src/device/cuda/p2p_combine.cu:106-107,245-246 — `dResult_f2`/`dResult_vp` (camelCase + `d`-prefix) vs `result_f2`/`result_vp` (snake) for the same buffers.
- [6.3][LOW] src/device/cuda/stream.hpp:88,160 — destroy-status `cudaError_t` named `e` (Stream) vs `err` (Event).
- [6.1][LOW] src/device/tier_select.hpp:86 — local `t` for the tile width (sibling kernel uses `max_tile_z`; `t` elsewhere is a byte size).
- [6.1][LOW] src/io/eigenstrat_format.cpp:49,84,91-92 — `ints` (generic) for the two header counts (n_ind, n_snp).
- [6.4][LOW] src/io/filter/mind_prepass.cpp:39,48,50 — `nm` for the per-sample non-missing count (feeds `out.nonmissing`).
- [6.4][LOW] src/io/ind_reader.cpp:54 — `std::istringstream ls(line)` (`ls` reads as "list"/"left-shift").
- [6.1][LOW] src/io/snp_reader.cpp:132 — `SnpTable t;` function-scope result object referenced across ~50 lines.

## 4. Cross-cutting patterns

1. **Abbreviation drift of one concept within a file/TU** (the single largest theme, spanning 6.3 and 6.4): `vp` vs `vpair` (p2p_combine, f2_blocks_out), `_rec` vs `_record` (decode_af_kernel, geno_reader), `n_ind` vs `n_individuals` (geno_reader), `e` vs `err` (stream.hpp), `int64_t` vs `long` (decode_af.hpp), `STEPPE_*` vs bare check macros. Almost all are one-side-only renames with no behavior change.
2. **One single-letter symbol carrying two unrelated meanings across functions in the same TU** (the `M`/`m` and `t`/`s` collisions): cuda_backend.cu and cpu_backend.cpp both overload `M` (SNP count vs widened nl·nr); small_linalg.hpp overloads `t` (tangent vs swap temp). Suggested direction codebase-wide: name the widened model-dim `mz`/`m_sz` and reserve `M` for the SNP axis.
3. **The jackknife SE scale `s`** appears opaque on BOTH the CPU/GPU host path (cuda_backend.cu:2381) and the kernel launcher (qpadm_fit_kernels.cu) — a coordinated rename to `jackknife_scale`/`se_scale` would fix both.
4. **Two-line scope-restore RAII guard `struct G { int d; }`** is duplicated verbatim in device_f2_blocks.cu and f2_blocks_out.cu (and referenced by p2p_combine.cu) — a shared `DeviceGuard` helper would resolve all three single-letter type/member notes at once.
5. **Count/extent named like a verb or boolean**: `fit` (vram_budget), `free_` (block_sink), `any` (ranktest) — noun-ify to state the quantity/predicate.

No correctness/overflow/UB/race issues surfaced under naming, as expected for this group. Several reviewers explicitly (and correctly) declined to flag parity-load-bearing AT2/LA symbols and documented intentional splits (e.g. the lowercase-field / uppercase-contract `q`/`v`/`n` ↔ `Q`/`V`/`N` and the RHS/LHS inversion that mirrors the R `solve(rhs, lhs)` idiom).

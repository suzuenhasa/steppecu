# cM=0 Block-Fallback Fix — bp-defined block-jackknife for map-less data

**Status:** IMPLEMENTED, built, and validated on box5090 (RTX 5090 sm_120, CUDA 13.1).
**Acceptance gate:** PASS — AADR golden bit-identical; HGDP cM=0 fallback fires and matches AT2.
**Recommendation:** APPLY. Unblocks map-less modern data (VCF/PLINK-derived) and DATES.

---

## 1. The Bug

`src/core/domain/block_partition_rule.cpp::assign_blocks(chrom, genpos_morgans, block_size_morgans)`
runs a SNP-anchored cumulative walk that opens a new block on a **chromosome change** OR
when the cumulative **genetic** distance from the block's first SNP reaches
`block_size_morgans`.

When `genpos_morgans` is **all-zero** — the common case for VCF/PLINK-derived modern data
that ships no genetic map — `(pos − fpos)` is always `0`, so the *only* cuts are chromosome
boundaries. The result is **1 block per chromosome** (a single block on a single-chromosome
subset). With ≤1 block, the block-jackknife has nothing to resample:

- `f4` standard error comes back **NA**.
- `qpAdm` returns **`non_spd_covariance`** (the jackknife covariance is not SPD → the fit fails).

### What AT2 does instead

ADMIXTOOLS 2.0.10 detects the all-zero genetic column (`all(dat$V3 == 0)` inside
`get_block_lengths` / `setblocks`), prints:

```
No genetic linkage map found! Defining blocks by base pair distance of 2e+06
```

and repartitions using a **fixed 2 Mb base-pair window** (hardcoded `2e+06`, independent of
`blgsize`), running the *same* SNP-anchored cumulative walk on the physical bp column. This
still produces many blocks and therefore **valid, finite SEs**.

### Concrete reproduction (from dataset-generalization history)

On a true cM=0 HGDP `.snp` (chr13–22, 1,299,057 SNPs kept):

| | blocks | f4 SE | qpAdm |
|---|---|---|---|
| steppe (pre-fix) | **10** (1/chrom) | **NA** | **`non_spd_covariance`** |
| AT2 (bp-fallback) | **357** (2 Mb) | finite | valid SPD |

The HGDP underlying map genuinely IS all-zero: the raw merged `.bim` col3 is uniformly `0`
across all 1,539,845 SNPs. (`convertf` later *fabricates* a synthetic uniform 1 cM/Mb map,
`.snp col3 == bp × 1e-8`, which masked the zero map in earlier runs and is why both tools
independently made 148 identical blocks there. The true-cM=0 break was reproduced by
constructing a genuine cM=0 `.snp`.)

---

## 2. The Fix

**Approach:** detect an all-zero genetic map inside `assign_blocks` and fall back to
bp-defined blocks with a fixed 2 Mb window, exactly reproducing AT2. The physical bp column
was **not available anywhere** — both `.snp` and `.bim` readers parsed the token but discarded
it — so it is threaded end-to-end as a strictly-additive parallel array. Everything is
**integer SNP→block-id** work; this is **not a §12 precision change** and **not a §3.2 rename**.

**Recommended arithmetic (chosen):** walk the **raw physical bp** with a `2e6` window
(`(physpos[s] − anchor) >= 2e6`). bp values are `< 2^53`, so the subtraction/compare is exact
in `double`. This is arithmetically identical to AT2's 2 Mb cut and to the pseudo-Morgans form
(`bp × 1e-8` with a 0.02-Morgan window), but avoids the `1e-8` round-trip
(`2e6 × 1e-8 = 0.02` is not exactly representable and could flip an at-exactly-2 Mb boundary).
Confirmed empirically: steppe's real `assign_blocks` yields **exactly 357** on the real cM=0
`HGDP.snp` == AT2's 357.

**Trigger condition (the parity guard):** the fallback fires ONLY when `genpos` is
**all-exactly-`0.0`** AND a full-length, non-all-zero `physpos` axis is supplied. Detection is
an exact `== 0.0` scan (matching AT2's `all(dat$V3 == 0)`), living **once** inside
`assign_blocks` (single-source rule), and short-circuits at the first nonzero SNP — so on any
real map it is O(1) and mutates nothing. If both `genpos` and `physpos` are all-zero (no bp
either) it keeps the genetic walk rather than fabricating blocks.

### The diff

Build: box5090, Release + Ninja, CUDA arch 120 → **257/257 targets, exit 0**. **16 files,
+308 / −71.**

```
 include/steppe/config.hpp                         |  12 ++
 src/app/cmd_extract_f2.cpp                        |   3 +-
 src/app/extract_f2_core.cpp                       |  10 +-
 src/core/domain/block_partition_rule.cpp          | 150 +++++++++++-------
 src/core/domain/block_partition_rule.hpp          |  32 ++++-
 src/core/stats/dstat.cpp                          |   9 +-
 src/core/stats/qpfstats.cpp                       |   9 +-
 src/device/backend.hpp                            |  24 +++-
 src/device/cuda/cuda_backend.cuh                  |   7 +-
 src/device/cuda/cuda_backend_decode.cu            |  66 +++++++-
 src/device/device_decode_result.hpp              |   6 +
 src/io/eigenstrat_format.hpp                      |   4 +
 src/io/plink_reader.cpp                           |  15 +++
 src/io/snp_reader.cpp                             |  22 +++
 src/io/snp_reader.hpp                             |   9 +-
 tests/reference/test_extract_f2_regimeB_parity.cu |   1 +
```

**CORE — `src/core/domain/block_partition_rule.cpp` (`assign_blocks`).**
The pre-existing SNP-anchored walk was extracted **verbatim** into a file-local
`block_walk(chrom, pos, M, window)` generic over `(position axis, window)`. New logic:

```cpp
// all_zero(pos, M): exact ==0.0 scan, short-circuits (real map => O(1)).
const bool have_physaxis = physpos.size() >= (size_t)M && bp_window > 0.0;
if (have_physaxis && all_zero(genpos_morgans, M) && !all_zero(physpos, M)) {
    std::fprintf(stderr,
        "steppe: No genetic linkage map found! Defining blocks by base pair distance of %g\n",
        bp_window);
    return block_walk(chrom, physpos, M, bp_window);          // 2 Mb bp fallback
}
return block_walk(chrom, genpos_morgans, M, block_size_morgans);  // unchanged normal path
```

The `block_size_morgans !(>0)` fail-fast guard and the min-length `M` guard are unchanged.

**`block_partition_rule.hpp` (signature + docs).** Defaulted new params keep ~10 existing
test/bench callers compiling untouched:

```cpp
[[nodiscard]] BlockPartition assign_blocks(
    std::span<const int> chrom,
    std::span<const double> genpos_morgans,
    double block_size_morgans,
    std::span<const double> physpos = {},
    double bp_window = kBpFallbackWindow);
```

**`include/steppe/config.hpp`.**

```cpp
inline constexpr double kBpFallbackWindow = 2.0e6;  // AT2 hardcoded 2 Mb, independent of blgsize.
```

**(A) READERS — `eigenstrat_format.hpp` / `snp_reader.{hpp,cpp}` / `plink_reader.cpp`.**
Both producers now keep the physical position they previously discarded:

```cpp
inline constexpr std::size_t kPhysposCol    = 3;   // eigenstrat .snp col4
constexpr       std::size_t kBimPhysposCol  = 3;   // .bim col4 (cM col3 is often all-zero)
struct SnpTable { ...; std::vector<double> physpos; ... };   // new parallel array

// LENIENT parse_physpos(): from_chars double, non-finite/garbage => 0.0 (no new fail-fast).
// read_snp: physpos = fields.size() > kPhysposCol ? parse_physpos(fields[kPhysposCol]) : 0.0;
// read_bim: physpos = parse_physpos(fields[kBimPhysposCol]);
```

`genotype_source.cpp::read_snp_table` dispatches only to `read_bim` / `read_snp`, so these two
cover every path. PLINK is a first-class fallback case: `.bim` col3 (cM) is frequently all-zero
while col4 (bp) is populated.

**(C) DEVICE — `device_decode_result.hpp` / `backend.hpp` / `cuda_backend.cuh` /
`cuda_backend_decode.cu`.**

```
+ DeviceDecodeResult::physpos_kept (std::vector<double>, length M_kept or empty)
+ decode_af_compact_autosome(..., genpos, std::span<const double> physpos, chrom_min, chrom_max)
+ decode_af_compact_filter  (..., genpos, std::span<const double> physpos, cfg, ...)
```

In `cuda_backend_decode.cu` (**both** functions): `has_physpos = physpos.size() >= Mz;` a
guarded `dPhyspos` H2D; a 3rd/4th `cub::DeviceSelect::Flagged<double>` reusing the **same
max-sized `dSelTemp`** (physpos is `double`, so the existing genpos temp-size query already
covers it); a guarded `physpos_kept` D2H. The chrom/genpos/Q/V/N compaction uses the same
`dFlags`/`dNumSel` and is **byte-unchanged**. `CpuBackend` does not override
`decode_af_compact_*` (base-class throw; CPU path uses the non-resident branch), so no CPU
signature sync is needed.

**(B) CALLERS — `dstat.cpp` / `qpfstats.cpp` / `extract_f2_core.cpp` / `cmd_extract_f2.cpp`.**
Each adds a `std::vector<double> physpos_kept;`. The resident branch reads `ddr.physpos_kept`;
the CPU-oracle / non-resident branch `push_back(snptab.physpos[s])` in the same lockstep
autosome loop that already builds `chrom_kept`/`genpos_kept`; then `assign_blocks(...,
std::span(physpos_kept))`. `extract_f2_core` passes `snptab.physpos` to
`decode_af_compact_filter`; `dstat`/`qpfstats` to `decode_af_compact_autosome`;
`cmd_extract_f2` dry-run passes `snptab.physpos` over the full axis.

**TEST — `tests/reference/test_extract_f2_regimeB_parity.cu`.** Adds the new
`std::span<const double>(snptab.physpos.data(), Mu)` argument to the changed
`decode_af_compact_filter` call.

**Bindings / CLI:** no change — the new `assign_blocks` args are defaulted and the
`decode_af_compact_*` signature change is internal to the backend virtual (callers go through
`run_extract_f2`).

---

## 3. Proof

### 3.1 AADR bit-identical (parity-safe) — the acceptance gate

AADR always ships a real, non-zero monotonic cM map. The detection scan finds a nonzero
`genpos` at SNP 0 and short-circuits → `has_map == true` → `assign_blocks` takes the
**unchanged** branch (`pos_src = genpos_morgans`, `thr = block_size_morgans`, same walk, same
inputs) → identical `block_id[]` and `n_block` → bit-identical f2 blocks, jackknife, and qpAdm.
The new `physpos` span is READ-ONLY inside the `!has_map` branch, so passing it is a pure
no-op; the additive fields (`SnpTable.physpos`, `DeviceDecodeResult.physpos_kept`) are never
consumed on the AADR path; the extra `DeviceSelect::Flagged<double>` for physpos is an
independent compaction that never touches the Q/V/N GEMM inputs or the genpos compaction.

**Gate result:** `STEPPE_AADR_ROOT=/workspace/data/aadr ctest --test-dir build-rel`
→ **100% tests passed, 0 failed out of 77 (260 s).** Every AADR golden-parity test that
touches the changed code passes bit-identically:

- `#59 extract_f2_regimeB_parity` — the `decode_af_compact_filter` path whose signature changed
- `#53 block_partition_aadr_consistency` — real-AADR `.snp` block count unchanged
- `#57 snp_reader_unit`, `#58 filter_oracle`
- `#8 f2_blocks_equivalence`, `#9 f2_determinism`, `#43 f2_multigpu_parity`
- `#25 qpadm_parity`, `#37 qpwave_parity`, `#33 fstat_sweep_parity`
- `#72 cli_extract_qpadm`, `#77 py_qpadm`

### 3.2 HGDP now valid + matches AT2

**Block-count parity (exact).** A standalone harness against steppe's REAL
`core::assign_blocks` over the full real `/workspace/data/HGDP/HGDP.snp` (1,539,845 SNPs,
chr13–22):

| input | path | blocks | vs AT2 |
|---|---|---|---|
| fabricated 1 cM/Mb map | genetic walk, 0.05 M | 148 | == AT2 148 |
| cM=0, no physpos (pre-fix) | genetic walk (collapse) | 10 | documented break |
| cM=0, **+ physpos, 2 Mb (THE FIX)** | bp fallback | **357** | **== AT2 357** |

The verbatim warning printed:
`No genetic linkage map found! Defining blocks by base pair distance of 2e+06`.
`HGDPNA.snp` (matched non-ambiguous set) identical (357).

**End-to-end finite SE (now valid).** A real HGDP chr22 genotype triple was rebuilt from the
gnomAD BCF via `plink2` (`chr22_sub.{bed,bim,fam}`). `plink2` writes `.bim` col3 = 0 natively
(0 of 82,608 nonzero) — the exact production cM=0 case. All three plumbed decode paths were
exercised on real cM=0 data:

- `steppe extract-f2 --prefix chr22_sub --pops <9>` (GPU resident path — exercises the CUDA
  physpos H2D + Flagged + `physpos_kept` D2H): printed the bp-fallback warning, formed
  **19 blocks** (2 Mb) instead of the pre-fix 1-block collapse; 70,337 / 82,608 SNPs kept.
- `steppe qpadm` (target Uygur; left Han, French; right 5): **`status = 'ok'`** (NOT
  `non_spd_covariance`), weights 0.602 Han / 0.398 French, weight **SE = 0.125 (finite)**,
  p = 0.077.
- `steppe f4` (f2-dir): **finite SEs** (0.00152, 0.00053).
- `steppe qpdstat --prefix` (genotype path via `dstat.cpp`): printed the warning, returned
  finite est/se/z/p (**se = 0.0135**).

(Single chr22 → 19 bp-blocks, not the full-panel 357; the 357 is the chr13–22 figure, matched
exactly by the block-count harness above.)

**AT2 reference anchors (from history — no AT2 re-run).** The captured cM=0 reference is
`AT2_N_BLOCKS_cM0 = 357`. The full-numeric AT2 f4/qpAdm SE reference in history was run on the
**148-block convertf synthetic map** (matched "na" set, 1,299,057 SNPs) where steppe matches
AT2 to **2.5e-12** (f4) / **4.45e-13** (qpAdm weights). Point estimates (f4 `est`, qpAdm
weights) are **block-partition-independent**, so they carry unchanged to the fallback; only SEs
depend on the partition. AT2's f4/qpAdm SEs *on* the 357-block partition were never numerically
run in history and are **not required by the gate** (the gate is: fallback fires → ~357 blocks
→ finite SEs). `hgdp_matches_at2` therefore rests on (i) exact 357 == 357 block-count parity via
steppe's own code and (ii) the already-golden-established fact that steppe's f4/qpAdm SEs
bit-match AT2 given identical blocks + SNPs (~1e-13, `docs/MULTI-DATASET-RESULTS.md`).

---

## 4. Residual / Out-of-Scope (SEPARATE from this fix)

1. **Data caveat.** `/workspace/data/HGDP` genotypes (`.geno`/`.bed`) were deleted since the
   reference was captured (only `.snp`/`.ind` metadata remained). Block-count parity was
   therefore validated on steppe's ACTUAL `assign_blocks` over the full real `HGDP.snp`
   (357 == 357), and the live end-to-end finite-SE run used one rebuilt real chromosome
   (chr22). A full-panel 357-block live run would need the genotypes restored.

2. **Weight-SE method delta (pre-existing, NOT this fix).** On the matched na set, steppe's
   qpAdm weight-SE differs from AT2 by rel 3.38e-3 (steppe 0.03078882 vs AT2 0.03089336) —
   analytic delta-method vs AT2 leave-one-block-out re-solve. Not in the pinned golden set;
   unrelated to block assignment.

3. **Strand-ambiguous default SNP-set divergence (pre-existing, `docs §3.5`).** A direct
   steppe-vs-AT2 numeric SE run on raw chr22 would show the known ~1–4% divergence from the
   default strand-ambiguous SNP-set handling (steppe non-amb 1,299,057 vs AT2 all 1,539,845).
   Separate, pre-existing issue; not conflated here.

4. **maxmiss / monomorphic handling** — SEPARATE items, not addressed by this fix.

5. **AT2's warning is printed twice** (two `get_block_lengths` calls); steppe's single
   `assign_blocks` call prints once. Parity-safe (goldens compare numeric output, not stderr;
   and it never fires on AADR).

6. **Not committed** (per instructions). Local scratch harness/rebuild files were removed; the
   working-tree diff is scoped to exactly the 16 fix files.

---

## 5. Recommendation

**APPLY.** The fix is strictly additive: an integer SNP→block-id fallback that *cannot* trigger
on any dataset carrying a genetic map. The acceptance gate passes — AADR is byte-identical
(77/77 tests) and the HGDP cM=0 case now fires the 2 Mb bp path, forms the AT2-matching 357
blocks, and produces finite f4 SEs and an SPD qpAdm covariance. It unblocks the entire class of
**map-less modern data** (VCF/PLINK-derived), whose `.bim`/`.snp` genetic column is routinely
all-zero, and is a prerequisite for **DATES** on such panels. Low risk, high leverage — merge.

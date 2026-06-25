# Phase 2 — Multi-format genotype reader (transpose-on-read)

Status: DESIGN (review before build). No code in this document; it is a
reviewable plan. Every load-bearing claim cites `file:line` against the tree at
the head of `phase2-fit-engine`. Where `architecture.md` and `src/` disagree,
`src/` is ground truth (per the project mandate) and the staleness is flagged.

---

## 1. Goal and the spec-vs-code gap

### 1.1 The goal

steppe must read the standard population-genetics genotype formats —
PACKEDANCESTRYMAP / `GENO` (SNP-major), EIGENSTRAT ASCII `.geno`, PLINK
`.bed`/`.bim`/`.fam`, and legacy ANCESTRYMAP — and feed them into the SAME
genotype pipeline that already serves the real AADR v66 TGENO data: decode →
per-pop allele frequency (Q/V/N) → f2 / qpDstat / DATES / qpfstats.

### 1.2 The seam every format must satisfy

The genotype front-end is byte-identical across all four tools. The chain is

```
io::GenoReader reader(geno)
  -> read_ind(ind, sel, n_present)        # io::IndPartition (pop selection + row order)
  -> read_snp(snp, SIZE_MAX)              # io::SnpTable (chrom/genpos/ref/alt)
  -> reader.read_tile(part, 0, M0)        # io::GenotypeTile   <-- THE CONTRACT
  -> io::detect_sample_ploidy(tile)       # per-sample ploidy (extract_f2 Auto only)
  -> backend.decode_af(view)              # or decode_af_compact_filter / _autosome
  -> assign_blocks / f2 / D / dates / qpfstats
```

This is identical at `src/app/extract_f2_core.cpp:92-99,125` (extract_f2),
`src/core/stats/dstat.cpp` (qpDstat-B), `src/core/stats/dates.cpp`, and
`src/app/qpfstats.cpp`. The ONLY object that crosses from the `io` leaf to the
backend is the `io::GenotypeTile` (`src/io/genotype_tile.hpp:47`), filled into a
CUDA-free `DecodeTileView` at `extract_f2_core.cpp:139-147`. So the entire
surface a new format must satisfy reduces to: **produce a `GenotypeTile` in the
canonical individual-major raw-value-2-bit packing.**

### 1.3 The canonical tile is HARD individual-major

`src/io/genotype_tile.hpp:40-46`: `packed` length is
`n_individuals * bytes_per_record`; gathered individual `g` occupies
`packed[g*bytes_per_record .. (g+1)*bytes_per_record)`, and SNP `s` of that
individual is the 2-bit code at byte `s/4`, position `s%4`, MSB-first
(`eigenstrat_format::code_in_byte`, `src/io/eigenstrat_format.hpp:160-163`). The
same packing is re-promised on the device view (`src/device/backend.hpp:371-375`).

### 1.4 The gap

`io::GenoReader::read_tile` THROWS on any non-TGENO format
(`src/io/geno_reader.cpp:81-85`): *"M1 decode targets TGENO (individual-major);
this file is GENO (SNP-major PACKEDANCESTRYMAP)."* PACKEDANCESTRYMAP magic IS
recognized at construction — `parse_geno_header` already derives the SNP-major
geometry `n_records=n_snp`, `bytes_per_record=max(48, ceil(n_ind/4))`
(`src/io/eigenstrat_format.cpp:114-119`, the EIGENSOFT rlen floor) — but
`read_tile` refuses it. So for PA the reader already knows nind / nsnp / stride /
record-axis; **only the gather + transpose is missing.** PLINK, EIGENSTRAT
ASCII, and ANCESTRYMAP have no reader at all: a grep for `plink`/`.bed`/`.bim`/
`.fam`/`0x6c1b` over `src/` finds only `filter_decision.hpp` (an unrelated hit),
no `plink_bed.cpp`.

> **Stale-doc flag.** `architecture.md` §5 (line ~251) still describes the OLD
> binary code mapping (`00->0,10->1,11->2,01->NA`) and §11.1 (~line 931) names a
> `plink_bed.cpp` that was never built. `src/io/eigenstrat_format.hpp:23-26`
> explicitly flags the binary mapping as WRONG ("mis-decodes; the raw-value
> mapping reproduces the oracle exactly"). This design follows `src/`. Both
> architecture.md lines should be updated as a follow-up; they do NOT drive this
> design.

---

## 2. The architecture: TRANSPOSE-ON-READ

### 2.1 The thesis

A new SNP-major reader gathers its raw bytes, then transposes them on-device
into the canonical individual-major `GenotypeTile`. After the transpose, ploidy
detection, `decode_af`, the regime-B filter, f2, D, DATES, and qpfstats are
**literally unchanged** — they only ever see a `GenotypeTile`.

### 2.2 Why transpose-on-read is correct, not merely convenient

The alternative — a native SNP-major *decode* path — must re-derive the byte
address `packed[g*bytes_per_record + s/4]`, which is hard-coded identically in
THREE places that would all fork:

- the GPU decode kernel: `decode_af_kernel.cu:127-134` reads
  `packed[g*bytes_per_record + byte_in_record]` with the warp riding the SNP axis
  for coalescing — a SNP-major buffer inverts which axis lives in a byte and
  which strides records, breaking the coalescing argument
  (`decode_af_kernel.cu:22-29,76-94`) AND the segmented reduction over
  `pop_offsets`;
- the CPU oracle: `cpu_backend.cpp` walks the SAME individual-major address
  (`tile.packed[g*bytes_per_record + byte_in_rec]`) via the SHARED core
  primitives (`genotype_code`/`accumulate_genotype_ploidy`/`finalize_af_counts`)
  — forking it breaks the bit-exact oracle gate pinned by
  `tests/reference/test_decode_equivalence.cu`;
- the host ploidy detector: `ploidy_detect.cpp:30,34` indexes
  `tile.packed.data() + g*tile.bytes_per_record` then `s/kCodesPerByte` — it
  needs a per-individual SNP prefix, impossible to express SNP-major without
  itself transposing.

A SNP-major decode would additionally fork `decode_af_compact_autosome` and
`decode_af_compact_filter` (`backend.hpp:782,824`, both built on
`decode_af_resident`, `cuda_backend.cu:1197`), every `DecodeTileView` field's
meaning, and require a parallel parity test per tool. That is 5+ kernels/paths ×
2 backends.

**The structural asymmetry is the argument.** Decode is *packing*-coupled (not
format-coupled) and fans out to 5 sites × 2 backends; the transpose is a single
localized op with three existing GPU precedents (§2.5). Transpose-on-read pays
ONE on-device transpose at read time and leaves the entire individual-major
contract untouched.

> **Precision note (honors §12).** Transpose, gather, encoding-map, and ploidy
> detection are pure INTEGER / bit operations (2-bit extract + remap + equality
> compare). The emulated-FP64-default / native-FP64-carve-out precision policy is
> **matmul-only** and does NOT apply here. The only FP64 in the whole path is the
> single `AC/N` divide in `finalize_af_counts` (decode), unchanged. Bit-exactness
> here is by construction, and the only correctness bar is exact equality.

### 2.3 The per-format plug-in shape

Each format is a small plug-in with four steps:

1. **Parse header / counts.** PA: already done (`eigenstrat_format.cpp:114-119`).
   EIGENSTRAT ASCII: no header — counts from `.snp`/`.ind` line counts. PLINK:
   3-byte magic check + counts from `.fam`/`.bim` line counts. ANCESTRYMAP:
   counts from `.snp`/`.ind`.
2. **Gather (host).** For a SNP range, read the SNP-major records
   (`header_bytes + snp*bytes_per_record` per SNP) for the selected individuals
   into a compact host SNP-major staging buffer. This mirrors the existing
   per-record seek/read loop (`geno_reader.cpp:202-213`) with the axis swapped,
   keeping the same fail-fast guards (records-present bound, checked multiply,
   short-read throw; `geno_reader.cpp:121-213`). The selection + pop-contiguous
   reorder is applied here (or folded into the transpose; see §2.4).
3. **Encoding-map.** Normalize the format's native 2-bit / ASCII code to the
   canonical raw-value `{0,1,2,3}` (§3.2). PA is identity; EIGENSTRAT/ANCESTRYMAP
   are an ASCII-digit map; PLINK is a 4-entry LUT plus a bit-order flip.
4. **Transpose (on-device).** Produce the canonical individual-major `packed`.

### 2.4 Where the transpose runs (layering)

The `io` leaf is CUDA-free and MUST stay so (`genotype_tile.hpp:23-24`,
`geno_reader.cpp:8-9`; architecture.md §4 — CUDA is PRIVATE to `steppe_device`).
So `io` cannot call a transpose kernel. The wiring is:

- `io` gathers + encoding-normalizes the selected individuals into a compact
  host SNP-major buffer (still SNP-major, but already canonical codes);
- the **app layer** (the only place `io` meets `device`, architecture.md §239)
  hands that buffer + (`n_ind`, `n_snp`, selected-individual gather list) to the
  backend through a NEW CUDA-free virtual on `ComputeBackend`, e.g.
  `transpose_to_canonical(SnpMajorTileView) -> {device-resident or host
  GenotypeTile}`. The base throws (the established pattern — `decode_af_compact_*`
  base throws at `backend.hpp:786,824`; `CpuBackend` is oracle-only);
- the CUDA backend H2D-uploads the SNP-major bytes and runs a 2-bit-granular
  transpose kernel: input element (individual `i`, SNP `s`) = code at PA-byte
  `s*bytes_per_record_PA + i/4`, pos `i%4` (`code_in_byte`); output packed into
  individual-major byte `g*ceil(n_snp/4) + s/4`, pos `s%4`. The SAME pass does
  the individual GATHER (select + pop-contiguous reorder, output column `g` ->
  source individual row from the `IndPartition`) and the encoding remap;
- the `CpuBackend` transpose is the oracle-only host nested loop (parity anchor).

> **Correction to one grounding framing.** The "GPU transpose machinery" inside
> `decode_af_kernel.cu:109-167` (the padded `__shared__` tQ/tV/tN tiles) is the
> *output* Q/V/N column-major store-coalescing transpose, NOT an input
> SNP-major->individual-major transpose. There is currently **no input-layout
> transpose kernel** anywhere. The transpose-on-read kernel is NEW machinery; the
> existing coalescing tiles are a *coding precedent* (the NVIDIA "Efficient Matrix
> Transpose" +1-pad bank-conflict recipe, already in this TU) and
> `transpose_small_kernel` (`qpadm_fit_kernels.cu:1105-1114`,
> `launch_transpose_small`) is a standalone element transpose to generalize from
> — but neither is reusable as-is. This is the honest distinction between
> "precedent exists" and "primitive exists."

### 2.5 What is reused vs new

**Reused (zero change):**
- the `GenotypeTile` + `DecodeTileView` contract (`genotype_tile.hpp`,
  `backend.hpp:362-400`);
- `decode_af` and both compact paths (`backend.hpp:782,824`;
  `cuda_backend.cu:1197-1266`);
- the `CpuBackend` decode oracle + the shared core decode primitives;
- `detect_sample_ploidy` (the host path stays as the CpuBackend oracle);
- `assign_blocks` / f2 / qpadm / dstat / dates / qpfstats;
- the format constants + BOTH header strides (`eigenstrat_format.hpp:51-163`,
  `eigenstrat_format.cpp:110-119`);
- `read_snp` / `read_ind` (`src/io/snp_reader.cpp`, `src/io/ind_reader.cpp`) —
  format-agnostic for the EIGENSTRAT/PA/ANCESTRYMAP family (they all share the
  `.snp`/`.ind` text files); only PLINK needs new `.bim`/`.fam` parsers;
- the transpose/coalescing *recipes* (precedent only, per §2.4).

**New (small, localized):**
- per-format header/count parse extensions (PLINK `.bed` magic + `.bim`/`.fam`;
  EIGENSTRAT/ANCESTRYMAP count-from-line);
- the SNP-major host gather + encoding-normalize;
- the on-device 2-bit transpose+gather+encoding-map kernel + its CUDA-free
  `ComputeBackend` virtual + the `CpuBackend` host-loop oracle;
- a `FormatReader` dispatch in the app wiring.

---

## 3. Per-format plan and phasing

### 3.1 The canonical 2-bit code (the normalization target)

`src/io/eigenstrat_format.hpp:23-26,78-88` + `core/internal/decode_af.hpp`:
RAW-VALUE mapping — code `0`->0 ref-allele copies, `1`->1 (het, `=kHetCode`),
`2`->2, `3`->MISSING (`=kMissingCode`). MSB-first extraction
(`(byte>>(6-2*(k%4)))&3`, `code_in_byte`). Decode folds non-missing as
`AC += code/(3.0-ploidy)`, `N += ploidy`.

### 3.2 Encoding normalization table (the load-bearing per-format map)

| Format | Axis | Bit order | Native code -> canonical | Source of counts |
|---|---|---|---|---|
| **PACKEDANCESTRYMAP / GENO** | SNP-major | MSB-first (same as TGENO) | **IDENTITY** (0/1/2 copies, 3 missing) | header (`eigenstrat_format.cpp:114-119`) |
| **EIGENSTRAT ASCII `.geno`** | SNP-major (line = SNP, char-col = individual) | n/a (ASCII char/genotype) | `'0'/'1'/'2'` -> 0/1/2; `'9'` -> 3 | `.snp`/`.ind` line counts |
| **PLINK `.bed`** | SNP-major (magic `0x6c 0x1b 0x01`) | **LSB-first** (sample 0 in bits 1-0) | LUT `{00->2, 01->3 missing, 10->1 het, 11->0}` + bit-order flip to MSB-first | `.fam`/`.bim` line counts |
| **ANCESTRYMAP (legacy unpacked)** | per-record ASCII | n/a (unpacked text) | same as EIGENSTRAT char map | `.snp`/`.ind` line counts |

PLINK is the only format whose code mapping AND bit order differ from steppe's
canonical convention; it is therefore the highest-risk plug-in (§8.4). PLINK
`.bim` is 6 cols (chrom, id, genpos-Morgans, physpos, A1, A2); `.fam` is 6 cols
(FID, IID, pat, mat, sex, pheno) — neither matches the EIGENSTRAT `.snp`/`.ind`
layout, so `read_snp`/`read_ind` do NOT apply to PLINK.

### 3.3 Phasing (PA first — strictly increasing risk, decreasing cross-check)

The phases are ordered so encoding complexity strictly INCREASES
(identity -> ASCII char -> 2-bit-LUT + bit-flip) while cross-check strength
strictly DECREASES (PA has an exact same-axes box transcode; PLINK has no
on-box golden and leans on the auto-covering downstream goldens once the
PA-validated transpose primitive is trusted).

- **P0 — PACKEDANCESTRYMAP / GENO.** Header geometry already done; `.snp`/`.ind`
  via existing readers; encoding is IDENTITY; the ONLY new logic is the
  SNP-major selected-individual gather + the device transpose. This validates the
  transpose primitive in isolation with **zero new golden generation** — the
  `golden_fitNA` PA<->TGENO pair (`golden_fitNA.json:7`) is a ready-made bit-exact
  cross-check (the f2 path was regenerated from a convertf-converted
  PACKEDANCESTRYMAP fixture because AT2 cannot read raw TGENO — PA and TGENO are
  the SAME dataset two ways).
- **P1 — EIGENSTRAT ASCII `.geno`.** Reuses `.snp`/`.ind` + `read_snp`/`read_ind`;
  adds the ASCII char gather + the `'9'->3` map; same transpose.
- **P2 — PLINK `.bed`.** New `.bim`/`.fam` parsers; `.bed` magic check + LSB-first
  gather; the `00/01/10/11` LUT + bit-order flip applied in the kernel's remap
  step; the distinct missing sentinel (`01`).
- **P3 — ANCESTRYMAP (legacy unpacked).** Same char map as EIGENSTRAT; unpacked,
  so no bit work — just normalize + pack into the canonical tile.

Each format = parse + gather + encoding-map + (shared) transpose. The transpose
kernel is written once in P0 and parameterized by encoding thereafter.

### 3.4 The GENO rlen floor (a P0 correctness detail)

GENO `bytes_per_record = max(48, ceil(n_ind/4))` (`eigenstrat_format.cpp:118`)
means small-`n_ind` files have PADDING bytes per SNP record beyond
`ceil(n_ind/4)`. The gather must read the full (possibly 48-byte-floored) record
but the transpose must consume only the first `ceil(n_ind/4)` bytes of codes —
i.e. bound individuals by `n_ind`, NOT by `bytes_per_record*4` (§8.3).

---

## 4. L2: ploidy detection on-device

### 4.1 The L2 violation today

`io::detect_sample_ploidy` is a pure-host loop
(`src/io/ploidy_detect.cpp:16-45`): every sample starts pseudo-haploid (ploidy
1), and for each gathered individual `g` it scans the first
`min(kPloidyDetectSnps=1000, n_snp)` SNPs of `packed.data()+g*bytes_per_record`,
setting ploidy 2 and breaking on the FIRST het (`code==kHetCode==1`). It is
consumed on exactly ONE path: `extract_f2_core.cpp:125` (the `Auto` branch).
dstat/dates/qpfstats force a constant all-diploid vector and never auto-detect,
so on-device detection is needed ONLY on the extract_f2 Auto path.

### 4.2 The prepass

Add `launch_detect_ploidy(d_packed, bytes_per_record, n_individuals, n_snp,
window=min(kPloidyDetectSnps,n_snp), d_sample_ploidy_out, stream)` in a new
CUDA TU (PRIVATE to `steppe_device`, mirroring `decode_compact_kernel.cu`).
Kernel: ONE thread per gathered individual (`n_individuals` is a few thousand —
a single 256-block sweep). Each thread is a literal port of the host outer-loop
body: init pseudo-haploid, scan `s=0..window`, read
`packed[g*bytes_per_record + s/kCodesPerByte]`, extract `genotype_code(byte,
s%kCodesPerByte)` (the SHARED core primitive — same extractor the host uses), set
diploid and break on the first `kHeterozygousGenotypeCode`. The first-het-wins
`break` is per-thread (one thread = one individual = the host's outer loop), so
there is no cross-thread reduction and no warp-ordering hazard; the result is
deterministic.

### 4.3 Wiring at the decode seam

The packed tile is already on the device as `dPacked` BEFORE the decode launch
(`cuda_backend.cu:1216-1218`). Insert the prepass there: on an Auto request,
allocate `dSamplePloidy` device-side and run `launch_detect_ploidy(...)` on the
SAME `stream_` before `launch_decode_af` (`cuda_backend.cu:1229`), instead of
H2D-copying a host-computed vector (`cuda_backend.cu:1222-1226`). Stream ordering
already guarantees upload -> prepass -> decode. The host `extract_f2_core.cpp:125`
Auto branch then stops calling `io::detect_sample_ploidy` and sets a request flag
— removing the host loop (the L2 fix). `CpuBackend` KEEPS the host
`detect_sample_ploidy` (the parity anchor; CUDA is forbidden there).

### 4.4 The layout-vanishes point

A SNP-major reader's transpose prepass produces a canonical individual-major
`dPacked` BEFORE decode. The ploidy prepass runs on that already-transposed
buffer, so it sees individual-contiguous records exactly like the TGENO path —
no per-format special-casing. Stream order: **transpose -> ploidy-detect ->
decode**, all reading the canonical layout. Ploidy and decode are
format-agnostic consumers of the transposed tile; the "is it SNP-major" question
never reaches them.

### 4.5 Parity with the host detector

Bit-identical by construction IF `genotype_code` (core) and `code_in_byte` (io)
extract the same bits at all 4 in-byte positions AND
`kHeterozygousGenotypeCode==kHetCode==1`. Both are pinned equal by
`tests/reference/test_decode_equivalence.cu` (§8.1 confirms this must be checked
at every position, not just the missing code). Keep `window =
min(kPloidyDetectSnps, n_snp)` (AT2 ntest-capped-at-nsnp) as a launch argument;
empty-tile guard: skip the launch when `n_individuals==0 || window==0` (mirrors
`ploidy_detect.cpp:27`).

> **Scope note.** This L2 prepass is independently valuable (it removes a host
> loop on the extract_f2 Auto path) and can land BEFORE or alongside the readers.
> Its covering goldens are `test_extract_f2_regimeB_parity.cu` (Auto path) +
> `test_decode_equivalence.cu`. The GENO==TGENO *ploidy* cross-check is gated on
> the P0 reader landing (read_tile throws on GENO today,
> `geno_reader.cpp:81-85`); do not cite it as already-runnable.

---

## 5. Validation strategy (reuse existing goldens, NO AT2 reruns)

Three nested tiers. The keystone is a bit-exact integer compare; the downstream
tiers are auto-covered.

### 5.1 Tier 1 — the bit-exact GENO==TGENO cross-check (the only new test)

The strongest assertion, and the only one requiring a new test file. Every AT2
golden was generated from the convertf-PA `v66_HO_pa` PACKEDANCESTRYMAP fixture
(`tests/reference/goldens/at2/README.md`); steppe reads the raw v66 TGENO
(`v66.p1_HO.aadr.patch.PUB`) — SAME inds, SAME 584131 SNPs, SAME `.ind`/`.snp`.
convertf is a lossless transcode, so the two files are the SAME genotypes on two
axes. A new CUDA self-checking TU `test_pa_decode_equivalence.cu` (cloned from
`test_decode_equivalence.cu`):

- Path A (oracle): `GenoReader(raw TGENO)` -> existing `read_tile` -> tile
  `T_tgeno`;
- Path B (new reader): `GenoReader(converted_pa/v66_HO_pa.geno)` -> new PA
  read path (parse + SNP-major gather of the SAME selected individuals +
  device transpose) -> tile `T_pa`;
- **ASSERT:** `memcmp(T_pa.packed, T_tgeno.packed) == 0` AND `pop_offsets` /
  `sample_ploidy` identical. Because PA and TGENO are provably the same data and
  the canonical packing is fully specified, a correct PA reader MUST produce the
  byte-identical tile. Integer/bit-only — **no float, no tolerance, no oracle
  file.** This is the format-reader analogue of `test_decode_equivalence.cu`'s
  GPU==CPU `max|Δ|==0`.
- Belt-and-suspenders: also run `detect_sample_ploidy` + `decode_af` on both
  tiles and assert Q/V/N `memcmp==0`.
- SKIP exit-0 if `converted_pa/` or `raw/` absent (mirrors
  `test_extract_f2_regimeB_parity.cu:147-162`).

### 5.2 Tier 2 — auto-coverage of the downstream goldens (no rerun, no new golden)

Once Tier 1 proves `T_pa == T_tgeno` bit-for-bit, EVERY existing genotype-path
golden is automatically a PA-path golden, because the pipeline downstream of the
tile consumes ONLY the `GenotypeTile`. Identical input tile => identical Q/V/N =>
identical f2 / D / smoothed-f2 / date. So `test_decode_equivalence.cu`,
`test_extract_f2_regimeB_parity.cu`, `cli_dstat_geno` (vs
`golden_fit0_dstat_geno.csv`), `cli_qpfstats` (vs `golden_qpfstats_geno.csv` /
`.rds`), `cli_dates`, `py_extract_f2`, and the entire f2-object
qpadm/qpwave/rotation layer (reading the committed `f2_fit0_9pop.bin` /
`f2_fitNA.bin` fixtures, all PA-derived) auto-cover the PA path with their
CURRENT goldens.

Minimal NEW wiring-proof tests (no new goldens): add a second `add_test`
invocation pointing the EXISTING `cli_dstat_geno` / `cli_qpfstats` / `cli_dates`
at the PA prefix instead of the raw TGENO prefix — same golden CSV, same
`rtol 1e-6`. A pass proves end-to-end PA->stat parity through the real CLI on the
GPU with zero new reference data.

### 5.3 Tier 3 — EIGENSTRAT / PLINK via reduction to PA

EIGENSTRAT and PLINK have no committed steppe golden. At TEST time (a build
fixture, not a steppe statistic — allowed) convert `v66_HO_pa` to EIGENSTRAT and
PLINK with convertf/plink2, then run the SAME Tier-1 bit-exact tile compare:
read the EIGENSTRAT/PLINK file -> transpose -> assert `memcmp`-equal to
`T_tgeno`. Each format is validated by reduction to the already-trusted
PA==TGENO identity — no new statistical golden, no AT2 rerun. The PLINK
encoding-map (the highest-risk plug-in, §8.4) is caught immediately by the
bit-exact compare if the LUT / bit-flip / allele polarity is wrong.

If convertf/plink2 are not on the box, Tier 3 SKIPs. The fully-CI-portable
alternative is to commit tiny hand-checkable EIGENSTRAT/PLINK fixtures (a few
inds × a few hundred SNPs, like `test_geno_reader.cpp`'s synthetic TGENO) — see
§8.5.

---

## 6. Compliance with the project constraints

- **§4 layering.** `io` stays a CUDA-free leaf emitting a plain `GenotypeTile`;
  the transpose + ploidy kernels are CUDA TUs PRIVATE to `steppe_device`; the new
  `ComputeBackend` virtual is CUDA-free (`backend.hpp` seam); the app layer is the
  only `io`<->`device` wiring point. No JAX / CuPy anywhere.
- **GPU-first.** Transpose, gather-into-individual-major, and ploidy detection
  run on-device; `CpuBackend` host loops are oracle-only (the established
  `decode_af_compact_*` base-throws pattern, `backend.hpp:786,824`). The transpose
  output can stay device-resident and feed `decode_af_resident` directly
  (`cuda_backend.cu:1197`), avoiding a transpose->D2H->H2D->decode round trip.
- **Single-GPU.** Everything is `--device 0`; no multi-GPU (parked).
- **Precision.** Transpose / gather / encoding-map / ploidy are integer/bit ops —
  the emulated-FP64-default + native carve-out policy is matmul-only and does NOT
  apply; bit-exactness is by construction (§2.2 precision note).
- **CUDA 13.** New kernels are standard CUDA 13 device code.

---

## 7. Build milestones (each small + golden-gated)

| M | Deliverable | Gate |
|---|---|---|
| **M-FR-0** | On-device ploidy prepass (`launch_detect_ploidy` + decode-seam wiring + flip extract_f2 Auto) — the L2 fix, independent of readers | `test_extract_f2_regimeB_parity.cu` (Auto) + `test_decode_equivalence.cu`, no new golden |
| **M-FR-1** | The on-device 2-bit transpose+gather+encoding kernel + the CUDA-free `transpose_to_canonical` virtual + `CpuBackend` host-loop oracle | unit: transpose oracle GPU==CPU `memcmp==0` on a synthetic SNP-major tile |
| **M-FR-2** | P0 PA reader: SNP-major gather + dispatch; wire `read_tile` to accept GENO via transpose | Tier-1 `test_pa_decode_equivalence.cu` (`memcmp==0`) |
| **M-FR-3** | PA wiring proof: PA-prefix variants of `cli_dstat_geno` / `cli_qpfstats` / `cli_dates` | Tier-2 auto-cover, existing goldens, `rtol 1e-6` |
| **M-FR-4** | P1 EIGENSTRAT ASCII reader (`'9'->3` map) | Tier-3 reduction-to-PA bit-exact compare |
| **M-FR-5** | P2 PLINK `.bed`/`.bim`/`.fam` (LUT + bit-flip + allele polarity) — **LANDED**: `read_bim`/`read_fam` (`plink_reader.cpp`) + `read_plink_snp_major_tile` (LSB→MSB flip + `kBedToCanon` LUT + ref:=A1) + the `GenoFormat::Plink` dispatch + the format-aware `resolve_genotype_triple`/`read_snp_table`/`read_ind_partition` front-door (`genotype_source.cpp`, the `.bed/.bim/.fam` vs `.geno/.snp/.ind` prefix fork). **The .fam population is column 6 (the phenotype), per AT2 `mcio.c:1180-1205` (NOT the FID — that is a convertf counter); convertf PACKEDPED writes it there with `outputgroup: YES`.** | Tier-1 `test_plink_decode_equivalence.cu` (`memcmp==0` PLINK==PA tile/ploidy/Q-V-N on GPU+CPU) + `cli_plink_prefix` (extract-f2 f2.bin BIT-IDENTICAL, qpdstat row-for-row) — both PASS on the real AADR convertf fixture |
| **M-FR-6** | P3 ANCESTRYMAP | Tier-3 reduction-to-PA bit-exact compare |

Commit between milestones; HIGH-impact reader changes are individually
golden-gated, lower-risk parse extensions group-batched.

---

## 8. Open questions / risks (for the user to weigh)

1. **Transpose output ownership.** Return a host `GenotypeTile` (simplest, drops
   into the existing 5 call sites unchanged — the safe P0 parity choice) OR a
   device-resident handle (fuses with `decode_af_resident`, the GPU-first
   optimization for P1+). `DecodeTileView` carries raw pointers
   (`backend.hpp:377`), so a resident variant needs a new view type or a
   resident-decode entry — confirm against the `decode_af_compact_*` signatures.
   **Recommendation:** host-tile for P0, resident for P1+.

2. **Ploidy request signaling.** A new explicit `bool detect_ploidy_on_device` on
   `DecodeTileView` (audit-greppable) vs a `kPloidyAuto` sentinel overloading the
   NULL-means-uniform contract (`backend.hpp:392-399`, more fragile). The legacy
   all-diploid callers (dstat/dates/qpfstats) pass a non-null constant vector and
   are unaffected either way. **Recommendation:** the explicit flag.

3. **GENO rlen-floor bounds.** Confirm the transpose kernel bounds individuals by
   `n_ind`, not by `bytes_per_record*4`, so padding bytes in small-`n_ind` GENO
   records are not decoded as phantom individuals (§3.4).

4. **PLINK encoding-map + allele polarity (highest risk).** PLINK `.bed` counts
   A1 (minor/ALT) copies with a code+missing convention that differs from
   EIGENSTRAT's ref-copy raw-value AND is LSB-first. The `.bim` ref/alt vs the
   `.snp` ref/alt must be reconciled or every SNP is allele-flipped. Recommend
   gating PLINK parity against the TGENO tile (Tier 1) so polarity is pinned to
   the trusted convention. Confirm the exact `.bed` LUT and missing sentinel
   before P2 — do NOT assume it matches GENO.

5. **Tier-3 external-tool dependency.** convertf/plink2 at test time means Tier 3
   SKIPs where they are absent. Decide: rely on box-local convertf, or commit
   tiny pre-converted EIGENSTRAT/PLINK fixtures (the only fully-CI-portable
   option, hand-checkable like the synthetic TGENO in `test_geno_reader.cpp`).

6. **Selected-individual gather correctness.** The SNP-major source interleaves
   ALL individuals within each SNP byte, so the transpose must do the
   selection+reorder (output column `g` -> source individual row from the
   `IndPartition`) and preserve the Q/V/N pop order (`extract_f2_core.cpp:116-118`,
   P-axis = read_ind sorted ASC by label). A reorder bug silently mis-assigns
   individuals — exactly what `geno_reader.cpp:114-117` guards against on the
   TGENO path. The Tier-1 byte-exact compare catches this.

7. **`snp_begin != 0` (the M5 tile loop).** `read_tile` rejects `snp_begin!=0`
   today (`geno_reader.cpp:86-90`, byte-alignment on the individual-major output).
   SNP-major input seeks whole per-SNP records, so nonzero `snp_begin` is actually
   EASIER for SNP-major (no sub-byte input offset). Decide whether the new reader
   lifts the restriction or keeps `snp_begin==0` for P0 parity.

8. **Data availability for Tier 1.** `converted_pa/v66_HO_pa` lives only on
   box5090; without it the Tier-1 cross-check is a permanently-skipped gate.
   Confirm the PA triple is present on the build box, and that its `.ind`/`.snp`
   row order byte-matches the raw TGENO prefix (the cross-check assumes
   apples-to-apples; assert `n_ind`/`n_snp` from both headers match before
   comparing tiles).

9. **CpuBackend transpose requirement.** Decide whether transpose-on-read must
   work on the CPU-only fallback (the oracle host loop) at all, or whether it is
   GPU-required like extract_f2 (`extract_f2_core.cpp:85-89` throws with no GPU).
   The CpuBackend transpose is needed at minimum as the Tier-1 oracle.

10. **Stale architecture.md.** §5 (~251, binary encoding) and §11.1 (~931,
    `plink_bed.cpp`) are wrong — flag for a doc update, do NOT let them drive the
    design.

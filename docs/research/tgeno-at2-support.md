# TGENO vs AT2 — SETTLED (lead synthesis of 3 lenses)

**HEADLINE**
- **Does AT2 (R admixtools, v2.0.10 — the one that built every golden) support v66 TGENO?** **NO.** It has no magic check on the decode path and **silently misreads** a TGENO file as SNP-major PACKEDANCESTRYMAP (transposed garbage) — no error thrown.
- **Verdict on the discrepancy:** **[b]** — the GOLDENS are AT2-misread-corrupted and **steppe's decode is correct**. (Not [a] steppe bug; not [c] missingness criterion; not [d] ambiguous on the decode itself.)
- **Are the goldens valid?** **NO.** Every golden made by AT2 `extract_f2` on this TGENO prefix (golden_fit0/golden_rot/golden_fitNA/golden_qpwave/golden_fit1) is built on an AT2 misread and must be regenerated from a format AT2 can read.

Confidence: **high.** Four independent sources agree — the AT2 C reader source, the DReichLab convertf README ("TGENO is a transposed format with 1 record / sample"), the on-box oracle, and exact file-size byte arithmetic that closes to the byte.

---

## (1) Does AT2 support the v66 "TGENO" format?

Two different tools share the name "admixtools" and must NOT be conflated:

| Tool | Supports TGENO? | Since | Evidence |
|---|---|---|---|
| **DReichLab/AdmixTools** (the C `convertf`) | **YES** | **v8.0.0** (written by Matthew Mah); v8.0.2 = bugfix for tgeno >2 GB | convertf README: "TGENO is a transposed format with 1 record / sample"; v66 README footnote 2 points here |
| **uqrmaie1/admixtools** (the **R package, v2.0.10** — the one that **made every golden**) | **NO** | n/a | reader has no magic check, SNP-major geometry; see below |

The v66 README is explicit (box: `aadr_v66.p1__README.docx`, word/document.xml text):
- ".geno: Genotypes (in the newer 'tgeno' format, see footnote 2 below)."
- Footnote 2: "genotypes are in a new supported 'transpose_packed' format described in Admixtools convertf.README ... **Please note that older versions of convertf/admixtools do not support this format.**"

**What AT2 R (v2.0.10) DOES with a TGENO file — SILENT MISREAD, not an error:**
- `R/io.R`: the actual extract_f2 path `packedancestrymap_to_afs` derives counts from the **`.snp`/`.ind` row counts** (`nindall = nrow(indfile)`, `nsnpall = nrow(snpfile)`) and calls the C reader with **`transpose = FALSE`** (which only transposes the OUTPUT matrix, never the on-disk axis). It does NOT consult the .geno magic at all on this path. (A separate display helper `read_packedancestrymap` reads `hd[2]`/`hd[3]` for n_ind/n_snp, but that is not the decode path.)
- `src/cpp_readgeno.cpp` (`cpp_read_packedancestrymap`, master == v2.0.10): the **header/magic read is commented out** (lines ~45–47: `// char* header = new char[bytespersnp]; // in.seekg(0...); // in.read(...)`), so no "TGENO"/"GENO"/"PACKEDANCESTRYMAP" magic is ever verified. It computes a **SNP-major** stride `bytespersnp = len/(nsnp+1)` (line ~36), iterates **per-SNP**, and decodes 4 codes/byte MSB-first (`>>6, >>4, >>2, &MASK0`), missing code = 3.
  - Note: the bit-shift order (MSB-first) and the missing=3 code are IDENTICAL to TGENO/steppe. **Only the AXIS is wrong** — AT2 reads 4 *individuals* per byte along a per-SNP record; TGENO packs 4 *SNPs* per byte along a per-individual record.

**Decisive arithmetic on the real HO file** (box: size 4,029,634,650; header `TGENO 27594 584131 ...`):
- TGENO (correct): `48 + n_ind*ceil(n_snp/4) = 48 + 27594*146033 = 4,029,634,650` = **exact file size** (verified).
- AT2 misread stride: `len/(nsnp+1) = 4029634650/(584131+1) = 6898`, remainder 292114 — does **not** divide cleanly. A valid PACKEDANCESTRYMAP record would be `max(48, ceil(27594/4)) = 6899`; AT2's 6898 is **off by one** and lands mid-record, so it walks across individual boundaries = transposed garbage.
- This is exactly the "impossible decode" the M(cli-4) verdict observed (`0xFF` → genotype 2 for one "sample" and 0 for another at the same "SNP"). **The verdict's own evidence mechanism was AT2 misreading TGENO** — it inverted cause and effect.

Repo negatives (uqrmaie1/admixtools): DESCRIPTION `Version: 2.0.10`; io vignette lists only PLINK / PACKEDANCESTRYMAP / EIGENSTRAT (no TGENO). No private/dev branch with TGENO support found as of 2026-06-21 (cannot 100% exclude a private branch — flagged).

---

## (2) The authoritative TGENO byte layout

From the real file header + DReichLab convertf README + EIGENSOFT packing + the on-box oracle (`build_tgeno_matrix.py`), all consistent:

- **Header:** 48-byte ASCII record: magic `"TGENO"`, then decimal `n_ind`, `n_snp`, then two hex hash tokens (ihash/shash, for .snp/.ind consistency checks), NUL-padded to 48 bytes.
- **Body:** `n_ind` records, **individual-major** (DReichLab README: "1 record / sample"). Each record is `ceil(n_snp/4)` bytes = one individual's genotypes across all SNPs.
- **Packing:** 4 SNPs/byte, 2 bits each, **MSB-first**: `snp0=(byte>>6)&3, snp1=(byte>>4)&3, snp2=(byte>>2)&3, snp3=byte&3`, i.e. `(byte >> (6 - 2*(k mod 4))) & 3`.
- **Code mapping (RAW VALUE):** 0/1/2 = copies of the **reference** allele (.snp col 5); **3 = MISSING**.
- **Stride / offset:** record r at byte `48 + r*ceil(n_snp/4)`.
- **Closure:** `48 + 27594*146033 = 4,029,634,650` = exact file size.

Independence note: the on-box oracle (`build_tgeno_matrix.py`) is same-author/self-derived ("confirmed from the file header"), so it is NOT an independent source on its own. The **independent corroboration** is: (i) DReichLab convertf README "1 record / sample"; (ii) AT2's own C reader uses the same MSB-first bit order and missing=3; (iii) the file-size arithmetic closes to the byte for individual-major and fails for SNP-major. (`mtdna_uncompress_v66.p1.py` is NOT relevant — mitochondrial FASTA reconstruction, not .geno packing.)

---

## (3) Does steppe's decode match the spec? Point-by-point — ALL MATCH

| Aspect | Authoritative TGENO | steppe | file:line | Match |
|---|---|---|---|---|
| Magic | "TGENO" | `kMagicTgeno = "TGENO"`, exact-token compare | `eigenstrat_format.hpp:58`, `.cpp:48-58` | YES |
| Header bytes | 48 | `kGenoHeaderBytes = 48` | `eigenstrat_format.hpp:51` | YES |
| Axis (transpose) | individual-major, n_ind records | `n_records = n_ind`, `bytes_per_record = packed_bytes(n_snp)` | `eigenstrat_format.cpp:110-113` | YES |
| Record offset | `48 + r*ceil(n_snp/4)` | `header_bytes + row * bytes_per_record` | `geno_reader.cpp:202-205` | YES |
| Stride | `ceil(n_snp/4)` | `packed_bytes(n) = ceil(n/4)` | `eigenstrat_format.hpp:136-141` | YES |
| Bit order | MSB-first 6/4/2/0 | `code_in_byte: (byte>>(6-2*(k%4)))&3` | `eigenstrat_format.hpp:147-150` | YES |
| Codes 0/1/2 = ref copies | raw value | raw-value mapping (not binary) | `eigenstrat_format.hpp:23-26` | YES |
| Missing code | 3 | `kMissingCode = 3` | `eigenstrat_format.hpp:81` | YES |
| GENO (SNP-major) handling | refuse to mis-decode | throws on non-TGENO `read_tile` | `geno_reader.cpp:81-85` | YES (guardrail) |

steppe is element-for-element correct, AND — unlike AT2 — it **refuses** to decode a non-TGENO file rather than silently misread it (`geno_reader.cpp:81-85`). (Implementation note, not a bug: `read_tile` reads only a byte-aligned SNP **prefix** — read length `packed_bytes(tile_snps)`, line 110 — while seeking with the full per-individual stride `header_.bytes_per_record`, line 205. Correct for the M1 `snp_begin==0` prefix path.)

---

## (4) Verdict on the M(cli-4) discrepancy

steppe `extract-f2`: 391333 SNPs, weights [0.866, 0.134]. golden_fit0: 500848 SNPs, [0.559, 0.441].

**[b] — the GOLDENS are AT2-misread-corrupted; steppe is right.**
- AT2 cannot read TGENO (Q1) and silently misreads it on the exact prefix the goldens were built from. golden_fit0's 500848 SNPs / [0.559,0.441] are computed on transposed garbage.
- steppe's decode matches the authoritative layout on every axis (Q3), confirmed by source + the file-size arithmetic that closes to the byte.
- The M(cli-4) verdict that blamed steppe is **invalid** — its evidence (the impossible 0xFF→2/0 decode) is itself the AT2 misread.

**Not [a]:** steppe's decode is verified correct.
**Not [c]:** a missingness-criterion difference cannot produce an axis transpose; the impossible-decode signature is an axis error, not a filter threshold.
**Caveat (honest):** the decode — the disputed item — is verified correct. steppe's downstream 391333-SNP count depends on filters (autosomes-only, allele/missingness gates) that this read-only task did not independently re-derive; those are separate from the decode and do not change the verdict that the decode is correct and the goldens are corrupt.

---

## (5) Are the goldens valid? — NO. Critical knock-on.

Every golden (golden_fit0 / golden_rot / golden_fitNA / golden_qpwave / golden_fit1) was generated by AT2 `extract_f2(pref="/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB", ...)` reading the TGENO `.geno`. Since AT2 (R v2.0.10) misreads TGENO, **all five goldens inherit the misread** — steppe parity has been measured against a corrupted reference. They must be regenerated from a format AT2 can read.

---

## (6) The fix path

**Do NOT change steppe's decode** — it is correct. Fix the reference.

1. **Quarantine** all AT2-on-TGENO goldens (fit0/rot/fitNA/qpwave/fit1); mark them invalid.
2. **Regenerate the reference** by giving AT2 correctly-axised data. Options, in order of preference:
   - **(recommended)** Convert the v66 TGENO `.geno` → classic PACKEDANCESTRYMAP (or EIGENSTRAT) with a **TGENO-aware DReichLab C `convertf` >= v8.0.0** (use v8.0.2 for the >2 GB bugfix), then run R admixtools `extract_f2` on the converted prefix. This yields a true AT2 reference on correctly-axised data.
   - Cross-check a handful of converted genotype cells against steppe's TGENO read (and against the on-box oracle) to confirm the conversion is faithful before trusting the new golden.
3. **Re-run M(cli-4) parity** against the regenerated golden. Expectation: it converges toward steppe's **391333 SNPs / [0.866, 0.134]**, NOT the corrupted 500848 / [0.559, 0.441].
4. **Guardrails:** keep steppe's magic-check refusal of non-TGENO files; add a test/CI guard that **rejects any golden built by AT2 R directly on a raw TGENO `.geno`**; document that AT2 R (v2.0.10) must never be pointed at a raw TGENO prefix.

**Cheapest decisive next experiment (confirms [b] before the full convert):**
Run the validated on-box oracle `build_tgeno_matrix.py` (and/or DReichLab `convertf` v8.0.2 → PACKEDANCESTRYMAP → AT2) on the SAME prefix and the SAME pop set as golden_fit0. If the oracle/converted-AT2 SNP count + weights match steppe (~391333 / [0.866,0.134]) and NOT the golden (500848 / [0.559,0.441]), [b] is confirmed end-to-end. This is one box read + one small conversion — no large recompute.

---

## Sources

- AADR v66 README (box): `/workspace/data/aadr/1240k/aadr_v66.p1__README.docx` (word/document.xml) — "newer 'tgeno' format"; footnote 2 "transpose_packed ... older versions of convertf/admixtools do not support this format."
- Real HO file (box): `/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB.geno` — size 4,029,634,650; header `TGENO 27594 584131 ...`.
- On-box oracle (box): `/workspace/data/aadr/raw/build_tgeno_matrix.py` lines 8-16, 28-38, 98-110 — 48-byte header, individual-major, MSB-first `(byte>>6)&3 ... &3`, value 3 = missing.
- AT2 R reader: https://raw.githubusercontent.com/uqrmaie1/admixtools/master/src/cpp_readgeno.cpp — magic read commented out (~45-47); `bytespersnp = len/(nsnp+1)` (~36); per-SNP iteration; MSB-first 4-codes/byte; missing=3.
- AT2 R io.R: https://raw.githubusercontent.com/uqrmaie1/admixtools/master/R/io.R — `packedancestrymap_to_afs` derives counts from `.snp`/`.ind` rows, calls reader with `transpose=FALSE`.
- AT2 R version: https://raw.githubusercontent.com/uqrmaie1/admixtools/master/DESCRIPTION — `Version: 2.0.10`.
- AT2 io vignette (formats list): https://uqrmaie1.github.io/admixtools/articles/io.html — PLINK / PACKEDANCESTRYMAP / EIGENSTRAT only.
- DReichLab convertf README: https://github.com/DReichLab/AdmixTools/blob/master/convertf/README — "TGENO is a transposed format with 1 record / sample"; "packed binary (2 bits per genotype)"; hash codes in header record.
- DReichLab TGENO support since v8.0.0 (Matthew Mah); v8.0.2 >2 GB bugfix: https://github.com/DReichLab/AdmixTools/releases
- steppe source: `/home/suzunik/steppe/src/io/eigenstrat_format.hpp`, `eigenstrat_format.cpp`, `geno_reader.cpp` (lines cited inline above).

## Flagged uncertainty
- Cannot 100% exclude a private/unreleased uqrmaie1 branch with TGENO support; public master (== v2.0.10) has none.
- The `0xFF→2/0` impossible-byte signature is from the prior M(cli-4) verdict, not re-run here; the load-bearing evidence (no magic check, wrong 6898 stride, exact file-size closure, format-support negatives, DReichLab "1 record/sample") is independently verified.
- WebFetch source snippets are paraphrase/line-approximate from the small-model reader, not byte-verified character-for-character; the structural claims (commented-out magic, `len/(nsnp+1)`, transpose=FALSE, axis) are consistent across all three lenses and the file-size math.
- steppe's downstream 391333-SNP count depends on filters not independently re-derived in this read-only task; the decode (the disputed item) is verified correct.

# steppe — Dataset Generalization Report (non-AADR parity)

**Question tested:** steppe's AT2 parity was only ever validated on AADR v66 (ancient DNA:
pseudo-haploid, a real cM map in the `.snp`). Does that parity **generalize** to modern,
all-diploid, non-AADR panels — reading the format *and* matching admixtools 2.0.10 (the R
oracle) at golden tolerance?

**Box:** box5090 (RTX 5090, sm_120, CUDA 13.1, emu-fp64 default). **Oracle:** admixtools 2.0.10
(R 4.3.3). **Datasets:** AdmixTools/HGDP "allmap" (Human Origins 2012), SGDP, gnomAD HGDP+1KGP,
1000G chr22. All REAL data, all decode paths on-device.

---

## 1. Headline

**steppe's core f-statistic math generalizes cleanly beyond AADR — but its out-of-the-box
default parity does NOT.**

Two things are simultaneously true and both matter for the release claim:

- **The MODERN-DIPLOID DECODE PATH IS CORRECT.** On four independent modern all-diploid panels
  (HGDP/HumanOrigins, SGDP, gnomAD HGDP+1KGP, 1000G), whenever steppe and AT2 are given the
  **identical SNP set**, steppe matches AT2 to **~1e-11 to ~1e-13** on f4, f3, and qpAdm
  weights/chisq/p — i.e. bit/golden parity. The pseudo-haploid→diploid branch switch (steppe
  `--ploidy auto` classifies every sample as diploid, matching AT2 `adjust_pseudohaploid`)
  produces correct statistics. f3 — which does **not** cancel the within-population diploid
  heterozygosity correction the way f4 does — matches to ~1e-14, which is the sharpest possible
  proof the diploid correction itself is right. **The AADR result is not an artifact of
  pseudo-haploid data.**

- **OUT-OF-THE-BOX, a user running steppe vs AT2 on the same modern EIGENSTRAT gets 1–4%
  different f-stats.** This is NOT a decode/math bug. It is a **default SNP-selection divergence**:
  steppe's `extract-f2` **unconditionally drops strand-ambiguous (palindromic A/T + C/G) SNPs**
  ("drop-not-flip", `src/io/filter/filter_decision.hpp`), which AT2 keeps. On modern panels that
  is **14–16% of all SNPs** (e.g. 50,819/352,987 on 1000G chr22; 240,788/1,539,845 on HGDP), and
  it shifts f4 by up to 4.1% and qpAdm weights/p by ~1–2%. Match the SNP sets and the gap
  collapses to 1e-12.

**Verdict:** steppe is **"a faster ADMIXTOOLS for any dataset" at the level of the underlying
math**, but today it is only **"a faster AADR tool" out-of-the-box** for reproducing AT2
numbers, because (a) its default strand-ambiguous filter silently changes the SNP set on modern
data, and (b) it has **no base-pair fallback when the cM map is genuinely zero** — a real hazard
for VCF/PLINK-derived data. Both are fixable/documentable (Section 4).

---

## 2. Matrix

| Dataset | Source | n_samp / n_snp | Real cM map? | steppe reads? | AT2 ran? | Parity (max reldiff) | Diploid path ok? | Jackknife-SE ok? | What breaks |
|---|---|---|---|---|---|---|---|---|---|
| **AdmixTools "allmap"** (Human Origins 2012 / HGDP) | Reich Lab `AdmixTools_Example_Data.tar.gz`; packedancestrymap, read natively | 836 / 621,038 (subset 139 / 8 pops) | **Yes** (real, monotonic col3) | **Yes** (native PA) | Yes | **PASS ~1.5e-10** (f4/f3 ~2–7e-13, qpAdm wt ~3e-14, chisq 4.1e-11, p 1.5e-10) | **Yes** (het frac 0.217; 0 PH / 139 diploid) | **Yes** (711 identical blocks, real map) | Only qpAdm **weight-SE** off 7.0e-4 (method diff). Acquisition premise wrong: R package ships **no** genotypes, only precomputed f2. |
| **SGDP** cteam_extended.v4.maf0.1perc | Reich Lab PLINK → convertf → PA/EIGENSTRAT | 55 / 212,231 (matched set) | **No** (raw .bim col3 = 0; convertf **fabricates** 1 cM/Mb) | **Yes** (PA + ASCII EIGENSTRAT identical) | Yes | **PASS ~1e-11** on matched set (f4/f3/qpAdm). **Naive default: up to ~6%** (f4 Q2 5.8%, Sardinian wt 6.6%) | **Yes** (real hets present; not 0/2 only) | **Yes** but on a **synthetic** uniform map | Default SNP-set divergence: strand-ambiguous drop (37,777/250,008 ≈15%) + `--maxmiss` semantics (per-indiv vs per-pop) + mono definition. convertf silently invents cM map. |
| **HGDP** (gnomAD HGDP+1KGP v2 phased) | gnomAD GCP bucket BCF chr13–22 → plink2 → convertf → EIGENSTRAT | 173 / 1,539,845 | **No** (BCF col3 = 0; convertf fabricates 1 cM/Mb) | **Yes** (EIGENSTRAT) | Yes | **FALSE default: 4.1e-2** (f4 z); **PASS 2.5e-12** on matched non-ambiguous set | **Yes** (`--ploidy auto` → 0 PH / all diploid) | **Yes on synthetic map; BREAKS on true cM=0** | Strand-ambiguous drop 240,788/1,539,845 (15.6%), no flag disables it. **cM=0 → steppe collapses to 10 blocks (1/chr); AT2 falls back to 2 Mb → 357 blocks.** qpAdm wt-SE 0.34%. |
| **1000G** phase3 chr22 | EBI 1000G VCF → plink2 → convertf → EIGENSTRAT; real hg19 recomb map tested separately | 663 / 352,987 (matched 302,168) | **No** (VCF col3 = 0; convertf fabricates; real map tested in regime b) | **Yes** (ASCII EIGENSTRAT) | Yes | **FALSE default: 2.9e-2** (f4); **PASS ~9e-13** f4, ~1.1e-11 qpAdm wt on matched set | **Yes** (663 diploid / 0 PH, AT2 parity) | **Yes ≥8 blocks; BREAKS at true cM=0 → 1 block** | Strand-ambiguous drop 50,819/352,987 (14.4%). **True cM=0 → steppe 1 block → f4 SE=NA, qpAdm `non_spd_covariance` (fails); AT2 → 2 Mb / 18 blocks.** qpAdm wt-SE 6.5%→3.2% (block-count dependent). |

*"Parity" column: max relative difference across the task's pinned golden set (f4/f3 est/se/z/p,
qpAdm weights/chisq/p). "Matched set" = SNP sets pre-aligned (palindromes removed, missing-filter
reconciled) so both tools see identical input.*

---

## 3. Findings

### 3.1 Modern-diploid path (RISK 1) — **PASS, decisively**

The primary risk was that AADR is pseudo-haploid, so steppe's ploidy adjustment had only ever
been exercised on the pseudo-haploid branch; a modern all-diploid panel hits a different decode
path.

- On all four datasets steppe's `--ploidy auto` correctly classified **every** sample as diploid
  (0 pseudo-haploid), matching AT2's `adjust_pseudohaploid` classification.
- Data confirmed genuinely diploid, not just labeled so: heterozygous calls (value=1) are present
  (allmap het-fraction 0.217; SGDP ~54k hets in a 20k×55 sample), unlike AADR pseudo-haploid
  which is 0/2 only.
- On the identical SNP set, **f4, f3, and qpAdm weights/chisq/p match AT2 to 1e-11…1e-14** on
  every dataset. f3's ~1e-14 match is the strongest evidence: f3 retains the diploid within-pop
  heterozygosity correction that f4 cancels, so a wrong diploid correction would surface there
  first — it doesn't.

**Conclusion:** the modern all-diploid f2/f4/f3/qpAdm math is correct. The AADR parity
generalizes to modern diploid panels.

### 3.2 cM genetic map (RISK 2) — **NUANCED; a real, silent hazard for map-less data**

The second risk was that block-jackknife SE and DATES need a real centiMorgan map in `.snp`
column 3; VCF/PLINK-derived data ships cM=0.

- **Every non-AADR source carries NO real recombination map** (raw `.bim`/BCF/VCF genetic-distance
  column = 0 for all SNPs — SGDP 34.4M, HGDP, 1000G).
- **EIGENSOFT `convertf` silently fabricates a uniform 1 cM/Mb map** when writing EIGENSTRAT/PA:
  `genetic_pos_Morgans = bp × 1e-8`, reset per chromosome (verified: `.snp` col3 == bp×1e-8 to 6
  decimals on 1000G and HGDP). So the natural convertf output has a **non-zero but synthetic**
  column. Because of this, in the main runs both tools independently partitioned into identical
  blocks and f4 SEs matched to ~1e-13 — **but the SE quality rests on a fabricated uniform map,
  not real recombination.** DATES and fine-block SE on such data would silently use the fake map.
- **The genuine hazard is a converter that preserves cM=0.** Tested directly on 1000G/HGDP by
  constructing a true cM=0 `.snp`: **steppe has no base-pair fallback** — `assign_blocks` reads
  `.snp` col3 directly, so it **collapses each chromosome to a single block** → f4 SE = NA, and
  qpAdm returns `non_spd_covariance` and **fails**. **AT2 instead warns** ("No genetic linkage
  map found! Defining blocks by base pair distance of 2e+06") and falls back to 2 Mb BP blocks
  (18 blocks on 1000G chr22, 357 on HGDP), still producing SEs.

**Conclusion:** a map-less dataset does **not** break steppe *in practice* only because convertf
masks it. On genuinely cM=0 input, steppe's jackknife SE / qpAdm SE / DATES **silently break**
(1 block per chromosome) where AT2 degrades gracefully. This is the single most important code-level
gap for general-dataset support.

### 3.3 Numeric drift vs AT2 — one consistent sub-golden item: qpAdm **weight SE**

Across every dataset, when SNP sets are matched, the **only** quantity outside golden tolerance is
the **qpAdm weight standard error** (allmap 7.0e-4, SGDP 9.4e-4, HGDP 3.4e-3, 1000G 6.5%→3.2% as
blocks grow 8→16). Crucially:
- qpAdm **point weights, chisq, and p all bit-match** AT2 (~1e-11…1e-14), and
- the underlying f4/f3 block-jackknife SEs themselves **bit-match** AT2 (~1e-13).

So this is not a decode error — it's a **weight-SE method nuance**: steppe's analytic/delta-method
Jacobian vs AT2's leave-one-block-out re-solve jackknife on the weights. It is consistent
(steppe ~0.07–0.3% smaller on real multi-block data), block-count dependent (the 6.5% on 1000G is
an 8-block single-chromosome artifact where one tiny 1,579-SNP block dominates), and **weight SE is
not in the task's pinned golden set** (weights/chisq/p, all of which pass). It would be negligible
on genome-wide (hundreds-of-blocks) runs but should be documented.

### 3.4 Format / acquisition gaps

- **The "admixtools example" is not zero-download.** The R package ships **no** geno/snp/ind — its
  `extdata` is legofit demo files, and the bundled `example_f2_blocks` is **precomputed f2**, not
  genotypes. The real example genotypes are the separate DReichLab `AdmixTools_Example_Data`
  tarball (Human Origins 2012). Minor, but the premise that the package carries example genotypes
  is false.
- **Portal rot.** The classic Stanford HGDP array portal (hagsc.org) is dead (404); cephb.fr
  offers only metadata. HGDP genotypes had to be pulled from the **gnomAD HGDP+1KGP GCP bucket**
  (public, no auth; box needed `curl -k` for the TLS chain). Reportable acquisition friction, not
  a steppe issue.
- **steppe reads every format tested** natively: packedancestrymap (allmap, native, no convertf),
  packed + unpacked ASCII EIGENSTRAT (SGDP verified byte-identical decode on both), EIGENSTRAT
  (HGDP, 1000G). No format-read failures. steppe was also **faster on ingest** (allmap extract-f2
  2.6s vs AT2 22.8s).

### 3.5 Default SNP-selection divergences (the real "what breaks")

Three default-behavior differences shift out-of-the-box numbers on modern data; all are
SNP-selection, none are math:
1. **Strand-ambiguous drop (dominant).** steppe unconditionally drops A/T + C/G SNPs; AT2 keeps
   them. 14–16% of SNPs on every modern panel. **No `extract-f2` flag disables it** (`--maf 0`,
   `--maxmiss 1`, `--no-drop-mono`, `--mind-max-miss 1` all still drop exactly the ambiguous
   count). Sole cause of the 1–4% out-of-the-box gap.
2. **`--maxmiss` semantics differ.** steppe `--maxmiss` = per-**individual** missing fraction;
   AT2 `maxmiss` = per-**population** fully-missing fraction (SGDP: steppe dropped 92,677 vs AT2
   968 at maxmiss=0).
3. **Monomorphic definition differs** (AT2 `poly='f2'` vs steppe drop-mono).

Reconciliation to golden parity: pre-filter to non-ambiguous SNPs + align the missing convention
(`plink2 --geno 0 --mac 1` then drop A/T & C/G) → both tools keep an identical set → f4/f3/qpAdm
match to golden tolerance. Biology sanity held throughout (e.g. qpAdm Uygur = 56% Han + 44% French;
1000G qpAdm CEU weight 0.603 bit-matched).

---

## 4. Release implication

**Claim today:** steppe is **"a faster ADMIXTOOLS whose f-statistic *math* is correct on any
diploid dataset,"** but it is **not yet "a drop-in faster ADMIXTOOLS that reproduces AT2 numbers
out-of-the-box on arbitrary datasets."** The gap between those two claims is entirely SNP-selection
defaults and a missing block fallback — not the compute engine.

To honestly claim **general-dataset support**, two things are needed:

**A. Code fixes (recommended before a "general dataset" claim):**
1. **Add a base-pair block fallback when the `.snp` cM column is all-zero** (mirror AT2: warn +
   partition by a BP window, default 2 Mb). This is the only place steppe **silently produces
   wrong/failed SE** (1 block/chr → SE=NA, qpAdm `non_spd_covariance`) where AT2 degrades
   gracefully. Highest-priority fix for VCF/PLINK-derived data and DATES.
2. **Expose a flag to keep (or flip) strand-ambiguous SNPs** — e.g. `--keep-ambiguous` /
   `--strand-mode {drop,keep,flip}`. Today the drop is unconditional with no override, which makes
   exact AT2 reproduction on modern panels impossible without pre-filtering outside steppe.

**B. Documented limitations (must ship regardless):**
1. steppe **drops strand-ambiguous SNPs by default**; AT2 keeps them → 1–4% differences on modern
   panels unless SNP sets are matched. Document the reconciliation recipe.
2. steppe `--maxmiss` is **per-individual**, AT2's is **per-population** — document the semantic
   difference.
3. **A real cM recombination map is required** for correct block-jackknife SE and DATES; VCF/PLINK
   sources have none, and `convertf` **fabricates a synthetic 1 cM/Mb map** — SEs will be computed
   off a fake map unless a real one is supplied. Warn users; recommend interpolating a real map
   (e.g. HapMap/Broad hg19) before DATES.
4. qpAdm **weight SE** can differ from AT2 by a few % on low-block-count datasets (method nuance);
   point weights, chisq, and p are exact.

**Bottom line:** the hard part — GPU-correct modern-diploid f2/f4/f3/qpAdm at AT2 parity — **works
and generalizes.** What stands between steppe and a defensible "faster ADMIXTOOLS for any dataset"
claim is one real code fix (cM=0 BP-block fallback), one ergonomics fix (strand-ambiguous flag),
and a short, honest limitations page. None of it is in the compute engine.

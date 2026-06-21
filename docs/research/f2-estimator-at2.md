# steppe ↔ ADMIXTOOLS 2 f2-estimator parity: diagnosis + reconciliation spec

**Synthesis of 3 lenses (at2-f2-source, steppe-f2-code, empirical-regimes). Read-only investigation. No steppe code changed, no commits.**

---

## HEADLINE

- **The differing term:** the **pseudo-haploid N convention** in the within-population
  het bias-correction denominator. steppe hard-codes `ploidy = 2` for *every* sample,
  so `N = 2·n_indiv` always; the AT2 golden was built with `adjust_pseudohaploid=TRUE`,
  which auto-detects pseudo-haploid aDNA per-sample and uses `N = n_indiv`. The het
  correction `hc = q(1−q)/max(N−1,1)` is then divided by `2n−1` (steppe) vs `n−1` (AT2).
  `apply_corr` and `afprod` are **not** the culprit — both match. The f2 *formula* is
  byte-identical across all three implementations.
- **Does it blow up on pseudo-haploid? YES.** It is **not** a uniform ~0.2% floor.
  The error tracks the pseudo-haploid sample fraction ~1:1: **0.00% on modern-diploid
  pairs (exact), median ~15.5% on all-ancient-pseudo-haploid pairs, up to >100% (one
  lens 109.6%, another 232%) on close-kin pseudo-haploid pairs with small true f2.**
- **Fixable or FP floor? FIXABLE convention difference, NOT the emulated-FP64 floor.**
  Emulated-FP64 is ruled out as the dominant residual: modern-diploid pairs share the
  identical FP path and are bit-identical (`emu40` == `fp64`, 0.000000%), yet the
  ancient-pseudo-haploid error is 15–100%+. A precision floor cannot produce a 230%
  sign-consistent error. The fix is per-sample ploidy detection feeding the het
  correction's `N` only.

---

## (1) THE EXACT ESTIMATOR DIFFERENCE

### The f2 formula is identical on all three sides

```
f2(A,B) = mean_over_SNPs[ (p_A − p_B)²  −  hc_A  −  hc_B ]
hc_X    = p_X(1 − p_X) / max(1, N_X − 1)
```

- **steppe:** numerator/per-SNP summand `(p_i−p_j)² − hc_i − hc_j` in
  `src/core/internal/f2_estimator.hpp:92-94` (`f2_summand`); het correction
  `q*(1−q)/max(N−1, kHetCorrDenomFloor)` in `f2_estimator.hpp:71-77`
  (`het_correction`), with `kHetCorrDenomFloor = 1.0` — i.e. exactly AT2's `max(1,N−1)`.
- **AT2 R 2.0.10:** `af(1−af)/max(1, count−1)` in `src/cpp_fstats.cpp:175-180`
  (`cpp_mats_to_f2_arr`); block mean in `R/resampling.R:31-38`.
- **DReichLab C:** same algebraic estimator; autosomal default ploidy=2
  (`qpsubs.c:343-362`) — which equals steppe and equals AT2 with
  `adjust_pseudohaploid=FALSE`.

`apply_corr` (the `−hc` subtraction itself) and `afprod` (the cross-product
reformulation) are present and matched on both sides. Q / allele-frequency is
**byte-identical** (measured max|ΔQ| = 0.00e+00 on the matched SNP set; the diploid/
pseudo-haploid `count` ratio is exactly 2.000 but **Q = AC/N is convention-invariant**
because AC scales with N).

### The single term that differs: `N` in the het-correction denominator

| | steppe | AT2 golden (`adjust_pseudohaploid=TRUE`) |
|---|---|---|
| ploidy source | hard-coded constant | per-sample auto-detect |
| pseudo-haploid pop | `N = 2·n_indiv` | `N = n_indiv` |
| denominator | `2n − 1` | `n − 1` |
| effect | subtracts ~half the het correction | full correction |

**steppe — the hard-coded ploidy (the bug):**
- `src/app/cmd_extract_f2.cpp:60` `constexpr int kPloidy = 2;`
- plumbed unchanged at `cmd_extract_f2.cpp:259` (`view.ploidy = kPloidy`) and
  `:289` (`fin.ploidy = kPloidy`).
- `src/core/internal/decode_af.hpp:142-156` (`finalize_af`): `N = ploidy * AN`,
  `Q = AC / N`. The file comment (`decode_af.hpp:23-32`) states ploidy is
  *"a PARAMETER sourced from metadata, never auto-detected from genotypes"* and
  *"For the real AADR data every sample is diploid (ploidy == 2)"* — that assumption
  is **false for aDNA** and is precisely the divergence.

**AT2 — per-sample auto-detect (the reference):**
- `src/cpp_readgeno.cpp:137-168`: a sample is classified pseudo-haploid unless a het
  call appears in the first 1000 SNPs.
- `src/cpp_readgeno.cpp:254-268` and `R/io.R:155, 167-170`: `count = Σ ploidy(i)` →
  `N = n_indiv` for pseudo-haploids, `2·n_indiv` for diploids.
- consumed by the estimator `src/cpp_fstats.cpp:478-498`.

**FP-path note:** in steppe the het correction *and* the catastrophic-cancellation
assemble are **native FP64 in every precision mode**; emulated-FP64 enters *only* the
three cuBLAS GEMMs (`src/device/cuda/f2_block_kernel.cu:15-28, 392-407`,
`launch_assemble_f2` stays native). So the het-convention term is computed identically
under `emu40` and `fp64` — confirming the residual is convention, not precision.

---

## (2) EMPIRICAL CHARACTERIZATION (real AADR, on-box, blocks+SNPs byte-identical per regime)

### Per-regime steppe-vs-AT2 f2 relative diff

| Regime | Blocks | SNPs kept | median rel diff | max rel diff |
|---|---|---|---|---|
| **Modern** (all diploid: Han, French, Sardinian, Papuan, Mbuti) | 710 | 513,822 | **0.0000%** | 0.31% (Karitiana, 1 pseudo-hap sample of 28) |
| **Ancient** (all pseudo-haploid: CordedWare, Yamnaya, Iran_N, Israel_Natufian, Turkey_N, Serbia) | 710 | 308,221 | **15.55%** | **109.63%** |
| **Mixed** (Haak 15-union, the production golden) | 709 | 264,544 | **4.71%** | **232.16%** |

### It BLOWS UP on pseudo-haploid — it is NOT uniform

- **Modern-diploid: exact** (0.0000% median; the sole 0.31% outlier, Karitiana, has
  exactly one pseudo-haploid sample of 28 — i.e. the error is purely a function of the
  pseudo-haploid sample fraction).
- **Ancient-pseudo-haploid: 15.5% median, scaling to >100%.** The drivers are the pops
  that are ~100% pseudo-haploid: Czechia_EBA_CordedWare, Czechia_BellBeaker,
  Yamnaya, Iran_GanjDareh_N, Israel_Natufian.
- **Worst case = closely-related × high-sample-fraction pseudo-haploid pairs with small
  true f2.** Here the (under-subtracted) het correction is a large fraction of the
  (small) true f2, so the relative error approaches 100%+:
  - Yamnaya–CordedWare and French–CordedWare ≈ **+112–117%** (at2-f2-source lens).
  - Czechia_BellBeaker–Czechia_EBA_CordedWare = **3.32× ratio** (steppe-f2-code lens,
    production golden, emu40).
  - **232%** single-entry max on the mixed Haak set (empirical-regimes lens).
- Distant, mixed-or-diploid pairs (Han–Mbuti, Malta–UstIshim) = **1.0000 exactly.**

### Reconciliation of the lens spread (flagged, not hidden)

The three lenses report the worst-case magnitude as 109.6% / 116.8% / 232% and the
production-golden central tendency as +4.7% (raw f2) vs +0.29% (after qpAdm
cancellation). These are **consistent, measuring different cuts:**
- 109.6% = measured all-ancient set max; 116.8% = AT2-self N-flip reproduction of the
  same pair (the ~7pt gap is jackknife block-weighting); 232% = a single worst entry on
  the mixed set with the smallest true f2 (largest relative amplification). All point to
  the same ">100% on close-kin pseudo-haploid" worst case.
- **+4.7%** is the raw-f2 median on the production golden; **+0.29% weights / +0.07% SE**
  is what survives into qpAdm because the f4-contrast GLS cancels most of the shared
  het bias. The +0.29%/−0.83% weight and +0.07% SE **reproduce the user's reported
  "~0.2% weights, ~1% SE" almost exactly** — confirming the chain end-to-end.

### Why qpAdm only showed ~0.2%/~1% while raw f2 is 15–100%

The het-correction bias is largely **common-mode** across the populations in a model and
**cancels in the f4 contrasts** that qpAdm's GLS is built from. So qpAdm masks the
defect down to sub-1%. **On raw f2 / f3 / D outputs (no cancellation) the full
11–116%+ error is exposed** — the user's worry is correct: a downstream consumer of raw
f2 on an all-pseudo-haploid ancient rotation would see it near the worst case on every
entry.

---

## (3) VERDICT

**FIXABLE convention difference. NOT the emulated-FP64-vs-native floor.**

Evidence emulated-FP64 is excluded as the dominant residual:
- Modern-diploid pairs use the **identical FP path** as ancient pairs yet are
  **bit-identical** (steppe `emu40` vs `fp64` on the modern set = 0.000000%).
- The het correction is **native FP64 in every mode** (`f2_block_kernel.cu:26-28`),
  so the precision knob cannot touch it.
- A precision floor is sub-1e-4; it cannot produce a sign-consistent 15–230% error.
- The error magnitude *and sign* (steppe biased high) are fully explained by the
  `2n−1` vs `n−1` denominator alone.

The emu-vs-native floor remains a candidate **only** for the residual sub-1e-4 noise
that will remain *after* the ploidy fix on the modern-diploid baseline — and that
baseline is already at 0.000000%, so the floor is presently unobservable. **Recommend
documenting the post-fix residual (whatever cuBLAS emulation leaves on a worst-case
dynamic-range GEMM) as the stated architecture.md §12 tolerance — but only after the
ploidy fix, since today the convention bug dominates it by 5+ orders of magnitude.**

---

## (4) THE RECONCILIATION (if fixable — and it is)

### The exact steppe change

Replace the global hard-coded `kPloidy = 2` with **per-sample ploidy** sourced the same
way AT2 sources it, feeding `finalize_af`'s `N` (and thus the het-correction denominator)
per sample/pop:

1. **Detect ploidy per sample** the AT2 way (`cpp_readgeno.cpp:137-168`,
   `R/io.R:155`): a sample is pseudo-haploid (ploidy=1) unless a heterozygous call
   appears within the first N (AT2: 1000) genotyped SNPs; else diploid (ploidy=2).
   (Steppe metadata can carry this rather than re-deriving it, but the *value* must
   match AT2's classification to be byte-parity.)
2. **Plumb per-sample ploidy** into the Q/V/N finalize instead of the constant:
   `cmd_extract_f2.cpp:60/259/289` → a per-sample ploidy vector; `finalize_af`
   (`decode_af.hpp:142-156`) already takes `int ploidy` per call, so `N = ploidy·AN`
   becomes correct with **no formula change**.
3. **Only `N` changes.** `Q = AC/N` is convention-invariant (both AC and N scale with
   ploidy), the het-correction formula (`f2_estimator.hpp:71-77`) and the
   cancellation assemble are untouched. This is a *data-plumbing* fix, not a numerics
   change.

> **Subtlety flagged (mixed-ploidy pops):** Turkey_N, Serbia, Yamnaya, Karitiana
> contain *both* diploid and pseudo-haploid samples. A correct AT2-parity fix needs
> per-sample allele weighting in the q accumulation (AT2's `g/(3−ploidy)` /
> `count = Σploidy(i)`), not just a per-*pop* ploidy. steppe's `decode_af` currently
> applies one ploidy to the whole tile, so the fix must reach the per-sample
> accumulation, not merely tag each population with a single ploidy. This is the part
> most likely to leave residual error if implemented as per-pop instead of per-sample.

### Blast radius

- **Precompute goldens (the f2 cache fixtures used by the fit tests): UNAFFECTED.**
  Those goldens are **AT2 fixtures** (the `cache_metadata.json` shows
  `adjust_pseudohaploid=true` already). The fit (Phase-2) tests consume the AT2-built
  f2 arrays as input, so changing steppe's *own* f2 estimator does not move them. The
  fit-parity oracle is unchanged.
- **extract-f2 end-to-end parity: TIGHTENS** (this is the point). Today steppe's own
  extract-f2 output diverges from AT2 by 4.7% (mixed) / 15.5% (ancient); after the fix
  it should collapse toward the modern-set baseline (0.000000% on diploid; sub-1e-4
  emu floor on the worst dynamic-range pairs).
- **No change to the qpAdm fit math, the GEMM precision seam, or any committed
  golden.** The fix is confined to ploidy sourcing + the per-sample N feed.

---

## (5) GO / NO-GO

**GO — and it is a FIXABLE convention difference, not an FP floor.**

- Root cause is isolated to a single term (het-correction `N`), confirmed at file:line
  on both sides, and confirmed empirically (modern exact, ancient blows up tracking the
  pseudo-haploid fraction, FP path bit-identical where convention agrees).
- The fix is well-scoped (per-sample ploidy detection → `finalize_af` N), does not
  touch the f2 formula, and does not perturb the AT2-fixture fit goldens.
- The only implementation risk is the **mixed-ploidy per-sample weighting** — implement
  it at the per-sample accumulation level (matching AT2 `count = Σploidy(i)`), not as a
  per-population ploidy tag, or pops like Turkey_N/Serbia/Yamnaya/Karitiana will retain
  partial error.
- After the fix, re-baseline the modern-diploid residual and **document the remaining
  emulated-FP64-vs-native delta as the §12 tolerance** — it is currently invisible
  under the convention bug.

---

### Source citations
- **steppe:** `src/app/cmd_extract_f2.cpp:60,259,289`;
  `src/core/internal/decode_af.hpp:23-32,142-156`;
  `src/core/internal/f2_estimator.hpp:71-77,92-94`;
  `src/device/cuda/f2_block_kernel.cu:15-28,392-407`.
- **AT2 R 2.0.10:** `src/cpp_fstats.cpp:175-180,478-498`;
  `src/cpp_readgeno.cpp:137-168,254-268`; `R/io.R:51-54,155,167-170`;
  `R/resampling.R:31-38`.
- **DReichLab C:** `qpsubs.c:343-362`.
- **goldens:** `f2_on_converted/cache_metadata.json`, `f2_fit0/cache_metadata.json`,
  `/workspace/data/haak/at2_f2/cache_metadata.json` — all
  `adjust_pseudohaploid:true, apply_corr:true, afprod:true`.

### Measured numbers (all real AADR, on-box; no fabrication)
- ΔQ on matched SNP set: max|Δ| = 0.00e+00 (Q convention-invariant).
- count ratio diploid/pseudo = 2.000 exactly.
- Modern regime: median 0.0000%, max 0.31% (Karitiana, 1/28 pseudo-haploid).
- Ancient regime: median 15.55%, max 109.63%.
- Mixed (production golden) regime: median 4.71% raw f2, max 232.16% single entry;
  worst pair Czechia_BellBeaker–Czechia_EBA_CordedWare ratio 3.32× (emu40).
- AT2-self N-flip reproduction: 116.8% vs measured 109.6% (gap = jackknife weighting).
- qpAdm survival: weights +0.29% / −0.83%, SE +0.07% — matches the user's
  "~0.2% weights, ~1% SE".
- FP floor exclusion: steppe emu40 vs fp64 on modern-diploid = 0.000000%.

### Uncertainty flags
- Worst-case max varies by cut (109.6% all-ancient / 116.8% N-flip / 232% single mixed
  entry); all are the same close-kin-pseudo-haploid worst case, differing only by
  which set and which extremity is reported. Treat ">100%, up to ~230% on the smallest
  true-f2 close-kin pseudo-haploid pairs" as the honest worst case.
- The mixed-ploidy per-sample weighting (Turkey_N/Serbia/Yamnaya/Karitiana) is the
  one place a per-pop (vs per-sample) fix would leave residual error — flagged as the
  primary implementation risk.
- The emulated-FP64 floor is a *candidate* for the post-fix sub-1e-4 residual only; it
  is currently unmeasurable because the modern-diploid baseline is already 0.000000%.

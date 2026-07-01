# steppe — Deeper Frontiers

Status: **exploration / vision doc.** Sibling to `INDIVIDUAL-LEVEL-FRONTIER.md`. Where that doc
takes the engine one rung *down* — from populations to a single genome — this one takes it one rung
*deeper*: along the chromosome (window-level ancestry heterogeneity), across the X (sex-biased
admixture), between individuals (pseudo-haploid kinship), through time (genetic dating and
selection-through-time), and up against the deep-time / ARG methods that steppe deliberately does
**not** enter. It is concept-first, grounded in steppe's real seams, and organized around one hard
line — the imputation-free / phasing-free lane versus the panel-dependent frontier. Nothing here is
committed scope.

---

## 1. The Vision

f-statistics are **imputation-free and phasing-free**: f2, f3, f4, D, qpAdm, qpWave, DATES and the
f4-ratio all run on pseudo-haploid pileups *directly* — allele-count deltas over a genetic map, no
haplotypes, no reference panel, no GLIMPSE. That single property draws the whole map of this doc.
Everything on steppe's side of the line — ancestry heterogeneity along the genome, X-versus-autosome
sex bias, pseudo-haploid kinship, admixture dating, selection-through-time, the archaic fraction — is
already inside the engine, because **steppe's jackknife blocks *are* genomic windows.**
`block_partition_rule.hpp` walks the SNPs in the AT2 `setblocks` convention (`blgsize = 0.05`
Morgans = 5 cM, reset at every chromosome boundary, ~711 autosome windows on AADR v66), and the
per-window f2 tensor `F2BlockTensor [P×P×n_block]` is *already sitting in VRAM as a stack of
per-window covariances*. Genome-wide f-statistics are just that tensor summed over the block axis;
**stop summing, and per-window and per-chromosome f-stats fall out as a re-reduction, not new math.**
On the far side of the line lives everything that genuinely needs phased haplotypes copied from a
reference panel — per-haplotype local ancestry (RFMix / MOSAIC / ChromoPainter), ROH and IBD
segments (hapROH / ancIBD), low-coverage imputation (GLIMPSE), and ARG/coalescent dating (Relate /
tsdate / MSMC2) — and that is *exactly* where the West-Eurasian reference-panel bias lives. steppe
cannot de-bias those methods; a GPU rewrite changes the wall-clock, not the bias. So its honest offer
is not "everything, faster" — it is the panel-free statistic, at coarser resolution, **labeled as
such**, and a clean handoff (the pseudo-haploid EIGENSTRAT those tools consume) at the wall.

---

## 2. The Areas

### Area 1 — Chromosome & Window-Level: *"local ancestry lite"*

**Vision.** Because f3 and f4 are exact linear contractions of the f2 basis, a *per-window* f4 or
qpAdm fit is the same statistic on one block's slice instead of the pooled sum. That buys a
genome-browser-style ancestry track — this arm 0.7 Yamnaya, that arm 0.3 — where the outlier window
is a candidate for selection, introgression, or an assembly artifact; it localizes gene flow
("the excess lives in one chr2 window, not smeared genome-wide"); and, with one chromosome-mask flip,
it contrasts qpAdm weights on the X against the autosomes. All of it imputation-free, panel-free,
valid on pseudo-haploid pileups — a re-reduction of a tensor already resident.

| Feature | What | Why it delights | Speed | Effort |
|---|---|---|---|---|
| **Ancestry-along-the-genome** | qpAdm per 5-cM window → a track of source weights across all ~711 windows | The outlier arm jumps out as a candidate selected/introgressed locus — straight off the resident `f2_blocks` | ⚡ sub-second | **Low** (reuse) |
| **f4 heterogeneity scan** | Sweep `f4(A,B; C,X)` per window to localize where gene flow concentrates | Rigorous "where does the signal live?" — reuses the quartet sweep, per block not pooled | ⚡ reuses sweep | **Low** |
| **X vs autosome** | The same qpAdm model re-run on X blocks (one mask flip + a 2nd f2 pass) | A source dropping ~0.5 autosomal → ~0.2 on X is the classic sex-bias signal, on pseudo-haploid | ⚡ | **Low–Med** (keep-X flag) |
| **Leave-one-chromosome-out** | Per-chromosome drop-refit | "Is my genome-wide p-value driven by one chromosome?" — latent in the block structure | ⚡ | **Low** |
| **True local ancestry** | RFMix / MOSAIC / Loter / ELAI / ChromoPainter / HAPMIX — per-haplotype tracts | Per-SNP breakpoints for one individual | — | **Out of scope** (phased panel) |

**Honesty rail.** *Local ancestry lite is not local ancestry* — it reports the **average** ancestry
composition of a window across the sampled populations, not per-haplotype tracts with breakpoints. The
binding constraint is not compute, it is the **standard error**: a single 5-cM window *is* one
jackknife block, so you cannot jackknife it against itself; a per-window SE needs sub-block resampling
or a genome-wide-calibrated noise band, never the naive genome-wide SE printed against one window.
Resolution is capped near 5 cM — shrink `blgsize` and the SNP-per-block count collapses, the SE
explodes, the covariance goes non-SPD. And the phasing-dependent lane cannot be de-biased: RFMix
et al. need statistically phased haplotypes copied from a labeled (Euro-leaning) panel; **ELAI** is
the phasing-free exception but still needs true *diploid* calls, so it does not rescue 0.5× pseudo-
haploid; **GLIMPSE** imputation error is *ancestry-correlated* — imputing toward a European panel and
then inferring ancestry is circular for exactly the divergent ancient samples where the question is
interesting. steppe stays in the coarse panel-free window lane, clearly badged.

### Area 2 — Sex-Bias, ROH & Kinship: *the individual-genomics layer*

**Vision.** This is the sharpest test of steppe's founding bet, because it runs straight into the two
methods where the imputation-free lane genuinely dead-ends. The honest map has three bands. **Sex-bias**
is pure f-stat reuse: run the same qpAdm/f4 model twice — once on autosomal SNPs, once on the X — and
compare; a target that is 71% steppe on the autosomes but 48% on the X is a male-biased pulse (males
carry one X, so X-linked ancestry over-weights the female contribution). **Kinship** from
pairwise allele-sharing (READ, TKGWV2) is the aDNA-native way to find families in a cemetery, and it is
a pairwise-mismatch reduction — the same N×N bridge object Area 3 of the parent doc plans. **ROH, IBD,
and Ne-through-time** are the panel wall.

| Feature | What | Why it delights | Speed | Effort |
|---|---|---|---|---|
| **Sex-bias in one command** | `qpadm --split-x`: model on autosomes and on the X, print both weight vectors + the X/A ratio + a jackknife Z | The Goldberg-vs-Lazaridis debate rerun on *your* samples, with honest error bars instead of a headline | ⚡ reuse f2/f4/qpAdm | **Low** |
| **Find the families** | Full pairwise READ/READv2 kinship matrix on-GPU | "I3 & I7 are 1st-degree, parent-offspring; I7 & I12 are 2nd-degree" — pseudo-haploid, no panel, per-window SE free | ⚡⚡ GEMM-shaped | **Med** (shared N×N reduction) |
| **Ultra-low-cov rescue** | A TKGWV2-style mode classifying relatives at ~0.03× via an allele-*frequency* table | Salvages a 0.05× skeleton as the son of a 0.4× one — panel exposure is mild (frequency weights, not haplotypes) | ⚡ | **Med** |
| **ROH / IBD / Ne** | hapROH, ancIBD, IBDNe, HapNe | Inbreeding, IBD segments, effective-size trajectories | — | **Out of scope** (panel wall) |

**Honesty rail.** X-vs-autosome sex-bias is **contested, not settled** — Lazaridis & Reich (2017)
challenged the Goldberg (2017) steppe result; the X carries an order of magnitude fewer SNPs, drifts
faster (Ne_X ≈ ¾ autosomal), and the X/A ratio is *model-dependent* (mapping it to a migrant sex ratio
needs a Goldberg-Rosenberg pulse model). steppe's contribution is a real jackknife *Z*, not a verdict;
and the X needs per-sex ploidy handling plus PAR exclusion or the frequencies are simply wrong.
**ROH is the one item imputation-free cannot reach**: pseudo-haploidization *erases* the heterozygosity
ROH is defined on, so hapROH's 1000G haplotype panel is load-bearing, not a convenience — refusing to
fake an imputation-free ROH is itself the feature. **ancIBD** starts from GLIMPSE imputation + 1000G
phasing before its HMM — the deepest Euro-bias exposure in the whole frontier. The right posture for the
whole third band is *interop, not reimplementation*: steppe already emits the pseudo-haploid EIGENSTRAT
these tools consume, so it hands off cleanly and refuses to launder GLIMPSE/1000G bias into its own
imputation-free output.

### Area 3 — Dating & Chronology

**Vision.** "Dating" hides three different clocks, and blurring them is the fastest way to mislead.
One is **metadata that already exists** — the radiocarbon and contextual dates in the AADR `.anno`.
One is **admixture dating from the DNA** — reading, off the decay of ancestry-linked LD, how many
generations ago two populations mixed — and this is the part steppe already *is*: **DATES**
(Chintalapati/Patterson/Moorjani 2022) ships GPU-native, the ALDER FFT reformulation turning the
~10¹²-pair object into an `O(G log G)` cuFFT autocorrelation, *flat in SNP count*, closed with a
leave-one-chromosome jackknife (golden: PUR ← {CEU, YRI} = 9.742 ± 0.317 generations). And one is the
genuinely hard business of reading a sample's calendar age off its genotypes.

| Feature | What | Why it delights | Speed | Effort |
|---|---|---|---|---|
| **Date this admixture** | DATES: generations-since-admixture ± jackknife SE + the exp-decay curve | Already shipped, GPU-native, imputation-free — the single strongest "no Euro-biased panel needed" story in the survey | ⚡ cuFFT `O(G log G)` | **Low** (exists) |
| **Ancestry through time** | Slice a region's samples by `.anno` date, run the qpAdm rotation per time-bin | Watch source fractions move across a transect — steppe/Yamnaya rising into the Bronze Age | ⚡ batched | **Low–Med** (`.anno` reader) |
| **Two-pulse dating** | A MALDER-style multi-exponential fit on the same decay curve | Resolves two admixture events at different dates — a *fitter* change, not a new kernel | ⚡ | **Med** |
| **Genetic age check** | The recombination clock (Moorjani 2016): DNA-implied calendar age vs the radiocarbon date | Disagreement flags contamination, mislabeling, or a bad context date — imputation-free, reuses the cuFFT engine | ⚡ | **Med** |
| **Split-time from drift** | Read a years estimate off a qpGraph drift edge, with the Ne / gen-time / mutation-rate knobs *visible* | The user *sees* how much of the "date" is model vs data | ⚡ reuse qpGraph | **Low** |
| **Painting / molecular clock** | GLOBETROTTER (phased donor panel); Fu 2014 missing-mutation clock (raw sequence) | — | — | **Out of scope** |

**Honesty rail.** The admixture-dating spine sits on a **third** already-built engine — the cuFFT
ancestry-covariance decay pipeline, distinct from both the f2 cache and the (to-be-built)
covariance/PCA substrate. MALDER identifiability is real: two nearby pulses can be statistically
inseparable (a fit limitation, not a data one). Calendar-age-from-genotypes is the niche weak point —
the recombination clock is imputation-free and reuses steppe's engine but is per-sample noisy and needs
an archaic reference; the accurate missing-mutation clock is defeated by the ascertained 1240k panel and
is out of scope. Drift → years is assumption-laden (Ne, generation time, mutation rate, each stacked
multiplicatively) — report drift and generations as the measurement, treat years as a clearly-labeled
derived quantity. And the AADR dates are heterogeneous (direct radiocarbon vs contextual, with real
SDs); the **genomic** block jackknife does *not* propagate date error, so that has to be surfaced on its
own. GLOBETROTTER's bias is avoided by *not needing haplotypes*, not by fixing them.

### Area 4 — Selection Through Time & Phenotype

**Vision.** This is the frontier where steppe's shape and the AADR's shape line up almost too neatly.
The AADR is a *dated* panel, and `launch_decode_af` (`decode_af_kernel.cu`: `Q = AC/(ploidy·AN)`, an
`N` count, a `V` non-missing mask, laid out `P × M` column-major) already turns any grouping into a
per-group, per-SNP allele-frequency matrix. Point `P` at **time-bins** — a custom `IndPartition` over
`.anno` dates — and the same kernel hands you an allele-frequency *trajectory panel*, resident on the
GPU, for free. This is the **cleanest imputation-free lane in the whole product**: selection scans and
single-locus trait readout run on pseudo-haploid pileups directly. The bias bites in exactly one place
— polygenic scores — and the honest move is to name it, not launder it.

| Feature | What | Why it delights | Speed | Effort |
|---|---|---|---|---|
| **`scan-selection --time-bins`** | One command → a genome-wide, jackknife-backed per-SNP selection table | The Mathieson 2015 scan, imputation-free, at sweep speed — LCT/MCM6, SLC24A5, SLC45A2, HERC2, FADS, the TLR cluster rise to the top | ⚡⚡ SNP-parallel (~14.5M/s shape) | **Low–Med** (one per-SNP kernel + `.anno` reader) |
| **Trajectory view** | Pick rs4988235, watch lactase persistence climb across dated bins, per region | Each point a real per-bin frequency with a block-jackknife error bar | ⚡ | **Low–Med** |
| **`readout <genome>`** | A phenotype card: lactase, pigmentation, earwax, alcohol-flush, diet FADS | Imputation-free single-locus lookups; an explicit low-coverage warning when depth can't call a het | ⚡ | **Low** (rsID catalog / IO) |
| **`bmws`-style `s(t)`** | An HMM over the AF time series → a per-SNP, time-resolved selection coefficient | Upgrades "was this selected?" to "how hard, and when?" — Bronze-Age Britain's vitamin-D story | ⚡ SNP-parallel | **High** (new HMM engine) |
| **Qx / PRS guardrail** | Qx (Berg & Coop 2014) on steppe's *own* imputation-free covariance, plus the caveat | Refuses to render a Euro-biased ancient-phenotype PRS as a verdict | ⚡ reuse f2 covariance | **Low–Med** |

**Honesty rail.** The scan is a *small new per-SNP test kernel on substrate steppe already has* — not
the f2/f4 contraction, not the covariance/PCA substrate — one thread/warp per SNP, time-bins in
registers, exactly the SNP-parallel shape the fstat-sweep already runs GPU-compute-bound. Caveats a
literate user will demand: a pseudo-haploid ancient sample shows one random read per site, so a *single*
trait call under-calls heterozygotes (fine for an AF average, risky for a per-individual verdict);
selection scans confound true sweeps with unmodeled admixture (a naive trend test must say so); time-bin
width and region grouping are researcher degrees of freedom; ~1.2M SNPs demand a genome-wide threshold.
And **PRS is the Euro-bias trap in another costume**: European-GWAS effect sizes port poorly (Martin
2019), and the ancient-height-selection claims (Mathieson 2015; Robinson 2015) **largely collapsed**
under stratification-controlled UK Biobank / sib estimates (Berg 2019; Sohail 2019) — Qx can be computed
correctly on steppe's honest covariance and *still* be garbage-in if the effect sizes carry residual
stratification. Note too that several canonical "hits" (EDAR, ABCC11) are region-specific, so a global
scan without regional stratification will mislead about where and when selection acted.

### Area 5 — Deep-Time / ARG & Individual QC

**Vision.** This frontier splits cleanly, and the split is the whole point. On one side sit the
**genealogical / coalescent** engines that reconstruct history down to individual tree nodes — allele
ages, TMRCAs, coalescence rates, population split times: Relate, tsinfer + tsdate, ARGweaver / SINGER,
MSMC2 / PSMC. The payoff is genuine and deep-time and *unreachable from f-statistics* — but every one of
them consumes **phased haplotypes and/or imputation against a West-Eurasian-leaning panel**, so a GPU
rewrite would *launder* the bias, not dissolve it. On the other side sits **individual QC** —
contamination, molecular sex, uniparental markers, damage — which is imputation-free, pseudo-haploid-
native, mostly *not GPU compute at all*, and largely already precomputed in the `.anno`. The honest
posture is symmetric: **decline the first group with a pointer; ingest the second as cheap metadata.**

| Feature | What | Why it delights | Speed | Effort |
|---|---|---|---|---|
| **Pre-run QC gate** | Read the `.anno` (contamination, molecular sex, coverage) and flag before any math | "I2345: 8.1% X-contamination (male); I2350: molecular sex F contradicts label M; I2360: 0.04× coverage" — no bad genome slips into a fit | ⚡ metadata | **Low** (ingest) |
| **Uniparental sanity strip** | mtDNA + Y haplogroup (from `.anno`) beside each qpAdm/f3 result | "Y-hg R1b, mt-hg H, consistent with the Yamnaya-source model" — a cheap cladality prior + sex-bias cross-check | ⚡ | **Low** |
| **Autosome-vs-X f4** | Off the existing block partition | A sex-biased-admixture hint without touching an ARG or a phased haplotype | ⚡ | **Low** |
| **The honest decline** | "When did they split?" → "that's MSMC2/Relate, and both phase against a Euro-biased panel; for admixture *timing* I'll run DATES now, imputation-free" | Names the scope wall instead of faking a coalescent answer | — | **Out of scope** + pointer |
| **ARG / coalescent** | Relate, tsinfer+tsdate, ARGweaver/SINGER, MSMC2, ancIBD | Allele ages, node dates, split times | — | **Out of scope** (phased panel) |

**Honesty rail.** The deep-time engines are not a matter of engineering effort — they are gated on
phasing/imputation, an SMC/HMM/MCMC compute shape that reuses *nothing* from the f2 contraction or the
SYRK/GEMM covariance substrate. Don't conflate "when did they *split*" (MSMC2's rCCR) with "when did they
*mix*" (DATES) — steppe answers the second, honestly and panel-free. Individual QC is fundamentally
BAM/pileup-level and *upstream* of steppe's genotype matrix; steppe should **ingest** it as provenance,
not claim a sequencing-QC pipeline it does not own. `hapROH`/`hapCon` and `DICE` are the pseudo-haploid-
friendly boundary case — aDNA-native and validated on non-European samples, but still **panel-only**
(1000G / reference frequencies), so classify them as panel-dependent, not fully independent.

---

## 3. What we're still missing

A completeness pass over both frontier docs surfaced eight gaps. Two are the cheapest wins in either
doc — *an already-shipped primitive that no doc exposes*.

- **Archaic (Neanderthal / Denisovan) ancestry per individual.** "What's my Neanderthal percentage?" is
  the single most-asked consumer question and a standard aDNA sanity check, yet neither doc mentions it —
  even though the workhorse estimator is the **f4-ratio** (Reich 2009/2010), `α = f4(X,Chimp;Archaic,Y) /
  f4(H,Chimp;Archaic,Y)`, a genome-wide fraction with a real jackknife SE, and **steppe already ships it**
  (`src/device/cuda/ratio_block_jackknife_kernel.cu`, exposed as the `f4ratio` binding). Pure family-B
  reuse with the individual wrapped as a size-1 pop; per-window archaic enrichment is nearly free on the
  jackknife blocks. Segment-level callers (S* / S-prime / IBDmix) need diploid calls and are a separate
  HMM family — out of scope.
- **Formal f3-admixture test & directionality.** The parent covers outgroup-f3 (affinity), f4/D
  (proxy), and qpAdm (modeling) but omits the classic *formal* admixture test: in-group `f3(X;A,B) < 0`
  is unambiguous proof X is admixed. Same `f2_blocks` contraction + jackknife already shipped, just the
  in-group sign convention and a Z-test — currently unsurfaced. A *population*-level test (the
  heterozygosity correction makes it fragile for one pseudo-haploid individual); directionality proper is
  largely covered by the shipped qpGraph + f4.
- **Diversity, heterozygosity & FST.** Every f2/f3 already contains a heterozygosity correction, yet the
  diversity layer is unexposed: per-population expected heterozygosity `h = mean 2p(1−p)`, within-group
  diversity, and pairwise **Hudson FST** (a ratio of f2-like numerator/denominator quantities). A trivial
  reduction over the AF matrix + a ratio-jackknife the `f4ratio` kernel already implements; per-window
  FST is free on the blocks. Honest limit: per-*individual* observed heterozygosity is **not** recoverable
  from a pseudo-haploid genome (one allele sampled per site).
- **Temporal continuity vs replacement.** The defining question of a dated panel — is the population at
  T2 the direct descendant of T1, or was it replaced? — is a distinct test neither doc frames: model the
  later bin as 100% the earlier one and ask whether the qpAdm/qpWave p-value survives (Bell Beaker
  replacement, Anatolian-farmer spread). Pure family-B reuse driven by a date-binned `IndPartition`; the
  blocks even let you ask *where* continuity breaks.
- **Genotype-likelihood-aware f-statistics** — *the honest "imputation done right."* The user distrusts
  imputation for good reason, but the docs' only answer is "stay pseudo-haploid." There is a third path:
  consume genotype **likelihoods** (ANGSD/BEAGLE GLs, ATLAS) directly and propagate per-site uncertainty
  into the f-statistics, instead of hard-calling into a Euro-biased panel *or* collapsing to one random
  read. Panel-free by construction (GLs come from the reads themselves), it strictly dominates
  pseudo-haploid above ~1× without ever phasing — the principled middle ground. An **input-layer** change
  (a GL reader + a GL-aware decode variant), then the entire f2/f4/qpAdm + jackknife stack is reused; a
  real numerics project (variance propagation through the f2 basis), not free. The most intellectually
  important item here.
- **Geographic / spatial provenance** — "where on the map is this genome from?" EEMS-style migration
  surfaces ride directly on the N×N individual-dissimilarity matrix Area 3 of the parent already plans,
  and a cheap k-nearest-neighbor guess is nearly free from the outgroup-f3 ranking (average the
  coordinates of the top-k closest ancient samples). Locator's neural net and SpaceMix's MCMC are separate
  engine families, out of scope. Distinct bias vector: **reference-geography imbalance**, not a phasing
  panel — a Euro-dense reference confidently mis-places under-sampled regions, and steppe cannot de-bias a
  lopsided map.
- **Mutational spectrum through time** (3-mer signatures, Harris & Pritchard 2017 — the European
  TCC→TTC excess). A distinct axis of history orthogonal to drift, but with a hard trap: the 1240k panel
  is an **ascertained** SNP set, and ascertainment grossly distorts the spectrum. Largely out of scope
  *and* actively misleading on capture data — worth naming precisely so users are warned off running it on
  1240k.
- **SFS-based demographic inference** (dadi, moments, fastsimcoal2, momi2 — split times in years,
  migration rates, Ne trajectories). The "put numbers on the model" step, entirely absent from both docs —
  but the raw SFS is *even more* ascertainment-fragile than f-statistics, which is precisely why the field
  leans on f-statistics for capture data. The honest framing is the reverse: steppe's f2/f3/f4 are the
  **ascertainment-tolerant substitute** for raw-SFS demography on ascertained aDNA. Out of scope as an
  engine.

---

## 4. Impact × Feasibility

### Instant wins (high impact, low effort) — *do these first*
- **Per-window & per-chromosome f-stats + qpAdm** ("local ancestry lite"). A re-reduction of the resident
  `F2BlockTensor` — no new kernels.
- **X-vs-autosome sex-bias** (`--split-x`). qpAdm run twice; the work is a keep-X flag + a second f2 pass.
- **Selection-through-time scan** (`--time-bins`). One small per-SNP kernel on the existing decode +
  block jackknife, plus the `.anno` reader — imputation-free at sweep speed.
- **Archaic ancestry via the *shipped* f4-ratio**, and the **formal f3-admixture test** — both are
  already-built primitives no doc surfaces; wrap a size-1 pop / flip a sign convention.
- **Temporal continuity test** and **diversity / heterozygosity / FST** — pure family-B reuse, nearly free
  on the AF matrix and the blocks.
- **Pseudo-haploid kinship** (READ/READv2/TKGWV2). The statistic is imputation-free engine reuse; it rides
  the one shared N×N allele-sharing reduction (see Big bets) and is nearly free once that lands.

### Big bets (high impact, high effort) — *the differentiators*
- **The GPU covariance / distance substrate** (Substrate B, inherited from the parent doc). One
  GEMM-shaped all-pairs f2 / 1−IBS reduction unlocks kinship matrices, EEMS geographic surfaces, and the
  parent's PCA/clustering together.
- **Genotype-likelihood-aware f-statistics.** The honest answer to imputation skepticism: use deeper
  coverage where it exists, panel-free, by propagating per-site uncertainty into f2. Input-layer + a real
  numerics project.
- **The `.anno` metadata / dates layer** feeding time transects, continuity tests, and ancestry-through-
  time (cross-link `EXPERIENCE-PLAN.md` Area 4).
- **`bmws`-style time-varying `s(t)`** — a new SNP-parallel HMM engine; and **MALDER multi-pulse** +
  **recombination-clock calendar dating** as fitter/reference extensions on the shipped cuFFT DATES engine.

### Steady value (medium impact, low–medium effort)
- **The N×N kinship / distance matrix** (the bridge object feeding kinship, EEMS, PCA).
- **Per-chromosome selection & continuity maps** (free off the block partition).
- **The phenotype-card readout** and the **uniparental QC strip** (IO plumbing + `.anno` ingest).
- **k-NN geographic guess** from the outgroup-f3 ranking.

### Lower priority / dependency-gated
- **True local ancestry** (RFMix / MOSAIC / Loter / ELAI / ChromoPainter / HAPMIX) — phased panel, new
  engine family, out of scope.
- **ROH / IBD / Ne** (hapROH, ancIBD, IBDNe, HapNe) and the **ARG / coalescent** engines (Relate, tsinfer
  + tsdate, ARGweaver / SINGER, MSMC2) — phasing/imputation, Euro-biased, out of scope; interop, not
  reimplementation.
- **PRS as a verdict**, **GLOBETROTTER painting dates**, the **missing-mutation molecular clock**, and
  **mutation-spectrum / raw-SFS demography** on 1240k — out of scope, and the last two are actively
  broken by ascertainment.

---

## 5. The imputation-free dividing line

The differentiator, stated sharply. For each question: what steppe answers **without a panel**
(imputation-free / phasing-free), what **genuinely needs phasing/imputation** (and inherits the
West-Eurasian panel bias), and — where one exists — the imputation-free alternative that answers most of
the same question.

| Question | steppe answers WITHOUT a panel | genuinely needs phasing / imputation (Euro-biased) | the imputation-free alternative |
|---|---|---|---|
| Ancestry along the genome | window qpAdm/f4 — the 5-cM window **average** | per-haplotype **tracts** with breakpoints (RFMix, MOSAIC, ELAI, ChromoPainter) | the coarse window profile *is* the panel-free "where does the signal live" answer |
| Sex-biased admixture | X-vs-autosome qpAdm/f4, pseudo-haploid | — | (nothing here needs a panel) |
| Kinship / families | READ / READv2 / TKGWV2 pairwise allele-sharing | — | READ *is* the panel-free field standard |
| Inbreeding / ROH / recent Ne | — (pseudo-haploidization erases heterozygosity) | hapROH (Li–Stephens copying vs 1000G) | windowed-heterozygosity ROH, but only on the rare *true-diploid* high-cov samples |
| IBD segments / recent co-ancestry | — | ancIBD (GLIMPSE→1000G + phasing + HMM), hap-IBD | — (no panel-free route on low-cov aDNA) |
| When did they **mix**? | DATES / ALDER / MALDER LD-decay (cuFFT) | GLOBETROTTER painting dates (phased donor panel) | DATES *is* the panel-free answer |
| When did they **split**? | drift edges (qpGraph) → generations | MSMC2 / PSMC rCCR (phased, high-cov) | drift/generations is the panel-free measurement; years is model-heavy |
| Calendar age from DNA | recombination clock (Moorjani 2016, cuFFT + archaic ref) | missing-mutation molecular clock (raw sequence) | the recombination clock — noisy, niche, but panel-free |
| Selection through time | Mathieson time-stratified scan + `bmws` `s(t)` | — | fully panel-free |
| Trait genotype (lactase, pigment…) | single-locus readout (pseudo-haploid, low-cov caveat) | — | direct lookup |
| Predict a polygenic phenotype (height…) | Qx on steppe's own covariance | PRS needs a Euro-biased GWAS effect panel + imputation | Qx *with* portability/stratification caveats — never a verdict |
| Archaic % (Neanderthal / Denisovan) | **f4-ratio (shipped)**, needs an archaic genome + chimp | segment-level (S* / S-prime / IBDmix, diploid) | the genome-wide fraction *is* the panel-free answer |
| Allele ages / TMRCA / node dates | — | Relate / tsinfer+tsdate / ARGweaver (phased) | — (no f-stat substitute) |
| Where on the map? | k-NN from outgroup-f3; EEMS on the N×N matrix | Locator (NN), SpaceMix (MCMC) | k-NN / EEMS — panel-free, but reference-*geography*-biased |
| Demographic parameters (Ne(t), migration) | f2/f3/f4 order the model qualitatively | dadi / moments / fastsimcoal2 raw-SFS | f-statistics *are* the ascertainment-tolerant substitute on capture data |

The rule the doc keeps returning to: **the bias is avoided by not needing haplotypes, not by fixing them.**
Every sub-genome answer should carry a badge — *imputation-free / panel-free* versus *needs a phased
reference panel (Euro-biased, unavailable here)* — so the user is never handed a window-average as a
tract map, or a panel-conditioned result without the flag.

---

## 6. Wild Ideas Worth Remembering

- **Selection-through-time, as a movie.** Press play and watch LCT-13910\*T climb across the dated bins,
  region by region — each frame a real jackknife-backed per-bin frequency, the Neolithic-to-now sweep
  rendered as animation, straight off the AF trajectory panel.
- **The dated-panel time-map.** An EEMS-style effective-migration surface computed *per time-slice* off
  the N×N distance matrix — watch the migration landscape of Europe redraw itself as the panel is filtered
  by date.
- **Per-individual admixture dating.** Run DATES on a single genome — "your steppe ancestry entered the
  line ~160 generations ago, ±" — one person, one number, a jackknife SE, no phasing.
- **Your Neanderthal number, done right.** The *shipped* f4-ratio on a kit as a size-1 pop — a real
  genome-wide archaic fraction with a jackknife SE, not a black-box percentage.
- **The genetic clock vs the carbon clock.** Run the recombination clock on every AADR sample and plot
  DNA-age against radiocarbon-age — the diagonal is the calibration, the outliers are the contaminated,
  mislabeled, or badly-contextualized samples.
- **The continuity map.** Per-region temporal-continuity qpWave across adjacent time-bins — a heat-map of
  where a population is the direct descendant of its predecessor and where it was replaced.
- **The honest imputation upgrade.** A genotype-likelihood decode path that *uses* deeper coverage where
  it exists — propagating per-site uncertainty into f2 — without ever touching a Euro-biased panel: the
  principled middle between pseudo-haploid and GLIMPSE.

---

## 7. Architecture & cross-links

**The concrete seams (the engine the research named):**

| Capability | Seam | File | Reuse vs new |
|---|---|---|---|
| Genomic 5-cM windows / chrom mask | `setblocks` walk (`blgsize=0.05`, per-chrom reset, ~711 blocks) | `src/core/domain/block_partition_rule.hpp` | reuse (windows already exist) |
| Per-window f2 tensor | `F2BlockTensor [P×P×n_block]`, device-resident | the `f2_blocks` cache | reuse (stop summing the block axis) |
| Time-bin / X regroup | custom `IndPartition` over `.anno` dates; `autosomes_only` mask flip for X | `src/io/ind_reader.{hpp,cpp}`; `autosomes_only` (`src/core/config/cli_args.hpp`); `decode_af_kernel.cu` | reuse + a keep-X flag |
| Time-bin AF trajectory | `P×M` allele-frequency matrix (`Q=AC/(ploidy·AN)`, `N`, `V`-mask) | `src/device/cuda/decode_af_kernel.cu` | reuse (set `P` = time-bins) |
| Archaic % / f4-ratio | on-device ratio block jackknife — **already shipped** | `src/device/cuda/ratio_block_jackknife_kernel.cu` (the `f4ratio` binding) | reuse (surface a shipped primitive) |
| Pairwise kinship / distance | all-pairs f2 / 1−IBS over per-individual genotypes | `src/io/genotype_tile.hpp` (`.packed` + `pop_offsets`) | **new** (Substrate B bridge object) |
| Per-SNP selection test | SNP-parallel per-column reduction over the AF matrix | new kernel over the `decode_af_kernel.cu` output | **new** (small kernel, existing substrate) |
| `.anno` dates + QC metadata | `Date_mean_in_BP`, SD, sex, mt/Y haplogroup, contamination, coverage | new `.anno` reader | **new** (IO plumbing) |
| Genotype-likelihood decode | GL-aware AF + variance propagation into f2 | new decode variant | **new** (input-layer + numerics) |
| ARG / coalescent / local-ancestry / ROH-IBD | — | — | **out of scope** (phased panel) |

**Sibling docs.** `INDIVIDUAL-LEVEL-FRONTIER.md` — the parent: the n=1 modeling / best-proxy / outlier /
clustering rungs and Substrates A/B/C this doc builds directly on (Substrate B's N×N distance object is
the shared dependency of this doc's kinship, EEMS, and the parent's PCA). `EXPERIENCE-PLAN.md` — **Area 4**
(the population catalog + the metadata/dates layer these transects consume) and the **DATES** tool
(admixture dating, the spine of Area 3 here), plus Area 6 (the STUDIO front-end for the selection movie and
the personal time-map). `DATASET-COMPATIBILITY.md` — the `--strand-mode` / palindromic-SNP ground truth a
consumer-kit merge hits. `USER-PAIN-WISHLIST.md` — the on-ramp for bringing your own data in. Deeper
engineering design belongs in `docs/design/` when a feature graduates from this doc to scope.

**Method provenance (external references, not internal steppe docs).** *Local ancestry / phasing:* RFMix,
MOSAIC, Loter, ELAI, ChromoPainter / fineSTRUCTURE (Lawson 2012), HAPMIX, GLIMPSE / GLIMPSE2. *Sex-bias &
kinship:* Goldberg 2017 & Goldberg-Rosenberg 2015, Lazaridis & Reich 2017; READ (Monroy Kuhn 2018),
READv2 (Alaçamlı 2024), TKGWV2 (Fernandes 2021), lcMLkin (Lipatov 2015), KIN (Popli 2023). *ROH / IBD /
Ne:* hapROH / hapCon (Ringbauer 2021; Huang & Ringbauer 2022), ancIBD (Ringbauer 2024), hap-IBD / IBDseq
(Browning), IBDNe (Browning & Browning 2015), HapNe (Fournier 2023). *Dating:* DATES (Chintalapati 2022),
ALDER (Loh 2013), ROLLOFF (Moorjani 2011), MALDER (Pickrell 2014), GLOBETROTTER (Hellenthal 2014),
recombination clock (Moorjani 2016), Fu 2014 (Ust'-Ishim). *Selection & phenotype:* Mathieson 2015, bmws
(Mathieson & Terhorst 2022) & Le 2023, WFABC / ApproxWF, Qx (Berg & Coop 2014; Berg 2019; Sohail 2019),
Martin 2019. *ARG / coalescent:* Relate (Speidel 2019), tsinfer + tsdate (Kelleher 2019; Wohns 2022;
Speidel 2024), ARGweaver (Rasmussen 2014; Hubisz 2020), SINGER (Deng 2024), MSMC2 / PSMC (Schiffels &
Durbin 2014; Li & Durbin 2011). *Archaic segments:* Reich 2009/2010 (f4-ratio), S* (Vernot & Akey),
S-prime (Browning 2018), IBDmix (Chen 2020). *Spatial & demography:* Locator (Battey 2020), EEMS (Petkova
2016), SpaceMix; dadi / moments / fastsimcoal2 / momi2; Harris & Pritchard 2017. *Individual QC:* ANGSD
X-contamination (Rasmussen 2011), contamMix (Fu 2013), Schmutzi (Renaud 2015), DICE (Racimo 2016),
AuthentiCT (Peyrégne & Peter 2020), mapDamage2 (Jónsson 2013), PMDtools (Skoglund 2014), Rx/Ry (Mittnik
2016; Skoglund 2013), HaploGrep, yhaplo, pathPhynder. *f-statistic foundations:* Peter 2016.

# steppe — The Individual-Level Frontier

Status: **exploration / vision doc.** Sibling to `EXPERIENCE-PLAN.md`. Where the experience
plan is about the *population-level* workflow surface (qpAdm rotation, f4 sweeps, qpGraph),
this doc explores the features that operate one rung lower — on the **individual**: modeling
your own genome, ranking the populations closest to a sample, flagging individuals that don't
belong to their label, and building data-driven populations from genetic similarity instead of
trusting the `.anno`. It is concept-first, but grounded in steppe's real seams and in what
population genetics actually does. Nothing here is committed scope.

---

## 1. The Vision

Today steppe answers questions about *populations* you hand it. The individual-level frontier
is the same engine pointed one level down: **"here is one genome — where does it sit?"** Point
steppe at a single ancient sample, or at a consumer DNA kit, and get back a qpAdm model of that
one person as a mixture of ancient sources — with a real p-value and jackknife SEs, not a
distance heuristic. Ask "which of these 500 populations is genetically closest to this sample?"
and get a ranked list in the time it takes the f3 sweep to run. Ask "does this individual
actually belong to the population it's labeled with?" and get a cladality test, not a hunch.
And when the curated labels fail you, cluster the individuals themselves and let steppe *propose*
populations from the data — then immediately re-test them with f-statistics so the grouping and
the inference stay honest. The unlock is the same one that makes the rest of steppe feel alive:
when a qpAdm fit is ~1 s and a full quartet sweep is on-device, individual-level analysis stops
being a batch job and becomes something you *interrogate*.

The whole space splits along one architectural fault line, and the doc is organized around it:

- **Reuses the engine steppe already has.** f3 and f4 are exact linear contractions of the
  pairwise-f2 basis — `f3(C;A,B) = ½[f2(A,C)+f2(B,C)−f2(A,B)]`,
  `f4(A,B;C,D) = ½[f2(A,D)+f2(B,C)−f2(A,C)−f2(B,D)]`. So *anything* posed as an f3/f4/qpAdm
  question about an individual is the device-resident `f2_blocks` cache + the block jackknife +
  the fit engine, with the individual wrapped as a population of size 1. **No new kernels.**
- **Needs a new (but GPU-native) substrate.** PCA, projection, individual×individual distance
  matrices, and clustering are a *different* object — genotype covariance and its
  eigendecomposition. steppe has **none** of this today (only a host-side Jacobi SVD used by the
  qpAdm ALS solve). But it is the same GEMM/SYRK shape as the f2 covariance path, so it is a
  natural — if genuinely new — GPU build.

Read the four Areas as "what the user gets"; read §3 as "what it costs to build," which is
mostly a function of which side of that fault line each feature lands on.

---

## 2. The Areas

### Area 1 — Personal-Genome Projection: *bring your own DNA*

**Vision.** The consumer-facing dream and the aDNA researcher's daily need are the same
operation: take one genome and model it against the ancient world. For an amateur it's "am I
Yamnaya or Anatolian, and how much?" For a researcher it's "model this newly sequenced
individual as a mixture of my candidate sources, with a p-value." steppe can serve both from one
path — because a single individual is just a population with one member, and the fit engine
already handles that.

| Feature | What | Why it delights | Speed | Effort |
|---|---|---|---|---|
| **Model *me*** | Wrap one individual as a size-1 target, run qpAdm against ancient sources | A real model with weights + p-value + SE, not a distance guess — "0.74 Yamnaya / 0.26 Anatolia_N, p=0.31" | ⚡ ~1 s fit | **Low** (reuse) |
| **Import your kit** | A reader for 23andMe / AncestryDNA raw text + VCF, merged onto the AADR panel by SNP intersection | Turns a 30–60 min convertf+merge chore into seconds (USER-PAIN-WISHLIST §W5) | ⚡ | **Med** (new reader + merge) |
| **Instant triage** | G25/Vahaduo-style: project the sample into a PCA space, rank + NNLS-model by Euclidean distance | Millisecond interactive "closest source" feedback before you commit to a real fit | ⚡⚡ batched | **High** (needs §3 PCA substrate) |
| **Honest confidence** | Block-jackknife SE on the weights; a rejected model *says so* | The thing G25/Vahaduo can't give you: a fit that can fail | ⚡ | **Low** (reuse) |

**Grounding.** The n=1 path is almost free: `IndPartition` / `PopGroup` (`src/io/ind_reader.hpp`)
already map individuals to populations by `.ind` column 3, and there is **no minimum-size check**
anywhere in the decode kernel or the f-stat code. Build a `PopGroup{label:"MyKit", rows:{i}}`,
let `decode_af_kernel.cu` compute its allele frequencies exactly as for any population, and it
flows into the existing `qpadm` rank/GLS solve with `L = n+1`. The math backs this up: Harney et
al. 2021 ("Assessing the Performance of qpAdm") explicitly stress-tested single-individual
targets at up to 90% missingness. f4 is an average of products over many SNPs, so a single
noisy `p̂ ∈ {0,½,1}` leaves it **unbiased in expectation** — the variance inflates, the SE grows,
but the estimator is honest (see §3, "the honesty rails"). The *triage* row is the one that costs:
G25 coordinates are `smartpca` projection (`lsqproject` for missingness, `shrinkmode` for the
projection-toward-origin bias) followed by a simplex-constrained NNLS per model — a fast
heuristic that needs the PCA substrate of §3, not the f2 engine. The consumer reader plugs into
`resolve_genotype_triple` / the format dispatch in `src/io/genotype_source.hpp` with a new
`GenoFormat` enum value; the AADR merge is a SNP-set intersection with strand handling (cross-link
`DATASET-COMPATIBILITY.md` — the `--strand-mode` question is exactly the palindromic-SNP problem a
23andMe merge hits).

### Area 2 — Best-Proxy & Closest-Population Search

**Vision.** "What's closest to this?" is the most-asked question in the field, and steppe is
built to answer it at sweep scale. Rank every population in the panel by genetic affinity to a
target; settle "is X a better proxy than Y?" with a symmetry test and a Z-score; rotate a pool of
sources to find which ones actually fit; and — the money feature — given a working model, find the
single **best replacement** source from a pool by refitting and ranking on model improvement. All
of it is the f-statistic engine steppe already ships.

| Feature | What | Why it delights | Speed | Effort |
|---|---|---|---|---|
| **Closest populations** | `outgroup-f3(O; T, X)` for all X, ranked descending | One command → "the 20 populations that share the most drift with this sample" | ⚡ reuses `f3-sweep` | **Low** |
| **Better proxy?** | `f4(T, O; X, Y)` / D-stat symmetry, Z-score arbiter | Rigorous "X beats Y" instead of eyeballing a PCA | ⚡ reuses `f4-sweep` | **Low** |
| **Which sources fit** | qpAdm rotation over a source pool | Feasible-model discovery, already wired as `qpadm-rotate` | ⚡ batched fit | **Low** (exists) |
| **Best replacement** | Swap each pool member into a slot, refit, rank by p-value gain / max residual | "Replace Turkey_N with *this* and the model goes from p=0.02 to p=0.41" | ⚡ batched refits | **Med** |

**Grounding.** This entire Area is family-B reuse. Outgroup-f3 ranking is a batched f3 — a linear
contraction of the `f2_blocks` cache — and `f3-sweep` already computes C(P,3) on-device. The
symmetry test is the existing all-quartets `f4-sweep` (the STPFST1 enumeration, GPU-compute-bound
at ~14.5M quartets/s). Rotation is `qpadm-rotate` verbatim; best-replacement is the same batched
fit engine with a ranking pass on the outputs. The one caveat to surface in the UI: outgroup-f3
**conflates shared drift with drift private to X** (Peter 2016) — a genuinely close but heavily
bottlenecked or single-sample population can rank *lower* than it should — so "closest by f3" is a
strong first pass that the f4/D symmetry test confirms. This Area is the strongest argument for
the memory's *finish-the-fit-engine-backend-first* sequencing and it lives squarely inside the S8
rotation envelope (thousands of models, f2 device-resident, fit solve batched) the design already
targets.

### Area 3 — Per-Individual Statistics & Outlier Detection

**Vision.** Before any population-level result is trustworthy, the individuals have to actually
belong to their labels. steppe can make QC a first-class, fast operation: flag the individuals
that sit too far from their group, test whether a sample is truly cladal with the population it's
assigned to, and render the individual×individual similarity matrix that everything else is built
on.

| Feature | What | Why it delights | Speed | Effort |
|---|---|---|---|---|
| **Does this sample belong?** | Per-individual `f4`-vs-centroid + pairwise-qpWave cladality vs the group | Catches contaminated / mislabeled / outlier samples with a test, not a plot | ⚡ reuses f4 + qpWave | **Low–Med** |
| **Flag the outliers** | `smartpca`-style iterative σ-threshold removal on the top PCs | The standard aDNA QC pass, on-GPU and interactive | ⚡ | **High** (needs §3 PCA) |
| **Who's similar to whom** | The full N×N pairwise distance matrix (all-pairs f2 or 1−IBS) | The most GPU-friendly object in pop-gen — one big reduction; feeds PCA + clustering | ⚡⚡ GEMM-shaped | **Med** |

**Grounding.** The cladality/centroid tests are family-B reuse — an individual as a size-1
population against a set of outgroups, through the existing f4/qpWave machinery. The outlier
iteration is the `smartpca` loop (`outliersigmathresh` default 6σ, `numoutlierevec` 10,
`numoutlieriter` 5): covariance → eig → drop → recompute, which needs the §3 PCA substrate. The
N×N distance matrix is the bridge object — an all-pairs f2 or allele-sharing (1−IBS) reduction is
a single GEMM-shaped kernel over the per-individual genotypes that `GenotypeTile`
(`src/io/genotype_tile.hpp`) already holds packed and pop-contiguous. Honesty rail (§3): every
pairwise entry is computed only on SNPs non-missing in *both* samples, so under heterogeneous aDNA
missingness different matrix cells rest on different SNP sets — the resulting distances can
violate the triangle inequality, and the UI should not pretend the matrix is metric.

### Area 4 — Custom Population Discovery by Genetic Similarity

**Vision.** The `.anno` labels are archaeology's best guess, not ground truth. Sometimes you want
the data to speak: cluster the individuals by genetic similarity, form candidate populations from
the clusters, and — crucially — hand those candidates straight back to the f-statistic engine to
*validate* them on independent criteria. steppe can make the propose→test loop tight enough to be
a genuine analysis mode, not a two-day detour.

| Feature | What | Why it delights | Speed | Effort |
|---|---|---|---|---|
| **Regroup by hand** | Build a custom `IndPartition` — any individuals → any labels | Redefine "populations" without touching the input files | ⚡ | **Low** (plumbing) |
| **Cluster into pops** | PCA + k-means / hierarchical on the top PCs → data-driven groups | "Here are 8 clusters the data actually supports" | ⚡ | **High** (needs §3 PCA) |
| **Split a messy label** | Cluster *within* one heterogeneous label to expose sub-structure | Find the outlier subgroup hiding inside "France_N" | ⚡ | **Med** (on §3) |
| **Propose → validate** | Auto-hand clusters to qpAdm/qpWave, report which survive as coherent pops | Closes the circularity gap the field warns about | ⚡ | **Med** (on the loop) |

**Grounding.** The regrouping seam is exact and cheap: `read_ind()` (`src/io/ind_reader.cpp`)
sorts groups by label and hands `PopGroup{label, rows}` to the decode; substitute a custom
`IndPartition` and every downstream stage (decode → f2 → qpAdm) just works on the new grouping.
The clustering itself is PCA+k-means on the §3 substrate (covariance/eig + iterative GEMM+argmin).
The load-bearing caution — and the reason the *propose→validate* row exists — is **circularity**:
if you cluster individuals by genetic similarity and then run f-statistics on those same clusters,
the grouping and the test are not independent and the significance is inflated. The field's answer
is exactly the loop above — use clustering as a *hypothesis generator* and re-test on independent
criteria — which steppe is unusually well-placed to automate. Explicitly out of scope for the
in-engine path: ADMIXTURE (a separate constrained-optimization SQP kernel family — better left to
the existing binary) and fineSTRUCTURE/ChromoPainter (sequential HMM painting + MCMC on phased
haplotypes — poorly GPU-batchable, and wrong data shape for sparse pseudo-haploid aDNA).

---

## 3. The Enabling Substrate — what it actually costs

Two compute families and one IO family carry everything above.

**Substrate A — the f2/f4/qpAdm reuse path (already built).** Because f3 and f4 are linear
contractions of pairwise f2, every inference-flavored feature here is the device-resident
`f2_blocks` cache + the block jackknife + the batched fit engine, with an individual expressed as
a size-1 `PopGroup`. This covers all of Area 2, the modeling half of Area 1, the cladality tests
of Area 3, and the validation half of Area 4. Cost: mostly plumbing (the size-1 wrap, a ranking
pass, output shaping) — no new kernels. This is why the individual-level frontier is a strong
argument for finishing the fit-engine backend first.

**Substrate B — the GPU covariance / PCA / distance path (new).** steppe has **no** PCA,
dimensionality reduction, or individual-distance code today (confirmed: zero hits; the only SVD is
the host-side one-sided Jacobi in `src/core/internal/small_linalg.hpp`, used solely by the qpAdm
ALS fit). This substrate unlocks the *triage* row of Area 1, the outlier iteration of Area 3, and
the clustering of Area 4 — all at once, which is why it's one build, not three. What it needs:
- a decode variant that keeps the **individual × SNP** matrix device-resident (the per-individual
  packed genotypes already exist in `GenotypeTile.packed` + `pop_offsets`; today the decode kernel
  reduces straight to per-population frequencies);
- **covariance / SYRK** over the centered matrix — the *same shape* as the existing f2 covariance
  path, so the numerics policy (EmulatedFp64{40} for the matmul, the cancellation carve-out for
  reductions) carries over;
- a **GPU eig / SVD** (cuSOLVER) for the PCs;
- **projection** (`lsqproject` = a per-sample least-squares solve; optional `shrinkmode` for the
  origin-bias correction) and a batched **NNLS** simplex-QP per sample/model for the G25/Vahaduo
  triage — both embarrassingly parallel across samples;
- **k-means** (GEMM + argmin) for Area 4.
All GEMM/SYRK/eig-shaped and GPU-native, but genuinely new surface area.

**Substrate C — the readers / the on-ramp (new IO).** Consumer formats (23andMe / AncestryDNA raw
text, VCF) need new readers hooked into `genotype_source.hpp`'s dispatch + a new `GenoFormat`
enum; the AADR merge is a SNP intersection with strand-ambiguity handling
(`DATASET-COMPATIBILITY.md`). This is the USER-PAIN-WISHLIST §W5 "import your kit → merged dataset"
on-ramp. IO, not compute — it can proceed in parallel with everything else.

**The honesty rails (non-negotiable, and a selling point).** The whole reason to build this on an
f-statistic engine rather than another distance calculator is that it can be *honest*:
- **n=1 is high-variance, not biased.** f4/qpAdm stay unbiased in expectation at one individual;
  the SEs inflate. Report the SE and let models fail — don't launder a wide interval into a point
  estimate.
- **The f3/f2 heterozygosity correction is ill-defined for one pseudo-haploid genotype**, so
  *in-group* f3-admixture on a singleton is fragile — but outgroup-f3 (correction lands on the big
  outgroup) and all-f4 qpAdm are fine. This is why single-individual work routes through f4/qpAdm,
  not f3-admixture.
- **PCA-distance (G25/Vahaduo) is triage, not inference** — no null, no SE, strong and opaque
  dependence on whoever defined the PCA and the shrinkage. Present it as a fast first look that
  qpAdm then tests, never as the verdict.
- **Clustering→f-stats is circular** unless the groups are re-validated on independent criteria.
  Build the loop; don't ship the shortcut.

---

## 4. Impact × Feasibility

### Instant wins (high impact, low effort) — *do these first*
- **Model one individual with qpAdm (n=1 target).** Reuse the fit engine + a size-1 `PopGroup`.
  The flagship consumer + researcher feature, nearly free.
- **Closest-population ranking via outgroup-f3.** Reuses `f3-sweep`; one command, ranked output.
- **Better-proxy f4/D symmetry test.** Reuses `f4-sweep`; the rigorous "X beats Y."
- **Custom regrouping via `IndPartition`.** The plumbing that makes Area 4 (and any relabeling)
  possible; a config seam, not a kernel.

### Big bets (high impact, high effort) — *the differentiators*
- **Substrate B: the GPU PCA / covariance / distance engine.** One build that unlocks projection
  triage, outlier iteration, and clustering together. The single highest-leverage new capability.
- **The consumer on-ramp (readers + AADR merge, §W5).** "Import your kit → model yourself in
  seconds" is the demo that makes the whole thing land for amateurs.
- **The interactive bring-your-own-DNA studio flow.** Project → triage → qpAdm-confirm, live.

### Steady value (medium impact, low–medium effort)
- **Best-replacement rotation sweep** (rotation engine + a ranking pass).
- **Per-individual cladality QC** (qpWave, n=1) — catches mislabeled/contaminated samples.
- **N×N individual distance matrix** (a GEMM-shaped all-pairs f2/IBS reduction) — also the bridge
  object feeding PCA + clustering.

### Lower priority / dependency-gated
- **In-engine ADMIXTURE** (separate SQP kernel family — better left to the external binary).
- **fineSTRUCTURE / ChromoPainter** (HMM + MCMC, out of scope for an f-stat GPU engine).
- **Full G25-coordinate parity** (`shrinkmode`/scaled coords) — only if we chase G25 compatibility
  specifically rather than our own projection.

---

## 5. Sequencing rationale

Ship the **reuse tier first** — n=1 qpAdm modeling, outgroup-f3 closest-population, f4/D symmetry,
and custom regrouping — because they ride the engine steppe already has and need no new kernels;
they turn "individual-level" from a slogan into working commands within the current backend.
Build **Substrate B once**, deliberately, because a single GPU covariance/PCA/distance path is the
shared dependency of Area 1's triage, Area 3's outliers, and Area 4's clustering — funding it once
unlocks three Areas. Run **Substrate C (the consumer on-ramp)** in parallel: it's IO, not compute,
so it doesn't contend with the engine work, and it's what makes the amateur story real. Keep the
**clustering→f-stats loop gated behind the honesty rail** — the propose→validate cycle is the
feature, not raw clustering. Dependencies to respect: the fit-engine backend should be finished
first (it's the substrate for the whole reuse tier — see the *build-sequence-backend-first* and
*design-for-scale / S8 rotation* notes), and Area 4's population catalog + discovery
(`EXPERIENCE-PLAN.md` Area 4) feeds the pop-picker these workflows lean on.

---

## 6. Wild Ideas Worth Remembering

- **Date your own admixture.** Run DATES on a single genome — "your steppe ancestry entered the
  line ~160 generations ago, ±." One person, one number, with a jackknife SE.
- **Your ancient relatives.** Rank the individual AADR samples (not populations) closest to a
  query genome — "the 10 ancient people you share the most drift with, and where they're from."
- **Ancestry through time.** Animate a personal genome's position as the reference panel is filtered
  by date — watch "you" move across the ancient PCA as the world changes around the coordinate.
- **A public "project your kit" demo.** The STUDIO (`EXPERIENCE-PLAN.md` Area 6) front-end: upload,
  project, triage, and a real qpAdm fit — the single best advertisement for the speed.
- **Close the loop automatically.** Cluster → auto-propose curated pops → auto-qpAdm-validate →
  keep only the groups that survive. Data-driven population definition that polices its own
  circularity.

---

## 7. Architecture & cross-links

**The concrete seams (from the engine scout):**

| Capability | Seam | File | Reuse vs new |
|---|---|---|---|
| Per-individual genotypes | `GenotypeTile.packed` + `pop_offsets` | `src/io/genotype_tile.hpp` | expose (reuse) |
| ind → pop grouping / regroup | `IndPartition` / `PopGroup`, `read_ind()` | `src/io/ind_reader.{hpp,cpp}` | custom partition (reuse) |
| n=1 decode | size-1 segment, no min-size check | `src/device/cuda/decode_af_kernel.cu` | reuse as-is |
| f3/f4/qpAdm on an individual | linear contractions of `f2_blocks` + jackknife + fit | the fit engine | reuse |
| PCA / covariance / distance | **absent**; host Jacobi SVD only | `src/core/internal/small_linalg.hpp` | **new (Substrate B)** |
| Consumer formats / merge | format dispatch + `GenoFormat` enum | `src/io/genotype_source.hpp` | **new (Substrate C)** |

**Sibling docs.** `DEEPER-FRONTIERS.md` — the companion that takes this same engine one level *deeper*
(chromosome/window-level "local ancestry lite", X-vs-autosome sex bias, pseudo-haploid kinship, genetic
dating, selection-through-time, and the phasing/panel-dependent frontier steppe deliberately stays out
of). `EXPERIENCE-PLAN.md` — Area 4 (population discovery / the pop-picker these
workflows consume), Area 2 (population-level PCA/MDS, the natural precursor to individual PCA),
Area 6 (the STUDIO front-end), Area 7 (the MCP surface — an LLM could drive "model my DNA" against
the real engine). `DATASET-COMPATIBILITY.md` — the bring-your-own-data + strand-ambiguity ground
truth. `USER-PAIN-WISHLIST.md` §W5 — the consumer-kit on-ramp this doc formalizes. Deeper
engineering design belongs in `docs/design/` when a feature graduates from this doc to scope.

**Method provenance.** G25/Vahaduo (smartpca projection + NNLS); Harney et al. 2021 (qpAdm
performance, single-individual & missingness); Peter 2016 (f-statistics, the outgroup-f3 drift
caveat); `smartpca`/EIGENSOFT (projection + outlier iteration knobs); Alexander et al. 2009/2011
(ADMIXTURE); Lawson et al. 2012 (fineSTRUCTURE). These are the external references a public
methods page should cite; they are not internal steppe docs.

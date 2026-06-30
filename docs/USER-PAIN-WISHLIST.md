# steppe — Voice of the User: Pain & Wishlist (evidence-ranked)

## 1. What this is

This is the **evidence-ranked voice of the user** for the steppe experience plan. It synthesizes five independent research angles into the recurring pains, feature wishes, and points of confusion that **real users of the ADMIXTOOLS family** (qpAdm, qpWave, qpGraph, f4/f3/D-stats, DATES) express across:

- **GitHub issue trackers** — [DReichLab/AdmixTools](https://github.com/DReichLab/AdmixTools/issues) (the original C tool, #61–#125 mined) and [uqrmaie1/admixtools](https://github.com/uqrmaie1/admixtools/issues) (ADMIXTOOLS 2 / AT2 R package, #18–#169 mined), which double as the de-facto Q&A forum where Nick Patterson and Robert Maier answer directly.
- **Q&A forums** — Biostars, Bioinformatics StackExchange (snippet-level; Cloudflare-blocked).
- **Amateur / genetic-genealogy community** — Eupedia (the richest current hub post-Anthrogenica), genoplot, somalispot, IllustrativeDNA/AdmixLab, vahaduo.
- **Academic critiques & tutorials** — Harney et al. 2021 (Genetics), Maier et al. 2023 (AT2, eLife 85492), "Testing Times" 2024, qpAdm-screens 2025, plus how-to blogs and workshops.
- **Adjacent-tool envy & wrapper motivation** — admixr (Bioinformatics 2019), pong, ADMIXTURE, smartpca — what people praise elsewhere that raw ADMIXTOOLS lacks.

steppe is a **GPU-fast, parity-validated reimplementation of ADMIXTOOLS 2**. That gives it two distinct levers: (a) **speed** — it can erase the compute/RAM walls that force users to downsample and make rigorous rotation/screening practical; and (b) **a fresh, from-scratch UX** — it can fix the structural ergonomics (install, formats, errors, output, reproducibility) that the field has already tried to patch twice (admixr, then AT2).

**The single most important strategic read:** the biggest UX win is **NOT a faster statistic** — it is eliminating the **pre-computation failure surface** (build, PATH, file formats, 32-bit overflow, silent NA, cryptic segfaults). Across *both* trackers and *both* audiences, the dominant traffic is people who can't even get the tool installed, running, and reading their files — and when it breaks, the error tells them nothing. Speed is the secondary, differentiating win that makes the now-recommended rigorous workflow (large rotation screens + bootstrap model comparison + reference-sensitivity sweeps) actually runnable.

---

## 2. The top annoyances (ranked by reach across angles)

Ranked by how often / how loudly each recurs, and how many of the five angles independently surfaced it.

### #1 — Install / compile / dependency-rot: "I can't even get it running" (all 5 angles, recurring-loud)

The **single largest issue cluster** on both trackers, and the most-cross-corroborated pain in this entire report. It splits cleanly by tool:

- **Original C ADMIXTOOLS** — a relentless stream of compile/link failures: macOS M1/M4 (`argp`/openblas), Arch, RHEL9, **no Windows build at all**, and even **spaces in the install path** break it.
- **ADMIXTOOLS 2 (R)** — gets **bricked by R dependency churn** (RcppArmadillo, `readr::read_table2` removal, igraph, dplyr) through no fault of the user's data.

> "Undefined symbols ... `_argp_parse` referenced from `_main` in transpose.o ... ld: symbol(s) not found" — DReichLab #90 (macOS M1)
> Maintainer's fix on macOS M4 is literally "try removing `-p` from the CFLAGS line" — DReichLab #110
> "`readr::read_table2()` has been removed" — caused a **CRITICAL FAILURE** where `qpadm()` "cannot complete any task" — uqrmaie1 #96 / #99
> "Admixtools cannot be run from a Windows system... install Oracle VirtualBox and install a Guest Linux system on it" — [a-genetics blog](https://a-genetics.blogspot.com/2021/11/install-admixtools.html)

Sources: DReichLab [#90](https://github.com/DReichLab/AdmixTools/issues/90), [#106](https://github.com/DReichLab/AdmixTools/issues/106), [#110](https://github.com/DReichLab/AdmixTools/issues/110), [#119](https://github.com/DReichLab/AdmixTools/issues/119), [#87](https://github.com/DReichLab/AdmixTools/issues/87), [#72](https://github.com/DReichLab/AdmixTools/issues/72); uqrmaie1 [#96](https://github.com/uqrmaie1/admixtools/issues/96), [#100](https://github.com/uqrmaie1/admixtools/issues/100), [#94](https://github.com/uqrmaie1/admixtools/issues/94), [#104](https://github.com/uqrmaie1/admixtools/issues/104).

**steppe opportunity:** Ship **one prebuilt GPU wheel (`pip install steppe`) plus a static Linux binary** — zero compilation, no LAPACK/openblas/argp/Perl/Rtools/gfortran to assemble, no R dependency tree to rot. Because the core is a self-contained compiled engine on a **pinned, vendored** stack, "my analysis stopped working after I updated a package" simply cannot happen. This neutralizes the largest, most-repeated complaint category **before any speed advantage even comes into play.** For Windows, a documented WSL2/container path beats "compile argp-standalone yourself."

### #2 — Opaque failure: bare segfaults / "command not found" with no actionable message (Angles A, B, D; recurring-loud)

The tool dies and the user has no idea why. Root causes are mundane — a sibling binary not on `$PATH`, too few SNP blocks, a single-sample population, a typo in a label file — but the surfaced error is a **core dump** or a `fatalx`. The most damaging instance: **qpWave/qpAdm silently shell out to a separate `qpfstats` binary** that must be on `$PATH`; if it isn't, you get a segfault and a multi-day support thread.

> "sh: 1: qpfstats: not found / bad open /tmp/fsx.7302 / **Segmentation fault (core dumped)**" ... Nick eventually: "I think I have run out of help… You need sysadmin help" — DReichLab #92
> "ERROR: insufficient number of blocks -- you need 5 blocks with non zero ABBA/BABA" — DReichLab #82
> "pop: ??? has sample size 1 and inbreed set" → Aborted core dumped — DReichLab #96
> "'zero popsize, Aborted (core dumped)' — check spelling in label files" — folk-knowledge fix, [qpAdm 101](https://a-genetics.blogspot.com/2021/11/qpadm-101.html)

Sources: DReichLab [#92](https://github.com/DReichLab/AdmixTools/issues/92), #93, #85, #91, #70, [#82](https://github.com/DReichLab/AdmixTools/issues/82), [#96](https://github.com/DReichLab/AdmixTools/issues/96).

**steppe opportunity:** Ship **ONE self-contained binary** — no shelling out to sibling binaries, no `$PATH` dependency. Treat **every `fatalx`/segfault class as a named, explained error with a remediation hint** ("population X has 1 sample; qpAdm needs ≥2 or set `--no-inbreed`", "only 1 block formed — check chromosome/position ordering or `blgsize`", "label 'Xyz' not found in .ind"). **Validate inputs up front and fail with a sentence, not a core dump.** A crash on bad input is a bug, not a user error.

### #3 — File-format wrangling & conversion is a minefield (all 5 angles; recurring-loud)

Inter-format conversion (EIGENSTRAT ↔ PLINK ↔ geno) silently swaps REF/ALT (allele-count convention), requires position-sorted .bim/.snp with genetic-map columns in the right units, trips "OOPS indiv file has changed", and the latest AADR `.geno` (>2GB) **overflows 32-bit `long` builds outright**. For the amateur audience, **merging your own 23andMe/Ancestry data into AADR is the single most time-consuming step.**

> "PLINK often chooses the count allele as the majority allele guaranteeing different datasets use different conventions" — Nick, DReichLab #76 (eigenstrat→PLINK is not the identity map)
> "your long is 32 bits… arithmetic overflow when reading the AADR genotype file (>2GB)" — Matthew Mah, DReichLab #113
> `extract_f2` silently yields **one block** because the .bim "is not ordered by position" — uqrmaie1 #47
> "The trickiest part of all this is the merging of our own data with the data from the Reich Lab" ... the conversion "takes the longest, between half hour and an hour" — Eupedia user Tautalus

Sources: DReichLab [#76](https://github.com/DReichLab/AdmixTools/issues/76), [#113](https://github.com/DReichLab/AdmixTools/issues/113), #74, #102; uqrmaie1 [#47](https://github.com/uqrmaie1/admixtools/issues/47), #74, #80, #20; [Eupedia 44759](https://www.eupedia.com/forum/threads/using-admixtools2-to-model-admixture.44759/); [Biostars 476988](https://www.biostars.org/p/476988/).

**steppe opportunity:** This is a concrete, defensible differentiator — steppe's **multi-format readers (TGENO/PA/EIGENSTRAT/PLINK/ANCESTRYMAP, validated bit-exact) already exist.** Lead with "point steppe at PLINK or EIGENSTRAT, no `convertf` step." **Auto-detect format, auto-sort by chrom/pos** (never silently emit 1 block), **normalize allele/count conventions to a canonical polarization, warn loudly on ambiguous strand**, use **64-bit indexing** so multi-GB AADR never overflows, and **decouple .ind labels from a genotype checksum** so relabeling never trips "OOPS." A "drop your 23andMe file in, we merge to AADR" on-ramp is the hobbyist killer feature.

### #4 — Non-tabular log output you must hand-parse; no programmatic result object (Angles A, D, E; recurring-loud)

Results are dumped to a free-text log to be scraped with sed/awk or manual copy-paste. **The existence of admixr — and the rewrite of AT2 itself — is the proof:** a whole wrapper exists specifically to fix this.

> "the user needs to extract relevant values from a non-tabular text file before they can be imported into software such as R"; admixr does "all the dirty work of parsing the output files... presenting the user with convenient R data structures" — [admixr paper, Bioinformatics 2019](https://academic.oup.com/bioinformatics/article/35/17/3194/5298728)
> The original workflow "often involves a combination of sed/awk/shell scripting and manual editing" and results must be "parsed by the user... often using command-line utilities again or by manual copy-pasting" — [admixr README](https://github.com/bodkan/admixr)

Sources: [admixr paper](https://academic.oup.com/bioinformatics/article/35/17/3194/5298728); [admixr README](https://github.com/bodkan/admixr); DReichLab #71 (feature request: "make labels available in the .log file").

**steppe opportunity:** Make **tidy/tabular the default output of every estimator** (f2/f3/f4/D/qpAdm/qpWave/qpGraph/DATES). Emit long-form CSV/Parquet/Arrow **and** return a pandas DataFrame from the Python binding, with stable column names (`pops, est, se, z, p, weights`), labels attached, no `.log` scraping, no par-file authoring. **Make the reproducible, scriptable path the only path** — be the thing you'd otherwise have to wrap.

### #5 — Memory / scale wall: OOM-killed, "downsample to survive" (all 5 angles; common-to-loud)

`qpfstats`/`extract_f2` are memory hogs; big AADR-scale or many-pop runs get OOM-killed; users are explicitly told to **downsample to ~1M SNPs just to finish**. The maintainers have **no timeline** to fix the 32-bit overflow.

> "qpfstats is a memory hog" — Nick, DReichLab #97
> 600 ind × 18M SNP needed **200GB RAM** to extract f2 — uqrmaie1 #30
> "It is very very unlikely you need 250M snps… downsample to about 1M"; "I recommend downsampling to no more than 1-2M snps" — Nick, DReichLab #121 / #122
> "this would cause arithmetic overflow... I'm not sure what the timeline will be to fix this" — DReichLab #113

Sources: DReichLab [#97](https://github.com/DReichLab/AdmixTools/issues/97), #121, #122, #70, #113; uqrmaie1 [#30](https://github.com/uqrmaie1/admixtools/issues/30), #25, #93, #73.

**steppe opportunity:** This is **steppe's home turf.** Device-resident, batched, **streamed** f2/f-stat compute sized for large models + the S8 rotation handles full AADR without 200GB host RAM or 32-bit overflow — directly answering complaints the upstream maintainers explicitly have no timeline to fix. Turn **"downsample to survive" into "run everything."** Lead marketing with **wall-clock on real AADR.**

### #6 — Rotation / screening is compute-bound: the rigorous workflow is now a multi-day batch (Angles A, C, D; recurring-loud among pros)

Credible qpAdm now means **tens of thousands of rotation models per analysis**, and the base tool is **single-core at ~1–2 min/model** — so a real screen is a multi-day, RAM-hungry batch nobody can iterate on interactively. This is the **strongest steppe-aligned signal in the academic literature.**

> "34,320 proximal and distal rotating models per simulation replicate" — [qpAdm-screens, Genetics 2025 (iyaf047)](https://academic.oup.com/genetics/article/230/1/iyaf047/8102970)
> "Running qpWave or qpAdm with this dataset takes 1-2 minutes, using a single core (no multithreading is available)" — [AT2 tutorial](https://uqrmaie1.github.io/admixtools/articles/admixtools.html)
> exhaustive rotation is "extremely tedious"; "many ADMIXTOOLS analyses run in parallel can consume a lot of memory!" — [admixr rotation vignette](https://bodkan.net/admixr/articles/vignette-02-qpAdm.html)

**steppe opportunity:** Exactly steppe's design envelope (the S8 rotation, batched-not-transliterated). GPU-batched, device-resident fitting of thousands of rotation models turns a 3-day CPU batch into an interactive session. **Make the rotation strategy the DEFAULT cheap path** and a first-class `run-the-whole-rotation` command, not a hand-rolled parallel loop.

### #7 — Results feel unreliable / non-reproducible: run-to-run variation, version drift, no tags (Angles A, B, C, D; common)

`allsnps=TRUE` gives run-to-run variation; `allsnps` semantics silently changed across versions; AT2 ≠ AT1 numbers; **there are no version tags/releases to pin a published analysis to.** Users openly distrust their own output.

> "What the `allsnps` option does in Admixtools 1 has changed over time. The `allsnps` option in Admixtools 2 isn't based on the latest version of Admixtools 1." — Maier, uqrmaie1 #82
> "may introduce some run-to-run variation. Therefore, repeating the analysis to replicate a viable model is advisable" — [Eupedia best-practices 45634](https://www.eupedia.com/forum/threads/best-practices-admixtools2-functions-extract_f2-f2_from_precomp-qpwave-qpadm.45634/)
> "please add tags/releases" / "Tagged Release soon?" — uqrmaie1 #64, #169 (recurring across years)

Sources: uqrmaie1 [#82](https://github.com/uqrmaie1/admixtools/issues/82), #49, #53, [#64](https://github.com/uqrmaie1/admixtools/issues/64), [#169](https://github.com/uqrmaie1/admixtools/issues/169); DReichLab #75, #64.

**steppe opportunity:** Cut **semver-tagged, citable releases from day one**; **stamp the exact version + parameter defaults into every output header** so a paper can cite "steppe vX.Y.Z" and a reviewer reproduces bit-for-bit. Couple with the **parity-validation (golden-gated)** story to make "same version → same number" a guarantee. Emit a **run manifest** (inputs, pop lists, SNP filters, seeds, version, f2-cache hash) for replayable, citable runs.

### #8 — No visualization: numbers, not figures (Angles C, E; common, distinctly amateur-weighted)

Raw ADMIXTOOLS gives a text table of weights/p-values; the amateur community **thinks and argues in pictures.** Even viewing a qpGraph result in the original tool required emitting a `.dot` file and running an external Graphviz `dot -Tps` pipeline.

> vahaduo's entire popularity is web-based G25 visualization: "Admixture JS", "Custom PCA", "3D PCA Viewer" — [vahaduo.github.io](https://vahaduo.github.io/)
> pong: "front end produces a D3.js visualization of maximum-weight alignments between runs", ranked into 'modes', interactive — [pong, Bioinformatics 2016](https://academic.oup.com/bioinformatics/article/32/18/2817/1744074)
> original qpGraph: "`-d dotfile`" then "`dot -Tps < dotfile > dotfile.ps`", "`dot` must be installed" — [DReichLab README.QPGRAPH](https://github.com/DReichLab/AdmixTools/blob/master/README.QPGRAPH)

**steppe opportunity:** Bundle **first-class plots**: source-weight bars with SE whiskers, p-value/feasibility traffic-light, rotation heatmaps, **native admixture-graph rendering (SVG/PNG, no Graphviz install)**, DATES decay curves, and a PCA/projection view. Visual output is **table-stakes for amateur adoption, not a nice-to-have.**

### #9 — qpGraph / find_graphs is fragile & non-deterministic (Angle A; common among pros)

It stack-overflows, throws opaque igraph/dplyr internal errors, and returns **inflated "worst residual" scores** because `find_graphs` evaluates each topology from a **single random start** and gets stuck in local optima — so the "winning" graph isn't reproducible.

> "evaluation nested too deeply: infinite recursion ... C stack usage 7972624 is too close to the limit" — uqrmaie1 #51 (13 comments)
> "`find_graphs()` only uses a single set of initial parameters for each graph by default, which can make topologies look worse than they really are" — fix is bumping `numstart` to 1000 — Maier, uqrmaie1 #65

Sources: uqrmaie1 [#51](https://github.com/uqrmaie1/admixtools/issues/51), [#65](https://github.com/uqrmaie1/admixtools/issues/65), #43, #33, #31, #23, #45, #21, #22, #46.

**steppe opportunity:** steppe's qpGraph enumerate path can run **many random starts cheaply on-GPU** — `numstart` isn't a cost tradeoff when the engine is fast. Make **global-optimum search the default** rather than an expert tuning knob, turning "is my best graph real or a local optimum?" from a gotcha into a non-issue. GPU-accelerated topology search (`find_graphs`-equivalent) is a natural differentiator — the search is embarrassingly parallel.

### #10 — No GUI for non-programmers (Angles C, E; common, distinctly amateur)

The largest amateur signal that there is **money on the table.** A **paid commercial GUI** (IllustrativeDNA's AdmixLab) and multiple **hosted web services** (genoplot's qpAdm service, vahaduo for G25) exist purely to wrap qpAdm and let hobbyists avoid the CLI.

> AdmixLab: "a browser based environment service... user-friendly interface... perform admixture analysis without needing to install command-line software locally" — [illustrativedna.com/admix-lab](https://illustrativedna.com/admix-lab/)
> AT2: "if you are not familiar with R, there is a browser application available" (`run_shiny_admixtools()`); amateur forums actively share it — [GenArchivist tid=1331](https://genarchivist.net/showthread.php?tid=1331)

**steppe opportunity:** There is a **proven, paying market** for an accessible qpAdm front end. steppe's fresh-UX mandate + GPU speed could offer a **free/open or self-hostable GUI** that is both faster and not paywalled — undercutting AdmixLab/genoplot on speed and price. Pair a searchable AADR population browser + one-click run + inline plot. This is the central UX bet for the amateur audience.

---

## 3. The wishlist (ranked feature asks)

⚡ = **steppe's SPEED uniquely enables this** (not just nicer ergonomics).

### W1 — A clean, no-config, programmatic API returning tidy tabular results (the thing admixr/AT2 were built to deliver)
Every estimator returns a DataFrame; results serialize to Parquet/CSV/JSON; no par-files, no log scraping. Evidence: admixr exists to "perform all stages of an ADMIXTOOLS analysis entirely from R" and "completely remove the need for low-level configuration"; results are "an R data frame which can be immediately used for further statistical analysis and plotting" ([admixr](https://github.com/bodkan/admixr), [paper](https://academic.oup.com/bioinformatics/article/35/17/3194/5298728)). **steppe:** build admixr-style ergonomics in **natively (Python-first, structured returns)** so no third-party wrapper is ever needed — and without an R dependency.

### W2 — ⚡ Fast, built-in rotation/screening WITH false-discovery control
Exhaustive source/outgroup sweeps and many-replicate confirmation, with FDR/multiple-testing reporting baked in. Evidence: "FDR exceeds 50% for many parameter combinations" ([qpAdm-screens](https://academic.oup.com/genetics/article/230/1/iyaf047/8102970)); only "44.8% of plausible models represent the true ancestry source" ([Testing Times](https://academic.oup.com/genetics/article/228/1/iyae110/7714968)); admixr's rotation is "extremely tedious." **steppe:** GPU-batched rotation makes the now-recommended rigorous practice runnable; **turn the academic critique into a guardrail feature** (FDR-aware ranked model table by default).

### W3 — A direct f2 / f3 / f4 / D utility from genotype files, each as a first-class CLI + binding
Evidence: DReichLab #117 ("F2 utility in admixtools?"); uqrmaie1 #79 ("computing f3- and f4-statistics directly from genotype files"), #83 ("dscores in admixtools2?"). **steppe:** already builds qpfstats + standalone f4/D/f3 on the genotype path — expose each as a first-class CLI+binding (per the build-sequence plan) so "why is there no direct f-stat utility" is answered out of the box.

### W4 — ⚡ A formal, cheap goodness-of-fit and pairwise model-comparison test
"Is model A actually better than model B?" as a one-call, default output. Evidence: AT2 was built to provide "formal goodness-of-fit tests" and tests for "whether the difference between the fits of any two models is statistically significant" — yet admits "We have not been entirely successful" on absolute fit; a reanalysis found "1,421 topologies fitting nominally or significantly better than the published model" ([eLife 85492](https://elifesciences.org/articles/85492)). **steppe:** make bootstrap-based fit + pairwise comparison a **GPU-cheap default**, run exhaustively rather than as a heroic compute effort.

### W5 — Built-in handling of consumer raw DNA + AADR merge as a one-click on-ramp (amateur)
Evidence: AdmixLab markets "non-destructive merging pipelines" accepting Ancestry/23andMe/MyHeritage as its headline value ([AdmixLab](https://illustrativedna.com/admix-lab/)); Tautalus describes the manual merge as the hours-long worst step ([Eupedia 44759](https://www.eupedia.com/forum/threads/using-admixtools2-to-model-admixture.44759/)). **steppe:** a guided "import your kit → merged dataset" flow leveraging steppe's fast readers converts a 30–60 min chore into seconds — the obvious funnel from consumer-DNA hobbyists into the tool.

### W6 — Built-in publication-ready plots + native graph rendering
Evidence: pong interactive barplots ([Bioinformatics 2016](https://academic.oup.com/bioinformatics/article/32/18/2817/1744074)); AT2's `plot_graph()`/`plot_comparison()` ggplot helpers; amateur forums post charts, not tables. **steppe:** ship a small plot module / plot-ready exports for the canonical figures; render admixture graphs natively without Graphviz.

### W7 — A point-and-click GUI / no-install hosted interface (amateur)
Evidence: AT2 ships `run_shiny_admixtools()` and is praised for it; a **paid** product (AdmixLab) and hosted services (genoplot) prove the demand. **steppe:** optional web GUI over steppe's engine with an AADR population browser; free/self-hostable undercuts the paywalled incumbents.

### W8 — Tagged, citable, reproducible releases (+ Windows path)
Evidence: uqrmaie1 #64/#169, DReichLab #64, #72. **steppe:** semver wheels on PyPI, stamped version in every output, a clear platform matrix (Linux+CUDA; documented WSL2 for Windows). "Reproducible & citable" as a core feature.

### W9 — ⚡ Reference-set decision support / sweep diagnostics
Help choosing right (reference) pops — the step everyone finds subjective and dangerous. Evidence: "no clear algorithm for proving that a certain reference set is optimal"; reference choice "affects results in a major way" ([qpAdm 101](https://a-genetics.blogspot.com/2021/11/qpadm-101.html)); Harney 2021's differential-relatedness requirement. **steppe:** cheap GPU sweeps to show sensitivity of weights/p to dropping each right pop, flag symmetric/non-informative right-sets per Harney — a decision-support feature **only speed makes practical.**

### W10 — Easier population grouping / relabeling that never invalidates the dataset
Evidence: uqrmaie1 #85 ("Quick way to group populations?"); Biostars 476988 (OOPS error from editing labels). **steppe:** first-class population/label management (group, alias, subset by metadata) that never requires regenerating the genotype file or trips a stale-checksum guard.

### W11 — Stable, locale-independent, reusable precomputed f2
Evidence: uqrmaie1 #20 ("set locale before writing/reading f2-statistics"); AT2's core selling point is reusing f2 across many models. **steppe:** keep `f2_blocks` device-resident and serialize them in a **locale-proof, versioned binary** the fit engine reads straight from VRAM (the fused-fit direction) — "extract once, fit thousands of models," fast and portable.

### W12 — Copy-paste runnable quickstarts/tutorials on real AADR for each stat
Evidence: Biostars 9548987 (request for a qpDstat tutorial); DReichLab #55; Maier's stated goal "make these methods a bit more accessible to non-experts." **steppe:** ship runnable quickstarts with population labels pre-resolved as a first-class part of the product, not a blog afterthought.

---

## 4. Common confusion / FAQ (what a good UX, docs, or `--explain` could fix)

These are the things **everyone gets wrong.** Each is a candidate for an inline explainer, a validation message, or a `--explain` flag.

1. **p-value direction is inverted vs. intuition.** You want **p > 0.05** (model NOT rejected = plausible), not p < 0.05. Users routinely read a low p as "good fit." Compounded by NA being mistaken for a pass. *"the opposite of how p-values are typically used in other fields"* — [Eupedia 44759](https://www.eupedia.com/forum/threads/using-admixtools2-to-model-admixture.44759/); uqrmaie1 #28, #78; Biostars 9568202. **And:** "P-value ranking ... should not be used to identify the best model" — [Harney 2021](https://academic.oup.com/genetics/article/217/4/iyaa045/6070149).

2. **The hard left/right constraint.** You need **more right (reference) than left (source+target)** populations, or the full-model p comes back **NA** (silently mistaken for a pass). The same pop can't be in both; `rank = nleft − 1`. "you don't have enough right populations (you need more right than left populations)" — uqrmaie1 #78. **steppe:** validate `nright > nleft` at call time; refuse with a plain-English message; show three visibly distinct states — **rejected (p<0.05) / not testable (NA) / passes.**

3. **D-statistic sign convention varies by paper.** Which of P1/P2 admixes with P3 when D>0 vs D<0. ADMIXTOOLS defines D=(BABA−ABBA)/(BABA+ABBA). *"the sign here varies with different papers (my fault). But the formula above has become standard"* — Nick, [DReichLab #124](https://github.com/DReichLab/AdmixTools/issues/124).

4. **Reference-set (right-pop) SELECTION SENSITIVITY.** People are surprised that reordering or changing the right/outgroup set changes qpAdm/f3 results. uqrmaie1 #26 ("The order of right population impact the qpAdm result"), #54; DReichLab #75.

5. **Negative f3 is the SIGNAL, not an error.** f3 < 0 (Z < −3) is strong evidence of admixture; a positive/zero f3 does NOT prove no admixture. Beginners read negatives as a bug. [compvar-workshop f3stats docs](https://compvar-workshop.readthedocs.io/en/latest/contents/03_f3stats/f3stats.html).

6. **High |Z| (e.g. Z=100) is clipped, and tiny SE is deflated by correlated SNPs.** "I clip Z scores to +- 100. I recommend downsampling to no more than 1-2M snps" — Nick, [DReichLab #122](https://github.com/DReichLab/AdmixTools/issues/122).

7. **Statistical significance ≠ historical reality.** A model can pass and still be biologically nonsense; at realistic FST the method converges on a *cluster* of plausible models. "statistically robust, but do not align with historical/pre-historical reality" — Eupedia; [Testing Times](https://academic.oup.com/genetics/article/228/1/iyae110/7714968).

8. **LD-pruning is NOT required.** The block jackknife handles LD — but everyone asks. "It's generally not necessary to do LD pruning" — Maier, uqrmaie1 #42.

9. **File-format / allele conventions.** EIGENSTRAT `.snp` REF/ALT vs PLINK `.bim`; ADMIXTOOLS treats column 5 as the **count allele**, not necessarily the reference — a frequent source of silently-corrupted merges. DReichLab #76. Also: input must be EIGENSTRAT/PACKEDANCESTRYMAP, which "normal" bioinformatics tools can't QC/filter ([admixr paper](https://academic.oup.com/bioinformatics/article/35/17/3194/5298728)).

10. **`blgsize` units** (Morgans vs base pairs; bp interpretation only kicks in when `blgsize ≥ 100` and needs a genetic-distance column). uqrmaie1 #47.

11. **`allsnps: YES` vs default (SNP intersection)** semantics — what it does changed across AT1/AT2 and across versions, so users can't reason about why numbers differ. uqrmaie1 #82.

12. **Minimum sample size / single-sample / pseudohaploid edge cases.** Whether qpAdm/qpDstat can run with 1 individual; the "sample size 1 and inbreed set" abort with `allsnps:YES`. DReichLab #96, #86; uqrmaie1 #44, #58.

13. **Terminology drift:** left/right/sources/references/outgroups are used inconsistently — admixr literally says "outgroup populations (also called references or right populations, depending on whom you talk to)." [admixr vignette](https://bodkan.net/admixr/articles/vignette-02-qpAdm.html).

14. **Population-file ordering is implicit & unguarded:** the first left population (top of file) is the target; the rest are references. No safeguard against misplacement. [CompPopGen workshop](https://comppopgenworkshop2019.readthedocs.io/en/latest/contents/05_qpwave_qpadm/qpwave_qpadm.html).

15. **Don't mix sample-prep types (.DG / .SG / noUDG) on the left** — a non-obvious data-hygiene rule that silently biases results. [Eupedia 44759].

16. **`tail prob` vs `taildiff`, rank/maxrank, A/B matrices** — output metrics are non-obvious and easy to misread. [CompPopGen workshop].

17. **maxmiss / missingness for aDNA vs modern data** — "Maxmiss = 1 ... optimal when dealing with aDNA ... stricter ... like 0.2 for modern." Eupedia 45634.

**steppe opportunity (cross-cutting):** A `--explain` mode and inline diagnostics can resolve most of these at the point of use: label the three p-value states explicitly, validate the left/right constraint, print the D sign convention in the f4/D output header, offer a reference-sensitivity sweep, and surface "negative f3 = admixture signal" as a note rather than letting users guess. Good defaults + a glossary in every output kill the FAQ.

---

## 5. The amateur vs pro split

Both audiences share the **install + format + interpretation** pain, but their headline needs diverge:

| | **Amateur / genetic-genealogy** (Eupedia, genoplot, somalispot, AdmixLab, vahaduo) | **Professional popgen / aDNA** (GitHub, methods papers, workshops) |
|---|---|---|
| **Top want** | **No-install GUI** + **visualization** (PCA/admixture plots) | **Speed at scale** (rotation/screen/graph-search) + **scripting/API** |
| **Killer feature** | **Consumer-DNA → AADR merge on-ramp** (23andMe/Ancestry) | **Reproducibility & citable versioning**; tidy DataFrame output |
| **Pop selection** | **Searchable AADR browser** (which labels exist, exact strings) | Reference-set **sensitivity diagnostics** + FDR control |
| **Tolerance for CLI** | Low — they **pay** to avoid it (AdmixLab/genoplot) | High — but want it scriptable, not par-file glue |
| **Where they live** | Eupedia (richest hub), genoplot, Discords, ethnic forums — **Reddit is thin** | GitHub issue trackers, eLife/Genetics, CompPopGen workshops |
| **Interpretation help** | Plain-language: p-value direction, "is my model real?" | Statistical rigor: FDR, model-comparison significance, non-identifiability |

**Key cross-overlaps (weight highest):** install-pain, file-format/merge pain, and p-value-inversion confusion surface in *both* columns and *across multiple angles* — these are the convergent signals. The **distinctly amateur** contributions are the **GUI/visualization demand** and the **consumer-DNA merge on-ramp**; the **distinctly pro** contributions are **compute-bound rotation/screening** and **reproducible/citable scripting.**

**Strategic note:** the amateur audience *is* steppe's own origin community (vahaduo). They are large, vocal, GPU-hungry (they run massive rotations on laptops), underserved by anything CLI-only, and **currently paying** IllustrativeDNA/genoplot for access. A free, fast, no-install, visual qpAdm directly disrupts that market — while the same fast engine + tidy/scriptable output captures the pros. steppe does not have to choose: **one engine, two front doors (Python/CLI for pros, GUI for amateurs).**

---

## 6. The 10 highest-leverage, evidence-backed UX moves for steppe

The intersection of **"users loudly want it"** × **"steppe can uniquely deliver it."** Ordered to feed/reprioritize the EXPERIENCE-PLAN sprint.

1. **One prebuilt artifact, zero compilation: a GPU wheel (`pip install steppe`) + static Linux binary.**
   Kills the #1 cross-angle pain (install/compile/dependency-rot) outright; no LAPACK/argp/Rtools/R tree to rot. *Evidence: DReichLab #90/#106/#110/#119/#72; uqrmaie1 #96/#100/#94.* Highest reach, lowest statistical risk — ship this first.

2. **Native multi-format reading + auto-sort + 64-bit indexing + canonical allele polarization — no `convertf`, no overflow.**
   steppe's readers already exist; lean into "point steppe at PLINK/EIGENSTRAT, it just works." *Evidence: DReichLab #76/#113; uqrmaie1 #47; Eupedia 44759.*

3. **Typed, explained errors with remediation hints — never a bare segfault/`fatalx`/NA.**
   Validate inputs up front (label exists, `nright > nleft`, ≥2 samples, block count). One self-contained binary, no `$PATH` shell-outs. *Evidence: DReichLab #92/#82/#96; uqrmaie1 #78; qpAdm-101 folk fixes.*

4. **Tidy/tabular output by default + Python DataFrame returns — be the tool you'd otherwise wrap.**
   Kills the entire "wrapper needed" category (admixr, AT2). Stable columns, CSV/Parquet/Arrow, labels attached, no log scraping. *Evidence: admixr paper + README; DReichLab #71.*

5. **⚡ GPU-batched rotation/screening as the DEFAULT path, with built-in FDR/model-comparison reporting.**
   Makes the now-recommended rigorous workflow (34,320 models/replicate) interactive instead of a 3-day batch. Speed turns the academic FDR critique into a guardrail. *Evidence: qpAdm-screens iyaf047; Testing Times iyae110; AT2 tutorial single-core ~1–2 min/model.*

6. **⚡ Run the FULL SNP set / full AADR — "never downsample," streamed device-resident f2.**
   Directly answers OOM/200GB/32-bit complaints the maintainers won't fix. Lead marketing with wall-clock on **real** AADR. *Evidence: DReichLab #97/#113/#121/#122; uqrmaie1 #30.*

7. **Semver-tagged, citable releases with version + params stamped in every output header; run manifest for replay.**
   "steppe vX.Y.Z" reproduces bit-for-bit, backed by the golden-gated parity story. *Evidence: uqrmaie1 #64/#169/#82; DReichLab #75.*

8. **A free/self-hostable GUI with a searchable AADR population browser + one-click run + inline plots.**
   Disrupts a **paying** market (AdmixLab/genoplot) and captures the large amateur funnel. *Evidence: AdmixLab product page + launch; genoplot; AT2 Shiny demand (GenArchivist tid=1331).*

9. **First-class visualization: weight bars + SE, p-value/feasibility traffic-light, rotation heatmaps, native graph rendering (no Graphviz), DATES decay curves.**
   Visual output is table-stakes for the amateur cohort that argues in pictures. *Evidence: vahaduo; pong; DReichLab README.QPGRAPH `dot` pipeline.*

10. **⚡ Cheap many-start qpGraph search (global-optimum default) + reference-set sensitivity sweeps + an inline `--explain`/FAQ layer.**
    `numstart=1000` and dropping-each-right-pop sweeps stop being expensive expert knobs; `--explain` resolves the p-value/left-right/D-sign/negative-f3 confusions at point of use. *Evidence: uqrmaie1 #65/#51/#26; Harney 2021; DReichLab #124; qpAdm-101.*

**Sequencing read:** moves **1–4** are pure UX-foundation wins with **no statistical risk** and the **broadest reach** — they neutralize the largest complaint category before speed even matters, and should anchor the sprint. Moves **5, 6, 10 (⚡)** are where steppe's GPU engine is **uniquely** load-bearing — they are the differentiation vs AT2 specifically (AT2 already closed the tidy-output/GUI/find_graphs/f2-cache gaps on CPU). Moves **8–9** capture the underserved, *paying* amateur market. Be honest in messaging (per the academic angle): **steppe's speed enables better practice — cheap sensitivity sweeps, exhaustive bootstrap comparison, large FDR-controlled screens — it does not "fix" qpAdm's inferential limits.** Overclaiming a statistical cure would be a credibility risk with the professional audience.

---

### Method & skepticism notes
- **Highest-confidence evidence** = the two GitHub trackers (fetched in full via `gh`; verbatim quotes) and the peer-reviewed papers/wrapper READMEs (admixr, AT2/eLife, Harney, qpAdm-screens). Biostars/StackExchange and several amateur forums (Eupedia/genoplot/somalispot/GenArchivist) are **Cloudflare/anti-bot-blocked** and cited via search snippets or reader-proxy — verbatim wording is faithful but worth a spot-check before publishing.
- **Intensity ratings** elevate only failure modes recurring across **multiple independent issues/angles** (install: ~10 issues across all 5 angles; segfault/PATH: 5+; format/convert: 5+). One-off bugs (uqrmaie1 #49 dplyr, #53 parse) count as reproducibility-churn evidence in aggregate, not as standalone top pains.
- **Convergent multi-angle signals (weight highest):** install/compile, file-format/merge, and compute-bound rotation surfaced in 4–5 angles each. **Distinctly amateur:** GUI + visualization + consumer-DNA merge. **Distinctly pro:** FDR/model-comparison rigor + citable reproducibility.
- **Honest caveat:** AT2 already captured much of the "tidy output + GUI + find_graphs + f2-cache + bootstrap" win in R. steppe's unique unclaimed ground is: **GPU scale for the rotation/screen/graph-search envelope + a Python/tabular-first CLI/API with native multi-format reads + a free GUI, all without an R dependency.** The `steppe_opportunity` lines map to project memory (multi-format readers done; qpfstats/f4/qpGraph built; GPU-wheel policy) as the target, not as independently re-verified-in-this-session capabilities.

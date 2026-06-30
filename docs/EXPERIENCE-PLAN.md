# steppe — The Experience Layer Plan

*A vision + a runnable pre-ship sprint. The compute is done and parity-validated against ADMIXTOOLS 2 on real AADR. What is barely built is everything about how a researcher **discovers, runs, sees, shares, and enjoys** steppe. This plan designs that layer.*

---

## 1. The Vision

For thirty years, population-genetics modeling has been a *batch* science. You write an R script, launch a qpAdm fit, walk away for coffee, come back to a CSV, and start the manual ggplot toil. Each fit is minutes, so the entire user experience — and the entire *mental* experience of the field — is necessarily write-wait-read-tweak-repeat. ADMIXTOOLS 2 is excellent, and it is structurally offline.

steppe breaks that constraint. A qpAdm fit is sub-second. A billion-quartet f4 sweep that would take AT2 days finishes in 177 seconds on one GPU. **The engine is fast enough to make population genetics interactive for the first time** — and that is not merely "a faster qpAdm," it is a different *category* of tool. When the latency between a hypothesis and its answer collapses to the blink of an eye, modeling stops being a chore and becomes a conversation: type a target and watch a model snap to a p-value; drag an outgroup in and out and watch feasibility flip; lasso a region and a millennium on a map and watch the rotation re-rank in front of you. Using steppe should feel like talking to a sharp colleague who happens to be instant — one who answers in a sentence (`Target = 0.74*Steppe + 0.26*Anatolia | p=0.31 | FEASIBLE`), draws the decay curve right there in your scrollback, knows every cryptic AADR label by heart, and hands you a publication-ready figure and a paste-able Methods paragraph on the way out. The two pains that gate adoption today — fiddly setup and the miserable hunt for the right population label — should be solved in the first two minutes. Everything after that should feel less like running software and more like *playing an instrument*.

The single architectural truth under all of it: the speed is the whole pitch, and right now it is invisible. The experience layer's job is to make the speed *felt* — in a live throughput bar, in a slider that re-fits as you drag it, in a terminal that answers before you've let go of the Enter key.

---

## 2. The Six Areas

### Area 1 — Terminal & CLI Experience

**Vision.** The same `steppe qpadm` that pipes byte-clean CSV into a script should, on a human's terminal, answer in a colored sentence with the numbers that matter triaged by significance. Errors should never end a session — a fat-fingered population name comes back with "did you mean Anatolia_Neolithic?", and finding a label in the 700-pop AADR haystack is a tab-completion away. Because a fit is sub-second and a sweep streams a live throughput bar, the terminal becomes an interactive instrument, not a batch submission form.

| Feature | What | Why it delights | Speed? | Effort |
|---|---|---|---|---|
| **isatty `auto` output mode** *(foundation)* | New default `--format auto` in the one output seam (`emit_to_destination`, cmd_emit.hpp): TTY → pretty tables, piped/`--out` → CSV byte-for-byte. `--color auto/always/never`, honor `NO_COLOR`. Pretty is strictly **additive** — csv/tsv/json stay golden-identical. | Today a human gets quoted CSV with `# section:` comments. This does the right thing automatically — readable for me, machine-clean for my pipeline, zero flags, zero risk to parity goldens. | No | M |
| **Pretty per-stat tables + significance color** | Aligned right-justified columns for f4/f3/f4-ratio/qpdstat/rank/rotation/sweep; `\|Z\|≥3` bold red, `2–3` yellow, qpAdm `p≥0.05` green vs red, weights outside [0,1] flagged. | aDNA researchers read Z and p for a living; this turns "which of 40 quartets is real" into a glance, matching the AT2-console mental model but actually formatted. | No | M |
| **The qpAdm model line** *(headline)* | One-line verdict above the detail: `Target = 0.74*Steppe + 0.26*Anatolia \| p=0.31 \| FEASIBLE`, color-coded, optional unicode weight bar. Built from fields already computed. | The single highest-delight change: collapses a four-section CSV dump into the exact sentence you'd write in a paper. The "qpAdm as a sentence" moment. | No | **S** |
| **`--explain` plain-language reading** | Narrates the result by the field's own rules: "p=0.31 > 0.05 — model NOT rejected; both weights in [0,1] (feasible); nested rank test p=0.002 says you can't drop a source." | qpAdm interpretation is genuinely error-prone (people misread p-direction + feasibility constantly). A built-in correct narrator lowers the expertise barrier and teaches students. | No | M |
| **Fuzzy "did you mean"** | On `PopResolver` failure, edit-distance + token-overlap against the label vector it already holds: `unknown 'Anatolia_N'. Did you mean: Anatolia_Neolithic, Iran_N? (3 close of 711)`. | AADR labels are notorious (`_N`/`.SG`/`.DG`). Kills the typo → grep pops.txt → retype loop dead — the stated #1 papercut. | No | **S** |
| **`steppe pops` discovery command** | `--grep steppe`, `--like 'Yamnaya*'`, fzf-style TTY picker with sample counts from `.ind`/`.anno`; obeys the auto/CSV split so it scripts too. | Directly attacks the named pain: the .anno is huge and cryptic. A first-class fast filterable label browser is the thing every AT2 user hand-rolls today. | No | M |
| **Live progress / throughput bars** | On **stderr** only when TTY: `14.5M quartets/s — 41% — ETA 1m44s` for sweep / rotation / qpgraph-search / dates / extract-f2. | A 177s sweep currently looks like a hang. This both reassures and lets the user **feel** the GPU speed — turns dead air into the product's best demo. | **Yes** | M |
| **Terminal sparkline / ASCII plots** | DATES decay curve as a braille sparkline with date±SE; `\|Z\|` histogram for sweep survivors; p-value strip for rotation. | Seeing the decay curve is how a dating researcher sanity-checks a fit. Drawing it in-terminal — no ggplot, no export — is delight + a "this tool gets my workflow" credibility signal. | No | M |
| **Shell completion + dynamic pop names** | `steppe completions zsh/bash/fish`: subcommands, flags, enum values, **and** dynamic completion of pop names by reading the `--f2-dir` on the line. | Tab-completing 700 cryptic AADR labels instead of copy-paste is transformative for the exact label pain this audience has. | No | M |
| **Examples-rich `--help` + `steppe examples`** | Per clig.dev: 2-3 copy-paste-runnable examples ending every subcommand help; a top-level cheat-sheet. | Examples are the fastest path from "installed" to "first real result" and answer "how do I say this here?" for AT2 migrants. | No | **S** |
| **Verbosity / timing hygiene** | `--quiet`/`--verbose`, a TTY stderr line `fit in 0.91s on RTX 5090, emu40`, clean exit codes. | The timing-on-device line quietly reinforces the speed story on every run without polluting pipelines. | **Yes** | **S** |

**Grounding:** all output already funnels through **one seam** — `emit_to_destination()` in `src/app/cmd_emit.hpp` — the single correct insertion point; no per-command surgery. Hard constraint: golden ctests diff CSV/JSON byte-for-byte, so pretty must be a *new* format, never a mutation of existing emitters. `PopResolver` already holds the full label vector + by_name map (Levenshtein drops straight in). Zero isatty/color/progress/completion code exists today (grep-confirmed).

---

### Area 2 — Visualization & Publication Figures: the `steppe.plot` layer

**Vision.** A result object already knows how to draw itself: `qpadm(...).plot()` yields a journal-ready stacked weight bar; `.plot(backend="plotly")` makes it hoverable; `.plot(backend="terminal")` sketches it over SSH. One coherent `steppe.plot` namespace speaks the popgen visual vocabulary natively, exports at exact journal specs, and stamps every figure with citable provenance — so "I ran the analysis" → "this is Figure 3" is minutes, not an afternoon of ggplot. And because a fit is instant, the same plotting code powers a live explorer where toggling a source re-fits and redraws in under a second.

| Feature | What | Why it delights | Speed? | Effort |
|---|---|---|---|---|
| **`steppe.plot` core + journal theme** *(foundation)* | Functions take result objects directly, dispatch on `backend=` (mpl default / **plotnine** / plotly / altair / terminal). `theme_steppe()` with a **deterministic shared source→color map** so "Yamnaya" is the same blue in every figure of a paper. | Kills the biggest figure tax: wiring result tables into a plot library by hand. plotnine is first-class because the audience is ex-R/ggplot2 — zero migration cost. | No | M |
| **qpAdm stacked weight bar** | `plot.weights(result)` → horizontal stacked bar per target, SE whiskers, **infeasible models hatched/red** from the existing `.feasible` flag, p + n_right annotated. | THE figure qpAdm exists to make, hand-rolled every time today. Auto-flagging infeasible models in the plot catches the #1 interpretation mistake before a reviewer does. | No | **S** |
| **f4 / D / f3 forest plot** | `plot.forest(...)`: row per quartet, point + ±3·SE whiskers, zero line, significance shading, negative-f3 "admixture signal" region shaded, auto-sorted by Z. | Forest plots are in nearly every aDNA paper and pure ggplot drudgery. The shading encodes the interpretation rules visually. | No | **S** |
| **DATES decay curve** | `plot.decay(result)`: binned covariance scatter + fitted exponential overlay, date±SE (+ optional years-BP), the >0.5 cM excluded region shaded per ALDER/DATES convention. | The figure that makes a DATES result credible to reviewers. steppe already returns the binned curve — this just draws the publication version. | No | **S** |
| **f2 / f3 clustered heatmap** | `plot.heatmap(F2Blocks)`: P×P from `to_numpy()` mean, optional hierarchical-clustering dendrograms so pops auto-group. | Turns the opaque (P,P,n_block) tensor into an instantly legible structure picture, no manual averaging. | No | **S** |
| **qpGraph admixture-graph DAG** | `plot.graph(result)` via graphviz: drift edges solid w/ lengths, admixture edges dashed w/ weight % + CI, **worst-residual edge highlighted** (`worst_residual_z` already exposed), fit score in corner. | Graph diagrams are notoriously painful to lay out by hand (people export to Inkscape). Auto-layout + surfacing the weak edge makes graph QC visual. | No | M |
| **MDS / PCA embedding from f2** | `plot.embed(F2Blocks, method=...)`: classical MDS/PCoA on the f2 matrix → 2D labeled population scatter, no genotypes needed. Live variant re-embeds as you toggle pops. | Gives the familiar "PCA cloud" straight off the f2 cache; the live re-embed (drop an outlier, watch the cloud relax) is the exploratory move batch UX forbids. | **Yes** | M |
| **Rotation "competition" ranked plot** | `plot.rotation(...)`: every model sorted by p, feasible colored / rejected greyed, p=0.05 line, hover for source set. Interactive: a live p-threshold slider. | Rotation produces a flood of models triaged in spreadsheets today. Seeing the whole landscape with a live threshold is the payoff of doing thousands of fits at once. | **Yes** | M |
| **Terminal tier (plots over SSH)** | `backend='terminal'` (plotext+rich): weight bar as colored blocks, forest as ascii whiskers, decay as braille. Wired as `steppe qpadm ... --plot`. | This audience lives in headless HPC SSH sessions. An instant in-terminal sketch where they ran it respects how they actually work. | No | M |
| **Journal export + multi-panel composer** | `plot.save(fig, 'fig3.pdf', journal='nature')`: exact mm column widths, 300+ dpi, embedded fonts, colorblind check. `plot.panel({'A':graph,'B':weights})` for labeled multi-panel. | Removes the last-mile fight with journal formatting that every figure goes through. One call → submission-ready. | No | **S** |
| **Figure provenance + auto methods/citation** | Every figure embeds metadata (version, command, data hash, pop sets, precision). `plot.caption(result)` emits a paste-ready caption + Methods paragraph + BibTeX. | Directly attacks reproducibility/citation pain — recover how a figure was made *from the figure file* — and quietly ensures steppe gets cited. | No | M |
| **Live qpAdm explorer widget** *(killer interactive)* | `steppe.explore.qpadm(...)`: ipywidgets/anywidget panel; toggling left/right pops re-runs the GPU fit (<1s over in-VRAM f2) and live-updates the weight bar + p + feasibility. "Pin to figure" snapshots via the same `plot.weights`. | The experience AT2-in-R structurally cannot offer. The single most compelling proof that steppe's speed changes the *science*, not just the runtime. | **Yes** | L |

**Grounding:** every result class **already** exposes exactly the tidy DataFrame a plot needs (`.weights`, `.table`, `.edges`+`.weights`+`.worst_residual_z`, `.curve_cm`/`.curve_corr`, `F2Blocks.to_numpy()`), so most of this layer is a presentation skin over existing outputs, **not new compute**. Keep graphviz an optional extra so the base wheel stays lean.

---

### Area 3 — Notebooks & Interactive Apps (marimo-first)

**Vision.** Open one marimo notebook, type "Bell_Beaker" into a searchable AADR picker, grab a slider, and watch qpAdm weights, p-value, and a residual-z heatmap redraw in real time — the modeling loop AT2's Shiny GUI could never close. Because marimo notebooks are pure git-diffable Python that deploy as web apps with one command, every exploration is also a reproducible, shareable artifact. The north star: the fastest path from "I wonder if X is a source for Y" to a defensible, citable answer, with nothing to install and nothing to wait for.

| Feature | What | Why it delights | Speed? | Effort |
|---|---|---|---|---|
| **`steppe explore` reactive qpAdm explorer** *(flagship)* | A marimo app wrapping `qpadm`+`read_f2`: searchable target, left/right multiselects, sliders (rank/fudge/als_iters). Any change re-runs only the fit cell → live weight bar + p-gauge + Altair residual-z heatmap. f2 loaded once via `persistent_cache`; debounced sliders fire on mouse-up. | A flight simulator for admixture modeling. Researchers iterate on right-set composition (the part that actually decides outcomes) and watch the residual heatmap light up — intuition no static table gives. | **Yes** | M |
| **Reproduced-study notebooks** (Haak 2015, Olalde 2018/19) | Curated, runnable marimo notebooks that reproduce headline qpAdm/f4 results end-to-end, asserting the numbers match the paper, then drop you into the explorer seeded with that paper's pops. Shipped under `steppe.examples`. | Answers the skeptic's first question — "does it reproduce the canon?" — runnably, in seconds. Doubles as the best onboarding and as living parity docs. | No | M |
| **AADR population picker widget** | anywidget/mo.ui element over the .anno: fuzzy type-ahead surfacing n, coverage, date range, region + a small map; returns a plain Python label list that composes into `qpadm(...)`. Reused everywhere. | Kills the most-cited day-to-day pain. "grep the .anno and pray" → "type three letters, see coverage and a map, click." | No | M |
| **Rotation leaderboard** | A sortable/filterable mo.ui table over `qpadm_search(as_dataframe=True)` (filter feasible / p>alpha / source count); click a row → drill into its weights + heatmap. Changing rights re-runs the whole rotation live. | Right-set selection is qpAdm's dark art. Making the whole competitive field recompute live as you tweak outgroups turns model selection into direct manipulation. | **Yes** | M |
| **`steppe app` one-command deploy** | Thin wrapper over `marimo run` launching any explorer in app mode (code hidden) on a host/port; ships a Dockerfile pinning CUDA + the wheel + notebooks. | Mirrors and out-classes AT2's Shiny GUI: same point-and-click for non-coders, but reactive and instant. One GPU box serves a whole lab a live modeling tool. | **Yes** | **S** |
| **Provenance & methods cell** | `steppe.provenance()` at a notebook's end emits version, AADR hash, pop sets, params, GPU + a paste-ready Methods paragraph + BibTeX. | Reproducibility and "how do I cite/describe this" are real pains; auto-generated from the actual run, the notebook becomes self-documenting. | No | **S** |
| **Sweep explorer** | Brushable Altair `\|Z\|` histogram + linked top-quartets table + per-population facets over the billions-scale all-quartets sweep output. | The 2.5B-quartet sweep is a uniquely-steppe capability with no exploration surface today. Makes the firehose actually browsable. | **Yes** | L *(gap: sweep has no Python facade yet)* |

**Grounding:** the Python facade already returns pandas and maps 1:1 onto marimo UI (dropdowns→pops, sliders→rank/fudge, checkbox→allow_negative_weights). `persistent_cache` fits the precompute-once/fit-many design: load f2 once, only the sub-second fit re-runs. **Constraint:** steppe is GPU-only — no WASM/pyodide real fits; apps run server-side on a CUDA box, in-docs interactivity must be precomputed "replay." **Gaps:** the f4/D sweep has no Python facade (CLI-only); the .anno parser isn't in the Python surface yet (shared with Area 4).

---

### Area 4 — Data & Population Discovery: the AADR metadata layer

**Vision.** A researcher should never again hand-grep a 27,594-row, 45-column .anno to guess the right Group ID. `steppe pops` indexes the annotation once and becomes a fast, fuzzy, version-aware search/browse/validate layer — and because a fit is sub-second, discovery and modeling fuse into one loop: find candidate pops, multi-select, watch the fit land live.

| Feature | What | Why it delights | Speed? | Effort |
|---|---|---|---|---|
| **`steppe pops index`** *(foundation)* | One-time builder parsing the real AADR .anno (45 cols, 27,594 rows, 4,265 Group IDs) into a columnar store (parquet/sqlite) keyed by Group ID with typed fields: n, date BP, coverage, SNPs-hit, country, lat/long, sex mix, ASSESSMENT quality, datatype suffix. Stamps the AADR version; cross-checks an f2-dir's pops.txt. | Turns a 45-column annotation horror (multiline quoted headers, 1,849 singletons) into something queryable in ms, and tells you up front which f2 pops are real vs typos. | No | M |
| **`steppe pops` search/filter/browse** | Composable AND filters: name substring, `--region/--country`, `--culture`, `--bp-from/-to` or `--period`, `--min-coverage/-snps/-n`, `--quality pass`, `--datatype`. Reuses existing csv/json/table emitters. | Replaces "open .anno in a spreadsheet, Ctrl-F, eyeball coverage, copy a label, hope it's spelled right." "Yamnaya" alone matches 52 groups across 9 countries. | No | M |
| **Fuzzy validation on EVERY pop-taking command** | Wire the catalog into `PopResolver`: a typo ranks suggestions by edit-distance + shared tokens against both pops.txt and the catalog, with sample counts: *"did you mean Russia_Samara_EBA_Yamnaya (n=10, cov 1.2x)?"* Plus `steppe pops validate pops.txt` to pre-flight a long run. | The highest-frequency papercut: compound labels are near-impossible to type, and AT2's failure is a cryptic mid-script R error. Pre-flights a 24s rotation so a misspelling doesn't cost the run. | No | **S** |
| **Population data cards — `steppe pops show`** | Rich per-pop card: n + sample IDs, date histogram sparkline, coverage distribution, SNPs-hit, country mini-map, sex split, quality breakdown, decoded suffix legend (.SG/.HO/-o). JSON variant too. | Answers the exact pre-trust questions — how many? how old? how good? any contamination flags? — without cross-referencing five columns by hand. | No | M |
| **Prefix listing + tab completion** | `steppe pops --prefix Italy_`, a `--tree` Country→Period→Culture outline, and generated completion so `--target <TAB>` offers real AADR groups. | The namespace is hierarchical by convention but nothing exposes it. Prefix + TAB makes the structure walkable instead of memorized. | No | **S** |
| **Python discovery API — `steppe.pops`** | `pops.search(...)`/`.card(...)`/`.samples(...)`/`.suggest(...)`/`.validate(...)` returning DataFrames, composing with the existing qpadm/f4 facade. | Meets the Jupyter/marimo half of the audience; the find→fit story becomes notebook-native. | No | M |
| **Version-aware registry** | Index several .anno versions; answer "pop X: in v62 (n=8) and v66 (n=12); renamed from Russia_Yamnaya". `steppe pops diff v62 v66`. | Reproducibility made concrete: the Reich lab DOES rename groups between releases, and papers cite labels that may not exist in your version. | No | M |
| **Interactive picker → live fit — `steppe pops pick`** | fzf-style full-screen picker: fuzzy-filter all 4,265 groups, preview-pane data card, multi-select into Target/Left/Right slots, "Fit now" pipes the selection straight into qpAdm and renders weights+p+feasibility inline — tweak rights, re-fit instantly. | The experience AT2 physically cannot have: discover → assemble → see if it works → swap a reference → re-fit, all in one screen at thought-speed. Like Poseidon's `trident` fused with instant compute. | **Yes** | L |
| **Sample-set builder — `steppe pops build`** | After filtering, emit selected sample IDs as an `.ind`/poplist ready for `extract-f2`, with an optional merged custom group label. | Closes "I found the right samples" → "now run it" (today manual awk/grep on the .ind). Bakes reproducible inclusion criteria into the artifact. | No | M |

**Grounding (real .anno on disk):** `/home/suzunik/steppe/aadr/v66.p1_HO.aadr.PUB.anno` = 45 tab cols, 27,594 rows, 4,265 Group IDs, 1,849 singletons; ASSESSMENT col = Pass 23,053 / Questionable 963 / CRITICAL 373. **Parser caveat:** headers + fields contain quoted commas *and embedded newlines* — needs a real quoted-TSV reader, not naive split. `PopResolver` is already the single name→index seam with fail-fast — "did you mean" drops in exactly there. Pure metadata search is **not** GPU-bound (catalog is tiny); the speed payoff is *indirect* — discovery is the on-ramp to the sub-second fit (picker→live-fit).

---

### Area 5 — Workflow, Reporting, Reproducibility & Onboarding

**Vision.** Go from a question — "is this population admixed, and from whom?" — to a cited, publication-ready answer (table, figure, plain-language reading, methods paragraph) in under a minute, without writing R or ggplot. Every run carries its own reproducible recipe, so reproducibility is a free byproduct of using the tool. The two adoption-gating pains (setup, label hunt) are solved in the first two minutes by a guided quickstart on real data.

| Feature | What | Why it delights | Speed? | Effort |
|---|---|---|---|---|
| **`steppe quickstart` + `steppe doctor`** | quickstart points at a small real-AADR subset, prints the exact command it will run, runs one qpAdm fit, annotates the output inline, and emits the report + manifest. `doctor` checks CUDA/driver/VRAM + data dirs with a green/red readiness summary. | The stated pains are "setup is fiddly" + onboarding. A copy-pasteable success on **real** data in two minutes converts skeptics; doctor kills the "why won't it find my GPU/data" tickets. | No | **S** |
| **Run → report generator (`--report html\|pdf`)** | Any fit-class command emits a self-contained HTML/PDF: result table + auto-figure + a plain-language interpretation paragraph + an auto-written Methods + citation block. No R, no ggplot. | Publication figures are manual toil today; this produces a credible shareable artifact straight from the run, trivial to drop in a slide or supplement. | **Yes** | M |
| **Provenance manifest + `steppe replay`** | Every run writes `run.json`: version+SHA, CUDA/GPU, exact subcommand+params, resolved pop lists, input content hashes, wall-clock, result summary. `steppe replay run.json` re-runs and **diffs**, flagging drift. The QIIME-2 Provenance-Replay pattern for popgen. | Reproducibility is hard and reviewers increasingly demand it. An automatic byproduct, not a discipline — and the data-hash check catches the classic "which AADR version did I actually use?" silent error. | No | M |
| **Result-aware diagnostics & guardrails** | Inline plain-language flags from the Status taxonomy: negative/>1 weights, rank-deficient/non-SPD covariance, p just below threshold, SE swamping the estimate, low SNP overlap — plus what to try next. | Interpreting qpAdm is a known stumbling block. Surfacing the "should I trust this fit?" heuristics inline prevents classic mistakes and makes steppe feel like it has expertise baked in. | No | **S** |
| **Cite-this-analysis** | `steppe cite run.json` → a Methods paragraph templated from the manifest (real params filled in) + BibTeX for steppe + AT2 + the AADR DOI; optional `steppe publish` deposits a bundle to Zenodo for a per-analysis DOI. | Citation/credit is underserved. Auto-writing the methods sentence with actual params reduces methods-section errors; the DOI gives the analysis a citable identity AT2-in-R lacks. | No | M |
| **Content-addressed f2/project cache** | Cache keyed on (genotype hash + pop set + extraction params) auto-reuses f2 blocks. A `steppe init` project dir + `steppe.lock` + a run log (`steppe ls`/`log`). | f2 extraction is the expensive I/O-bound front; transparent caching makes the exploratory loop feel instant because only the cheap GLS solve re-runs. The keystone the interactive features build on. | **Yes** | M |
| **Analysis recipes — `steppe ask`** | Named recipes mapping a QUESTION to the right multi-step chain: `steppe ask is-admixed --target X --pool …` runs an f3/D screen → a rotation → ranks feasible models → renders the report. Also shareable `recipe.yaml`. | Users think in questions, not f-stat-call sequences. Encoding the community's accepted workflows (rotation, cladality, qpGraph scaffolding) embeds best-practice methodology newcomers need. | **Yes** | L |
| **`steppe explore` live-refit TUI/notebook** | f2 resident in VRAM; edit target/left/right and the fit re-runs live — add/drop a source, watch weights + p + the ancestry bar update; a slider over the right-set re-fits on every tick. | The headline thing AT2-in-R can NEVER do. Model selection becomes tactile — develop intuition by wiggling the model, not editing a script and waiting. | **Yes** | L |

**Grounding:** the Status taxonomy (RANK_DEFICIENT/NON_SPD_COVARIANCE) is result *values*, not exceptions — directly reusable for guardrails. `result_emit.cpp` has a clean single emit path but **no** provenance/manifest/citation today (greenfield); a `--config`/RunConfig hook is the natural manifest anchor. Measured speed enabling the interactive set (real AADR, single 5090): qpAdm 0.92s, rotate C(18,5)=8568 in 24.2s, DATES 2.5s — sub-second GLS solves over in-VRAM f2 are exactly what makes live-refit viable *if* f2 stays device-resident (the cache feature is the enabler).

---

### Area 6 — The North Star: the dream interactive steppe

**Vision.** You open `steppe studio` in a browser, your AADR f2-cache already resident in VRAM, and you **think out loud with the data** — type a target and watch a qpAdm model snap to a p-value, drag an outgroup and watch feasibility flip, lasso a region+millennium on a map and watch the rotation re-rank, pull an edge on an admixture graph and watch the score recolor. steppe stops being "a faster qpAdm" and becomes the place the field goes to ask questions of the ancient past in real time.

**The linchpin (build first, everything depends on it): `steppe serve` — a long-lived daemon** holding the f2 tensor + cached Resources (ADR-0005) in VRAM, exposing a thin local websocket/HTTP API over the Python facade. Cold qpadm is 0.9s but the GPU compute is sub-200ms; the daemon pays the load *once* so every interaction is the sub-200ms hit. Without it, "interactive" is a lie. *(Effort M; FastAPI/uvicorn + websockets, a resident F2Blocks handle, jsonl streaming.)*

| Feature | What | Why it delights | Speed? | Effort |
|---|---|---|---|---|
| **steppe STUDIO** — qpAdm-as-you-type workbench *(flagship)* | Browser app: target box + draggable Source / Right chips; center live weights bar + p-gauge + residual table; everything re-fits through the daemon in <0.3s as you type/drag. A rotation slider sweeps the outgroup set, feasible models glowing. | Kills the write→wait→parse→tweak loop that defines the field's day. Turns the most-used command into a Tableau-slider-grade live instrument. | **Yes** | L |
| **Population Relationship Explorer** | Pick any AADR pop → ranked top affinities (outgroup-f3 partners, attracted f4 quartets) as a force-directed graph; click a node and the sweep re-runs centered on it — browse the web like Spotify "fans also like." Live `min \|Z\|` slider. | Answers "what is this pop most like / what's a good source or outgroup for it?" — today only approximated by reading papers. Pure discovery, impossible at R speeds. | **Yes** | M–L |
| **Live Admixture-Graph Builder** | Figma-style node editor; draw a drift edge or add an admixture node and the graph re-scores on every edit; residuals recolor edges live; a "suggest next edge" fires the bounded topology search and previews ghost edges. | Graph fitting is today's most artisanal, most painful AT2 task — blind edge-list editing then waiting. Makes "what if I add an edge here?" instant: design, not guesswork. | **Yes** | L |
| **AADR Atlas** — map + time-scrub | A geographic+temporal map; scrub a time slider and watch samples appear/migrate; **lasso a region+window → that selection becomes your source pool**, the rotation runs live and annotates the map. The map is both input and output. | aDNA is inherently spatiotemporal but the tooling makes you think in text files. Turns geography into a query. | **Yes** | L |
| **Reproducible-session / `steppe share`** | One click exports a runnable script that reproduces the exact fit + a styled publication figure + a citation block (version, AADR version, pops/args, content hash); optional read-only shareable URL. | Kills two named pains (manual figures, hard reproducibility) and is the credibility bridge — reviewers can re-run what they see. Makes playful exploration publishable. | No | M |

**Guardrail (state once):** making qpAdm this fast risks industrializing p-hacking. The honest design answer — and a differentiator — is to bake **stability/bootstrap halos** (a shaking weight bar / confidence ribbon, cheap on GPU) and **"candidate, not answer"** framing into the UI so the speed produces *better* science, not just more model-fishing.

---

## 3. Prioritized Matrix — Impact × Feasibility

### Instant wins (high impact, low effort) — *do these first*
These are mostly app-layer, parity-safe, and hit the named pains directly.

| Feature | Area | Effort | Why it's a win |
|---|---|---|---|
| **qpAdm model line** | CLI | S | Highest delight-per-line in the whole plan; "qpAdm as a sentence." |
| **Fuzzy "did you mean"** | CLI/Data | S | Kills the #1 daily papercut; `PopResolver` already holds the data. |
| **Pretty `auto` output + tables** | CLI | M | Foundation everything visual sits on; one seam, zero parity risk. |
| **qpAdm weight bar + f4 forest + DATES decay + f2 heatmap** | Viz | S each | The four core figures, pure presentation over existing DataFrames. |
| **`steppe pops` search + index** | Data | M | The front door; the most-requested missing capability. |
| **Examples-rich `--help` + `steppe examples`** | CLI | S | Fastest path from installed → first result. |
| **`steppe quickstart` + `doctor`** | Workflow | S | Solves setup + onboarding in two minutes on real data. |
| **Result-aware guardrails** | Workflow | S | Prevents the classic interpretation mistakes inline. |
| **`steppe explore` marimo notebook** | Notebooks | M | The cheapest demonstration that the speed changes the science. |

### Big bets (high impact, high effort) — *the differentiators, post-0.1.0*
- **`steppe serve` daemon** (Area 6 linchpin) → unlocks everything live.
- **steppe STUDIO** browser workbench (qpAdm-as-you-type).
- **Live qpAdm explorer widget / interactive picker→live-fit** (`steppe pops pick`).
- **Live Admixture-Graph Builder** + **Population Relationship Explorer**.
- **AADR Atlas** (map + time-scrub → lasso-to-rotation).
- **`steppe ask` recipes** + the content-addressed f2/project cache (the cache is the enabler under all live features).

### Steady value (medium impact, low–medium effort) — *fill in around the wins*
Progress bars, terminal sparklines, shell completion, `--explain`, plain `steppe pops show` cards, the journal-export presets, the provenance manifest + `replay`, `cite`, figure provenance, the rotation/leaderboard plots, the terminal-tier plots.

### Lower priority / dependency-gated
Sweep explorer (needs the sweep Python facade first), version-aware registry, Zenodo `publish`, sample-set builder, collaborative multiplayer.

**The 2×2, in one line:** the *instant wins* are where IMPACT and FEASIBILITY both peak — ship them. The *big bets* are where steppe becomes a different category of tool — but every one of them depends on the daemon + the f2 cache, so those two infrastructure pieces are the highest-leverage post-ship investment.

---

## 4. The Recommended Pre-Ship Experience Sprint (ship with / around 0.1.0)

A tight, sequenced, mostly-app-layer set that makes steppe feel **finished and credible** on day one without touching the golden-gated engine. Everything here is S/M effort, parity-safe, and hits a named pain.

**Phase A — The terminal stops being hostile** *(the foundation + the headline)*
1. **isatty `auto` output mode** in `emit_to_destination` (M) — the enabling primitive; csv/json stay byte-identical.
2. **Pretty significance-colored tables** (M) + **the qpAdm model line** (S) — the "qpAdm as a sentence" moment.
3. **Fuzzy "did you mean"** in `PopResolver` (S) — kills the typo loop.
4. **Examples-rich `--help` + `steppe examples`** (S).

**Phase B — Finding populations stops being miserable**
5. **`steppe pops index`** over the real v66 .anno (M) — the catalog.
6. **`steppe pops` search/filter** (M) + **fuzzy validation wired into every pop-taking command** (S, reuses Phase A's scorer).

**Phase C — Results become figures and stories**
7. The **`steppe.plot` core + journal theme** (M) and the **four core figures**: qpAdm weight bar, f4/D forest, DATES decay, f2 heatmap (S each) — with the **shared deterministic palette** and **journal export presets** (S).
8. **Result-aware diagnostics/guardrails** (S) + **`--explain`** (M).

**Phase D — Onboarding + one interactive showpiece**
9. **`steppe quickstart` + `steppe doctor`** (S) — real-AADR success in two minutes.
10. **One `steppe explore` marimo notebook** (M) wrapping the existing `qpadm` facade with the AADR picker — the single demo that shows the speed changing the science.
11. **Provenance manifest (`run.json`) + a paste-ready Methods/citation block** (M) — reproducibility as a free byproduct.

**Sequencing logic:** A is the foundation (one seam, zero risk) and ships the single highest-delight change (the model line). B attacks the most-cited pain and *reuses A's edit-distance scorer*. C makes the tool publication-credible with pure presentation code over DataFrames that already exist. D gives onboarding + exactly one taste of the interactive future to anchor the pitch. Live progress bars (M) and terminal sparklines (M) are stretch goals for this sprint if Phase A lands early.

**Explicitly deferred to post-1.0 (flag as "coming"):** the `steppe serve` daemon and **steppe STUDIO**; the live qpAdm explorer *widget* and `steppe pops pick` live-fit; the graph builder, relationship explorer, and AADR Atlas; `steppe ask` recipes + the f2/project cache; `steppe app` deploy + Docker; the sweep explorer (needs the sweep facade); Zenodo `publish` + version-aware registry. These are the differentiators — but they ride on infrastructure (daemon + cache) that shouldn't be rushed into 0.1.0.

**What this sprint buys:** a researcher downloads steppe, runs `steppe doctor` then `steppe quickstart`, gets a real fit annotated in plain English in two minutes, finds their populations with `steppe pops --culture Yamnaya`, runs a fit that prints a colored one-line verdict, calls `.plot()` for a journal-ready figure with a paste-able Methods paragraph, and opens one marimo notebook to *feel* the speed. That is a complete, credible, delightful first hour — and it ships without touching a single golden test.

---

## 5. The Wild Ideas Worth Remembering

The ambitious bets that, even if parked, define where steppe is going. Flagged as wild.

- **qpAdm AUTOPILOT / "find me the model" button** — given just a target, brute-force the full rotation across thousands of source×outgroup combos and return a *ranked shortlist* of defensible models (feasible, p>threshold, parsimonious, stable under SNP resampling), each with a one-line rationale. Turns the field's hardest judgement call into a reviewable menu. **Guardrail-critical:** frame as "candidates to scrutinize," never "the answer."

- **steppe STUDIO** — the browser workbench where qpAdm re-fits as you type and drag. The flagship of the whole vision; the thing people *see* working on a projector and decide to adopt.

- **The AADR Atlas / time-map** — lasso a region and a millennium on a map and *that becomes your source pool*; the rotation runs live and annotates the map. Geography as a query. "Ancient-DNA Google Earth."

- **Natural-language front door** — `steppe ask "was Bronze Age Britain mostly Beaker-derived?"` where an LLM only *plans* (resolves fuzzy pop names + picks args from the AADR index) and the GPU computes the real, parity-validated numbers. No hallucinated statistics.

- **Stability-as-you-go (bootstrap halos)** — because SNP-block resampling is cheap on GPU, every live result carries a *shaking* weight bar / confidence ribbon so you SEE how fragile a model is the instant you build it. The honest antidote to speed-enabled p-hacking — and a genuine scientific differentiator.

- **Live Admixture-Graph Builder** — a Figma-style canvas that re-scores on every edge edit, with "suggest next edge" previewing the topology search as ghost edges. Graph fitting becomes design, not guesswork.

- **Real-time collaborative sessions** — a PI and a student on different continents drag the same admixture graph / map and watch each other's edits re-score. Review and teaching happen *in* the live tool, not over emailed CSVs.

- **Conference / live-demo mode** — answer an audience's "but what if you swap in Yamnaya?" on stage, live, because the re-fit is instant. The way tools get adopted is being seen working.

- **Figure-diff for re-analysis** — re-run on a new AADR release and overlay the new weight bars / decay curve / graph on the old (ghosted) to see *what changed* between dataset versions. Turns re-analysis into a visual diff.

- **steppe-studio as a local web app** (FastAPI + JS, the fast answer to AT2's Shiny) tying Studio + the explorer + the f2 MDS cloud + the rotation leaderboard into one real-time "explore demographic history" loop.

---

*The compute is done. This is the one last sprint that decides whether anyone gets to enjoy it. The speed is the product — the experience layer is how the world finally feels it.*

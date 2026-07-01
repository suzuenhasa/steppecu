# Area 2 — Visualization & Publication Figures: `steppe.plot`

**Engineering-architecture doc.** This is the *how* (folder scaffolding, tech-stack decisions,
interfaces to existing code, packaging) for the visualization layer. The *what/why* — the
feature list, the delight rationale, the pre-ship sprint sequencing — lives in the product plan,
`docs/EXPERIENCE-PLAN.md` § "Area 2 — Visualization & Publication Figures". The two are
complementary: read the plan for the features, read this for the module boundaries and the deps.
This doc does **not** restate the plan's feature table — but § 2 grounds the *demand* (the canonical
figure kinds and their current pain) that fixes the module boundaries below.

---

## 1. Role of this area

`steppe.plot` is the presentation skin over the *already-tidy* result objects the nanobind facade
returns. The core insight the plan states and the code confirms: **every result class already
exposes exactly the DataFrame a figure needs**, so this layer is presentation-over-existing-data,
not new compute:

| Figure | Consumes (facade, `bindings/steppe/__init__.py`) |
|---|---|
| qpAdm stacked weight bar | `QpAdmResult.weights` DF `[target,left,weight,se,z]` + `.feasible`, `.p`, `.status` |
| f4 / D / f3 forest | `F4Result.table` / `F3Result.table` `[pop1..,est,se,z,p]` |
| DATES decay curve | `DatesResult.curve` DF `[cm,corr]` + `.date_gen`, `.se` |
| f2 / f3 clustered heatmap | `F2Blocks.to_numpy()` → `(P,P,n_block)` float64, `.pops` |
| qpGraph admixture DAG | `QpGraphResult.edges` `[from,to,length]` + `.weights` `[from,to,weight,low,high]` + `.worst_residual_z`, `.score` |
| MDS/PCoA embedding | `F2Blocks.to_numpy()` mean-over-blocks → classical MDS |
| Rotation "competition" | `qpadm_search(..., as_dataframe=True)` → `[model_index,left,p,feasible,status,...]` |

The area owns three cross-cutting concerns beyond "draw a chart": (a) a **backend-dispatch**
mechanism (`backend="mpl"|"plotnine"|"plotly"|"altair"|"terminal"`), (b) a **journal theme +
deterministic pop→color palette** so "Yamnaya" is the same blue in every panel of a paper, and
(c) **publication export + figure provenance** (exact mm widths, embedded fonts, a paste-ready
Methods/BibTeX caption recovered from the figure). See the plan for why each matters.

**Non-goals / boundaries** (owned elsewhere, referenced not rebuilt):
- The reactive *notebook app* and the marimo `steppe explore` loop → **Area 3**. Area 2 supplies
  the redraw primitives (a figure that re-renders in <1 render); Area 3 orchestrates the
  interaction.
- The resident-f2 **daemon** that makes "re-fit as you drag" sub-200 ms → **Area 6** (`steppe
  serve`). Area 2's *live* variants (live re-embed, live rotation threshold) call the facade or
  the daemon; they do not implement the fit loop.
- The **AADR pops catalog** (`steppe.pops`) → **Area 4**. Area 2 *optionally* colors an embedding
  by region/period if the catalog is importable, but has no hard dependency on it.
- The **provenance manifest** `run.json` schema → **Area 5**. Area 2's `caption()` /
  figure-metadata embedding reads that schema when present and falls back to a self-contained
  minimal stamp otherwise.
- The **C++ CLI `--plot` flag** wiring → **Area 1** (the CLI is C++; it cannot import the Python
  renderer in-process — see § 12 open questions).

---

## 2. The visualization vocabulary — what practitioners want

This section is the **demand grounding** for the seven `PlotKind`s the architecture lists — *why*
those kinds exist and what fixes the module boundaries below — not a restatement of the plan's
feature table. The field lives on ~9 canonical figures, every one hand-built in static R/ggplot;
the recurring pains (no hover, no live re-fit, inconsistent per-pop colors, silent uncertainty, no
model/version diff, no escape from static R) are exactly the cross-cutting concerns § 1 claims.

| Figure | Made today with | What hurts | steppe's answer |
|---|---|---|---|
| **PCA / MDS cloud** | smartpca (EIGENSOFT) + hand ggplot | static PNG, no hover ("which dot is this sample?"); `lsqproject` shrinkage + missing-data uncertainty never drawn, so a dot's position reads as exact; re-do after dropping an outlier = full smartpca rerun | `plot.embed` — classical MDS off the resident f2, genotype-free; re-embeds live (§ 10) |
| **ADMIXTURE / STRUCTURE Q-bar** | pong / DISTRUCT / CLUMPAK | label-switching (pong exists only to align components), no principled K, hand sample-ordering | *adjacent* (steppe is qpAdm-not-ADMIXTURE) — the ordering / color / alignment lessons transfer to the qpAdm weight bar |
| **qpAdm weight bar + model table** | AT2 in R/ggplot, hand-wired each time | infeasible models (neg / >1 weight, p just under α) **not visually flagged** — ships to reviewers looking identical to a good fit; p-direction (want p>0.05) inverted vs intuition, unmarked | `plot.weights` — `masks["hatch"]=~feasible` off the existing `.feasible` flag; the three p-states labeled |
| **f4 / D forest** | AT2 ggplot drudgery, one comparison at a time | the D-sign convention (who admixed when D>0) is never printed; negative-f3 "admixture signal" shading is folk knowledge | `plot.forest` — zero line, sort by \|Z\|, neg-f3 region shaded by default; convention stamped in the caption |
| **qpGraph admixture DAG** | raw `dot` pipeline → Inkscape by hand | ugly default layout, static, and — the killer — no side-by-side topology compare (a reanalysis found **1,421 graphs** fitting ≥ a published one) | `plot.graph` via graphviz, worst-residual edge highlighted; the compare/diff view is a § 10 dream |
| **TreeMix tree + residual heatmap** | TreeMix + custom R | migration-edge count *m* unresolved (OptM/ΔK); residual read impressionistically | *adjacent* — the residual-heatmap-as-QC idiom maps onto qpGraph / qpAdm residuals |
| **f2 / f3 heatmap** | hand-rolled R heatmap | the `(P,P,n_block)` tensor block-averaged by hand; clustering reorder wired by the user; dendrograms rarely included | `plot.heatmap` off `F2Blocks` — auto block-mean + `scipy.cluster.hierarchy` reorder + dendrogram |
| **DATES / ALDER decay** | ALDER/DATES emit the binned curve to a log; figure rebuilt by hand | the figure that makes a *date* credible to a reviewer, and it is pure manual ggplot every time | `plot.decay` — fitted A·exp(−λd)+c overlay, <0.5 cM region shaded |
| **Geographic sample map** | ggplot2 + maps / rnaturalearth / sf, static | no time axis (workaround = a grid of one static map per period); manual `.anno` lat/long + `Date_mean_in_BP` munging | a new **MAP** kind (§ 11) — a first-class temporal map, deferred behind the Area-4 catalog + Area-6 daemon |

The nine collapse almost exactly onto the seven specced `PlotKind`s
(`WEIGHTS/FOREST/DECAY/HEATMAP/ROTATION/EMBED/GRAPH`): the geographic map is the one that needs a
**new MAP kind** (§ 11), and ADMIXTURE-Q / TreeMix are *adjacent vocabulary* whose lessons transfer
rather than native kinds. Nothing in the established vocabulary needs a new abstraction — the
survey's first finding: the architecture is aimed correctly.

**The cross-cutting wants** (why this area owns more than a draw call, and the through-line to
§ 10–§ 11):

- **Interactivity / hover** — answer "which sample is this dot?" and "what is this quartet's exact Z?" — the § 1 backend-dispatch to plotly/altair.
- **Live re-fit** — the figure as an instrument, not a snapshot (§ 10, riding the Area-6 daemon).
- **Publication polish + reproducible colors** — "Yamnaya" the same blue in every panel of a paper — the deterministic palette + journal export § 1 already owns.
- **Honest uncertainty** — draw the replicate *spread*, not a bare ±SE scalar the reader must trust (§ 10.1, § 10.4).
- **Model / version diff** — overlay two fits, or two AADR releases, band-by-band (§ 10.3).
- **Escaping static R** — the migrating AT2 audience wants its ggplot muscle memory (plotnine) *and* the interactivity R never gave it.

The nine kinds are covered today; the *wants* are what § 10 (the dream tier) and § 11 (the
rendering engineering) build out.

---

## 3. Tech-stack decision

The design principle from the plan and the packaging memo: **the base wheel stays `numpy`-only**;
every plotting dependency is an optional extra, imported lazily (the exact pattern the facade
already uses for `pandas` via `_require_pandas`). steppe is GPU-only and server-side, so there is
**no WASM/pyodide constraint on the plotting libs** — they run on the same CUDA box as the fit.

### 3.1 The rendering libraries (weighed)

| Backend | Library | Chosen role | Why over the alternatives |
|---|---|---|---|
| **`mpl`** *(default)* | **matplotlib ≥3.7** | The universal default for `.plot()`; publication raster/vector export | The one library that (a) is publication-grade (EPS/PDF vector, font embedding via `pdf.fonttype=42`), (b) runs on **py3.9** (steppe's floor), and (c) is the substrate every other static lib sits on (plotnine renders *through* mpl). Zero-friction default: if a user has `steppe[viz]`, `.plot()` just works. |
| **`plotnine`** | **plotnine ≥0.13** | The ggplot2-grammar backend for the ex-R / AT2-in-R audience | The audience migrating from ADMIXTOOLS 2 lives in `ggplot2`; plotnine is a faithful grammar-of-graphics port (actively maintained — 0.15.x as of mid-2026, MIT) so their `+ geom_* + theme_*` muscle memory transfers with **zero migration cost**. Caveat: plotnine requires **py≥3.10**, so it is a *separate* extra (`viz-ggplot`), not folded into `viz` (which must stay py3.9-installable). This is the whole reason mpl, not plotnine, is the default. |
| **`plotly`** | **plotly ≥5.20** | Hoverable/interactive figures for notebooks and Studio | For "hover a quartet to read its exact Z", 3D, and web embedding, plotly is the mature interactive choice and integrates with marimo/anywidget. Chosen over Bokeh for the richer hover/HTML-export story and the larger popgen-notebook install base. |
| **`altair`** | **altair ≥5** | Declarative Vega-Lite specs for the reactive residual/heatmap panels (Area 3's chosen interactive grammar) | Declarative, reproducible, and the grammar Area 3's marimo explorer already targets (residual-z heatmap, brushable `|Z|` histogram). **Hard limit noted**: Vega-Lite errors past ~5000 rows — so altair is fine for weights/forest/rotation-of-hundreds and the residual heatmap, but the billions-scale sweep must be pre-aggregated *before* it reaches altair (row-guard in the backend, § 6.3). Chosen alongside (not instead of) plotly because it is the natural fit for the marimo tier. |
| **`terminal`** | **plotext ≥5.2 + rich ≥13** | ASCII/braille plots over SSH; the headless-HPC tier | plotext is the current, actively-maintained terminal-plot lib (braille 4×2 markers for the DATES decay, bar/scatter/hist for weights/forest, native `rich` integration); rich draws the colored blocks/tables. Chosen over `termplotlib` (which merely shells to gnuplot — an extra system dep) and over hand-rolled ANSI. This audience lives in SSH sessions, so a `res.plot(backend="terminal")` sketch-where-you-ran-it is a credibility signal. |
| **`graphviz`** *(DAG only)* | **graphviz ≥0.20** (pure-Python DOT interface) | The qpGraph admixture-graph layout | Auto-layout of a DAG is `dot`'s job. The pure-Python **`graphviz`** package (emits DOT, shells to the system `dot` binary) is chosen over **`pygraphviz`** (which compiles against libgvc — a heavy build-time dep and a common install failure) because it is a tiny pure-python wheel with richer edge/node styling and string-keyed nodes that map 1:1 to pop labels. The only cost is a **system `dot` requirement** (documented like the CUDA runtime — see § 7). |

### 3.2 The embedding math (`plot.embed`)

Classical MDS / PCoA on the f2 matrix is an **eigendecomposition of the double-centered f2
matrix** — f2 is already a squared-distance-like quantity, so classical MDS ≈ PCA of the
populations with no genotypes needed. Decision: **implement classical MDS in-house with
`numpy`/`scipy.linalg.eigh`** (`embed.py`, ~30 lines) rather than pull `scikit-learn`. Rationale:
scipy is already a light, common dep (and needed for the heatmap's hierarchical clustering,
`scipy.cluster.hierarchy`); classical MDS is a 4-step closed form; and avoiding sklearn keeps
`viz` lean. `scikit-learn` (metric MDS, t-SNE, UMAP) stays an *opt-in* the user installs if they
ask for `method="tsne"` — the backend raises a clear "install scikit-learn" then.

### 3.3 The backend-dispatch design (the portfolio-grade core)

Modeled on **pandas' plotting-backend mechanism** (`_get_plot_backend` + the
`pandas_plotting_backends` entry-point group) — the proven pattern for "one call, swappable
renderer". steppe adopts a **spec/render split**:

```
public figure fn  ──build──▶  PlotSpec (backend-agnostic: tidy DF + semantic roles + annotations)
                                   │
                            resolve(backend=…)          # arg > env > global > "matplotlib"
                                   │
                    Backend.render_<kind>(spec, **kw) ──▶  a Figure object (mpl.Figure / plotly.Figure / str / …)
```

- A figure builder (`figures/weights.py`) turns a result into a **`PlotSpec`** — pure data +
  encoding roles (`x`, `y`, `fill`, `error`, `annotations`, `hatch_mask`, `theme`), *no library
  imported*. This is testable without any plotting dep and is the single place the "infeasible →
  hatched/red", "sort by Z", "shade negative-f3 region" *semantics* live.
- A **`Backend`** (Protocol in `backends/base.py`) renders a `PlotSpec` of a given kind. Backends
  declare a **capability set** (not every backend does every plot — `graphviz` only does `graph`,
  `terminal` skips `graph`); the dispatcher raises a clear "backend X cannot render kind Y; try
  matplotlib" instead of a deep `AttributeError`.
- **Resolution order** (mirrors pandas' `plotting.backend` option): explicit `backend=` arg →
  `STEPPE_PLOT_BACKEND` env → `steppe.plot.set_backend(...)` global → default `"matplotlib"`.
- **Extensibility**: third-party backends register via the `steppe.plot_backends` entry-point
  group, discovered lazily on first miss — so a lab can ship a house style without forking.

Why the spec/render split over "each figure fn imports mpl directly": it makes the *popgen
semantics* (the shading rules, the significance colors, the sort order) backend-independent and
unit-testable on the DataFrame alone, and makes adding plotly/altair a matter of one render method
per kind rather than re-deriving the encoding. It is more architecture than a 4-figure MVP
strictly needs, but it is the right abstraction for a namespace that must speak "the popgen visual
vocabulary" across five backends and stay a portfolio piece.

---

## 4. Fit decision — **hybrid** (new subpackage + a minimal facade extension)

**New subpackage** `bindings/steppe/plot/` is the home for essentially all of the code — it is a
pure-Python presentation layer that consumes the facade's DataFrames, so it does not belong in the
compiled `_core` or the C++ CLI. It ships *inside* the existing wheel package (scikit-build-core's
`wheel.packages = ["bindings/steppe"]` already copies the whole subtree), so **no packaging change
is needed to include the code** — only the new optional extras (§ 7) gate the *dependencies*.

**Minimal facade extension** (the "hybrid" part): add a lazy `.plot` accessor to the result
classes so `qpadm(...).plot()` works as the plan promises. This is the single touch to existing
code — additive, lazy, and modeled on the existing `_require_pandas` pattern:

```python
# bindings/steppe/__init__.py  — one new mixin, added to each result class's bases (additive)
class PlottableMixin:
    """Lazy `.plot` accessor. Imports steppe.plot ONLY when first accessed, so `import steppe`
    stays numpy-only and the [viz] deps are optional (mirrors _require_pandas)."""
    @property
    def plot(self):
        from .plot import PlotAccessor      # lazy: no matplotlib at import time
        return PlotAccessor(self)

class QpAdmResult(PlottableMixin): ...      # + QpWaveResult, F4Result, F3Result,
                                            #   F4RatioResult, DatesResult, QpGraphResult, F2Blocks
```

`PlotAccessor` (defined in `steppe.plot`, pandas-`CachedAccessor`-style) is both callable and
namespaced, so all three plan spellings work:

```python
qpadm(...).plot()                       # __call__  -> the canonical figure for this result type
qpadm(...).plot(backend="plotly")       # __call__  with a backend
qpadm(...).plot.weights(se=True)        # explicit kind
f4(...).plot.forest(sort="z")
```

Why hybrid and not "extend the facade in place": keeping the ~15 figure/backend/export modules out
of the 1000-line `__init__.py` keeps the facade a thin marshalling layer (its stated job) and lets
`steppe.plot` be imported, tested, and versioned as a unit. Why not a *separate package*
(`steppe-plot` on PyPI): the figures are meaningless without the result classes they render;
co-shipping them (code in the base wheel, deps behind an extra) is the lean, discoverable choice
and matches how `pandas` ships its own `.plot` accessor.

---

## 5. Folder scaffolding

```
bindings/steppe/
├── __init__.py                 # (EXISTING) + PlottableMixin  → 8 result classes gain `.plot`
└── plot/                       # (NEW subpackage — pure Python, deps behind [viz]*)
    ├── __init__.py             # public API + PlotAccessor + backend registry + resolve()
    ├── spec.py                 # PlotSpec dataclasses (backend-agnostic data + encoding roles)
    ├── theme.py                # theme_steppe(); journal presets (nature/science/cell/pnas mm+dpi)
    ├── palette.py              # deterministic pop→color map (hash-stable, Okabe-Ito base);
    │                           #   significance colors (|Z|≥3 red, 2–3 amber, p<0.05 green)
    ├── embed.py                # classical MDS / PCoA on the f2 matrix (numpy + scipy.linalg.eigh)
    ├── export.py               # save(fig, path, journal=…): mm width, dpi, font embed, cb-check
    ├── panel.py                # panel({"A": fig, …}) labeled multi-panel composer
    ├── provenance.py           # caption(result) → caption + Methods + BibTeX; embed metadata
    ├── figures/                # backend-AGNOSTIC builders: result → PlotSpec (the semantics)
    │   ├── __init__.py
    │   ├── weights.py          #  QpAdmResult      → stacked-bar spec (infeasible hatched)
    │   ├── forest.py           #  F4/F3Result      → forest spec (zero line, sorted by Z)
    │   ├── decay.py            #  DatesResult      → decay spec (>0.5 cM region shaded)
    │   ├── heatmap.py          #  F2Blocks         → P×P heatmap + optional dendrogram spec
    │   ├── rotation.py         #  search DataFrame → ranked-p spec (feasible colored)
    │   ├── embed.py            #  F2Blocks         → 2D MDS scatter spec (via ../embed.py)
    │   └── graph.py            #  QpGraphResult    → DOT graph model (graphviz-specific)
    └── backends/
        ├── __init__.py         # registry: name→module, entry-point discovery, capability check
        ├── base.py             # Backend Protocol; PlotKind enum; CapabilityError
        ├── _mpl.py             # matplotlib (default) — renders ALL non-graph kinds
        ├── _plotnine.py        # plotnine (ggplot) — renders the static kinds
        ├── _plotly.py          # plotly — weights/forest/rotation/embed/heatmap (interactive)
        ├── _altair.py          # altair — same, ROW-GUARDED (Vega-Lite ~5k cap)
        ├── _graphviz.py        # graphviz — the DAG (graph()); the only graph renderer
        └── _terminal.py        # plotext + rich — weights/forest/decay ascii/braille

tests/python/plot/             # (NEW) spec-level unit tests (no plotting dep) + smoke renders
    ├── test_spec.py            #  result → PlotSpec correctness (hatch mask, sort, shading rules)
    ├── test_palette.py         #  determinism: same pop → same color across calls/processes
    ├── test_embed.py           #  classical MDS numerics vs a known small case
    └── test_backends_smoke.py  #  @pytest.mark.viz: each installed backend renders without error
```

---

## 6. Key interfaces

### 6.1 Public API (`steppe/plot/__init__.py`)

```python
# ── figure builders (each returns a live Figure of the resolved backend's native type) ──
def weights(result, *, backend=None, show_se=True, weight_bar=False, ax=None, **kw): ...
def forest (results, *, backend=None, sort="z", ci=3.0, ax=None, **kw): ...   # f4/f3/D
def decay  (result, *, backend=None, years_per_gen=None, ax=None, **kw): ...  # DatesResult
def heatmap(blocks, *, backend=None, cluster=True, metric="f2", ax=None, **kw): ...  # F2Blocks
def rotation(search_df, *, backend=None, alpha=0.05, ax=None, **kw): ...      # qpadm_search DF
def embed  (blocks, *, backend=None, method="mds", k=2, color_by=None, ax=None, **kw): ...
def graph  (result, *, engine="dot", highlight_worst=True, **kw): ...         # QpGraphResult → DOT

# ── publication last-mile ──
def save(fig, path, *, journal=None, width_mm=None, dpi=600, embed_fonts=True,
         colorblind_check=True, provenance=None): ...
def panel(items: dict[str, "Figure"], *, layout="auto", labels=True, **kw): ...
def caption(result, *, style="methods") -> "Caption": ...   # .text / .methods / .bibtex

# ── theme / palette / backend control ──
def theme_steppe(base="paper"): ...
def color_of(pop: str) -> str: ...           # the deterministic per-pop color
def set_backend(name: str) -> None: ...       # global default (resolution order: arg>env>global>mpl)
def get_backend() -> str: ...
def available_backends() -> dict[str, set["PlotKind"]]:  # capability matrix

# ── the accessor bound onto result objects via PlottableMixin ──
class PlotAccessor:
    def __init__(self, obj): self._obj = obj
    def __call__(self, *, backend=None, **kw):    # canonical figure for type(obj)
        return _DEFAULT_KIND[type(self._obj)](self._obj, backend=backend, **kw)
    weights = forest = decay = heatmap = rotation = embed = graph = ...  # thin delegators
```

### 6.2 The spec/backend contract (`spec.py`, `backends/base.py`)

```python
# spec.py — backend-agnostic; imports only dataclasses + typing (NO plotting lib)
@dataclass(frozen=True)
class PlotSpec:
    kind: PlotKind                    # WEIGHTS | FOREST | DECAY | HEATMAP | ROTATION | EMBED | GRAPH
    data: "pandas.DataFrame"          # the tidy frame straight off the result accessor
    encoding: dict[str, str]          # role → column: {"x":"weight","fill":"left","error":"se",...}
    annotations: dict[str, Any]       # {"vlines":[0.0], "shade":[(0.5, inf)], "title":..., "p":0.31}
    masks: dict[str, "Series"]        # {"hatch": ~feasible, "grey": rejected}  — the semantics
    theme: ThemeSpec                  # colors, font sizes, palette map (deterministic pop→color)

# backends/base.py
class Backend(Protocol):
    name: str
    capabilities: set[PlotKind]
    def render(self, spec: PlotSpec, *, ax=None, **kw) -> Any: ...   # → native Figure/str

def resolve(name: str | None) -> Backend: ...        # arg>env>global>"matplotlib"; entry-point miss
class CapabilityError(RuntimeError): ...             # "backend 'terminal' cannot render GRAPH"
```

### 6.3 Notable per-figure semantics (why the spec layer earns its keep)

- **`weights`**: `masks["hatch"] = ~result.feasible` drives the "infeasible model hatched/red"
  rule from the *existing* `.feasible` flag; `annotations` carries `p`, `n_right`, `status`.
- **`forest`**: `annotations["vlines"]=[0.0]`, `sort="z"` sorts rows by `|z|`; f3 forest shades the
  negative-`est` region ("admixture signal") via `annotations["shade"]`.
- **`decay`**: `annotations["shade"]=[(0.0, 0.5)]` greys the excluded <0.5 cM region per
  ALDER/DATES; the fitted `A·exp(-λd)+c` overlay is computed from `date_gen` in the builder.
- **`heatmap`**: `blocks.to_numpy().mean(axis=2)` → P×P; `cluster=True` runs
  `scipy.cluster.hierarchy` to reorder + emit dendrograms.
- **`rotation`** / sweep: the **altair backend row-guards** — if `len(data) > 5000` it raises
  "altair caps at ~5k rows; pre-aggregate or use backend='plotly'/'mpl'" rather than emit a broken
  Vega spec (the documented Vega-Lite limit).
- **`graph`**: builds a `graphviz.Digraph` — drift edges solid with `length` labels, admixture
  edges dashed with `weight% [low,high]`, and the `worst_residual_z` edge recolored; the fit
  `score` printed in a corner. This is the one kind that bypasses the generic `PlotSpec` render
  and emits DOT directly (graphviz is not a "chart on axes").

### 6.4 Export & provenance

```python
# export.py — journal presets are exact mm column widths + dpi + font-embed rcParams
JOURNAL = {"nature": Preset(single_mm=89, double_mm=183, dpi=300, font="Arial",
                            pdf_fonttype=42), "science": ..., "cell": ..., "pnas": ...}
def save(fig, path, *, journal=None, width_mm=None, ...):
    # sets pdf.fonttype/ps.fonttype=42 (TrueType embed), scales to exact mm, runs a
    # colorblind simulation pass (Okabe-Ito palette is cb-safe by construction), and
    # writes provenance metadata into the file (PDF /Keywords, PNG tEXt, SVG <metadata>).

# provenance.py — the figure "knows how it was made"
def caption(result, *, style="methods") -> Caption:
    # pulls version/params/data-hash from the Area-5 run.json schema if present, else a
    # self-contained minimal stamp; returns .text (figure legend), .methods (paragraph),
    # .bibtex (steppe + ADMIXTOOLS 2 + AADR DOI).
```

---

## 7. Packaging — the extras

Base wheel stays **`numpy`-only**. The `steppe.plot` *code* ships in the base wheel (pure Python,
lazily importing its deps); the *dependencies* are optional extras added to
`[project.optional-dependencies]` in `pyproject.toml`. Split so a py3.9 publication user is not
forced to take plotnine (py≥3.10) or the interactive stack:

```toml
[project.optional-dependencies]
# ── the publication + DAG core (py3.9-safe; the Phase-C sprint deliverable) ──
viz = [
    "matplotlib>=3.7",   # default backend + all raster/vector export
    "scipy>=1.10",       # classical MDS (embed) + hierarchical clustering (heatmap)
    "pandas>=1.5",       # the result DataFrames every figure consumes
    "graphviz>=0.20",    # the qpGraph DAG (pure-Python DOT; needs system `dot`, see note)
]
# ── the ggplot backend for the ex-R / AT2 audience (SEPARATE: plotnine needs py>=3.10) ──
viz-ggplot = ["plotnine>=0.13"]
# ── the interactive tier (feeds Area 3 marimo + Area 6 Studio) ──
viz-interactive = ["plotly>=5.20", "altair>=5"]
# ── the SSH / headless-HPC terminal tier (also usable by Area 1's CLI --plot) ──
viz-terminal = ["plotext>=5.2", "rich>=13"]
# ── the scale + big-data render path (datashader raster + WebGL geo + Arrow transport) ──
#    heavy, opt-in; feeds the sweep firehose + AADR Atlas (§ 10–§ 11). base wheel stays numpy-only.
viz-scale = [
    "datashader>=0.16",    # server-side rasterization of the billion-quartet sweep (RASTER kind)
    "holoviews>=1.18",     # zoom-triggered re-aggregation wrapper (Python-only interactive sweep)
    "pydeck>=0.9",         # deck.gl geo/temporal layers (the AADR Atlas MAP kind)
    "pyarrow>=15",         # Arrow IPC transport of reduced results to the browser
]
# ── optional on-GPU aggregation (rasterize straight off VRAM; needs a matching CUDA) ──
viz-scale-gpu = ["cudf-cu13"]
# ── convenience aggregate (grows to include app/mcp when those areas land) ──
all = ["steppe[viz,viz-ggplot,viz-interactive,viz-terminal,viz-scale]"]
```

- **`graphviz` needs the system `dot` binary** (like the CUDA-13 runtime, it is a documented
  system requirement, not a pip dep). `graph()` raises a clear "graphviz `dot` not found; install
  the Graphviz system package" if `dot` is absent.
- **Lazy-import contract** (enforced by a test): `import steppe` and `import steppe.plot` must
  succeed with only `numpy` installed; the missing-dep error is raised at *first `.plot()` call*,
  naming the exact extra (`"install steppe[viz]"`), exactly as `_require_pandas` does today.
- No `[tool.scikit-build]` change: `wheel.packages = ["bindings/steppe"]` already includes the new
  `plot/` subtree; the CLI stays `STEPPE_BUILD_CLI=OFF`.
- **`viz-scale` is the first CUDA-coupled plotting extra.** The pure-`datashader` path rasterizes
  host-side and installs anywhere; only `viz-scale-gpu`'s `cudf-cu13` needs a CUDA matching the
  wheel. The datashader raster is a *reduce-for-display* (like the heatmap's block-mean), so it
  stays presentation, not new compute — but see § 12, open-question 7.

---

## 8. Shared-infra dependencies (referenced, not re-scaffolded)

| Shared infra | Owner | How Area 2 uses it |
|---|---|---|
| **(vi) result classes / DataFrames** | facade (exists) | **Primary input.** Every figure reads a `.weights`/`.table`/`.curve`/`.edges`/`to_numpy()` DF. Area 2 adds the `PlottableMixin` `.plot` accessor onto them. |
| **(v) provenance manifest `run.json`** | **Area 5** | `provenance.py`/`caption()` read the manifest schema for version/params/data-hash and embed it in the figure file; minimal self-contained stamp until Area 5 lands. |
| **(ii) AADR pops catalog `steppe.pops`** | **Area 4** | *Optional* enrichment: `embed(color_by="region")` / `heatmap` can color/annotate by region/period if `steppe.pops` is importable. Soft dep, no hard requirement. |
| **(iv) resident-f2 daemon `steppe serve`** | **Area 6** | Powers the *live* variants: live re-embed (drop a pop → cloud relaxes), live rotation p-threshold. Area 2 provides the redraw primitive; the sub-200 ms re-fit is the daemon's. Static figures need nothing from it. |
| **(iii) content-addressed f2 cache** | **Area 5** | Under the live variants (via the daemon). Not touched by static export. |
| **(i) CLI output seam `emit_to_destination`** | **Area 1** | The `steppe qpadm … --plot` terminal render is an Area-1 CLI decision that calls Area 2's `_terminal` backend — but the C++ CLI cannot import Python in-process (see § 12). |

**The live qpAdm explorer widget** (plan lists it under Area 2, "killer interactive"): the
*rendering* primitive (a `weights` figure that re-renders in place, ideally via **anywidget** so it
works in both Jupyter and marimo) belongs here; the *interaction loop* (searchable pop pickers,
sliders, the re-fit trigger) is **Area 3** (marimo) riding the **Area 6** daemon. Area 2's
contribution is a `plot.live.weights(handle)` that returns an anywidget whose `update(result)`
swaps the bar without a full re-layout. Flagged as a boundary (§ 12).

---

## 9. Intra-area build order

1. **Dispatch core + default backend + theme** — `spec.py`, `backends/base.py`, `backends/_mpl.py`,
   `theme.py`, `palette.py`, `__init__.py` registry + `PlotAccessor`, and the `PlottableMixin`
   edit to the facade. *(foundation; nothing renders without this.)*
2. **The four core figures on mpl** — `figures/weights|forest|decay|heatmap.py`. This is the
   **Phase-C pre-ship sprint** deliverable (`docs/EXPERIENCE-PLAN.md` § 4, Phase C). Ship with the
   deterministic palette from step 1.
3. **Publication last-mile** — `export.py` (journal presets, font embed, cb-check), `panel.py`,
   `provenance.py` `caption()`. Makes step 2 submission-ready.
4. **`plotnine` backend** — `backends/_plotnine.py` (ggplot parity of the four core figures). The
   ex-R zero-migration path (`viz-ggplot` extra).
5. **The qpGraph DAG** — `backends/_graphviz.py` + `figures/graph.py`. Independent of the
   spec/backend generic path.
6. **MDS embedding** — `embed.py` (classical MDS) + `figures/embed.py`. Pure numpy/scipy.
7. **Terminal tier** — `backends/_terminal.py` (plotext + rich). The SSH sketch.
8. **Interactive tier** — `backends/_plotly.py`, `backends/_altair.py` (row-guarded),
   `figures/rotation.py`. Feeds Area 3/6.
9. **Live redraw primitives** — the anywidget `plot.live.*`. Rides the Area-6 daemon; sequenced
   last because it depends on the linchpin.

Steps 1–3 are the parity-safe, no-daemon, ship-with-0.1.0 core (the plan's Phase C). Steps 4–8 are
steady-value fill-in. Step 9 is a big-bet that waits on Area 6.

---

## 10. Dream visualizations — the speed / scale / time tier

The seven kinds cover the vocabulary; steppe's three differentiating properties each unlock a
*second* tier no batch-R tool can render — **SPEED** (a fit is ~0.9 s cold, sub-200 ms resident via
the Area-6 daemon), **SCALE** (2.5 B quartets in ~177 s at 14.5 M q/s), **TIME** (the
radiocarbon-dated AADR panel). Organized below by the unlock. Every figure is
presentation-over-existing-data or a thin new builder over an existing kind — none re-derives
compute, per the area's founding constraint.

### 10.1 SPEED → the figure becomes an instrument

SNP-block resampling is embarrassingly parallel on-device (cuRAND Philox per-replicate substreams),
so the replicate cloud is free in the same sub-second budget.

| Figure | What it is | Unlocked by | Demo / wow |
|---|---|---|---|
| **Live qpAdm Studio** | target box + draggable source/right chips; a live weight bar + p-gauge + feasibility light re-fits in place on every drag; "pin to figure" snapshots through the *same* static `plot.weights` builder | resident-f2 daemon → <200 ms re-fit; spec/render split → live and paper views share one builder | "what if you swap in Yamnaya?" answered live, on a conference projector — the write→`source()`→wait→parse loop collapsed to a slider drag |
| **Breathing bootstrap bar** | the weight bar re-fit with the whole replicate cloud as a translucent overlay (or an animation cycling replicate heights) so a fragile model visibly *shakes* while a robust one sits still; feasibility flips color live | cheap on-device resampling → the `[n_rep×n_source]` draw matrix, not a scalar | the honest antidote to speed-enabled p-hacking — you *feel* a weight pinned at 0.5 that jitters 0.1–0.9. **Highest value / lowest cost**: needs only a facade tweak exposing the draw matrix (today reduced to `d["se"]`), and is independent of the daemon |
| **Live rotation leaderboard** | the whole rotation field (hundreds–thousands of models) as a ranked strip; drag the p-threshold or edit the outgroup pool and the board re-ranks live, feasible glowing, just-rejected fading | C(18,5)=8568 rotation in ~24 s → the whole competitive field at once; daemon → live re-rank | right-set selection (qpAdm's dark art) becomes direct manipulation instead of an Excel sort |
| **Drag-to-edit graph builder** | Figma-style node canvas over `QpGraphResult`; draw an edge / drop an admixture node → the graph re-scores on every edit, worst-residual edge recoloring live; "suggest next edge" previews the bounded topology search as ghost edges | ~1 s re-score → "what if I add an edge here?" instant | watching a bad edge flush the graph red the instant you draw it — turns qpGraph's most artisanal task into design |
| **Relaxing MDS cloud** | classical MDS that physically relaxes into its new eigen-layout when you drop/add a pop, WebGL-animated (a coordinate tween between old and new layouts) | MDS is a closed-form eigendecomposition of the small P×P mean-f2 matrix — no genotypes, no re-read | pull the one weird sample out and watch the structure it was distorting snap into focus |
| **Brushable residual heatmap** | the residual-z heatmap cross-filtered with the weight bar + leaderboard; brush a hot cell → highlights the culprit pop and (optionally) re-fits without it so the misfit resolves in one gesture | the counterfactual drop-and-refit is only possible at <200 ms | "the model doesn't fit" → "*this* pop is why, and here's the fit without it" |

### 10.2 SCALE → dense fields from what are sparse tables

| Figure | What it is | Unlocked by | Demo / wow |
|---|---|---|---|
| **Sweep Galaxy / affinity firehose** | every C(P,4) f4/D quartet as a datashaded density field (\|Z\| vs effect, or a projected quartet space); bright ridges are clustered signal; zoom re-aggregates, lasso a ridge → survivor quartets stream into a linked WebGL top-K table | 2.5 B quartets at 14.5 M q/s — a firehose with **no exploration surface today** (altair dies at 5k, mpl scatter far earlier) | "a billion statistics at once," pan-zoom-re-aggregate like Google Maps over the space of all quartets — open-question 6's concrete answer (§ 11) |
| **Population affinity network** | force-directed graph, nodes = AADR pops, edges = strongest outgroup-f3 / attracted-f4; click a node → the sweep re-centers on it, edges re-weighting live | the whole quartet space is a sub-second on-device sweep; recentering is a fresh batch, trivial | "browse populations like Spotify fans-also-like" — answers "who is a good source / outgroup for this pop?" |
| **Full-panel f2 manifold** | every pop embedded from resident f2 (classical MDS, opt-in UMAP/t-SNE), a WebGL point cloud with pan/zoom, hover-to-label, lasso-to-subset-into-a-fit | f2 already resident in VRAM → re-embedding after dropping outliers is nearly free | the whole aDNA world as one navigable star-chart instead of a 40-pop static blob |

### 10.3 TIME → the dated panel as a first-class axis

The AADR panel is radiocarbon-dated (`Date_mean_in_BP` in the `.anno`); no other qpAdm tool treats
age as a plotting axis. Needs the date + lat/long columns joined from the Area-4 catalog (the
`.anno` gap).

| Figure | What it is | Unlocked by | Demo / wow |
|---|---|---|---|
| **Ancestry-through-time river** | streamgraph where x = calibrated age and each band is a qpAdm source weight per rolling temporal bin, with a bootstrap ribbon on each band's edge | a fit per time-slice is sub-second → regenerate the whole stream when you swap a source | the figure the field describes in prose ("steppe rose from 0 % to 62 %") but never draws — the blue band widening from the Yamnaya horizon |
| **Spatiotemporal time-map (AADR Atlas)** | deck.gl geo + temporal map; a millennium slider scrubs dated samples in/out; lasso a region+window and *that* becomes the rotation source pool — the fit runs live and annotates the map. The map is simultaneously query and answer | speed + scale + time | "ancient-DNA Google Earth." Prior art (spread.gl, ARGscape put time on a deck.gl axis) proves the render; nobody wired it to a sub-second rotation |
| **Selection-through-time scrubber** | derived-allele-frequency trajectories across dated samples (LCT/rs4988235, SLC45A2) with per-bin CIs — a live, playable Mathieson-2015 | the dated panel + on-device genotype reduction → per-millennium frequency instant; jackknife CIs cheap | the field's most iconic result (rise of lactase persistence) turned from a static panel into something you scrub and rewind |
| **Figure-diff for re-analysis** | overlay a new-AADR result on the ghosted old one across any kind (weights/forest/decay) so the v66→next-release delta *moves in place* | re-fitting the old model on new data is a second, not an afternoon | a quiet reproducibility superpower — the change moves, not a re-read of two PDFs |

### 10.4 Further novel directions (the completeness critic)

Two rate highest — novel *and* near-term:

- **Joint weight-simplex confidence blob** — plot the full cloud of bootstrap replicate weight
  vectors *inside* the source simplex with a density contour, exposing the **joint** geometry of
  uncertainty (source A and B trading off along a ridge, the cloud poking across the zero-weight
  infeasibility face) that per-weight ±SE bars structurally cannot show. It fixes a real, widespread
  misrepresentation — independent error bars on compositional weights that are anti-correlated.
  Cheap because GPU bootstrap fills the simplex densely; needs only `mpltern`/`python-ternary` + a
  datashader 2-D KDE. *(SPEED)*
- **The re-runnable figure** — a clickable computational-DAG inset (`dataset → f2 blocks → fit →
  weights`, each node carrying its exact params + hash, click to re-emit an intermediate or re-run
  from there) beside the panel, so the figure literally contains a re-executable recipe of itself.
  Extends the Area-2 `caption()` / Area-5 `run.json` plumbing into a clickable graph rather than
  static file metadata — a reproducibility/credibility win for a parity-validated tool.

The rest, grouped by the property they exploit (SPEED → instruments, SCALE → dense fields, TIME →
spatiotemporal):

- **Force-directed / physics admixture graph** — drift edges as springs (rest-length ∝ branch length), drag a pop while the worst-residual edge glows and the layout relaxes; scales to the thousand-pop affinity web with hierarchical edge-bundling → cosmograph / regl GPU d3-force in an anywidget. *(SPEED/SCALE)*
- **Rotation feasibility manifold** — the model-*space* map: every point a candidate source set, color = p, so you *see* the contiguous feasible island and the boundary where a source flips negative, not a thousand-row table → barycentric/UMAP embedding + datashader density. *(SCALE)*
- **Continuous spatial-ancestry surface with kriging uncertainty** — interpolate a per-sample ancestry component into a geographic field whose **opacity is the kriging variance**, so it fades honestly to "unknown" where samples are sparse → verde/PyKrige + deck.gl. *(SCALE+TIME)*
- **Edge-bundled migration-flow map** — bundled directional admixture arcs on geography, width = weight, color = epoch; a temporal scrub reveals corridors switching on/off → deck.gl ArcLayer. *(SCALE+TIME)*
- **Model-vs-model / version-vs-version diff** — a slopegraph/dumbbell of weights + a paired residual "fit-fingerprint" heatmap showing *where* one model fails and the other rescues (the *comparative* residual view; § 10.1's is the single-model one). *(SPEED)*
- **Data-real explanatory animation** — animate the *actual* (a−b)(c−d) product accumulating SNP-by-SNP and the block-jackknife visibly dropping each block (the wobble *is* the SE), off the device-resident per-block terms → matplotlib `FuncAnimation` / manim; a teaching mode that reads true numbers, not a cartoon. *(SPEED)*
- **Accessibility beyond color** — every figure emits an ARIA/SVG `<title>+<desc>` + an equivalent tidy data table + an auto-verbal caption ("target X is 62 %±4 source A, 38 %±4 source B; p=0.3"), plus redundant hatch/shape encoding for grayscale (past the existing Okabe-Ito CVD pass); optional **sonification** of decay/temporal curves (astronify / maidr) where the ear catches rate changes a flat curve hides.
- **Speculative tail** — collaborative/multiplayer live-fit sessions (one daemon, shared cursors, Yjs CRDT into an anywidget) and a WebXR embedding cloud with a scrubbable time axis.

**Feasibility (honest).** The § 10.1 live figures are HIGH-value but gated on the Area-6 daemon
(and, for residual brushing, a per-quartet residual accessor) — *except* the breathing-bootstrap
bar, which is a facade tweak that ships now. The SCALE tier is gated on the sweep Python facade
(open-question 6) plus the datashader path (§ 11); the TIME tier on the Area-4 date/lat-long columns
plus a per-timebin frequency reducer. Honest caveat: none of these *fix* qpAdm's inferential limits;
they make honest uncertainty and sensitivity cheap to **show**, which is the credible claim to the
pro audience.

---

## 11. Scale & real-time rendering architecture

The engineering that resolves open-question 6 (§ 12): the billions-scale sweep and the live tier
need paths the five static backends cannot serve. The key recognition — most of this is **not a
sixth chart backend**; it is new *paths* that feed the backends and the Area-6 daemon steppe already
has, registered through the same entry-point/capability mechanism (§ 3.3) so the dispatcher raises
the existing `CapabilityError`, never a broken Vega spec.

### 11.1 Rendering millions–billions of points — server-side rasterization

**datashader is a pre-aggregation RENDERING PATH upstream of `PlotSpec`, not a new backend.** A new
`figures/sweep.py` projects + aggregates every quartet into a fixed H×W density grid (count / max /
any per pixel) *once, regardless of N*, and emits a new `PlotSpec` kind (`RASTER`/`DENSITY`) that
the **existing** backends already draw — mpl `imshow`, plotly `go.Image`, or a served PNG. The
browser only ever receives a canvas-sized image, never the billion points; this sails past altair's
~5k Vega-Lite cap and mpl's scatter wall (the exact gaps § 3.1 flags), and it is § 6.3's row-guard
escape hatch ("pre-aggregate before it reaches altair") made concrete.

**Pre-aggregate-on-GPU vs stream — the ~1M-row threshold.** datashader accepts a **cuDF GPU
DataFrame** as a drop-in for pandas, so the sweep survivors rasterize *on the same CUDA device that
computed them* — zero host round-trip, the "aggregate on the box, ship only the reduction"
principle. Below ~1M points (a filtered sweep, a rotation landscape) you can instead **stream** the
reduced rows to the client GPU and let WebGL (deck.gl / regl — 60 fps to ~1M, hover/picking for
free) draw them; above ~1M — and certainly at the 2.5 B firehose — you **must** rasterize (Chrome
caps one allocation at 1 GB; deck.gl layers crash 10 M–100 M). That ~1M-row line is the
pre-agg-vs-stream boundary. hexbin/density + LOD tiling (re-rasterize on zoom via HoloViews
`rasterize()`/`datashade()` over a Bokeh callback) give slippy-map zoom with no bespoke front end —
a Python-only interactive-sweep path, gated heavy because Bokeh/Panel is effectively a second server
beside the FastAPI daemon (so it is the fallback, not the flagship).

### 11.2 The GPU→browser data path

Principle: **keep f2 resident in VRAM and ship only reduced results.** Two wire regimes:

- **≤~1M rows (interactive tier)** — the daemon serializes the reduced result as **Apache Arrow
  IPC** record batches; `arrow-js` decodes them zero-copy in the browser and hands the typed buffers
  straight to deck.gl's binary-attribute path. Arrow is columnar end-to-end, needs no
  deserialization on receipt, streams as batches, and benchmarks ~20–30× over ODBC — the honest
  answer to "ship only the reduction."
- **>~1M (firehose)** — ship the datashader **raster** (a small image) + its extents, not the
  points; a zoom sends new extents and the box re-rasterizes.

**Snapshot vs streaming**: static publication figures are a one-shot reduce→render; the live tier
streams reduced frames (weight vectors, replicate clouds) over a websocket (msgpack via `msgspec`,
or Arrow; control messages JSON). This transport is **Area-6's** — the deck.gl + Arrow front end
lives in `web/`, not `steppe.plot`. `steppe.plot`'s responsibility ends at "a reduced DataFrame or a
raster."

### 11.3 Real-time / reactive

**anywidget** is the substrate for every live figure — the `plot.live.*` primitive already scoped in
§ 8 and build-order step 9 (§ 9). A small traitlets model carries only the tidy frame (weights, se,
replicate draws, p) across the wire; the JS `render({model, el})` subscribes to
`model.on('change:…')` and mutates the SVG/canvas in place — the
`plot.live.weights(handle).update(result)` primitive. One widget spans Jupyter + marimo + Panel and,
for the sweep, carries the server-rendered raster + a re-rasterize-on-zoom callback, so a single
component spans the small-live and billion-scale cases. The **Area-6 daemon** is the hard dependency
for the sub-200 ms loop: a resident-f2 session + a **latest-wins coalescing queue** (a slider
dragged across 40 values dispatches *one* fit — the coalescing *is* the debounce).

**anywidget vs plotly `FigureWidget`** (the open-question-5 tradeoff): `FigureWidget` is
zero-JS-build — hold it, mutate traces inside `fig.batch_update()`, and Plotly.react diffs/repaints
only the changed marks — the pragmatic Jupyter default; anywidget adds a JS build step but is the
portable standard (Jupyter/marimo/Panel) and the right home for a reusable primitive both Area 3 and
Area 6 consume. **FigureWidget for the notebook, anywidget for the shared component.**

### 11.4 3D + temporal animation

A new kind (`EMBED3D`) + an animation wrapper on the **existing plotly backend**, not a new stack:
plotly `Scatter3d` is the rotatable 3-component MDS scene (zero-JS-build, self-contained-HTML export
— the natural artifact for a talk/supplement), `plotly-resampler` downsamples large series, and it
holds ~100k WebGL points before the datashader raster takes over. Temporal scrubbing is plotly
`animation_frame` (an interactive slider bound to `Date_mean_in_BP` bins) for notebooks and
`matplotlib.animation` (`FuncAnimation`) for the MP4/GIF export to a supplement — standard,
dependency-light ways to turn "a fit per time-slice" into a scrubbable animation and a citable video.

### 11.5 Packaging + the dream-figure → path map

The heavy deps go behind a **new `viz-scale` extra** (added to § 7), parallel to the existing
`viz-interactive`/`viz-terminal` split; the base wheel stays `numpy`-only, unchanged. cuxfilter /
RAPIDS crossfilter and the deck.gl + Arrow React front end are Studio-tier / Area-6, not base
extras.

| Dream figure (§ 10) | Rendering path | Extra / dep |
|---|---|---|
| Sweep Galaxy / firehose | datashader `RASTER` kind → mpl/plotly draws it; HoloViews re-agg on zoom | `viz-scale` (+ `viz-scale-gpu` for on-VRAM) |
| Survivor top-K constellation | WebGL scatter (deck.gl / jscatter), ≤1M rows over Arrow | `viz-scale` + Area-6 `web/` |
| Full-panel f2 manifold | plotly scattergl (≤100k) → datashader raster above | `viz-interactive` / `viz-scale` |
| AADR Atlas time-map (`MAP`) | pydeck / deck.gl geo + temporal; lasso → daemon | `viz-scale` + Area-4 dates + Area-6 daemon |
| Ancestry river (`RIVER`) | new `RIVER` kind, mpl `stackplot` + plotly; existing palette | `viz` (+ Area-4 dates) |
| Selection-through-time | plotly `animation_frame` / mpl `FuncAnimation` | `viz-interactive` + a timebin-frequency reducer |
| Live Studio / breathing bar / leaderboard | anywidget or plotly `FigureWidget`, `update(result)` | `viz-interactive` + Area-6 daemon |
| Drag-to-edit graph | React Flow node editor over `QpGraphResult` (static stays graphviz) | Area-6 `web/` |
| 3D MDS + time axis (`EMBED3D`) | plotly `Scatter3d` + `animation_frame`, self-contained HTML | `viz-interactive` |
| Figure-diff | PlotSpec **diff render mode** (two frames, one ghosted) — every static backend inherits it | `viz` (no new dep) |

**Net:** four new `PlotSpec` kinds (`RASTER`/`DENSITY`, `MAP`, `RIVER`, `EMBED3D`), one new builder
(`sweep`) plus the river/embed3d builders, a plotly-backend extension for 3D/time, a
`datashader`/`pydeck` backend pair behind `viz-scale`, and the heavy WebGL/Arrow work pushed into
Area 6 — the spec/render split and the § 3.3 entry-point/capability mechanism hold verbatim.

---

## 12. Open questions

1. **C++ CLI `--plot` (Area 1 boundary).** The plan wires the terminal tier as `steppe qpadm …
   --plot`, but the CLI is a pure-C++ binary that cannot import the Python `_terminal` backend
   in-process (and the wheel ships no `steppe` command — `STEPPE_BUILD_CLI=OFF`). Options: (a) a
   Python `console_scripts` CLI (`steppe.__main__`) that wraps the facade *and* `steppe.plot` and
   owns `--plot`; (b) the C++ CLI shells out to a small `steppe-plot` Python helper; (c) `--plot`
   lives only on the Python surface. This is fundamentally **Area 1's CLI-in-wheel decision** —
   Area 2 supplies the renderer either way.
2. **The live-widget owner (Area 2 vs 3).** Should the `steppe.explore.qpadm(...)` widget live in
   `steppe.plot.live` (Area 2, rendering-centric) or `steppe.explore` (Area 3, app-centric)?
   Recommendation: the **anywidget component + `update(result)` primitive** in Area 2; the
   **orchestration** (pickers, sliders, debounce, daemon calls) in Area 3. Needs a joint sign-off.
3. **plotnine in `viz` vs `viz-ggplot`.** Splitting keeps `viz` py3.9-installable but means the
   ex-R default (`backend="plotnine"`) is a *second* `pip install`. Alternative: bump the whole
   `viz` extra (and its docs) to py≥3.10. Leaning **split** to honor `requires-python>=3.9`, but if
   analytics show ~all users are ≥3.10 the simpler single extra wins.
4. **Palette determinism scope.** `color_of(pop)` must be stable *across processes and papers*
   (hash → Okabe-Ito-extended ramp). Open: do we also want a *project-pinned* override
   (`theme.pins.toml`) so a lab can freeze "Yamnaya = #0072B2" regardless of the hash? Ties into
   Area 5's project dir.
5. **anywidget vs a plotly `FigureWidget` for the live tier.** anywidget is the portable standard
   (Jupyter+marimo+Panel) but adds a JS build; a plotly `FigureWidget` is zero-JS but Jupyter-only
   ergonomics. Decide when step 9 starts, informed by Area 3's marimo choice.
6. **Sweep-scale rendering — proposed, not unsolved.** § 11 gives the 2.5 B-quartet sweep a concrete
   path: datashader server-side rasterization to a fixed-size image (a `RASTER`/`DENSITY` `PlotSpec`
   kind, optionally cuDF-on-VRAM), deck.gl for the ≤~1M interactive tier, Arrow IPC for the
   GPU→browser transport, all behind a new `viz-scale` extra. Remaining unknowns: (a) the sweep
   still has **no Python facade** — it is CLI-only (an Area-3 gap), and the raster path is
   upstream-blocked on it; (b) the **pre-aggregate-vs-stream threshold** (~1M rows is the working
   guess) needs measuring on real sweep output; (c) whether the interactive re-aggregation rides the
   **FastAPI daemon** or a **second Bokeh/Panel server** (the heavy-dep fork in § 11.1); (d) whether
   the **AADR Atlas / time-map** is Area-2 territory (a `MAP` kind + pydeck backend) or
   Area-6/Studio territory (it needs the daemon, the Area-4 date columns, *and* a `web/` front end) —
   leaning **Area-6 owns the app, Area-2 owns the `MAP` spec + static render**.
7. **The `RASTER` kind's home, and the "no new compute" line.** datashader is modeled as a pre-agg
   *path* emitting a `RASTER` spec the existing backends draw, not a sixth backend — but on-VRAM
   aggregation (cuDF) is a compute step with a real CUDA dependency living inside a layer whose
   founding constraint is "presentation-over-existing-data, no new compute." Is `figures/sweep.py` a
   legitimate reduce-for-display (like the heatmap's block-mean) or does it cross that line? Leaning
   legitimate — but `viz-scale`/`viz-scale-gpu` is the first plotting extra that is CUDA-coupled,
   which deserves an explicit call in the packaging story.
8. **Exposing the replicate cloud (blocks the cheapest honest-uncertainty win).** The
   breathing-bootstrap bar (§ 10.1) and the joint weight-simplex blob (§ 10.4) need the
   `[n_rep×n_source]` draw matrix, which exists on-device (`JackknifeMode{Delete1,Bootstrap}`, Philox
   substreams) but is reduced to the scalar `d["se"]` before it reaches Python. Surfacing it is a
   small, high-value facade extension — but it is a **facade/bindings decision (Area 1)**, not
   Area 2's, and needs sign-off on the accessor shape (a `.replicates` DataFrame? a lazy accessor
   mirroring `_require_pandas`?).

---

*Sources for the tech research:*
[plotnine (PyPI / plotnine.org)](https://plotnine.org/),
[matplotlib publication figure guidelines](https://plotivy.app/blog/nature-journal-figure-guidelines-2025),
[Altair vs Plotly vs plotnine comparison](https://towardsdatascience.com/4-key-players-in-python-data-visualization-ecosystem-matplotlib-seaborn-altair-and-plotly-23ae37a68227/),
[plotext terminal plotting](https://github.com/piccolomo/plotext),
[graphviz vs pygraphviz](https://graphviz.readthedocs.io/en/stable/manual.html),
[anywidget / marimo](https://anywidget.dev/en/jupyter-widgets-the-good-parts/),
[pandas plotting-backend entry-point pattern](https://pandas.pydata.org/docs/dev/development/extending.html),
[scikit-learn ClassicalMDS / PCoA](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.ClassicalMDS.html),
[datashader — billion-point server-side rasterization](https://datashader.org/),
[deck.gl / pydeck — WebGL geospatial + temporal layers](https://deck.gl/),
[Apache Arrow / Arrow Flight — zero-copy columnar transport](https://arrow.apache.org/),
[HoloViews — interactive datashader (rasterize-on-zoom)](https://holoviews.org/),
[Observable Plot — fast declarative re-render past Vega-Lite's row wall](https://observablehq.com/plot/).
</content>
</invoke>

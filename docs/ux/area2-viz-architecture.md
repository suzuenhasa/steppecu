# Area 2 — Visualization & Publication Figures: `steppe.plot`

**Engineering-architecture doc.** This is the *how* (folder scaffolding, tech-stack decisions,
interfaces to existing code, packaging) for the visualization layer. The *what/why* — the
feature list, the delight rationale, the pre-ship sprint sequencing — lives in the product plan,
`docs/EXPERIENCE-PLAN.md` § "Area 2 — Visualization & Publication Figures". The two are
complementary: read the plan for the features, read this for the module boundaries and the deps.
This doc does **not** restate the feature table.

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
  renderer in-process — see § 8 open questions).

---

## 2. Tech-stack decision

The design principle from the plan and the packaging memo: **the base wheel stays `numpy`-only**;
every plotting dependency is an optional extra, imported lazily (the exact pattern the facade
already uses for `pandas` via `_require_pandas`). steppe is GPU-only and server-side, so there is
**no WASM/pyodide constraint on the plotting libs** — they run on the same CUDA box as the fit.

### 2.1 The rendering libraries (weighed)

| Backend | Library | Chosen role | Why over the alternatives |
|---|---|---|---|
| **`mpl`** *(default)* | **matplotlib ≥3.7** | The universal default for `.plot()`; publication raster/vector export | The one library that (a) is publication-grade (EPS/PDF vector, font embedding via `pdf.fonttype=42`), (b) runs on **py3.9** (steppe's floor), and (c) is the substrate every other static lib sits on (plotnine renders *through* mpl). Zero-friction default: if a user has `steppe[viz]`, `.plot()` just works. |
| **`plotnine`** | **plotnine ≥0.13** | The ggplot2-grammar backend for the ex-R / AT2-in-R audience | The audience migrating from ADMIXTOOLS 2 lives in `ggplot2`; plotnine is a faithful grammar-of-graphics port (actively maintained — 0.15.x as of mid-2026, MIT) so their `+ geom_* + theme_*` muscle memory transfers with **zero migration cost**. Caveat: plotnine requires **py≥3.10**, so it is a *separate* extra (`viz-ggplot`), not folded into `viz` (which must stay py3.9-installable). This is the whole reason mpl, not plotnine, is the default. |
| **`plotly`** | **plotly ≥5.20** | Hoverable/interactive figures for notebooks and Studio | For "hover a quartet to read its exact Z", 3D, and web embedding, plotly is the mature interactive choice and integrates with marimo/anywidget. Chosen over Bokeh for the richer hover/HTML-export story and the larger popgen-notebook install base. |
| **`altair`** | **altair ≥5** | Declarative Vega-Lite specs for the reactive residual/heatmap panels (Area 3's chosen interactive grammar) | Declarative, reproducible, and the grammar Area 3's marimo explorer already targets (residual-z heatmap, brushable `|Z|` histogram). **Hard limit noted**: Vega-Lite errors past ~5000 rows — so altair is fine for weights/forest/rotation-of-hundreds and the residual heatmap, but the billions-scale sweep must be pre-aggregated *before* it reaches altair (row-guard in the backend, § 5.3). Chosen alongside (not instead of) plotly because it is the natural fit for the marimo tier. |
| **`terminal`** | **plotext ≥5.2 + rich ≥13** | ASCII/braille plots over SSH; the headless-HPC tier | plotext is the current, actively-maintained terminal-plot lib (braille 4×2 markers for the DATES decay, bar/scatter/hist for weights/forest, native `rich` integration); rich draws the colored blocks/tables. Chosen over `termplotlib` (which merely shells to gnuplot — an extra system dep) and over hand-rolled ANSI. This audience lives in SSH sessions, so a `res.plot(backend="terminal")` sketch-where-you-ran-it is a credibility signal. |
| **`graphviz`** *(DAG only)* | **graphviz ≥0.20** (pure-Python DOT interface) | The qpGraph admixture-graph layout | Auto-layout of a DAG is `dot`'s job. The pure-Python **`graphviz`** package (emits DOT, shells to the system `dot` binary) is chosen over **`pygraphviz`** (which compiles against libgvc — a heavy build-time dep and a common install failure) because it is a tiny pure-python wheel with richer edge/node styling and string-keyed nodes that map 1:1 to pop labels. The only cost is a **system `dot` requirement** (documented like the CUDA runtime — see § 6). |

### 2.2 The embedding math (`plot.embed`)

Classical MDS / PCoA on the f2 matrix is an **eigendecomposition of the double-centered f2
matrix** — f2 is already a squared-distance-like quantity, so classical MDS ≈ PCA of the
populations with no genotypes needed. Decision: **implement classical MDS in-house with
`numpy`/`scipy.linalg.eigh`** (`embed.py`, ~30 lines) rather than pull `scikit-learn`. Rationale:
scipy is already a light, common dep (and needed for the heatmap's hierarchical clustering,
`scipy.cluster.hierarchy`); classical MDS is a 4-step closed form; and avoiding sklearn keeps
`viz` lean. `scikit-learn` (metric MDS, t-SNE, UMAP) stays an *opt-in* the user installs if they
ask for `method="tsne"` — the backend raises a clear "install scikit-learn" then.

### 2.3 The backend-dispatch design (the portfolio-grade core)

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

## 3. Fit decision — **hybrid** (new subpackage + a minimal facade extension)

**New subpackage** `bindings/steppe/plot/` is the home for essentially all of the code — it is a
pure-Python presentation layer that consumes the facade's DataFrames, so it does not belong in the
compiled `_core` or the C++ CLI. It ships *inside* the existing wheel package (scikit-build-core's
`wheel.packages = ["bindings/steppe"]` already copies the whole subtree), so **no packaging change
is needed to include the code** — only the new optional extras (§ 6) gate the *dependencies*.

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

## 4. Folder scaffolding

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

## 5. Key interfaces

### 5.1 Public API (`steppe/plot/__init__.py`)

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

### 5.2 The spec/backend contract (`spec.py`, `backends/base.py`)

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

### 5.3 Notable per-figure semantics (why the spec layer earns its keep)

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

### 5.4 Export & provenance

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

## 6. Packaging — the extras

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
# ── convenience aggregate (grows to include app/mcp when those areas land) ──
all = ["steppe[viz,viz-ggplot,viz-interactive,viz-terminal]"]
```

- **`graphviz` needs the system `dot` binary** (like the CUDA-13 runtime, it is a documented
  system requirement, not a pip dep). `graph()` raises a clear "graphviz `dot` not found; install
  the Graphviz system package" if `dot` is absent.
- **Lazy-import contract** (enforced by a test): `import steppe` and `import steppe.plot` must
  succeed with only `numpy` installed; the missing-dep error is raised at *first `.plot()` call*,
  naming the exact extra (`"install steppe[viz]"`), exactly as `_require_pandas` does today.
- No `[tool.scikit-build]` change: `wheel.packages = ["bindings/steppe"]` already includes the new
  `plot/` subtree; the CLI stays `STEPPE_BUILD_CLI=OFF`.

---

## 7. Shared-infra dependencies (referenced, not re-scaffolded)

| Shared infra | Owner | How Area 2 uses it |
|---|---|---|
| **(vi) result classes / DataFrames** | facade (exists) | **Primary input.** Every figure reads a `.weights`/`.table`/`.curve`/`.edges`/`to_numpy()` DF. Area 2 adds the `PlottableMixin` `.plot` accessor onto them. |
| **(v) provenance manifest `run.json`** | **Area 5** | `provenance.py`/`caption()` read the manifest schema for version/params/data-hash and embed it in the figure file; minimal self-contained stamp until Area 5 lands. |
| **(ii) AADR pops catalog `steppe.pops`** | **Area 4** | *Optional* enrichment: `embed(color_by="region")` / `heatmap` can color/annotate by region/period if `steppe.pops` is importable. Soft dep, no hard requirement. |
| **(iv) resident-f2 daemon `steppe serve`** | **Area 6** | Powers the *live* variants: live re-embed (drop a pop → cloud relaxes), live rotation p-threshold. Area 2 provides the redraw primitive; the sub-200 ms re-fit is the daemon's. Static figures need nothing from it. |
| **(iii) content-addressed f2 cache** | **Area 5** | Under the live variants (via the daemon). Not touched by static export. |
| **(i) CLI output seam `emit_to_destination`** | **Area 1** | The `steppe qpadm … --plot` terminal render is an Area-1 CLI decision that calls Area 2's `_terminal` backend — but the C++ CLI cannot import Python in-process (see § 8). |

**The live qpAdm explorer widget** (plan lists it under Area 2, "killer interactive"): the
*rendering* primitive (a `weights` figure that re-renders in place, ideally via **anywidget** so it
works in both Jupyter and marimo) belongs here; the *interaction loop* (searchable pop pickers,
sliders, the re-fit trigger) is **Area 3** (marimo) riding the **Area 6** daemon. Area 2's
contribution is a `plot.live.weights(handle)` that returns an anywidget whose `update(result)`
swaps the bar without a full re-layout. Flagged as a boundary (§ 8).

---

## 8. Intra-area build order

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

## 9. Open questions

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
6. **Sweep-scale rendering.** The 2.5B-quartet sweep has no Python facade yet (an Area-3 gap) and
   would blow past altair's 5k cap and matplotlib's practical scatter limit. The forest/rotation
   figures assume hundreds–thousands of rows; a datashader-style aggregation path for the sweep is
   out of scope until the sweep facade exists.

---

*Sources for the tech research:*
[plotnine (PyPI / plotnine.org)](https://plotnine.org/),
[matplotlib publication figure guidelines](https://plotivy.app/blog/nature-journal-figure-guidelines-2025),
[Altair vs Plotly vs plotnine comparison](https://towardsdatascience.com/4-key-players-in-python-data-visualization-ecosystem-matplotlib-seaborn-altair-and-plotly-23ae37a68227/),
[plotext terminal plotting](https://github.com/piccolomo/plotext),
[graphviz vs pygraphviz](https://graphviz.readthedocs.io/en/stable/manual.html),
[anywidget / marimo](https://anywidget.dev/en/jupyter-widgets-the-good-parts/),
[pandas plotting-backend entry-point pattern](https://pandas.pydata.org/docs/dev/development/extending.html),
[scikit-learn ClassicalMDS / PCoA](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.ClassicalMDS.html).
</content>
</invoke>

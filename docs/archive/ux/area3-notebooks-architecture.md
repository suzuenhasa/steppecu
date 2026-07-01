# Area 3 — Notebooks & Interactive Apps (marimo-first): Engineering Architecture

> **Scope of this document.** `docs/EXPERIENCE-PLAN.md` is the **product plan** — *what* Area 3
> ships and *why* (the flagship `steppe explore` explorer, the AADR picker widget, the
> reproduced-study notebooks, `steppe app` deploy, the sweep explorer). *This* document is the
> **engineering architecture** — *how*: the tech-stack decisions, the concrete folder tree and
> file names, the key interface signatures, how they plug into the existing facade, the packaging
> extras, and the shared-infra dependencies. It references the plan's feature tables; it does not
> restate them.

---

## 1. Area role (from the plan)

Area 3 turns steppe's sub-second fit into a **reactive modeling loop**: open a notebook, type a
target into a searchable AADR picker, drag a slider over the right-set, and watch the qpAdm weight
bar / p-gauge / residual-z heatmap redraw live — the loop AT2's offline R/Shiny structure cannot
close (EXPERIENCE-PLAN.md §Area 3). Every exploration is *also* a git-diffable, deployable,
citable artifact.

The area owns four deliverables from the plan:

1. **`steppe explore`** — the flagship reactive qpAdm explorer (searchable target, left/right
   multiselects, rank/fudge/als sliders → live weight bar + p-gauge + Altair residual heatmap).
2. **Reproduced-study notebooks** (Haak 2015, Olalde 2018/19) — runnable, assert-the-numbers,
   then drop into the seeded explorer; the "does it reproduce the canon?" onboarding.
3. **AADR population picker widget** + **rotation leaderboard** — reusable reactive components
   over the facade.
4. **`steppe app`** — one-command `marimo run` deploy (code hidden, one GPU box serves a lab) +
   a pinned CUDA Dockerfile.

**Hard constraint the whole area designs around (plan §Area 3 "Constraint").** steppe is
**GPU-only**: there is no WASM/pyodide path for real fits — the CUDA compute must run
**server-side** on a CUDA box. This bifurcates the delivery model (§4 below): *live* apps are
`marimo run` on a GPU host; *in-docs* interactivity is precomputed **replay**, never a real fit in
the browser.

---

## 2. Tech-stack decision

### 2.1 Notebook / app runtime — **marimo** (not Jupyter)

**Decision: marimo is the substrate for every Area-3 deliverable.**

marimo is a reactive Python notebook stored as **pure `.py`** (an `app = marimo.App()` with
`@app.cell` functions). Three properties make it the correct — not merely trendy — choice here:

- **Reactive dataflow, not top-to-bottom cells.** Changing a UI element re-runs *only the
  dependent cells*. This is the exact execution model Area 3 needs: the f2 tensor loads **once**
  (an expensive cell), and only the sub-second fit cell re-runs when a picker/slider changes.
  Jupyter's linear model would either re-run everything or rely on brittle manual `if` gating.
- **`git`-diffable, deployable as an app for free.** `marimo run notebook.py` serves the same
  file as a read-only web app with code hidden — this *is* the `steppe app` feature, no second
  framework. Jupyter needs Voilà/Panel/nbconvert plumbing to reach the same place; the plan's
  "every exploration is also a reproducible, shareable artifact" is a native marimo property, not
  something we build.
- **Runs as a plain script.** `python notebook.py` executes the cells top-to-bottom — so the
  reproduced-study notebooks double as **CI parity canaries** (assert the published numbers) with
  zero notebook-execution harness (`nbmake`/`papermill`), mirroring the existing
  `examples/python/quickstart_qpadm.py` "living API canary" pattern.

**Alternatives weighed.**

| Option | Verdict | Why |
|---|---|---|
| **Jupyter + ipywidgets + Voilà** | Rejected | JSON `.ipynb` diffs badly; hidden execution-order state fights reactivity; deploy needs a separate stack (Voilà/Panel). The plan explicitly targets "AT2's Shiny GUI could never close [the loop]" — marimo's reactive+deploy story is the direct answer. |
| **Streamlit / Panel / Shiny-for-Python** | Rejected | App-only, not notebook-first — you lose the "exploration *is* the artifact" duality and the `python notebook.py` script/CI path. Full re-run-script model wastes the load-once/fit-many structure. |
| **Solara** | Rejected | Reactive and nice, but component-code-first (not notebook-first), smaller ecosystem, and no first-class `run-as-script` for the CI parity canaries. |
| **marimo** | **Chosen** | Reactive + pure-`.py` (git/CI) + `marimo run` deploy + `marimo edit --sandbox` (PEP-723 pinned reproduction) in one tool; standardized on **anywidget** for custom widgets (see §2.3). |

### 2.2 Reactive charts — **Altair/Vega-Lite** (primary), matplotlib fallback

**Decision: Altair is the chart layer for the *interactive* surfaces** (residual heatmap, the
rotation leaderboard, the sweep |Z| histogram), because marimo's `mo.ui.altair_chart` turns a
frontend brush/point selection into a **pandas DataFrame in Python automatically** — the reactive
primitive the sweep explorer and leaderboard drill-downs need. Altair also supports selections
across a much wider chart class than Plotly (whose marimo selections are limited to
scatter/bar/histogram/heatmap), which matters for the brushable-histogram → linked-table pattern.

Static, non-interactive figures inside a notebook (a one-shot weight bar for a "pin to figure"
snapshot) delegate to **Area 2's `steppe.plot`** (matplotlib/plotnine default) — Area 3 does not
re-implement publication figures; it renders *reactive* charts and calls Area 2 for the frozen
snapshot.

**Alternatives.** Plotly (kept as an optional per-user choice via `mo.ui.plotly`, but not the
default — narrower selection support, heavier JS payload); plotnine/matplotlib (the *static*
tier, owned by Area 2). This split — **Altair for live, Area-2's plot for pinned** — avoids
duplicating the figure layer while giving the reactive surfaces the selection primitive they need.

### 2.3 Custom widgets — **anywidget** (the AADR picker)

**Decision: the AADR population picker (and any bespoke widget) is built on anywidget.** marimo has
**standardized on anywidget** as its third-party widget API; anywidget widgets extend ipywidgets,
so the same picker works in marimo *and* Jupyter/Colab (the notebook half of the audience). This is
the "write the widget once, runs everywhere" property.

However — most of Area 3's "widgets" do **not** need a custom JS bundle. marimo's built-in
`mo.ui.*` (`dropdown`, `multiselect`, `slider`, `table`, `text` type-ahead) already cover the
target/left/right selects and the rotation leaderboard table. **anywidget is reserved for the one
element that needs a custom frontend**: the picker's fuzzy type-ahead + coverage/date/region
tooltip + the small map (plan's AADR picker row). We start with a pure-`mo.ui` picker (ships day
one) and upgrade *that one element* to anywidget when the map/tooltip is added — the interface
(`.value -> list[str]` of Group IDs) is identical, so nothing downstream changes.

### 2.4 State / caching — marimo `persistent_cache` + `mo.state`, and the daemon

The load-once/fit-many structure maps onto three marimo mechanisms:

- **`mo.persistent_cache` / `@mo.cache`** — caches the expensive `read_f2(...)` (and, later, an
  Area-5 content-addressed f2 dir) to disk/memory so the f2 tensor survives kernel restarts and is
  paid once. This is the notebook-local echo of Area 5's f2 cache.
- **`mo.state`** — holds the resident `F2Blocks` handle / the `FitClient` across reactive cells.
- **The Area-6 daemon (`steppe serve`)** — for a *deployed* app serving a lab, the f2 tensor lives
  **resident in VRAM in the daemon**, and each fit is the <200 ms compute hit (cold `read_f2` +
  upload is paid once, by the daemon, not per interaction). Area 3 talks to it through a thin
  client (§5.3) with a **`LocalClient` fallback** so notebooks run today, before the daemon lands.

**Debounce caveat (open question, §9).** marimo sliders re-run dependent cells on *every* tick.
For an expensive fit we gate with **`mo.ui.form` / `mo.ui.run_button`** (submit-on-confirm) or the
`mo.ui.slider` that fires on release; a native per-element debounce is not guaranteed across marimo
versions, so the design uses form-gating as the reliable path and treats "fire on mouse-up" as a
version-dependent nicety.

### 2.5 Deploy — `marimo run --headless --host --token-password` + a CUDA Dockerfile

`steppe app` is a **thin wrapper over `marimo run`** (edit mode via `marimo edit`), passing
`--headless` (no browser auto-open, for the box), `--host 0.0.0.0` (serve outside the container),
and marimo's built-in **token/password auth** (`--token-password`, on by default). A shipped
`Dockerfile` pins CUDA 13 + the GPU wheel (`steppe[app]`) + the notebooks, so one GPU box serves a
whole lab — the plan's "mirrors and out-classes AT2's Shiny GUI." No custom web server is needed;
marimo's server (Starlette/uvicorn under the hood) is the runtime, and Area 6's FastAPI daemon is a
*separate* process the app's `FitClient` connects to for resident-f2 speed.

---

## 3. Fit decision — **HYBRID: a new subpackage + an extended examples tree + packaging extras**

Grounded in the real code, Area 3 is **not** a CLI extension (that is Area 1's C++ seam) and not a
change to the facade result classes (those are frozen, and Area 3 *consumes* them). It is:

1. **A new Python subpackage `bindings/steppe/explore/`** — the reusable app-building machinery:
   the reactive-component factories (`widgets.py`), the fit client with daemon + local fallback
   (`client.py`), the shipped marimo notebooks (`notebooks/`), and the `steppe app` launcher
   (`_launch.py`). It ships inside the existing `wheel.packages = ["bindings/steppe"]` (pure
   Python), and every heavy import (`marimo`, `altair`, `anywidget`) is **lazy** — mirroring the
   facade's `_require_pandas()` pattern (`bindings/steppe/__init__.py:83`) — so `import steppe`
   and the base wheel stay numpy-only.

2. **An extended `examples/` tree** — `examples/notebooks/` (alongside the existing
   `examples/python/` and `examples/cpp/`), holding the reproduced-study + demo marimo notebooks.
   These are the source-of-truth, git-diffable, CI-runnable notebooks (the "living canary" pattern
   the repo already uses). A thin `bindings/steppe/examples/` accessor exposes them post-install
   (via a `force-include` map, §7) so `steppe app haak2015` resolves after `pip install`.

3. **Packaging extras** `steppe[notebook]` and `steppe[app]` (§7), plus a `[project.scripts]`
   launcher entry (or a subcommand of Area 1's Python `steppe` dispatcher, §7.2).

**Why a subpackage and not the facade top-level.** The facade (`bindings/steppe/__init__.py`) is a
**frozen-shape, 23-name public surface** returning numpy/pandas; adding marimo/altair machinery to
it would (a) pull heavy transitive imports toward the base wheel and (b) churn a deliberately
stable surface. A submodule `steppe.explore` isolates the reactive layer behind a lazy import,
exactly as the plan intends ("New subpackage `bindings/steppe/explore` + examples/notebooks +
`steppe[notebook]`/`[app]` extras").

---

## 4. Delivery model (the GPU-only constraint, made concrete)

Three delivery tiers, each honoring "no real fit in the browser":

| Tier | Runs where | Compute | Used by |
|---|---|---|---|
| **Live app** (`steppe app`, `marimo run`) | Server-side on a CUDA box | Real GPU fits via `FitClient` (daemon-resident f2 → <200 ms, or local `read_f2`) | the flagship explorer, rotation leaderboard, sweep explorer |
| **Live notebook** (`marimo edit`) | The researcher's own CUDA box | Real GPU fits, in-process `LocalClient` | interactive exploration, the reproduced-study drop-in |
| **Docs replay** (`marimo export html` / `html-wasm`, mkdocs-marimo) | The reader's browser | **Precomputed only** — the notebook is run once on a GPU box at build time; the exported HTML replays cached outputs, no live fit | the website / tutorials / blog embeds |

The docs tier deliberately uses marimo's **static/WASM export as replay**: the notebook is
executed on a GPU box during the docs build (outputs baked in), and the browser shows the frozen
interactive-looking result. This is the *only* honest way to put "interactivity" in docs given
GPU-only compute — stated as a constraint, implemented as a build step, never a live fit.

---

## 5. Folder scaffolding + key interfaces

### 5.1 The subpackage tree

```
bindings/steppe/
├── __init__.py                     # (existing, unchanged) the 23-name facade
├── explore/                        # NEW — the reactive app layer (lazy-imports marimo/altair)
│   ├── __init__.py                 # public API: qpadm_app / rotation_app / sweep_app / run / connect
│   ├── _marimo.py                  # _require_marimo() / _require_altair() guards (mirror _require_pandas)
│   ├── client.py                   # FitClient protocol; LocalClient (in-proc) + DaemonClient (Area 6)
│   ├── widgets.py                  # pop_picker / weight_bar / p_gauge / residual_heatmap / rotation_table
│   ├── _launch.py                  # run(): the `steppe app` body — subprocess over `marimo run/edit`
│   ├── _provenance_cell.py         # thin re-export of steppe.provenance() (Area 5) for the methods cell
│   └── notebooks/                  # the SHIPPED, deployable marimo notebooks (pure .py, app = marimo.App())
│       ├── qpadm_explorer.py       # FLAGSHIP: picker + sliders → live weight bar + p-gauge + heatmap
│       ├── rotation_leaderboard.py # sortable mo.ui.table over qpadm_search(as_dataframe=True) + drilldown
│       └── sweep_explorer.py       # brushable Altair |Z| histogram + linked table  (GATED: needs sweep facade)
└── examples/                       # NEW — post-install discovery of the reproduced-study notebooks
    ├── __init__.py                 # list() / path(name) / run(name) over the bundled notebooks
    └── catalog.py                  # study -> (notebook path, expected numbers) for the parity asserts

examples/                          # (existing tree — SOURCE OF TRUTH, git-diffable, CI-run)
├── python/quickstart_qpadm.py      # (existing)
├── cpp/                            # (existing)
└── notebooks/                      # NEW
    ├── haak2015_qpadm.py           # reproduce Haak 2015 headline qpAdm; assert; drop into explorer
    ├── olalde2018_beaker.py        # reproduce Olalde 2018/19; assert; drop into explorer
    ├── explore_demo.py             # the "feel the speed" sprint demo (plan §4 Phase D item 10)
    └── README.md                   # how to run / edit / deploy the notebooks

deploy/                            # NEW — the `steppe app` deploy assets
├── Dockerfile                      # CUDA 13 base + steppe[app] wheel + notebooks; ENTRYPOINT marimo run --headless
└── README.md                       # host/port/token, one-box-serves-a-lab runbook
```

### 5.2 The app builders — `steppe/explore/__init__.py`

```python
# Every function lazy-imports marimo/altair via _require_marimo(); the base wheel stays numpy-only.

def qpadm_app(*, f2: "F2Blocks | str | None" = None,
              prefer_daemon: bool = True) -> "marimo.App":
    """Build the flagship reactive qpAdm explorer as a marimo.App.
    `f2` = an F2Blocks handle, an f2-dir path, or None (picked in-app).
    Composes pop_picker + sliders → client.qpadm(...) → weight_bar/p_gauge/residual_heatmap."""

def rotation_app(*, f2: "F2Blocks | str | None" = None,
                 prefer_daemon: bool = True) -> "marimo.App":
    """The rotation leaderboard: mo.ui.table over qpadm_search(as_dataframe=True),
    filterable (feasible / p>alpha / source count); row-click drills into weights + heatmap."""

def sweep_app(*, sweep_path: str) -> "marimo.App":
    """Brushable |Z| histogram + linked top-quartets table over a sweep output.
    GATED on the sweep Python facade (currently CLI-only — see §9 open questions)."""

def connect(f2: "F2Blocks | str | None" = None, *,
            prefer_daemon: bool = True) -> "FitClient":
    """Return a DaemonClient if `steppe serve` (Area 6) is reachable, else a LocalClient
    that holds the resident F2Blocks in this process (read_f2 paid once via persistent_cache)."""

def run(app: str = "qpadm", *, mode: str = "run", host: str = "127.0.0.1",
        port: int = 2718, headless: bool = True, token: "str | None" = None) -> int:
    """The `steppe app` command body: launch a shipped notebook via `marimo run/edit`.
    `app` ∈ {"qpadm","rotation","sweep","haak2015",...}; resolves to a notebooks/ path."""
```

### 5.3 The fit client — `steppe/explore/client.py`

The abstraction that lets one notebook run **either** against the resident daemon (Area 6, fast)
**or** in-process (works today, before the daemon lands). This is Area-3-owned code; it *references*
Area 6's server contract, it does not implement the server.

```python
from typing import Protocol
from .. import QpAdmResult, QpWaveResult, F4Result, F2Blocks   # the frozen facade result classes

class FitClient(Protocol):
    def qpadm(self, *, target: str, left: list[str], right: list[str], **opts) -> QpAdmResult: ...
    def qpadm_search(self, *, target: str, models: list, right: list[str], **opts) -> list[QpAdmResult]: ...
    def f4(self, quartets: list, **opts) -> F4Result: ...
    @property
    def pops(self) -> list[str]: ...          # the resident P-axis labels (for the picker fallback)

class LocalClient(FitClient):
    """In-process: holds an F2Blocks (read_f2 cached via mo.persistent_cache) and calls the
    facade directly. Cold-load once; every fit is the facade call. Works with NO daemon."""
    def __init__(self, f2: "F2Blocks | str", *, device: int = 0): ...

class DaemonClient(FitClient):
    """Talks to the Area-6 `steppe serve` daemon over its local websocket/HTTP contract;
    the f2 tensor stays VRAM-resident in the daemon, so each call is the <200 ms compute path.
    A drop-in for LocalClient — SAME method signatures returning the SAME result classes."""
    def __init__(self, url: str = "ws://127.0.0.1:8787"): ...
```

### 5.4 The reactive components — `steppe/explore/widgets.py`

```python
def pop_picker(*, catalog=None, source: "list[str] | None" = None,
               multi: bool = True, label: str = "population") -> "mo.ui.Element":
    """AADR fuzzy type-ahead. Backed by steppe.pops (Area 4) when available — surfacing n,
    coverage, date range, region (+ a small map via anywidget, later). Falls back to the
    resident client.pops label list when the catalog isn't installed. .value -> list[str] of
    Group IDs, composable straight into qpadm(target=..., left=..., right=...)."""

def weight_bar(result: "QpAdmResult", *, backend: str = "altair"):
    """The live stacked weight bar (SE whiskers, infeasible flagged from result.feasible).
    backend="altair" for the reactive tier; delegates to steppe.plot.weights (Area 2) for a
    pinned/publication snapshot."""

def p_gauge(p: float, *, alpha: float = 0.05): ...          # green/red p-value gauge vs alpha
def residual_heatmap(result: "QpAdmResult"):                # Altair residual-z heatmap (the "aha" view)
def rotation_table(df) -> "mo.ui.table":                    # sortable/filterable leaderboard over the search DataFrame
```

### 5.5 A shipped notebook (shape) — `steppe/explore/notebooks/qpadm_explorer.py`

A real marimo notebook (`app = marimo.App()`; `@app.cell` functions). The load-once/fit-many
reactive structure, verbatim in spirit:

```python
import marimo
app = marimo.App()

@app.cell                        # loaded ONCE (persistent_cache); survives kernel restarts
def _load():
    import steppe
    from steppe.explore import connect, widgets
    client = connect(F2_DIR, prefer_daemon=True)        # DaemonClient or LocalClient
    return client, widgets

@app.cell                        # the UI — changing any of these re-runs ONLY the fit cell
def _controls(widgets, client):
    target = widgets.pop_picker(source=client.pops, multi=False, label="target")
    left   = widgets.pop_picker(source=client.pops, multi=True,  label="left")
    right  = widgets.pop_picker(source=client.pops, multi=True,  label="right")
    return mo.ui.form(mo.vstack([target, left, right]))   # form-gated submit (§2.4 debounce)

@app.cell                        # the FIT — the ONLY cell that re-runs; sub-second
def _fit(client, target, left, right):
    res = client.qpadm(target=target.value, left=left.value, right=right.value)
    return res

@app.cell                        # the live render
def _view(widgets, res):
    mo.vstack([widgets.weight_bar(res), widgets.p_gauge(res.p), widgets.residual_heatmap(res)])
```

### 5.6 A reproduced-study notebook (CI canary) — `examples/notebooks/haak2015_qpadm.py`

Runs as a marimo app *and* as `python haak2015_qpadm.py` in CI. Reads a small real-AADR subset,
fits, **asserts vs the published numbers** (from `steppe.examples.catalog`), then instantiates
`qpadm_app(f2=..., ...)` seeded with the paper's pops — "does it reproduce the canon?" made
runnable, doubling as living parity docs (plan §Area 3).

---

## 6. How it plugs into existing code (the concrete seams)

- **Facade result classes** (`bindings/steppe/__init__.py`): `QpAdmResult` (`.weights`,
  `.feasible`, `.p`, `.rankdrop` — lines 113–190), `QpWaveResult`, `F4Result`, `F2Blocks`
  (`read_f2`, `.pops`, `.to_numpy()`). The widgets read these **directly**; no new compute, no new
  result type. The rotation leaderboard is `qpadm_search(..., as_dataframe=True)` (line 1023) →
  `mo.ui.table`, exactly the 1:1 mapping the plan's grounding note calls out.
- **The lazy-import pattern**: `explore/_marimo.py`'s `_require_marimo()` is a copy of the facade's
  `_require_pandas()` (`__init__.py:83`) — same clear "install `steppe[notebook]`" error, same
  base-wheel-stays-lean guarantee.
- **The launcher**: `run()` shells out to `marimo run <notebooks/…>.py --headless --host --port
  --token-password` (or `marimo edit` for `mode="edit"`). It does **not** touch the C++ `steppe`
  binary (which the wheel does not build — pyproject `-DSTEPPE_BUILD_CLI=OFF`); it is a
  Python-side entry (§7.2).

---

## 7. Packaging

### 7.1 New extras (added to `[project.optional-dependencies]` in `pyproject.toml`)

```toml
# Reactive notebooks + widgets + charts. Lazy-imported inside steppe.explore, so the BASE wheel
# stays numpy-only (mirrors the existing `pandas` soft-dep pattern).
notebook = [
    "marimo>=0.9",      # the reactive notebook/app runtime (edit + run + script + export)
    "altair>=5",        # reactive Vega-Lite charts; mo.ui.altair_chart selections -> DataFrames
    "anywidget>=0.9",   # the custom AADR picker (map/tooltip tier); marimo's standard widget API
    "pandas>=1.5",      # the DataFrame surface the widgets + leaderboard consume
]
# The one-command deploy surface (`steppe app`): same runtime, intended for `marimo run` on a box.
app = ["steppe[notebook]"]
# (owned by the MASTER doc) all = ["steppe[viz,notebook,app,mcp]"]
```

`marimo`, `altair`, `anywidget` are **never** base deps — the base wheel's sole hard dep stays
`numpy>=1.22` (pyproject line 42–44). Bundling the reproduced-study notebooks into the wheel uses
scikit-build-core's **`force-include`** so `examples/notebooks/` is the single source of truth *and*
importable post-install:

```toml
[tool.scikit-build.wheel.force-include]
"examples/notebooks" = "steppe/examples/notebooks"   # one source-of-truth file, bundled into the wheel
```

### 7.2 The `steppe app` / `steppe explore` command — packaging decision (defer to Area 1)

The C++ `steppe` binary is **not** in the wheel, and it cannot launch marimo anyway (it is C++).
So `steppe app` / `steppe explore` are **Python** entry points. Two options, and the choice is an
Area-1/Area-5 decision (the Python `steppe` dispatcher), so Area 3 designs to slot into either:

1. **Preferred** — if Area 1/5 add a Python `steppe.__main__:main` console_scripts dispatcher over
   the facade, Area 3 **registers `app` / `explore` as subcommands** of it (calling
   `steppe.explore.run(...)`). One `steppe` command, consistent UX.
2. **Fallback / standalone** — Area 3 ships its own `[project.scripts]` entry so the feature exists
   independently:

   ```toml
   [project.scripts]
   steppe-app = "steppe.explore._launch:main"     # `steppe-app qpadm --host 0.0.0.0 --port 2718`
   ```

   plus the always-available `python -m steppe.explore <app> [--edit]` module runner.

Either way, the entry only functions when `steppe[app]` (hence marimo) is installed; without it,
`run()` raises the same clear `_require_marimo()` message.

---

## 8. Shared-infra dependencies (referenced, **not** re-scaffolded)

| Shared infra | Owner | How Area 3 depends on it | Fallback so Area 3 ships first |
|---|---|---|---|
| **`steppe serve` resident-f2 daemon** | **Area 6** | `DaemonClient` (`client.py`) connects to it for <200 ms live fits — the linchpin for *deployed* apps serving a lab. | **`LocalClient`** (in-proc `read_f2` + `persistent_cache`) makes every notebook run today, before the daemon exists. The `FitClient` protocol makes the swap invisible. |
| **AADR pops catalog `steppe.pops`** | **Area 4** | `pop_picker` reads it for n/coverage/date/region/map + fuzzy type-ahead. | Falls back to `client.pops` (the resident F2Blocks P-axis labels) so the picker works with just an f2-dir. |
| **Content-addressed f2 cache** | **Area 5** | The persistent-cache enabler: an Area-5 cached f2-dir path feeds `connect(f2=...)`; `mo.persistent_cache` is the notebook-local echo. | `read_f2` of any f2-dir works uncached. |
| **Provenance manifest `run.json` / `steppe.provenance()`** | **Area 5** | The notebook's closing "methods cell" calls `steppe.provenance()` (re-exported via `_provenance_cell.py`) for version/AADR-hash/pops/params + a paste-ready Methods paragraph + BibTeX. | Notebook runs without it (the cell is additive). |
| **Facade result classes / DataFrames** | **already in the facade** | Consumed directly by widgets + notebooks (`QpAdmResult.weights/.feasible`, `qpadm_search(as_dataframe=True)`, `F2Blocks`). | n/a — exists today. |
| **`steppe.plot` (static/publication figures)** | **Area 2** | `weight_bar(..., backend="mpl")` / "pin to figure" delegates to `plot.weights`; the reactive Altair tier is Area-3-owned. | Altair-only rendering works without Area 2. |
| **CLI output seam** | **Area 1** | Only via the Python `steppe` dispatcher decision (§7.2). | The standalone `steppe-app` console_script is the fallback. |

---

## 9. Intra-area build order

1. **`explore/client.py` — `FitClient` + `LocalClient`** (+ `_marimo.py` guards). Everything
   reactive needs a fit source; `LocalClient` works against the facade **today**, no daemon. Stub
   `DaemonClient` against Area 6's contract.
2. **`explore/widgets.py` — `pop_picker` (mo.ui fallback) + `weight_bar` + `p_gauge` +
   `residual_heatmap`.** Reusable across every notebook; picker starts on `client.pops`, upgrades to
   `steppe.pops` (Area 4) when present.
3. **`explore/notebooks/qpadm_explorer.py` (flagship) + `_launch.py` + the `steppe-app`
   console_script + `notebook`/`app` extras.** The plan's Phase-D showpiece; the first thing a
   user *feels*.
4. **`explore/notebooks/rotation_leaderboard.py`** over `qpadm_search(as_dataframe=True)`.
5. **`examples/notebooks/` reproduced-study notebooks + `steppe.examples` discovery + the CI
   parity assert** (script-mode `python …_qpadm.py`).
6. **`deploy/Dockerfile` (CUDA 13 + `steppe[app]`) + the docs replay export** (`marimo export
   html` at docs-build time).
7. **`explore/notebooks/sweep_explorer.py`** — **last, dependency-gated**: needs the sweep Python
   facade (currently CLI-only, §9 gap).

Upgrade the `pop_picker` to the **anywidget** map/tooltip version (§2.3) whenever Area 4's catalog
carries lat/long — a drop-in on the same `.value -> list[str]` interface.

---

## 10. Open questions

1. **Sweep Python facade gap.** The f4/D all-quartets sweep is **CLI-only** (`cmd_fstat_sweep.cpp`,
   the STPFST1 shards); there is no `steppe.sweep(...)` / `read_fstats` in the facade. The sweep
   explorer (step 7) is **blocked** on adding that facade — is it in Area 3's scope or a
   compute-team item? (Plan flags this explicitly as the Area-3 gap.)
2. **The daemon contract (Area 6).** `DaemonClient` needs a frozen wire protocol: websocket JSONL
   vs HTTP, the request/response schema for `qpadm`/`qpadm_search`, how the daemon keys resident
   `F2Blocks` handles, and auth. Area 6 owns it; Area 3's `FitClient` shape must be co-designed so
   `LocalClient`↔`DaemonClient` stay drop-in.
3. **`steppe app` command home.** Standalone `steppe-app` console_script vs a subcommand of a
   unified Python `steppe` dispatcher (Area 1/5). Affects UX consistency and the `[project.scripts]`
   table.
4. **Debounce.** marimo re-runs on every slider tick; is form-gating (`mo.ui.form`/`run_button`)
   acceptable UX for the "drag a slider and watch it re-fit" pitch, or do we need a real
   client-side debounce (version-dependent marimo feature)? Affects how "live" the slider *feels*.
5. **Reproduced-study data provisioning.** The Haak/Olalde notebooks need a small real-AADR subset
   on disk (no synthetic data, per project policy). Do we ship a curated f2-dir fixture, or
   `extract-f2` from a documented AADR subset at first run? Ties into Area 4/5 (the AADR version
   stamp + the content-addressed cache).
6. **marimo version pinning.** marimo is pre-1.0 with an evolving `mo.ui`/anywidget API; do we pin a
   tested minor (`marimo==0.x.y`) for reproducibility and bump deliberately, or float `>=0.9`?
   Affects the `--sandbox` PEP-723 reproduction story.
7. **Docs replay freshness.** The HTML export bakes numbers at docs-build time on a GPU box — how do
   we keep the replayed figures from drifting from the live engine across releases (a CI re-export
   gate)?

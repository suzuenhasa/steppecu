# Area 5 — Workflow, Reporting, Reproducibility & Onboarding: Engineering Architecture

> **What this doc is.** The *engineering architecture* (the **how**) for Area 5 of the steppe
> experience layer: the folder scaffolding, the tech-stack decisions, the interfaces to the
> existing code, and the packaging. It is the complement to
> [`docs/EXPERIENCE-PLAN.md`](../EXPERIENCE-PLAN.md) §Area 5, which is the *product plan* (the
> **what / why** — the feature table, the delight, the pre-ship sprint). **This doc does not
> restate the feature table**; read the plan for the vision and read this for the build.

> **Grounding rule.** Every decision below is anchored in the real tree: the C++ CLI seam
> (`src/app/cmd_emit.hpp`, `cli_parse.cpp`), the content-addressed f2 writer
> (`src/app/f2_dir_writer.{hpp,cpp}` — which **already** emits `meta.json` + `sha256:` digests),
> the frozen `RunConfig` (`src/core/config/run_config.hpp`), the nanobind facade + result classes
> (`bindings/steppe/__init__.py`), and the packaging (`pyproject.toml`). Line references are to
> those files as of this branch.

---

## 1. Role of this area (reference, not restatement)

Area 5 turns a *run* into a *finished, cited, reproducible artifact* and gets a new user to a
real result in two minutes. Per the plan (§Area 5), it covers: `steppe quickstart` + `doctor`;
the run→report generator (`--report html|pdf`); the provenance manifest `run.json` + `steppe
replay`; result-aware guardrails; `steppe cite`/`publish`; the analysis recipes (`steppe ask`);
and the live-refit surface.

**Area 5 additionally OWNS two pieces of shared infrastructure** that the whole live/interactive
half of the plan sits on (Areas 3/6/7 depend on them — this doc scaffolds them once, here):

- **(iii) the content-addressed f2/project cache** — the enabler that makes the exploratory loop
  feel instant because only the cheap GLS solve re-runs.
- **(v) the provenance manifest `run.json`** — the reproducibility anchor consumed by Areas
  2/5/6/7 for citation, replay, and figure provenance.

**Scope discipline / de-duplication with sibling areas.** Two plan features appear in more than
one area; Area 5 deliberately does **not** re-own them:

- The *live-refit widget / TUI* (`steppe explore`) is architected by **Area 3** (marimo) and
  **Area 6** (the `steppe serve` daemon). Area 5 provides the **cache + manifest** those surfaces
  consume and stops there. Area 5's `steppe ask` recipes run **batch** on the facade in the
  pre-ship sprint; the *interactive* recipe variant rides on the Area 6 daemon post-1.0.
- *Figure provenance / auto-caption* is an **Area 2** (`steppe.plot`) feature; Area 5 supplies the
  `RunManifest` it stamps into the figure and consumes Area 2's figures into the report.

---

## 2. The load-bearing discovery: the content-addressing already exists in C++

The single most important grounding fact for this area is that **steppe already writes a
content-addressed provenance sidecar** for every extracted f2 dir. `write_f2_dir()`
(`src/app/f2_dir_writer.cpp:373`) emits `meta.json` (schema `kF2MetaSchemaVersion`,
`f2_dir_writer.hpp:51`) containing:

- `f2_cache_id` = `sha256:` of the **complete `f2.bin`** (always computed —
  `f2_dir_writer.cpp:442`),
- `pops_sha256` = `sha256:` of the exact `pops.txt` name→index map,
- `source.{geno,snp,ind}_sha256` = dataset content hashes (opt-in via `--hash`),
- the **full extraction params**: `blgsize_cm`, `precision_tag`, `precision_mantissa_bits`, and
  the resolved `FilterConfig` (`maf_min`, `maxmiss`, `mind_max_missing`, `autosomes_only`,
  `drop_monomorphic`, `transversions_only`), plus the `pop_selection` echo
  (`f2_dir_writer.cpp:484-532`).

The SHA-256 is a real, `sha256sum`-compatible, SHA-NI-accelerated implementation
(`sha256_file()`, `f2_dir_writer.cpp:356`). **This is exactly the cache key material and exactly
the manifest input-hash material.** Area 5 therefore **reuses** this rather than rebuilding it:

- the **f2 cache key** = a stable digest over `{source shas, sorted pop set, extraction params}`
  — every ingredient is already in `meta.json`;
- the **manifest `inputs[]`** = the `f2_cache_id` + `pops_sha256` + source shas read straight
  out of the f2 dir's `meta.json`.

The Python side mirrors the same `sha256:` scheme (`hashlib.sha256`) so a key computed in Python
matches a digest written in C++ byte-for-byte — content addressing is only sound if both languages
agree on the bytes.

---

## 3. Fit decision: **HYBRID** (new Python subpackage + a unified Python front door), zero new C++

| Candidate | Verdict |
|---|---|
| Extend the C++ CLI (`src/app`) | **No** for the bulk. The report generator (Jinja2 + WeasyPrint), the recipe engine (YAML), the Zenodo client (HTTP), and pydantic manifest validation are Python-native and would drag Pango/Cairo/jinja into the lean compute binary — a category error against `cpu-is-test-only` / lean-wheel. |
| Extend the nanobind facade (`bindings/steppe/__init__.py`) | **No.** That module is the frozen-shape result/DataFrame facade (23 `__all__` names). Workflow orchestration is a *layer over* it, not more of it. |
| New Python subpackage `bindings/steppe/workflow/` | **Yes — primary.** All of manifest / cache / report / recipes / cite / doctor / quickstart is Python over the existing facade + the C++ `meta.json`. |
| Separate package/app | **No.** It ships in the same wheel behind extras (`steppe[workflow]`, `steppe[report]`), so `import steppe.workflow` is one install. |

**The one genuinely new seam is a CLI front door.** The plan's verbs — `quickstart`, `doctor`,
`replay`, `cite`, `publish`, `ask`, `init`, `ls`, `log`, and the `--report` flag — **do not exist
in the C++ binary and should not** (they are Python-native). The C++ `steppe` binary is also **not
built into the wheel** (`pyproject.toml` sets `-DSTEPPE_BUILD_CLI=OFF`, `build.targets=["_core"]`),
so `pip install steppe` yields no `steppe` command today. Area 5 therefore proposes a **unified
Python `steppe` console-script front door** (`[project.scripts]`, a Typer app) that:

- handles the **workflow verbs natively** in Python (report/replay/cite/ask/doctor/quickstart/…);
- **dispatches compute verbs** (`qpadm`, `f4`, …) to the C++ `steppe` binary **if it is on `PATH`**,
  else to the facade — so `steppe qpadm … --report html` works whether or not the C++ binary is
  installed.

This is the concrete realization of the **explicit packaging decision Area 1 flagged** ("ship a
`steppe` command from `pip install`… a Python `console_scripts` entry that dispatches to the
facade, OR flip the wheel to bundle the C++ binary"). **Ownership boundary:** Area 1 owns the C++
output/emit seam and the golden-safe CSV/JSON contract; Area 5 owns the Python front door and its
Python-native verbs. The naming co-existence is a cross-area open question (§9, Q1).

---

## 4. Tech-stack decisions (researched; alternatives weighed)

| Concern | Choice | Why (vs the alternatives) |
|---|---|---|
| **Manifest model + schema** | **pydantic v2** | Versioned, validated, round-trippable, and gives **free JSON-Schema export** for the `run.json` contract (so `replay` can validate an old manifest). `dataclasses` have no runtime validation or schema; `attrs`+`cattrs` validate but need extra wiring and still no schema gen. pydantic is the standard for exactly this "typed, versioned wire object" job and is a light pure-Python dep. |
| **Report templating** | **Jinja2** | The de-facto Python HTML templating engine: autoescape (XSS-safe on pop labels), sandbox, template inheritance for the shared report shell. `string.Template` is too weak for tables/loops; Mako is heavier and less standard. |
| **HTML → PDF** | **WeasyPrint** (default), Playwright *optional/parked* | WeasyPrint needs **no headless browser**, produces tiny deterministic print-CSS PDFs (measured ~21 KB vs Playwright's 59–125 KB for a complex doc), and its footprint is Pango/Cairo (now shippable as pip binary wheels) vs Playwright's **~550 MB Chromium**. Reports are server-rendered, JS-free, print-shaped — WeasyPrint's exact sweet spot. Playwright wins only when templates need JS/modern-grid; parked behind a future `report-browser` extra. `wkhtmltopdf` is deprecated/unmaintained. **Typst** was evaluated: it is a *markup language*, not an HTML→PDF path, so it does not fit the "render the same HTML report to PDF" model; noted as a future high-fidelity typesetting option (§9, Q3). |
| **Recipe format** | **YAML (`PyYAML` `safe_load`) validated by a pydantic schema** | Recipes are declarative *question → step-chain* specs that must be **human-editable and shareable as data** (`recipe.yaml`). A full workflow engine (**Snakemake/Nextflow/CWL**) is the wrong abstraction and massive overkill — those model file-DAG dataflow across processes; a steppe recipe is a short ordered list of facade calls with parameter binding. A bespoke Python DSL is not shareable as a file. So: YAML for the surface, pydantic for the schema, a small interpreter (`recipes/steps.py`) that maps step verbs to facade functions. `ruamel.yaml` is the round-trip alternative if we later *emit* comment-annotated recipes. |
| **Cache store** | **Content-addressed filesystem dir under `platformdirs.user_cache_dir("steppe")`** + a small JSONL index | The cache *value* is a multi-file STPF2BK1 f2 dir already on disk with a `meta.json` provenance sidecar. A SQLite/polars catalog would **duplicate the filesystem truth**; a lightweight JSONL index (for `steppe ls`/`log`) over content-addressed dirs is enough and stays inspectable. `platformdirs` gives XDG-correct, cross-platform cache placement (over hard-coding `~/.cache`). |
| **Content hashing** | **`hashlib.sha256`**, mirroring the C++ `sha256:` scheme | The cache key must be reproducible **across the C++ writer and the Python cache**; both hash the same bytes with SHA-256. blake3 is faster but the C++ side is already committed to sha256 (`f2_dir_writer.cpp`), and *key parity* dominates *hash speed* for content addressing. |
| **Citation / DOI** | **`requests` → Zenodo REST API** + **CFF** (`CITATION.cff`) + hand-templated **BibTeX** | Zenodo's deposition API is simple REST (create deposition → upload files → publish → DataCite DOI), with a `sandbox.zenodo.org` test env — no heavy SDK warranted. CFF is the emerging repo-citation standard Zenodo itself consumes. BibTeX for steppe + ADMIXTOOLS 2 + the AADR DOI is a fill-a-template job. |
| **GPU/env doctor** | **`nvidia-ml-py` (`pynvml`)** + a **`ctypes` `dlopen` probe** of `libcudart.so.13` | pynvml gives driver version, per-GPU name + VRAM free/total, and device count via NVML — structured, no fragile `nvidia-smi` text parsing. The `ctypes` probe of `libcudart.so.13` (the wheel's `DT_NEEDED` runtime floor) checks the **CUDA-13 system requirement** the wheel documents but does not bundle. Both are optional: `doctor` degrades to a clear "install `steppe[workflow]`" message if pynvml is absent. |
| **CLI framework** | **Typer** (+ **Rich** for the readiness tables) | Type-hint-driven commands, auto-generated `--help` with examples, and built-in shell-completion — matching the clig.dev ergonomics the plan wants for the Python-native verbs. Rich renders the green/red `doctor` summary. `argparse` is verbose; raw `click` is more boilerplate. (Area 1 owns the *C++* CLI ergonomics; this mirrors them Python-side.) |

**Provenance-pattern reference (not a dep).** The manifest + `replay` design follows the
**QIIME 2 Provenance-Replay** pattern the plan cites: a self-describing record with **checksum-based
validation** that can regenerate an executable reproduction and **detect post-hoc alteration**.
We adopt the *pattern* (content hashes + a replay-and-diff), not the framework. **RO-Crate / W3C
PROV** were considered as the manifest interchange standard; deferred as heavyweight for v1 (a
flat, versioned pydantic JSON is enough) but flagged as the export target if inter-tool provenance
becomes a requirement (§9, Q7).

---

## 5. Folder scaffolding (the actual tree to create)

New Python subpackage under the existing wheel package (`wheel.packages=["bindings/steppe"]`, so it
ships automatically):

```
bindings/steppe/workflow/
├── __init__.py            # public API surface (re-exports the callables below)
├── _hashing.py            # sha256 helpers that MATCH src/app/f2_dir_writer.cpp's "sha256:" scheme
├── provenance.py          # ✦ SHARED INFRA (v): the run.json manifest — RunManifest + capture/replay
├── cache.py               # ✦ SHARED INFRA (iii): content-addressed f2/project cache + steppe init/ls/log
├── diagnostics.py         # result-aware guardrails: Status taxonomy + heuristics -> plain-language flags
├── doctor.py              # CUDA/driver/VRAM/data-dir readiness (pynvml + ctypes cudart probe)
├── quickstart.py          # guided first-run on a bundled real-AADR subset -> fit + report + manifest
├── cite.py                # methods paragraph + BibTeX + CITATION.cff; Zenodo publish (requests)
├── cli.py                 # the Typer "steppe" front door (workflow verbs + compute dispatch)
├── report/
│   ├── __init__.py        # generate_report(result, fmt=...) -> Path
│   ├── render.py          # Jinja2 env + WeasyPrint PDF path + figure embedding (calls steppe.plot)
│   ├── interpret.py       # narrator: builds the plain-language reading from result fields
│   └── templates/
│       ├── base.html.j2   # shared shell + theme_steppe CSS (print @page rules for PDF)
│       ├── qpadm.html.j2  # the qpAdm model-line + weights table + interpretation + methods
│       ├── f4.html.j2     # f4/f3/qpdstat forest + table
│       └── dates.html.j2  # DATES decay curve + date±SE
├── recipes/
│   ├── __init__.py        # run_recipe(name_or_path, ...) -> RecipeResult
│   ├── schema.py          # the recipe.yaml pydantic schema (question -> params -> steps)
│   ├── steps.py           # step-verb -> facade-call interpreter (f3-screen / rotate / rank / report)
│   └── builtin/
│       ├── is-admixed.yaml
│       ├── reproduce-paper.yaml
│       └── build-a-graph.yaml
└── data/
    └── quickstart/        # tiny REAL-AADR subset (or a fetch-manifest pointing at it) for quickstart
```

Packaging touch-point: **`pyproject.toml`** (new extras — §7). **No `src/app` or `_core` changes
are required for v1**; the only optional C++ touch is the cross-area `--manifest` flag (§9, Q2),
which belongs to Area 1's emit seam.

---

## 6. Key interfaces (signatures + how they plug into existing code)

### 6.1 Provenance manifest — `workflow/provenance.py` (shared infra v)

```python
from pydantic import BaseModel

SCHEMA_VERSION = 1  # bump on a field-set change; replay() validates against it

class InputHash(BaseModel):
    role: str            # "f2_dir" | "geno" | "snp" | "ind" | "pops"
    path: str
    sha256: str          # "sha256:<hex>" — read from the f2 dir's meta.json (C++-written)

class DeviceRecord(BaseModel):
    gpu_name: str; cuda_runtime: str; driver: str
    precision: str; mantissa_bits: int          # mirrors meta.json precision_tag/bits

class RunManifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    steppe_version: str                          # steppe.__version__ (single-sourced, D2)
    steppe_git_sha: str | None
    created_utc: str
    command: str                                 # "qpadm" | "f4" | "dates" | "ask:is-admixed"
    params: dict                                 # resolved kwargs (target/left/right/rank/fudge/...)
    resolved_pops: dict                          # {"target":..., "left":[...], "right":[...]}
    inputs: list[InputHash]
    device: DeviceRecord
    wall_ms: float
    result_summary: dict                         # p/chisq/dof/feasible/status/weights (from the result class)

def capture(result, *, command: str, params: dict, f2, wall_ms: float) -> RunManifest: ...
def write_manifest(m: RunManifest, path: str = "run.json") -> "Path": ...      # m.model_dump_json(indent=2)
def load(path: str = "run.json") -> RunManifest: ...                            # RunManifest.model_validate_json
def replay(path: str = "run.json", *, rerun=True, tol: float = 0.0) -> "ReplayReport":
    """Re-run the recorded command via the facade and DIFF result_summary; flag drift
       (QIIME-2 pattern). Data-hash mismatch -> the 'which AADR version?' error surfaced loudly."""
```

*Plugs into:* the facade result classes (`QpAdmResult`, `F4Result`, …) supply `result_summary`;
`f2: F2Blocks` supplies `.pops`/`.P`/`.device`; the input hashes are read from the f2 dir's
`meta.json` (C++ `f2_cache_id`/`pops_sha256`/source shas). `steppe.__version__` is the
already-single-sourced version.

### 6.2 Content-addressed cache — `workflow/cache.py` (shared infra iii)

```python
class ExtractParams(BaseModel):    # the subset of FilterConfig/knobs that affect f2 bytes
    blgsize: float; maf: float; maxmiss: float
    autosomes_only: bool; drop_monomorphic: bool; transversions_only: bool
    ploidy: str; strand_mode: str; precision: str | None

def cache_key(*, source_shas: dict[str, str], pops: list[str], params: ExtractParams) -> str:
    """sha256 over canonical-JSON of {sorted source shas, SORTED pops, params}. 'sha256:<hex>'.
       Mirrors the C++ meta.json ingredients so the key is stable and reproducible."""

class F2Cache:
    def __init__(self, root: "Path | None" = None):   # default platformdirs.user_cache_dir("steppe")
        ...
    def get_or_extract(self, prefix, *, pops: list[str], **extract_kwargs) -> "steppe.F2Blocks":
        """HIT -> steppe.read_f2(cached_dir); MISS -> steppe.extract_f2(prefix, pops=..., out=dir)
           then read_f2. The C++ writer drops meta.json into the dir = the cache entry sidecar."""
    def path_for(self, key: str) -> "Path": ...
    def index(self) -> list["CacheEntry"]: ...          # backs `steppe ls`

# project layer (steppe init / lock / log):
def init_project(directory: str = ".") -> "Path": ...   # writes steppe.lock + .steppe/
def log_run(m: RunManifest, directory: str = ".") -> None: ...   # append .steppe/runlog.jsonl (steppe log)
```

*Plugs into:* wraps the **existing** `steppe.extract_f2(..., out=dir)` /
`steppe.read_f2(dir)` facade (both already implemented, `__init__.py:413,407`). **This is the
object Areas 3/6/7 call** so their live surfaces never re-extract — they call
`F2Cache.get_or_extract(...)` once and fit-many over the resident handle.

### 6.3 Report generator — `workflow/report/`

```python
def generate_report(result, *, fmt: str = "html", manifest: RunManifest | None = None,
                    out: str | None = None, journal: str | None = None) -> "Path":
    """result: any facade result class. Renders a self-contained report:
       result table + Area-2 figure (steppe.plot) + interpret.reading(result) +
       cite.methods_paragraph(manifest) + BibTeX. fmt='html' -> Jinja2;
       fmt='pdf' -> WeasyPrint(HTML)."""
```

*Plugs into:* dispatches on the result class to pick the template + the Area-2 plotting call
(`steppe.plot.weights` / `.forest` / `.decay`); `interpret.py` reads the same fields the CLI
`--explain` reads (`p`, `feasible`, `rankdrop.p_nested`, `status`). Figures are inlined as base64
SVG/PNG so the HTML/PDF is self-contained.

### 6.4 Recipes — `workflow/recipes/`

```python
class Step(BaseModel):
    verb: str                # "f3_screen" | "rotate" | "rank_feasible" | "report"
    with_: dict = {}         # bound params (YAML 'with:')

class Recipe(BaseModel):
    question: str
    params: dict             # declared inputs (target, pool, right, ...)
    steps: list[Step]

def run_recipe(name_or_path: str, *, f2, report: bool = True, **bindings) -> "RecipeResult":
    """Load a builtin/<name>.yaml or a user recipe.yaml, bind params, run the step chain over the
       facade (f3/D screen -> qpadm_search rotation -> rank feasible p>alpha -> generate_report)."""
```

*Plugs into:* `steps.py` maps verbs to `steppe.f3`/`steppe.qpdstat`/`steppe.qpadm_search`; the
final `report` step calls §6.3. **Batch** in v1; a future daemon-backed live variant is Area 6.

### 6.5 Doctor / quickstart / cite

```python
# doctor.py
def doctor(data_dir: str | None = None) -> "DoctorReport":
    """pynvml: driver + per-GPU VRAM; ctypes dlopen('libcudart.so.13'); facade `import steppe`
       smoke; optional data-dir + pops.txt cross-check (Area 4 catalog). Rich green/red table."""

# quickstart.py
def quickstart(data: str | None = None) -> "Path":
    """Point at data/quickstart/ (real-AADR subset), print the exact command, run ONE qpadm fit
       via the facade, annotate inline (interpret.py), emit report + run.json. Returns the report."""

# cite.py
def methods_paragraph(m: RunManifest) -> str: ...
def bibtex(m: RunManifest) -> str: ...                 # steppe + ADMIXTOOLS 2 + AADR DOI
def citation_cff() -> str: ...
def publish_zenodo(bundle_dir: str, *, token: str, sandbox: bool = False) -> str:  # -> DOI URL
```

### 6.6 CLI front door — `workflow/cli.py`

Typer app registered as `[project.scripts] steppe = "steppe.workflow.cli:main"`. Verbs:
`doctor`, `quickstart`, `init`, `ls`, `log`, `replay <run.json>`, `cite <run.json> [--publish]`,
`ask <recipe> …`, and a `--report {html,pdf}` option on the compute verbs (which dispatch to the
C++ `steppe` binary if on `PATH`, else the facade, then post-process to a report + `run.json`).

---

## 7. Packaging (the extras + their deps)

Add to `pyproject.toml` `[project.optional-dependencies]` (base wheel stays **numpy-only**):

```toml
# Area 5 core workflow layer — light, pure-Python deps only.
workflow = [
    "pydantic>=2",        # manifest + recipe schema (validation + JSON-Schema export)
    "pyyaml>=6",          # recipe.yaml surface
    "platformdirs>=4",    # XDG-correct cache dir
    "typer>=0.12",        # the `steppe` front door
    "rich>=13",           # doctor readiness tables / annotated quickstart
    "requests>=2.28",     # Zenodo REST publish
    "nvidia-ml-py>=12",   # doctor GPU/VRAM/driver probe (import-guarded; optional at runtime)
]
# Adds the HTML/PDF report generator; depends on the Area-2 viz layer for the figures.
report = [
    "steppe[workflow]",
    "steppe[viz]",        # Area 2: steppe.plot (figures embedded into the report)
    "jinja2>=3",
    "weasyprint>=62",     # HTML -> PDF (pulls Pango/Cairo binary wheels)
]
# steppe[all] (union, defined once — coordinate with the master doc) includes workflow + report.
```

Rationale: a user who only needs manifests + the cache + `doctor`/`quickstart` installs
`steppe[workflow]` (no native PDF deps). `steppe[report]` layers Jinja2 + WeasyPrint + the Area-2
viz. The `steppe` console-script is registered by the base package but its Python-native verbs
raise a clear "install `steppe[workflow]`" if the extra is missing (import-guarded, mirroring the
facade's lazy `_require_pandas` pattern, `__init__.py:83`).

---

## 8. Shared-infrastructure dependencies (reference, do NOT re-scaffold)

| Infra | Owner | Area 5 relationship |
|---|---|---|
| **(iii) content-addressed f2 cache** | **Area 5 (this doc)** | **OWNED here** (`workflow/cache.py`). Reuses the C++ `meta.json`/`sha256:` content-addressing (`f2_dir_writer.cpp`). Consumed by Areas 3/6/7. |
| **(v) provenance manifest `run.json`** | **Area 5 (this doc)** | **OWNED here** (`workflow/provenance.py`). Consumed by Areas 2 (figure provenance), 6 (session share), 7 (anti-hallucination stamp). |
| (i) CLI output seam (`cmd_emit.hpp`) | Area 1 | The Python front door **dispatches to** the C++ binary and reuses its `--format json` bytes / `meta.json` content hashes. Cross-area packaging decision (the `steppe` script) is co-owned; §9 Q1/Q2. |
| (ii) AADR pops catalog (`steppe.pops`) | Area 4 | **Depended on** by `doctor` (data-dir cross-check), `quickstart` (pop validation), and recipes (resolving fuzzy pop names before a fit). |
| (iv) resident-f2 daemon (`steppe serve`) | Area 6 | **Optional dependency.** Recipes + `steppe explore` run batch on the facade in v1; the *live* variant rides the daemon post-1.0. Area 5 does not build the daemon. |
| (vi) result classes / DataFrames | facade (built) | **Depended on** — `capture()`, `generate_report()`, recipes all consume `QpAdmResult`/`F4Result`/… directly (`__init__.py`). |
| Area 2 `steppe.plot` | Area 2 | **Depended on** by `report/render.py` for the embedded figures (hence `steppe[report]` → `steppe[viz]`). |

---

## 9. Intra-area build order

1. **`_hashing.py` + `provenance.py`** — the `RunManifest` schema is the anchor everything else
   stamps; build and freeze it first (shared infra v).
2. **`cache.py`** — the enabler under all live features (shared infra iii); depends on `_hashing`,
   wraps the existing `extract_f2`/`read_f2`.
3. **`doctor.py`** — standalone, cheap, high-value onboarding win; no report dependency.
4. **`diagnostics.py` + `report/`** — the report generator; depends on the manifest, the Area-2
   viz layer, and the result classes.
5. **`quickstart.py`** — the guided first-run; depends on the fit path + report + manifest.
6. **`cite.py`** — methods/BibTeX/CFF from the manifest (offline first); Zenodo `publish` last.
7. **`recipes/`** — the L-effort `steppe ask`; depends on the cache + facade + report.
8. **`cli.py`** — grows verb-by-verb alongside 3–7; it is the integration surface, not a
   prerequisite for any single piece.

Matches the plan's pre-ship sprint sequencing: **doctor+quickstart (Phase D, S)** and the
**manifest (Phase D, M)** land in 0.1.0; **report/recipes/publish** are the steady-value /
big-bet follow-ons.

---

## 10. Open questions

1. **CLI naming co-existence (co-owned with Area 1).** The Python `[project.scripts] steppe`
   front door vs a possible future C++ `steppe` binary shipped in the wheel. Proposal: the Python
   `steppe` is the front door and shells to a system C++ binary (found on `PATH`); if the wheel
   later bundles the compiled binary, install it as `steppe-core` (or `_steppe_compute`) and have
   the front door invoke that. Needs a joint Area-1/Area-5 sign-off.
2. **`run.json` emission point.** Python-front-door-only vs also adding a `--manifest run.json`
   flag to the C++ `emit_to_destination` seam so a pure-C++ `steppe qpadm` (no Python) also drops
   provenance. The C++ side already has `RunConfig` + the SHA infra; the schema owner stays Python.
   Decision belongs to Area 1's seam.
3. **WeasyPrint vs Typst / browser tier.** WeasyPrint (Pango/Cairo) on a headless CUDA/HPC box:
   confirm the pip binary wheels resolve without system libs, or gate PDF behind a documented
   `apt install` note. Parked `report-browser` (Playwright) and `report-typst` tiers for
   pixel-perfect / advanced-typesetting needs.
4. **Cache eviction + format-bump invalidation.** LRU/size-cap GC policy for
   `platformdirs`-cached f2 dirs, and how aggressively a `kF2DiskVersion` (`f2.bin`) bump
   invalidates cached entries. Also: cache-key inclusion of `strand_mode`/`ploidy` (both affect
   f2 bytes — already in `ExtractParams`).
5. **Zenodo bundle contents + AADR licensing.** Depositing the raw f2 dir may run into AADR
   redistribution terms; likely deposit only the **manifest + report + reproduction script**, not
   the genotypes. Token UX + sandbox-vs-prod flow.
6. **Recipe engine scope for 0.1.** `steppe ask` batch-only (facade) vs requiring the Area-6
   daemon. Recommendation: batch-only in the pre-ship sprint; defer the live variant.
7. **Manifest interchange standard.** Whether to export the flat pydantic `run.json` as
   **RO-Crate / W3C PROV** for inter-tool provenance, or keep it steppe-native. v1 keeps it native;
   revisit if a collaborator toolchain demands the standard.
```

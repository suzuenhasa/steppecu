# Area 4 — Data & Population Discovery (`steppe.pops`): Engineering Architecture

> **Scope of this doc.** `docs/EXPERIENCE-PLAN.md` is the **product** plan — *what* the
> `.anno` catalog should feel like and *why* (the feature tables in its "Area 4" section,
> the pain it kills, the prioritization). **This** doc is the **engineering architecture** —
> *how* it is built: the tech-stack decisions, the folder scaffolding, the interface
> signatures, how they plug into the real code, and the packaging. It references the plan's
> features; it does not restate them.
>
> Read the plan's Area 4 table first (EXPERIENCE-PLAN.md §"Area 4"). This doc assumes it.

---

## 1. Area role & boundary

Area 4 owns **shared infrastructure (ii): the AADR population catalog**, exposed as the
Python subpackage `steppe.pops` and the `steppe pops` CLI command group. It turns the real
AADR annotation file — `/home/suzunik/steppe/aadr/v66.p1_HO.aadr.PUB.anno`, a **quoted-TSV,
49 columns, 27,594 sample rows, 4,265 Group IDs, 1,849 singletons** — into a fast,
version-aware, fuzzy **search / filter / validate / card** layer.

This catalog is a **producer of data for four other areas** and a **consumer of none of
their live compute** (pure metadata; **not** GPU-bound — see the plan's grounding note "the
speed payoff is *indirect*"). Its consumers:

| Consumer | What it reads from `steppe.pops` |
|---|---|
| **Area 1** (CLI) | rich catalog-backed "did you mean" suggestions *with sample counts* (`suggest`), `steppe pops validate` pre-flight |
| **Area 3** (marimo) | the AADR picker widget (fuzzy type-ahead over the catalog → a plain label list) |
| **Area 6** (STUDIO / Atlas) | source/right chips, and lat/long + date-BP for the map + time-scrub |
| **Area 7** (MCP) | the catalog exposed as an MCP **resource**; `pops.search` / `pops.validate` as MCP tools |

**The boundary with Area 1's `PopResolver` (state it explicitly).** The plan says "wire the
catalog into `PopResolver`." `PopResolver` is **C++** (`src/app/pop_resolver.{hpp,cpp}`) and
holds only the ≤ P labels of one f2-dir's `pops.txt`; the catalog is **Python + SQLite** and
holds all 4,265 AADR Group IDs plus per-sample metadata. The C++ binary cannot call Python,
so the responsibility splits cleanly:

- **Area 1 owns the *cheap* in-binary suggestion**: on a `PopResolver` miss, rank the
  `pops.txt` label vector it already holds by edit distance (Levenshtein/token overlap) —
  no catalog, no dependency. This ships in the C++ CLI.
- **Area 4 owns the *rich* catalog-backed suggestion**: `Catalog.suggest()` /
  `Catalog.validate()` rank against the **full 4,265-group AADR namespace** and annotate each
  candidate with `n`, coverage, and date range (*"did you mean `Russia_Samara_EBA_Yamnaya`
  (n=10, cov 1.2×)?"*). This serves the **Python** facade, the `steppe pops validate` CLI,
  the marimo picker, and the MCP tools.

That split is an open question to confirm (§9), but it is the only design that respects the
"C++ can't import Python" reality without reimplementing rapidfuzz + a columnar store in C++.

---

## 2. Tech-stack decision

The catalog is **tiny** (15 MB TSV, 27 k rows) but has four hard requirements that drive the
stack: (a) a **correct quoted-TSV parse** (embedded commas *and newlines* in quoted fields —
naive `split('\t')` is wrong), (b) **composable AND filters** + per-pop **aggregation**
(group-by over 27 k rows → 4,265 cards), (c) **fast fuzzy/substring** ranking over 4,265
labels, (d) a **version-stamped, content-addressed** artifact so `pops diff v62 v66` and
reproducibility work. The overriding constraint: **keep the base wheel numpy-only** — every
heavy dep is an optional extra with a graceful stdlib fallback.

### 2.1 The quoted-TSV parser → **stdlib `csv`** (zero dep)

**Decision: Python's standard-library `csv.reader(f, delimiter='\t', quotechar='"')`.**

The `.anno` embeds newlines inside quoted header/field cells (the "Genetic ID" header alone
spans several visual lines; the `ASSESSMENT WARNINGS` header is a paragraph). Verified on the
real file: stdlib `csv` parses it to exactly **49 columns / 27,594 data rows / 4,265 Group
IDs / 1,849 singletons** — the numbers the plan cites — whereas a `tr '\t' '\n' | head -1`
split reports a wrong column count because it splits *inside* the quoted header.

- *Alternatives weighed.* **polars `read_csv`** (typed, fast) and **pyarrow.csv** both parse
  quoted TSV correctly and faster — but they are **not needed at 27 k rows** (the whole build
  is <200 ms in pure Python) and would force a heavy dep into the *build* path. **Reject** as
  a hard dep; keep polars as an **optional accelerator** only if the multi-version registry
  ever indexes many large `.anno` files at once (§9). stdlib `csv` keeps the index builder in
  the **base wheel** with zero new deps — the right default for a one-time build.

### 2.2 The columnar / queryable store → **stdlib `sqlite3` + FTS5 (trigram)** (zero dep)

**Decision: SQLite (stdlib `sqlite3`), a single-file content-addressed catalog, with an FTS5
`trigram` virtual table for substring/prefix search.**

Why SQLite over the columnar-file options:

| Option | Verdict |
|---|---|
| **SQLite (stdlib)** | **Chosen.** Zero new dep (stdlib `sqlite3`), single-file artifact, SQL expresses the composable AND filters + the per-pop card **joins** natively, **FTS5 trigram** gives fast substring/prefix over 4,265 labels, multi-version registry via a `version` column / `ATTACH DATABASE`. Verified: this box's `sqlite3` (3.39.3) has FTS5 + the trigram tokenizer. |
| **Parquet + polars/pyarrow** | Great columnar group-by, tiny file — but reading it needs polars/pyarrow (**heavy dep**), has **no built-in text index** (fuzzy still needs rapidfuzz on top), and Parquet is append-hostile for a version registry. Wrong for a *queryable* store; right only as an interchange/export format. |
| **DuckDB** | "SQLite for analytics" — excellent, columnar, can query Parquet in place. But it is a ~20 MB dep and **overkill for 27 k rows**; its win (vectorized analytics over millions of rows) never materializes here. |
| **pandas/polars in-memory only** | No persistence, no content-addressing, rebuilt every process. Fine as the *return* type, not the *store*. |

**FTS5 trigram** is the right search primitive precisely for the AADR label problem: the
trigram tokenizer treats every 3-char window as a token, so `Yamnaya` substring-matches
`Russia_Samara_EBA_Yamnaya` and `Beaker` finds `Britain_Bell_Beaker` — the exact "type three
letters" experience the plan wants. **Graceful fallback:** if a user's CPython `sqlite3` was
built without FTS5, the search layer degrades to `WHERE group_id LIKE '%q%'` (linear over
4,265 rows = sub-ms) — correctness preserved, only the index dropped.

**Content-addressing.** The catalog file is named by the **SHA-256 of the `.anno` bytes** +
a parser-schema version, so a given AADR release always maps to one immutable catalog and
`pops diff` can trust identity. This **reuses Area 5's content-addressed cache convention**
(§8) rather than inventing a second hashing scheme.

### 2.3 Fuzzy ranking → **`rapidfuzz`** (optional `[pops]` extra, `difflib` fallback)

**Decision: `rapidfuzz` for the "did you mean" scorer, in the `steppe[pops]` extra, with a
stdlib `difflib.get_close_matches` fallback so the base install still suggests.**

rapidfuzz is the clear 2025/26 winner: a C++ core, ~40 % faster than the field, **MIT**
(vs TheFuzz's GPL — matters for a redistributable wheel), and it bundles `WRatio` /
`token_sort_ratio` / `process.extract` which are exactly the compound-label metrics AADR
needs (`Anatolia_N` → `Anatolia_Neolithic`). Over 4,265 labels a `process.extract` call is
sub-millisecond, so no index is required — the group vocabulary is loaded from SQLite once
and ranked in memory.

- *Alternatives.* **TheFuzz/fuzzywuzzy** — same API, slower, GPL. **jellyfish** — phonetic
  (Soundex/Metaphone), wrong tool for structured underscore labels. **stdlib `difflib`** —
  no dep, `SequenceMatcher`-based, "good enough" ranking → the **fallback** so `steppe.pops`
  suggests even in the bare base wheel.

### 2.4 The cache-dir resolver → **`platformdirs`** (optional `[pops]` extra, env fallback)

**Decision: `platformdirs.user_cache_path("steppe")` for the catalog location**, in the
`[pops]` extra, falling back to `$STEPPE_CACHE` → `~/.cache/steppe` when absent.

platformdirs is the maintained successor to `appdirs`, is the de-facto standard for exactly
this ("regenerable data that improves performance" — a rebuildable index is textbook
`user_cache_dir`), and is a pure-Python featherweight. The fallback keeps the base wheel from
needing it.

### 2.5 The `steppe pops` CLI framework → **Typer** (on the *shared* launcher, not owned here)

**Decision: the `pops` command group is a `typer.Typer()` sub-app**, mounted onto the shared
Python `steppe` launcher (§7), and independently runnable as `python -m steppe.pops`.

Typer is the right pick for this group: it is type-hint-first (matches the facade's style),
gives **shell completion for free** (which Area 1's plan explicitly wants for pop names), and
is Click underneath (mature, 38.7 % adoption) so nothing is lost. It weighs one dep, which
belongs to the **shared `[cli]` extra owned by Area 1**, not to `[pops]` — the `steppe.pops`
*Python API* has no CLI dep at all. (Argparse was rejected: the composable-filter surface
`--region/--country/--culture/--bp-from/--min-coverage/--quality/--datatype` is exactly the
boilerplate Typer removes; Click is the acceptable fallback if the launcher standardizes on
it.)

---

## 3. Fit decision

**Hybrid**: a **new Python subpackage `bindings/steppe/pops/`** (the catalog engine + the
`steppe.pops` public API) **plus** a **Typer sub-app** contributed to the shared `steppe` CLI
launcher.

Grounded in the real code:

- **Not C++.** The compute CLI (`src/app/`, 14 subcommands) is a pure-host C++ binary that
  `std::exit()`s inside each CLI11 callback and is **not even built into the wheel**
  (`pyproject.toml`: `-DSTEPPE_BUILD_CLI=OFF`, `build.targets=["_core"]`). Reimplementing a
  quoted-TSV reader + a columnar store + rapidfuzz in C++ to add a `pops` subcommand there
  would be a large lift for a **metadata, non-GPU, non-perf-critical** feature — precisely
  the work Python exists to absorb. The whole experience layer is "Python over the facade;
  only Area 1 is primarily C++," and Area 4 is the archetype of that.
- **New subpackage, not a flat facade extension.** The existing facade
  (`bindings/steppe/__init__.py`, ~1,100 lines, 23 `__all__` names) is a single flat module
  of result classes + stat functions. The catalog is a *cohesive, multi-file subsystem*
  (parser, schema, store, fuzzy, registry, cards, CLI) — it belongs in its own package
  `bindings/steppe/pops/`, re-exported as `steppe.pops`, exactly as the plan names it
  (`pops.search(...)` / `.card(...)` / `.suggest(...)` / `.validate(...)`).
- **DataFrames match the house style.** `steppe.pops` returns **pandas** (the facade's
  existing lazy-`_require_pandas` soft dep), so `search().query(...)` composes with the rest
  of the facade and drops into notebooks natively.

---

## 4. Folder scaffolding

```
bindings/steppe/pops/
├── __init__.py        # public API surface (§5.1): build_index, open_catalog, and the
│                      #   module-level search/card/samples/suggest/validate/versions/diff
│                      #   that delegate to the default (latest) catalog. Re-exported as steppe.pops.
├── anno.py            # THE quoted-TSV reader. read_anno(path) -> Iterator[SampleRow] via
│                      #   csv.reader(quotechar='"'); COLUMN_MAP (49 cols -> typed fields),
│                      #   coerce() (BP int, coverage float w/ '..'/'n/a'/'' -> None, lat/long),
│                      #   header-hash detection (schema drift guard).
├── suffixes.py        # decode_suffix('.SG'|'.DG'|'.HO'|'.AG'|'.TW'|'-o' ...) -> legend text,
│                      #   parsed once from the Genetic-ID header cell. Used by cards + validate.
├── regions.py         # derive_region(political_entity, lat, long) -> region. THERE IS NO
│                      #   region column (verified) — a curated country->macro-region lookup.
├── schema.py          # SQLite DDL + build SQL: `sample` (27,594 typed rows), `pop`
│                      #   (4,265 aggregates), `meta` (aadr_version, anno_sha256, parser_ver,
│                      #   counts, built_at), FTS5 `pop_fts` (tokenize='trigram').
├── index.py           # build_index(anno, out=None, version=None, force=False):
│                      #   read_anno -> INSERT sample -> aggregate -> INSERT pop -> build FTS5
│                      #   -> stamp meta. Content-addressed path from anno_sha256 (refs Area 5).
├── catalog.py         # class Catalog: open + query. search()/card()/samples()/prefixes()/
│                      #   tree(). Lazy pandas assembly. Holds the group vocab for fuzzy.py.
├── fuzzy.py           # suggest(name, ...) via rapidfuzz.process.extract (difflib fallback);
│                      #   validate(labels)/validate_poplist(path) -> ValidationReport with
│                      #   per-unknown suggestions + sample counts.
├── cards.py           # @dataclass PopCard + sparkline DATA (date-BP histogram bins,
│                      #   coverage-dist bins, sex split, quality breakdown). DATA ONLY —
│                      #   terminal rendering is Area 1's rich/plotext layer.
├── registry.py        # multi-version registry over the platformdirs cache dir:
│                      #   versions(), register(version, path), diff(v_a, v_b) -> DataFrame.
├── build.py           # build_ind(labels|filter, out, merge_label=None): sample-set -> an
│                      #   .ind / poplist ready for steppe.extract_f2 (facade).
├── paths.py           # cache_root() via platformdirs (env / ~/.cache/steppe fallback);
│                      #   content-address helpers (reuse Area 5's hashing).
├── _emit.py           # Python-side csv/tsv/json/auto emitter mirroring result_emit.cpp's
│                      #   `# section:`/JSON shapes + the isatty TTY-split (see §8, dep on Area 1).
└── cli.py             # typer.Typer() `pops` sub-app: index/search/show/validate/build/diff
                       #   (+ `pick`, deferred). Mounted on the shared launcher; `python -m`.

bindings/steppe/pops/__main__.py   # `python -m steppe.pops` -> cli.app()

tests/python/
├── test_py_pops_anno.py       # parse -> 49 cols, 27,594 rows, 4,265 groups, 1,849 singletons,
│                              #   embedded-newline header, '..' coverage sentinel -> None.
├── test_py_pops_index.py      # build_index determinism; anno_sha256 stable; meta counts.
├── test_py_pops_search.py     # AND-filter composition; 'Yamnaya' matches; prefix; FTS5 + LIKE fallback.
├── test_py_pops_validate.py   # suggest ranking (Anatolia_N -> Anatolia_Neolithic); validate_poplist.
└── test_py_pops_card.py       # card aggregates for a known group; samples() shape.
```

Nothing large is bundled: the 15 MB `.anno` stays user-supplied; the built SQLite catalog
lives in the cache dir, not in the wheel.

---

## 5. Key interfaces

### 5.1 Python API (`steppe.pops`)

```python
# bindings/steppe/pops/__init__.py

def build_index(
    anno: str | os.PathLike,
    *,
    out: str | None = None,          # None -> content-addressed path in the cache dir
    version: str | None = None,      # AADR label ("v66"); default = parsed from filename
    force: bool = False,             # rebuild even if a catalog for this anno_sha256 exists
) -> "Catalog": ...
    # One-time builder (plan: `steppe pops index`). Parses the quoted-TSV, aggregates per
    # Group ID, writes SQLite (+FTS5), stamps anno_sha256 + version, registers it. Idempotent.

def open_catalog(version: str | None = None) -> "Catalog": ...
    # Open the named (or latest-registered) catalog. Raises if none built.

class Catalog:
    version: str
    anno_sha256: str
    n_samples: int
    n_groups: int

    def search(
        self,
        query: str | None = None,           # substring/fuzzy over Group ID (FTS5 trigram)
        *,
        region: str | None = None,
        country: str | None = None,         # matches Political Entity
        culture: str | None = None,         # token in Group ID
        bp_from: int | None = None,         # date_bp_mean >= (years BP)
        bp_to: int | None = None,
        period: str | None = None,          # named window -> (bp_from, bp_to)
        min_coverage: float | None = None,
        min_snps: int | None = None,
        min_n: int | None = None,           # group size
        quality: str | None = None,         # 'pass' | 'questionable' | 'critical'
        datatype: str | None = None,        # '1240K' | 'HO' | 'Shotgun' ...
        limit: int | None = None,
        as_dataframe: bool = True,
    ) -> "pandas.DataFrame | list[PopRow]": ...
        # Composable AND filters over the `pop` table (+FTS5 for `query`). One row / Group ID.

    def card(self, group_id: str) -> "PopCard": ...
        # Rich per-pop card: n, sample_ids, date-BP histogram, coverage dist, snps-hit,
        # country/lat/long, sex split, quality breakdown, decoded suffix legend. DATA only.

    def samples(self, group_id: str, *, as_dataframe: bool = True) -> "pandas.DataFrame | list[SampleRow]": ...
        # Per-sample rows for a group (Genetic ID, date, coverage, quality, suffix).

    def suggest(self, name: str, *, limit: int = 5, min_score: float = 60.0) -> "list[Suggestion]": ...
        # rapidfuzz ranking over the 4,265-group vocab (difflib fallback). Each Suggestion:
        # (group_id, score, n, coverage, date_range). Powers "did you mean".

    def validate(self, labels: "list[str]") -> "ValidationReport": ...
    def validate_poplist(self, path: str) -> "ValidationReport": ...
        # Resolve labels against the namespace; unknowns carry ranked suggestions + counts.
        # Pre-flights a long run (plan: `steppe pops validate pops.txt`).

    def prefixes(self, prefix: str) -> "list[str]": ...     # `Italy_` -> children
    def tree(self) -> "dict": ...                           # Country -> Period -> Culture outline

    def build_ind(self, labels_or_filter, *, out: str, merge_label: str | None = None) -> str: ...
        # Sample-set builder: selected Genetic IDs -> an .ind/poplist for steppe.extract_f2.

# Module-level convenience (delegate to the default catalog) — mirrors the plan verbatim:
def search(*a, **k): ...
def card(gid): ...
def samples(gid, **k): ...
def suggest(name, **k): ...
def validate(labels): ...
def versions() -> "list[str]": ...
def diff(v_a: str, v_b: str) -> "pandas.DataFrame": ...   # `steppe pops diff v62 v66`
```

### 5.2 SQLite schema (the store, `schema.py`)

```sql
CREATE TABLE sample (            -- one row per Genetic ID (27,594)
  genetic_id TEXT PRIMARY KEY, group_id TEXT, individual_id TEXT,
  date_bp_mean INTEGER, date_bp_sd INTEGER,
  coverage_1240k REAL, snps_1240k INTEGER, snps_ho INTEGER,
  locality TEXT, political_entity TEXT, lat REAL, long REAL,
  molecular_sex TEXT, datatype TEXT, pulldown TEXT, suffix TEXT,
  assessment TEXT, publication TEXT, doi TEXT, y_hap TEXT, mt_hap TEXT
);
CREATE INDEX sample_group ON sample(group_id);

CREATE TABLE pop (               -- one row per Group ID (4,265), aggregated from `sample`
  group_id TEXT PRIMARY KEY, n INTEGER, is_singleton INTEGER,
  date_bp_min INTEGER, date_bp_max INTEGER, date_bp_median INTEGER,
  coverage_median REAL, snps_median INTEGER,
  countries TEXT, region TEXT,             -- region derived (regions.py), not a source col
  n_pass INTEGER, n_questionable INTEGER, n_critical INTEGER
);

CREATE VIRTUAL TABLE pop_fts USING fts5(group_id, tokenize='trigram');  -- substring/prefix

CREATE TABLE meta (              -- one row: provenance stamp
  aadr_version TEXT, anno_sha256 TEXT, parser_version INTEGER,
  n_samples INTEGER, n_groups INTEGER, built_at TEXT
);
```

### 5.3 The parser contract (`anno.py`) — the load-bearing detail

The single riskiest interface. It MUST use a real quoted reader; verified numbers are the
regression oracle:

```python
COLUMN_MAP = {  # 0-based csv column index -> (field, coercer). 49 cols total.
    0: ("genetic_id", str),        14: ("group_id", str),      15: ("locality", str),
    16: ("political_entity", str), 17: ("lat", _float_or_none), 18: ("long", _float_or_none),
    10: ("date_bp_mean", _int_or_none), 11: ("date_bp_sd", _int_or_none),
    23: ("coverage_1240k", _cov_or_none),  # '..' (8,299 rows), 'n/a', '' -> None
    47: ("assessment", str), ...
}
def read_anno(path) -> "Iterator[SampleRow]":
    r = csv.reader(open(path, newline=""), delimiter="\t", quotechar='"')
    header = next(r); assert len(header) == 49   # schema guard
    for row in r: yield _coerce(row)
```

Invariants the tests pin: `sum(1 for _ in read_anno(anno)) == 27_594`;
`len({s.group_id for s in ...}) == 4_265`; singletons `== 1_849`;
`assessment` histogram `Pass 23,053 / PROVISIONAL_PASS 2,908 / Questionable 963 /
CRITICAL 373 / MERGE_PASS 183 / PROVISIONAL_CRITICAL 83`.

### 5.4 CLI (`cli.py`, Typer sub-app `pops`)

```
steppe pops index [ANNO] [--out DIR] [--version LABEL] [--force]
steppe pops search [QUERY] [--region --country --culture --bp-from --bp-to --period
                            --min-coverage --min-snps --min-n --quality --datatype --limit]
                   [--format auto|csv|tsv|json]      # obeys the Area-1 TTY split
steppe pops show   GROUP_ID [--json]                 # renders a PopCard (Area-1 rich)
steppe pops validate [POPLIST | --f2-dir DIR]        # pre-flight; cross-check pops.txt
steppe pops build  [filters] --out prefix.ind [--label NAME]
steppe pops diff   V_A V_B
steppe pops pick   [filters]                         # fzf-style TUI -> live fit (deferred, §6)
```

---

## 6. Packaging

Add ONE extra; the base wheel stays **numpy-only**. The `steppe.pops` **API** runs on the
base wheel (stdlib `csv` + `sqlite3` + lazy `pandas`) with graceful fallbacks; the extra only
*upgrades* fuzzy quality and cache-dir resolution.

```toml
[project.optional-dependencies]
# ... existing: pandas, test, lint, cuda ...
pops = [
    "rapidfuzz>=3.9",     # fuzzy "did you mean" (difflib fallback if absent)
    "platformdirs>=4",    # catalog cache dir  (~/.cache/steppe fallback if absent)
]
# The `steppe pops` CLI additionally needs the shared [cli] extra (typer, rich) owned by
# Area 1's launcher — the `pops` Typer sub-app is mounted there, not shipped standalone.
# `steppe[all]` aggregates pops + viz + app + mcp.
```

- **No change to `[tool.scikit-build]`** — this is pure Python, no `_core` / CMake / CLI
  build impact; `-DSTEPPE_BUILD_CLI=OFF` stays OFF. The subpackage ships under the existing
  `wheel.packages=["bindings/steppe"]` (it is `bindings/steppe/pops/`).
- **polars is deliberately NOT a dep** — stdlib `csv` builds the index; polars would only be
  a future optional accelerator for a many-version bulk reindex (§9).

---

## 7. Shared-infra dependencies (reference — do NOT re-scaffold)

| # | Shared piece | Owner | How Area 4 uses it |
|---|---|---|---|
| i | **CLI output seam** `emit_to_destination` / `--format` / `# section:`/JSON shapes | **Area 1** (`src/app/cmd_emit.hpp`, `result_emit.cpp`) | `steppe pops` is **Python**, so the C++ seam can't serve it. `_emit.py` **mirrors** its format tokens + isatty TTY-split so `pops search` output is CSV/JSON-clean when piped and pretty on a TTY. Card rendering (`pops show`) reuses Area-1's rich/plotext terminal layer. Depends on Area 1 fixing the **shared Python `steppe` launcher** (console_scripts) that mounts the `pops` sub-app. |
| ii | **The AADR pops catalog** | **Area 4 — THIS AREA OWNS IT** | Produced here; consumed by Areas 1/3/6/7 (table in §1). |
| iii | **Content-addressed f2 cache** | **Area 5** | The catalog artifact **reuses Area 5's content-addressing/hashing convention** (SHA-256 keying) and lives near the same cache root, rather than inventing a parallel scheme. Reference only. |
| v | **Provenance manifest `run.json`** | **Area 5** | `pops validate` / `pops build` and the resolved Group IDs are **stamped into `run.json`** (which AADR version + `anno_sha256` was used) — closing the "which AADR release did I use?" reproducibility gap. Reference only. |
| vi | **Result classes / DataFrames** | facade (`bindings/steppe/__init__.py`) | `steppe.pops` returns **pandas** via the same lazy `_require_pandas` soft dep; `build_ind()` output feeds `steppe.extract_f2(prefix, pops=...)` (facade). |
| — | **`PopResolver`** (C++) | **Area 1** | Boundary in §1: Area 1's cheap in-binary Levenshtein over `pops.txt`; Area 4's rich catalog-backed `suggest`/`validate`. |
| — | **Resident-f2 daemon `steppe serve`** | **Area 6** | Only the deferred `pops pick` → live-fit rides on it (§6, `Effort L`). The rest of Area 4 has **no daemon dependency**. |

---

## 8. Intra-area build order

Sequenced so each step is independently testable and the highest-pain feature lands early:

1. **`anno.py`** — the quoted-TSV reader + `COLUMN_MAP` + coercers. *Foundational risk;* gate
   on the §5.3 invariants (49/27,594/4,265/1,849 + the assessment histogram). Zero deps.
2. **`schema.py` + `index.py`** — the SQLite store + build/aggregate + content-addressed path
   + `meta` stamp. Delivers `steppe pops index` (the plan's *foundation* feature).
3. **`catalog.py`** — `search()` / `card()` / `samples()` / `prefixes()` / `tree()` over the
   store (composable AND filters; FTS5 with LIKE fallback). Delivers `pops search` + `pops show`.
4. **`fuzzy.py`** — `suggest()` / `validate()` (rapidfuzz + difflib fallback). Delivers the
   **highest-frequency papercut** fix; wire the Python facade's pop-taking functions + the
   marimo picker + MCP tools to it.
5. **`_emit.py` + `cli.py` + `__main__.py`** — the Typer `pops` sub-app on the shared launcher
   (needs Area-1's `[cli]` launcher; `python -m steppe.pops` works before it lands).
6. **`registry.py`** — multi-version registry + `diff` (the version-aware feature).
7. **`build.py`** — the sample-set builder → `.ind`/poplist → `extract_f2` (closes find→run).
8. **`pops pick`** *(deferred, `Effort L`)* — the fzf-style TUI live-fit; rides on the Area-6
   daemon + f2 cache. Post-0.1.0, matching the plan's deferral.

Steps 1–4 (+5's `python -m`) are the pre-ship Phase-B slice the plan sequences; 6–8 are the
"steady value / dependency-gated" tail.

---

## 9. Open questions

1. **The `steppe` front-door (shared, Area 1).** Does the wheel ship a **Python
   `steppe` console-script launcher** that owns the experience subcommands (`pops`, `plot`,
   `serve`, `mcp`, `explore`) and coexists with / supersedes the C++ `steppe_app` binary
   (currently `STEPPE_BUILD_CLI=OFF`, not in the wheel)? Both wanting `steppe` on `PATH` is a
   collision Area 1 must resolve. Until then, `python -m steppe.pops` is Area 4's entry.
2. **Is the C++/Python "did you mean" split (§1) accepted?** i.e. C++ CLI = cheap `pops.txt`
   Levenshtein; catalog-backed rich suggestions = Python-only. The alternative (an out-of-band
   catalog lookup from the C++ binary) is a much larger, uglier bridge.
3. **Cache-root coordination with Area 5.** Should the catalog live *inside* Area 5's
   content-addressed cache tree (one cache root, one hashing scheme) or in its own
   `platformdirs` dir that merely *reuses the convention*? Prefer the former; needs Area 5's
   root API.
4. **`region` derivation.** There is **no region column** (verified). The `--region` filter
   needs a curated `political_entity`/lat-long → macro-region lookup (`regions.py`). Where
   does that mapping live and who curates it? (Candidate: a small bundled JSON.)
5. **Field-schema spec.** The `.anno` is **49 columns, not the 45 the plan cites**, with `..`
   / `n/a` / empty sentinels (coverage `..` on 8,299 rows) and multi-value cells (multiple
   SNP-panel columns). The exact typed subset + coercion rules need to be frozen as the
   `parser_version` contract (bump it on any column-mapping change → new `anno_sha256`-keyed
   catalog).
6. **FTS5 availability.** Present on this box (`sqlite3` 3.39.3), but some minimal CPython
   `sqlite3` builds omit it — the LIKE fallback covers correctness; do we want a build-time
   probe that logs which path is active?
7. **polars accelerator.** Keep the builder stdlib-only, or add an optional polars fast-path
   for a bulk multi-version reindex if the registry ever grows to many large `.anno` files?

---

*This area is pure metadata — not GPU-bound — but it is the on-ramp to the sub-second fit:
find candidate pops → multi-select → watch the fit land. The catalog is the shared spine
Areas 1, 3, 6 and 7 all lean on to turn a 27,594-row annotation horror into a three-letter
type-ahead.*

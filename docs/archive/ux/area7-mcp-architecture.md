# Area 7 — the MCP / Agent Surface (`steppe.mcp`) — Engineering Architecture

> **What this document is.** `docs/EXPERIENCE-PLAN.md` is the **product** plan — the *what/why* of
> the MCP surface: an LLM that resolves fuzzy population names, runs a **real** GPU qpAdm, and answers
> with actual weights / p-values / feasibility, "the LLM plans, the GPU computes, nothing is
> hallucinated" (plan §2, Area 7). **This** document is the **engineering architecture** — the *how*:
> the MCP SDK decision (weighed against alternatives), the transport model, the tool/resource/prompt
> mapping onto the existing facade, the stateless-tool ↔ resident-handle resolution problem and its
> answer, the anti-hallucination *mechanism* (not just the promise), the folder scaffolding, the exact
> interface signatures, the packaging extra, and the dependency edges. It does **not** restate the
> plan's feature table; it references it and designs the machine underneath.

---

## 1. Area role — the trustworthy compute backend for AI-driven popgen

Area 7 exposes the entire steppe facade to any **MCP** (Model Context Protocol) client — Claude
Desktop, an IDE agent, or a hosted agent framework — as a set of **tools** (the analyses), **resources**
(the AADR catalog + the validation record + run manifests), and **prompts** (the `steppe ask` recipes).
The plan's five features (plan §2, Area 7 table — *not restated*) reduce to four engineering
responsibilities:

1. **A 1:1 tool surface over the facade** — `qpadm`/`qpwave`/`qpgraph`/`dates`/`f4`/`f3`/`f4ratio`/
   `qpdstat`/`qpadm_rotate`, each returning the structured result **plus a provenance stamp**.
2. **The pops catalog as a readable resource + fuzzy `search`/`validate` tools** so the model turns
   "Beaker" into real Group IDs *before* it computes (Area 4).
3. **The grounding / anti-hallucination contract** — every result carries params + data-hash +
   version, and the parity-vs-AT2 validation record is a first-class resource the model can read and
   cite. **This is the differentiator, and it is a mechanism, not a marketing line** (§6.5).
4. **The agentic loop riding the daemon** — the fit → read → refine → re-fit loop runs at
   conversational speed because the compute is sub-second and f2 stays VRAM-resident (Area 6).

The one architectural truth this area must honor: **the model plans; steppe computes; nothing is
fabricated.** An LLM wired to ADMIXTOOLS-2-in-R has neither the speed to loop nor a grounding contract
— that gap is the entire pitch. Everything below is in service of making "the number the user sees was
computed on the GPU and is parity-validated against AT2" a *structural* guarantee, not a hope.

The plan is explicit about phasing (§3 "Big bets", §4 deferred): **a read-only surface — catalog
search + a single fit — is near-term and cheap** (a thin stdio wrapper over the facade); **the full
agentic loop rides on the daemon + f2 cache.** The scaffolding below builds the cheap thing first and
grows into the daemon-backed thing without a rewrite (§4 backend seam, §9 build order).

---

## 2. The grounding facts this rides on

Three already-built surfaces make this area "reuse, not rebuild":

- **The facade is already the right shape.** `bindings/steppe/__init__.py` exposes 23 public names
  (`__all__`) returning typed result classes — `QpAdmResult` (`.weights`/`.rankdrop`/`.popdrop`
  DataFrames + `p`/`chisq`/`dof`/`f4rank`/`feasible`/`status`/`precision`), `F4Result`/`F3Result`/
  `F4RatioResult` (`.table`), `QpGraphResult`/`QpGraphSearchResult`, `DatesResult`, `F2Blocks`, and
  `qpadm_search(...)`. Every field an MCP tool needs to serialize already exists on these objects. **No
  compute and no new result type is built here** — the MCP layer is a *serializer + a stateless
  front door*.

- **The stateless-tool problem, and why it dictates the design.** An MCP tool call is
  JSON-in / JSON-out; the model **cannot hold a Python `F2Blocks` object** across calls the way a
  notebook can. But every fit function takes an `f2: F2Blocks` first argument
  (`qpadm(f2, ...)`, `f4(f2, ...)`, …). The resolution: **tools take a string `f2_ref`** — either an
  f2-dir path or an **Area-5 content-address cache key** — and the server resolves it to a live handle
  *server-side* (via the Area-5 `F2Cache` in-process, or the Area-6 `HandleRegistry` when
  daemon-backed). An `extract_f2` / `qpfstats` tool **returns** a `cache_key`; subsequent fit tools
  **consume** it. This makes the agentic loop *stateless on the wire but stateful server-side* — the
  canonical MCP pattern, and the reason Area 5's cache and Area 6's registry are hard dependencies,
  not nice-to-haves (§8).

- **The provenance material already exists.** Area 5's `RunManifest` (`workflow/provenance.py`:
  `steppe_version` + `command` + `params` + `resolved_pops` + `inputs[]` content-hashes + `device`
  + `result_summary`) is *exactly* the provenance stamp every tool result must carry. The MCP layer
  does not invent a provenance format — it **embeds `capture(...)`'s output** into each structured
  tool result (§6.4). Likewise the "data hash" is the C++ `meta.json` `sha256:` content-address
  (`f2_dir_writer.cpp`) surfaced through Area 5's `cache_key`.

---

## 3. Tech-stack decision

### 3.1 The MCP framework — **FastMCP 2.x (the standalone `fastmcp` package)**

The Model Context Protocol has three Python paths. Weighed:

| Candidate | Verdict | Why |
|---|---|---|
| **FastMCP 2.x** (`pip install fastmcp`, `from fastmcp import FastMCP`; jlowin → PrefectHQ) | **CHOSEN** | The actively-developed superset. Decorator model (`@mcp.tool` / `@mcp.resource(uri)` / `@mcp.prompt`) is the lowest-boilerplate way to register the 1:1 surface. **Three decisive wins for steppe:** (1) a first-class in-memory **`Client(server)`** so the tool surface is testable in-process with **zero subprocess** — the exact pattern our `tests/python` facade gates use; (2) `mcp.http_app(path=..., transport="streamable-http")` returns a **Starlette ASGI app that mounts into Area 6's Uvicorn** sharing one process / one CUDA context (the alignment Area 6 §3.1 already committed to); (3) `mcp.run()` (stdio) ↔ `mcp.run(transport="streamable-http")` is a **one-line transport switch**, so the same tool code serves a local Claude Desktop *and* a daemon-mounted remote surface. Pure-Python, gated behind `steppe[mcp]`. |
| **Official `mcp` SDK's bundled `mcp.server.fastmcp.FastMCP` (FastMCP 1.0)** | Rejected (as the primary) | FastMCP **1.0 was contributed into the official SDK and then frozen** there; 2.0 is where clients, auth, server composition, richer resource templates, and the ergonomic test client live. FastMCP 2.x depends on the official `mcp` package under the hood, so choosing 2.x is a **superset, not a fork** — we still get the reference protocol impl, plus the parts we need. Picking the frozen 1.0 to save one transitive dep is a false economy. |
| **The low-level `mcp.server.Server`** (explicit `list_tools`/`call_tool`/`list_resources`/`read_resource` handlers) | Rejected | Maximum control, maximum boilerplate — we'd hand-write the JSON-Schema for 14 tools that FastMCP derives from type hints + pydantic models for free. No upside for a surface that is fundamentally "call a facade function, serialize the result". Keep it in the back pocket only if a tool ever needs protocol-level behavior FastMCP hides. |
| **FastAPI-MCP / auto-expose-an-OpenAPI-as-MCP** | Rejected | Auto-generating tools from the Area-6 REST routes would leak HTTP-shaped params (path refs, status codes) into the tool schemas and couple the agent surface to the daemon's wire format. The tool surface must be **curated** — hand-picked names, grounding-aware descriptions, provenance-stamped returns — not a mechanical mirror of REST. |

**Structured output is load-bearing here.** The MCP `2025-06-18` spec added **structured tool output**
(a validated JSON object alongside the text content block) and **output schemas**. FastMCP derives both
automatically from a **pydantic return model** (§6.4): a tool that returns a `QpAdmToolResult`
pydantic model emits a machine-checkable schema *and* a structured payload the client can parse without
regexing prose. This is exactly what lets the provenance stamp be a *typed field the model must
surface*, not a sentence it can drop.

### 3.2 Transport — **stdio by default, streamable-HTTP for remote / daemon-mounted**

| Transport | Role | Why |
|---|---|---|
| **stdio** | **Default; the near-term read-only surface.** | How local MCP clients (Claude Desktop, Cursor, IDE agents) launch a server — they spawn `steppe-mcp` and speak MCP over stdin/stdout. Zero network, zero auth surface, trivial to configure. The stdio server runs the **in-process backend** (§4): each process holds one CUDA context and composes the facade directly. Perfect for "search the catalog + run one fit". |
| **streamable-HTTP** (the `2025-03/06` successor to HTTP+SSE) | **Remote clients + the daemon-backed agentic loop.** | For a hosted agent, or to hit the **resident f2** loop, the MCP server runs over streamable-HTTP. Its ASGI app **mounts into Area 6's Uvicorn** (`app.mount("/mcp", mcp.http_app(...))`, passing the FastMCP lifespan through — the documented nested-lifespan requirement) so tool calls submit to the **same `GpuWorker`** and hit the sub-200 ms path with **no second CUDA context**. This is the plan's "Daemon-backed resident f2" feature, realized as a sub-app mount, not a second daemon. |
| Plain HTTP+SSE (legacy two-endpoint) | Not targeted | Superseded by streamable-HTTP in the current spec; FastMCP speaks streamable-HTTP natively. No reason to target the deprecated transport. |

The consequence for the code: **transport is a runtime flag, never a fork of the tool code.** The same
`build_server(cfg)` registers the same tools/resources/prompts; `serve()` chooses `mcp.run()` vs
`mcp.run(transport="streamable-http")`; and `http_app()` hands the ASGI app to Area 6 for mounting
(§6.6).

### 3.3 The compute-routing seam — a **`ComputeBackend` protocol** (in-process ↔ daemon)

The single most important internal abstraction. A tool must not care *how* the fit runs — only that it
gets a result. So all device work goes through one protocol with two implementations:

- **`InProcessBackend`** — composes the facade directly (`steppe.qpadm(f2, ...)`), resolving `f2_ref`
  through the **Area-5 `F2Cache`**. Cold-ish (per-process CUDA context; the plain facade re-uploads f2
  per fit), but *correct* and *self-contained* — the near-term read-only surface with no daemon
  required.
- **`DaemonBackend`** — an `httpx` client to the **Area-6 daemon** `/v1/fit`, `/v1/rotate`,
  `/v1/graph/score`, `/v1/pops` routes (or, when the MCP app is *mounted inside* the daemon process,
  a direct in-process handle to the shared `GpuWorker` + `HandleRegistry`). This is the resident,
  sub-200 ms, multi-session path.

This seam is why "read-only surface now, agentic loop later" is a *config change*, not a rewrite: the
tools are written once against `ComputeBackend`; the deployment picks the impl (§6.2).

### 3.4 Structured schemas + validation — **pydantic v2** (shared with Area 5)

Tool return types and the `Provenance` stamp are **pydantic v2** models (§6.4). This is already the
Area-5 `RunManifest` stack, so the manifest embeds losslessly, the JSON-Schema is free (the MCP output
schema), and validation is one `model_validate`. No second serialization library is introduced; the
Area-6 `msgspec` wire types stay on the daemon side of the `DaemonBackend` boundary and never leak into
the MCP schema.

---

## 4. The fit decision — **new subpackage `bindings/steppe/mcp/` + `steppe[mcp]` extra + a `steppe-mcp` console script**

The plan calls it exactly: *"the MCP server is a thin Python wrapper (official `mcp` SDK / FastMCP)
over the existing facade + the Area-4 pops catalog + the Area-6 daemon — it reuses, not rebuilds"*
(plan §2, Area 7 grounding). Concretely:

1. **A new Python subpackage `bindings/steppe/mcp/`** — the server, tools, resources, prompts, the
   `ComputeBackend` seam, the schemas. Pure Python over the facade + Area-4 `steppe.pops` + (optionally)
   the Area-6 daemon. Ships *inside* the wheel but its heavy dep (`fastmcp`) is imported only under the
   `steppe[mcp]` extra (guarded import → a clear "install steppe[mcp]" error; **`import steppe` stays
   numpy-only**).
2. **A `[project.scripts]` console entry `steppe-mcp = "steppe.mcp.__main__:main"`** (and
   `python -m steppe.mcp`). Because the C++ CLI is not built into the wheel (`STEPPE_BUILD_CLI=OFF`,
   `pyproject.toml`), there is no `steppe` binary to hang a subcommand on — the console script is the
   launch surface, exactly as Area 6 ships `steppe-serve`. An MCP client's config points at
   `steppe-mcp` (stdio) or a mounted URL (HTTP).

**Why not extend an existing seam.** Area 1's C++ `emit_to_destination` is byte-stream file/stdout
output — the wrong abstraction for a stateful protocol server. The facade is *consumed* by the MCP
layer but is a stateless function library, not a protocol server. And the MCP surface is a distinct
responsibility (protocol handshake, tool schemas, resource URIs, prompt templates, the grounding
contract) → a new subpackage, cleanly dependency-gated. This mirrors Area 6's own "new subpackage +
extra + console script" decision and keeps the base wheel lean.

---

## 5. Folder scaffolding

```
bindings/steppe/
  __init__.py                 # (exists) the facade; add a guarded `from . import mcp` re-export note only
  mcp/                        # NEW subpackage — the MCP server (heavy dep fastmcp → steppe[mcp])
    __init__.py               #   public: build_server(), serve(), McpConfig; guarded "install steppe[mcp]" ImportError
    __main__.py               #   `python -m steppe.mcp` / console_script main(): argparse → transport+backend → serve()
    server.py                 #   build_server(cfg) -> FastMCP; wires backend, registers tools/resources/prompts, sets instructions
    config.py                 #   McpConfig (pydantic): transport{stdio,http}, backend{inproc,daemon}, f2_ref default, device, daemon_url, host/port, token
    backend.py                #   ComputeBackend Protocol + InProcessBackend (facade+F2Cache) + DaemonBackend (httpx → Area 6)
    context.py                #   resolve_f2(ref) -> steppe.F2Blocks | cache_key handling (Area-5 cache / Area-6 registry)
    schemas.py                #   pydantic result models w/ embedded Provenance: QpAdmToolResult, FStatToolResult, GraphToolResult, DatesToolResult, PopHit, Halo, Provenance
    provenance.py             #   stamp(result, *, command, params, f2, wall_ms) -> Provenance  (wraps Area-5 workflow.provenance.capture)
    grounding.py              #   GROUNDING_INSTRUCTIONS (server-level) + the validation-record resource body (parity vs AT2 + reproduced studies)
    tools/
      __init__.py             #     register_tools(mcp, backend) — the single wiring entry
      qpadm.py                #     qpadm / qpwave / qpadm_rotate (search) tools
      fstats.py               #     f4 / f3 / f4ratio / qpdstat / dstat tools
      graph.py                #     qpgraph / qpgraph_search tools
      dates.py                #     dates tool
      pops.py                 #     pops_search / pops_validate / pops_card  (thin over steppe.pops — Area 4)
      extract.py              #     extract_f2 / qpfstats — BUILD an f2 handle, RETURN a cache_key ref
    resources.py              #   register_resources(mcp, backend): steppe://pops/catalog, steppe://pops/{gid},
                              #     steppe://validation/parity, steppe://runs/{id}, steppe://f2/{cache_key}
    prompts.py                #   register_prompts(mcp): the `steppe ask` recipes as MCP prompts (Area-5 recipes)
    py.typed

tests/python/mcp/             # NEW — MCP surface tests (fastmcp in-memory Client, no subprocess)
  test_tools_inproc.py        #   every tool callable in-process; result shapes match the facade
  test_f2_ref_resolution.py   #   extract_f2 → cache_key → qpadm(cache_key) round-trip (the stateless-loop contract)
  test_provenance_stamp.py    #   EVERY tool result carries a Provenance block (the anti-hallucination invariant)
  test_resources.py           #   catalog / validation / run-manifest / f2 resources readable + well-formed
  test_prompts.py             #   the ask recipes render to messages with the right slots
  test_grounding.py           #   the validation resource content is present + the server instructions assert "compute, never invent"
  test_daemon_backend.py      #   (gpu/daemon-marked) DaemonBackend routes to a live Area-6 /v1/fit

docs/ux/
  area7-mcp-architecture.md   # THIS doc
```

---

## 6. Key interfaces

### 6.1 The `f2_ref` resolution (`context.py`) — the stateless-loop keystone

```python
# A tool NEVER receives an F2Blocks; it receives a string the server resolves.
def resolve_f2(ref: str, backend: "ComputeBackend", *, device: int = 0) -> "steppe.F2Blocks":
    """ref is one of:
         - an f2-dir path            -> steppe.read_f2(ref)               (direct)
         - an Area-5 cache key       -> F2Cache.path_for(key) -> read_f2  (content-addressed)
         - a resident handle id      -> Area-6 HandleRegistry.get(key)    (daemon-backed, no re-load)
       InProcessBackend uses the Area-5 F2Cache; DaemonBackend defers resolution to the daemon."""
```

### 6.2 The compute-routing seam (`backend.py`)

```python
class ComputeBackend(Protocol):
    """The single seam between the (stateless) MCP tools and the (stateful) GPU."""
    def load_f2(self, ref: str, *, device: int = 0) -> "steppe.F2Blocks": ...
    def extract_f2(self, prefix: str, *, pops: list[str], **kw) -> str: ...   # returns a cache_key ref
    def qpadm(self, f2, *, target, left, right, **opts) -> "steppe.QpAdmResult": ...
    def qpadm_rotate(self, f2, *, target, models, right, **opts) -> list["steppe.QpAdmResult"]: ...
    def qpwave(self, f2, *, left, right, **opts) -> "steppe.QpWaveResult": ...
    def f4(self, f2, quartets) -> "steppe.F4Result": ...
    def f3(self, f2, triples) -> "steppe.F3Result": ...
    def f4ratio(self, f2, tuples) -> "steppe.F4RatioResult": ...
    def qpdstat(self, f2, quartets) -> "steppe.F4Result": ...
    def qpgraph(self, f2, edges, **opts) -> "steppe.QpGraphResult": ...
    def qpgraph_search(self, f2, *, pops, **opts) -> "steppe.QpGraphSearchResult": ...
    def dates(self, prefix, target, source1, source2, **kw) -> "steppe.DatesResult": ...
    def stability(self, f2, *, target, left, right, **opts) -> "Halo | None": ...   # Area-6 bootstrap halo (safety flow-through)

class InProcessBackend:
    """Composes the facade directly; resolves f2_ref via the Area-5 F2Cache. One CUDA context
    per process. The near-term read-only surface — correct, self-contained, no daemon."""

class DaemonBackend:
    """httpx client to the Area-6 daemon (/v1/fit, /v1/rotate, /v1/graph/score, /v1/pops), or a
    direct GpuWorker handle when the MCP app is mounted inside the daemon process. The resident,
    sub-200 ms, multi-session path."""
```

### 6.3 A tool, shaped (`tools/qpadm.py`)

```python
def register_tools(mcp: "FastMCP", backend: "ComputeBackend") -> None:

    @mcp.tool
    def qpadm(f2_ref: str, target: str, left: list[str], right: list[str],
              rank: int = -1, allow_negative_weights: bool = True) -> QpAdmToolResult:
        """Run a REAL qpAdm GLS admixture fit on the GPU (parity-validated vs ADMIXTOOLS 2).
        Returns the computed weights, p-value, feasibility, and a provenance stamp. The numbers
        are computed on the GPU — report them verbatim; do NOT estimate or round-invent statistics.
        `f2_ref` is an f2-dir path or a cache key from `extract_f2`; resolve pop names with
        `pops_search`/`pops_validate` FIRST."""
        t0 = time.perf_counter()
        f2 = backend.load_f2(f2_ref)
        res = backend.qpadm(f2, target=target, left=left, right=right,
                            rank=rank, allow_negative_weights=allow_negative_weights)
        halo = backend.stability(f2, target=target, left=left, right=right)
        return to_qpadm_result(res, f2=f2, stability=halo,
                               params=locals_without(f2, backend),
                               wall_ms=1e3 * (time.perf_counter() - t0))
    # …qpwave, qpadm_rotate similarly; fstats/graph/dates in sibling modules.
```

`to_qpadm_result` reuses the **Area-1 model-line sentence** (`Target = 0.74*Steppe + 0.26*Anatolia |
p=0.31 | FEASIBLE`) as the human-readable `model_line` field, so the model narrates the exact sentence
a researcher would write — built from fields already on `QpAdmResult`.

### 6.4 The structured return + the provenance stamp (`schemas.py`, `provenance.py`)

```python
from pydantic import BaseModel

class Provenance(BaseModel):
    """The anti-hallucination stamp on EVERY tool result. Mirrors the Area-5 RunManifest fields
    so a run.json can be reconstructed from a tool result and vice-versa."""
    steppe_version: str
    precision: str                 # e.g. "emulated_fp64_40" — the arithmetic tier the number was computed in
    data_hash: str                 # the Area-5 cache_key / C++ meta.json 'sha256:<hex>' — WHICH f2 produced this
    resolved_pops: dict            # {"target":…, "left":[…], "right":[…]} — the names actually used
    params: dict                   # rank/fudge/als_iterations/… the fit ran with
    wall_ms: float
    grounding: str = "steppe://validation/parity"   # the resource proving these numbers are defensible

class WeightRow(BaseModel):  left: str; weight: float; se: float; z: float
class Halo(BaseModel):       weight_lo: list[float]; weight_hi: list[float]; n_resample: int   # bootstrap ribbon

class QpAdmToolResult(BaseModel):
    model_line: str                # the Area-1 "qpAdm as a sentence" verdict
    target: str; left: list[str]; weights: list[WeightRow]
    p: float; chisq: float; dof: int; f4rank: int
    feasible: bool; status: str
    stability: Halo | None         # safety flow-through (plan §2: "candidate, not answer")
    provenance: Provenance

# provenance.py — reuses Area-5, does NOT reinvent it:
def stamp(result, *, command: str, params: dict, f2, wall_ms: float) -> Provenance:
    """Thin adapter over workflow.provenance.capture(...): pull steppe_version/precision/data_hash/
    resolved_pops from the Area-5 manifest capture and shape a Provenance for the tool result."""
```

Because these are pydantic models, FastMCP emits the MCP **output schema** for free — the client sees
that `provenance` is a required field, so a well-behaved agent surfaces "computed at precision emu40
from f2 sha256:… — see steppe://validation/parity" alongside the weights. The stamp is **structurally
unavoidable**, which is the whole point.

### 6.5 The grounding / anti-hallucination contract (`grounding.py`, `resources.py`)

The contract is three concrete mechanisms, not a promise:

1. **Server-level instructions** (`FastMCP(instructions=GROUNDING_INSTRUCTIONS)`) state the invariant
   to the model up front: *"steppe tools return real, GPU-computed, ADMIXTOOLS-2-parity-validated
   statistics. Never fabricate, estimate, or 'recall' an f-statistic, weight, p-value, or date — if
   you need a number, call the tool. Resolve population names with `pops_validate` before computing.
   Every result carries a `provenance` block; report it."*
2. **The validation record as a readable resource** — `steppe://validation/parity` returns the
   parity-vs-AT2 evidence (the golden-gated tests, the reproduced-study numbers, the precision policy)
   so the model can *read why* the numbers are defensible and tell the user. This is the plan's
   "validation record is a readable resource so the model knows — and tells the user — the numbers are
   defensible, not invented."
3. **The provenance stamp on every result** (§6.4) — params + data-hash + version + precision, so any
   claim is traceable back to a specific f2 and a specific command.

```python
def register_resources(mcp, backend):
    @mcp.resource("steppe://validation/parity")
    def validation_record() -> str:
        """The anti-hallucination evidence: parity vs ADMIXTOOLS 2 (golden-gated), the reproduced
        studies, and the precision policy. The model reads this to ground its trust claim."""
        return grounding.VALIDATION_RECORD

    @mcp.resource("steppe://pops/catalog")
    def pops_catalog() -> str: ...                 # Area-4 catalog summary (browsable)

    @mcp.resource("steppe://pops/{group_id}")      # resource template — a data card per pop
    def pops_card(group_id: str) -> dict:          # steppe.pops.card(group_id): n, coverage, date BP, region…
        ...

    @mcp.resource("steppe://runs/{run_id}")        # an Area-5 run.json, addressable
    def run_manifest(run_id: str) -> dict: ...

    @mcp.resource("steppe://f2/{cache_key}")       # metadata about a resident/cached f2
    def f2_info(cache_key: str) -> dict: ...        # pops, P, n_block, device, precision
```

### 6.6 The prompts + the server + the transport (`prompts.py`, `server.py`)

```python
def register_prompts(mcp):
    @mcp.prompt
    def is_admixed(target: str, candidate_sources: list[str], f2_ref: str) -> list["Message"]:
        """The `steppe ask is-admixed` recipe as a prompt: screen with f3/D → rotate over the
        candidate sources → rank feasible models → summarize with provenance. Executes via the tools."""
    # + reproduce_study(paper, f2_ref) and build_graph(pops, f2_ref) — the other Area-5 recipes.

# server.py
def build_server(cfg: McpConfig) -> "FastMCP":
    from fastmcp import FastMCP                     # guarded import (steppe[mcp])
    mcp = FastMCP("steppe", instructions=grounding.GROUNDING_INSTRUCTIONS)
    backend = make_backend(cfg)                     # InProcessBackend | DaemonBackend
    register_tools(mcp, backend)
    register_resources(mcp, backend)
    register_prompts(mcp)
    return mcp

def serve(cfg: McpConfig | None = None, **kw) -> None:
    mcp = build_server(resolve_config(cfg, **kw))
    if cfg.transport == "stdio":
        mcp.run()                                   # local clients (Claude Desktop, IDEs)
    else:
        mcp.run(transport="streamable-http", host=cfg.host, port=cfg.port)

def http_app(cfg: McpConfig):
    """The ASGI app Area 6 mounts into its Uvicorn (app.mount('/mcp', http_app(cfg)), passing the
    FastMCP lifespan through). Daemon-backed → the same GpuWorker, one CUDA context."""
    return build_server(cfg).http_app(path="/mcp", transport="streamable-http")
```

---

## 7. Packaging — the `steppe[mcp]` extra

Add to `pyproject.toml` `[project.optional-dependencies]` (base wheel stays **numpy-only** — the
`fastmcp` import is guarded, so `import steppe` never pulls it):

```toml
mcp = [
  "fastmcp>=2.3",       # FastMCP 2.x: tools/resources/prompts, stdio + streamable-HTTP, ASGI http_app, in-memory test Client
  "pydantic>=2",        # structured tool-output schemas + the Provenance stamp (shared with Area 5's manifest)
  "httpx>=0.27",        # DaemonBackend client → the Area-6 daemon (only exercised in daemon-backed mode)
]
# convenience roll-up (owned by whichever area lands last; kept consistent across the sibling docs):
all = ["steppe[app,viz,mcp]"]   # app=Area 6, viz=Area 2, mcp=Area 7
```

```toml
[project.scripts]
steppe-mcp = "steppe.mcp.__main__:main"   # the MCP launcher (guarded: errors clearly without [mcp])
```

Notes:
- **`fastmcp` transitively pulls the official `mcp` SDK** (it is a superset), so we get the reference
  protocol impl without declaring it separately.
- **No Node / no build step** — unlike Area 6's `web/` front-end, the MCP surface is pure Python; the
  wheel change is one extra + one console script.
- **`httpx` is only needed for `DaemonBackend`.** It is light and pure-Python, so it lives in the
  `mcp` extra; a purist could split a `steppe[mcp-daemon]` extra, but the cost is not worth the extra
  surface — the read-only in-process path simply never imports it at runtime.
- **`pandas` is *not* required.** The MCP schemas serialize from the result classes' *scalar/list*
  fields (`res.weight`, `res.p`, …), not their pandas accessors — so the surface works without the
  `pandas` extra, and only pulls it transitively if a tool chooses to emit a DataFrame-derived view.

---

## 8. Shared-infra dependencies (reference — do not re-scaffold)

Area 7 **owns no shared infra**; it is a pure consumer. Its edges:

- **(vi) the existing result classes / DataFrames — the facade.** Tools serialize straight from
  `QpAdmResult` / `F4Result` / `QpGraphResult` / `DatesResult` etc. (`bindings/steppe/__init__.py`).
  **No new result type, no new compute** — the hard invariant of this area.
- **(ii) the AADR pops catalog — `steppe.pops` (Area 4).** `tools/pops.py` and the `steppe://pops/*`
  resources are thin proxies over `pops.search`/`pops.card`/`pops.validate`/`pops.suggest`; the
  fuzzy-name grounding ("turn 'Beaker' into the right Group IDs before computing") is Area-4's scorer,
  surfaced as a tool. **Hard dependency** — the pops surface is half of the value.
- **(iii) the content-addressed f2 cache — Area 5.** `InProcessBackend.load_f2` resolves an `f2_ref`
  cache key through the Area-5 `F2Cache`; `extract_f2` returns a cache key. **This is what makes the
  stateless tool loop work** (§2, §6.1) — without it, `f2_ref` degrades to raw dir paths only.
- **(v) the provenance manifest `run.json` — Area 5.** `provenance.stamp` wraps
  `workflow.provenance.capture(...)`; the `steppe://runs/{id}` resource serves saved manifests. The
  MCP layer **does not define a provenance format** — it embeds Area 5's.
- **(iv) the `steppe serve` daemon — Area 6.** `DaemonBackend` routes to `/v1/fit` etc.; the
  streamable-HTTP MCP app **mounts into the Area-6 Uvicorn** sharing the `GpuWorker` +
  `HandleRegistry` (Area 6 §3.1, §8 already reserve this). The full agentic loop's sub-200 ms speed is
  the daemon's; the MCP layer just calls it. **Performance dependency, not correctness** — the
  in-process backend works standalone.
- **(i) the CLI output seam — Area 1.** *Minimal* — the MCP layer reuses the facade, not the C++
  emitters. It **borrows Area-1's model-line sentence** (`to_qpadm_result.model_line`, §6.3) for the
  human-readable verdict; that formatter should live in a shared Python helper Area 1 also targets, or
  be re-derived from the same `QpAdmResult` fields.

**Area 6's stability service** (`serve/services/stability.py`, the bootstrap-halo weight ribbon) is the
source of `backend.stability(...)` — the plan's "safety flow-through" (§2, Area 7). Standalone
(no-daemon) the halo can be computed in-process by resampling SNP blocks over the facade, or omitted
(`stability=None`) until Area 6 lands it.

---

## 9. Intra-area build order

1. **`schemas.py` + `provenance.py`** — the pydantic tool-return models + the `Provenance` stamp
   (wrapping Area-5 `capture`). The stamp is the anchor every tool must carry; define it first so no
   tool ships without it.
2. **`backend.py` `InProcessBackend` + `context.resolve_f2`** — the facade composition + `f2_ref`
   resolution over the Area-5 cache. This is the whole read-only compute path.
3. **`tools/pops.py` + `tools/qpadm.py` (single `qpadm` fit) + the `steppe://validation/parity` +
   `steppe://pops/*` resources + `grounding.py` instructions** — **the plan's near-term "read-only
   surface: search + a single fit"** (plan §3), stdio, standalone. The smallest thing that proves the
   anti-hallucination contract end-to-end. Test with FastMCP's in-memory `Client`.
4. **The rest of `tools/*`** — `qpwave`, `qpadm_rotate`, `fstats` (f4/f3/f4ratio/qpdstat/dstat),
   `graph` (qpgraph/qpgraph_search), `dates`, and `extract_f2`/`qpfstats` (the cache-key producers) —
   completing the 1:1 facade surface.
5. **`prompts.py`** — the `steppe ask` recipes as MCP prompts (gated on Area-5 recipes landing; until
   then, hand-write the is-admixed / reproduce-study / build-graph templates).
6. **`steppe://runs/{id}` + `steppe://f2/{key}` resources** — the provenance + resident-f2 resources
   (gated on Area-5 manifests / the cache being addressable).
7. **`DaemonBackend` + streamable-HTTP transport + the Area-6 mount** — the daemon-backed agentic loop
   (gated on the Area-6 daemon; §8). The safety-flow-through halo wires in here (`backend.stability`).

Sequencing logic: steps 1–3 are the **cheap, high-value read-only surface the plan says to ship first**
— a complete, grounded, standalone MCP server with catalog search + one real fit. Steps 4–6 complete
the surface. Step 7 is the resident agentic loop, dependency-gated on Area 6. **No step requires a
compute change** — the entire area is Python over the facade.

---

## 10. Open questions

1. **stdio-in-process CUDA context lifetime.** A stdio server spawned by Claude Desktop holds one CUDA
   context for the client's session; a cold first fit pays the ~0.9 s load. Acceptable for read-only
   (one fit), but does the stdio server pre-warm a default `f2_ref` at startup (fast first fit, VRAM
   held idle) or lazy-load (slow first fit, no idle VRAM)? Likely a `McpConfig.preload` flag —
   **needs a measurement on real AADR** (memory *real-data-only-all-results*).
2. **`f2_ref` ergonomics for the model.** A cache key (`sha256:…`) is opaque to an LLM; an f2-dir path
   is guessable but leaks the filesystem. Do we expose a **named-alias** layer (`f2_ref="v66-HO"` →
   a configured cache key) so the model traffics in human names, with `steppe://f2/{key}` resolving
   the metadata? Improves the agent UX materially; needs a small alias registry in `McpConfig`.
3. **Which backend is default.** In-process (self-contained, no daemon, cold-ish) vs daemon (fast,
   resident, requires `steppe-serve` running). Proposed: **auto-detect** — use `DaemonBackend` if a
   daemon is reachable at the configured URL, else fall back to `InProcessBackend` — so the same
   `steppe-mcp` config works with or without Area 6. Confirm the fallback is silent-with-a-log, not an
   error.
4. **Auth / exposure for streamable-HTTP.** The `2025-06-18` spec standardized OAuth 2.1 and FastMCP
   ships first-class OAuth helpers, but for a single-box lab daemon that is heavy. Local stdio needs
   none; a mounted remote surface likely wants at least the Area-6 bearer-token + CORS allowlist.
   How much auth is in first-cut scope vs "run it behind the SSH tunnel Area 6 assumes"?
5. **The sweep facade gap.** The f4/D **all-quartets sweep is CLI-only** (no Python facade — the same
   gap Areas 3/6 flag). So a `sweep`/`neighbors` tool for the Relationship-Explorer-style agentic
   query can't be built until that facade exists. Out of scope for Area 7; flagged as an upstream
   prerequisite. Until then the neighborhood query degrades to explicit `f3`/`f4` tool calls.
6. **Elicitation / sampling.** The MCP spec supports server→client **elicitation** (ask the user to
   confirm) and **sampling** (the server asks the client's model to reason). Could power a "this model
   is infeasible — try dropping a source?" confirm-loop, or let a tool ask the client to disambiguate
   a pop name. Genuinely useful for the agentic UX but adds protocol surface; parked for v1 (the
   grounding contract + structured returns are the priority) and noted so the design doesn't foreclose
   it.
7. **Should the `model_line` formatter be shared with Area 1?** The "qpAdm as a sentence" verdict is
   wanted by both the C++ CLI (Area 1) and this Python tool result. C++ and Python can't share one
   implementation cheaply; either it is re-derived from `QpAdmResult` fields in Python (small
   duplication) or a single Python formatter becomes the source and Area 1's C++ version mirrors it.
   Decide the ownership so the sentence reads identically in the terminal and in the chat.
```

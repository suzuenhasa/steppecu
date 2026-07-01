# Area 6 — the `steppe serve` daemon + STUDIO — Engineering Architecture

> **What this document is.** `docs/EXPERIENCE-PLAN.md` is the **product** plan — the *what/why* of
> the North Star: STUDIO's qpAdm-as-you-type workbench, the Relationship Explorer, the live
> Admixture-Graph Builder, the AADR Atlas, and `steppe share` (plan §2, "Area 6"). **This** document
> is the **engineering architecture** — the *how*: the process/concurrency model of the daemon, the
> tech-stack decisions (weighed against alternatives), the folder scaffolding, the exact interface
> signatures, the packaging extra, and the dependency edges to the rest of the codebase. It does not
> restate the plan's feature tables; it references them and designs the machine underneath.

---

## 1. Area role — the linchpin

Area 6 owns **shared infra (iv): the `steppe serve` daemon** — a long-lived process that holds the
f2 tensor + cached `Resources` in VRAM and exposes a thin local websocket/HTTP API over the existing
Python facade. The plan states the thesis exactly (§2, Area 6): *cold `qpadm` is ~0.9 s but the GPU
compute is sub-200 ms; the daemon pays the load once so every interaction is the sub-200 ms hit.*
Without a resident process, "interactive" is a lie — every fit re-reads the f2-dir from disk and
re-enumerates the device.

This area is **the linchpin every live feature rides on**: Area 3's marimo explorers, Area 7's
MCP agent surface, and Area 6's own browser front-ends (STUDIO / Relationship Explorer / Graph
Builder / Atlas) all obtain their sub-second loop by talking to *this* daemon. It is therefore the
highest-leverage post-0.1.0 investment (plan §3 "Big bets"; §4 "explicitly deferred to post-1.0").

The five front-end features it must serve are enumerated in the plan (§2, Area 6 table) and are **not
restated here**. Architecturally they reduce to four server capabilities: (a) a **debounced live
fit** (STUDIO + Atlas rotation), (b) a **neighborhood sweep** centered on a pop (Relationship
Explorer), (c) an **incremental graph re-score** (Graph Builder), and (d) a **session snapshot →
reproducible artifact** (`steppe share`). Everything else is presentation on the browser side.

---

## 2. The resident-state model this rides on (the critical grounding)

The whole area lives or dies on one already-built mechanism, so it is worth stating precisely.

`bindings/internal/bind_common.hpp` defines the opaque `F2Handle` (lines 73–83): it holds the **host**
`F2BlockTensor` (FP64), the P pop labels, the target device ordinal, and a lazily-built,
**cached `std::unique_ptr<Resources>`** (device streams + cuBLAS/cuSOLVER/cuFFT handles). `read_f2()`
in `bind_f2handle.cpp` loads `f2.bin`+`pops.txt` into it; every fit entry
(`bind_qpadm.cpp::run_qpadm_py`, etc.) drives the fit through `with_device_f2(h, run)`
(`bind_common.hpp:203–214`), which:

1. `ensure_resources(h)` — **builds `Resources` once**, then reuses the cached pointer on all
   subsequent calls (the ADR-0005 "precompute-once / fit-many" path, commented at `bind_common.hpp:70`);
2. **re-uploads the host tensor to the GPU inside every call** (`upload_f2_blocks_to_device`,
   line 209) — the `DeviceF2Blocks` is created and freed within the call and never crosses into Python
   (the deliberate M(py-1) "spike #1 / VRAM ownership" constraint, `bind_common.hpp:15–18`).

**What the daemon gets for free:** holding a live Python `steppe.F2Blocks` object keeps the host
tensor and the cached `Resources` resident, so the daemon amortizes the disk read *and* the device
enumeration / handle construction — the dominant slice of the ~0.9 s cold cost. Every subsequent
`qpadm(handle, ...)` reuses them.

**What the daemon still pays, and the one facade change it motivates:** the host→device **re-upload
of the f2 tensor on every fit** (spike #1 keeps `DeviceF2Blocks` call-scoped). For the tiny golden
that is nanoseconds; for a production model (P≈300–500, n_block≈700 → a 0.7–1.4 GB FP64 tensor) it is
tens of ms of PCIe traffic *per fit* — which for a slider dragged at 30 fps is the difference between
"live" and "stutters". The clean architectural answer (consistent with memory
*m45-d2h-incidental-fused-fit*: "keep f2 device-resident, the real fix is the fit engine reading
f2_blocks from VRAM") is a **resident-`DeviceF2Blocks` session handle**: a small, additive binding
that lifts spike #1 *only for the daemon path* — upload once, pin in VRAM, fit many. This is the
single piece of **new compiled/binding surface** Area 6 needs; everything else is Python over the
existing facade. It is scoped in §6.1 and flagged as a dependency in §8. **Design note:** this handle
is created and owned by the daemon's single GPU worker (§3.2) so the pinned `DeviceF2Blocks` never
races a second CUDA stream.

The facade functions the daemon composes are already exactly the shape the front-ends want:
`read_f2`, `qpadm`, `qpwave`, `qpadm_search` (batched rotation → `as_dataframe`), `f3`/`f4`
(neighborhood affinities), `qpgraph`/`qpgraph_search` (graph builder), `dates`, and the result
classes with their tidy DataFrames (`bindings/steppe/__init__.py`). No compute is rebuilt.

---

## 3. Tech-stack decision

### 3.1 The daemon web framework — **FastAPI + Uvicorn + the `websockets` transport**

| Candidate | Verdict | Why |
|---|---|---|
| **FastAPI (on Starlette/Uvicorn)** | **CHOSEN** | The lingua franca of Python GPU-serving; async endpoints + native `WebSocket` + dependency-injection for the session/handle registry; first-class lifespan hooks to build the resident handle at startup. Decisive interop: **Area 7's MCP server is FastMCP, which is itself Starlette/ASGI** — mounting the MCP app and the STUDIO websocket API under **one Uvicorn process sharing one GPU worker** is a sub-app mount, not a second daemon. |
| Litestar 2.x | Rejected (for now) | Genuinely faster (msgspec-native, ~40–120% higher throughput in migration reports) and its DTO/msgpack story is excellent. But **our bottleneck is the GPU round-trip, not framework serialization** — a 200 ms fit dwarfs a sub-ms encode. FastAPI's ubiquity, the FastMCP/Starlette alignment for Area 7, and the larger example corpus for "FastAPI + GPU model server" outweigh raw RPS here. (Revisit if the sweep-streaming firehose ever becomes serialization-bound.) |
| Raw Starlette / bare `websockets` | Rejected | We'd hand-roll routing, validation, and OpenAPI that FastAPI gives free; no upside at our scale. |
| Flask / Django Channels | Rejected | Sync-first (Flask) or far too heavy (Django) for a single-box local daemon. |

Uvicorn is the ASGI server; the `websockets` library backs the WS transport (preferred over `wsproto`
for throughput). **Single Uvicorn worker, single process** — this is a *local, single-GPU* daemon, not
a horizontally-scaled service; one CUDA context lives in one process (memory *multi-gpu-parked*:
default `--device 0`). Concurrency is handled *inside* the process (§3.2), not by forking workers.

### 3.2 GPU concurrency — an **async endpoint + a single-worker GPU executor with a latest-wins coalescing queue**

The hard constraint: a fit is a **blocking, GIL-holding** nanobind call against **one CUDA context**.
Two rules follow, and the research confirms both (FastAPI async docs; "single-GPU → serialize with a
semaphore/single worker"):

1. **Never block the event loop.** Endpoints are `async`; the actual fit is dispatched to a
   **single-thread `ThreadPoolExecutor` (`max_workers=1`)** via `loop.run_in_executor` /
   `anyio.to_thread` with a bound capacity limiter. One worker = the CUDA context is touched by exactly
   one thread, so no stream/context race; the event loop stays free to accept new WS frames while a
   fit runs.
2. **Coalesce, don't queue.** A slider dragged across 40 values must not enqueue 40 fits. The GPU
   worker is fronted by a **per-session latest-wins mailbox** (`asyncio.Queue(maxsize=1)` semantics):
   a newer request for a session *supersedes* the pending one; the in-flight one is allowed to finish
   (CUDA has no cheap safe cancel) but its result is dropped if a newer request has landed. This is
   what makes the "drag and watch it re-fit" feel instant instead of laggy — it is the debounce the
   plan's Area-3 grounding calls "sliders fire on mouse-up", pushed server-side so *every* client gets
   it.

This gives fair, deterministic GPU access, natural back-pressure, and the debounce the live UX needs —
without subprocess-per-request (rejected: it would defeat the resident-VRAM premise, the entire point
of the daemon).

### 3.3 Wire serialization — **`msgspec` (msgpack frames for data, JSON for control), `orjson` fallback**

Control messages (subscribe, set-model, ping) are small and human-debuggable → JSON. Result payloads
carry numeric arrays (weight vectors, SE, per-quartet Z, sweep survivors, residual matrices) → **msgpack
via `msgspec`**, which the research puts at ~9× JSON / ~4× orjson on encode and has **`extras=[numpy]`
zero-copy typed-array support** — exactly our residual-matrix / sweep-vector shapes. `msgspec.Struct`
schemas double as the typed protocol contract (§6.4). A single decoder/encoder is instantiated at
startup (the "-20% latency" tip). Plain `orjson` is the dependency-light fallback if `msgspec` is
undesirable, but `msgspec` is chosen because it is *also* Litestar's engine and the natural Struct
layer for the protocol types.

### 3.4 The front-end — **React 19 + TypeScript + Vite**, with purpose-built viz libs

The plan lists "React/Svelte/SolidJS + a viz lib". Weighed:

| Candidate | Verdict | Why |
|---|---|---|
| **React 19 + Vite + TS** | **CHOSEN** | The viz ecosystem we need is **React-first**: `@xyflow/react` (**React Flow**) is the mature node-editor for the Graph Builder; **deck.gl**'s canonical bindings are React (`@deck.gl/react`) for the Atlas. Our runtime hot-path is the **GPU round-trip, not DOM diffing**, so SolidJS/Svelte's fine-grained-reactivity edge is irrelevant to felt latency; React 19's compiler already closes most of the historical perf gap. Biggest ecosystem, hiring pool, and example corpus — correct for a portfolio-grade, maintainable front-end. |
| SolidJS | Rejected | Fastest runtime, but the win is on DOM-heavy churn we don't have (updates arrive at fit cadence, ~5–30 Hz, already debounced server-side). Would force wrapping React Flow / deck.gl. |
| Svelte 5 (runes) | Close 2nd | Best DX and a real **Svelte Flow** exists; genuinely viable. Loses only on deck.gl being React-native and on the smaller node-editor/geo corpus. Documented as the fallback. |

Per-surface viz libraries (all MIT, all lazy-loaded so a bare STUDIO page ships light):

- **Graph Builder** → **React Flow (`@xyflow/react`)** — drag nodes, draw drift/admixture edges,
  recolor edges by residual-Z; the "Figma-style canvas" the plan describes, out-of-the-box.
- **AADR Atlas** → **deck.gl `ScatterplotLayer`/`ArcLayer` over MapLibre GL JS** (open-source basemap,
  no Mapbox token). GPU-filtered points + a time-scrub slider; lasso via `@deck.gl` `PolygonLayer` +
  a selection callback → the selected pop set POSTs back as a rotation source pool.
- **Relationship Explorer** → **sigma.js + graphology** (WebGL force-graph) — scales to thousands of
  affinity nodes where an SVG/D3 force sim would choke; `graphology` supplies layout + centrality.
- **Live charts** (weight bar, p-gauge, decay, `|Z|` histogram) → **uPlot** (2 kB, canvas, built for
  streaming re-render) for the hot live panels; **Observable Plot** for the richer static insets.
- **State/transport** → **TanStack Query** for REST reads (pops catalog, session list) + a thin
  **native `WebSocket`** hook decoding `msgspec` frames for the live channel.

Build tooling: **Vite + pnpm + TypeScript**, output a static bundle. **Hard constraint (memory
`cpu-is-test-only`, plan §3):** steppe is GPU-only — **no WASM/pyodide real fits**; the browser is a
*thin renderer*, every statistic comes from the daemon on the CUDA box. The SPA is static assets;
all compute is server-side.

---

## 4. The fit decision — **hybrid**

Three distinct artifacts, each placed where it belongs:

1. **New Python subpackage `bindings/steppe/serve/`** — the daemon: FastAPI app, the GPU worker,
   the session registry, the WS/REST protocol. Pure Python over the existing facade (+ the one
   resident-handle binding, §6.1). Ships *inside* the wheel but imports its heavy deps only under the
   `steppe[app]` extra (guarded imports; `import steppe` stays numpy-only).
2. **Separate front-end at repo-root `web/`** — the Vite/React SPA. **Not** a Python package; its own
   `package.json`/`pnpm-lock.yaml`, its own CI. Its built static assets are *optionally* embedded into
   the wheel (§7) so `steppe serve` can serve STUDIO with zero extra install, but the source tree is a
   separate app.
3. **A launch entry point** — because the CLI is **pure C++ and the wheel does not build it**
   (`STEPPE_BUILD_CLI=OFF`, `pyproject.toml`), there is no `steppe` binary on PATH from `pip install`.
   The daemon therefore ships as a **`[project.scripts]` console entry `steppe-serve =
   "steppe.serve.__main__:main"`** (and `python -m steppe.serve`). We deliberately do **not** try to
   make `steppe serve` a C++ subcommand shelling out to Python — that couples the two build systems for
   no benefit. `steppe-serve` is the documented invocation; if/when the C++ CLI is later shipped in the
   wheel, a `serve` subcommand can `exec` the console script.

**Why not extend an existing seam.** Area 1's C++ `emit_to_destination` seam is byte-stream file/stdout
output — the wrong abstraction for a stateful, bidirectional, long-lived socket. The nanobind facade is
*consumed* by the daemon but is a stateless function library, not a server. A daemon is a new
responsibility → a new subpackage, cleanly dependency-gated. This matches the plan's own framing:
"new subpackage (`bindings/steppe/serve`) + a separate web/ front-end + `steppe[app]` extra."

---

## 5. Folder scaffolding

```
bindings/steppe/
  __init__.py                      # (exists) the facade; add steppe.serve re-export guarded
  _resident.py                     # NEW — thin py wrapper over the resident-DeviceF2Blocks handle (§6.1)
  serve/                           # NEW subpackage — the daemon (heavy deps → steppe[app])
    __init__.py                    #   public: serve(), ServeConfig; guarded "install steppe[app]" error
    __main__.py                    #   `python -m steppe.serve` / console_script main(): argparse → serve()
    app.py                         #   build_app(cfg) -> FastAPI; lifespan builds worker+registry; mounts routers + static
    config.py                      #   ServeConfig (msgspec.Struct): host, port, f2_dir|cache_key, device, cors, token
    worker.py                      #   GpuWorker: single-thread executor owning the CUDA context (§3.2)
    queue.py                       #   SessionMailbox: per-session latest-wins coalescing (§3.2)
    registry.py                    #   HandleRegistry: content-addressed resident F2Blocks handles (Area-5 cache keys)
    session.py                     #   Session: current target/left/right/params + provenance accumulator
    protocol.py                    #   msgspec.Struct wire types: ClientMsg/ServerMsg unions (§6.4)
    routes/
      __init__.py
      ws.py                        #   /ws/{session_id}: the live channel (fit / sweep / graph re-score / stream)
      fit.py                       #   POST /v1/fit, /v1/rotate  (one-shot REST, non-live clients incl. MCP)
      pops.py                      #   GET  /v1/pops?...        (thin proxy over steppe.pops — Area 4)
      graph.py                     #   POST /v1/graph/score, /v1/graph/suggest  (Graph Builder)
      sweep.py                     #   WS   /ws/sweep/{id}      (Relationship Explorer / Atlas streaming survivors)
      share.py                     #   POST /v1/share -> run.json + script + figure bundle (Area-5 provenance)
      health.py                    #   GET  /v1/health, /v1/session   (doctor: device, VRAM, resident handles)
    services/
      fit_service.py               #   compose steppe.qpadm/qpwave over the resident handle; build result payloads
      sweep_service.py             #   steppe.f3/f4 neighborhood + (future) sweep-facade streaming
      graph_service.py             #   steppe.qpgraph/qpgraph_search incremental re-score
      stability.py                 #   bootstrap-halo: SNP-block resample → weight ribbons (plan §2 guardrail)
    static/                        #   (build-time) the compiled web/ bundle is copied here for wheel embedding
    py.typed

web/                               # NEW separate front-end app (NOT a python package)
  package.json  pnpm-lock.yaml  vite.config.ts  tsconfig.json  index.html
  src/
    main.tsx  App.tsx
    api/            ws.ts  rest.ts  msgpack.ts  types.ts     # generated from serve/protocol.py schema
    state/          session.ts  store.ts
    studio/         Studio.tsx  TargetBox.tsx  SourceChips.tsx  WeightBar.tsx  PGauge.tsx  ResidualTable.tsx  RotationSlider.tsx
    explorer/       Relationship.tsx  AffinityGraph.tsx        # sigma.js
    graph/          GraphBuilder.tsx  EdgeInspector.tsx        # React Flow
    atlas/          Atlas.tsx  TimeScrub.tsx  LassoLayer.tsx    # deck.gl + MapLibre
    share/          ShareDialog.tsx
    components/     charts/ (uPlot wrappers)  StabilityHalo.tsx
  public/           maplibre style json, fonts

docs/ux/
  area6-daemon-architecture.md     # THIS doc

tests/python/serve/                # NEW — daemon tests (httpx.AsyncClient + starlette TestClient WS)
  test_worker_serialization.py  test_session_coalesce.py  test_ws_protocol.py  test_fit_route.py
```

---

## 6. Key interfaces

### 6.1 The resident-`DeviceF2Blocks` handle — the one new binding (motivated §2)

A small additive nanobind entry + Python wrapper. It lifts spike #1 *only* for the daemon: the
`DeviceF2Blocks` is uploaded once and pinned; fits take the resident device blocks directly.

```cpp
// bindings/bind_qpadm.cpp (additive) — a resident session variant.
// Uploads h.tensor to VRAM ONCE and stashes DeviceF2Blocks on the F2Handle (or a new
// ResidentF2 object); subsequent fits reuse it. Owned/driven by the daemon's single GPU
// worker so the pinned device buffer never races a 2nd stream.
struct ResidentF2 { /* holds DeviceF2Blocks + &Resources; move-only, RAII frees VRAM */ };
ResidentF2* pin_device_f2(F2Handle& h);                       // upload-once
nb::dict    run_qpadm_resident(ResidentF2&, target, left, right, /*opts*/…);
nb::list    run_qpadm_search_resident(ResidentF2&, target, lefts, right, /*opts*/…);
```

```python
# bindings/steppe/_resident.py
class ResidentF2:
    """A pinned, upload-once device f2 handle for the daemon's sub-200ms loop.
    Created from an F2Blocks; must be used on the thread that created it (one CUDA context)."""
    @classmethod
    def pin(cls, f2: "steppe.F2Blocks") -> "ResidentF2": ...
    def qpadm(self, *, target, left, right, **opts) -> "steppe.QpAdmResult": ...
    def qpadm_search(self, *, target, models, right, **opts) -> list["steppe.QpAdmResult"]: ...
    def close(self) -> None: ...          # frees VRAM
```

> If shipping this binding change slips, the daemon still works using the plain
> `steppe.qpadm(f2, ...)` path (resources cached, tensor re-uploaded per call) — correct, just not
> optimally sub-200 ms on large models. So this is a *performance* dependency, not a *correctness* one.

### 6.2 The GPU worker + coalescing queue (`serve/worker.py`, `serve/queue.py`)

```python
class GpuJob(Protocol):
    session_id: str
    seq: int                              # monotonic; a higher seq for a session supersedes lower
    def run(self, res: "ResidentF2") -> "ServerMsg": ...

class GpuWorker:
    """Single-thread executor owning the CUDA context. All device calls funnel here."""
    def __init__(self, device: int): ...
    async def submit(self, job: GpuJob) -> "ServerMsg": ...   # run_in_executor(max_workers=1)
    async def aclose(self) -> None: ...

class SessionMailbox:
    """Per-session latest-wins: enqueue(job) drops any pending job for the same session_id
    so a dragged slider fits only the newest value. In-flight jobs finish; stale results dropped."""
    def enqueue(self, job: GpuJob) -> None: ...
    async def drain(self, worker: GpuWorker) -> AsyncIterator["ServerMsg"]: ...
```

### 6.3 The handle registry (`serve/registry.py`) — Area-5 cache keys

```python
class HandleRegistry:
    """Content-addressed resident handles. Keyed on the Area-5 f2-cache key
    (genotype hash + pop set + extraction params). Loads once, shares across sessions/clients."""
    def get_or_load(self, key: CacheKey) -> ResidentF2: ...   # read_f2 -> ResidentF2.pin (on the worker thread)
    def evict(self, key: CacheKey) -> None: ...
    def list(self) -> list[HandleInfo]: ...                   # for /v1/health (VRAM budget)
```

### 6.4 The wire protocol (`serve/protocol.py`) — `msgspec.Struct` unions

```python
# client -> server
class SetModel(Struct, tag="set_model"):   target: str; left: list[str]; right: list[str]; params: FitParams
class SetRight(Struct, tag="set_right"):    add: list[str] = []; remove: list[str] = []   # drag an outgroup
class RotateSweep(Struct, tag="rotate"):    pool: list[str]; min_size: int; max_size: int
class NeighborsOf(Struct, tag="neighbors"): pop: str; min_abs_z: float
class GraphEdit(Struct, tag="graph_edit"):  op: Literal["add_edge","del_edge","add_admix"]; a: str; b: str
ClientMsg = Union[SetModel, SetRight, RotateSweep, NeighborsOf, GraphEdit, Ping]

# server -> client   (msgpack; numeric arrays zero-copy via msgspec extras=[numpy])
class FitResult(Struct, tag="fit"):   seq:int; target:str; left:list[str]; weight:list[float]; se:list[float]
                                       # + p, chisq, dof, f4rank, feasible, status, precision, wall_ms
class Stability(Struct, tag="halo"):  seq:int; weight_lo:list[float]; weight_hi:list[float]   # bootstrap ribbon
class SweepTick(Struct, tag="sweep"): survivors: list[Quartet]; done: bool                     # streamed jsonl-over-ws
class GraphScore(Struct, tag="score"): score:float; worst_residual_z:float; edges:list[EdgeZ]
class Progress(Struct, tag="progress"): frac:float; rate:float; eta_s:float
class Err(Struct, tag="error"):       code:str; message:str; did_you_mean:list[str] = []       # Area-4 fuzzy
ServerMsg = Union[FitResult, Stability, SweepTick, GraphScore, Progress, Err]
```

### 6.5 The FastAPI surface (`serve/app.py`, `serve/routes/*`)

```python
def build_app(cfg: ServeConfig) -> "FastAPI":
    """Assemble the daemon. lifespan: build GpuWorker(cfg.device) + HandleRegistry, preload
    cfg.f2_dir|cache_key into a resident handle, mount routers + the static web/ bundle, and
    (optionally) mount the Area-7 FastMCP ASGI sub-app sharing the same worker."""

# WS   /ws/{session_id}            live channel: ClientMsg in -> ServerMsg stream out (STUDIO/Atlas)
# WS   /ws/sweep/{session_id}      streamed sweep survivors (Relationship Explorer)
# POST /v1/fit        {model} -> FitResult                (one-shot; MCP + non-live clients)
# POST /v1/rotate     {pool}  -> [FitResult]              (rotation leaderboard)
# GET  /v1/pops?q=&region=&min_n=  -> catalog rows        (proxy → steppe.pops, Area 4)
# POST /v1/graph/score  {edges} -> GraphScore
# POST /v1/graph/suggest {edges} -> [ghost edges]         (bounded topology search preview)
# POST /v1/share      {session} -> {run_json, script, figure_url}   (Area-5 provenance + Area-2 figure)
# GET  /v1/health     -> {device, vram_used, vram_total, handles:[…], version}   (doctor)

def serve(cfg: ServeConfig | None = None, **kw) -> None:
    """Public entry: uvicorn.run(build_app(resolve_config(cfg, **kw)), host, port). Called by
    steppe.serve.__main__:main and the `steppe-serve` console script."""
```

---

## 7. Packaging — the `steppe[app]` extra

Add to `pyproject.toml` `[project.optional-dependencies]` (base wheel stays **numpy-only** —
guarded imports mean `import steppe` never pulls these):

```toml
app = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",   # includes the `websockets` + httptools speedups
  "websockets>=12",
  "msgspec>=0.18",             # msgpack + numpy zero-copy wire types
  "anyio>=4",                  # to_thread capacity limiter for the GPU executor
  "pandas>=1.5",               # result DataFrames the services shape
]
# convenience roll-up once the sibling areas land their extras:
all = ["steppe[app,viz,mcp]"]   # viz=Area 2, mcp=Area 7
```

```toml
[project.scripts]
steppe-serve = "steppe.serve.__main__:main"   # the daemon launcher (guarded: errors w/o [app])
```

**Front-end bundling decision.** `web/` builds a static bundle via Vite. Two ship modes, both
supported: (1) **embedded** — a CMake/scikit-build step copies `web/dist` → `bindings/steppe/serve/static`
so `wheel.packages` picks it up and `steppe serve` serves STUDIO with zero extra install (the default
for a release wheel; keeps the "pip install and go" story); (2) **decoupled** — for dev, `steppe-serve
--dev` proxies to the Vite dev server (`web/` running `pnpm dev`) for HMR. The bundle is built in CI,
**not** on the user's machine (no Node required to `pip install steppe[app]`). If bundle size becomes a
concern, embedding can move behind a `steppe[studio]` extra that fetches a prebuilt asset tarball.

---

## 8. Shared-infra dependencies (reference — do not re-scaffold)

Area 6 **owns** shared infra **(iv) the daemon**. It **depends on**:

- **(iii) the content-addressed f2 cache — Area 5.** The `HandleRegistry` keys resident handles on the
  Area-5 cache key `(genotype hash + pop set + extraction params)` so multiple sessions/clients share
  one VRAM-resident f2. *The daemon is the primary consumer of the cache; the cache is the enabler
  under the daemon.* If the cache isn't ready, the registry degrades to keying on the raw f2-dir path.
- **(ii) the AADR pops catalog — `steppe.pops` (Area 4).** `routes/pops.py` is a thin proxy; the "did
  you mean" in `Err.did_you_mean` reuses Area 4's fuzzy scorer. STUDIO's target/source pickers and the
  Atlas's lat/long + date come from the catalog.
- **(v) the provenance manifest `run.json` — Area 5.** `routes/share.py` assembles `run.json` from the
  session's accumulated params + content hashes; `steppe share` = provenance bundle + an Area-2 figure.
- **(vi) the existing result classes / DataFrames — the facade.** `services/*` build wire payloads
  straight from `QpAdmResult` / `F4Result` / `QpGraphResult` etc. (§2). No new result types.
- **(i) the CLI output seam — Area 1.** *Minimal* — the daemon reuses the facade, not the C++ emitters;
  it only shares Area 1's `isatty`/`--json` convention for its own startup banner on stderr.

Areas that **ride on this daemon (downstream):**

- **Area 3 (marimo)** — the explorers can talk to `/ws` for the resident loop instead of holding their
  own `read_f2` (or fall back to in-process for a standalone notebook).
- **Area 7 (MCP)** — "Daemon-backed resident f2" (plan §2, Area 7): the FastMCP app **mounts into this
  same Uvicorn process** (§3.1) and submits jobs to the **same `GpuWorker`**, so every tool call hits
  the sub-200 ms path with no second CUDA context.
- **Area 2 (viz)** — `routes/share.py` calls the Area-2 figure renderer for the shareable artifact.

---

## 9. Intra-area build order

1. **`ResidentF2` binding + `_resident.py`** (§6.1) — the perf enabler; validate upload-once vs
   per-call on a real AADR handle (memory *real-data-only-all-results*: measure the sub-200 ms claim on
   real AADR, not the tiny golden). *Correctness-decoupled: the daemon can start on the plain facade.*
2. **`GpuWorker` + `SessionMailbox`** (§6.2) — the concurrency core; unit-test serialization +
   latest-wins coalescing headless (no HTTP).
3. **`protocol.py` + `HandleRegistry` + `Session`** — the typed contract + resident-handle sharing.
4. **FastAPI skeleton + `/v1/health` + `/v1/fit`** (§6.5) — the one-shot REST path; this alone unblocks
   **Area 7's read-only MCP surface** (plan §3: "a read-only tool surface — search + a single fit — is
   near-term and cheap").
5. **`/ws/{session}` live channel + `fit_service` + `stability`** — STUDIO's core loop + bootstrap
   halos (the plan §2 guardrail: "candidate, not answer").
6. **`web/` STUDIO** (React + uPlot) against the live channel — the flagship.
7. **`sweep_service` + `/ws/sweep` + Relationship Explorer** (sigma.js) — *gated on the sweep Python
   facade, which does not exist yet* (plan §2 Area 3 gap).
8. **`graph_service` + Graph Builder** (React Flow) and **AADR Atlas** (deck.gl+MapLibre) — the two
   largest front-ends, last.
9. **`share.py`** — folds in Area-5 provenance + Area-2 figures once those land.

Sequencing logic: the **worker + resident handle** are the load-bearing core (steps 1–2); a **thin
REST fit** (step 4) is the smallest thing that proves the daemon and unblocks MCP; the **live WS +
STUDIO** (steps 5–6) are the flagship; the heavy explorers (7–8) come after, two of them dependency-
gated.

---

## 10. Open questions

1. **Ship the `ResidentF2` binding, or accept per-call re-upload for 0.1-of-Area-6?** §6.1 makes the
   daemon *correct* without it; the question is whether the measured large-model re-upload cost on real
   AADR (P≈300–500) actually breaches the sub-200 ms budget enough to justify lifting spike #1 now vs.
   after the "fit engine reads f2 from VRAM" work (memory *m45-d2h-incidental-fused-fit*). **Needs a
   measurement on real AADR before committing.**
2. **Auth / exposure.** Local-only (`127.0.0.1`) by default is safe and simple, but "one GPU box serves
   a whole lab" (plan §2, Area 3) implies LAN exposure → a bearer token + CORS allowlist in
   `ServeConfig`. How much auth is in-scope for the first cut vs. "run it behind an SSH tunnel"?
3. **Multi-session VRAM budgeting.** The registry can pin several resident handles (different f2 sets).
   What is the eviction policy when VRAM is tight — LRU by handle, or refuse-new with a clear error?
   `/v1/health` exposes the budget; the policy is undecided.
4. **Cancellation semantics.** CUDA has no cheap safe cancel; latest-wins *drops results* but the
   in-flight fit still burns the GPU. For a fast slider that's fine (fits are sub-200 ms); for a long
   rotation/sweep, do we need cooperative cancellation checkpoints in the core (a real core change) or
   is "let it finish, drop the payload" acceptable?
5. **Frontend framework re-vote.** React is chosen for the deck.gl/React-Flow ecosystem; **Svelte 5 +
   Svelte Flow** is a legitimate close second (§3.4). Lock this before `web/` scaffolding hardens — the
   choice is cheap now and expensive after 20 components exist.
6. **Sweep facade dependency.** The Relationship Explorer / Atlas streaming needs a Python surface for
   the f4/D sweep, which is **CLI-only today** (plan §2 Area 3 gap). Whose backlog builds it — a
   prerequisite for steps 7–8, and it belongs upstream of Area 6, not inside it.
7. **Real-time collaboration (plan §5 "wild").** Multiple clients on one session is a small step from
   the current per-session broadcast, but conflict resolution on the Graph Builder canvas (two people
   dragging edges) is a CRDT-shaped problem explicitly parked; note it so the session model doesn't
   foreclose it.
```

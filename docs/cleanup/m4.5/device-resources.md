# M4.5 unit review — `device/resources` (`Resources` / `PerGpuResources` + `build_resources`)

Files audited (read fully): `src/device/resources.hpp`, `src/device/resources.cpp`.
Directly-related context read line-by-line for cross-checking: `src/device/backend.hpp` (the `ComputeBackend`
interface + `BackendCapabilities` POD + the per-device-instance contract), `src/device/backend_factory.hpp`
(`make_cuda_backend`), `src/device/cuda/cuda_backend.cu` (the `CudaBackend` ctor lines 80-108, the
`capabilities()` probe lines 450-547, `set_and_return_device` 569-572, the member-decl block 574-599, and the
factory body 609-611), `include/steppe/config.hpp` (`DeviceConfig`, lines 198-311), the consumer
`src/core/fstats/f2_blocks_multigpu.{hpp,cpp}` (the entry point that **mutates** `Resources::last_combine_path`),
`src/device/p2p_combine.{hpp,cu}` (the P2P combine the duplicate-ordinal case would feed — cu lines 230-309),
`src/device/shard_plan.hpp`, `src/core/fstats/f2_combine.hpp`, `src/device/CMakeLists.txt`, and the gate test
`tests/reference/test_resources_build.cu`. Standard judged against: architecture.md §2/§4/§7/§9/§11.2/§11.4/§12/§13
and cleanup `00-overview.md` §(2).

Authoritative docs verified this pass (cited inline):
- **CUDA 12.0 made `cudaSetDevice` eagerly initialize the runtime + the device's primary context** — confirmed
  against NVIDIA's runtime docs (pre-12.0 it deferred init until the first runtime call; the `cudaFree(0)`
  warm-up workaround is no longer needed). This is the load-bearing fact under P1.
- **`cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice)` returns `cudaSuccess` or `cudaErrorInvalidDevice`,
  and the same-device pair (`device == peerDevice`) is invalid** (NVIDIA CUDA Runtime API §Peer Device Memory
  Access + corroborated on the NVIDIA dev forum "Peer access not supported between a device and itself"). This
  is why the probe at cuda_backend.cu:517 skips `peer == device_id_`, and it is the fact that lets me CORRECT a
  wrong sub-claim in the prior C1 (see below).
- `cudaMemGetInfo(size_t* free, size_t* total)` writes `free` then `total` (the order cuda_backend.cu:500-503
  reads them in) — confirmed.

---

## Role & layering

`device/resources.{hpp,cpp}` is the M4.5 **dependency-injection home** for the single-node multi-GPU (SPMG)
precompute: it owns one `PerGpuResources` per device (RAII `unique_ptr<ComputeBackend>` + the probed
`BackendCapabilities` + the physical ordinal), pins the fixed `g=0..G-1` combine order, and holds the
out-of-band `last_combine_path` tag. `build_resources` is the builder; `resolve_device_order` (file-local) is
the order policy.

The layering is the **cleanest expression of the §4 CUDA-free seam in the M4.5 set**, and it deserves to be
called out before the findings:

- The header names only `ComputeBackend` (by `unique_ptr`), the CUDA-free `BackendCapabilities` POD, and
  `DeviceConfig` — it compiles into `core`/the tests with no CUDA toolkit (§4). Verified: the only includes are
  `<cstddef>/<memory>/<vector>`, `steppe/config.hpp`, `device/backend.hpp`. Correct.
- The `.cpp` reaches the GPU **only** through `make_cuda_backend` + `capabilities()`. It includes no
  `<cuda_runtime.h>`, and (notably) gets even the visible-device *count* out-of-band from
  `capabilities().device_count` rather than calling `cudaGetDeviceCount` directly. Confirmed by CMakeLists:33-39
  (resources.cpp is a plain `.cpp`; CUDA libs PRIVATE at line 49).
- The combine *policy* (the §4 P2P-vs-host fork) correctly lives **above** this unit, in
  `core/fstats/f2_blocks_multigpu.cpp:156-201`; this unit only *records* the discovered tag. That is exactly the
  cleanup §(2).2/§(2).3 split.

There are **no parity risks** in anything proposed here — this unit touches device *binding* and *probing* only,
never the arithmetic. Every field it owns (`device_id`, `caps`, `last_combine_path`) is parity-neutral by
construction (§12), and the parity-critical invariant it *must* uphold — `gpus` in the fixed `g=0..G-1`
`device_order` — is upheld structurally (resources.cpp:88-94 pushes in `device_order` order).

---

## Score: 8.5/10 — clean layering and RAII, but a redundant device-0 context spin-up on the default auto path and an unfilled §9 validation contract keep it off the 9.5–10 bar

The header is excellent; the builder is correct and fail-fast on the paths it *does* guard, but it **silently
drops the two validation duties the §9 `build()` contract names explicitly** (duplicate / out-of-range
ordinals) and pays an avoidable full primary-context init + cuBLAS create/destroy + 64 MiB workspace
allocate/free on device 0 in the auto-enumerate path — *and* probes device 0 twice on that path. Neither breaks
parity; both are squarely "what a 9.5–10 senior bar requires."

The re-audit **confirmed the two headline findings (P1, C1)** and **everything in the layering / RAII / capability-
tier sections**, but it **corrected one wrong sub-claim inside C1** (the duplicate-`{0,0}` "self-peer
`cudaMemcpyPeer`" mechanism does not actually occur — verified against the P2P `.cu` skip condition and the
`cudaErrorInvalidDevice` self-pair fact), **demoted P3/P4 to non-findings**, **split E3's contradiction into a
verified doc-vs-code lie**, and **added five new substantiated findings** (a leaked ambient-device side effect in
the "pure-sounding" resolver, a double-probe of device 0, an unverified-but-real exception-safety property worth
documenting, a `device_count` width/`int` observation, and the `Resources::config` carrying two knobs the unit's
own doc lists but only one of which is ever read downstream).

---

## Findings

### Performance (first-class this pass)

#### P1 [HIGH perf, PARITY-SAFE: yes] `resolve_device_order` auto path spins up, then tears down, a full device-0 CUDA context (cuBLAS handle + 64 MiB workspace) just to read one `int` — and then probes/builds device 0 a SECOND time
Location: `resolve_device_order`, `resources.cpp:52-53` (`auto probe = make_cuda_backend(0); const int visible = probe->capabilities().device_count;`) + the RAII teardown at scope exit (line 64), then the construction loop `resources.cpp:88-94` re-builds device 0 as the real `gpus[0]`.

The throwaway `make_cuda_backend(0)` is **not** a cheap query. `make_cuda_backend` (cuda_backend.cu:609-611) does
`std::make_unique<CudaBackend>(0)`, and the `CudaBackend` ctor (cuda_backend.cu:94-108):
1. `set_and_return_device(0)` → `cudaSetDevice(0)`, which **as of CUDA 12.0 eagerly initializes the runtime and
   device-0's primary context** (verified above; pre-12.0 this was deferred);
2. constructs `blas_` (`CublasHandle`) → `cublasCreate`, whose context "is tightly coupled with the CUDA context
   current at the time of the `cublasCreate()` call" (the ctor doc cites cuBLAS §2.1.2);
3. constructs `workspace_` = `DeviceBuffer<std::byte>{kCublasWorkspaceBytes}` → a **64 MiB `cudaMalloc`**
   (config.hpp `kCublasWorkspaceBytes`);
4. `blas_.set_workspace(...)` + `blas_.set_stream(stream_)`.

All of that is built **only** to call `capabilities()` for a single `int` (`device_count`), then destroyed at
scope exit (the workspace `cudaFree`s 64 MiB; `cublasDestroy` synchronizes). Worse, on the auto path the build
loop's *first* iteration then does `make_cuda_backend(0)` **again** — re-initializing device-0 state and
re-`cudaMalloc`-ing the 64 MiB workspace a second time — **and** `capabilities()` runs the full peer-scan loop +
`cudaGetDeviceProperties` + `cudaMemGetInfo` on device 0 **twice** (once in the throwaway, once in the loop;
see new finding P5).

Why it matters: this is pure setup waste — a ~64 MiB allocate+free, a cuBLAS create+destroy (the destroy
synchronizes), and a primary-context spin — on the **default** path the budget 2×5090 box always takes (empty
`devices` ⇒ auto-enumerate). It has no functional return and it muddies any cold-start timing (§11.3 is about hot
loops, but this is gratuitous cold-start overhead with zero benefit).

Concrete fix (CUDA-free preserved): `device_count` is a process-global query that needs no bound backend. Add a
CUDA-free factory query beside `make_cuda_backend` in `backend_factory.hpp` —
`[[nodiscard]] int visible_device_count();` — defined in `cuda_backend.cu` as a one-line
`STEPPE_CUDA_CHECK(cudaGetDeviceCount(&n))` wrapper, and call it in `resolve_device_order` instead of building a
backend. This keeps resources.cpp CUDA-free (the symbol resolves through the device target exactly like
`make_cuda_backend`; no `<cuda_runtime.h>` here), and removes the 64 MiB alloc/free, the cuBLAS create/destroy,
the duplicate device-0 primary-context spin, AND the double-probe (P5). Severity: high (default path). Effort: S.
Risk: none — `cudaGetDeviceCount` returns the same value `capabilities().device_count` already captures
(cuda_backend.cu:478).

Adversarial check 1 — does `cudaGetDeviceCount` itself init a context, making the win illusory? No: even if the
runtime does some lightweight bookkeeping, `cudaGetDeviceCount` does NOT create a cuBLAS context and does NOT
allocate the 64 MiB workspace, so the bulk of the cost (the alloc/free + cuBLAS create/destroy) is unambiguously
eliminated. Adversarial check 2 — is the throwaway load-bearing as a context warm-up so the later
`make_cuda_backend(0)` is faster? No: CUDA 12 primary contexts are refcounted; the throwaway's destruction drops
the refcount, so there is no warm-cache benefit, and the workspace alloc is wasted regardless. The comment at
resources.cpp:49-51 even *acknowledges* the throwaway is destroyed "before the real per-device backends are
constructed below." Confirmed real, not a false positive.

#### P2 [LOW perf, PARITY-SAFE: yes-if-careful] Per-device backend construction (and the probe) is fully sequential; the G primary-context inits + G×64 MiB workspace allocs do not overlap
Location: `build_resources` loop, `resources.cpp:88-94`.

Each iteration does a `cudaSetDevice(ordinal)` (eager context init, CUDA 12), `cublasCreate`, a 64 MiB
`cudaMalloc`, then a `capabilities()` probe that itself does a save/restore `cudaSetDevice` round-trip +
`cudaGetDeviceProperties` + `cudaMemGetInfo` + a peer-scan loop (cuda_backend.cu:470-547). For G=2 this is small;
the *shape* of the loop is "initialize device g completely, then device g+1," and the per-device context
creations + workspace allocs are independent and serialize needlessly.

Honest verdict: **do not optimize this yet.** This is build-time latency only (never the hot path), context init
is driver-serialized internally on many drivers, and the §11.4 SPMG model has each device driven by its own host
context as a *later* workflow (f2_blocks_multigpu.hpp:46-50 explicitly defers per-device host threads as "a later
performance workflow, NOT a parity concern"). Listed because the user asked for *every* serialization point.

Concrete fix (when the per-device-thread workflow lands): construct the G backends on G host threads (one per
device), each binding its own device. PARITY-SAFE **yes-if-careful**: the resulting `gpus` vector MUST still be
ordered g=0..G-1 — join then place by index, NEVER by completion order, or the fixed combine order (§12) is
scrambled. Severity: low. Effort: M. (Note: the prior draft's parenthetical "§11.4 mentions G up to 8" is dropped
— §11.4 says only "the G devices," no specific cap.)

#### P3 [LOW perf, PARITY-SAFE: yes — REJECTED as a non-finding] `std::vector<int>` copy of `config.devices` on the explicit branch
Location: `resources.cpp:45-46` (`return config.devices;`).

`resolve_device_order` takes `const DeviceConfig&`, so the explicit branch copies `config.devices` (cannot move
out of a const ref), and `build_resources` separately copies the whole `config` into `resources.config` at line
80 — so the device list is copied twice on the explicit path. This is a **G-int** copy (single-digit ints);
negligible, and any "fix" (pass-by-value to enable a move) would copy the rest of `DeviceConfig` — the filter
vectors, the Precision, the knobs — anyway. **No action; demoted to a rejection** (see Considered & rejected).

### Correctness & bugs

#### C1 [MED, PARITY-SAFE: yes] `build_resources` does NOT validate the §9 contract: duplicate or out-of-range device ordinals in `config.devices` are not rejected here
Location: `build_resources`, `resources.cpp:69-97`; contract at architecture.md §9 ("**Validation at `build()`**
rejects ... any device id in `DeviceConfig::devices` that is absent or duplicated (empty ⇒ auto-enumerate, never
invalid)" — verified verbatim, architecture.md line 625).

The §9 contract is explicit that the resolution-time layer rejects (a) an ordinal absent on the box and (b) a
duplicated ordinal. This unit is the M4.5 realization of that resolution step (cleanup §(2).1 "owned at
`Resources` construction"), and `ConfigBuilder::build()` does not yet exist — so `build_resources` is **the only
enforceable site today**, exactly the situation B13 named for `assign_blocks`.

What actually happens on the two bad inputs:
- **Out-of-range ordinal** (e.g. `devices={0,5}` on a 2-GPU box): caught, but *indirectly* and with a worse
  message — `make_cuda_backend(5)` → `set_and_return_device(5)` → `cudaSetDevice(5)` throws a `CudaError`
  ("invalid device ordinal") from deep in the backend (cuda_backend.cu:569-572). It is fail-fast (good), but the
  error names a CUDA ordinal failure, not "configured device 5 is not among the N visible devices" — the §9-grade
  diagnostic the contract wants. Acceptable-but-not-9.5.
- **Duplicate ordinal** (e.g. `devices={0,0}`): **NOT caught at all.** `build_resources` builds two distinct
  `CudaBackend` instances both bound to device 0 and returns a G=2 `Resources` with
  `gpus[0].device_id == gpus[1].device_id == 0`. The user asked for 2 GPUs and silently got one GPU doing double
  the work serially, with no warning. §9 says reject it; this unit must.

**CORRECTION to the prior draft (verified against the P2P `.cu`).** The prior C1 claimed a duplicate non-root
device-0 entry "would be staged as a 'peer' of root 0 — a self-peer copy the P2P unit's contract did not
anticipate." **That is false.** `combine_f2_partials_p2p` (cuda/p2p_combine.cu:243) branches on
`owning_device == root_device_id`, **not** on `g == 0`. For `{0,0}` the second partial has
`owning_device == device_ids[1] == 0 == root_device_id`, so it takes the **no-peer-hop H2D branch**
(cuda/p2p_combine.cu:248-251) — exactly like the root's own partial. There is **no** `cudaMemcpyPeer(dst,0,src,0)`
self-peer DMA (which would, per the verified `cudaErrorInvalidDevice` self-pair fact, fail). So the duplicate
`{0,0}` would in fact run to completion on *both* combine paths and produce a numerically valid (if wasteful)
result. The footgun is therefore purely a **silent semantic violation** (2 lanes serialized on 1 GPU; the user's
intent ignored; and a genuine second device masked), not a P2P crash. Severity stays MED on the silent-contract
ground; the crash sub-claim is withdrawn.

Concrete fix: after `resolve_device_order` returns, validate before the construction loop (composes with P1's
count query — ONE CUDA-free call serves both):
```cpp
// §9 build() contract: reject duplicate or out-of-range ordinals (fail-fast, §2).
const int visible = visible_device_count();            // the P1 CUDA-free count query
std::vector<char> seen(static_cast<std::size_t>(visible < 0 ? 0 : visible), 0);
for (const int ord : device_order) {
    if (ord < 0 || ord >= visible)
        throw std::runtime_error("build_resources: configured device ordinal " +
            std::to_string(ord) + " is not among the " + std::to_string(visible) +
            " visible CUDA devices (architecture.md §9)");
    if (std::exchange(seen[static_cast<std::size_t>(ord)], 1))
        throw std::runtime_error("build_resources: configured device ordinal " +
            std::to_string(ord) + " is duplicated in DeviceConfig::devices "
            "(architecture.md §9 — the combine order must pin DISTINCT devices)");
}
```
Severity: med (silent contract violation on the duplicate; weak message on out-of-range). Effort: S. Risk: none —
it only rejects inputs that were never valid; parity unaffected. PARITY-SAFE: yes.

Adversarial check: is duplicate-rejection wrong because someone legitimately wants 2 lanes on 1 GPU? No —
`DeviceConfig::devices` is documented (config.hpp:202-209) as the SINGLE source of truth that "PINS both the set
AND the ordering, which is the fixed combine order"; a set with a repeated member is ill-formed by that contract,
and §9 explicitly lists "duplicated" as a reject. Confirmed real.

#### C2 [LOW, PARITY-SAFE: yes] `resolve_device_order`'s `visible < 1` guard is dead on the CUDA path today; it becomes the *real* primary gate once P1 lands
Location: `resources.cpp:52-58`.

On a box with zero CUDA devices, `make_cuda_backend(0)` → `cudaSetDevice(0)` throws first (invalid device), so the
`visible < 1` check at line 54 is unreachable via the CUDA backend. It is a correct belt-and-suspenders (a future
mock factory could return a backend reporting `device_count == 0`), so keep it — but the comment block
(lines 49-53) implies it is the primary "no visible device" gate, which is misleading: today the primary gate is
the throw *inside* `cudaSetDevice`. Once `resolve_device_order` uses the P1 `visible_device_count()` query, the
`visible < 1` guard becomes the *actual* primary gate (with a precise message) — strictly better. Severity: low.
Effort: folded into P1. PARITY-SAFE: yes.

#### C3 [LOW, PARITY-SAFE: yes — optional polish] No guard that `make_cuda_backend(ordinal)` returned non-null before `entry.backend->capabilities()` dereferences it
Location: `resources.cpp:91-92`.

`entry.backend = make_cuda_backend(ordinal); entry.caps = entry.backend->capabilities();` — if a future/alternate
factory returned `nullptr`, line 92 is a null deref. Today this is **not a live bug**: `make_cuda_backend` is
`std::make_unique<CudaBackend>` (cuda_backend.cu:610), which either returns non-null or throws `bad_alloc`. The
gate test (test_resources_build.cu:114-115, 168-169) already asserts `backend != nullptr` as a contract; a
`STEPPE_ASSERT` (debug-only, the host_device.hpp macro the multigpu entry uses) would document the invariant at
zero release cost. Severity: low. Effort: trivial. PARITY-SAFE: yes. Adversarial: arguably noise since
`make_unique` can't return null — optional polish, not a defect.

### Edge cases & failure modes

#### E1 [covered — good] Empty resolved order → fail-fast
`build_resources:73-77` throws on an empty `device_order` with a §9-cited message. Correct and well-placed. The
auto path's `visible < 1` also guards it; the explicit path can never produce empty because the
`!config.devices.empty()` branch is taken (resources.cpp:45-46). Note: this means the `device_order.empty()`
throw at line 73 is, like C2's guard, currently only reachable via a hypothetical mock — but it is correct
defensive fail-fast. Good.

#### E2 [N/A — handled above this unit] Empty shard / single-GPU / G==1
The single-GPU structural no-op and empty-shard handling live in the *consumer*
(`f2_blocks_multigpu.cpp:88-91` and the backend's degenerate guard cuda_backend.cu:135-137), not in
`build_resources`. `build_resources` correctly produces a valid 1-entry `Resources` for `devices={0}` (gate test
LANE A). Nothing for this unit to do. Covered.

#### E3 [MED, PARITY-SAFE: yes] VERIFIED doc-vs-code lie: `CombinePath::None`'s doc claims it covers the G==1 fast path, but no code path ever *sets* `None` after a run
Location: `resources.hpp:46-49` (the `None` enumerator doc) vs the consumer's write sites.

The `None` doc (resources.hpp:47-49) says: "No multi-GPU combine has run on this Resources yet (the
value-initialized default), **OR the last run was the G==1 single-GPU fast path**." I traced every write of
`last_combine_path`: the ONLY assignments are `HostStaged` (f2_blocks_multigpu.cpp:200) and `P2pDeviceResident`
(f2_blocks_multigpu.cpp:183). The G==1 fast path (f2_blocks_multigpu.cpp:88-91) returns **without touching the
field**. So `last_combine_path` is `None` *only* as the value-initialized default — it is **never assigned `None`
by any run**. Consequence: after a G==2 P2P run followed by a G==1 run on the *same* `Resources` (re-binding to a
single device is unusual but legal), the tag stale-reads `P2pDeviceResident` even though the last run combined
nothing. The doc and code therefore **disagree**: the second clause of the `None` doc describes behavior that does
not exist.

Why it matters: it is purely observability (parity-neutral), but it is a contradiction in exactly the "which-path
tag" the cleanup §(2).2 design exists to make trustworthy; a test reading the tag after a mixed G sequence would
see a lie. Concrete fix — two correct options, pick one and make doc+code agree:
- **(preferred)** the consumer sets `resources.last_combine_path = CombinePath::None;` on the G==1 early-return
  (one line at f2_blocks_multigpu.cpp:88-91), making the doc true; OR
- **(doc-only, in this unit)** change the `None` doc to "the value-initialized default; a G==1 run does NOT update
  it" and delete the false second clause.
Since the *contract* (the enum doc) lives in this file, this unit must at minimum fix the comment; the cleaner fix
is the consumer setting `None`. Severity: med (doc/code contract mismatch on the trust-bearing tag). Effort: S.
PARITY-SAFE: yes (tag is off the numeric payload).

#### E4 [LOW, PARITY-SAFE: yes] No documented thread-safety contract on the mutable `last_combine_path` vs the §11.4 future per-device-thread model
Location: `resources.hpp:104-110`.

When the P2 per-device-thread workflow lands, `Resources&` will be shared across the orchestrator;
`last_combine_path` is written by exactly one place (the single-threaded combine-selection step) so there is no
race today, and the per-device *compute* threads do not touch it. A one-line "written only by the single-threaded
combine-selection step, never by per-device worker threads" note would keep the future workflow from introducing
a data race. Severity: low. Effort: trivial (doc only). PARITY-SAFE: yes.

#### E5 [NEW — LOW, PARITY-SAFE: yes] `resolve_device_order`'s auto path leaks a `cudaSetDevice(0)` ambient side effect out of a function whose name reads as a pure resolution
Location: `resources.cpp:52` (the throwaway `make_cuda_backend(0)`).

`resolve_device_order` reads as a pure "decide the ordering" helper, but on the auto path it constructs a
`CudaBackend`, whose ctor does `cudaSetDevice(0)` (cuda_backend.cu:95 via `set_and_return_device`) and **does not
restore** the previously-current device. The throwaway's destruction frees its resources but does NOT un-set the
current device, so `resolve_device_order` returns with **device 0 current as a side effect** — unlike the
explicit branch (a pure `return config.devices;`) which changes nothing. Today this is harmless: the
construction loop immediately re-binds via each `make_cuda_backend(ordinal)`, and `capabilities()` save/restores.
But it is a hidden, asymmetric side effect in a "pure-sounding" resolver, and it is exactly the kind of latent
footgun that bites when the function is reused. **P1 removes it entirely** (a `cudaGetDeviceCount` count query
does not require a bound, device-selecting backend, and `cudaGetDeviceCount` does not change the current device).
Severity: low. Effort: folded into P1. PARITY-SAFE: yes.

### Numerical / precision (§12)

#### N1 [N/A — correctly parity-neutral by construction]
This unit performs no arithmetic. Every field it owns (`device_id`, `caps`, `last_combine_path`) is a
data-movement/observability datum (§12 parity-neutral; cleanup §(2).5). The fixed `g=0..G-1` order is preserved
structurally: `build_resources` pushes `gpus` in `device_order` order, which is `config.devices` verbatim on the
explicit path (resources.cpp:45-46) and dense `0..count-1` on the auto path (resources.cpp:59-64) — both the
§11.4/§12 fixed combine order. **The one parity-critical invariant this unit must uphold is upheld.** The only
place a future change could leak parity is reordering `gpus` (e.g. the P2 threaded build placing by completion
order) — flagged in P2 as yes-if-careful. Explicitly justified N/A.

### CUDA idioms / RAII / async semantics / layering (§7)

#### R1 [GOOD] RAII ownership and move-only semantics are correct
`PerGpuResources` holds `unique_ptr<ComputeBackend>` ⇒ move-only with synthesized move ops; `Resources` holds
`vector<PerGpuResources>` ⇒ move-only by composition. The doc (resources.hpp:70-73, 93) is accurate: copy is
correctly ill-formed (would clone a deleted-copy backend). `gpus.reserve(device_order.size())` before the push
loop (resources.cpp:81) avoids reallocation-moves of the `unique_ptr`-holding entries during construction. The
factory returns `unique_ptr<ComputeBackend>` (abstract) so no concrete type is named. Textbook §7/§9.

#### R2 [NEW — LOW, PARITY-SAFE: yes] `build_resources` is strongly exception-safe (mid-loop throw cleans up partial state) but this is undocumented
Location: `build_resources`, `resources.cpp:88-94`.

If `make_cuda_backend(ordinal)` throws fail-fast mid-loop (e.g. ordinal 1 of `{0,1}` is invalid), the
already-constructed `gpus[0]` entry (and its RAII backend: cuBLAS handle + 64 MiB workspace) is correctly
destroyed when the `Resources resources` local unwinds — and the in-progress `PerGpuResources entry` (which owns
the partially-built backend, or none) likewise unwinds. So `build_resources` is **strongly exception-safe**: a
throw leaks no device handle or VRAM. This is a genuinely good property worth a one-line note ("`build_resources`
is strongly exception-safe — a mid-loop fail-fast unwinds all already-bound backends via RAII"), because it is
load-bearing for the C1 validation (which throws after partial work is possible) and for any future caller doing
retry. Severity: low (doc only — the property already holds). Effort: trivial. PARITY-SAFE: yes.

#### R3 [GOOD] `device_count()` is `[[nodiscard]] ... const noexcept`
resources.hpp:113 — correct const-correctness and noexcept on the trivial accessor.

#### R4 [LOW — cross-unit observation, no action in resources.cpp] The probe's save/restore `cudaSetDevice` round-trip is a no-op pair given the construction loop already left that device current
The probe (cuda_backend.cu:481-545) saves `cudaGetDevice`, sets `device_id_`, queries, restores `entry_device`.
In the `build_resources` loop, `make_cuda_backend(ordinal)` left `ordinal == device_id_` current, so on entry to
`capabilities()` `entry_device == device_id_` already, and the save/set/restore is a redundant pair on the build
hot path. But the probe is `const` and callable from *any* ambient-device context (its doc, cuda_backend.cu:465-469,
is accurate), so the defensive save/restore is the right design for the method — the redundancy is the *caller's*
(this loop), not the method's. No action on resources.cpp; once P5's double-probe is gone this is the only
per-device probe and the redundancy is one no-op pair per device — negligible. Noted for the cost picture.

#### R5 [LOW, PARITY-SAFE: yes] `build_resources` documents its throw set in prose only — no `@throws` tag
Location: `resources.hpp:116-130`.

`build_resources` is correctly `[[nodiscard]]` (resources.hpp:130) and correctly NOT `noexcept` (it throws
fail-fast). The throw set ("bad ordinal / empty order / no visible device") is in the prose banner
(resources.hpp:122-129) but there is no `@throws` tag and no enumerated exception type. For a §9 builder a
one-line `@throws std::runtime_error` (empty/no-device, and the C1 validation) `/ CudaError` (bad ordinal via the
backend ctor) would match the rigor of p2p_combine.hpp:100-103 (which enumerates) and f2_combine.hpp. Severity:
low. Effort: trivial. PARITY-SAFE: yes.

### Magic numbers & hardcoded values (§4)

#### M1 [N/A — clean] No magic numbers in this unit
The only literals are `0` (the device-0 throwaway ordinal — which P1 removes), `1` (the `visible < 1` floor, a
true mathematical "at least one device"), and the loop init `0`. The 64 MiB workspace and the mantissa default
live in config.hpp, not here. Clean.

### Decomposition / single-responsibility / function size (§2)

#### D1 [GOOD] Clean two-function decomposition
`resolve_device_order` (the order policy) is correctly separated from `build_resources` (the construction +
probe). Both are short and single-purpose. Adding the C1 validation grows `build_resources` by ~12 lines, which
is fine; if preferred it can be its own `validate_device_order(order, visible)` free function for testability
(see T1). No over-decomposition.

### Readability, naming, const-correctness, comment density (§2/§7)

#### Cmt1 [MED — the dominant cleanliness issue] Comment density is ~6:1 prose-to-code; several comments are stale, speculative, or duplicate the gate condition four times
Location: throughout — esp. resources.hpp:35-59 (a 25-line doc-comment on a 4-value enum), resources.hpp:116-129,
resources.cpp:7-22, 37-43, 49-53.

The unit is correct but **buried** in commentary. Specific drift / over-claim:
- resources.hpp:9-22 and resources.cpp:7-22 both narrate "CUDA-FREE BY CONTRACT" at length, and the *identical*
  "CUDA-FREE BY CONTRACT, like device/backend.hpp" paragraph is also re-spelled in shard_plan.hpp:13-18 and
  p2p_combine.hpp:13-22. Per §8 DRY the rule belongs once (a one-line `// CUDA-free seam: see device/backend.hpp`
  cross-ref), not a re-explained paragraph per file. Same "duplicated WHY prose, single-home it" smell L21
  flagged elsewhere.
- resources.cpp:49-53 narrates the throwaway-probe mechanism that P1 deletes — the comment goes with the code.
- resources.hpp:47-49 (`None` covers the G==1 path) is **contradicted by the code** (E3) — the most important one
  to fix, because it is a *false* comment, not just a verbose one.
- The enum doc (resources.hpp:35-45, 53, 57) re-states the §11.4 gate condition (`prefer_p2p_combine &&
  can_access_peer && G >= 2`). The SAME gate is stated in config.hpp:262-287, in f2_blocks_multigpu.cpp:156-167,
  and in p2p_combine.hpp:64-67 — the gate is now documented in **four** places. Per §8 it should have ONE
  authoritative home (the entry point that evaluates it, f2_blocks_multigpu.cpp) and cross-refs elsewhere.

Why it matters: §2 values readability; over-commentary that drifts is worse than terse code because a reviewer
must reconcile four copies of the gate and spot the one (E3) that lies. Concrete fix: trim the enum/struct docs to
the contract + a single cross-ref; delete the throwaway-probe narration with P1; single-home the gate condition;
fix the E3 enum doc. Severity: med (largest standing cleanliness debt, and a doc/code contradiction is hiding in
it). Effort: M. PARITY-SAFE: yes (comments only).

#### Cmt2 [LOW — correct pattern, not a defect] `caps{}` value-init brace on `PerGpuResources::caps`
resources.hpp:87 — `BackendCapabilities caps{};` correctly value-initializes (all-zero/false = "unknown" default,
matching the backend.hpp:144-185 contract). Good; noted as a correct pattern.

#### Nm1 [LOW] `resolve_device_order` local `visible` vs the field name `device_count`
resources.cpp:53 names the count `visible`; the source field is `BackendCapabilities::device_count`. Minor
inconsistency — `visible` is arguably clearer ("visible device count") but a reader cross-referencing the probe
sees two names for one quantity. Once P1's `visible_device_count()` query lands, the helper's local naming aligns
with the query name. Trivial; leave or rename to `visible_count`. Severity: low. Effort: trivial.

### Performance — see the dedicated section above (P1–P3) + P5 (new) below

#### P5 [NEW — MED perf, PARITY-SAFE: yes] On the auto path, device 0's full `capabilities()` probe runs TWICE (the throwaway + the real `gpus[0]`)
Location: `resources.cpp:52-53` (throwaway probe) + `resources.cpp:92` (`gpus[0]`'s probe).

This is the *probe-cost* half of P1, called out separately because it is its own waste even setting aside the
backend construction. `capabilities()` (cuda_backend.cu:470-547) is not free: it does `cudaGetDeviceCount`, a
`cudaGetDevice`/`cudaSetDevice` save/set, `cudaGetDeviceProperties`, `cudaMemGetInfo`, a **peer-scan loop over all
visible devices** (`cudaDeviceCanAccessPeer` per peer — up to `count-1` driver calls), the
`emulation_honorable` predicate, and a `cudaSetDevice` restore. On the auto path this entire sequence runs once in
the throwaway `probe->capabilities()` (line 53, only `device_count` is read — the peer scan, props, and meminfo
are computed and **discarded**) and again at line 92 for the real `gpus[0]`. Replacing the throwaway with the P1
`visible_device_count()` (a single `cudaGetDeviceCount`) eliminates the discarded probe entirely. Severity: med
(it is per-build on the default path, and the peer scan is O(count) driver round-trips). Effort: S (subsumed by
P1). PARITY-SAFE: yes — the probe is parity-neutral observability.

### Layering / API / ABI (§4)

#### L1 [GOOD, with one nuance — fixed by P1] The CUDA-free split is clean; the device-count-via-probe trick is the one place it is *over*-engineered
The header/cpp are CUDA-free (verified against the includes + CMakeLists). The one nuance: `resolve_device_order`
builds a full throwaway backend *specifically to avoid* calling `cudaGetDeviceCount` directly and keep the body
CUDA-free (resources.cpp:40-43 narrates this). P1's fix (a CUDA-free `visible_device_count()` factory query)
achieves the *same* layering cleanliness with none of the cost — the layering goal is preserved while the
over-engineering is removed. There is no actual layering violation; this is the rare case where the perf fix and
the cleanliness fix are the same change. PARITY-SAFE: yes.

#### L2 [OBSERVATION — intentional, better-than-sketch divergence from the §9 reference shape] `Resources` here departs from the architecture.md §9 sketch
The §9 sketch (architecture.md:606-620) puts a single top-level `unique_ptr<ComputeBackend> backend` (the
"multi-GPU aware" backend) plus a `vector<PerGpuResources>` whose entries hold `StreamPool`, `DeviceAllocator*`,
`CublasHandle`, `CusolverDnHandle`, `NcclComm`. The M4.5 implementation instead:
- moves the backend *into* `PerGpuResources` (one backend per device) and drops the top-level `backend` —
  **correct**: the §9 sketch predates the per-device-instance contract (backend.hpp:193-202) where one backend ==
  one device; a top-level "multi-GPU aware backend" would re-own per-device state the backend already holds (DRY
  §8). The header's own doc (resources.hpp:16-22) argues this well.
- does NOT carry `StreamPool`/`allocator`/`blas`/`solver`/`NcclComm` in `PerGpuResources` — **correct**: those
  live *inside* `CudaBackend` (stream_/blas_/workspace_, cuda_backend.cu:596-598); re-owning them here is
  DRY-violating, and NcclComm is intentionally absent (no NCCL on the parity path, §11.4/§12).
- ADDS `config` and `last_combine_path` to `Resources` (not in the §9 sketch) — **correct and required** by
  cleanup §(2).2 (the discovered tag must live on `Resources`).

This is a *good* divergence (the impl is more correct than the older sketch), but it is undocumented as a
deliberate deviation. A 9.5-bar move: a one-line header note ("`Resources` here intentionally departs from the
architecture.md §9 sketch: backend is per-device per the backend.hpp per-device-instance contract; the
stream/handle/workspace are owned inside `CudaBackend`, not re-owned here") so a future reader does not "fix" it
back to the sketch. Severity: low (doc only). Effort: trivial. PARITY-SAFE: yes. NOT a defect — flagged so the
divergence is recorded.

### Testability (§13)

#### T1 [MED, PARITY-SAFE: yes] `resolve_device_order` and the (missing) ordinal validation are not GPU-free-unit-testable because the auto path requires a real device
Location: `resolve_device_order` (resources.cpp:44-65); the gate test is a `.cu` requiring a live GPU.

`resolve_device_order`'s *explicit* branch is pure (`return config.devices;`) and trivially testable, but the
*auto* branch and any ordinal validation are entangled with `make_cuda_backend`, so the only test
(test_resources_build.cu) is a GPU `.cu` that must run on the box. The C1 validation logic (duplicate/out-of-range
detection) is **pure host arithmetic over `(device_order, visible_count)`** and should be a free function
`validate_device_order(std::span<const int> order, int visible)` unit-tested GPU-free in a plain `.cpp` (the §13
"exercisable GPU-free" principle; same gap B11/B20 flagged for units whose only coverage is a data-box `.cu`). The
auto-enumeration *ordering* logic (`0..count-1`) is likewise pure given the count and could be tested by injecting
the count. Severity: med (a §13 host-test gap on the exact validation C1 adds — landing C1 without a host test
repeats the "only a .cu oracle" anti-pattern). Effort: S. PARITY-SAFE: yes.

#### T2 [LOW] The gate test does not exercise the fail-fast paths (bad/duplicate ordinal, empty order)
test_resources_build.cu asserts only happy paths (LANE A `{0}`, LANE B auto, LANE C `{0,1}`). There is no negative
case (`devices={0,99}` should throw; `devices={0,0}` should throw once C1 lands; empty order on a 0-device box).
The §2 fail-fast guards (E1, the C1 validation, the out-of-range throw) are untested. Add a
`try { build_resources(bad); FAIL } catch` lane. Severity: low (the guards exist for out-of-range; they're just
unverified, and the duplicate guard does not exist yet). Effort: S. PARITY-SAFE: yes.

### Capability-tier coherence (§(2))

#### K1 [GOOD] The probe is owned at `Resources` construction, the tag is off the numeric payload, and the override/discovered split is correct
Verified against cleanup §(2).1-§(2).4:
- §(2).1 "one capability probe, owned at `Resources` construction": `build_resources:92` stores the ONE
  `capabilities()` per device. Correct (modulo the P5 double-probe on the auto path, which is a perf waste, not a
  structural break).
- §(2).2 "tag never on `F2BlockTensor`": `last_combine_path` is on `Resources`, not the tensor. Correct.
- §(2).3 "two knob types": override intent (`prefer_p2p_combine`, `enable_peer_access`, `deterministic`) on
  `DeviceConfig`; discovered tag (`last_combine_path`) on `Resources`. Correct.
- §(2).4 "non-throwing tagged degrade": the probe's P2P answer is non-throwing (the WARN lives in `capabilities()`,
  cuda_backend.cu:519-527; resources.cpp correctly does NOT throw on `can_access_peer == false`). Correct.

The cleanest §(2) realization in M4.5. The one gap is E3 (the tag's G==1 contract), a doc/reset issue, not a
structural one.

#### K2 [LOW, PARITY-SAFE: yes — cross-unit] `Resources::config` carries `enable_peer_access`, and the header doc (resources.hpp:101) lists it as a frozen intent lever, but it is consulted NOWHERE — the combine gate ignores it
Location: `Resources::config` holds `enable_peer_access` (config.hpp:255-260); resources.hpp:101 lists it as one
of the frozen levers. Verified by grep: `enable_peer_access` appears in the codebase ONLY in comments/docs
(backend.hpp:168, resources.hpp:101) — it is read by **no** code. The combine gate (f2_blocks_multigpu.cpp:171-172)
checks only `prefer_p2p_combine && can_access_peer`, and the P2P `.cu` calls `cudaDeviceEnablePeerAccess`
**unconditionally** (cuda/p2p_combine.cu:265, routed through the non-throwing WARN).

So the §11.4 "MAY-WE" knob is a dead lever today: a user who sets `enable_peer_access=false` (forbid peer
enabling) but leaves `prefer_p2p_combine=true` (the default) would still take the P2P path and still call
`cudaDeviceEnablePeerAccess` — contradicting the documented MAY-WE semantic (config.hpp:255-259). This is **not** a
resources.cpp bug (the gate and the enable-call are in the consumer + the P2P `.cu`), but `Resources::config` is
the carrier of this dead knob, and resources.hpp:101 *advertises* it as a meaningful frozen lever. Flag for the
entry-point review: the gate should be `prefer_p2p_combine && enable_peer_access && can_access_peer && G>=2`, and
the unconditional `cudaDeviceEnablePeerAccess` at p2p_combine.cu:265 should respect it. Severity: low (functional
only for the unusual `enable_peer_access=false, prefer_p2p_combine=true` combo). Effort: S (in the consumer + the
`.cu`). PARITY-SAFE: yes (both branches are bit-identical; this only governs which transport).

---

## Considered & rejected (incl. rejected-for-parity)

- **REJECTED-FOR-PARITY: build the G backends and place `gpus` in completion order to overlap context init.**
  Tempting for P2, but placing by completion order would scramble the fixed `g=0..G-1` combine order §12 parity
  rests on. The threaded build must join and place strictly by index. Kept the perf idea (P2) but explicitly
  constrained it yes-if-careful.

- **REJECTED-FOR-PARITY: dedupe duplicate ordinals by silently collapsing `{0,0}`→`{0}` instead of throwing.**
  Collapsing changes G (2→1), which changes the shard plan and the combine layout — not a parity *break* per se,
  but it silently alters what the user asked for. §9 says *reject*, not collapse. Throw (C1).

- **WITHDRAWN sub-claim (prior C1): "a duplicate non-root device 0 would trigger a self-peer `cudaMemcpyPeer`."**
  Verified false against cuda/p2p_combine.cu:243 (the skip is `owning_device == root_device_id`, not `g == 0`) and
  the documented `cudaErrorInvalidDevice` self-pair behavior. A duplicate `{0,0}` takes the no-peer-hop H2D branch
  for *both* partials and runs to completion. The duplicate is still a silent-contract footgun (C1 stays MED) but
  it does not crash the P2P path. This correction is the kind of false sub-claim the re-audit exists to catch.

- **Rejected (demoted from P3/P4): the double-copy of `config.devices` and the by-value-helper micro-opt.** G-int
  copies; passing `resolve_device_order` a by-value `DeviceConfig` to enable a move would copy the rest of the
  struct anyway. No action.

- **Rejected: store the visible device count on `Resources`.** Not needed post-build; `device_count()` gives G;
  the visible count is used only during resolution. No action.

- **Rejected: add a hard runtime null-check (throw) on `make_cuda_backend`'s return.** `std::make_unique` cannot
  return null (throws `bad_alloc`), so a runtime throw is dead code; a debug `STEPPE_ASSERT` documenting the
  invariant is the most warranted (C3), and even that is optional.

- **Rejected: move the combine-path *selection* (the §4 gate) into this unit.** The gate belongs in the consumer
  (`f2_blocks_multigpu.cpp`) per §(2).3 / the layering; this unit only *records* the chosen path. Moving selection
  here would couple `Resources` construction to runtime combine policy.

- **Rejected: have `build_resources` call `cudaDeviceEnablePeerAccess` eagerly when
  `enable_peer_access && can_access_peer`.** Peer enabling is the P2P combine's job (it does it lazily,
  p2p_combine.cu:265); doing it at build time would (a) leak a device-state side effect out of a "build the
  bundle" function and (b) be wasted on the host-staged path. K2 is about the *gate* reading the knob, not about
  enabling here.

- **Rejected: cache `capabilities()` lazily instead of probing every device at build.** The §(2).1 contract is
  "ONE probe, owned at `Resources` construction" — eager-at-build is the design (the tier is known before any
  compute). Eager is correct. (P5 is about the *redundant second* probe of device 0 on the auto path, not about
  the eager-per-device policy.)

- **Rejected-as-noise: rename `visible` → `device_count` for exact field-name parity.** Logged as Nm1 (trivial,
  optional); not worth a finding on its own.

---

## What it takes to reach 10/10

Performance:
1. **P1 + P5 (high, S):** delete the throwaway `make_cuda_backend(0)` in `resolve_device_order`; replace with a
   CUDA-free `visible_device_count()` factory query. Removes a 64 MiB alloc/free + cuBLAS create/destroy + a
   redundant device-0 primary-context spin + the discarded second full probe (peer scan included) on the default
   (auto-enumerate) path. The single highest-value change in the unit; simultaneously satisfies L1, E5, and
   improves C2.
2. **P2 (low, M, later):** when the §11.4 per-device-thread workflow lands, construct the G backends on G threads
   — but place `gpus` strictly by index (parity).

Cleanliness / correctness:
3. **C1 (med, S):** add the §9 duplicate + out-of-range ordinal validation in `build_resources` (composes with
   P1's count query). Turn the silent `{0,0}` footgun into fail-fast; give the out-of-range case a §9-grade
   message.
4. **E3 (med, S):** fix the verified `last_combine_path` G==1 lie — either reset to `None` on the G==1
   early-return (consumer) or correct the `CombinePath::None` doc here; make doc and code agree.
5. **T1 + T2 (med, S):** extract `validate_device_order(order, visible)` as a pure free function and unit-test it
   GPU-free; add negative-path lanes (bad/dup ordinal, empty order) to the gate test.
6. **Cmt1 (med, M):** cut comment density ~3×; single-home the §11.4 gate condition (currently in four files);
   delete the throwaway-probe narration with P1; fix the false E3 enum doc.
7. **L2 + R2 + R5 + E4 (low, trivial):** record the deliberate divergence from the §9 `Resources` sketch; note
   `build_resources`'s strong exception-safety; add `@throws` to `build_resources`; add the one-line
   thread-safety note on `last_combine_path`.
8. **K2 (low, S, consumer + `.cu`):** thread `enable_peer_access` into the combine gate and the
   `cudaDeviceEnablePeerAccess` call so the MAY-WE knob `Resources::config` carries is actually honored
   (cross-unit; flag for the entry-point review).

None of 1–8 touch the arithmetic or the fixed combine order; all are PARITY-SAFE (P2 yes-if-careful).

---

## Good patterns to keep

- **The CUDA-free seam is exemplary.** Header + cpp reach the GPU only through `make_cuda_backend` +
  `capabilities()`; no `<cuda_runtime.h>`; `device_count` obtained out-of-band. This is the §4 dependency-direction
  done right — the template for the rest of the device-layer CUDA-free plumbing (shard_plan, p2p_combine decl all
  follow it). (P1 *preserves* this property while removing the over-engineering.)
- **RAII move-only ownership** of the per-device backends (`unique_ptr<ComputeBackend>` in `PerGpuResources`,
  vector-composed into `Resources`) with `reserve` before the push loop, and **strong exception safety** on a
  mid-loop throw — textbook §7/§9 (document the latter per R2).
- **The per-device-instance refactor** (backend owned *inside* `PerGpuResources`, stream/handle/workspace not
  re-owned) is more correct than the architecture.md §9 sketch and avoids a DRY violation — keep it (and document
  it per L2).
- **The capability-tier realization** (one probe owned at construction; discovered tag on `Resources` off the
  numeric payload; override intent on `DeviceConfig`; non-throwing P2P degrade) is the cleanest §(2)
  implementation in M4.5 — keep the structure; only fix the E3 contract.
- **Fixed g=0..G-1 order is structural**: `gpus` is pushed in `device_order` order, the §11.4/§12 combine order on
  both the explicit and auto paths. Keep this invariant front-and-center in any future threaded build.
- **Fail-fast where it guards** (empty resolved order, bad ordinal via `cudaSetDevice`) with §-cited messages —
  extend this rigor to the duplicate/out-of-range cases (C1) and the pattern is complete.

# M4.5 unit review (adversarial second pass) — `device/cuda/handles` (`src/device/cuda/handles.hpp`)

Unit under review: the **M4.5 delta** to `src/device/cuda/handles.hpp` (branch `m4.5-multigpu`). The delta is exactly
two additions, and *only these are in scope* (the pre-M4.5 `CublasHandle` (stream, workspace) core was already
reviewed at 8.5/10 in `docs/cleanup/device-cuda-handles.md` and is NOT re-litigated except where the delta touches it
or where a pre-existing in-file comment is load-bearing to a delta claim):

1. **The device-ordinal record-and-assert.** The ctor does `STEPPE_CUDA_CHECK(cudaGetDevice(&device_id_))` *before*
   `cublasCreate`; a new private `assert_on_creation_device()` (debug-only, `noexcept`) is called at the head of
   `set_workspace()` (line 111) and `set_stream()` (line 125); a new `[[nodiscard]] int device_id() const noexcept`
   accessor (line 136); the ordinal is carried through both move operations.
2. **`MathModeScope`** (lines 218–263) — a move-only RAII class that captures a handle's current `cublasMath_t`
   (`cublasGetMathMode`), applies a requested mode, and restores the captured mode at scope exit, with the same
   non-throwing-dtor / teardown-warn discipline as `CublasHandle`.

**This is the adversarial second pass.** The first pass scored 8.5/10 with 28 findings + 4 perf-findings. I re-read
`handles.hpp` line by line, re-fetched the official NVIDIA docs, and *re-verified every finding against the actual
code*. The first pass holds up well — it is an unusually careful first pass — but it **(a) under-stated the
`device_id()` deadness** (the accessor is not merely "not yet wired," it is architecturally *unreachable* from the
consumer it names, because `PerGpuResources` holds the abstract `ComputeBackend`, not the concrete `CudaBackend`);
**(b) mis-framed the 9.2 asymmetry** (the sibling hazard is NOT symmetric — `cudaFree` is device-independent, only
`cudaMalloc`/stream-launch are device-bound); and **(c) missed three real findings** — a brand-new device-ordinal
hazard the delta *created* in `MathModeScope` (N4, the strongest new finding), an internally-inconsistent
declaration-order rationale in the `set_workspace` doc (N1), and a moved-from-stale-ordinal robustness gap (N2). It
also **over-praised the move test** (the NRVO-elided path means the move-ctor inertness is not *guaranteed* exercised
— N3-test). Net: the score is unchanged at **8.5/10**, but the path to 10/10 is sharper and one item (N4) is a
delta-internal consistency defect, not just a wiring gap.

### Cross-file context verified this pass (line by line)
- `core/internal/host_device.hpp` — confirmed `STEPPE_DEBUG_ONLY(...)` is `((void)0)` under NDEBUG with the argument
  *unevaluated* (line 64), and `STEPPE_ASSERT(cond,msg)` is `((void)0)` under NDEBUG (line 79) / `assert((cond)&&(msg))`
  in debug (line 81). The assert is genuinely release-free.
- `device/cuda/check.cuh` — confirmed `CublasError::status_name` is `[[nodiscard]] static … noexcept` and
  allocation-free (lines 84–98); `cuda_check` throws `CudaError` on any non-`cudaSuccess` (line 114); `cuda_warn` is the
  non-throwing recoverable sibling (lines 134–149).
- `device/cuda/cuda_backend.cu` — the **sole production consumer**. Ctor (lines 94–108): `device_id_` is initialized
  *first* via `set_and_return_device` (which `cudaSetDevice`s, line 570) so the device is current before `blas_`/
  `workspace_` construct; `set_workspace` + `set_stream` are called **exactly once each** (lines 106–107). Every compute
  entry re-selects via `guard_device()` (`cudaSetDevice(device_id_)`, line 561). `engage_f2_precision(blas_.get(), …)`
  is called raw at line 339 (and the M0 path's `run_f2_gemms` calls it at `f2_block_kernel.cu:337`) — **never wrapped
  in a `MathModeScope`**. Member declaration order: `blas_` (597) before `workspace_` (598) — see N1.
- `device/cuda/f2_block_kernel.cu` — `engage_f2_precision` (lines 278–301) sets `cublasSetMathMode` raw
  (`CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` or `CUBLAS_PEDANTIC_MATH`) and **never restores** — the open hazard `MathModeScope`
  documents (3.1, confirmed).
- `device/resources.hpp` — `PerGpuResources` holds `int device_id` (line 76) and the backend as
  `std::unique_ptr<ComputeBackend>` (line 81, the *abstract* interface, "Held as the abstract interface so this struct
  stays CUDA-free"). The capability tier lives in `caps` (line 87), out-of-band. **No code path can reach
  `CublasHandle::device_id()` from here** (N3 / strengthened 7.2).
- `device/cuda/p2p_combine.cu`, `device/cuda/device_buffer.cuh`, `device/cuda/stream.hpp`,
  `tests/reference/test_handles.cu`, `tests/CMakeLists.txt` — all read (siblings have no device-ordinal record; the
  test is one single-device CTest target).

### Device-behavior claims re-fetched and CONFIRMED against the official docs this pass
- **`cudaGetDevice`** is `__host__ __device__ cudaError_t cudaGetDevice(int* device)`, "Returns which device is
  currently being used," return values `cudaSuccess`/`cudaErrorInvalidValue`, and the docs **explicitly** state "Note
  that this function may also return error codes from previous, asynchronous launches"
  ([CUDA Runtime API — Device Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)).
  `cudaSetDevice` carries the *identical* "may also return error codes from previous, asynchronous launches" note —
  load-bearing for 1.1.
- **`cublasSetStream`** "unconditionally resets the cuBLAS library workspace back to the default workspace pool" and
  the doc mentions **only** stream + workspace — it does **not** reset the math mode (cuBLAS §2.4.7,
  [cuBLAS Library docs, CUDA 13.x](https://docs.nvidia.com/cuda/cublas/index.html)). Confirms `MathModeScope` is a
  *distinct* concern from the (stream, workspace) re-apply.
- **`cublasMath_t`**: `CUBLAS_DEFAULT_MATH` = "the default and highest-performance mode … Tensor Cores will be used
  whenever possible"; `CUBLAS_PEDANTIC_MATH` = "uses the prescribed precision and standardized arithmetic for all
  phases of calculations and is primarily intended for numerical robustness studies, testing, and debugging" (same doc).
- **"A cuBLAS library context is tightly coupled with the CUDA context that is current at the time of the
  `cublasCreate()` call"** (cuBLAS §2.1.2, same doc) — the load-bearing premise for the device-ordinal scaffold,
  confirmed verbatim.
- **`cudaFree`**: the docs say it "Frees the memory space pointed to by devPtr, which must have been returned by a
  previous call to" an allocator, with **no current-device requirement stated** — confirming that freeing is
  device-independent (a device pointer carries its own device association), whereas `cudaMalloc` "allocate[s] memory on
  the device" (the current device). This asymmetry is the crux of the *refined* 9.2.

---

## Role & layering

The delta keeps the file exactly where §4 puts it (CUDA-private, `#include <cublas_v2.h>`/`<cuda_runtime.h>`, consumed
only behind `steppe_device PRIVATE`). Both additions are pure scaffolding for the **§11.4 multi-GPU pass** and the
**§12 oracle gate** the prior 8.5/10 review explicitly asked for (its items 2.3 and 3.2). They are, by construction,
**parity-neutral**: neither touches the arithmetic, the reduction order, the EmulatedFp64 bit-identity, or the fixed
g=0..G−1 combine. The record-and-assert is observability-only (release-compiled-out, verified against
`host_device.hpp`); `MathModeScope` only *restores* state `engage_f2_precision` was already setting imperatively. **No
finding in this review proposes anything that would change the locked bit-identity contract**, and the one tempting
"fix" that could is captured under Considered & rejected.

The structural problem the two sibling reviews did not have to confront: **the delta ships two abstractions, and
neither is wired into the production path it was built for.** `MathModeScope` has **zero production callers** (only
`test_handles.cu` and its own definition — re-confirmed by grep); `engage_f2_precision` still sets the math mode raw at
`f2_block_kernel.cu:280/299` and never restores, so the determinism hazard the scope documents *is still open in
production*. And `CublasHandle::device_id()` has **zero production readers** — and (the second-pass sharpening) it is
not merely unwired but **architecturally unreachable** from the consumer its comment names: `PerGpuResources` holds the
backend as `std::unique_ptr<ComputeBackend>` (the abstract, CUDA-free interface), so there is no path from the §11.4
orchestration to a concrete `CublasHandle::device_id()` without breaking the CUDA-free layering. (Note: the
architecture.md §9 *reference sketch* at line 613 has `CublasHandle blas;` *directly* in `PerGpuResources`, which would
make `device_id()` reachable — but the *shipped* `resources.hpp` deliberately diverges to stay CUDA-free, so the comment
describes a design that the implementation chose not to build.) So the delta is two correct, well-tested,
well-documented scaffolds that are not yet load-bearing — defensible for a "scaffold" pass, but the gap between this and
a 9.5–10.

---

## Score: 8.5/10 — both additions are correct, idiomatic RAII, well-tested, and honestly documented; held off 9.5–10 by (a) `MathModeScope` shipping unused while the determinism hazard it closes stays open in `engage_f2_precision`; (b) a public `device_id()` accessor with no production reader AND that is architecturally unreachable from its documented consumer; (c) a real ctor `cudaGetDevice` sticky-error-laundering subtlety; (d) the new `MathModeScope` mutating cuBLAS context state WITHOUT the device-ordinal guard the same delta just added to `CublasHandle` (a delta-internal asymmetry, N4); and (e) the single-wrapper device-ordinal scaffold leaving its `DeviceBuffer`/`Stream`/`Event` siblings — which have a *related but not identical* device-binding hazard — asymmetric.

The mechanics are right: the move operations carry the new members; the assert is correctly debug-only, query-only, and
never `cudaSetDevice` (honoring §7's no-hidden-global-mutable-state rule); `MathModeScope`'s ctor-may-throw /
dtor-never-throws split is exactly the §7 contract; the moved-from-is-inert semantics are correct and *tested*. The
documentation is dense and cited. This is solidly the same 8.5 tier as its two M4.5 siblings, for the same reason:
clean, correct, well-justified scaffolding that stops short of the senior bar on wiring / asymmetry / two subtleties,
none of it parity-affecting. The second pass does **not** move the number, but it adds N4 (a delta-internal defect, not
just a wiring gap), strengthens 7.2 to "unreachable," and corrects the 9.2 framing.

---

## Findings

### (8) Performance — first-class this pass

**P1 [GOOD / verified] The device-ordinal assert is NOT on any hot path; cost is two debug-only `cudaGetDevice` queries per backend lifetime.**
Location: `assert_on_creation_device()` (147–159), called from `set_workspace()` (111) and `set_stream()` (125). I
re-grepped every call site: in production they are called **exactly once each, in the `CudaBackend` ctor**
(`cuda_backend.cu:106–107`), never per-`compute_f2`, never per-iteration, never per-GEMM. So the assert fires at most
twice per backend construction, in debug only. `cudaGetDevice` is a lightweight thread-local current-device query
([CUDA Device Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)), and under NDEBUG
the whole body is `((void)0)` with the argument unevaluated (verified `host_device.hpp:64`). No release cost, no
hot-path cost even in debug. The *correct* placement. **PARITY-SAFE: yes.** Effort: N/A.

**P2 [LOW] The ctor adds a `cudaGetDevice` sequenced before `cublasCreate` — unavoidable (no API to read a `cublasHandle_t`'s bound device) and free.**
Location: ctor (77–80). Two serialized runtime calls where there was one; `cudaGetDevice` is ~free (thread-local read)
and the handle is created once at startup (§7). cuBLAS exposes `cublasGetStream` but **not** a device-of-context query,
so capturing the ordinal at the ctor via `cudaGetDevice` is the *only* mechanism — the right and unavoidable design,
not a redundant query. No change. **PARITY-SAFE: yes.** Severity: low. Effort: N/A.

**P3 [GOOD / N-A] `MathModeScope` is free on the determinism path it is *meant* for and would not regress throughput if wired.**
The intended consumer (§12 mandatory gate: "recompute the covariance for a sample of jackknife blocks in native `Fp64`
… `cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)`") runs on a *sampled* subset of blocks, not the hot precompute. Two
`cublasGetMathMode`/`cublasSetMathMode` calls per scope are trivial next to the GEMMs they bracket. No allocation, no
sync, no copy. Even when wired (F1) there is no perf concern — this is a correctness/wiring item (3.1/F1), not a perf
item. **PARITY-SAFE: yes.**

**P4 [N-A with reason] No data-bouncing / no missing grid-stride / no sequential-P2P / no type-casting-noise in this unit.**
This header owns library-handle *state*, not kernels or transfers: **no kernels** (so no grid-stride loops to add — the
perf-relevant grid-stride work is in `f2_*_kernel.cu`), **no `cudaMemcpy`/`cudaMemcpyPeer`** (so no data-bouncing or
sequential-transfer-overlap to fix — that lives in `p2p_combine.cu`, a separate unit), and **no
`int`↔`long`↔`size_t` casts** (the only integral member is `int device_id_`, matching `cudaGetDevice`'s `int*`
out-param and the backend's `int device_id_` exactly — no width mismatch, no narrowing). The §7 "create-once"
discipline means no allocations on a hot path. The performance axis is genuinely **N/A for this unit's additions**; the
perf-relevant M4.5 work (P2P combine fan-in, pinned staging, transfer/compute overlap, sequential-`cudaMemcpyPeer`
pipelining) is in the combine units. Recorded for completeness per the mandate. **PARITY-SAFE: yes.** Severity: low.

### (1) Correctness & bugs

**1.1 [MED] The ctor's `STEPPE_CUDA_CHECK(cudaGetDevice(&device_id_))` can THROW on a *previous* asynchronous fault unrelated to this handle — laundering a sticky error into a `CudaError` mis-attributed to handle construction, even in release.** *(CONFIRMED against the runtime docs this pass.)*
Location: ctor, line 78. The CUDA Runtime docs are explicit that `cudaGetDevice` "may also return error codes from
previous, asynchronous launches"
([CUDA Device Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)). So a kernel fault
*before* this handle was constructed can be the sticky error `cudaGetDevice` returns, and `STEPPE_CUDA_CHECK` throws a
`CudaError` whose message points at `cudaGetDevice(&device_id_)` inside `CublasHandle()` — attributing an unrelated
prior fault to handle construction. `cublasCreate` (next line) has the same sticky-error exposure, so the *net new*
risk is small, but it is real and it is in *release* (unlike the debug-only assert). Two narrowing points keep this MED
not HIGH: (a) in production the backend ctor calls `set_and_return_device` → `cudaSetDevice` (`cuda_backend.cu:570`)
*immediately before* `blas_` constructs — and `cudaSetDevice` carries the *same* "may return prior async errors" note,
so a pending sticky error would already have surfaced there, on a call that *is* a legitimate fault attribution point;
(b) a sticky error genuinely *should* surface somewhere — throwing is not wrong, only *mis-located*.
Why it matters: §7's `STEPPE_CUDA_CHECK` is for *fault* calls whose failure is unrecoverable and *attributable*; using
it on a query that can return a stale unrelated status muddies the §10 diagnostic.
Concrete fix (two small options): (a) leave it and add a one-line note that this can rethrow a *prior* fault (cheapest;
the message already carries the call site); or (b) capture the ordinal with the non-throwing
`detail::cuda_warn(cudaGetDevice(&device_id_), …)` and let `cublasCreate` be the authoritative fail — but that loses
fail-fast on a genuinely broken device. I lean (a)+comment. **PARITY-SAFE: yes.** Severity: med. Effort: S.

**1.2 [LOW] The `device_id_ = -1` sentinel comment "never observed" is slightly too strong vs the throwing-ctor case.** *(CONFIRMED; doc-precision nit.)*
Location: line 187 comment ("the -1 is a defensive sentinel never observed: the ctor always sets it before any use, and
the body throws if cudaGetDevice fails"). If `cudaGetDevice` throws (1.1), the partially-constructed object is abandoned
(no destructor runs for a ctor that throws before completing — correct), so `-1` is indeed never *read*. The comment is
therefore *correct* but invites the reader to wonder; "never observed on a successfully-constructed handle; if
`cudaGetDevice` throws, the object never completes construction and is never used" would be airtight. Pure
documentation precision. **PARITY-SAFE: yes.** Severity: low. Effort: S.

**1.3 [GOOD] The move operations correctly carry `device_id_` and `MathModeScope::prev_`; move-from leaves the source inert; self-assignment is guarded.** *(VERIFIED line by line.)*
`CublasHandle` move-ctor (82–86) and move-assign (88–97) both copy `device_id_` by value (an `int`); move-assign
`destroy()`s first, then steals, self-guarded. `MathModeScope` move-ctor (228–229) `std::exchange(o.h_, nullptr)` +
copies `prev_`; move-assign (231–238) self-guards, `restore()`s the current, then steals. A moved-from `MathModeScope`
has `h_ == nullptr` so its `restore()` is a no-op — the single-restore invariant. Correct. No change. (But see N3-test:
the test does not *guarantee* it exercises the move-ctor path.)

**1.4 [GOOD] `MathModeScope` ctor-throws / dtor-never-throws is the right §7 split, matching `CublasHandle`'s teardown discipline.** *(VERIFIED.)*
Ctor (223–226) uses `CUBLAS_CHECK` (throws `CublasError` on get/set failure — a ctor may throw); `restore()` (246–259)
routes a nonzero `cublasSetMathMode` to `STEPPE_LOG_WARN` and is `noexcept`-in-practice (called only from `~MathModeScope()`/
move-assign, never throws). `CublasError::status_name` is `static … noexcept` (verified `check.cuh:84`), so the warn
path cannot throw. Identical pattern to `CublasHandle::destroy()`. Correct. No change.

**N2 [LOW — NEW this pass] A moved-from `CublasHandle` retains a STALE `device_id_`, so `assert_on_creation_device()` on it checks against the dead ordinal before the null-handle cuBLAS call fails.** *(NEW; CONFIRMED line by line.)*
Location: move-ctor line 86 / move-assign line 94 copy `device_id_` by value and do **not** reset the source's
`device_id_` (only `h_`/`ws_`/`ws_bytes_` are `std::exchange`d to null/0). So a moved-from handle has `h_ == nullptr`
but `device_id_ == <source ordinal>`. If a caller then (incorrectly) calls `set_stream()`/`set_workspace()` on the
moved-from handle, `assert_on_creation_device()` runs `STEPPE_ASSERT(current == device_id_, …)` against the *stale*
ordinal — which can spuriously *pass* (if the same device is current) or *fail-abort* (if not) before the
`cublasSetStream(nullptr, …)` would have returned `CUBLAS_STATUS_NOT_INITIALIZED` and thrown a clean `CublasError`. The
net effect: using a moved-from handle is already a contract violation and always fails, but the *first* thing it does
(in debug) is assert against a meaningless ordinal rather than reach the authoritative null-handle fault. This is
benign in practice (the test never does it; production never does it) but it is a small RAII-correctness smell: the
moved-from object's `device_id_` is dead state that the assert nonetheless trusts. Fix (optional): `std::exchange(o.device_id_, -1)`
in both moves, so a moved-from handle's assert compares against the `-1` sentinel (which can never match a real current
device) and fails loudly+consistently — matching the "moved-from owns nothing" contract for *all* members, not just the
pointer ones. **PARITY-SAFE: yes** (moved-from handles never run on a parity path). Severity: low. Effort: S.

### (2) Edge cases & failure modes

**2.1 [LOW] `MathModeScope` over a moved-from / null `CublasHandle` calls `cublasGetMathMode(nullptr, …)` and throws — acceptable (caller contract), but undocumented.** *(CONFIRMED; the docs did not pin the exact null-handle status, but cuBLAS returns `CUBLAS_STATUS_NOT_INITIALIZED` for an uninitialized handle — standard behavior.)*
Location: ctor (223). `MathModeScope` takes a raw `cublasHandle_t` and immediately calls `cublasGetMathMode(h_, &prev_)`.
If a caller passes `get()` from a moved-from `CublasHandle` (`h_ == nullptr`), the ctor throws — a clean fail, not UB,
and the test constructs scopes only over a live handle. The class doc does not state the precondition "the handle must
be live (non-moved-from)." A one-line precondition note (mirroring `CublasHandle`'s "moved-from owns nothing") closes
it. **PARITY-SAFE: yes.** Severity: low. Effort: S.

**2.2 [LOW] The debug assert downgrades a `cudaGetDevice` failure to `STEPPE_ASSERT` (abort), not a throw — correct for a debug-only `noexcept` path.** *(CONFIRMED; intended.)*
Location: `assert_on_creation_device()` 152–154. The query failure is asserted, not thrown, so in debug a failing query
SIGABRTs rather than raising `CudaError`. This is intentional and correct: the method is `noexcept` (it must not throw),
and the whole block is debug-only, so an abort under a debugger is the right fail-fast. No change. **PARITY-SAFE: yes.**
Severity: low.

**2.3 [N-A] Empty/zero/negative/overflow inputs: not applicable to this unit.** No extents, sizes, counts, or
arithmetic here — `device_id_` is an opaque ordinal echoed from `cudaGetDevice`, `prev_` is an opaque enum. The
integer-width / wraparound concerns live in the kernel/transfer units (`compute_f2`'s `M > INT_MAX` guard at
`cuda_backend.cu:157`, the z-extent tiling). The one degenerate M4.5 case — an empty SNP shard — is handled in
`CudaBackend::compute_f2`/`compute_f2_blocks` (`P<=0 || M<=0` early return, `cuda_backend.cu:137`), not in the handle.
N/A with reason.

### (3) Numerical / precision vs §12

**3.1 [MED] `MathModeScope` closes a real §12 determinism hazard — but it is UNUSED, so the hazard is still open in production. The headline finding.** *(CONFIRMED by grep: zero production callers.)*
Location: `MathModeScope` (218–263) vs `engage_f2_precision` (`f2_block_kernel.cu:278–301`). The class doc (193–210)
names the exact hazard: `cublasSetMathMode` is sticky and is **not** reset by `cublasSetStream` (cuBLAS §2.4.7,
re-confirmed — only the workspace is reset), so when the §12 oracle gate recomputes a sample of jackknife blocks in
native `Fp64` (PEDANTIC) on the **same shared handle** an `EmulatedFp64` run uses, "whichever ran last silently leaks
its math mode into the next." `MathModeScope` is the fix. But: the only references to `MathModeScope` are its own
definition and `test_handles.cu`; `engage_f2_precision` still calls raw `cublasSetMathMode(handle, …)` (lines 280, 299)
and **never restores**, and it is invoked raw at `cuda_backend.cu:339` and `f2_block_kernel.cu:337`. So the leak the
scope is built to prevent is *still present in the production code path* the moment an oracle-recompute and an
EmulatedFp64 run share a handle.
Why it matters: §12 names this as the *mandatory* gate every release runs; the scaffold without the wiring documents a
hazard it does not actually close in the shipping path. The prior review's 3.2 asked precisely for this to be *owned*;
the delta built the owner but left the imperative setter in place.
Concrete fix (parity-NEUTRAL, see Considered & rejected): have `engage_f2_precision` *return* a `MathModeScope` (so the
math mode is *always* scoped, never leaked), OR add `CublasHandle::engage_precision(const Precision&) -> MathModeScope`
and route the §12 oracle recompute through it. Because the oracle gate is not itself implemented yet (M4.5 *scaffold*
pass), this is MED not HIGH — but a 9.5–10 submission does not add the lock without the door it locks.
**PARITY-SAFE: yes** (restoring a mode that was being set imperatively cannot change any computed value; it only
prevents a *future* leak — see the explicit parity analysis in Considered & rejected). Severity: med. Effort: M.

**3.2 [GOOD] The scaffold does not, and cannot, perturb the locked bit-identity.** *(VERIFIED against §12.)*
Neither addition touches accumulation order, the EmulatedFp64 slice count, the fixed g=0..G−1 combine, or any GEMM. The
assert is release-compiled-out; `MathModeScope` only *restores* prior state. The bit-identity GATE (host-staged + P2P,
EmulatedFp64{40}, real AADR) is provably unaffected. No change.

### (4) CUDA idioms / RAII / stream & async / launch config vs §7

**4.1 [GOOD] `MathModeScope` is the §7 RAII shape verbatim — move-construct + move-assign + non-throwing `restore()` + deleted copy.** *(VERIFIED.)* Mirrors `CublasHandle`/`Stream`/`DeviceBuffer` exactly, including the
move-assign-with-self-guard §7 calls out as the historically-buggy one. The non-owning `h_` comment (261) correctly
states "the `CublasHandle` owns the context." Idiomatic. No change.

**4.2 [GOOD] Record-and-assert, never `cudaSetDevice` — exactly §7.** *(VERIFIED.)* `assert_on_creation_device()` only
*queries* (`cudaGetDevice`) and *checks*; it never mutates the current device. The class doc (34–37) and the method doc
(143–146) both state this loudly: "selecting the device is the caller's / `Resources`' job." This is the §7 "no hidden
global mutable state in wrappers" rule honored, and the *reason* the wrapper is safe to hold per-device in
`PerGpuResources`. The backend's `guard_device()` (`cuda_backend.cu:561`) is the legitimate selector; the assert
validates the backend did its job. Correct division of labor. No change.

**4.3 [GOOD / no issue] `assert_on_creation_device()` is host-only despite `cudaGetDevice` being `__host__ __device__`.** *(CONFIRMED no defect.)*
`handles.hpp` is nvcc-compiled. `cudaGetDevice` is `__host__ __device__`. `assert_on_creation_device()` is a plain
(host) member with no `__device__` annotation, called only from host code (`set_stream`/`set_workspace`), so it is only
instantiated for the host — the `__host__ __device__` annotation on `cudaGetDevice` does not force a device
instantiation. No defect; listed for the adversarial record. No change.

**4.4 [GOOD] The new `#include <cuda_runtime.h>` is correct and minimal.** `cudaGetDevice`, `cudaError_t`,
`cudaSuccess` all come from `<cuda_runtime.h>`; previously pulled transitively via `check.cuh`, but including it
directly with the IWYU comment (line 49) is the right hygiene. No change.

**N4 [MED — NEW this pass, the strongest new finding] `MathModeScope` mutates cuBLAS context state (`cublasGetMathMode`/`cublasSetMathMode`) but has NO device-ordinal record-and-assert — the EXACT guard the same delta just added to `CublasHandle::set_stream`/`set_workspace`. The delta closes the wrong-device hazard on one context-mutating surface and OPENS it on the new one.** *(NEW; CONFIRMED line by line — `MathModeScope` carries only `h_`/`prev_`, no `device_id_`.)*
Location: `MathModeScope` ctor (223–226), `restore()` (251–257). The delta's stated thesis (handles.hpp 26–39) is that
"every use that mutates the cuBLAS context debug-ASSERTs the current device still matches" the creation device, because
"a cuBLAS library context is tightly coupled with the CUDA context current at `cublasCreate`" (cuBLAS §2.1.2,
confirmed). `cublasSetMathMode`/`cublasGetMathMode` are *exactly* such context-mutating/context-querying calls — they
configure the handle's context. Yet `MathModeScope` takes a *raw* `cublasHandle_t`, stores no device ordinal, and runs
`cublasGetMathMode`+`cublasSetMathMode` in its ctor and `cublasSetMathMode` in `restore()` **with no check that the
creation device is current**. In the very §11.4 multi-GPU world this delta is scaffolding for — where the §12 oracle
recompute (the documented `MathModeScope` consumer) runs per-device and `cudaSetDevice` switches between handles — a
`MathModeScope` constructed (or, worse, *destroyed* via `restore()` at scope exit) while a *different* device is current
is precisely the wrong-device cuBLAS bug `CublasHandle` now guards against. The dtor case is the nastier one: the scope
may have been created with device g current, then a `cudaSetDevice(g')` happens, then the scope's `restore()` fires
`cublasSetMathMode` against handle-g while g' is current — and because `restore()` is `noexcept` and routes failures to
a *debug-only* warn, a wrong-device restore could even be silently swallowed in release.
Why it matters: this is a **delta-internal consistency defect**, not merely a missing future wiring — the same author,
in the same diff, established the "guard every context mutation with the device-ordinal assert" rule for `CublasHandle`
and then violated it for the new `MathModeScope`. §2 (DRY/consistency) and the delta's own §11.4 thesis demand the guard
on both context-mutating surfaces.
Concrete fix (parity-SAFE, debug-only): give `MathModeScope` the same `assert_on_creation_device()` discipline — either
(a) take a `CublasHandle&` (which carries `device_id_`) instead of (or in addition to) the raw handle, and assert in
ctor + `restore()` that the recorded ordinal is current (this is the *better* design and dovetails with the
`engage_precision(const Precision&) -> MathModeScope` consolidation of F1/3.1); or (b) record `cudaGetDevice` at the
`MathModeScope` ctor and debug-assert it in `restore()`. The raw-handle ctor can remain for composition, but the
device-current precondition should be asserted on the same debug terms `CublasHandle` uses. **PARITY-SAFE: yes**
(debug-only assert, release no-op; and the only consumer is the off-hot-path oracle gate). Severity: med (it is the
single inconsistency *inside* the delta — the delta's whole premise applied to one new surface and not the other).
Effort: S–M (S for option (b), M if folded into the `CublasHandle&`/`engage_precision` consolidation).

### (5) Magic numbers & hardcoded values vs §4

**5.1 [GOOD / N-A] No magic numbers introduced.** The only literals in the delta are `-1` (the documented
device-ordinal sentinel — a true sentinel, not a tunable), `nullptr`/`0` in the move members, and `CUBLAS_DEFAULT_MATH`
(a vendor enum, the defensive default for `prev_`). No hardcoded device count, workspace size, or tier threshold. §4
satisfied. No change.

### (6) Decomposition / single-responsibility / function size vs §2

**6.1 [GOOD] Both additions are tiny and single-purpose.** `assert_on_creation_device()` is ~6 lines of body;
`MathModeScope` is one class with ctor / two moves / dtor / one private `restore()`, nothing over ~10 lines. The §2
decomposition target. No change.

**6.2 [LOW] `MathModeScope` lives in `handles.hpp` but its only documented composition target (`engage_f2_precision`) lives in `f2_block_kernel.cu` — the math-mode policy is split across two files with no wiring between them (the §8 single-home smell).** *(CONFIRMED; pre-existing, now sharper.)*
The class doc (215–217) says it "composes with `engage_f2_precision`, which already operates on the raw handle." But
`engage_f2_precision` is in a *kernel* file and is the thing that *should* be returning or using the scope (3.1). So the
math-mode policy has: the *setter* in `f2_block_kernel.cu`, the *scoped restorer* in `handles.hpp`, and *no wiring*. The
prior review's 3.2/4.3 recommended consolidating behind `CublasHandle::engage_precision(const Precision&)`;
`MathModeScope` is the right primitive but the consolidation was not done. Not a defect in `MathModeScope` per se — a
placement/SRP observation about the delta. **PARITY-SAFE: yes.** Severity: low. Effort: M (folds into 3.1's wiring and
N4's `CublasHandle&` ctor).

### (7) Readability, naming, const-correctness, [[nodiscard]] / noexcept, comments

**7.1 [GOOD] `[[nodiscard]]` / `noexcept` placement is correct on the new surface.** `device_id()` is `[[nodiscard]] …
const noexcept` (pure accessor); `assert_on_creation_device()` is `const noexcept` (a checker that must not throw); the
moves are `noexcept`; `MathModeScope`'s ctor is correctly NOT `noexcept` (it throws). No change.

**7.2 [MED — strengthened this pass] `device_id()`'s comment over-claims a consumer that does not exist AND is architecturally UNREACHABLE.** *(CONFIRMED by grep — zero production readers; STRENGTHENED via `resources.hpp`.)*
Location: 132–136. No `.device_id()` read exists in `src/` outside `test_handles.cu`. The comment says "the §11.4
multi-GPU orchestration uses it to log which tier each per-device handle is on and to assert the right device is current
before use," but: (a) the "assert the right device is current" part is done by the *private*
`assert_on_creation_device()`, which does not need the public accessor; (b) the "log which tier" part is done by
`PerGpuResources::caps` + `PerGpuResources::device_id` (`resources.hpp:76,87`), sourced from `DeviceConfig::devices[g]`
— not from `CublasHandle::device_id()`. **The second-pass sharpening:** `PerGpuResources` holds the backend as
`std::unique_ptr<ComputeBackend>` (the *abstract* CUDA-free interface, `resources.hpp:81`), so the §11.4 layer
*cannot* reach a concrete `CublasHandle::device_id()` at all without breaking the CUDA-free layering. So this is not
"not yet wired" — it is a public accessor whose documented consumer is *structurally impossible* in the shipped design.
(The architecture.md §9 sketch at line 613 puts `CublasHandle blas;` directly in `PerGpuResources`, which would make it
reachable — but the implementation deliberately diverged to stay CUDA-free, and the comment was not updated to match.)
Fix: (a) trim the comment to the truth ("exposed for diagnostics / the unit test's cross-check; not consumed in
production — `PerGpuResources` tracks its own `device_id` and cannot reach this through the abstract backend"), or (b)
drop the public accessor entirely until a reader exists (the private assert does not need it; the test could read
`device_id_` via a friend or be content asserting `set_stream` doesn't abort). A 9.5–10 header does not ship a public
method whose documented consumer is fictional *and* unreachable. **PARITY-SAFE: yes.** Severity: med (doc-vs-reality
drift on a public API surface — the exact thing this audit exists to catch). Effort: S.

**7.3 [LOW] The `MathModeScope` class doc (193–210) is excellent but ~18 lines for a ~45-line class, and it re-states the §2.4.7 "only the workspace is reset" cuBLAS fact that already appears in the `CublasHandle` header comment (16–18) and in `engage_f2_precision`'s comment — three homes for one fact (§8 single-home).** *(CONFIRMED; style, not defect.)*
The comment is accurate, cited, and justifies a non-obvious hazard, so this is not "delete it." But the thrice-stated
cuBLAS fact is a §8 single-home miss; stating it once (a property of cuBLAS — arguably `check.cuh`'s cuBLAS notes or the
§12 doc) and cross-referencing would be cleaner. Minor. **PARITY-SAFE: yes.** Severity: low. Effort: S.

**7.4 [GOOD / no change] Naming: `MathModeScope`, `prev_`, `requested` are clear and consistent with house style.**
`requested` (223) is fine and consistent with the sibling wrappers' terseness. The prior review's 7.4 explicitly says
do not rename in isolation for consistency. No change. Listed for completeness.

**7.5 [GOOD] The "moved-from owns nothing" sentence the prior review asked for is present on both classes.**
`CublasHandle` (67–68) and `MathModeScope` (213–214, "A moved-from scope is inert (it restores nothing)") both carry it.
A prior finding closed. No change. (N2 notes the one member — `device_id_` — that is *not* reset on move, which is the
only crack in the "owns nothing" story.)

**N1 [LOW — NEW this pass] The `set_workspace` doc (107–109) gives an internally-inconsistent rationale for the declaration order — it says the workspace "must outlive the handle's use of it … so the backend declares the handle BEFORE the workspace buffer," but those two clauses imply OPPOSITE orderings.** *(NEW; CONFIRMED against `cuda_backend.cu:591–598`. Pre-existing core comment, but in-file and load-bearing to a delta cross-file claim the first pass said it verified.)*
Location: lines 107–109. "must outlive the handle's use of it" implies the workspace buffer should be destroyed *after*
the handle, i.e. declared *before* it (earlier members are destroyed last under reverse-order destruction). But the next
clause says "declares the handle BEFORE the workspace buffer" — which destroys the workspace *first*. The two clauses
contradict. The *actual* backend (`cuda_backend.cu:591–595`) gets the order right and for the right reason: `blas_`
(597) before `workspace_` (598) → workspace freed first → and that is *safe specifically because `cublasDestroy` does
not read the workspace* (it only synchronizes), **not** because the workspace "outlives the handle's use." So the
handles.hpp comment reaches the correct declaration order via incorrect reasoning ("must outlive"), while the backend
comment reaches it via the correct reasoning ("cublasDestroy doesn't read it"). A reader who trusts the handles.hpp
rationale would conclude the *opposite* declaration order is required. Fix: align the handles.hpp comment with the
backend's correct rationale — "the buffer is non-owning; `cublasDestroy` only synchronizes and does not read the
workspace, so the backend may declare/destroy the workspace before the handle (it does: `blas_` before `workspace_`)."
**PARITY-SAFE: yes.** Severity: low. Effort: S.

### (8) Performance

Covered first-class above (P1–P4). Summary: the device-ordinal assert is debug-only and off the hot path (P1,
verified); the ctor query is unavoidable and free (P2); `MathModeScope` is free on its intended sampled path (P3); no
data-bouncing / grid-stride / sequential-P2P / cast-noise surface in this unit (P4, N/A with reason). **No performance
regression and no missed performance opportunity in the delta.**

### (9) Layering / API / ABI vs §4

**9.1 [GOOD] The CUDA-private boundary is intact; the combine policy is NOT in this layer.** *(VERIFIED.)* The delta
adds only CUDA-toolkit symbols (`cudaGetDevice`, `cublasGetMathMode`/`cublasSetMathMode`, `cublasMath_t`) behind the
existing `<cublas_v2.h>`/`<cuda_runtime.h>` includes — `core`/`api`/`app` still cannot and do not include this header
(the combine *policy* correctly lives in `core/fstats/f2_blocks_multigpu.cpp`). The device-ordinal machinery is the
right layer (a property of a cuBLAS handle, which is device-private). No change.

**9.2 [LOW — REFINED this pass] The device-ordinal scaffold landed on `CublasHandle` ONLY — its `DeviceBuffer`/`Stream`/`Event` siblings have a RELATED device-binding hazard but NOT an identical one, so the asymmetry is real but milder than the first pass framed.** *(REFINED; the "same class of hazard" framing was too strong.)*
Location: `device_buffer.cuh` / `stream.hpp` (verified: neither records or asserts a creation device). The first pass
called this "the same class of multi-GPU bug." The docs say otherwise, and the distinction matters:
- **`cudaMalloc` allocates on the *current* device** — so a `DeviceBuffer` *constructed* while the wrong device is
  current allocates on the wrong GPU. That is a real hazard, **but it bites at construction**, where the backend's
  `cudaSetDevice` (`cuda_backend.cu:570`, sequenced before the buffers build) already makes the right device current —
  so it is guarded *externally*, the same way `CublasHandle`'s creation is.
- **`cudaFree` is device-INDEPENDENT** — the docs say it frees "the memory space pointed to by devPtr … returned by a
  previous call," with no current-device requirement (re-confirmed this pass; a device pointer carries its own device
  association). So `DeviceBuffer::reset()` calling `cudaFree` at teardown while the wrong device is current is **safe** —
  it is *not* a hazard, and a record-and-assert there would guard a non-bug. Same for `cudaStreamDestroy`/
  `cudaEventDestroy`.
- **A `Stream`/`Event` *used in a launch* while a different device is current** *is* a real bug
  (`cudaErrorInvalidResourceHandle`), but `Stream`/`Event` are passed to launches by the backend, which `guard_device()`s
  first — again externally guarded.
So the genuinely-symmetric hazard is narrow: a `CublasHandle` GEMM on the wrong device fails *loudly*
(`CUBLAS_STATUS_ARCH_MISMATCH`) and is the one that empirically bites, which is a *defensible* reason it got the assert
first. The finding survives as a LOW: a senior multi-GPU scaffold either (a) factors a shared `DeviceScoped`
record-and-assert mixin for the *construction/use* surfaces that are actually device-bound (the `cudaMalloc` site, the
launch sites), or (b) documents *why* only `CublasHandle` warrants it (the hard `ARCH_MISMATCH` vs the externally-guarded
softer cases). Right now it is silent asymmetry. **PARITY-SAFE: yes.** Severity: low. Effort: M.

### (10) Testability vs §13

**10.1 [GOOD, with one caveat — see N3-test] `test_handles.cu` is a strong gate for both additions.** *(VERIFIED line by line.)*
It pins (1) capture+restore (94–122), (2) **nesting** — inner restores to outer, proving the ctor captures the *live*
mode not a default (124–141), (3) **move-leaves-moved-from-inert** — restore fires exactly once (143–163), and (4)
device_id() == creation device + the record-and-assert does not abort on the matching device (175–202). It fails-fast
(not "no-GPU PASS") if there is no device (210–216). These are the properties that matter, and the nesting case in
particular is the kind a shallow test skips. Good.

**N3-test [LOW — NEW this pass] The move test (case 3, 148–163) relies on `return src;` from a lambda, which is NRVO-eligible — so it does NOT *guarantee* it exercises the move-ctor's inertness; under copy-elision no move construction happens.** *(NEW; the first pass praised this case as "proves the restore fires exactly once" without noting the elision.)*
Location: test_handles.cu 149–152: `MathModeScope moved_into = [&]{ MathModeScope src(...); return src; }();`. `return src;`
where `src` is a named local of the function's return type is **NRVO-eligible** (C++17+ permits, does not mandate,
copy-elision of a named return). If NRVO fires, `src` is constructed *directly* in `moved_into`'s storage and **no
move-constructor runs** — so the property the test claims to pin (a *moved-from* `MathModeScope` restores nothing) is
*not exercised* on that build; only the in-place single-construction is. The observable result (mode restored exactly
once) is identical either way, so the test still *passes* — but it does not *prove* the move-ctor inertness it is
documented to prove, because the compiler may have elided the move. To force the move-ctor path, use an explicit
`std::move` through a non-elidable seam — e.g. construct `src`, then `MathModeScope moved_into(std::move(src));` in the
same scope with `src` outliving to its own dtor, asserting the restore fires only at `moved_into`'s dtor. (The
move-assign path is separately untested altogether — the test only exercises the move-*ctor* seam.) **PARITY-SAFE: yes.**
Severity: low. Effort: S.

**10.2 [LOW] The assert's *failure* path (a handle used on the WRONG device) is untested — the single-GPU box only exercises the always-true case; the multi-GPU negative case is unpinned. On the 2×5090 box it IS testable.** *(CONFIRMED; inherent on 1-GPU, available on the box.)*
Location: `expect_device_ordinal` (175–202) only proves the *matching* device passes. The whole point of the assert is
to SIGABRT on a *mismatch*; that path is untestable on one GPU (and a passing assert is a no-op, a failing one aborts —
a death-test). On the current 2×5090 box it *is* testable: construct on device 0, `cudaSetDevice(1)`, call `set_stream`
in a forked-child death-test, expect SIGABRT. Not done. A multi-GPU pass that ships a multi-GPU assert and only tests
the always-true single-GPU case has not proven the assert *fires*. (Note: the box's P2P is driver-disabled, but the
device-ordinal assert is independent of P2P — the death-test only needs two devices, which the 5090 box has.)
**PARITY-SAFE: yes.** Severity: low. Effort: M.

**10.3 [LOW] `MathModeScope`'s production property (does the §12 oracle recompute leave the handle's EmulatedFp64 mode intact?) is untestable until wired (3.1).** *(CONFIRMED; cross-ref 3.1.)*
The unit test proves the *primitive* restores. The *property that matters* — "after the oracle's PEDANTIC recompute, the
surrounding EmulatedFp64 run still sees its emulated math mode" — has no test because the production wiring does not
exist (3.1). Adding the wiring (F1) creates the seam: a GPU test that engages EmulatedFp64, runs an oracle recompute
inside a scope, asserts the handle's mode is back to emulated. **PARITY-SAFE: yes.** Severity: low. Effort: M (enabled
by 3.1).

### (11) Capability-tier coherence

**11.1 [GOOD] The tag is correctly OFF the numeric payload.** *(VERIFIED.)* The device-ordinal and math-mode state are
handle properties; neither is stamped onto `F2BlockTensor` or any result. The capability *tier* (`can_access_peer`,
`emulated_fp64_honorable`) lives in `BackendCapabilities` / `PerGpuResources::caps` (out-of-band,
`resources.hpp:83–87`). This header does not pollute the numeric artifact. Correct separation. No change.

**11.2 [LOW] `device_id()` was *intended* (per its comment) to feed §11.4 capability-tier logging but does not, and cannot (7.2) — so the capability-tier wiring this accessor was added for is incomplete and structurally unreachable.** *(Cross-ref 7.2; CONFIRMED + strengthened.)*
The capability-tier facet of 7.2: the comment ties `device_id()` to tier logging, but the tier logging reads
`PerGpuResources::caps` + `PerGpuResources::device_id`, never `blas_.device_id()` — and cannot, via the abstract
backend. Either trim the claim or drop the accessor. The probe + tagged-degrade + which-path-recording machinery itself
is clean and lives in the right places (resources / combine units); this header's only capability-tier contribution is
the (unused, unreachable) accessor and the (correct, used-by-the-assert) ordinal. **PARITY-SAFE: yes.** Severity: low.
Effort: S.

---

## Considered & rejected (incl. rejected-for-parity)

- **"Have the ctor `cudaSetDevice` to a fixed/expected device before `cublasCreate` so the handle is always on a known
  device."** REJECTED (parity-neutral but architecture-violating): §7 forbids hidden global mutable state in wrappers,
  and `cudaSetDevice` inside a ctor leaves the wrong device current after construction. The record-and-assert design is
  exactly right — the *caller* (`PerGpuResources`/`set_and_return_device`) selects; the handle only verifies.

- **"Move `MathModeScope` into `engage_f2_precision` and have it return the scope — but does that change the math mode
  the GEMMs see and thus break bit-identity?"** REJECTED-as-a-parity-risk, ACCEPTED-as-safe after analysis: returning a
  `MathModeScope` does **not** change the mode *during* the GEMMs (the scope holds the requested mode for its lifetime;
  it restores only at scope *exit*, after the GEMMs). The GEMMs still run under the exact same mode
  `engage_f2_precision` sets today. The only change is the *previous* mode is restored afterward — which cannot affect
  the current run's bytes, only prevent a *future* leak. So F1 is parity-SAFE. Flagged explicitly because "touching the
  math-mode engagement on the parity path" warrants the adversarial check, and it passes.

- **"`device_id()` should return the ordinal via `cublasGetStream` + a device lookup instead of caching it."**
  REJECTED: cuBLAS exposes the *stream* (`cublasGetStream`) but not the *device* a context bound to; there is no API to
  recover the creation device from a `cublasHandle_t` (re-confirmed against the cuBLAS docs this pass). Caching it at the
  ctor via `cudaGetDevice` (P2) is the only mechanism. Keep it.

- **"Make `assert_on_creation_device()` throw (`STEPPE_CUDA_CHECK`) instead of `STEPPE_ASSERT` so a wrong-device use
  fails in release too."** REJECTED: the method is `noexcept` and a wrong-device use is a *programming* bug in the
  orchestration layer, not a runtime fault — §7's record-and-**assert** (debug) is the prescribed shape, and a release
  throw would put a `cudaGetDevice` on the (cold) `set_stream`/`set_workspace` path for a condition the backend's
  `guard_device()` already prevents. The debug assert is the right cost.

- **"Drop the `device_id_ = -1` default since the ctor always sets it."** REJECTED: the in-class default is defensive
  and free; it makes a default-constructed-then-moved-into object well-defined and matches the sibling wrappers' style.
  Keep it (1.2 only asks to tighten the comment; N2 asks to *also* exchange it on move).

- **"`MathModeScope` should take a `CublasHandle&` (type-safe) not a raw `cublasHandle_t`."** PARTIALLY REJECTED, now
  RECOMMENDED-AS-AN-ADDITION (see N4): requiring it would break the documented composition with `engage_f2_precision`
  (which holds the raw handle), so the raw-handle ctor must stay. BUT N4 shows the raw-handle form has *no device-ordinal
  guard*, which the delta's own thesis demands — so an *additional* `CublasHandle&` ctor (carrying `device_id_` for the
  assert) is no longer just a nice-to-have; it is the clean fix for N4 and folds into F1's `engage_precision`
  consolidation. So: keep the raw ctor for composition, *add* the `CublasHandle&` ctor for the guarded path.

- **"Record the device ordinal on `DeviceBuffer`/`Stream`/`Event` too (9.2) — but is that a parity risk?"** NOT a parity
  risk (asserts are observability-only). Re-examined this pass and REFINED, not rejected: a record-and-assert on
  `cudaFree`/`cudaStreamDestroy`/`cudaEventDestroy` would guard a **non-bug** (those are device-independent — confirmed
  against the docs), so a blanket "add it to all three" is *wrong*. The real device-bound surfaces are `cudaMalloc`
  (construction) and the launch sites (use), and those are already externally guarded by the backend's `cudaSetDevice`.
  9.2 survives as a documentation/symmetry LOW, not a "add the assert everywhere" mandate.

- **"The ctor `cudaGetDevice` should be removed and the ordinal read lazily on first `set_stream`."** REJECTED: cuBLAS
  binds the context to the device current at `cublasCreate` (§2.1.2, re-confirmed), so the ordinal MUST be captured at
  construction — a lazy read on first use could observe a *different* current device than the one the context actually
  bound to, defeating the invariant.

- **"`MathModeScope::restore()` should `cublasGetMathMode` and skip the set if `prev_` already equals the current mode,
  to avoid a redundant `cublasSetMathMode`."** REJECTED (not worth it, and risks an extra query): `cublasSetMathMode` is
  a trivial handle-state write; adding a `cublasGetMathMode` to *maybe* skip it trades one cheap write for a query +
  branch, on a cold path. No benefit. Keep the unconditional restore.

## What it takes to reach 10/10

1. **(3.1 / 6.2 / 10.3 — the headline) Wire `MathModeScope` to the path it protects.** Have `engage_f2_precision`
   *return* a `MathModeScope` (math mode always scoped, never leaked), or introduce
   `CublasHandle::engage_precision(const Precision&) -> MathModeScope` and route the §12 oracle recompute through it.
   Add the GPU test (10.3) proving the EmulatedFp64 mode survives an in-scope PEDANTIC recompute. Parity-SAFE. Turns a
   documented-but-inert lock into a closed door.
2. **(N4 — the new delta-internal defect) Give `MathModeScope` the device-ordinal guard the delta added to `CublasHandle`.**
   Add a `CublasHandle&` ctor that carries `device_id_` and debug-asserts the creation device is current in the ctor
   *and* in `restore()` (the dtor case is the dangerous one in a multi-GPU world). This closes the inconsistency where
   the delta guards one context-mutating surface and opens the other. Folds naturally into item 1.
3. **(7.2 / 11.2) Make `device_id()` honest — and reckon with its unreachability.** Trim the comment to the truth (no
   fictional consumer; `PerGpuResources` tracks its own `device_id` and *cannot* reach this through the abstract
   backend), or drop the public accessor until a reader exists (the private assert does not need it). No fictional
   *and unreachable* consumer in a public-API comment.
4. **(1.1 / 1.2 / N1 / N2) Tighten the ctor query and the doc.** Note that `STEPPE_CUDA_CHECK(cudaGetDevice(...))` can
   rethrow a *prior* asynchronous fault mis-attributed to handle construction; tighten the "-1 never observed" comment
   for the throwing-ctor case (1.2); fix the internally-inconsistent `set_workspace` declaration-order rationale to
   match the backend's correct one (N1); and `std::exchange(o.device_id_, -1)` on move so a moved-from handle's assert
   compares against the sentinel, not stale state (N2).
5. **(9.2) Resolve the device-ordinal asymmetry *correctly*.** Either factor a shared `DeviceScoped` record-and-assert
   for the surfaces that are *actually* device-bound (the `cudaMalloc` construction site, the launch sites) — NOT the
   device-independent `cudaFree`/destroy sites — or document why only `CublasHandle` (the loud `ARCH_MISMATCH`) warrants
   it while the others are externally guarded by the backend's `cudaSetDevice`.
6. **(10.2 / N3-test) Pin the assert's *failure* path** on the 2-GPU box with a death-test (construct on device 0,
   switch to device 1, expect SIGABRT), and fix the NRVO-elided move test so the move-ctor inertness is *guaranteed*
   exercised (explicit `std::move` through a non-elidable seam; add a move-*assign* case too).
7. **(2.1 / 7.3) Doc polish:** state `MathModeScope`'s "handle must be live" precondition; collapse the thrice-stated
   "cublasSetStream resets only the workspace" cuBLAS fact to one home (§8) and cross-reference.

Items 1–2 are the must-dos that separate this from its 8.5 siblings (wire the scaffold / close the delta-internal N4
asymmetry). 3 is the honest-public-surface item. 4–6 are the senior-bar correctness/robustness items. 7 is cheap polish.

## Good patterns to keep

- **Record-and-assert, never `cudaSetDevice` (4.2)** — the textbook §7 way to own a per-device invariant in a move-only
  wrapper without hidden global mutable state. The *right* multi-GPU primitive — it just needs to be applied to the new
  context-mutating surface too (N4).
- **Debug-only, off-hot-path assert (P1)** — `STEPPE_DEBUG_ONLY` + `STEPPE_ASSERT`, fired only at the two ctor-time
  `set_*` calls, zero release cost. Correctly *not* on a per-GEMM path.
- **`MathModeScope` ctor-throws / dtor-never-throws (1.4)** and the moved-from-restores-nothing single-restore invariant
  (1.3) — the §7 RAII shape verbatim, and *tested* (modulo the NRVO caveat in N3-test).
- **The strong, adversarial unit test (10.1)** — nesting and move-inertness cases a shallow test would skip; fail-fast
  on no-GPU rather than a false PASS.
- **Tag off the numeric payload (11.1)** — handle state stays handle state; the capability tier lives out-of-band in
  `PerGpuResources::caps`.
- **Honest, cited documentation** — the device-ordinal and math-mode hazards are explained with verbatim cuBLAS
  citations; the prior review's "moved-from owns nothing" ask was picked up (7.5). The doc defects are the over-claim on
  `device_id()` (7.2), the inconsistent `set_workspace` order rationale (N1), and the thrice-stated cuBLAS fact (7.3).
- **Correct layering (9.1)** — CUDA-private, combine policy kept in `core`, no leak into `api`/`app`.

---

Sources (official docs, re-fetched and verified this pass):
- [CUDA Runtime API — Device Management (NVIDIA)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)
  — `cudaGetDevice` signature (`__host__ __device__ cudaError_t cudaGetDevice(int*)`), "Returns which device is
  currently being used," return values `cudaSuccess`/`cudaErrorInvalidValue`, and the load-bearing "Note that this
  function may also return error codes from previous, asynchronous launches" (1.1); `cudaSetDevice` carries the identical
  note.
- [CUDA Runtime API — Memory Management (NVIDIA)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
  — `cudaFree` frees "the memory space pointed to by devPtr … returned by a previous call" with **no current-device
  requirement** (the basis for the refined 9.2); `cudaMalloc` allocates "on the device" (the current device).
- [cuBLAS Library Documentation (NVIDIA, CUDA 13.x)](https://docs.nvidia.com/cuda/cublas/index.html)
  — §2.1.2 ("A cuBLAS library context is tightly coupled with the CUDA context that is current at the time of the
  `cublasCreate()` call" — the device-ordinal premise, verbatim), `cublasMath_t` (`CUBLAS_DEFAULT_MATH` "the default and
  highest-performance mode … Tensor Cores will be used whenever possible"; `CUBLAS_PEDANTIC_MATH` "uses the prescribed
  precision and standardized arithmetic … for numerical robustness studies, testing, and debugging"), §2.4.7
  (`cublasSetStream` "unconditionally resets the cuBLAS library workspace back to the default workspace pool" — resets
  the workspace **only**, not the math mode; the basis for `MathModeScope` being a distinct concern), and
  `cublasSetMathMode`/`cublasGetMathMode`.
- `docs/architecture.md` §2/§4/§7/§9/§11.4/§12/§13 and `docs/cleanup/device-cuda-handles.md` (the pre-M4.5 8.5/10
  baseline whose 2.3 + 3.2 this delta implements).
- Verified in-repo cross-files: `core/internal/host_device.hpp`, `device/cuda/check.cuh`, `device/cuda/cuda_backend.cu`,
  `device/cuda/f2_block_kernel.cu`, `device/resources.hpp`, `device/cuda/device_buffer.cuh`, `device/cuda/stream.hpp`,
  `tests/reference/test_handles.cu`, `tests/CMakeLists.txt`.

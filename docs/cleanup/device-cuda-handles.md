# Review — `device-cuda-handles` (`src/device/cuda/handles.hpp`)

Unit under review: the RAII cuBLAS handle wrapper `steppe::device::CublasHandle`.
Reviewed in full, line by line (the whole file is 87 lines), against `docs/architecture.md`
§2/§4/§7/§8/§9/§11/§12/§13, `docs/ROADMAP.md` §4/§5/§6, `docs/TODO.md` capability tiers, and the
official cuBLAS 13.x / CUDA Runtime docs. **Every device-behavior claim below is cited to the
cuBLAS docs and was re-fetched and verified verbatim during this adversarial pass** (the load-bearing
"`cublasSetStream` resets the workspace" claim in particular — see §2.4.7 quote in 1.1).

This is the ADVERSARIAL second pass over the first-pass draft (8.5/10, 24 findings). I re-verified
every existing finding against the actual code and the official docs, confirmed the real ones, tightened
two that were imprecise, added six new findings the first pass missed (N5 macro-leak, N6 NDEBUG-arg
inconsistency, plus 1.1 strengthened with a second consumer path and a sharper doc citation, the precise
workspace-recommendation correction in 11.1, a `valid()`/observability gap, and an include/`<cublas_v2.h>`
hygiene confirmation), and moved one dismissed candidate into "Considered & rejected." Net verdict is
unchanged in spirit but the score moves to 8.5 with a clearer reason: the file is a correct RAII core,
but it is the natural home for a §12-determinism invariant that its own consumers provably defeat.

Directly-related context read in full and used to verify cross-file claims: `check.cuh` (the
`CUBLAS_CHECK` / `CublasError::status_name` home it depends on — confirmed `status_name` is
`static … noexcept`, so the dtor cannot throw through it), `device_buffer.cuh` and `stream.hpp` (the
sibling RAII wrappers it mirrors — confirmed the duplicated teardown macro and the NDEBUG-branch
inconsistency), `cuda_backend.cu` (its sole consumer — the `blas_{stream_}` / `workspace_` members and
the ctor `cublasSetWorkspace`), `f2_block_kernel.cu` (`engage_f2_precision` + `run_f2_gemms`, which call
`cublasSetStream`), `f2_blocks_kernel.cu` (`run_f2_gemms_group`, the M4 batched path — **also** calls
`cublasSetStream`), and `include/steppe/config.hpp` (`kCublasWorkspaceBytes = 64 MiB`).

## Role & layering

`handles.hpp` owns one CUDA library handle (`cublasHandle_t`) with the move-only,
debug-warn-on-teardown shape mandated by architecture §2 ("RAII everywhere") and §7
("Library handles follow the same fully-movable shape; create them once at startup"). It is a
near-verbatim realization of the §7 reference snippet (architecture.md lines 425–451; I diffed
the two — the production class adds only the typed `CublasError::status_name` teardown text over
the doc's `int(s)`) and the §8 single-home rule for the cuBLAS-handle resource. It is correctly a
CUDA-private header (`#include <cublas_v2.h>` — the v2 API, confirmed correct, not the legacy
`cublas.h`) and so lives behind the `steppe_device` `PRIVATE` link boundary (§4): `core`/`api`/`app`
cannot include it and physically cannot compile against cuBLAS. Its only non-experiment, non-sibling
consumer is `CudaBackend::blas_` in `cuda_backend.cu` (verified by grep: the only hits outside
`experiments/` are `cuda_backend.cu`, plus a comment in `stream.hpp` and the file itself). The
layering placement is exactly right; the header is small and the RAII mechanics are correct.

The findings below are mostly about (a) a real, doc-confirmed footgun the wrapper's *minimal surface*
enables in **both** of its consumer GEMM paths, (b) missing siblings/abstractions the architecture
explicitly names (`CusolverDnHandle`, `PerGpuResources`, capability tagging), and (c) polish (a leaked
macro and an inconsistent NDEBUG branch the first pass missed).

## Score: 8.5/10 — solid, correct RAII core; loses points for an under-abstracted surface that lets a doc-confirmed §12 determinism footgun (`cublasSetStream` discards the emulation workspace) sit one call away in BOTH consumer paths, a doc-vs-code drift on "where the stream is bound," a leaked-and-inconsistent teardown macro, and missing capability-tier/multi-GPU hooks the rest of the codebase will need.

The class is correct in isolation — move semantics, the non-throwing destructor, the
`exit()`→typed-`CUBLAS_CHECK` upgrade over the spike, and the teardown-warning sink are all right and
match the architecture's reference shape almost verbatim. It does not earn 9.5–10 because: the handle is
a *bare* `cublasHandle_t` carrier whose mutable cuBLAS state (stream, workspace, math mode) is
configured entirely *outside* the wrapper in `cuda_backend.cu` / `f2_block_kernel.cu` /
`f2_blocks_kernel.cu`, and that split lets a documented cuBLAS hazard — `cublasSetStream`
**unconditionally resets the workspace to the default pool** (cuBLAS §2.4.7, verified verbatim below) —
sit unguarded and **actually fire before every emulated-FP64 GEMM batch in both the M0 and M4 paths**;
the file-header comment claims a property ("bound to a stream at construction … single statistic
stream") that the sole consumer never exercises (it constructs with a `nullptr` stream and then
re-`cublasSetStream`s per call); the teardown-warning macro is duplicated across the three device
headers, leaks (no `#undef`), and has an inconsistent NDEBUG body; and there is no capability-tier
surface (device-ordinal binding, math-mode encapsulation, a `CusolverDnHandle` sibling §7/§9 explicitly
names) even as scaffolding. None of these are *bugs originating in this file's code*, which is why this
is high-8 not low-7 — but a senior reviewer reading §12's determinism contract will not call this header
"done" until the workspace/stream invariant it enables is owned somewhere, and this header is the
natural home.

---

## Findings

### (1) Correctness & bugs

**1.1 [HIGH] The wrapper's minimal surface enables a documented cuBLAS footgun that fires in BOTH consumer paths: `cublasSetStream` unconditionally resets the user workspace, defeating the §12 emulation-determinism workspace before every GEMM batch.** *(CONFIRMED and STRENGTHENED — the first pass cited only the M0 path; the M4 path defeats it too.)*
Location: `CublasHandle` whole class (it exposes only `get()`); manifests in `cuda_backend.cu:51`
(ctor binds the workspace) vs **two** consumer call sites that then discard it:
`f2_block_kernel.cu:242` (`run_f2_gemms` → `cublasSetStream(handle, stream)`) and
`f2_blocks_kernel.cu:190` (`run_f2_gemms_group`, the M4 strided-batched path → the same
`cublasSetStream(handle, stream)`).
The cuBLAS 13.x docs state, verbatim (re-fetched this pass): **"Additionally this function
unconditionally resets the cuBLAS library workspace back to the default workspace pool"**
(cuBLAS §2.4.7 `cublasSetStream`,
[cuBLAS Library docs](https://docs.nvidia.com/cuda/cublas/index.html)). And for the emulation case:
**"This guarantee no longer holds when multiple CUDA streams are active or fixed-point emulation is
used … To avoid this effect, users can provide workspace via `cublasSetWorkspace()` to meet
fixed-point emulation workspace requirements"** (cuBLAS §2.1.4 Results Reproducibility, same doc).
The `CudaBackend` ctor (`cuda_backend.cu:48–53`) calls `cublasSetWorkspace(blas_.get(), …)` *exactly*
to satisfy that requirement. But the per-call `cublasSetStream` in both GEMM routines runs *after* the
ctor and *before* the GEMMs (`run_f2_gemms`: `cublasSetStream` line 242 → `engage_f2_precision` 243 →
three `cublasGemmEx` 253–268; `run_f2_gemms_group`: `cublasSetStream` 190 → three
`cublasGemmStridedBatchedEx` 203–222). The ctor's determinism workspace is therefore provably reverted
to the default pool before every emulated-FP64 GEMM in production.
Why it matters: this is the precise failure mode §12 names ("emulated-FP64 needs an explicit
`cublasSetWorkspace` workspace"; architecture.md line 31, 71, 723) and a `build()`-time validation §9
promises. Bit-stable goldens (§13, architecture.md line 770) depend on it. The wrapper is implicated
because it is the type that *should* make stream+workspace inseparable — a bare `get()` makes the
order-dependence invisible to the caller.
Concrete fix: give `CublasHandle` a `set_stream(cudaStream_t)` method that re-applies the owned/known
workspace immediately after `cublasSetStream` (store the workspace `void*`+size, or hold a
non-owning `cuda::std::span<std::byte>`), OR an `attach_workspace(span<std::byte>)` that the class
re-installs after any stream change. Either way the *class* owns the invariant "stream change ⇒ re-bind
workspace," not the consumer. Then forbid raw `cublasSetStream(get(), …)` in both `run_f2_gemms` and
`run_f2_gemms_group` (route them through the wrapper). This is the single highest-value change to this
file.
Severity: high. Effort: M. Before M4.5: **yes** — it directly undermines the §12 determinism claim the
M0/M4 paths assert and the goldens depend on, and it is now a two-path defect.

**1.2 [MED] The ctor `stream` argument is dead in the sole consumer (passed `nullptr` by design at M0), so the header's documented "bound to a stream at construction" contract is unexercised.** *(CONFIRMED; framing tightened — there is no UB here, the param is simply dead-by-design.)*
Location: `CublasHandle(cudaStream_t stream = nullptr)` (`handles.hpp:44–47`), exercised by
`cuda_backend.cu:338–339`:
```
cudaStream_t stream_ = nullptr;          // member 0 (default-init to nullptr)
CublasHandle blas_{stream_};             // member 1 (reads stream_ — which IS nullptr)
```
Members are initialized in declaration order, so `stream_` *is* properly initialized (to its
default-member-initializer `nullptr`) before `blas_`'s constructor reads it — there is **no
read-before-init / UB** here (I considered and rejected that stronger claim; see Considered & rejected).
The point is narrower and still valid: `stream_` is `nullptr` *by design* at M0 (the
`cuda_backend.cu:335–337` comment says the default stream suffices until the streaming pipeline lands),
so the ctor's `if (stream) cublasSetStream(...)` branch never fires, and the header's doc line 39–41
("Optionally bound to a stream at construction (single statistic stream on the bit-stable path)")
describes a path the only consumer does not take. Worse, the moment a real RAII `Stream` lands (§11.1,
which the `cuda_backend.cu:337` comment promises), whoever wires it must *also* notice that both GEMM
routines re-`cublasSetStream` every call (1.1), making the ctor binding doubly moot.
Why it matters: a constructor parameter the sole call site cannot meaningfully use is a
readability/maintenance trap (§2 clarity; §7 "create once"). Not a bug in *this* file, but the file's
contract over-promises relative to how it is consumed.
Concrete fix: prefer (b): drop the ctor stream arg in favor of the `set_stream()` method from 1.1 and
delete the "bound at construction" language; (b) composes with 1.1. Alternatively (a) keep the arg but
have the consumer construct the handle *after* its stream is a real value and stop re-binding per GEMM.
Severity: med. Effort: S. Before M4.5: yes (small, and it removes a live trap as Streams land).

**1.3 [LOW] `get()` returns a mutable `cublasHandle_t` from a `const` method, so a `const CublasHandle&` grants full mutation of the underlying cuBLAS context (stream/math-mode/workspace).** *(CONFIRMED — accurate, and unavoidable with the C API.)*
Location: `[[nodiscard]] cublasHandle_t get() const noexcept` (`handles.hpp:64`).
`cublasHandle_t` is an opaque pointer (`typedef struct cublasContext* cublasHandle_t`), and every cuBLAS
mutator (`cublasSetStream`, `cublasSetWorkspace`, `cublasSetMathMode`) takes it by value — so the
`const` on `get()` is nominal: a `const CublasHandle&` can mutate the cuBLAS context. This is the *same*
pattern `DeviceBuffer::data() const` and `Stream::get() const` use, the architecture reference snippet
(§7 line 439) does exactly this, and it is internally consistent — but worth flagging that
`const`-correctness is here a fiction, and combined with 1.1 it means determinism-critical state can be
changed through a `const` reference. Acceptable given the C API; the real mitigation is to not hand out
`get()` for mutation at all (route stream/workspace/math-mode changes through wrapper methods, 1.1/4.3),
leaving `get()` only for the per-call `cublas*` *compute* entry points.
Severity: low. Effort: S (folds into 1.1). Before M4.5: no.

### (2) Edge cases & failure modes

**2.1 [GOOD / N-A] Double-free / self-move / moved-from teardown are all handled correctly.** *(CONFIRMED line by line.)*
Move-ctor (`handles.hpp:49`) `std::exchange(o.h_, nullptr)` leaves the source null; move-assign
(`51–57`) self-assignment guard `if (this != &o)`, then `destroy()` the current handle, then steal —
correct and leak-free. `destroy()` (`67–79`) guards on `if (h_)` and resets `h_ = nullptr`, so a second
destructor call is a no-op. A default- or moved-from handle (`h_ == nullptr`) destroys cleanly with no
cuBLAS call. No edge-case bug here.

**2.2 [LOW] Construction failure surfaces only as a thrown `CublasError` — which is correct RAII, but is a cross-unit note: the API must translate it at `build()` per §9/§10.** *(CONFIRMED; downgraded from MED — it is purely a documentation/cross-unit note, not a defect in this file.)*
Location: ctor `handles.hpp:44–47` + `CudaBackend` member init `cuda_backend.cu:338–340`.
If `cublasCreate` fails (`CUBLAS_STATUS_NOT_INITIALIZED`, `ALLOC_FAILED`), the ctor throws `CublasError`
and `h_` is never set non-null, so RAII is sound (no partial object). The §9/§11.2 contract is that the
backend fails fast at `build()` with a typed status; a cuBLAS init failure here propagates out of
`CudaBackend`'s constructor (via `make_cuda_backend`'s `make_unique`), which `core`/`api` must translate
to `STEPPE_ERR_CUDA_RUNTIME` (§10). That translation is the api's job, not this header's. The header
correctly does **not** mark the ctor `noexcept` (a throwing `noexcept` ctor would `std::terminate`) —
the right call; do not change it.
Severity: low. Effort: S. Before M4.5: no.

**2.3 [LOW] No CUDA-context / device-ordinal capture — a latent multi-GPU hazard for the `PerGpuResources` the architecture mandates day-one (§0, §9 line 604–609, §11.4).** *(CONFIRMED; doc-cited.)*
Location: ctor `handles.hpp:44`. The cuBLAS docs: **"A cuBLAS library context is tightly coupled with
the CUDA context that is current at the time of the `cublasCreate()` call"** (cuBLAS §2.1.2,
[cuBLAS Library docs](https://docs.nvidia.com/cuda/cublas/index.html), re-verified verbatim). §9's
`PerGpuResources` (architecture.md line 604) holds a handle *per device*, and §11.4 switches devices
with `cudaSetDevice`. A `CublasHandle` constructed under the wrong current device binds to the wrong
device's context, and every GEMM through it runs on the wrong GPU or fails with
`CUBLAS_STATUS_ARCH_MISMATCH`. The wrapper has no notion of which device it belongs to, no stored
ordinal to validate against. Fine at M0 (single device); a latent correctness bug the moment §11.4
lands.
Concrete fix: capture `int device_; STEPPE_CUDA_CHECK(cudaGetDevice(&device_));` in the ctor
(record-and-assert, **not** `cudaSetDevice` — §7 forbids hidden global mutable state in wrappers); add a
debug-only assert in a future `set_stream`/used path that the current device matches `device_`.
Severity: low (latent; not exercised until M4.5+ multi-GPU). Effort: S. Before M4.5: optional — do it
when `PerGpuResources` is wired; the hook costs almost nothing now.

### (3) Numerical / precision vs §12

**3.1 [HIGH — cross-listed with 1.1] The handle does not own the workspace, so the §12 determinism precondition for `EmulatedFp64` is split across files and is currently defeated; the workspace lifetime is also not tied to the handle.** *(CONFIRMED.)*
Location: workspace lives as a *separate* `DeviceBuffer<std::byte> workspace_` (`cuda_backend.cu:340`)
bound via raw `cublasSetWorkspace(blas_.get(), …)` in the backend ctor (`cuda_backend.cu:51`). §12 is
explicit (architecture.md line 31/71/723): emulated-FP64 reproducibility needs an explicit
`cublasSetWorkspace` workspace; the cuBLAS docs confirm the same and add that the workspace must be
**"intact between the invocations"** of the multiple kernels a cuBLAS call may launch (cuBLAS §2.4.8,
[cuBLAS Library docs](https://docs.nvidia.com/cuda/cublas/index.html)). Because the workspace is not
part of `CublasHandle`: (a) nothing re-establishes it after the `cublasSetStream` reset (1.1), and
(b) nothing ties the workspace lifetime to the handle lifetime. On (b): destruction order is reverse
declaration order, so `workspace_` (declared last, line 340) is destroyed *before* `blas_` (line 339) —
which is the *correct* order here only because `cublasDestroy` synchronizes first (4.2), but it is
**incidental, not designed**: reorder those two members and you get a teardown where `cublasDestroy`'s
implicit sync runs after the workspace VRAM is already freed. The precision contract §12 makes
load-bearing for parity is therefore structurally fragile.
Concrete fix: make `CublasHandle` optionally *own* (or hold a non-owning span to) its workspace and
re-install it on every stream change (1.1). The §12 belt-and-suspenders `CUBLAS_WORKSPACE_CONFIG=:4096:8`
(architecture.md line 732) is environmental and orthogonal; the *programmatic* workspace must survive
stream binding, which only the class can guarantee.
Severity: high. Effort: M (same change as 1.1). Before M4.5: yes.

**3.2 [LOW] Math-mode + emulation tuning is also handle state set entirely outside the wrapper (`engage_f2_precision`, `f2_block_kernel.cu:201–217`) — but it is NOT reset by `cublasSetStream`, so no active bug; it is a single-home (§8) / SRP (§2) smell.** *(CONFIRMED, including the negative claim.)*
Location: `engage_f2_precision` mutates the handle's math mode (`cublasSetMathMode`) and, under
`STEPPE_HAVE_EMU_TUNING`, the emulation strategy and FIXED mantissa-bit count (`cublasSetEmulationStrategy`
/ `cublasSetFixedPointEmulationMantissaControl` / `…MaxMantissaBitCount`). Like the workspace and stream,
this is determinism/precision-critical handle state the wrapper neither owns nor guards. Unlike the
workspace, the math mode is **not** reset by `cublasSetStream` — the §2.4.7 doc text resets only the
*workspace*, not the math mode (verified by reading the section; no mention of math-mode reset). So there
is no active bug. But the architectural smell is the same: `CublasHandle` is a bare carrier and all its
semantically-meaningful state lives elsewhere — and notably in a *kernel* file (`f2_block_kernel.cu`),
an odd home for cuBLAS-handle policy. A `CublasHandle::engage_precision(const Precision&)` would
centralize §12's precision policy behind the owning type. §2 single-responsibility / §8 single-home
observation, not a correctness bug.
Severity: low. Effort: M (touches the kernel file's API). Before M4.5: no (do it when 1.1/3.1 land — they
all converge on "the handle owns its mutable state").

### (4) CUDA idioms / RAII / stream & async / launch config vs §7

**4.1 [GOOD] Move-only, full move-construct + move-assign, non-throwing destructor — matches §7 verbatim.**
Lines 49–62 implement the §7 reference shape (architecture.md lines 426–448) essentially verbatim,
including the move-assign that §7 calls out as "the old draft's deleted move-assign was a bug."
`noexcept` on both moves; copy deleted; destructor delegates to a `noexcept` `destroy()`. Correct and
idiomatic. No change.

**4.2 [GOOD] `cublasDestroy` called once, at teardown, never per-iteration — and the doc claim is verified.**
The header comment (lines 6–8) asserts "`cublasDestroy` implicitly synchronizes, so a handle is never
recreated per-iteration." Confirmed verbatim: **"the release of those resources by calling
`cublasDestroy()` will implicitly call `cudaDeviceSynchronize()`"** (cuBLAS §2.4.1,
[cuBLAS Library docs](https://docs.nvidia.com/cuda/cublas/index.html)). The create-once/reuse design
(one `blas_` member, `cuda_backend.cu:339`) honors the docs' "minimize the number of times
`cublasCreate()` and `cublasDestroy()` are called." Good. (This fact is also what makes the
member-ordering in 3.1(b) accidentally safe.)

**4.3 [MED] The wrapper is too thin — a `cublasHandle_t` newtype, not an abstraction that owns the cuBLAS context's mutable configuration; this is the root cause of 1.1/3.1/3.2.** *(CONFIRMED.)*
Location: whole class. Per §7 an owning wrapper's job is to make the resource safe and its invariants
enforceable. Here every piece of stateful cuBLAS configuration the codebase uses — stream
(`cublasSetStream`), workspace (`cublasSetWorkspace`), math mode + emulation tuning — is poked into the
raw handle from `cuda_backend.cu`, `f2_block_kernel.cu`, and `f2_blocks_kernel.cu`. The wrapper
guarantees only *destruction*. Defensible as MVP, but it is the root cause of 1.1/3.1/3.2: because the
class does not own the invariant "(stream, workspace) must be set together," a caller can — and in two
places does — break it. The senior-quality fix is a small, cohesive surface:
- `void set_stream(cudaStream_t s)` → `cublasSetStream` **then re-apply** the owned workspace.
- `void set_workspace(cuda::std::span<std::byte>)` (store ptr+size; reapply after stream changes).
- `void engage_precision(const Precision&)` (move §12's policy out of the kernel file).
- keep `get()` purely for handing the configured handle to `cublasGemmEx` / `cublasGemmStridedBatchedEx`.
This is the difference between "owns a handle" and "owns the handle's contract." Single change that takes
this header to 9.5+.
Severity: med. Effort: M. Before M4.5: the `set_stream`+workspace part, yes (it fixes 1.1); the rest can
follow.

**4.4 [LOW] No `cuda::std::span` currency for the (future) workspace, per §7's "span/mdspan are the interface currency."** *(CONFIRMED; not actionable until 3.1.)*
If the workspace moves into the handle (3.1/4.3), it should be passed as `cuda::std::span<std::byte>`
(matching `DeviceBuffer::view()` in the §7 reference, architecture.md line 409), not a raw `void*`+size
like the spike. Flagged so the encapsulation uses the right currency.
Severity: low. Effort: S. Before M4.5: no.

### (5) Magic numbers & hardcoded values vs §4 / ROADMAP §4

**5.1 [GOOD / N-A] No magic numbers in this file.** The only literals are `nullptr` (a true language
constant) and the format string in the debug macro. The cuBLAS workspace size correctly lives as
`kCublasWorkspaceBytes` in `config.hpp:88` (not here), and the handle does not reference it. ROADMAP §4
("no literal may survive except true mathematical constants") is satisfied for this unit. If the
workspace is encapsulated (3.1), keep sourcing its size from `kCublasWorkspaceBytes`, never re-literal it.

### (6) Decomposition / single-responsibility / function size vs §2

**6.1 [GOOD] Single responsibility, tiny functions.** One class, one resource; ctor, two move ops, dtor,
one accessor, one private `destroy()` — nothing exceeds ~6 lines. The §2 decomposition target. The only
SRP critique is the inverse of "too big": the class is arguably too *small* for its responsibility (4.3)
— it owns the handle but not the handle's contract. A design-completeness point, not a function-size one.

**6.2 [LOW] The teardown-warning macro is duplicated three times across the device headers, violating §2 DRY — and (NEW, see N5/N6) the three copies are not even identical and none is `#undef`'d.** *(CONFIRMED and EXPANDED.)*
Location: `handles.hpp:29–35` (`STEPPE_HANDLES_WARN_ON_TEARDOWN`), `device_buffer.cuh:33–40`
(`STEPPE_DEVICE_BUFFER_WARN_ON_TEARDOWN`), `stream.hpp:38–44` (`STEPPE_STREAM_WARN_ON_TEARDOWN`). All
three are the same `#if NDEBUG / fprintf(stderr, "[steppe][warn] %s at teardown: %s\n", …)` pattern with
a per-header name. §2 ("CUDA error checking, error propagation, logging … implemented exactly once") and
§8 (the teardown-warning sink is named a single-home concern: "internal/log.hpp … also the
teardown-warning sink (§7)", architecture.md line 523) both say this should be one thing. All three even
carry the same comment promising to "swap the body for `STEPPE_LOG_WARN` once log.hpp lands" — a known,
acknowledged stopgap, but it is *three* copies of it, and they have drifted (see N5/N6). It should be one
shared inline/macro in a single device-private header (e.g. `teardown_warn.cuh`) or, better, routed
through the eventual `internal/log.hpp` facade.
Severity: low. Effort: S (extract once, include in all three). Before M4.5: no, but cheap and removes a
§2 violation flagged by the project's own grep-gate philosophy.

### (7) Readability, naming, const-correctness, [[nodiscard]] / noexcept, comments

**7.1 [GOOD] `[[nodiscard]]` and `noexcept` are placed correctly.** `get()` is
`[[nodiscard]] … const noexcept` (a pure accessor — correct). Moves are `noexcept`. `destroy()` is
`noexcept`. The ctor is correctly *not* `noexcept` (it throws on `cublasCreate` failure). Const-correctness
is as good as the C API allows (1.3 caveat).

**7.2 [LOW] The header doc over-claims the stream binding (cross-ref 1.2) and, minor, omits the "moved-from owns nothing" line its sibling has.** *(CONFIRMED.)* Lines 39–41 ("Optionally bound to a stream
at construction (single statistic stream on the bit-stable path)") describe a behavior the sole consumer
defeats (1.2). Tighten to match reality, or fix the consumer and keep the comment. Minor: `stream.hpp:48`
documents "a moved-from Stream owns nothing and is safe to destroy"; `handles.hpp` has no equivalent
sentence for a moved-from handle. A nice-to-have for symmetry, not a defect.
Severity: low. Effort: S. Before M4.5: no.

**7.3 [GOOD] `CublasError::status_name` reached cross-class from the dtor is the right DRY move and cannot throw.** `handles.hpp:74–75` calls `CublasError::status_name(s)`; the mapping lives once in
`check.cuh:74–88`, and `status_name` is `[[nodiscard]] static … noexcept` returning a `const char*`
from a `switch` (no allocation), so calling it from a `noexcept` `destroy()` is safe. Confirmed by
reading `check.cuh`. No change.

**7.4 [GOOD / N-A] Naming nit: `o`/`h_` are terse but consistent with the sibling wrappers
(`device_buffer.cuh`, `stream.hpp`).** Consistency wins; do not rename in isolation. Listed for
completeness, not a defect.

### (8) Performance

**8.1 [GOOD / N-A] Nothing performance-sensitive in this file.** The handle is created once and
destroyed once (§7); no hot path. The one performance-adjacent fact is *negative*: the workspace/stream
footgun (1.1) does not cost performance, it costs *determinism* — and the fix (re-applying the workspace
on stream change) costs one `cublasSetWorkspace` per stream change, negligible. The 64 MiB
`kCublasWorkspaceBytes` is *ample* vs the cuBLAS-recommended **32 MiB for sm_12x Blackwell** (corrected
in 11.1) — comfortably above, not below, the recommendation — so there is no determinism risk from an
undersized workspace, only the reset risk (1.1).

### (9) Layering / API / ABI vs §4

**9.1 [GOOD] Correctly CUDA-private; correctly placed under `src/device/cuda/`.** Includes `<cublas_v2.h>`
(the v2 API, correct) and `check.cuh` (itself CUDA-private); consumed only by `cuda_backend.cu`. The
file-header comment (lines 12–15) correctly justifies the `.hpp` extension despite being a CUDA header
(matches §4's repo-layout entry, architecture.md line 871). No `core`/`api`/`app` TU includes it
(verified by grep — the only non-experiment hits are `cuda_backend.cu` and a comment in `stream.hpp`).
§4 layering honored.

**9.2 [LOW] The architecture (§7 line 449, §9 line 609 `CusolverDnHandle solver;`) names a sibling `CusolverDnHandle` following "this exact shape" — it does not exist yet; design 1.1/4.3 so it can share the pattern.** *(CONFIRMED.)*
Not a defect in *this* file, but `handles.hpp` is the named single-home for "library handles." When
`CusolverDnHandle` lands (S5 SVD / S6 GLS, §5), it belongs here and must replicate the move/dtor shape
*and* inherit whatever workspace/stream/determinism encapsulation 1.1/4.3 introduce (cuSOLVER has the
same `cusolverDnSetStream` + workspace + `cusolverDnSetDeterministicMode` story — architecture.md line
733 enumerates the deterministic-covered routines). Design 1.1/4.3 so the cuSOLVER sibling shares the
pattern (a tiny `destroy_or_warn` helper, or a CRTP base) rather than copy-pasting a third handle and a
third teardown macro (N5).
Severity: low. Effort: M (future). Before M4.5: no.

### (10) Testability vs §13

**10.1 [MED] The handle is untestable GPU-free (inherent), and there is no seam to unit-test the workspace-survives-stream-change invariant (1.1) that matters most.** *(CONFIRMED.)*
§13 leans on GPU-free testing and a `CudaTest` fixture that "resets the device per suite and asserts a
clean error state in `TearDown`" (architecture.md line 757). A bare `CublasHandle` can only be exercised
on a real GPU (it calls `cublasCreate` in the ctor) — inherent to a cuBLAS-handle wrapper and acceptable.
But the *invariant* worth testing — "after `set_stream`, the workspace is still mine, not the default
pool" — has no API to test against today because there is no `set_stream`/`set_workspace` on the wrapper
(it's all raw calls in the backend). Adding the methods from 1.1/4.3 *also* creates the testable seam: a
GPU test can construct a handle, set a workspace, change the stream, and assert (behaviorally, via a
reproducibility re-run of an `EmulatedFp64` GEMM and a bit-diff) that determinism held. Without those
methods, the most important property of this unit is only testable end-to-end in the goldens — a unit
test pinning it would have caught 1.1.
Severity: med. Effort: M (enabled by 1.1/4.3). Before M4.5: yes for the seam, since the §12 determinism
gate silently depends on the workspace surviving the per-call `cublasSetStream`.

### (11) Capability tiers (PRO-6000-capable vs budget-5090) — what this unit should add

Per `docs/TODO.md` (lines 125–137) the codebase must runtime-detect capability and "degrade with an
explicit logged tag," with parity on both tiers, and the cross-cutting item is "a capability probe +
capability-tagged results … slotting into `DeviceConfig`/`Resources`." This handle wrapper touches that
boundary in three concrete ways, none surfaced today.

**11.1 [LOW] Workspace sizing should be capability-aware and logged — and the first pass's recommended-size figure was imprecise; the doc says 32 MiB for sm_12x, not 4 MiB.** *(CORRECTED this pass.)*
The cuBLAS docs' recommended workspace by architecture (cuBLAS §2.4.8, re-verified verbatim):
sm_90 (Hopper) = 32 MiB, sm_10x = 32 MiB, **sm_12x (Blackwell) = 32 MiB**, other = 4 MiB
([cuBLAS Library docs](https://docs.nvidia.com/cuda/cublas/index.html)). **Both** the full-host RTX PRO
6000 and the budget 5090 are sm_120, so both want 32 MiB — the first pass's "4 MiB otherwise" framing was
a distractor (neither steppe box is "otherwise"). `kCublasWorkspaceBytes` is a flat 64 MiB, comfortably
above the 32 MiB floor on both tiers, so there is **no under-sizing risk** (revising 8.1 accordingly).
The capability angle is only mild over-provisioning: on the budget tier (consumer 5090, less VRAM
headroom, the §11.2 budget check fighting for every MB), 64 MiB per handle × G handles (multi-GPU) is not
free; on the capable tier (96 GB) it is noise. When the handle owns its workspace (3.1), make the size a
parameter derivable from a runtime device-property probe (`cudaGetDeviceProperties` → arch, choose
`max(32 MiB, configured)`) and log the chosen value with a capability tag
(`[steppe][cap] cublas workspace = 64MiB (sm_120)`), per TODO's explicit-tag requirement (line 137).
Severity: low (was med — corrected since there is no under-sizing risk). Effort: M. Before M4.5: no;
design 3.1 so the size is a parameter, not a hardcode, to keep this open.

**11.2 [LOW] `STEPPE_HAVE_EMU_TUNING` is a *build-time* flag, but the capability tier wants *runtime* detection with a logged fallback.** *(CONFIRMED.)*
Mostly in `f2_block_kernel.cu` (`engage_f2_precision`, `#if STEPPE_HAVE_EMU_TUNING`, lines 204–212), not
this header. But if precision engagement moves into `CublasHandle::engage_precision` (3.2/4.3), the
wrapper becomes the natural place to *runtime-probe* whether `cublasSetEmulationStrategy` /
`cublasSetFixedPointEmulationMantissaControl` are available on this driver/arch and to emit a
`[steppe][cap] emu-tuning {on | off → default emulation}` tag (per TODO line 137) rather than silently
`(void)precision` in the `#else` (`f2_block_kernel.cu:211`). The §12 caveat "without it the emulated math
mode still engages emulation (just not the pinned slice count)" (f2_block_kernel.cu:196–198,
architecture.md line 89 [UNCERTAIN] handle-level only) is exactly the "degrade with an explicit logged
tag" case.
Severity: low. Effort: M (cross-file, future). Before M4.5: no.

**11.3 [LOW] No P2P / device-ordinal awareness — relevant to §11.4's capability-tiered combine.** *(CONFIRMED; cross-ref 2.3.)*
The multi-GPU combine (§11.4, TODO line 133) has a capable path (`cudaMemcpyPeer` / aikitoria-patched
P2P) and a budget path (host-staged). Per-device `CublasHandle`s must be created under the right device
(2.3); recording the device ordinal lets the multi-GPU orchestration log which tier each handle is on.
Cheap hook; defer to when multi-GPU lands.
Severity: low. Effort: S. Before M4.5: no.

### (12) NEW findings the first pass missed (macro hygiene / observability)

**N5 [LOW] The teardown macro `STEPPE_HANDLES_WARN_ON_TEARDOWN` is never `#undef`'d, so it leaks into the preprocessor namespace of every TU that includes `handles.hpp` (and the same is true of all three sibling macros).** *(NEW — missed by the first pass.)*
Location: `handles.hpp:29–35`; verified by grep that there is **no `#undef`** of the teardown macro in
`handles.hpp`, `stream.hpp`, or `device_buffer.cuh`. A function-like macro defined in a widely-included
header and left defined pollutes the macro namespace of every consumer (`cuda_backend.cu` includes all
three) — a classic header-hygiene defect and a potential collision with any future symbol of that name.
Because the macro is `#if`-conditional on `NDEBUG`, it also means a downstream TU silently inherits
whichever build's definition was first seen. The fix dovetails with 6.2: once the teardown sink is a
single inline function (or routed through `STEPPE_LOG_WARN`), the macro disappears entirely. As an
interim, `#undef STEPPE_HANDLES_WARN_ON_TEARDOWN` at the bottom of the header (after the class, before the
`#endif`) confines it. Note this contradicts the file-header comment's "must never throw" discipline only
in spirit — a leaked macro is not a runtime hazard, but it is exactly the kind of detail a 9.5/10 header
gets right.
Severity: low. Effort: S. Before M4.5: no (fold into the 6.2 extraction).

**N6 [LOW] The three teardown macros' NDEBUG (release) branches are inconsistent in whether they evaluate their arguments — a latent trap if an argument ever gains a side effect.** *(NEW — missed by the first pass.)*
Location: `handles.hpp:30` (`((void)0)` — does NOT evaluate `what`/`statusname`); `stream.hpp:39`
(`((void)0)` — does NOT evaluate); `device_buffer.cuh:34` (`((void)(errstr))` — DOES evaluate `errstr`
to suppress unused-variable). For `handles.hpp` the live argument is `CublasError::status_name(s)`, a
side-effect-free `noexcept` pure function, so non-evaluation in release is currently harmless — and in
fact slightly *better* (it avoids the `switch` call in release). But the inconsistency across three
sibling headers that otherwise claim to "mirror" each other (the `handles.hpp:25` comment says "mirroring
device_buffer.cuh") is a smell, and `((void)0)` is a footgun the moment someone passes an argument with a
side effect (e.g. a function that also clears `cudaGetLastError()`), since it would silently not run in
release. Unifying via the single shared sink (6.2) removes the discrepancy. If kept as macros, prefer the
`((void)(arg))` form for all three so the contract ("arguments are always evaluated") is uniform — or, if
non-evaluation is the intended optimization, document it.
Severity: low. Effort: S. Before M4.5: no.

**N7 [LOW] No `valid()` / `explicit operator bool()` to query whether the handle owns a live cuBLAS context.** *(NEW — minor; sibling-consistent omission.)*
Location: whole class. After a move, `h_ == nullptr`; there is no public way to ask the handle whether it
is live (only `get() == nullptr`, which conflates "moved-from" with "default-constructed"). The sibling
`DeviceBuffer`/`Stream` have the same omission, so this is consistent, and at M0 nothing needs it (the
handle is always live in the one consumer). But when `PerGpuResources` holds a vector of handles and the
multi-GPU teardown/relocation paths move them around (§11.4), a cheap `[[nodiscard]] explicit operator
bool() const noexcept { return h_ != nullptr; }` would let callers assert liveness without leaking
`get()`. Nice-to-have, not a defect.
Severity: low. Effort: S. Before M4.5: no.

---

## Considered & rejected

- **"Member-init order makes `stream_` read while uninitialized (UB)."** Rejected (and the first pass's
  1.2 wording, which leaned toward this, was tightened): members init in declaration order and `stream_`
  (declared first, with `= nullptr`) is fully initialized before `blas_`'s ctor reads it. There is no
  read-before-init. The real point is narrower: the arg is dead-by-design because `stream_` is `nullptr`
  at M0 (1.2).

- **"The destructor should `cudaSetDevice` to the handle's device before `cublasDestroy`."** Rejected:
  §7 forbids hidden global mutable state in wrappers, and changing the current device inside a destructor
  is a surprising side effect (it leaves the wrong device current after scope exit). The owning scope
  (`PerGpuResources` teardown) sets the device; the handle at most *records-and-asserts* its ordinal
  (2.3).

- **"`get()` should return `const cublasHandle_t`."** Rejected as meaningless: `cublasHandle_t` is
  `typedef struct cublasContext* cublasHandle_t`; `const cublasHandle_t` is a const *pointer*, not a
  pointer-to-const, so it does not prevent mutating the cuBLAS context (1.3). The C API takes the handle
  by value into mutators regardless. The real mitigation is to not expose `get()` for mutation (4.3).

- **"The ctor should make `cublasSetStream` mandatory / never accept a null stream."** Rejected: cuBLAS
  legitimately runs on the NULL/default stream if `cublasSetStream` is never called ("If the cuBLAS
  library stream is not set, all kernels use the default NULL stream" — cuBLAS docs), and the M0 backend
  deliberately uses the default stream (`cuda_backend.cu:335–337`). The issue is "stream changes silently
  drop the workspace" (1.1) and "the ctor arg is dead in the consumer" (1.2), not "null stream forbidden."

- **"Make `CublasHandle` copyable via a shared/ref-counted handle."** Rejected: move-only is mandated by
  §2/§7 and correct for an owning resource; `PerGpuResources` is itself move-only and holds the handle by
  value. The cuBLAS docs explicitly warn against sharing a handle and re-configuring it from multiple
  threads ("extreme care … It is even more true for the destruction of the handle. So it is not
  recommended that multiple thread share the same cuBLAS handle" — cuBLAS §2.1.3, verified verbatim).
  Keep move-only.

- **"The teardown warning should throw / abort on a nonzero destroy status to be fail-fast."** Rejected:
  §7/§10 are explicit that destructors never throw but warn in debug; an abort at teardown can mask the
  original error during stack unwinding. The current debug-`fprintf`-in-`NDEBUG`-noop is the
  architecture-prescribed behavior. The only improvement is DRY-ing/routing the macro (6.2/N5) through
  `STEPPE_LOG_WARN` once it exists.

- **"`destroy()` should check `cudaGetLastError()` / sync before `cublasDestroy`."** Rejected:
  `cublasDestroy` already implies `cudaDeviceSynchronize()` (verified, 4.2), so an extra sync is
  redundant; and clearing a sticky error in a destructor would hide a fault the next runtime call should
  surface (§7/§10).

- **"Default workspace is fine; drop the explicit `cublasSetWorkspace` entirely."** Rejected: §12 and the
  cuBLAS docs require an explicit workspace specifically because fixed-point emulation voids run-to-run
  reproducibility otherwise, and the docs prescribe `cublasSetWorkspace` "to meet fixed-point emulation
  workspace requirements" (cuBLAS §2.1.4, verified). The explicit workspace is load-bearing; the bug is
  that it is *defeated* by the stream reset (1.1), not that it is unnecessary.

- **"`get()` lacking a null check is a bug (could hand out `nullptr` after move)."** Rejected as
  not-a-defect: a moved-from handle legitimately holds `nullptr`, and using a moved-from object is a
  caller contract violation (standard move semantics). Documenting "moved-from owns nothing" (mirroring
  `stream.hpp:48`) is a nice-to-have folded into 7.2; a `valid()`/`operator bool` is the constructive
  version (N7).

- **"The NDEBUG `((void)0)` branch in `handles.hpp` is a bug because it doesn't evaluate
  `CublasError::status_name(s)`."** Rejected as a *bug*, kept as a *smell* (N6): `status_name` is a
  side-effect-free `noexcept` pure function, so not calling it in release is harmless (even marginally
  better). It is only a latent trap if an argument ever gains a side effect — hence N6 is LOW and framed
  as a consistency/contract issue, not a correctness bug.

- **"64 MiB workspace is undersized for Blackwell (docs say 32 MiB)."** Rejected — backwards: the docs say
  32 MiB is the *recommended minimum* for sm_12x, and 64 MiB is *above* it (8.1, 11.1). No under-sizing
  risk. The only angle is mild over-provisioning on the budget tier, captured as a LOW capability note.

## What it takes to reach 10/10

1. **(1.1 / 3.1 / 4.3) Own the (stream, workspace) invariant in the class.** Add
   `set_stream(cudaStream_t)` that re-applies the owned workspace after `cublasSetStream` (which the docs
   confirm resets it to the default pool — §2.4.7), and `set_workspace(span<std::byte>)` that stores
   ptr+size. Remove the raw `cublasSetStream`/`cublasSetWorkspace` calls from `cuda_backend.cu`,
   `f2_block_kernel.cu` (`run_f2_gemms`), **and** `f2_blocks_kernel.cu` (`run_f2_gemms_group`) and route
   them through the wrapper. **This single change closes the only high-severity gap, in both the M0 and
   M4 paths, and restores the §12 determinism guarantee the goldens depend on.**
2. **(3.2) Move §12 precision engagement into `CublasHandle::engage_precision(const Precision&)`** so the
   handle owns *all* its determinism/precision-relevant state, and the kernel files stop being the home
   for cuBLAS-handle policy.
3. **(1.2 / 7.2) Reconcile the ctor stream arg with reality.** Drop the ctor arg in favor of
   `set_stream` and fix the header comment (no dead parameters, no aspirational comments), or reorder
   `CudaBackend` members so the handle is constructed with a real non-null stream and stop re-binding per
   GEMM. Add the "moved-from owns nothing" sentence for symmetry with `stream.hpp`.
4. **(6.2 / N5 / N6) DRY the teardown-warning macro** into one device-private home (or route through
   `internal/log.hpp` once it lands), shared by `device_buffer.cuh` / `stream.hpp` / `handles.hpp` — which
   simultaneously removes the macro leak (N5) and the NDEBUG-branch inconsistency (N6).
5. **(2.3 / 9.2 / 11.x) Add the capability/multi-GPU scaffolding the architecture already names:** record
   the device ordinal at construction (record-and-assert, never `cudaSetDevice` inside the wrapper); make
   the encapsulated workspace size capability-derived (`max(32 MiB, configured)`) and logged with a tag;
   and factor the move/dtor/destroy-or-warn shape so the mandated `CusolverDnHandle` sibling shares it.
   Add `explicit operator bool()` (N7) for liveness queries the multi-GPU path will want.
6. **(10.1) Add a GPU unit test** for the workspace-survives-stream-change invariant once the
   `set_stream`/`set_workspace` seam exists — a test that would have caught 1.1.

Items 1–3 are the before-M4.5 must-dos (correctness/determinism). 4 is a cheap quality+hygiene win that a
10/10 submission would not leave drifting across three headers. 5–6 are forward-compat that a senior
submission would not defer to "later."

## Good patterns to keep

- **Exact §7 RAII shape:** move-construct + move-assign (with self-assignment guard) + non-throwing
  `destroy()` + deleted copy. The reference implementation; correct.
- **`exit()` → typed `CUBLAS_CHECK`/`CublasError`** upgrade over the spike — the ctor throws a catchable,
  located exception (`check.cuh`), which the api can map to `steppe_status_t` (§10).
- **Create-once / reuse / never-per-iteration**, justified by the verified
  `cublasDestroy`-implies-`cudaDeviceSynchronize` fact (4.2, cuBLAS §2.4.1).
- **Debug-only teardown warning** that turns "destructor never throws" into "never *silent* either"
  (§7/§10) — the right non-fatal behavior; just needs DRY-ing + un-leaking (6.2/N5/N6).
- **Correct, `noexcept`, allocation-free cross-class `status_name` use in the dtor** (7.3).
- **Honest, dense, cited file-header comment** that explains the `.hpp`-yet-CUDA-private decision (§4)
  and the destructor contract — strong documentation discipline; tighten only the stream-binding claim
  (7.2).
- **Correct layering:** CUDA-private (`<cublas_v2.h>`), consumed only via the device backend, never
  leaking into `core`/`api`/`app` (§4) — verified by grep.
- **Zero magic numbers** in the file; the workspace size correctly lives as a named `config.hpp`
  constant (5.1).

---

Sources (official docs, every device-behavior claim re-fetched and verified verbatim this pass):
- [cuBLAS Library Documentation (NVIDIA, CUDA 13.x)](https://docs.nvidia.com/cuda/cublas/index.html)
  — §2.1.2 (handle/CUDA-context coupling), §2.1.3 (thread safety / handle sharing), §2.1.4 (Results
  Reproducibility — fixed-point emulation voids the guarantee; provide `cublasSetWorkspace`), §2.4.1
  (`cublasDestroy` implies `cudaDeviceSynchronize`), §2.4.7 (`cublasSetStream` **unconditionally resets
  the workspace to the default pool**), §2.4.8 (`cublasSetWorkspace`; recommended sizes 32 MiB for
  sm_90/sm_10x/sm_12x, 4 MiB otherwise; workspace must stay intact between a call's kernels).
- [CUDA C++ Programming Guide / Runtime API (NVIDIA)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
  — stream/default-stream semantics, `cudaGetDevice`/`cudaSetDevice` current-device model (2.3).

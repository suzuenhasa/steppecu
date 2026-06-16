# Code review — `src/device/cuda/device_buffer.cuh` (`DeviceBuffer<T>`)

Reviewer: adversarial second pass (senior CUDA/C++). Scope: the single translation unit
`src/device/cuda/device_buffer.cuh` (101 lines), read line-by-line, judged against
`docs/architecture.md` (§2, §4, §7, §9, §10, §11, §12, §13), `docs/ROADMAP.md` (§4, §5, §6),
`docs/TODO.md` (the M4.5 plan, the `wxz1fiiln` capability-tier section, the `wn01sl1wz` pre-M4.5
cleanup list, and the "Keeping it GPU-dominant" perf section), `docs/cleanup/00-overview.md`
(the cross-cutting cleanup ledger), and `.clang-tidy`. Verified against its direct context, all
of which I re-read this pass: `check.cuh` (the `STEPPE_CUDA_CHECK` home), the sibling RAII wrappers
`stream.hpp` / `handles.hpp` (re-read **in full** this pass — the first pass mis-stated their
release-macro bodies, corrected in 7.3), the sole consumer `cuda_backend.cu` (re-read in full),
the `launch_*` wrapper signatures in `f2_blocks_kernel.cuh` (relevant to 4.1), the §7 `DeviceBuffer`
reference in `architecture.md` (lines 388–421), `include/steppe/config.hpp`, and the `tests/` tree
+ `tests/CMakeLists.txt`.

Every CUDA device-behavior claim below is cited to the **official NVIDIA docs fetched this pass**
(CUDA Runtime API — Memory Management `cudaFree`/`cudaMalloc`; CUDA C++ Best Practices Guide —
Memory Optimizations / alignment; CUDA C++ Programming Guide §3.4 Multi-Device). Where the first
pass asserted a behavior or a doc citation from memory and I could not reproduce it verbatim, I
corrected the wording or moved the citation to the doc that actually states it (e.g. the 256-byte
alignment guarantee lives in the **Best Practices Guide**, not the Programming Guide as the first
pass implied).

**What this second pass changed vs the prior draft (score 8.5, 21 findings):**
- **Corrected a factual error in 7.3.** The prior draft claimed `stream.hpp`/`handles.hpp` release
  macros are `((void)0)` *and never touch their arguments*; verified they are `((void)0)` but take
  **two** params (`what`, `errstr`) — the inconsistency vs `device_buffer.cuh`'s one-param
  `((void)(errstr))` is real, but the framing "device_buffer evaluates, siblings don't" is sharper
  than stated: in release `device_buffer.cuh` evaluates `cudaGetErrorString(e)` *only on the
  teardown-error path*, the siblings discard their args entirely. Re-scoped (still low).
- **Sharpened 6.3 with the actual `cppcoreguidelines-macro-usage` default.** Verified the check's
  `AllowedRegexp` default is `^DEBUG_*` (clang-tidy docs, fetched) — `STEPPE_DEVICE_BUFFER_WARN_ON_TEARDOWN`
  does **not** match it and **is** function-like, so under `.clang-tidy`'s `WarningsAsErrors:'*'` it
  would trip the check. **But** so would `STEPPE_CUDA_CHECK`, `CUBLAS_CHECK`, `STEPPE_CUDA_CHECK_KERNEL`
  and every sibling teardown macro — i.e. this is a **project-wide tidy-config question, not a
  device_buffer-specific defect**, which weakens it as a finding *for this file* and means a
  suppression/`NOLINT` convention must already exist or the tidy gate isn't actually green on the
  device layer. Downgraded accordingly and reframed.
- **Confirmed `STEPPE_DEBUG_ONLY` does not exist** (grep over `src/`/`include/`/`tests/` = 0 hits;
  corroborated by `00-overview.md` and `core-internal-views.md`), so `device_buffer.cuh` rolling its
  own `#if defined(NDEBUG)` instead of the §7-sketched `STEPPE_DEBUG_ONLY(...)` is **correct, not a
  defect** — added to Considered & rejected.
- **Confirmed 4.1 (`view()`) is real, project-acknowledged, AND currently un-consumable** — the
  `launch_*` wrappers in `f2_blocks_kernel.cuh` take bare `const double*`/`double*`, so even if
  `view()` were restored today nothing could take a span until those signatures go span-first
  (TODO line 92: "restore `DeviceBuffer::view()` + new M4.5 wrappers span-first"). Sharpened.
- **Confirmed 1.1** against the actual §11.2 budget text ("sums the resident terms … P²·B·8 …")
  and the unchecked `std::size_t` products in `cuda_backend.cu` (61–64, 125–126, 179–180, 253–258).
- **Re-confirmed the 4.6 downgrade** against §11.4 verbatim (`device_id` on `PerGpuResources`,
  `cudaMemcpyPeer` the only residue, partials owned at resource level) — the prior high→low–med
  call stands.
- **Added 1 finding** the prior draft missed: the ctor is not `[[nodiscard]]`-protected against a
  discarded temporary `DeviceBuffer<T>(n)` that allocates then immediately frees (8.3 — low,
  recorded, partly rejected). Net vs the 8.5 draft: **+1 added / −0 rejected; 22 substantiated
  findings**, several re-scoped.

---

## Role & layering

`DeviceBuffer<T>` is the move-only RAII owner of device memory and one of the **three allowlisted
translation units** permitted to call the `cudaMalloc`/`cudaFree` allocation family directly
(architecture.md §2 DRY grep gate: `device_buffer.cuh`, `allocator.cu`, `pinned_buffer.cuh`). Every
other unit takes non-owning views and never touches the allocation family. It is a CUDA header
(`.cuh`, includes `<cuda_runtime.h>`), therefore `PRIVATE` to `steppe_device` and invisible to
`core`/`api`/CLI (architecture.md §4). It is the type architecture.md §7 sketches (lines 388–421):
move-construct **and** move-assign, `reset()`, debug-only teardown warning — **minus** the
`cuda::std::span view()` accessors the reference shows (4.1). The layering is correct: it depends
only on `<cuda_runtime.h>`, `<cstddef>`, `<utility>`, and the in-layer `check.cuh`; no upward
dependency, no global state, no I/O.

Two facts kept honest about the role section: `pinned_buffer.cuh` and `allocator.cu` are named on the
allowlist but **do not exist yet** (verified: `src/device/cuda/` holds only `check.cuh`,
`device_buffer.cuh`, `stream.hpp`, `handles.hpp`, and the kernel/backend `.cu` files). They are
spec'd siblings, not current context — which matters for the async-pool findings (4.3, 8.1): the
pooled owner the architecture promises is genuinely *absent*, not merely un-wired.

This is a small, near-textbook RAII header. The *implementation* is essentially correct and idiomatic;
the substantive findings are **gaps against the project's own stated standards** — the §7-mandated
`view()` (which the TODO tracks for restoration), the absent unit test, the unchecked
construction-time multiply — not latent crashes. The file is good; it is not yet 9.5–10 against the §7
reference and the §13 test mandate.

---

## Score: 8.5/10 — solid, idiomatic, layering-clean RAII; held back by the §7-mandated missing `view()` (TODO-tracked, and currently un-consumable because the `launch_*` wrappers are bare-pointer), no dedicated unit test, an unchecked construction-time `n * sizeof(T)` multiply, and the still-placeholder `fprintf` teardown warning

A genuinely well-built wrapper that matches the architecture's intent and its sibling wrappers
bit-for-bit on the move/dtor/`noexcept` discipline. The deductions are concrete and mostly pre-M4.5:
it diverges from the §7 reference by **omitting the `cuda::std::span view()` accessors**
(architecture.md line 478: "owning types hand out views; kernels accept only views, never owners or
bare pointers"; TODO line 92: "restore `DeviceBuffer::view()`"); it has **no dedicated test** despite
being `[BUILT]` and §13 making move/double-free exactly the kind of behavior compute-sanitizer
protects (and the §13 sanitizer only covers code a test runs — line 757/762); and the **construction
multiply `n * sizeof(T)` is unchecked**, so a malformed size can wrap and under-allocate silently —
against §2 "fail-fast, not silent corruption." None require restructuring the type. The first pass's
high-severity multi-GPU framing was overstated and is corrected below.

---

## Findings

### 1. Correctness & bugs

**1.1 — The construction multiply `n * sizeof(T)` (and `bytes()`) is an unchecked `std::size_t`
product; a wrapped value under-allocates silently rather than failing fast. [med, S, before-M4.5: yes]**
Location: ctor (line 54) — `cudaMalloc(&ptr_, n * sizeof(T))`; `bytes()` (line 77) — `size_ * sizeof(T)`.
Both are unchecked `std::size_t` multiplications. Unsigned overflow is *defined* (modular), so this is
not C++ UB — it is a **semantically wrong modular reduction**: if `n * sizeof(T)` wraps to a small
value, `cudaMalloc` succeeds with a buffer far smaller than `size_` advertises, and every downstream
kernel/copy over-runs it — silent heap corruption, the exact opposite of §2's "fail-fast … not silent
corruption." For `double` (`sizeof==8`) the wrap point is `n > SIZE_MAX/8 ≈ 2.3e18` — unreachable on
today's hardware, **but `n` is not bounded by hardware at this layer**: `cuda_backend.cu` forms buffer
sizes as products of three host values (`P·s_pad·nb`, lines 253–255; `total = slab · n_block`, line
126; `pm = P·M`, line 179) using exactly this `std::size_t` arithmetic, and the products are formed
*before* any §11.2 VRAM-budget check sees them (verified: the §11.2 `build()` budget that "sums the
resident terms … `P²·B·8`" is not yet wired — there is no `allocator.cu`/budget summation).
Adversarial check — is this real or theatre? Reachability is genuinely low today and the §11.2 budget
check is the primary intended guard. But (a) §11.2 is not yet wired, and (b) the right place to make a
*single-source* allocation owner safe is in the owner, cheaply, once. So: real, low reachability today,
fix is ~3 lines and removes a whole class of silent-under-allocation.
Concrete fix: in the ctor, before the multiply,
`if (n != 0 && n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw …` — throw a **typed**
error, not a bare `throw` (the codebase convention is `CudaError` via `STEPPE_CUDA_CHECK`, or a domain
error mapping to `STEPPE_ERR_DEVICE_OOM`/`STEPPE_ERR_INVALID_CONFIG` per §10); requires
`#include <limits>` (not currently included — verified). With the ctor invariant established, `bytes()`
is then always exact (it can never observe an `n` whose product overflows, because such an `n` never
constructed a non-empty buffer). Severity med (real corruption class, low reachability today);
before-M4.5 yes because M4.5 multiplies these product-forming call sites across G devices.

*Sub-note (the old 1.2, folded):* the §11.2 budget code "sums the resident terms" (architecture.md
line 681) — if it sums `buf.bytes()` across many buffers, a single wrapped `bytes()` corrupts the
total in the *too-small* (safe-looking) direction, defeating the fail-fast. Same root cause; the
budget summation should additionally use a checked/saturating accumulate. Not a separate finding — the
arithmetic is fixed once at construction.

**1.2 — Move/dtor/self-assignment are correct. (verified, no bug)**
Move-ctor (57–58) and move-assign (60–67) both null the source via `std::exchange`; move-assign
self-guards (`if (this != &o)`) and `reset()`s before stealing — no double-free, no self-move leak.
`reset()` (80–93) guards `cudaFree` on `ptr_ != nullptr` and re-nulls both fields. `cudaFree(nullptr)`
is documented a no-op — **CUDA Runtime API, Memory Management, `cudaFree`: "If devPtr is 0, no
operation is performed"** (verified this pass) — so the `if (ptr_)` guard is belt-and-suspenders for
*correctness*. But it is **load-bearing for behavior**: the same `cudaFree` doc states **"Note that this
function may also return error codes from previous, asynchronous launches"** (verified this pass), so a
stray `cudaFree` on the empty/moved-from path could surface a prior sticky error into the teardown
warning (debug) or trip the §13 `CudaTest` clean-error-state `TearDown`. The guard avoids that runtime
call entirely. Correct and worth keeping. Verified.

### 2. Edge cases & failure modes

**2.1 — `n == 0` is handled and documented (no-op, null ptr, zero size). (verified, good)**
Ctor 53–55 short-circuits `if (n)`. A zero-size buffer is `{nullptr, 0}`, `bytes()==0`. This is the
right call: I re-read the Runtime API Memory Management page this pass and **`cudaMalloc`'s behavior at
`size == 0` is not documented** (no defined-return statement), so *not calling it* is strictly safer
and deterministic. The header states the contract. Good — and `cuda_backend.cu` relies on it for the
`P==0`/`M==0` degenerate paths (130, 298), where `total`/`pm` can be 0.

**2.2 — `data()` on a zero-length / moved-from buffer returns `nullptr` with no debug invariant check.
[low, S, before-M4.5: no]**
Location: `data()` (74–75). After move-out or for `n==0`, `data()` returns `nullptr`. Handing that to
`cudaMemcpyAsync(..., size=0, ...)` is fine; handing it to a kernel with nonzero extent dereferences
null on device. There is no debug-only `assert(ptr_ != nullptr || size_ == 0)` to catch host-side
misuse, and — verified — no `STEPPE_DEBUG_ONLY`/`STEPPE_ASSERT` facility exists in the codebase to host
such a guard (grep = 0 hits; tracked as a phantom in `00-overview.md`). The sibling wrappers don't
assert either (consistent). Adversarial check: a guard can't protect the escaped pointer on the device
side, so its value is limited to catching host-side use-after-move early, and it depends on a facility
that doesn't exist. Low, optional; mirrors the §7 "forced sync in debug" defensive philosophy but is
not mandated and is blocked on the phantom assert facility.

**2.3 — Allocation-failure path is fail-fast and typed. (verified, good)**
The ctor routes `cudaMalloc` through `STEPPE_CUDA_CHECK` (line 54), which throws `CudaError` carrying
file:line and the runtime name/string (check.cuh:101–105, 40–56). `CudaError::status()` exposes
`cudaErrorMemoryAllocation`, which the public API maps to `STEPPE_ERR_DEVICE_OOM` (architecture.md §10
table). On a throwing ctor no object is constructed, `ptr_` is never set, nothing leaks (default member
init `ptr_ = nullptr`, line 95). Exactly the intended behavior.

**2.4 — The `fprintf`-to-stderr teardown warning is a §10 `printf`-in-library violation, and in release
the warning is fully dropped. [low–med, S, before-M4.5: no]**
Location: macro 33–40; call site 88.
(a) The destructor-must-not-throw rule is correct (architecture.md §7), and routing a nonzero `cudaFree`
to a warning instead of throwing is the documented design. But the warning is `std::fprintf(stderr, …)`,
which architecture.md §10 (line 643) explicitly forbids: **"Never `printf`/`std::cout` in library
code."** The header openly flags this as a tracked placeholder ("Swap the body for STEPPE_LOG_WARN once
log.hpp lands", lines 28–32), so it is *intentional debt* — and `internal/log.hpp` does not exist yet
(verified: grep returns only the three siblings' comments referencing it). `00-overview.md` B7 already
tracks creating `internal/log.hpp` to replace "the 3 `fprintf` macros." Severity low–med: a known TODO
with a named migration target (§8 lists `internal/log.hpp` as the single logging home).
(b) In release (`NDEBUG`) the macro is `((void)(errstr))` (line 34), so the only observability of a
teardown fault in production is the *next* runtime call re-reporting the sticky error elsewhere. §7
sanctions release silence for the *warning text*, so this is compliant — worth stating, not a defect.
See 7.3 for the macro-hygiene half (the release expansion needlessly evaluates `cudaGetErrorString` on
the error path).
Concrete fix: route through `STEPPE_LOG_WARN` once `log.hpp` lands (removes the §10 violation and
collapses the three duplicated prefixes — 5.1).

**2.5 — Over-/under-aligned `T` is a non-issue. (verified, N/A — citation corrected)**
`cudaMalloc`'s return is aligned to **at least 256 bytes** — **CUDA C++ Best Practices Guide, Memory
Optimizations ("A Sequential but Misaligned Access Pattern"): "Memory allocated through the CUDA Runtime
API, such as via `cudaMalloc()`, is guaranteed to be aligned to at least 256 bytes"** (verified this
pass; modern GPUs empirically give 512, corroborated by community measurement). *Citation note:* the
first pass attributed this to the **Programming Guide**; the verbatim sentence lives in the **Best
Practices Guide** — corrected. The instantiated `T`s are `double`, `long`, `int`, `std::size_t`,
`std::byte`, `std::uint8_t` (verified via grep of `src/`) — all `alignof ≤ 16`. The runtime guarantee
dominates; no `alignof(T)` assert needed. N/A, recorded for completeness.

### 3. Numerical / precision vs §12

**N/A — `DeviceBuffer<T>` performs no arithmetic.** Pure storage; precision, accumulation order,
native-FP64-vs-Ozaki, and determinism are properties of the consuming kernels/cuBLAS calls.
*Adjacent, not a defect:* §12 requires an explicit `cublasSetWorkspace` workspace for emulated-FP64
run-to-run reproducibility, and that workspace **is** a `DeviceBuffer<std::byte>
workspace_{steppe::kCublasWorkspaceBytes}` member of `CudaBackend` (cuda_backend.cu:340, fed by
`config.hpp::kCublasWorkspaceBytes`, bound in the ctor at line 51). So this type is the *vehicle* for
the §12 determinism workspace — its RAII correctness and the 1.1 no-under-allocation invariant are
therefore mildly load-bearing for determinism — but the file itself has no numerical content.

### 4. CUDA idioms / RAII / stream & async semantics / launch config / occupancy vs §7

**4.1 — MISSING `view()` returning `cuda::std::span<T>` / `cuda::std::span<const T>` — and the
`launch_*` wrappers it would feed are still bare-pointer. [med, M, before-M4.5: yes]**
Location: accessor block 74–77.
Compare the §7 reference for *this exact type* (architecture.md lines 409–410): `cuda::std::span<T>
view() noexcept { return {ptr_, size_}; }` plus a `const` overload; and the §7 rule (line 478):
**"owning types hand out views; kernels accept only views, never owners or bare pointers."** This file
ships only raw-pointer `data()`. This is the single largest divergence from the architecture's own spec
for `DeviceBuffer`. **Project-acknowledged**: TODO line 92's pre-M4.5 cleanup reads "restore
`DeviceBuffer::view()` + new M4.5 wrappers span-first" — i.e. `view()` was present and was dropped.
**Sharpened this pass:** restoring `view()` *alone* buys nothing today, because the consumers — the
`launch_*` wrappers in `f2_blocks_kernel.cuh` (verified: `launch_gather_group`, `run_f2_gemms_group`,
`launch_assemble_blocks_group` all take `const double*`/`double*`/`const int*`/`const long*`, lines
45–80) — are bare-pointer signatures. So the §7 view-first contract requires *both* `view()` on the
owner *and* span-typed launch wrappers; the TODO phrases this as one item ("+ new M4.5 wrappers
span-first"). Doing it *after* the M4.5/M5 call sites proliferate is a larger refactor.
Adversarial check — defensible as YAGNI? No: the architecture names `view()` a defining member, the
consumers exist, and the TODO lists its restoration. This is a deliberate omission to reverse.
Concrete fix: add `[[nodiscard]] cuda::std::span<T> view() noexcept { return {ptr_, size_}; }` + the
`const` overload; `#include <cuda/std/span>`. **Caveat:** `cuda::std::span` is *declared* available
(CCCL ≥ 3.1, architecture.md §3) but is **not yet used anywhere in `src/`** (verified — `views.hpp`
only *mentions* it in a comment; `span_view.hpp` does not exist), so this is the first adopter and
should land alongside the §8 `internal/span_view.hpp` home rather than as a lone include. Keep `data()`
for the cuBLAS/`cudaMemcpyAsync` sites that need a raw `T*`/`void*`.

**4.2 — `data()` const/non-const overloads are correct; the only gap is the missing `view() const`'s
`span<const T>` constness. [low, S, before-M4.5: no]**
Location: 74–75. The pair is correct C++ const-correctness. Lacking `view()` (4.1), a consumer wanting a
read-only device view must take `data()` and re-wrap, losing the `span<const T>` constness `view()
const` gives for free. Folded into 4.1; recorded so the const story is complete.

**4.3 — Synchronous `cudaFree` in the destructor is a device-wide sync point; the async-free lifetime
contract is undocumented and becomes a hazard under the (not-yet-built) pool allocator. [med, M,
before-M4.5: no]**
Location: whole class; `reset()` 86.
`cudaFree` is **not** stream-ordered. **CUDA Runtime API, Memory Management, `cudaFree` (verified this
pass): "This API will not perform any implicit synchronization when the pointer was allocated with
cudaMallocAsync or cudaMallocFromPoolAsync … For all other pointers, this API may perform implicit
synchronization."** `DeviceBuffer` uses plain `cudaMalloc`, so its `cudaFree` is in the "all other
pointers" class and **may implicitly synchronize the device.** Today (M0–M4, one f2 call at a time, with
an explicit `cudaStreamSynchronize` per chunk — cuda_backend.cu:208, 271, 282) this is invisible. In the
M5 double-buffered streaming pipeline (§11.1) a buffer going out of scope mid-pipeline would stall the
copy/compute overlap that pipeline exists to achieve. The architecture's answer is the pool-backed
sibling using `cudaMallocAsync`/`cudaFreeAsync` (architecture.md §7 line 487, in `allocator.cu`) —
**which does not exist yet** (verified). The gap is that the synchronous semantics are *undocumented* on
the class.
Adversarial check — in scope for *this* file? The synchronous semantics are correct *for this type*; the
finding is the missing one-line contract note. Concrete fix: document on the class that destruction is
an implicit device sync point and that hot-loop/streaming buffers should come from the pool allocator
once `allocator.cu` lands; do **not** add a stream to `DeviceBuffer` (that is the pooled sibling's job —
see Considered & rejected). Med, M5-relevant; not before-M4.5.

**4.4 — `reset()` uses bare `cudaFree` + manual status inspection, not the throwing `STEPPE_CUDA_CHECK`,
so the `noexcept` dtor cannot throw. (verified, good — subtle, done right)**
The ctor uses throwing `STEPPE_CUDA_CHECK(cudaMalloc(...))` (54); `reset()` calls bare `cudaFree` and
inspects status manually (86–89) because `STEPPE_CUDA_CHECK` throws and `reset()`/the dtor are `noexcept`
(80, 72). Using `STEPPE_CUDA_CHECK` here would `std::terminate` on a teardown fault. This asymmetry is
exactly §7 ("destructors never throw … route a nonzero destroy status to a debug log"). Identical to the
discipline in `stream.hpp`/`handles.hpp` (`destroy()` does the same). Verified correct.

**4.5 — `noexcept` discipline is complete and correct. (verified, good)**
Move-ctor (57), move-assign (60), `data()`×2/`size()`/`bytes()` (74–77), `reset()` (80) are all
`noexcept`. The only non-`noexcept` member is the allocating ctor (53), which *must* be able to throw
`CudaError`. Matches §7. Move-assign's unconditional `noexcept` is correct — only pointer/size
bookkeeping and the `noexcept` `reset()` run; nothing `T`-dependent can throw (the buffer stores raw
bytes, never `T` objects).

**4.6 — `DeviceBuffer` records no device id; the construction-device binding is implicit. Optional
defense-in-depth for the M4.5 SPMG path, NOT a mandated correctness chokepoint. [low–med, S,
before-M4.5: no]**
Location: whole class (ctor 53–55 binds; `reset()` 86 frees).
**Mechanism:** CUDA C++ Programming Guide §3.4 documents that device memory allocations and kernel
launches are made on the *current* device (the current device defaults to device 0). So `cudaMalloc` in
the ctor binds the buffer to whatever device was current at construction, and that binding is invisible
to the object (it stores no `device_id`).
**Why the first pass over-rated this (corrected to low–med):**
1. The architecture *deliberately* homes device identity elsewhere: `PerGpuResources` carries `int
   device_id` and the allocator is "on THIS device" (architecture.md line 605, verified this pass), and
   the M4.5 capability probe "slots into `DeviceConfig`/`Resources`" (TODO line 137). `DeviceBuffer`
   floating with the current device is *by design*; the SPMG driver owns the `cudaSetDevice` context. A
   `device_id` on `DeviceBuffer` would duplicate state the architecture keeps at the resource level.
2. The failure mode is **not silent**: §11.4 uses per-device streams ("on THIS device"), and per §3.4 a
   kernel launch issued to a stream not associated with the current device **fails the launch** — caught
   by `STEPPE_CUDA_CHECK_KERNEL()` — rather than silently reading a foreign-device pointer. (Copies are
   device-resolved by pointer, so teardown `cudaFree` and `cudaMemcpyPeer` are safe regardless of current
   device.) *Citation honesty:* I could not re-fetch §3.4's exact sentences this pass (the Programming
   Guide URL returns only the TOC to the fetcher); the multi-device current-device model is corroborated
   by §11.4's own design text and is well-established documented behavior, but I have **not** verbatim-
   confirmed the "a kernel launch will fail …" sentence the prior draft quoted — I therefore state the
   conclusion (loud failure, not silent corruption) and flag the quote as un-re-verified rather than
   asserting it as a verbatim citation.
So this is a *convenience/observability* nicety, not a silent-corruption hazard. Concrete fix (if
adopted): `cudaGetDevice(&device_id_)` in the ctor; `[[nodiscard]] int device() const noexcept`. The one
place it has real downstream value is the `cudaMemcpyPeer(dst, dstDevice, src, srcDevice, bytes)`
source-device argument in the M4.5 P2P combine (TODO line 128 / §11.4) — but those partials are owned at
the `PerGpuResources` level where `device_id` already lives, so even there the caller has the id without
`DeviceBuffer` carrying it. Low–med; *not* before-M4.5 (the resource-level design already covers the
contract).

**4.7 — The function-like teardown macro leaks unguarded from a header into every including TU. [low, S,
before-M4.5: no]**
Location: macro 33–40 (no `#undef`).
`STEPPE_DEVICE_BUFFER_WARN_ON_TEARDOWN` is a function-like macro defined in a header and never `#undef`'d,
so it pollutes the preprocessor namespace of every TU that includes `device_buffer.cuh` (and transitively
`cuda_backend.cu`). The long `STEPPE_DEVICE_BUFFER_*` prefix makes collision unlikely, and the two
siblings (`stream.hpp`'s `STEPPE_STREAM_WARN_ON_TEARDOWN`, `handles.hpp`'s
`STEPPE_HANDLES_WARN_ON_TEARDOWN`) leak their analogues the same way (consistent), so this is minor. The
clean resolution is the same as 2.4/5.1: the `STEPPE_LOG_WARN` migration removes all three header macros
entirely. Low.

### 5. Magic numbers & hardcoded values vs §4 (ROADMAP §4)

**5.1 — The teardown-warning prefix literal is duplicated across the three sibling wrappers. [low, S,
before-M4.5: no]**
Location: macro body 38–39 — `"[steppe][warn] cudaFree at DeviceBuffer teardown: %s\n"`.
ROADMAP §4's "no literal may survive except true mathematical constants" targets *numeric* config
constants, not log-format strings, so this is out of strict §4 scope — but the prefix shape
`[steppe][warn] … teardown: %s\n` is duplicated in `stream.hpp` (line 41:
`"[steppe][warn] %s at teardown: %s\n"`) and `handles.hpp` (line 30, same shape): three near-identical
literals = a minor §2 DRY smell. **Project-acknowledged**: TODO line 92 lists "triplicated
teardown-warning macros → one shared," and `00-overview.md` B7 lists the same. Resolution: collapse into
`STEPPE_LOG_WARN` (one prefix in the facade) when `log.hpp` lands; until then the duplication is bounded
and tracked. Low. *Minor inconsistency noted:* `device_buffer.cuh`'s prefix bakes "cudaFree at
DeviceBuffer teardown" into the format string and takes one arg (`errstr`), whereas the siblings take two
args (`what`, `errstr`) and parameterize the operation name — so the trio is not even uniform in *shape*,
not just in the literal. The `log.hpp` migration fixes both.

**5.2 — No surviving numeric magic numbers. (verified, good)**
The only literals are `0`/`nullptr` initializers (true defaults) and the `sizeof(T)` factor (a language
operator). The one constant a buffer could have inlined — the cuBLAS workspace size — correctly lives as
`steppe::kCublasWorkspaceBytes` in `config.hpp` (verified: `64u*1024u*1024u`, doc-commented with the §12
rationale) and is injected by `cuda_backend.cu:340`, keeping `DeviceBuffer` size-agnostic. Clean against
ROADMAP §4.

### 6. Decomposition / single-responsibility / function size vs §2

**6.1 — Single responsibility, minimal surface, right size. (verified, good)**
One class, one job, ~55-line body. `reset()` is the single internal helper shared by move-assign and the
dtor — the DRY factoring §2 wants (no duplicated free logic). Nothing to decompose; adding anything would
be over-engineering.

**6.2 — Debug/release warning policy encapsulated in one macro. (verified, good)**
`STEPPE_DEVICE_BUFFER_WARN_ON_TEARDOWN` (33–40) localizes the `NDEBUG` split so `reset()` reads linearly —
good separation of policy from mechanism. See 6.3 / 7.3 for the macro caveats.

**6.3 — A function-like macro is used where an `inline` function would satisfy
`cppcoreguidelines-macro-usage` — but this is a project-wide tidy-config question, not a device_buffer
defect. [low, S, before-M4.5: no]**
Location: macro 33–40.
`.clang-tidy` enables `cppcoreguidelines-*` with `WarningsAsErrors: '*'` (verified). I verified the
check's behavior this pass: `cppcoreguidelines-macro-usage` flags **function-like macros** and its
`AllowedRegexp` default is **`^DEBUG_*`** (clang-tidy docs, fetched). `STEPPE_DEVICE_BUFFER_WARN_ON_TEARDOWN`
is function-like and does **not** match `^DEBUG_*`, so on paper it would trip the check and, under
`WarningsAsErrors:'*'`, block the build.
**But the adversarial conclusion is that this is *not* a device_buffer-specific finding:** by the same
rule, `STEPPE_CUDA_CHECK`, `CUBLAS_CHECK`, `STEPPE_CUDA_CHECK_KERNEL` (check.cuh), and the two sibling
teardown macros are *all* non-`^DEBUG_*` function-like macros. If the tidy gate genuinely fired on
function-like macros, the entire device layer would already fail to build — which means either (a) a
project-wide `cppcoreguidelines-macro-usage` suppression / `AllowedRegexp` override / per-TU `NOLINT`
already exists, or (b) the tidy gate isn't actually green on `.cu`/`.cuh` (plausible: the `.clang-tidy`
comment says CUDA checks are "tuned per-milestone, not blanket-off"). Either way, the right fix is at the
project tidy-config level, uniformly for all the `STEPPE_*` check/warn macros, not a one-off rewrite here.
A `static inline void warn_on_teardown(const char*) noexcept` guarded by `#ifndef NDEBUG` *would* type the
argument and kill 4.7/7.3 — but only if applied consistently, and the `log.hpp` migration supersedes it
anyway. Low; recorded with the sharpened default but downgraded as not-this-file-specific.

### 7. Readability, naming, const-correctness, [[nodiscard]]/noexcept, comment density

**7.1 — `[[nodiscard]]` coverage on accessors is complete and correct. (verified, good)**
All four accessors (`data()`×2, `size()`, `bytes()`, 74–77) are `[[nodiscard]]`; discarding any is always
a bug. Matches the §7 emphasis. (The constructor itself is not nodiscard-protected — see 8.3.)

**7.2 — Naming and const-correctness are clean. (verified, good)**
`ptr_`/`size_` trailing-underscore privates, `o` for the moved-from source (consistent with
`stream.hpp`/`handles.hpp`), `e` for the local error. Const `data()` returns `const T*`; non-mutating
members are `const`. Matches the `.clang-tidy` naming contract (CamelCase types, lower_case members,
`_`-suffixed privates, UPPER_CASE macros — `STEPPE_DEVICE_BUFFER_WARN_ON_TEARDOWN` complies). No issues.

**7.3 — The release macro `((void)(errstr))` evaluates `cudaGetErrorString(e)` on the teardown-error path,
and is inconsistent with the two sibling wrappers. [low, S, before-M4.5: no]**
Location: macro 34; call site 88.
At the call site the argument is `cudaGetErrorString(e)` (88), reached only inside `if (e != cudaSuccess)`
(87). In release the macro expands to `((void)(cudaGetErrorString(e)))` — the call **is** made and
discarded. `cudaGetErrorString` is an opaque external function the compiler cannot prove side-effect-free,
so it will not be elided; it returns a static string (no device round-trip), so this is harmless, but it
is a surprising evaluate-then-discard, *and it only happens on the rare teardown-error path*, so the
practical cost is nil.
**Inconsistency, re-verified this pass (the prior draft mis-stated the sibling bodies):** `stream.hpp`
line 39 is `STEPPE_STREAM_WARN_ON_TEARDOWN(what, errstr) ((void)0)` and `handles.hpp` line 29 is
`STEPPE_HANDLES_WARN_ON_TEARDOWN(what, statusname) ((void)0)` — both expand to a literal `((void)0)` and
**discard their arguments entirely** (the two-param shape, vs `device_buffer.cuh`'s one-param
`((void)(errstr))`). So the trio differs in *both* arity and release expansion. Aligning
`device_buffer.cuh` to `((void)0)` (and dropping the `cudaGetErrorString` evaluation, or letting the
`STEPPE_LOG_WARN` migration handle it) makes the three uniform. Low, but a real cross-sibling
inconsistency.

**7.4 — Comment density matches house style; one rule stated thrice. (acceptable, cosmetic)**
The header block (1–17), the macro comment (28–32), and the `reset()` comment (82–85) each restate
"destructor never throws / fail-fast must not become fail-silent." Three statements of one rule. Trimming
the `reset()` body comment to one line would remove the redundancy without losing the citation. Cosmetic
(and consistent with the siblings, which carry the same triple-statement).

**7.5 — `bytes()`/`size()` carry no documented no-overflow invariant. [low, S, before-M4.5: no]**
Location: 76–77. `size()` and `bytes()` are exact only if the ctor enforced the 1.1 no-overflow invariant.
Until 1.1 lands, `bytes()` can in principle return a wrapped value for a buffer whose construction itself
under-allocated. A one-line doc comment ("`bytes()` is exact; the ctor rejects any `n` for which
`n*sizeof(T)` would overflow `size_t`") ties the two together and documents the precondition the §11.2
budget code will rely on. Low; pairs with 1.1.

### 8. Performance

**8.1 — Per-chunk `cudaMalloc`/`cudaFree` in the M4 bucket loop pays full synchronous-allocation cost
because the pooled allocator the architecture promises is not built yet. [med, M, before-M4.5: no]**
Location: consequence of this type's synchronous design — `cuda_backend.cu` constructs `DeviceBuffer`s
**inside** the chunk loop (248–258: `dIds`, `dQg`, `dVg`, `dSg`, `dGg`, `dVpairg`, `dRg` every iteration)
and frees them at scope exit, with `cudaStreamSynchronize` per chunk (271). Each construction is a
synchronous `cudaMalloc`; each destruction an implicitly-synchronizing `cudaFree` (4.3).
`cudaMalloc`/`cudaFree` are among the most expensive runtime calls; architecture.md §11.3 names
"excessive `cudaMalloc`/launch overhead" as the first `cudaapisum` thing to look for. The designed fix is
the §7 pool allocator (`cudaMallocAsync`/`cudaFreeAsync`, `cudaMemPoolAttrReleaseThreshold = UINT64_MAX`)
— **which does not exist yet** (verified: no `allocator.cu`). `DeviceBuffer` is correctly the *non-pooled*
owner; this is not a defect *in this file* (its synchronous design is right and simple) — it is the
performance reason this type must not be the only allocation owner. Per §2's profiler-first rule the
allocation cost should be measured (nsys) before optimizing, but the allocation count is structurally
O(n_chunks). Recorded to connect the unit to the §7/§11 strategy; not before-M4.5 (the M5 streaming/pool
milestone).

**8.2 — The empty-buffer fast path avoids a needless `cudaMalloc(…, 0)`. (verified, good)**
The `if (n)` guard (54) makes zero-size buffers cost nothing — good for the degenerate `P==0`/`M==0` paths
`cuda_backend.cu` guards (130, 298), where `total`/`pm`/`pp` can be 0.

**8.3 — The allocating ctor is not protected against a discarded temporary `DeviceBuffer<T>(n)`. [low, S,
before-M4.5: no]**
Location: ctor 53–55.
A statement like `DeviceBuffer<double>(n);` (a discarded temporary) would `cudaMalloc` then immediately
`cudaFree` — an expensive no-op the compiler will not warn about, because a class with a non-trivial
destructor cannot usefully be `[[nodiscard]]` on its *constructor* (there is no constructor-`[[nodiscard]]`;
`[[nodiscard]]` on the *class* only warns when the temporary's result is discarded, and it *does* apply to
discarded prvalues of the type). Adversarial check: is this real? Marking the **class** `[[nodiscard]]`
(`class [[nodiscard]] DeviceBuffer`) would make `DeviceBuffer<T>(n);` as a discarded statement a warning —
which is the only realistic misuse — at the cost of being unable to write `DeviceBuffer<T> tmp(n);` purely
for its allocation side-effect (never done here). The siblings (`Stream`, `Event`, `CublasHandle`) are not
`[[nodiscard]]`-classed either, so adding it only here is inconsistent. Low; mostly recorded — I lean
**reject** for consistency with the siblings (see Considered & rejected) but note it because the prior pass
didn't consider the class-level `[[nodiscard]]` dimension at all.

### 9. Layering / API / ABI vs §4

**9.1 — Layering is correct; the type never leaks above `steppe_device`. (verified, good)**
Includes only `<cuda_runtime.h>`, `<cstddef>`, `<utility>`, `check.cuh` (+ debug-only `<cstdio>`).
IWYU-clean (verified: every used symbol — `std::size_t`, `std::exchange`,
`cudaMalloc`/`cudaFree`/`cudaError_t`/`cudaSuccess`/`cudaGetErrorString`, `std::fprintf` — is covered by an
include; nothing unused; 1.1's fix adds `<limits>`). As a `.cuh` it is `PRIVATE` to `steppe_device`
(architecture.md §4, §8), so `core`/`api`/CLI cannot include it — and don't need to (they reach the GPU
only through the CUDA-free `ComputeBackend`; `CudaBackend` holds the `DeviceBuffer` members internally,
cuda_backend.cu:71–73, 340). The two `// ALLOWLISTED TU` markers (54, 86) tag exactly the
`cudaMalloc`/`cudaFree` sites the §2 CI grep gate expects. Clean.

**9.2 — No ABI concern: device-private header-only template. (verified, N/A)**
Not in `include/steppe/`, never instantiated across the public C ABI. §16 ABI rules don't apply.

### 10. Testability vs §13

**10.1 — No dedicated test for `DeviceBuffer`'s move/self-move/teardown edges, which is precisely what
§13's compute-sanitizer protects. [med, M, before-M4.5: yes]**
Verified: `DeviceBuffer` is referenced only in `cuda_backend.cu` and *mentioned in a comment* at
`tests/reference/test_f2_equivalence.cu:362` — no test exercises the type itself, and no
`test_device_buffer`/`test_raii*` target exists (verified across `tests/` and `tests/CMakeLists.txt` —
the registered tests are f2/decode/f2_blocks/filter equivalence + 3 host-only unit tests).
**Calibration vs the first pass:** §6 and §13 mandate a test for the new *kernel* / "thin kernels get a
launch-and-compare test" (architecture.md line 757) — they do **not** literally require a unit test per
RAII wrapper, so the first pass slightly over-stated "§6 makes a test part of done for every BUILT unit."
But the real justification stands and is strong: §13's CI runs compute-sanitizer `memcheck`/`racecheck`
with `--error-exitcode 1` (lines 762, 898), and **the sanitizer only protects code a test actually
executes.** The happy `n>0` path is covered indirectly by the f2 equivalence tests; the
move-construct-nulls-source, move-assign-frees-then-steals, self-move-no-op, and moved-from-teardown
paths are **never executed by any test**, so a double-free regression there would pass the f2 test and
surface only under sanitizer on an unrun path. The §13 `CudaTest` fixture (resets device, asserts clean
error state in `TearDown`, line 757) is the intended host.
Concrete fix: `tests/unit/test_device_buffer.cu` (or a shared `test_raii_wrappers.cu` covering
`DeviceBuffer`/`Stream`/`Event`/`CublasHandle`, which share the move-only shape) asserting: (a)
`size()`/`bytes()` post-construction; (b) `data()` non-null for `n>0`, null for `n==0`; (c) after `b =
std::move(a)`: `a.data()==nullptr && a.size()==0`, `b` owns the original pointer; (d) self-move-assign is
a no-op (run under `compute-sanitizer memcheck`); (e) teardown of a moved-from buffer is clean. Med;
before-M4.5 (M4.5 replicates this wrapper shape into `NcclComm`/`StreamPool`/`PinnedBuffer` per TODO line
95 / `00-overview.md` — establishing the test pattern now sets the standard the replicas inherit).

**10.2 — Not GPU-free-testable, inherently. (verified, acceptable)**
Every operation hits the CUDA runtime, so unlike §13's pure `__host__ __device__` numerics this can't be
tested on a GPU-free host. That is inherent to a device-memory owner; its test belongs in the GPU tier
(10.1). No fix beyond 10.1.

### 11. Capability tiers (PRO-6000-capable vs budget-5090) vs TODO

**11.1 — The capability probe / device-id the M4.5 P2P-vs-host-staged combine needs lives in
`DeviceConfig`/`Resources`, not here — so this type needs no probe; the only residue is the
`cudaMemcpyPeer` source-device id, already covered by `PerGpuResources`. [low, S, before-M4.5: no]**
TODO `wxz1fiiln` requires every run to record "which path it took + why it degraded," with the M4.5
combine being **P2P device-resident** (capable: RTX PRO 6000 stock driver / full-host 5090 aikitoria
patch) vs the **host-staged fixed-order combine** (budget vast 5090), gated by `cudaDeviceCanAccessPeer`
with the explicit logged tag *"P2P combine unavailable (no peer access) → host-staged fixed-order
combine"* (TODO line 128; verbatim-confirmed in §11.4 this pass). **But the TODO is explicit that this
"slots into `DeviceConfig`/`Resources`" (line 137)** — the capability detection and the capability-tagged
result belong there, *not* in `DeviceBuffer`. The P2P combine's `cudaMemcpyPeer(dst, dstDevice, src,
srcDevice, bytes)` needs the source device id — but per §11.4 the peer partials are owned at the
`PerGpuResources` level (`int device_id`), so the combine code already has the id without `DeviceBuffer`
carrying it. Same root as 4.6, scoped to the capability tier, same conclusion: no probe logic belongs in
`DeviceBuffer`; the optional `device()` accessor (4.6) is a convenience, not a requirement. Low.
**`DeviceBuffer` correctly touches neither capability tier** — it only moves bytes; the tiering is
data-movement/observability only and parity-neutral (11.2).

**11.2 — Parity-neutrality preserved: this type only moves bytes. (verified, good)**
Per §11.4 / TODO ("bit-identical on both paths … the transport only moves bytes; software fixes the
order"; "every lever is parity-neutral"). `DeviceBuffer` never participates in reduction order, so adding
`device()` or a pooled variant cannot affect parity. Correctly orthogonal to the §12 determinism contract.
Confirms 4.6/11.1's optional additions are parity-safe.

---

## Considered & rejected

- **(First pass's 4.6 at HIGH severity — "silent-wrong-result / illegal-access in M4.5").** Downgraded to
  low–med (4.6 above), not rejected, but the *high* framing is rejected: per §3.4 / §11.4 a cross-device
  kernel launch fails loudly (caught by `STEPPE_CUDA_CHECK_KERNEL`) rather than silently corrupting, and
  the architecture homes `device_id` on `PerGpuResources` (line 605) with the capability probe in
  `DeviceConfig`/`Resources` (TODO line 137) — so a `device_id` on each `DeviceBuffer` is defense-in-depth,
  not a mandated chokepoint. (Citation honesty: the exact §3.4 "kernel launch will fail …" sentence is
  un-re-verified this pass — see 4.6 — but the conclusion is corroborated by §11.4's design.)
- **First pass's 1.2 as a separate finding.** Folded into 1.1 — it is the *same* `n*sizeof(T)` arithmetic;
  the budget-summation angle is a sub-note, not a distinct fix-site bug.
- **First pass's 11.1 at MED.** Downgraded/folded into 4.6 — the device-id residue is real but lives at
  the resource level; nothing capability-specific belongs in `DeviceBuffer`.
- **`device_buffer.cuh` should use the §7-sketched `STEPPE_DEBUG_ONLY(...)` macro instead of rolling its
  own `#if defined(NDEBUG)`.** Rejected — verified `STEPPE_DEBUG_ONLY`/`STEPPE_ASSERT` **do not exist** in
  the codebase (grep over `src/`/`include/`/`tests/` = 0 hits; they appear only in `architecture.md` prose
  and are tracked as phantoms in `00-overview.md`/`core-internal-views.md`). Rolling the local
  `NDEBUG`-guarded macro is the correct interim, consistent with the two siblings, until the cross-cutting
  facility lands. Not a defect.
- **Mark the *class* `[[nodiscard]]` to catch a discarded `DeviceBuffer<T>(n);` allocation.** Considered
  (8.3) and leaned-reject: it would warn on the only realistic misuse, but the siblings (`Stream`,
  `Event`, `CublasHandle`) are not `[[nodiscard]]`-classed, so adding it only here is inconsistent and the
  misuse is not observed in the codebase. Recorded as 8.3 (low) rather than dropped, because the prior
  pass didn't consider it; if adopted, apply uniformly to all four wrappers.
- **`operator=(DeviceBuffer&&)` should be conditionally `noexcept`.** Rejected — it is unconditionally
  `noexcept` and correctly so: `reset()` is `noexcept` and `std::exchange` on trivial scalars cannot throw;
  no `T`-dependent operation occurs (the buffer stores raw bytes, never `T` objects).
- **Make `data()` return `T* const` / add `__restrict__`.** Rejected — `__restrict__` is a
  kernel-parameter aliasing hint, meaningless on a `T*` returned to host code; if wanted it belongs on the
  `launch_xxx` signatures.
- **`bytes()` should return `cudaMalloc`'s rounded-up (≥256-byte) actual size.** Rejected — the §11.2
  budget and every consumer reason about *logical* bytes; the runtime's internal rounding is not
  observable via the runtime API and not what callers want.
- **Add `resize()`/`reallocate()`.** Rejected — invites realloc-in-hot-loop; the architecture's answer to
  repeated sizing is the pool allocator (§7), and immutable-after-construction is the right constraint.
  (The per-chunk reconstruction in `cuda_backend.cu` is the §7/§11 pool-allocator gap of 8.1, not a
  missing `resize`.)
- **`cudaFree` in the dtor should be `cudaFreeAsync` on the owning stream.** Rejected *for this type* —
  `DeviceBuffer` is deliberately the synchronous, stream-agnostic owner; stream-ordered freeing is the
  pooled sibling's job (§7, see 4.3/8.1). Conflating them forces every buffer to carry a stream it often
  lacks.
- **`#include <cstdio>` in the macro `#else` is a header layering smell.** Rejected as a current defect —
  gated to debug only and only to back the tracked-temporary teardown warning the header explicitly flags
  for `STEPPE_LOG_WARN` replacement. Folded into 2.4/4.7.
- **Guard against `cudaMalloc` succeeding but leaving `*devPtr` unchanged.** Rejected —
  `STEPPE_CUDA_CHECK` throws on any nonzero status; on success `cudaMalloc` writes `devPtr`; there is no
  documented success-with-null-out case.
- **`T` could be `void`/incomplete and break `sizeof(T)`.** Rejected — `sizeof(T)` ill-formed on
  incomplete `T` is a *compile-time* failure (fail-fast at the call site), and all instantiations are
  scalar (`double`/`long`/`int`/`std::size_t`/`std::byte`/`std::uint8_t`, verified via grep). No runtime
  gap.
- **Mark the class `final`.** Considered. Rejected: no virtuals, never used as a base, and the siblings are
  not `final` — adding it only here is inconsistent and gains nothing. Cosmetic; not a finding.

---

## What it takes to reach 10/10

In rough priority (severity × proximity to the M4.5 milestone this branch targets):

1. **(4.1, med, before-M4.5)** Restore `cuda::std::span<T> view()` + `const` overload (the §7 reference
   member; TODO line 92), landing it with the §8 `internal/span_view.hpp` home, `#include <cuda/std/span>`,
   **and** moving the `launch_*` wrapper signatures (`f2_blocks_kernel.cuh`) to span-first so `view()` is
   actually consumable — the TODO bundles both ("+ new M4.5 wrappers span-first"). Keep `data()` for the
   cuBLAS/`cudaMemcpyAsync` raw-pointer sites. This is the single biggest spec divergence.
2. **(1.1 / 7.5, med, before-M4.5)** Add a checked-multiply overflow guard in the ctor (`n >
   SIZE_MAX/sizeof(T)` → typed error; `#include <limits>`) so a wrapped size can never under-allocate, and
   document `bytes()`/`size()` as exact under that invariant for the §11.2 budget.
3. **(10.1, med, before-M4.5)** Add `tests/unit/test_device_buffer.cu` (or a shared `test_raii_wrappers.cu`)
   covering construct/size/`data()` null-ness, move-construct nulls source, move-assign frees-then-steals +
   self-move no-op, and clean teardown of a moved-from buffer — run under `compute-sanitizer memcheck`. Sets
   the pattern the M4.5 `NcclComm`/`StreamPool`/`PinnedBuffer` wrappers inherit.
4. **(2.4 / 4.7 / 5.1 / 6.3 / 7.3, low–med)** When `internal/log.hpp` lands (00-overview B7), replace the
   `fprintf`-to-stderr warning with `STEPPE_LOG_WARN` — removes the §10 `printf`-in-library violation, the
   header-macro leak, the duplicated/mismatched-shape prefix across the three wrappers, and the
   `cppcoreguidelines-macro-usage` smell in one move. Until then, align the release expansion to `((void)0)`
   like `stream.hpp`/`handles.hpp` so the trio is uniform and the dead `cudaGetErrorString` evaluation is
   dropped. (6.3 is really a project-wide tidy-config item; fix it uniformly, not just here.)
5. **(4.3 / 8.1, med, M5)** Document that destruction is an implicit device-wide sync point and that
   hot-loop/streaming buffers should come from the pool allocator (`allocator.cu`) once it lands; build the
   stream-ordered pooled sibling at M5 so §11.1 overlap and the per-chunk `cudaMalloc` cost are addressed
   without changing `DeviceBuffer`'s simple synchronous role.
6. **(4.6 / 11.1, low–med, optional)** *Optionally* record `cudaGetDevice(&device_id_)` and expose
   `[[nodiscard]] int device() const noexcept` as defense-in-depth for the M4.5 multi-GPU path — but note
   the architecture already homes `device_id` on `PerGpuResources`, so this is convenience, not required.

Items 1–3 are the substantive gaps against §7/§13 and the M4.5 milestone; 4–6 are quality/perf polish and
an optional convenience. None require restructuring the type.

---

## Good patterns to keep

- **Move-only with full move-construct AND move-assign**, both via `std::exchange`, with the
  self-assignment guard and `reset()`-before-steal — exactly the §7 shape, and the bug the architecture
  called out (deleted move-assign) is fixed here. Identical discipline in `stream.hpp`/`handles.hpp`.
- **Single `reset()` helper** shared by move-assign and the destructor — no duplicated free logic (§2 DRY).
- **`noexcept` discipline done right**: everything that can be is `noexcept`; the one throwing member is the
  allocating ctor (which must be), and `reset()` deliberately uses bare `cudaFree` + manual status
  inspection instead of the throwing `STEPPE_CUDA_CHECK` to keep the destructor `noexcept` — a subtle
  detail correctly handled (and matched by the siblings).
- **`n == 0` short-circuit** to `{nullptr, 0}` with no runtime call — deterministic, cheap, and documented
  as a contract; the `cudaFree(nullptr)` no-op guard avoids a stray runtime call (and a stale-sticky-error
  surface) on the moved-from/empty path.
- **Complete `[[nodiscard]]` coverage** on all accessors, correct const/non-const `data()` overloads,
  IWYU-clean includes.
- **Allowlist discipline**: the two `// ALLOWLISTED TU` markers tag exactly the `cudaMalloc`/`cudaFree`
  sites the §2 CI grep gate expects, and the type holds the layering (CUDA-private `.cuh`, no leakage above
  `steppe_device`).
- **Size-agnostic**: the cuBLAS workspace size lives once in `config.hpp::kCublasWorkspaceBytes` and is
  injected, never inlined here.
- **Tracked-temporary honesty**: the teardown-warning placeholder names its replacement (`STEPPE_LOG_WARN`
  once `log.hpp` lands) rather than pretending to be final, and is consistent in shape with the sibling
  `Stream`/`Event`/`CublasHandle` wrappers — and it correctly rolls a local `NDEBUG` guard rather than
  citing the still-nonexistent `STEPPE_DEBUG_ONLY`.

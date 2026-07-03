# `handles.hpp` reference

## 1. Purpose

`src/device/cuda/handles.hpp` is the home for the small, owning C++ wrappers
around the raw GPU-library handles steppe uses: cuBLAS (matrix multiply),
cuSOLVER (dense linear algebra), and cuFFT (fast Fourier transforms). Each raw
handle is a resource that must be explicitly created and destroyed. Left as bare
pointers, they leak whenever an error is thrown between the create call and the
matching destroy call. The wrappers here turn every one of those handles into an
object that frees itself automatically, on every exit path — normal return and
error unwind alike.

The file holds five wrapper types and a few supporting pieces:

- **`CublasHandle`** — an owning cuBLAS handle that also enforces the two
  invariants below (stream/workspace and device ordinal).
- **`MathModeScope`** — a temporary, self-restoring change to a cuBLAS handle's
  math mode.
- **`CusolverDnHandle`** — an owning dense-cuSOLVER handle.
- **`GesvdjInfo`** — an owning wrapper for a cuSOLVER Jacobi-SVD parameter
  structure.
- **`CufftPlan`** — an owning wrapper for a cuFFT plan.
- **`CusolverMathModeScope`** plus the `engage_solver_precision` helper and a
  capability-probe macro — the seam that can promote a cuSOLVER solve to the
  faster emulated-double-precision path.

Because it includes `cublas_v2.h`, `cufft.h`, and `cusolverDn.h`, this is a CUDA
header. It is private to the GPU layer of steppe and is never compiled into the
core library, the public API, or the command-line tool.

Two rules recur through every wrapper in this file. They are stated once here and
referenced by the class sections:

- **Move-only.** Each wrapper can be moved (ownership transferred) but not
  copied. A moved-from wrapper owns nothing and is safe to destroy — it holds a
  null/zero handle so its destructor does nothing. This mirrors how steppe's
  other resource types (`Stream`, `DeviceBuffer`) behave.
- **Teardown never throws.** Every destructor is written so it cannot throw. If
  the underlying destroy call reports a nonzero status, that status is routed to
  a debug warning sink rather than thrown, so that "fail fast" during normal
  operation never turns into "fail silent" or a crash during stack unwinding.
  All of these wrappers share the same one warning sink for teardown messages.

---

## 2. The stream-and-workspace invariant

This is the single most important, most non-obvious rule in the file, and the
reason `CublasHandle` is a custom type rather than a thin pointer holder.

The emulated-double-precision matrix-multiply path (the fast "f2" path) needs
cuBLAS to use a fixed, explicitly-supplied scratch buffer — its *workspace* — in
order to produce the exact same result on every run. That workspace is pinned
once by calling `cublasSetWorkspace`.

The trap: cuBLAS's own documentation states that calling `cublasSetStream`
"unconditionally resets the cuBLAS library workspace back to the default
workspace pool." In other words, **any** change to the stream that happens
*after* the workspace was pinned silently throws the pinned workspace away and
reverts to cuBLAS's default pool. The moment that happens, the run-to-run
determinism the reference results depend on is quietly defeated.

This hazard cannot be caught at the call sites. Whether it fires depends on the
order in which the stream and workspace happen to be set, and that ordering is
invisible when you look at any single call. So the fix is owned by the type
itself: `CublasHandle` remembers the pinned workspace (as a non-owning pointer
and byte count), and its `set_stream()` method **re-applies** that workspace
immediately after every `cublasSetStream`. Callers must route all stream changes
through `set_stream()` and must never call raw `cublasSetStream` on the handle
returned by `get()`.

Note that this reset behavior is specific to the workspace. A cuBLAS handle's
*math mode* is not reset by a stream change — see section 6.

cuSOLVER does not have this hazard: `cusolverDnSetStream` does not touch any
pinned workspace (in the modern dense cuSOLVER API the workspace is passed per
call), so `CusolverDnHandle::set_stream` does not need to re-apply anything.

---

## 3. The device-ordinal invariant

A cuBLAS (and likewise cuSOLVER) context is bound to whichever CUDA device is
"current" at the moment the context is created. If a handle is later used while a
*different* device is current, its work either runs on the wrong GPU or fails
outright with an architecture-mismatch error.

To catch that class of bug, each owning handle **records** the device ordinal
that was current when it was created (via `cudaGetDevice` in the constructor).
Then, every operation that configures or issues work on the handle
**debug-asserts** that the currently-current device still matches the recorded
one.

Two deliberate properties of this scheme:

- It is **record-and-assert only** — it never calls `cudaSetDevice`. Selecting
  the device is the caller's job (in practice, the per-GPU resource holder).
  Making the wrapper silently switch devices would introduce hidden global state,
  which is exactly what this design avoids. The wrapper only verifies the
  precondition.
- It is **debug-only**. The whole query-and-assert is compiled out under
  `NDEBUG`, so a release build pays nothing for it. On a single-GPU run the
  assertion is always trivially true (the ordinal is always 0), and it changes no
  reported result — it is purely an observability guard, and it is scaffolding
  for the eventual multi-GPU work.

The moved-from device ordinal is carried through moves, because the moved-into
handle owns the same underlying context. The default `-1` is a defensive
sentinel that is never actually observed: the constructor always sets the real
ordinal before any use, and throws if the query fails.

---

## 4. Shared error handling and the warning sink

The wrappers split their error handling by phase, and it is worth understanding
the split:

- **Acquire and configure calls may throw.** Constructors and configuring
  methods (`set_workspace`, `set_stream`, `make`, the math-mode scope
  constructors) route failures through the checking macros, which throw a typed
  error. A constructor is allowed to throw; that is how an acquisition failure is
  surfaced.
- **Teardown never throws.** Every destructor (and the `restore()` /`destroy()`
  helpers behind them) checks the destroy/restore status and, if it is nonzero,
  writes it to the shared debug warning sink instead of throwing. The status is
  turned into a human-readable name for the message. This keeps a failing
  teardown from either being swallowed entirely or from throwing during stack
  unwinding.

The debug device-ordinal assert (section 3) also runs at teardown for the two
context-owning handles, so the creation device is checked at destroy time too.
Today the underlying destroy calls tolerate a different current device, but the
multi-GPU teardown path runs under whatever device happens to be ambient, so this
assert would catch a future toolkit that starts minding the current device at
destroy time.

---

## 5. CublasHandle

An owning, move-only cuBLAS handle. It is the workhorse for every matrix
multiply. The intended lifecycle:

1. **Create once** — construct it while the intended device is current
   (constructor records that device ordinal, section 3).
2. **Pin the workspace once** via `set_workspace(ptr, bytes)`. The buffer is
   non-owning: the caller (a `DeviceBuffer` in the backend) must outlive the
   handle's use of it. Because destruction runs in reverse declaration order, the
   backend declares the handle *before* the workspace buffer, so the buffer is
   still alive when the handle is torn down.
3. **Bind the statistics stream once** via `set_stream(stream)`, which also
   re-applies the workspace (section 2).
4. **Reuse it for every matrix multiply**, handing `get()` only to the cuBLAS
   compute entry points — never to a raw `cublasSetStream`.

Creating the handle once and reusing it is deliberate: destroying a cuBLAS handle
implicitly synchronizes the device, so a handle is never recreated per iteration.

Public surface:

- `get()` — the raw `cublasHandle_t` for the compute calls.
- `device_id()` — the recorded creation-device ordinal (always 0 on single-GPU),
  used by multi-GPU orchestration for logging and the device assert.

A moved-from `CublasHandle` holds a null handle and owns nothing.

---

## 6. MathModeScope

A small stack-scoped object that changes a cuBLAS handle's *math mode* for the
duration of a scope and then puts it back exactly as it was.

The reason it exists: a cuBLAS math mode is **sticky** handle state. Unlike the
workspace, it is *not* reset by a stream change — once set, it stays set until
something changes it again. The f2 path sets a math mode once per compute call
(an emulated-double-precision mode for an honored emulated request, otherwise a
pedantic native mode) and never restores it. That is harmless while a single
precision owns the handle, but it becomes a determinism hazard the moment the
mandatory verification pass recomputes a sample of jackknife blocks in native
double precision on the *same shared handle* that an emulated run is using —
whichever ran last would silently leak its math mode into the next.

`MathModeScope` makes such a change observably scoped: the constructor captures
the handle's current mode (`cublasGetMathMode`) and applies the requested one;
the destructor restores the captured mode. So a native recompute can engage its
mode for its own scope and leave the handle exactly as it found it.

Details worth noting:

- It takes a raw `cublasHandle_t`, not a `CublasHandle&`, so it composes with the
  existing f2 code that already operates on the raw handle.
- It is **non-owning** and safe only under strictly stack-scoped use: it must be
  constructed inside a live owning handle's lifetime and always destruct before
  that handle. If it were ever promoted to a long-lived member or moved out and
  held, the owning handle would have to be proven to outlive it, or the restore
  would run against a torn-down context.
- Move-only and never-throwing teardown per section 1. A moved-from scope is
  inert and restores nothing.

---

## 7. CusolverDnHandle

An owning, move-only dense-cuSOLVER handle — the home for the qpAdm fit's small
linear-algebra primitives: Cholesky factor/inverse for the symmetric
positive-definite matrices, LU solves (including batched) for the alternating and
weighted least-squares and leave-one-out steps, and Jacobi SVD for the rank test.

It mirrors `CublasHandle`: RAII, move-only, and the same device-ordinal
record-and-assert (section 3). The caller selects the device before constructing
it. The differences from `CublasHandle`:

- **No workspace-reset hazard.** As noted in section 2, `set_stream` just binds
  the stream; there is nothing to re-apply.
- **Deterministic mode pinned at creation.** The constructor calls
  `cusolverDnSetDeterministicMode(handle, CUSOLVER_DETERMINISTIC_RESULTS)` once,
  right after create. On the current toolkit that is *already* the default, so
  this is a purely **defensive** pin — it changes nothing today (the reference
  results do not move) but forecloses a future toolkit silently flipping the
  default and de-determinizing the rank-test SVD. This deterministic mode is
  sticky handle state and governs **only** the general/Jacobi SVD family
  (`gesvd`/`gesvdj`). It does *not* cover the Cholesky and LU routines
  (`potrf`/`potri`/`getrf`/`getrs`), which stay in native double precision and
  are outside the deterministic set. The primary bit-exact rank test for the
  small reference model does not use cuSOLVER at all — it uses a custom
  fixed-order on-device Jacobi routine that is deterministic by construction and
  untouched by this setting.

Public surface:

- `get()`, `device_id()` — as in `CublasHandle`.
- `set_stream(stream)` — bind the stream (no workspace re-apply).
- `deterministic_mode()` — reads back the currently-set deterministic mode
  (`cusolverDnGetDeterministicMode`). This lets a unit test assert the pin
  literally. Unlike the teardown paths, this is a state query, so it is allowed
  to throw on a cuSOLVER fault (it is not marked `noexcept`).

---

## 8. GesvdjInfo

An owning, move-only wrapper for a cuSOLVER `gesvdjInfo_t` — the parameter
structure for the Jacobi (one-sided) SVD routine. Creating it heap-allocates the
structure (which can fail with an allocation error) and there is a matching
destroy call; the two are a paired create/destroy just like the handle
create/destroy.

Why it needs to be RAII: the bare-pointer idiom (create, then run throwing SVD
and error-check calls, then destroy at the end) leaks the structure if any of
those throwing checks unwinds past the destroy line. This was the only non-RAII
resource left in the backend translation unit, which otherwise holds every handle
under an RAII owner. Wrapping it means an exception anywhere in the Jacobi branch
still frees the structure on unwind.

Two things that make it simpler than the handle wrappers:

- It is created and destroyed **per SVD shape**, not reused across calls. It is a
  cheap host-side configuration struct, not a per-call device allocation, so
  there is no reuse contract to honor — each Jacobi branch makes a fresh one.
- It carries **no device ordinal**. A `gesvdjInfo_t` is plain host-side
  configuration and is not bound to a CUDA context, so it needs no
  record-and-assert.

The constructor uses cuSOLVER's default tolerance and sweep count (callers leave
them at default). `get()` returns the raw structure. Move-only, never-throwing
teardown; a moved-from wrapper owns nothing.

---

## 9. CufftPlan

An owning, move-only wrapper for a cuFFT plan — used by the DATES
autocorrelation-curve engine's batched forward and inverse transforms.

The stakes are higher than for a plain handle: a cuFFT plan *backs a device
workspace*. Per cuFFT's documentation, the intermediate buffer allocations happen
during planning and are released when the plan is destroyed. So a leaked plan
leaks GPU memory, and the leak scales with retries.

Why it needs RAII: the engine previously held two plans as bare handles and freed
them with a bare destroy pair at the end of the function — past a throw window.
The stream-set call between the two creates can throw, the transform-exec calls
inside the per-sample loop can throw, and any error check between create and
destroy can throw. On any throw the function unwinds and the tail destroy is never
reached, leaking both plans. This was the one remaining resource in the backend
not behind an RAII owner. Making the plans `CufftPlan` members tears them down on
every exit path.

Design points:

- Like `GesvdjInfo`, a plan is created and destroyed **per FFT shape**, not reused
  across calls (one plan is set up per curve call and reused across all target
  samples), and it carries **no device ordinal** (the engine is single-stream on
  the backend's device, and destroy carries the plan's own device).
- A `cufftHandle` is a plain integer; `0` is the unambiguous "no plan" sentinel,
  because the cuFFT API never returns `0` as a live plan. A default-constructed
  `CufftPlan` holds `0` and destroys nothing until `make()` populates it.

Public surface:

- `make(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)`
  — creates a batched plan into this owner (the argument order matches
  `cufftPlanMany`). It first destroys any plan already held; on a throw the owner
  is left holding no plan, so nothing leaks.
- `set_stream(stream)` — binds a stream to the plan.
- `get()` — the raw `cufftHandle`.

---

## 10. The emulated-FP64 cuSOLVER capability probe

Near the top of the file, a compile-time probe decides whether the current CUDA
toolkit even *has* an emulated-double-precision math mode for cuSOLVER, and
publishes the answer as the macro `STEPPE_HAVE_CUSOLVER_FP64_EMULATED` (`1` if
present, `0` if not).

The background: the cuBLAS f2 path uses an emulated-double-precision fixed-point
math mode, and that enum exists in the cuBLAS headers of the current toolkit. But
cuSOLVER's math-mode enum is a **separate, narrower** enum. On the current toolkit
it exposes only a default mode and a BF16-based single-precision-emulation mode —
there is *no* emulated-double-precision cuSOLVER mode yet. So the promotion seam
(section 11) cannot hardcode an emulated-double cuSOLVER enum, because it would
not compile.

The macro is a single, forward-compatible point of truth:

- If a future toolkit adds an explicit emulated-double cuSOLVER enumerator, the
  macro becomes `1` and the seam genuinely promotes to it.
- Absent it today, the macro is `0` and the seam degrades to the native default
  mode instead. The probe deliberately does **not** invent a numeric value —
  feeding cuSOLVER an out-of-range mode would be undefined behavior, the opposite
  of the behavior-preserving intent.

### The one-shot downgrade warning

Compiled only on the "no emulated mode" build, `warn_cusolver_emulated_fp64_unavailable_once`
emits a single warning, at most once per process, the first time an honored
emulated-double solve is requested but the toolkit has no emulated-double
cuSOLVER mode. It exists so the downgrade to native is observable without spamming
the per-solve hot path. The one-shot guard uses an atomic flag so it is safe when
more than one host thread engages it. It routes through the same one warning sink
as the teardown messages, and it states plainly that the reported precision in
that case is native double precision, not emulated — and that only the target
math mode is gated, so a newer cuSOLVER promotes with no code change. On a build
whose toolkit *does* have the mode, this helper is not compiled at all, so there
is no unused-function warning.

---

## 11. CusolverMathModeScope and the precision-promotion seam

The cuSOLVER analogue of `MathModeScope` (section 6), plus the reason it exists.

The motivation is throughput. Native double precision is essentially free when a
run fits a single model. But the large model rotation runs millions of small
Cholesky/SVD/least-squares solves, where native double precision on the target
hardware is a tensor-core-throughput wall. This scope is the seam that lets an
individual solve stage be *promoted* to the emulated-double (tensor-core) path
**without changing the default**. The default qpAdm code still passes native
precision, so the reference parity is unchanged; the deliverable is the
*capability* to promote, not a default flip.

Like the cuBLAS math mode, `cusolverDnSetMathMode` is **sticky** handle state, and
the per-device solver handle is shared across stages and across models in the
rotation. So an unscoped, imperative promote would silently leak the emulated mode
into the next native (oracle or ill-conditioned) solve. `CusolverMathModeScope`
makes the change observably scoped: the constructor captures the current mode and
applies the requested one; the destructor restores the captured mode.

How the requested mode is chosen — the **honorability** flag:

- The constructor takes a pre-decided `honorable` boolean. The caller computes it
  with the same single predicate the cuBLAS f2 path uses, so the two paths can
  never disagree about whether an emulated request is honored.
- `honorable == true` requests the emulated-double cuSOLVER mode; `false` (native
  double, TF32, or an unhonorable emulated request) requests the native default.
- The emulated-double mode is *additionally* gated on toolkit support (section
  10). When it is unavailable — the case today — an honored request degrades to
  native and emits the one-shot tag. Even then the seam is **live**: it still
  makes a real `cusolverDnSetMathMode` call and restores the prior mode; only the
  target mode is native. This is not a no-op stub — the get/apply/restore
  round-trip exercises the real cuSOLVER API on every scope.

Public surface:

- `promoted()` — true only when the scope *actually* engaged the emulated-double
  mode (an honored request on a toolkit that has the mode). When it is false for
  an honored request, the solve ran native and the one-shot tag was emitted. This
  is the observable promotion state, for tests and logging.

Same non-owning, strictly-stack-scoped safety contract as `MathModeScope`: it
takes a raw `cusolverDnHandle_t` (so it composes at solve sites that already hold
the raw handle), must be constructed inside the owning handle's lifetime, and must
destruct before it. Move-only; a moved-from scope is inert. Teardown never throws.

---

## 12. engage_solver_precision

A one-line inline helper that builds a `CusolverMathModeScope` from a typed
`Precision` value. It is the call the qpAdm solve sites use to engage the seam.

It routes the honorability decision through the **same** predicate the f2 path
uses, but takes that predicate as a function pointer so this header does not have
to include the device-private kernel header where the predicate lives (the caller
passes the predicate's address).

The practical effect: the default qpAdm code passes native double precision, so
the predicate returns "not honorable," the scope targets native, and the
reference golden parity is unchanged. A future per-stage policy can pass an
emulated-double `Precision` to promote that one solve — and, on a toolkit that has
the emulated-double cuSOLVER mode, get the faster path for free.

I read through this carefully. This is **not slop** — it's clearly written by someone who understands RAII, CUDA context lifetime, and the sharp edges of cuBLAS/cuSOLVER/cuFFT. A senior developer would respect the engineering but would also flag the **extreme comment density**, some **release-silent invariants**, and a few **API rough spots**.

## What's genuinely good

- **The cuBLAS workspace/stream invariant is handled correctly.** The comments at lines 12–24 and the implementation at `set_stream()` (lines 157–161) show real awareness of the cuBLAS §2.4.7 footgun: `cublasSetStream` silently resets the workspace, so the wrapper re-applies it. This is the kind of subtle vendor behavior that leaks bugs for months.
- **RAII wrappers are move-only, non-copyable, and destructor-noexcept.** All four handle classes (`CublasHandle`, `CusolverDnHandle`, `GesvdjInfo`, `CufftPlan`) get this right: `std::exchange` on move, deleted copy ops, and `destroy()` is `noexcept` with warnings routed to `STEPPE_LOG_WARN` instead of throwing.
- **The math-mode scope guards are the right abstraction.** `MathModeScope` (lines 264–316) and `CusolverMathModeScope` (lines 634–714) capture sticky handle state and restore it. This is exactly how you avoid silent precision leakage between stages that share a handle.
- **The FP64-emulated promotion seam is forward-compatible without inventing enum values.** The `#if defined(CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH)` probe (lines 85–89) and the downgrade-with-one-shot-warning path (lines 674–690) are disciplined. The code does not feed an out-of-range int to `cusolverDnSetMathMode`.
- **Move constructors and assignments are exception-safe and self-assignment safe.** The `if (this != &o)` guard and `destroy()`-then-move pattern are consistent across all classes.
- **`GesvdjInfo` fixes a real leak hazard.** The comment at lines 405–417 correctly identifies that bare `cusolverDnCreateGesvdjInfo` / `cusolverDnDestroyGesvdjInfo` pairs leak if a throwing check unwinds between them.

## What a senior developer would flag

**The comment-to-code ratio is out of control.**

This file is 734 lines, roughly 500+ of which are comments. Many are excellent, but many are repetitions of the same invariant across multiple classes. For example, the "record-and-assert / never `cudaSetDevice`" paragraph appears almost verbatim for `CublasHandle` (lines 26–39), `CusolverDnHandle` (lines 319–329), and again inside `assert_on_creation_device()` (lines 171–192, 362–379). A senior reviewer would say: "Extract the invariant, or trust the reader to read it once."

**The device-ordinal invariant is a debug-only no-op in release.**

```cpp
void assert_on_creation_device() const noexcept {
    STEPPE_DEBUG_ONLY(
        int current = -1;
        const cudaError_t e = cudaGetDevice(&current);
        STEPPE_ASSERT(e == cudaSuccess, ...);
        STEPPE_ASSERT(current == device_id_, ...));
}
```

This is fine for observability, but the comment promises it "catches" multi-GPU bugs. In a release build it catches nothing — the handle will silently be used on the wrong device. That's a valid engineering tradeoff (the query is not free), but the prose oversells it.

**`CufftPlan::make` has an awkward two-phase initialization API.**

```cpp
CufftPlan plan;
plan.make(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
```

A constructor that throws on failure is cleaner than a default-constructed empty shell followed by `make()`. The comment says it "mirrors a default-constructed owning wrapper," but default-constructed owning wrappers usually acquire in a constructor or factory. The raw `int*` arguments are also inherited from `cufftPlanMany` and not the wrapper's fault, but the two-phase pattern pushes lifetime complexity onto the caller.

**`CusolverMathModeScope` and `MathModeScope` are largely duplicated.**

They differ in handle type, enum type, and the honorable/downgrade logic, but the capture/restore/move/dtor structure is near-identical. A template or a small helper would cut the duplication and reduce the surface area for copy-paste drift.

**`engage_solver_precision` takes a C-style function pointer.**

```cpp
[[nodiscard]] inline CusolverMathModeScope engage_solver_precision(
    cusolverDnHandle_t handle, const Precision& precision,
    bool (*honorable_predicate)(const Precision&) noexcept) {
```

It works and avoids a header include, but a template predicate or a tiny type-erased callable would be more idiomatic C++. As written, `noexcept` on a function pointer is easy to get wrong at the call site.

**Internal cleanup ticket references are noise for external readers.**

Phrases like "cleanup X-1/B1", "cleanup device-cuda-handles 2.3/11.x", "TODO M4.5 line 98", and "cleanup [17.5]" appear repeatedly. They are useful inside the team but make the file read like a project tracker rather than source code. At minimum, keep them in one place (the file header) instead of sprinkling them through every docstring.

**The `CufftPlan` empty-state sentinel is inconsistent with the pointer-sentinel style used elsewhere.**

`CufftPlan` uses `plan_ == 0` because `cufftHandle` is an integer, while the other wrappers use `nullptr`. This is technically correct, but a reader has to remember which handle type uses which convention. A named constant or `static_cast<cufftHandle>(0)` would help.

**The one-shot warning helper uses `std::atomic_flag` without a relaxed-ordering justification.**

```cpp
static std::atomic_flag emitted;
if (!emitted.test_and_set(std::memory_order_relaxed)) {
```

`memory_order_relaxed` is fine for a one-shot warning, but the comment doesn't explain why weaker ordering is acceptable. A `std::once_flag` / `std::call_once` pattern would be clearer and harder to misuse.

## The "slop" test

**Not slop.** Slop would be:
- Bare `cublasCreate`/`cublasDestroy` pairs with no RAII
- Copy-pasted handle wrappers that disagree on move semantics
- Ignoring the cuBLAS workspace-reset hazard
- Hardcoding a nonexistent cuSOLVER enum value

None of that is here. The code is careful, correct, and well-reasoned. The verbosity is a style choice, not incompetence.

## What it actually looks like

This looks like **solid, defensive systems code written by an engineer who has been burned by CUDA context/state bugs before and is determined not to be burned again.** The RAII wrappers are correct, the vendor-documented footguns are addressed, and the forward-compatible promotion seam is thoughtful.

A senior C++ reviewer would say: "Competent, ship it — but cut the comments in half and turn the duplicate scope guards into a template." A senior CUDA reviewer would say: "The handle lifetime and workspace invariants are right; the debug-only device check is fine, but don't pretend it protects release builds."

The main risk to a job-application showcase is that the file **reads as if it's trying very hard to prove the author is careful**. The content supports that claim, but the density of architecture.md references and cleanup-ticket citations can make a reviewer wonder whether the code can stand on its own.

**Verdict:** B+ — technically sound and clearly experienced, but overly verbose and with some release-silent invariants that the prose oversells.

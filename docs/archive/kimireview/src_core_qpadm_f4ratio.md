# Review: `src/core/qpadm/f4ratio.cpp`

I read this carefully. It is **not slop**, and it is clearly written by someone who understands both the qpAdm/f4-ratio math and the project's backend seam. That said, a senior C++ developer would flag several convention, API, and robustness issues that keep it from being pristine showcase code.

## What's genuinely good

- **The design reuses the right seams.** Building one interleaved 2N-quartet flat array and calling `assemble_f4_quartets` once, then delegating the actual ratio jackknife to the shared `RatioBlockJackknife` backend, is the correct architecture. It avoids duplicating AT2 math on the host and keeps the CUDA path device-resident.
- **Empty-batch handling is clean.** Returning `Status::Ok` with no rows for `N <= 0` (lines 90-95) is the right contract for a batch API.
- **Constants are named and localized.** `kPrimaryGpu` and `kSetmissThresh` are `inline constexpr` in an anonymous namespace, not scattered as magic numbers.
- **The two public overloads are thin, type-dispatched forwarders** (lines 148-160), mirroring the `run_f4`/`run_f3` pattern and keeping per-source logic in one templated implementation.
- **Vector pre-allocation is sensible.** `res.p1.reserve(...)` and `flat.reserve(N * 8)` (lines 101-107) avoid the obvious reallocation trap for large batches.

## What a senior developer would flag

**Ignoring `opts` with `(void)opts` is a smell.**

```cpp
// opts is accepted for API symmetry with run_qpadm/run_qpwave/run_f4/run_f3, but a bare
// f4-ratio SE uses fudge=0 ALWAYS (no Q is inverted), so opts.fudge is deliberately not
// consulted here. Acknowledge it so -Werror is satisfied.
(void)opts;
```

Lines 81-84. Accepting a parameter just to throw it away is a contract lie. If `f4-ratio` truly never uses `fudge`, the public API should not take `QpAdmOptions`. If the overload is required for generic dispatch, use `[[maybe_unused]]` or omit the name in C++17 rather than the C-style cast. The comment is honest, but honest about a design wart.

**No guard on `resources.gpus`.**

```cpp
[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}
```

Lines 61-63. `at(0)` will throw `std::out_of_range` if no GPUs are present, and dereferencing `backend` will crash if it is null. The file's own header comment says "Domain outcomes ... are a per-call status VALUE / per-row NaN sentinel, never an exception (architecture.md §10)," but this helper violates that policy. At minimum there should be a null/empty check returning `Status` or a contract assertion.

**Result fields may be uninitialized in the empty/error paths.**

```cpp
F4RatioResult res;
res.precision_tag = core::qpadm::honored_tag(prec, be);
```

Lines 87-88. `res.alpha`, `res.se`, and `res.z` are not set before the early `N <= 0` return or before the backend call. If `F4RatioResult` has no constructor that zeroes them, callers who check `status == Ok` and then read numeric fields can see garbage. The backend assignment at lines 138-140 covers the happy path, but the empty path is silent.

**The `N <= 0` check conflates zero and negative.**

```cpp
const int N = static_cast<int>(tuples.size());
if (N <= 0) {
```

Line 90. `std::span::size()` returns unsigned; it cannot be negative. Using `<= 0` instead of `== 0` is slightly defensive but also suggests the author is not fully comfortable with the unsigned-to-signed conversion they just performed. More importantly, `static_cast<int>` silently truncates if a caller passes more than 2³¹-1 tuples, which would then be passed as a negative-ish `N` to the backend. That is unlikely in genomics, but it is the kind of narrowing a senior dev would explicitly guard or use `std::size_t` for.

**Comment-to-code ratio is high, and some comments are fragile.**

The 33-line header block (lines 1-33) is informative but dense with project-internal acronyms (AT2, OQ-5, OQ-12, S3, S4, M1) and line-number references like `backend.hpp:113-126` that will drift. Lines 128-135 repeat much of that explanation inline. The intent is good, but a senior reviewer would worry about copy-paste drift between this file, `f4.cpp`, and `f3.cpp` — the comments claim they mirror each other, which is exactly the kind of claim that rots first.

**No input validation on population indices.**

The code trusts `tuples` to contain valid indices. That may be the project's convention, but a single out-of-range index in the flat array will propagate into `assemble_f4_quartets` and the backend. Given the file's emphasis on "never an exception," silent bad-index behavior is a mismatch.

## The "slop" test

**Not slop.**

There are no magic numbers without explanation, no obviously wrong algorithms, no copy-pasted host loops, and no missing error paths in the nominal flow. The comments are verbose but mostly explain *why*, and the code delegates the hard math to the right shared backend rather than reimplementing it.

## What it actually looks like

This looks like **solid, domain-knowledgeable C++ written by someone who is good at threading the project's existing architecture but is still learning to be paranoid about edge cases.** The math plumbing is correct and the backend integration is clean, but the surface API (`QpAdmOptions` ignored), the unchecked `resources.gpus.at(0)` dereference, and the uninitialized result fields would make a senior reviewer pause before calling it production-hardened.

A senior C++ reviewer would say: "Good structure, good reuse — now fix the option contract, harden the resource lookup, and zero your result struct." A CUDA/backend reviewer would say: "The host side is fine; the real work happens in the shared seam, which is the right place for it."

## Verdict

**B+.** Competent, well-integrated code that correctly implements the f4-ratio batch flow, marred by a few API/robustness warts that would be quick to fix and should be fixed before it is held up as showcase-quality code.

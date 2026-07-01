I read through this carefully. This is **not slop** — it's the work of someone who understands CUDA launch constraints and wants the math in exactly one place. A senior reviewer would find it competent and mostly correct, but would flag a few sharp edges and a lot of comment bulk.

## What's genuinely good

- **Single-home-of-truth for launch-grid math.** The file's stated purpose — ceiling division, hardware caps, and named block dimensions in one header — is exactly the right separation of concerns. Kernel TUs should not be re-deriving `dim3` math.
- **`STEPPE_HD constexpr` integer arithmetic.** Everything that can be compile-time foldable is. Keeping this header host/device-agnostic and CUDA-runtime-free (it only pulls in macros) is good layering; it can be consumed by both host-pure `core` and device code without dragging in `<cuda_runtime.h>`.
- **Named constants for hardware limits.** `kMaxGridX`, `kMaxGridY`, and `kMaxGridZ = kMaxGridY` single-source the shared y/z cap instead of duplicating the 65 535 literal. That is a real DRY win and makes future hardware changes edit one site.
- **Two `cdiv` overloads for two scales.** Providing `int` and `long` overloads is the right call: the SNP/`M` axis can exceed `INT_MAX`, and the overload set forces the caller to think about which one they need (lines 85–94).
- **Fail-fast debug assertions.** `grid_for` and `grid_z_extent` use `STEPPE_ASSERT` with explicit, actionable messages about re-orienting the large axis or tiling the batch (lines 125–128, 151–154). In debug builds this turns an opaque `cudaErrorInvalidConfiguration` into a clear invariant failure.

## What a senior developer would flag

**The comments are spec-fragile and extremely dense.**

This 178-line file contains more prose than code. Comments cite `architecture.md §4 line 466`, `§8 DRY-internals table line 525`, `ROADMAP §4`, `cleanup X-4`, `cleanup X-7/B6`, etc. (lines 3–39). That is useful context the first time you read it, but it is a maintenance hazard: line numbers and cleanup ticket IDs drift. A senior reviewer would ask whether the file is documenting itself or compensating for external docs that are hard to keep in sync.

**Signed-integer overflow is possible in `cdiv`.**

```cpp
[[nodiscard]] STEPPE_HD constexpr int cdiv(int n, int block) noexcept {
    return (n + block - 1) / block;
}
```

Line 86. The precondition says `n` is non-negative and `block > 0`, but there is no assert or `if` enforcing it. For a pathological caller passing `n` near `INT_MAX`, `n + block - 1` is signed overflow — undefined behavior. The `long` overload has the same issue. A defensive implementation would either assert the precondition or use unsigned arithmetic for the addition.

**`grid_for`'s `max_grid` bound is a debug-only check.**

```cpp
STEPPE_ASSERT(extent >= 0 && static_cast<unsigned>(extent) <= max_grid, ...);
```

Line 125. Because `STEPPE_ASSERT` compiles out under `NDEBUG`, a release build can still launch with an over-limit grid and get the same opaque `cudaErrorInvalidConfiguration` this helper is meant to prevent. A senior reviewer would want at least a comment admitting that, or a `static_assert`/compile-time guard where possible. The current comment says "fail-fast in debug," which is honest, but it also claims the wrapper "fails-fast" in a way that is not true in release.

**The `[[maybe_unused]]` on `max_grid` is noisy.**

```cpp
[[nodiscard]] STEPPE_HD constexpr int grid_for(int n, int block = kCdivBlock,
                                               [[maybe_unused]] unsigned max_grid = kMaxGridY) noexcept
```

Line 117–118. The three-line justification (lines 119–123) is correct: `max_grid` is only read by the assert. But a cleaner API would make the bound unavoidable — for example, by returning `unsigned` and always validating, or by providing two named functions (`grid_for_yz`, `grid_for_x`) so the parameter is not conditionally used at all.

**Forward-looking hardware claims are unverifiable here.**

```cpp
// ... including Blackwell sm_120 (CUDA C++ Programming Guide, Compute Capabilities); verified for CUDA 13.x).
```

Line 54. If this code is being read today, "CUDA 13.x" and "sm_120" may or may not match the toolchain in the room. It is fine to cite a spec, but "verified for CUDA 13.x" is the kind of claim that ages badly and can look like hallucination if the reader cannot reproduce it. A senior reviewer would prefer "as of CUDA X.Y" with a date or build note, or simply cite the Programming Guide without claiming verification.

**`grid_z_extent` is host-only and not `constexpr`.**

```cpp
[[nodiscard]] inline unsigned grid_z_extent(int n_in_group) noexcept
```

Line 146. This is fine if the call sites are always host-side launch wrappers, but it is a visible asymmetry next to the `STEPPE_HD constexpr` helpers. The reason is presumably `STEPPE_ASSERT`, but the file does not explain why this one helper is different. A reader may wonder whether it is intentional or an oversight.

## The "slop" test

**Not slop.** Slop would be magic numbers, copy-pasted `(n + block - 1) / block` everywhere, missing bounds checks, and comments that only restate the code. This file has none of that. The constants are named, the overloads are scale-aware, and the assertions are explicit. The code is small, focused, and mostly correct.

## What it actually looks like

This looks like **careful, spec-driven systems code written by someone who has been bitten by silent launch failures before.** The author clearly understands why launch-grid math should live in one place, why y/z caps differ from x, and why the SNP axis needs a wider integer type. It is the kind of file that would pass a code review on correctness grounds.

A senior reviewer would probably say: "Ship it, but trim the comments by half and harden the signed overflow cases." A junior maintainer would find it educational; a rushed one might skip the wall of text and miss the important invariants buried in it.

## Verdict

**B+.** Solid, correct, well-intentioned header. The architecture and DRY choices are good, but the comment density, debug-only bounds checks, and missing overflow guards keep it from being pristine.

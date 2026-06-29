I read through `f2_block_kernel.cu` carefully. This is **not slop** — it's clearly written by someone who understands both the genomics numerics and CUDA well enough to avoid the big traps. But a senior reviewer would have mixed reactions: some parts are excellent, others are over-defended or carry subtle gotchas.

## What's genuinely good

- **The coalescing fix in `f2_feeder_kernel` is well-explained and correct.** Lines 103–115 and 129–130 show the author actually understood the problem: the original mapping had consecutive lanes striding by `P`, and the fix maps `threadIdx.x` to the population (unit-stride) axis while keeping `M` on `gridDim.x`. The comment explains *why* the block tile is transposed without changing the grid orientation.
- **Shared primitives with the CPU oracle.** Using `het_correction`, `assemble_f2_numerator`, and `finalize_f2` from `core/internal/f2_estimator.hpp` (lines 84–86, 150, 203–205) is the right way to guarantee the GPU and CPU cannot diverge on the formula.
- **The `emulation_honorable` predicate as a single source of truth.** Lines 228–235 and the surrounding design close the C-2 split: both `f2_compute_type` and `engage_f2_precision` consult the same predicate, so the math mode and compute type cannot disagree. That's good defensive API design.
- **The cuBLAS stream/workspace note at lines 366–371.** Avoiding a per-call `cublasSetStream` because it resets the workspace pool is exactly the kind of CUDA footgun a senior dev wants to see handled deliberately.
- **The `INT_MAX` narrowing guard at lines 377–387.** `M` is `long` but `cublasGemmEx` takes `int`; the code documents that the caller guards it and pins the invariant with `STEPPE_ASSERT`.

## What a senior developer would flag

**The comment density and tone.** This file has ~100 lines of code and ~300 lines of comments, and some comments read like manifestos:

```cpp
// PRECISION POLICY (MEASURED on real AADR — architecture.md §12, ROADMAP §0; this
// is the law):
```

(lines 17–50). A senior dev would say: "If you need fifty lines to explain why you're not doing dynamic mantissa control, something is wrong — either the design isn't self-explanatory or you're writing apology comments."

**`ATOMIC_FLAG_INIT` is deprecated in C++20.** Line 253:

```cpp
static std::atomic_flag emitted = ATOMIC_FLAG_INIT;
```

This is a small but real convention mismatch. In modern C++ you default-construct the flag.

**The downgrade warning is routed through a debug-only sink.** Lines 245–260 say `STEPPE_LOG_WARN` is "NDEBUG-silent." For a capability downgrade that the comments insist must be "observable," silently dropping the warning in release builds is odd. The rationale — "future verbose sink still fires at most once" — is reasonable but worth flagging.

**`STEPPE_ASSERT` for the `INT_MAX` check.** Line 384:

```cpp
STEPPE_ASSERT(M <= static_cast<long>(std::numeric_limits<int>::max()), ...);
```

If this is a real `assert` (debug-only), then in release builds the narrowing guard disappears. The caller is supposed to guard it, but relying on a distant caller plus a debug-only assert for a silent-wrong-result bug is a gotcha.

**Tf32 silently runs native FP64.** `f2_compute_type` at lines 283–287 returns `CUBLAS_COMPUTE_64F` for `Precision::Kind::Tf32`, with a comment that the TF32 path is "screening-only" and "not wired into this FP64-storage GEMM path." A senior dev would want either an explicit error or a clearer contract: if the caller passes Tf32 here, they're getting FP64 math whether they realize it or not.

**`STEPPE_CUDA_CHECK_KERNEL()` after async launches.** Lines 359 and 420:

```cpp
f2_feeder_kernel<<<grid, block, 0, stream>>>(...);
STEPPE_CUDA_CHECK_KERNEL();
```

This checks the synchronous launch error, but actual kernel runtime errors are asynchronous. If the kernel segfaults or hits a math error, you won't catch it here unless `STEPPE_CUDA_CHECK_KERNEL` synchronizes (which would stall the pipeline). Most projects accept this pattern, but a senior CUDA reviewer would verify the macro.

**The exact-zero validity check.** Line 137:

```cpp
const bool valid = (V_raw[idx] != 0.0);
```

If `V` is ever a very small non-zero value rather than a true zero, this misclassifies it as valid. This may be fine per the Q/V/N contract, but it's the kind of assumption a senior dev circles.

## The "slop" test

**Not slop.** There's no copy-paste drift, no unexplained magic numbers (16 is behind `kCdivBlock`, 2P is behind `kF2StackedBlocks`), no missing error checking, and no obviously wrong algorithm. The comments are verbose but accurate and explain *why*, not just *what*.

## What it actually looks like

This looks like **solid production engineering code written by a domain expert who is competent at CUDA and extremely careful about numerical correctness.** The author has clearly been burned by precision regressions and cuBLAS workspace bugs, and the file is structured to prevent those specific failures. The style is defensive and heavily annotated — not messy, but not minimalist either. A senior CUDA specialist would probably say: "Correct and safe — ship it, but I'd trim the commentary and tighten the warning/assert semantics." A senior C++ person would say: "A bit C-ish in places, and the comments are doing too much work, but the logic is sound."

**Verdict:** B+. Ship after tightening the deprecated `ATOMIC_FLAG_INIT`, clarifying whether the downgrade warning should really be NDEBUG-silent, and auditing `STEPPE_CUDA_CHECK_KERNEL` for async error handling.
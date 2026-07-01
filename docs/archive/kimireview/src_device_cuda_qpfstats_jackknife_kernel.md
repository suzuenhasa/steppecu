I read through this carefully. This is **not slop** — it's clearly written by someone who understands both the genomics math and the CUDA porting task, but a senior developer would flag a handful of polish, consistency, and performance-paper-cut issues.

## What's genuinely good

- **The top-level design rationale is excellent.** The header comment (lines 1–25) explains *why* the work moved to the GPU, the precision rationale (native FP64 vs. emulated), the one-thread-per-comb/pair shape, and why `cub::BlockReduce` was deliberately avoided. This is the kind of context that saves future maintainers hours.
- **Faithful host-to-device port with explicit contracts.** The comments explicitly tie the kernels to `core::matrix_jackknife_est_col` and `core::f2blocks_pair_est` (and the qpAdm golden). That kind of traceability matters for a correctness-sensitive stats backend.
- **Appropriate use of `__restrict__`** on all pointer arguments and explicit `cudaStream_t` threading.
- **Grid-stride loops** are used correctly (lines 47–49, 113–115), so the kernels tolerate any grid size and the launch wrapper can cap at `kMaxGridX`.
- **Launch wrapper isolation is clean.** The kernels live in an anonymous namespace; only the two `launch_*` functions are exposed, matching the stated architecture contract.
- **Error checking is present** via `STEPPE_CUDA_CHECK_KERNEL()` after each launch (lines 165, 178).

## What a senior developer would flag

**Inconsistent launch-config guarding between the two wrappers:**

```cuda
void launch_qpfstats_numer_jackknife(...) {
    ...
    long blocks = (static_cast<long>(npopcomb) + kThreads - 1) / kThreads;
    if (blocks > static_cast<long>(core::kMaxGridX)) blocks = core::kMaxGridX;
    ...
}

void launch_qpfstats_recenter_shift(...) {
    ...
    long blocks = (static_cast<long>(npairs) + kThreads - 1) / kThreads;
    if (blocks < 1) blocks = 1;
    if (blocks > static_cast<long>(core::kMaxGridX)) blocks = core::kMaxGridX;
    ...
}
```

Both are protected by `npopcomb/npairs > 0` upfront, so `blocks` is already ≥1. The extra `if (blocks < 1)` in the second wrapper is harmless but the mismatch reads like copy-paste drift. A senior reviewer would want both wrappers to look identical in policy.

**`nan("")` instead of a typed quiet-NaN:**

```cuda
const double knan = nan("");   // line 46
```

It works in CUDA, but `std::numeric_limits<double>::quiet_NaN()` (or at least `NAN`) is more idiomatic modern C++. `nan("")` looks like C-style habit.

**The `shift[p] = bglob[p] - 0.0;` line:**

```cuda
if (!(sum_bl > 0.0)) { shift[p] = bglob[p] - 0.0; continue; }   // line 124
```

The `- 0.0` serves no purpose. Just write `shift[p] = bglob[p];`. This is the kind of oddity that makes a reader pause and wonder if there used to be a reason.

**Device lambda with implicit by-reference capture:**

```cuda
const auto arr = [&](int blk) -> double {
    return b[p + static_cast<long>(npairs) * blk];
};   // lines 116–118
```

CUDA device lambdas are allowed, but implicit `[&]` capture in device code can produce surprising lifetime/duplication behavior and tends to trigger warnings with stricter compiler flags. A plain helper function or explicit `[=]` capture would be cleaner. It also adds a small abstraction tax for a single multiply-add.

**Undocumented `kThreads = 128`:**

```cuda
constexpr int kThreads = 128;   // lines 160, 172
```

128 is a defensible default, but there's no comment on why 128 was chosen versus 64, 256, or occupancy-tuned values. Given that each thread does a short ~711-element reduction in registers, block size matters for occupancy.

**Multiple passes over the same small row:**

The numer kernel loops over `n_block` several times (lines 53, 62, 67–73, 80–95). With `n_block` ≈ 711 this is small enough to keep in registers or L1, but the kernel never explicitly tiles or reuses the loaded `cnt`/`numer` values across loops. It's correct and simple, but a performance reviewer would at least ask whether a single fused pass or shared-memory cache was considered.

**The "ascending-b" claim vs. the actual loop order:**

The header comment says the kernel reproduces "the EXACT ascending-b operand order" (line 9) and "same ascending-b accumulation" (line 38). But the code simply loops `for (int b = 0; b < n_block; ++b)` without any ascending sort of block sizes or operands. If the host reference actually sorts, this is a real discrepancy; if it doesn't, the comment is misleading. Either way, a senior reviewer would ask for clarification.

**Strided row-major reads across a warp:**

Each thread owns one comb/pair and reads a contiguous row (`base + b`), but consecutive threads in a warp read rows separated by `n_block` doubles. With `n_block` ~711 that is not coalesced. It's a common tradeoff for row-owner parallelism, but for a "PERF path" kernel it deserves either a comment or a tiling strategy.

## The "slop" test

**Not slop.** Slop is unexplained magic numbers, stale copy-pasted comments, missing error checks, or algorithms that only happen to pass. This file has none of those. The comments are dense but they explain *why* decisions were made and link to host-side contracts. The math is carefully transliterated, and the launch/error-handling plumbing is present.

## What it actually looks like

This looks like **solid research-engineering CUDA written by a domain expert who is competent on the GPU but not obsessing over every last warp.** The author clearly knows the jackknife statistics cold, made deliberate choices about precision and parallelism, and avoided the common disasters (atomic storms, unnecessary `cub::BlockReduce`, emulated FP64 in a cancellation-sensitive pass).

A senior CUDA specialist would say: "Correct and well-explained — now let's profile occupancy and memory coalescing before we call it done." A senior C++ person would say: "A little C-ish in places, and the launch wrappers need to be consistent, but I'd rather read this than most hand-rolled CUDA I see."

**Verdict:** B+ to A-. Production-ready with minor polish issues; the "ascending-b" comment and the launch-wrapper inconsistency are the only things that would make a reviewer stop and ask questions in a showcase setting.

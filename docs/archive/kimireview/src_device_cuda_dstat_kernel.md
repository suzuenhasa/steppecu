I read through this carefully. This is **not slop** — it's clearly written by someone who understands both the genomics problem and CUDA performance, but a senior developer would still have specific nits, mostly around the hot-path micro-optimization and hardcoded architecture assumptions.

## What's genuinely good

- **The shared-memory pairwise reuse design is real and well-quantified.** The comment at lines 14–26 explains the ~11.6 TB global-load problem and how the tiled kernel fixes it with one coalesced Q+V read per SNP. This is a meaningful algorithmic improvement, not a cosmetic refactor.
- **The bit-exact/golden reasoning is careful and specific.** Lines 28–36 document IEEE-754 negation exactness, symmetric `het[ij]`, and ascending-`s` ownership so the tiled path matches the legacy kernel bit-for-bit. That's exactly the kind of analysis you want in numerical CUDA code.
- **The fallback path is clean.** Lines 38–41 and 279–286 retain the legacy per-(combo,block) kernel inside the same TU and branch at runtime. No forked codebase, no duplicated wrappers, and the fallback is automatically golden-safe because it's byte-identical to the historical kernel.
- **CUDA fundamentals are handled correctly:** `__restrict__` pointers, cooperative coalesced column loads (`Qsh[i] = Q[col + i]` at line 179), `__syncthreads()` barriers around shared-memory phases, and proper `extern __shared__ double s_mem[]` usage.
- **Good encapsulation.** The kernel bodies live only in this TU; the backend reaches them through the narrow `launch_dstat_block_reduce` wrapper in `dstat_kernel.cuh`, consistent with `architecture.md §7`.
- **`(void)M;` at lines 78 and 139** is a small but competent signal: the documented dimension is preserved in the signature even though the block layout bounds the walk.

## What a senior developer would flag

**The triangular index decode in the hot path (lines 189–193):**

```cuda
int i = 0;
int rem = idx;
int row = P - 1;  // entries in row 0
while (rem >= row) { rem -= row; ++i; --row; }
const int j = i + 1 + rem;
```

This is an O(P) integer loop executed for every pair index, for every SNP, on the cooperative fill path. For P near 112, each thread may do ~50–100 iterations per pair. A senior CUDA reviewer would immediately ask whether this has been profiled against a precomputed `(i,j)` lookup table in constant memory or a small device-side buffer. It may be memory-latency-hidden, but it's the obvious next optimization target.

**Hardcoded 99 KB opt-in shared-memory budget tied to a specific architecture:**

```cuda
constexpr std::size_t kOptinSmem = 99u * 1024u;     // 101376 (sharedMemPerBlockOptin)
```

The comment at lines 254–255 explicitly pins this to "sm_120 / CUDA 13 (RTX 5090)." If this code runs on an older GPU, `cudaFuncSetAttribute` will reject the opt-in and `STEPPE_CUDA_CHECK` will throw. A senior dev would prefer a runtime query of `cudaDevAttrMaxSharedMemoryPerBlockOptin` with a graceful fallback, rather than a constant that encodes one SKU.

**Magic constants without motivation:**

```cuda
constexpr int kThreads = 256;
```

256 is a reasonable default, but there's no comment explaining why 256 was chosen over 128 or 512 for occupancy on the target hardware. Given how much this file documents *why* for the math, the launch config is under-explained.

**Implicit `min`/`max` overloads in device code (lines 218, 221):**

```cuda
if (p1 != p2) h1 = het[pair_index_lo_hi(min(p1, p2), max(p1, p2), P)];
```

This works because `cuda_runtime.h` provides `min`/`max` for `int`, but it's implicit and easy to misread. A senior reviewer would prefer an explicit ternary (`p1 < p2 ? p1 : p2`) or a small local helper, especially in a file that otherwise spells out bit-exact semantics.

**Possible `unsigned` overflow in the legacy launch (line 283):**

```cuda
const long blocks = (total + kThreads - 1) / kThreads;
dstat_block_reduce_legacy_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(...)
```

For realistic genomics sizes this is fine, but `static_cast<unsigned>` of a `long` is narrow and undefined if `blocks` ever exceeds `UINT_MAX`. On a 64-bit host `long` can hold much larger values. Casting down silently is a footgun if the input scales unexpectedly.

**Use of `long` for 64-bit indices (lines 74, 75, 135, etc.):**

```cuda
long M;
```

On Linux x86_64 this is 64-bit, but `long` is 32-bit on Windows. If the project is strictly Linux/CUDA this is acceptable, but a senior C++ reviewer would prefer `std::int64_t` for portable intent.

**`__syncthreads()` after divergent work (line 227):**

```cuda
if (active) { ... }
__syncthreads();
```

In this specific kernel it is correct because *all* threads in the block reach the barrier regardless of `active`. But the pattern is a classic CUDA footgun; a reviewer would double-check it, and a defensive author might add a comment reaffirming that inactive threads still participate.

**No precondition enforcement on `pair_index_lo_hi` (lines 63–65):**

The function documents `lo < hi` and assumes the caller passes the smaller index first, but there is no `assert` or runtime check. In a debug build, a check would catch misuse early.

## The "slop" test

**Not slop.** Slop is unexplained magic numbers, copy-pasted stale comments, missing error handling, and algorithms that only happen to pass. This file has none of that. The comments are dense but they explain mathematical invariants, host-side contracts, and bit-exactness arguments that aren't visible in the CUDA code alone.

## What it actually looks like

This looks like **solid production CUDA written by a domain expert who cares about correctness and performance.** The author clearly understands the admixture-statistic math, the golden/bit-exact requirements, and the memory-bandwidth bottleneck they are optimizing. They also know enough CUDA to avoid the big mistakes: coalesced access, shared-memory reuse, barriers, and async error checking.

A senior CUDA specialist would say: "Good architecture — now let me profile that pair-index decode and replace the hardcoded 99 KB with a runtime query." A senior C++ person would say: "A bit verbose, slightly C-ish in its use of `long` and `min`/`max`, but clearly reasoned and well-structured."

The comment density is high, but here it is mostly earned: the comments justify a non-obvious tiling scheme, the bit-exactness argument, and the fallback boundary. It does not read like someone compensating for unclear code.

## Verdict

**B+ to A-**. Correct, performant, and well-explained. The main demerits are the O(P) pair decode in the hot loop and the architecture-specific hardcoded shared-memory cap. Fix those and it is showcase-quality CUDA.

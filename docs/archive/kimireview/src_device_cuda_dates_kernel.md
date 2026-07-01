I read through this carefully. This is **not slop** — it's clearly written by someone who understands both the admixture genetics and enough CUDA to avoid the obvious pitfalls. A senior reviewer would find a lot to respect, but also a few places where the implementation could be tightened.

## What's genuinely good

- **The architectural header comment is excellent.** It explains exactly what this TU owns (elementwise / scatter / reduce glue around cuFFT), what it does *not* own (the f2 GEMM, host O(M²) loops), and the reference C source it is pinned to. That kind of boundary-setting prevents a lot of maintenance grief.
- **Bit-exact reproduction of host logic in `repack_target_kernel`.** The race-condition analysis (one thread per dest byte because 4 codes/byte would race) and the explicit mirroring of `dates.cpp:304-311` show careful, paranoid engineering.
- **`regress_dots_kernel` uses a proper shared-memory tree reduction** with one `atomicAdd` per block, not a thread-per-element atomic storm.
- **The elementwise kernels are clean and obvious.** `power_spectrum_kernel` and `cross_power_kernel` do exactly one thing, with no hidden state or surprising side effects.
- **Launch wrappers are uniform and stream-aware**, and they all call `STEPPE_CUDA_CHECK(cudaGetLastError())` immediately after the kernel launch.
- **`__restrict__` and `const` are applied consistently** on device pointers, which matters for the compiler's load/store analysis.

## What a senior developer would flag

**Magic numbers in the exp-fit kernel with no named constants:**

```cuda
const int coarse = 4000;
// ...
const int ternary_iters = 200;
```

Both values are explained in the surrounding prose, but they are not `constexpr` names. A senior reviewer would expect `kExpFitCoarseGrid = 4000` and `kExpFitTernaryIters = 200` so that tuning does not require hunting through loop bodies.

**The `__syncthreads()` after a conditional block in `regress_dots_kernel`:**

```cuda
for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off) {
        sh12[threadIdx.x] += sh12[threadIdx.x + off];
        sh22[threadIdx.x] += sh22[threadIdx.x + off];
    }
    __syncthreads();
}
```

In this specific loop it is safe because every thread hits the barrier. But the *pattern* of placing `__syncthreads()` immediately after a divergent `if` is a classic CUDA footgun; a senior reviewer will pause and verify it every time.

**The `grid_for` helper is cavalier about overflow:**

```cuda
inline int grid_for(long n) { return static_cast<int>((n + kBlock - 1) / kBlock); }
```

`int` grids are fine for genomics sizes, but a silent narrowing cast from `long` to `int` without an assert or `if (n > ...)` is the kind of thing that becomes a bug when someone reuses this helper for a much larger problem. More importantly, for `launch_dates_fit_curves` where `n_curves ~ 23`, this launches one block of 256 threads, leaving ~233 idle threads. The comment admits this, but a senior CUDA dev would probably use a smaller fixed block size for that launch.

**The `+=` accumulation contract in `extract_lags_kernel` is invisible here:**

```cuda
dd[cell] += v / static_cast<double>(n_fft);
```

The comment says "summed (+=)", which is intentional, but the reader has to trust that the caller zeroed `dd` before launching this kernel. That contract should either be in the wrapper comment or the kernel should use `=` if this is the only writer.

**The `dev_linfit_2x2` NaN signaling is C-ish:**

```cuda
*co0 = nan(""); *c = nan(""); return nan("");
```

It works, but `std::numeric_limits<double>::quiet_NaN()` or at least `NAN` would be more idiomatic in modern C++.

**FP64 `atomicAdd` assumptions are not guarded.** `scatter_kernel`, `accumulate_bins_kernel`, and the regress reduction all call `atomicAdd` on `double`. That requires sm_60+. The project may guarantee that architecture elsewhere, but this file does not mention it, and a portable CUDA TU would at least note the assumption.

**`extract_lags_kernel` does redundant work for `lag >= n_fft`:**

```cuda
const double v = (lag < n_fft) ? inv[static_cast<long>(kc) * n_fft + lag] : 0.0;
```

Since the launch grid is sized as `n_chrom * (diffmax + 1)`, launching threads for `lag >= n_fft` only to write zero is a small waste. It is benign, but a senior reviewer would ask why the grid is not sized to `min(diffmax + 1, n_fft)`.

**The `accumulate_bins_kernel` threshold `0.5` is a magic number:**

```cuda
if (c00 < 0.5) return;
```

This is presumably a minimum count cutoff, but there is no named constant or comment explaining why 0.5 and not 0.0 or 1.0.

**Input validation in launch wrappers is minimal.** None of the wrappers check for null pointers or nonsensical sizes (e.g., `n_chrom <= 0`, `M <= 0`, `n_fft <= 0`). In a research codebase that is often acceptable, but a senior production reviewer would want at least `assert`s or early returns with clear contracts.

## The "slop" test

**Not slop.** Slop is unexplained magic numbers, copy-pasted drift, missing error checking, or obviously wrong algorithms. This file has a few unexplained constants and some C-ish habits, but the comments are accurate, the math is documented, the CUDA patterns are correct, and the host-reference mirroring is deliberate. There is no stale copy-paste or hand-waving.

## What it actually looks like

This looks like **solid research/engineering code written by a domain expert who knows the DATES algorithm well and knows enough CUDA to implement it correctly.** The author understands coalesced access, shared-memory reductions, grid-stride loops, and FP64 cuFFT autocorrelation, but is not necessarily optimizing for every last SM cycle.

A senior CUDA specialist would likely say: "Correct and well-reasoned — ship it, but let me spend a day on occupancy, register pressure, and the magic-number cleanup if this is on the hot path." A senior C++ reviewer would say: "Competent, a bit C-flavored in places, and slightly over-commented, but I would rather read this than most CUDA TUs that cross my desk."

The biggest risk is not correctness but **future drift**: the host-reference comments (`dates.cpp:304-311`, `dates.c:585-665`) are valuable now but will become stale if the host code moves.

**Verdict:** B+ to A- — respectable production CUDA, clearly written, with only minor tuning and style issues to address.

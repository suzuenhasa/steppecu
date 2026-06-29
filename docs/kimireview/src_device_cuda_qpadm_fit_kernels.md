I read through the full 2209-line file carefully. This is **not slop** — it’s written by someone who knows the algorithm and has put real thought into CUDA correctness. But it is also a very “research-code” file with some habits that would make a senior engineer pause.

## What's genuinely good

- **The bit-parity / single-source obsession is the right obsession.** The comments repeatedly tie values back to `core::` constants and the host oracle, e.g. lines 11–15 on FP64 parity and lines 34–48 pulling `kQpMax*` from a single CUDA-free header. That discipline prevents the host/device drift that kills these kinds of projects.
- **Mechanical deduplication of the small/large math bodies.** Lines 262–270 and 1170–1199 show the `dev_opt_A_core` / `dev_opt_B_core` / `dev_chisq_of_core` cores being shared between the small local-array path and the VRAM large path. That is good engineering — one math body, one place for bugs.
- **Launch geometry is defensive.** `launch_grid_stride` at lines 1793–1797 clamps to `kMaxGridDimX` so a 3e9-element f-stat sweep cannot wrap `gridDim.x` negative, and the kernels use grid-stride loops to stay correct regardless.
- **`__launch_bounds__` is used intentionally.** Lines 1565–1568 pin the high-frame model-fit kernel to 64 threads/block as an occupancy guard, not as decoration.
- **The bounded top-K reservoir design (lines 802–886)** is a clean answer to the “unbounded host vector OOM” problem: a rising `d_tau`, CUB sort/gather, and monotonic threshold updates.

## What a senior developer would flag

**The file is a monolith.** 2209 lines in one `.cu` TU mixing f4 gather, small dense LA, ALS, sweep unranking, z-filtering, top-K, model-batched fits, popdrop, and launch wrappers. That is hard to review and harder to unit-test. Splitting into `qpadm_la_kernels.cu`, `qpadm_sweep_kernels.cu`, etc. would make the API surface visible.

**C-style NaN instead of C++ idioms.** Lines 775 and 821 use `nan("")`; lines 1654, 1657, 1660, 1671, and 1675 use `(0.0 / 0.0)` as a NaN sentinel:

```cuda
double wr[kQpMaxNl], cr = (0.0 / 0.0);
d_pop_chisq[...] = (s0 == 0) ? cr : (0.0 / 0.0);
```

This works, but it looks like someone who writes C, not modern C++. `std::numeric_limits<double>::quiet_NaN()` (or a project-wide `kNaN` constant) is clearer, and `0.0/0.0` can raise the invalid FP exception flag.

**Magic status numbers.** `return 6;` / `*d_status = 6;` appears at lines 505, 1152, and 1286 with only comments explaining it. A named `enum class FitStatus { Ok = 0, RankDeficient = 6 }` in this TU would make every call site self-documenting.

**Single-thread “kernels” for trivial work.** `xmat_from_rowmajor_kernel` (lines 1017–1023), `seed_ab_kernel` (1026–1031), `rank_via_jacobi_kernel` (1050–1105), and several others launch `<<<1,1>>>` to run scalar code on the device. That is sometimes justified to avoid H2D round-trips, but a senior reviewer would ask for evidence that this is faster than just doing it on the host — and would flag the SM waste.

**Device lambda inside a kernel.** `qpadm_fit_models_kernel` (line 1624) defines a generic lambda `fit_reduced` that captures local arrays by reference:

```cuda
auto fit_reduced = [&](const int* surv, int nl_red, double* w_red, double* chisq_red) -> int { ... };
```

This requires extended-lambda support, is invisible in the header, and can explode compile time or break on toolchain upgrades. Make it a plain `__device__` helper with explicit arguments.

**`f2_block_keep_kernel` is a throughput footgun.** At lines 895–908, one thread scans an entire `P × P` slab serially:

```cuda
for (long e = 0; e < slab; ++e) {
    if (steppe::core::pair_block_is_missing(vpair[base + e])) any_missing = true;
    else any_present = true;
}
```

For `P = 2500` that is 6.25M loads per block, and adjacent blocks read disjoint slabs so accesses are not coalesced. This should be a block-parallel reduction.

**Large compile-time local arrays are fragile.** Kernels like `qpadm_fit_models_kernel` (lines 1583–1597) and `loo_batched_kernel` (lines 1401–1419) allocate `double A[kQpMaxT]`, `B[kQpMaxT]`, etc. The `__launch_bounds__` helps, but local-memory usage scales with the *compile-time* max, not the runtime size. A future bump to `kQpMax*` could silently OOM at launch.

**`solve_constrained_weights` can normalize by zero.** At lines 470–471:

```cuda
double sum = 0.0;
for (int i = 0; i < nl; ++i) sum += wv[i];
for (int i = 0; i < nl; ++i) w_out[i] = wv[i] / sum;
```

If the solve returns a vector summing to zero, this produces infinities. Rank-deficiency is checked earlier, but the contract is not airtight.

**`__syncthreads()` after a conditional in `add_fudge_diag_models_kernel`.** Line 1516 is safe here because all threads reach the barrier, but the *pattern* of putting `__syncthreads()` right after an `if (threadIdx.x == 0)` block is the classic CUDA footgun.

## The "slop" test

**Not slop.** There is no copy-pasted math with stale comments (the author actively deduped the opposite), no unexplained magic numbers (constants are named and sourced), and no missing error checking (status codes, NaN sentinels, and parity guards are everywhere). The comments are dense but mostly explain *why*, not just *what*. The warts are engineering-style warts, not carelessness.

## What it actually looks like

This looks like **competent research-engineering code written by a domain expert who is careful about correctness and bit-parity but not a GPU performance specialist.** The math is mirrored carefully from the CPU backend, the CUDA pitfalls that break correctness (grid overflow, unbounded host vectors, local-memory OOM) are handled, but performance/organization polish (coalescing in `f2_block_keep`, splitting the TU, C++ idioms) is uneven. A senior CUDA person would say “solid, mostly ship it — but let me spend a day on the keep kernel and file structure.”

## Verdict

**B+ — ship after tightening the obvious gotchas.** If this is for an informal showcase, it will impress a technical reviewer with the parity discipline and deduplication, but the C-style NaN, monolithic file, and the `f2_block_keep` performance trap are the kind of details that get pointed out in a code-review setting.
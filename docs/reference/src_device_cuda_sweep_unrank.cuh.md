# `sweep_unrank.cuh` reference

## 1. Purpose

`src/device/cuda/sweep_unrank.cuh` is a tiny, header-only piece of combinatorics
shared by two very different GPU sweeps: the f-statistic all-combinations sweep
(`qpadm_fit_kernels.cu`) and the READv2 all-pairs relatedness kernel
(`readv2_mismatch_kernel.cu`). It answers one question, on the device, for a lot of
threads at once: **given a flat rank `r`, which combination is that?**

Both sweeps launch one thread per combination. Thread `t` needs to know *which*
populations (or which pair of samples) it is responsible for, and it needs to work
that out from nothing but its own index — there is no table of combinations in
memory to look up, because materializing one for a large sweep would cost more than
the compute. So each thread runs a small "unrank" routine that turns its flat rank
into the actual tuple. This header holds those routines.

It defines three functions, all `static __device__ __forceinline__`:

| Function | What it does |
|---|---|
| `sweep_choose(n, kk)` | The binomial coefficient `C(n, kk)`, in `double`. |
| `sweep_unrank(r, P, k, c)` | The general combinatorial-number-system unrank: rank `r` → the `k`-subset `c[0] < c[1] < … < c[k-1]` of `{0..P-1}`. |
| `readv2_unrank_pair(r, N, c)` | The `k = 2` special case, in closed form: rank `r` → the pair `(i, j)` with `i < j`. |

There is no host code here and nothing allocates. This is arithmetic that runs
inside a kernel, on a per-thread rank, and returns a handful of indices in a small
stack array the caller passes in.

---

## 2. Why this is a shared header (and why `static __forceinline__`)

`sweep_choose` and `sweep_unrank` were originally born inside
`qpadm_fit_kernels.cu`, in that file's anonymous namespace, serving the f-stat
sweep. When the READv2 all-pairs kernel needed the exact same rank arithmetic, the
choice was to copy it or to share it. Copying would leave two definitions free to
drift apart — a subtle divergence in the rank convention between two sweeps is the
kind of bug that produces plausible-but-wrong results. So the definitions were
lifted **verbatim** into this header and both translation units now include it.
There is one definition of the rank convention, not two.

The `static __device__ __forceinline__` on each function is deliberate and
load-bearing. `static` gives every CUDA translation unit that includes the header
its own internal-linkage copy — which is exactly the linkage the functions had back
when they lived in an anonymous namespace. That matters because the project builds
with `CUDA_SEPARABLE_COMPILATION ON`: without internal linkage, two TUs each
emitting an external-linkage `sweep_unrank` would be an ODR violation at device
link. Internal linkage sidesteps that entirely — each TU is self-contained.
`__forceinline__` keeps the little routine from becoming a real device call in the
inner loop; it folds straight into the thread's code.

---

## 3. `sweep_choose` — the binomial coefficient in `double`

```
C(n, kk) = (n)(n-1)…(n-kk+1) / kk!
```

`sweep_choose` computes exactly that: it multiplies the `kk` descending terms of the
numerator, multiplies the `kk` ascending terms of the factorial denominator, and
divides. It guards the degenerate inputs first — `kk < 0` or `n < kk` returns `0.0`,
which is the mathematically correct "no such subsets" answer and, more importantly,
is the terminating value the unrank scan in section 4 relies on.

The choice of `double` rather than integer arithmetic is inherited from the qpAdm
original and is intentional: the ranks these sweeps address can be very large (a
C(P,4) quartet space over hundreds of populations runs to billions), and doing the
running product in `double` keeps the intermediate `C(v, kk)` values representable
across the whole scan without overflowing a 64-bit integer mid-computation. The rank
`r` itself is compared against these `double` values, so the whole comparison ladder
lives in the same numeric world.

---

## 4. `sweep_unrank` — the general combinatorial-number-system unrank

`sweep_unrank(r, P, k, c)` maps a flat rank `r` in `[0, C(P, k))` to the specific
`k`-element subset `c[0] < c[1] < … < c[k-1]` drawn from `{0..P-1}`. It writes the
`k` chosen indices into the caller's `c` array, ascending.

The method is the classic combinatorial number system. Every `k`-subset has a unique
rank, and you can recover the subset digit by digit, from the **most significant
position down**:

- For the top position (`pos = k-1`, so `kk = k`), find the largest `v` such that
  `C(v, kk) <= r`. That `v` is the value at this position: `c[pos] = v`. There are
  `C(v, kk)` subsets whose top element is below `v`, so those ranks come first;
  subtract them off (`r -= C(v, kk)`) and the residual is the rank *within* the block
  that fixes this position.
- Drop to the next position (`kk` decreases by one) and repeat with the residual
  rank, now choosing a value below the one just fixed.

The inner `while (sweep_choose(v + 1, kk) <= r) ++v;` is the "find the largest `v`"
search — it walks `v` upward from its minimum (`kk - 1`) until pushing one higher
would overshoot `r`. This is a **linear scan**, and its cost is the reason the `k=2`
special case in section 5 exists. `P` is accepted for signature symmetry with
`readv2_unrank_pair` and to document the ambient set size, but the routine does not
actually read it — the scan is bounded by `r` and `kk`, not by `P` — hence the
`(void)P;`.

**Callers.** `sweep_unrank_quartets_kernel` calls it with `k = 4` and
`sweep_unrank_triples_kernel` with `k = 3` (both in `qpadm_fit_kernels.cu`). Each
thread passes a small stack array `int c[4]` / `int c[3]`, then maps the returned
positions back to real f2 population indices through the sweep's optional subset
table. The rank offset `c0 + t` lets a sweep be launched in tiled chunks — `c0` is
the chunk's base rank and `t` is the thread's offset within it.

---

## 5. `readv2_unrank_pair` — the O(1) closed form for pairs

The general `sweep_unrank` does an O(P) linear scan per thread. For the READv2
all-pairs case (`k = 2`, over `N` samples) that would make the whole sweep O(N³):
C(N,2) ≈ N²/2 threads each scanning O(N). `readv2_unrank_pair` removes the scan
entirely with a closed form, so each thread is genuinely O(1).

The trick is that the pair space enumerated the way `sweep_unrank` does it — `j` as
the outer (more significant) digit, `i` as the residual — groups naturally by `j`:

- The block for a given `j` (where `j` runs `1..N-1`) starts at rank
  `C(j, 2) = j·(j-1)/2` and holds exactly `j` pairs, namely `i = 0..j-1`.
- So to unrank `r`: find the largest `j` with `C(j, 2) <= r`, then `i = r - C(j, 2)`.

Solving `j·(j-1)/2 <= r` for the largest such `j` is a quadratic, giving the
closed-form seed `j = floor((1 + sqrt(1 + 8r)) / 2)`. The routine computes that
seed, then **corrects it with two short clamp loops**:

```
while (j·(j-1)/2 > r) --j;      // nudge down if the sqrt overshot a block edge
while ((j+1)·j/2 <= r) ++j;     // nudge up if it undershot
```

These loops exist because the `sqrt` is done in `double` and floating-point rounding
can land the seed one block off exactly at a block boundary. The loops are the safety
net: they run at most a step or two and pin `j` to the exact block. It then returns
`c[0] = i` and `c[1] = j`, so `c[0] < c[1]` holds by construction. Like the general
routine, `N` is accepted for signature symmetry and documentation but not read
(`(void)N;`) — the answer depends only on `r`.

---

## 6. The rank convention is a contract shared across three files

The single most important invariant here is that everyone agrees on **how a pair
maps to a rank**. `readv2_unrank_pair` encodes: `rank(i, j) = C(j, 2) + i` for
`i < j`. That formula is not private to this header — three places depend on it
matching, byte for byte:

1. **`readv2_mismatch_direct_kernel`** (one thread per pair) calls
   `readv2_unrank_pair(r, …)` to turn its thread rank into `(i, j)` and writes its
   result to output slot `r`.
2. **`readv2_mismatch_tiled_kernel`** does *not* call the unrank — it already knows
   its `(I, J)` from the tile geometry — but it must write to the *same* output slot.
   So it recomputes the rank the other direction: `r = C(J, 2) + I`. Its inline
   comment says as much ("matches `readv2_unrank_pair`"). If the two disagreed, the
   direct and tiled paths would scatter the same pair to different slots.
3. **`unrank_pair_host`** in `src/core/readv2/readv2.cpp` is a host CPU mirror of the
   exact same routine — same seed, same two clamp loops — so the host side can
   reason about which pair a given output row belongs to without re-deriving it.

Because the rank formula is a cross-file contract, this header is the one place it is
allowed to be defined for the device, and any change to the convention would have to
land in all three at once.

---

## 7. Contracts, invariants, and edge cases

- **Rank range.** `sweep_unrank` expects `r` in `[0, C(P, k))` and
  `readv2_unrank_pair` expects `r` in `[0, C(N, 2))`. Callers guarantee this by
  bounds-checking the thread rank against the combination count before calling
  (`if (r >= n_pairs) return;` in the READv2 kernel, `if (t >= C) return;` in the
  sweep kernels). An out-of-range `r` is not defended against inside these functions
  — it is a precondition, not a runtime check.
- **Ascending output.** Both routines return indices in strictly increasing order:
  `c[0] < c[1] < … < c[k-1]`. This is what lets callers treat the tuple as an
  upper-triangular / ordered combination without re-sorting.
- **Caller-owned output.** `c` is a small stack array the caller supplies and sizes
  (`int c[4]`, `int c[3]`, `int c[2]`). Nothing here allocates.
- **Overflow discipline is the caller's job.** These routines take `r` as
  `long long` and the callers are careful to *widen before they multiply* when they
  build that rank — for example `static_cast<long long>(blockIdx.x) * blockDim.x + …`
  in the READv2 kernel, so a large sweep's flat rank never overflows 32-bit on the
  way in. This header trusts that a valid `long long r` arrives.
- **Floating-point at block edges.** The `double`-based comparisons in
  `sweep_choose` and the `sqrt` seed in `readv2_unrank_pair` can land exactly on a
  block boundary where rounding matters. `sweep_unrank`'s `++v` scan is
  self-correcting (it stops at the first `v` that doesn't overshoot); the pair
  routine's two clamp loops are the explicit fix. Both are robust to the rounding,
  which is why the arithmetic can stay in `double` for the large ranks it must reach.
- **Degenerate `sweep_choose`.** `C(n, kk)` returns `0.0` for `kk < 0` or `n < kk`,
  which is both the correct combinatorial value and the sentinel the unrank scan
  needs to terminate cleanly at the low end of each position.

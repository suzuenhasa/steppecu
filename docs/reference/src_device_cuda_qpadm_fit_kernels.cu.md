# `qpadm_fit_kernels.cu` reference

## 1. Purpose

This file holds the CUDA kernels that carry out the qpAdm model fit and the related
f-statistic computations on the GPU. It is the element-wise and small-reduction half
of the fit: the steps where one GPU thread handles one small piece of work, or one
thread runs an entire small linear-algebra problem end to end.

The heavier dense linear algebra — factorizations, matrix inverses, singular-value
decompositions of large matrices, and the big matrix multiplications — does **not**
live here. Those go through the vendor libraries (cuBLAS and cuSOLVER) and are driven
from the GPU backend file instead. What stays here is everything that is naturally a
custom kernel: reading the resident f2 tensor and combining it into f-statistics, the
block-jackknife bookkeeping, the alternating least-squares refinement, the constrained
weight solve, the numerical-rank count, and the large combinatorial sweep.

The file is a private CUDA translation unit. The rest of the codebase never sees the
kernel bodies or the `<<<>>>` launch syntax. It sees only a set of narrow
`launch_*` wrapper functions (declared in the companion `.cuh` header), each of which
picks the grid and block geometry and fires one kernel. Host code calls those
wrappers; the kernels themselves are confined to this file.

A quick orientation to the qpAdm problem these kernels serve: qpAdm models a target
population as a mixture of several *left* (source) populations, measured against a set
of *right* (reference) populations. The core data is a matrix of f4-statistics; a
block jackknife over genome blocks gives both the point estimate and a covariance
matrix; a low-rank factorization of that matrix yields the admixture weights; a
chi-square measures how well the model fits; and a sequence of rank tests and
source-drop tests judges how many sources the data actually supports.

---

## 2. Precision: native double precision throughout, and why

Every kernel in this file computes in **native** double precision (FP64). This is a
deliberate choice and it is different from the rest of the GPU pipeline, where the
matrix-multiply-heavy f2 stages default to a faster *emulated* form of double
precision.

There are two reasons native double precision is used here instead:

1. **The f4 and f3 combines are cancellation-sensitive.** An f4-statistic is a
   difference of four f2 values (an f3 is a difference of three). When those values
   are close, the subtraction cancels most of the significant digits, and the small
   surviving difference is exactly the quantity that matters. Emulated arithmetic is
   accuracy-*approximate*; on a difference like this it is not trustworthy, so these
   combines use the real hardware double.

2. **The reductions must reproduce a specific operation order.** The leave-one-out,
   total, and pseudo-value reductions are written to fold their terms in the *same
   order* as the CPU reference implementation. The CPU reference accumulates in long
   double; these kernels accumulate the same sequence in FP64. On real data (708
   jackknife blocks, a 10-element statistic vector) the FP64 result matches the
   committed reference result to the required tolerance. Emulated arithmetic would
   break that bit-for-bit correspondence.

There is also a plain performance argument: these steps are small per-item scalar
sums, not large matrix multiplications. Emulated double precision only pays off on
big GEMMs, so it would offer nothing here even setting accuracy aside.

The upshot is a clean split of responsibility. The emulated-precision path speeds up
the big f2 matrix multiplies elsewhere; the native double path in this file keeps the
delicate differences and the parity-critical reductions exact, and it doubles as the
reference that the emulated path is validated against.

---

## 3. The model-size envelope: small path, large path, and launch-time memory limits

Several of the kernels run one entire small linear-algebra problem per thread, using
fixed-size arrays declared inside the thread. That design has a hard constraint
behind it that is worth understanding.

### Why per-thread local arrays are size-limited

CUDA reserves a kernel's per-thread local memory for the device's *maximum* resident
thread count, not for the threads that actually run. So a large per-thread stack frame
is multiplied by tens of thousands of potential threads, and the launch can fail with
an out-of-memory error even if you only ever launch a single thread. This was measured
directly: pushing the per-thread arrays up to a right-population count of 40 tripped a
launch-time allocation failure.

Because of this, the kernels that stack their working arrays in per-thread local
memory serve only a bounded **small model envelope**:

| Bound | Meaning | Value |
|---|---|---|
| `kQpMaxNl` | Maximum number of left (source) populations | 5 |
| `kQpMaxNr` | Maximum number of right (reference) populations | 10 |
| `kQpMaxR` | Maximum fit rank | 4 |
| `kQpMaxM` | Maximum statistic-vector length, `nl·nr` | 50 |
| `kQpMaxT` | Maximum factor dimension, `max(nl,nr)·r` | 40 |

These bounds are defined once, in a small CUDA-free header shared by the host model
search, the GPU backend, and this file, so all three agree on the same envelope and
none of them can drift. A model that fits inside the envelope runs on the fast
stacked-local kernels; a model that exceeds it runs on a different set of kernels (see
below). The host code checks which case applies before launching.

### The two paths: small (stacked-local) and large (VRAM scratch)

- **Small path.** For models inside the envelope, the work runs entirely in
  per-thread local arrays and registers, with no scratch traffic to global memory.
  These are the fast, bit-parity kernels, and they are what the batched
  many-models-at-once kernels use.

- **Large path.** For models that exceed the envelope, the identical math runs but
  every working array is a pointer into a scratch arena in GPU memory (VRAM) that the
  host sizes and allocates at runtime. Because nothing lives in the per-thread frame,
  any left/right/rank size fits and there is no launch-time memory failure. These
  kernels and helpers carry a `_large` suffix.

Crucially, the two paths share their math. Each core computation (the two
alternating-least-squares half-steps, the chi-square, the constrained weight solve) is
written **once** as a device function that takes its working storage as pointers. The
small path passes pointers to per-thread local arrays; the large path passes pointers
into the VRAM arena. The arithmetic and the order of operations are identical, so the
two paths produce the same result and there is only one body to maintain.

### The small wrappers are non-inlined on purpose

The small-path wrappers that declare the local arrays are marked `__noinline__`. This
keeps the first half-step's stack frame from stacking on top of the second half-step's
frame during the refinement loop. If they inlined and their frames coexisted, the
combined per-thread frame would be large enough to bring back the launch-time
out-of-memory failure. Keeping them separate calls bounds the peak frame to one
half-step at a time.

---

## 4. Launch geometry and numeric constants

A handful of named constants set the launch geometry and the convergence behavior of
the in-kernel solvers. They are single-sourced here (or in a shared header) so that,
for example, a change to a block edge automatically moves the matching grid divisor.

| Constant | Value | What it's for |
|---|---|---|
| `kSymTile` | `16` | The edge length of the square 2-D thread block used by the symmetrize kernel: a 16×16 block tiled over the n×n matrix. The grid divisor is derived from this same constant, so a block-size change cannot silently under-cover the matrix. |
| `kWarpSize` | `32` | The GPU warp size, used as the rounding granularity when picking a small block size for the per-model diagonal-fudge kernel (round `m·m` up to a whole number of warps, floored at one warp). |
| `kMaxGridDimX` | `2^31 − 1` (equals `INT_MAX`) | The hardware limit on the x-dimension of a 1-D launch grid, taken from the shared launch-config header. Every 1-D grid-stride kernel clamps its computed grid to this. Note that only the x-dimension reaches `2^31−1`; the y- and z-dimensions are capped at 65535. This value equals `INT_MAX`, so the clamped grid always fits in an `int`. |
| `kOffConvergence` | `1e-30` | The stopping threshold for the in-kernel one-sided Jacobi SVD: the sweep stops when the summed squared off-diagonal magnitude falls below this. **Frozen** — the value sets the exact iteration count that matches the reference, so it may be renamed but its value must not change. |
| `kJacobiTol` | `1e-15` | The per-rotation relative tolerance for the in-kernel Jacobi SVD (a rotation is skipped when the off-diagonal term is negligible relative to the two column norms). **Frozen** for the same parity reason. Shared by both routines that run the identical sweep so they cannot drift apart. |
| `kJacobiMaxSweeps` | `60` | The maximum number of Jacobi sweeps before giving up. **Frozen** for parity. |

The three Jacobi constants are called out as frozen because they jointly determine how
many iterations the sweep runs, and therefore the exact singular values it produces.
Changing any of them would change a reported numerical rank. They are safe to rename,
not to re-value.

---

## 5. Device linear-algebra helpers: LU factorization, solve, and one-sided Jacobi SVD

These single-thread helpers are the building blocks the fit kernels call. Each is a
line-for-line transliteration of the corresponding routine in the CPU reference, so
the GPU fit matches the reference bit for bit. All matrices are stored column-major:
element `(i, j)` of an `n`-row matrix lives at index `i + n·j`.

### LU factorization (`dev_lu_factor`)

Factors an n×n matrix in place using partial pivoting (it picks the largest-magnitude
pivot in each column and records the row swaps). It returns false if a pivot is exactly
zero, meaning the matrix is singular.

### Linear solve (`dev_solve`)

Solves `A·x = b`. It copies `A` into caller-provided scratch, factors that copy,
applies the recorded row swaps to `b`, then does the forward and back substitution.
The caller supplies all scratch buffers so the routine allocates nothing. On a singular
matrix it returns false and leaves the answer untouched, which lets each caller apply
its own "rank-deficient" policy (write zeros, write NaN, or set a status code).

### One-sided Jacobi SVD (`dev_jacobi_svd_V`)

This computes the right singular vectors of an m×n matrix without any library call. It
works by repeatedly sweeping over all pairs of columns and applying a plane rotation to
each pair that drives their mutual inner product toward zero (this is the "one-sided
Jacobi" method — it orthogonalizes the columns in place). After enough sweeps the
columns are mutually orthogonal; each column's norm is then a singular value, and the
accumulated rotations form the right singular vectors. The columns are sorted by
descending singular value, and the leading `r` right singular vectors are written out.

Two robustness details: the output buffer is fully zero-initialized first, so any
columns beyond the true rank read as clean zeros rather than uninitialized memory; and
the tolerance and sweep-cap constants it uses are the frozen Jacobi constants from the
previous section, so it produces exactly the iteration count the reference does.

---

## 6. Seeding and refining a fit: seed, ALS, chi-square, and constrained weights

The heart of qpAdm is a low-rank factorization of the f4 matrix `X` (dimensions nl×nr)
into two smaller matrices `A` (nl×r) and `B` (r×nr) such that `A·B` approximates `X`
under a metric given by the inverse covariance. This is solved by alternating least
squares (ALS): fix one factor, solve for the other, and repeat.

### Seeding (`seed_ab_from_V`, `dev_seed_ab`)

The ALS loop needs a starting point. The seed comes from the singular-value
decomposition of `X`: `B` is the transpose of the leading `r` right singular vectors,
and `A` is `X·Bᵀ`. There are two ways to get the singular vectors: `dev_seed_ab` runs
the in-kernel Jacobi SVD from the previous section (used on the small path and the
nr-heavy fallback), while the large path computes them with cuSOLVER on the host and
passes them in. Either way, the final "build A and B from the vectors" step is the same
shared function, so both seeding routes agree.

### The two ALS half-steps and the chi-square (`dev_opt_A_core`, `dev_opt_B_core`, `dev_chisq_of_core`)

`opt_A` holds `B` fixed and solves for the best `A`; `opt_B` holds `A` fixed and solves
for the best `B`. Each builds a small normal-equations system weighted by the inverse
covariance, adds a small "fudge" term to the diagonal for numerical stability (a fixed
fraction of the diagonal's trace), and solves it with the LU solve above. The
chi-square routine forms the residual `E = X − A·B`, vectorizes it row-major, and
returns `vec(E)ᵀ · Qinv · vec(E)` — the fit's goodness-of-fit number.

As described in section 3, each of these three is a single core body that takes its
scratch as pointers. The small-path wrappers wrap it with per-thread local arrays; the
`_large` variants wrap it with VRAM-arena pointers.

### The constrained weight solve (`solve_constrained_weights`)

Once `A` is refined, the admixture weights come from a small constrained solve. It
builds an nl×nl cross-product system whose extra constraint row forces the weights to
sum to one, solves it, and normalizes so the weights sum to exactly one. It returns
false on a singular system (a rank-deficient model), leaving the output untouched so
the caller can mark the model as rank-deficient. This one body replaces four
previously-duplicated copies across the small, large, and batched kernels.

### The full single-model fit (`dev_als_weights`)

This ties the pieces together for one model: optionally seed `A` and `B` from the SVD,
run the ALS loop for a fixed number of iterations, solve for the constrained weights,
and compute the chi-square. It returns a status code — success, or rank-deficient. A
rank-zero model is a special case (all weights equal, only the chi-square is computed).
It is templated on the size bounds so the same code serves both the small batched
kernels (small bounds, many threads) and the single-thread fallback (larger bounds).

---

## 7. Building f4 and f3 statistics from the resident f2 tensor

These gather kernels read the f2 tensor that already lives in GPU memory — a
column-major `P×P×nb` array (P populations, nb genome blocks) — and combine its entries
into f4 or f3 statistics per block. They never copy the tensor back to the host.

### The qpAdm f4 gather (`assemble_f4_gather_kernel`)

For the qpAdm fit, one launch fills the whole nl×nr per-block matrix. Each output
element indexes a left population `Li` and a right population `Rj` against a shared base
pair `(L0, R0)`, and computes the four-slab f4 identity:

```
X[j + nr·i, b] = 0.5 · ( f2(Li,R0,b) + f2(L0,Rj,b) − f2(L0,R0,b) − f2(Li,Rj,b) )
```

The `0.5` factor and the four-term difference are the standard f4 formula. This
difference is the cancellation-sensitive quantity that motivates native double
precision (section 2).

### Standalone quartet and triple gathers (`assemble_f4_quartets_gather_kernel`, `assemble_f3_triples_gather_kernel`)

The standalone f-statistic tools (computing many independent f4 or f3 values) use two
sibling kernels. The quartet kernel gives each output column its own four populations
`(p1,p2,p3,p4)` and applies the same four-slab identity `0.5·(f2(p2,p3)+f2(p1,p4)
−f2(p1,p3)−f2(p2,p4))`. The triple kernel is the three-slab analogue for f3:
`0.5·(f2(C,A)+f2(C,B)−f2(A,B))`. Both read the resident tensor directly and write a
dense per-block output.

All three gathers understand *survivor compaction* for missing blocks; that mechanism
is described in section 12.

---

## 8. Leave-one-out values, totals, and the jackknife pseudo-values

qpAdm quantifies uncertainty with a block jackknife: it recomputes the statistic with
each genome block left out in turn, and reads the spread of those recomputations as the
error. These kernels do that bookkeeping. They reproduce the CPU reference's
computation, and its exact operation order, in native double precision.

### Leave-one-out and totals per row (`f4_loo_total_row`, `f4_loo_total_kernel`)

For each row `k` of the statistic vector, this computes:

- the **total** across all blocks, a block-size-weighted average;
- the **leave-one-out** value for every block `b`, `loo[k,b]`, which is what the total
  would be with block `b` removed;
- the **weighted-mean line** `tot_line`, a weighted average of the leave-one-out
  values;
- the **jackknife estimate** for the row.

There is a small memory-access optimization here that is written to be exactly
parity-preserving. The leave-one-out values must be written out to global memory (there
are too many blocks — up to hundreds — to keep a whole row in registers). Two of the
downstream sums do not depend on `tot_line`, so they are accumulated on the fly from
each leave-one-out value as it is computed, folding what used to be a separate pass into
the main loop. Only the one sum that genuinely needs `tot_line` re-reads the
leave-one-out values in a second pass. This cuts the global-memory touches per
leave-one-out value from three to two while running the additions in the identical order
as before, so the result is bit-for-bit unchanged.

### The jackknife pseudo-values (`f4_xtau_elem`, `f4_xtau_kernel`)

From the estimate, the leave-one-out values, and `tot_line`, each block yields a
*pseudo-value* (`xtau`). The pseudo-values are laid out column-major so that a single
matrix multiply of the pseudo-value matrix against its own transpose, divided by the
block count, produces the jackknife covariance matrix `Q` that the fit uses. The formula
per block uses `h = n/bl` (the total sample size over the block size) and scales by
`sqrt(h−1)`.

### The diagonal-only jackknife variance (`f4_diag_var_kernel`)

The standalone f-statistic tools do not need the full covariance matrix — they only
need each item's own variance. This kernel computes exactly the diagonal of `Q`,
`var[k] = (1/nb)·Σ_b xtau[k,b]²`, directly as a per-item sum of squares. It avoids ever
forming the dense m×m matrix, the matrix multiply, the fudge step, and the Cholesky
factor/inverse — so it costs `O(m·nb)` work and `O(m)` memory instead of `O(m²)`, which
is what lets a sweep of billions of items avoid running out of memory. Because it reads
the very same pseudo-value buffer the full path produces, it reproduces the existing
reference results by construction. The accumulation is a well-conditioned sum of
squares, so plain native double precision is exact enough.

---

## 9. Numerical rank of the covariance matrix

The fit reports a diagnostic — the numerical rank of the m×m covariance matrix `Q` —
that indicates how well-determined the model is. `rank_via_jacobi_kernel` computes it
on the GPU (it used to be the last small piece of per-model work still done on the
host).

It reuses the exact same one-sided Jacobi sweep as the SVD helper, with the same frozen
tolerances and sweep cap, so its singular values match the CPU reference's exactly.
Two simplifications apply: because only the singular values are needed (not the
vectors), the vector accumulation is dropped — the singular values are the column norms
of the rotated matrix, which do not depend on the vectors. And it works in a small VRAM
scratch buffer rather than stacked per-thread arrays, so any production matrix size fits
without a launch-time memory failure.

The rank is the count of singular values above a tolerance `smax · m · eps`, where
`smax` is the largest singular value and `eps` is machine epsilon passed in from the
host so the constant is identical to the reference's. This is the exact tolerance
formula the CPU reference uses.

---

## 10. The full model-fit kernel: rank sweep, popdrop, and staged standard errors

`qpadm_fit_models_kernel` runs a complete qpAdm fit for one model per thread, with many
models batched across the grid. Each thread reads its own model's slice of a strided
arena and writes its own slice of the outputs. Its stages, in order:

1. **Reshape** the row-major total vector into the column-major fit matrix.
2. **The main fit** at the requested rank: run `dev_als_weights`, store the weights (or
   zeros if rank-deficient), store the chi-square and the status code.
3. **The rank sweep**: refit at every rank from 0 up to the maximum and store each
   chi-square, so the caller can pick the number of sources the data supports.
4. **Popdrop**: refit the model with each single left source removed in turn (plus the
   full model), each at its own reduced rank, and store the resulting chi-squares and
   the full model's weights. This is the "does dropping this source hurt the fit?" test.
   The ordering of the drops is the parity ordering[^at2]. A rank-deficient reduced
   fit stores NaN, matching the weights' not-a-number-on-singular contract.

Two design notes on this kernel:

- **The standard error is deliberately not computed here.** Computing the jackknife
  standard error means refitting the model once per genome block — hundreds of extra
  ALS fits — which, if done inside this per-model thread, would serialize into a
  throughput wall. Instead the standard error runs as a separate batched kernel with one
  thread per (model, block) pair, followed by a deterministic variance reduction (see
  section 11).

- **The memory-access pattern is strided by design.** Adjacent threads own adjacent
  *models*, and each model's data is stored contiguously, so consecutive threads stride
  by a full per-model slice rather than reading consecutive addresses. This is not the
  ideal coalesced pattern, and it is accepted knowingly: the only remedy is a structural
  data-layout change (a struct-of-arrays arena laid out across models) that would have
  to be matched in every producer and consumer of these arenas — a cross-cutting rewrite
  well beyond a local fix. The per-model math and its operation order are unchanged, so
  this is a performance trade-off, not a correctness issue. It is flagged as the follow-up
  to make if this kernel ever profiles as the bottleneck.

The batched fit kernels carry a `__launch_bounds__` hint that pins the per-thread
register budget to the occupancy target for their single fixed block size. Because the
block size is compile-time fixed and the kernels use a grid-stride loop, the bound is
never exceeded, so it only guards occupancy and can never cause a launch failure. There
are single-thread, non-batched versions of the seed, ALS, and weight/chi-square kernels
as well, used by the host when it drives the fit one stage at a time rather than as one
fused batched kernel.

---

## 11. The leave-one-out standard error and the large-model refit path

The jackknife standard error is produced by refitting the model with each genome block
left out, then measuring the spread of the resulting weights. Because those refits are
independent of one another, they are done in parallel rather than in a host-side loop.

### The batched refit (`qpadm_loo_models_kernel`, `qpadm_se_from_wmat_kernel`)

One kernel runs one thread per (model, block): each thread fits the leave-one-out
weights for one block of one model and writes a scaled weight vector into a matrix. A
second kernel then reduces that matrix into the per-weight standard error, using the
sample variance across blocks. The reduction runs in a fixed ascending-block order with
no atomics, so it is deterministic and gives identical results regardless of how the
work is spread across GPUs.

### The large-model parallel refit (`loo_large_batched_kernel`)

For models outside the small envelope, the same idea runs with VRAM scratch: one thread
per (model, block), each thread reusing the `_large` device helpers and carving its own
non-overlapping slice of a scratch arena (so the threads never race). One important
parity detail: the SVD seed for each block is precomputed on the host with cuSOLVER and
passed in, *not* recomputed with the in-kernel Jacobi. Recomputing it would shift the
ALS fixed point in the last bits and change the standard error, so the host-computed
seed keeps each refit bit-identical to the serial reference. The chi-square is unused
output for these refits, so it is simply skipped.

---

## 12. The all-combinations sweep and the bounded top-K reservoir

The standalone f-statistic tools can sweep over *every* combination of populations — for
example, every possible quartet out of many hundreds. That can be an astronomical number
of items. These kernels run the whole sweep on the GPU, with no host-side enumeration
loop at all, and keep memory bounded no matter how many combinations there are.

### Generating combinations directly (`sweep_unrank_quartets_kernel`, `sweep_unrank_triples_kernel`)

Rather than enumerate combinations on the host and copy them over, each thread computes
its own combination from a single integer index. This uses the *combinatorial number
system* in colexicographic order: the item at global rank `r` is the unique k-subset
whose members satisfy `r = C(c₀,1) + C(c₁,2) + … + C(c_{k−1},k)`. The kernel peels off
the largest index first (find the largest `c` with `C(c,k) ≤ r`, subtract it, recurse),
so every thread recovers its whole quartet or triple independently with no shared state.
The small binomials involved are computed in double precision and are exact over the
sweep's population range. An optional subset map lets the sweep run over a chosen subset
of populations rather than all of them; by default it is the identity. The kernels write
the exact flat index layout that the gather kernels of section 7 read, so generating a
chunk of work replaces what used to be a host-to-device copy of an enumerated chunk.

### Filtering by significance (`sweep_zfilter_kernel`)

After each chunk's estimates and variances are computed, this kernel derives the
standard error and the z-score per item and flags which items survive. In the
minimum-z mode it keeps items whose absolute z-score is at least a threshold; a
degenerate item (non-positive variance) yields a not-a-number z-score, which fails the
comparison and is therefore dropped — the honest behavior. In the keep-all mode every
item is flagged, and the top-K selection happens afterward.

### The bounded top-K reservoir with a rising threshold

A naive sweep would collect all survivors into an unbounded host vector and run out of
memory on a huge sweep. The fix is a bounded device-resident reservoir that keeps only
the K most significant items, using a threshold that rises as the sweep proceeds:

- `sweep_zfilter_tau_kernel` filters each chunk against a *device-resident* threshold
  read from memory, rather than a fixed host constant. It also writes each survivor's
  absolute z-score as the sort key.
- `sweep_topk_iota_kernel`, `sweep_topk_gather_kernel` support sorting and reordering the
  reservoir: an index array is sorted alongside the z-scores, then used to gather every
  column of the reservoir into descending order so each item's full tuple stays together,
  and the reservoir is truncated back to K.
- `sweep_topk_raise_tau_kernel` raises the threshold to the new K-th-largest absolute
  z-score after each truncation, guarded so it can only ever *rise*.

The monotone-rise guard is what makes the bounded reservoir exact for the global top-K:
because the threshold never drops, no item that was filtered out at an earlier, lower
threshold could possibly have beaten the current K-th-place item. Only the K rows ever
travel back to the host, so a multi-billion-item sweep can never blow up host memory.
The threshold rises only in top-K mode; in minimum-z mode it stays pinned at the floor.

### Splitting keys for compaction (`sweep_deinterleave_keys_kernel`)

The library primitive that stream-compacts the survivors takes one input array at a
time, so this small kernel splits the interleaved per-item key array into separate
contiguous columns (up to four, one per population index) that can each be compacted
with the same survivor flags.

---

## 13. Missing-block handling: keep-mask and survivor compaction

Genome blocks with missing data must be excluded from the f-statistics, and the two
backends must agree exactly on which blocks to drop.

`f2_block_keep_kernel` builds the keep-mask: one thread per resident block scans that
block's paired-variance slab and marks the block dropped only if it is *partially*
covered — that is, it has at least one pair with zero variance and at least one pair with
positive variance. A fully-zero slab is a "no variance information" sentinel from the
legacy zero-fill, not a genuinely missing block, so it is kept. This mirrors the CPU
reference's drop rule exactly and shares the single-source predicate that defines
"missing," so the two backends cannot diverge.

The host reads the mask and builds an ascending list of surviving block ids. The gather
kernels of section 7 then take that list and a survivor count: each thread maps its
compacted survivor index to the original resident block id, reads the resident tensor at
the original block, and writes into the dense compacted output. When there are no missing
blocks the list is a null pointer meaning "identity," and the gather is bit-identical to
the version before missing-block handling existed.

---

## 14. Small matrix utilities, the gather helper, and launch geometry

A set of small helper kernels round out the file.

- **Symmetrize (`symmetrize_lower_to_full_kernel`).** Copies a matrix's lower triangle
  into its upper triangle in place, using a 2-D block that strides over both axes. The
  stride matters at large sizes: the covariance in the qpfstats path is `npairs × npairs`
  with `npairs = P·(P−1)/2`, and for a few thousand populations the natural tile count on
  the y-axis would exceed the hardware cap of 65535. The launch wrapper clamps the y-grid
  to that cap, and the 2-D stride still covers every element — identical to the unclamped
  mapping when no clamp is needed.
- **Diagonal fudge (`add_fudge_diag_kernel`, `add_fudge_diag_models_kernel`).** Adds a
  small multiple of the trace to a matrix's diagonal for numerical stability, in the
  single-matrix and per-model batched forms.
- **Reshape and transpose (`xmat_from_rowmajor_kernel`, `transpose_small_kernel`).**
  Convert between the row-major statistic layout and the column-major fit layout.
- **Identity fill (`fill_identity_batched_kernel`).** Fills a batched identity-matrix
  arena, used as the right-hand side for batched inverses.
- **Survivor gather (`qpadm_gather_loo_qinv_kernel`).** Compacts the surviving models'
  leave-one-out and inverse-covariance slices into dense output in one launch, replacing
  a per-survivor copy loop whose launch overhead dominated at large survivor counts. It
  is pure data movement — it moves bits, it computes nothing — and the ascending survivor
  order keeps each compacted slice bit-identical to its original.

### Grid geometry and the grid-stride safety net

Most launch wrappers use a shared helper (`launch_grid_stride`) that ceil-divides the
total work by the block size and clamps the result to the maximum x-grid dimension. This
clamp is not merely defensive: on the largest sweeps the total item count reaches several
billion, so the raw grid would overflow a signed 32-bit integer and wrap negative. The
clamp caps the grid at the maximum, and because every one of these kernels is written as
a grid-stride loop (each thread strides forward by the whole grid until the work is
exhausted), a clamped grid still covers the full range. The result is identical to what a
hypothetical non-overflowing launch would produce. Every wrapper also guards against a
non-positive work count before launching, and checks for launch errors afterward.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

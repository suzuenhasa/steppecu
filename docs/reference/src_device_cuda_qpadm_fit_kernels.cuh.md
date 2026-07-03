# `qpadm_fit_kernels.cuh` reference

## 1. Purpose

`src/device/cuda/qpadm_fit_kernels.cuh` is the private header that lists the GPU
kernel *launchers* for the qpAdm model-fitting stage. It contains only narrow
`void launch_*(...)` function declarations. Each of these functions is a thin
wrapper: the host-side orchestration code calls `launch_something(...)`, and the
actual kernel body and its `<<<grid, block>>>` launch configuration live in the
matching `.cu` file, never here. This split keeps the host code from having to
compile any device kernel code directly.

Because these declarations name a CUDA type (`cudaStream_t`), this header is
internal to the GPU part of steppe. It is the seam between the GPU backend and its
own kernels — not the public, CUDA-free interface that the rest of the library
talks to.

Every kernel declared here does its arithmetic in **native double precision**
(native FP64), not the faster emulated double precision used for the bulk
matrix-multiply work elsewhere. The reason is numerical: the core f-statistic is a
difference of nearly-equal quantities, so it loses precision to cancellation, and
the jackknife reductions have to add up terms in a specific order. Running these in
native FP64 lets the GPU reproduce the operation order of the CPU reference
implementation closely enough that, at the reference problem size (708 jackknife
blocks, a 10-entry statistic vector), the GPU result matches the trusted reference
to the required validation tolerance (relative tolerance around 1e-6).

The CPU reference implementation (called the CPU backend) computes the same
quantities in extra-wide long-double arithmetic and serves as the correctness
oracle. It is a development and test tool only — never a user-facing runtime. Many
kernels here are described as "transliterating" the CPU backend: they perform the
exact same scalar operations in the exact same order, just on the GPU, so the GPU
answer is bit-for-bit (or tolerance-bounded) identical to that oracle.

---

## 2. Shared conventions and invariants

The same handful of ideas recur across almost every function in this header.
Understanding them once makes each individual declaration easy to read.

### Symbol glossary

| Symbol | Meaning |
|---|---|
| `P` | Number of populations in the f2 tensor. |
| `nb` | Number of jackknife blocks. After missing-block dropping (below) this is the *survivor* block count. |
| `nl` | Number of left (source) populations beyond a shared reference. There are `nl+1` left indices, index 0 being the reference `L0`. |
| `nr` | Number of right (reference) populations beyond a shared reference. There are `nr+1` right indices, index 0 being the reference `R0`. |
| `m` | The length of a flattened f4 statistic vector, `m = nl * nr`. |
| `r` | The model rank being fitted. |
| `n` | Total SNP count across all blocks, `n = Σ_b bl_b`. |
| `bl_b` | The SNP count of block `b` (the block size / weight). |

### The resident f2 tensor and its layout

The heavy first-stage output — the f2 tensor — stays resident in GPU memory (VRAM)
for the whole fit. It is a `P × P × nb` array stored **column-major** with the flat
index `i + P*j + P*P*b` (population `i`, population `j`, block `b`). Every gather
kernel reads this tensor directly out of VRAM. None of them copy it back to the
host first; the comments call this out as "NO D2H" (no device-to-host copy).

### The row-major f4 vectorization

An f4 matrix for one block is a small `nl × nr` grid of statistics. It is flattened
**row-major** into a single index `k = j + nr*i`, and then stored across blocks as
`X[k + m*b]` with `m = nl*nr`. So `X` is column-major in `(k, b)`: all `m` entries
of one block are contiguous. This is the layout the downstream jackknife and
covariance steps expect.

### Native FP64 (the cancellation carve-out)

Every kernel here is native FP64. The rest of steppe defaults to emulated double
precision for speed, but the f-statistic difference and the jackknife reductions are
the numerically delicate parts, so they are carved out to native double precision to
keep the GPU aligned with the FP64 oracle.

### Survivor-block compaction (`d_surv`)

A jackknife block is treated as *missing* when any population pair has zero valid SNP
overlap in that block (this mirrors ADMIXTOOLS 2 reading f2 with `remove_na = TRUE`).
Missing blocks are dropped from the analysis. The mechanism is:

- A keep-mask kernel (section 4) flags which resident blocks survive.
- The host reads that mask and builds `d_surv`, an **ascending** list, length `nb`,
  mapping a compacted survivor index to the original resident block id.
- The gather kernels then read the resident f2 tensor at the *original* block id but
  write their output densely into the *compacted* survivor positions.

Passing `d_surv == nullptr` means identity — no block is dropped, and the output is
bit-identical to the path that predates missing-block handling. This is the common
"no missing blocks" case.

### The stream argument

Every launcher's final parameter is a `cudaStream_t`. The host enqueues all fit work
on a single statistics stream so the ordering is deterministic; the launchers just
forward that stream to their `<<<>>>` launch.

### Single-thread transliteration kernels

At the reference (small) model sizes — for example `m = 10`, `nl = 2`, `nr = 5`,
`r = 1` — the linear-algebra steps are tiny. Several kernels here therefore run as
**single-thread device kernels**: one GPU thread executes the CPU reference's exact
scalar operations in the exact same order. This is slow per-model but keeps the
result bit-exact against the FP64 oracle while still running on the GPU (device
resident), which is the required production shape. Larger models and batched models
get dedicated variants (sections 9 and 10).

---

## 3. Building f4 and f3 matrices from the f2 tensor

Three "gather" kernels turn the resident f2 tensor into the per-block statistic
matrix that the fit consumes. They all read f2 from VRAM, all write native FP64, and
all honor the `d_surv` survivor compaction from section 2.

### `launch_assemble_f4_gather`

Builds the per-block f4 matrix `dX[k + m*b]` for a **single** shared reference pair:
one reference left population `L0` and one reference right population `R0`, spanning
the full `nl × nr` grid of the other left/right populations. It is batched over all
blocks in one launch (the grid runs over `(k, b)`). The per-entry formula is the
four-slab f4 identity:

```
X[j + nr*i, b] = 0.5 * ( f2(Li, R0, b) + f2(L0, Rj, b)
                       - f2(L0, R0, b) - f2(Li, Rj, b) )
```

with `Li = d_left[i+1]`, `Rj = d_right[j+1]`, `L0 = d_left[0]`, `R0 = d_right[0]`.
`d_left` (length `nl+1`) and `d_right` (length `nr+1`) are small index buffers
copied up to the device. Output is `dX[m*nb]`, `m = nl*nr`.

### `launch_assemble_f4_quartets_gather`

The standalone f4 gather whose statistic axis is `N` arbitrary **quartets**. Unlike
the grid form above, every column has its own four populations `(p1, p2, p3, p4)`
instead of sharing one `(L0, R0)`. `d_quartets` is a flattened index array of length
`4*N`, with quartet `k` at positions `[4*k .. 4*k+3]`. Per entry:

```
dX[k + N*b] = 0.5 * ( f2(p2, p3, b) + f2(p1, p4, b)
                    - f2(p1, p3, b) - f2(p2, p4, b) )
```

This is the same four-slab identity specialized to a single pair per column — no new
math, just a per-column quad rather than a shared reference pair.

### `launch_assemble_f3_triples_gather`

The three-slab sibling for f3 statistics. Its statistic axis is `N` arbitrary
**triples**, each with its own `(C, A, B)`. `d_triples` is a flattened index array of
length `3*N`, triple `k` at `[3*k .. 3*k+2]` = `{C, A, B}`. Per entry:

```
dX[k + N*b] = 0.5 * ( f2(C, A, b) + f2(C, B, b) - f2(A, B, b) )
```

The three-slab f3 identity is the one piece of new math f3 adds over f4; everything
else (VRAM read, native FP64, survivor compaction) is identical.

---

## 4. The missing-block keep mask

### `launch_f2_block_keep`

Decides which resident jackknife blocks survive. One thread per resident block writes
`d_keep[b] = 0` when block `b` is **partially** covered — meaning it has at least one
population pair with zero valid overlap *and* at least one pair with real data — and
`d_keep[b] = 1` otherwise. A block that is *entirely* zero is treated as a
"no-information" sentinel from the legacy zero-fill path, not a genuine missing block,
so it is kept. `d_vpair` is the resident `P × P × nb` pair-overlap tensor; `d_keep`
has length `nb`. The host reads `d_keep` back and turns it into the ascending
survivor-id list (`d_surv`) that section 3's gathers use. The kernel uses the same
single predicate as the CPU reference's survivor-block logic, so the two agree by
construction.

---

## 5. Block-jackknife statistics

These three kernels turn the per-block f4/f3 matrix `dX` into the point estimate, the
jackknife pseudo-values, and (optionally) the per-item variance. They reduce over the
`nb` blocks and reproduce the CPU reference's operation order in native FP64.

### `launch_f4_loo_total`

Computes the weighted total, the leave-one-out estimates, the "total line", and the
final estimate. It uses one thread per statistic entry `k` (the statistic vector is
small). For each `k`, with `rel_b = bl_b / n`:

```
tot_ij   = ( Σ_b X[k,b] * bl_b ) / n
loo[k,b] = ( tot_ij - X[k,b] * rel_b ) / ( 1 - rel_b )
tot_line = ( Σ_b loo[k,b] * (1 - bl_b/n) ) / ( Σ_b (1 - bl_b/n) )
est[k]   = ( Σ_b (tot_line - loo[k,b]) ) + ( Σ_b loo[k,b] * bl_b ) / n
```

`d_block_sizes` holds the per-block SNP counts `bl_b` (length `nb`); `n` is their sum.
Outputs are `dLoo[m*nb]`, `dTotal[m]`, and `dTotLine[m]`. The CPU reference's
`mean(...) * nb` simplifies to the bare sum shown above, and this kernel reproduces
that simplification exactly so the numbers line up.

### `launch_f4_xtau`

Computes the jackknife pseudo-values used to form the covariance. One thread per
`(k, b)`, with `h = n / bl_b` and `sh = sqrt(h - 1)`:

```
xtau[k,b] = ( est[k]*h - loo[k,b]*(h-1) - tot_line[k] ) / sh
```

The output `dXtau[m*nb]` is laid out **column-major** as `k + m*b` on purpose: in that
layout a symmetric matrix-multiply (cuBLAS `Dsyrk`, no-transpose, leading dimension
`m`, contraction length `nb`) forms the covariance `Q = xtau · xtauᵀ` directly. Here
`dEst` is the same buffer as `dTotal` (length `m`), `dTotLine` has length `m`, and
`dLoo` is `m*nb`.

### `launch_f4_diag_var`

Computes only the **diagonal** of the jackknife covariance — the per-item variance —
without ever forming the dense `m × m` covariance. One thread per item `k`:

```
var[k] = (1/nb) * Σ_b dXtau[k + m*b]²
```

This equals the diagonal entry `Q[k + m*k]` that the f4/f3 per-item standard error
reads, but it skips the dense `Dsyrk`, the regularization step, and the Cholesky
factor/inverse. That makes it `O(m·nb)` work and `O(m)` memory instead of `O(m²)`,
which is what keeps a very large all-combinations sweep from running out of memory. It
reads the exact same `dXtau` buffer `launch_f4_xtau` produced, so it re-passes the
existing FP64 goldens by construction.

---

## 6. Covariance-matrix helpers

Two small element-wise kernels finish the covariance before it is inverted.

### `launch_symmetrize_lower_to_full`

Mirrors the lower triangle of an `n × n` column-major matrix into the upper triangle,
in place. A `Dsyrk` or Cholesky-inverse result fills only one triangle; the CPU
reference writes both, so this makes the GPU matrix fully symmetric to match. One
thread per `(i, j)` with `i > j` copies `(i, j)` into `(j, i)`.

### `launch_add_fudge_diag`

Adds a regularization term to the diagonal of an `n × n` column-major matrix:

```
dM[k + n*k] += fudge * tr
```

`tr` is the matrix trace, computed by the host with a separate trace reduction and
passed in. One thread per diagonal entry. This is the "fudge" ridge step the CPU
reference applies before inverting the covariance.

---

## 7. The small-dense fit core

For models inside the bit-parity envelope (small `nl`, `nr`, `r`), the whole fit is a
sequence of single-thread device kernels that transliterate the CPU reference's scalar
linear algebra — the same operations, the same order, native FP64 — so the GPU fit is
bit-exact against the FP64 oracle *and* stays device resident (no host round-trip).
They consume only device buffers and write only device outputs.

### `launch_qpadm_xmat_from_rowmajor`

Reshapes a row-major statistic slice (from the total estimate or one leave-one-out
slice) into the `nl × nr` column-major `xmat` the fit works with: entry `k = j + nr*i`
becomes `xmat(i,j)` at `i + nl*j`. For the full-data fit the source is `dTotal` (one
slice); for the batched jackknife re-fits a per-block slice is passed. Single thread.

### `launch_qpadm_seed_ab`

Produces the initial admixture factors. It runs a one-sided Jacobi singular-value
decomposition of the `nl × nr` column-major `xmat`, then sets `B = t(V[:, 0:r])`
(shape `r × nr`) and `A = xmat · t(B)` (shape `nl × r`). Outputs `dA[nl*r]` and
`dB[r*nr]` column-major. Single thread (the matrix is small).

### `launch_qpadm_rank_via_jacobi`

Computes the numerical **rank** of the `m × m` column-major covariance `Q`, which is
the model-observability check ("is this model well determined"). It runs the same
one-sided Jacobi sweep as the seed step and counts the singular values above the
threshold `tol = smax * m * eps`, where `eps` is machine epsilon for double precision
passed in from the host so the constant is bit-identical to the host's. The
singular-vector accumulation is skipped because the singular values do not depend on
it, so the count is exact. `dScratch` holds `W[m*m] | sigma[m]` (that is `m*m + m`
doubles); `dIntScratch` holds an `order[m]` index array; `dRank` is the single-int
output. Single thread.

### `launch_qpadm_als`

Refines `A` and `B` by alternating least squares for `als_iters` iterations. Each
iteration solves for one factor holding the other fixed, using the covariance inverse
`dQinv` (`m × m`, `m = nl*nr`, column-major). It transliterates the CPU reference's
Kronecker-coefficient and right-hand-side build, the LU solve, and the by-row reshape,
in native FP64, single thread. It seeds `A`/`B` in place from the caller's `dA`/`dB`
(filled by the seed kernel) and overwrites them with the refined factors. If an ALS
subsystem is singular, that factor is left as zeros, matching the CPU reference.

### `launch_qpadm_weights_chisq`

Solves for the final admixture weights and the goodness-of-fit chi-square. From the
refined `A` (`nl × r`) it builds `RHS = crossprod(cbind(A, 1))` (`nl × nl`) with an
all-ones left-hand side, LU-solves, and normalizes so the weights sum to 1, writing
`dW[nl]`. It then computes `chisq = vec(E)' · Qinv · vec(E)` where `E = xmat - A·B`,
writing `dchisq[0]`. `d_status` (one int) is set to `0` for Ok or `6` for
rank-deficient (the weight solve was singular), matching the CPU reference. When
`r == 0` it takes the trivial path (weights all equal, chi-square computed with empty
`A`, `B`). Single thread.

---

## 8. Batched leave-one-block-out re-fits

### `launch_qpadm_loo_batched`

Runs the entire per-block re-fit — build `xmat` from that block's leave-one-out slice,
seed, run the ALS iterations, solve the weights, normalize — for **all** `nb` blocks in
one launch (one thread or thread-group per block), transliterating the same operations
as the single fit. This replaces what would otherwise be hundreds of separate host
calls (708 at the reference size) with a single batched device kernel. Output is
`dWmat[nb*nl]` stored **row-major** as `b*nl + i` — the replicate weight matrix the
host reduces into a jackknife standard error. Native FP64.

---

## 9. The large-model path

Models that exceed the small bit-parity envelope (`nl > 5`, or `nr > 10`, or `r > 4`;
for example a run with `nr = 39`) cannot keep their working set in per-thread local
memory — CUDA reserves that local memory for the device's maximum resident-thread
count, so a big model would fail to launch with an out-of-memory error. The large-path
kernels move the per-model working arrays into caller-provided VRAM scratch buffers
(`dScratch` / `dIntScratch`) instead. The math and operation order match the small
kernels exactly; native FP64. These run one model per launch for now, and the scratch
offset layout is the seam that a future model-batched version reuses.

For the large path the singular-value decomposition is *not* a kernel here. It is done
with cuSOLVER on the host side, followed by the cheap reshaping kernel below.

### `launch_transpose_small`

Transposes the `nl × nr` column-major `xmat` into the `nr × nl` column-major `dXt`
(`dXt[j + nr*i] = dXmat[i + nl*j]`), one thread per element. This orients the matrix
handed to cuSOLVER so that it has at least as many rows as columns — a determinism
requirement of the routine — and for the common large case `nr >= nl` the right
singular vectors of `xmat` come out as the (more convenient) left singular vectors of
the transpose.

### `launch_qpadm_seed_from_V`

The cheap, matrix-multiply-shaped tail of the seed, run **after** the host's cuSOLVER
SVD. Given the right singular vectors `V[:, 0:r]` (`nr × r` column-major, descending
order) in `dVout`, it sets `B[p,j] = V[j,p]` (`r × nr`) and `A = xmat · t(B)`
(`nl × r`). No large local arrays are needed. Single thread, native FP64. Writes
`dA[nl*r]` and `dB[r*nr]` column-major.

### `launch_qpadm_als_large`

The large-path ALS loop. Same math as `launch_qpadm_als`, but every per-model working
array (`xvec`, `Wm`, `coeffs`, `rhs`, the LU factor, the solve vector, the pivots, the
temporary factor copies) lives in caller VRAM scratch, so any `nl`/`nr`/`m`/`r` fits.
Single thread, native FP64. Seeds in place from `dA`/`dB` (filled by
`launch_qpadm_seed_from_V`) and overwrites them. A singular ALS subsystem leaves that
factor zero, matching the CPU reference.

### `launch_qpadm_weights_chisq_large`

The large-path weight solve and chi-square. Same math as
`launch_qpadm_weights_chisq` with VRAM scratch for its working arrays. Sets `d_status`
to `0` (Ok) or `6` (rank-deficient), and takes the trivial path for `r == 0`. Single
thread, native FP64.

### `launch_qpadm_loo_large_batched`

The parallel large-model leave-one-out re-fits — the throughput-scaled version of the
large jackknife standard error. One thread per `(model, block)` runs the same
large-path ALS and weight-solve math, each seeded from a per-`(model, block)` cuSOLVER
SVD slice (`dAseed`/`dBseed`, precomputed on the host so the seed is bit-identical).
Each thread uses its own slice of a runtime-sized VRAM arena, with stride `dbl_refit`
doubles and `int_refit` ints per re-fit and the layout
`xmat[m] | A[nl*r] | B[r*nr] | union[scratch]`. It writes the unscaled normalized
weights to `dWmat[(model*nb + b)*nl + i]` (all zeros when the status is nonzero). This
replaces the host's serial per-block loop with `nb` concurrent re-fits while keeping
the standard error bit-identical — only the parallelism changes, and the host-side
variance reduction is unchanged. Native FP64.

---

## 10. Model-batched (rotation) kernels

These lift the single-model launchers to a **model-batch** axis. Each kernel adds a
`model` grid dimension and reads or writes a per-model slice of a strided arena, so one
batched dispatch fits a whole bucket of same-shape `(nl, nr, r)` models at once —
instead of a host loop calling a single-model kernel repeatedly. The math, operation
order, and native FP64 are unchanged; the model axis is the only addition. They serve
the small bit-parity bucket (`nl <= 5`, `nr <= 10`, `r <= 4`), which is the common case
for a large model-space rotation; the oversized tail still runs the per-model path. The
covariance `Q` (a strided-batched GEMM) and its inverse (batched cuSOLVER Cholesky) are
formed elsewhere; these kernels are the element-wise, small-reduction, and
single-thread-per-model steps.

### `launch_assemble_f4_gather_models_batched`

The f4 gather, batched over models. The grid runs over `(k + m*b, model)`. It reads the
shared resident f2 tensor with **per-model** index arenas: `d_left_arena` (`[B][nl+1]`
row-major) and `d_right_arena` (`[B][nr+1]`). It writes the strided output arena `dX`,
where model `model`'s slice is `dX + model*(m*nb)` in the `k + m*b` layout. The survivor
list `d_surv` is shared across all models. Native FP64.

### `launch_f4_loo_total_models_batched`

The leave-one-out / total / total-line reduction, batched over models. One thread per
`(k, model)` reduces over the `nb` blocks of that model's `dX` slice, reproducing the
FP64 operation order, and writes per-model slices of `dLoo[m*nb]`, `dTotal[m]`, and
`dTotLine[m]`.

### `launch_f4_xtau_models_batched`

The pseudo-value kernel, batched over models. One thread per `(k + m*b, model)` reads
this model's `dLoo` / `dEst` (= `dTotal`) / `dTotLine` slices and writes its `dXtau`
slice in the `k + m*b` column-major layout the batched symmetric matrix-multiply
consumes.

### `launch_add_fudge_diag_models_batched`

The per-model fudge-diagonal step. For each model it computes the trace of that model's
`Q` in-thread, then writes `Qf.diag += fudge * tr`. `dQ` is the input strided arena
(`m*m` per model, column-major); `dQf` is the output copy. One block per model (or a
grid-stride loop). Native FP64.

### `launch_fill_identity_batched`

Builds a batched identity right-hand-side arena `[m*m*B]` — one `m × m` column-major
identity per model — for the batched cuSOLVER Cholesky inverse. One thread per
`(element, model)`. Native FP64.

### `launch_qpadm_fit_models_batched`

The heart of the batched path: **one thread per model** runs the *whole* small-path
qpAdm fit for its model. It reads per-model slices of `dTotal` (the point estimate, `m`
per model), `dQinv` (`m*m` per model, the batched inverse), `dLoo` (`m*nb` per model),
and the shared `d_block_sizes`. It transliterates the full single-model pipeline: seed,
ALS, constrained weight solve, chi-square; the rank sweep over `r = 0 .. rmax`; the
per-block leave-one-out re-fits that give the standard-error diagonal; and the
"pop-drop" reduced fits that drop one source population at a time. Per model it writes:

| Output | Shape | Meaning |
|---|---|---|
| `d_weight` | `B*nl` | Full-rank weights, summing to 1 (zeros if rank-deficient). |
| `d_se` | `B*nl` | Jackknife standard-error diagonal (native-FP64 sample variance over `nb-1`). |
| `d_chisq` | `B` | Chi-square at the fitted rank. |
| `d_status` | `B` | `0` Ok, `6` rank-deficient. |
| `d_rank_chisq` | `B*(rmax+1)` | Chi-square for each rank `r = 0 .. rmax`. |
| `d_pop_chisq` | `B*(nl+1)` | Pop-drop chi-squares: the full row, then each single-source drop. |
| `d_pop_wfull` | `B*nl` | Full-model pop-drop-row weights at rank `nl-1` (the feasibility source). |

The host assembles the final result fields — the p-value from the chi-square, the
nested rank-drop table, and the pop-drop pattern strings and feasibility — from these
outputs, exactly as the single-model path does. `rmax = min(nl, nr) - 1`; the reported
fit rank `r_fit` defaults to `nl-1`. Because there are many threads, each thread's
local working set is bounded to the small envelope, and the host's bucketer guarantees
`nl <= 5`, `nr <= 10`, `r <= 4` before this launches. Native FP64.

### `launch_qpadm_loo_models_batched`

The per-block leave-one-out re-fits, batched across `(model, block)` — the
throughput-scaled standard-error source. One thread per `(model, b)` fits that block's
leave-one-out weights and writes the **scaled** weight vector (multiplied by
`s = (nb-1)/sqrt(nb)`) to `dWmat[model*nb*nl + b*nl + i]`. This runs `B·nb` threads in
parallel instead of a per-model serial block loop. Native FP64.

### `launch_qpadm_se_from_wmat_batched`

Turns the replicate weight matrix into the standard-error diagonal, batched across
`(model, weight column)`. One thread per `(model, i)`:

```
SE[i] = sqrt( Σ_b (wmat[b,i] - mean_i)² / (nb - 1) )
```

The reduction runs in a **fixed** order with no atomics, so the result is bit-identical
regardless of how many GPUs are used. Writes `d_se[B*nl]`. Native FP64.

### `launch_qpadm_gather_loo_qinv`

A pure data-movement compaction used by the two-pass standard-error path. It compacts
the `dLoo` (`m*nb`) and `dQinv` (`m*m`) slices of the `n_surv` survivor positions listed
in `d_surv` (ascending) into dense destination arenas `dLooDst` / `dQinvDst`, in one
launch rather than a per-survivor device-to-device copy loop. It is parity-neutral — a
survivor's compacted slice is byte-identical to its slice in the full arena — so the
standard-error kernels run unchanged over `n_models = n_surv`. It only copies doubles;
native FP64.

---

## 11. All-combinations sweep kernels

These kernels support sweeping every combination of populations (for example every
group of 4 out of many). They exist to remove a host-side bottleneck: enumerating the
combinations on the CPU and copying each chunk down to the device was the slow part, so
the enumeration is done on the device instead.

### `launch_sweep_unrank_quartets`

On-device combinatorial **unranking** of a chunk of quartets. One thread per local item
`t` in `[0, C)` computes the global colexicographic rank `c0 + t` and writes the
sorted quad `dQuartets[4*t .. 4*t+3]` = `(c0 < c1 < c2 < c3)` — exactly the flat `4*C`
layout the quartet gather (section 3) reads as a device pointer. This replaces copying a
host-enumerated quartet list down for each chunk; there is no host enumeration at all.
Integer math. `d_subset` restricts the sweep to a chosen index subset, and `range` is
its size.

### `launch_sweep_unrank_triples`

The f3 sibling of the quartet unrank — the same on-device unranking for chunks of
triples (`k = 3`).

### `launch_sweep_zfilter`

The significance (`|z|`) filter for the sweep. One thread per item `k` computes
`est = dXtotal[k]`, `se = sqrt(dVar[k])`, and `z = est/se` (written to `dEst` / `dSe` /
`dZ`), and sets the survivor flag `d_flags[k]` (a `0`/`1` byte). In mode `0` the flag is
`|z| >= min_z` (the minimum-z filter, applied on the device); in mode `1` it keeps
everything (top-K or all-results selection, which is ranked on the host over the
compacted set). A NaN `z` (from a degenerate variance) flags `0` under the minimum-z
filter, because NaN comparisons are false. Native FP64.

### `launch_sweep_deinterleave_keys`

Splits the interleaved key array `d_items[k*t + c]` into `k` separate contiguous integer
columns `d_c0 .. d_c3`, so each column can be stream-compacted by the CUB "Flagged"
primitive with the same survivor flags (CUB Flagged takes one input column per call).
`k <= 4`; for `k = 3` the fourth column is set to zero.

---

## 12. Bounded device top-K reservoir

When a sweep asks only for the top-K most significant results, keeping every survivor in
an unbounded host vector can exhaust memory. This set of kernels keeps a fixed-size,
device-resident reservoir with a *rising* threshold, so memory stays bounded no matter
how many combinations are computed.

### `launch_sweep_zfilter_tau`

Like `launch_sweep_zfilter`, but the cutoff is read from `d_tau[0]` — the live top-K
threshold, which is raised each time the reservoir is compacted — instead of a host
constant. The `|z|` sort key is written into `dAbsZ`, and the flag is
`d_flags[t] = (|z| > d_tau[0])`. A NaN `z` (degenerate variance) flags `0`, so it never
enters the top-K. Native FP64.

### `launch_sweep_topk_iota`

Fills `d_idx[0..n) = 0..n-1` — the value array (a starting permutation) for CUB's
descending sort-by-key.

### `launch_sweep_topk_gather`

Reorders the reservoir by a permutation: `out[r] = in[d_perm[r]]` for `r` in `[0, m)`,
applied across every column of the reservoir (estimate, standard error, z, absolute z,
and the four key columns) with the *same* permutation, so each row's tuple stays intact.
This puts the reservoir into descending `|z|` order (after which it is truncated to K).
Out of place.

### `launch_sweep_topk_raise_tau`

Raises the rising threshold to the new K-th-largest `|z|`:
`d_tau[0] = max(d_tau[0], d_sorted_absz[K-1])` when the mode is `1` (top-K) and `K > 0`.
It is monotone — it never lowers the threshold — and it is a no-op in minimum-z mode
(mode `0`, where `tau` stays pinned at the `min_z` floor). A single device thread.

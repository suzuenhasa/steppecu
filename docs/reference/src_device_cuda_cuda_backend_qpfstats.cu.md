# `cuda_backend_qpfstats.cu` reference

## 1. Purpose

This file implements the GPU side of qpfstats — the step that turns raw
genome-wide f-statistics into smoothed, bias-corrected estimates with error bars.
It is one translation unit split out of the larger CUDA backend; it holds nothing
but the qpfstats solve family, and the bodies were moved here verbatim from the
combined backend file, so no math, precision choice, or ordering changed in the
move.

There are four functions:

- `qpfstats_smooth` — the core solve, taking the already-prepared inputs (the
  design matrix, the per-block values, and the global values) as plain host arrays.
- `qpfstats_blocks_smooth` (two overloads) — the full pipeline from genotype data
  to smoothed output. One overload uploads the genotype/variance arrays from host
  memory; the other borrows them where they already sit in GPU memory.
- `qpfstats_blocks_smooth_device` — the shared engine both overloads call once the
  genotype and variance data are resident on the GPU.

The whole file reproduces the qpfstats regression math[^at2], but
reshaped so it runs as a handful of large GPU operations instead of the
CPU-bound, one-block-at-a-time loop the reference uses. That reshaping — solving
every jackknife block's column at once through a single shared factorization — is
the central idea and is described in section 3.

## 2. The smoothing regression this solves

qpfstats fits a set of f-statistics jointly rather than one at a time. The inputs
that arrive at the solve are:

- **`x`** — a design matrix, shape `npopcomb × npairs`, stored column-major. Each
  row is one population combination; each column is one of the `npairs` basis
  quantities being fit. This matrix is the same for every block, which is the fact
  the whole design exploits.
- **`ymat`** — the per-block source values, shape `npopcomb × n_block`. Column `b`
  holds the statistic for every combination with jackknife block `b` left out.
- **`y`** — the global source values, one per combination (`npopcomb` of them),
  computed with no block left out.
- **`ridge`** — a small number added to the diagonal of the normal-equations
  matrix (Tikhonov regularization) so the solve stays well-conditioned.

The job is a regularized least-squares fit of `x` against each of those value
columns. Solving it for every block column gives the per-block fitted
coefficients `b` (shape `npairs × n_block`); solving it for the global column
gives `bglob` (shape `npairs`). Together those feed the block-jackknife variance
and the centering correction (section 8).

## 3. The shared-factor batched solve

The reference implementation loops over jackknife blocks and does a separate
regression for each one on the CPU. Because the design matrix `x` is identical
across all blocks, the normal-equations matrix

```
A_shared = x'·x + ridge·I
```

is also identical across all blocks. It only has to be built and factored **once**.
Every block column, plus the global column, is then just another right-hand side
against that one factorization. That turns a per-block loop into five GPU
operations over the whole problem at once:

1. **Build the matrix.** `A_shared = x'·x` via one symmetric rank-k update
   (`cublasDsyrk`, lower triangle). Because `x` is column-major with shape
   `npopcomb × npairs`, the product `x'·x` is the transpose-times-normal form:
   `syrk` with the transpose operation, `n = npairs`, `k = npopcomb`, leading
   dimension `npopcomb`. Then the lower triangle is mirrored to the full matrix and
   `ridge` is added to the diagonal.
2. **Build all right-hand sides.** The RHS source is the block values and the
   global values laid side by side: columns `0 .. n_block-1` are `ymat`, and the
   last column is `y`, giving a `npopcomb × (n_block+1)` block. One matrix multiply
   (`cublasDgemm`, transpose-times-normal) forms `RHS = x'·RhsSrc`, shape
   `npairs × (n_block+1)`.
3. **Factor once.** `L = chol(A_shared)` via one Cholesky factorization
   (`cusolverDnDpotrf`, lower). If the matrix is not positive definite the routine
   reports it and the call returns a "non-SPD covariance" status.
4. **Solve every column together.** With `A_shared = L·Lᵀ`, the system `A·B = RHS`
   is two triangular solves: a forward solve `L·Z = RHS` (lower, no transpose) and
   a back solve `Lᵀ·B = Z` (lower, transpose). Each solve handles **all**
   `n_block+1 columns in a single call`, so the entire block axis is covered in
   two operations with no host loop.
5. **Split the answer.** The solution `B` overwrites the RHS buffer. Columns
   `0 .. n_block-1` are the per-block coefficients `b`; the last column is the
   global coefficients `bglob`.

A deliberate choice in step 4: the code uses a **pair of triangular solves**
(`cublasDtrsm`) rather than the batched Cholesky-solve routine, because that
routine only handles a single right-hand-side column at a time. Two triangular
solves cover all columns at once, which is what makes the whole-block-axis batching
possible.

## 4. Precision: emulated for the multiplies, native for the factorization

The two matrix-multiply steps (building `A_shared` and building the right-hand
sides) run in **emulated double precision** by default — a faster arithmetic that
is essentially as accurate as native double precision for well-conditioned
multiplies. This is engaged the same way as elsewhere in the backend: a scoped
math-mode change plus a call that turns on the requested precision, both undone
automatically when the surrounding block ends. The scope is closed before the
Cholesky runs so the emulated mode never leaks into the factorization.

The **factorization and the triangular solves run in native double precision** by
default. These are the numerically delicate, potentially ill-conditioned parts of
the computation, so they get the exact reference arithmetic rather than the
emulated approximation. That default is read from a shared setting
(`solve_precision_`, which starts at native double precision); it can be promoted
to the faster mode by a separate setter when a run has been validated to tolerate
it. The precision is set through a local scope so it never leaks into the next
operation.

The takeaway: the heavy multiplies get the fast path; the sensitive linear algebra
gets the safe path. This split matches how the rest of the backend treats its
matrix work.

## 5. NaN handling and the partial-NaN downdate fallback

Missing data shows up as NaN entries in the source columns. Before the solve, a
kernel zeroes every NaN entry and, at the same time, counts how many combinations
were NaN in each column. That count is copied back to the host and drives two
cases:

- **An all-NaN column** (every combination missing — for example a jackknife block
  with no usable SNPs) becomes an all-zero right-hand side after zeroing. The
  shared solve then returns `b = 0` for that column automatically, because
  `A·b = 0`. This exactly reproduces the reference's "all-missing block gives zero
  coefficients" rule, with no special-casing.
- **A partial-NaN column** (some but not all combinations missing) is not correct
  under simple zeroing, because the zeroed rows still implicitly contributed to
  `A_shared`. Those columns need the matrix itself adjusted: `A_b = A_shared -
  x[nan]'·x[nan]`, subtracting out the outer products of exactly the missing rows
  ("downdating"). Only those few columns are re-solved, on the host, with a small
  dense solver, matching the reference's generic-solve branch bit-for-bit.

The partial-NaN path is a genuinely cold fallback. On the real reference dataset
no block is partial-NaN — the one empty block is all-NaN and handled by the
zero-right-hand-side case above — so this loop runs **zero iterations in
production**. The host-side recompute (rebuilding `A_shared` in extended precision,
downdating per missing row, and solving each affected column) exists only for
correctness on data that does hit the partial case; the fast path never pays for
it. In the fused pipeline (section 7) the same fallback additionally copies the
source values back from the GPU only when it actually fires, and then recomputes
the centering correction on the corrected coefficients.

## 6. The three public entry points

- **`qpfstats_smooth`** takes the design matrix `x`, the per-block values `ymat`,
  and the global values `y` as ready-made host arrays. It uploads them, runs the
  solve of section 3, applies the NaN handling of section 5, and returns `b` and
  `bglob`. Use this when the caller has already produced the per-block and global
  values (for example, the CPU reference path against which the GPU is checked).

- **`qpfstats_blocks_smooth` (host-pointer overload)** takes the raw
  allele-frequency and variance arrays in host memory, uploads them into GPU
  buffers, and calls the fused core. This is the entry for callers that don't
  already have the data on the GPU.

- **`qpfstats_blocks_smooth` (resident overload)** takes a decode result whose
  allele-frequency and variance arrays are **already in GPU memory** and calls the
  fused core directly, with no upload. This is the fast door for the normal
  pipeline, where the decode step already left the data resident. Both overloads
  produce byte-identical results — they differ only in whether they copy the
  genotype data to the GPU first.

Each entry point guards against degenerate inputs (zero populations, pairs,
blocks, or an empty quad list) by returning an OK result with empty outputs
instead of touching the GPU.

## 7. The fused device pipeline (the blocks-smooth core)

`qpfstats_blocks_smooth_device` is the single shared engine behind both
`qpfstats_blocks_smooth` overloads. Its inputs are already-resident device
pointers to the allele-frequency array `dQ` and variance array `dV` (both shape
`P × M`), the per-SNP block assignment, the population quadruples, the design
matrix `x`, and the jackknife block sizes. It never frees `dQ`/`dV` — those are
borrowed, owned by whoever uploaded or decoded them.

The whole computation stays on the GPU from genotypes to final small outputs, in
four stages:

1. **Numerator reduce.** A per-block reduction computes the f-statistic numerator
   sums and the SNP counts for every population combination and every block, using
   the same block layout the rest of the backend uses (the per-block contiguous SNP
   ranges, rebuilt from the block-id array). These sums stay resident.

2. **Block jackknife.** An on-device kernel turns the block sums into the
   leave-one-out per-block values `ymat` and the global values `y`, writing them
   straight into one combined right-hand-side buffer (block columns first, the
   global column last) laid out exactly as the solve expects. This replaces what
   used to be a large host loop and runs in native double precision because the
   difference-of-sums here is cancellation-prone.

3. **The shared solve.** The identical five-step solve of section 3 runs directly
   on that buffer in GPU memory — no copy of the per-block or global values back to
   the host and up again. The solution stays resident.

4. **Recenter.** A per-pair centering correction (section 8) is computed on-device
   from the resident solution.

Only three small arrays cross back to the host: the per-block coefficients `b`, the
global coefficients `bglob`, and the recenter shift. The large intermediate
arrays — the numerator and count sums, which can be well over a gigabyte — never
leave the GPU. That eliminated copy is the reason this path exists.

## 8. The recenter jackknife

After the solve, each of the `npairs` fitted quantities gets a centering
correction. For one pair, the correction is the global coefficient minus the
block-jackknife point estimate of that pair's per-block coefficient series
(weighted by block sizes). This is the standard jackknife bias/centering step: it
lines up the global estimate with what the per-block estimates imply, so the
reported value and its jackknife error bar are consistent.

On the fast path this is one on-device kernel reading the resident solved
coefficients and the block sizes. On the partial-NaN fallback path (section 5),
where some coefficients were re-solved on the host, the correction is recomputed on
the host over the corrected series using the same reference estimator, so the two
paths agree.

## 9. The pooled Cholesky workspace

The Cholesky factorization needs a scratch buffer whose size the routine reports at
call time. Rather than allocating and freeing that buffer on every call, this file
keeps one **persistent, grow-only** workspace (a class member) and enlarges it only
when a call needs more than it currently holds; it never shrinks. Feeding the same
bytes to the factorization keeps the result bit-identical to a fresh allocation, so
the pooling is purely a speed and allocation-churn improvement with no effect on the
numbers.

One ordering detail matters: the required workspace size is queried **after** the
factorization's precision mode is engaged, so the size reflects the mode actually
in effect. Querying it before the mode was set could return a size for the wrong
mode.

## 10. Shared class state and how it's used

The solve family keeps its mutable state as members of the backend class rather
than as locals, and this file is a reader of most of it:

- **`solve_precision_`** — the precision used for the Cholesky and triangular
  solves (section 4). Read here; written by a separate setter elsewhere. Defaults
  to native double precision.
- **`blas_` / `solver_`** — the handles for the matrix-multiply, triangular-solve,
  and factorization libraries. Every GPU linear-algebra call in this file goes
  through them.
- **`solver_work_`** — the pooled factorization workspace of section 9.

Keeping these on the class is what lets the workspace persist across calls and lets
the precision setting be configured once and read by every solve.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

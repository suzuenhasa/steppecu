# `qpfstats_kernel.cu` reference

## 1. Purpose

`src/device/cuda/qpfstats_kernel.cu` holds two tiny GPU kernels that *prepare*
the qpfstats smoothing solve. They are not where the heavy work happens: they are
not the f2 matrix multiplication and not the f2 cache. They are the small setup
steps that arrange the data just before the backend runs a batched least-squares
solve that smooths the genotype-path joint f2 estimates.

The file owns exactly two kernels and their launch wrappers:

1. **Zero the not-a-number entries.** One kernel walks the right-hand-side matrix,
   replaces every non-finite value (a NaN or an infinity) with zero *in place*, and
   counts, for each genome block, how many of its rows were non-finite.
2. **Add the ridge to the diagonal.** The other kernel adds a small constant to
   every diagonal entry of the solve's coefficient matrix — a standard
   regularization step that keeps the least-squares solve well-behaved.

Everything here is private to the GPU backend. The kernel bodies and their
`<<<>>>` launches live only in this file; the rest of steppe reaches them through
two narrow wrapper functions declared in the companion header
`qpfstats_kernel.cuh` and never includes this body.

---

## 2. The smoothing solve these kernels feed

The qpfstats smoother stabilizes many f2 estimates at once by solving a single
least-squares system whose coefficient matrix is shared across all the columns it
has to solve for. Two facts about that solve explain why these two kernels exist:

- **The coefficient matrix is `A = x'x + ridge·I`.** It is the normal-equations
  matrix `x'x` (the design matrix multiplied by its own transpose) plus a small
  constant on the diagonal. That constant is the *ridge*, and adding it is the
  second kernel's whole job (section 5).
- **One factorization solves every column.** The matrix `A` is factored once, and
  then *all* the right-hand-side columns are solved against that single factored
  form in one batched pair of triangular solves. Because the factorization is
  shared across every column, this is called the shared-factor solve. The
  right-hand-side matrix is `ymat`, and getting `ymat` ready — cleaning out its
  non-finite entries — is the first kernel's whole job (sections 3 and 4).

Both kernels run on the GPU, on the same stream as the solve that consumes their
output, so no data has to leave the device between the prep and the solve.

---

## 3. The not-a-number-zeroing kernel and the all-missing shortcut

`qpfstats_zero_nan_ymat_kernel` sweeps the right-hand-side matrix `ymat` and does
two things at each cell: if the value is not finite, it overwrites it with zero,
and it adds one to a per-block counter.

### The `ymat` layout

`ymat` is stored **column-major** with shape `[npopcomb × n_block]`: `npopcomb`
rows (one per population combination) and `n_block` columns (one per jackknife
genome block). The cell for combination `c` in block `b` lives at index
`c + npopcomb*b`. The kernel derives a cell's block from its flat index as
`b = cell / npopcomb`.

### Why zeroing a column is enough

The reason a non-finite value can simply be set to zero — rather than needing any
special case in the solve — is a property of the shared-factor solve itself.
Solving `A·b = 0` gives `b = 0`. So if a whole column of `ymat` is non-finite and
therefore becomes all zeros, that column flows through the exact same shared
factorization as every other column and naturally comes out as a zero result. No
branch, no separate code path. This exactly reproduces the rule that an
all-missing block yields a zero estimate[^at2]: the reference sets `ymat_chunk[nan] = 0`
and records a per-block missing count `k_i = sum(nan_i)`, and this kernel produces
the same zeroed matrix and the same per-block counts.

### The per-block count

The kernel atomically increments `nan_per_block[b]` for every non-finite cell it
zeros. That array must be zeroed before the kernel runs; the launch wrapper does
this with an asynchronous memory-set (section 6). The resulting counts are what
the solve uses to decide, block by block, how to handle missing data — described
next.

### Grid-stride coverage

The kernel walks its cells with a grid-stride loop: each thread starts at its
global index and steps forward by the total number of threads until it runs off
the end of the matrix. This makes the kernel cover any input size no matter how
the launch grid is sized — there is no hidden assumption that the grid exactly
spans the matrix — so the wrapper is free to pick a fixed, simple grid size.

---

## 4. The three-case missing-data protocol

The per-block count the first kernel produces drives a three-way decision in the
solve. For a block whose count is `k` out of `npopcomb` rows:

| Count `k` | Meaning | How the solve handles it |
|---|---|---|
| `k == npopcomb` | Every row is missing (all-NaN block). | Its `ymat` column is now all zeros, so the shared solve returns exactly zero for it — the all-missing rule, realized for free by the zeroed column (section 3). |
| `k == 0` | No row is missing. | The block takes the plain shared-factor path with no adjustment to the coefficient matrix. |
| `0 < k < npopcomb` | Some but not all rows are missing. | The block needs a genuine correction: the missing rows must be removed from the coefficient matrix (a downdate) and the block re-solved with that adjusted matrix. Because these blocks are few, that downdate-and-re-solve is done on the host side for just those blocks. |

So the counts sort every block into one of three buckets, and only the mixed
middle bucket needs the more expensive per-block work; the all-missing and
no-missing buckets are both served by the single shared solve.

---

## 5. The ridge-diagonal kernel

`qpfstats_add_ridge_diag_kernel` adds the ridge constant to the diagonal of the
coefficient matrix: it computes `A[i + n*i] += ridge` for each `i`. `A` is the
column-major `[n × n]` matrix, so `i + n*i` is its `i`-th diagonal entry. The
kernel launches one thread per diagonal entry and returns immediately for any
thread whose index is past `n`.

Adding a small constant to the diagonal turns the plain normal-equations matrix
`x'x` into `x'x + ridge·I`. This is ridge regularization: it keeps the solve
stable and well-defined even when `x'x` is close to singular, which is what lets
the shared factorization succeed and be reused across every column.

---

## 6. Launch wrappers and geometry

Host code never issues a raw kernel launch; two wrapper functions do, and they are
the only symbols the rest of steppe can see from this file.

### `launch_qpfstats_zero_nan_ymat`

- Returns immediately (does nothing) when the matrix is empty (`npopcomb * n_block
  <= 0`).
- Marks a coarse profiling range around the work so the smoother-prep phase shows
  up as one labeled span in a trace. The marker compiles to nothing unless
  profiling is enabled at build time.
- Zeroes the whole `nan_per_block` array first, with an asynchronous memory-set on
  the same stream, so the kernel's atomic increments start from a clean slate.
- Launches the zeroing kernel with `kZeroNanThreads = 256` threads per block and
  enough blocks to cover the matrix (`ceil(total / 256)`). The grid-stride loop
  (section 3) makes this fixed geometry correct for any matrix size.

### `launch_qpfstats_add_ridge_diag`

- Returns immediately when `n <= 0`.
- Launches the ridge kernel with `kRidgeThreads = 64` threads per block and
  `ceil(n / 64)` blocks — one thread per diagonal entry.

Both wrappers check for launch errors immediately after issuing the kernel, and
both take the CUDA stream to run on, so the prep stays ordered with the solve that
consumes it.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

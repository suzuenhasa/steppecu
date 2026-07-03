# `qpfstats_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/qpfstats_kernel.cuh` is the narrow, public-facing seam through
which the rest of steppe reaches two tiny GPU kernels that prepare the qpfstats
"smoothing solve." The kernels themselves — the actual GPU code and the `<<<>>>`
launch syntax — live privately in `qpfstats_kernel.cu`. This header exposes only
two plain C++ wrapper functions. Callers include this header and call the
wrappers; they never see or include the kernel bodies.

Keeping the seam this thin matters because CUDA kernel code can only be compiled
by the GPU compiler. By hiding the kernel bodies behind ordinary function
declarations, the backend can call them from regular C++ translation units
without every caller having to be a CUDA file.

What the two kernels are *for*: the qpfstats feature fits a smoothed set of
f-statistics by solving a batched least-squares system that shares one factored
matrix across every genome block. Before that solve can run, two small
preparation steps are needed — cleaning up undefined (non-finite) values in the
right-hand-side data, and adding a small regularization term to the shared
matrix. Each wrapper is one of those two steps. Neither kernel does the heavy
matrix-multiply or the solve itself; they only groom the inputs.

Both functions take a `cudaStream_t` and do their work asynchronously on that
stream, and both operate **in place** — they overwrite the buffer you hand them
rather than returning a new one.

---

## 2. `launch_qpfstats_zero_nan_ymat` — clean up non-finite values and count them per block

```cpp
void launch_qpfstats_zero_nan_ymat(double* d_ymat, int npopcomb, int n_block,
                                   int* d_nan_per_block, cudaStream_t stream);
```

This wrapper does two things in a single pass over a matrix on the GPU:

1. **Zeroes every non-finite entry in place.** It scans `d_ymat` and, wherever it
   finds a value that is not finite (a NaN or an infinity), it overwrites that
   entry with `0.0`. The rest of the matrix is left untouched.
2. **Counts, per block, how many rows were non-finite.** For each column (block)
   it writes the number of non-finite entries it found into
   `d_nan_per_block`. The caller does not have to pre-zero this counter buffer —
   the wrapper clears it before the pass begins.

### The matrix layout

`d_ymat` is a **column-major** matrix with `npopcomb` rows and `n_block` columns.
Each row is one population combination; each column is one genome block. In
column-major layout, the entry at row `c`, column `b` sits at index
`c + npopcomb * b`, so an entire column (all combinations for one block) is a
contiguous run of `npopcomb` doubles. The counter buffer `d_nan_per_block` has
one integer per column.

The `n_block` argument names the number of columns, but the kernel treats it
purely as "however many columns the buffer has." In practice the backend hands
it a right-hand-side buffer with one extra column beyond the genome blocks (an
additional whole-dataset column tacked on the end), and passes that larger column
count here. The kernel does not care — it simply cleans and counts every column
it is given.

### Why zero the non-finite entries — and what the per-block count is for

A population combination can be undefined for a given block (for example, a block
with no usable data for that combination), which shows up as a NaN. The
downstream solve reuses one shared, pre-factored matrix for every block at once,
which is only valid when every block's right-hand side is well-formed. Zeroing
the non-finite entries and recording *how many* there were per block is what lets
the solve stay on that fast shared path in the common cases and only fall back to
slower per-block work when it genuinely has to.

The per-block count sorts every block into one of three cases:

| Per-block non-finite count | What it means | How the solve handles it |
|---|---|---|
| `count == npopcomb` (the whole column was non-finite) | The block is entirely undefined. | Zeroing the column makes its right-hand side all zeros, so the shared solve returns a zero result for that block automatically (a system with a zero right-hand side has a zero solution). No special case is needed. |
| `count == 0` (nothing was non-finite) | The block is fully defined. | It rides the shared, no-adjustment fast path — the common case. |
| `0 < count < npopcomb` (a few rows were non-finite) | The block is partly defined. | The shared factor no longer matches this block exactly, so it needs a per-block correction (dropping the non-finite rows from the shared matrix and re-solving just those columns). This is handled on the host afterward, using the counts to find exactly which columns need it. |

### Matches ADMIXTOOLS 2

Zeroing the non-finite entries and taking the per-block count reproduces
ADMIXTOOLS 2's behavior exactly. In particular, the all-non-finite case
(`count == npopcomb`) reproduces ADMIXTOOLS 2's "an entirely undefined block
contributes a zero result" rule, and it does so through the very same shared
solve rather than a separate branch — the zeroed column simply produces a zero
solution.

### Behavior on an empty input

If the matrix has no entries (the product of `npopcomb` and `n_block` is zero or
negative), the wrapper returns immediately and launches nothing.

---

## 3. `launch_qpfstats_add_ridge_diag` — add a regularization term to the diagonal

```cpp
void launch_qpfstats_add_ridge_diag(double* d_A, int n, double ridge, cudaStream_t stream);
```

This wrapper adds the scalar `ridge` to every diagonal entry of the
column-major, `n` by `n` matrix `d_A`, in place. It uses one GPU thread per
diagonal entry.

The matrix `d_A` is the shared "normal-equations" matrix of the smoothing
solve — conceptually `x` transposed times `x`. Adding a constant to its diagonal
turns it into `x'x + ridge·I` (the identity matrix scaled by `ridge`). This is
standard ridge regularization: the small added term keeps the matrix well-behaved
enough to factor and solve reliably, even when the underlying data would
otherwise make the matrix nearly singular. The `ridge` amount is supplied by the
caller.

Because `d_A` is column-major and square with leading dimension `n`, diagonal
entry `i` lives at index `i + n * i`, which is the only element each thread
touches.

### Behavior on an empty input

If `n` is zero or negative the wrapper returns immediately and launches nothing.

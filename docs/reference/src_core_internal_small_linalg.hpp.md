# `small_linalg.hpp` reference

## 1. Purpose

`src/core/internal/small_linalg.hpp` is the small dense linear-algebra kit used by
the reference (CPU) path of the qpAdm model-fit engine. It provides three things a
fit needs and nothing more:

1. **A linear solve** — `solve(A, b)` finds `x` such that `A·x = b`.
2. **An explicit matrix inverse** — `inverse(A)` produces the full inverse matrix
   (used to form the weight matrix the fit calls `Qinv`).
3. **A singular-value decomposition** — `jacobi_svd(A)` factors a matrix into
   `U · diag(S) · Vᵀ`, and supplies the starting guess ("seed") for the fit's
   iterative solver.

These are the *correctness-first* solvers. They favour being obviously right over
being fast, and they exist so the rest of the fit engine can be checked against a
reference whose answers are trusted. The GPU backend does the same math at speed
with the vendor cuSOLVER library; this file is the plain, readable version those
GPU results are validated against.

The header is deliberately self-contained: no CUDA, no third-party libraries,
header-only, and only the C++ standard library. That lets it compile into both the
device target (where the CPU reference lives) and the core fit code without dragging
in any GPU toolkit. Because the matrices involved are small — up to a few hundred
rows for the weight matrix, and only a handful of columns for the SVD — a
textbook O(n³) LU factorization and a one-sided Jacobi SVD are more than fast
enough and easy to verify by eye.

Everything lives in the `steppe::core` namespace.

---

## 2. Storage conventions (column-major)

Every matrix passed through this file is a flat `std::vector<double>` laid out in
**column-major** order. That means the element in row `i`, column `j` of a matrix
with `rows` rows sits at flat index `i + rows·j` — whole columns are contiguous in
memory, and you step by `rows` to move across a row.

This is the same layout BLAS, LAPACK, and cuSOLVER use. Choosing it here means the
CPU reference and the GPU backend agree on how a matrix is stored, so no data has to
be transposed when a value crosses between them.

The addressing rule lives in exactly one place — the `cm_index(i, j, ld)` helper,
which returns `i + ld·j` — so that every access site computes the same flat index
the same way and none of them can drift apart. (`ld`, the "leading dimension", is
just the row count of the matrix being indexed.) See section 8 for the helpers.

---

## 3. LinAlgStatus — the singular-matrix result

`LinAlgStatus` is a one-field struct returned by `solve`, `inverse`, and the
internal factorization:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `ok` | `bool` | `true` | `true` on success. `false` means the matrix was singular — a zero pivot survived partial pivoting, so there is no unique solution or inverse. |

The important contract is what happens on failure: a singular matrix is **not** an
exception. The routine returns `ok == false` and leaves the caller's output
untouched, and the caller is expected to turn that into an ordinary domain-level
error status. Nothing in this file throws.

---

## 4. solve — linear solve A·x = b

```
LinAlgStatus solve(const std::vector<double>& A, int n,
                   const std::vector<double>& b, std::vector<double>& x)
```

Solves the `n×n` system `A·x = b` for the vector `x`.

- `A` is an `n×n` column-major matrix and is **not modified** — the routine works on
  an internal copy.
- `b` is the right-hand side, length `n`.
- `x` receives the solution, length `n`.

The method is LU factorization with partial pivoting (row swaps to keep the
arithmetic stable), followed by forward and back substitution. This mirrors the
behaviour of R's `solve(A, b)` — the reference tool this project reproduces — so the
two agree on the same well-conditioned problems.

If `A` is singular, `solve` returns `ok == false` and leaves `x` untouched.

---

## 5. inverse — explicit matrix inverse

```
LinAlgStatus inverse(const std::vector<double>& A, int n, std::vector<double>& inv)
```

Computes the full explicit inverse of an `n×n` column-major matrix `A`, writing the
result into `inv` (also `n×n`, column-major). This mirrors R's `solve(A)` with a
single argument.

Internally it factors `A` once with LU, then solves the system against each column
of the identity matrix in turn; the resulting columns are the columns of the
inverse. As with `solve`, `A` itself is left unchanged, and a singular `A` returns
`ok == false`.

The fit engine uses this to build the weight matrix it refers to as `Qinv`. Note
that forming an explicit inverse is done here because the reference path wants the
inverse matrix itself, not just the solution to one system.

---

## 6. jacobi_svd and SvdResult — the SVD seed

```
SvdResult jacobi_svd(const std::vector<double>& A, int m, int n)
```

Computes the singular-value decomposition of an `m×n` column-major matrix `A`,
returning matrices `U`, singular values `S`, and matrix `V` such that
`A = U · diag(S) · Vᵀ`. All three outputs are column-major.

### The `SvdResult` struct

| Field | Type | Meaning |
|---|---|---|
| `U` | `vector<double>` | The left singular vectors, `m × k`, column-major. |
| `S` | `vector<double>` | The `k` singular values, in **descending** order. |
| `V` | `vector<double>` | The right singular vectors, `n × k`, column-major. |
| `m` | `int` | Row count of the input. |
| `n` | `int` | Column count of the input. |
| `k` | `int` | `min(m, n)` — the number of singular values / vectors returned. |

### Why the fit needs it

The SVD supplies the **starting guess** for the fit's iterative (alternating
least-squares) solver. Only the leading right singular vectors are used as that
seed — the first `r` columns of `V`, transposed, become the initial estimate the
iteration refines. For the smallest reference case the input is a 2×5 matrix, so the
routine works on 5 columns.

### How it works

The routine uses **one-sided Jacobi**, the simplest SVD method for small matrices
and the easiest to trust as a reference. It repeatedly applies plane rotations to
pairs of columns of a working copy of `A` until those columns are mutually
orthogonal (their pairwise inner products drop to essentially zero). Once that
happens:

- the length (norm) of each final column is a singular value,
- each column normalized to unit length gives a column of `U`,
- the accumulated rotations, applied to the identity, give `V`.

The columns are then sorted so the singular values come out largest-first.

### Determinism

The result is reproducible run to run because the algorithm uses a **fixed sweep
order** over the column pairs and **fixed numeric tolerances** (section 7) to decide
when a rotation is negligible and when the whole process has converged. No random
starting points, no data-dependent ordering.

---

## 7. The SVD tolerance constants

`jacobi_svd` defines three compile-time constants. Two of them are frozen — their
*values* are checked against the reference SVD (both ADMIXTOOLS 2 and cuSOLVER) and
must not change; only their names may be edited. The third is a plain safety cap.

| Constant | Value | Role | Frozen? |
|---|---|---|---|
| `kTol` | `1e-15` | The relative threshold for skipping a rotation. A rotation on a column pair `(p, q)` is skipped when their inner product `|gamma|` is at or below `kTol · sqrt(alpha·beta)` (where `alpha`, `beta` are the two column norms squared). This decides *which* rotations actually fire, so it directly shapes the result. | **Yes — value must stay `1e-15`.** Name-only edits allowed. |
| `kOffDiagTol` | `1e-30` | The convergence floor. After each full sweep, the routine sums the squared off-diagonal inner products; when that total drops below this floor, the columns are considered orthogonal and the loop stops. | **Yes — value must stay `1e-30`.** Name-only edits allowed. |
| `kMaxSweeps` | `60` | A safety cap on the number of sweeps. It is **not** load-bearing: for the small shapes this file handles (a few dozen columns at most), one-sided Jacobi converges in far fewer sweeps, and the cap only exists so a pathological, non-converging input cannot spin forever. | No — this is a guard, not a parity number. |

Both frozen tolerances are the kind of value that would silently change reported
results if edited, which is why they carry the "do not change the magnitude"
warning.

---

## 8. Internal building blocks

The public routines above are built from a set of small shared helpers. Each helper
exists so that a piece of logic used in more than one place is written exactly once
and cannot drift between copies. Their math is deliberately identical to the
open-coded loops they replaced — same operations, same left-to-right accumulation
order — so results are bit-for-bit the same.

| Helper | What it does |
|---|---|
| `cm_index(i, j, ld)` | Returns the flat column-major index `i + ld·j` for element (row `i`, col `j`). The single home of the storage-addressing rule from section 2. All three operands are widened to `std::size_t` before multiplying, so the index cannot overflow. |
| `apply_rotation(x, y, len, c, s)` | Applies one `(c, s)` plane rotation in place to a pair of length-`len` columns: `x' = c·x − s·y`, `y' = s·x + c·y`. Called twice per Jacobi rotation — once for the working columns (length `m`) and once for the accumulating `V` columns (length `n`). |
| `col_sqnorm(x, len)` | The squared Euclidean norm `Σ x[i]²` of a column. Used both for the Jacobi rotation math and for turning the final columns into singular values. |
| `col_dot(x, y, len)` | The inner product `Σ x[i]·y[i]` of two columns — the cross term the Jacobi step drives toward zero. |
| `lu_solve_rhs(...)` | The forward-then-back substitution for one right-hand side through an already-LU-factored matrix. Shared by `solve` (writing into a contiguous solution vector) and `inverse` (writing into one column of the result), which differ only in where the answer is written. |
| `lu_factor(a, n, piv)` | In-place LU factorization with partial pivoting of an `n×n` column-major matrix, recording the row permutation in `piv`. Returns `ok == false` if a pivot is essentially zero (singular). This is the shared engine underneath both `solve` and `inverse`; it is an internal helper, not part of the public surface. |

This is a careful, reference-quality header for small dense linear algebra, but it has a few habits that would make a numerical-C++ reviewer pause before calling it pristine.

## What's genuinely good

- **The architectural intent is exactly right.** Header-only, standard-library-only, CUDA-free, and column-major to match the GPU backend (lines 1–16). This is the correct way to ship a host reference oracle that compiles everywhere.
- **Column-major indexing is centralized.** `cm_index` (lines 40–43) is a single source of truth for the storage convention, and the casts to `std::size_t` before the multiply avoid overflow.
- **Good DRY factoring.** `apply_rotation`, `col_sqnorm`, `col_dot`, and `lu_solve_rhs` each extract one loop that used to be copy-pasted across LU solve/inverse and Jacobi SVD (lines 52–105). That makes the file easier to verify and harder to drift.
- **Sensible error model.** `LinAlgStatus` returns `ok = false` for singular matrices rather than throwing (lines 29–31), matching the project's "caller maps to a domain Status" contract.
- **One-sided Jacobi is a defensible oracle.** For the stated `2×5` golden-path input, a fixed-sweep Jacobi with explicit Givens rotations is the kind of obviously-correct reference a reviewer is happy to see.

## What a senior developer would flag

**Exact-zero pivot test in `lu_factor` (line 123):**

```cpp
if (best == 0.0) return {false};
```

This is the classic floating-point mistake. A matrix can be numerically singular with a pivot of `1e-300`, and this check will blithely divide by it on the next iteration. For a reference solver, a tolerance-based singularity test (e.g., `best <= eps * norm`) is more robust. The `inverse` and `solve` functions inherit this fragility.

**Hand-rolled bubble sort for singular-value ordering (lines 258–265):**

```cpp
for (int i = 0; i < n - 1; ++i)
    for (int j = i + 1; j < n; ++j)
        if (sigma[...] > sigma[...]) { ... swap ... }
```

`n` is small, but a senior C++ reviewer will still wince at reimplementing sort. This is a convention mismatch in a file that otherwise leans on `std::vector`. It should be `std::sort(order.begin(), order.end(), [&](int a, int b){ return sigma[a] > sigma[b]; });`, which is clearer and less error-prone.

**Redundant exact-zero guard in Jacobi (line 234):**

```cpp
if (std::fabs(gamma) <= kTol * std::sqrt(alpha * beta) || gamma == 0.0)
    continue;
```

The `gamma == 0.0` branch is already covered by the tolerance test unless `alpha` or `beta` is zero. It reads like defensive C code that doesn't fully trust the tolerance expression. A small comment explaining the zero-column case would be better than a second condition.

**Silent non-convergence in `jacobi_svd`:**

The loop runs for at most `kMaxSweeps = 60` sweeps and then returns whatever it has (lines 219, 247). There is no `LinAlgStatus`, no `bool converged`, and no way for the caller to know the SVD stopped early. For a "golden" reference path, that is a gap in observability.

**`kOffDiagTol = 1e-30` (line 222):**

Double-precision machine epsilon is around `2e-16`. An off-diagonal tolerance of `1e-30` is far below the noise floor and may as well be zero for most inputs. The comment calls it "parity-frozen" and oracle-diffable, which is fair if the value is load-bearing for regression tests, but it is numerically odd and should probably come with a note about why it is not `epsilon`-scaled.

**No size preconditions on public APIs.** `solve`, `inverse`, and `jacobi_svd` assume the caller passed vectors whose sizes match `n*n` or `m*n`. For a header marked "internal" that may be acceptable, but adding `assert(A.size() == static_cast<std::size_t>(n) * n)` style checks would catch integration mistakes cheaply.

**Comment density.** The comments are accurate and explain *why*, but a few are meta-commentary about DRY and naming-style standards (e.g., lines 33–39, 49–51, 63–64). A senior reviewer may wonder if the code is compensating for itself.

## The "slop" test

**Not slop.** There are no unexplained magic numbers, no copy-pasted loops with stale comments, no raw owning pointers, no `printf`/`FILE*` mixed with streams, and no obviously wrong numerics. The code is organized, documented, and intentional.

## What it actually looks like

This looks like **solid reference code written by someone who understands the admixture-genetics numerics and knows enough C++ to keep it clean.** The LU and Jacobi implementations are textbook-correct, the separation of helpers is thoughtful, and the "CUDA-free, header-only" packaging is the right call.

A senior numerical C++ reviewer would say: "Competent, ship it as the reference oracle — but tighten the singularity test, replace the bubble sort, and expose non-convergence before I'd call it exemplary." A CUDA specialist would have little to say here because the file is deliberately host-side and toolkit-free, which is itself a good sign.

## Verdict

**B+.** Correct, well-structured reference code with a few old-school numerical habits that keep it short of exemplary.

# Review findings — src__core__internal__small_linalg

Files: src/core/internal/small_linalg.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

- [5.1][LOW] src/core/internal/small_linalg.hpp:220 — `if (off < 1e-30) break;` uses a bare unnamed literal for the squared-off-diagonal convergence threshold of the Jacobi sweep, while the adjacent, closely-related per-pair tolerance is the named `constexpr double kTol = 1e-15` (line 179). The inconsistency (one named, one not) makes the convergence criterion harder to reason about (1e-30 ≈ kTol²). Suggested: hoist to a named `constexpr double kOffDiagTol = 1e-30;` (or derive from kTol) beside kTol/kMaxSweeps.

## Group 6 — Naming

- [6.3][LOW] src/core/internal/small_linalg.hpp:235-242 — the descending selection-sort over `order[]` uses `a` and `b` as its index/loop variables (`for (int a = 0; ...) for (int b = a + 1; ...)`), breaking the file's otherwise-consistent index naming convention (`i`, `j`, `k`, `p`, `q` used everywhere else for loop indices). It is also confusing because `a`/`b` are used as matrix-element value temporaries elsewhere in the same function (lines 189-190, 205-206, 213-214), so the reader must context-switch on what `a`/`b` mean. Suggested: rename the sort indices to `i`/`j` (or `p`/`q`) to match the file convention.
- [6.1][LOW] src/core/internal/small_linalg.hpp:239 — the swap temporary in the sort is named `t`, but `t` is also the Jacobi rotation tangent at line 200 within the same SVD function. Two unrelated meanings of a single-letter `t` in one function adds avoidable ambiguity (a third `t` swap temp also appears at line 55 in `lu_factor`). Suggested: use a clearer swap temp name (e.g. `tmp_idx` / `swap`) for the index swap so `t` stays the rotation tangent.

## Group 7 — Duplication

- [7.1][LOW] src/core/internal/small_linalg.hpp:40-43, 81-84, 113-116 — the column-major element-access lambda `at(i,j) = a[i + n*j]` (with the identical triple `static_cast<std::size_t>` cast pattern) is copy-pasted three times across `lu_factor`, `solve`, and `inverse`, differing only by the backing vector name (`a`/`lu`) and the return type (`double&` vs `double`). Suggested: extract a single header-local helper (e.g. a `cm_index(i, j, n)` flat-index fn, or a tiny column-major view template) and reuse it in all three.
- [7.1][LOW] src/core/internal/small_linalg.hpp:89-101 vs 125-139 — the forward-substitution block is essentially identical between `solve` (89-94) and `inverse` (125-130), and the back-substitution block differs only by where the result is written (`x[i]` vs `inv[i + n*col]`) and the source it reads (`x[j]` vs `inv[j + n*col]`). Both implement the same per-RHS LU triangular solve. Suggested: extract a `lu_solve_rhs(at, n, y, out, out_stride/out_offset)` helper invoked once per column by `inverse` and once by `solve`.
- [7.1][LOW] src/core/internal/small_linalg.hpp:204-209 vs 210-217 — the two Jacobi rotation-application loops apply the same `(c, s)` plane rotation to a pair of columns, differing only by the column pointers (`wp/wq` vs `vp/vq`) and the loop length (`m` vs `n`). Suggested: extract `apply_rotation(double* x, double* y, int len, double c, double s)` and call it twice.
- [7.2][LOW] src/core/internal/small_linalg.hpp:188-194 and 226-231 — the squared-column-norm accumulation (`alpha`/`beta` over W columns at 189-192, then the `sigma` squared-norm at 227-230) is the same `sum(w[i]*w[i])` reduction computed in two places; `alpha`/`beta` are exactly the squared column norms that line 230's `sigma` recomputes after the sweeps. Suggested: a small `col_dot(x, y, len)` / `col_sqnorm(x, len)` helper would fold both the (p,q) inner-product accumulation and the final sigma pass.
- [7.3][LOW] src/core/internal/small_linalg.hpp:135-138, 167-170, 254-264 — the flat column-major index expression `i + n*j` spelled out with three explicit `static_cast<std::size_t>` casts recurs verbatim outside the access lambdas (the `inv` writes in `inverse`, the `Vfull` identity init, and the `Vfull`/`out.V`/`out.U` column copies in `jacobi_svd`). Suggested: the same `cm_index(i, j, ld)` helper from 7.1 hoists the cast/index boilerplate here too.

## Group 8 — Comments

- [8.3][LOW] src/core/internal/small_linalg.hpp:220 — the Jacobi sweep-convergence break `if (off < 1e-30) break;` has no comment, and the bare `1e-30` magnitude is non-obvious: it is the threshold on the accumulated *squared* off-diagonal mass (`off += gamma*gamma`), i.e. the squared analogue of the named per-pair tolerance `kTol = 1e-15` (line 179), so `1e-30 ≈ kTol²`. Without a note the reader cannot tell why this constant differs from kTol by a factor of 1e15. Suggested: add a one-line rationale (and/or name it as in Group 5 finding 5.1) stating it is the squared-off-diagonal convergence floor, kTol².
- [8.3][LOW] src/core/internal/small_linalg.hpp:180 — `constexpr int kMaxSweeps = 60;` carries no rationale for the value 60. One-sided Jacobi converges in far fewer sweeps for the small matrices here; 60 is a safety cap, but that intent is undocumented. Suggested: brief comment noting it is a convergence safety cap (typical convergence ≪60 for these small shapes), so the choice is not read as load-bearing.

## Group 9 — Constants & configuration

- [9.2][LOW] src/core/internal/small_linalg.hpp:220 — the Jacobi sweep has three tunable convergence knobs, but only two are surfaced together: `kTol = 1e-15` and `kMaxSweeps = 60` are named `constexpr` at the top of the sweep (lines 179-180), while the third, the squared-off-diagonal convergence floor `if (off < 1e-30) break;`, is a bare literal buried in the inner loop body, away from its siblings. A reader tuning the SVD convergence sees only two of the three knobs at the top. Suggested: hoist the `1e-30` to a named `constexpr` (e.g. `kOffDiagTol`, ≈ kTol²) beside kTol/kMaxSweeps so all three convergence knobs are surfaced in one place rather than one being tangled into the loop logic. (Overlaps Group 5 5.1 / Group 8 8.3 from the magic-number/comment angle; flagged here for the config-surfacing angle.)

## Group 10 — Initialization

No Group 10 issues found.


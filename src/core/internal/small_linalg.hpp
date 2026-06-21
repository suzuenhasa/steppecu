// src/core/internal/small_linalg.hpp
//
// Host-pure, native-FP64 small dense linear algebra for the qpAdm fit engine
// reference path (CpuBackend): the SVD seed, the general `solve(A, b)` (LU with
// partial pivoting, matching R's `solve`), and the explicit matrix inverse used
// for Qinv. CUDA-FREE, header-only, standard-library-only — it compiles into the
// device target (where CpuBackend lives) AND into core/qpadm without dragging in
// any toolkit. These are the M(fit-1) reference solvers (correctness over speed,
// design §2); the CUDA backend uses cuSOLVER (M(fit-4)).
//
// CONVENTIONS. Dense matrices are passed as std::vector<double> in COLUMN-MAJOR
// order: A(i,j) at i + rows*j. This is the BLAS/LAPACK/cuSOLVER layout the GPU
// backend will use, so the reference and the GPU agree on storage with no
// transpose at the seam. Small matrices only (m up to low hundreds for Q; nl×nr
// for the SVD), so an O(n^3) dense LU + one-sided Jacobi SVD are ample and
// obviously-correct (the oracle property).
#ifndef STEPPE_CORE_INTERNAL_SMALL_LINALG_HPP
#define STEPPE_CORE_INTERNAL_SMALL_LINALG_HPP

#include <cmath>
#include <cstddef>
#include <vector>

namespace steppe::core {

/// Result of a linear solve / inverse: `ok == false` ⇒ the matrix was singular
/// (a zero pivot survived partial pivoting) — the caller maps this to a domain
/// Status value, never an exception.
struct LinAlgStatus {
    bool ok = true;
};

/// Flat column-major index of element (row `i`, col `j`) in a matrix with leading
/// dimension `ld`: `i + ld·j`. The single home of the column-major addressing rule
/// (BLAS/LAPACK/cuSOLVER layout, file header CONVENTIONS) so the three `at(i,j)`
/// access lambdas and the open-coded `i + n*j` writes in lu_factor/solve/inverse/
/// jacobi_svd cannot drift on the widening-cast pattern (DRY; NAMING-STYLE-STANDARD
/// §2.5 single-source; findings group-7 7.1/7.3). All three operands widen to
/// std::size_t before the multiply, matching the existing per-site casts exactly.
[[nodiscard]] inline std::size_t cm_index(int i, int j, int ld) noexcept {
    return static_cast<std::size_t>(i) +
           static_cast<std::size_t>(ld) * static_cast<std::size_t>(j);
}

/// Apply one (c, s) plane rotation in place to the column pair (`x`, `y`) of length
/// `len`: `x' = c·x − s·y`, `y' = s·x + c·y`. The single home of the Jacobi
/// rotation-application loop, called twice by jacobi_svd (once for the W columns of
/// length m, once for the V columns of length n) so the two identical loops cannot
/// drift (DRY; NAMING-STYLE-STANDARD §2.5; findings group-7 7.1). Math unchanged:
/// the temporaries `a`/`b` snapshot the pre-rotation values exactly as the inlined
/// loops did, so the result is bit-identical.
inline void apply_rotation(double* x, double* y, int len, double c, double s) noexcept {
    for (int i = 0; i < len; ++i) {
        const double a = x[i];
        const double b = y[i];
        x[i] = c * a - s * b;
        y[i] = s * a + c * b;
    }
}

/// Squared Euclidean norm Σ x[i]² of a length-`len` column. Single home of the
/// squared-column-norm reduction used for the Jacobi `alpha`/`beta` accumulation
/// and the final `sigma` pass (DRY; NAMING-STYLE-STANDARD §2.5; findings group-7
/// 7.2). Same left-to-right accumulation order as the inlined loops, so bit-identical.
[[nodiscard]] inline double col_sqnorm(const double* x, int len) noexcept {
    double acc = 0.0;
    for (int i = 0; i < len; ++i) acc += x[i] * x[i];
    return acc;
}

/// Inner product Σ x[i]·y[i] of two length-`len` columns. Companion to col_sqnorm
/// for the Jacobi (p,q) cross term `gamma` (DRY; NAMING-STYLE-STANDARD §2.5;
/// findings group-7 7.2). Same left-to-right accumulation order, bit-identical.
[[nodiscard]] inline double col_dot(const double* x, const double* y, int len) noexcept {
    double acc = 0.0;
    for (int i = 0; i < len; ++i) acc += x[i] * y[i];
    return acc;
}

/// Solve one RHS through an already-LU-factored matrix accessed via `at(i,j)`
/// (unit-lower L then upper U, both stored in `at`), reading the permuted RHS from
/// `y` (overwritten by the forward-substitution result) and writing the solution to
/// `out[out_offset + i*out_stride]`. The single home of the per-column triangular
/// solve shared by solve (out = x, contiguous) and inverse (out = inv column,
/// offset by n·col) — the forward-substitution block was identical and the
/// back-substitution differed only by the write/read target (DRY;
/// NAMING-STYLE-STANDARD §2.5; findings group-7 7.1). Math unchanged.
template <typename At>
inline void lu_solve_rhs(const At& at, int n, std::vector<double>& y,
                         std::vector<double>& out, std::size_t out_offset,
                         std::size_t out_stride) {
    // Forward substitution (unit lower L).
    for (int i = 0; i < n; ++i) {
        double s = y[static_cast<std::size_t>(i)];
        for (int j = 0; j < i; ++j) s -= at(i, j) * y[static_cast<std::size_t>(j)];
        y[static_cast<std::size_t>(i)] = s;
    }
    // Back substitution (upper U) into out[out_offset + i*out_stride].
    for (int i = n - 1; i >= 0; --i) {
        double s = y[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < n; ++j)
            s -= at(i, j) * out[out_offset + static_cast<std::size_t>(j) * out_stride];
        out[out_offset + static_cast<std::size_t>(i) * out_stride] = s / at(i, i);
    }
}

/// In-place LU factorization with partial pivoting of an n×n COLUMN-MAJOR matrix
/// `a` (overwritten with L\U), recording the row permutation in `piv`. Returns
/// ok=false if a pivot is ~0 (singular). Internal helper for solve/inverse.
[[nodiscard]] inline LinAlgStatus lu_factor(std::vector<double>& a, int n,
                                            std::vector<int>& piv) {
    piv.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) piv[static_cast<std::size_t>(i)] = i;
    const auto at = [&](int i, int j) -> double& { return a[cm_index(i, j, n)]; };
    for (int k = 0; k < n; ++k) {
        // Pivot: largest |a(i,k)| for i >= k.
        int p = k;
        double best = std::fabs(at(k, k));
        for (int i = k + 1; i < n; ++i) {
            const double v = std::fabs(at(i, k));
            if (v > best) { best = v; p = i; }
        }
        if (best == 0.0) return {false};
        if (p != k) {
            for (int j = 0; j < n; ++j) {
                double t = at(k, j); at(k, j) = at(p, j); at(p, j) = t;
            }
            int tp = piv[static_cast<std::size_t>(k)];
            piv[static_cast<std::size_t>(k)] = piv[static_cast<std::size_t>(p)];
            piv[static_cast<std::size_t>(p)] = tp;
        }
        const double inv_pivot = 1.0 / at(k, k);
        for (int i = k + 1; i < n; ++i) {
            const double f = at(i, k) * inv_pivot;
            at(i, k) = f;
            for (int j = k + 1; j < n; ++j) at(i, j) -= f * at(k, j);
        }
    }
    return {true};
}

/// Solve A·x = b for x, A an n×n COLUMN-MAJOR matrix (a copy is factored; A is not
/// modified), b length n. Mirrors R's `solve(A, b)` (LU with partial pivoting).
/// On a singular A returns ok=false and leaves x untouched.
[[nodiscard]] inline LinAlgStatus solve(const std::vector<double>& A, int n,
                                        const std::vector<double>& b,
                                        std::vector<double>& x) {
    std::vector<double> lu = A;
    std::vector<int> piv;
    const LinAlgStatus st = lu_factor(lu, n, piv);
    if (!st.ok) return st;
    const auto at = [&](int i, int j) -> double { return lu[cm_index(i, j, n)]; };
    // Apply the row permutation to b.
    std::vector<double> y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        y[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(piv[static_cast<std::size_t>(i)])];
    // Solve L·U·x = Py: forward then back substitution into the contiguous x.
    x.assign(static_cast<std::size_t>(n), 0.0);
    lu_solve_rhs(at, n, y, x, /*out_offset=*/0, /*out_stride=*/1);
    return {true};
}

/// Explicit inverse of an n×n COLUMN-MAJOR matrix A, written into `inv` (n×n
/// column-major). Mirrors R's `solve(A)`. On a singular A returns ok=false.
[[nodiscard]] inline LinAlgStatus inverse(const std::vector<double>& A, int n,
                                          std::vector<double>& inv) {
    std::vector<double> lu = A;
    std::vector<int> piv;
    const LinAlgStatus st = lu_factor(lu, n, piv);
    if (!st.ok) return st;
    const auto at = [&](int i, int j) -> double { return lu[cm_index(i, j, n)]; };
    inv.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    std::vector<double> y(static_cast<std::size_t>(n));
    for (int col = 0; col < n; ++col) {
        // RHS = e_col permuted by piv.
        for (int i = 0; i < n; ++i) {
            const int src = piv[static_cast<std::size_t>(i)];
            y[static_cast<std::size_t>(i)] = (src == col) ? 1.0 : 0.0;
        }
        // Forward then back substitution into inv(:, col) = inv[col*n + i].
        lu_solve_rhs(at, n, y, inv, /*out_offset=*/cm_index(0, col, n), /*out_stride=*/1);
    }
    return {true};
}

/// One-sided Jacobi SVD of an m×n COLUMN-MAJOR matrix A (m,n small), producing
/// U (m×min(m,n)), singular values S (length min(m,n), descending), and V
/// (n×min(m,n)), all column-major, with A = U·diag(S)·Vᵀ. This is the SEED for
/// the qpAdm ALS (only the leading r right singular vectors V[:,0:r] are used as
/// B = t(V[:,0:r]); design §4 S6). One-sided Jacobi is the obviously-correct
/// small-matrix oracle (it orthogonalizes the columns of A by Givens rotations).
///
/// For the M(fit-1) golden the input is 2×5; one-sided Jacobi operates on the n=5
/// columns. Determinism: a fixed sweep order + a fixed convergence tolerance.
struct SvdResult {
    std::vector<double> U;  ///< m × k column-major
    std::vector<double> S;  ///< length k (descending)
    std::vector<double> V;  ///< n × k column-major
    int m = 0;
    int n = 0;
    int k = 0;  ///< min(m, n)
};

[[nodiscard]] inline SvdResult jacobi_svd(const std::vector<double>& A, int m, int n) {
    // Work on W = A (m×n, column-major) and accumulate right rotations into Vfull
    // (n×n). One-sided Jacobi rotates pairs of COLUMNS of W to mutual orthogonality;
    // the column norms become the singular values and the normalized columns are U.
    std::vector<double> W = A;
    std::vector<double> Vfull(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) Vfull[cm_index(i, i, n)] = 1.0;

    const auto Wcol = [&](int j) -> double* { return &W[cm_index(0, j, m)]; };
    const auto Vcol = [&](int j) -> double* { return &Vfull[cm_index(0, j, n)]; };

    constexpr double kTol = 1e-15;
    // Convergence safety cap, NOT load-bearing: one-sided Jacobi converges in far
    // fewer sweeps for these small (≤ a few dozen) shapes; 60 only bounds a pathological
    // non-converging input so the loop cannot spin.
    constexpr int kMaxSweeps = 60;
    // PARITY-FROZEN (§3.2/§12): the Jacobi off-diagonal convergence floor. Name only —
    // the magnitude 1e-30 is oracle-diffable against the AT2/cuSOLVER SVD and must not change.
    constexpr double kOffDiagTol = 1e-30;
    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double* wp = Wcol(p);
                double* wq = Wcol(q);
                const double alpha = col_sqnorm(wp, m);
                const double beta = col_sqnorm(wq, m);
                const double gamma = col_dot(wp, wq, m);
                off += gamma * gamma;
                if (std::fabs(gamma) <= kTol * std::sqrt(alpha * beta) || gamma == 0.0)
                    continue;
                // Jacobi rotation angle that zeros the (p,q) inner product.
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t = (zeta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = c * t;
                // Apply the (c, s) rotation to the W column pair (length m) and the
                // accumulated V column pair (length n).
                apply_rotation(wp, wq, m, c, s);
                apply_rotation(Vcol(p), Vcol(q), n, c, s);
            }
        }
        if (off < kOffDiagTol) break;
    }

    // Singular values = column norms of W; U columns = W columns normalized.
    const int k = (m < n) ? m : n;
    std::vector<double> sigma(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j)
        sigma[static_cast<std::size_t>(j)] = std::sqrt(col_sqnorm(Wcol(j), m));
    // Sort columns by descending singular value.
    std::vector<int> order(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) order[static_cast<std::size_t>(j)] = j;
    for (int i = 0; i < n - 1; ++i)
        for (int j = i + 1; j < n; ++j)
            if (sigma[static_cast<std::size_t>(order[static_cast<std::size_t>(j)])] >
                sigma[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])]) {
                int tmp_idx = order[static_cast<std::size_t>(i)];
                order[static_cast<std::size_t>(i)] = order[static_cast<std::size_t>(j)];
                order[static_cast<std::size_t>(j)] = tmp_idx;
            }

    SvdResult out;
    out.m = m; out.n = n; out.k = k;
    out.S.resize(static_cast<std::size_t>(k));
    out.U.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(k), 0.0);
    out.V.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(k), 0.0);
    for (int jj = 0; jj < k; ++jj) {
        const int src = order[static_cast<std::size_t>(jj)];
        const double sv = sigma[static_cast<std::size_t>(src)];
        out.S[static_cast<std::size_t>(jj)] = sv;
        // V column.
        const double* vsrc = &Vfull[cm_index(0, src, n)];
        for (int i = 0; i < n; ++i) out.V[cm_index(i, jj, n)] = vsrc[i];
        // U column = W column / sigma (0 if degenerate).
        const double* wsrc = &W[cm_index(0, src, m)];
        if (sv > 0.0) {
            for (int i = 0; i < m; ++i) out.U[cm_index(i, jj, m)] = wsrc[i] / sv;
        }
    }
    return out;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_SMALL_LINALG_HPP

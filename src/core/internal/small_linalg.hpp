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

/// In-place LU factorization with partial pivoting of an n×n COLUMN-MAJOR matrix
/// `a` (overwritten with L\U), recording the row permutation in `piv`. Returns
/// ok=false if a pivot is ~0 (singular). Internal helper for solve/inverse.
[[nodiscard]] inline LinAlgStatus lu_factor(std::vector<double>& a, int n,
                                            std::vector<int>& piv) {
    piv.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) piv[static_cast<std::size_t>(i)] = i;
    const auto at = [&](int i, int j) -> double& {
        return a[static_cast<std::size_t>(i) + static_cast<std::size_t>(n) *
                                                   static_cast<std::size_t>(j)];
    };
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
    const auto at = [&](int i, int j) -> double {
        return lu[static_cast<std::size_t>(i) + static_cast<std::size_t>(n) *
                                                    static_cast<std::size_t>(j)];
    };
    // Apply the row permutation to b.
    std::vector<double> y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        y[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(piv[static_cast<std::size_t>(i)])];
    // Forward substitution (unit lower L).
    for (int i = 0; i < n; ++i) {
        double s = y[static_cast<std::size_t>(i)];
        for (int j = 0; j < i; ++j) s -= at(i, j) * y[static_cast<std::size_t>(j)];
        y[static_cast<std::size_t>(i)] = s;
    }
    // Back substitution (upper U).
    x.assign(static_cast<std::size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double s = y[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < n; ++j) s -= at(i, j) * x[static_cast<std::size_t>(j)];
        x[static_cast<std::size_t>(i)] = s / at(i, i);
    }
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
    const auto at = [&](int i, int j) -> double {
        return lu[static_cast<std::size_t>(i) + static_cast<std::size_t>(n) *
                                                    static_cast<std::size_t>(j)];
    };
    inv.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    std::vector<double> y(static_cast<std::size_t>(n));
    for (int col = 0; col < n; ++col) {
        // RHS = e_col permuted by piv.
        for (int i = 0; i < n; ++i) {
            const int src = piv[static_cast<std::size_t>(i)];
            y[static_cast<std::size_t>(i)] = (src == col) ? 1.0 : 0.0;
        }
        // Forward substitution (unit lower L).
        for (int i = 0; i < n; ++i) {
            double s = y[static_cast<std::size_t>(i)];
            for (int j = 0; j < i; ++j) s -= at(i, j) * y[static_cast<std::size_t>(j)];
            y[static_cast<std::size_t>(i)] = s;
        }
        // Back substitution into inv(:, col).
        for (int i = n - 1; i >= 0; --i) {
            double s = y[static_cast<std::size_t>(i)];
            for (int j = i + 1; j < n; ++j)
                s -= at(i, j) * inv[static_cast<std::size_t>(j) +
                                    static_cast<std::size_t>(n) * static_cast<std::size_t>(col)];
            inv[static_cast<std::size_t>(i) +
                static_cast<std::size_t>(n) * static_cast<std::size_t>(col)] = s / at(i, i);
        }
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
    for (int i = 0; i < n; ++i)
        Vfull[static_cast<std::size_t>(i) + static_cast<std::size_t>(n) *
                                                static_cast<std::size_t>(i)] = 1.0;

    const auto Wcol = [&](int j) -> double* {
        return &W[static_cast<std::size_t>(m) * static_cast<std::size_t>(j)];
    };
    const auto Vcol = [&](int j) -> double* {
        return &Vfull[static_cast<std::size_t>(n) * static_cast<std::size_t>(j)];
    };

    constexpr double kTol = 1e-15;
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
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int i = 0; i < m; ++i) {
                    const double a = wp[i];
                    const double b = wq[i];
                    alpha += a * a;
                    beta += b * b;
                    gamma += a * b;
                }
                off += gamma * gamma;
                if (std::fabs(gamma) <= kTol * std::sqrt(alpha * beta) || gamma == 0.0)
                    continue;
                // Jacobi rotation angle that zeros the (p,q) inner product.
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t = (zeta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = c * t;
                for (int i = 0; i < m; ++i) {
                    const double a = wp[i];
                    const double b = wq[i];
                    wp[i] = c * a - s * b;
                    wq[i] = s * a + c * b;
                }
                double* vp = Vcol(p);
                double* vq = Vcol(q);
                for (int i = 0; i < n; ++i) {
                    const double a = vp[i];
                    const double b = vq[i];
                    vp[i] = c * a - s * b;
                    vq[i] = s * a + c * b;
                }
            }
        }
        if (off < kOffDiagTol) break;
    }

    // Singular values = column norms of W; U columns = W columns normalized.
    const int k = (m < n) ? m : n;
    std::vector<double> sigma(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        double nrm = 0.0;
        const double* wj = Wcol(j);
        for (int i = 0; i < m; ++i) nrm += wj[i] * wj[i];
        sigma[static_cast<std::size_t>(j)] = std::sqrt(nrm);
    }
    // Sort columns by descending singular value.
    std::vector<int> order(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) order[static_cast<std::size_t>(j)] = j;
    for (int a = 0; a < n - 1; ++a)
        for (int b = a + 1; b < n; ++b)
            if (sigma[static_cast<std::size_t>(order[static_cast<std::size_t>(b)])] >
                sigma[static_cast<std::size_t>(order[static_cast<std::size_t>(a)])]) {
                int t = order[static_cast<std::size_t>(a)];
                order[static_cast<std::size_t>(a)] = order[static_cast<std::size_t>(b)];
                order[static_cast<std::size_t>(b)] = t;
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
        const double* vsrc = &Vfull[static_cast<std::size_t>(n) * static_cast<std::size_t>(src)];
        for (int i = 0; i < n; ++i)
            out.V[static_cast<std::size_t>(i) + static_cast<std::size_t>(n) *
                                                    static_cast<std::size_t>(jj)] = vsrc[i];
        // U column = W column / sigma (0 if degenerate).
        const double* wsrc = &W[static_cast<std::size_t>(m) * static_cast<std::size_t>(src)];
        if (sv > 0.0) {
            for (int i = 0; i < m; ++i)
                out.U[static_cast<std::size_t>(i) + static_cast<std::size_t>(m) *
                                                        static_cast<std::size_t>(jj)] =
                    wsrc[i] / sv;
        }
    }
    return out;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_SMALL_LINALG_HPP

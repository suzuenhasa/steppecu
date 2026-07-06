// src/core/internal/small_linalg.hpp
//
// Host-pure, CUDA-free, header-only small dense linear algebra (solve, inverse,
// Jacobi SVD) for the CpuBackend qpAdm-fit reference path. Every matrix is a flat
// std::vector<double> in column-major (BLAS/LAPACK/cuSOLVER) order.
//
// Reference: docs/reference/src_core_internal_small_linalg.hpp.md
#ifndef STEPPE_CORE_INTERNAL_SMALL_LINALG_HPP
#define STEPPE_CORE_INTERNAL_SMALL_LINALG_HPP

#include <cmath>
#include <cstddef>
#include <vector>

#include "core/internal/index_cast.hpp"

namespace steppe::core {

// LinAlgStatus — reference §3
struct LinAlgStatus {
    bool ok = true;
};

// Internal building blocks — reference §8
[[nodiscard]] inline std::size_t cm_index(int i, int j, int ld) noexcept {
    return idx(i) +
           idx(ld) * idx(j);
}

inline void apply_rotation(double* x, double* y, int len, double c, double s) noexcept {
    for (int i = 0; i < len; ++i) {
        const double a = x[i];
        const double b = y[i];
        x[i] = c * a - s * b;
        y[i] = s * a + c * b;
    }
}

[[nodiscard]] inline double col_sqnorm(const double* x, int len) noexcept {
    double acc = 0.0;
    for (int i = 0; i < len; ++i) acc += x[i] * x[i];
    return acc;
}

[[nodiscard]] inline double col_dot(const double* x, const double* y, int len) noexcept {
    double acc = 0.0;
    for (int i = 0; i < len; ++i) acc += x[i] * y[i];
    return acc;
}

template <typename At>
inline void lu_solve_rhs(const At& at, int n, std::vector<double>& y,
                         std::vector<double>& out, std::size_t out_offset,
                         std::size_t out_stride) {
    for (int i = 0; i < n; ++i) {
        double s = y[idx(i)];
        for (int j = 0; j < i; ++j) s -= at(i, j) * y[idx(j)];
        y[idx(i)] = s;
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = y[idx(i)];
        for (int j = i + 1; j < n; ++j)
            s -= at(i, j) * out[out_offset + idx(j) * out_stride];
        out[out_offset + idx(i) * out_stride] = s / at(i, i);
    }
}

[[nodiscard]] inline LinAlgStatus lu_factor(std::vector<double>& a, int n,
                                            std::vector<int>& piv) {
    piv.resize(idx(n));
    for (int i = 0; i < n; ++i) piv[idx(i)] = i;
    const auto at = [&](int i, int j) -> double& { return a[cm_index(i, j, n)]; };
    for (int k = 0; k < n; ++k) {
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
            int tp = piv[idx(k)];
            piv[idx(k)] = piv[idx(p)];
            piv[idx(p)] = tp;
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

// solve — reference §4
[[nodiscard]] inline LinAlgStatus solve(const std::vector<double>& A, int n,
                                        const std::vector<double>& b,
                                        std::vector<double>& x) {
    std::vector<double> lu = A;
    std::vector<int> piv;
    const LinAlgStatus st = lu_factor(lu, n, piv);
    if (!st.ok) return st;
    const auto at = [&](int i, int j) -> double { return lu[cm_index(i, j, n)]; };
    std::vector<double> y(idx(n));
    for (int i = 0; i < n; ++i)
        y[idx(i)] = b[idx(piv[idx(i)])];
    x.assign(idx(n), 0.0);
    lu_solve_rhs(at, n, y, x, /*out_offset=*/0, /*out_stride=*/1);
    return {true};
}

// inverse — reference §5
[[nodiscard]] inline LinAlgStatus inverse(const std::vector<double>& A, int n,
                                          std::vector<double>& inv) {
    std::vector<double> lu = A;
    std::vector<int> piv;
    const LinAlgStatus st = lu_factor(lu, n, piv);
    if (!st.ok) return st;
    const auto at = [&](int i, int j) -> double { return lu[cm_index(i, j, n)]; };
    inv.assign(idx(n) * idx(n), 0.0);
    std::vector<double> y(idx(n));
    for (int col = 0; col < n; ++col) {
        for (int i = 0; i < n; ++i) {
            const int src = piv[idx(i)];
            y[idx(i)] = (src == col) ? 1.0 : 0.0;
        }
        lu_solve_rhs(at, n, y, inv, /*out_offset=*/cm_index(0, col, n), /*out_stride=*/1);
    }
    return {true};
}

// SvdResult / jacobi_svd — reference §6
struct SvdResult {
    std::vector<double> U;
    std::vector<double> S;
    std::vector<double> V;
    int m = 0;
    int n = 0;
    int k = 0;
};

[[nodiscard]] inline SvdResult jacobi_svd(const std::vector<double>& A, int m, int n) {
    std::vector<double> W = A;
    std::vector<double> Vfull(idx(n) * idx(n), 0.0);
    for (int i = 0; i < n; ++i) Vfull[cm_index(i, i, n)] = 1.0;

    const auto Wcol = [&](int j) -> double* { return &W[cm_index(0, j, m)]; };
    const auto Vcol = [&](int j) -> double* { return &Vfull[cm_index(0, j, n)]; };

    // SVD tolerance constants (values parity-frozen) — reference §7
    constexpr double kTol = 1e-15;
    constexpr int kMaxSweeps = 60;
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
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t = (zeta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = c * t;
                apply_rotation(wp, wq, m, c, s);
                apply_rotation(Vcol(p), Vcol(q), n, c, s);
            }
        }
        if (off < kOffDiagTol) break;
    }

    const int k = (m < n) ? m : n;
    std::vector<double> sigma(idx(n));
    for (int j = 0; j < n; ++j)
        sigma[idx(j)] = std::sqrt(col_sqnorm(Wcol(j), m));
    std::vector<int> order(idx(n));
    for (int j = 0; j < n; ++j) order[idx(j)] = j;
    for (int i = 0; i < n - 1; ++i)
        for (int j = i + 1; j < n; ++j)
            if (sigma[idx(order[idx(j)])] >
                sigma[idx(order[idx(i)])]) {
                int tmp_idx = order[idx(i)];
                order[idx(i)] = order[idx(j)];
                order[idx(j)] = tmp_idx;
            }

    SvdResult out;
    out.m = m; out.n = n; out.k = k;
    out.S.resize(idx(k));
    out.U.assign(idx(m) * idx(k), 0.0);
    out.V.assign(idx(n) * idx(k), 0.0);
    for (int jj = 0; jj < k; ++jj) {
        const int src = order[idx(jj)];
        const double sv = sigma[idx(src)];
        out.S[idx(jj)] = sv;
        const double* vsrc = &Vfull[cm_index(0, src, n)];
        for (int i = 0; i < n; ++i) out.V[cm_index(i, jj, n)] = vsrc[i];
        const double* wsrc = &W[cm_index(0, src, m)];
        if (sv > 0.0) {
            for (int i = 0; i < m; ++i) out.U[cm_index(i, jj, m)] = wsrc[i] / sv;
        }
    }
    return out;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_SMALL_LINALG_HPP

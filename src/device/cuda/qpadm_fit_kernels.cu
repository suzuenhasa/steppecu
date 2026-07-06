// src/device/cuda/qpadm_fit_kernels.cu
//
// The custom CUDA kernels for the qpAdm fit and its f-statistics: element-wise
// gathers of the resident f2 tensor, block-jackknife bookkeeping, in-thread
// linear-algebra solvers, and the combinatorial sweep. The heavy dense linear
// algebra (factorizations, inverses, SVDs, GEMMs) lives in cuda_backend.cu. A
// private CUDA TU: the codebase sees only the launch_* wrappers in the .cuh.
//
// Reference: docs/reference/src_device_cuda_qpadm_fit_kernels.cu.md

#include <cuda_runtime.h>

#include "core/internal/f2_estimator.hpp"
#include "core/internal/launch_config.hpp"
#include "core/qpadm/qpadm_bounds.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/qpadm_fit_kernels.cuh"

namespace steppe::device {

namespace {

// Small-path size envelope — reference §3
using core::qpadm::kQpMaxNl;
using core::qpadm::kQpMaxNr;
using core::qpadm::kQpMaxR;
using core::qpadm::kQpMaxM;
using core::qpadm::kQpMaxT;

// Launch-geometry & numeric constants — reference §4
constexpr int kSymTile = 16;
constexpr int kWarpSize = 32;
constexpr int kBlock64 = 64;
constexpr int kBlock128 = 128;
constexpr int kBlock256 = 256;
constexpr double kOffConvergence = 1e-30;
constexpr double kJacobiTol = 1e-15;
constexpr int kJacobiMaxSweeps = 60;

// Device LA helpers: LU factor, solve, one-sided Jacobi SVD — reference §5
__device__ inline bool dev_lu_factor(double* a, int n, int* piv) {
    for (int i = 0; i < n; ++i) piv[i] = i;
    for (int k = 0; k < n; ++k) {
        int p = k;
        double best = fabs(a[k + n * k]);
        for (int i = k + 1; i < n; ++i) {
            const double v = fabs(a[i + n * k]);
            if (v > best) { best = v; p = i; }
        }
        if (best == 0.0) return false;
        if (p != k) {
            for (int j = 0; j < n; ++j) {
                const double t = a[k + n * j]; a[k + n * j] = a[p + n * j]; a[p + n * j] = t;
            }
            const int tp = piv[k]; piv[k] = piv[p]; piv[p] = tp;
        }
        const double inv_pivot = 1.0 / a[k + n * k];
        for (int i = k + 1; i < n; ++i) {
            const double f = a[i + n * k] * inv_pivot;
            a[i + n * k] = f;
            for (int j = k + 1; j < n; ++j) a[i + n * j] -= f * a[k + n * j];
        }
    }
    return true;
}

__device__ inline bool dev_solve(const double* A, int n, const double* b,
                                 double* x, double* lu, int* piv, double* y) {
    for (int i = 0; i < n * n; ++i) lu[i] = A[i];
    if (!dev_lu_factor(lu, n, piv)) return false;
    for (int i = 0; i < n; ++i) y[i] = b[piv[i]];
    for (int i = 0; i < n; ++i) {
        double s = y[i];
        for (int j = 0; j < i; ++j) s -= lu[i + n * j] * y[j];
        y[i] = s;
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = y[i];
        for (int j = i + 1; j < n; ++j) s -= lu[i + n * j] * x[j];
        x[i] = s / lu[i + n * i];
    }
    return true;
}

__device__ inline void dev_jacobi_svd_V(const double* A, int m, int n, int r,
                                        double* W, double* Vfull, double* sigma,
                                        int* order, double* Vout) {
    for (int i = 0; i < n * r; ++i) Vout[i] = 0.0;
    for (int i = 0; i < m * n; ++i) W[i] = A[i];
    for (int i = 0; i < n * n; ++i) Vfull[i] = 0.0;
    for (int i = 0; i < n; ++i) Vfull[i + n * i] = 1.0;
    for (int sweep = 0; sweep < kJacobiMaxSweeps; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double* wp = &W[m * p];
                double* wq = &W[m * q];
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int i = 0; i < m; ++i) {
                    const double a = wp[i]; const double b = wq[i];
                    alpha += a * a; beta += b * b; gamma += a * b;
                }
                off += gamma * gamma;
                if (fabs(gamma) <= kJacobiTol * sqrt(alpha * beta) || gamma == 0.0) continue;
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t = (zeta >= 0.0 ? 1.0 : -1.0) /
                                 (fabs(zeta) + sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / sqrt(1.0 + t * t);
                const double s = c * t;
                for (int i = 0; i < m; ++i) {
                    const double a = wp[i]; const double b = wq[i];
                    wp[i] = c * a - s * b; wq[i] = s * a + c * b;
                }
                double* vp = &Vfull[n * p];
                double* vq = &Vfull[n * q];
                for (int i = 0; i < n; ++i) {
                    const double a = vp[i]; const double b = vq[i];
                    vp[i] = c * a - s * b; vq[i] = s * a + c * b;
                }
            }
        }
        if (off < kOffConvergence) break;
    }
    for (int j = 0; j < n; ++j) {
        double nrm = 0.0;
        const double* wj = &W[m * j];
        for (int i = 0; i < m; ++i) nrm += wj[i] * wj[i];
        sigma[j] = sqrt(nrm);
    }
    for (int j = 0; j < n; ++j) order[j] = j;
    for (int a = 0; a < n - 1; ++a)
        for (int b = a + 1; b < n; ++b)
            if (sigma[order[b]] > sigma[order[a]]) {
                const int t = order[a]; order[a] = order[b]; order[b] = t;
            }
    const int k = (m < n) ? m : n;
    for (int jj = 0; jj < r && jj < k; ++jj) {
        const int src = order[jj];
        const double* vsrc = &Vfull[n * src];
        for (int i = 0; i < n; ++i) Vout[i + n * jj] = vsrc[i];
    }
}

// Seed A,B from the right singular vectors — reference §6
__device__ inline void seed_ab_from_V(const double* xmat, const double* V,
                                      int nl, int nr, int r, double* A, double* B) {
    for (int p = 0; p < r; ++p)
        for (int j = 0; j < nr; ++j)
            B[p + r * j] = V[j + static_cast<long>(nr) * p];
    for (int i = 0; i < nl; ++i)
        for (int p = 0; p < r; ++p) {
            double acc = 0.0;
            for (int j = 0; j < nr; ++j)
                acc += xmat[i + static_cast<long>(nl) * j] * B[p + r * j];
            A[i + static_cast<long>(nl) * p] = acc;
        }
}

template <int MAXNL, int MAXNR, int MAXR>
__device__ inline void dev_seed_ab(const double* xmat, int nl, int nr, int r,
                                   double* A, double* B) {
    double W[MAXNL * MAXNR];
    double Vfull[MAXNR * MAXNR];
    double sigma[MAXNR];
    int order[MAXNR];
    double Vout[MAXNR * MAXR];
    dev_jacobi_svd_V(xmat, nl, nr, r, W, Vfull, sigma, order, Vout);
    seed_ab_from_V(xmat, Vout, nl, nr, r, A, B);
}

// ALS half-steps, chi-square, constrained-weight solve, full fit — reference §6
__device__ inline void dev_opt_A_core(const double* B, const double* xmat,
                                      int nl, int nr, int r, const double* qinv,
                                      double fudge, double* Aout,
                                      double* xvec, double* Wm, double* coeffs,
                                      double* rhs, double* A2, double* lu, double* y,
                                      int* piv) {
    const int m = nl * nr;
    const int t = nl * r;
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j)
            xvec[i * nr + j] = xmat[i + nl * j];
    auto B2 = [&](int i, int p, int k) -> double {
        const int ii = k / nr, j = k % nr;
        return (i == ii) ? B[p + r * j] : 0.0;
    };
    for (int kr = 0; kr < m; ++kr)
        for (int a = 0; a < t; ++a) {
            const int i = a / r, p = a % r;
            double acc = 0.0;
            for (int kc = 0; kc < m; ++kc) acc += qinv[kr + m * kc] * B2(i, p, kc);
            Wm[kr + m * a] = acc;
        }
    for (int a = 0; a < t; ++a) {
        const int i = a / r, p = a % r;
        for (int c = 0; c < t; ++c) {
            double acc = 0.0;
            for (int k = 0; k < m; ++k) acc += B2(i, p, k) * Wm[k + m * c];
            coeffs[a + t * c] = acc;
        }
        double rr = 0.0;
        for (int k = 0; k < m; ++k) rr += xvec[k] * Wm[k + m * a];
        rhs[a] = rr;
    }
    double tr = 0.0;
    for (int a = 0; a < t; ++a) tr += coeffs[a + t * a];
    for (int a = 0; a < t; ++a) coeffs[a + t * a] += fudge * tr;
    for (int i = 0; i < nl * r; ++i) Aout[i] = 0.0;
    if (dev_solve(coeffs, t, rhs, A2, lu, piv, y)) {
        for (int i = 0; i < nl; ++i)
            for (int p = 0; p < r; ++p)
                Aout[i + nl * p] = A2[i * r + p];
    }
}

__device__ inline void dev_opt_B_core(const double* A, const double* xmat,
                                      int nl, int nr, int r, const double* qinv,
                                      double fudge, double* Bout,
                                      double* xvec, double* Wm, double* coeffs,
                                      double* rhs, double* B2v, double* lu, double* y,
                                      int* piv) {
    const int m = nl * nr;
    const int t = r * nr;
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j)
            xvec[i * nr + j] = xmat[i + nl * j];
    auto A2f = [&](int k, int p, int jc) -> double {
        const int i = k / nr, j = k % nr;
        return (j == jc) ? A[i + nl * p] : 0.0;
    };
    for (int kr = 0; kr < m; ++kr)
        for (int c = 0; c < t; ++c) {
            const int p = c / nr, jc = c % nr;
            double acc = 0.0;
            for (int kc = 0; kc < m; ++kc) acc += qinv[kr + m * kc] * A2f(kc, p, jc);
            Wm[kr + m * c] = acc;
        }
    for (int a = 0; a < t; ++a) {
        const int pa = a / nr, jca = a % nr;
        for (int c = 0; c < t; ++c) {
            double acc = 0.0;
            for (int k = 0; k < m; ++k) acc += A2f(k, pa, jca) * Wm[k + m * c];
            coeffs[a + t * c] = acc;
        }
        double rr = 0.0;
        for (int k = 0; k < m; ++k) rr += xvec[k] * Wm[k + m * a];
        rhs[a] = rr;
    }
    double tr = 0.0;
    for (int a = 0; a < t; ++a) tr += coeffs[a + t * a];
    for (int a = 0; a < t; ++a) coeffs[a + t * a] += fudge * tr;
    for (int i = 0; i < r * nr; ++i) Bout[i] = 0.0;
    if (dev_solve(coeffs, t, rhs, B2v, lu, piv, y)) {
        for (int p = 0; p < r; ++p)
            for (int j = 0; j < nr; ++j)
                Bout[p + r * j] = B2v[p * nr + j];
    }
}

__device__ inline double dev_chisq_of_core(const double* xmat, const double* A,
                                           const double* B, int nl, int nr, int r,
                                           const double* qinv, double* e) {
    const int m = nl * nr;
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j) {
            double ab = 0.0;
            for (int p = 0; p < r; ++p) ab += A[i + nl * p] * B[p + r * j];
            e[i * nr + j] = xmat[i + nl * j] - ab;
        }
    double acc = 0.0;
    for (int a = 0; a < m; ++a) {
        double row = 0.0;
        for (int b = 0; b < m; ++b) row += qinv[a + m * b] * e[b];
        acc += e[a] * row;
    }
    return acc;
}

template <int MAXM, int MAXT>
__device__ __noinline__ void dev_opt_A(const double* B, const double* xmat,
                                       int nl, int nr, int r, const double* qinv,
                                       double fudge, double* Aout) {
    double xvec[MAXM];
    double Wm[MAXM * MAXT];
    double coeffs[MAXT * MAXT];
    double rhs[MAXT];
    double A2[MAXT], lu[MAXT * MAXT], y[MAXT];
    int piv[MAXT];
    dev_opt_A_core(B, xmat, nl, nr, r, qinv, fudge, Aout,
                   xvec, Wm, coeffs, rhs, A2, lu, y, piv);
}

template <int MAXM, int MAXT>
__device__ __noinline__ void dev_opt_B(const double* A, const double* xmat,
                                       int nl, int nr, int r, const double* qinv,
                                       double fudge, double* Bout) {
    double xvec[MAXM];
    double Wm[MAXM * MAXT];
    double coeffs[MAXT * MAXT];
    double rhs[MAXT];
    double B2v[MAXT], lu[MAXT * MAXT], y[MAXT];
    int piv[MAXT];
    dev_opt_B_core(A, xmat, nl, nr, r, qinv, fudge, Bout,
                   xvec, Wm, coeffs, rhs, B2v, lu, y, piv);
}

template <int MAXM>
__device__ inline double dev_chisq_of(const double* xmat, const double* A,
                                      const double* B, int nl, int nr, int r,
                                      const double* qinv) {
    double e[MAXM];
    return dev_chisq_of_core(xmat, A, B, nl, nr, r, qinv, e);
}

__device__ inline bool solve_constrained_weights(const double* A, int nl, int r,
                                                 double* RHS, double* lhs, double* wv,
                                                 double* lu, double* y, int* piv,
                                                 double* w_out) {
    const int rp = r + 1;
    auto xm = [&](int i, int p) -> double { return (p < r) ? A[i + nl * p] : 1.0; };
    for (int i = 0; i < nl; ++i) {
        for (int ip = 0; ip < nl; ++ip) {
            double acc = 0.0;
            for (int p = 0; p < rp; ++p) acc += xm(i, p) * xm(ip, p);
            RHS[i + nl * ip] = acc;
        }
        lhs[i] = 1.0;
    }
    if (!dev_solve(RHS, nl, lhs, wv, lu, piv, y)) return false;
    double sum = 0.0;
    for (int i = 0; i < nl; ++i) sum += wv[i];
    for (int i = 0; i < nl; ++i) w_out[i] = wv[i] / sum;
    return true;
}

template <int MAXNL, int MAXNR, int MAXR>
__device__ inline int dev_als_weights(const double* xmat, int nl, int nr, int r,
                                      const double* qinv, double fudge, int als_iters,
                                      bool seed, double* A, double* B,
                                      double* w_out, double* chisq_out) {
    constexpr int MAXM = MAXNL * MAXNR;
    constexpr int MAXT = (MAXNL > MAXNR ? MAXNL : MAXNR) * MAXR;
    if (r == 0) {
        for (int i = 0; i < nl; ++i) w_out[i] = 1.0;
        *chisq_out = dev_chisq_of<MAXM>(xmat, A, B, nl, nr, 0, qinv);
        return 0;
    }
    if (seed) dev_seed_ab<MAXNL, MAXNR, MAXR>(xmat, nl, nr, r, A, B);
    double Atmp[MAXT], Btmp[MAXT];
    for (int it = 0; it < als_iters; ++it) {
        dev_opt_A<MAXM, MAXT>(B, xmat, nl, nr, r, qinv, fudge, Atmp);
        for (int i = 0; i < nl * r; ++i) A[i] = Atmp[i];
        dev_opt_B<MAXM, MAXT>(A, xmat, nl, nr, r, qinv, fudge, Btmp);
        for (int i = 0; i < r * nr; ++i) B[i] = Btmp[i];
    }
    double RHS[MAXNL * MAXNL], LHS[MAXNL], wv[MAXNL], lu[MAXNL * MAXNL], y[MAXNL];
    int piv[MAXNL];
    if (!solve_constrained_weights(A, nl, r, RHS, LHS, wv, lu, y, piv, w_out))
        return 6;
    *chisq_out = dev_chisq_of<MAXM>(xmat, A, B, nl, nr, r, qinv);
    return 0;
}

// f4 4-slab gather element — reference §7
__device__ inline double f4_gather_elem(const double* f2, int P, const int* lft,
                                        const int* rgt, int nr, int b, int k) {
    const int i = k / nr;
    const int j = k % nr;
    const int L0 = lft[0];
    const int R0 = rgt[0];
    const int Li = lft[i + 1];
    const int Rj = rgt[j + 1];
    const long slab = static_cast<long>(P) * static_cast<long>(P) * b;
    const auto at = [&](int a, int c) -> double {
        return f2[static_cast<long>(a) + static_cast<long>(P) * c + slab];
    };
    return 0.5 * (at(Li, R0) + at(L0, Rj) - at(L0, R0) - at(Li, Rj));
}

// Leave-one-out / total / tot_line + pseudo-value element cores — reference §8
__device__ inline void f4_loo_total_row(const double* dX_slice,
                                        const int* d_block_sizes, int m, int nb,
                                        double n, int k, double* dLoo_slice,
                                        double* tot_line_out, double* total_out) {
    double num = 0.0;
    for (int b = 0; b < nb; ++b)
        num += dX_slice[k + static_cast<long>(m) * b] *
               static_cast<double>(d_block_sizes[b]);
    const double tot_ij = num / n;
    double wln = 0.0, wld = 0.0, wbn = 0.0;
    for (int b = 0; b < nb; ++b) {
        const double bl = static_cast<double>(d_block_sizes[b]);
        const double rel = bl / n;
        const double xv = dX_slice[k + static_cast<long>(m) * b];
        const double loo = (tot_ij - xv * rel) / (1.0 - rel);
        dLoo_slice[k + static_cast<long>(m) * b] = loo;
        const double w = 1.0 - rel;
        wln += loo * w;
        wld += w;
        wbn += loo * bl;
    }
    const double tot_line = wln / wld;
    *tot_line_out = tot_line;
    double diffsum = 0.0;
    for (int b = 0; b < nb; ++b) {
        const double loo = dLoo_slice[k + static_cast<long>(m) * b];
        diffsum += tot_line - loo;
    }
    *total_out = diffsum + wbn / n;
}

__device__ inline double f4_xtau_elem(double est, double loo, double tot_line,
                                      double bl, double n) {
    const double h = n / bl;
    const double sh = sqrt(h - 1.0);
    return (est * h - loo * (h - 1.0) - tot_line) / sh;
}

// qpAdm f4 gather + standalone quartet / triple gather kernels — reference §7
__global__ void assemble_f4_gather_kernel(const double* __restrict__ f2, int P,
                                          const int* __restrict__ d_left,
                                          const int* __restrict__ d_right,
                                          int nl, int nr, int nb,
                                          const int* __restrict__ d_surv,
                                          double* __restrict__ dX) {
    const int m = nl * nr;
    const long total = static_cast<long>(m) * static_cast<long>(nb);
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<long>(gridDim.x) * blockDim.x) {
        const int k  = static_cast<int>(idx % m);
        const int bs = static_cast<int>(idx / m);
        const int b  = d_surv ? d_surv[bs] : bs;
        dX[idx] = f4_gather_elem(f2, P, d_left, d_right, nr, b, k);
    }
}

__global__ void assemble_f4_quartets_gather_kernel(const double* __restrict__ f2, int P,
                                                   const int* __restrict__ d_quartets,
                                                   int N, int nb,
                                                   const int* __restrict__ d_surv,
                                                   double* __restrict__ dX) {
    const long total = static_cast<long>(N) * static_cast<long>(nb);
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<long>(gridDim.x) * blockDim.x) {
        const int k  = static_cast<int>(idx % N);
        const int bs = static_cast<int>(idx / N);
        const int b  = d_surv ? d_surv[bs] : bs;
        const int p1 = d_quartets[4 * k + 0];
        const int p2 = d_quartets[4 * k + 1];
        const int p3 = d_quartets[4 * k + 2];
        const int p4 = d_quartets[4 * k + 3];
        const long slab = static_cast<long>(P) * static_cast<long>(P) * b;
        const auto at = [&](int a, int c) -> double {
            return f2[static_cast<long>(a) + static_cast<long>(P) * c + slab];
        };
        dX[idx] = 0.5 * (at(p2, p3) + at(p1, p4) - at(p1, p3) - at(p2, p4));
    }
}

__global__ void assemble_f3_triples_gather_kernel(const double* __restrict__ f2, int P,
                                                  const int* __restrict__ d_triples,
                                                  int N, int nb,
                                                  const int* __restrict__ d_surv,
                                                  double* __restrict__ dX) {
    const long total = static_cast<long>(N) * static_cast<long>(nb);
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<long>(gridDim.x) * blockDim.x) {
        const int k  = static_cast<int>(idx % N);
        const int bs = static_cast<int>(idx / N);
        const int b  = d_surv ? d_surv[bs] : bs;
        const int pc = d_triples[3 * k + 0];
        const int pa = d_triples[3 * k + 1];
        const int pb = d_triples[3 * k + 2];
        const long slab = static_cast<long>(P) * static_cast<long>(P) * b;
        const auto at = [&](int a, int c) -> double {
            return f2[static_cast<long>(a) + static_cast<long>(P) * c + slab];
        };
        dX[idx] = 0.5 * (at(pc, pa) + at(pc, pb) - at(pa, pb));
    }
}

// All-combinations sweep: combinatorial-number-system unrank — reference §12
__device__ __forceinline__ double sweep_choose(int n, int kk) {
    if (kk < 0 || n < kk) return 0.0;
    double num = 1.0;
    for (int i = 0; i < kk; ++i) num *= static_cast<double>(n - i);
    double den = 1.0;
    for (int i = 1; i <= kk; ++i) den *= static_cast<double>(i);
    return num / den;
}

__device__ __forceinline__ void sweep_unrank(long long r, int P, int k, int* c) {
    for (int pos = k - 1; pos >= 0; --pos) {
        const int kk = pos + 1;
        int v = kk - 1;
        while (sweep_choose(v + 1, kk) <= static_cast<double>(r)) ++v;
        c[pos] = v;
        r -= static_cast<long long>(sweep_choose(v, kk));
    }
}

__device__ __forceinline__ int sweep_to_f2(int pos, const int* d_subset) {
    return d_subset ? d_subset[pos] : pos;
}

__global__ void sweep_unrank_quartets_kernel(long long c0, int C, int range,
                                             const int* __restrict__ d_subset,
                                             int* __restrict__ dQuartets) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= C) return;
    int c[4];
    sweep_unrank(c0 + static_cast<long long>(t), range, 4, c);
    dQuartets[4 * t + 0] = sweep_to_f2(c[0], d_subset);
    dQuartets[4 * t + 1] = sweep_to_f2(c[1], d_subset);
    dQuartets[4 * t + 2] = sweep_to_f2(c[2], d_subset);
    dQuartets[4 * t + 3] = sweep_to_f2(c[3], d_subset);
}

__global__ void sweep_unrank_triples_kernel(long long c0, int C, int range,
                                            const int* __restrict__ d_subset,
                                            int* __restrict__ dTriples) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= C) return;
    int c[3];
    sweep_unrank(c0 + static_cast<long long>(t), range, 3, c);
    dTriples[3 * t + 0] = sweep_to_f2(c[0], d_subset);
    dTriples[3 * t + 1] = sweep_to_f2(c[1], d_subset);
    dTriples[3 * t + 2] = sweep_to_f2(c[2], d_subset);
}

// Significance z-filter — reference §12
__global__ void sweep_zfilter_kernel(const double* __restrict__ dXtotal,
                                     const double* __restrict__ dVar,
                                     int C, int mode, double min_z,
                                     double* __restrict__ dEst, double* __restrict__ dSe,
                                     double* __restrict__ dZ,
                                     unsigned char* __restrict__ d_flags) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= C) return;
    const double est = dXtotal[k];
    const double var = dVar[k];
    const double se = (var > 0.0) ? sqrt(var) : nan("");
    const double z = est / se;
    dEst[k] = est;
    dSe[k] = se;
    dZ[k] = z;
    unsigned char keep = 1;
    if (mode == 0) keep = (fabs(z) >= min_z) ? 1 : 0;
    d_flags[k] = keep;
}

// Deinterleave keys for compaction — reference §12
__global__ void sweep_deinterleave_keys_kernel(const int* __restrict__ d_items, int C, int k,
                                               int* __restrict__ d_c0, int* __restrict__ d_c1,
                                               int* __restrict__ d_c2, int* __restrict__ d_c3) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= C) return;
    d_c0[t] = d_items[k * t + 0];
    d_c1[t] = d_items[k * t + 1];
    d_c2[t] = d_items[k * t + 2];
    d_c3[t] = (k >= 4) ? d_items[k * t + 3] : 0;
}

// Bounded top-K reservoir with rising threshold — reference §12
__global__ void sweep_zfilter_tau_kernel(const double* __restrict__ dXtotal,
                                         const double* __restrict__ dVar,
                                         int C, const double* __restrict__ d_tau,
                                         double* __restrict__ dEst, double* __restrict__ dSe,
                                         double* __restrict__ dZ, double* __restrict__ dAbsZ,
                                         unsigned char* __restrict__ d_flags) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= C) return;
    const double est = dXtotal[k];
    const double var = dVar[k];
    const double se = (var > 0.0) ? sqrt(var) : nan("");
    const double z = est / se;
    const double az = fabs(z);
    dEst[k] = est;
    dSe[k] = se;
    dZ[k] = z;
    dAbsZ[k] = az;
    const double tau = d_tau[0];
    d_flags[k] = (az >= tau) ? 1 : 0;
}

__global__ void sweep_topk_iota_kernel(int* __restrict__ d_idx, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    d_idx[t] = t;
}

__global__ void sweep_topk_gather_kernel(const int* __restrict__ d_perm, int m,
                                         const double* __restrict__ inEst,
                                         const double* __restrict__ inSe,
                                         const double* __restrict__ inZ,
                                         const double* __restrict__ inAbsZ,
                                         const int* __restrict__ inC0,
                                         const int* __restrict__ inC1,
                                         const int* __restrict__ inC2,
                                         const int* __restrict__ inC3,
                                         double* __restrict__ outEst, double* __restrict__ outSe,
                                         double* __restrict__ outZ, double* __restrict__ outAbsZ,
                                         int* __restrict__ outC0, int* __restrict__ outC1,
                                         int* __restrict__ outC2, int* __restrict__ outC3) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;
    const int src = d_perm[r];
    outEst[r] = inEst[src];
    outSe[r] = inSe[src];
    outZ[r] = inZ[src];
    outAbsZ[r] = inAbsZ[src];
    outC0[r] = inC0[src];
    outC1[r] = inC1[src];
    outC2[r] = inC2[src];
    outC3[r] = inC3[src];
}

__global__ void sweep_topk_raise_tau_kernel(const double* __restrict__ d_sorted_absz,
                                            int K, int mode, double* __restrict__ d_tau) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (mode != 1 || K <= 0) return;
    const double kth = d_sorted_absz[K - 1];
    if (kth > d_tau[0]) d_tau[0] = kth;
}

// Missing-block keep-mask — reference §13
__global__ void f2_block_keep_kernel(const double* __restrict__ vpair, int P, int nb,
                                     int* __restrict__ d_keep) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const long slab = static_cast<long>(P) * static_cast<long>(P);
    const long base = slab * b;
    bool any_missing = false, any_present = false;
    for (long e = 0; e < slab; ++e) {
        if (steppe::core::pair_block_is_missing(vpair[base + e])) any_missing = true;
        else any_present = true;
        if (any_missing && any_present) break;
    }
    d_keep[b] = (any_missing && any_present) ? 0 : 1;
}

// Leave-one-out / total + pseudo-value + diagonal-variance kernels — reference §8
__global__ void f4_loo_total_kernel(const double* __restrict__ dX,
                                    const int* __restrict__ d_block_sizes,
                                    int m, int nb, double n,
                                    double* __restrict__ dLoo,
                                    double* __restrict__ dTotal,
                                    double* __restrict__ dTotLine) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= m) return;
    f4_loo_total_row(dX, d_block_sizes, m, nb, n, k, dLoo, &dTotLine[k], &dTotal[k]);
}

__global__ void f4_xtau_kernel(const double* __restrict__ dLoo,
                               const double* __restrict__ dEst,
                               const double* __restrict__ dTotLine,
                               const int* __restrict__ d_block_sizes,
                               int m, int nb, double n,
                               double* __restrict__ dXtau) {
    const long total = static_cast<long>(m) * static_cast<long>(nb);
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<long>(gridDim.x) * blockDim.x) {
        const int k = static_cast<int>(idx % m);
        const int b = static_cast<int>(idx / m);
        const double bl = static_cast<double>(d_block_sizes[b]);
        dXtau[idx] = f4_xtau_elem(dEst[k], dLoo[idx], dTotLine[k], bl, n);
    }
}

__global__ void f4_diag_var_kernel(const double* __restrict__ dXtau,
                                   int m, int nb, double* __restrict__ dVar) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < m;
         k += gridDim.x * blockDim.x) {
        double acc = 0.0;
        for (int b = 0; b < nb; ++b) {
            const double x = dXtau[static_cast<long>(k) + static_cast<long>(m) * b];
            acc += x * x;
        }
        dVar[k] = acc / static_cast<double>(nb);
    }
}

// Small matrix utilities: symmetrize, fudge-diag, reshape — reference §14
__global__ void symmetrize_lower_to_full_kernel(double* __restrict__ dM, int n) {
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int istride = gridDim.x * blockDim.x;
    const int jstride = gridDim.y * blockDim.y;
    for (int j = j0; j < n; j += jstride) {
        for (int i = i0; i < n; i += istride) {
            if (i > j) {
                dM[j + static_cast<long>(n) * i] = dM[i + static_cast<long>(n) * j];
            }
        }
    }
}

__global__ void add_fudge_diag_kernel(double* __restrict__ dM, int n,
                                      double fudge, double tr) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) dM[k + static_cast<long>(n) * k] += fudge * tr;
}

__global__ void xmat_from_rowmajor_kernel(const double* __restrict__ src,
                                          int nl, int nr, double* __restrict__ dXmat) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j)
            dXmat[i + nl * j] = src[j + nr * i];
}

// Single-thread seed kernel — reference §6
__global__ void seed_ab_kernel(const double* __restrict__ dXmat, int nl, int nr,
                               int r, double* __restrict__ dA, double* __restrict__ dB) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (r <= 0) return;
    dev_seed_ab<kQpMaxNl, kQpMaxNr, kQpMaxR>(dXmat, nl, nr, r, dA, dB);
}

// Numerical rank of the covariance via one-sided Jacobi — reference §9
__global__ void rank_via_jacobi_kernel(const double* __restrict__ dQ, int m, double eps,
                                       double* __restrict__ sW, double* __restrict__ sSigma,
                                       int* __restrict__ sOrder, int* __restrict__ dRank) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (m <= 0) { *dRank = 0; return; }
    for (int i = 0; i < m * m; ++i) sW[i] = dQ[i];
    for (int sweep = 0; sweep < kJacobiMaxSweeps; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < m - 1; ++p) {
            for (int q = p + 1; q < m; ++q) {
                double* wp = &sW[m * p];
                double* wq = &sW[m * q];
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int i = 0; i < m; ++i) {
                    const double a = wp[i]; const double b = wq[i];
                    alpha += a * a; beta += b * b; gamma += a * b;
                }
                off += gamma * gamma;
                if (fabs(gamma) <= kJacobiTol * sqrt(alpha * beta) || gamma == 0.0) continue;
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t = (zeta >= 0.0 ? 1.0 : -1.0) /
                                 (fabs(zeta) + sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / sqrt(1.0 + t * t);
                const double s = c * t;
                for (int i = 0; i < m; ++i) {
                    const double a = wp[i]; const double b = wq[i];
                    wp[i] = c * a - s * b; wq[i] = s * a + c * b;
                }
            }
        }
        if (off < kOffConvergence) break;
    }
    for (int j = 0; j < m; ++j) {
        double nrm = 0.0;
        const double* wj = &sW[m * j];
        for (int i = 0; i < m; ++i) nrm += wj[i] * wj[i];
        sSigma[j] = sqrt(nrm);
    }
    for (int j = 0; j < m; ++j) sOrder[j] = j;
    for (int a = 0; a < m - 1; ++a)
        for (int b = a + 1; b < m; ++b)
            if (sSigma[sOrder[b]] > sSigma[sOrder[a]]) {
                const int t = sOrder[a]; sOrder[a] = sOrder[b]; sOrder[b] = t;
            }
    const double smax = sSigma[sOrder[0]];
    const double tol = smax * static_cast<double>(m) * eps;
    int rk = 0;
    for (int j = 0; j < m; ++j) if (sSigma[j] > tol) ++rk;
    *dRank = rk;
}

// Single-thread ALS loop + weight / chi-square kernels — reference §6
__global__ void als_kernel(const double* __restrict__ dXmat, const double* __restrict__ dQinv,
                           int nl, int nr, int r, double fudge, int als_iters,
                           double* __restrict__ dA, double* __restrict__ dB) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (r <= 0) return;
    constexpr int MAXM = kQpMaxM;
    constexpr int MAXT = kQpMaxT;
    double A[MAXT], B[MAXT], Atmp[MAXT], Btmp[MAXT];
    for (int i = 0; i < nl * r; ++i) A[i] = dA[i];
    for (int i = 0; i < r * nr; ++i) B[i] = dB[i];
    for (int it = 0; it < als_iters; ++it) {
        dev_opt_A<MAXM, MAXT>(B, dXmat, nl, nr, r, dQinv, fudge, Atmp);
        for (int i = 0; i < nl * r; ++i) A[i] = Atmp[i];
        dev_opt_B<MAXM, MAXT>(A, dXmat, nl, nr, r, dQinv, fudge, Btmp);
        for (int i = 0; i < r * nr; ++i) B[i] = Btmp[i];
    }
    for (int i = 0; i < nl * r; ++i) dA[i] = A[i];
    for (int i = 0; i < r * nr; ++i) dB[i] = B[i];
}

__global__ void weights_chisq_kernel(const double* __restrict__ dXmat,
                                     const double* __restrict__ dQinv,
                                     const double* __restrict__ dA,
                                     const double* __restrict__ dB,
                                     int nl, int nr, int r,
                                     double* __restrict__ dW,
                                     double* __restrict__ dchisq,
                                     int* __restrict__ d_status) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    constexpr int MAXM = kQpMaxM;
    constexpr int MAXT = kQpMaxT;
    double A[MAXT], B[MAXT];
    for (int i = 0; i < nl * r; ++i) A[i] = dA[i];
    for (int i = 0; i < r * nr; ++i) B[i] = dB[i];
    if (r == 0) {
        for (int i = 0; i < nl; ++i) dW[i] = 1.0;
        *dchisq = dev_chisq_of<MAXM>(dXmat, A, B, nl, nr, 0, dQinv);
        *d_status = 0;
        return;
    }
    double RHS[kQpMaxNl * kQpMaxNl], LHS[kQpMaxNl], wv[kQpMaxNl], lu[kQpMaxNl * kQpMaxNl], y[kQpMaxNl];
    int piv[kQpMaxNl];
    if (!solve_constrained_weights(A, nl, r, RHS, LHS, wv, lu, y, piv, dW)) {
        *d_status = 6; return;
    }
    *dchisq = dev_chisq_of<MAXM>(dXmat, A, B, nl, nr, r, dQinv);
    *d_status = 0;
}

// Large-path (VRAM-scratch) opt_A / opt_B / chisq forwarders — reference §3
__device__ inline void dev_opt_A_large(const double* B, const double* xmat,
                                       int nl, int nr, int r, const double* qinv,
                                       double fudge, double* Aout,
                                       double* xvec, double* Wm, double* coeffs,
                                       double* rhs, double* A2, double* lu, double* y,
                                       int* ipiv) {
    dev_opt_A_core(B, xmat, nl, nr, r, qinv, fudge, Aout,
                   xvec, Wm, coeffs, rhs, A2, lu, y, ipiv);
}

__device__ inline void dev_opt_B_large(const double* A, const double* xmat,
                                       int nl, int nr, int r, const double* qinv,
                                       double fudge, double* Bout,
                                       double* xvec, double* Wm, double* coeffs,
                                       double* rhs, double* B2v, double* lu, double* y,
                                       int* ipiv) {
    dev_opt_B_core(A, xmat, nl, nr, r, qinv, fudge, Bout,
                   xvec, Wm, coeffs, rhs, B2v, lu, y, ipiv);
}

__device__ inline double dev_chisq_of_large(const double* xmat, const double* A,
                                            const double* B, int nl, int nr, int r,
                                            const double* qinv, double* e) {
    return dev_chisq_of_core(xmat, A, B, nl, nr, r, qinv, e);
}

// Large-path transpose — reference §14
__global__ void transpose_small_kernel(const double* __restrict__ dXmat,
                                       int nl, int nr, double* __restrict__ dXt) {
    const long total = static_cast<long>(nl) * static_cast<long>(nr);
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<long>(gridDim.x) * blockDim.x) {
        const int i = static_cast<int>(idx % nl);
        const int j = static_cast<int>(idx / nl);
        dXt[j + static_cast<long>(nr) * i] = dXmat[i + static_cast<long>(nl) * j];
    }
}

// Large-path seed / ALS / weight-chisq kernels — reference §6
__global__ void seed_from_V_kernel(const double* __restrict__ dXmat,
                                   const double* __restrict__ dVout,
                                   int nl, int nr, int r,
                                   double* __restrict__ dA, double* __restrict__ dB) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (r <= 0) return;
    seed_ab_from_V(dXmat, dVout, nl, nr, r, dA, dB);
}

__global__ void als_large_kernel(const double* __restrict__ dXmat,
                                 const double* __restrict__ dQinv,
                                 int nl, int nr, int r, double fudge, int als_iters,
                                 double* __restrict__ dA, double* __restrict__ dB,
                                 double* __restrict__ dScratch,
                                 int* __restrict__ dIntScratch) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (r <= 0) return;
    const int m = nl * nr;
    const int t = (nl > nr ? nl : nr) * r;
    double* xvec   = dScratch;
    double* Wm     = xvec + m;
    double* coeffs = Wm + static_cast<long>(m) * t;
    double* rhs    = coeffs + static_cast<long>(t) * t;
    double* tmp    = rhs + t;
    double* lu     = tmp + t;
    double* y      = lu + static_cast<long>(t) * t;
    int*    ipiv   = dIntScratch;
    for (int it = 0; it < als_iters; ++it) {
        dev_opt_A_large(dB, dXmat, nl, nr, r, dQinv, fudge, dA,
                        xvec, Wm, coeffs, rhs, tmp, lu, y, ipiv);
        dev_opt_B_large(dA, dXmat, nl, nr, r, dQinv, fudge, dB,
                        xvec, Wm, coeffs, rhs, tmp, lu, y, ipiv);
    }
}

__global__ void weights_chisq_large_kernel(const double* __restrict__ dXmat,
                                           const double* __restrict__ dQinv,
                                           const double* __restrict__ dA,
                                           const double* __restrict__ dB,
                                           int nl, int nr, int r,
                                           double* __restrict__ dW,
                                           double* __restrict__ dchisq,
                                           int* __restrict__ d_status,
                                           double* __restrict__ dScratch,
                                           int* __restrict__ dIntScratch) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double* RHS = dScratch;
    double* wv  = RHS + static_cast<long>(nl) * nl;
    double* lu  = wv + nl;
    double* y   = lu + static_cast<long>(nl) * nl;
    double* e   = y + nl;
    int*    piv = dIntScratch;
    if (r == 0) {
        for (int i = 0; i < nl; ++i) dW[i] = 1.0;
        *dchisq = dev_chisq_of_large(dXmat, dA, dB, nl, nr, 0, dQinv, e);
        *d_status = 0;
        return;
    }
    if (!solve_constrained_weights(dA, nl, r, RHS, e, wv, lu, y, piv, dW)) {
        *d_status = 6; return;
    }
    *dchisq = dev_chisq_of_large(dXmat, dA, dB, nl, nr, r, dQinv, e);
    *d_status = 0;
}

// Large-model parallel LOO refits — reference §11
__global__ void __launch_bounds__(64)
loo_large_batched_kernel(const double* __restrict__ dLoo,
                                         const double* __restrict__ dQinv,
                                         const double* __restrict__ dAseed,
                                         const double* __restrict__ dBseed,
                                         int nl, int nr, int r, double fudge,
                                         int als_iters, int nb, int n_models,
                                         long dbl_refit, long int_refit,
                                         double* __restrict__ dScratch,
                                         int* __restrict__ dIntScratch,
                                         double* __restrict__ dWmat) {
    const long total = static_cast<long>(nb) * n_models;
    for (long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         gid < total; gid += static_cast<long>(gridDim.x) * blockDim.x) {
        const int model = static_cast<int>(gid / nb);
        const int b     = static_cast<int>(gid % nb);
        const int m = nl * nr;
        const double* qinv = dQinv + static_cast<long>(model) * m * m;
        const double* loo  = dLoo  + static_cast<long>(model) * m * nb;
        double* base = dScratch + gid * dbl_refit;
        int*    ibase = dIntScratch + gid * int_refit;
        double* xmat = base;
        double* A    = xmat + m;
        double* B    = A + static_cast<long>(nl) * (r > 0 ? r : 1);
        double* sc   = B + static_cast<long>(r > 0 ? r : 1) * nr;
        for (int i = 0; i < nl; ++i)
            for (int j = 0; j < nr; ++j)
                xmat[i + nl * j] = loo[(j + nr * i) + static_cast<long>(m) * b];
        if (r > 0) {
            const double* As = dAseed + gid * static_cast<long>(nl) * r;
            const double* Bs = dBseed + gid * static_cast<long>(r) * nr;
            for (int i = 0; i < nl * r; ++i) A[i] = As[i];
            for (int i = 0; i < r * nr; ++i) B[i] = Bs[i];
            const int t = (nl > nr ? nl : nr) * r;
            double* xvec   = sc;
            double* Wm     = xvec + m;
            double* coeffs = Wm + static_cast<long>(m) * t;
            double* rhs    = coeffs + static_cast<long>(t) * t;
            double* tmp    = rhs + t;
            double* lu     = tmp + t;
            double* y      = lu + static_cast<long>(t) * t;
            int*    ipiv   = ibase;
            for (int it = 0; it < als_iters; ++it) {
                dev_opt_A_large(B, xmat, nl, nr, r, qinv, fudge, A,
                                xvec, Wm, coeffs, rhs, tmp, lu, y, ipiv);
                dev_opt_B_large(A, xmat, nl, nr, r, qinv, fudge, B,
                                xvec, Wm, coeffs, rhs, tmp, lu, y, ipiv);
            }
        }
        double* row = dWmat + (static_cast<long>(model) * nb + b) * nl;
        double* RHS = sc;
        double* wv  = RHS + static_cast<long>(nl) * nl;
        double* lu  = wv + nl;
        double* y   = lu + static_cast<long>(nl) * nl;
        double* e   = y + nl;
        int*    piv = ibase;
        if (r == 0) {
            for (int i = 0; i < nl; ++i) row[i] = 1.0;
            continue;
        }
        if (!solve_constrained_weights(A, nl, r, RHS, e, wv, lu, y, piv, row)) {
            for (int i = 0; i < nl; ++i) row[i] = 0.0;
            continue;
        }
    }
}

// Batched LOO refits — reference §11
__global__ void __launch_bounds__(64)
loo_batched_kernel(const double* __restrict__ dLoo,
                                   const double* __restrict__ dQinv,
                                   int nl, int nr, int r, double fudge, int als_iters,
                                   int nb, double* __restrict__ dWmat) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const int m = nl * nr;
    double xmat[kQpMaxM];
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j)
            xmat[i + nl * j] = dLoo[(j + nr * i) + static_cast<long>(m) * b];
    double A[kQpMaxT], B[kQpMaxT], w[kQpMaxNl], chisq = 0.0;
    const int st = dev_als_weights<kQpMaxNl, kQpMaxNr, kQpMaxR>(
        xmat, nl, nr, r, dQinv, fudge, als_iters, true, A, B, w, &chisq);
    for (int i = 0; i < nl; ++i)
        dWmat[static_cast<long>(b) * nl + i] = (st == 0) ? w[i] : 0.0;
}

// Model-batched f4 gather — reference §7
__global__ void assemble_f4_gather_models_kernel(const double* __restrict__ f2, int P,
                                                 const int* __restrict__ d_left_arena,
                                                 const int* __restrict__ d_right_arena,
                                                 int nl, int nr, int nb, int n_models,
                                                 const int* __restrict__ d_surv,
                                                 double* __restrict__ dX) {
    const long m = static_cast<long>(nl) * nr;
    const long per = m * nb;
    const long total = per * n_models;
    for (long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         gid < total; gid += static_cast<long>(gridDim.x) * blockDim.x) {
        const int model = static_cast<int>(gid / per);
        const long idx = gid - static_cast<long>(model) * per;
        const int k  = static_cast<int>(idx % m);
        const int bs = static_cast<int>(idx / m);
        const int b  = d_surv ? d_surv[bs] : bs;
        const int* lft = d_left_arena + static_cast<long>(model) * (nl + 1);
        const int* rgt = d_right_arena + static_cast<long>(model) * (nr + 1);
        dX[gid] = f4_gather_elem(f2, P, lft, rgt, nr, b, k);
    }
}

// Model-batched loo / total + xtau — reference §8
__global__ void f4_loo_total_models_kernel(const double* __restrict__ dX,
                                           const int* __restrict__ d_block_sizes,
                                           int m, int nb, double n, int n_models,
                                           double* __restrict__ dLoo,
                                           double* __restrict__ dTotal,
                                           double* __restrict__ dTotLine) {
    const long total = static_cast<long>(m) * n_models;
    for (long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         gid < total; gid += static_cast<long>(gridDim.x) * blockDim.x) {
        const int model = static_cast<int>(gid / m);
        const int k = static_cast<int>(gid % m);
        const long base = static_cast<long>(model) * m * nb;
        const long out = static_cast<long>(model) * m + k;
        f4_loo_total_row(dX + base, d_block_sizes, m, nb, n, k, dLoo + base,
                         &dTotLine[out], &dTotal[out]);
    }
}

__global__ void f4_xtau_models_kernel(const double* __restrict__ dLoo,
                                      const double* __restrict__ dEst,
                                      const double* __restrict__ dTotLine,
                                      const int* __restrict__ d_block_sizes,
                                      int m, int nb, double n, int n_models,
                                      double* __restrict__ dXtau) {
    const long per = static_cast<long>(m) * nb;
    const long total = per * n_models;
    for (long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         gid < total; gid += static_cast<long>(gridDim.x) * blockDim.x) {
        const int model = static_cast<int>(gid / per);
        const long idx = gid - static_cast<long>(model) * per;
        const int k = static_cast<int>(idx % m);
        const int b = static_cast<int>(idx / m);
        const double bl = static_cast<double>(d_block_sizes[b]);
        const double est = dEst[static_cast<long>(model) * m + k];
        const double tl = dTotLine[static_cast<long>(model) * m + k];
        dXtau[gid] = f4_xtau_elem(est, dLoo[gid], tl, bl, n);
    }
}

// Model-batched fudge-diag + identity fill — reference §14
__global__ void add_fudge_diag_models_kernel(const double* __restrict__ dQ,
                                             double* __restrict__ dQf, int m,
                                             double fudge, int n_models) {
    const int model = blockIdx.x;
    if (model >= n_models) return;
    const long base = static_cast<long>(model) * m * m;
    __shared__ double s_tr;
    if (threadIdx.x == 0) {
        double tr = 0.0;
        for (int k = 0; k < m; ++k) tr += dQ[base + k + static_cast<long>(m) * k];
        s_tr = tr;
    }
    __syncthreads();
    const double add = fudge * s_tr;
    for (int e = threadIdx.x; e < m * m; e += blockDim.x) {
        const int col = e / m, row = e % m;
        dQf[base + e] = dQ[base + e] + ((row == col) ? add : 0.0);
    }
}

__global__ void fill_identity_batched_kernel(double* __restrict__ dI, int m,
                                             int n_models) {
    const long per = static_cast<long>(m) * m;
    const long total = per * n_models;
    for (long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         gid < total; gid += static_cast<long>(gridDim.x) * blockDim.x) {
        const long e = gid % per;
        const int col = static_cast<int>(e / m), row = static_cast<int>(e % m);
        dI[gid] = (row == col) ? 1.0 : 0.0;
    }
}

// Model-batched full fit: rank sweep + popdrop — reference §10
__global__ void __launch_bounds__(64)
qpadm_fit_models_kernel(const double* __restrict__ dTotal,
                                        const double* __restrict__ dQinv,
                                        const double* __restrict__ dLoo,
                                        const int* __restrict__ d_block_sizes,
                                        int nl, int nr, int r_fit, int rmax,
                                        double fudge, int als_iters, int nb, int n_models,
                                        double* __restrict__ d_weight,
                                        double* __restrict__ d_se,
                                        double* __restrict__ d_chisq,
                                        int* __restrict__ d_status,
                                        double* __restrict__ d_rank_chisq,
                                        double* __restrict__ d_pop_chisq,
                                        double* __restrict__ d_pop_wfull) {
    const int model = blockIdx.x * blockDim.x + threadIdx.x;
    if (model >= n_models) return;
    const int m = nl * nr;
    const double* total = dTotal + static_cast<long>(model) * m;
    const double* qinv  = dQinv  + static_cast<long>(model) * m * m;
    (void)dLoo; (void)d_se;

    double xmat[kQpMaxM];
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j)
            xmat[i + nl * j] = total[j + nr * i];

    double A[kQpMaxT], B[kQpMaxT], w[kQpMaxNl], chisq = 0.0;
    int st = dev_als_weights<kQpMaxNl, kQpMaxNr, kQpMaxR>(
        xmat, nl, nr, r_fit, qinv, fudge, als_iters, true, A, B, w, &chisq);
    d_status[model] = st;
    for (int i = 0; i < nl; ++i)
        d_weight[static_cast<long>(model) * nl + i] = (st == 0) ? w[i] : 0.0;
    d_chisq[model] = chisq;

    for (int rr = 0; rr <= rmax; ++rr) {
        double Ar[kQpMaxT], Br[kQpMaxT], wr[kQpMaxNl], cr = 0.0;
        dev_als_weights<kQpMaxNl, kQpMaxNr, kQpMaxR>(
            xmat, nl, nr, rr, qinv, fudge, als_iters, true, Ar, Br, wr, &cr);
        d_rank_chisq[static_cast<long>(model) * (rmax + 1) + rr] = cr;
    }

    auto fit_reduced = [&](const int* surv, int nl_red, double* w_red, double* chisq_red) -> int {
        const int m_red = nl_red * nr;
        double xr[kQpMaxM];
        for (int ii = 0; ii < nl_red; ++ii)
            for (int j = 0; j < nr; ++j)
                xr[ii + nl_red * j] = total[j + nr * surv[ii]];
        double qr[kQpMaxM * kQpMaxM];
        for (int a = 0; a < m_red; ++a) {
            const int ia = surv[a / nr] * nr + (a % nr);
            for (int bb = 0; bb < m_red; ++bb) {
                const int ib = surv[bb / nr] * nr + (bb % nr);
                qr[a + m_red * bb] = qinv[ia + m * ib];
            }
        }
        const int r_red = nl_red - 1;
        double Ar[kQpMaxT], Br[kQpMaxT];
        return dev_als_weights<kQpMaxNl, kQpMaxNr, kQpMaxR>(
            xr, nl_red, nr, r_red, qr, fudge, als_iters, true, Ar, Br,
            w_red, chisq_red);
    };

    {
        int surv[kQpMaxNl];
        for (int i = 0; i < nl; ++i) surv[i] = i;
        double wr[kQpMaxNl], cr = (0.0 / 0.0);
        const int s0 = fit_reduced(surv, nl, wr, &cr);
        d_pop_chisq[static_cast<long>(model) * (nl + 1) + 0] =
            (s0 == 0) ? cr : (0.0 / 0.0);
        for (int i = 0; i < nl; ++i)
            d_pop_wfull[static_cast<long>(model) * nl + i] =
                (s0 == 0) ? wr[i] : (0.0 / 0.0);
    }
    if (nl >= 2) {
        for (int drop = nl - 1; drop >= 0; --drop) {
            int surv[kQpMaxNl];
            int cnt = 0;
            for (int i = 0; i < nl; ++i) if (i != drop) surv[cnt++] = i;
            double wr[kQpMaxNl], cr = (0.0 / 0.0);
            const int sd = fit_reduced(surv, cnt, wr, &cr);
            const int row = 1 + (nl - 1 - drop);
            d_pop_chisq[static_cast<long>(model) * (nl + 1) + row] =
                (sd == 0) ? cr : (0.0 / 0.0);
        }
    }
}

// Model-batched LOO refits + SE reduction — reference §11
__global__ void __launch_bounds__(128)
qpadm_loo_models_kernel(const double* __restrict__ dLoo,
                                        const double* __restrict__ dQinv,
                                        int nl, int nr, int r_fit, double fudge,
                                        int als_iters, int nb, int n_models, double s,
                                        double* __restrict__ dWmat) {
    const long total = static_cast<long>(nb) * n_models;
    for (long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         gid < total; gid += static_cast<long>(gridDim.x) * blockDim.x) {
        const int model = static_cast<int>(gid / nb);
        const int b = static_cast<int>(gid % nb);
        const int m = nl * nr;
        const double* qinv = dQinv + static_cast<long>(model) * m * m;
        const double* loo = dLoo + static_cast<long>(model) * m * nb;
        double xb[kQpMaxM];
        for (int i = 0; i < nl; ++i)
            for (int j = 0; j < nr; ++j)
                xb[i + nl * j] = loo[(j + nr * i) + static_cast<long>(m) * b];
        double Ab[kQpMaxT], Bb[kQpMaxT], wb[kQpMaxNl], cb;
        const int sb = dev_als_weights<kQpMaxNl, kQpMaxNr, kQpMaxR>(
            xb, nl, nr, r_fit, qinv, fudge, als_iters, true, Ab, Bb, wb, &cb);
        double* row = dWmat + (static_cast<long>(model) * nb + b) * nl;
        for (int i = 0; i < nl; ++i) row[i] = (sb == 0) ? s * wb[i] : 0.0;
    }
}

__global__ void qpadm_se_from_wmat_kernel(const double* __restrict__ dWmat,
                                          int nl, int nb, int n_models,
                                          double* __restrict__ d_se) {
    const long total = static_cast<long>(nl) * n_models;
    for (long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         gid < total; gid += static_cast<long>(gridDim.x) * blockDim.x) {
        const int model = static_cast<int>(gid / nl);
        const int i = static_cast<int>(gid % nl);
        const double* w = dWmat + static_cast<long>(model) * nb * nl;
        if (nb < 2) { d_se[gid] = 0.0; continue; }
        double mean = 0.0;
        for (int b = 0; b < nb; ++b) mean += w[static_cast<long>(b) * nl + i];
        mean /= static_cast<double>(nb);
        double var = 0.0;
        for (int b = 0; b < nb; ++b) {
            const double d = w[static_cast<long>(b) * nl + i] - mean;
            var += d * d;
        }
        d_se[gid] = sqrt(var / static_cast<double>(nb - 1));
    }
}

// Survivor gather of loo / qinv slices — reference §14
__global__ void qpadm_gather_loo_qinv_kernel(const double* __restrict__ dLooSrc,
                                             const double* __restrict__ dQinvSrc,
                                             const int* __restrict__ d_surv,
                                             int m, int nb, int n_surv,
                                             double* __restrict__ dLooDst,
                                             double* __restrict__ dQinvDst) {
    const long loo_per = static_cast<long>(m) * nb;
    const long qinv_per = static_cast<long>(m) * m;
    const long total = (loo_per + qinv_per) * n_surv;
    for (long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         gid < total; gid += static_cast<long>(gridDim.x) * blockDim.x) {
        const long per = loo_per + qinv_per;
        const int k = static_cast<int>(gid / per);
        const long e = gid % per;
        const int j = d_surv[k];
        if (e < loo_per) {
            dLooDst[static_cast<long>(k) * loo_per + e] =
                dLooSrc[static_cast<long>(j) * loo_per + e];
        } else {
            const long q = e - loo_per;
            dQinvDst[static_cast<long>(k) * qinv_per + q] =
                dQinvSrc[static_cast<long>(j) * qinv_per + q];
        }
    }
}

}  // namespace

// Launch wrappers — reference §14
void launch_assemble_f4_gather_models_batched(const double* d_f2, int P,
                                              const int* d_left_arena,
                                              const int* d_right_arena,
                                              int nl, int nr, int nb, int n_models,
                                              const int* d_surv,
                                              double* dX, cudaStream_t stream) {
    const long total = static_cast<long>(nl) * nr * nb * n_models;
    if (total <= 0) return;
    const int block = kBlock256;
    const int grid = core::grid_stride_extent(total, block);
    assemble_f4_gather_models_kernel<<<grid, block, 0, stream>>>(
        d_f2, P, d_left_arena, d_right_arena, nl, nr, nb, n_models, d_surv, dX);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_f4_loo_total_models_batched(const double* dX, const int* d_block_sizes,
                                        int m, int nb, double n, int n_models,
                                        double* dLoo, double* dTotal, double* dTotLine,
                                        cudaStream_t stream) {
    const long total = static_cast<long>(m) * n_models;
    if (total <= 0 || nb <= 0) return;
    const int block = kBlock128;
    const int grid = core::grid_stride_extent(total, block);
    f4_loo_total_models_kernel<<<grid, block, 0, stream>>>(
        dX, d_block_sizes, m, nb, n, n_models, dLoo, dTotal, dTotLine);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_f4_xtau_models_batched(const double* dLoo, const double* dEst,
                                   const double* dTotLine, const int* d_block_sizes,
                                   int m, int nb, double n, int n_models,
                                   double* dXtau, cudaStream_t stream) {
    const long total = static_cast<long>(m) * nb * n_models;
    if (total <= 0) return;
    const int block = kBlock256;
    const int grid = core::grid_stride_extent(total, block);
    f4_xtau_models_kernel<<<grid, block, 0, stream>>>(
        dLoo, dEst, dTotLine, d_block_sizes, m, nb, n, n_models, dXtau);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_add_fudge_diag_models_batched(const double* dQ, double* dQf, int m,
                                          double fudge, int n_models, cudaStream_t stream) {
    if (m <= 0 || n_models <= 0) return;
    int block = (m * m < 256) ? ((m * m + kWarpSize - 1) / kWarpSize * kWarpSize) : 256;
    if (block < kWarpSize) block = kWarpSize;
    add_fudge_diag_models_kernel<<<n_models, block, 0, stream>>>(
        dQ, dQf, m, fudge, n_models);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_fill_identity_batched(double* dI, int m, int n_models, cudaStream_t stream) {
    const long total = static_cast<long>(m) * m * n_models;
    if (total <= 0) return;
    const int block = kBlock256;
    const int grid = core::grid_stride_extent(total, block);
    fill_identity_batched_kernel<<<grid, block, 0, stream>>>(dI, m, n_models);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_fit_models_batched(const double* dTotal, const double* dQinv,
                                     const double* dLoo, const int* d_block_sizes,
                                     int nl, int nr, int r_fit, int rmax,
                                     double fudge, int als_iters, int nb, int n_models,
                                     double* d_weight, double* d_se, double* d_chisq,
                                     int* d_status, double* d_rank_chisq,
                                     double* d_pop_chisq, double* d_pop_wfull,
                                     cudaStream_t stream) {
    if (n_models <= 0) return;
    const int block = kBlock64;
    const int grid = core::cdiv(n_models, block);
    qpadm_fit_models_kernel<<<grid, block, 0, stream>>>(
        dTotal, dQinv, dLoo, d_block_sizes, nl, nr, r_fit, rmax, fudge, als_iters,
        nb, n_models, d_weight, d_se, d_chisq, d_status, d_rank_chisq,
        d_pop_chisq, d_pop_wfull);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_loo_models_batched(const double* dLoo, const double* dQinv,
                                     int nl, int nr, int r_fit, double fudge,
                                     int als_iters, int nb, int n_models, double s,
                                     double* dWmat, cudaStream_t stream) {
    const long total = static_cast<long>(nb) * n_models;
    if (total <= 0) return;
    const int block = kBlock128;
    const int grid = core::grid_stride_extent(total, block);
    qpadm_loo_models_kernel<<<grid, block, 0, stream>>>(
        dLoo, dQinv, nl, nr, r_fit, fudge, als_iters, nb, n_models, s, dWmat);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_se_from_wmat_batched(const double* dWmat, int nl, int nb,
                                       int n_models, double* d_se, cudaStream_t stream) {
    const long total = static_cast<long>(nl) * n_models;
    if (total <= 0) return;
    const int block = kBlock128;
    const int grid = core::grid_stride_extent(total, block);
    qpadm_se_from_wmat_kernel<<<grid, block, 0, stream>>>(dWmat, nl, nb, n_models, d_se);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_gather_loo_qinv(const double* dLooSrc, const double* dQinvSrc,
                                  const int* d_surv, int m, int nb, int n_surv,
                                  double* dLooDst, double* dQinvDst, cudaStream_t stream) {
    const long per = static_cast<long>(m) * nb + static_cast<long>(m) * m;
    const long total = per * n_surv;
    if (total <= 0) return;
    const int block = kBlock256;
    const int grid = core::grid_stride_extent(total, block);
    qpadm_gather_loo_qinv_kernel<<<grid, block, 0, stream>>>(
        dLooSrc, dQinvSrc, d_surv, m, nb, n_surv, dLooDst, dQinvDst);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_assemble_f4_gather(const double* d_f2, int P,
                               const int* d_left, const int* d_right,
                               int nl, int nr, int nb, const int* d_surv,
                               double* dX, cudaStream_t stream) {
    const long total = static_cast<long>(nl) * nr * nb;
    if (total <= 0) return;
    const int block = kBlock256;
    const int grid = core::grid_stride_extent(total, block);
    assemble_f4_gather_kernel<<<grid, block, 0, stream>>>(d_f2, P, d_left, d_right,
                                                          nl, nr, nb, d_surv, dX);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_assemble_f4_quartets_gather(const double* d_f2, int P,
                                        const int* d_quartets, int N, int nb,
                                        const int* d_surv,
                                        double* dX, cudaStream_t stream) {
    const long total = static_cast<long>(N) * nb;
    if (total <= 0) return;
    const int block = kBlock256;
    const int grid = core::grid_stride_extent(total, block);
    assemble_f4_quartets_gather_kernel<<<grid, block, 0, stream>>>(
        d_f2, P, d_quartets, N, nb, d_surv, dX);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_assemble_f3_triples_gather(const double* d_f2, int P,
                                       const int* d_triples, int N, int nb,
                                       const int* d_surv,
                                       double* dX, cudaStream_t stream) {
    const long total = static_cast<long>(N) * nb;
    if (total <= 0) return;
    const int block = kBlock256;
    const int grid = core::grid_stride_extent(total, block);
    assemble_f3_triples_gather_kernel<<<grid, block, 0, stream>>>(
        d_f2, P, d_triples, N, nb, d_surv, dX);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_f2_block_keep(const double* d_vpair, int P, int nb, int* d_keep,
                          cudaStream_t stream) {
    if (nb <= 0) return;
    const int block = kBlock128;
    const int grid = core::cdiv(nb, block);
    f2_block_keep_kernel<<<grid, block, 0, stream>>>(d_vpair, P, nb, d_keep);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_f4_loo_total(const double* dX, const int* d_block_sizes,
                         int m, int nb, double n,
                         double* dLoo, double* dTotal, double* dTotLine,
                         cudaStream_t stream) {
    if (m <= 0 || nb <= 0) return;
    const int block = kBlock64;
    const int grid = core::cdiv(m, block);
    f4_loo_total_kernel<<<grid, block, 0, stream>>>(dX, d_block_sizes, m, nb, n,
                                                    dLoo, dTotal, dTotLine);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_f4_xtau(const double* dLoo, const double* dEst, const double* dTotLine,
                    const int* d_block_sizes, int m, int nb, double n,
                    double* dXtau, cudaStream_t stream) {
    const long total = static_cast<long>(m) * nb;
    if (total <= 0) return;
    const int block = kBlock256;
    const int grid = core::grid_stride_extent(total, block);
    f4_xtau_kernel<<<grid, block, 0, stream>>>(dLoo, dEst, dTotLine, d_block_sizes,
                                               m, nb, n, dXtau);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_f4_diag_var(const double* dXtau, int m, int nb, double* dVar,
                        cudaStream_t stream) {
    if (m <= 0 || nb <= 0) return;
    const int block = kBlock256;
    const int grid = core::cdiv(m, block);
    f4_diag_var_kernel<<<grid, block, 0, stream>>>(dXtau, m, nb, dVar);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_sweep_unrank_quartets(long long c0, int C, int range, const int* d_subset,
                                  int* dQuartets, cudaStream_t stream) {
    if (C <= 0) return;
    const int block = kBlock256;
    const int grid = core::cdiv(C, block);
    sweep_unrank_quartets_kernel<<<grid, block, 0, stream>>>(c0, C, range, d_subset, dQuartets);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_sweep_unrank_triples(long long c0, int C, int range, const int* d_subset,
                                 int* dTriples, cudaStream_t stream) {
    if (C <= 0) return;
    const int block = kBlock256;
    const int grid = core::cdiv(C, block);
    sweep_unrank_triples_kernel<<<grid, block, 0, stream>>>(c0, C, range, d_subset, dTriples);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_sweep_zfilter(const double* dXtotal, const double* dVar, int C, int mode,
                          double min_z, double* dEst, double* dSe, double* dZ,
                          unsigned char* d_flags, cudaStream_t stream) {
    if (C <= 0) return;
    const int block = kBlock256;
    const int grid = core::cdiv(C, block);
    sweep_zfilter_kernel<<<grid, block, 0, stream>>>(dXtotal, dVar, C, mode, min_z,
                                                     dEst, dSe, dZ, d_flags);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_sweep_deinterleave_keys(const int* d_items, int C, int k,
                                    int* d_c0, int* d_c1, int* d_c2, int* d_c3,
                                    cudaStream_t stream) {
    if (C <= 0) return;
    const int block = kBlock256;
    const int grid = core::cdiv(C, block);
    sweep_deinterleave_keys_kernel<<<grid, block, 0, stream>>>(d_items, C, k,
                                                               d_c0, d_c1, d_c2, d_c3);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_sweep_zfilter_tau(const double* dXtotal, const double* dVar, int C,
                              const double* d_tau, double* dEst, double* dSe, double* dZ,
                              double* dAbsZ, unsigned char* d_flags, cudaStream_t stream) {
    if (C <= 0) return;
    const int block = kBlock256;
    const int grid = core::cdiv(C, block);
    sweep_zfilter_tau_kernel<<<grid, block, 0, stream>>>(dXtotal, dVar, C, d_tau,
                                                         dEst, dSe, dZ, dAbsZ, d_flags);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_sweep_topk_iota(int* d_idx, int n, cudaStream_t stream) {
    if (n <= 0) return;
    const int block = kBlock256;
    const int grid = core::cdiv(n, block);
    sweep_topk_iota_kernel<<<grid, block, 0, stream>>>(d_idx, n);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_sweep_topk_gather(const int* d_perm, int m,
                              const double* inEst, const double* inSe, const double* inZ,
                              const double* inAbsZ, const int* inC0, const int* inC1,
                              const int* inC2, const int* inC3,
                              double* outEst, double* outSe, double* outZ, double* outAbsZ,
                              int* outC0, int* outC1, int* outC2, int* outC3,
                              cudaStream_t stream) {
    if (m <= 0) return;
    const int block = kBlock256;
    const int grid = core::cdiv(m, block);
    sweep_topk_gather_kernel<<<grid, block, 0, stream>>>(
        d_perm, m, inEst, inSe, inZ, inAbsZ, inC0, inC1, inC2, inC3,
        outEst, outSe, outZ, outAbsZ, outC0, outC1, outC2, outC3);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_sweep_topk_raise_tau(const double* d_sorted_absz, int K, int mode,
                                 double* d_tau, cudaStream_t stream) {
    sweep_topk_raise_tau_kernel<<<1, 1, 0, stream>>>(d_sorted_absz, K, mode, d_tau);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_symmetrize_lower_to_full(double* dM, int n, cudaStream_t stream) {
    if (n <= 0) return;
    const dim3 block(kSymTile, kSymTile);
    const unsigned tiles = (static_cast<unsigned>(n) + kSymTile - 1) / kSymTile;
    const unsigned grid_dim = tiles > core::kMaxGridY ? core::kMaxGridY : tiles;
    const dim3 grid(grid_dim, grid_dim);
    symmetrize_lower_to_full_kernel<<<grid, block, 0, stream>>>(dM, n);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_add_fudge_diag(double* dM, int n, double fudge, double tr,
                           cudaStream_t stream) {
    if (n <= 0) return;
    const int block = kBlock64;
    const int grid = core::cdiv(n, block);
    add_fudge_diag_kernel<<<grid, block, 0, stream>>>(dM, n, fudge, tr);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_xmat_from_rowmajor(const double* dTotalSrc, int nl, int nr,
                                     double* dXmat, cudaStream_t stream) {
    xmat_from_rowmajor_kernel<<<1, 1, 0, stream>>>(dTotalSrc, nl, nr, dXmat);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_seed_ab(const double* dXmat, int nl, int nr, int r,
                          double* dA, double* dB, cudaStream_t stream) {
    seed_ab_kernel<<<1, 1, 0, stream>>>(dXmat, nl, nr, r, dA, dB);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_rank_via_jacobi(const double* dQ, int m, double eps,
                                  double* dScratch, int* dIntScratch, int* dRank,
                                  cudaStream_t stream) {
    double* sW = dScratch;
    double* sSigma = dScratch + static_cast<long>(m) * m;
    rank_via_jacobi_kernel<<<1, 1, 0, stream>>>(dQ, m, eps, sW, sSigma, dIntScratch, dRank);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_als(const double* dXmat, const double* dQinv,
                      int nl, int nr, int r, double fudge, int als_iters,
                      double* dA, double* dB, cudaStream_t stream) {
    als_kernel<<<1, 1, 0, stream>>>(dXmat, dQinv, nl, nr, r, fudge, als_iters, dA, dB);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_weights_chisq(const double* dXmat, const double* dQinv,
                                const double* dA, const double* dB,
                                int nl, int nr, int r,
                                double* dW, double* dchisq, int* d_status,
                                cudaStream_t stream) {
    weights_chisq_kernel<<<1, 1, 0, stream>>>(dXmat, dQinv, dA, dB, nl, nr, r,
                                              dW, dchisq, d_status);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_loo_batched(const double* dLoo, const double* dQinv,
                              int nl, int nr, int r, double fudge, int als_iters,
                              int nb, double* dWmat, cudaStream_t stream) {
    if (nb <= 0) return;
    const int block = kBlock64;
    const int grid = core::cdiv(nb, block);
    loo_batched_kernel<<<grid, block, 0, stream>>>(dLoo, dQinv, nl, nr, r, fudge,
                                                   als_iters, nb, dWmat);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_transpose_small(const double* dXmat, int nl, int nr,
                            double* dXt, cudaStream_t stream) {
    const long total = static_cast<long>(nl) * nr;
    if (total <= 0) return;
    const int block = kBlock256;
    const int grid = static_cast<int>(core::cdiv(total, static_cast<long>(block)));
    transpose_small_kernel<<<grid, block, 0, stream>>>(dXmat, nl, nr, dXt);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_seed_from_V(const double* dXmat, const double* dVout,
                              int nl, int nr, int r,
                              double* dA, double* dB, cudaStream_t stream) {
    seed_from_V_kernel<<<1, 1, 0, stream>>>(dXmat, dVout, nl, nr, r, dA, dB);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_als_large(const double* dXmat, const double* dQinv,
                            int nl, int nr, int r, double fudge, int als_iters,
                            double* dA, double* dB,
                            double* dScratch, int* dIntScratch, cudaStream_t stream) {
    als_large_kernel<<<1, 1, 0, stream>>>(dXmat, dQinv, nl, nr, r, fudge, als_iters,
                                          dA, dB, dScratch, dIntScratch);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_weights_chisq_large(const double* dXmat, const double* dQinv,
                                      const double* dA, const double* dB,
                                      int nl, int nr, int r,
                                      double* dW, double* dchisq, int* d_status,
                                      double* dScratch, int* dIntScratch,
                                      cudaStream_t stream) {
    weights_chisq_large_kernel<<<1, 1, 0, stream>>>(dXmat, dQinv, dA, dB, nl, nr, r,
                                                    dW, dchisq, d_status,
                                                    dScratch, dIntScratch);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_loo_large_batched(const double* dLoo, const double* dQinv,
                                    const double* dAseed, const double* dBseed,
                                    int nl, int nr, int r, double fudge, int als_iters,
                                    int nb, int n_models, long dbl_refit, long int_refit,
                                    double* dScratch, int* dIntScratch, double* dWmat,
                                    cudaStream_t stream) {
    const long total = static_cast<long>(nb) * n_models;
    if (total <= 0) return;
    const int block = kBlock64;
    const int grid = core::grid_stride_extent(total, block);
    loo_large_batched_kernel<<<grid, block, 0, stream>>>(
        dLoo, dQinv, dAseed, dBseed, nl, nr, r, fudge, als_iters, nb, n_models,
        dbl_refit, int_refit, dScratch, dIntScratch, dWmat);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

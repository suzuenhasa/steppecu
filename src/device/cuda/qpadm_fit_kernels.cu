// src/device/cuda/qpadm_fit_kernels.cu
//
// qpAdm fit (M(fit-4)) device kernels (the FROZEN CONTRACT §2). The element-wise /
// small-reduction steps of the GPU fit: the S3 f4 4-slab gather (reading the
// RESIDENT f2 tensor in VRAM), the est_to_loo / x_total / tot_line reduction, the
// S4 xtau pseudo-values, and the small symmetrize / fudge-diag helpers the
// cuBLAS/cuSOLVER small-LA in cuda_backend.cu composes with. The dense factor/solve
// / SVD steps (potrf/potri, getrf/getrs[Batched], gesvdj) and the GEMMs live in
// cuda_backend.cu (cuSOLVER/cuBLAS), not here.
//
// PRECISION (§12): every kernel here is NATIVE FP64. The f4 4-slab combine is the
// cancellation-sensitive f-stat difference; the loo/total/xtau reductions reproduce
// the CpuBackend long-double oracle's operation ORDER in FP64 (the parity anchor is
// FP64/long-double; at nb=708, m=10 the FP64 result matches the af6a8c2 golden to
// the gate tier). No Ozaki / emulated lane on this path for the gate.
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). The narrow
// launch wrappers (the only thing cuda_backend.cu calls) are declared in
// qpadm_fit_kernels.cuh; the kernel bodies + <<<>>> are confined here (§7).
#include <cuda_runtime.h>

#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK, STEPPE_CUDA_CHECK_KERNEL
#include "device/cuda/qpadm_fit_kernels.cuh"

namespace steppe::device {

// Compile-time scratch bounds for the small on-device LA (the gate is m=10, nl=2,
// nr=5, r=1, t<=5; these cover typical qpAdm models with generous headroom). A fit
// exceeding them would index out of the fixed local arrays; the host caller asserts
// the model fits before launching (cuda_backend.cu).
namespace {
// Bounds chosen to cover common qpAdm models (left sources nl<=5, right outgroups
// nr<=10, rank r<=4) while keeping the single-thread LA's per-thread LOCAL-MEMORY
// frame small enough to launch (the dominant scratch is Wm[m*t] and coeffs[t*t];
// at these bounds m<=50, t<=40 ⇒ Wm<=50*40=2000 doubles=16 KB, coeffs<=40*40=1600
// doubles=12.5 KB — modest per-thread local memory). CUDA local memory is reserved
// per-thread across the device, so over-large fixed arrays trip
// cudaErrorMemoryAllocation at launch even for a 1-thread kernel; these bounds keep
// it well under that. The golden is nl=2, nr=5, r=1 (m=10, t<=5) — far inside. The
// host caller fails fast if a model exceeds them (cuda_backend.cu qpAdm fit guard).
constexpr int kQpMaxNl = 5;
constexpr int kQpMaxNr = 10;
constexpr int kQpMaxR  = 4;
constexpr int kQpMaxM  = kQpMaxNl * kQpMaxNr;  // 50
constexpr int kQpMaxT  = (kQpMaxNl > kQpMaxNr ? kQpMaxNl : kQpMaxNr) * kQpMaxR;  // max(nl,nr)*r = 40

// --- Device small-dense LA (transliterating small_linalg.hpp, native FP64) --------
// Single-thread helpers used inside the rank-test / ALS / weight / chisq / LOO
// kernels. They reproduce the CpuBackend's exact scalar operations + order so the
// GPU fit is bit-exact vs the FP64 oracle. Column-major A(i,j) at i + n*j.

/// LU factor (partial pivoting) of an n×n column-major `a` (overwritten L\U), piv
/// recorded. Returns true on success, false if a pivot is exactly 0 (singular).
/// Transliterates core::lu_factor (small_linalg.hpp:36-69).
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

/// Solve A·x=b (A n×n column-major, factored into a copy) — core::solve
/// (small_linalg.hpp:74-103). `lu`/`piv`/`y` are caller-provided scratch (lu is a
/// copy of A, mutated). Returns true on success; on singular, false (x untouched).
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

/// One-sided Jacobi SVD of m×n column-major A → V (n×k col-major, k=min(m,n),
/// descending), and the W columns (normalized) are NOT needed by the seed (only V).
/// Transliterates core::jacobi_svd (small_linalg.hpp:162-267). `W`/`Vfull`/`sigma`/
/// `order` are caller scratch. Writes the leading r columns of V into `Vout` (n×r
/// col-major). Native FP64.
__device__ inline void dev_jacobi_svd_V(const double* A, int m, int n, int r,
                                        double* W, double* Vfull, double* sigma,
                                        int* order, double* Vout) {
    for (int i = 0; i < m * n; ++i) W[i] = A[i];
    for (int i = 0; i < n * n; ++i) Vfull[i] = 0.0;
    for (int i = 0; i < n; ++i) Vfull[i + n * i] = 1.0;
    const double kTol = 1e-15;
    const int kMaxSweeps = 60;
    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
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
                if (fabs(gamma) <= kTol * sqrt(alpha * beta) || gamma == 0.0) continue;
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
        if (off < 1e-30) break;
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

/// Seed A,B from svd(xmat) at rank r — core::seed_AB (cpu_backend.cpp:626-644).
/// B = t(V[:,0:r]) (r×nr): B[p,j]=V[j,p]; A = xmat·t(B) (nl×r). Native FP64.
__device__ inline void dev_seed_ab(const double* xmat, int nl, int nr, int r,
                                   double* A, double* B) {
    double W[kQpMaxNl * kQpMaxNr];
    double Vfull[kQpMaxNr * kQpMaxNr];
    double sigma[kQpMaxNr];
    int order[kQpMaxNr];
    double Vout[kQpMaxNr * kQpMaxR];
    dev_jacobi_svd_V(xmat, nl, nr, r, W, Vfull, sigma, order, Vout);
    for (int p = 0; p < r; ++p)
        for (int j = 0; j < nr; ++j)
            B[p + r * j] = Vout[j + nr * p];
    for (int i = 0; i < nl; ++i)
        for (int p = 0; p < r; ++p) {
            double acc = 0.0;
            for (int j = 0; j < nr; ++j)
                acc += xmat[i + nl * j] * B[p + r * j];
            A[i + nl * p] = acc;
        }
}

/// opt_A — core::opt_A (cpu_backend.cpp:652-709). Given B (r×nr), returns A (nl×r).
/// __noinline__ so its large local frame (Wm/coeffs) does NOT stack with opt_B's in
/// the ALS-loop kernels (keeps the per-thread local-memory reservation small enough
/// to launch — see the kQpMax* bound rationale).
__device__ __noinline__ void dev_opt_A(const double* B, const double* xmat,
                                       int nl, int nr, int r, const double* qinv,
                                       double fudge, double* Aout) {
    const int m = nl * nr;
    const int t = nl * r;
    double xvec[kQpMaxM];
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j)
            xvec[i * nr + j] = xmat[i + nl * j];
    // B2(a,k): a=i*r+p, k=ii*nr+j ; (i==ii)?B[p,j]:0.
    auto B2 = [&](int a, int k) -> double {
        const int i = a / r, p = a % r;
        const int ii = k / nr, j = k % nr;
        return (i == ii) ? B[p + r * j] : 0.0;
    };
    double Wm[kQpMaxM * kQpMaxT];  // m×t
    for (int kr = 0; kr < m; ++kr)
        for (int a = 0; a < t; ++a) {
            double acc = 0.0;
            for (int kc = 0; kc < m; ++kc) acc += qinv[kr + m * kc] * B2(a, kc);
            Wm[kr + m * a] = acc;
        }
    double coeffs[kQpMaxT * kQpMaxT];
    double rhs[kQpMaxT];
    for (int a = 0; a < t; ++a) {
        for (int c = 0; c < t; ++c) {
            double acc = 0.0;
            for (int k = 0; k < m; ++k) acc += B2(a, k) * Wm[k + m * c];
            coeffs[a + t * c] = acc;
        }
        double rr = 0.0;
        for (int k = 0; k < m; ++k) rr += xvec[k] * Wm[k + m * a];
        rhs[a] = rr;
    }
    double tr = 0.0;
    for (int a = 0; a < t; ++a) tr += coeffs[a + t * a];
    for (int a = 0; a < t; ++a) coeffs[a + t * a] += fudge * tr;
    double A2[kQpMaxT], lu[kQpMaxT * kQpMaxT], y[kQpMaxT];
    int piv[kQpMaxT];
    for (int i = 0; i < nl * r; ++i) Aout[i] = 0.0;
    if (dev_solve(coeffs, t, rhs, A2, lu, piv, y)) {
        for (int i = 0; i < nl; ++i)
            for (int p = 0; p < r; ++p)
                Aout[i + nl * p] = A2[i * r + p];
    }
}

/// opt_B — core::opt_B (cpu_backend.cpp:715-768). Given A (nl×r), returns B (r×nr).
/// __noinline__ (see dev_opt_A) so its frame does not stack with opt_A's.
__device__ __noinline__ void dev_opt_B(const double* A, const double* xmat,
                                       int nl, int nr, int r, const double* qinv,
                                       double fudge, double* Bout) {
    const int m = nl * nr;
    const int t = r * nr;
    double xvec[kQpMaxM];
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j)
            xvec[i * nr + j] = xmat[i + nl * j];
    // A2(k,c): k=i*nr+j, c=p*nr+jc ; (j==jc)?A[i,p]:0.
    auto A2f = [&](int k, int c) -> double {
        const int i = k / nr, j = k % nr;
        const int p = c / nr, jc = c % nr;
        return (j == jc) ? A[i + nl * p] : 0.0;
    };
    double Wm[kQpMaxM * kQpMaxT];
    for (int kr = 0; kr < m; ++kr)
        for (int c = 0; c < t; ++c) {
            double acc = 0.0;
            for (int kc = 0; kc < m; ++kc) acc += qinv[kr + m * kc] * A2f(kc, c);
            Wm[kr + m * c] = acc;
        }
    double coeffs[kQpMaxT * kQpMaxT];
    double rhs[kQpMaxT];
    for (int a = 0; a < t; ++a) {
        for (int c = 0; c < t; ++c) {
            double acc = 0.0;
            for (int k = 0; k < m; ++k) acc += A2f(k, a) * Wm[k + m * c];
            coeffs[a + t * c] = acc;
        }
        double rr = 0.0;
        for (int k = 0; k < m; ++k) rr += xvec[k] * Wm[k + m * a];
        rhs[a] = rr;
    }
    double tr = 0.0;
    for (int a = 0; a < t; ++a) tr += coeffs[a + t * a];
    for (int a = 0; a < t; ++a) coeffs[a + t * a] += fudge * tr;
    double B2v[kQpMaxT], lu[kQpMaxT * kQpMaxT], y[kQpMaxT];
    int piv[kQpMaxT];
    for (int i = 0; i < r * nr; ++i) Bout[i] = 0.0;
    if (dev_solve(coeffs, t, rhs, B2v, lu, piv, y)) {
        for (int p = 0; p < r; ++p)
            for (int j = 0; j < nr; ++j)
                Bout[p + r * j] = B2v[p * nr + j];
    }
}

/// chisq = vec(E)'·qinv·vec(E), E = xmat - A·B, row-major vec — core::chisq_of
/// (cpu_backend.cpp:833-858). Native FP64 (CpuBackend uses long double; FP64 matches
/// the golden chisq to the gate tier).
__device__ inline double dev_chisq_of(const double* xmat, const double* A,
                                      const double* B, int nl, int nr, int r,
                                      const double* qinv) {
    const int m = nl * nr;
    double e[kQpMaxM];
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

/// Full als_weights body — core::als_weights (cpu_backend.cpp:773-826). Refines A,B
/// (seeded externally if r>0), runs the constrained weight solve, normalizes Σw=1,
/// computes chisq. Returns 0=Ok / 6=RankDeficient. `A`/`B` are caller scratch
/// (refined in place). When `seed`==true, seeds A,B from svd(xmat) first.
__device__ inline int dev_als_weights(const double* xmat, int nl, int nr, int r,
                                      const double* qinv, double fudge, int als_iters,
                                      bool seed, double* A, double* B,
                                      double* w_out, double* chisq_out) {
    if (r == 0) {
        for (int i = 0; i < nl; ++i) w_out[i] = 1.0;
        *chisq_out = dev_chisq_of(xmat, A, B, nl, nr, 0, qinv);
        return 0;
    }
    if (seed) dev_seed_ab(xmat, nl, nr, r, A, B);
    double Atmp[kQpMaxT], Btmp[kQpMaxT];
    for (int it = 0; it < als_iters; ++it) {
        dev_opt_A(B, xmat, nl, nr, r, qinv, fudge, Atmp);
        for (int i = 0; i < nl * r; ++i) A[i] = Atmp[i];
        dev_opt_B(A, xmat, nl, nr, r, qinv, fudge, Btmp);
        for (int i = 0; i < r * nr; ++i) B[i] = Btmp[i];
    }
    // Constrained weight solve: xm(i,p)=A(i,p) for p<r, xm(i,r)=1.
    const int rp = r + 1;
    auto xm = [&](int i, int p) -> double { return (p < r) ? A[i + nl * p] : 1.0; };
    double RHS[kQpMaxNl * kQpMaxNl];
    double LHS[kQpMaxNl];
    for (int i = 0; i < nl; ++i) {
        for (int ip = 0; ip < nl; ++ip) {
            double acc = 0.0;
            for (int p = 0; p < rp; ++p) acc += xm(i, p) * xm(ip, p);
            RHS[i + nl * ip] = acc;
        }
        LHS[i] = xm(i, r);  // = 1
    }
    double wv[kQpMaxNl], lu[kQpMaxNl * kQpMaxNl], y[kQpMaxNl];
    int piv[kQpMaxNl];
    if (!dev_solve(RHS, nl, LHS, wv, lu, piv, y)) return 6;  // RankDeficient
    double sum = 0.0;
    for (int i = 0; i < nl; ++i) sum += wv[i];
    for (int i = 0; i < nl; ++i) w_out[i] = wv[i] / sum;
    *chisq_out = dev_chisq_of(xmat, A, B, nl, nr, r, qinv);
    return 0;
}

// --- S3 f4-gather kernel ---------------------------------------------------------
// One thread per (k, b), k = j + nr*i in 0..m-1, b in 0..nb-1. Reads the resident
// f2 tensor (column-major i + P*j + P*P*b) and writes dX[k + m*b]. Native FP64; the
// 4-slab subtraction is the cancellation-sensitive f-stat difference (§12).
__global__ void assemble_f4_gather_kernel(const double* __restrict__ f2, int P,
                                          const int* __restrict__ d_left,
                                          const int* __restrict__ d_right,
                                          int nl, int nr, int nb,
                                          double* __restrict__ dX) {
    const int m = nl * nr;
    const long total = static_cast<long>(m) * static_cast<long>(nb);
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<long>(gridDim.x) * blockDim.x) {
        const int k = static_cast<int>(idx % m);
        const int b = static_cast<int>(idx / m);
        const int i = k / nr;  // left source index (0..nl-1)
        const int j = k % nr;  // right index       (0..nr-1)
        const int L0 = d_left[0];
        const int R0 = d_right[0];
        const int Li = d_left[i + 1];
        const int Rj = d_right[j + 1];
        const long slab = static_cast<long>(P) * static_cast<long>(P) * b;
        const auto at = [&](int a, int c) -> double {
            return f2[static_cast<long>(a) + static_cast<long>(P) * c + slab];
        };
        const double x = 0.5 * (at(Li, R0) + at(L0, Rj) - at(L0, R0) - at(Li, Rj));
        dX[idx] = x;
    }
}

// --- S3 est_to_loo + x_total + tot_line ------------------------------------------
// One thread per k (m is tiny — m=10 at the golden). Reproduces CpuBackend
// compute_loo_and_total cpu_backend.cpp:553-592 in FP64 (the long-double accumulators
// become FP64). The CpuBackend term1 = mean(tot_line - loo)*nb = (Σ (tot_line-loo)/nb)*nb
// = Σ (tot_line - loo) exactly; reproduced as the bare sum.
__global__ void f4_loo_total_kernel(const double* __restrict__ dX,
                                    const int* __restrict__ d_block_sizes,
                                    int m, int nb, double n,
                                    double* __restrict__ dLoo,
                                    double* __restrict__ dTotal,
                                    double* __restrict__ dTotLine) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= m) return;
    // tot_ij = (Σ_b X[k,b]*bl_b) / n.
    double num = 0.0;
    for (int b = 0; b < nb; ++b)
        num += dX[k + static_cast<long>(m) * b] *
               static_cast<double>(d_block_sizes[b]);
    const double tot_ij = num / n;
    // loo[k,b] = (tot_ij - X*rel_b)/(1-rel_b).
    for (int b = 0; b < nb; ++b) {
        const double bl = static_cast<double>(d_block_sizes[b]);
        const double rel = bl / n;
        const double xv = dX[k + static_cast<long>(m) * b];
        dLoo[k + static_cast<long>(m) * b] = (tot_ij - xv * rel) / (1.0 - rel);
    }
    // tot_line[k] = weighted.mean(loo[k,:], 1 - bl/n).
    double wln = 0.0, wld = 0.0;
    for (int b = 0; b < nb; ++b) {
        const double w = 1.0 - static_cast<double>(d_block_sizes[b]) / n;
        wln += dLoo[k + static_cast<long>(m) * b] * w;
        wld += w;
    }
    const double tot_line = wln / wld;
    dTotLine[k] = tot_line;
    // est[k] = Σ_b (tot_line - loo) + (Σ_b loo*bl)/n.
    double diffsum = 0.0, wbn = 0.0;
    for (int b = 0; b < nb; ++b) {
        const double loo = dLoo[k + static_cast<long>(m) * b];
        diffsum += tot_line - loo;
        wbn += loo * static_cast<double>(d_block_sizes[b]);
    }
    dTotal[k] = diffsum + wbn / n;
}

// --- S4 xtau pseudo-values -------------------------------------------------------
// One thread per (k,b). xtau[k,b] = (est[k]*h - loo[k,b]*(h-1) - tot_line[k]) / sh,
// h = n/bl_b, sh = sqrt(h-1). Column-major (k + m*b) for the SYRK. Native FP64.
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
        const double h = n / bl;
        const double sh = sqrt(h - 1.0);
        const double loo = dLoo[idx];
        dXtau[idx] = (dEst[k] * h - loo * (h - 1.0) - dTotLine[k]) / sh;
    }
}

// --- symmetrize lower -> full (in place) -----------------------------------------
__global__ void symmetrize_lower_to_full_kernel(double* __restrict__ dM, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // row
    const int j = blockIdx.y * blockDim.y + threadIdx.y;  // col
    if (i < n && j < n && i > j) {
        // copy lower (i,j) into upper (j,i): col-major idx = row + n*col.
        dM[j + static_cast<long>(n) * i] = dM[i + static_cast<long>(n) * j];
    }
}

// --- add fudge*tr to the diagonal (in place) -------------------------------------
__global__ void add_fudge_diag_kernel(double* __restrict__ dM, int n,
                                      double fudge, double tr) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) dM[k + static_cast<long>(n) * k] += fudge * tr;
}

// --- xmat from row-major slice ---------------------------------------------------
__global__ void xmat_from_rowmajor_kernel(const double* __restrict__ src,
                                          int nl, int nr, double* __restrict__ dXmat) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j)
            dXmat[i + nl * j] = src[j + nr * i];
}

// --- S5 seed (single thread) -----------------------------------------------------
__global__ void seed_ab_kernel(const double* __restrict__ dXmat, int nl, int nr,
                               int r, double* __restrict__ dA, double* __restrict__ dB) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (r <= 0) return;
    dev_seed_ab(dXmat, nl, nr, r, dA, dB);
}

// --- S6 ALS opt_A/opt_B loop (single thread; seeds already in dA/dB) -------------
__global__ void als_kernel(const double* __restrict__ dXmat, const double* __restrict__ dQinv,
                           int nl, int nr, int r, double fudge, int als_iters,
                           double* __restrict__ dA, double* __restrict__ dB) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (r <= 0) return;
    double A[kQpMaxT], B[kQpMaxT], Atmp[kQpMaxT], Btmp[kQpMaxT];
    for (int i = 0; i < nl * r; ++i) A[i] = dA[i];
    for (int i = 0; i < r * nr; ++i) B[i] = dB[i];
    for (int it = 0; it < als_iters; ++it) {
        dev_opt_A(B, dXmat, nl, nr, r, dQinv, fudge, Atmp);
        for (int i = 0; i < nl * r; ++i) A[i] = Atmp[i];
        dev_opt_B(A, dXmat, nl, nr, r, dQinv, fudge, Btmp);
        for (int i = 0; i < r * nr; ++i) B[i] = Btmp[i];
    }
    for (int i = 0; i < nl * r; ++i) dA[i] = A[i];
    for (int i = 0; i < r * nr; ++i) dB[i] = B[i];
}

// --- S6 weight solve + chisq (single thread) -------------------------------------
__global__ void weights_chisq_kernel(const double* __restrict__ dXmat,
                                     const double* __restrict__ dQinv,
                                     const double* __restrict__ dA,
                                     const double* __restrict__ dB,
                                     int nl, int nr, int r,
                                     double* __restrict__ dW,
                                     double* __restrict__ dchisq,
                                     int* __restrict__ d_status) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double A[kQpMaxT], B[kQpMaxT];
    for (int i = 0; i < nl * r; ++i) A[i] = dA[i];
    for (int i = 0; i < r * nr; ++i) B[i] = dB[i];
    if (r == 0) {
        for (int i = 0; i < nl; ++i) dW[i] = 1.0;
        *dchisq = dev_chisq_of(dXmat, A, B, nl, nr, 0, dQinv);
        *d_status = 0;
        return;
    }
    const int rp = r + 1;
    auto xm = [&](int i, int p) -> double { return (p < r) ? A[i + nl * p] : 1.0; };
    double RHS[kQpMaxNl * kQpMaxNl], LHS[kQpMaxNl];
    for (int i = 0; i < nl; ++i) {
        for (int ip = 0; ip < nl; ++ip) {
            double acc = 0.0;
            for (int p = 0; p < rp; ++p) acc += xm(i, p) * xm(ip, p);
            RHS[i + nl * ip] = acc;
        }
        LHS[i] = xm(i, r);
    }
    double wv[kQpMaxNl], lu[kQpMaxNl * kQpMaxNl], y[kQpMaxNl];
    int piv[kQpMaxNl];
    if (!dev_solve(RHS, nl, LHS, wv, lu, piv, y)) { *d_status = 6; return; }
    double sum = 0.0;
    for (int i = 0; i < nl; ++i) sum += wv[i];
    for (int i = 0; i < nl; ++i) dW[i] = wv[i] / sum;
    *dchisq = dev_chisq_of(dXmat, A, B, nl, nr, r, dQinv);
    *d_status = 0;
}

// --- S7 batched LOO re-fits (one thread per replicate block) ---------------------
__global__ void loo_batched_kernel(const double* __restrict__ dLoo,
                                   const double* __restrict__ dQinv,
                                   int nl, int nr, int r, double fudge, int als_iters,
                                   int nb, double* __restrict__ dWmat) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const int m = nl * nr;
    // xmat_b from x_loo[:,:,b]: row-major k=j+nr*i ⇒ xmat(i,j) at i+nl*j.
    double xmat[kQpMaxM];
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j)
            xmat[i + nl * j] = dLoo[(j + nr * i) + static_cast<long>(m) * b];
    double A[kQpMaxT], B[kQpMaxT], w[kQpMaxNl], chisq;
    const int st = dev_als_weights(xmat, nl, nr, r, dQinv, fudge, als_iters,
                                   /*seed=*/true, A, B, w, &chisq);
    for (int i = 0; i < nl; ++i)
        dWmat[static_cast<long>(b) * nl + i] = (st == 0) ? w[i] : 0.0;
}

}  // namespace

void launch_assemble_f4_gather(const double* f2, int P,
                               const int* d_left, const int* d_right,
                               int nl, int nr, int nb,
                               double* dX, cudaStream_t stream) {
    const long total = static_cast<long>(nl) * nr * nb;
    if (total <= 0) return;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    assemble_f4_gather_kernel<<<grid, block, 0, stream>>>(f2, P, d_left, d_right,
                                                          nl, nr, nb, dX);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_f4_loo_total(const double* dX, const int* d_block_sizes,
                         int m, int nb, double n,
                         double* dLoo, double* dTotal, double* dTotLine,
                         cudaStream_t stream) {
    if (m <= 0 || nb <= 0) return;
    const int block = 64;
    const int grid = (m + block - 1) / block;
    f4_loo_total_kernel<<<grid, block, 0, stream>>>(dX, d_block_sizes, m, nb, n,
                                                    dLoo, dTotal, dTotLine);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_f4_xtau(const double* dLoo, const double* dEst, const double* dTotLine,
                    const int* d_block_sizes, int m, int nb, double n,
                    double* dXtau, cudaStream_t stream) {
    const long total = static_cast<long>(m) * nb;
    if (total <= 0) return;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    f4_xtau_kernel<<<grid, block, 0, stream>>>(dLoo, dEst, dTotLine, d_block_sizes,
                                               m, nb, n, dXtau);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_symmetrize_lower_to_full(double* dM, int n, cudaStream_t stream) {
    if (n <= 0) return;
    const dim3 block(16, 16);
    const dim3 grid((static_cast<unsigned>(n) + 15) / 16,
                    (static_cast<unsigned>(n) + 15) / 16);
    symmetrize_lower_to_full_kernel<<<grid, block, 0, stream>>>(dM, n);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_add_fudge_diag(double* dM, int n, double fudge, double tr,
                           cudaStream_t stream) {
    if (n <= 0) return;
    const int block = 64;
    const int grid = (n + block - 1) / block;
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
    const int block = 64;
    const int grid = (nb + block - 1) / block;
    loo_batched_kernel<<<grid, block, 0, stream>>>(dLoo, dQinv, nl, nr, r, fudge,
                                                   als_iters, nb, dWmat);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

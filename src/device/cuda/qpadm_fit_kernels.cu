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

// NOTE (the small-bound envelope): these stacked-LOCAL templated kernels serve ONLY
// the bit-parity SMALL path (nl<=5, nr<=10, r<=4 — the 9-pop golden is far inside).
// CUDA reserves a kernel's per-thread LOCAL memory for the device's MAX resident-
// thread count, so a large per-thread frame (Wm[m*t] grows with nr) trips
// cudaErrorMemoryAllocation at launch even single-threaded (measured on box5090:
// kQpBigNr=40 ⇒ OOM). So a model exceeding the envelope (e.g. NRBIG nr=39) does NOT
// run on these kernels — it runs on the LARGE path (the cuSOLVER SVD `large_svd_V`
// in cuda_backend.cu + the VRAM-scratch `*_large` kernels at the bottom of this file,
// the FROZEN CONTRACT §1/§2). CudaBackend dispatches on `model_fits_small_path`. The
// nr>32 path is no longer a PENDING seam — it RUNS ON THE GPU (fit-engine.md §1.4).

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
/// Templated on the per-thread local-array bound (MAXNL/MAXNR/MAXR): the single-
/// thread sweep kernels instantiate it at a BIG bound (nr>32 fallback) while the
/// many-thread LOO batched kernel uses the SMALL bound — so widening the sweep's
/// scratch never bloats the batched kernel's per-thread local memory (the launch
/// constraint; the kQpMax* rationale at the top of this file).
template <int MAXNL, int MAXNR, int MAXR>
__device__ inline void dev_seed_ab(const double* xmat, int nl, int nr, int r,
                                   double* A, double* B) {
    double W[MAXNL * MAXNR];
    double Vfull[MAXNR * MAXNR];
    double sigma[MAXNR];
    int order[MAXNR];
    double Vout[MAXNR * MAXR];
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
/// to launch — see the kQpMax* bound rationale). Templated on the local-array bound
/// (MAXM/MAXT) so the single-thread sweep can use a big bound while the batched LOO
/// kernel keeps the small one.
template <int MAXM, int MAXT>
__device__ __noinline__ void dev_opt_A(const double* B, const double* xmat,
                                       int nl, int nr, int r, const double* qinv,
                                       double fudge, double* Aout) {
    const int m = nl * nr;
    const int t = nl * r;
    double xvec[MAXM];
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j)
            xvec[i * nr + j] = xmat[i + nl * j];
    // B2(a,k): a=i*r+p, k=ii*nr+j ; (i==ii)?B[p,j]:0.
    auto B2 = [&](int a, int k) -> double {
        const int i = a / r, p = a % r;
        const int ii = k / nr, j = k % nr;
        return (i == ii) ? B[p + r * j] : 0.0;
    };
    double Wm[MAXM * MAXT];  // m×t
    for (int kr = 0; kr < m; ++kr)
        for (int a = 0; a < t; ++a) {
            double acc = 0.0;
            for (int kc = 0; kc < m; ++kc) acc += qinv[kr + m * kc] * B2(a, kc);
            Wm[kr + m * a] = acc;
        }
    double coeffs[MAXT * MAXT];
    double rhs[MAXT];
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
    double A2[MAXT], lu[MAXT * MAXT], y[MAXT];
    int piv[MAXT];
    for (int i = 0; i < nl * r; ++i) Aout[i] = 0.0;
    if (dev_solve(coeffs, t, rhs, A2, lu, piv, y)) {
        for (int i = 0; i < nl; ++i)
            for (int p = 0; p < r; ++p)
                Aout[i + nl * p] = A2[i * r + p];
    }
}

/// opt_B — core::opt_B (cpu_backend.cpp:715-768). Given A (nl×r), returns B (r×nr).
/// __noinline__ (see dev_opt_A) so its frame does not stack with opt_A's. Templated
/// on the local-array bound (MAXM/MAXT), see dev_opt_A.
template <int MAXM, int MAXT>
__device__ __noinline__ void dev_opt_B(const double* A, const double* xmat,
                                       int nl, int nr, int r, const double* qinv,
                                       double fudge, double* Bout) {
    const int m = nl * nr;
    const int t = r * nr;
    double xvec[MAXM];
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j)
            xvec[i * nr + j] = xmat[i + nl * j];
    // A2(k,c): k=i*nr+j, c=p*nr+jc ; (j==jc)?A[i,p]:0.
    auto A2f = [&](int k, int c) -> double {
        const int i = k / nr, j = k % nr;
        const int p = c / nr, jc = c % nr;
        return (j == jc) ? A[i + nl * p] : 0.0;
    };
    double Wm[MAXM * MAXT];
    for (int kr = 0; kr < m; ++kr)
        for (int c = 0; c < t; ++c) {
            double acc = 0.0;
            for (int kc = 0; kc < m; ++kc) acc += qinv[kr + m * kc] * A2f(kc, c);
            Wm[kr + m * c] = acc;
        }
    double coeffs[MAXT * MAXT];
    double rhs[MAXT];
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
    double B2v[MAXT], lu[MAXT * MAXT], y[MAXT];
    int piv[MAXT];
    for (int i = 0; i < r * nr; ++i) Bout[i] = 0.0;
    if (dev_solve(coeffs, t, rhs, B2v, lu, piv, y)) {
        for (int p = 0; p < r; ++p)
            for (int j = 0; j < nr; ++j)
                Bout[p + r * j] = B2v[p * nr + j];
    }
}

/// chisq = vec(E)'·qinv·vec(E), E = xmat - A·B, row-major vec — core::chisq_of
/// (cpu_backend.cpp:833-858). Native FP64 (CpuBackend uses long double; FP64 matches
/// the golden chisq to the gate tier). Templated on MAXM (the residual-vector bound).
template <int MAXM>
__device__ inline double dev_chisq_of(const double* xmat, const double* A,
                                      const double* B, int nl, int nr, int r,
                                      const double* qinv) {
    const int m = nl * nr;
    double e[MAXM];
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
/// (refined in place). When `seed`==true, seeds A,B from svd(xmat) first. Templated
/// on the local-array bounds (MAXNL/MAXNR/MAXR ⇒ MAXM/MAXT) so the single-thread
/// sweep and the many-thread batched LOO can pick different per-thread budgets.
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
    // Constrained weight solve: xm(i,p)=A(i,p) for p<r, xm(i,r)=1.
    const int rp = r + 1;
    auto xm = [&](int i, int p) -> double { return (p < r) ? A[i + nl * p] : 1.0; };
    double RHS[MAXNL * MAXNL];
    double LHS[MAXNL];
    for (int i = 0; i < nl; ++i) {
        for (int ip = 0; ip < nl; ++ip) {
            double acc = 0.0;
            for (int p = 0; p < rp; ++p) acc += xm(i, p) * xm(ip, p);
            RHS[i + nl * ip] = acc;
        }
        LHS[i] = xm(i, r);  // = 1
    }
    double wv[MAXNL], lu[MAXNL * MAXNL], y[MAXNL];
    int piv[MAXNL];
    if (!dev_solve(RHS, nl, LHS, wv, lu, piv, y)) return 6;  // RankDeficient
    double sum = 0.0;
    for (int i = 0; i < nl; ++i) sum += wv[i];
    for (int i = 0; i < nl; ++i) w_out[i] = wv[i] / sum;
    *chisq_out = dev_chisq_of<MAXM>(xmat, A, B, nl, nr, r, qinv);
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
    dev_seed_ab<kQpMaxNl, kQpMaxNr, kQpMaxR>(dXmat, nl, nr, r, dA, dB);
}

// --- S6 ALS opt_A/opt_B loop (single thread; seeds already in dA/dB) -------------
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
    *dchisq = dev_chisq_of<MAXM>(dXmat, A, B, nl, nr, r, dQinv);
    *d_status = 0;
}

// ---------------------------------------------------------------------------------
// LARGE-path device helpers (the FROZEN CONTRACT §2.2): VRAM-scratch versions of
// opt_A / opt_B / chisq / seed-from-V. SAME math + SAME op order as the templated
// small kernels above, but every per-model working array is a `double*` into caller-
// provided VRAM (so arbitrary nl/nr/m/r fit — no per-thread local frame to OOM at
// launch). Single thread per model. Native FP64. The scratch layout (offsets the
// launch wrapper precomputes) is documented at each call.
// ---------------------------------------------------------------------------------

/// opt_A (large) — core::opt_A (cpu_backend.cpp:652-709) with VRAM scratch. B (r×nr)
/// in, A (nl×r) out. Scratch: xvec[m], Wm[m*t], coeffs[t*t], rhs[t], A2[t], lu[t*t],
/// y[t]; ipiv[t] (int). t = nl*r. Same op order as dev_opt_A.
__device__ inline void dev_opt_A_large(const double* B, const double* xmat,
                                       int nl, int nr, int r, const double* qinv,
                                       double fudge, double* Aout,
                                       double* xvec, double* Wm, double* coeffs,
                                       double* rhs, double* A2, double* lu, double* y,
                                       int* ipiv) {
    const int m = nl * nr;
    const int t = nl * r;
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j)
            xvec[i * nr + j] = xmat[i + nl * j];
    auto B2 = [&](int a, int k) -> double {
        const int i = a / r, p = a % r;
        const int ii = k / nr, j = k % nr;
        return (i == ii) ? B[p + r * j] : 0.0;
    };
    for (int kr = 0; kr < m; ++kr)
        for (int a = 0; a < t; ++a) {
            double acc = 0.0;
            for (int kc = 0; kc < m; ++kc) acc += qinv[kr + m * kc] * B2(a, kc);
            Wm[kr + m * a] = acc;
        }
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
    for (int i = 0; i < nl * r; ++i) Aout[i] = 0.0;
    if (dev_solve(coeffs, t, rhs, A2, lu, ipiv, y)) {
        for (int i = 0; i < nl; ++i)
            for (int p = 0; p < r; ++p)
                Aout[i + nl * p] = A2[i * r + p];
    }
}

/// opt_B (large) — core::opt_B (cpu_backend.cpp:715-768) with VRAM scratch. A (nl×r)
/// in, B (r×nr) out. Scratch as dev_opt_A_large; t = r*nr.
__device__ inline void dev_opt_B_large(const double* A, const double* xmat,
                                       int nl, int nr, int r, const double* qinv,
                                       double fudge, double* Bout,
                                       double* xvec, double* Wm, double* coeffs,
                                       double* rhs, double* B2v, double* lu, double* y,
                                       int* ipiv) {
    const int m = nl * nr;
    const int t = r * nr;
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j)
            xvec[i * nr + j] = xmat[i + nl * j];
    auto A2f = [&](int k, int c) -> double {
        const int i = k / nr, j = k % nr;
        const int p = c / nr, jc = c % nr;
        return (j == jc) ? A[i + nl * p] : 0.0;
    };
    for (int kr = 0; kr < m; ++kr)
        for (int c = 0; c < t; ++c) {
            double acc = 0.0;
            for (int kc = 0; kc < m; ++kc) acc += qinv[kr + m * kc] * A2f(kc, c);
            Wm[kr + m * c] = acc;
        }
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
    for (int i = 0; i < r * nr; ++i) Bout[i] = 0.0;
    if (dev_solve(coeffs, t, rhs, B2v, lu, ipiv, y)) {
        for (int p = 0; p < r; ++p)
            for (int j = 0; j < nr; ++j)
                Bout[p + r * j] = B2v[p * nr + j];
    }
}

/// chisq (large) — core::chisq_of (cpu_backend.cpp:833-858) with VRAM scratch e[m].
__device__ inline double dev_chisq_of_large(const double* xmat, const double* A,
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

// --- LARGE-path transpose (nl×nr col-major xmat -> nr×nl col-major Xt) ------------
__global__ void transpose_small_kernel(const double* __restrict__ dXmat,
                                       int nl, int nr, double* __restrict__ dXt) {
    const long total = static_cast<long>(nl) * static_cast<long>(nr);
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<long>(gridDim.x) * blockDim.x) {
        const int i = static_cast<int>(idx % nl);   // xmat row    (0..nl-1)
        const int j = static_cast<int>(idx / nl);   // xmat col    (0..nr-1)
        dXt[j + static_cast<long>(nr) * i] = dXmat[i + static_cast<long>(nl) * j];
    }
}

// --- LARGE-path seed from V[:,0:r] (single thread) -------------------------------
// dVout is nr×r col-major (the leading r right singular vectors, descending). B = t(V):
// B[p,j] = V[j,p]; A = xmat·t(B). No SVD here (cuSOLVER did it on the host).
__global__ void seed_from_V_kernel(const double* __restrict__ dXmat,
                                   const double* __restrict__ dVout,
                                   int nl, int nr, int r,
                                   double* __restrict__ dA, double* __restrict__ dB) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (r <= 0) return;
    for (int p = 0; p < r; ++p)
        for (int j = 0; j < nr; ++j)
            dB[p + r * j] = dVout[j + static_cast<long>(nr) * p];
    for (int i = 0; i < nl; ++i)
        for (int p = 0; p < r; ++p) {
            double acc = 0.0;
            for (int j = 0; j < nr; ++j)
                acc += dXmat[i + static_cast<long>(nl) * j] * dB[p + r * j];
            dA[i + static_cast<long>(nl) * p] = acc;
        }
}

// --- LARGE-path ALS opt_A/opt_B loop (single thread; VRAM scratch) ---------------
// Scratch layout (dScratch, all double): t = max(nl,nr)*r is the wide upper bound for
// opt_A's t=nl*r AND opt_B's t=r*nr (both ≤ max(nl,nr)*r), so one layout serves both.
//   xvec[m] | Wm[m*t] | coeffs[t*t] | rhs[t] | A2/B2[t] | lu[t*t] | y[t]
// dIntScratch: ipiv[t].
__global__ void als_large_kernel(const double* __restrict__ dXmat,
                                 const double* __restrict__ dQinv,
                                 int nl, int nr, int r, double fudge, int als_iters,
                                 double* __restrict__ dA, double* __restrict__ dB,
                                 double* __restrict__ dScratch,
                                 int* __restrict__ dIntScratch) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (r <= 0) return;
    const int m = nl * nr;
    const int t = (nl > nr ? nl : nr) * r;  // wide bound covering nl*r and r*nr
    double* xvec   = dScratch;
    double* Wm     = xvec + m;
    double* coeffs = Wm + static_cast<long>(m) * t;
    double* rhs    = coeffs + static_cast<long>(t) * t;
    double* tmp    = rhs + t;        // A2 / B2v
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

// --- LARGE-path weight solve + chisq (single thread; VRAM scratch) ---------------
// Scratch layout (dScratch): RHS[nl*nl] | wv[nl] | lu[nl*nl] | y[nl] | e[m].
// dIntScratch: piv[nl].
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
    const int rp = r + 1;
    auto xm = [&](int i, int p) -> double { return (p < r) ? dA[i + nl * p] : 1.0; };
    for (int i = 0; i < nl; ++i) {
        for (int ip = 0; ip < nl; ++ip) {
            double acc = 0.0;
            for (int p = 0; p < rp; ++p) acc += xm(i, p) * xm(ip, p);
            RHS[i + nl * ip] = acc;
        }
    }
    // LHS = ones (xm(i,r) == 1); solve RHS·wv = 1.
    double* LHS = e;  // reuse e as the nl-length RHS vector (e has m >= nl slots)
    for (int i = 0; i < nl; ++i) LHS[i] = 1.0;
    if (!dev_solve(RHS, nl, LHS, wv, lu, piv, y)) { *d_status = 6; return; }
    double sum = 0.0;
    for (int i = 0; i < nl; ++i) sum += wv[i];
    for (int i = 0; i < nl; ++i) dW[i] = wv[i] / sum;
    *dchisq = dev_chisq_of_large(dXmat, dA, dB, nl, nr, r, dQinv, e);
    *d_status = 0;
}

// --- LARGE-path PARALLEL LOO re-fits (one thread per (model, block)) --------------
// The large-model jackknife SE was the throughput wall: the host ran nb (=701 for
// NRBIG) leave-one-block-out refits SERIALLY, each a cuSOLVER SVD seed + a single-
// thread ALS chain + weight solve, with a per-block host round-trip. The refits are
// INDEPENDENT, so this kernel runs all nb (and the future B·nb) refits CONCURRENTLY:
// one thread per (model, block), each reusing the SAME `*_large` device helpers as the
// serial path — only WHERE the scratch lives changes (a per-thread slice of a runtime-
// sized VRAM arena instead of one single-model buffer) and that the loop is parallel.
//
// PARITY (bit-identical SE): the SVD seed is precomputed on the host (Stage A) with the
// SAME cuSOLVER gesvd `large_svd_V`+`seed_from_V` per block (it is NOT recomputed here —
// an on-device Jacobi seed would shift the ALS fixed point in the LSBs). Each thread
// copies its dAseed/dBseed slice, then runs the EXACT als_large + weights_chisq_large
// math in the SAME op order ⇒ each refit's normalized weights are BIT-IDENTICAL to the
// serial path's. chisq is dead output here (gls_weights_loo_batched discards it) ⇒ the
// dev_chisq_of_large call is skipped (an omission of unused output, not a math change).
//
// Per-thread scratch (dScratch slice, stride = `dbl_refit` the host sizer returns):
//   xmat[m] | A[nl*r] | B[r*nr] | union[large_dbl_scratch(nl,nr,r)]
// where `union` is the SAME max(als,wc) layout als_large_kernel / weights_chisq_large_
// kernel carve (t = max(nl,nr)*r). dIntScratch slice stride = `int_refit` =
// large_int_scratch (max(t,nl) pivots). Writes UNSCALED w to dWmat (status!=0 ⇒ zeros,
// matching the serial path's zero-init + status guard). One model here (n_models=1 for
// NRBIG); the model axis is the S8 batching seam.
__global__ void loo_large_batched_kernel(const double* __restrict__ dLoo,
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
        // This thread's scratch slice (no overlap across threads ⇒ race-free).
        double* base = dScratch + gid * dbl_refit;
        int*    ibase = dIntScratch + gid * int_refit;
        double* xmat = base;                                   // m
        double* A    = xmat + m;                               // nl*r
        double* B    = A + static_cast<long>(nl) * (r > 0 ? r : 1);   // r*nr
        double* sc   = B + static_cast<long>(r > 0 ? r : 1) * nr;     // union scratch
        // xmat_b from loo[:,:,b]: row-major k=j+nr*i ⇒ xmat(i,j) at i+nl*j (same reshape
        // as the serial xmat_from_rowmajor + the small loo_batched_kernel).
        for (int i = 0; i < nl; ++i)
            for (int j = 0; j < nr; ++j)
                xmat[i + nl * j] = loo[(j + nr * i) + static_cast<long>(m) * b];
        // ---- the ALS opt_A/opt_B loop (seeded from this block's dAseed/dBseed) -------
        if (r > 0) {
            const double* As = dAseed + gid * static_cast<long>(nl) * r;
            const double* Bs = dBseed + gid * static_cast<long>(r) * nr;
            for (int i = 0; i < nl * r; ++i) A[i] = As[i];
            for (int i = 0; i < r * nr; ++i) B[i] = Bs[i];
            // SAME union layout als_large_kernel carves (t = max(nl,nr)*r).
            const int t = (nl > nr ? nl : nr) * r;
            double* xvec   = sc;
            double* Wm     = xvec + m;
            double* coeffs = Wm + static_cast<long>(m) * t;
            double* rhs    = coeffs + static_cast<long>(t) * t;
            double* tmp    = rhs + t;          // A2 / B2v
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
        // ---- the constrained weight solve + normalize (weights_chisq_large math) -----
        // SAME wc layout weights_chisq_large_kernel carves: RHS[nl*nl]|wv[nl]|lu[nl*nl]|
        // y[nl]|e[m] (e reused as the nl-length LHS = ones, e has m>=nl slots). chisq is
        // dead output here, so dev_chisq_of_large is skipped.
        double* row = dWmat + (static_cast<long>(model) * nb + b) * nl;
        double* RHS = sc;
        double* wv  = RHS + static_cast<long>(nl) * nl;
        double* lu  = wv + nl;
        double* y   = lu + static_cast<long>(nl) * nl;
        double* e   = y + nl;
        int*    piv = ibase;
        if (r == 0) {
            for (int i = 0; i < nl; ++i) row[i] = 1.0;   // status Ok, w = ones (Σ=1 after norm)
            continue;
        }
        const int rp = r + 1;
        auto xm = [&](int i, int p) -> double { return (p < r) ? A[i + nl * p] : 1.0; };
        for (int i = 0; i < nl; ++i) {
            for (int ip = 0; ip < nl; ++ip) {
                double acc = 0.0;
                for (int p = 0; p < rp; ++p) acc += xm(i, p) * xm(ip, p);
                RHS[i + nl * ip] = acc;
            }
        }
        double* LHS = e;
        for (int i = 0; i < nl; ++i) LHS[i] = 1.0;
        if (!dev_solve(RHS, nl, LHS, wv, lu, piv, y)) {
            for (int i = 0; i < nl; ++i) row[i] = 0.0;   // RankDeficient ⇒ zeros (status guard)
            continue;
        }
        double sum = 0.0;
        for (int i = 0; i < nl; ++i) sum += wv[i];
        for (int i = 0; i < nl; ++i) row[i] = wv[i] / sum;  // UNSCALED (host scales by (nb-1)/√nb)
    }
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
    double A[kQpMaxT], B[kQpMaxT], w[kQpMaxNl], chisq;            // many-thread ⇒ SMALL bound
    const int st = dev_als_weights<kQpMaxNl, kQpMaxNr, kQpMaxR>(
        xmat, nl, nr, r, dQinv, fudge, als_iters, /*seed=*/true, A, B, w, &chisq);
    for (int i = 0; i < nl; ++i)
        dWmat[static_cast<long>(b) * nl + i] = (st == 0) ? w[i] : 0.0;
}

// =================================================================================
// M(fit-6) S8 MODEL-BATCHED kernels (the ROTATION primitive). Each kernel adds a
// `model` grid axis and reads/writes a per-model SLICE of a strided arena. Same math,
// same op order, native FP64 — the model axis is the only addition. They serve the
// SMALL-path bit-parity bucket (nl<=5, nr<=10, r<=4); the host bucketer guarantees
// that envelope before launching, so the kQpMax* per-thread local bound holds even
// with one thread per model (MANY resident threads ⇒ the SMALL bound, like the
// proven loo_batched_kernel). See the .cuh for the per-kernel contracts.
// =================================================================================

// --- S3 f4-gather (model-batched): grid over (k+m*b, model) ----------------------
__global__ void assemble_f4_gather_models_kernel(const double* __restrict__ f2, int P,
                                                 const int* __restrict__ d_left_arena,
                                                 const int* __restrict__ d_right_arena,
                                                 int nl, int nr, int nb, int n_models,
                                                 double* __restrict__ dX) {
    const long m = static_cast<long>(nl) * nr;
    const long per = m * nb;                         // elements per model
    const long total = per * n_models;
    for (long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         gid < total; gid += static_cast<long>(gridDim.x) * blockDim.x) {
        const int model = static_cast<int>(gid / per);
        const long idx = gid - static_cast<long>(model) * per;  // k + m*b within model
        const int k = static_cast<int>(idx % m);
        const int b = static_cast<int>(idx / m);
        const int i = k / nr;  // left source index (0..nl-1)
        const int j = k % nr;  // right index       (0..nr-1)
        const int* lft = d_left_arena + static_cast<long>(model) * (nl + 1);
        const int* rgt = d_right_arena + static_cast<long>(model) * (nr + 1);
        const int L0 = lft[0];
        const int R0 = rgt[0];
        const int Li = lft[i + 1];
        const int Rj = rgt[j + 1];
        const long slab = static_cast<long>(P) * static_cast<long>(P) * b;
        const auto at = [&](int a, int c) -> double {
            return f2[static_cast<long>(a) + static_cast<long>(P) * c + slab];
        };
        dX[gid] = 0.5 * (at(Li, R0) + at(L0, Rj) - at(L0, R0) - at(Li, Rj));
    }
}

// --- S3 est_to_loo + x_total + tot_line (model-batched): one thread per (k, model) -
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
        const long base = static_cast<long>(model) * m * nb;  // this model's dX/dLoo slice
        double num = 0.0;
        for (int b = 0; b < nb; ++b)
            num += dX[base + k + static_cast<long>(m) * b] *
                   static_cast<double>(d_block_sizes[b]);
        const double tot_ij = num / n;
        for (int b = 0; b < nb; ++b) {
            const double bl = static_cast<double>(d_block_sizes[b]);
            const double rel = bl / n;
            const double xv = dX[base + k + static_cast<long>(m) * b];
            dLoo[base + k + static_cast<long>(m) * b] = (tot_ij - xv * rel) / (1.0 - rel);
        }
        double wln = 0.0, wld = 0.0;
        for (int b = 0; b < nb; ++b) {
            const double w = 1.0 - static_cast<double>(d_block_sizes[b]) / n;
            wln += dLoo[base + k + static_cast<long>(m) * b] * w;
            wld += w;
        }
        const double tot_line = wln / wld;
        dTotLine[static_cast<long>(model) * m + k] = tot_line;
        double diffsum = 0.0, wbn = 0.0;
        for (int b = 0; b < nb; ++b) {
            const double loo = dLoo[base + k + static_cast<long>(m) * b];
            diffsum += tot_line - loo;
            wbn += loo * static_cast<double>(d_block_sizes[b]);
        }
        dTotal[static_cast<long>(model) * m + k] = diffsum + wbn / n;
    }
}

// --- S4 xtau (model-batched): one thread per (k+m*b, model) -----------------------
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
        const long idx = gid - static_cast<long>(model) * per;  // k + m*b
        const int k = static_cast<int>(idx % m);
        const int b = static_cast<int>(idx / m);
        const double bl = static_cast<double>(d_block_sizes[b]);
        const double h = n / bl;
        const double sh = sqrt(h - 1.0);
        const double loo = dLoo[gid];
        const double est = dEst[static_cast<long>(model) * m + k];
        const double tl = dTotLine[static_cast<long>(model) * m + k];
        dXtau[gid] = (est * h - loo * (h - 1.0) - tl) / sh;
    }
}

// --- per-model fudge-diag (model-batched): one block per model -------------------
__global__ void add_fudge_diag_models_kernel(const double* __restrict__ dQ,
                                             double* __restrict__ dQf, int m,
                                             double fudge, int n_models) {
    const int model = blockIdx.x;
    if (model >= n_models) return;
    const long base = static_cast<long>(model) * m * m;
    // Thread 0 computes the trace (m tiny); copy + fudge handled by all threads.
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

// --- batched identity RHS arena --------------------------------------------------
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

// --- the MODEL-BATCHED full fit (one thread per model) ---------------------------
// Reduced-row popdrop bound: nl<=kQpMaxNl rows ⇒ a reduced model keeps nl_red<=nl
// rows. The reduced xmat (m_red<=kQpMaxM) and reduced Qinv (m_red*m_red<=kQpMaxM^2)
// live in per-thread local — at the SMALL bound m<=50 ⇒ m_red^2<=2500 doubles=20 KB.
// Combined with dev_als_weights' SMALL-bound scratch this stays inside the per-thread
// frame the loo_batched_kernel already proved launchable. The host bucketer enforces
// the envelope.
__global__ void qpadm_fit_models_kernel(const double* __restrict__ dTotal,
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
    (void)dLoo; (void)d_se;  // the LOO (SE) now runs in a separate batched kernel.

    // ---- xmat (col-major) from x_total (row-major k = j + nr*i) -------------------
    double xmat[kQpMaxM];
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nr; ++j)
            xmat[i + nl * j] = total[j + nr * i];

    // ---- the full-rank fit at r_fit (weights + chisq) ----------------------------
    double A[kQpMaxT], B[kQpMaxT], w[kQpMaxNl], chisq;
    int st = dev_als_weights<kQpMaxNl, kQpMaxNr, kQpMaxR>(
        xmat, nl, nr, r_fit, qinv, fudge, als_iters, /*seed=*/true, A, B, w, &chisq);
    d_status[model] = st;
    for (int i = 0; i < nl; ++i)
        d_weight[static_cast<long>(model) * nl + i] = (st == 0) ? w[i] : 0.0;
    d_chisq[model] = chisq;

    // ---- the rank sweep r = 0..rmax (chisq(r); reuses the SAME xmat + qinv) -------
    for (int rr = 0; rr <= rmax; ++rr) {
        double Ar[kQpMaxT], Br[kQpMaxT], wr[kQpMaxNl], cr;
        dev_als_weights<kQpMaxNl, kQpMaxNr, kQpMaxR>(
            xmat, nl, nr, rr, qinv, fudge, als_iters, /*seed=*/true, Ar, Br, wr, &cr);
        d_rank_chisq[static_cast<long>(model) * (rmax + 1) + rr] = cr;
    }

    // NOTE: the LOO SE is NOT computed here (it would serialize nb=702 ALS fits per
    // model-thread — the throughput wall). It runs as a SEPARATE batched kernel with
    // one thread per (model, block) (qpadm_loo_models_kernel) + a deterministic
    // variance reduction (qpadm_se_from_wmat_kernel) — good occupancy, deterministic
    // op order (no atomics) so G=1==G=2 bit-identical. d_se is filled by that kernel.

    // ---- popdrop (leave-one-LEFT-SOURCE-out) over the per-model X + Qinv ----------
    // Row 0 = the FULL model (all nl rows). Rows 1..nl = drop source (nl-1), (nl-2),
    // ..., 0 (the AT2 order). Each reduced model is fit at rank nl_red-1 (the FITTED
    // rank, per ranktest.cpp popdrop_one). The FULL row's weights are the feasibility
    // source (d_pop_wfull). Reduced xmat / Qinv are extracted from this model's slices.
    auto fit_reduced = [&](const int* surv, int nl_red, double* w_red, double* chisq_red) -> int {
        const int m_red = nl_red * nr;
        double xr[kQpMaxM];
        for (int ii = 0; ii < nl_red; ++ii)
            for (int j = 0; j < nr; ++j)
                xr[ii + nl_red * j] = total[j + nr * surv[ii]];  // col-major reduced xmat
        double qr[kQpMaxM * kQpMaxM];
        // ind[a] = surv[a/nr]*nr + (a%nr); qr[a + m_red*b] = qinv[ind[a] + m*ind[b]].
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
            xr, nl_red, nr, r_red, qr, fudge, als_iters, /*seed=*/true, Ar, Br,
            w_red, chisq_red);
    };

    // Full row (index 0).
    {
        int surv[kQpMaxNl];
        for (int i = 0; i < nl; ++i) surv[i] = i;
        double wr[kQpMaxNl], cr;
        const int s0 = fit_reduced(surv, nl, wr, &cr);
        d_pop_chisq[static_cast<long>(model) * (nl + 1) + 0] = cr;
        for (int i = 0; i < nl; ++i)
            d_pop_wfull[static_cast<long>(model) * nl + i] =
                (s0 == 0) ? wr[i] : (0.0 / 0.0);  // NaN on a singular reduced fit
    }
    // Single-source drops (only for nl>=2). Row index = 1 + (nl-1-drop) so the array
    // order is drop=(nl-1), (nl-2), ..., 0 (the AT2 order).
    if (nl >= 2) {
        for (int drop = nl - 1; drop >= 0; --drop) {
            int surv[kQpMaxNl];
            int cnt = 0;
            for (int i = 0; i < nl; ++i) if (i != drop) surv[cnt++] = i;
            double wr[kQpMaxNl], cr;
            fit_reduced(surv, cnt, wr, &cr);
            const int row = 1 + (nl - 1 - drop);
            d_pop_chisq[static_cast<long>(model) * (nl + 1) + row] = cr;
        }
    }
}

// --- S7 LOO per-block re-fits (model-batched): one thread per (model, block) ------
// Each thread fits ONE block's leave-one-out weights for ONE model, writing the
// SCALED weight vector (s = (nb-1)/sqrt(nb)) into dWmat[model*nb*nl + b*nl + i]. This
// replaces the per-model serial nb-loop (the throughput wall) with B*nb parallel
// threads. dQinv/dLoo are per-model slices. Native FP64. (st!=0 ⇒ zeros for that row.)
__global__ void qpadm_loo_models_kernel(const double* __restrict__ dLoo,
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
            xb, nl, nr, r_fit, qinv, fudge, als_iters, /*seed=*/true, Ab, Bb, wb, &cb);
        double* row = dWmat + (static_cast<long>(model) * nb + b) * nl;
        for (int i = 0; i < nl; ++i) row[i] = (sb == 0) ? s * wb[i] : 0.0;
    }
}

// --- S7 SE from the wmat (model-batched): one thread per (model, weight col) ------
// SE[i] = sqrt( Σ_b (wmat[b,i] - mean_i)^2 / (nb-1) ), mean_i = Σ_b wmat[b,i]/nb. The
// reduction order is FIXED (ascending b) ⇒ deterministic (no atomics) so G=1==G=2
// bit-identical. Matches se_from_loo's sample_cov_diag (FP64; the long-double oracle
// accumulator becomes FP64 — SE is not in the rotation parity gate, only determinism).
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

// --- SE-policy gather (model-batched): compact survivor per-model slices ----------
// Pure data movement (parity-NEUTRAL: it moves bits, it does not compute) for the
// two-pass SE. For each compacted output model k, copy the slice of the source model
// d_surv[k] (the chunk position of the k-th survivor) into the dense output slot k.
// One launch gathers ALL survivor dLoo (m*nb each) + dQinv (m*m each) slices, replacing
// the per-survivor cudaMemcpyAsync D2D loop (its launch overhead dominated at large
// survivor counts). Ascending-survivor order ⇒ a survivor's compacted slice is bit-
// identical to its full-arena slice (the SE kernels then see identical inputs).
__global__ void qpadm_gather_loo_qinv_kernel(const double* __restrict__ dLooSrc,
                                             const double* __restrict__ dQinvSrc,
                                             const int* __restrict__ d_surv,
                                             int m, int nb, int n_surv,
                                             double* __restrict__ dLooDst,
                                             double* __restrict__ dQinvDst) {
    const long loo_per = static_cast<long>(m) * nb;   // dLoo slice size per model
    const long qinv_per = static_cast<long>(m) * m;   // dQinv slice size per model
    const long total = (loo_per + qinv_per) * n_surv; // all elements, both arenas
    for (long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         gid < total; gid += static_cast<long>(gridDim.x) * blockDim.x) {
        const long per = loo_per + qinv_per;
        const int k = static_cast<int>(gid / per);    // compacted output model
        const long e = gid % per;                     // element within the pair of slices
        const int j = d_surv[k];                      // source chunk position
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

void launch_assemble_f4_gather_models_batched(const double* f2, int P,
                                              const int* d_left_arena,
                                              const int* d_right_arena,
                                              int nl, int nr, int nb, int n_models,
                                              double* dX, cudaStream_t stream) {
    const long total = static_cast<long>(nl) * nr * nb * n_models;
    if (total <= 0) return;
    const int block = 256;
    const long grid_l = (total + block - 1) / block;
    const int grid = static_cast<int>(grid_l > 65535 ? 65535 : grid_l);
    assemble_f4_gather_models_kernel<<<grid, block, 0, stream>>>(
        f2, P, d_left_arena, d_right_arena, nl, nr, nb, n_models, dX);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_f4_loo_total_models_batched(const double* dX, const int* d_block_sizes,
                                        int m, int nb, double n, int n_models,
                                        double* dLoo, double* dTotal, double* dTotLine,
                                        cudaStream_t stream) {
    const long total = static_cast<long>(m) * n_models;
    if (total <= 0 || nb <= 0) return;
    const int block = 128;
    const long grid_l = (total + block - 1) / block;
    const int grid = static_cast<int>(grid_l > 65535 ? 65535 : grid_l);
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
    const int block = 256;
    const long grid_l = (total + block - 1) / block;
    const int grid = static_cast<int>(grid_l > 65535 ? 65535 : grid_l);
    f4_xtau_models_kernel<<<grid, block, 0, stream>>>(
        dLoo, dEst, dTotLine, d_block_sizes, m, nb, n, n_models, dXtau);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_add_fudge_diag_models_batched(const double* dQ, double* dQf, int m,
                                          double fudge, int n_models, cudaStream_t stream) {
    if (m <= 0 || n_models <= 0) return;
    int block = (m * m < 256) ? ((m * m + 31) / 32 * 32) : 256;
    if (block < 32) block = 32;
    add_fudge_diag_models_kernel<<<n_models, block, 0, stream>>>(
        dQ, dQf, m, fudge, n_models);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_fill_identity_batched(double* dI, int m, int n_models, cudaStream_t stream) {
    const long total = static_cast<long>(m) * m * n_models;
    if (total <= 0) return;
    const int block = 256;
    const long grid_l = (total + block - 1) / block;
    const int grid = static_cast<int>(grid_l > 65535 ? 65535 : grid_l);
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
    const int block = 64;
    const int grid = (n_models + block - 1) / block;
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
    const int block = 128;
    const long grid_l = (total + block - 1) / block;
    const int grid = static_cast<int>(grid_l > 65535 ? 65535 : grid_l);
    qpadm_loo_models_kernel<<<grid, block, 0, stream>>>(
        dLoo, dQinv, nl, nr, r_fit, fudge, als_iters, nb, n_models, s, dWmat);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_se_from_wmat_batched(const double* dWmat, int nl, int nb,
                                       int n_models, double* d_se, cudaStream_t stream) {
    const long total = static_cast<long>(nl) * n_models;
    if (total <= 0) return;
    const int block = 128;
    const long grid_l = (total + block - 1) / block;
    const int grid = static_cast<int>(grid_l > 65535 ? 65535 : grid_l);
    qpadm_se_from_wmat_kernel<<<grid, block, 0, stream>>>(dWmat, nl, nb, n_models, d_se);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_qpadm_gather_loo_qinv(const double* dLooSrc, const double* dQinvSrc,
                                  const int* d_surv, int m, int nb, int n_surv,
                                  double* dLooDst, double* dQinvDst, cudaStream_t stream) {
    const long per = static_cast<long>(m) * nb + static_cast<long>(m) * m;
    const long total = per * n_surv;
    if (total <= 0) return;
    const int block = 256;
    const long grid_l = (total + block - 1) / block;
    const int grid = static_cast<int>(grid_l > 65535 ? 65535 : grid_l);
    qpadm_gather_loo_qinv_kernel<<<grid, block, 0, stream>>>(
        dLooSrc, dQinvSrc, d_surv, m, nb, n_surv, dLooDst, dQinvDst);
    STEPPE_CUDA_CHECK_KERNEL();
}

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

void launch_transpose_small(const double* dXmat, int nl, int nr,
                            double* dXt, cudaStream_t stream) {
    const long total = static_cast<long>(nl) * nr;
    if (total <= 0) return;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
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
    const int block = 64;  // large per-thread VRAM-scratch refit ⇒ small block, grid-stride
    const long grid_l = (total + block - 1) / block;
    const int grid = static_cast<int>(grid_l > 65535 ? 65535 : grid_l);
    loo_large_batched_kernel<<<grid, block, 0, stream>>>(
        dLoo, dQinv, dAseed, dBseed, nl, nr, r, fudge, als_iters, nb, n_models,
        dbl_refit, int_refit, dScratch, dIntScratch, dWmat);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

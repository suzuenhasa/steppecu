// src/device/cuda/qpadm_fit_kernels.cuh
//
// Narrow launch-wrapper declarations for the qpAdm fit (M(fit-4)) device kernels
// (the FROZEN CONTRACT §2). Host orchestration (cuda_backend.cu) calls these
// `void launch_*` functions; the kernel bodies + `<<<>>>` live only in the .cu —
// the architecture.md §7 "host code never includes kernel bodies or <<<>>>" rule,
// mirroring f2_blocks_kernel.cuh's style. All math here is NATIVE FP64: the f4
// 4-slab combine is the cancellation-sensitive f-stat difference, and the
// jackknife/xtau reductions reproduce the CpuBackend long-double oracle's operation
// order in FP64 (at nb=708, m=10 this matches the golden to the gate tier; §12).
//
// This header names CUDA types (cudaStream_t) and so is PRIVATE to steppe_device
// (architecture.md §4) — the device-internal seam between the backend and the
// kernels, NOT the CUDA-free public ComputeBackend seam.
#ifndef STEPPE_DEVICE_CUDA_QPADM_FIT_KERNELS_CUH
#define STEPPE_DEVICE_CUDA_QPADM_FIT_KERNELS_CUH

#include <cuda_runtime.h>

namespace steppe::device {

/// S3 f4-gather (the FROZEN CONTRACT §2a; CpuBackend cpu_backend.cpp:400-412). Build
/// the per-block f4 matrix X[k + m*b] for k = j + nr*i (ROW-MAJOR vectorization),
/// batched over ALL nb blocks in one launch (grid over (k, b)). Reads the RESIDENT
/// f2 tensor `f2` (column-major i + P*j + P*P*b, VRAM) directly — NO D2H. The
/// 4-slab combine
///   X[j+nr*i, b] = 0.5*( f2(Li,R0,b) + f2(L0,Rj,b) - f2(L0,R0,b) - f2(Li,Rj,b) )
/// with Li = d_left[i+1], Rj = d_right[j+1], L0 = d_left[0], R0 = d_right[0]. Native
/// FP64 (the cancellation-sensitive f-stat difference). `d_left`/`d_right` are the
/// small H2D'd index buffers (length nl+1 / nr+1). Output dX[m*nb], m = nl*nr.
void launch_assemble_f4_gather(const double* f2, int P,
                               const int* d_left, const int* d_right,
                               int nl, int nr, int nb,
                               double* dX, cudaStream_t stream);

/// S3 est_to_loo + x_total + tot_line (the FROZEN CONTRACT §2a; CpuBackend
/// compute_loo_and_total cpu_backend.cpp:540-593). One thread per k (m = nl*nr
/// small) reduces over the nb blocks, reproducing the CpuBackend's operation order
/// in FP64 (the long-double accumulators become FP64; at nb=708 this matches the
/// golden to rtol 1e-6, the gate tier). For each k:
///   tot_ij    = (Σ_b X[k,b]*bl_b) / n
///   loo[k,b]  = (tot_ij - X[k,b]*rel_b) / (1 - rel_b),  rel_b = bl_b/n
///   tot_line  = (Σ_b loo[k,b]*(1-bl_b/n)) / (Σ_b (1-bl_b/n))
///   est[k]    = (Σ_b (tot_line - loo[k,b])) + (Σ_b loo[k,b]*bl_b)/n
/// (the CpuBackend's `mean(...)*nb` simplifies to the bare sum; reproduced exactly).
/// `d_block_sizes` is the H2D'd per-block SNP count (length nb), `n` = Σ bl.
/// Outputs dLoo[m*nb], dTotal[m], dTotLine[m].
void launch_f4_loo_total(const double* dX, const int* d_block_sizes,
                         int m, int nb, double n,
                         double* dLoo, double* dTotal, double* dTotLine,
                         cudaStream_t stream);

/// S4 xtau pseudo-values (the FROZEN CONTRACT §2b; CpuBackend cpu_backend.cpp:448-458).
/// One thread per (k, b): h = n/bl_b, sh = sqrt(h-1),
///   xtau[k,b] = (est[k]*h - loo[k,b]*(h-1) - tot_line[k]) / sh
/// laid out COLUMN-MAJOR (k + m*b) so cublasDsyrk(OP_N, lda=m, k=nb) forms
/// Q = xtau·xtauᵀ. Native FP64. dEst = dTotal (length m), dTotLine length m,
/// dLoo[m*nb]. Output dXtau[m*nb].
void launch_f4_xtau(const double* dLoo, const double* dEst, const double* dTotLine,
                    const int* d_block_sizes, int m, int nb, double n,
                    double* dXtau, cudaStream_t stream);

/// Mirror the LOWER triangle of an n×n COLUMN-MAJOR matrix into the UPPER (so a
/// SYRK/potri result that fills one triangle becomes the full symmetric matrix the
/// CpuBackend writes both triangles of). In-place. One thread per (i,j), i>j writes
/// (j,i) = (i,j). Native FP64.
void launch_symmetrize_lower_to_full(double* dM, int n, cudaStream_t stream);

/// Add `fudge * tr` to the diagonal of an n×n COLUMN-MAJOR matrix (the FROZEN
/// CONTRACT §2b fudge step; CpuBackend cpu_backend.cpp:480-481). `dM[k + n*k] +=
/// fudge*tr`. `tr` is supplied by the host (computed via a trace reduction). One
/// thread per diagonal entry. Native FP64.
void launch_add_fudge_diag(double* dM, int n, double fudge, double tr,
                           cudaStream_t stream);

// ---------------------------------------------------------------------------------
// Small-dense on-device LA for the rank-test / ALS / weight / chisq (S5/S6) — the
// FROZEN CONTRACT §2c/§2d. At the golden sizes everything is tiny (m=10, nl=2,
// nr=5, r=1, t≤5). These run as SINGLE-THREAD device kernels that TRANSLITERATE the
// CpuBackend's exact native-FP64 scalar operations (small_linalg.hpp jacobi_svd /
// lu_factor / solve; cpu_backend.cpp opt_A/opt_B/als_weights/chisq_of) — same ops,
// same order, on the GPU — so the GPU fit is bit-exact vs the FP64 oracle AND runs
// device-resident (the production path; the binding requirement). They consume only
// device buffers (dXmat from x_total, dQinv) and write device outputs (dW/dA/dB,
// dchisq). The constrained weight solve + chisq are folded into one kernel.
// ---------------------------------------------------------------------------------

/// S5 SVD seed (the FROZEN CONTRACT §2c; CpuBackend seed_AB cpu_backend.cpp:626-644).
/// One-sided Jacobi SVD of the nl×nr COLUMN-MAJOR `dXmat` (transliterating
/// core::jacobi_svd), then B = t(V[:,0:r]) (r×nr), A = xmat·t(B) (nl×r). dA[nl*r],
/// dB[r*nr] written column-major. Single-thread kernel (nl,nr small). Native FP64.
void launch_qpadm_seed_ab(const double* dXmat, int nl, int nr, int r,
                          double* dA, double* dB, cudaStream_t stream);

/// S6 ALS opt_A then opt_B for `als_iters` iterations (the FROZEN CONTRACT §2d;
/// CpuBackend als_weights loop cpu_backend.cpp:788-791, opt_A 652-709, opt_B
/// 715-768). Transliterates the Kronecker coeffs/rhs build + the LU solve
/// (core::solve) + the byrow reshape, in native FP64, single-thread. Seeds A,B in
/// place from the caller's dA/dB (filled by launch_qpadm_seed_ab) and overwrites
/// them with the refined factors. `dQinv` is the m×m (m=nl*nr) column-major inverse.
/// On a singular ALS system the corresponding factor is left zero (CpuBackend
/// cpu_backend.cpp:704). Native FP64.
void launch_qpadm_als(const double* dXmat, const double* dQinv,
                      int nl, int nr, int r, double fudge, int als_iters,
                      double* dA, double* dB, cudaStream_t stream);

/// S6 constrained weight solve + chisq (the FROZEN CONTRACT §2d/§2c; CpuBackend
/// als_weights cpu_backend.cpp:798-825, chisq_of 833-858). From the refined dA
/// (nl×r) build RHS=crossprod(cbind(A,1)) (nl×nl), LHS=ones, LU-solve, normalize
/// Σw=1 → dW[nl]; and chisq = vec(E)'·Qinv·vec(E), E = xmat - A·B → dchisq[0].
/// `d_status` (length 1, int) is set to 0=Ok, 6=RankDeficient (the weight solve was
/// singular), matching CpuBackend cpu_backend.cpp:820. Single-thread, native FP64.
/// For r==0 the trivial path (w=ones, chisq=chisq_of with empty A,B) is taken.
void launch_qpadm_weights_chisq(const double* dXmat, const double* dQinv,
                                const double* dA, const double* dB,
                                int nl, int nr, int r,
                                double* dW, double* dchisq, int* d_status,
                                cudaStream_t stream);

/// Build the nl×nr COLUMN-MAJOR xmat from the ROW-MAJOR x_total / loo slice
/// (k = j + nr*i ⇒ xmat(i,j) at i + nl*j; CpuBackend xmat_from_total
/// cpu_backend.cpp:598-605). For the full-data fit `dTotalSrc` = dTotal (one slice);
/// for batched S7 a per-block slice is passed. Single-thread kernel. Native FP64.
void launch_qpadm_xmat_from_rowmajor(const double* dTotalSrc, int nl, int nr,
                                     double* dXmat, cudaStream_t stream);

// ---------------------------------------------------------------------------------
// S7 BATCHED leave-one-block-out re-fits (the FROZEN CONTRACT §2e). The whole
// per-block fit (xmat from x_loo[:,:,b] → seed → 20 ALS iters → weight solve →
// normalize) is run for ALL nb blocks in ONE launch (one thread/block-of-threads
// per replicate), transliterating the same CpuBackend ops as the single fit. This
// replaces the 708 host gls_weights calls with a single batched device kernel
// (NOT a host loop; the binding requirement). Output dWmat[nb*nl] ROW-MAJOR
// (b*nl + i, the AT2 replicate matrix). Native FP64.
void launch_qpadm_loo_batched(const double* dLoo, const double* dQinv,
                              int nl, int nr, int r, double fudge, int als_iters,
                              int nb, double* dWmat, cudaStream_t stream);

// ---------------------------------------------------------------------------------
// LARGE-path kernels (the FROZEN CONTRACT §1/§2): models exceeding the bit-parity
// small envelope (nl>5 || nr>10 || r>4, e.g. NRBIG nr=39). The per-model working set
// moves OUT of per-thread LOCAL memory (which CUDA reserves for the device's max
// resident-thread count ⇒ OOM at launch for big nr) into caller-provided VRAM
// scratch (`dScratch`/`dIntScratch`). The SVD seed is NOT a kernel here — it is the
// cuSOLVER `large_svd_V` on the host (cuda_backend.cu) followed by
// `launch_qpadm_seed_from_V` (the cheap GEMM-shaped part of the seed). Same math /
// same op order as the small templated kernels; native FP64. Single-model-per-launch
// (grid=1) for THIS milestone; the `dScratch` offset layout is the S8 batching seam.
// ---------------------------------------------------------------------------------

/// Transpose the nl×nr COLUMN-MAJOR `dXmat` into the nr×nl COLUMN-MAJOR `dXt`
/// (dXt[j + nr*i] = dXmat[i + nl*j]). One thread per element. Used to orient the
/// matrix handed to cuSOLVER gesvd as rows>=cols (the §1.3 determinism rule): for the
/// common large case nr>=nl, Xt is nr×nl (rows>=cols), and U(Xt) == V(Xmat). Native FP64.
void launch_transpose_small(const double* dXmat, int nl, int nr,
                            double* dXt, cudaStream_t stream);

/// LARGE-path seed from the cuSOLVER right singular vectors V[:,0:r] (nr×r col-major
/// in `dVout`, descending). Only the cheap GEMM-shaped part of `dev_seed_ab` AFTER the
/// SVD: B[p,j] = V[j,p] (r×nr), A = xmat·t(B) (nl×r). No Vfull/W local arrays. Single
/// thread (one model). Native FP64. Writes dA[nl*r], dB[r*nr] column-major.
void launch_qpadm_seed_from_V(const double* dXmat, const double* dVout,
                              int nl, int nr, int r,
                              double* dA, double* dB, cudaStream_t stream);

/// LARGE-path ALS opt_A/opt_B loop — same math as launch_qpadm_als but the per-model
/// working set (xvec[m], Wm[m*t], coeffs[t*t], rhs[t], lu[t*t], y[t], piv[t], A2/B2[t])
/// lives in caller-provided VRAM scratch (dScratch / dIntScratch), so arbitrary
/// nl/nr/m/r fit. Single-thread kernel (one model). Native FP64. Seeds in dA/dB
/// (filled by launch_qpadm_seed_from_V); overwrites them with the refined factors. On
/// a singular ALS system the corresponding factor is left zero (CpuBackend parity).
void launch_qpadm_als_large(const double* dXmat, const double* dQinv,
                            int nl, int nr, int r, double fudge, int als_iters,
                            double* dA, double* dB,
                            double* dScratch, int* dIntScratch, cudaStream_t stream);

/// LARGE-path weight solve + chisq — same math as launch_qpadm_weights_chisq with
/// VRAM scratch (RHS[nl*nl], LHS[nl], wv[nl], lu[nl*nl], y[nl], piv[nl], e[m]). Sets
/// d_status 0=Ok / 6=RankDeficient. For r==0 the trivial path (w=ones, chisq with empty
/// A,B). Single-thread, native FP64.
void launch_qpadm_weights_chisq_large(const double* dXmat, const double* dQinv,
                                      const double* dA, const double* dB,
                                      int nl, int nr, int r,
                                      double* dW, double* dchisq, int* d_status,
                                      double* dScratch, int* dIntScratch,
                                      cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_QPADM_FIT_KERNELS_CUH

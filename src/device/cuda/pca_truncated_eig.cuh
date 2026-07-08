#pragma once
// src/device/cuda/pca_truncated_eig.cuh
//
// Randomized subspace-iteration top-K eigensolver for a device-resident, column-major
// N x N SPD covariance (the `steppe pca` GRM). It replaces the full-spectrum
// cusolverDnDsyevd(CUSOLVER_EIG_MODE_VECTOR, N) whose ~2N^2*8B workspace (~8.5 GB at
// N=23,089) is the marginal allocation that OOMs the full AADR cohort for a top-K=10 ask.
// The randomized Rayleigh-Ritz solve here allocates only O(N*(K+p)) extra workspace
// (~190 MB at N=23k, the default L=256), so the eigensolve stops being the memory wall.
//
// Math (Halko 2011, Alg. 4.4 + 5.3): draw a deterministic Gaussian sketch Omega (N x L,
// L = min(K+oversample, N)); Y = C*Omega; thin (Householder) QR -> Q; a few
// subspace-iteration passes (Y=C*Q, re-QR) to sharpen the closely-spaced trailing PCs;
// project B = Q^T C Q (L x L); Dsyevd the tiny B; lift V = Q * B_evecs (N x L). The L
// Ritz values are the top-L eigenvalue estimates of C (ascending); the top K well-separated
// ones are the PCs. Precision mirrors the SYRK/eigen carve-out: the O(N^2*L) sketch GEMMs
// run emulated-FP64 (matmul-heavy, the default), while the small B formation, the QR, and
// the L x L Dsyevd run native-FP64 (the cancellation/eigen carve-out, now on L=256 by default,
// not the 23k GRM).
//
// Factored out of cuda_backend_pca.cu so the truncation math is unit-testable on a planted
// spectrum. Private to steppe_device; a CUDA header (pulls in cuBLAS/cuSOLVER).

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include "steppe/config.hpp"  // Precision
#include "steppe/error.hpp"   // Status

namespace steppe::device {

struct PcaTruncatedEig {
    int L = 0;                    // actual subspace width used (= min(K+oversample, N))
    Status status = Status::Ok;
};

// Truncated top-K eigensolve of the column-major N x N SPD matrix dC (READ-ONLY — not
// overwritten, so trace(dC) stays valid for the caller's var_explained denominator).
//
//   dC       : device, column-major N x N, SPD (leading dim N).
//   N, K     : matrix size and number of requested top eigenpairs (K <= N).
//   oversample, subspace_iters : randomized-SVD tuning (p and q). L = min(K+oversample, N).
//   precision: the sketch-GEMM precision policy (emulated-FP64 default).
//   d_evecL  : device out, column-major N x L — the L lifted Ritz vectors, ASCENDING Ritz
//              value (so column L-1 is the top PC). Caller sizes it N*L.
//   d_evalL  : device out, length L — the ascending Ritz values. Caller sizes it L.
//   out_L    : host out, the actual L (== result.L).
//
// blas must be bound to `stream`; solver's stream is set here. Returns Status::NonSpdCovariance
// if any cuSOLVER factorization reports a nonzero info.
PcaTruncatedEig pca_truncated_topk(cublasHandle_t blas, cusolverDnHandle_t solver,
                                   cudaStream_t stream, const double* dC, int N, int K,
                                   int oversample, int subspace_iters,
                                   const Precision& precision, double* d_evecL,
                                   double* d_evalL, int* out_L);

}  // namespace steppe::device

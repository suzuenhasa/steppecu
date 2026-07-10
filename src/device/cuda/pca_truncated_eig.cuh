#pragma once
// src/device/cuda/pca_truncated_eig.cuh
// Reference: docs/reference/src_device_cuda_pca_truncated_eig.cuh.md
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

#include <functional>

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

// PcaCovMatvec — the covariance operator the randomized solver applies. It computes
// CQ_out (column-major N x ncols, leading dim N) = C * Q_in (column-major N x ncols, ld N),
// where C is the sample x sample covariance. The IMPLEMENTATION is free to form the product
// any way it likes and NEVER has to materialize C: the dense path multiplies a resident N x N
// Gram (byte-identical to the pre-factoring code), while the matrix-free/biobank path streams
// CQ = Σ_tiles Z_t (Z_t^T Q) over SNP tiles, so no N x N object is ever allocated. `ncols` is
// the subspace width L on every call. The callee is issued on the solver's stream (the caller
// binds `blas`/`solver` to `stream`); it must leave `blas`'s math mode as it found it or set
// it per call (the dense/streamed impls both scope their own MathModeScope).
using PcaCovMatvec = std::function<void(const double* Q_in, double* CQ_out, int ncols)>;

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

// Operator-driven form of the same Halko pipeline: identical to pca_truncated_topk except the
// covariance action is supplied by `matvec` (see PcaCovMatvec) instead of a resident N x N
// Gram, so the biobank-scale (matrix-free) PCA path can plug a streamed Z(Z^T Q) sweep here
// and NEVER form C. `matvec` is invoked q+2 times (the range-finder sketch, `subspace_iters`
// power passes, and the Rayleigh-Ritz projection). All other arguments/outputs match
// pca_truncated_topk; `precision` governs only the final emulated-FP64 lift GEMM (the QR, the
// L x L B formation, and the L x L Dsyevd stay native — the eigen carve-out on L, not N).
PcaTruncatedEig pca_truncated_topk_op(cublasHandle_t blas, cusolverDnHandle_t solver,
                                      cudaStream_t stream, const PcaCovMatvec& matvec, int N,
                                      int K, int oversample, int subspace_iters,
                                      const Precision& precision, double* d_evecL,
                                      double* d_evalL, int* out_L);

}  // namespace steppe::device

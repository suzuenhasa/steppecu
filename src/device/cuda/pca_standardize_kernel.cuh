#pragma once
// src/device/cuda/pca_standardize_kernel.cuh
//
// Declarations for the PCA standardization + projection kernel launchers. The kernel
// bodies live in pca_standardize_kernel.cu and share the Patterson center/scale/standardize
// arithmetic with the CPU oracle via core/internal/pca_standardize.hpp, so the GPU and
// reference paths cannot diverge. The covariance SYRK and the eigendecomposition are issued
// by cuda_backend_pca.cu (cuBLAS / cuSOLVER); these launchers cover the three custom kernels:
// the per-SNP scale fold, the standardize-into-Z write, and the top-K coordinate projection.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// Per-SNP Patterson scale over one SNP tile [s_lo, s_lo+tileM): one thread per SNP folds all
// N individuals' diploid codes into the allele count -> center (2p) and inv_scale
// (1/sqrt(p(1-p)), or 0 for a monomorphic/all-missing SNP). Atomically adds the count of
// `used` (polymorphic, non-empty) SNPs into *d_used_count.
void launch_pca_snp_scale(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                          std::size_t N, long s_lo, long tileM,
                          double* d_center, double* d_inv_scale,
                          unsigned long long* d_used_count, cudaStream_t stream);

// Standardize one SNP tile into the column-major Z operand: one thread per (individual i,
// local SNP sl) writes Z[i + sl*N] = (code - center[sl]) * inv_scale[sl] (missing -> 0). The
// column-major N x tileM layout is exactly the operand cublasDsyrk reads (leading dim N) to
// accumulate C += Z*Z^T.
void launch_pca_standardize(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                            std::size_t N, long s_lo, long tileM,
                            const double* d_center, const double* d_inv_scale,
                            double* d_Z, cudaStream_t stream);

// Project the top-K eigenvectors to sample PC coordinates: one thread per (sample i, PC k)
// writes the row-major coords[i*K + k] = evec[i + (N-1-k)*N] * sqrt(max(eval[N-1-k], 0)).
// d_evec is the column-major N x N eigenvector matrix cuSOLVER Dsyevd overwrote into the
// covariance buffer; d_eval is the ascending eigenvalue vector (so PC k reads column N-1-k).
void launch_pca_coords(const double* d_evec, const double* d_eval, int N, int K,
                       double* d_coords, cudaStream_t stream);

}  // namespace steppe::device

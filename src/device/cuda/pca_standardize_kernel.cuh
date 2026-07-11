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

// -------- real-valued (BGEN) dosage launchers --------
//
// The FP32-dosage twins of launch_pca_snp_scale / launch_pca_standardize: they read a
// DosageTileView (individual-major dosage[i*n_snp + s], NaN = missing) instead of the 2-bit
// packed tile, but produce the IDENTICAL column-major center/inv_scale + Z operand the SYRK
// path consumes. The Patterson fold/standardize arithmetic is shared with the integer kernels
// (and the CPU) via core/internal/pca_standardize.hpp (pca_snp_scale_f / pca_standardize_one_f).

// Per-SNP Patterson scale over one SNP tile [s_lo, s_lo+tileM): one thread per SNP folds all N
// individuals' ALT dosages (NaN skipped) into the dosage sum -> center (2p) and inv_scale
// (1/sqrt(p(1-p)), or 0 for a monomorphic/all-missing SNP). Atomically adds the `used` count.
void launch_pca_dosage_snp_scale(const float* d_dosage, std::size_t N, std::size_t n_snp,
                                 long s_lo, long tileM, double* d_center, double* d_inv_scale,
                                 unsigned long long* d_used_count, cudaStream_t stream);

// Standardize one SNP tile into the column-major Z operand: one thread per (individual i, local
// SNP sl) writes Z[i + sl*N] = (dosage - center[sl]) * inv_scale[sl] (NaN -> 0). Identical N x
// tileM operand shape cublasDsyrk reads (leading dim N).
void launch_pca_dosage_standardize(const float* d_dosage, std::size_t N, std::size_t n_snp,
                                   long s_lo, long tileM, const double* d_center,
                                   const double* d_inv_scale, double* d_Z, cudaStream_t stream);

// Project the top-K eigenvectors to sample PC coordinates: one thread per (sample i, PC k)
// writes the row-major coords[i*K + k] = evec[i + (ncol-1-k)*N] * sqrt(max(eval[ncol-1-k], 0)).
// d_evec is a column-major N x ncol eigenvector block with leading dim N (`ncol` = N for the
// full solve, ncol = L=k+p for the truncated solve); d_eval is the ascending eigenvalue
// vector of length ncol (so PC k reads column ncol-1-k — the largest pairs live at the tail).
void launch_pca_coords(const double* d_evec, const double* d_eval, int N, int ncol, int K,
                       double* d_coords, cudaStream_t stream);

// Fill d_omega[0..n_elem) with deterministic N(0,1) draws (counter-based splitmix64 +
// Box-Muller keyed on `seed`, one thread per element). Reproducible across runs for a fixed
// seed — the rotation-invariant full-rank Gaussian sketch the randomized eigensolver assumes.
// Native FP64 (the values are inputs, not a precision-sensitive reduction).
void launch_pca_fill_omega(double* d_omega, std::size_t n_elem, unsigned long long seed,
                           cudaStream_t stream);

// trace(C) = sum of the column-major diagonal C[i + i*N], i in [0,N) -> *d_out (native FP64,
// the var_explained denominator / cancellation carve-out class). Caller zeros *d_out first.
void launch_pca_trace(const double* d_C, int N, double* d_out, cudaStream_t stream);

// Accumulate Σ_e d_Z[e]^2 over the first n_elem elements of a standardized Z tile into *d_out
// (native FP64, block reduction + one atomicAdd/block). The matrix-free PCA path uses this to
// build trace(C) = trace(Z Z^T) = ||Z||_F^2 by summing squares over each SNP tile as it is
// standardized — the exact same quantity the dense path reads as Σ diag(dC), but WITHOUT ever
// forming the N x N Gram. Caller zeros *d_out before the first tile.
void launch_pca_accumulate_sumsq(const double* d_Z, long n_elem, double* d_out,
                                 cudaStream_t stream);

// -------- lsqproject (`steppe pca --project-*`) launchers --------

// Per-SNP Patterson scale over ONE SNP tile, folding ONLY the `n_rows` reference rows named by
// d_rows (indices into the tile's individual axis). One thread per SNP -> center (2p) and
// inv_scale (1/sqrt(p(1-p)), or 0 monomorphic) computed over the reference rows alone, and the
// count of `used` (ref-polymorphic) SNPs atomically added into *d_used_count. This is the
// eigenbasis fold: targets are excluded from the covariance AND the allele frequencies.
void launch_pca_snp_scale_gather(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                                 const int* d_rows, long n_rows, long s_lo, long tileM,
                                 double* d_center, double* d_inv_scale,
                                 unsigned long long* d_used_count, cudaStream_t stream);

// Standardize one SNP tile into the column-major Z operand, GATHERING the `n_rows` rows named
// by d_rows: Z[r + sl*n_rows] = (code(d_rows[r], s) - center[sl]) * inv_scale[sl] (missing ->
// 0). Column-major n_rows x tileM (leading dim n_rows) — the SYRK / GEMM operand.
void launch_pca_standardize_gather(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                                   const int* d_rows, long n_rows, long s_lo, long tileM,
                                   const double* d_center, const double* d_inv_scale,
                                   double* d_Z, cudaStream_t stream);

// Standardize the TARGET block for one SNP tile AND emit its observed mask. For each target row
// r (d_rows[r]) and local SNP sl: Z_tgt[r + sl*n_rows] = pca_standardize_one(code) (missing ->
// 0), and mask[r + sl*n_rows] = 1 iff the code is a valid genotype AND inv_scale[sl] != 0 (a
// ref-polymorphic site), else 0. The mask marks the sites that enter A = W_O^T W_O; the
// missing->0 in Z makes W^T Z_tgt exactly W_O^T z_O with no explicit masking of b.
void launch_pca_standardize_target(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                                   const int* d_rows, long n_rows, long s_lo, long tileM,
                                   const double* d_center, const double* d_inv_scale,
                                   double* d_Ztgt, std::uint8_t* d_mask, cudaStream_t stream);

// Pack the top-K reference eigenbasis: from the column-major N_ref x L Ritz block (ascending
// eigenvalue, largest at the tail) write U (N_ref x K, column-major, column k <- Ritz column
// L-1-k) and inv_S[k] = 1/sqrt(max(eval[L-1-k], tiny)). K threads set inv_S; N_ref*K threads
// pack U (a separate launch each). The loadings use U (not U*S) plus inv_S.
void launch_pca_pack_basis(const double* d_evecL, const double* d_evalL, int N_ref, int L, int K,
                           double* d_U, double* d_inv_S, cudaStream_t stream);

// Scale the loadings tile W (column-major tileM x K, leading dim tileM) in place by 1/S_k:
// W[s + k*tileM] *= inv_S[k]. One thread per (local SNP s, PC k).
void launch_pca_scale_loadings(double* d_W, long tileM, int K, const double* d_inv_S,
                               cudaStream_t stream);

// Accumulate the per-target normal matrices A_j (K x K each, column-major, N_tgt of them) over
// one SNP tile: A[j*K*K + k + kp*K] += Σ_{s in tile} mask[j + s*N_tgt] * W[s+k*tileM] *
// W[s+kp*tileM]. One thread per (target j, k, kp); caller zeros d_A before the first tile.
void launch_pca_accumulate_A(const double* d_W, const std::uint8_t* d_mask, long tileM,
                             int N_tgt, int K, double* d_A, cudaStream_t stream);

// Accumulate per-target usable-site counts over one SNP tile: m_obs[j] += Σ_s mask[j + s*N_tgt].
// One thread per target j; caller zeros d_m_obs before the first tile.
void launch_pca_accumulate_mobs(const std::uint8_t* d_mask, long tileM, int N_tgt,
                                unsigned long long* d_m_obs, cudaStream_t stream);

}  // namespace steppe::device

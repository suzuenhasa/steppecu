// src/device/cuda/admixture_kernels.cuh
//
// Host launch wrappers for the ADMIXTURE Q/F block-EM elementwise kernels (`steppe
// admixture`). The GEMMs (A=Q F^T, Q^T R, R F) are issued by cuBLAS from
// cuda_backend_admixture.cu; these are the non-GEMM pieces: the per-individual dosage
// decode, the per-SNP mean allele frequency, the native-FP64 binomial responsibility map,
// the multiplicative F/Q updates + row-simplex renormalize, and the native-FP64
// log-likelihood reduction. All matrices are COLUMN-MAJOR (cuBLAS-native).
#ifndef STEPPE_DEVICE_CUDA_ADMIXTURE_KERNELS_CUH
#define STEPPE_DEVICE_CUDA_ADMIXTURE_KERNELS_CUH

#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// Decode the packed individual-major tile into G [N x M] dosage (0/1/2) + validity mask V
// (1.0 valid / 0.0 missing), both column-major (element (i,s) at i + N*s). Forces diploid
// (code == dosage), mirroring the PCA decode.
void launch_admix_decode(const std::uint8_t* d_packed, std::size_t bytes_per_record, long N,
                         long M, double* d_G, double* d_V, cudaStream_t stream);

// Per-SNP mean allele-2 frequency phat[s] = (sum_i V*G) / (2 * sum_i V) over valid samples
// (0.5 when a SNP has no valid sample). d_phat is length M.
void launch_admix_snp_mean(const double* d_G, const double* d_V, long N, long M, double* d_phat,
                           cudaStream_t stream);

// Binomial responsibilities for one SNP tile [N x t] at global SNP offset s0 (native FP64):
//   a = clamp(A, eps, 1-eps); R2 = V*G/a; R1 = V*(2-G)/(1-a).
// G/V are read at the GLOBAL column (d_Gtile/d_Vtile already offset by N*s0); A/R2/R1 are the
// local tile (ld N). Missing samples (V=0) contribute 0.
void launch_admix_responsibility(const double* d_Gtile, const double* d_Vtile, const double* d_A,
                                 long N, long t, double eps, double* d_R2, double* d_R1,
                                 cudaStream_t stream);

// Multiplicative F-update over all (s,k) in [M x K] (column-major F(s,k)=d_F[s+M*k]):
//   f' = clamp( f*S2 / (f*S2 + (1-f)*S1), eps, 1-eps ),  S2/S1 are [K x M] (S(k,s)=d_S[k+K*s]).
void launch_admix_update_f(double* d_F, const double* d_S2, const double* d_S1, long M, int K,
                           double eps, cudaStream_t stream);

// Complement d_Fc = 1 - d_F over [M x K] (feeds the Q-update T1 = R1 (1-F) GEMM).
void launch_admix_complement(const double* d_F, double* d_Fc, long M, int K, cudaStream_t stream);

// Multiplicative Q-update + row-simplex renormalize over [N x K] (column-major Q(i,k)=
// d_Q[i+N*k]): q' = q*(T2+T1); then each row i is divided by sum_k q'(i,k). One block per
// individual (K threads); a degenerate all-zero row is reset to uniform 1/K.
void launch_admix_update_q(double* d_Q, const double* d_T2, const double* d_T1, long N, int K,
                           cudaStream_t stream);

// Log-likelihood accumulation over one SNP tile [N x t] (native FP64, atomicAdd to d_ll):
//   L += sum_valid [ G*log(a) + (2-G)*log(1-a) ],  a = clamp(A, eps, 1-eps).
void launch_admix_loglik(const double* d_Gtile, const double* d_Vtile, const double* d_A, long N,
                         long t, double eps, double* d_ll, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_ADMIXTURE_KERNELS_CUH

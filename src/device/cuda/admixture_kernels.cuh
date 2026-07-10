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

// Decode one SNP TILE [N x t] at global SNP offset s0 from the resident 2-bit packed tile into
// a tile-local column-major G/V pair (element (i,j) at i + N*j, global SNP s0 + j). Produces
// values BYTE-IDENTICAL to launch_admix_decode for the same (i, s0+j) — it is the Tier-1 wall
// fix: the fit keeps only dPacked resident (~32x smaller than the two full N x M FP64 buffers)
// and decodes a [N x tileM] slice per SNP-tile inside the existing s0 loops, mirroring how PCA
// re-standardizes from d_packed per SNP-block. d_Gt/d_Vt are the tile-local buffers (offset 0).
void launch_admix_decode_tile(const std::uint8_t* d_packed, std::size_t bytes_per_record, long N,
                              long s0, long t, double* d_Gt, double* d_Vt, cudaStream_t stream);

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

// ---------------------------------------------------------------------------------------------
// SQUAREM + batched-restart layer (`--accel squarem`). S seeds run CONCURRENTLY as one resident
// stack: every kernel below gains a leading batch dimension (blockIdx.z = seed) and a per-seed
// stride, sharing the single seed-independent G/V decode. The SQUAREM control layer is a
// SWAPPABLE wrapper over the SAME base map M — these primitives are the map's elementwise pieces
// re-expressed batched, plus the three accelerator-only kernels (norms/combine/project). An
// optional `d_active` mask (length S; nullptr = all active) freezes converged/settled seeds to
// no-ops so the batch stays uniform for the strided-batched GEMMs. All buffers are column-major.

// Batched binomial responsibilities. G/V are seed-independent (shared, indexed by the tile idx);
// A/R2/R1 are the per-seed tile at stride `stride_tile` (= N*tileM). One block-z per seed.
void launch_admix_responsibility_b(const double* d_Gtile, const double* d_Vtile, const double* d_A,
                                   long N, long t, double eps, double* d_R2, double* d_R1, int S,
                                   long stride_tile, cudaStream_t stream);

// Batched multiplicative F-update. F stride = M*K; S2/S1 stride = K*M. Seeds with
// d_active[s]==0 are skipped (frozen).
void launch_admix_update_f_b(double* d_F, const double* d_S2, const double* d_S1, long M, int K,
                             double eps, int S, long stride_F, long stride_S,
                             const double* d_active, cudaStream_t stream);

// Batched complement Fc = 1 - F over [M x K] per seed (stride = M*K). Frozen seeds skipped.
void launch_admix_complement_b(const double* d_F, double* d_Fc, long MK, int S, long stride,
                               const double* d_active, cudaStream_t stream);

// Batched multiplicative Q-update + row-simplex renormalize. Q stride = N*K. Frozen seeds skipped.
void launch_admix_update_q_b(double* d_Q, const double* d_T2, const double* d_T1, long N, int K,
                             int S, long stride_Q, const double* d_active, cudaStream_t stream);

// Batched log-likelihood: accumulates each seed's L into d_ll[s] (length S). A stride = N*tileM.
void launch_admix_loglik_b(const double* d_Gtile, const double* d_Vtile, const double* d_A, long N,
                           long t, double eps, double* d_ll, int S, long stride_tile,
                           cudaStream_t stream);

// SQUAREM fused norms: reads theta0/theta1/theta2 directly (no r,v materialize) and accumulates
//   rr[s] += sum(theta1-theta0)^2,  vv[s] += sum(theta2-2*theta1+theta0)^2   (native FP64,
// segmented by seed). Call once per state part (Q then F) into the SAME rr/vv (memset first).
void launch_admix_squarem_norms(const double* d_t0, const double* d_t1, const double* d_t2,
                                long len, long stride, double* d_rr, double* d_vv, int S,
                                cudaStream_t stream);

// SQUAREM extrapolation combine: out = c0*t0 + c1*t1 + c2*t2 with per-seed alpha (d_alpha[s]):
//   c0=(1+a)^2, c1=-2a(1+a), c2=a^2  (alpha=-1 => (0,0,1) => out=t2 exactly). Frozen seeds skipped.
void launch_admix_squarem_combine(const double* d_t0, const double* d_t1, const double* d_t2,
                                  double* d_out, long len, long stride, const double* d_alpha,
                                  const double* d_active, int S, cudaStream_t stream);

// Feasibility projection of theta' before the stabilizing map. project_f: clamp F to [eps,1-eps]
// + refresh Fc. project_q: per-row clamp-to-nonneg + row-renormalize (uniform 1/K on a dead row).
void launch_admix_project_f(double* d_F, double* d_Fc, long MK, int S, long stride, double eps,
                            const double* d_active, cudaStream_t stream);
void launch_admix_project_q(double* d_Q, long N, int K, int S, long stride_Q,
                            const double* d_active, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_ADMIXTURE_KERNELS_CUH

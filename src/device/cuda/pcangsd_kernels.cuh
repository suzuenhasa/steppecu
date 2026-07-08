#pragma once
// src/device/cuda/pcangsd_kernels.cuh
//
// Launch declarations for the PCAngsd (`steppe pcangsd`) elementwise device
// kernels. The matmul-heavy steps (the E^T E gram, the V V^T projector, the
// reconstruction P = W E, and the final covariance gram) are issued by
// cuda_backend_pcangsd.cu via cuBLAS SYRK/GEMM + cuSOLVER Dsyevd (reused from the
// `steppe pca` path); these launchers cover the native-FP64 elementwise kernels the
// IAF EM adds. All frequencies are allele-2 (the tile stores l[base+g] = P(g copies
// of A1), so Lrr=l[base+2], Lhet=l[base+1], Laa=l[base+0]). See pcangsd_em.hpp for
// the reference math these reproduce.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// emMAF: one thread per site over the WHOLE resident tensor. Fixed-point EM for the
// population allele-2 frequency f[site] (init 0.25, f = (1/2N) Σ_i E_dosage(L,f)),
// stopping at rmse1d < maf_tol or maf_iter iterations. Native FP64.
void launch_pcangsd_emmaf(const double* d_l, long n_site, int n_sample, int maf_iter,
                          double maf_tol, double* d_f_all, cudaStream_t stream);

// build E (N x Mw column-major, dE[i + jj*N]) over the kept sites. When d_P is null
// the weight frequency is the population f (updateNormal init); otherwise it is the
// clipped individual frequency pi = clip((P[i+jj*N]+2f)/2, 1e-4, 1-1e-4)
// (updatePCAngsd). E is centered by 2*f_j always; when standardize, also scaled by
// dj = 1/sqrt(2 f_j(1-f_j)) (covPCAngsd). Native FP64.
void launch_pcangsd_build_E(const double* d_l, const int* d_kept, const double* d_fk,
                            const double* d_P, long n_sample_l, long Mw, int n_sample,
                            bool standardize, double* d_E, cudaStream_t stream);

// dCov diagonal: one thread per individual, reduces over the kept sites the
// per-individual variance term dCov[i] = Σ_j [(-2f)^2 p0/pSum + (1-2f)^2 p1/pSum +
// (2-2f)^2 p2/pSum]/(2 f(1-f)), weights p0/p1/p2 from the clipped individual freq.
void launch_pcangsd_dcov(const double* d_l, const int* d_kept, const double* d_fk,
                         const double* d_P, long n_sample_l, long Mw, int n_sample,
                         double* d_dcov, cudaStream_t stream);

// Sum of squared differences Σ (P - Pprev)^2 over n elements -> *d_acc (grid-stride
// block reduction + atomicAdd). The caller forms the pi-scale rmse2d from it.
void launch_pcangsd_sqdiff(const double* d_P, const double* d_Pprev, long n, double* d_acc,
                           cudaStream_t stream);

// Finalize the covariance: set the diagonal C[i+i*N] = dCov[i] and scale every entry
// by inv_M. One thread per (a,b) over the full N x N (row/col-major symmetric).
void launch_pcangsd_finalize_cov(double* d_C, const double* d_dcov, int N, double inv_M,
                                 cudaStream_t stream);

// Write the individual allele-2 frequency pi (M_used x N row-major, site-major) from
// the reconstruction P and the population f: pi[jj*N+i] = clip((P[i+jj*N]+2f)/2).
void launch_pcangsd_pi(const double* d_P, const double* d_fk, long Mw, int n_sample,
                       double* d_pi, cudaStream_t stream);

}  // namespace steppe::device

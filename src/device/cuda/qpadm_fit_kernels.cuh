// src/device/cuda/qpadm_fit_kernels.cuh
//
// Narrow `void launch_*` wrapper declarations for the qpAdm fit device kernels;
// the kernel bodies and <<<>>> launches live in the matching .cu. Names CUDA
// types (cudaStream_t), so this header is private to steppe_device.
//
// Reference: docs/reference/src_device_cuda_qpadm_fit_kernels.cuh.md
#ifndef STEPPE_DEVICE_CUDA_QPADM_FIT_KERNELS_CUH
#define STEPPE_DEVICE_CUDA_QPADM_FIT_KERNELS_CUH

#include <cuda_runtime.h>

namespace steppe::device {

// f4/f3 gather kernels — reference §3
void launch_assemble_f4_gather(const double* d_f2, int P,
                               const int* d_left, const int* d_right,
                               int nl, int nr, int nb, const int* d_surv,
                               double* dX, cudaStream_t stream);

void launch_assemble_f4_quartets_gather(const double* d_f2, int P,
                                        const int* d_quartets, int N, int nb,
                                        const int* d_surv,
                                        double* dX, cudaStream_t stream);

void launch_assemble_f3_triples_gather(const double* d_f2, int P,
                                       const int* d_triples, int N, int nb,
                                       const int* d_surv,
                                       double* dX, cudaStream_t stream);

// Missing-block keep mask — reference §4
void launch_f2_block_keep(const double* d_vpair, int P, int nb, int* d_keep,
                          cudaStream_t stream);

// Block-jackknife statistics — reference §5
void launch_f4_loo_total(const double* dX, const int* d_block_sizes,
                         int m, int nb, double n,
                         double* dLoo, double* dTotal, double* dTotLine,
                         cudaStream_t stream);

void launch_f4_xtau(const double* dLoo, const double* dEst, const double* dTotLine,
                    const int* d_block_sizes, int m, int nb, double n,
                    double* dXtau, cudaStream_t stream);

void launch_f4_diag_var(const double* dXtau, int m, int nb, double* dVar,
                        cudaStream_t stream);


// All-combinations sweep kernels — reference §11
void launch_sweep_unrank_quartets(long long c0, int C, int range, const int* d_subset,
                                  int* dQuartets, cudaStream_t stream);

void launch_sweep_unrank_triples(long long c0, int C, int range, const int* d_subset,
                                 int* dTriples, cudaStream_t stream);

void launch_sweep_zfilter(const double* dXtotal, const double* dVar, int C, int mode,
                          double min_z, double* dEst, double* dSe, double* dZ,
                          unsigned char* d_flags, cudaStream_t stream);

void launch_sweep_deinterleave_keys(const int* d_items, int C, int k,
                                    int* d_c0, int* d_c1, int* d_c2, int* d_c3,
                                    cudaStream_t stream);


// Bounded device top-K reservoir — reference §12
void launch_sweep_zfilter_tau(const double* dXtotal, const double* dVar, int C,
                              const double* d_tau, double* dEst, double* dSe, double* dZ,
                              double* dAbsZ, unsigned char* d_flags, cudaStream_t stream);

void launch_sweep_topk_iota(int* d_idx, int n, cudaStream_t stream);

void launch_sweep_topk_gather(const int* d_perm, int m,
                              const double* inEst, const double* inSe, const double* inZ,
                              const double* inAbsZ, const int* inC0, const int* inC1,
                              const int* inC2, const int* inC3,
                              double* outEst, double* outSe, double* outZ, double* outAbsZ,
                              int* outC0, int* outC1, int* outC2, int* outC3,
                              cudaStream_t stream);

void launch_sweep_topk_raise_tau(const double* d_sorted_absz, int K, int mode,
                                 double* d_tau, cudaStream_t stream);

// Covariance-matrix helpers — reference §6
void launch_symmetrize_lower_to_full(double* dM, int n, cudaStream_t stream);

void launch_add_fudge_diag(double* dM, int n, double fudge, double tr,
                           cudaStream_t stream);


// Small-dense fit core — reference §7
void launch_qpadm_seed_ab(const double* dXmat, int nl, int nr, int r,
                          double* dA, double* dB, cudaStream_t stream);

void launch_qpadm_rank_via_jacobi(const double* dQ, int m, double eps,
                                  double* dScratch, int* dIntScratch, int* dRank,
                                  cudaStream_t stream);

void launch_qpadm_als(const double* dXmat, const double* dQinv,
                      int nl, int nr, int r, double fudge, int als_iters,
                      double* dA, double* dB, cudaStream_t stream);

void launch_qpadm_weights_chisq(const double* dXmat, const double* dQinv,
                                const double* dA, const double* dB,
                                int nl, int nr, int r,
                                double* dW, double* dchisq, int* d_status,
                                cudaStream_t stream);

void launch_qpadm_xmat_from_rowmajor(const double* dTotalSrc, int nl, int nr,
                                     double* dXmat, cudaStream_t stream);

// Batched leave-one-block-out re-fits — reference §8
void launch_qpadm_loo_batched(const double* dLoo, const double* dQinv,
                              int nl, int nr, int r, double fudge, int als_iters,
                              int nb, double* dWmat, cudaStream_t stream);


// Large-model path — reference §9
void launch_transpose_small(const double* dXmat, int nl, int nr,
                            double* dXt, cudaStream_t stream);

void launch_qpadm_seed_from_V(const double* dXmat, const double* dVout,
                              int nl, int nr, int r,
                              double* dA, double* dB, cudaStream_t stream);

void launch_qpadm_als_large(const double* dXmat, const double* dQinv,
                            int nl, int nr, int r, double fudge, int als_iters,
                            double* dA, double* dB,
                            double* dScratch, int* dIntScratch, cudaStream_t stream);

void launch_qpadm_weights_chisq_large(const double* dXmat, const double* dQinv,
                                      const double* dA, const double* dB,
                                      int nl, int nr, int r,
                                      double* dW, double* dchisq, int* d_status,
                                      double* dScratch, int* dIntScratch,
                                      cudaStream_t stream);

void launch_qpadm_loo_large_batched(const double* dLoo, const double* dQinv,
                                    const double* dAseed, const double* dBseed,
                                    int nl, int nr, int r, double fudge, int als_iters,
                                    int nb, int n_models, long dbl_refit, long int_refit,
                                    double* dScratch, int* dIntScratch, double* dWmat,
                                    cudaStream_t stream);


// Model-batched (rotation) kernels — reference §10
void launch_assemble_f4_gather_models_batched(const double* d_f2, int P,
                                              const int* d_left_arena,
                                              const int* d_right_arena,
                                              int nl, int nr, int nb, int n_models,
                                              const int* d_surv,
                                              double* dX, cudaStream_t stream);

void launch_f4_loo_total_models_batched(const double* dX, const int* d_block_sizes,
                                        int m, int nb, double n, int n_models,
                                        double* dLoo, double* dTotal, double* dTotLine,
                                        cudaStream_t stream);

void launch_f4_xtau_models_batched(const double* dLoo, const double* dEst,
                                   const double* dTotLine, const int* d_block_sizes,
                                   int m, int nb, double n, int n_models,
                                   double* dXtau, cudaStream_t stream);

void launch_add_fudge_diag_models_batched(const double* dQ, double* dQf, int m,
                                          double fudge, int n_models, cudaStream_t stream);

void launch_fill_identity_batched(double* dI, int m, int n_models, cudaStream_t stream);

void launch_qpadm_fit_models_batched(const double* dTotal, const double* dQinv,
                                     const double* dLoo, const int* d_block_sizes,
                                     int nl, int nr, int r_fit, int rmax,
                                     double fudge, int als_iters, int nb, int n_models,
                                     double* d_weight, double* d_se, double* d_chisq,
                                     int* d_status, double* d_rank_chisq,
                                     double* d_pop_chisq, double* d_pop_wfull,
                                     cudaStream_t stream);

void launch_qpadm_loo_models_batched(const double* dLoo, const double* dQinv,
                                     int nl, int nr, int r_fit, double fudge,
                                     int als_iters, int nb, int n_models, double s,
                                     double* dWmat, cudaStream_t stream);

void launch_qpadm_se_from_wmat_batched(const double* dWmat, int nl, int nb,
                                       int n_models, double* d_se, cudaStream_t stream);

void launch_qpadm_gather_loo_qinv(const double* dLooSrc, const double* dQinvSrc,
                                  const int* d_surv, int m, int nb, int n_surv,
                                  double* dLooDst, double* dQinvDst, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_QPADM_FIT_KERNELS_CUH

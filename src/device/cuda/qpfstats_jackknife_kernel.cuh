// src/device/cuda/qpfstats_jackknife_kernel.cuh — the narrow launch seam for the qpfstats
// ON-DEVICE block-JACKKNIFE kernels (the PERF path; the CUDA-private bodies live in
// qpfstats_jackknife_kernel.cu; architecture.md §7). The backend reaches them ONLY through
// these wrappers. NATIVE FP64 — the §12 cancellation carve-out for the leave-one-out
// jackknife difference (NEVER EmulatedFp64), reproducing the host long-double reference
// (core/internal/qpfstats_jackknife.hpp) at the golden tier.
//
// This header names a CUDA type (cudaStream_t) ⇒ PRIVATE to steppe_device (architecture.md §4).
#ifndef STEPPE_DEVICE_CUDA_QPFSTATS_JACKKNIFE_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_QPFSTATS_JACKKNIFE_KERNEL_CUH

#include <cuda_runtime.h>

namespace steppe::device {

/// The per-comb numer + ymat materialize + the GLOBAL block-jackknife est y, ON-DEVICE, from
/// the RESIDENT numsum/cnt (ROW-MAJOR [npopcomb × n_block]). One thread owns one comb c
/// (embarrassingly parallel over the ~305k combs; reduces over n_block). Reproduces AT2
/// matrix_jackknife_est_col EXACTLY (same NaN/cnt<=0 mask, rel>=1 skip, loo isfinite guard,
/// the ascending-b accumulation order) in native FP64. Writes:
///   d_numer  ROW-MAJOR [npopcomb × n_block]: numsum/cnt (NaN where cnt<=0).
///   d_ymat   COL-MAJOR [npopcomb × n_block]: ymat[c + npopcomb*b] = the same mean.
///   d_y      [npopcomb]: the per-comb global jackknife estimate (the bglob RHS source).
void launch_qpfstats_numer_jackknife(const double* d_numsum, const double* d_cnt,
                                     int npopcomb, int n_block, double* d_numer,
                                     double* d_ymat, double* d_y, cudaStream_t stream);

/// The per-pair RECENTER jackknife, ON-DEVICE, from the RESIDENT smoothed b (COL-MAJOR
/// [npairs × n_block]) + bglob [npairs] + the per-block SNP counts block_sizes [n_block].
/// One thread owns one pair p. Reproduces AT2 f2blocks_pair_est (the f2(f2blocks)$est: tot =
/// weighted.mean(arr, bl), loo = (tot - arr·rel)/(1-rel), est = weighted.mean(loo, 1-1/h)) in
/// native FP64, then writes d_shift[p] = bglob[p] - est (the per-pair recenter constant).
void launch_qpfstats_recenter_shift(const double* d_b, const double* d_bglob,
                                    const int* d_block_sizes, int npairs, int n_block,
                                    double* d_shift, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_QPFSTATS_JACKKNIFE_KERNEL_CUH

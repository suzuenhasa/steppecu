// src/device/cuda/ratio_block_jackknife_kernel.cuh — the narrow launch seam for the SHARED
// on-device RATIO-block-jackknife kernel (the f4ratio M1 + qpDstat M2 common engine; the
// CUDA-private body lives in ratio_block_jackknife_kernel.cu; architecture.md §7). The backend
// reaches the kernel ONLY through this wrapper. NATIVE FP64 — the §12 cancellation carve-out
// for the ratio jackknife difference (NEVER EmulatedFp64), reproducing the host long-double
// reference (f4ratio.cpp ratio_jackknife / dstat.cpp dstat_jackknife) at the golden tier.
//
// This header names a CUDA type (cudaStream_t) ⇒ PRIVATE to steppe_device (architecture.md §4).
#ifndef STEPPE_DEVICE_CUDA_RATIO_BLOCK_JACKKNIFE_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_RATIO_BLOCK_JACKKNIFE_KERNEL_CUH

#include <cuda_runtime.h>

namespace steppe::device {

/// Device-passable indexing descriptor for ONE per-(item,block) input array (the CUDA twin of
/// backend.hpp RatioJackArray). The kernel reads element (item k, block b) at
///   data[base + k*item_stride + b*block_stride].
/// A null `data` ⇒ the array is absent (the dstat x_blocks pair). A 0 item_stride BROADCASTS
/// across items (the f4ratio per-block block_sizes weight). All pointers are DEVICE pointers.
struct DRatioJackArray {
    const double* data;
    long base;
    long item_stride;
    long block_stride;
};

/// One thread per item k, grid-stride over the N items, each owning a SHORT register reduction
/// over the n_block axis. Reproduces the host long-double ratio block-jackknife EXACTLY (same
/// survivor mask, same ascending-b operand order, same $est / xtau-variance form) in native
/// FP64. tot_mode selects f4ratio (0: block-sum tot; num/den ARE the est_to_loo replicates;
/// reads xblk_num/xblk_den) vs dstat (1: LOO-ratio tot; num/den ARE the per-block sums with
/// weight=cnt; the LOO replicate is built in-kernel). setmiss_thresh>0 ⇒ |xblk_den|<thresh
/// drops the block (f4ratio AT2 setmiss); else weight<=0 drops it (dstat cnt>0). When
/// compute_p the kernel writes p[k]=erfc(|z|/sqrt2) (AT2 ztop / f4_two_sided_p); else p untouched.
/// Outputs est/se/z (and optionally p), length N each.
void launch_ratio_block_jackknife(const DRatioJackArray& num, const DRatioJackArray& den,
                                  const DRatioJackArray& weight, const DRatioJackArray& xblk_num,
                                  const DRatioJackArray& xblk_den, int N, int n_block,
                                  int tot_mode, double setmiss_thresh, bool compute_p,
                                  double* d_est, double* d_se, double* d_z, double* d_p,
                                  cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_RATIO_BLOCK_JACKKNIFE_KERNEL_CUH

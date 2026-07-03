// src/device/cuda/ratio_block_jackknife_kernel.cuh — launch seam for the shared on-device
// ratio-block-jackknife kernel (the f4ratio + qpDstat common engine; body in the paired .cu).
// Names a CUDA type (cudaStream_t), so this header is private to steppe_device.
//
// Reference: docs/reference/src_device_cuda_ratio_block_jackknife_kernel.cuh.md
#ifndef STEPPE_DEVICE_CUDA_RATIO_BLOCK_JACKKNIFE_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_RATIO_BLOCK_JACKKNIFE_KERNEL_CUH

#include <cuda_runtime.h>

namespace steppe::device {

// Per-(item,block) input array descriptor — reference §3
struct DRatioJackArray {
    const double* data;
    long base;
    long item_stride;
    long block_stride;
};

// Kernel launch seam — reference §4
void launch_ratio_block_jackknife(const DRatioJackArray& num, const DRatioJackArray& den,
                                  const DRatioJackArray& weight, const DRatioJackArray& xblk_num,
                                  const DRatioJackArray& xblk_den, int N, int n_block,
                                  int tot_mode, double setmiss_thresh, bool compute_p,
                                  double* d_est, double* d_se, double* d_z, double* d_p,
                                  cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_RATIO_BLOCK_JACKKNIFE_KERNEL_CUH

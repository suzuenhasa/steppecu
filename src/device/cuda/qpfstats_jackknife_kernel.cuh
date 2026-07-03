// src/device/cuda/qpfstats_jackknife_kernel.cuh
//
// The host-callable launch seam for the two qpfstats on-device block-jackknife
// kernels; the kernel bodies live in the matching .cu file. Names a CUDA type
// (cudaStream_t), so this header is private to the device library.
//
// Reference: docs/reference/src_device_cuda_qpfstats_jackknife_kernel.cuh.md
#ifndef STEPPE_DEVICE_CUDA_QPFSTATS_JACKKNIFE_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_QPFSTATS_JACKKNIFE_KERNEL_CUH

#include <cuda_runtime.h>

namespace steppe::device {

// numer / global-estimate jackknife launch — reference §6
void launch_qpfstats_numer_jackknife(const double* d_numsum, const double* d_cnt,
                                     int npopcomb, int n_block, double* d_numer,
                                     double* d_ymat, double* d_y, cudaStream_t stream);

// recenter-shift jackknife launch — reference §7
void launch_qpfstats_recenter_shift(const double* d_b, const double* d_bglob,
                                    const int* d_block_sizes, int npairs, int n_block,
                                    double* d_shift, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_QPFSTATS_JACKKNIFE_KERNEL_CUH

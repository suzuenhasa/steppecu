// src/device/cuda/readv2_mismatch_kernel.cuh
//
// Reference: docs/reference/src_device_cuda_readv2_mismatch_kernel.cuh.md
//
// launch_readv2_mismatch: the all-pairs AND-mask / XOR / __popcll windowed-mismatch
// reduction over the resident [sample x SNP-window] bit-matrix — steppe's first
// __popc kernel. For every unordered sample pair it produces the four per-pair
// reductions the host turns into P0_mean / SE:
//   sum_p0     : sum over windows of per-window mismatch/comparable (double)
//   sum_p0_sq  : sum over windows of that ratio squared     (double, for the SE)
//   n_win_used : count of windows with >0 comparable sites  (int   -> n_windows)
//   tot_comp   : total comparable sites across all windows   (int64 -> n_overlap_sites)
//
// Two kernels behind the launcher (mirroring dstat_kernel's dual): a baseline
// one-thread-per-pair reduction (the correctness gate) and a shared-memory tiled
// reduction that caches sample rows per window (the throughput follow). Private to
// steppe_device.
#ifndef STEPPE_DEVICE_CUDA_READV2_MISMATCH_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_READV2_MISMATCH_KERNEL_CUH

#include <cstdint>

#include <cuda_runtime.h>

#include "device/cuda/readv2_layout.cuh"

namespace steppe::device {

void launch_readv2_mismatch(const Readv2Word* d_words,
                            long words_per_sample,
                            int wpw,
                            long n_win,
                            int n_samples,
                            long long n_pairs,
                            double* d_sum_p0,
                            double* d_sum_p0_sq,
                            int* d_n_win_used,
                            long long* d_tot_comp,
                            bool tiled,
                            cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_READV2_MISMATCH_KERNEL_CUH

// src/device/cuda/roh_fb_kernel.cuh
//
// Launch-wrapper declaration for the hapROH (K+1)-state copying forward-backward GPU
// kernel (the `steppe roh` FB core). A device-internal seam (names cudaStream_t),
// private to steppe_device; the body + <<<>>> live in the .cu.
//
// The kernel runs ONE CUDA block per target (the M-sequential scan is confined to a
// block; __syncthreads() is the per-column barrier) and materializes the per-SNP ROH
// posterior with the LS engine's always-on checkpoint/recompute memory scheme (a
// target's O(K*M) forward table is never fully resident — large K forces this). The
// K ROH states spread over threadIdx.x; the distinguished non-ROH state 0 is carried in
// a shared scalar so the pooled ROH-mass reduction cleanly excludes it. Only the per-SNP
// ROH posterior (1 - gamma_0) leaves the device.
//
// Reference: docs/planning/haproh-face-spec.md §3
#ifndef STEPPE_DEVICE_CUDA_ROH_FB_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_ROH_FB_KERNEL_CUH

#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// Batched (K+1)-state scaled forward-backward. One block per target (grid.x = n_target),
// K ROH states spread over threadIdx.x. All buffers are device pointers owned by the
// backend:
//   d_ob        n_target*M      u8   target observed alleles {0,1,missing} (target-major)
//   d_refhaps   K*M             u8   reference-haplotype panel, donor-major (shared by targets)
//   d_p         M               f64  per-SNP panel allele frequency
//   d_T         M*9             f64  per-SNP transition tensor t[l*9 + r*3 + c]
//   d_proh      n_target*M      f64  OUTPUT ROH posterior 1 - gamma_0, target-major
//   d_check_roh n_target*nck*K  f64  normalized ROH alpha at each checkpoint column
//   d_check0    n_target*nck    f64  normalized state-0 alpha at each checkpoint column
//   d_alphaA/B  n_target*K      f64  ROH forward ping-pong columns
//   d_alpha_blk n_target*C*K    f64  recomputed backward ROH tile (C columns)
//   d_a0_blk    n_target*C      f64  recomputed backward state-0 tile (C columns)
//   d_betaA/B   n_target*K      f64  ROH backward ping-pong columns
// C is the checkpoint stride (= ceil(sqrt(M))); nck = ceil(M/C) blocks.
void launch_roh_fb(const std::uint8_t* d_ob, const std::uint8_t* d_refhaps, const double* d_p,
                   const double* d_T, int K, long M, int n_target, int C, int nck, double e_rate,
                   double in_val, double* d_proh, double* d_check_roh, double* d_check0,
                   double* d_alphaA, double* d_alphaB, double* d_alpha_blk, double* d_a0_blk,
                   double* d_betaA, double* d_betaB, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_ROH_FB_KERNEL_CUH

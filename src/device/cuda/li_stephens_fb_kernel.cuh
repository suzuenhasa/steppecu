// src/device/cuda/li_stephens_fb_kernel.cuh
//
// Launch-wrapper declaration for the Li-Stephens copying forward-backward GPU
// kernel (the `steppe paint` FB core, Phase 1). This header names CUDA types
// (cudaStream_t) and so is private to steppe_device — the device-internal seam
// between the backend and the FB kernel, whose body and <<<>>> live in the .cu.
//
// The kernel runs ONE CUDA block per recipient (the M-sequential scan is confined
// to a block; __syncthreads() is the per-column barrier) and materializes the
// per-column copying posterior gamma with an always-on checkpoint/recompute
// memory scheme (a recipient's O(K*M) forward table is never fully resident).
// Reference: docs/planning/li-stephens-engine-scope.md §2a.
#ifndef STEPPE_DEVICE_CUDA_LI_STEPHENS_FB_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_LI_STEPHENS_FB_KERNEL_CUH

#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// Batched forward-backward launch. One block per recipient (grid.x = n_recip),
// donors spread over threadIdx.x (grid-stride over K). All buffers are device
// pointers owned by the backend:
//   d_recipient  n_recip*M   u8   recipient alleles (recipient-major)
//   d_donors     K*M         u8   donor panel, donor-major (shared by recipients)
//   d_pi         n_recip*K   f64  copying prior per recipient
//   d_rho        M           f64  recombination probs (rho[0] ignored)
//   d_mu         M           f64  per-site emission rate
//   d_gamma      n_recip*K*M f64  OUTPUT, donor-major, each SNP column normalized
//   d_checkpts   n_recip*nck*K f64  normalized alpha at each checkpoint column
//   d_alphaA/B   n_recip*K   f64  forward ping-pong columns
//   d_alpha_blk  n_recip*C*K f64  recomputed backward tile (C columns)
//   d_betaA/B    n_recip*K   f64  backward ping-pong columns
// C is the checkpoint stride (= ceil(sqrt(M))); nck = ceil(M/C) blocks.
void launch_ls_forward_backward(const std::uint8_t* d_recipient, const std::uint8_t* d_donors,
                                const double* d_pi, const double* d_rho, const double* d_mu,
                                int K, long M, int n_recip, int C, int nck, double* d_gamma,
                                double* d_checkpts, double* d_alphaA, double* d_alphaB,
                                double* d_alpha_blk, double* d_betaA, double* d_betaB,
                                cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_LI_STEPHENS_FB_KERNEL_CUH

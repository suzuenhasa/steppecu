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
// Reference: docs/reference/src_device_cuda_li_stephens_fb_kernel.cuh.md
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
//
// The final four (defaulted-null) buffers select the PAINT coancestry sink (Phase 2):
//   d_w             M             f64  per-SNP genetic-length weight (chunklengths)
//   d_acc_cnt       n_recip*K     f64  OUTPUT, expected #chunks per donor (recipient-major)
//   d_acc_len       n_recip*K     f64  OUTPUT, expected copied length per donor (Morgans)
//   d_checkpts_prev n_recip*nck*K f64  companion checkpoints (alpha at column bi*C-1)
// When they are null the kernel runs the GATE path and writes d_gamma (byte-identical to
// Phase 1); when d_acc_cnt is non-null it runs the paint sink and d_gamma is ignored
// (pass nullptr).
//
// The final three (defaulted-null) buffers select the LOCALANC per-SNP sink (Phase 3):
//   d_donor_group  K             i32  ancestry-label index per donor (in [0,P))
//   d_post         n_recip*M*P   f64  OUTPUT, per-SNP per-label posterior, layout
//                                     post[(rid*M + l)*P + g] (recipient-major)
//   P              scalar             number of ancestry labels
// When d_post is non-null the kernel runs the localanc sink (d_gamma AND d_acc_cnt pass
// nullptr; d_checkpts_prev is unused). Exactly one of {d_gamma, d_acc_cnt, d_post} is the
// live output — the host orchestrators guarantee it.
void launch_ls_forward_backward(const std::uint8_t* d_recipient, const std::uint8_t* d_donors,
                                const double* d_pi, const double* d_rho, const double* d_mu,
                                int K, long M, int n_recip, int C, int nck, double* d_gamma,
                                double* d_checkpts, double* d_alphaA, double* d_alphaB,
                                double* d_alpha_blk, double* d_betaA, double* d_betaB,
                                cudaStream_t stream, const double* d_w = nullptr,
                                double* d_acc_cnt = nullptr, double* d_acc_len = nullptr,
                                double* d_checkpts_prev = nullptr,
                                const int* d_donor_group = nullptr, double* d_post = nullptr,
                                int P = 0);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_LI_STEPHENS_FB_KERNEL_CUH

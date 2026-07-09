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

// Wave-batched (K+1)-state scaled forward-backward: ONE grid of W item-blocks
// (blockIdx.x = item index within the wave), each block indexing the ONCE-resident donor-
// major panel DIRECTLY through the per-item site_map (no per-item compacted refhaps gather).
// Same block-per-item FB body as launch_roh_fb — bit-identical math — but hundreds of items
// resident concurrently instead of the 3-stream pipeline's ~2-3. All device pointers owned
// by the backend; per-item M/C/nck come from the d_M/d_C/d_nck arrays, buffers are strided
// by the batch-wide Mstride (= Mmax) / Cstride (= Cmax = nck bound); site_map is strided by
// Mstride too. The panel accessor returns panel[donor_map[k]*Mp + site_map[l]] — byte-for-
// byte the value roh_gather wrote into the old compacted refhaps, so the FB input is identical.
//   d_ob        W*Mstride    u8   per-item observed alleles (item-major)
//   d_panel     Kpanel*Mp    u8   ONCE-resident donor-major panel bytes
//   d_donor_map K            i32  resident-panel row per selected donor
//   d_site_map  W*Mstride    i32  per-item panel column (KeptSite::panel_l) per kept site
//   d_p/d_T     W*Mstride(*9) f64 per-item allele frequency / transition tensor
//   d_M/d_C/d_nck  W         i32  per-item kept-site count / checkpoint stride / block count
void launch_roh_fb_wave(const std::uint8_t* d_ob, const std::uint8_t* d_panel,
                        const int* d_donor_map, const int* d_site_map, const double* d_p,
                        const double* d_T, const int* d_M, const int* d_C, const int* d_nck,
                        int K, long Mp, long Mstride, long Cstride, int W, double e_rate,
                        double in_val, double* d_proh, double* d_check_roh, double* d_check0,
                        double* d_alphaA, double* d_alphaB, double* d_alpha_blk, double* d_a0_blk,
                        double* d_betaA, double* d_betaB, cudaStream_t stream);

// Gather one work-item's compacted donor-major reference-haplotype scratch from the
// ONCE-resident panel (the batch-overlap residency — replaces the per-item host K*M gather
// + re-upload with a single device gather over the shared panel). For k in [0,K), l in
// [0,M):  d_refhaps[k*M + l] = d_panel[d_donor_map[k]*Mp + d_site_map[l]]  — byte-for-byte
// the panel bytes the serial host loop copied, so the FB input is identical. K<=0 || M<=0
// launches nothing (empty items produce no ROH posterior).
//   d_panel     Kpanel*Mp    u8   resident donor-major panel bytes
//   d_donor_map K            i32  resident-panel row per selected donor
//   d_site_map  M            i32  panel column (KeptSite::panel_l) per kept site
//   d_refhaps   K*M          u8   OUTPUT compacted donor-major scratch (FB input)
void launch_roh_gather(const std::uint8_t* d_panel, const int* d_donor_map,
                       const int* d_site_map, int K, long M, long Mp, std::uint8_t* d_refhaps,
                       cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_ROH_FB_KERNEL_CUH

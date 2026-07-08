// src/device/cuda/ancibd_fb_kernel.cuh
//
// Launch-wrapper declarations for the ancIBD 5-state pairwise forward-backward GPU
// kernels (the `steppe ibd` FB core). A device-internal seam (names cudaStream_t),
// private to steppe_device; the bodies + <<<>>> live in the .cu.
//
// Two kernels:
//   (1) launch_ancibd_derive_hts — derive the per-haplotype ancestral-allele
//       probability table d_hts[(2*sample+h)*M + site] from the device-resident GP
//       tensor + phased-GT bits (LoadH5Multi2.get_haplo_prob), one thread per
//       (site, sample).
//   (2) launch_ancibd_fb — the 5-state scaled forward-backward: ONE CUDA block
//       (one warp) per pair, batched over all pairs; native-FP64 per-column rescale;
//       gathers the pair's four haplotype rows from d_hts and writes the per-SNP IBD
//       posterior d_pibd[pair*M + site] = 1 - post[0]. Only fwd[0] and the column
//       normalizers are kept resident (2*M doubles/pair) — no 5*M table, no
//       checkpoint/recompute (M is small enough).
//
// Reference: docs/planning/ancibd-face-spec.md §3
#ifndef STEPPE_DEVICE_CUDA_ANCIBD_FB_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_ANCIBD_FB_KERNEL_CUH

#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// Derive d_hts[(2*sample+h)*M + site] (h in {0,1}) from the GP tensor + phased bits.
//   d_gp3     M*n_sample*3  f64  site-major GP triplet (VCF-native order; g0,g1 used)
//   d_phased2 M*n_sample*2  u8   site-major phased GT allele bits {0,1} per haplotype
//   d_hts     (2*n_sample)*M f64 OUTPUT haplotype ancestral probs, hap-row-major
void launch_ancibd_derive_hts(const double* d_gp3, const std::uint8_t* d_phased2, double* d_hts,
                              int n_sample, long M, double min_error, cudaStream_t stream);

// Batched 5-state scaled forward-backward. One block (warp) per pair, grid.x = n_pair.
//   d_hts      (2*n_sample)*M f64  haplotype ancestral probs (from derive_hts)
//   d_p        M              f64  per-SNP derived (ALT) allele frequency
//   d_T        M*9            f64  per-SNP transition tensor t[l*9 + r*3 + c]
//   d_pair_idx n_pair*2       i32  the two sample indices per pair
//   d_fwd0     n_pair*M       f64  scratch: state-0 forward, per pair (resident)
//   d_c        n_pair*M       f64  scratch: per-column normalizers, per pair
//   d_pibd     n_pair*M       f64  OUTPUT IBD posterior 1 - post[0], per pair
void launch_ancibd_fb(const double* d_hts, const double* d_p, const double* d_T,
                      const int* d_pair_idx, int n_sample, long M, int n_pair, double in_val,
                      double p_min, double* d_fwd0, double* d_c, double* d_pibd,
                      cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_ANCIBD_FB_KERNEL_CUH

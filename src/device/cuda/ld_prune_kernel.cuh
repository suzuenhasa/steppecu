// src/device/cuda/ld_prune_kernel.cuh
//
// GPU kernels for the windowed-r2 LD pruner (`steppe ... --ld-prune WIN:STEP:R2`), the plink2
// --indep-pairwise analogue. Three thin launch wrappers over one SNP-tile:
//   1. launch_ld_dosage_decode_snpmajor — unpack the individual-major 2-bit tile into a compact
//      SNP-major dosage pane out[(s - s_lo)*N + g] (contiguous samples per SNP -> coalesced
//      pairwise reductions), missing left as kMissingGenotypeCode.
//   2. launch_ld_variant_stats — per target SNP, the GLOBAL {nm, Σg, Σg²} over the N samples
//      (=> major-allele frequency + the monomorphic pre-removal flag on the host).
//   3. launch_ld_pairwise_over — for every within-window same-chromosome pair (index distance
//      1..window-1) the plink2 pairwise-complete integer r^2 decision cov12² > thresh·var1·var2.
// The greedy backward-scan selection itself is a cheap host loop over these device-computed
// per-pair booleans (see cuda_backend_ld_prune.cu) — the genotype-scale r^2 work is on-GPU.
#ifndef STEPPE_DEVICE_CUDA_LD_PRUNE_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_LD_PRUNE_KERNEL_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// SNP-major dosage decode of SNPs [s_lo, s_lo + n_dec) into d_dos[(s - s_lo)*N + g].
void launch_ld_dosage_decode_snpmajor(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                                      int N, long s_lo, long n_dec, std::uint8_t* d_dos,
                                      cudaStream_t stream);

// Per-target-SNP global stats: for t in [0, n_tgt) over d_dos row t and all N samples, the
// nonmissing count, dosage sum, and dosage sum-of-squares.
void launch_ld_variant_stats(const std::uint8_t* d_dos, int N, long n_tgt, long* d_nm, long* d_sum,
                             long* d_ssq, cudaStream_t stream);

// Pairwise over-threshold flags. For target t in [0, n_tgt) and distance d in [1, window-1] with
// i = s_lo + t and j = i + d, writes d_over[t*(window-1) + (d-1)] = 1 iff j < M, d_chrom[i] ==
// d_chrom[j], and cov12² > r2_thresh_eps · var1 · var2 over the pairwise-nonmissing samples; else 0.
// d_dos must cover rows [0, n_dec) (n_dec >= n_tgt + window - 1 clamped at M - s_lo). r2_thresh_eps
// is the user r^2 pre-scaled by (1 + plink kSmallEpsilon).
void launch_ld_pairwise_over(const std::uint8_t* d_dos, int N, long s_lo, long n_tgt, long n_dec,
                             long M, const int* d_chrom, int window, double r2_thresh_eps,
                             std::uint8_t* d_over, cudaStream_t stream);

// dp4a int-packed, warp-per-pair variant of launch_ld_pairwise_over — the DEFAULT pairwise path.
// Byte-identical output to launch_ld_pairwise_over (the 6 sufficient statistics are exact integers,
// so int32/dp4a accumulation + warp-shuffle reduction produces the SAME block sums as the emulated-
// int64 scalar path, and the final int64 cross-products + double test are character-for-character
// identical). One warp folds one (target, distance) pair (kLdWarpsPerBlock pairs batched per block),
// packing four samples per dp4a and masking pairwise-missing samples to zero — no emulated int64 mul
// in the hot per-sample loop, no shared-memory reduction. STEPPE_LD_KERNEL=old selects the scalar
// launch_ld_pairwise_over above as the byte-identity gate oracle + fallback.
void launch_ld_pairwise_over_dp4a(const std::uint8_t* d_dos, int N, long s_lo, long n_tgt,
                                  long n_dec, long M, const int* d_chrom, int window,
                                  double r2_thresh_eps, std::uint8_t* d_over, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_LD_PRUNE_KERNEL_CUH

// src/device/cuda/decode_compact_kernel.cuh
//
// Launch-wrapper declarations for the device-resident decode-and-compact seam:
// an on-GPU keep-mask plus column gather that reproduce the old CPU filter path
// bit-for-bit. Kernel bodies and <<<>>> live only in decode_compact_kernel.cu;
// private to steppe_device (the signatures name cudaStream_t).
//
// Reference: docs/reference/src_device_cuda_decode_compact_kernel.cuh.md
#ifndef STEPPE_DEVICE_CUDA_DECODE_COMPACT_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_DECODE_COMPACT_KERNEL_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "steppe/config.hpp"

namespace steppe::device {

// Autosome keep mask (regime A) — reference §3
void launch_autosome_keep_mask(const int* d_chrom, long M, int chrom_min,
                               int chrom_max, std::uint8_t* d_flags,
                               cudaStream_t stream);

// Full extract_f2 keep mask (regime B) — reference §4
void launch_regimeb_keep_mask(const double* d_Q, const double* d_N, int P, long M,
                              const char* d_ref, const char* d_alt,
                              const int* d_chrom, const steppe::FilterConfig& cfg,
                              double ploidy, double total_indiv, double maxmiss,
                              std::uint8_t* d_flags, cudaStream_t stream);

// Column gather and compaction — reference §5
void launch_compact_columns_gather(const double* d_in, int P, long M,
                                   const std::uint8_t* d_flags,
                                   const long* d_keep_idx, double* d_out,
                                   cudaStream_t stream);

// Packed 2-bit column compaction (apply_snp_filter device path).
// Gather the kept SNP columns of a device-resident individual-major 2-bit tile into a gap-free
// compacted tile: one thread per (individual row, output byte) reads `d_kept_cols` (the ascending
// source-column list from cub::DeviceSelect::Flagged, == repack_tile_columns' kept_cols) and packs
// the four output codes of its byte MSB-first — byte-for-byte identical to the host repack, no
// atomics (each output byte is written exactly once). `src_bytes_per_record` is the SOURCE row
// stride, `out_bytes_per_record` = ceil(n_kept/4) the destination row stride.
void launch_compact_packed_columns(const std::uint8_t* d_src,
                                   std::size_t src_bytes_per_record,
                                   const long* d_kept_cols, long n_kept,
                                   std::uint8_t* d_out,
                                   std::size_t out_bytes_per_record,
                                   std::size_t n_individuals, cudaStream_t stream);

// Device-resident pooled-per-SNP summary reduction — reference §7.
// Runs derive_pooled_summary_one over the resident [P×M] Q/N (the SHARED STEPPE_HD
// reduction, so it is bit-for-bit identical to the host loop) and writes the four
// O(M) summary planes, so the caller D2Hs 4*M doubles instead of 3*P*M. This is the
// device lever that removes the singleton-P (KING/kinship) host-decode wall.
void launch_pooled_summary(const double* d_Q, const double* d_N, int P, long M,
                           double ploidy_d, double total_indiv_d,
                           double* d_ref_af, double* d_minor_af, double* d_missing,
                           double* d_allele_count, cudaStream_t stream);

}  // namespace steppe::device
#endif  // STEPPE_DEVICE_CUDA_DECODE_COMPACT_KERNEL_CUH

// src/device/cuda/decode_compact_kernel.cuh
//
// Narrow launch-wrapper declarations for the DEVICE-RESIDENT decode seam
// (host-compute audit C2/M3/M4 cure; the device-resident decode-and-compact path):
//   1. launch_autosome_keep_mask  — the per-SNP autosome keep predicate
//      (chr in [kAutosomeChromMin, kAutosomeChromMax]) → a uint8 flag buffer, the
//      CUB DeviceSelect::Flagged flag input. Regime (A) of the host-filter audit:
//      INTEGER-EXACT (chrom is an int from .snp), reproduces the host autosome loop
//      (qpfstats.cpp:268-278 / dstat.cpp:245-255) bit-for-bit.
//   2. launch_compact_columns_gather — gather the kept P-length COLUMNS of the
//      column-major [P × M] Q/V tensors into the compacted [P × M_kept] tensors,
//      keyed by the EXCLUSIVE-SCAN compacted column index of the keep flags. This
//      is the [P×M] companion to CUB DeviceSelect::Flagged (which is 1-D, used for
//      the parallel chrom/genpos): the scan+gather preserves FILE ORDER exactly
//      (the same kept axis as the host lockstep push_back), so assign_blocks over
//      the compacted axis reproduces the host block_id.
//
// PRIVATE to steppe_device (architecture.md §4) — names cudaStream_t. The kernel
// bodies and <<<>>> live only in decode_compact_kernel.cu.
#ifndef STEPPE_DEVICE_CUDA_DECODE_COMPACT_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_DECODE_COMPACT_KERNEL_CUH

#include <cstdint>

#include <cuda_runtime.h>

#include "steppe/config.hpp"  // FilterConfig (the regime-B keep-mask thresholds)

namespace steppe::device {

/// Per-SNP autosome keep-mask: one thread per SNP s in [0, M). Writes
/// d_flags[s] = (d_chrom[s] >= chrom_min && d_chrom[s] <= chrom_max) ? 1 : 0.
/// INTEGER-EXACT (the host predicate `if (chr < 1 || chr > 22) continue;`), so the
/// on-device kept SET is bit-identical to the host autosome loop. d_flags is the
/// uint8 flag buffer consumed by CUB DeviceSelect::Flagged (value castable to bool).
void launch_autosome_keep_mask(const int* d_chrom, long M, int chrom_min,
                               int chrom_max, std::uint8_t* d_flags,
                               cudaStream_t stream);

/// REGIME-B per-SNP keep-mask (the extract_f2 FULL filter, on-device): one thread
/// per SNP s in [0, M). Reproduces the host extract_f2 keep decision EXACTLY — the
/// pooled-MAF reduction Σ_pop Q·N (FFMA-immune, the SHARED derive_pooled_summary_one,
/// p = 0..P-1 sequential ⇒ bit-identical to the host loop) + the SHARED
/// keep_decision_pooled (multiallelic / strand-ambiguous / MAF / geno / monomorphic
/// ==0.0 exact / transversion / autosome) + the SEPARATE pop-coverage maxmiss (drop
/// if the fraction of pops with N<=0 STRICTLY exceeds maxmiss; the `<=0.0` and strict
/// `>` of extract_f2_core.cpp). Membership is the extract_f2 no-op (always true).
/// `cfg` carries the thresholds (geno_max_missing FORCED to 1.0 by the caller; the
/// pop-axis maxmiss is the separate `maxmiss` arg). Writes the uint8 CUB flag.
/// d_ref/d_alt are the per-SNP .snp allele chars; d_chrom the per-SNP chromosome.
void launch_regimeb_keep_mask(const double* d_Q, const double* d_N, int P, long M,
                              const char* d_ref, const char* d_alt,
                              const int* d_chrom, const steppe::FilterConfig& cfg,
                              double ploidy, double total_indiv, double maxmiss,
                              std::uint8_t* d_flags, cudaStream_t stream);

/// Gather the kept COLUMNS of a column-major [P × M] tensor into a compacted
/// [P × M_kept] tensor. d_keep_idx is the EXCLUSIVE prefix sum of the keep flags
/// (length M): for a kept SNP s, d_keep_idx[s] is its compacted column position;
/// d_flags[s] selects whether the column is copied. One thread per (pop, src SNP);
/// kept columns are written to d_out[i + P·d_keep_idx[s]] = d_in[i + P·s]. FILE
/// ORDER preserved (the scan is monotone over s), so the compacted axis matches the
/// host lockstep subset exactly. d_in/d_out may NOT alias.
void launch_compact_columns_gather(const double* d_in, int P, long M,
                                   const std::uint8_t* d_flags,
                                   const long* d_keep_idx, double* d_out,
                                   cudaStream_t stream);

}  // namespace steppe::device
#endif  // STEPPE_DEVICE_CUDA_DECODE_COMPACT_KERNEL_CUH

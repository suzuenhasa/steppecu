#pragma once
// src/device/cuda/king_allpairs_kernel.cuh
//
// Declarations for the two kernels behind `steppe kinship` (KING-robust). Both share the
// KING classification arithmetic with the host finalize + the CPU unit-test oracle via
// core/internal/king_kinship.hpp (so the swept counts cannot drift), and the all-pairs
// enumeration via the closed-form k=2 unrank in sweep_unrank.cuh. The kernel bodies live in
// king_allpairs_kernel.cu. Mirrors fst_allpairs_kernel.cuh.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// Dosage decode: one thread per (sample g, SNP s_local) over the current SNP-tile
// [s_lo, s_lo+tm). Unpacks the 2-bit code into a compact N x tm BYTE tensor (individual-
// major: out_code[g*tm + s]); missing is kept as kMissingGenotypeCode. The individual
// partition is a singleton per sample, so record g == sample g (identity pop offsets).
// Decoding ONCE amortizes the unpack across the N-1 pairs each sample joins.
void launch_king_dosage_decode(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                               int N, long s_lo, long tm, std::uint8_t* d_code,
                               cudaStream_t stream);

// KING accumulate: ONE BLOCK per pair. When d_pairs_i / d_pairs_j are non-null the block maps
// its chunk index to that explicit (i, j) pair; otherwise it maps the flat rank r via the O(1)
// readv2_unrank_pair. The block's threads stride-share this tile's SNPs through the SHARED
// king_classify over the two samples' decoded-code slices, tree-reduce the block's five
// partial counts in shared memory, and thread 0 ADDS the block total into the persistent
// per-pair accumulators (each BLOCK owns a distinct r -> no atomics; integer -> deterministic).
// Called per SNP-tile; the += reduces across tiles. `d_include` (or nullptr) is the GLOBAL
// autosome mask, indexed by s_lo + s.
void launch_king_allpairs_accumulate(const std::uint8_t* d_code, int N, long tm, long s_lo,
                                     const std::uint8_t* d_include, const int* d_pairs_i,
                                     const int* d_pairs_j, long long pair0, long long C,
                                     long long out_offset, long* d_nsnp, long* d_hethet,
                                     long* d_ibs0, long* d_het_i, long* d_het_j,
                                     cudaStream_t stream);

}  // namespace steppe::device

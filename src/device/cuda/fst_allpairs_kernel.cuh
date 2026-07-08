#pragma once
// src/device/cuda/fst_allpairs_kernel.cuh
//
// Declarations for the two kernels behind `steppe fst --all-pairs`. Both share the WC
// variance-component arithmetic with the single-pair path and the CPU oracle via
// core/internal/wc_fst.hpp (so the matrix values cannot drift from the single-pair kernel),
// and the pair enumeration via the closed-form k=2 unrank in sweep_unrank.cuh. The kernel
// bodies live in fst_allpairs_kernel.cu.
//
// Reference: docs/reference/src_device_cuda_fst_allpairs_kernel.cuh.md

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// Sufficient-stat decode: one thread per (pop p, SNP s_local) over the current SNP-tile
// [s_lo, s_lo+tm). Folds each pop's diploid codes into the WcPerPop accumulator and STORES
// {n, ac, het} into the three P x tm tensors (pop-major: buf[p*tm + s]). This is the
// single-pair fst_wc_kernel per-pop loop generalized to all P pops and stored, not
// finalized — decoded ONCE per tile and reused across every C(P,2) pair.
void launch_fst_suffstat_decode(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                                const std::size_t* d_pop_offsets, int P, long s_lo, long tm,
                                double* d_n, double* d_ac, double* d_het, cudaStream_t stream);

// All-pairs accumulate: ONE BLOCK per pair r in the chunk [pair0, pair0+C). Maps r to the
// tile pop indices (i, j) via the O(1) readv2_unrank_pair, then the block's threads stride-share
// this tile's SNPs through the SHARED wc_finalize over the two pops' sufficient-stat slices,
// tree-reduce the block's partial Σnum/Σden/Σvalid in shared memory, and thread 0 ADDS the block
// total into the persistent per-pair accumulator (each BLOCK owns a distinct r — no atomics).
// Blocking over the SNP axis (rather than v1's one-thread-per-pair) keeps the launch occupied at
// low pair counts. Called per SNP-tile; the += reduces across tiles. `d_include` (or nullptr) is
// the GLOBAL summary mask, indexed by s_lo + s.
void launch_fst_allpairs_accumulate(const double* d_n, const double* d_ac, const double* d_het,
                                    int P, long tm, long s_lo, const std::uint8_t* d_include,
                                    long long pair0, long long C, double* d_pair_num,
                                    double* d_pair_den, long* d_pair_cnt, cudaStream_t stream);

}  // namespace steppe::device

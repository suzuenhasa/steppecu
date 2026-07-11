#pragma once
// src/device/cuda/king_streamed_kernel.cuh
//
// The one NEW kernel behind the biobank-scale STREAMED KING path (`steppe kinship --min-kinship`
// / `--king-cutoff`). The streamed driver reuses the dense decode + one-block-per-pair fold
// (king_allpairs_kernel) VERBATIM but, instead of D2H-ing 5*C(N,2) counts, it processes the
// pair space in BLOCKS and, per block, folds phi via the SHARED king_phi and raises an emit flag
// so cub::DeviceSelect::Flagged keeps only the above-threshold survivors. This kernel is that
// per-block phi + flag stage: it unranks each block-local pair to (i, j), computes phi from the
// block-local integer accumulators, and writes the compaction inputs. Because king_phi is the
// same STEPPE_HD inline the dense HOST finalize calls, the streamed phi is BIT-IDENTICAL to the
// dense path (the values gate).

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// king_streamed_flag: one thread per block-local pair t in [0, blockC). Unranks the global rank
// r = block_pair0 + t to (i < j), computes phi = king_phi(counts[t]) from the block-local
// accumulators, and sets out_flag[t] = (strict ? phi > threshold : phi >= threshold). A NaN phi
// (min-het == 0) fails BOTH comparisons and is dropped, matching the dense host finalize. The
// integer count arrays are the block's persistent accumulators (block-local index); out_i/out_j/
// out_phi are the parallel compaction inputs cub selects on with out_flag.
void launch_king_streamed_flag(const long* d_nsnp, const long* d_hethet, const long* d_ibs0,
                               const long* d_het_i, const long* d_het_j, int N,
                               long long block_pair0, long long blockC, double threshold,
                               bool strict, int* d_out_i, int* d_out_j, double* d_out_phi,
                               std::uint8_t* d_out_flag, cudaStream_t stream);

}  // namespace steppe::device

// src/device/cuda/king_streamed_kernel.cu
//
// The per-block phi + emit-flag kernel for the streamed KING path (see king_streamed_kernel.cuh).
// Reuses the SHARED king_phi (core/internal/king_kinship.hpp) so the emitted phi is bit-identical
// to the dense host finalize, and the O(1) closed-form pair unrank (sweep_unrank.cuh) so the
// (i, j) it recovers matches the dense all-pairs enumeration exactly.
#include "device/cuda/king_streamed_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/king_kinship.hpp"   // KingCounts, king_phi
#include "core/internal/launch_config.hpp"  // cdiv
#include "device/cuda/check.cuh"
#include "device/cuda/sweep_unrank.cuh"     // readv2_unrank_pair

namespace steppe::device {

using core::KingCounts;
using core::king_phi;

namespace {

constexpr int kKingStreamFlagBlock = 256;

__global__ void king_streamed_flag_kernel(const long* __restrict__ nsnp,
                                          const long* __restrict__ hethet,
                                          const long* __restrict__ ibs0,
                                          const long* __restrict__ het_i,
                                          const long* __restrict__ het_j, int N,
                                          long long block_pair0, long long blockC, double threshold,
                                          int strict, int* __restrict__ out_i,
                                          int* __restrict__ out_j, double* __restrict__ out_phi,
                                          std::uint8_t* __restrict__ out_flag) {
    const long long t =
        static_cast<long long>(blockIdx.x) * blockDim.x + static_cast<long long>(threadIdx.x);
    if (t >= blockC) return;
    const long long r = block_pair0 + t;
    int c[2];
    readv2_unrank_pair(r, N, c);  // c[0] = i < c[1] = j
    out_i[t] = c[0];
    out_j[t] = c[1];

    KingCounts kc;
    kc.nsnp = nsnp[t];
    kc.hethet = hethet[t];
    kc.ibs0 = ibs0[t];
    kc.het_i = het_i[t];
    kc.het_j = het_j[t];
    const double phi = king_phi(kc);  // SHARED -> bit-identical to the dense finalize
    out_phi[t] = phi;

    // NaN phi (min-het <= 0) fails BOTH comparisons and is dropped, matching finalize's phi>=cut.
    const bool keep = (strict != 0) ? (phi > threshold) : (phi >= threshold);
    out_flag[t] = keep ? std::uint8_t{1} : std::uint8_t{0};
}

}  // namespace

void launch_king_streamed_flag(const long* d_nsnp, const long* d_hethet, const long* d_ibs0,
                               const long* d_het_i, const long* d_het_j, int N,
                               long long block_pair0, long long blockC, double threshold,
                               bool strict, int* d_out_i, int* d_out_j, double* d_out_phi,
                               std::uint8_t* d_out_flag, cudaStream_t stream) {
    if (blockC <= 0) return;
    const unsigned block = static_cast<unsigned>(kKingStreamFlagBlock);
    const unsigned grid = static_cast<unsigned>(
        core::cdiv(static_cast<long>(blockC), static_cast<long>(kKingStreamFlagBlock)));
    king_streamed_flag_kernel<<<grid, block, 0, stream>>>(
        d_nsnp, d_hethet, d_ibs0, d_het_i, d_het_j, N, block_pair0, blockC, threshold,
        strict ? 1 : 0, d_out_i, d_out_j, d_out_phi, d_out_flag);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

// src/device/cuda/king_allpairs_kernel.cu
//
// GPU kernels for `steppe kinship` (KING-robust between-family kinship), plus their thin host
// launch wrappers. The design mirrors fst_allpairs_kernel.cu exactly: decode the per-individual
// diploid dosage ONCE per SNP-tile into a compact N x tm byte tensor, then fold every pair's
// five KING counts (nsnp, hetHet, IBS0, het_i, het_j) block-per-pair. Both stages reuse the
// SHARED primitives so the swept counts cannot drift from the host finalize / CPU oracle:
//   * king_classify (core/internal/king_kinship.hpp) — the per-SNP KING count fold.
//   * genotype_code / kCodesPerByte (core/internal/decode_af.hpp) — the 2-bit unpack.
//   * readv2_unrank_pair (sweep_unrank.cuh) — the O(1) closed-form k=2 pair unrank.
#include "device/cuda/king_allpairs_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"      // genotype_code, kCodesPerByte
#include "core/internal/king_kinship.hpp"   // KingCounts, king_classify
#include "core/internal/launch_config.hpp"  // kDecodeBlockX/Y, cdiv, kMaxGridX
#include "device/cuda/check.cuh"
#include "device/cuda/sweep_unrank.cuh"     // readv2_unrank_pair

namespace steppe::device {

using core::genotype_code;
using core::KingCounts;
using core::king_classify;
using core::kDecodeBlockX;
using core::kDecodeBlockY;

namespace {

// Dosage decode — one thread per (sample g, SNP s_local). Unpacks the 2-bit code into the
// compact N x tm byte tensor (individual-major: out[g*tm + s]). Missing stays as its code.
__global__ void king_dosage_decode_kernel(const std::uint8_t* __restrict__ packed,
                                          std::size_t bytes_per_record, int N, long s_lo, long tm,
                                          std::uint8_t* __restrict__ out_code) {
    const long g_l = static_cast<long>(blockIdx.y) * blockDim.y + threadIdx.y;
    const long s = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g_l >= N || s >= tm) return;
    const long gs = s_lo + s;
    const std::size_t byte_in_record =
        static_cast<std::size_t>(gs) / static_cast<std::size_t>(core::kCodesPerByte);
    const int pos_in_byte = static_cast<int>(gs % core::kCodesPerByte);
    const std::uint8_t byte =
        packed[static_cast<std::size_t>(g_l) * bytes_per_record + byte_in_record];
    out_code[static_cast<std::size_t>(g_l) * static_cast<std::size_t>(tm) +
             static_cast<std::size_t>(s)] = genotype_code(byte, pos_in_byte);
}

// Accumulate block width — must match the launch's block dim (the shared-mem reduction below
// is sized to it). 256 = kDecodeBlockX * kDecodeBlockY, a bandwidth-friendly power of two.
constexpr int kKingAccumBlock = 256;

// KING accumulate — ONE BLOCK per pair. The block's threads stride-share the tile's SNPs, each
// holding a private KingCounts partial, then a shared-memory tree reduction folds the block's
// partials and thread 0 ADDS the block total into the persistent per-pair counts. One block
// owns each distinct r across a launch, so the += needs no atomic and the integer reduction is
// order-independent -> fully deterministic. Blocking over the SNP axis (rather than one thread
// per pair) keeps the launch occupied at low pair counts (few samples) — the fst all-pairs fix.
__global__ void king_allpairs_accumulate_kernel(const std::uint8_t* __restrict__ code, int N,
                                                long tm, long s_lo,
                                                const std::uint8_t* __restrict__ include,
                                                const int* __restrict__ pairs_i,
                                                const int* __restrict__ pairs_j, long long pair0,
                                                long long C, long long out_offset,
                                                long* __restrict__ out_nsnp,
                                                long* __restrict__ out_hethet,
                                                long* __restrict__ out_ibs0,
                                                long* __restrict__ out_het_i,
                                                long* __restrict__ out_het_j) {
    const long long idx = static_cast<long long>(blockIdx.x);  // one block per pair in the chunk
    if (idx >= C) return;
    const long long r = pair0 + idx;
    // Output index. Dense all-pairs/--pairs pass out_offset==0 so the accumulators are the full
    // C(N,2)/list array indexed by the global rank r (unchanged). The STREAMED path sizes the
    // accumulators to one pair-BLOCK and passes out_offset = block_pair0, so o = r - block_pair0
    // is the block-local slot — the only change vs the dense fold, which stays bit-identical.
    const long long o = r - out_offset;

    int i, j;
    if (pairs_i != nullptr) {  // explicit-pair mode: r indexes the given list
        i = pairs_i[r];
        j = pairs_j[r];
    } else {  // all-pairs mode: unrank the flat C(N,2) rank
        int c[2];
        readv2_unrank_pair(r, N, c);  // c[0] = i < c[1] = j
        i = c[0];
        j = c[1];
    }
    const std::size_t ti = static_cast<std::size_t>(i) * static_cast<std::size_t>(tm);
    const std::size_t tj = static_cast<std::size_t>(j) * static_cast<std::size_t>(tm);

    // Each thread reduces a strided slice of the tile's SNPs into a private count partial.
    KingCounts acc;  // {0,0,0,0,0}
    for (long s = static_cast<long>(threadIdx.x); s < tm; s += static_cast<long>(blockDim.x)) {
        if (include != nullptr && include[s_lo + s] == 0) continue;
        king_classify(code[ti + static_cast<std::size_t>(s)],
                      code[tj + static_cast<std::size_t>(s)], acc);
    }

    // Block-wide tree reduction (blockDim.x == kKingAccumBlock, a power of two).
    __shared__ long s_nsnp[kKingAccumBlock];
    __shared__ long s_hh[kKingAccumBlock];
    __shared__ long s_ib[kKingAccumBlock];
    __shared__ long s_hi[kKingAccumBlock];
    __shared__ long s_hj[kKingAccumBlock];
    const unsigned t = threadIdx.x;
    s_nsnp[t] = acc.nsnp;
    s_hh[t] = acc.hethet;
    s_ib[t] = acc.ibs0;
    s_hi[t] = acc.het_i;
    s_hj[t] = acc.het_j;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) {
            s_nsnp[t] += s_nsnp[t + stride];
            s_hh[t] += s_hh[t + stride];
            s_ib[t] += s_ib[t + stride];
            s_hi[t] += s_hi[t + stride];
            s_hj[t] += s_hj[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        out_nsnp[o] += s_nsnp[0];  // block owns r across this launch -> no atomic
        out_hethet[o] += s_hh[0];
        out_ibs0[o] += s_ib[0];
        out_het_i[o] += s_hi[0];
        out_het_j[o] += s_hj[0];
    }
}

}  // namespace

void launch_king_dosage_decode(const std::uint8_t* d_packed, std::size_t bytes_per_record, int N,
                               long s_lo, long tm, std::uint8_t* d_code, cudaStream_t stream) {
    if (N <= 0 || tm <= 0) return;
    const dim3 block(static_cast<unsigned>(kDecodeBlockX), static_cast<unsigned>(kDecodeBlockY));
    const dim3 grid(static_cast<unsigned>(core::cdiv(tm, static_cast<long>(kDecodeBlockX))),
                    static_cast<unsigned>(core::cdiv(N, kDecodeBlockY)));
    king_dosage_decode_kernel<<<grid, block, 0, stream>>>(d_packed, bytes_per_record, N, s_lo, tm,
                                                          d_code);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_king_allpairs_accumulate(const std::uint8_t* d_code, int N, long tm, long s_lo,
                                     const std::uint8_t* d_include, const int* d_pairs_i,
                                     const int* d_pairs_j, long long pair0, long long C,
                                     long long out_offset, long* d_nsnp, long* d_hethet,
                                     long* d_ibs0, long* d_het_i, long* d_het_j,
                                     cudaStream_t stream) {
    if (C <= 0 || tm <= 0) return;
    // ONE BLOCK per pair (grid.x = C): the caller chunks C to <= the pair-chunk clamp so grid.x
    // stays under kMaxGridX. block must equal kKingAccumBlock so the shared-mem reduction covers
    // exactly the launched threads. out_offset shifts the persistent accumulator index (0 for the
    // dense full-array path; block_pair0 for the streamed pair-block path).
    STEPPE_ASSERT(static_cast<unsigned long long>(C) <= core::kMaxGridX,
                  "king all-pairs gridDim.x (one block per pair) exceeds kMaxGridX — chunk the pairs");
    const unsigned grid = static_cast<unsigned>(C);
    king_allpairs_accumulate_kernel<<<grid, static_cast<unsigned>(kKingAccumBlock), 0, stream>>>(
        d_code, N, tm, s_lo, d_include, d_pairs_i, d_pairs_j, pair0, C, out_offset, d_nsnp,
        d_hethet, d_ibs0, d_het_i, d_het_j);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

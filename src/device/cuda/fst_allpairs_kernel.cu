// src/device/cuda/fst_allpairs_kernel.cu
// Reference: docs/reference/src_device_cuda_fst_allpairs_kernel.cu.md
//
// GPU kernels for the all-pairs Weir & Cockerham 1984 FST matrix (`steppe fst --all-pairs`),
// plus their thin host launch wrappers. The design (docs/planning/fst-all-pairs-scope.md):
// decode the per-(pop, SNP) sufficient statistic {n, ac, het} ONCE per SNP-tile, then fold
// every C(P,2) pair's wc_finalize into the genome-wide per-pair Σ. Both stages reuse the
// SHARED primitives so the matrix cannot drift from the single-pair path:
//   * wc_accumulate / wc_finalize (core/internal/wc_fst.hpp) — the WC math, native FP64.
//   * genotype_code / kCodesPerByte (core/internal/decode_af.hpp) — the 2-bit unpack.
//   * readv2_unrank_pair (sweep_unrank.cuh) — the O(1) closed-form k=2 pair unrank.
#include "device/cuda/fst_allpairs_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"      // genotype_code, kCodesPerByte
#include "core/internal/launch_config.hpp"  // kDecodeBlockX/Y, cdiv
#include "core/internal/wc_fst.hpp"         // WcPerPop, WcSite, wc_accumulate, wc_finalize
#include "device/cuda/check.cuh"
#include "device/cuda/sweep_unrank.cuh"     // readv2_unrank_pair

namespace steppe::device {

using core::genotype_code;
using core::kDecodeBlockX;
using core::kDecodeBlockY;
using core::wc_accumulate;
using core::wc_finalize;
using core::WcPerPop;
using core::WcSite;

namespace {

// Sufficient-stat decode — reference §2. One thread per (pop p, SNP s_local).
__global__ void fst_suffstat_decode_kernel(const std::uint8_t* __restrict__ packed,
                                           std::size_t bytes_per_record,
                                           const std::size_t* __restrict__ pop_offsets, int P,
                                           long s_lo, long tm, double* __restrict__ out_n,
                                           double* __restrict__ out_ac,
                                           double* __restrict__ out_het) {
    const long p_l = static_cast<long>(blockIdx.y) * blockDim.y + threadIdx.y;
    const long s = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p_l >= P || s >= tm) return;
    const int p = static_cast<int>(p_l);

    const long gs = s_lo + s;
    const std::size_t byte_in_record =
        static_cast<std::size_t>(gs) / static_cast<std::size_t>(core::kCodesPerByte);
    const int pos_in_byte = static_cast<int>(gs % core::kCodesPerByte);

    const std::size_t g_begin = pop_offsets[static_cast<std::size_t>(p)];
    const std::size_t g_end = pop_offsets[static_cast<std::size_t>(p) + 1];

    WcPerPop acc;  // {0, 0, 0}
    for (std::size_t g = g_begin; g < g_end; ++g) {
        const std::uint8_t byte = packed[g * bytes_per_record + byte_in_record];
        wc_accumulate(genotype_code(byte, pos_in_byte), acc);
    }
    const std::size_t off = static_cast<std::size_t>(p) * static_cast<std::size_t>(tm) +
                            static_cast<std::size_t>(s);
    out_n[off] = acc.n;
    out_ac[off] = acc.ac;
    out_het[off] = acc.het;
}

// Accumulate block width — must match the launch's block dim (the shared-mem reduction below
// is sized to it). 256 = kDecodeBlockX * kDecodeBlockY, a memory-bandwidth-friendly power of two.
constexpr int kFstAccumBlock = 256;

// All-pairs accumulate — reference §3. ONE BLOCK per pair (was one THREAD per pair): the block's
// threads stride-share the tile's SNPs, each holds a private partial (num/den/cnt), then a
// shared-memory tree reduction folds the block's partials and thread 0 ADDS the block total into
// the persistent per-pair Σ. One block owns each distinct r, so the += needs no atomic and the
// reduction order is deterministic (bit-stable vs the single-pair path within FP64 round-off).
// v1 mapped one THREAD to each pair, so the accumulate launched only C threads and was
// occupancy-starved at low pair counts (few pops) — each of the ~C threads then serially looped
// all M SNPs, which set steppe's ~flat low-K wall. Blocking over the SNP axis restores occupancy
// (C blocks x kFstAccumBlock threads) and shortens each thread's loop to ~M/blockDim.
__global__ void fst_allpairs_accumulate_kernel(const double* __restrict__ dN,
                                               const double* __restrict__ dAc,
                                               const double* __restrict__ dHet, int P, long tm,
                                               long s_lo, const std::uint8_t* __restrict__ include,
                                               long long pair0, long long C,
                                               double* __restrict__ pair_num,
                                               double* __restrict__ pair_den,
                                               long* __restrict__ pair_cnt) {
    const long long idx = static_cast<long long>(blockIdx.x);  // one block per pair
    if (idx >= C) return;
    const long long r = pair0 + idx;

    int c[2];
    readv2_unrank_pair(r, P, c);  // c[0] = i < c[1] = j
    const long ti = static_cast<long>(c[0]) * tm;
    const long tj = static_cast<long>(c[1]) * tm;

    // Each thread reduces a strided slice of the tile's SNPs into a private partial.
    double an = 0.0, ad = 0.0;
    long cn = 0;
    for (long s = static_cast<long>(threadIdx.x); s < tm; s += static_cast<long>(blockDim.x)) {
        if (include != nullptr && include[s_lo + s] == 0) continue;
        const std::size_t oi = static_cast<std::size_t>(ti + s);
        const std::size_t oj = static_cast<std::size_t>(tj + s);
        const WcPerPop A{dN[oi], dAc[oi], dHet[oi]};
        const WcPerPop B{dN[oj], dAc[oj], dHet[oj]};
        const WcSite w = wc_finalize(A, B);
        if (w.valid) {
            an += w.num;
            ad += w.den;
            ++cn;
        }
    }

    // Block-wide tree reduction (blockDim.x == kFstAccumBlock, a power of two).
    __shared__ double s_num[kFstAccumBlock];
    __shared__ double s_den[kFstAccumBlock];
    __shared__ long s_cnt[kFstAccumBlock];
    const unsigned t = threadIdx.x;
    s_num[t] = an;
    s_den[t] = ad;
    s_cnt[t] = cn;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) {
            s_num[t] += s_num[t + stride];
            s_den[t] += s_den[t + stride];
            s_cnt[t] += s_cnt[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        pair_num[r] += s_num[0];  // block owns r across this launch -> no atomic
        pair_den[r] += s_den[0];
        pair_cnt[r] += s_cnt[0];
    }
}

}  // namespace

void launch_fst_suffstat_decode(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                                const std::size_t* d_pop_offsets, int P, long s_lo, long tm,
                                double* d_n, double* d_ac, double* d_het, cudaStream_t stream) {
    if (P <= 0 || tm <= 0) return;
    const dim3 block(static_cast<unsigned>(kDecodeBlockX), static_cast<unsigned>(kDecodeBlockY));
    const dim3 grid(static_cast<unsigned>(core::cdiv(tm, static_cast<long>(kDecodeBlockX))),
                    static_cast<unsigned>(core::cdiv(P, kDecodeBlockY)));
    fst_suffstat_decode_kernel<<<grid, block, 0, stream>>>(
        d_packed, bytes_per_record, d_pop_offsets, P, s_lo, tm, d_n, d_ac, d_het);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_fst_allpairs_accumulate(const double* d_n, const double* d_ac, const double* d_het,
                                    int P, long tm, long s_lo, const std::uint8_t* d_include,
                                    long long pair0, long long C, double* d_pair_num,
                                    double* d_pair_den, long* d_pair_cnt, cudaStream_t stream) {
    if (C <= 0 || tm <= 0) return;
    // ONE BLOCK per pair (grid.x = C): the block's kFstAccumBlock threads stride-share the tile's
    // SNPs. The caller chunks C to <= kFstPairChunkClamp (2^30), so grid.x = C stays under kMaxGridX
    // (2^31-1) with headroom — no grid-stride loop needed. block must equal kFstAccumBlock so the
    // kernel's shared-mem reduction (sized to kFstAccumBlock) covers exactly the launched threads.
    STEPPE_ASSERT(static_cast<unsigned long long>(C) <= core::kMaxGridX,
                  "fst all-pairs gridDim.x (one block per pair) exceeds kMaxGridX — chunk the pairs");
    const unsigned grid = static_cast<unsigned>(C);
    fst_allpairs_accumulate_kernel<<<grid, static_cast<unsigned>(kFstAccumBlock), 0, stream>>>(
        d_n, d_ac, d_het, P, tm, s_lo, d_include, pair0, C, d_pair_num, d_pair_den, d_pair_cnt);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

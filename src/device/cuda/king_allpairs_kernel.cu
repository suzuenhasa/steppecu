// src/device/cuda/king_allpairs_kernel.cu
//
// GPU kernels for `steppe kinship` (KING-robust between-family kinship), plus their thin host
// launch wrappers. The five KING counts (nsnp, hetHet, IBS0, het_i, het_j) are folded by BITPLANE
// POPCOUNT (see king_allpairs_kernel.cuh): the packed diploid tile is packed ONCE per SNP-tile into
// three per-sample class bitplanes (HOMREF, HET, HOMALT) over the SNP axis, and each pair's counts
// become AND + __popc over the words. This reproduces the shared king_classify fold EXACTLY
// (core/internal/king_kinship.hpp — the CPU oracle the counts cannot drift from), 32 genotypes per
// instruction instead of one, and reads each sample O(N/TB) times (tiled) rather than O(N).
//
// Exactness vs king_classify (per SNP, codes ci,cj in {0=homref,1=het,2=homalt,3=missing}):
//   * A SNP is "considered" iff BOTH non-missing. HOMREF/HET/HOMALT bits are set only for a
//     non-missing, unmasked call, so NM = HOMREF|HET|HOMALT has a bit iff considered on that side,
//     and NM_i & NM_j is the considered mask -> nsnp = popc(NM_i & NM_j)         (== acc.nsnp+=1).
//   * het_i += 1 iff considered and ci==1: HET_i implies NM_i, so the condition is HET_i & NM_j
//     -> het_i = popc(HET_i & NM_j)  (symmetrically het_j = popc(HET_j & NM_i)).
//   * hetHet += 1 iff ci==1 & cj==1 -> popc(HET_i & HET_j) (het implies both non-missing).
//   * IBS0 is king_classify's ELSE-IF branch: both non-missing, both homozygous, ci!=cj — i.e.
//     opposite homozygotes, which is DISJOINT from hetHet, so the independent
//     popc((HOMREF_i & HOMALT_j) | (HOMALT_i & HOMREF_j)) equals the branch count.
// All five counts are integer and order-independent -> the reductions are bit-identical. An
// autosome/include mask and a missing call both zero every class bit for that SNP, so it drops out
// of every popcount — the same effect as king_classify skipping it plus the fold's include `continue`.
#include "device/cuda/king_allpairs_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"      // genotype_code, kCodesPerByte, kHet/kMissing codes
#include "core/internal/launch_config.hpp"  // kDecodeBlockX/Y, cdiv, grid_for_x, kMaxGridY
#include "device/cuda/check.cuh"
#include "device/cuda/sweep_unrank.cuh"     // readv2_unrank_pair (r -> i<j) and C(J,2)+I forward rank

namespace steppe::device {

using core::genotype_code;

namespace {

// ---- Bitplane pack -----------------------------------------------------------------------------

// One thread per (sample g, word w). Builds HOMREF/HET/HOMALT words over the 32 SNPs the word spans.
// Missing (code==3) and masked (include==0) SNPs leave all three bits 0 -> the derived NM bit is 0.
__global__ void king_bitplane_build_kernel(const std::uint8_t* __restrict__ packed,
                                           std::size_t bytes_per_record, int N, long s_lo, long tm,
                                           long W, const std::uint8_t* __restrict__ include,
                                           std::uint32_t* __restrict__ homref,
                                           std::uint32_t* __restrict__ het,
                                           std::uint32_t* __restrict__ homalt) {
    const int g = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    const long w = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= N || w >= W) return;

    std::uint32_t v_hr = 0, v_ht = 0, v_ha = 0;
    const long base_s = w * 32;
    for (int b = 0; b < 32; ++b) {
        const long s = base_s + b;
        if (s >= tm) break;  // last (partial) word: SNPs >= tm leave their bits 0
        const long gs = s_lo + s;
        if (include != nullptr && include[gs] == 0) continue;  // masked -> all class bits 0
        const std::size_t byte_in_record =
            static_cast<std::size_t>(gs) / static_cast<std::size_t>(core::kCodesPerByte);
        const int pos_in_byte = static_cast<int>(gs % core::kCodesPerByte);
        const std::uint8_t byte =
            packed[static_cast<std::size_t>(g) * bytes_per_record + byte_in_record];
        const std::uint8_t code = genotype_code(byte, pos_in_byte);
        const std::uint32_t bit = 1u << b;
        // codes: 0=homref, 1=het (kHeterozygousGenotypeCode), 2=homalt, 3=missing (kMissing -> none)
        if (code == 0)
            v_hr |= bit;
        else if (code == core::kHeterozygousGenotypeCode)
            v_ht |= bit;
        else if (code == 2)
            v_ha |= bit;
    }
    const long off = static_cast<long>(g) * W + w;
    homref[off] = v_hr;
    het[off] = v_ht;
    homalt[off] = v_ha;
}

// ---- The per-word KING fold (shared by the tiled and per-pair kernels) -------------------------

struct KingWordAcc {
    long long nsnp = 0, hethet = 0, ibs0 = 0, het_i = 0, het_j = 0;
};

__device__ __forceinline__ void king_fold_word(std::uint32_t hr_i, std::uint32_t ht_i,
                                               std::uint32_t ha_i, std::uint32_t hr_j,
                                               std::uint32_t ht_j, std::uint32_t ha_j,
                                               KingWordAcc& a) {
    const std::uint32_t nm_i = hr_i | ht_i | ha_i;  // non-missing (and unmasked) on side i
    const std::uint32_t nm_j = hr_j | ht_j | ha_j;
    a.nsnp += __popc(nm_i & nm_j);
    a.hethet += __popc(ht_i & ht_j);
    a.het_i += __popc(ht_i & nm_j);
    a.het_j += __popc(ht_j & nm_i);
    a.ibs0 += __popc((hr_i & ha_j) | (ha_i & hr_j));
}

// ---- Dense all-pairs: TB x TB sample-tile GEMM with shared-memory reuse -------------------------

constexpr int kKingTile = 32;        // TB: samples per tile dim (TB*TB == 1024 threads/block)
constexpr int kKingChunkWords = 32;  // words staged into shared per chunk (32*33*4*6 = 25 KiB static)

__global__ void king_tiled_accumulate_kernel(const std::uint32_t* __restrict__ homref,
                                             const std::uint32_t* __restrict__ het,
                                             const std::uint32_t* __restrict__ homalt, int N, long W,
                                             int n_tiles, long* __restrict__ out_nsnp,
                                             long* __restrict__ out_hethet,
                                             long* __restrict__ out_ibs0,
                                             long* __restrict__ out_het_i,
                                             long* __restrict__ out_het_j) {
    // Decode this block's upper-triangular tile-pair (tileA <= tileB) from its flat id (mirrors
    // readv2_mismatch_tiled_kernel): rows of decreasing length n_tiles, n_tiles-1, ... .
    int tileA = 0, rem = static_cast<int>(blockIdx.x), row = n_tiles;
    while (rem >= row) {
        rem -= row;
        ++tileA;
        --row;
    }
    const int tileB = tileA + rem;
    const int A_base = tileA * kKingTile;
    const int B_base = tileB * kKingTile;

    const int a = static_cast<int>(threadIdx.y);  // local row in tile A -> I (smaller index)
    const int b = static_cast<int>(threadIdx.x);  // local row in tile B -> J (larger index)
    const int I = A_base + a;
    const int J = B_base + b;
    // I<J covers both the off-diagonal tiles (I in A < J in B always) and the diagonal (a<b);
    // it also drops the self-pair and, with I<N/J<N, the partial-tile out-of-range rows.
    const bool active = (I < N) && (J < N) && (I < J);

    // Padded word dim (+1) makes the per-warp shared reads bank-conflict-free: within a warp a is
    // uniform (broadcast on tile A) and b = 0..31 varies, and (b*(TB+1)) mod 32 == b is a permutation.
    __shared__ std::uint32_t sAhr[kKingTile][kKingChunkWords + 1];
    __shared__ std::uint32_t sAht[kKingTile][kKingChunkWords + 1];
    __shared__ std::uint32_t sAha[kKingTile][kKingChunkWords + 1];
    __shared__ std::uint32_t sBhr[kKingTile][kKingChunkWords + 1];
    __shared__ std::uint32_t sBht[kKingTile][kKingChunkWords + 1];
    __shared__ std::uint32_t sBha[kKingTile][kKingChunkWords + 1];

    const int tid = static_cast<int>(threadIdx.y) * kKingTile + static_cast<int>(threadIdx.x);
    constexpr int kNThreads = kKingTile * kKingTile;

    KingWordAcc acc;
    for (long w0 = 0; w0 < W; w0 += kKingChunkWords) {
        const long rem_w = W - w0;
        const int cw = (rem_w < kKingChunkWords) ? static_cast<int>(rem_w) : kKingChunkWords;

        // Cooperatively stage cw words x TB samples of both tiles into shared (coalesced: consecutive
        // threads load consecutive words of the same sample row).
        for (int e = tid; e < kKingTile * cw; e += kNThreads) {
            const int lr = e / cw;  // local sample row 0..TB-1
            const int lw = e % cw;  // word within this chunk
            const long gw = w0 + lw;
            const int gA = A_base + lr;
            const int gB = B_base + lr;
            if (gA < N) {
                const long off = static_cast<long>(gA) * W + gw;
                sAhr[lr][lw] = homref[off];
                sAht[lr][lw] = het[off];
                sAha[lr][lw] = homalt[off];
            } else {
                sAhr[lr][lw] = 0;
                sAht[lr][lw] = 0;
                sAha[lr][lw] = 0;
            }
            if (gB < N) {
                const long off = static_cast<long>(gB) * W + gw;
                sBhr[lr][lw] = homref[off];
                sBht[lr][lw] = het[off];
                sBha[lr][lw] = homalt[off];
            } else {
                sBhr[lr][lw] = 0;
                sBht[lr][lw] = 0;
                sBha[lr][lw] = 0;
            }
        }
        __syncthreads();

        if (active) {
            for (int wl = 0; wl < cw; ++wl) {
                king_fold_word(sAhr[a][wl], sAht[a][wl], sAha[a][wl], sBhr[b][wl], sBht[b][wl],
                               sBha[b][wl], acc);
            }
        }
        __syncthreads();  // reuse shared for the next chunk
    }

    if (active) {
        // Forward upper-triangular rank of (I<J) == readv2_unrank_pair's inverse (host pair_ij).
        // One thread owns this rank across the launch -> plain += (no atomic); reduces across tiles.
        const long long rank = static_cast<long long>(J) * (J - 1) / 2 + I;
        out_nsnp[rank] += static_cast<long>(acc.nsnp);
        out_hethet[rank] += static_cast<long>(acc.hethet);
        out_ibs0[rank] += static_cast<long>(acc.ibs0);
        out_het_i[rank] += static_cast<long>(acc.het_i);
        out_het_j[rank] += static_cast<long>(acc.het_j);
    }
}

// ---- Explicit-pairs / streamed contiguous rank range: one warp per pair ------------------------

constexpr int kKingPerPairWarps = 8;  // 8 warps == 256 threads/block

__global__ void king_perpair_accumulate_kernel(
    const std::uint32_t* __restrict__ homref, const std::uint32_t* __restrict__ het,
    const std::uint32_t* __restrict__ homalt, int N, long W, const int* __restrict__ pairs_i,
    const int* __restrict__ pairs_j, long long pair0, long long C, long long out_offset,
    long* __restrict__ out_nsnp, long* __restrict__ out_hethet, long* __restrict__ out_ibs0,
    long* __restrict__ out_het_i, long* __restrict__ out_het_j) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const long long idx = static_cast<long long>(blockIdx.x) * kKingPerPairWarps + warp;
    if (idx >= C) return;  // warp-uniform: the whole warp stays active for the shuffle below
    const long long r = pair0 + idx;
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
    const long bi = static_cast<long>(i) * W;
    const long bj = static_cast<long>(j) * W;

    KingWordAcc acc;
    for (long w = lane; w < W; w += 32) {  // coalesced: 32 lanes read 32 consecutive words
        king_fold_word(homref[bi + w], het[bi + w], homalt[bi + w], homref[bj + w], het[bj + w],
                       homalt[bj + w], acc);
    }
    // Warp reduction (the whole warp is active; mask 0xffffffff is valid).
    for (int off = 16; off > 0; off >>= 1) {
        acc.nsnp += __shfl_down_sync(0xffffffffu, acc.nsnp, off);
        acc.hethet += __shfl_down_sync(0xffffffffu, acc.hethet, off);
        acc.ibs0 += __shfl_down_sync(0xffffffffu, acc.ibs0, off);
        acc.het_i += __shfl_down_sync(0xffffffffu, acc.het_i, off);
        acc.het_j += __shfl_down_sync(0xffffffffu, acc.het_j, off);
    }
    if (lane == 0) {  // one warp owns o across the launch -> plain += (no atomic); reduces across tiles
        out_nsnp[o] += static_cast<long>(acc.nsnp);
        out_hethet[o] += static_cast<long>(acc.hethet);
        out_ibs0[o] += static_cast<long>(acc.ibs0);
        out_het_i[o] += static_cast<long>(acc.het_i);
        out_het_j[o] += static_cast<long>(acc.het_j);
    }
}

}  // namespace

void launch_king_bitplane_build(const std::uint8_t* d_packed, std::size_t bytes_per_record, int N,
                                long s_lo, long tm, long W, const std::uint8_t* d_include,
                                std::uint32_t* d_homref, std::uint32_t* d_het,
                                std::uint32_t* d_homalt, cudaStream_t stream) {
    if (N <= 0 || W <= 0 || tm <= 0) return;
    const dim3 block(static_cast<unsigned>(core::kDecodeBlockX),
                     static_cast<unsigned>(core::kDecodeBlockY));
    const dim3 grid(static_cast<unsigned>(core::cdiv(W, static_cast<long>(core::kDecodeBlockX))),
                    static_cast<unsigned>(core::cdiv(N, core::kDecodeBlockY)));
    king_bitplane_build_kernel<<<grid, block, 0, stream>>>(d_packed, bytes_per_record, N, s_lo, tm,
                                                           W, d_include, d_homref, d_het, d_homalt);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_king_tiled_accumulate(const std::uint32_t* d_homref, const std::uint32_t* d_het,
                                  const std::uint32_t* d_homalt, int N, long W, long* d_nsnp,
                                  long* d_hethet, long* d_ibs0, long* d_het_i, long* d_het_j,
                                  cudaStream_t stream) {
    if (N <= 1 || W <= 0) return;
    const int n_tiles = core::cdiv(N, kKingTile);
    const long n_tile_pairs = static_cast<long>(n_tiles) * (n_tiles + 1) / 2;
    const dim3 block(static_cast<unsigned>(kKingTile), static_cast<unsigned>(kKingTile));  // 32x32
    const int grid = core::grid_for_x(
        n_tile_pairs, 1,
        "king tiled gridDim.x (upper-triangular tile-pair axis) exceeds kMaxGridX");
    king_tiled_accumulate_kernel<<<static_cast<unsigned>(grid), block, 0, stream>>>(
        d_homref, d_het, d_homalt, N, W, n_tiles, d_nsnp, d_hethet, d_ibs0, d_het_i, d_het_j);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_king_perpair_accumulate(const std::uint32_t* d_homref, const std::uint32_t* d_het,
                                    const std::uint32_t* d_homalt, int N, long W,
                                    const int* d_pairs_i, const int* d_pairs_j, long long pair0,
                                    long long C, long long out_offset, long* d_nsnp, long* d_hethet,
                                    long* d_ibs0, long* d_het_i, long* d_het_j,
                                    cudaStream_t stream) {
    if (C <= 0 || W <= 0) return;
    const long long blocks = core::cdiv(static_cast<long>(C), static_cast<long>(kKingPerPairWarps));
    STEPPE_ASSERT(static_cast<unsigned long long>(blocks) <= core::kMaxGridX,
                  "king per-pair gridDim.x exceeds kMaxGridX — chunk the pairs");
    king_perpair_accumulate_kernel<<<static_cast<unsigned>(blocks),
                                     static_cast<unsigned>(kKingPerPairWarps * 32), 0, stream>>>(
        d_homref, d_het, d_homalt, N, W, d_pairs_i, d_pairs_j, pair0, C, out_offset, d_nsnp,
        d_hethet, d_ibs0, d_het_i, d_het_j);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

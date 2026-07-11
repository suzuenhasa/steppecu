#pragma once
// src/device/cuda/king_allpairs_kernel.cuh
//
// Declarations for the three kernels behind `steppe kinship` (KING-robust). The counts are
// folded by BITPLANE POPCOUNT: the packed diploid tile is packed ONCE (per SNP-tile) into three
// per-sample bitplanes over the SNP axis — HOMREF, HET, HOMALT (one bit per SNP per class, in
// uint32 words) — with the non-missing plane NM = HOMREF|HET|HOMALT derived in-register. The five
// KING counts then reduce to AND + __popc over the words, reproducing king_classify
// (core/internal/king_kinship.hpp) EXACTLY, 32 genotypes per instruction instead of one:
//     nsnp   = popc(NM_i & NM_j)                                  (both non-missing == "considered")
//     hetHet = popc(HET_i & HET_j)
//     het_i  = popc(HET_i & NM_j),  het_j = popc(HET_j & NM_i)    (het over the considered set)
//     IBS0   = popc((HOMREF_i & HOMALT_j) | (HOMALT_i & HOMREF_j))(opposite homozygotes)
// The dense all-pairs fold is a GEMM-style TB x TB sample-tile that cooperatively stages the tile's
// bitplane words into shared memory, so each sample's words are read O(N/TB) times, not O(N) — this
// replaces the memory-bound one-block-per-pair global re-read. The explicit-`--pairs` list and the
// streamed (`--min-kinship`) contiguous rank range fold warp-per-pair (coalesced bitplane loads).
// The kernel bodies live in king_allpairs_kernel.cu; the include-mask, singleton-pop and stable
// emission invariants are unchanged. Mirrors readv2_mismatch_kernel.cuh (the same bitplane idiom).

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// Bitplane pack: for the SNP-tile [s_lo, s_lo+tm) build, per sample g, the three class bitplanes
// (HOMREF, HET, HOMALT) over the tm SNPs, packed into W = ceil(tm/32) uint32 words each, sample-
// major (plane[g*W + w], bit b of word w == local SNP w*32+b). A masked SNP (d_include[gs]==0) and
// a missing call (code == kMissingGenotypeCode) leave ALL class bits 0, so NM = OR of the three is
// 0 there and the SNP contributes zero to every popcount — the exact effect of king_classify's
// "skip the site" plus the fold's `if (include[...]==0) continue`. Decoding once amortizes the
// 2-bit unpack across the N-1 pairs each sample joins.
void launch_king_bitplane_build(const std::uint8_t* d_packed, std::size_t bytes_per_record, int N,
                                long s_lo, long tm, long W, const std::uint8_t* d_include,
                                std::uint32_t* d_homref, std::uint32_t* d_het,
                                std::uint32_t* d_homalt, cudaStream_t stream);

// Dense all-pairs accumulate: a TB x TB sample-tile GEMM over the current SNP-tile's W words. Each
// CTA owns one upper-triangular tile-pair (tileA <= tileB); each thread owns one pair (I<J), stages
// the tile's bitplane words through shared memory in word-chunks, folds the five counts by AND +
// __popc, and ADDS its block-total into the persistent per-pair accumulators at rank C(J,2)+I
// (each rank has exactly one writer per launch -> no atomics; the += reduces across SNP-tiles).
void launch_king_tiled_accumulate(const std::uint32_t* d_homref, const std::uint32_t* d_het,
                                  const std::uint32_t* d_homalt, int N, long W, long* d_nsnp,
                                  long* d_hethet, long* d_ibs0, long* d_het_i, long* d_het_j,
                                  cudaStream_t stream);

// Per-pair accumulate: ONE WARP per pair. When d_pairs_i / d_pairs_j are non-null the warp maps its
// chunk index to that explicit (i, j); otherwise it maps the flat rank r via the O(1)
// readv2_unrank_pair. The 32 lanes stride the current SNP-tile's W words (coalesced bitplane loads),
// each folding the five counts by AND + __popc, then a warp-shuffle reduction has lane 0 ADD the
// warp-total into the persistent accumulators at o = r - out_offset (out_offset==0 for the dense
// explicit list; block_pair0 for the streamed pair-block). The += reduces across SNP-tiles.
void launch_king_perpair_accumulate(const std::uint32_t* d_homref, const std::uint32_t* d_het,
                                    const std::uint32_t* d_homalt, int N, long W,
                                    const int* d_pairs_i, const int* d_pairs_j, long long pair0,
                                    long long C, long long out_offset, long* d_nsnp, long* d_hethet,
                                    long* d_ibs0, long* d_het_i, long* d_het_j, cudaStream_t stream);

}  // namespace steppe::device

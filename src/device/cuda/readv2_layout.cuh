// src/device/cuda/readv2_layout.cuh
//
// The packed [sample x SNP-window] bit-matrix cell + window geometry shared by the
// READv2 pack kernel and mismatch kernel. Private to steppe_device (a .cuh, never on
// the CUDA-free seam).
//
// Cell: one 128-bit AoS word per (sample, 64-SNP block) — `allele` and `valid` bit
// planes packed together so the mismatch inner loop fetches both in one coalesced
// ulonglong2 transaction. Bits are LSB-first: SNP local index l in [0,64) -> bit l.
//   allele bit = the single pseudo-haploid allele (0 for ref-copy code 0, 1 for
//                ref-copy code 2). Meaningful only where valid == 1.
//   valid  bit = 1 iff the site is a genuine 0/2 hardcall for this sample; a missing
//                (code 3), a het (code 1 — dropped, no encoding in a 1-bit layout,
//                scope T10), or a padding SNP beyond the window/genome all set 0.
//
// Windows are non-overlapping SNP-count windows and are WORD-ALIGNED: window g owns
// words [g*wpw, g*wpw+wpw). The last word of a window carries window_snps % 64 real
// SNPs; its unused high bits stay valid=0. That is the single biggest correctness
// simplifier — the mismatch kernel never splits a popcount across a window edge, and
// the both-valid AND doubles as the padding mask (scope T2/T5).
#ifndef STEPPE_DEVICE_CUDA_READV2_LAYOUT_CUH
#define STEPPE_DEVICE_CUDA_READV2_LAYOUT_CUH

#include <cstdint>

namespace steppe::device {

// SNPs packed per 64-bit plane word.
inline constexpr int kReadv2SnpsPerWord = 64;

// One (sample, word) cell — 16 B, loaded as one 128-bit transaction (ulonglong2).
struct Readv2Word {
    std::uint64_t allele;
    std::uint64_t valid;
};

// Words per window = ceil(window_snps / 64). Windows start on a word boundary.
[[nodiscard]] __host__ __device__ inline int readv2_wpw(int window_snps) {
    return (window_snps + kReadv2SnpsPerWord - 1) / kReadv2SnpsPerWord;
}

// Windows tiling the SNP axis = ceil(M0 / window_snps) (partial last window kept).
[[nodiscard]] __host__ __device__ inline long readv2_n_win(long m0, int window_snps) {
    return (m0 + window_snps - 1) / window_snps;
}

// Total words per sample = n_win * wpw.
[[nodiscard]] __host__ __device__ inline long readv2_words_per_sample(long m0, int window_snps) {
    return readv2_n_win(m0, window_snps) *
           static_cast<long>(readv2_wpw(window_snps));
}

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_READV2_LAYOUT_CUH

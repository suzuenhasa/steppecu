// src/device/cuda/readv2_mismatch_kernel.cu
//
// The READv2 all-pairs windowed-mismatch reduction. The per-word primitive is the
// whole point (scope §2.1):
//     comp = Wi.valid & Wj.valid;                 // sites valid in BOTH samples
//     mism = (Wi.allele ^ Wj.allele) & comp;      // differing alleles among those
//     comparable += __popcll(comp);  mismatches += __popcll(mism);
// __popcll (64-bit), never __popc (scope T4); popcount the MASKED word (scope T5),
// and because window padding carries valid=0 the both-valid AND doubles as the
// padding mask. P0 is the MEAN over windows of per-window mismatch/comparable, so the
// reduction flushes at each word-aligned window boundary (scope T6), never a single
// genome-wide ratio.
#include "device/cuda/readv2_mismatch_kernel.cuh"

#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/launch_config.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/readv2_layout.cuh"
#include "device/cuda/sweep_unrank.cuh"

namespace steppe::device {

namespace {

// Per-window primitive over one aligned word pair — the AND/XOR/__popcll core.
__device__ __forceinline__ void word_counts(const Readv2Word& wi, const Readv2Word& wj,
                                             int& comp, int& mism) {
    const std::uint64_t c = wi.valid & wj.valid;
    const std::uint64_t m = (wi.allele ^ wj.allele) & c;
    comp += __popcll(c);
    mism += __popcll(m);
}

// Reduce one pair given its two row bases, folding per-window ratios into the four
// scalars. Row bases point at the sample's first Readv2Word in the bit-matrix.
__device__ __forceinline__ void reduce_pair(const Readv2Word* __restrict__ ri,
                                            const Readv2Word* __restrict__ rj,
                                            int wpw, long n_win,
                                            double& sum_p0, double& sum_p0_sq,
                                            int& n_used, long long& tot_comp) {
    sum_p0 = 0.0;
    sum_p0_sq = 0.0;
    n_used = 0;
    tot_comp = 0;
    for (long g = 0; g < n_win; ++g) {
        const long base = g * static_cast<long>(wpw);
        int win_comp = 0;
        int win_mism = 0;
        for (int wl = 0; wl < wpw; ++wl) {
            word_counts(ri[base + wl], rj[base + wl], win_comp, win_mism);
        }
        if (win_comp > 0) {
            const double p0 = static_cast<double>(win_mism) / static_cast<double>(win_comp);
            sum_p0 += p0;
            sum_p0_sq += p0 * p0;
            ++n_used;
            tot_comp += win_comp;
        }
    }
}

// (a) Baseline: one thread per pair, O(1) closed-form unrank (scope §2.3a + critic fix).
__global__ void readv2_mismatch_direct_kernel(const Readv2Word* __restrict__ d_words,
                                              long words_per_sample, int wpw, long n_win,
                                              int n_samples, long long n_pairs,
                                              double* __restrict__ d_sum_p0,
                                              double* __restrict__ d_sum_p0_sq,
                                              int* __restrict__ d_n_win_used,
                                              long long* __restrict__ d_tot_comp) {
    // Cast BEFORE the multiply so the flat rank never overflows 32-bit (critic fix).
    const long long r =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= n_pairs) return;
    int c[2];
    readv2_unrank_pair(r, n_samples, c);  // c[0] = i < c[1] = j
    const Readv2Word* ri = d_words + static_cast<long>(c[0]) * words_per_sample;
    const Readv2Word* rj = d_words + static_cast<long>(c[1]) * words_per_sample;

    double sum_p0, sum_p0_sq;
    int n_used;
    long long tot_comp;
    reduce_pair(ri, rj, wpw, n_win, sum_p0, sum_p0_sq, n_used, tot_comp);
    d_sum_p0[r] = sum_p0;
    d_sum_p0_sq[r] = sum_p0_sq;
    d_n_win_used[r] = n_used;
    d_tot_comp[r] = tot_comp;
}

// (b) Tiled: a block owns a TS x TS tile-pair (A,B) of samples; TS<=32 so TS*TS<=1024
// threads fit one block (critic fix). Each thread owns one local pair (a,b) and loops
// windows, reading the window's rows from shared memory that the block cooperatively
// loads once per window — the dstat-style row reuse. Upper-triangular rank gives the
// global output slot; diagonal tiles skip a>=b (scope §2.3b).
constexpr int kTileSamples = 32;  // TS

__global__ void readv2_mismatch_tiled_kernel(const Readv2Word* __restrict__ d_words,
                                             long words_per_sample, int wpw, long n_win,
                                             int n_samples, long long n_pairs,
                                             int n_tiles,
                                             double* __restrict__ d_sum_p0,
                                             double* __restrict__ d_sum_p0_sq,
                                             int* __restrict__ d_n_win_used,
                                             long long* __restrict__ d_tot_comp) {
    // Decode this block's tile-pair (tileA <= tileB) from its flat upper-triangular id.
    const int block_id = static_cast<int>(blockIdx.x);
    int tileA = 0;
    int rem = block_id;
    int row = n_tiles;
    while (rem >= row) { rem -= row; ++tileA; --row; }
    const int tileB = tileA + rem;  // tileB >= tileA

    const int A_base = tileA * kTileSamples;
    const int B_base = tileB * kTileSamples;

    extern __shared__ Readv2Word s_rows[];
    Readv2Word* sA = s_rows;                                  // [TS * wpw]
    Readv2Word* sB = s_rows + static_cast<long>(kTileSamples) * wpw;  // [TS * wpw]

    const int a = static_cast<int>(threadIdx.y);
    const int b = static_cast<int>(threadIdx.x);
    const int I = A_base + a;
    const int J = B_base + b;
    const bool same_tile = (tileA == tileB);
    const bool active = (a < kTileSamples) && (b < kTileSamples) &&
                        (I < n_samples) && (J < n_samples) && (I < J);

    double sum_p0 = 0.0;
    double sum_p0_sq = 0.0;
    int n_used = 0;
    long long tot_comp = 0;

    const int tid = static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
    const int nthreads = static_cast<int>(blockDim.x * blockDim.y);
    const int tile_words = kTileSamples * wpw;

    for (long g = 0; g < n_win; ++g) {
        const long wbase = g * static_cast<long>(wpw);
        // Cooperatively load window g's rows of tile A and tile B into shared.
        for (int e = tid; e < tile_words; e += nthreads) {
            const int lr = e / wpw;   // local row within tile
            const int lw = e % wpw;   // word within window
            const int gA = A_base + lr;
            const int gB = B_base + lr;
            sA[e] = (gA < n_samples)
                        ? d_words[static_cast<long>(gA) * words_per_sample + wbase + lw]
                        : Readv2Word{0, 0};
            sB[e] = (gB < n_samples)
                        ? d_words[static_cast<long>(gB) * words_per_sample + wbase + lw]
                        : Readv2Word{0, 0};
        }
        __syncthreads();

        if (active && !(same_tile && a >= b)) {
            int win_comp = 0;
            int win_mism = 0;
            const Readv2Word* ra = sA + static_cast<long>(a) * wpw;
            const Readv2Word* rb = sB + static_cast<long>(b) * wpw;
            for (int wl = 0; wl < wpw; ++wl) {
                word_counts(ra[wl], rb[wl], win_comp, win_mism);
            }
            if (win_comp > 0) {
                const double p0 =
                    static_cast<double>(win_mism) / static_cast<double>(win_comp);
                sum_p0 += p0;
                sum_p0_sq += p0 * p0;
                ++n_used;
                tot_comp += win_comp;
            }
        }
        __syncthreads();
    }

    if (active && !(same_tile && a >= b)) {
        // Upper-triangular rank of (I<J): C(J,2) + I (matches readv2_unrank_pair).
        const long long r = static_cast<long long>(J) * (J - 1) / 2 + I;
        if (r < n_pairs) {
            d_sum_p0[r] = sum_p0;
            d_sum_p0_sq[r] = sum_p0_sq;
            d_n_win_used[r] = n_used;
            d_tot_comp[r] = tot_comp;
        }
    }
}

}  // namespace

void launch_readv2_mismatch(const Readv2Word* d_words, long words_per_sample, int wpw,
                            long n_win, int n_samples, long long n_pairs,
                            double* d_sum_p0, double* d_sum_p0_sq, int* d_n_win_used,
                            long long* d_tot_comp, bool tiled, cudaStream_t stream) {
    if (n_pairs <= 0) return;

    if (tiled) {
        const int n_tiles = core::cdiv(n_samples, kTileSamples);
        const long n_tile_pairs = static_cast<long>(n_tiles) * (n_tiles + 1) / 2;
        const std::size_t smem =
            static_cast<std::size_t>(2) * kTileSamples * static_cast<std::size_t>(wpw) *
            sizeof(Readv2Word);
        constexpr std::size_t kDefaultSmem = 48u * 1024u;
        constexpr std::size_t kOptinSmem = 99u * 1024u;
        if (smem <= kOptinSmem) {
            if (smem > kDefaultSmem) {
                STEPPE_CUDA_CHECK(cudaFuncSetAttribute(
                    readv2_mismatch_tiled_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
            }
            const dim3 block(kTileSamples, kTileSamples);  // 32x32 = 1024 threads
            const int grid_x = core::grid_for_x(
                n_tile_pairs, 1,
                "readv2 mismatch tiled gridDim.x (tile-pair axis) exceeds kMaxGridX");
            readv2_mismatch_tiled_kernel<<<static_cast<unsigned>(grid_x), block, smem, stream>>>(
                d_words, words_per_sample, wpw, n_win, n_samples, n_pairs, n_tiles,
                d_sum_p0, d_sum_p0_sq, d_n_win_used, d_tot_comp);
            STEPPE_CUDA_CHECK_KERNEL();
            return;
        }
        // Window too wide for shared — fall through to the always-correct baseline.
    }

    constexpr int kThreads = 256;
    const int grid = core::grid_for_x(
        static_cast<long>(n_pairs), kThreads,
        "readv2 mismatch gridDim.x (pair axis) exceeds kMaxGridX — restrict --samples");
    readv2_mismatch_direct_kernel<<<static_cast<unsigned>(grid), kThreads, 0, stream>>>(
        d_words, words_per_sample, wpw, n_win, n_samples, n_pairs,
        d_sum_p0, d_sum_p0_sq, d_n_win_used, d_tot_comp);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

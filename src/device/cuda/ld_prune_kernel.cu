// src/device/cuda/ld_prune_kernel.cu
//
// GPU kernels for the windowed-r2 LD pruner (`--ld-prune WIN:STEP:R2`) — the plink2
// --indep-pairwise math. See ld_prune_kernel.cuh for the three-stage contract. All read the
// SAME 2-bit unpack (genotype_code / kMissingGenotypeCode) the rest of steppe uses, so a decoded
// dosage cannot drift from the FST/KING/PCA paths; the r^2 arithmetic mirrors plink2's exact
// integer form (int64 sufficient statistics -> double cov/variance -> the cov²>thresh·var·var
// test) so a pair's keep/drop decision is bit-identical to plink2's.
//
// The pairwise fold ships in two byte-identical forms: ld_pairwise_over_kernel (scalar, one block
// per pair, emulated-int64 accumulate + shared-memory tree reduction — the STEPPE_LD_KERNEL=old
// gate oracle + fallback) and ld_pairwise_over_dp4a_kernel (the DEFAULT: dp4a int-packed accumulate,
// warp-per-pair, warp-shuffle reduction). The six sufficient statistics {n, Σa, Σb, Σa², Σb², Σab}
// are exact integers that fit int32 for any N < ~5e8 (max 4·N ≈ 92k at the AADR N=23089), so the
// int32/dp4a sums equal the int64 sums bit-for-bit regardless of reduction order (integer add is
// exactly associative), and both paths run the SAME final int64 cross-products + double test —
// making the two kernels' per-pair keep/drop flags provably byte-identical. Only the emulated int64
// muls (Blackwell has no native int64 multiply) leave the hot per-sample loop.
#include "device/cuda/ld_prune_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"      // genotype_code, kCodesPerByte, kMissingGenotypeCode
#include "core/internal/launch_config.hpp"  // kDecodeBlockX/Y, cdiv, kMaxGridX
#include "device/cuda/check.cuh"

namespace steppe::device {

using core::genotype_code;
using core::kMissingGenotypeCode;

namespace {

// Block width for the per-SNP / per-pair reductions (a bandwidth-friendly power of two, matching
// the KING accumulate block). The shared-memory reductions below are sized to it.
constexpr int kLdReduceBlock = 256;

// Pairs (one warp each) batched per block in the dp4a pairwise fold: 256 threads = 8 warps = 8
// (target, distance) pairs, so the 8 independent warp-shuffle reductions replace the scalar path's
// single 8-step / 6-array shared-memory tree — no shared memory, no block-wide __syncthreads.
constexpr int kLdWarpsPerBlock = kLdReduceBlock / 32;

// SNP-major dosage decode — 1D grid-stride over the n_dec x N pane (SNP-major idx = sl*N + g), so
// no grid dimension carries the (unbounded) SNP count. Writes the 2-bit code (0/1/2 dosage,
// 3 = missing) into out_dos[idx]; consecutive threads hit consecutive samples g -> coalesced writes.
__global__ void ld_dosage_decode_snpmajor_kernel(const std::uint8_t* __restrict__ packed,
                                                 std::size_t bytes_per_record, int N, long s_lo,
                                                 long n_dec, std::uint8_t* __restrict__ out_dos) {
    const long total = n_dec * static_cast<long>(N);
    const long stride = static_cast<long>(gridDim.x) * blockDim.x;
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += stride) {
        const long sl = idx / static_cast<long>(N);
        const long g = idx - sl * static_cast<long>(N);
        const long gs = s_lo + sl;
        const std::size_t byte_in_record =
            static_cast<std::size_t>(gs) / static_cast<std::size_t>(core::kCodesPerByte);
        const int pos_in_byte = static_cast<int>(gs % core::kCodesPerByte);
        const std::uint8_t byte =
            packed[static_cast<std::size_t>(g) * bytes_per_record + byte_in_record];
        out_dos[static_cast<std::size_t>(idx)] = genotype_code(byte, pos_in_byte);
    }
}

// Per-target-SNP global stats — ONE BLOCK per target SNP t. Threads stride-reduce the N samples of
// d_dos row t into (nm, Σg, Σg²), a shared-memory tree reduction folds them, thread 0 writes.
__global__ void ld_variant_stats_kernel(const std::uint8_t* __restrict__ dos, int N, long n_tgt,
                                        long* __restrict__ out_nm, long* __restrict__ out_sum,
                                        long* __restrict__ out_ssq) {
    const long t = static_cast<long>(blockIdx.x);
    if (t >= n_tgt) return;
    const std::size_t row = static_cast<std::size_t>(t) * static_cast<std::size_t>(N);

    long nm = 0, sum = 0, ssq = 0;
    for (int g = static_cast<int>(threadIdx.x); g < N; g += static_cast<int>(blockDim.x)) {
        const std::uint8_t c = dos[row + static_cast<std::size_t>(g)];
        if (c != kMissingGenotypeCode) {
            const long v = static_cast<long>(c);
            ++nm;
            sum += v;
            ssq += v * v;
        }
    }

    __shared__ long s_nm[kLdReduceBlock];
    __shared__ long s_sum[kLdReduceBlock];
    __shared__ long s_ssq[kLdReduceBlock];
    const unsigned tx = threadIdx.x;
    s_nm[tx] = nm;
    s_sum[tx] = sum;
    s_ssq[tx] = ssq;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tx < stride) {
            s_nm[tx] += s_nm[tx + stride];
            s_sum[tx] += s_sum[tx + stride];
            s_ssq[tx] += s_ssq[tx + stride];
        }
        __syncthreads();
    }
    if (tx == 0) {
        out_nm[t] = s_nm[0];
        out_sum[t] = s_sum[0];
        out_ssq[t] = s_ssq[0];
    }
}

// Pairwise over-threshold flags — ONE BLOCK per (target t, distance d) pair. Threads stride-reduce
// the pairwise-nonmissing sufficient statistics (n, Σa, Σb, Σa², Σb², Σab) over the N samples of
// rows (t) and (t+d), fold them, and thread 0 applies plink2's exact integer r^2 test.
__global__ void ld_pairwise_over_kernel(const std::uint8_t* __restrict__ dos, int N, long s_lo,
                                       long n_tgt, long n_dec, long M,
                                       const int* __restrict__ chrom, int window,
                                       double r2_thresh_eps, std::uint8_t* __restrict__ out_over) {
    const long W1 = static_cast<long>(window) - 1;
    const long idx = static_cast<long>(blockIdx.x);
    if (idx >= n_tgt * W1) return;
    const long t = idx / W1;
    const long dd = idx - t * W1;          // 0-based distance slot
    const long d = dd + 1;                 // pair distance in [1, window-1]
    const long i = s_lo + t;               // earlier SNP (global index)
    const long j = i + d;                  // later SNP (global index)

    const std::size_t out_idx =
        static_cast<std::size_t>(t) * static_cast<std::size_t>(W1) + static_cast<std::size_t>(dd);

    // Out of range or a cross-chromosome pair: not a candidate (never enters the greedy).
    if (j >= M || (t + d) >= n_dec || chrom[i] != chrom[j]) {
        if (threadIdx.x == 0) out_over[out_idx] = 0;
        return;
    }

    const std::size_t ri = static_cast<std::size_t>(t) * static_cast<std::size_t>(N);
    const std::size_t rj = static_cast<std::size_t>(t + d) * static_cast<std::size_t>(N);

    long n = 0, sa = 0, sb = 0, sqa = 0, sqb = 0, dot = 0;
    for (int g = static_cast<int>(threadIdx.x); g < N; g += static_cast<int>(blockDim.x)) {
        const std::uint8_t ca = dos[ri + static_cast<std::size_t>(g)];
        const std::uint8_t cb = dos[rj + static_cast<std::size_t>(g)];
        if (ca != kMissingGenotypeCode && cb != kMissingGenotypeCode) {
            const long a = static_cast<long>(ca);
            const long b = static_cast<long>(cb);
            ++n;
            sa += a;
            sb += b;
            sqa += a * a;
            sqb += b * b;
            dot += a * b;
        }
    }

    __shared__ long s_n[kLdReduceBlock];
    __shared__ long s_sa[kLdReduceBlock];
    __shared__ long s_sb[kLdReduceBlock];
    __shared__ long s_sqa[kLdReduceBlock];
    __shared__ long s_sqb[kLdReduceBlock];
    __shared__ long s_dot[kLdReduceBlock];
    const unsigned tx = threadIdx.x;
    s_n[tx] = n;
    s_sa[tx] = sa;
    s_sb[tx] = sb;
    s_sqa[tx] = sqa;
    s_sqb[tx] = sqb;
    s_dot[tx] = dot;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tx < stride) {
            s_n[tx] += s_n[tx + stride];
            s_sa[tx] += s_sa[tx + stride];
            s_sb[tx] += s_sb[tx + stride];
            s_sqa[tx] += s_sqa[tx + stride];
            s_sqb[tx] += s_sqb[tx + stride];
            s_dot[tx] += s_dot[tx + stride];
        }
        __syncthreads();
    }
    if (tx != 0) return;

    // plink2 IndepPairwise integer form: cov12 = dot·n - Σa·Σb, var_k = Σk²·n - (Σk)², all exact in
    // int64, cast to double for the ratio test cov12² > thresh·var1·var2. n == 0 -> 0 > 0 = drop.
    const long nn = s_n[0];
    const long SA = s_sa[0], SB = s_sb[0], SQA = s_sqa[0], SQB = s_sqb[0], DOT = s_dot[0];
    const double cov12 = static_cast<double>(DOT * nn - SA * SB);
    const double var1 = static_cast<double>(SQA * nn - SA * SA);
    const double var2 = static_cast<double>(SQB * nn - SB * SB);
    out_over[out_idx] =
        (cov12 * cov12 > r2_thresh_eps * var1 * var2) ? std::uint8_t{1} : std::uint8_t{0};
}

// dp4a int-packed pairwise-over-threshold flags — ONE WARP per (target t, distance d) pair,
// kLdWarpsPerBlock pairs per block. Each lane folds four samples at a time via __dp4a: it packs
// four consecutive-coalesced dosages of rows (t) and (t+d) into a 32-bit word, zeros any sample
// where EITHER dosage is missing (pairwise-complete deletion, via the __vcmpeq4 byte compare), and
// accumulates the six sufficient statistics {n, Σa, Σb, Σa², Σb², Σab} in int32 — no emulated int64
// multiply in the hot loop. A warp-shuffle reduction folds the 32 lanes; lane 0 then runs the SAME
// int64 cross-products + double test as ld_pairwise_over_kernel, so the flag is byte-identical.
__global__ void ld_pairwise_over_dp4a_kernel(const std::uint8_t* __restrict__ dos, int N, long s_lo,
                                            long n_tgt, long n_dec, long M,
                                            const int* __restrict__ chrom, int window,
                                            double r2_thresh_eps,
                                            std::uint8_t* __restrict__ out_over) {
    const long W1 = static_cast<long>(window) - 1;
    const int warp_in_block = static_cast<int>(threadIdx.x) >> 5;  // threadIdx.x / 32
    const int lane = static_cast<int>(threadIdx.x) & 31;           // threadIdx.x % 32
    const long idx = static_cast<long>(blockIdx.x) * kLdWarpsPerBlock + warp_in_block;
    if (idx >= n_tgt * W1) return;  // whole-warp uniform: every lane shares idx
    const long t = idx / W1;
    const long dd = idx - t * W1;   // 0-based distance slot
    const long d = dd + 1;          // pair distance in [1, window-1]
    const long i = s_lo + t;        // earlier SNP (global index)
    const long j = i + d;           // later SNP (global index)

    const std::size_t out_idx =
        static_cast<std::size_t>(t) * static_cast<std::size_t>(W1) + static_cast<std::size_t>(dd);

    // Out of range or a cross-chromosome pair: not a candidate (whole-warp uniform).
    if (j >= M || (t + d) >= n_dec || chrom[i] != chrom[j]) {
        if (lane == 0) out_over[out_idx] = 0;
        return;
    }

    const std::uint8_t* __restrict__ arow = dos + static_cast<std::size_t>(t) * static_cast<std::size_t>(N);
    const std::uint8_t* __restrict__ brow =
        dos + static_cast<std::size_t>(t + d) * static_cast<std::size_t>(N);

    // int32 accumulators (max block sum is 4·N ≈ 92k at N=23089, far under 2^31 — exact, no overflow).
    unsigned un = 0, usa = 0, usb = 0, usqa = 0, usqb = 0, udot = 0;

    // Full 128-sample groups: lane l packs samples {g, g+32, g+64, g+96} so each of the four
    // sub-loads is a warp-coalesced 32-byte read (lanes 0..31 hit consecutive bytes).
    const int nfull = N >> 7;  // N / 128
    for (int it = 0; it < nfull; ++it) {
        const int g = (it << 7) + lane;  // it*128 + lane
        const unsigned a0 = arow[g], a1 = arow[g + 32], a2 = arow[g + 64], a3 = arow[g + 96];
        const unsigned b0 = brow[g], b1 = brow[g + 32], b2 = brow[g + 64], b3 = brow[g + 96];
        const unsigned a4 = a0 | (a1 << 8) | (a2 << 16) | (a3 << 24);
        const unsigned b4 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        // __vcmpeq4 -> 0xFF per byte where the dosage == kMissingGenotypeCode (3). A sample is used
        // only when BOTH are present, so OR the two miss masks; ~miss keeps present bytes (0/1/2).
        const unsigned miss = __vcmpeq4(a4, 0x03030303u) | __vcmpeq4(b4, 0x03030303u);
        const unsigned vmask = ~miss;                    // 0xFF per pairwise-present byte
        const unsigned am = a4 & vmask;                  // present: 0/1/2, missing: 0
        const unsigned bm = b4 & vmask;
        const unsigned vones = vmask & 0x01010101u;      // 0x01 per pairwise-present byte
        un = __dp4a(vones, 0x01010101u, un);             // += count of present pairs
        usa = __dp4a(am, 0x01010101u, usa);              // += Σa
        usb = __dp4a(bm, 0x01010101u, usb);              // += Σb
        usqa = __dp4a(am, am, usqa);                     // += Σa² (missing bytes are 0)
        usqb = __dp4a(bm, bm, usqb);                     // += Σb²
        udot = __dp4a(am, bm, udot);                     // += Σab
    }
    // Scalar tail over the last N%128 samples, lane-strided (each sample handled exactly once).
    for (int g = (nfull << 7) + lane; g < N; g += 32) {
        const unsigned ca = arow[g], cb = brow[g];
        if (ca != 3u && cb != 3u) {
            un += 1u;
            usa += ca;
            usb += cb;
            usqa += ca * ca;
            usqb += cb * cb;
            udot += ca * cb;
        }
    }

    // Warp-shuffle reduction of the six int32 partials (all 32 lanes active; exact integer add).
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        un += __shfl_down_sync(0xffffffffu, un, off);
        usa += __shfl_down_sync(0xffffffffu, usa, off);
        usb += __shfl_down_sync(0xffffffffu, usb, off);
        usqa += __shfl_down_sync(0xffffffffu, usqa, off);
        usqb += __shfl_down_sync(0xffffffffu, usqb, off);
        udot += __shfl_down_sync(0xffffffffu, udot, off);
    }
    if (lane != 0) return;

    // Identical to ld_pairwise_over_kernel: exact int64 cov/var cross-products, then the double
    // ratio test cov12² > thresh·var1·var2 (n == 0 -> 0 > 0 = drop). The int32 block sums promote
    // to the SAME int64 values the scalar path holds, so this is byte-for-byte the same decision.
    const long nn = static_cast<long>(un);
    const long SA = static_cast<long>(usa), SB = static_cast<long>(usb);
    const long SQA = static_cast<long>(usqa), SQB = static_cast<long>(usqb), DOT = static_cast<long>(udot);
    const double cov12 = static_cast<double>(DOT * nn - SA * SB);
    const double var1 = static_cast<double>(SQA * nn - SA * SA);
    const double var2 = static_cast<double>(SQB * nn - SB * SB);
    out_over[out_idx] =
        (cov12 * cov12 > r2_thresh_eps * var1 * var2) ? std::uint8_t{1} : std::uint8_t{0};
}

}  // namespace

void launch_ld_dosage_decode_snpmajor(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                                      int N, long s_lo, long n_dec, std::uint8_t* d_dos,
                                      cudaStream_t stream) {
    if (N <= 0 || n_dec <= 0) return;
    const long total = n_dec * static_cast<long>(N);
    const unsigned grid = static_cast<unsigned>(core::grid_stride_extent(total, kLdReduceBlock));
    ld_dosage_decode_snpmajor_kernel<<<grid, static_cast<unsigned>(kLdReduceBlock), 0, stream>>>(
        d_packed, bytes_per_record, N, s_lo, n_dec, d_dos);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_ld_variant_stats(const std::uint8_t* d_dos, int N, long n_tgt, long* d_nm, long* d_sum,
                             long* d_ssq, cudaStream_t stream) {
    if (N <= 0 || n_tgt <= 0) return;
    STEPPE_ASSERT(static_cast<unsigned long long>(n_tgt) <= core::kMaxGridX,
                  "ld_prune variant-stats gridDim.x (one block per SNP) exceeds kMaxGridX");
    ld_variant_stats_kernel<<<static_cast<unsigned>(n_tgt), static_cast<unsigned>(kLdReduceBlock), 0,
                              stream>>>(d_dos, N, n_tgt, d_nm, d_sum, d_ssq);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_ld_pairwise_over(const std::uint8_t* d_dos, int N, long s_lo, long n_tgt, long n_dec,
                             long M, const int* d_chrom, int window, double r2_thresh_eps,
                             std::uint8_t* d_over, cudaStream_t stream) {
    if (N <= 0 || n_tgt <= 0 || window <= 1) return;
    const long W1 = static_cast<long>(window) - 1;
    const long nblocks = n_tgt * W1;
    STEPPE_ASSERT(static_cast<unsigned long long>(nblocks) <= core::kMaxGridX,
                  "ld_prune pairwise gridDim.x (n_tgt*(window-1)) exceeds kMaxGridX — reduce the "
                  "SNP tile or window");
    ld_pairwise_over_kernel<<<static_cast<unsigned>(nblocks), static_cast<unsigned>(kLdReduceBlock),
                              0, stream>>>(d_dos, N, s_lo, n_tgt, n_dec, M, d_chrom, window,
                                           r2_thresh_eps, d_over);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_ld_pairwise_over_dp4a(const std::uint8_t* d_dos, int N, long s_lo, long n_tgt,
                                  long n_dec, long M, const int* d_chrom, int window,
                                  double r2_thresh_eps, std::uint8_t* d_over, cudaStream_t stream) {
    if (N <= 0 || n_tgt <= 0 || window <= 1) return;
    const long W1 = static_cast<long>(window) - 1;
    const long npairs = n_tgt * W1;
    // One warp per pair, kLdWarpsPerBlock warps per block. gridDim.x = ceil(npairs / warps) <= the
    // scalar path's npairs, itself already asserted <= kMaxGridX by the caller's tile sizing.
    const long nblocks = (npairs + kLdWarpsPerBlock - 1) / kLdWarpsPerBlock;
    STEPPE_ASSERT(static_cast<unsigned long long>(nblocks) <= core::kMaxGridX,
                  "ld_prune dp4a pairwise gridDim.x exceeds kMaxGridX — reduce the SNP tile or "
                  "window");
    ld_pairwise_over_dp4a_kernel<<<static_cast<unsigned>(nblocks),
                                   static_cast<unsigned>(kLdReduceBlock), 0, stream>>>(
        d_dos, N, s_lo, n_tgt, n_dec, M, d_chrom, window, r2_thresh_eps, d_over);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

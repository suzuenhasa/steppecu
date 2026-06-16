// src/device/cuda/decode_af_kernel.cu
//
// S0 + S1 — GPU genotype decode → per-pop allele-frequency reduction
// (architecture.md §5 S0 Format decode + S1 Allele-freq reduction, §7, §11.3;
// ROADMAP M1). The 2-bit unpack (raw-value mapping) + segmented reduction over
// the individual RECORDS within each population segment → integer AC/AN, then a
// single FP64 divide → the Q/V/N contract.
//
// What this TU owns (architecture.md §5 S0/S1 — the decode front-end's kernel):
//   1. decode_af_kernel  — one thread per (population, SNP); segmented reduction
//                          over the population's individuals; AC/AN → Q/V/N via
//                          the SHARED core primitive (core/internal/decode_af.hpp).
//   2. launch_decode_af  — the narrow launch wrapper (no <<<>>> in host code).
//
// PRECISION (architecture.md §11.3, §12): the decode is BANDWIDTH-BOUND; the
// accumulation is INTEGER (AC/AN) and the finalize is a single native FP64 divide
// — reduced precision buys nothing and would break the bit-for-bit oracle match.
// Q is exact because AC/AN are integer-accumulated and divided ONCE.
//
// COALESCING (architecture.md §11.3): threadIdx.x runs over the SNP axis, so
// adjacent threads in a warp read adjacent bytes of the SAME individual record on
// each step of the individual loop (byte index = s/4) — coalesced reads on the
// within-record SNP axis, as the brief requires.
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). It includes
// the SHARED host/device decode primitive so the CPU oracle and this path cannot
// diverge on the unpack/missing/divide (architecture.md §13; ROADMAP §5).
#include "device/cuda/decode_af_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"       // genotype_code, genotype_valid, finalize_af
#include "core/internal/launch_config.hpp"   // cdiv, kDecodeBlockX/Y (the launch-math home)
#include "device/cuda/check.cuh"             // STEPPE_CUDA_CHECK_KERNEL

namespace steppe::device {

using core::AfResult;
using core::finalize_af;
using core::genotype_code;
using core::genotype_valid;
using core::kDecodeBlockX;  // SNP axis (32 = one warp, warp-aligned for coalescing)
using core::kDecodeBlockY;  // population axis (32*8 = 256 threads/block)

namespace {

/// Decode one (population, SNP) entry. One thread owns (i = population row, s =
/// SNP). It reduces over population i's individual segment [seg_begin, seg_end),
/// unpacking SNP s's 2-bit code from each record's byte (s/4) at position (s%4),
/// accumulating AC (ref-allele copies) / AN (non-missing individuals), then writes
/// Q/V/N via the SHARED finalize. Column-major [P × M]: element (i,s) at i + P·s.
__global__ void decode_af_kernel(const std::uint8_t* __restrict__ packed,
                                 std::size_t bytes_per_record,
                                 const std::size_t* __restrict__ pop_offsets,
                                 int P, long M, int ploidy,
                                 double* __restrict__ Q,
                                 double* __restrict__ V,
                                 double* __restrict__ N) {
    const long s = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;  // SNP
    const int i = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;    // population
    if (s >= M || i >= P) return;

    const std::size_t seg_begin = pop_offsets[static_cast<std::size_t>(i)];
    const std::size_t seg_end = pop_offsets[static_cast<std::size_t>(i) + 1];
    const std::size_t byte_in_rec = static_cast<std::size_t>(s) / 4u;
    const int pos_in_byte = static_cast<int>(s & 3);

    long ac = 0;  // Σ ref-allele copies over non-missing individuals
    long an = 0;  // count of non-missing individuals
    for (std::size_t g = seg_begin; g < seg_end; ++g) {
        const std::uint8_t byte = packed[g * bytes_per_record + byte_in_rec];
        const std::uint8_t code = genotype_code(byte, pos_in_byte);
        if (genotype_valid(code)) {
            ac += static_cast<long>(code);
            ++an;
        }
    }

    const AfResult r = finalize_af(ac, an, ploidy);
    const std::size_t off =
        static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
    Q[off] = r.q;
    V[off] = r.v;
    N[off] = r.n;
}

}  // namespace

void launch_decode_af(const std::uint8_t* d_packed,
                      std::size_t bytes_per_record,
                      const std::size_t* d_pop_offsets,
                      int P, long M, int ploidy,
                      double* d_Q, double* d_V, double* d_N,
                      cudaStream_t stream) {
    // Grid math via the single launch-config home (architecture.md §4, §8): the
    // SNP axis (M, `long`) MUST use the long cdiv overload; the population axis
    // (P, `int`) uses the int one. The 32×8 block is NON-square, so we pass the
    // explicit per-axis dims (kDecodeBlockX/Y) — NOT grid_for's square default.
    const dim3 block(kDecodeBlockX, kDecodeBlockY);
    const dim3 grid(static_cast<unsigned>(core::cdiv(M, static_cast<long>(kDecodeBlockX))),
                    static_cast<unsigned>(core::cdiv(P, kDecodeBlockY)));
    decode_af_kernel<<<grid, block, 0, stream>>>(d_packed, bytes_per_record,
                                                 d_pop_offsets, P, M, ploidy,
                                                 d_Q, d_V, d_N);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

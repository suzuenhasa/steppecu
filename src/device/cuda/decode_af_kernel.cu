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
// COALESCING (architecture.md §11.3; cleanup 20.1/MED): threadIdx.x runs over the
// SNP axis, so adjacent threads in a warp read adjacent bytes of the SAME individual
// record on each step of the individual loop (byte index = s/kCodesPerByte) —
// coalesced reads on the within-record SNP axis, as the brief requires. The OUTPUT
// arena, however, is column-major [P×M] (unit-stride in the population i, NOT s), so
// the naive per-thread store would be P-strided / uncoalesced. The kernel therefore
// stages the finalized Q/V/N into a padded shared tile and re-emits them through a
// thread remap whose consecutive lanes vary the population axis, turning the three
// stores into coalesced population-contiguous bursts (see the kernel comment) — the
// coalesced input reads are untouched.
//
// WITHIN-WARP READ REUSE (cleanup 20.3/MED — DEFERRED, profile-gated): with the SNP
// axis on threadIdx.x, byte_in_record = s/kCodesPerByte takes only 8 distinct values
// across a warp's 32 lanes ({0,0,0,0,1,1,1,1,…}), so each record byte is fetched by 4
// consecutive lanes, and that 8-byte window is re-fetched every individual-loop step.
// The reads are CONTIGUOUS so L1/L2 serves the duplicates — not a correctness or
// coalescing bug, and this is the documented bandwidth-bound design (above). A warp
// could instead cooperatively load the 8-byte window once and broadcast the 2-bit
// extraction (shuffle/shared), but that adds a barrier/shuffle hazard surface and is
// ONLY worth it if a profiler shows L1 is not already absorbing the duplicate fetch.
// No such profile exists, so this is left as-is by design (cleanup task 20.3).
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). It includes
// the SHARED host/device decode primitive so the CPU oracle and this path cannot
// diverge on the unpack/missing/divide (architecture.md §13; ROADMAP §5).
#include "device/cuda/decode_af_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"       // genotype_code, accumulate_genotype, finalize_af
#include "core/internal/launch_config.hpp"   // cdiv, kDecodeBlockX/Y (the launch-math home)
#include "device/cuda/check.cuh"             // STEPPE_CUDA_CHECK_KERNEL

namespace steppe::device {

using core::accumulate_genotype;
using core::AfResult;
using core::finalize_af;
using core::genotype_code;
using core::kDecodeBlockX;  // SNP axis (32 = one warp, warp-aligned for coalescing)
using core::kDecodeBlockY;  // population axis (32*8 = 256 threads/block)

namespace {

/// Decode one (population, SNP) entry. One thread owns (i = population row, s =
/// SNP). It reduces over population i's individual segment [seg_begin, seg_end),
/// unpacking SNP s's 2-bit code from each record's byte (s/kCodesPerByte) at
/// position (s%kCodesPerByte),
/// folding it into AC (ref-allele copies) / AN (non-missing individuals) via the
/// SHARED accumulate_genotype, then writing Q/V/N via the SHARED finalize.
/// Column-major [P × M]: element (i,s) at i + P·s.
///
/// COALESCING (cleanup 20.1/MED — uncoalesced output stores):
///   * INPUT reads: threadIdx.x rides the SNP axis s, so a warp's lanes read
///     ADJACENT packed bytes of the same individual record (byte = s/kCodesPerByte)
///     — coalesced, and load-bearing: the 32-wide x edge (kDecodeBlockX) is FORCED
///     to stay on the SNP axis to keep these reads contiguous, so we cannot swap the
///     thread→element map the way the (square-block) f2 feeder does. This is the
///     dominant traffic (per-individual loop), kept exactly as before.
///   * OUTPUT stores: the [P×M] arena is column-major ⇒ the unit-stride dimension is
///     the population i, NOT s. With s on threadIdx.x the three Q/V/N stores were
///     P-strided across a warp (a 32-way one-sector-per-lane scatter). FIX: stage
///     each thread's finalized (q,v,n) into a shared tile keyed by (pop-local,
///     snp-local), __syncthreads, then re-emit through a thread→element remap whose
///     consecutive lanes vary the POPULATION (unit-stride) axis ⇒ coalesced
///     population-contiguous bursts. The tile inner dim is padded +1 to break the
///     32-bank stride on the transposed shared read (NVIDIA, "An Efficient Matrix
///     Transpose in CUDA C/C++": pad the tile width so a column of data no longer
///     maps to a single bank, eliminating the worst-case 32-way conflict). The
///     finalized per-element math and the written global addresses are IDENTICAL —
///     only the store ACCESS PATTERN changes (§12 parity unchanged).
// [21.4] __launch_bounds__(kDecodeBlockX * kDecodeBlockY) == 32*8 = 256 pins the
// register cap to the SOLE launch block (the fixed dim3(kDecodeBlockX, kDecodeBlockY),
// launch_decode_af). Occupancy here is now multi-resource bounded (the three __shared__
// transpose tiles tQ/tV/tN added by the group-20 coalescing fix), so the register pin is
// a valid forward-compat guard; the fixed block never exceeds the bound (Prog Guide §5.4).
__global__ void __launch_bounds__(kDecodeBlockX * kDecodeBlockY)
decode_af_kernel(const std::uint8_t* __restrict__ packed,
                                 std::size_t bytes_per_record,
                                 const std::size_t* __restrict__ pop_offsets,
                                 int P, long M, int ploidy,
                                 double* __restrict__ Q,
                                 double* __restrict__ V,
                                 double* __restrict__ N) {
    // Shared staging tiles, one per output, padded +1 on the snp-local (inner) dim
    // so the transposed store-phase read (consecutive lanes vary pop-local, the
    // outer index) does not 32-bank-conflict (NVIDIA efficient-transpose recipe).
    // [kDecodeBlockY = 8 pops] × [kDecodeBlockX = 32 SNPs] + pad.
    __shared__ double tQ[kDecodeBlockY][kDecodeBlockX + 1];
    __shared__ double tV[kDecodeBlockY][kDecodeBlockX + 1];
    __shared__ double tN[kDecodeBlockY][kDecodeBlockX + 1];

    // Tile origin (column-major [P×M]): pops [pop0, pop0+8), SNPs [snp0, snp0+32).
    const int pop0 = static_cast<int>(blockIdx.y) * blockDim.y;
    const long snp0 = static_cast<long>(blockIdx.x) * blockDim.x;

    // --- compute phase: SNP on threadIdx.x (coalesced packed-byte reads kept) -----
    const long s = snp0 + threadIdx.x;        // SNP
    const int i = pop0 + static_cast<int>(threadIdx.y);  // population
    if (s < M && i < P) {
        const std::size_t seg_begin = pop_offsets[static_cast<std::size_t>(i)];
        const std::size_t seg_end = pop_offsets[static_cast<std::size_t>(i) + 1];
        const std::size_t byte_in_record =
            static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
        const int pos_in_byte = static_cast<int>(s % core::kCodesPerByte);

        std::int64_t ac = 0;  // Σ ref-allele copies over non-missing individuals
        std::int64_t an = 0;  // count of non-missing individuals
        for (std::size_t g = seg_begin; g < seg_end; ++g) {
            const std::uint8_t byte = packed[g * bytes_per_record + byte_in_record];
            const std::uint8_t code = genotype_code(byte, pos_in_byte);
            accumulate_genotype(code, ac, an);  // shared inner step (A-1/B27)
        }
        const AfResult r = finalize_af(ac, an, ploidy);
        tQ[threadIdx.y][threadIdx.x] = r.q;
        tV[threadIdx.y][threadIdx.x] = r.v;
        tN[threadIdx.y][threadIdx.x] = r.n;
    }

    __syncthreads();

    // --- store phase: remap so consecutive lanes vary the POPULATION axis ---------
    // Flat thread id; (pop-local p, snp-local q) with p = tid % blockDim.y so
    // consecutive lanes increment p (the column-major unit-stride dim) ⇒ the global
    // stores hit consecutive addresses i + P·s (coalesced pop-contiguous bursts).
    const int bdx = static_cast<int>(blockDim.x);
    const int bdy = static_cast<int>(blockDim.y);
    const int tid = static_cast<int>(threadIdx.y) * bdx + static_cast<int>(threadIdx.x);
    const int p = tid % bdy;                              // pop-local  (0..7)
    const int q = tid / bdy;                              // snp-local  (0..31)
    const int io = pop0 + p;                              // global population
    const long so = snp0 + static_cast<long>(q);          // global SNP
    if (io < P && so < M) {
        const std::size_t off = static_cast<std::size_t>(io) +
                                static_cast<std::size_t>(P) * static_cast<std::size_t>(so);
        Q[off] = tQ[p][q];
        V[off] = tV[p][q];
        N[off] = tN[p][q];
    }
}

}  // namespace

void launch_decode_af(const std::uint8_t* d_packed,
                      std::size_t bytes_per_record,
                      const std::size_t* d_pop_offsets,
                      int P, long M, int ploidy,
                      double* d_Q, double* d_V, double* d_N,
                      cudaStream_t stream) {
    // Grid math via the single launch-config home (architecture.md §4, §7, §8): the
    // SNP axis (M, `long`) rides gridDim.x via the long cdiv overload — x is the only
    // 2^31 axis, the correct home for the large SNP count. The population axis (P,
    // `int`) rides gridDim.y through `grid_for(P, kDecodeBlockY)`, whose y/z-cap
    // assert (kMaxGridY = 65 535) now applies (cleanup X-7/B6: the decode grid.y clamp
    // is folded into grid_for); P ≤ ~4266 ≪ that, so it is always satisfied. The 32×8
    // block is NON-square, so the y axis passes the explicit kDecodeBlockY edge — NOT
    // grid_for's square default; the SNP/x axis keeps the long cdiv overload grid_for
    // (int-only) cannot provide.
    const dim3 block(kDecodeBlockX, kDecodeBlockY);
    const dim3 grid(static_cast<unsigned>(core::cdiv(M, static_cast<long>(kDecodeBlockX))),
                    static_cast<unsigned>(core::grid_for(P, kDecodeBlockY)));
    decode_af_kernel<<<grid, block, 0, stream>>>(d_packed, bytes_per_record,
                                                 d_pop_offsets, P, M, ploidy,
                                                 d_Q, d_V, d_N);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

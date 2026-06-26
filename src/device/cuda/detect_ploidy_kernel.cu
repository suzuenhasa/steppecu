// src/device/cuda/detect_ploidy_kernel.cu
//
// M-FR-0 — the AT2 pseudo-haploid PER-SAMPLE PLOIDY prepass on the GPU (the L2
// host-compute fix). A literal bit-parity port of the host io::detect_sample_ploidy
// loop (src/io/ploidy_detect.cpp): one thread per gathered individual scans the
// first min(kPloidyDetectSnps, n_snp) SNPs of that individual's packed record for a
// HETEROZYGOUS call and emits 2 (diploid) on the first het, else 1 (pseudo-haploid).
//
// What this TU owns:
//   1. detect_ploidy_kernel  — one thread per individual; the het-scan over the
//                              detection window via the SHARED core::genotype_code.
//   2. launch_detect_ploidy  — the narrow launch wrapper (no <<<>>> in host code).
//
// BIT-PARITY (the M-FR-0 gate): the detection is INTEGER / BIT ops only — the 2-bit
// unpack (core::genotype_code), the het comparison (== kHeterozygousGenotypeCode),
// and the window cap (min(kPloidyDetectSnps, n_snp)) are EXACTLY the host loop's,
// using the SAME shared core primitives (core/internal/decode_af.hpp). No floating
// point, no reduction order, no precision lane — so the device ploidy vector is
// bit-identical to the host detector by construction (architecture.md §13). It reads
// the SAME individual-major packed bytes decode_af reads (byte s/4, position s%4,
// MSB-first), so it is the on-device twin of the host scan over the same tile.
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). It includes the
// SHARED host/device decode primitive so the host scan and this path cannot diverge.
#include "device/cuda/detect_ploidy_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"       // genotype_code, kHeterozygousGenotypeCode, kPloidyDetectSnps, kPloidy*
#include "core/internal/launch_config.hpp"   // cdiv, kMaxGridX
#include "device/cuda/check.cuh"             // STEPPE_CUDA_CHECK_KERNEL

namespace steppe::device {

namespace {

constexpr int kPloidyBlock = 256;

/// One thread per gathered individual g in [0, n_individuals). Scans the first
/// `window = min(kPloidyDetectSnps, n_snp)` SNPs of g's packed record (the SAME byte
/// layout decode_af reads) for a het call; writes 2 (diploid) on the first het, else
/// 1 (pseudo-haploid). A literal port of io::detect_sample_ploidy (ploidy_detect.cpp)
/// over core::genotype_code / kHeterozygousGenotypeCode — integer/bit ops, bit-exact.
__global__ void detect_ploidy_kernel(const std::uint8_t* __restrict__ packed,
                                     std::size_t bytes_per_record,
                                     std::size_t n_individuals, std::size_t window,
                                     int* __restrict__ ploidy) {
    const std::size_t g =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= n_individuals) return;

    // Initialize to PSEUDO-HAPLOID (ploidy 1, the AT2 default); a het in the window
    // bumps to DIPLOID (ploidy 2). Mirrors the host loop's `ploidy(g, 1)` + break.
    int pl = core::kPloidyPseudoHaploid;
    const std::uint8_t* rec = packed + g * bytes_per_record;
    for (std::size_t s = 0; s < window; ++s) {
        const std::size_t byte_in_rec =
            s / static_cast<std::size_t>(core::kCodesPerByte);
        const int pos_in_byte =
            static_cast<int>(s % static_cast<std::size_t>(core::kCodesPerByte));
        // First het call (code == 1) ⇒ diploid; a haploid genome cannot be het (AT2).
        if (core::genotype_code(rec[byte_in_rec], pos_in_byte) ==
            core::kHeterozygousGenotypeCode) {
            pl = core::kPloidyDiploid;
            break;
        }
    }
    ploidy[g] = pl;
}

}  // namespace

void launch_detect_ploidy(const std::uint8_t* d_packed,
                          std::size_t bytes_per_record,
                          std::size_t n_individuals, std::size_t n_snp,
                          int* d_ploidy, cudaStream_t stream) {
    if (n_individuals == 0) return;
    // AT2 ntest capped at the SNPs the tile carries (a shorter record scans its whole
    // prefix — exactly what the host loop does when kPloidyDetectSnps > n_snp).
    const std::size_t window =
        (static_cast<std::size_t>(core::kPloidyDetectSnps) < n_snp)
            ? static_cast<std::size_t>(core::kPloidyDetectSnps)
            : n_snp;
    // window == 0 ⇒ no SNPs to scan: the kernel still launches unconditionally and
    // every sample stays the AT2 default pseudo-haploid (the het-scan loop body never
    // runs for window=0), matching the host loop's all-1 init — so the output is still
    // written without a special-case branch here.
    const long grid_x =
        core::cdiv(static_cast<long>(n_individuals), static_cast<long>(kPloidyBlock));
    STEPPE_ASSERT(grid_x >= 0 &&
                      static_cast<unsigned long long>(grid_x) <= core::kMaxGridX,
                  "detect-ploidy gridDim.x (individual axis) exceeds kMaxGridX "
                  "(architecture.md §7) — tile the individual axis");
    detect_ploidy_kernel<<<static_cast<unsigned>(grid_x), kPloidyBlock, 0, stream>>>(
        d_packed, bytes_per_record, n_individuals, window, d_ploidy);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

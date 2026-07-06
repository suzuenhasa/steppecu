// src/device/cuda/detect_ploidy_kernel.cu
//
// GPU per-individual ploidy prepass: one thread per gathered individual scans the
// leading SNPs of its packed record and writes 2 (diploid) on the first heterozygous
// call, else 1 (pseudo-haploid). The on-device twin of the host detector in
// src/io/ploidy_detect.cpp, bit-identical by construction (integer/bit ops via the
// shared core primitives).
//
// Reference: docs/reference/src_device_cuda_detect_ploidy_kernel.cu.md
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
    const int grid_x = core::grid_for_x(
        static_cast<long>(n_individuals), kPloidyBlock,
        "detect-ploidy gridDim.x (individual axis) exceeds kMaxGridX "
        "(architecture.md §7) — tile the individual axis");
    detect_ploidy_kernel<<<static_cast<unsigned>(grid_x), kPloidyBlock, 0, stream>>>(
        d_packed, bytes_per_record, n_individuals, window, d_ploidy);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device

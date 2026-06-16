// src/core/internal/decode_af.hpp
//
// THE shared, per-element genotype-decode + allele-frequency primitive — the
// single source of the decode math so the CPU reference oracle and the GPU
// kernel CANNOT diverge (architecture.md §5 S0/S1, §7, §8, §13; ROADMAP §2 Q/V/N
// contract, M1).
//
// Mirrors core/internal/f2_estimator.hpp's role: thin __host__ __device__ scalar
// functions that are (a) the per-element numerics the GPU kernel calls inside its
// segmented reduction and (b) the exact same numerics the CPU reference calls in
// its scalar loop — so the 2-bit unpack, the raw-value→genotype mapping, the
// missing-handling, and the AC/AN→Q/V/N finalize are identical on both paths.
//
// THE DECODE CONVENTION (verified bit-for-bit against the on-box oracle
// build_tgeno_matrix.py; the corrected M1 facts):
//   * 2-bit code per genotype, RAW VALUE: 0→0 ref-allele copies, 1→1, 2→2,
//     3→MISSING. (NOT the binary mapping 00→0,10→1,11→2,01→missing, which
//     mis-decodes — the raw-value mapping reproduces the oracle, max|Δ|=0.)
//   * Per (population, SNP): over the individuals of that population,
//       AC += code            for each NON-missing individual (code != 3)
//       AN += 1               counting NON-missing INDIVIDUALS (not alleles)
//     Missing individuals are excluded from BOTH AC and AN.
//   * Then, with the per-sample PLOIDY factor `ploidy` (2 diploid / 1 pseudo-
//     haploid — a PARAMETER sourced from metadata, never auto-detected from
//     genotypes; ROADMAP Q/V/N contract):
//       N = ploidy * AN       (non-missing HAPLOID count: 2×non-missing diploids)
//       Q = AC / N            (ref-allele frequency in [0,1]; 0 where AN == 0)
//       V = (AN > 0) ? 1 : 0  (validity mask)
//     For the real AADR data every sample is diploid (ploidy == 2), so every N is
//     even — exactly the oracle's `N = 2*non-missing`, `Q = AC/(2*AN)`.
//
// __host__ __device__ portability: like f2_estimator.hpp, STEPPE_HD expands to the
// CUDA qualifiers under nvcc and to nothing otherwise, so the SAME functions
// compile and unit-test on the CPU and run on the GPU (the point of one shared
// primitive). CUDA-free-compilable (it includes no CUDA header).
#ifndef STEPPE_CORE_INTERNAL_DECODE_AF_HPP
#define STEPPE_CORE_INTERNAL_DECODE_AF_HPP

#include <cstdint>

#if defined(__CUDACC__)
#  define STEPPE_HD __host__ __device__
#else
#  define STEPPE_HD
#endif

namespace steppe::core {

// ===========================================================================
// Per-element decode numerics — the shared primitive.
// ===========================================================================

/// The 2-bit code that denotes a MISSING genotype (RAW-VALUE mapping). Codes
/// 0/1/2 are reference-allele copy counts; code 3 is missing. The single home of
/// the missing sentinel for the decode primitive (mirrors io::kMissingCode, kept
/// here so the core decoder does not depend on the `io` leaf — architecture.md §4
/// layering: core does not depend on io; the value is a domain constant of the
/// 2-bit packing, identical on both sides by construction and pinned by the test).
inline constexpr std::uint8_t kMissingGenotypeCode = 3;

/// Extract the 2-bit code for SNP position `k` (0-based) within a packed byte,
/// MSB-first: position 0 → bits 7-6, 1 → 5-4, 2 → 3-2, 3 → 1-0. This is the
/// `(byte >> (6 - 2*(k mod 4))) & 3` rule. SAME bit order as
/// io::code_in_byte (the host reader) — pinned by the equivalence test; kept here
/// so the device/CPU decoders share ONE bit-extraction with no io dependency.
[[nodiscard]] STEPPE_HD inline std::uint8_t genotype_code(std::uint8_t packed_byte,
                                                          int k) noexcept {
    const int shift = (3 - (k & 3)) * 2;  // 6, 4, 2, 0 for k%4 = 0,1,2,3
    return static_cast<std::uint8_t>((packed_byte >> shift) & 0x3u);
}

/// Whether a 2-bit code is a NON-MISSING genotype (code != 3). Missing genotypes
/// are excluded from BOTH the allele count and the sample count.
[[nodiscard]] STEPPE_HD inline bool genotype_valid(std::uint8_t code) noexcept {
    return code != kMissingGenotypeCode;
}

/// The decoded per-(population, SNP) result: reference-allele frequency Q in
/// [0,1] (0 where no data), the non-missing HAPLOID count N, and the validity V.
/// Plain POD so it crosses the CUDA-free seam unchanged.
struct AfResult {
    double q = 0.0;  ///< ref-allele frequency in [0,1] (0 where an == 0)
    double n = 0.0;  ///< non-missing haploid count = ploidy * an
    double v = 0.0;  ///< validity mask: 1.0 if an > 0, else 0.0
};

/// Finalize Q/V/N for one (population, SNP) from the integer accumulators:
///   `ac` = Σ genotype codes over NON-missing individuals (ref-allele copies),
///   `an` = count of NON-missing INDIVIDUALS,
///   `ploidy` = per-sample haploid multiplier (2 diploid, 1 pseudo-haploid — a
///              metadata parameter, never inferred from genotypes).
/// Yields N = ploidy*an, Q = ac/N (0 where an==0), V = (an>0). This is the EXACT
/// oracle math (`N = 2*an`, `Q = ac/(2*an)`, `V = an>0` for ploidy 2). The single
/// FP64 divide here is what keeps Q exact: AC/AN are integer-accumulated, divided
/// once. Used by BOTH the CPU reference and the GPU kernel.
[[nodiscard]] STEPPE_HD inline AfResult finalize_af(long ac, long an, int ploidy) noexcept {
    AfResult r;
    if (an > 0) {
        const double n = static_cast<double>(ploidy) * static_cast<double>(an);
        r.n = n;
        r.q = static_cast<double>(ac) / n;
        r.v = 1.0;
    } else {
        r.n = 0.0;
        r.q = 0.0;
        r.v = 0.0;
    }
    return r;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_DECODE_AF_HPP

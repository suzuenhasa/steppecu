// src/core/internal/decode_af.hpp
//
// The shared genotype-decode + allele-frequency primitive: unpacks 2-bit genotype
// codes into a per-(population, SNP) reference-allele frequency. Every helper is a
// STEPPE_HD inline so the CPU reference oracle and the GPU kernel call the exact
// same math and cannot drift; several packing constants are kept in sync with the
// io reader by construction (core does not depend on io).
//
// Reference: docs/reference/src_core_internal_decode_af.hpp.md
#ifndef STEPPE_CORE_INTERNAL_DECODE_AF_HPP
#define STEPPE_CORE_INTERNAL_DECODE_AF_HPP

#include <cstdint>

#include "core/internal/host_device.hpp"  // STEPPE_HD

namespace steppe::core {

// Two-bit genotype packing — reference §2
inline constexpr std::uint8_t kMissingGenotypeCode = 3;

inline constexpr int kCodesPerByte = 4;
inline constexpr int kBitsPerCode = 2;

inline constexpr std::uint8_t kCodeMask = static_cast<std::uint8_t>((1u << kBitsPerCode) - 1u);

inline constexpr std::uint8_t kHeterozygousGenotypeCode = 1;

// Ploidy and pseudo-haploid auto-detection — reference §3
inline constexpr int kPloidyDetectSnps = 1000;

inline constexpr int kPloidyPseudoHaploid = 1;
inline constexpr int kPloidyDiploid = 2;

inline constexpr double kPloidyDivisorBase = 3.0;

// Extracting and testing a single code — reference §4
[[nodiscard]] STEPPE_HD inline std::uint8_t genotype_code(std::uint8_t packed_byte,
                                                          int k) noexcept {
    const int shift = (kCodesPerByte - 1 - (k % kCodesPerByte)) * kBitsPerCode;
    return static_cast<std::uint8_t>((packed_byte >> shift) & kCodeMask);
}

[[nodiscard]] STEPPE_HD inline bool genotype_valid(std::uint8_t code) noexcept {
    return code != kMissingGenotypeCode;
}

// The two accumulation folds — reference §5
STEPPE_HD inline void accumulate_genotype(std::uint8_t code,
                                          std::int64_t& ac,
                                          std::int64_t& an) noexcept {
    if (genotype_valid(code)) {
        ac += static_cast<std::int64_t>(code);
        ++an;
    }
}

STEPPE_HD inline void accumulate_genotype_ploidy(std::uint8_t code, int ploidy,
                                                 double& ac,
                                                 std::int64_t& n) noexcept {
    if (genotype_valid(code) && (ploidy == kPloidyDiploid || ploidy == kPloidyPseudoHaploid)) {
        ac += static_cast<double>(code) / (kPloidyDivisorBase - static_cast<double>(ploidy));
        n += ploidy;
    }
}

// AfResult — reference §6
struct AfResult {
    double q = 0.0;
    double n = 0.0;
    double v = 0.0;
};

// Finalizing a frequency — reference §7
[[nodiscard]] STEPPE_HD inline AfResult finalize_af(std::int64_t ac, std::int64_t an,
                                                    int ploidy) noexcept {
    AfResult r;
    if (an > 0 && ploidy > 0) {
        const double n = static_cast<double>(ploidy) * static_cast<double>(an);
        r.n = n;
        r.q = static_cast<double>(ac) / n;
        r.v = 1.0;
    }
    return r;
}

[[nodiscard]] STEPPE_HD inline AfResult finalize_af_counts(double ac,
                                                           std::int64_t n) noexcept {
    AfResult r;
    if (n > 0) {
        const double nd = static_cast<double>(n);
        r.n = nd;
        r.q = ac / nd;
        r.v = 1.0;
    }
    return r;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_DECODE_AF_HPP

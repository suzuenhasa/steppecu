// src/core/internal/king_kinship.hpp
//
// The shared KING-robust kinship primitive: the per-SNP diploid classification fold, the
// per-pair phi estimator, and the KING degree bands. Every helper is a STEPPE_HD inline so
// the GPU pair kernel (classification) and the host finalize + the CPU unit-test oracle
// (phi + degree) call the exact same arithmetic and cannot drift.
//
// Estimator (Manichaikul et al. 2010, KING-robust; == plink2 --make-king-table KINSHIP,
// bit-for-bit verified vs plink2 v2.0.0-a on 29,646 real AADR pairs): over autosomal biallelic
// SNPs where BOTH calls are non-missing ("considered"), with diploid dosage codes ci,cj in
// {0=hom-ref, 1=het, 2=hom-alt}:
//   N_hetHet = #{ci==1 & cj==1}
//   N_IBS0   = #{opposite homozygotes: (ci==0 & cj==2) or (ci==2 & cj==0)}
//   N_het_i  = #{ci==1},  N_het_j = #{cj==1}   (over the considered set)
//   phi = 1/2 + (N_hetHet - 2*N_IBS0 - (N_het_i + N_het_j)/2) / (2*min(N_het_i, N_het_j))
// This is plink2's exact estimator: the numerator carries the -(het_i+het_j)/2 term and the
// denominator is 2*min(het_i, het_j), NOT (het_i + het_j). The two forms COINCIDE only when
// het_i == het_j (e.g. duplicate/MZ -> phi = 0.5); they diverge for the asymmetric het counts
// typical of cross-population pairs, where the naive (hetHet-2*IBS0)/(het_i+het_j) form is
// systematically too NON-negative (a +0.012 bias that flips unrelated pairs to spurious 3rd-degree).
// Unrelated -> phi ~ 0 (can go slightly negative — emit raw); min(N_het_i, N_het_j) == 0 -> phi
// = NaN (undefined). The counts are INVARIANT under REF<->ALT relabeling (het stays het,
// opposite-hom stays opposite), so allele polarity does not bind.
#ifndef STEPPE_CORE_INTERNAL_KING_KINSHIP_HPP
#define STEPPE_CORE_INTERNAL_KING_KINSHIP_HPP

#include <cstdint>

#include "core/internal/decode_af.hpp"    // kMissingGenotypeCode, kHeterozygousGenotypeCode
#include "core/internal/host_device.hpp"  // STEPPE_HD

namespace steppe::core {

// Per-pair KING count accumulator (integer, order-independent -> bit-exact reductions).
struct KingCounts {
    long nsnp = 0;    // considered sites (both calls non-missing)
    long hethet = 0;  // ci==1 & cj==1
    long ibs0 = 0;    // opposite homozygotes
    long het_i = 0;   // ci==1 over the considered set
    long het_j = 0;   // cj==1 over the considered set
};

// KING degree bands (2^-(k+3/2)): the standard between-family relatedness ladder.
enum class KingDegree : int {
    Duplicate = 0,  // phi >= 0.354  (MZ twins / duplicate sample)
    First = 1,      // [0.177, 0.354)
    Second = 2,     // [0.0884, 0.177)
    Third = 3,      // [0.0442, 0.0884)
    Unrelated = 4,  // < 0.0442
    Undefined = 5,  // phi is NaN (no shared het)
};

// Fold one pair of diploid codes into the accumulator. A missing code on EITHER side skips
// the site entirely (it is not "considered"). Codes 0/1/2 are diploid dosages; 1 is the het.
STEPPE_HD inline void king_classify(std::uint8_t ci, std::uint8_t cj, KingCounts& acc) noexcept {
    if (ci == kMissingGenotypeCode || cj == kMissingGenotypeCode) return;
    acc.nsnp += 1;
    const bool hi = (ci == kHeterozygousGenotypeCode);
    const bool hj = (cj == kHeterozygousGenotypeCode);
    if (hi) acc.het_i += 1;
    if (hj) acc.het_j += 1;
    if (hi && hj) acc.hethet += 1;
    // Opposite homozygotes: both non-missing, both homozygous, and different (0 vs 2).
    else if (!hi && !hj && ci != cj) acc.ibs0 += 1;
}

// A portable quiet NaN usable in both host and device code.
[[nodiscard]] STEPPE_HD inline double king_nan() noexcept { return __builtin_nan(""); }

// The KING-robust kinship phi from a pair's counts (plink2 --make-king-table convention).
// phi = 1/2 + (hetHet - 2*IBS0 - (het_i + het_j)/2) / (2*min(het_i, het_j)).
// NaN when EITHER sample shares no heterozygote (min == 0 -> the 2*min denominator vanishes).
[[nodiscard]] STEPPE_HD inline double king_phi(const KingCounts& c) noexcept {
    const long het_min = (c.het_i < c.het_j) ? c.het_i : c.het_j;
    if (het_min <= 0) return king_nan();
    const double num = static_cast<double>(c.hethet) - 2.0 * static_cast<double>(c.ibs0) -
                       0.5 * static_cast<double>(c.het_i + c.het_j);
    return 0.5 + num / (2.0 * static_cast<double>(het_min));
}

// KING degree band from phi (matches the standard 2^-(k+3/2) cut points). A NaN phi
// (undefined) reports Undefined.
[[nodiscard]] STEPPE_HD inline KingDegree degree_from_phi(double phi) noexcept {
    if (!(phi == phi)) return KingDegree::Undefined;  // NaN
    if (phi >= 0.354) return KingDegree::Duplicate;
    if (phi >= 0.177) return KingDegree::First;
    if (phi >= 0.0884) return KingDegree::Second;
    if (phi >= 0.0442) return KingDegree::Third;
    return KingDegree::Unrelated;
}

// The frozen degree label (the emitted `degree` column value).
[[nodiscard]] inline const char* king_degree_label(KingDegree d) noexcept {
    switch (d) {
        case KingDegree::Duplicate: return "dup";
        case KingDegree::First: return "1st";
        case KingDegree::Second: return "2nd";
        case KingDegree::Third: return "3rd";
        case KingDegree::Unrelated: return "unrelated";
        case KingDegree::Undefined: return "undefined";
    }
    return "undefined";
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_KING_KINSHIP_HPP

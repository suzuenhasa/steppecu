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
//       V = (AN > 0 && ploidy > 0) ? 1 : 0  (validity mask; a non-positive ploidy
//                             is misset metadata and masks the cell OUT — see
//                             finalize_af, cleanup B10/X-11)
//     For the real AADR data every sample is diploid (ploidy == 2), so every N is
//     even — exactly the oracle's `N = 2*non-missing`, `Q = AC/(2*AN)`.
//
// __host__ __device__ portability: like f2_estimator.hpp, STEPPE_HD (from the
// single home core/internal/host_device.hpp) expands to the CUDA qualifiers under
// nvcc and to nothing otherwise, so the SAME functions compile and unit-test on
// the CPU and run on the GPU (the point of one shared primitive). CUDA-free-
// compilable (it includes no CUDA header).
#ifndef STEPPE_CORE_INTERNAL_DECODE_AF_HPP
#define STEPPE_CORE_INTERNAL_DECODE_AF_HPP

#include <cstdint>

#include "core/internal/host_device.hpp"  // STEPPE_HD

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

/// The 2-bit packing radix: kCodesPerByte genotype codes per byte, kBitsPerCode bits
/// each (8 = kCodesPerByte·kBitsPerCode). The single home of the packing radix for
/// the core decode primitive — the byte/position split `s/kCodesPerByte`,
/// `s%kCodesPerByte` and the MSB-first shift below derive from these, so the kernel
/// (decode_af_kernel.cu) and the CPU oracle (cpu_backend.cpp) cannot re-pick bare
/// 4/2/3 literals. Mirrors io::kCodesPerByte / io::kBitsPerCode (eigenstrat_format.hpp)
/// by construction, kept here so the core decoder does not depend on the `io` leaf
/// (architecture.md §4 layering: core does not depend on io); the two are pinned equal
/// by the cross-leaf equivalence test (tests/reference/test_decode_equivalence.cu).
inline constexpr int kCodesPerByte = 4;
inline constexpr int kBitsPerCode = 2;

/// Extract the 2-bit code for SNP position `k` (0-based) within a packed byte,
/// MSB-first: position 0 → bits 7-6, 1 → 5-4, 2 → 3-2, 3 → 1-0. This is the
/// `(byte >> (6 - 2*(k mod 4))) & 3` rule. SAME bit order as
/// io::code_in_byte (the host reader) — pinned by the equivalence test; kept here
/// so the device/CPU decoders share ONE bit-extraction with no io dependency.
[[nodiscard]] STEPPE_HD inline std::uint8_t genotype_code(std::uint8_t packed_byte,
                                                          int k) noexcept {
    const int shift = (kCodesPerByte - 1 - (k % kCodesPerByte)) * kBitsPerCode;  // 6,4,2,0
    return static_cast<std::uint8_t>((packed_byte >> shift) & 0x3u);
}

/// Whether a 2-bit code is a NON-MISSING genotype (code != 3). Missing genotypes
/// are excluded from BOTH the allele count and the sample count.
[[nodiscard]] STEPPE_HD inline bool genotype_valid(std::uint8_t code) noexcept {
    return code != kMissingGenotypeCode;
}

/// Fold one individual's 2-bit `code` into the per-(population, SNP) integer
/// accumulators: a NON-MISSING code adds its raw value to `ac` (Σ ref-allele
/// copies) and increments `an` (count of non-missing individuals); a MISSING code
/// is excluded from BOTH. This is THE accumulation convention of the decode
/// reduction's inner step — single-homed here so the GPU kernel's segmented
/// reduction and the CPU oracle's scalar loop CANNOT diverge on it (the last
/// per-element decode semantic to join genotype_code/genotype_valid/finalize_af in
/// the shared primitive; architecture.md §8 single-source, §13; cleanup A-1/B27).
/// `std::int64_t` accumulators (== `long` on the LP64 target) give ample headroom:
/// max AC = ploidy·n_individuals ≪ 2^53 (cleanup E-3 width note).
STEPPE_HD inline void accumulate_genotype(std::uint8_t code,
                                          std::int64_t& ac,
                                          std::int64_t& an) noexcept {
    if (genotype_valid(code)) {
        ac += static_cast<std::int64_t>(code);
        ++an;
    }
}

/// The decoded per-(population, SNP) result: reference-allele frequency Q in
/// [0,1] (0 where no data), the non-missing HAPLOID count N, and the validity V.
/// Plain POD so it crosses the CUDA-free seam unchanged.
struct AfResult {
    double q = 0.0;  ///< ref-allele frequency in [0,1] (0 where an == 0)
    double n = 0.0;  ///< non-missing haploid count = ploidy * an
    double v = 0.0;  ///< validity mask: 1.0 if an > 0 && ploidy > 0, else 0.0
};

/// Finalize Q/V/N for one (population, SNP) from the integer accumulators:
///   `ac` = Σ genotype codes over NON-missing individuals (ref-allele copies),
///   `an` = count of NON-missing INDIVIDUALS,
///   `ploidy` = per-sample haploid multiplier (2 diploid, 1 pseudo-haploid — a
///              metadata parameter, never inferred from genotypes; the only
///              meaningful values are {1, 2}).
/// Yields N = ploidy*an, Q = ac/N (0 where an==0), V = (an>0). This is the EXACT
/// oracle math (`N = 2*an`, `Q = ac/(2*an)`, `V = an>0` for ploidy 2). The single
/// FP64 divide here is what keeps Q exact: AC/AN are integer-accumulated, divided
/// once. Used by BOTH the CPU reference and the GPU kernel.
///
/// PLOIDY VALIDITY (fail-soft, the single home of the Q/V/N contract — cleanup
/// B10/X-11): `ploidy` is folded INTO the validity test, so a non-positive ploidy
/// (`ploidy <= 0` — uninitialized/misset metadata) degrades to the masked-out
/// `{q:0, n:0, v:0}` result, NOT a NaN/Inf-with-validity. Without this guard,
/// `ploidy == 0` with `an > 0` would give `n == 0` and `q = ac/0` — well-defined
/// IEEE-754 (this target is `is_iec559`): `+inf`/`NaN` — while still reporting
/// `v == 1`, so a non-finite Q would slip past the `an > 0` "we have data" branch
/// UNMASKED and poison every f2 GEMM touching that column (architecture.md §2
/// fail-fast: degrade, never silently corrupt). The upstream `io` filter leaf
/// REJECTS a bad ploidy outright (snp_filter.cpp validates `ploidy ∈ {1, 2}` and
/// throws), so the two single-source primitives now AGREE on the illegal-ploidy
/// contract: this device-side per-element primitive — which has no throw path on
/// the GPU and no `ConfigBuilder::build()`-time validation (ploidy lives on the
/// per-call DecodeTileView, not RunConfig) — masks out; the host filter throws.
[[nodiscard]] STEPPE_HD inline AfResult finalize_af(std::int64_t ac, std::int64_t an,
                                                    int ploidy) noexcept {
    // `AfResult r;` already yields the masked-out {q:0, n:0, v:0} via its in-class
    // default member initializers, so the masked-out path needs no else branch — the
    // re-assignment of the same constants was a dead store (DRY/dead-code;
    // NAMING-STYLE-STANDARD §4 dead-store; findings group-7 7.4). Only the
    // has-data arm assigns.
    AfResult r;
    if (an > 0 && ploidy > 0) {
        const double n = static_cast<double>(ploidy) * static_cast<double>(an);
        r.n = n;
        r.q = static_cast<double>(ac) / n;
        r.v = 1.0;
    }
    return r;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_DECODE_AF_HPP

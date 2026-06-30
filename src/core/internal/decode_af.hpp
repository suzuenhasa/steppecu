// src/core/internal/decode_af.hpp
//
// THE shared, per-element genotype-decode + allele-frequency primitive — the single source
// of the decode math (2-bit unpack, raw-value→genotype mapping, missing-handling,
// AC/AN→Q/V/N finalize, AT2 per-sample pseudo-haploid ploidy). Full decode convention +
// the AT2 auto-detection narrative: see docs/design/decode-af.md (architecture.md §5, §13).
//
// PARITY INVARIANT: these __host__ __device__ scalars are the SINGLE source so the CPU
// reference oracle and the GPU kernel CANNOT diverge — verified bit-for-bit against the
// on-box oracle (build_tgeno_matrix.py), max|Δ|=0.
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

/// The 2-bit code-extraction mask, DERIVED from the packing radix kBitsPerCode (the
/// low kBitsPerCode bits set: (1 << 2) - 1 == 0x3 == 3). Single-homed alongside the
/// radix so the mask and the shift in genotype_code below share ONE source — the
/// kernel/oracle cannot re-pick a bare `0x3` while the shift derives from the named
/// radix (value 0x3 unchanged; NAMING-STYLE-STANDARD §3.3 derive-from-named-radix).
inline constexpr std::uint8_t kCodeMask = static_cast<std::uint8_t>((1u << kBitsPerCode) - 1u);

/// The 2-bit code that denotes a HETEROZYGOUS genotype (RAW-VALUE mapping: 1
/// reference-allele copy). Used by the AT2 pseudo-haploid auto-detection: a sample
/// is diploid iff it has at least one het call in the detection prefix (a haploid
/// genome can never be heterozygous). The single home of the het sentinel.
inline constexpr std::uint8_t kHeterozygousGenotypeCode = 1;

/// AT2 pseudo-haploid auto-detection window (adjust_pseudohaploid=TRUE; admixtools
/// cpp_*_ploidy default ntest = 1000): a sample is classified by scanning its FIRST
/// kPloidyDetectSnps SNPs for any heterozygous call. The single home of the AT2
/// detection-window constant (geno_reader / the oracle build_tgeno_matrix.py mirror
/// it). A SNP count below this just scans fewer SNPs (the whole record).
inline constexpr int kPloidyDetectSnps = 1000;

/// Per-sample ploidy values (the only meaningful ones; AT2 emits {1, 2}).
inline constexpr int kPloidyPseudoHaploid = 1;  ///< pseudo-haploid (ancient DNA): N contributes 1
inline constexpr int kPloidyDiploid = 2;        ///< diploid (modern / het-bearing): N contributes 2

/// AT2's haploidization-divisor base in `val / (3.0 - ploidy(i))` (admixtools 2.0.10
/// src/cpp_readgeno.cpp cpp_*_to_afs). PARITY-FROZEN literal: name-only, value MUST stay
/// 3.0 (NAMING-STYLE-STANDARD §3.2 AT2 literals — name only, never change value). FLOAT
/// is load-bearing: ploidy 1 ⇒ 3.0-1 == 2.0 ⇒ code/2.0 == 0.5 for an out-of-window het
/// (an integer divide would drop it — see accumulate_genotype_ploidy). Single-homed here.
inline constexpr double kPloidyDivisorBase = 3.0;

/// Extract the 2-bit code for SNP position `k` (0-based) within a packed byte,
/// MSB-first: position 0 → bits 7-6, 1 → 5-4, 2 → 3-2, 3 → 1-0. This is the
/// `(byte >> (6 - 2*(k mod 4))) & 3` rule. SAME bit order as
/// io::code_in_byte (the host reader) — pinned by the equivalence test; kept here
/// so the device/CPU decoders share ONE bit-extraction with no io dependency.
[[nodiscard]] STEPPE_HD inline std::uint8_t genotype_code(std::uint8_t packed_byte,
                                                          int k) noexcept {
    const int shift = (kCodesPerByte - 1 - (k % kCodesPerByte)) * kBitsPerCode;  // 6,4,2,0
    return static_cast<std::uint8_t>((packed_byte >> shift) & kCodeMask);
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

/// PER-SAMPLE-PLOIDY fold — AT2's adjust_pseudohaploid accumulation, BYTE-FAITHFUL
/// to admixtools 2.0.10 src/cpp_readgeno.cpp:
///   altalleles(pop)      += val / (3.0 - ploidy(i));   // NOTE: FLOAT 3.0
///   observedalleles(pop) += ploidy(i);
/// Folds one individual's 2-bit `code` and its per-sample `ploidy` (1 pseudo-haploid
/// / 2 diploid) into:
///   `ac` (DOUBLE) += code / (3.0 - ploidy)   — the AT2-WEIGHTED ref-allele count.
///   `n`  (int)    += ploidy                  — the per-sample-summed HAPLOID count.
///
/// THE FLOAT IS LOAD-BEARING (not an integer divide): a sample CLASSIFIED pseudo-
/// haploid (no het in the first kPloidyDetectSnps SNPs) can STILL carry a het call
/// (code == 1) at a SNP OUTSIDE that detection window. For such a SNP, ploidy == 1 ⇒
/// 3.0 - 1 == 2.0 ⇒ code/2.0 == 0.5 — AT2 adds 0.5 (the het contributes half a ref
/// allele to the haploidized count). An INTEGER `1/2 == 0` would silently drop it,
/// diverging from AT2 by exactly that 0.5 (the measured Q max|Δ| == 0.004 = 1/250
/// before this fix). 0.5-multiples are exact in double, so the sum stays exact and Q
/// is still the single exact divide AC/N in finalize_af_counts — no precision loss,
/// just the AT2-correct value. For ploidy == 2, code/1.0 == code (integer-valued
/// double), so an ALL-DIPLOID segment is BIT-IDENTICAL to the legacy
/// accumulate_genotype + `N = 2·AN`.
///
/// A MISSING code (code == 3) is excluded from BOTH. A ploidy outside {1,2} (misset
/// per-sample metadata) is excluded from BOTH (fail-soft: it contributes nothing
/// rather than dividing by 3-ploidy ≤ 0 / fabricating a count) — mirrors the
/// finalize ploidy guard. This is THE per-element accumulation for the auto-ploidy
/// decode, shared by the GPU kernel and the CPU oracle (architecture.md §8/§13).
STEPPE_HD inline void accumulate_genotype_ploidy(std::uint8_t code, int ploidy,
                                                 double& ac,
                                                 std::int64_t& n) noexcept {
    if (genotype_valid(code) && (ploidy == kPloidyDiploid || ploidy == kPloidyPseudoHaploid)) {
        ac += static_cast<double>(code) / (kPloidyDivisorBase - static_cast<double>(ploidy));
        n += ploidy;
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

/// Finalize Q/V/N from the PER-SAMPLE-PLOIDY accumulators (accumulate_genotype_ploidy):
///   `ac` = Σ_i code_i/(3.0-ploidy_i) (the AT2-weighted ref-allele count, DOUBLE —
///                                     0.5-multiples from PH het calls are exact),
///   `n`  = Σ_i ploidy_i              (the per-sample-summed haploid count, integer).
/// Yields N = n, Q = ac/n (0 where n==0), V = (n>0). The ploidy is ALREADY folded
/// into `ac`/`n` by accumulate_genotype_ploidy, so this just does the single FP64
/// divide AC/N — the analogue of finalize_af for the per-sample path. An all-diploid
/// segment gives n = 2·non-missing and ac = Σcode (integer-valued double), so this is
/// BIT-IDENTICAL to finalize_af(ac, an, 2) on the modern-data path. Used by BOTH the
/// CPU reference and the GPU kernel on the auto-ploidy decode.
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

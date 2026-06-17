// tests/unit/test_decode.cpp
//
// Host-only unit test of the SHARED genotype-decode / allele-frequency primitive
// (architecture.md §13 "Unit tests"; ROADMAP §5; cleanup B10/T-1). Pure C++ TU,
// NO GPU and NO data: it proves the `__host__ __device__` per-element primitives
// in core/internal/decode_af.hpp (genotype_code, genotype_valid, finalize_af)
// compile and are correct on the host path (STEPPE_HD expands to empty under the
// host compiler), so the same functions the GPU decode kernel and the CPU oracle
// share are independently pinned — the divergence-prevention thesis the header
// rests on, tested directly rather than only through the GPU/4GB-data equivalence
// .cu (the §13 gap this closes; sibling test_f2.cpp does the same for f2_estimator).
//
// THE B10 VERDICT GATE (cleanup B10/X-11 — ploidy folded into finalize_af
// validity): finalize_af must produce
//   * ploidy == 0  ->  MASKED OUT  {q:0, n:0, v:0}   (degrade, never NaN/Inf-with-validity)
//   * ploidy == 1  ->  HAPLOID     N == 1 * AN        (1x)
//   * ploidy == 2  ->  DIPLOID     N == 2 * AN        (2x)
// plus the existing an==0 masked-out boundary and the negative-ploidy mask-out.
//
// Dual harness (build contract, matches test_f2.cpp): with -DSTEPPE_TEST_WITH_GTEST
// it uses GoogleTest; otherwise it is a self-checking main() that returns non-zero
// on the first failure — all CTest needs to gate (tests/CMakeLists.txt). No CUDA.
#include <cstdio>

#include "core/internal/decode_af.hpp"  // genotype_code, genotype_valid, finalize_af, AfResult, kMissingGenotypeCode

namespace {

using steppe::core::AfResult;
using steppe::core::finalize_af;
using steppe::core::genotype_code;
using steppe::core::genotype_valid;
using steppe::core::kMissingGenotypeCode;

// All finalize_af arithmetic here is EXACTLY representable in double (small integer
// AC/AN, ploidy in {1,2}; the single divide is exact for these operands), so the
// equality checks are exact — no tolerance. q == AC/N is asserted only where the
// quotient is exact in binary (e.g. 3/6 == 0.5), never on a repeating fraction.

// ---- genotype_code: the 2-bit MSB-first unpack, all four positions -----------
// Byte 0b11100100 = positions [k0=11=3, k1=10=2, k2=01=1, k3=00=0] (MSB-first).
[[nodiscard]] bool test_genotype_code_positions() {
    const std::uint8_t b = 0b11100100u;
    return genotype_code(b, 0) == 3u && genotype_code(b, 1) == 2u &&
           genotype_code(b, 2) == 1u && genotype_code(b, 3) == 0u;
}

// k is taken mod 4 (the within-byte index), so position 4 aliases position 0.
[[nodiscard]] bool test_genotype_code_mod4() {
    const std::uint8_t b = 0b11100100u;
    return genotype_code(b, 4) == genotype_code(b, 0) &&
           genotype_code(b, 7) == genotype_code(b, 3);
}

// ---- genotype_valid: raw-value mapping 0/1/2 valid, 3 == MISSING -------------
[[nodiscard]] bool test_genotype_valid_mapping() {
    return genotype_valid(0) && genotype_valid(1) && genotype_valid(2) &&
           !genotype_valid(kMissingGenotypeCode) && kMissingGenotypeCode == 3u;
}

// ---- finalize_af: the B10 verdict gate ---------------------------------------

// ploidy == 0 with an > 0 -> MASKED OUT {0,0,0}. Pre-B10 this gave n==0 and
// q = ac/0 == +inf (a non-finite Q reported with v==1, UNMASKED). The fold of
// `ploidy > 0` into the validity test degrades it to a total, pure masked cell.
[[nodiscard]] bool test_finalize_ploidy0_masked() {
    const AfResult r = finalize_af(/*ac=*/4, /*an=*/3, /*ploidy=*/0);
    return r.q == 0.0 && r.n == 0.0 && r.v == 0.0;
}

// ploidy == 0 with ac == 0 (pre-B10: q = 0/0 == NaN, v==1) -> also masked out.
// (Asserts the masked-out branch with the would-be-NaN operands, not just inf.)
[[nodiscard]] bool test_finalize_ploidy0_zeroac_masked() {
    const AfResult r = finalize_af(/*ac=*/0, /*an=*/5, /*ploidy=*/0);
    return r.q == 0.0 && r.n == 0.0 && r.v == 0.0;
}

// Negative ploidy is also misset metadata -> masked out (would otherwise give a
// negative N and a wrong-signed Q with v==1).
[[nodiscard]] bool test_finalize_negative_ploidy_masked() {
    const AfResult r = finalize_af(/*ac=*/4, /*an=*/3, /*ploidy=*/-2);
    return r.q == 0.0 && r.n == 0.0 && r.v == 0.0;
}

// ploidy == 1 -> HAPLOID: N == 1 * AN (1x). With ac=2, an=4: N=4, Q=0.5, V=1.
[[nodiscard]] bool test_finalize_ploidy1_haploid() {
    const AfResult r = finalize_af(/*ac=*/2, /*an=*/4, /*ploidy=*/1);
    return r.n == 4.0 && r.q == 0.5 && r.v == 1.0;
}

// ploidy == 2 -> DIPLOID: N == 2 * AN (2x). With ac=3, an=3: N=6, Q=0.5, V=1.
[[nodiscard]] bool test_finalize_ploidy2_diploid() {
    const AfResult r = finalize_af(/*ac=*/3, /*an=*/3, /*ploidy=*/2);
    return r.n == 6.0 && r.q == 0.5 && r.v == 1.0;
}

// The 1x-vs-2x relation pinned directly on the SAME (ac, an): N doubles from
// ploidy 1 to ploidy 2, and Q halves (the oracle's N = 2*non-missing for diploid).
[[nodiscard]] bool test_finalize_haploid_vs_diploid_factor() {
    const AfResult h = finalize_af(/*ac=*/4, /*an=*/5, /*ploidy=*/1);  // N=5,  Q=0.8
    const AfResult d = finalize_af(/*ac=*/4, /*an=*/5, /*ploidy=*/2);  // N=10, Q=0.4
    return h.n == 5.0 && d.n == 10.0 && d.n == 2.0 * h.n &&
           h.q == 0.8 && d.q == 0.4 && h.v == 1.0 && d.v == 1.0;
}

// an == 0 (whole-population-missing SNP) -> masked out regardless of ploidy. The
// normal, expected domain outcome; {0,0,0} is the §5-S2 representation the GEMM
// consumes. (Also confirms the guard is `an > 0 AND ploidy > 0`, not one or other.)
[[nodiscard]] bool test_finalize_an0_masked() {
    const AfResult d = finalize_af(/*ac=*/0, /*an=*/0, /*ploidy=*/2);
    const AfResult h = finalize_af(/*ac=*/0, /*an=*/0, /*ploidy=*/1);
    return d.q == 0.0 && d.n == 0.0 && d.v == 0.0 &&
           h.q == 0.0 && h.n == 0.0 && h.v == 0.0;
}

// an == 1: a single non-missing individual still produces a valid cell.
[[nodiscard]] bool test_finalize_an1_valid() {
    const AfResult r = finalize_af(/*ac=*/2, /*an=*/1, /*ploidy=*/2);  // N=2, Q=1.0
    return r.n == 2.0 && r.q == 1.0 && r.v == 1.0;
}

struct Case {
    const char* name;
    bool (*fn)();
};

constexpr Case kCases[] = {
    {"genotype_code MSB-first positions 0..3", test_genotype_code_positions},
    {"genotype_code k taken mod 4", test_genotype_code_mod4},
    {"genotype_valid raw-value mapping (3 == missing)", test_genotype_valid_mapping},
    {"finalize_af ploidy=0 -> masked {0,0,0}", test_finalize_ploidy0_masked},
    {"finalize_af ploidy=0, ac=0 -> masked (no NaN)", test_finalize_ploidy0_zeroac_masked},
    {"finalize_af negative ploidy -> masked {0,0,0}", test_finalize_negative_ploidy_masked},
    {"finalize_af ploidy=1 -> haploid (N == 1*AN)", test_finalize_ploidy1_haploid},
    {"finalize_af ploidy=2 -> diploid (N == 2*AN)", test_finalize_ploidy2_diploid},
    {"finalize_af haploid vs diploid 1x/2x factor", test_finalize_haploid_vs_diploid_factor},
    {"finalize_af an=0 -> masked {0,0,0}", test_finalize_an0_masked},
    {"finalize_af an=1 -> valid cell", test_finalize_an1_valid},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(DecodeAf, AllPrimitives) {
    for (const auto& c : kCases) {
        EXPECT_TRUE(c.fn()) << "failed: " << c.name;
    }
}
#else
int main() {
    int failures = 0;
    for (const auto& c : kCases) {
        const bool ok = c.fn();
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (!ok) ++failures;
    }
    if (failures != 0) {
        std::fprintf(stderr, "test_decode: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_decode: all %zu decode-primitive checks PASS\n",
                sizeof(kCases) / sizeof(kCases[0]));
    return 0;
}
#endif

// tests/unit/test_decode.cpp
//
// Host-only unit test of the SHARED genotype-decode / allele-frequency primitive
// (architecture.md §13 "Unit tests"; ROADMAP §5; cleanup B10/T-1). Pure C++ TU,
// NO GPU and NO data: it proves the `__host__ __device__` per-element primitives
// in core/internal/decode_af.hpp (genotype_code, genotype_valid,
// accumulate_genotype, finalize_af) compile and are correct on the host path
// (STEPPE_HD expands to empty under the host compiler), so the same functions the
// GPU decode kernel and the CPU oracle share are independently pinned — the
// divergence-prevention thesis the header rests on, tested directly rather than
// only through the GPU/4GB-data equivalence .cu (the §13 gap this closes; sibling
// test_f2.cpp does the same for f2_estimator).
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
#include <cstdint>
#include <cstdio>

#include "core/internal/decode_af.hpp"  // genotype_code, genotype_valid, accumulate_genotype, finalize_af, AfResult, kMissingGenotypeCode

namespace {

using steppe::core::accumulate_genotype;
using steppe::core::accumulate_genotype_ploidy;
using steppe::core::AfResult;
using steppe::core::finalize_af;
using steppe::core::finalize_af_counts;
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

// ---- accumulate_genotype: the AC/AN inner-step convention (A-1/B27) ----------
// The shared accumulation step both decode loops now call: a NON-MISSING code
// adds its raw value to ac and bumps an by 1; a MISSING code touches neither. This
// is the last per-element decode semantic that used to be hand-duplicated in the
// GPU kernel and the CPU oracle — pinning it here closes that divergence surface.

// A valid code adds its raw value to ac and increments an by exactly 1.
[[nodiscard]] bool test_accumulate_valid_adds_code() {
    std::int64_t ac = 0;
    std::int64_t an = 0;
    accumulate_genotype(2u, ac, an);
    return ac == 2 && an == 1;
}

// code 0 is VALID (a homozygous-ref individual): an increments, ac stays 0. Pins
// that "missing" is the sentinel 3, not "code == 0".
[[nodiscard]] bool test_accumulate_code0_is_valid() {
    std::int64_t ac = 7;  // pre-loaded to prove ac is unchanged by a 0-code
    std::int64_t an = 0;
    accumulate_genotype(0u, ac, an);
    return ac == 7 && an == 1;
}

// A MISSING code (3) is excluded from BOTH accumulators — neither moves.
[[nodiscard]] bool test_accumulate_missing_excluded() {
    std::int64_t ac = 5;
    std::int64_t an = 4;
    accumulate_genotype(kMissingGenotypeCode, ac, an);
    return ac == 5 && an == 4;
}

// Running accumulation over a segment matches the convention AC = Σ valid codes,
// AN = count of valid: codes {0,1,2,3,2} -> ac = 0+1+2+2 = 5, an = 4 (the 3 skipped).
[[nodiscard]] bool test_accumulate_segment_running() {
    const std::uint8_t seg[] = {0u, 1u, 2u, kMissingGenotypeCode, 2u};
    std::int64_t ac = 0;
    std::int64_t an = 0;
    for (const std::uint8_t code : seg) accumulate_genotype(code, ac, an);
    return ac == 5 && an == 4;
}

// The whole inner step composes to the SAME AfResult as feeding the convention's
// AC/AN straight into finalize_af — accumulate then finalize == the decode contract.
[[nodiscard]] bool test_accumulate_then_finalize() {
    const std::uint8_t seg[] = {2u, 2u, kMissingGenotypeCode, 1u};  // ac=5, an=3
    std::int64_t ac = 0;
    std::int64_t an = 0;
    for (const std::uint8_t code : seg) accumulate_genotype(code, ac, an);
    const AfResult got = finalize_af(ac, an, /*ploidy=*/2);
    const AfResult want = finalize_af(/*ac=*/5, /*an=*/3, /*ploidy=*/2);  // N=6, Q=5/6
    return ac == 5 && an == 3 && got.q == want.q && got.n == want.n && got.v == want.v;
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

// ---- accumulate_genotype_ploidy: AT2 adjust_pseudohaploid per-sample fold ------
// AT2 (cpp_readgeno.cpp): AC += code/(3-ploidy); N += ploidy; missing/bad-ploidy
// excluded from BOTH. This is the per-sample pseudo-haploid f2 fix.

// DIPLOID (ploidy 2): code/(3.0-2)=code, N += 2. So {0,1,2} fold to ac=3, n=6 — the
// SAME ac as the legacy accumulate_genotype, and N = 2·(non-missing count).
[[nodiscard]] bool test_accumulate_ploidy_diploid_matches_legacy() {
    const std::uint8_t seg[] = {0u, 1u, 2u, kMissingGenotypeCode};  // 3 valid
    double ac = 0.0; std::int64_t n = 0;
    for (std::uint8_t c : seg) accumulate_genotype_ploidy(c, /*ploidy=*/2, ac, n);
    // Legacy: ac=0+1+2=3, an=3 ⇒ N=2·3=6. Per-sample diploid must match exactly.
    std::int64_t lac = 0, lan = 0;
    for (std::uint8_t c : seg) accumulate_genotype(c, lac, lan);
    return ac == 3.0 && n == 6 && ac == static_cast<double>(lac) && n == 2 * lan;
}

// PSEUDO-HAPLOID (ploidy 1) homozygous: code∈{0,2}, code/(3.0-1)=code/2 ∈{0,1}, N += 1.
// {0,2,2,missing} ⇒ ac = 0+1+1 = 2, n = 3 (3 non-missing × 1).
[[nodiscard]] bool test_accumulate_ploidy_pseudohaploid() {
    const std::uint8_t seg[] = {0u, 2u, 2u, kMissingGenotypeCode};
    double ac = 0.0; std::int64_t n = 0;
    for (std::uint8_t c : seg) accumulate_genotype_ploidy(c, /*ploidy=*/1, ac, n);
    return ac == 2.0 && n == 3;  // Q = 2/3
}

// PSEUDO-HAPLOID with an OUT-OF-WINDOW het (code 1): AT2 adds code/2.0 = 0.5 (the
// FLOAT divide is load-bearing — an integer 1/2 would drop it, the Q=0.004 bug).
// {2,1} at ploidy 1 ⇒ ac = 1 + 0.5 = 1.5, n = 2 ⇒ Q = 0.75.
[[nodiscard]] bool test_accumulate_ploidy_pseudohaploid_het_half() {
    double ac = 0.0; std::int64_t n = 0;
    accumulate_genotype_ploidy(2u, /*ploidy=*/1, ac, n);  // ref/ref ⇒ +1.0
    accumulate_genotype_ploidy(1u, /*ploidy=*/1, ac, n);  // stray het ⇒ +0.5 (NOT 0)
    return ac == 1.5 && n == 2;  // Q = 1.5/2 = 0.75
}

// A MISSING code is excluded from BOTH accumulators regardless of ploidy.
[[nodiscard]] bool test_accumulate_ploidy_missing_excluded() {
    double ac = 5.0; std::int64_t n = 4;
    accumulate_genotype_ploidy(kMissingGenotypeCode, /*ploidy=*/2, ac, n);
    accumulate_genotype_ploidy(kMissingGenotypeCode, /*ploidy=*/1, ac, n);
    return ac == 5.0 && n == 4;
}

// A bad ploidy (∉ {1,2}) is fail-soft: the sample contributes NOTHING (no divide by
// 3.0-ploidy ≤ 0, no fabricated count) — mirrors the finalize ploidy guard.
[[nodiscard]] bool test_accumulate_ploidy_bad_excluded() {
    double ac = 0.0; std::int64_t n = 0;
    accumulate_genotype_ploidy(2u, /*ploidy=*/0, ac, n);
    accumulate_genotype_ploidy(2u, /*ploidy=*/3, ac, n);
    accumulate_genotype_ploidy(2u, /*ploidy=*/-1, ac, n);
    return ac == 0.0 && n == 0;
}

// MIXED-PLOIDY pop (the real aDNA case): one diploid (code 1) + one pseudo-haploid
// (code 2). AC = 1/(3.0-2) + 2/(3.0-1) = 1 + 1 = 2; N = 2 + 1 = 3 ⇒ Q = 2/3. This is
// what the per-pop ploidy tag could NOT express.
[[nodiscard]] bool test_accumulate_ploidy_mixed_pop() {
    double ac = 0.0; std::int64_t n = 0;
    accumulate_genotype_ploidy(1u, /*ploidy=*/2, ac, n);  // diploid het
    accumulate_genotype_ploidy(2u, /*ploidy=*/1, ac, n);  // pseudo-haploid ref/ref
    const AfResult r = finalize_af_counts(ac, n);
    return ac == 2.0 && n == 3 && r.n == 3.0 && r.v == 1.0 &&
           (r.q > 0.6666 && r.q < 0.6667);  // 2/3
}

// ---- finalize_af_counts: divide the already-ploidy-weighted AC/N ---------------

// finalize_af_counts(ac, n) == finalize_af(ac, an, 2) when n == 2·an (the diploid
// equivalence the modern-data bit-identity rests on).
[[nodiscard]] bool test_finalize_counts_matches_diploid() {
    const AfResult c = finalize_af_counts(/*ac=*/3.0, /*n=*/6);
    const AfResult d = finalize_af(/*ac=*/3, /*an=*/3, /*ploidy=*/2);  // N=6
    return c.q == d.q && c.n == d.n && c.v == d.v && c.n == 6.0 && c.q == 0.5;
}

// n == 0 (whole-pop-missing) ⇒ masked {0,0,0}, no divide-by-zero.
[[nodiscard]] bool test_finalize_counts_n0_masked() {
    const AfResult r = finalize_af_counts(/*ac=*/0.0, /*n=*/0);
    return r.q == 0.0 && r.n == 0.0 && r.v == 0.0;
}

// A single pseudo-haploid sample: ac=1, n=1 ⇒ N=1 (ODD — the convention that was
// impossible under the all-diploid 2×count rule), Q=1.0, V=1.
[[nodiscard]] bool test_finalize_counts_single_pseudohaploid() {
    const AfResult r = finalize_af_counts(/*ac=*/1.0, /*n=*/1);
    return r.n == 1.0 && r.q == 1.0 && r.v == 1.0;
}

struct Case {
    const char* name;
    bool (*fn)();
};

constexpr Case kCases[] = {
    {"genotype_code MSB-first positions 0..3", test_genotype_code_positions},
    {"genotype_code k taken mod 4", test_genotype_code_mod4},
    {"genotype_valid raw-value mapping (3 == missing)", test_genotype_valid_mapping},
    {"accumulate_genotype valid code adds raw value", test_accumulate_valid_adds_code},
    {"accumulate_genotype code 0 is valid (an++, ac stays)", test_accumulate_code0_is_valid},
    {"accumulate_genotype missing excluded from ac and an", test_accumulate_missing_excluded},
    {"accumulate_genotype running segment AC/AN convention", test_accumulate_segment_running},
    {"accumulate_genotype then finalize == decode contract", test_accumulate_then_finalize},
    {"finalize_af ploidy=0 -> masked {0,0,0}", test_finalize_ploidy0_masked},
    {"finalize_af ploidy=0, ac=0 -> masked (no NaN)", test_finalize_ploidy0_zeroac_masked},
    {"finalize_af negative ploidy -> masked {0,0,0}", test_finalize_negative_ploidy_masked},
    {"finalize_af ploidy=1 -> haploid (N == 1*AN)", test_finalize_ploidy1_haploid},
    {"finalize_af ploidy=2 -> diploid (N == 2*AN)", test_finalize_ploidy2_diploid},
    {"finalize_af haploid vs diploid 1x/2x factor", test_finalize_haploid_vs_diploid_factor},
    {"finalize_af an=0 -> masked {0,0,0}", test_finalize_an0_masked},
    {"finalize_af an=1 -> valid cell", test_finalize_an1_valid},
    {"accumulate_genotype_ploidy diploid == legacy", test_accumulate_ploidy_diploid_matches_legacy},
    {"accumulate_genotype_ploidy pseudo-haploid AC/N", test_accumulate_ploidy_pseudohaploid},
    {"accumulate_genotype_ploidy PH out-of-window het = +0.5", test_accumulate_ploidy_pseudohaploid_het_half},
    {"accumulate_genotype_ploidy missing excluded", test_accumulate_ploidy_missing_excluded},
    {"accumulate_genotype_ploidy bad ploidy fail-soft", test_accumulate_ploidy_bad_excluded},
    {"accumulate_genotype_ploidy mixed-ploidy pop", test_accumulate_ploidy_mixed_pop},
    {"finalize_af_counts == diploid finalize_af", test_finalize_counts_matches_diploid},
    {"finalize_af_counts n=0 -> masked", test_finalize_counts_n0_masked},
    {"finalize_af_counts single pseudo-haploid (N=1 odd)", test_finalize_counts_single_pseudohaploid},
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

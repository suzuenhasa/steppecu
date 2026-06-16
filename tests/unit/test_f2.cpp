// tests/unit/test_f2.cpp
//
// Host-only unit test of the SHARED f2 estimator primitive (architecture.md §13
// "Unit tests"; ROADMAP §5). Pure C++ TU, NO GPU: it proves the
// `__host__ __device__` per-element primitives in core/internal/f2_estimator.hpp
// compile and are correct on the host path (STEPPE_HD expands to empty under the
// host compiler), so the same functions the GPU feeder and the CPU oracle share
// are independently pinned. These are the per-element numerics, not the structure
// of the hot path (architecture.md §7).
//
// Dual harness (build contract): with -DSTEPPE_TEST_WITH_GTEST it uses GoogleTest;
// otherwise it is a self-checking main() that returns non-zero on the first
// failure — all CTest needs to gate (tests/CMakeLists.txt). No CUDA here.
#include <cmath>
#include <cstdio>

#include "core/internal/f2_estimator.hpp"  // het_correction, f2_term, assemble_f2_numerator, finalize_f2, cdiv, grid_for
#include "steppe/config.hpp"               // kHetCorrDenomFloor, kCdivBlock

namespace {

using steppe::core::assemble_f2_numerator;
using steppe::core::cdiv;
using steppe::core::f2_term;
using steppe::core::finalize_f2;
using steppe::core::grid_for;
using steppe::core::het_correction;

// Relative-closeness check for the f2 algebra (the only tolerance this host test
// needs; all arithmetic is exact double here, but the difference-of-products
// assemble can carry a few ULP). 1e-12 is generous for these small magnitudes.
constexpr double kEps = 1e-12;

[[nodiscard]] bool close(double a, double b, double eps = kEps) {
    const double denom = std::fmax(std::fabs(b), 1.0);
    return std::fabs(a - b) <= eps * denom;
}

// ---- het_correction --------------------------------------------------------
// Invalid entry contributes nothing (carries its own validity bit).
[[nodiscard]] bool test_het_invalid() {
    return het_correction(0.5, 100.0, /*valid=*/false) == 0.0;
}

// Valid: hc = q(1-q)/max(N-1, floor). For q=0.5, N=101 -> 0.25/100 = 0.0025.
[[nodiscard]] bool test_het_valid() {
    const double hc = het_correction(0.5, 101.0, /*valid=*/true);
    return close(hc, 0.25 / 100.0);
}

// Denominator floor (the `1` in AT2's max(N-1,1)): with a single non-missing
// haploid (N=1), N-1=0 would divide by zero; the floor kHetCorrDenomFloor
// rescues it -> hc = q(1-q)/floor.
[[nodiscard]] bool test_het_denom_floor() {
    const double hc = het_correction(0.5, 1.0, /*valid=*/true);
    return close(hc, (0.25) / steppe::kHetCorrDenomFloor);
}

// ---- f2_term: cancellation-free per-SNP summand ----------------------------
// (p_i - p_j)^2 - hc_i - hc_j. With p_i=0.7, p_j=0.2, hc_i=0.01, hc_j=0.02:
// 0.5^2 - 0.03 = 0.25 - 0.03 = 0.22.
[[nodiscard]] bool test_f2_term() {
    return close(f2_term(0.7, 0.2, 0.01, 0.02), 0.22);
}

// f2_term is symmetric in the squared difference: term(a,b,.,.) == term(b,a,.,.).
[[nodiscard]] bool test_f2_term_symmetry() {
    return close(f2_term(0.7, 0.2, 0.01, 0.02), f2_term(0.2, 0.7, 0.02, 0.01));
}

// ---- assemble_f2_numerator: expanded GEMM form == cancellation-free form ----
// The GPU builds sumsq_i=Σp_i², sumsq_j=Σp_j², cross=Σp_i p_j, hsum_i=Σhc_i,
// hsum_j=Σhc_j and assembles Σp_i² + Σp_j² - 2Σp_i p_j - Σhc_i - Σhc_j. Over a
// few SNPs this must equal Σ[(p_i-p_j)² - hc_i - hc_j] (the oracle's sum of
// f2_term). This is the load-bearing equivalence the GPU/CPU agreement rests on.
[[nodiscard]] bool test_assemble_matches_term_sum() {
    const double pi[] = {0.7, 0.3, 0.9};
    const double pj[] = {0.2, 0.5, 0.1};
    const double hci[] = {0.01, 0.02, 0.005};
    const double hcj[] = {0.02, 0.01, 0.03};

    double sumsq_i = 0.0, sumsq_j = 0.0, cross = 0.0, hsum_i = 0.0, hsum_j = 0.0;
    double term_sum = 0.0;
    for (int s = 0; s < 3; ++s) {
        sumsq_i += pi[s] * pi[s];
        sumsq_j += pj[s] * pj[s];
        cross += pi[s] * pj[s];
        hsum_i += hci[s];
        hsum_j += hcj[s];
        term_sum += f2_term(pi[s], pj[s], hci[s], hcj[s]);
    }
    const double num = assemble_f2_numerator(sumsq_i, sumsq_j, cross, hsum_i, hsum_j);
    return close(num, term_sum);
}

// ---- finalize_f2: divide with the Vpair==0 guard ---------------------------
[[nodiscard]] bool test_finalize_divides() {
    return close(finalize_f2(2.2, 10.0), 0.22);
}

[[nodiscard]] bool test_finalize_vpair_zero_guard() {
    // Vpair==0 -> 0 (no SNP jointly valid; AT2 convention). Numerator is also 0
    // in practice, but the guard must return 0 regardless to avoid 0/0 = NaN.
    return finalize_f2(0.0, 0.0) == 0.0 && finalize_f2(5.0, 0.0) == 0.0;
}

// ---- launch-config helpers (cdiv / grid_for) -------------------------------
[[nodiscard]] bool test_cdiv() {
    return cdiv(10, 4) == 3 && cdiv(8, 4) == 2 && cdiv(0, 4) == 0 && cdiv(1, 4) == 1;
}

[[nodiscard]] bool test_grid_for() {
    // grid_for(n) == cdiv(n, kCdivBlock); covering n elements with kCdivBlock tiles.
    return grid_for(50) == cdiv(50, steppe::kCdivBlock) &&
           grid_for(steppe::kCdivBlock) == 1;
}

struct Case {
    const char* name;
    bool (*fn)();
};

constexpr Case kCases[] = {
    {"het_correction invalid -> 0", test_het_invalid},
    {"het_correction valid", test_het_valid},
    {"het_correction denom floor", test_het_denom_floor},
    {"f2_term cancellation-free summand", test_f2_term},
    {"f2_term squared-difference symmetry", test_f2_term_symmetry},
    {"assemble (expanded) == sum of f2_term", test_assemble_matches_term_sum},
    {"finalize_f2 divides by Vpair", test_finalize_divides},
    {"finalize_f2 Vpair==0 guard", test_finalize_vpair_zero_guard},
    {"cdiv ceiling division", test_cdiv},
    {"grid_for == cdiv(n, kCdivBlock)", test_grid_for},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(F2Estimator, AllPrimitives) {
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
        std::fprintf(stderr, "test_f2_estimator: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_f2_estimator: all %zu primitive checks PASS\n",
                sizeof(kCases) / sizeof(kCases[0]));
    return 0;
}
#endif

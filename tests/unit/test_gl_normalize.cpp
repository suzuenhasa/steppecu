// tests/unit/test_gl_normalize.cpp
//
// Pure host unit test of the GL/PL/GP -> normalized-linear math (io/gl_normalize.hpp).
// No VCF text, no device, no data — just the numerically-stable normalization the
// bit-exact decode gate spot-checks against a hand computation. Self-checking
// executable; CTest gates on the exit code.
#include <array>
#include <cmath>
#include <cstdio>

#include "io/gl_normalize.hpp"

namespace {

int g_failures = 0;

void approx(const char* what, double got, double want, double tol = 1e-9) {
    if (std::abs(got - want) > tol) {
        std::printf("  [FAIL] %-32s got=%.12g want=%.12g (tol=%.1e)\n", what, got, want, tol);
        ++g_failures;
    }
}

}  // namespace

int main() {
    std::printf("=== gl_normalize unit test (PL/GL/GP -> normalized linear) ===\n");
    namespace gn = steppe::io::glnorm;

    // --- PL: hand computation. PL=(0,10,20): m=0; e=(1, 10^-1, 10^-2); sum=1.11.
    {
        const std::array<double, 3> L = gn::normalize_pl(0, 10, 20);
        const double s = 1.0 + 0.1 + 0.01;
        approx("pl(0,10,20)[0]", L[0], 1.0 / s);
        approx("pl(0,10,20)[1]", L[1], 0.1 / s);
        approx("pl(0,10,20)[2]", L[2], 0.01 / s);
        approx("pl sums to 1", L[0] + L[1] + L[2], 1.0);
    }
    // --- PL: min not at index 0 (subtract-the-min stability). PL=(100,0,150).
    {
        const std::array<double, 3> L = gn::normalize_pl(100, 0, 150);
        // e = (10^-10, 1, 10^-15); the ML genotype (index 1) dominates.
        approx("pl(100,0,150)[1]~1", L[1], 1.0, 1e-9);
        if (!(L[1] > L[0] && L[1] > L[2])) {
            std::printf("  [FAIL] pl(100,0,150) argmax != index 1\n");
            ++g_failures;
        }
    }
    // --- PL: flat (0,0,0) -> uniform 1/3.
    {
        const std::array<double, 3> L = gn::normalize_pl(0, 0, 0);
        approx("pl(0,0,0) uniform", L[0], 1.0 / 3.0);
    }

    // --- GL: log10 likelihoods, subtract-the-max. GL=(-1,0,-2) -> e=(0.1,1,0.01).
    {
        const std::array<double, 3> L = gn::normalize_gl(-1.0, 0.0, -2.0);
        const double s = 0.1 + 1.0 + 0.01;
        approx("gl(-1,0,-2)[0]", L[0], 0.1 / s);
        approx("gl(-1,0,-2)[1]", L[1], 1.0 / s);
        approx("gl(-1,0,-2)[2]", L[2], 0.01 / s);
    }
    // PL and GL agree when PL = -10*GL: PL(0,10,20) == GL(0,-1,-2).
    {
        const std::array<double, 3> Lp = gn::normalize_pl(0, 10, 20);
        const std::array<double, 3> Lg = gn::normalize_gl(0.0, -1.0, -2.0);
        approx("pl==gl [0]", Lp[0], Lg[0]);
        approx("pl==gl [1]", Lp[1], Lg[1]);
        approx("pl==gl [2]", Lp[2], Lg[2]);
    }

    // --- GP: posterior passthrough + renormalize. (0.8,0.15,0.05) already sums 1.
    {
        const std::array<double, 3> L = gn::normalize_gp(0.8, 0.15, 0.05);
        approx("gp(0.8,0.15,0.05)[0]", L[0], 0.8);
        approx("gp(0.8,0.15,0.05)[1]", L[1], 0.15);
        approx("gp(0.8,0.15,0.05)[2]", L[2], 0.05);
    }
    // GP with rounding drift renormalizes to exactly 1.
    {
        const std::array<double, 3> L = gn::normalize_gp(0.5, 0.5, 0.5);  // sums 1.5
        approx("gp drift[0]", L[0], 1.0 / 3.0);
        approx("gp sums to 1", L[0] + L[1] + L[2], 1.0);
    }
    // GP degenerate (all-zero) -> uninformative.
    {
        const std::array<double, 3> L = gn::normalize_gp(0.0, 0.0, 0.0);
        approx("gp(0,0,0) uninformative[0]", L[0], 1.0 / 3.0);
    }

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (PL/GL/GP normalization matches the hand computations)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}

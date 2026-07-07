// tests/unit/test_readv2_classify.cpp
//
// Host-only, device-free unit test of the READv2 numeric heart (src/core/readv2/
// readv2_classify.cpp): the degree classifier pinned at its three normalized-P0 cut
// points and the z statistic on canned per-window distributions. No CUDA, no GPU, no
// data — pure arithmetic, so it runs on every lane. Self-checking main; CTest gates on
// the exit code.
#include <cmath>
#include <cstdio>
#include <cstring>

#include "core/readv2/readv2_classify.hpp"

namespace {

namespace rc = steppe::core::readv2;

int g_failures = 0;

void expect_degree(double p0_norm, const char* want) {
    const char* got = rc::degree_from_p0norm(p0_norm);
    if (std::strcmp(got, want) != 0) {
        std::printf("  [FAIL] degree(%.6f) got=%s want=%s\n", p0_norm, got, want);
        ++g_failures;
    }
}

void expect_close(const char* what, double got, double want) {
    const double tol = 1e-9 + 1e-6 * std::fabs(want);
    if (std::fabs(got - want) > tol) {
        std::printf("  [FAIL] %-24s got=% .10g want=% .10g\n", what, got, want);
        ++g_failures;
    }
}

}  // namespace

int main() {
    std::printf("=== READv2 classifier + z unit test (host-only, no GPU) ===\n");

    // --- degree_from_p0norm at and around the cut points 0.625 / 0.8125 / 0.90625 ---
    expect_degree(0.00, rc::kDegreeIdentical);
    expect_degree(0.50, rc::kDegreeIdentical);
    expect_degree(0.6249, rc::kDegreeIdentical);
    expect_degree(0.6250, rc::kDegreeFirst);   // >= cut -> next band
    expect_degree(0.70, rc::kDegreeFirst);
    expect_degree(0.8124, rc::kDegreeFirst);
    expect_degree(0.8125, rc::kDegreeSecond);
    expect_degree(0.88, rc::kDegreeSecond);
    expect_degree(0.90624, rc::kDegreeSecond);
    expect_degree(0.90625, rc::kDegreeUnrelated);
    expect_degree(1.00, rc::kDegreeUnrelated);
    expect_degree(std::nan(""), rc::kDegreeUnrelated);  // undefined -> most-distant token

    // --- boundary_for_degree ---
    expect_close("bnd identical", rc::boundary_for_degree(rc::kDegreeIdentical), rc::kCutIdentical);
    expect_close("bnd first", rc::boundary_for_degree(rc::kDegreeFirst), rc::kCutFirst);
    expect_close("bnd second", rc::boundary_for_degree(rc::kDegreeSecond), rc::kCutSecond);
    expect_close("bnd unrelated", rc::boundary_for_degree(rc::kDegreeUnrelated), rc::kCutSecond);

    // --- readv2_z on a canned 4-window distribution p0 = {0.6,0.7,0.7,0.8} ------------
    // mean = 0.7, sum_p0_sq = 0.36+0.49+0.49+0.64 = 1.98, background = 1.0.
    // var_windows = (1.98 - 4*0.49)/3 = 0.02/3; se_mean = sqrt(var/4); z below.
    {
        const double p0_mean = 0.7, background = 1.0, sum_p0_sq = 1.98;
        const int n_windows = 4;
        const double p0_norm = p0_mean / background;  // 0.7 -> degree "first", boundary 0.8125
        const double var = (sum_p0_sq - n_windows * p0_mean * p0_mean) / (n_windows - 1);
        const double se_mean = std::sqrt(var / n_windows);
        const double want_z = (rc::kCutFirst - p0_norm) / (se_mean / background);
        const double got_z = rc::readv2_z(p0_mean, p0_norm, background, n_windows, sum_p0_sq);
        expect_close("z first-band", got_z, want_z);
        if (!(got_z > 0.0)) { std::printf("  [FAIL] z should be > 0\n"); ++g_failures; }
    }

    // --- readv2_z undefined for n_windows < 2 -> NaN ----------------------------------
    {
        const double z = rc::readv2_z(0.7, 0.7, 1.0, 1, 0.49);
        if (!std::isnan(z)) { std::printf("  [FAIL] z(n_windows=1) should be NaN\n"); ++g_failures; }
    }

    // --- readv2_z with zero background -> NaN (degenerate guard) -----------------------
    {
        const double z = rc::readv2_z(0.7, 0.7, 0.0, 4, 1.98);
        if (!std::isnan(z)) { std::printf("  [FAIL] z(background=0) should be NaN\n"); ++g_failures; }
    }

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (classifier + z arithmetic verified)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}

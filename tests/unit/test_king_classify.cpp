// tests/unit/test_king_classify.cpp
//
// Host-only, device-free unit test of the KING-robust numeric heart
// (src/core/internal/king_kinship.hpp): the per-SNP classification fold, the phi estimator
// at its verified limits (duplicate -> 0.5, unrelated -> ~0, the opposite-hom penalty), the
// degree bands at their cut points, and the REF<->ALT relabel invariance. No CUDA, no GPU,
// no data — pure arithmetic. Self-checking main; CTest gates on the exit code.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "core/internal/king_kinship.hpp"

namespace {

using steppe::core::degree_from_phi;
using steppe::core::king_classify;
using steppe::core::king_degree_label;
using steppe::core::king_phi;
using steppe::core::KingCounts;

int g_failures = 0;

// Fold two whole diploid-code vectors (0/1/2, or 3 = missing) into a KingCounts.
KingCounts fold(const std::vector<std::uint8_t>& ci, const std::vector<std::uint8_t>& cj) {
    KingCounts c;
    for (std::size_t s = 0; s < ci.size(); ++s) king_classify(ci[s], cj[s], c);
    return c;
}

void expect_close(const char* what, double got, double want) {
    const double tol = 1e-12 + 1e-9 * std::fabs(want);
    if (std::fabs(got - want) > tol) {
        std::printf("  [FAIL] %s got=%.12g want=%.12g\n", what, got, want);
        ++g_failures;
    }
}

void expect_eq_long(const char* what, long got, long want) {
    if (got != want) {
        std::printf("  [FAIL] %s got=%ld want=%ld\n", what, got, want);
        ++g_failures;
    }
}

void expect_degree(double phi, const char* want) {
    const char* got = king_degree_label(degree_from_phi(phi));
    if (std::strcmp(got, want) != 0) {
        std::printf("  [FAIL] degree(%.6f) got=%s want=%s\n", phi, got, want);
        ++g_failures;
    }
}

}  // namespace

int main() {
    // 1) Duplicate / MZ: ci == cj => IBS0 = 0, hetHet = het_i = het_j = #het => phi = 0.5.
    {
        const std::vector<std::uint8_t> g{0, 1, 2, 1, 0, 2, 1};
        const KingCounts c = fold(g, g);
        expect_eq_long("dup ibs0", c.ibs0, 0);
        expect_eq_long("dup hethet", c.hethet, 3);
        expect_eq_long("dup het_i", c.het_i, 3);
        expect_eq_long("dup het_j", c.het_j, 3);
        expect_close("dup phi", king_phi(c), 0.5);
        expect_degree(king_phi(c), "dup");
    }

    // 2) Missing on either side skips the site entirely (not "considered").
    {
        const std::vector<std::uint8_t> ci{1, 3, 1, 2};
        const std::vector<std::uint8_t> cj{1, 1, 3, 2};
        const KingCounts c = fold(ci, cj);
        expect_eq_long("miss nsnp", c.nsnp, 2);  // only sites 0 and 3 considered
        expect_eq_long("miss hethet", c.hethet, 1);
        expect_eq_long("miss ibs0", c.ibs0, 0);
    }

    // 3) Opposite-homozygote penalty pushes phi negative; and phi is INVARIANT under the
    //    REF<->ALT relabel (0<->2). Build a pair with some hetHet and some IBS0.
    {
        //         s: 0    1    2    3    4
        const std::vector<std::uint8_t> ci{1, 1, 0, 2, 1};
        const std::vector<std::uint8_t> cj{1, 1, 2, 0, 1};
        const KingCounts c = fold(ci, cj);
        expect_eq_long("pen hethet", c.hethet, 3);
        expect_eq_long("pen ibs0", c.ibs0, 2);
        expect_eq_long("pen het_i", c.het_i, 3);
        expect_eq_long("pen het_j", c.het_j, 3);
        // phi = (3 - 2*2) / (3 + 3) = -1/6.
        expect_close("pen phi", king_phi(c), -1.0 / 6.0);

        // Relabel 0<->2 on BOTH samples: het stays het, opposite-hom stays opposite.
        auto relabel = [](std::vector<std::uint8_t> v) {
            for (auto& x : v)
                if (x == 0) x = 2;
                else if (x == 2) x = 0;
            return v;
        };
        const KingCounts cr = fold(relabel(ci), relabel(cj));
        expect_close("relabel phi", king_phi(cr), king_phi(c));
        expect_eq_long("relabel ibs0", cr.ibs0, c.ibs0);
        expect_eq_long("relabel hethet", cr.hethet, c.hethet);
    }

    // 3b) ASYMMETRIC het counts (het_i != het_j): this is where the plink2 within-family
    //     estimator diverges from the naive (hetHet-2*IBS0)/(het_i+het_j) form. Build a pair
    //     whose considered sites give het_i=4, het_j=5, hetHet=3, IBS0=1.
    //         s: 0(HH) 1(HH) 2(HH) 3(i-het,j-hom) 4(IBS0) 5(j-het,i-hom) 6(j-het,i-hom)
    {
        const std::vector<std::uint8_t> ci{1, 1, 1, 1, 0, 0, 2};
        const std::vector<std::uint8_t> cj{1, 1, 1, 0, 2, 1, 1};
        const KingCounts c = fold(ci, cj);
        expect_eq_long("asym hethet", c.hethet, 3);
        expect_eq_long("asym ibs0", c.ibs0, 1);
        expect_eq_long("asym het_i", c.het_i, 4);
        expect_eq_long("asym het_j", c.het_j, 5);
        // plink2 within-family: 1/2 + (3 - 2*1 - (4+5)/2) / (2*min(4,5))
        //                     = 0.5 + (1 - 4.5) / 8 = 0.5 - 0.4375 = 0.0625.
        expect_close("asym phi (within-family)", king_phi(c), 0.0625);
        // The naive between-family form would give (3-2)/(4+5)=+0.1111 — a DIFFERENT answer, so
        // this case pins the min-denominator estimator against a silent regression.
    }

    // 4) No shared het on EITHER sample => phi is undefined (NaN) => degree "undefined".
    {
        const std::vector<std::uint8_t> ci{0, 2, 0};
        const std::vector<std::uint8_t> cj{0, 2, 2};
        const KingCounts c = fold(ci, cj);
        expect_eq_long("undef denom", c.het_i + c.het_j, 0);
        if (!std::isnan(king_phi(c))) {
            std::printf("  [FAIL] undefined phi should be NaN, got=%.6g\n", king_phi(c));
            ++g_failures;
        }
        expect_degree(king_phi(c), "undefined");
    }

    // 5) Degree bands at their cut points (2^-(k+3/2)).
    expect_degree(0.40, "dup");
    expect_degree(0.354, "dup");
    expect_degree(0.353, "1st");
    expect_degree(0.177, "1st");
    expect_degree(0.176, "2nd");
    expect_degree(0.0884, "2nd");
    expect_degree(0.0883, "3rd");
    expect_degree(0.0442, "3rd");
    expect_degree(0.0441, "unrelated");
    expect_degree(0.0, "unrelated");
    expect_degree(-0.2, "unrelated");

    if (g_failures == 0) {
        std::printf("test_king_classify: ALL PASS\n");
        return 0;
    }
    std::printf("test_king_classify: %d FAILURE(S)\n", g_failures);
    return 1;
}

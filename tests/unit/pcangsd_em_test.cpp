// tests/unit/pcangsd_em_test.cpp
//
// Host unit gate for the PCAngsd EM + GL-weighted covariance reference math
// (core::pcangsd_reference — the CpuBackend oracle the CUDA kernels reproduce). Two
// legs: (1) the posterior / expected-allele-2-dosage hand value, and (2) a full fit
// on a tiny CERTAIN-genotype fixture (N=4, individuals 0 and 1 identical) checking
// the covariance is symmetric, the two identical individuals get equal covariance
// rows AND equal PC coordinates, the spectrum is descending/finite, and the
// individual allele frequencies are in the [1e-4, 1-1e-4] clip range. Device-free;
// the real numeric concordance vs the pcangsd package is the box gate.
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/stats/pcangsd_em.hpp"

using steppe::core::pcangsd_reference;
using steppe::core::PcangsdRef;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}
bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }
}  // namespace

int main() {
    // --- 1) posterior / E_dosage hand value --------------------------------------
    // Lrr=0.8, Lhet=0.15, Laa=0.05, q=0.3:
    //   p0=0.8*0.49=0.392, p1=0.15*0.42=0.063, p2=0.05*0.09=0.0045, pSum=0.4595
    //   E_dosage=(0.063+2*0.0045)/0.4595 = 0.072/0.4595 = 0.1566920...
    {
        double p0, p1, p2, ps, ed;
        steppe::core::pcangsd_detail::posterior(0.8, 0.15, 0.05, 0.3, p0, p1, p2, ps, ed);
        check(close(ed, 0.072 / 0.4595, 1e-12), "E_dosage hand value");
        // certain genotypes: E_dosage == the allele-2 copy count regardless of q.
        double a0, a1, a2, as_, e0, e1, e2;
        double d;
        steppe::core::pcangsd_detail::posterior(1.0, 0.0, 0.0, 0.37, a0, a1, a2, as_, e0);
        steppe::core::pcangsd_detail::posterior(0.0, 1.0, 0.0, 0.37, a0, a1, a2, as_, e1);
        steppe::core::pcangsd_detail::posterior(0.0, 0.0, 1.0, 0.37, a0, a1, a2, as_, e2);
        (void)d;
        check(close(e0, 0.0, 1e-12) && close(e1, 1.0, 1e-12) && close(e2, 2.0, 1e-12),
              "certain E_dosage == copy count");
    }

    // --- 2) full fit on a certain-genotype fixture (individuals 0 and 1 identical) --
    const int N = 4;
    const long M = 40;
    // Per-site allele-2 copy counts: cols 0 and 1 IDENTICAL; f in {0.375,0.5,0.625}.
    std::vector<double> l(static_cast<std::size_t>(M) * static_cast<std::size_t>(N) * 3, 0.0);
    auto set_certain = [&](long j, int i, int g2) {  // g2 = copies of allele 2 (0/1/2)
        const std::size_t b = (static_cast<std::size_t>(j) * static_cast<std::size_t>(N) +
                               static_cast<std::size_t>(i)) * 3;
        // Tile order: l[b+0]=P(A2A2)=[g2==2], l[b+1]=het=[g2==1], l[b+2]=P(A1A1)=[g2==0].
        l[b + 0] = (g2 == 2) ? 1.0 : 0.0;
        l[b + 1] = (g2 == 1) ? 1.0 : 0.0;
        l[b + 2] = (g2 == 0) ? 1.0 : 0.0;
    };
    for (long j = 0; j < M; ++j) {
        const int g01 = static_cast<int>(j % 3);
        set_certain(j, 0, g01);
        set_certain(j, 1, g01);  // identical to individual 0
        set_certain(j, 2, static_cast<int>((j + 1) % 3));
        set_certain(j, 3, static_cast<int>((j + 2) % 3));
    }

    const PcangsdRef r = pcangsd_reference(l.data(), M, N, /*e=*/2, /*max_iter=*/100, /*tol=*/1e-8,
                                           /*maf=*/0.05, /*maf_iter=*/500, /*maf_tol=*/1e-8,
                                           /*want_pi=*/true);
    check(r.ok, "fit ok");
    check(r.N == 4 && r.e == 2, "N/e");
    check(r.M_used == 40 && r.M_total == 40, "all sites survive MAF 0.05");
    check(static_cast<long>(r.cov.size()) == N * N, "cov size N*N");

    // Symmetry.
    bool sym = true;
    for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b)
            if (!close(r.cov[static_cast<std::size_t>(a) * N + static_cast<std::size_t>(b)],
                       r.cov[static_cast<std::size_t>(b) * N + static_cast<std::size_t>(a)], 1e-9))
                sym = false;
    check(sym, "cov symmetric");

    // Identical individuals -> equal covariance rows and equal PC coordinates.
    bool rows_eq = true;
    for (int k = 0; k < N; ++k)
        if (!close(r.cov[0 * N + static_cast<std::size_t>(k)],
                   r.cov[1 * N + static_cast<std::size_t>(k)], 1e-6))
            rows_eq = false;
    check(rows_eq, "identical individuals -> equal cov rows");
    check(close(r.coords[0 * 2 + 0], r.coords[1 * 2 + 0], 1e-6) &&
              close(r.coords[0 * 2 + 1], r.coords[1 * 2 + 1], 1e-6),
          "identical individuals -> equal PC coords");

    // Spectrum descending + finite; leading variance in (0,1].
    check(r.eigenvalues.size() == 2 && r.eigenvalues[0] >= r.eigenvalues[1], "eigenvalues descending");
    check(std::isfinite(r.eigenvalues[0]) && std::isfinite(r.eigenvalues[1]), "eigenvalues finite");
    check(r.var_explained[0] > 0.0 && r.var_explained[0] <= 1.0 + 1e-12, "PC1 var in (0,1]");
    check(r.iters_run >= 1, "ran >= 1 iteration");

    // Individual allele frequencies in the clip range.
    check(static_cast<long>(r.pi.size()) == M * N, "pi size M_used*N");
    bool pi_ok = true;
    for (double v : r.pi)
        if (!(v >= 1e-4 - 1e-15 && v <= 1.0 - 1e-4 + 1e-15)) pi_ok = false;
    check(pi_ok, "pi in [1e-4, 1-1e-4]");

    if (g_fail == 0) {
        std::printf("RESULT: PASS (pcangsd EM reference: posterior + covariance + PCA properties)\n");
        return 0;
    }
    std::printf("RESULT: FAIL (%d check(s) failed)\n", g_fail);
    return 1;
}

// tests/unit/test_fst_all_pairs.cpp
//
// Host-only unit test of the `steppe fst --all-pairs` REFORMULATION on a tiny known matrix.
// It reproduces, in pure C++, the exact two-stage device algorithm (decode the per-(pop,SNP)
// sufficient statistic {n, ac, het} ONCE, then combine every C(P,2) pair by the SAME shared
// wc_finalize the single-pair kernel uses, indexed by the flat pair rank r = j*(j-1)/2 + i
// that readv2_unrank_pair inverts) and checks it three ways:
//   (1) against HAND-COMPUTED anchors on a 3-pop x 2-SNP tile,
//   (2) MATRIX == the direct single-pair computation for every pair (the parity claim: the
//       decode-once path yields bit-for-bit what recomputing each pair from scratch gives),
//   (3) the matrix is symmetric with a zero diagonal.
// The GPU kernel (fst_allpairs_kernel.cu) is a transliteration of this loop over the SAME
// core/internal/wc_fst.hpp primitives, so this pins the reformulation host-side (the
// divergence-prevention thesis, mirroring test_fst_wc.cpp). Pure C++ TU: NO CUDA, NO data.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "core/internal/decode_af.hpp"   // genotype_code, kCodesPerByte
#include "core/internal/wc_fst.hpp"      // WcPerPop, WcSite, wc_accumulate, wc_finalize

namespace {

using steppe::core::genotype_code;
using steppe::core::wc_accumulate;
using steppe::core::wc_finalize;
using steppe::core::WcPerPop;
using steppe::core::WcSite;

int g_failures = 0;

void check_true(const char* what, bool ok) {
    if (!ok) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}

void check_close(const char* what, double got, double want, double tol) {
    const double d = std::fabs(got - want);
    if (!(d <= tol)) {
        std::printf("  [FAIL] %-40s got=% .12e want=% .12e |d|=% .3e\n", what, got, want, d);
        ++g_failures;
    }
}

std::uint8_t pack4(std::uint8_t c0, std::uint8_t c1, std::uint8_t c2, std::uint8_t c3) {
    return static_cast<std::uint8_t>((c0 << 6) | (c1 << 4) | (c2 << 2) | c3);
}

// The sufficient-stat decode (host mirror of launch_fst_suffstat_decode): fold pop p's
// individuals [g0,g1) at SNP s into {n, ac, het}.
WcPerPop suffstat(const std::vector<std::uint8_t>& packed, std::size_t bpr, long s,
                  std::size_t g0, std::size_t g1) {
    const std::size_t byte = static_cast<std::size_t>(s) / steppe::core::kCodesPerByte;
    const int pos = static_cast<int>(s % steppe::core::kCodesPerByte);
    WcPerPop acc;
    for (std::size_t g = g0; g < g1; ++g) wc_accumulate(genotype_code(packed[g * bpr + byte], pos), acc);
    return acc;
}

}  // namespace

int main() {
    std::printf("=== all-pairs WC FST matrix reformulation (host) ===\n");

    // 3 pops x 2 individuals each (6 individuals), 2 SNPs, 1 byte/record.
    //   SNP0: pop0={2,2}  pop1={0,0}  pop2={1,1}
    //   SNP1: pop0={1,1}  pop1={1,1}  pop2={0,0}
    const std::vector<std::uint8_t> packed = {
        pack4(2, 1, 0, 0),  // ind0 (pop0)
        pack4(2, 1, 0, 0),  // ind1 (pop0)
        pack4(0, 1, 0, 0),  // ind2 (pop1)
        pack4(0, 1, 0, 0),  // ind3 (pop1)
        pack4(1, 0, 0, 0),  // ind4 (pop2)
        pack4(1, 0, 0, 0),  // ind5 (pop2)
    };
    const std::size_t bpr = 1;
    const int P = 3;
    const long M = 2;
    const std::size_t pop_off[P + 1] = {0, 2, 4, 6};

    // ---- Stage 1: decode the P x M sufficient-stat tensor ONCE. ----
    std::vector<WcPerPop> suff(static_cast<std::size_t>(P) * static_cast<std::size_t>(M));
    for (int p = 0; p < P; ++p)
        for (long s = 0; s < M; ++s)
            suff[static_cast<std::size_t>(p) * static_cast<std::size_t>(M) +
                 static_cast<std::size_t>(s)] =
                suffstat(packed, bpr, s, pop_off[p], pop_off[p + 1]);

    // ---- Stage 2: combine every C(P,2) pair over the tensor, r = j*(j-1)/2 + i. ----
    const std::size_t Pz = static_cast<std::size_t>(P);
    std::vector<double> fst(Pz * Pz, 0.0), num(Pz * Pz, 0.0), den(Pz * Pz, 0.0);
    for (int j = 1; j < P; ++j) {
        for (int i = 0; i < j; ++i) {
            double an = 0.0, ad = 0.0;
            for (long s = 0; s < M; ++s) {
                const WcPerPop& A = suff[static_cast<std::size_t>(i) * static_cast<std::size_t>(M) + s];
                const WcPerPop& B = suff[static_cast<std::size_t>(j) * static_cast<std::size_t>(M) + s];
                const WcSite w = wc_finalize(A, B);
                if (w.valid) { an += w.num; ad += w.den; }
            }
            const double f = (ad != 0.0) ? an / ad : std::nan("");
            const std::size_t ij = static_cast<std::size_t>(i) * Pz + static_cast<std::size_t>(j);
            const std::size_t ji = static_cast<std::size_t>(j) * Pz + static_cast<std::size_t>(i);
            fst[ij] = fst[ji] = f;
            num[ij] = num[ji] = an;
            den[ij] = den[ji] = ad;
        }
    }

    // ---- (1) hand-computed anchors ----
    // pair(0,1): SNP0 {2,2}vs{0,0} -> num 0.5 den 0.5 ; SNP1 {1,1}vs{1,1} -> num 0 den 0.25.
    //   ratio = 0.5 / 0.75 = 2/3.
    check_close("cell(0,1) num", num[0 * Pz + 1], 0.5, 1e-15);
    check_close("cell(0,1) den", den[0 * Pz + 1], 0.75, 1e-15);
    check_close("cell(0,1) FST == 2/3", fst[0 * Pz + 1], 2.0 / 3.0, 1e-15);

    // ---- (2) MATRIX cell == direct single-pair recomputation (from raw codes). ----
    for (int a = 0; a < P; ++a) {
        for (int b = a + 1; b < P; ++b) {
            double an = 0.0, ad = 0.0;
            for (long s = 0; s < M; ++s) {
                const WcPerPop A = suffstat(packed, bpr, s, pop_off[a], pop_off[a + 1]);
                const WcPerPop B = suffstat(packed, bpr, s, pop_off[b], pop_off[b + 1]);
                const WcSite w = wc_finalize(A, B);
                if (w.valid) { an += w.num; ad += w.den; }
            }
            const double f = (ad != 0.0) ? an / ad : std::nan("");
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "cell(%d,%d) == single-pair", a, b);
            check_close(lbl, fst[static_cast<std::size_t>(a) * Pz + static_cast<std::size_t>(b)], f,
                        0.0);  // bit-exact: identical inputs + shared wc_finalize
        }
    }

    // ---- (3) symmetry + zero diagonal ----
    for (int i = 0; i < P; ++i) {
        check_true("zero diagonal", fst[static_cast<std::size_t>(i) * Pz + static_cast<std::size_t>(i)] == 0.0);
        for (int j = 0; j < P; ++j) {
            const double a = fst[static_cast<std::size_t>(i) * Pz + static_cast<std::size_t>(j)];
            const double b = fst[static_cast<std::size_t>(j) * Pz + static_cast<std::size_t>(i)];
            check_true("symmetric", a == b || (std::isnan(a) && std::isnan(b)));
        }
    }

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (all-pairs matrix == single-pair, symmetric, hand anchors)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}

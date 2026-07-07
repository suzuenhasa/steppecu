// tests/unit/test_fst_wc.cpp
//
// Host-only unit test of the SHARED Weir & Cockerham 1984 per-site FST primitive
// (core/internal/wc_fst.hpp) — the exact wc_accumulate / wc_finalize the GPU kernel
// (fst_wc_kernel.cu) and the CpuBackend oracle (cpu_backend.cpp fst_wc_per_site) both
// call, pinned directly (the divergence-prevention thesis, mirroring test_decode.cpp for
// decode_af). Pure C++ TU: NO CUDA, NO data.
//
// Two faces:
//  (A) DIRECT wc_finalize on hand-computed anchors — the a/b/c algebra:
//        * fixed opposite alleles  A={2,2} B={0,0}  -> a=0.5, den=0.5, FST=1.0
//        * all-het identical       A={1,1} B={1,1}  -> a=0,   den=0.25, FST=0.0
//        * degenerate n1=n2=1                        -> INVALID (n_bar<=1, num=den=0)
//        * all-missing                               -> INVALID (num=den=0)
//  (B) a tiny hand-built 2-pop packed tile driven through the SAME per-site reduction the
//      backend runs (genotype_code + wc_accumulate + wc_finalize): a monomorphic-across-
//      the-pair site is INVALID (valid=0), and a MISSING genotype is excluded per-site
//      (adding one missing individual to a pop does not change num/den/fst).
//
// Self-checking main(): returns non-zero on the first failure — all CTest needs to gate.

#include <cmath>
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

// One byte holds 4 MSB-first codes for SNP0..3 of one individual (bytes_per_record == 1).
std::uint8_t pack4(std::uint8_t c0, std::uint8_t c1, std::uint8_t c2, std::uint8_t c3) {
    return static_cast<std::uint8_t>((c0 << 6) | (c1 << 4) | (c2 << 2) | c3);
}

// Per-site WC reduction over a population-contiguous packed tile — the exact loop shape
// the CpuBackend oracle and the GPU kernel run.
WcSite site_fst(const std::vector<std::uint8_t>& packed, std::size_t bpr, long s,
                std::size_t a0, std::size_t a1, std::size_t b0, std::size_t b1) {
    const std::size_t byte = static_cast<std::size_t>(s) / steppe::core::kCodesPerByte;
    const int pos = static_cast<int>(s % steppe::core::kCodesPerByte);
    WcPerPop A;
    for (std::size_t g = a0; g < a1; ++g) wc_accumulate(genotype_code(packed[g * bpr + byte], pos), A);
    WcPerPop B;
    for (std::size_t g = b0; g < b1; ++g) wc_accumulate(genotype_code(packed[g * bpr + byte], pos), B);
    return wc_finalize(A, B);
}

}  // namespace

int main() {
    std::printf("=== Weir & Cockerham 1984 per-site FST primitive (host) ===\n");

    // ---- (A) direct wc_finalize anchors ------------------------------------------------
    {
        // Fixed opposite alleles: A={2,2}, B={0,0}.
        const WcSite r = wc_finalize(WcPerPop{2.0, 4.0, 0.0}, WcPerPop{2.0, 0.0, 0.0});
        check_true("anchor1 valid", r.valid);
        check_close("anchor1 num (a)", r.num, 0.5, 1e-15);
        check_close("anchor1 den (a+b+c)", r.den, 0.5, 1e-15);
        check_close("anchor1 FST", r.fst, 1.0, 1e-15);
    }
    {
        // All-het identical: A={1,1}, B={1,1} -> ac=2, het=2 each.
        const WcSite r = wc_finalize(WcPerPop{2.0, 2.0, 2.0}, WcPerPop{2.0, 2.0, 2.0});
        check_true("anchor2 valid", r.valid);
        check_close("anchor2 num (a)", r.num, 0.0, 1e-15);
        check_close("anchor2 den (a+b+c)", r.den, 0.25, 1e-15);
        check_close("anchor2 FST", r.fst, 0.0, 1e-15);
    }
    {
        // Degenerate n1=n2=1 -> n_bar==1 -> invalid, num/den forced to 0 (no NaN/inf poison).
        const WcSite r = wc_finalize(WcPerPop{1.0, 2.0, 0.0}, WcPerPop{1.0, 0.0, 0.0});
        check_true("degenerate n1=n2=1 invalid", !r.valid);
        check_true("degenerate num==0", r.num == 0.0);
        check_true("degenerate den==0", r.den == 0.0);
        check_true("degenerate FST is NaN", std::isnan(r.fst));
    }
    {
        // All-missing (both pops empty) -> invalid.
        const WcSite r = wc_finalize(WcPerPop{0.0, 0.0, 0.0}, WcPerPop{0.0, 0.0, 0.0});
        check_true("all-missing invalid", !r.valid);
        check_true("all-missing num==0", r.num == 0.0);
        check_true("all-missing den==0", r.den == 0.0);
    }

    // ---- (B) tiny hand-built 2-pop tile -------------------------------------------------
    // pop A = ind0,ind1 ; pop B = ind2,ind3 ; 4 SNPs (1 byte/record).
    // SNP0: A={2,2} B={0,0}  -> FST 1.0
    // SNP1: A={1,1} B={1,1}  -> FST 0.0 (all het)
    // SNP2: A={0,0} B={0,0}  -> monomorphic across pair -> INVALID
    // SNP3: A={2,missing} B={0,0} -> missing excluded (n1=1)
    const std::uint8_t kMiss = steppe::core::kMissingGenotypeCode;  // 3
    std::vector<std::uint8_t> packed = {
        pack4(2, 1, 0, 2),      // ind0 (A)
        pack4(2, 1, 0, kMiss),  // ind1 (A) — SNP3 missing
        pack4(0, 1, 0, 0),      // ind2 (B)
        pack4(0, 1, 0, 0),      // ind3 (B)
    };
    const std::size_t bpr = 1;

    const WcSite s0 = site_fst(packed, bpr, 0, 0, 2, 2, 4);
    check_true("SNP0 valid", s0.valid);
    check_close("SNP0 FST == 1", s0.fst, 1.0, 1e-15);

    const WcSite s1 = site_fst(packed, bpr, 1, 0, 2, 2, 4);
    check_true("SNP1 valid", s1.valid);
    check_close("SNP1 FST == 0", s1.fst, 0.0, 1e-15);

    const WcSite s2 = site_fst(packed, bpr, 2, 0, 2, 2, 4);
    check_true("SNP2 monomorphic -> INVALID", !s2.valid);
    check_true("SNP2 num==0", s2.num == 0.0);
    check_true("SNP2 den==0", s2.den == 0.0);
    check_true("SNP2 FST is NaN", std::isnan(s2.fst));

    // SNP3: the missing genotype must be excluded per-site. The result over A={2,missing}
    // must equal the result over A={2} alone (missing changes nothing).
    const WcSite s3 = site_fst(packed, bpr, 3, 0, 2, 2, 4);
    check_true("SNP3 valid (missing excluded, n1=1,n2=2)", s3.valid);
    const WcSite s3_ref = wc_finalize(WcPerPop{1.0, 2.0, 0.0}, WcPerPop{2.0, 0.0, 0.0});
    check_close("SNP3 num == no-missing ref", s3.num, s3_ref.num, 1e-15);
    check_close("SNP3 den == no-missing ref", s3.den, s3_ref.den, 1e-15);
    check_close("SNP3 FST == no-missing ref", s3.fst, s3_ref.fst, 1e-15);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (WC anchors + tile monomorphic/missing edges)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}

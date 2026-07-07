// tests/unit/test_pca_standardize.cpp
//
// Host-only unit test of the SHARED Patterson-2006 PCA standardization primitive
// (core/internal/pca_standardize.hpp) — the exact pca_snp_scale / pca_standardize_one the
// GPU kernel (pca_standardize_kernel.cu) and the CpuBackend oracle (cpu_backend.cpp
// pca_covariance_eig) both call, pinned directly (the divergence-prevention thesis, mirroring
// test_fst_wc.cpp). Pure C++ TU: NO CUDA, NO data.
//
// Anchors:
//   * dosages {0,1,2,1} -> ac=4, n=4 -> p=0.5, center=1, pq=0.25, used, inv_scale=1/0.5=2;
//     standardized {0,1,2,1} -> {-2,0,+2,0} (mean-centered by 2p, scaled by 1/sqrt(pq)).
//   * monomorphic {2,2,2} -> p=1, pq=0, used=false, inv_scale=0 -> every code standardizes
//     to 0 (a zeroed column, equivalent to dropping the SNP).
//   * all-missing (n=0) -> used=false, inv_scale=0.
//   * one missing genotype is EXCLUDED from p: {0,2,missing} folds as ac=2,n=2 -> p=0.5,
//     and the missing code standardizes to 0 (mean-imputed after centering).
//
// Self-checking main(): returns non-zero on the first failure.
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "core/internal/decode_af.hpp"       // kMissingGenotypeCode
#include "core/internal/pca_standardize.hpp"  // pca_snp_scale, pca_standardize_one

namespace {

using steppe::core::kMissingGenotypeCode;
using steppe::core::pca_snp_scale;
using steppe::core::pca_standardize_one;
using steppe::core::PcaSnpScale;

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

double inv_scale_of(const PcaSnpScale& s) { return s.used ? (1.0 / std::sqrt(s.pq)) : 0.0; }

}  // namespace

int main() {
    std::printf("=== Patterson 2006 PCA standardization primitive (host) ===\n");

    // ---- p = 0.5 polymorphic site: dosages {0,1,2,1} ----------------------------------
    {
        const PcaSnpScale s = pca_snp_scale(/*ac=*/4, /*n=*/4);
        check_close("p=0.5 p", s.p, 0.5, 1e-15);
        check_close("p=0.5 center(2p)", s.center, 1.0, 1e-15);
        check_close("p=0.5 pq", s.pq, 0.25, 1e-15);
        check_true("p=0.5 used", s.used);
        const double inv = inv_scale_of(s);
        check_close("p=0.5 inv_scale (1/sqrt(pq))", inv, 2.0, 1e-15);
        check_close("std code 0", pca_standardize_one(0, s.center, inv), -2.0, 1e-15);
        check_close("std code 1", pca_standardize_one(1, s.center, inv), 0.0, 1e-15);
        check_close("std code 2", pca_standardize_one(2, s.center, inv), 2.0, 1e-15);
        check_close("std missing -> 0", pca_standardize_one(kMissingGenotypeCode, s.center, inv),
                    0.0, 1e-15);
    }

    // ---- monomorphic {2,2,2}: p=1, pq=0 -> unused, zeroed column ----------------------
    {
        const PcaSnpScale s = pca_snp_scale(/*ac=*/6, /*n=*/3);
        check_close("mono p", s.p, 1.0, 1e-15);
        check_close("mono pq", s.pq, 0.0, 1e-15);
        check_true("mono unused", !s.used);
        const double inv = inv_scale_of(s);
        check_close("mono inv_scale 0", inv, 0.0, 1e-15);
        check_close("mono std code 2 -> 0", pca_standardize_one(2, s.center, inv), 0.0, 1e-15);
    }

    // ---- monomorphic {0,0}: p=0, pq=0 -> unused ---------------------------------------
    {
        const PcaSnpScale s = pca_snp_scale(/*ac=*/0, /*n=*/2);
        check_close("mono0 p", s.p, 0.0, 1e-15);
        check_true("mono0 unused", !s.used);
    }

    // ---- all-missing (n=0) -> unused --------------------------------------------------
    {
        const PcaSnpScale s = pca_snp_scale(/*ac=*/0, /*n=*/0);
        check_true("all-missing unused", !s.used);
        check_close("all-missing inv_scale 0", inv_scale_of(s), 0.0, 1e-15);
    }

    // ---- one missing excluded from p: {0,2,missing} folds as ac=2,n=2 -----------------
    {
        // The fold: two non-missing codes (0 and 2) -> ac=2, n=2 -> p=0.5.
        const PcaSnpScale s = pca_snp_scale(/*ac=*/2, /*n=*/2);
        check_close("miss-excluded p=0.5", s.p, 0.5, 1e-15);
        check_true("miss-excluded used", s.used);
        const double inv = inv_scale_of(s);
        check_close("miss-excluded std 0", pca_standardize_one(0, s.center, inv), -2.0, 1e-15);
        check_close("miss-excluded std 2", pca_standardize_one(2, s.center, inv), 2.0, 1e-15);
        check_close("miss-excluded std missing", pca_standardize_one(kMissingGenotypeCode,
                                                                     s.center, inv), 0.0, 1e-15);
    }

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (Patterson standardization primitive pins center/scale/"
                    "missing/monomorphic)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}

// tests/unit/test_sfs_hist.cpp
//
// Host-only unit test of the SHARED 2D joint-SFS histogram primitive
// (core/internal/sfs_hist.hpp) — the exact sfs_site_complete / sfs_axis_extent /
// sfs_axis_index / sfs_linear_index the GPU joint-histogram kernel (sfs_hist_kernel.cu) and
// the CpuBackend oracle (cpu_backend.cpp joint_sfs_2pop) both call, driven over a tiny
// hand-built 2-pop packed tile through the SAME per-pop fold (wc_accumulate) + cell math
// the backend runs (the divergence-prevention thesis, mirroring test_fst_wc.cpp). Pure C++
// TU: NO CUDA, NO data.
//
// Tile: pop A = ind0,ind1 (NA=2, 2N=4); pop B = ind2,ind3 (NB=2, 2N=4); 4 SNPs (1 byte/rec):
//   SNP0: A={2,2} B={0,0}  -> acA=4 acB=0   (complete)
//   SNP1: A={2,1} B={0,1}  -> acA=3 acB=1   (complete)
//   SNP2: A={0,0} B={0,0}  -> acA=0 acB=0   (complete, monomorphic across the pair)
//   SNP3: A={2,MISSING} B={0,0} -> pop A incomplete -> DROPPED (complete-data restriction)
//
// UNFOLDED (extA=extB=5): cells (4,0),(3,1),(0,0). FOLDED (extA=extB=3): SNP0 (4,0) folds to
// (0,0) and collapses with SNP2 (0,0) -> cell (0,0) count 2; SNP1 (3,1) folds to (1,1).
// Both drop SNP3 (missing) -> n_complete == 3.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "core/internal/decode_af.hpp"   // genotype_code, kCodesPerByte, kMissingGenotypeCode
#include "core/internal/sfs_hist.hpp"    // sfs_site_complete/axis_extent/axis_index/linear_index
#include "core/internal/wc_fst.hpp"      // WcPerPop, wc_accumulate

namespace {

using steppe::core::genotype_code;
using steppe::core::sfs_axis_extent;
using steppe::core::sfs_axis_index;
using steppe::core::sfs_linear_index;
using steppe::core::sfs_site_complete;
using steppe::core::wc_accumulate;
using steppe::core::WcPerPop;

int g_failures = 0;

void check_true(const char* what, bool ok) {
    if (!ok) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}

void check_eq(const char* what, long got, long want) {
    if (got != want) {
        std::printf("  [FAIL] %-44s got=%ld want=%ld\n", what, got, want);
        ++g_failures;
    }
}

std::uint8_t pack4(std::uint8_t c0, std::uint8_t c1, std::uint8_t c2, std::uint8_t c3) {
    return static_cast<std::uint8_t>((c0 << 6) | (c1 << 4) | (c2 << 2) | c3);
}

// Accumulate the joint SFS over the tile — the exact loop the CpuBackend oracle + GPU kernel
// run (fold both segments, drop incomplete sites, place into the mixed-radix cell).
struct Hist {
    std::vector<std::int64_t> grid;
    long extA = 0, extB = 0, n_complete = 0;
};

Hist joint_sfs(const std::vector<std::uint8_t>& packed, std::size_t bpr, long M,
               std::size_t a0, std::size_t a1, std::size_t b0, std::size_t b1, bool folded) {
    const long NA = static_cast<long>(a1 - a0);
    const long NB = static_cast<long>(b1 - b0);
    Hist h;
    h.extA = sfs_axis_extent(NA, folded);
    h.extB = sfs_axis_extent(NB, folded);
    h.grid.assign(static_cast<std::size_t>(h.extA * h.extB), 0);
    for (long s = 0; s < M; ++s) {
        const std::size_t byte = static_cast<std::size_t>(s) / steppe::core::kCodesPerByte;
        const int pos = static_cast<int>(s % steppe::core::kCodesPerByte);
        WcPerPop A, B;
        for (std::size_t g = a0; g < a1; ++g) wc_accumulate(genotype_code(packed[g * bpr + byte], pos), A);
        for (std::size_t g = b0; g < b1; ++g) wc_accumulate(genotype_code(packed[g * bpr + byte], pos), B);
        if (!sfs_site_complete(A.n, NA, B.n, NB)) continue;
        const long idx[2] = {sfs_axis_index(A.ac, NA, folded), sfs_axis_index(B.ac, NB, folded)};
        const long ext[2] = {h.extA, h.extB};
        h.grid[static_cast<std::size_t>(sfs_linear_index(idx, ext, 2))] += 1;
        ++h.n_complete;
    }
    return h;
}

std::int64_t cell(const Hist& h, long i, long j) {
    return h.grid[static_cast<std::size_t>(i) * static_cast<std::size_t>(h.extB) +
                  static_cast<std::size_t>(j)];
}

std::int64_t grid_total(const Hist& h) {
    std::int64_t t = 0;
    for (std::int64_t v : h.grid) t += v;
    return t;
}

}  // namespace

int main() {
    std::printf("=== 2D joint-SFS histogram primitive (host) ===\n");

    // ---- (A) inline anchors -----------------------------------------------------------
    check_eq("unfolded extent 2N+1 (N=2)", sfs_axis_extent(2, false), 5);
    check_eq("folded extent N+1 (N=2)", sfs_axis_extent(2, true), 3);
    check_eq("unfolded index ac=4", sfs_axis_index(4.0, 2, false), 4);
    check_eq("unfolded index ac=1", sfs_axis_index(1.0, 2, false), 1);
    check_eq("folded index ac=4 -> min(4,0)=0", sfs_axis_index(4.0, 2, true), 0);
    check_eq("folded index ac=3 -> min(3,1)=1", sfs_axis_index(3.0, 2, true), 1);
    check_eq("folded index ac=2 -> min(2,2)=2", sfs_axis_index(2.0, 2, true), 2);
    check_true("complete when n==N both", sfs_site_complete(2.0, 2, 2.0, 2));
    check_true("incomplete when nA<NA", !sfs_site_complete(1.0, 2, 2.0, 2));
    {
        const long idx[2] = {3, 1};
        const long ext[2] = {5, 5};
        check_eq("linear index (3,1) over 5x5", sfs_linear_index(idx, ext, 2), 16);
    }

    // ---- (B) tiny hand-built 2-pop tile -----------------------------------------------
    const std::uint8_t kMiss = steppe::core::kMissingGenotypeCode;  // 3
    const std::vector<std::uint8_t> packed = {
        pack4(2, 2, 0, 2),      // ind0 (A)
        pack4(2, 1, 0, kMiss),  // ind1 (A) — SNP3 missing
        pack4(0, 0, 0, 0),      // ind2 (B)
        pack4(0, 1, 0, 0),      // ind3 (B)
    };
    const std::size_t bpr = 1;
    const long M = 4;

    // Unfolded.
    const Hist u = joint_sfs(packed, bpr, M, 0, 2, 2, 4, /*folded=*/false);
    check_eq("unfolded dims A", u.extA, 5);
    check_eq("unfolded dims B", u.extB, 5);
    check_eq("unfolded n_complete (SNP3 dropped)", u.n_complete, 3);
    check_eq("unfolded total == n_complete", static_cast<long>(grid_total(u)), 3);
    check_eq("unfolded cell (4,0)=1 [SNP0]", static_cast<long>(cell(u, 4, 0)), 1);
    check_eq("unfolded cell (3,1)=1 [SNP1]", static_cast<long>(cell(u, 3, 1)), 1);
    check_eq("unfolded cell (0,0)=1 [SNP2]", static_cast<long>(cell(u, 0, 0)), 1);

    // Folded — SNP0 (4,0) reflects to (0,0) and collapses with SNP2 (0,0).
    const Hist f = joint_sfs(packed, bpr, M, 0, 2, 2, 4, /*folded=*/true);
    check_eq("folded dims A", f.extA, 3);
    check_eq("folded dims B", f.extB, 3);
    check_eq("folded n_complete (SNP3 dropped)", f.n_complete, 3);
    check_eq("folded total == n_complete", static_cast<long>(grid_total(f)), 3);
    check_eq("folded cell (0,0)=2 [SNP0 reflected + SNP2]", static_cast<long>(cell(f, 0, 0)), 2);
    check_eq("folded cell (1,1)=1 [SNP1]", static_cast<long>(cell(f, 1, 1)), 1);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (SFS inline anchors + tile fold/missing/reflection)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}

// src/core/internal/sfs_hist.hpp
//
// The shared 2D joint site-frequency-spectrum (SFS) histogram primitive: given the
// per-population per-SNP non-missing count n and A1-copy count ac (== the FST path's
// WcPerPop{n, ac}) for two populations, it decides whether a site is usable under the
// complete-data convention and maps the pair of counts to one linear cell in the joint
// grid. Every helper is a STEPPE_HD inline so the GPU joint-histogram kernel and the CPU
// reference oracle call the exact same integer arithmetic and cannot drift.
//
// The SFS is a pure INTEGER-count stat (no floating-point estimator), so it is gated
// BIT-EXACT (cell-by-cell equality) against scikit-allel joint_sfs / joint_sfs_folded,
// NOT ADMIXTOOLS2 — no FP parity policy binds. It reuses wc_accumulate's per-pop fold
// verbatim: ac == the sum of diploid dosages 0/1/2 == copies of A1 (EIGENSTRAT .geno /
// PLINK .bed A1 axis), and missing (kMissingGenotypeCode) advances nothing.
//
// SEMANTICS (matched to scikit-allel exactly):
//   * Complete-data restriction (§4): scikit-allel bincounts derived-allele counts at a
//     FIXED chromosome count and does NOT hypergeometric-project. So a site is usable iff
//     every individual in BOTH pops is non-missing (n == N per pop), fixing the per-site
//     chromosome count at exactly 2*N. Incomplete sites contribute nothing.
//   * UNFOLDED cell (i, j): i = copies of A1 in pop A (0..2NA), j = copies of A1 in pop B
//     (0..2NB). Axis extent 2N+1. Polarity matters: the counted allele is A1 on both
//     sides (the gate aligns the scikit-allel column to A1 explicitly).
//   * FOLDED cell (i, j): each pop folded on ITS OWN within-population minor allele —
//     i = min(acA, 2NA-acA), j = min(acB, 2NB-acB) — matching scikit-allel's
//     np.amin(ac, axis=1) per-pop fold (NOT a combined-minor fold). Axis extent N+1.
//     Polarity-free (min is invariant to the A1<->A2 swap): the robust primary gate.
//
// v1 is 2D (two pops). The mixed-radix sfs_linear_index generalizes cleanly to K>2 (3D is
// a documented follow-up), but only 2D is built + gated now.
#ifndef STEPPE_CORE_INTERNAL_SFS_HIST_HPP
#define STEPPE_CORE_INTERNAL_SFS_HIST_HPP

#include "core/internal/host_device.hpp"   // STEPPE_HD
#include "core/internal/wc_fst.hpp"         // WcPerPop, wc_accumulate (the shared per-pop fold)

namespace steppe::core {

// A site is usable iff every individual in BOTH pops is non-missing (complete-data
// convention, §4). n = WcPerPop.n (non-missing count), N = individuals in the pop segment.
[[nodiscard]] STEPPE_HD inline bool sfs_site_complete(double nA, long NA,
                                                      double nB, long NB) noexcept {
    return (nA == static_cast<double>(NA)) && (nB == static_cast<double>(NB));
}

// Per-axis category extent. Unfolded: 2N+1 (A1 copies 0..2N). Folded: N+1 (minor 0..N).
[[nodiscard]] STEPPE_HD inline long sfs_axis_extent(long N, bool folded) noexcept {
    return folded ? (N + 1) : (2 * N + 1);
}

// Per-axis category index from a pop's A1-copy count ac (== WcPerPop.ac) over 2N chroms.
// Unfolded: a (== copies of A1). Folded: min(a, 2N-a) (per-pop minor, polarity-free).
[[nodiscard]] STEPPE_HD inline long sfs_axis_index(double ac, long N, bool folded) noexcept {
    const long a = static_cast<long>(ac);            // exact: ac is an integer sum of 0/1/2
    if (!folded) return a;
    const long minor = 2 * N - a;
    return (a < minor) ? a : minor;                  // min(a, 2N-a)
}

// Mixed-radix linear index over K axes (row-major, last axis fastest) — 2D uses K=2, but
// the signature generalizes cleanly to 3D+ (only 2D is BUILT in v1). idx[k] in
// [0, extent[k]); returns sum_k idx[k] * prod_{j>k} extent[j] (for K=2: idx[0]*ext[1]+idx[1]).
[[nodiscard]] STEPPE_HD inline long sfs_linear_index(const long* idx, const long* extent,
                                                     int K) noexcept {
    long lin = 0;
    for (int k = 0; k < K; ++k) lin = lin * extent[k] + idx[k];
    return lin;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_SFS_HIST_HPP

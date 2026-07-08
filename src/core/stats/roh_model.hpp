// src/core/stats/roh_model.hpp
//
// Host-pure model math for the hapROH runs-of-homozygosity copying HMM (`steppe
// roh`) — the CUDA-free numerics shared by the CPU reference oracle and the device
// forward-backward kernel. Mirrors ancibd_model.hpp / li_stephens.hpp for their
// engine faces.
//
// This is a faithful port of the reference hapROH (hringbauer/hapROH@master, imports
// as `hapsburg`; the production `hapsb_ind` path pins e_model="haploid",
// t_model="standard" Model_Transitions, h_model="FiveStateScaled"
// fwd_bkwd_scaled_lowmem):
//
//   * The MODEL is a (K+1)-state copying HMM: state 0 = non-ROH (Hardy-Weinberg
//     background emission from the panel allele frequency), states 1..K = "the target
//     copies reference haplotype k" (a ROH state per phased reference haplotype;
//     K = 2*n_ref for a diploid 1000G panel). emissions.py Model_Emissions.
//   * The K ROH states are SYMMETRIC, so the TRANSITION is run on a 3x3 symmetry-
//     collapsed continuous-time rate matrix Q (transitions.py Model_Transitions +
//     hmm_inference.prep_3x3matrix). The per-SNP transition T_l = expm(Q*dg_l) is
//     precomputed HOST-side and uploaded device-resident, exactly as ancibd_model.hpp
//     does for ancIBD and cmd_paint does for rho/mu/w. pre_compute_transition_matrix
//     divides the collapsed off-diagonal [:2,2] entries by (K-1) after the expm.
//
// The transition half is a DIRECT clone of ancibd_model.hpp (the machinery is
// state-symmetric and identical), only with the ROH rates and n = K instead of
// ancIBD's n = 4; we reuse ancibd_expm3 / ancibd_mat3_mul verbatim (they are already
// host-only, well tested, and independent of the IBD semantics).
//
// The forward-backward that CONSUMES this (see roh_fb.hpp) tracks all K+1 states but
// collapses to an O(K)-per-column pooled recursion. Native FP64 throughout (the FB is
// a sub-one product scan that underflows — the reduction carve-out, NOT the emulated-
// matmul default).
//
// Reference: docs/planning/haproh-face-spec.md §2
#ifndef STEPPE_CORE_STATS_ROH_MODEL_HPP
#define STEPPE_CORE_STATS_ROH_MODEL_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "core/stats/ancibd_model.hpp"  // ancibd_expm3 / ancibd_mat3_mul (host-only, reused)

#if defined(__CUDACC__)
#define STEPPE_HD __host__ __device__
#else
#define STEPPE_HD
#endif

namespace steppe::core {

// The haploid missing-allele sentinel the emission treats as uninformative (matches
// li_stephens.hpp kLsMissingAllele and the paint decode). A decoded pseudo-haploid
// allele is 0/1; anything > 1 is missing.
inline constexpr std::uint8_t kRohMissingAllele = 0xFFu;

// Locked hapROH default parameters (hapsb_ind, hapsburg_run.py; "defaults finetuned
// for pseudo-haploid 1240k aDNA data"). Every value is exposed as a `steppe roh`
// flag defaulting to these so the model math and the CLI defaults never drift apart.
// Rates are per Morgan; segment-length thresholds are in Morgan.
struct RohParams {
    double e_rate = 0.01;    // genotype/miscopy error (haploid emission)
    double roh_in = 1.0;     // rate of jumping INTO a ROH copying state
    double roh_out = 20.0;   // rate of jumping OUT of a ROH state
    double roh_jump = 300.0; // rate of jumping within ROH to another reference haplotype
    double in_val = 1e-4;    // initial per-ROH-state probability (fwd col 0 prior)
    double min_gap = 1e-10;  // per-SNP genetic gap floor (Morgan), prepare_rmap clamp
    double max_gap = 0.05;   // per-SNP genetic gap ceiling (Morgan), prepare_rmap clamp

    // Per-chromosome segment calling (postprocessing.call_roh + merge_called_blocks_custom).
    double cutoff_post = 0.999;       // per-SNP ROH-posterior calling threshold
    double roh_min_l_initial = 0.02;  // pre-merge length cut (Morgan)
    double max_gap_merge = 0.005;     // adjacent-block merge gap (Morgan)
    double min_len1 = 0.02;           // merge gate: min(len_a,len_b) >= min_len1 (Morgan)
    double min_len2 = 0.04;           // merge gate: max(len_a,len_b) >= min_len2 (Morgan)
    double roh_min_l_final = 0.04;    // post-merge length cut (Morgan)
};

// ---------------------------------------------------------------------------
// Emission (Model_Emissions.give_emission, emissions.py, e_model="haploid").
//
// Per SNP l: the target observation ob in {0,1} (the single pseudo-haploid allele) or
// missing; p = panel allele frequency (mean of the K reference-haplotype bits at l);
// e = e_rate. A missing target/reference allele emits 1.0 (uninformative), matching
// hapROH's only_calls site drop and ls_emission's missing handling.

// State 0 (non-ROH): the Hardy-Weinberg allele-frequency emission with error.
[[nodiscard]] STEPPE_HD inline double roh_emission0(std::uint8_t ob, double p, double e) {
    if (ob > 1u) return 1.0;  // missing target -> uninformative
    if (ob == 1u) return p * (1.0 - e) + (1.0 - p) * e;
    return (1.0 - p) * (1.0 - e) + p * e;
}

// ROH state k: match (1-e) / mismatch (e) of the target against reference haplotype k.
[[nodiscard]] STEPPE_HD inline double roh_emission_copy(std::uint8_t ref_hap, std::uint8_t ob,
                                                        double e) {
    if (ob > 1u || ref_hap > 1u) return 1.0;  // missing -> uninformative
    return (ref_hap == ob) ? (1.0 - e) : e;
}

// ---------------------------------------------------------------------------
// Transition: the collapsed 3x3 continuous-time rate matrix Q (host-only build) and
// the per-SNP T_l = expm(Q * dg_l) precompute — the ancibd_model.hpp machinery with
// ROH rates and n = K.

// Build the 3x3 prepped rate matrix Q (calc_transition_rate submat33 + prep_3x3matrix)
// for K symmetric ROH states. Collapsed order [non-ROH, this-ROH, other-ROH-pool];
// the pool column carries R[.,>0]*(K-1). Requires K >= 2.
[[nodiscard]] inline std::array<double, 9> roh_rate_matrix(const RohParams& pr, int K) {
    const double n = static_cast<double>(K);
    const double di = -pr.roh_out - pr.roh_jump + pr.roh_jump / n;  // ROH-state diagonal
    std::array<double, 9> q{};
    q[0] = -pr.roh_in;                    // Q[0][0]
    q[1] = pr.roh_in / n;                 // Q[0][1]  state0 -> this-ROH
    q[2] = (pr.roh_in / n) * (n - 1.0);   // Q[0][2]  state0 -> other-ROH-pool
    q[3] = pr.roh_out;                    // Q[1][0]  ROH -> state0
    q[4] = di;                            // Q[1][1]
    q[5] = (pr.roh_jump / n) * (n - 1.0); // Q[1][2]  this-ROH -> other-ROH-pool
    q[6] = pr.roh_out;                    // Q[2][0]
    q[7] = pr.roh_jump / n;              // Q[2][1]
    q[8] = -(pr.roh_out + pr.roh_jump / n);  // Q[2][2]
    return q;
}

// Per-SNP genetic gap dg_l = clamp(g_l - g_{l-1}, min_gap, max_gap), Morgan
// (hmm_inference.prepare_rmap). dg[0] = 0 clamped up to min_gap.
[[nodiscard]] inline std::vector<double> roh_gaps(const std::vector<double>& genpos_morgans,
                                                  const RohParams& pr) {
    const std::size_t M = genpos_morgans.size();
    std::vector<double> g(M, 0.0);
    for (std::size_t l = 0; l < M; ++l) {
        double d = (l == 0) ? 0.0 : (genpos_morgans[l] - genpos_morgans[l - 1]);
        if (d < pr.min_gap) d = pr.min_gap;
        if (d > pr.max_gap) d = pr.max_gap;
        g[l] = d;
    }
    return g;
}

// Precompute the per-SNP transition tensor T [M x 3 x 3], row-major t[l*9 + r*3 + c],
// as the FB consumes it: expm(Q * dg_l) with the collapsed off-diagonal [:2,2] entries
// divided by (K-1) (pre_compute_transition_matrix). Requires K >= 2.
[[nodiscard]] inline std::vector<double> roh_build_transition(
    const std::vector<double>& genpos_morgans, const RohParams& pr, int K) {
    const std::array<double, 9> Q = roh_rate_matrix(pr, K);
    const std::vector<double> gaps = roh_gaps(genpos_morgans, pr);
    const double denom = static_cast<double>(K - 1);
    const std::size_t M = genpos_morgans.size();
    std::vector<double> T(M * 9, 0.0);
    for (std::size_t l = 0; l < M; ++l) {
        std::array<double, 9> A;
        for (int i = 0; i < 9; ++i) A[i] = Q[i] * gaps[l];
        std::array<double, 9> E = ancibd_expm3(A);
        E[2] /= denom;  // [0][2] state0 -> per-other-ROH
        E[5] /= denom;  // [1][2] this-ROH -> per-other-ROH
        for (int i = 0; i < 9; ++i) T[l * 9 + static_cast<std::size_t>(i)] = E[i];
    }
    return T;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_STATS_ROH_MODEL_HPP

// src/core/stats/ancibd_model.hpp
//
// Host-pure model math for the ancIBD IBD-detection HMM (`steppe ibd`) — the
// CUDA-free numerics shared by the CPU reference oracle and the device forward-
// backward kernel. Mirrors li_stephens.hpp's role for the paint engine.
//
// This is a faithful port of the reference ancIBD (hringbauer/ancIBD@master, the
// production CLI path run_ancIBD.py:104-112 which pins e_model='haploid_gl2',
// t_model='standard' FiveStateTransitions, h_model='FiveStateScaled'):
//
//   * The EMISSION (emission.py HaplotypeSharingEmissions2.give_emission_matrix,
//     :127-138) is a 5-vector [1, e1, e2, e3, e4], already normalized to the
//     non-IBD Hardy-Weinberg baseline (e[0]=1). e[j] is the likelihood RATIO that
//     one haplotype of A and one of B are IBD, over four sharing configs.
//   * The four IBD states are symmetric, so the TRANSITION is run on a 3x3
//     symmetry-collapsed continuous-time rate matrix Q (transition.py
//     FiveStateTransitions.calc_transition_rate + prep_3x3matrix, :45-112). The
//     per-SNP transition T_l = expm(Q * dg_l) is precomputed HOST-side (one 3x3
//     matrix exponential per SNP gap) and uploaded device-resident, exactly as
//     cmd_paint precomputes rho/mu/w. full_transition_matrix (:75-94) normalizes
//     the collapsed off-diagonal [:2,2] entries by 1/(n-1)=1/3 after the expm.
//
// The forward-backward that CONSUMES this (fwd_bkwd_scaled, cfunc.pyx:357-469)
// tracks all FIVE states (emission is 5-row); the 3x3 is only the rate table. See
// ancibd_fb.hpp for the FB port. Native FP64 throughout (the FB is a sub-one
// product scan that underflows — the gl_normalize.hpp reduction carve-out, NOT the
// emulated-matmul default).
//
// Reference: docs/planning/ancibd-face-spec.md, docs/reference/src_core_stats_ancibd_model.hpp.md
#ifndef STEPPE_CORE_STATS_ANCIBD_MODEL_HPP
#define STEPPE_CORE_STATS_ANCIBD_MODEL_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#if defined(__CUDACC__)
#define STEPPE_HD __host__ __device__
#else
#define STEPPE_HD
#endif

namespace steppe::core {

// Locked ancIBD default parameters (run_ancIBD.py:104-112, transition.py). Every
// value is exposed as a `steppe ibd` flag defaulting to these; they are gathered
// here so the model math and the CLI defaults never drift apart.
struct AncibdParams {
    double ibd_in = 1.0;       // rate of jumping INTO an IBD copying state (Morgan^-1)
    double ibd_out = 10.0;     // rate of jumping OUT of an IBD state (run_ancIBD.py:28 default 10)
    double ibd_jump = 400.0;   // rate of jumping within IBD to the other haplotype pair
    double in_val = 1e-4;      // initial per-IBD-state probability (fwd col 0 prior)
    double min_error = 1e-3;   // haplotype ancestral-prob cap [min_error, 1-min_error]
    double p_min = 1e-3;       // derived-AF clamp in the emission (emission.py:106)
    double min_gap = 1e-10;    // per-SNP genetic gap floor (Morgan)
    double max_gap = 0.05;     // per-SNP genetic gap ceiling (Morgan)
    double cutoff_post = 0.99; // per-SNP IBD-posterior calling threshold
    double max_gap_merge = 0.0075;  // adjacent-block merge gap (Morgan) = 0.75 cM
    double min_cm = 8.0;       // called-segment length floor (cM)
    int n_ibd = 4;             // number of symmetric IBD states (2x2 copying configs)
};

// ---------------------------------------------------------------------------
// hts_p: the per-haplotype ANCESTRAL-allele probability from GP + phased GT.
//
// A faithful port of LoadH5Multi2.get_haplo_prob (loaddata.py:268-334, the
// PRODUCTION l_model='h5' path — NOT the legacy max(GP) path). The two returned
// values are P(haplotype A carries the ancestral/allele-0), P(haplotype B ...),
// where haplotype A is the first phased GT allele and B the second. All inputs are
// on the VCF-NATIVE REF/ALT axis (allele 0 == REF == ancestral, allele 1 == ALT ==
// derived), exactly matching ancIBD (which never applies a panel-A1 swap).
//
//   g00 = GP[0]            (P homozygous ref)
//   gp1 = GP[1]            (P heterozygous)
//   het prob routed by the phased hardcall: g01 gets it when GT is 0|1, g10 when
//   1|0, and it is split 50/50 for the homozygous calls 0|0 and 1|1.
//   h_A = g00 + g01 ;  h_B = g00 + g10 ;  both clamped to [min_error, 1-min_error].
STEPPE_HD inline void ancibd_haplo_prob(double gp0, double gp1, std::uint8_t gt0, std::uint8_t gt1,
                                        double min_error, double& hA, double& hB) {
    double g01, g10;
    if (gt0 == 0u && gt1 == 1u) {
        g01 = gp1;
        g10 = 0.0;
    } else if (gt0 == 1u && gt1 == 0u) {
        g01 = 0.0;
        g10 = gp1;
    } else {  // homozygous (0|0 or 1|1) -> split the het mass 50/50 (loaddata.py:305-311)
        g01 = gp1 * 0.5;
        g10 = gp1 * 0.5;
    }
    double a = gp0 + g01;
    double b = gp0 + g10;
    a = a < min_error ? min_error : (a > 1.0 - min_error ? 1.0 - min_error : a);
    b = b < min_error ? min_error : (b > 1.0 - min_error ? 1.0 - min_error : b);
    hA = a;
    hB = b;
}

// ---------------------------------------------------------------------------
// Emission (HaplotypeSharingEmissions2, emission.py:100-138).
//
// share(hx, hy, p) = hx*hy/(1-pt) + (1-hx)*(1-hy)/pt, pt = clamp(p, p_min, 1-p_min)
// The 5-vector is e[0]=1 (non-IBD baseline) and the four IBD sharing configs, in
// the reference's it.product([0,1],repeat=2) order over shared=[j, k+2]:
//   e[1] = share(A.hapA, B.hapA)   shared (0,2)
//   e[2] = share(A.hapA, B.hapB)   shared (0,3)
//   e[3] = share(A.hapB, B.hapA)   shared (1,2)
//   e[4] = share(A.hapB, B.hapB)   shared (1,3)
[[nodiscard]] STEPPE_HD inline double ancibd_share(double hx, double hy, double p, double p_min) {
    double pt = p < p_min ? p_min : (p > 1.0 - p_min ? 1.0 - p_min : p);
    return hx * hy / (1.0 - pt) + (1.0 - hx) * (1.0 - hy) / pt;
}

// Fill e[0..4] for a pair at one SNP. hAa/hAb are individual A's two haplotype
// ancestral probs, hBa/hBb individual B's. p is the derived (ALT) allele freq.
STEPPE_HD inline void ancibd_emission5(double hAa, double hAb, double hBa, double hBb, double p,
                                       double p_min, double e[5]) {
    e[0] = 1.0;
    e[1] = ancibd_share(hAa, hBa, p, p_min);
    e[2] = ancibd_share(hAa, hBb, p, p_min);
    e[3] = ancibd_share(hAb, hBa, p, p_min);
    e[4] = ancibd_share(hAb, hBb, p, p_min);
}

// ---------------------------------------------------------------------------
// Transition: the collapsed 3x3 continuous-time rate matrix Q (host-only build)
// and the per-SNP T_l = expm(Q * dg_l) precompute.

// Build the 3x3 prepped rate matrix Q (calc_transition_rate submat33 + prep_3x3matrix).
[[nodiscard]] inline std::array<double, 9> ancibd_rate_matrix(const AncibdParams& pr) {
    const double n = static_cast<double>(pr.n_ibd);
    const double di = -pr.ibd_out - pr.ibd_jump + pr.ibd_jump / n;  // IBD-state diagonal (rate mat)
    // Rate matrix R (submat33) rows 0..2:
    // R[0] = {-ibd_in, ibd_in/n, ibd_in/n}
    // R[1] = { ibd_out, di, ibd_jump/n}
    // R[2] = { ibd_out, ibd_jump/n, di}
    // prep_3x3matrix collapses the 3rd state:
    std::array<double, 9> q{};
    q[0] = -pr.ibd_in;                 // Q[0][0]
    q[1] = pr.ibd_in / n;              // Q[0][1]
    q[2] = (pr.ibd_in / n) * (n - 1);  // Q[0][2] = R[0][2]*(n-1)
    q[3] = pr.ibd_out;                 // Q[1][0]
    q[4] = di;                         // Q[1][1]
    q[5] = (pr.ibd_jump / n) * (n - 1);  // Q[1][2] = R[1][2]*(n-1)
    q[6] = pr.ibd_out;                 // Q[2][0]
    q[7] = pr.ibd_jump / n;            // Q[2][1]
    q[8] = -(pr.ibd_out + pr.ibd_jump / n);  // Q[2][2] = -(R[2][0]+R[2][1])
    return q;
}

// 3x3 matrix multiply, row-major.
inline void ancibd_mat3_mul(const double* a, const double* b, double* out) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += a[i * 3 + k] * b[k * 3 + j];
            out[i * 3 + j] = s;
        }
}

// expm of a 3x3 matrix by scaling-and-squaring + a Taylor series (scaled to
// infinity-norm < 0.5, then ~18 Taylor terms, then squared back). Accurate to well
// below 1e-14 for the rate-matrix norms this model produces (||Q*dg|| <~ 30), far
// under the concordance tolerance; robust to complex eigenvalues (no eig needed).
[[nodiscard]] inline std::array<double, 9> ancibd_expm3(const std::array<double, 9>& A) {
    // infinity norm (max abs row sum)
    double nrm = 0.0;
    for (int i = 0; i < 3; ++i) {
        double rs = std::abs(A[i * 3 + 0]) + std::abs(A[i * 3 + 1]) + std::abs(A[i * 3 + 2]);
        nrm = std::max(nrm, rs);
    }
    int s = 0;
    double scale = 1.0;
    while (nrm * scale > 0.5) {
        scale *= 0.5;
        ++s;
    }
    double B[9];
    for (int i = 0; i < 9; ++i) B[i] = A[i] * scale;

    // Taylor: E = I + B + B^2/2! + ...
    double E[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double term[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // running B^k / k!
    double tmp[9];
    for (int k = 1; k <= 18; ++k) {
        ancibd_mat3_mul(term, B, tmp);
        const double inv = 1.0 / static_cast<double>(k);
        for (int i = 0; i < 9; ++i) term[i] = tmp[i] * inv;
        for (int i = 0; i < 9; ++i) E[i] += term[i];
    }
    // square s times
    for (int i = 0; i < s; ++i) {
        ancibd_mat3_mul(E, E, tmp);
        for (int j = 0; j < 9; ++j) E[j] = tmp[j];
    }
    std::array<double, 9> out;
    for (int i = 0; i < 9; ++i) out[i] = E[i];
    return out;
}

// Per-SNP genetic gap dg_l = clamp(g_l - g_{l-1}, min_gap, max_gap), in Morgan
// (transition.py rmap_to_gaps). dg[0] uses the reference convention gaps[0]=0 then
// clamped up to min_gap. `genpos_morgans` is the per-kept-site map in SNP order.
[[nodiscard]] inline std::vector<double> ancibd_gaps(const std::vector<double>& genpos_morgans,
                                                     const AncibdParams& pr) {
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
// as the FB consumes it: expm(Q * dg_l) with the collapsed off-diagonal [:2,2]
// entries divided by (n_ibd - 1) (full_transition_matrix, transition.py:92-93).
[[nodiscard]] inline std::vector<double> ancibd_build_transition(
    const std::vector<double>& genpos_morgans, const AncibdParams& pr) {
    const std::array<double, 9> Q = ancibd_rate_matrix(pr);
    const std::vector<double> gaps = ancibd_gaps(genpos_morgans, pr);
    const double denom = static_cast<double>(pr.n_ibd - 1);
    const std::size_t M = genpos_morgans.size();
    std::vector<double> T(M * 9, 0.0);
    for (std::size_t l = 0; l < M; ++l) {
        std::array<double, 9> A;
        for (int i = 0; i < 9; ++i) A[i] = Q[i] * gaps[l];
        std::array<double, 9> E = ancibd_expm3(A);
        E[2] /= denom;  // [0][2]
        E[5] /= denom;  // [1][2]
        for (int i = 0; i < 9; ++i) T[l * 9 + static_cast<std::size_t>(i)] = E[i];
    }
    return T;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_STATS_ANCIBD_MODEL_HPP

// src/core/stats/ancibd_fb.hpp
//
// Host reference 5-state scaled forward-backward for `steppe ibd` — a faithful,
// scalar, native-FP64 port of ancIBD's compiled fwd_bkwd_scaled (the production
// h_model='FiveStateScaled', cfunc.pyx:357-469). It is BOTH the CpuBackend
// reference oracle and the exact recurrence the CUDA kernel (ancibd_fb_kernel.cu)
// mirrors block-per-pair.
//
// The HMM has FIVE hidden states (state 0 = non-IBD; states 1-4 = the four
// haplotype-sharing configs, each with its OWN emission e1..e4). The transition is
// the 3x3 symmetry-collapsed T tensor from ancibd_model.hpp; the FB nonetheless
// carries all five forward/backward values because the four IBD emissions differ
// per SNP. Per-column rescaling (divide each forward column by its sum c_i; rescale
// the backward column by the SAME c_i) keeps the sub-one product from underflowing;
// the posterior is post = fwd * bwd directly (the shared c_i make each column sum to
// 1 — no explicit renormalization, matching the reference). The IBD posterior is
// 1 - post[0] (postprocessing.py roh_posterior).
//
// Reference: docs/planning/ancibd-face-spec.md §2c
#ifndef STEPPE_CORE_STATS_ANCIBD_FB_HPP
#define STEPPE_CORE_STATS_ANCIBD_FB_HPP

#include <cstddef>
#include <vector>

#include "core/stats/ancibd_model.hpp"

namespace steppe::core {

// Run the 5-state scaled FB for ONE pair over M SNPs and write p_ibd[l] = 1 - post0.
//   hAa/hAb: individual A's two haplotype ancestral probs [M] (VCF-native axis).
//   hBa/hBb: individual B's two haplotype ancestral probs [M].
//   p:       per-SNP derived (ALT) allele frequency [M].
//   T:       per-SNP transition tensor [M*9], row-major t[l*9 + r*3 + c]
//            (ancibd_build_transition).
//   p_ibd_out: OUTPUT [M].
inline void ancibd_fb_pair(const double* hAa, const double* hAb, const double* hBa,
                           const double* hBb, const double* p, const double* T, long M,
                           const AncibdParams& pr, double* p_ibd_out) {
    if (M <= 0) return;
    const std::size_t Ms = static_cast<std::size_t>(M);
    const double in_val = pr.in_val;
    const double p_min = pr.p_min;

    // Precompute the 5-row emission (cheap, M small per chromosome).
    std::vector<double> e_mat(5 * Ms, 0.0);
    for (long l = 0; l < M; ++l) {
        const std::size_t ls = static_cast<std::size_t>(l);
        double e[5];
        ancibd_emission5(hAa[ls], hAb[ls], hBa[ls], hBb[ls], p[ls], p_min, e);
        for (int j = 0; j < 5; ++j) e_mat[static_cast<std::size_t>(j) * Ms + ls] = e[j];
    }
    auto E = [&](int j, long l) -> double {
        return e_mat[static_cast<std::size_t>(j) * Ms + static_cast<std::size_t>(l)];
    };
    auto TT = [&](long l, int r, int c) -> double {
        return T[static_cast<std::size_t>(l) * 9 + static_cast<std::size_t>(r) * 3 +
                 static_cast<std::size_t>(c)];
    };

    // Forward table (5 x M), per-column normalized; c[l] the column normalizers.
    std::vector<double> fwd(5 * Ms, 0.0);
    std::vector<double> c(Ms, 1.0);
    auto F = [&](int j, long l) -> double& {
        return fwd[static_cast<std::size_t>(j) * Ms + static_cast<std::size_t>(l)];
    };

    // Column 0: the in_val prior, NO emission (matches cfunc.pyx:391-392, c[0]=1).
    F(0, 0) = 1.0 - 4.0 * in_val;
    for (int j = 1; j < 5; ++j) F(j, 0) = in_val;
    c[0] = 1.0;

    for (long i = 1; i < M; ++i) {
        const double stay = TT(i, 1, 1) - TT(i, 1, 2);
        const double fwd0_prev = F(0, i - 1);
        const double fl = 1.0 - fwd0_prev;  // sum of the four IBD forwards (normalized)
        double temp[5];
        temp[0] = E(0, i) * (fwd0_prev * TT(i, 0, 0) + fl * TT(i, 1, 0));
        const double x1 = fwd0_prev * TT(i, 0, 1);  // coming from state 0
        const double x2 = fl * TT(i, 1, 2);         // coming from another IBD state
        for (int j = 1; j < 5; ++j) temp[j] = E(j, i) * (x1 + x2 + F(j, i - 1) * stay);
        double ci = 0.0;
        for (int j = 0; j < 5; ++j) ci += temp[j];
        const double inv = (ci > 0.0) ? 1.0 / ci : 0.0;
        for (int j = 0; j < 5; ++j) F(j, i) = temp[j] * inv;
        c[i] = ci;
    }

    // Backward pass (streamed): bwd column, init at M-1 = ones; posterior on the fly.
    double bwd[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    for (long i = M - 1; i >= 1; --i) {
        // Posterior at column i uses the CURRENT bwd (== column i).
        p_ibd_out[static_cast<std::size_t>(i)] = 1.0 - F(0, i) * bwd[0];

        const double stay = TT(i, 1, 1) - TT(i, 1, 2);
        double fl = 0.0;
        for (int k = 1; k < 5; ++k) fl += bwd[k] * E(k, i);
        double ntemp[5];
        ntemp[0] = bwd[0] * TT(i, 0, 0) * E(0, i) + fl * TT(i, 0, 1);
        const double x1 = E(0, i) * bwd[0] * TT(i, 1, 0);
        const double x2 = fl * TT(i, 1, 2);
        for (int j = 1; j < 5; ++j) ntemp[j] = x1 + x2 + E(j, i) * bwd[j] * stay;
        const double inv = (c[static_cast<std::size_t>(i)] > 0.0)
                               ? 1.0 / c[static_cast<std::size_t>(i)]
                               : 0.0;
        for (int j = 0; j < 5; ++j) bwd[j] = ntemp[j] * inv;  // now column i-1
    }
    // Column 0 posterior with bwd == column 0.
    p_ibd_out[0] = 1.0 - F(0, 0) * bwd[0];
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_STATS_ANCIBD_FB_HPP

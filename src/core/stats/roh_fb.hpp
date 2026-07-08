// src/core/stats/roh_fb.hpp
//
// Host reference (K+1)-state scaled forward-backward for `steppe roh` — a faithful,
// scalar, native-FP64 port of hapROH's compiled fwd_bkwd_scaled_lowmem (the production
// h_model="FiveStateScaled", cfunc.pyx). It is BOTH the CpuBackend reference oracle and
// the exact recurrence the CUDA kernel (roh_fb_kernel.cu) mirrors block-per-target.
//
// The HMM has K+1 hidden states (state 0 = non-ROH; states 1..K = "the target copies
// reference haplotype k"). All K ROH states are symmetric, so the (K+1)-state recursion
// collapses to O(K) per column using a pooled ROH-mass sum f = sum_{k>=1} alpha[k] and
// three transition scalars from the collapsed 3x3 T tensor (roh_model.hpp). Per-column
// rescaling (divide each forward/backward column by its own sum) keeps the sub-one
// product from underflowing; the ROH posterior is scale-invariant:
//   p_roh(l) = 1 - gamma_0(l),  gamma_0(l) = alpha_0(l)*beta_0(l) / sum_s alpha_s(l)beta_s(l)
// (postprocessing.modify_posterior0, 1 - posterior0).
//
// Column 0 is the in_val prior ONLY — NO emission, NO rescale (c[0]=1), matching
// cfunc.pyx (the first emission is applied at column 1) and the shipped ancibd_fb.hpp.
//
// Reference: docs/planning/haproh-face-spec.md §2d, §3
#ifndef STEPPE_CORE_STATS_ROH_FB_HPP
#define STEPPE_CORE_STATS_ROH_FB_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/stats/roh_model.hpp"

namespace steppe::core {

// Run the (K+1)-state scaled FB for ONE target over M SNPs and write p_roh[l].
//   ob:       target observed alleles [M] in {0,1, missing} (single pseudo-haploid call)
//   refhaps:  donor-major reference-haplotype bytes [K*M], refhaps[k*M + l] in {0,1,missing}
//   p:        per-SNP panel allele frequency [M]
//   T:        per-SNP transition tensor [M*9], row-major t[l*9 + r*3 + c]
//             (roh_build_transition)
//   p_roh_out: OUTPUT [M].
inline void roh_fb_target(const std::uint8_t* ob, const std::uint8_t* refhaps, const double* p,
                          const double* T, int K, long M, const RohParams& pr, double* p_roh_out) {
    if (M <= 0 || K <= 0) return;
    const std::size_t Ms = static_cast<std::size_t>(M);
    const std::size_t Ks = static_cast<std::size_t>(K);
    const double e = pr.e_rate;
    const double in_val = pr.in_val;

    // Full forward table: state-0 vector [M] + the ROH block [K*M] (state-major k*M+l).
    // Large-K production uses checkpoint/recompute on the GPU; the CPU oracle stores the
    // table (K is modest for the unit gate + the parity spot-check).
    std::vector<double> a0(Ms, 0.0);
    std::vector<double> aroh(Ks * Ms, 0.0);

    auto TT = [&](long l, int r, int c) -> double {
        return T[static_cast<std::size_t>(l) * 9 + static_cast<std::size_t>(r) * 3 +
                 static_cast<std::size_t>(c)];
    };

    // Column 0: the in_val prior, NO emission, NO rescale (c[0]=1). Sum is exactly 1.
    a0[0] = 1.0 - static_cast<double>(K) * in_val;
    for (int k = 0; k < K; ++k) aroh[static_cast<std::size_t>(k) * Ms + 0] = in_val;

    for (long l = 1; l < M; ++l) {
        const std::size_t ls = static_cast<std::size_t>(l);
        const double t00 = TT(l, 0, 0), t01 = TT(l, 0, 1);
        const double t10 = TT(l, 1, 0), t11 = TT(l, 1, 1), t12 = TT(l, 1, 2);
        const double stay = t11 - t12;
        const double a0p = a0[ls - 1];
        double f = 0.0;  // pooled ROH forward mass at l-1
        for (int k = 0; k < K; ++k) f += aroh[static_cast<std::size_t>(k) * Ms + (ls - 1)];
        const std::uint8_t obl = ob[ls];
        const double e0 = roh_emission0(obl, p[ls], e);
        const double na0 = e0 * (a0p * t00 + f * t10);
        const double x1 = a0p * t01;
        const double x2 = f * t12;
        double s = na0;
        for (int k = 0; k < K; ++k) {
            const std::uint8_t rk = refhaps[static_cast<std::size_t>(k) * Ms + ls];
            const double ek = roh_emission_copy(rk, obl, e);
            const double v = ek * (x1 + x2 + aroh[static_cast<std::size_t>(k) * Ms + (ls - 1)] * stay);
            aroh[static_cast<std::size_t>(k) * Ms + ls] = v;
            s += v;
        }
        const double inv = (s > 0.0) ? 1.0 / s : 0.0;
        a0[ls] = na0 * inv;
        for (int k = 0; k < K; ++k) aroh[static_cast<std::size_t>(k) * Ms + ls] *= inv;
    }

    // Backward (streamed): beta column init 1; posterior via explicit denom on the fly.
    double b0 = 1.0;
    std::vector<double> broh(Ks, 1.0);
    std::vector<double> nbroh(Ks, 0.0);
    for (long l = M - 1; l >= 0; --l) {
        const std::size_t ls = static_cast<std::size_t>(l);
        double denom = a0[ls] * b0;
        for (int k = 0; k < K; ++k)
            denom += aroh[static_cast<std::size_t>(k) * Ms + ls] * broh[static_cast<std::size_t>(k)];
        const double gamma0 = (denom > 0.0) ? a0[ls] * b0 / denom : 0.0;
        p_roh_out[ls] = 1.0 - gamma0;
        if (l == 0) break;

        const double t00 = TT(l, 0, 0), t01 = TT(l, 0, 1);
        const double t10 = TT(l, 1, 0), t11 = TT(l, 1, 1), t12 = TT(l, 1, 2);
        const double stay = t11 - t12;
        const std::uint8_t obl = ob[ls];
        const double e0 = roh_emission0(obl, p[ls], e);
        double fl = 0.0;  // sum_k beta_k * e_k at column l
        for (int k = 0; k < K; ++k) {
            const std::uint8_t rk = refhaps[static_cast<std::size_t>(k) * Ms + ls];
            fl += broh[static_cast<std::size_t>(k)] * roh_emission_copy(rk, obl, e);
        }
        const double nb0 = b0 * t00 * e0 + fl * t01;
        const double x1 = e0 * b0 * t10;
        double s = nb0;
        for (int k = 0; k < K; ++k) {
            const std::uint8_t rk = refhaps[static_cast<std::size_t>(k) * Ms + ls];
            const double ek = roh_emission_copy(rk, obl, e);
            const double v = x1 + fl * t12 + ek * broh[static_cast<std::size_t>(k)] * stay;
            nbroh[static_cast<std::size_t>(k)] = v;
            s += v;
        }
        const double inv = (s > 0.0) ? 1.0 / s : 0.0;
        b0 = nb0 * inv;
        for (int k = 0; k < K; ++k) broh[static_cast<std::size_t>(k)] = nbroh[static_cast<std::size_t>(k)] * inv;
    }
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_STATS_ROH_FB_HPP

// src/core/stats/li_stephens.cpp — host model helpers for `steppe paint`.
//
// CUDA-free; the forward-backward itself is a ComputeBackend virtual. Reference:
// docs/planning/li-stephens-engine-scope.md §2a, §2c.

#include "core/stats/li_stephens.hpp"

#include <cmath>
#include <cstddef>

#include "core/internal/decode_af.hpp"

namespace steppe::core {

std::uint8_t haploid_allele_from_code(std::uint8_t code) noexcept {
    if (code == 0u) return 0u;
    if (code == 2u) return 1u;
    return kLsMissingAllele;  // missing (code 3) or the illegal het code 1
}

double watterson_emission_rate(int K) noexcept {
    if (K < 2) return 0.0;
    double harmonic = 0.0;
    for (int i = 1; i < K; ++i) harmonic += 1.0 / static_cast<double>(i);
    const double theta_hat = 1.0 / harmonic;
    return 0.5 * theta_hat / (theta_hat + static_cast<double>(K));
}

double ls_recomb_prob(double delta_morgans, double Ne, int K) noexcept {
    if (K < 1 || !(Ne > 0.0) || !(delta_morgans > 0.0)) return 0.0;
    const double expected = 4.0 * Ne * delta_morgans / static_cast<double>(K);
    double rho = 1.0 - std::exp(-expected);
    if (rho < 0.0) rho = 0.0;
    if (rho > 1.0) rho = 1.0;
    return rho;
}

std::vector<double> build_recomb_probs(const std::vector<int>& chrom,
                                       const std::vector<double>& genpos_morgans, double Ne,
                                       int K) {
    const std::size_t M = genpos_morgans.size();
    std::vector<double> rho(M, 1.0);
    const bool have_chrom = chrom.size() == M;
    for (std::size_t l = 1; l < M; ++l) {
        const bool same_chrom = !have_chrom || (chrom[l] == chrom[l - 1]);
        if (!same_chrom) {
            rho[l] = 1.0;  // unlinked across a chromosome boundary
            continue;
        }
        const double delta = genpos_morgans[l] - genpos_morgans[l - 1];
        rho[l] = ls_recomb_prob(delta > 0.0 ? delta : 0.0, Ne, K);
    }
    return rho;
}

std::vector<double> build_genetic_weights(const std::vector<int>& chrom,
                                          const std::vector<double>& genpos_morgans) {
    const std::size_t M = genpos_morgans.size();
    std::vector<double> w(M, 0.0);
    const bool have_chrom = chrom.size() == M;
    for (std::size_t l = 0; l < M; ++l) {
        double gap_left = 0.0;
        double gap_right = 0.0;
        if (l > 0) {
            const bool same = !have_chrom || (chrom[l] == chrom[l - 1]);
            if (same) {
                const double d = genpos_morgans[l] - genpos_morgans[l - 1];
                gap_left = d > 0.0 ? d : 0.0;
            }
        }
        if (l + 1 < M) {
            const bool same = !have_chrom || (chrom[l] == chrom[l + 1]);
            if (same) {
                const double d = genpos_morgans[l + 1] - genpos_morgans[l];
                gap_right = d > 0.0 ? d : 0.0;
            }
        }
        w[l] = 0.5 * (gap_left + gap_right);
    }
    return w;
}

std::vector<double> build_uniform_pi(int K, int self) {
    if (K <= 0) return {};
    std::vector<double> pi(static_cast<std::size_t>(K), 0.0);
    if (self >= 0 && self < K && K > 1) {
        const double p = 1.0 / static_cast<double>(K - 1);
        for (int k = 0; k < K; ++k)
            pi[static_cast<std::size_t>(k)] = (k == self) ? 0.0 : p;
    } else {
        const double p = 1.0 / static_cast<double>(K);
        for (int k = 0; k < K; ++k) pi[static_cast<std::size_t>(k)] = p;
    }
    return pi;
}

}  // namespace steppe::core

// src/core/stats/li_stephens.hpp
//
// Host-side model helpers for the Li-Stephens haplotype-copying engine (`steppe
// paint`) — the CUDA-free numerics that turn a genotype triple's cM map + panel size
// into the forward-backward inputs (recombination probabilities rho, the copying
// prior pi, the emission/mutation rate theta) and that map a packed haploid genotype
// code to its {0,1} allele. Pure host C++20, trivially unit-testable.
//
// The forward-backward itself is a ComputeBackend virtual (ls_forward_backward); in
// Phase 0 only the CpuBackend reference oracle implements it. The full streaming
// paint pipeline (tile loop over the panel + the coancestry face) lands in Phase 1/2;
// this header carries the shared, testable pieces both the reference gate and that
// pipeline build on.
//
// Reference: docs/planning/li-stephens-engine-scope.md §2a, §2c
#ifndef STEPPE_CORE_STATS_LI_STEPHENS_HPP
#define STEPPE_CORE_STATS_LI_STEPHENS_HPP

#include <cstdint>
#include <vector>

namespace steppe::core {

// The haploid allele alphabet the FB emission compares over: 0 / 1, or "missing".
inline constexpr std::uint8_t kLsMissingAllele = 0xFFu;

// haploid_allele_from_code — map a packed 2-bit genotype code {0,1,2,missing} to the
// haploid {0,1} allele. A phased haploid column carries hom-ref (code 0 -> allele 0)
// or hom-alt (code 2 -> allele 1); the missing code -> kLsMissingAllele. The
// heterozygous code 1 must not appear in phased haploid input (the validator rejects
// a diploid triple), so it maps to missing defensively.
[[nodiscard]] std::uint8_t haploid_allele_from_code(std::uint8_t code) noexcept;

// watterson_emission_rate — the Li & Stephens (2003) per-site miscopy ("mutation")
// rate over a panel of K donors: theta_hat = 1 / H_{K-1} with H the (K-1)th harmonic
// number, and the emission rate mu = (1/2) * theta_hat / (theta_hat + K). This is the
// `--theta auto` default; a user-supplied --theta overrides it. Returns 0 for K < 2.
[[nodiscard]] double watterson_emission_rate(int K) noexcept;

// ls_recomb_prob — the recombination ("switch") probability across one SNP gap for a
// panel of K donors: with an expected 4*Ne*Δg pairwise recombinations over a gap of
// Δg Morgans, rho = 1 - exp(-4*Ne*Δg / K), clamped to [0,1]. `delta_morgans` is the
// same-chromosome cM gap (>= 0); pass a boundary as unlinked via the caller (rho=1).
// Reference: docs/planning/li-stephens-engine-scope.md §2a.
[[nodiscard]] double ls_recomb_prob(double delta_morgans, double Ne, int K) noexcept;

// build_recomb_probs — the full per-SNP rho vector over an ordered SNP list. rho[0] is
// 1.0 (the first column has no predecessor -> fully unlinked, alpha_0 = pi*e_0), and
// any chromosome change resets rho to 1.0 (unlinked across the boundary, the DATES
// inter-chromosome convention). `chrom` and `genpos_morgans` are the .snp columns in
// SNP order; both must be length M. Ne > 0, K >= 1.
[[nodiscard]] std::vector<double> build_recomb_probs(const std::vector<int>& chrom,
                                                     const std::vector<double>& genpos_morgans,
                                                     double Ne, int K);

// build_uniform_pi — the copying prior over K donors. Uniform 1/K, unless a
// leave-one-out self index is given (0 <= self < K): that donor's prior is 0 and the
// remaining K-1 share 1/(K-1) (panel-vs-self painting, §3 decision 4). `self < 0`
// means no leave-one-out. Returns K probabilities summing to 1.
[[nodiscard]] std::vector<double> build_uniform_pi(int K, int self);

}  // namespace steppe::core

#endif  // STEPPE_CORE_STATS_LI_STEPHENS_HPP

// src/core/internal/admix_init.hpp
//
// Deterministic, seeded init for the ADMIXTURE Q/F EM (`steppe admixture`), shared
// verbatim by the CUDA backend host code and the CpuBackend reference oracle so the two
// paths init from the identical Q/F and cannot drift on the RNG. Pure host math (no CUDA,
// no cuRAND) — a splitmix64 counter stream feeds a Dirichlet(1,...,1) simplex draw for the
// per-individual Q rows and a per-SNP-mean + jitter draw for the F columns.
//
// Layout note: both Q and F are produced COLUMN-MAJOR to hand straight to cuBLAS
//   Q(i,k) = Qcm[i + N*k]   (rows on the simplex: sum_k Q(i,k) = 1)
//   F(s,k) = Fcm[s + M*k]   (allele-2 frequencies clamped to [eps, 1-eps])
#ifndef STEPPE_CORE_INTERNAL_ADMIX_INIT_HPP
#define STEPPE_CORE_INTERNAL_ADMIX_INIT_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace steppe::core {

// splitmix64 — the counter-based reproducible substream generator (the architecture's
// stated Philox-class intent; no cuRAND linked, so a small in-tree stream stands in).
[[nodiscard]] inline std::uint64_t admix_splitmix64(std::uint64_t& x) noexcept {
    x += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// A uniform double in [0, 1).
[[nodiscard]] inline double admix_next_unit(std::uint64_t& s) noexcept {
    return static_cast<double>(admix_splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
}

// Fill Q [N x K] column-major with per-row Dirichlet(1,...,1) draws (uniform on the simplex
// via normalized Exp(1) variates), so every row is >= 0 and sums to 1.
inline void admix_init_q(std::uint64_t seed, long N, int K, double* Qcm) {
    std::uint64_t s = seed ^ 0x0123456789ABCDEFull;
    std::vector<double> r(static_cast<std::size_t>(K), 0.0);
    for (long i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            const double u = admix_next_unit(s);
            const double e = -std::log(1.0 - (u < 1.0 ? u : 0.999999999999));  // Exp(1)
            r[static_cast<std::size_t>(k)] = e;
            sum += e;
        }
        if (!(sum > 0.0)) sum = 1.0;
        for (int k = 0; k < K; ++k)
            Qcm[static_cast<std::size_t>(i) + static_cast<std::size_t>(N) * static_cast<std::size_t>(k)] =
                r[static_cast<std::size_t>(k)] / sum;
    }
}

// Fill F [M x K] column-major from the per-SNP mean allele-2 frequency phat[M] plus a small
// per-(SNP,cluster) jitter so the K columns start distinct, clamped to [eps, 1-eps].
inline void admix_init_f(std::uint64_t seed, long M, int K, const double* phat, double eps,
                         double* Fcm) {
    std::uint64_t s = seed ^ 0x00D1CE5B9C0FFEE1ull;
    for (long j = 0; j < M; ++j) {
        const double p = phat[static_cast<std::size_t>(j)];
        for (int k = 0; k < K; ++k) {
            const double jit = (admix_next_unit(s) - 0.5) * 0.1;
            double f = p + jit;
            if (f < eps) f = eps;
            if (f > 1.0 - eps) f = 1.0 - eps;
            Fcm[static_cast<std::size_t>(j) + static_cast<std::size_t>(M) * static_cast<std::size_t>(k)] = f;
        }
    }
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_ADMIX_INIT_HPP

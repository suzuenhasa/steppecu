// src/core/stats/pcangsd_em.hpp
//
// Host reference for the PCAngsd (Meisner & Albrechtsen 2018) individual-allele-
// frequency EM + GL-weighted covariance + top-K PCA. This is the correctness
// ORACLE for `steppe pcangsd`: the CpuBackend override and the unit test both call
// it, and the CUDA kernels reproduce the SAME math on the device (gated by
// concordance, since pcangsd uses float32 internally).
//
// Ported verbatim from Rosemeis/pcangsd (covariance_cy.pyx / shared_cy.pyx):
//   emMAF        : per-site population allele-2 freq EM, init 0.25, f = (1/2N) Σ_i
//                  E_dosage(L, f); iterate to rmse1d < maf_tol or maf_iter.
//   updateNormal : E[j,i] = E_dosage(L, f_j) - 2 f_j   (weights use f_j; centered by f_j)
//   estimatePi   : V = top-K right singular vectors of E (= eigvecs of E^T E);
//                  P = E V V^T   (rank-K reconstruction)
//   updatePCAngsd: pi = clip((P[j,i]+2 f_j)/2, 1e-4, 1-1e-4); weights use pi; E
//                  centered by f_j (NOT pi).  <-- the critic-verified pin.
//   covPCAngsd   : E_std[j,i] = (E_dosage(L, pi) - 2 f_j) * dj, dj = 1/sqrt(2 f_j(1-f_j));
//                  C = E_std^T E_std with the diagonal REPLACED by dCov[i], then C *= 1/M.
//
// ALL frequencies below are allele-2 (minor / non-reference) — the tile stores
// l[base+g] = P(g copies of A1), so for site j / sample i:
//   Lrr = l[base+2] (0 copies A2), Lhet = l[base+1], Laa = l[base+0] (2 copies A2).
//
// Native double throughout (the GL path's cancellation carve-out; not emulated).
#ifndef STEPPE_CORE_STATS_PCANGSD_EM_HPP
#define STEPPE_CORE_STATS_PCANGSD_EM_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace steppe::core {

struct PcangsdRef {
    int N = 0;                         // individuals
    int e = 0;                         // PCs reported
    long M_used = 0;                   // sites kept after the MAF filter
    long M_total = 0;                  // sites in the input
    int iters_run = 0;
    double final_rmse = 0.0;
    std::vector<double> cov;           // N*N row-major (the .cov)
    std::vector<double> coords;        // N*e row-major (eigvec*sqrt(eval), top-e descending)
    std::vector<double> eigenvalues;   // e, descending
    std::vector<double> var_explained; // e, ratio over all N eigenvalues of C
    std::vector<double> freq;          // M_used, population allele-2 freq (kept sites)
    std::vector<double> pi;            // M_used*N individual allele-2 freq (optional)
    bool ok = true;
};

namespace pcangsd_detail {

// Expected allele-2 dosage E[g|L,q] under the HWE prior (q^2, 2q(1-q), (1-q)^2),
// returning the posterior weights so callers can also form the dCov terms.
inline void posterior(double Lrr, double Lhet, double Laa, double q, double& p0, double& p1,
                      double& p2, double& psum, double& edos) {
    const double omq = 1.0 - q;
    p0 = Lrr * omq * omq;   // 0 copies of A2
    p1 = Lhet * 2.0 * q * omq;
    p2 = Laa * q * q;       // 2 copies of A2
    psum = p0 + p1 + p2;
    edos = (psum > 0.0) ? (p1 + 2.0 * p2) / psum : 2.0 * q;
}

// Cyclic-Jacobi symmetric eigensolve of the N x N matrix A (row-major, overwritten).
// On return evals[] (length N, ascending after the caller sorts) hold the diagonal
// and V (row-major, columns are eigenvectors) holds the rotations.
inline void jacobi_eigen(std::vector<double>& A, int N, std::vector<double>& V) {
    const std::size_t Nz = static_cast<std::size_t>(N);
    V.assign(Nz * Nz, 0.0);
    for (int i = 0; i < N; ++i) V[static_cast<std::size_t>(i) * Nz + static_cast<std::size_t>(i)] = 1.0;
    auto at = [Nz](std::vector<double>& X, int r, int c) -> double& {
        return X[static_cast<std::size_t>(r) * Nz + static_cast<std::size_t>(c)];
    };
    constexpr int kMaxSweeps = 100;
    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < N; ++p)
            for (int q = p + 1; q < N; ++q) off += at(A, p, q) * at(A, p, q);
        if (off <= 1e-30) break;
        for (int p = 0; p < N; ++p) {
            for (int q = p + 1; q < N; ++q) {
                const double apq = at(A, p, q);
                if (apq == 0.0) continue;
                const double phi = 0.5 * (at(A, q, q) - at(A, p, p)) / apq;
                const double t = (phi >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(phi) + std::sqrt(phi * phi + 1.0));
                const double cs = 1.0 / std::sqrt(t * t + 1.0);
                const double sn = t * cs;
                for (int i = 0; i < N; ++i) {
                    const double aip = at(A, i, p);
                    const double aiq = at(A, i, q);
                    at(A, i, p) = cs * aip - sn * aiq;
                    at(A, i, q) = sn * aip + cs * aiq;
                }
                for (int i = 0; i < N; ++i) {
                    const double api = at(A, p, i);
                    const double aqi = at(A, q, i);
                    at(A, p, i) = cs * api - sn * aqi;
                    at(A, q, i) = sn * api + cs * aqi;
                }
                for (int i = 0; i < N; ++i) {
                    const double vip = at(V, i, p);
                    const double viq = at(V, i, q);
                    at(V, i, p) = cs * vip - sn * viq;
                    at(V, i, q) = sn * vip + cs * viq;
                }
            }
        }
    }
}

}  // namespace pcangsd_detail

// Full PCAngsd reference. l is site-major [n_site*n_sample*3], g = copies of A1.
inline PcangsdRef pcangsd_reference(const double* l, long n_site, int n_sample, int e,
                                    int max_iter, double tol, double maf, int maf_iter,
                                    double maf_tol, bool want_pi) {
    using namespace pcangsd_detail;
    PcangsdRef out;
    out.M_total = n_site;
    const long M = n_site;
    const int N = n_sample;
    if (M <= 0 || N <= 0 || e <= 0) { out.ok = false; return out; }

    auto base = [N](long j, int i) -> std::size_t {
        return (static_cast<std::size_t>(j) * static_cast<std::size_t>(N) +
                static_cast<std::size_t>(i)) * 3;
    };
    auto Lrr = [&](long j, int i) { return l[base(j, i) + 2]; };
    auto Lhet = [&](long j, int i) { return l[base(j, i) + 1]; };
    auto Laa = [&](long j, int i) { return l[base(j, i) + 0]; };

    // --- emMAF: per-site population allele-2 frequency ------------------------
    std::vector<double> f_all(static_cast<std::size_t>(M), 0.25);
    const double invN2 = 1.0 / (2.0 * static_cast<double>(N));
    for (long j = 0; j < M; ++j) {
        double fj = 0.25;
        for (int it = 0; it < maf_iter; ++it) {
            double acc = 0.0;
            for (int i = 0; i < N; ++i) {
                double p0, p1, p2, ps, ed;
                posterior(Lrr(j, i), Lhet(j, i), Laa(j, i), fj, p0, p1, p2, ps, ed);
                acc += ed;
            }
            const double fnew = acc * invN2;
            const double d = fnew - fj;
            fj = fnew;
            if (std::sqrt(d * d) < maf_tol) break;  // rmse1d over a scalar
        }
        f_all[static_cast<std::size_t>(j)] = fj;
    }

    // --- MAF filter: keep sites with min(f, 1-f) >= maf ----------------------
    std::vector<long> kept;
    kept.reserve(static_cast<std::size_t>(M));
    for (long j = 0; j < M; ++j) {
        const double fj = f_all[static_cast<std::size_t>(j)];
        if (std::min(fj, 1.0 - fj) >= maf) kept.push_back(j);
    }
    const long Mw = static_cast<long>(kept.size());
    out.M_used = Mw;
    if (Mw <= 0) { out.ok = false; return out; }
    const int K = std::min(e, N);

    std::vector<double> fk(static_cast<std::size_t>(Mw));
    for (long j = 0; j < Mw; ++j) fk[static_cast<std::size_t>(j)] = f_all[static_cast<std::size_t>(kept[static_cast<std::size_t>(j)])];
    out.freq = fk;

    const std::size_t Nz = static_cast<std::size_t>(N);
    const std::size_t MN = static_cast<std::size_t>(Mw) * Nz;

    // Emat[j,i] (M x N) and the rank-K reconstruction P[j,i].
    std::vector<double> Emat(MN, 0.0);
    std::vector<double> P(MN, 0.0);
    std::vector<double> Pprev(MN, 0.0);

    // build_E(use_pi): weights use pi (from P) when use_pi, else f; centered by f.
    auto build_E = [&](bool use_pi) {
        for (long jj = 0; jj < Mw; ++jj) {
            const long j = kept[static_cast<std::size_t>(jj)];
            const double fj = fk[static_cast<std::size_t>(jj)];
            for (int i = 0; i < N; ++i) {
                const std::size_t idx = static_cast<std::size_t>(jj) * Nz + static_cast<std::size_t>(i);
                double q = fj;
                if (use_pi) {
                    q = (P[idx] + 2.0 * fj) * 0.5;
                    q = std::min(std::max(q, 1e-4), 1.0 - 1e-4);
                }
                double p0, p1, p2, ps, ed;
                posterior(Lrr(j, i), Lhet(j, i), Laa(j, i), q, p0, p1, p2, ps, ed);
                Emat[idx] = ed - 2.0 * fj;
            }
        }
    };

    // gram G = Emat^T Emat (N x N), then top-K eigenvectors V (N x K), then P = Emat V V^T.
    std::vector<double> Gvec(Nz * Nz, 0.0);
    auto gram_reconstruct = [&]() {
        std::fill(Gvec.begin(), Gvec.end(), 0.0);
        for (long jj = 0; jj < Mw; ++jj) {
            const std::size_t r = static_cast<std::size_t>(jj) * Nz;
            for (int a = 0; a < N; ++a) {
                const double ea = Emat[r + static_cast<std::size_t>(a)];
                if (ea == 0.0) continue;
                for (int b = 0; b < N; ++b)
                    Gvec[static_cast<std::size_t>(a) * Nz + static_cast<std::size_t>(b)] +=
                        ea * Emat[r + static_cast<std::size_t>(b)];
            }
        }
        std::vector<double> Gtmp = Gvec;
        std::vector<double> V;
        jacobi_eigen(Gtmp, N, V);
        std::vector<double> evals(Nz);
        for (int i = 0; i < N; ++i) evals[static_cast<std::size_t>(i)] = Gtmp[static_cast<std::size_t>(i) * Nz + static_cast<std::size_t>(i)];
        std::vector<int> order(Nz);
        for (int i = 0; i < N; ++i) order[static_cast<std::size_t>(i)] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return evals[static_cast<std::size_t>(a)] > evals[static_cast<std::size_t>(b)];
        });
        // W = V_K V_K^T (N x N) over the top-K eigenvectors.
        std::vector<double> W(Nz * Nz, 0.0);
        for (int a = 0; a < N; ++a)
            for (int b = 0; b < N; ++b) {
                double w = 0.0;
                for (int kk = 0; kk < K; ++kk) {
                    const int col = order[static_cast<std::size_t>(kk)];
                    w += V[static_cast<std::size_t>(a) * Nz + static_cast<std::size_t>(col)] *
                         V[static_cast<std::size_t>(b) * Nz + static_cast<std::size_t>(col)];
                }
                W[static_cast<std::size_t>(a) * Nz + static_cast<std::size_t>(b)] = w;
            }
        // P = Emat W  (M x N).
        for (long jj = 0; jj < Mw; ++jj) {
            const std::size_t r = static_cast<std::size_t>(jj) * Nz;
            for (int i = 0; i < N; ++i) {
                double acc = 0.0;
                for (int b = 0; b < N; ++b)
                    acc += Emat[r + static_cast<std::size_t>(b)] *
                           W[static_cast<std::size_t>(b) * Nz + static_cast<std::size_t>(i)];
                P[r + static_cast<std::size_t>(i)] = acc;
            }
        }
    };

    // --- init: updateNormal (centered by f, weights by f) then estimatePi ----
    build_E(/*use_pi=*/false);
    gram_reconstruct();

    // --- main loop -----------------------------------------------------------
    // Plain fixed-point EM (linear convergence). The reference pcangsd accelerates the
    // same update with SQUAREM, converging in ~1/8 the iterations to the SAME fixed
    // point; at the default tol this loop commonly runs to max_iter. Kept unaccelerated
    // for the cleanest oracle match; SQUAREM is a documented follow-up.
    int it = 0;
    double rmse = 0.0;
    for (it = 0; it < max_iter; ++it) {
        Pprev = P;
        build_E(/*use_pi=*/true);
        gram_reconstruct();
        double s = 0.0;
        for (std::size_t idx = 0; idx < MN; ++idx) {
            const double d = P[idx] - Pprev[idx];
            s += d * d;
        }
        rmse = 0.5 * std::sqrt(s / static_cast<double>(MN));  // pi-scale rmse2d
        if (rmse < tol) { ++it; break; }
    }
    out.iters_run = it;
    out.final_rmse = rmse;

    // --- final covariance: covPCAngsd (standardized E) + dCov diagonal -------
    std::vector<double> C(Nz * Nz, 0.0);
    std::vector<double> dCov(Nz, 0.0);
    std::vector<double> Estd(Nz, 0.0);
    if (want_pi) out.pi.assign(MN, 0.0);
    for (long jj = 0; jj < Mw; ++jj) {
        const long j = kept[static_cast<std::size_t>(jj)];
        const double fj = fk[static_cast<std::size_t>(jj)];
        const double dj = 1.0 / std::sqrt(2.0 * fj * (1.0 - fj));
        const double denom = 2.0 * fj * (1.0 - fj);
        for (int i = 0; i < N; ++i) {
            const std::size_t idx = static_cast<std::size_t>(jj) * Nz + static_cast<std::size_t>(i);
            double q = (P[idx] + 2.0 * fj) * 0.5;
            q = std::min(std::max(q, 1e-4), 1.0 - 1e-4);
            if (want_pi) out.pi[idx] = q;
            double p0, p1, p2, ps, ed;
            posterior(Lrr(j, i), Lhet(j, i), Laa(j, i), q, p0, p1, p2, ps, ed);
            Estd[static_cast<std::size_t>(i)] = (ed - 2.0 * fj) * dj;
            if (ps > 0.0) {
                const double t0 = (-2.0 * fj) * (-2.0 * fj) * (p0 / ps);
                const double t1 = (1.0 - 2.0 * fj) * (1.0 - 2.0 * fj) * (p1 / ps);
                const double t2 = (2.0 - 2.0 * fj) * (2.0 - 2.0 * fj) * (p2 / ps);
                dCov[static_cast<std::size_t>(i)] += (t0 + t1 + t2) / denom;
            }
        }
        for (int a = 0; a < N; ++a) {
            const double ea = Estd[static_cast<std::size_t>(a)];
            for (int b = 0; b < N; ++b)
                C[static_cast<std::size_t>(a) * Nz + static_cast<std::size_t>(b)] +=
                    ea * Estd[static_cast<std::size_t>(b)];
        }
    }
    for (int i = 0; i < N; ++i)
        C[static_cast<std::size_t>(i) * Nz + static_cast<std::size_t>(i)] = dCov[static_cast<std::size_t>(i)];
    const double invM = 1.0 / static_cast<double>(Mw);
    for (std::size_t idx = 0; idx < Nz * Nz; ++idx) C[idx] *= invM;
    out.cov = C;

    // --- eigendecompose C, project top-e coords ------------------------------
    std::vector<double> Ctmp = C;
    std::vector<double> Vc;
    jacobi_eigen(Ctmp, N, Vc);
    std::vector<double> cevals(Nz);
    for (int i = 0; i < N; ++i) cevals[static_cast<std::size_t>(i)] = Ctmp[static_cast<std::size_t>(i) * Nz + static_cast<std::size_t>(i)];
    std::vector<int> corder(Nz);
    for (int i = 0; i < N; ++i) corder[static_cast<std::size_t>(i)] = i;
    std::sort(corder.begin(), corder.end(), [&](int a, int b) {
        return cevals[static_cast<std::size_t>(a)] > cevals[static_cast<std::size_t>(b)];
    });
    double total = 0.0;
    for (int i = 0; i < N; ++i) total += cevals[static_cast<std::size_t>(i)];
    out.coords.assign(Nz * static_cast<std::size_t>(K), 0.0);
    out.eigenvalues.assign(static_cast<std::size_t>(K), 0.0);
    out.var_explained.assign(static_cast<std::size_t>(K), 0.0);
    for (int kk = 0; kk < K; ++kk) {
        const int col = corder[static_cast<std::size_t>(kk)];
        const double lam = cevals[static_cast<std::size_t>(col)];
        const double sc = lam > 0.0 ? std::sqrt(lam) : 0.0;
        out.eigenvalues[static_cast<std::size_t>(kk)] = lam;
        out.var_explained[static_cast<std::size_t>(kk)] = (total != 0.0) ? lam / total : 0.0;
        for (int i = 0; i < N; ++i)
            out.coords[static_cast<std::size_t>(i) * static_cast<std::size_t>(K) + static_cast<std::size_t>(kk)] =
                Vc[static_cast<std::size_t>(i) * Nz + static_cast<std::size_t>(col)] * sc;
    }
    out.N = N;
    out.e = K;
    out.ok = true;
    return out;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_STATS_PCANGSD_EM_HPP

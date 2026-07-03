// src/core/qpadm/qpgraph_objective.hpp
//
// Host (CPU) reference for the qpGraph GLS fit objective: given a graph topology and
// mixture proportions, score how well the graph reproduces the observed f-statistics.
// The device kernel mirrors this math per-thread; this header is the authoritative
// statement of it, matching ADMIXTOOLS 2's optimweightsfun step for step.
//
// Reference: docs/reference/src_core_qpadm_qpgraph_objective.hpp.md
#ifndef STEPPE_CORE_QPADM_QPGRAPH_OBJECTIVE_HPP
#define STEPPE_CORE_QPADM_QPGRAPH_OBJECTIVE_HPP

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/internal/small_linalg.hpp"
#include "core/qpadm/qpgraph_model.hpp"

namespace steppe::core::qpadm {

// build_ppwts_2d — reference §3
inline void build_ppwts_2d(const QpGraphModel& m, const std::vector<double>& pwts_c,
                           std::vector<double>& ppwts) {
    const int ne = m.nedge_norm, np = m.npair;
    ppwts.assign(static_cast<std::size_t>(np) * static_cast<std::size_t>(ne), 0.0);
    for (int k = 0; k < np; ++k) {
        const int ca = m.cmb1[static_cast<std::size_t>(k)];
        const int cb = m.cmb2[static_cast<std::size_t>(k)];
        for (int e = 0; e < ne; ++e) {
            const double wa = pwts_c[static_cast<std::size_t>(e) + static_cast<std::size_t>(ne) * static_cast<std::size_t>(ca)];
            const double wb = pwts_c[static_cast<std::size_t>(e) + static_cast<std::size_t>(ne) * static_cast<std::size_t>(cb)];
            ppwts[static_cast<std::size_t>(k) + static_cast<std::size_t>(np) * static_cast<std::size_t>(e)] = wa * wb;
        }
    }
}

// nnls_active_set — reference §4
inline bool nnls_active_set(const std::vector<double>& A, const std::vector<double>& q,
                            int n, std::vector<double>& bl) {
    bl.assign(static_cast<std::size_t>(n), 0.0);
    std::vector<char> passive(static_cast<std::size_t>(n), 0);
    const double nnls_eps = 1e-12;
    const int max_iter = 3 * n + 30;
    for (int outer = 0; outer < max_iter; ++outer) {
        std::vector<double> w(static_cast<std::size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            double s = q[static_cast<std::size_t>(i)];
            for (int j = 0; j < n; ++j)
                s -= A[static_cast<std::size_t>(i) + static_cast<std::size_t>(n) * static_cast<std::size_t>(j)] * bl[static_cast<std::size_t>(j)];
            w[static_cast<std::size_t>(i)] = s;
        }
        int t = -1; double best = nnls_eps;
        for (int i = 0; i < n; ++i)
            if (!passive[static_cast<std::size_t>(i)] && w[static_cast<std::size_t>(i)] > best) { best = w[static_cast<std::size_t>(i)]; t = i; }
        if (t < 0) return true;
        passive[static_cast<std::size_t>(t)] = 1;
        for (int inner = 0; inner < max_iter; ++inner) {
            std::vector<int> P;
            for (int i = 0; i < n; ++i) if (passive[static_cast<std::size_t>(i)]) P.push_back(i);
            const int p = static_cast<int>(P.size());
            std::vector<double> Ap(static_cast<std::size_t>(p) * static_cast<std::size_t>(p), 0.0), qp(static_cast<std::size_t>(p), 0.0);
            for (int a = 0; a < p; ++a) {
                qp[static_cast<std::size_t>(a)] = q[static_cast<std::size_t>(P[static_cast<std::size_t>(a)])];
                for (int b = 0; b < p; ++b)
                    Ap[static_cast<std::size_t>(a) + static_cast<std::size_t>(p) * static_cast<std::size_t>(b)] =
                        A[static_cast<std::size_t>(P[static_cast<std::size_t>(a)]) + static_cast<std::size_t>(n) * static_cast<std::size_t>(P[static_cast<std::size_t>(b)])];
            }
            std::vector<double> z;
            const LinAlgStatus st = solve(Ap, p, qp, z);
            if (!st.ok) return false;
            bool all_pos = true;
            for (int a = 0; a < p; ++a) if (z[static_cast<std::size_t>(a)] <= nnls_eps) { all_pos = false; break; }
            if (all_pos) {
                for (int i = 0; i < n; ++i) bl[static_cast<std::size_t>(i)] = 0.0;
                for (int a = 0; a < p; ++a) bl[static_cast<std::size_t>(P[static_cast<std::size_t>(a)])] = z[static_cast<std::size_t>(a)];
                break;
            }
            double alpha = 1.0;
            for (int a = 0; a < p; ++a) {
                if (z[static_cast<std::size_t>(a)] <= nnls_eps) {
                    const double blp = bl[static_cast<std::size_t>(P[static_cast<std::size_t>(a)])];
                    const double denom = blp - z[static_cast<std::size_t>(a)];
                    if (denom > nnls_eps) { const double r = blp / denom; if (r < alpha) alpha = r; }
                }
            }
            for (int a = 0; a < p; ++a) {
                const int idx = P[static_cast<std::size_t>(a)];
                bl[static_cast<std::size_t>(idx)] += alpha * (z[static_cast<std::size_t>(a)] - bl[static_cast<std::size_t>(idx)]);
            }
            for (int i = 0; i < n; ++i)
                if (passive[static_cast<std::size_t>(i)] && bl[static_cast<std::size_t>(i)] <= nnls_eps) {
                    passive[static_cast<std::size_t>(i)] = 0; bl[static_cast<std::size_t>(i)] = 0.0;
                }
        }
    }
    return true;
}

// opt_edge_lengths — reference §5
inline bool opt_edge_lengths(const std::vector<double>& ppwts, const std::vector<double>& ppinv,
                             const std::vector<double>& f_obs, int npair, int nedge,
                             double fudge, bool constrained, std::vector<double>& bl) {
    const int np = npair, ne = nedge;
    std::vector<double> W(static_cast<std::size_t>(np) * static_cast<std::size_t>(ne), 0.0);
    for (int e = 0; e < ne; ++e)
        for (int r = 0; r < np; ++r) {
            double acc = 0.0;
            for (int c = 0; c < np; ++c)
                acc += ppinv[static_cast<std::size_t>(r) + static_cast<std::size_t>(np) * static_cast<std::size_t>(c)] *
                       ppwts[static_cast<std::size_t>(c) + static_cast<std::size_t>(np) * static_cast<std::size_t>(e)];
            W[static_cast<std::size_t>(r) + static_cast<std::size_t>(np) * static_cast<std::size_t>(e)] = acc;
        }
    std::vector<double> cc(static_cast<std::size_t>(ne) * static_cast<std::size_t>(ne), 0.0), q(static_cast<std::size_t>(ne), 0.0);
    for (int e1 = 0; e1 < ne; ++e1) {
        for (int e2 = 0; e2 < ne; ++e2) {
            double acc = 0.0;
            for (int r = 0; r < np; ++r)
                acc += ppwts[static_cast<std::size_t>(r) + static_cast<std::size_t>(np) * static_cast<std::size_t>(e1)] *
                       W[static_cast<std::size_t>(r) + static_cast<std::size_t>(np) * static_cast<std::size_t>(e2)];
            cc[static_cast<std::size_t>(e1) + static_cast<std::size_t>(ne) * static_cast<std::size_t>(e2)] = acc;
        }
        double rr = 0.0;
        for (int r = 0; r < np; ++r) {
            double qf = 0.0;
            for (int c = 0; c < np; ++c)
                qf += ppinv[static_cast<std::size_t>(r) + static_cast<std::size_t>(np) * static_cast<std::size_t>(c)] * f_obs[static_cast<std::size_t>(c)];
            rr += ppwts[static_cast<std::size_t>(r) + static_cast<std::size_t>(np) * static_cast<std::size_t>(e1)] * qf;
        }
        q[static_cast<std::size_t>(e1)] = rr;
    }
    double trm = 0.0;
    for (int e = 0; e < ne; ++e) trm += cc[static_cast<std::size_t>(e) + static_cast<std::size_t>(ne) * static_cast<std::size_t>(e)];
    const double ridge = fudge * trm / static_cast<double>(ne);
    for (int e = 0; e < ne; ++e) cc[static_cast<std::size_t>(e) + static_cast<std::size_t>(ne) * static_cast<std::size_t>(e)] += ridge;
    std::vector<double> sc(static_cast<std::size_t>(ne));
    for (int e = 0; e < ne; ++e) sc[static_cast<std::size_t>(e)] = std::sqrt(cc[static_cast<std::size_t>(e) + static_cast<std::size_t>(ne) * static_cast<std::size_t>(e)]);
    std::vector<double> ccs(static_cast<std::size_t>(ne) * static_cast<std::size_t>(ne)), q1(static_cast<std::size_t>(ne));
    for (int e1 = 0; e1 < ne; ++e1) {
        q1[static_cast<std::size_t>(e1)] = q[static_cast<std::size_t>(e1)] / sc[static_cast<std::size_t>(e1)];
        for (int e2 = 0; e2 < ne; ++e2)
            ccs[static_cast<std::size_t>(e1) + static_cast<std::size_t>(ne) * static_cast<std::size_t>(e2)] =
                cc[static_cast<std::size_t>(e1) + static_cast<std::size_t>(ne) * static_cast<std::size_t>(e2)] /
                (sc[static_cast<std::size_t>(e1)] * sc[static_cast<std::size_t>(e2)]);
    }
    std::vector<double> bls;
    if (constrained) {
        if (!nnls_active_set(ccs, q1, ne, bls)) return false;
    } else {
        const LinAlgStatus st = solve(ccs, ne, q1, bls);
        if (!st.ok) return false;
    }
    bl.assign(static_cast<std::size_t>(ne), 0.0);
    for (int e = 0; e < ne; ++e) bl[static_cast<std::size_t>(e)] = bls[static_cast<std::size_t>(e)] / sc[static_cast<std::size_t>(e)];
    return true;
}

// qpgraph_score — reference §6
inline double qpgraph_score(const QpGraphModel& m, const double* theta,
                            const std::vector<double>& f_obs, const std::vector<double>& ppinv,
                            double fudge, bool constrained,
                            std::vector<double>* out_bl = nullptr,
                            std::vector<double>* out_fit = nullptr) {
    std::vector<double> pwts_c;
    fill_pwts_centered(m, theta, pwts_c);
    std::vector<double> ppwts;
    build_ppwts_2d(m, pwts_c, ppwts);
    std::vector<double> bl;
    if (!opt_edge_lengths(ppwts, ppinv, f_obs, m.npair, m.nedge_norm, fudge, constrained, bl))
        return std::numeric_limits<double>::infinity();
    const int np = m.npair, ne = m.nedge_norm;
    std::vector<double> res(static_cast<std::size_t>(np));
    for (int k = 0; k < np; ++k) {
        double fit = 0.0;
        for (int e = 0; e < ne; ++e)
            fit += ppwts[static_cast<std::size_t>(k) + static_cast<std::size_t>(np) * static_cast<std::size_t>(e)] * bl[static_cast<std::size_t>(e)];
        res[static_cast<std::size_t>(k)] = f_obs[static_cast<std::size_t>(k)] - fit;
        if (out_fit) (*out_fit)[static_cast<std::size_t>(k)] = fit;
    }
    double score = 0.0;
    for (int a = 0; a < np; ++a) {
        double row = 0.0;
        for (int b = 0; b < np; ++b)
            row += ppinv[static_cast<std::size_t>(a) + static_cast<std::size_t>(np) * static_cast<std::size_t>(b)] * res[static_cast<std::size_t>(b)];
        score += res[static_cast<std::size_t>(a)] * row;
    }
    if (out_bl) *out_bl = bl;
    return score;
}

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_QPGRAPH_OBJECTIVE_HPP

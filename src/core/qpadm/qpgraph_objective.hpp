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

#include "core/internal/index_cast.hpp"
#include "core/internal/small_linalg.hpp"
#include "core/qpadm/qpgraph_model.hpp"

namespace steppe::core::qpadm {

// build_ppwts_2d — reference §3
inline void build_ppwts_2d(const QpGraphModel& m, const std::vector<double>& pwts_c,
                           std::vector<double>& ppwts) {
    const int ne = m.nedge_norm, np = m.npair;
    ppwts.assign(idx(np) * idx(ne), 0.0);
    for (int k = 0; k < np; ++k) {
        const int ca = m.cmb1[idx(k)];
        const int cb = m.cmb2[idx(k)];
        for (int e = 0; e < ne; ++e) {
            const double wa = pwts_c[idx(e) + idx(ne) * idx(ca)];
            const double wb = pwts_c[idx(e) + idx(ne) * idx(cb)];
            ppwts[idx(k) + idx(np) * idx(e)] = wa * wb;
        }
    }
}

// nnls_active_set — reference §4
inline bool nnls_active_set(const std::vector<double>& A, const std::vector<double>& q,
                            int n, std::vector<double>& bl) {
    bl.assign(idx(n), 0.0);
    std::vector<char> passive(idx(n), 0);
    const double nnls_eps = 1e-12;
    const int max_iter = 3 * n + 30;
    for (int outer = 0; outer < max_iter; ++outer) {
        std::vector<double> w(idx(n), 0.0);
        for (int i = 0; i < n; ++i) {
            double s = q[idx(i)];
            for (int j = 0; j < n; ++j)
                s -= A[idx(i) + idx(n) * idx(j)] * bl[idx(j)];
            w[idx(i)] = s;
        }
        int t = -1; double best = nnls_eps;
        for (int i = 0; i < n; ++i)
            if (!passive[idx(i)] && w[idx(i)] > best) { best = w[idx(i)]; t = i; }
        if (t < 0) return true;
        passive[idx(t)] = 1;
        for (int inner = 0; inner < max_iter; ++inner) {
            std::vector<int> P;
            for (int i = 0; i < n; ++i) if (passive[idx(i)]) P.push_back(i);
            const int p = static_cast<int>(P.size());
            std::vector<double> Ap(idx(p) * idx(p), 0.0), qp(idx(p), 0.0);
            for (int a = 0; a < p; ++a) {
                qp[idx(a)] = q[idx(P[idx(a)])];
                for (int b = 0; b < p; ++b)
                    Ap[idx(a) + idx(p) * idx(b)] =
                        A[idx(P[idx(a)]) + idx(n) * idx(P[idx(b)])];
            }
            std::vector<double> z;
            const LinAlgStatus st = solve(Ap, p, qp, z);
            if (!st.ok) return false;
            bool all_pos = true;
            for (int a = 0; a < p; ++a) if (z[idx(a)] <= nnls_eps) { all_pos = false; break; }
            if (all_pos) {
                for (int i = 0; i < n; ++i) bl[idx(i)] = 0.0;
                for (int a = 0; a < p; ++a) bl[idx(P[idx(a)])] = z[idx(a)];
                break;
            }
            double alpha = 1.0;
            for (int a = 0; a < p; ++a) {
                if (z[idx(a)] <= nnls_eps) {
                    const double blp = bl[idx(P[idx(a)])];
                    const double denom = blp - z[idx(a)];
                    if (denom > nnls_eps) { const double r = blp / denom; if (r < alpha) alpha = r; }
                }
            }
            for (int a = 0; a < p; ++a) {
                const int pidx = P[idx(a)];
                bl[idx(pidx)] += alpha * (z[idx(a)] - bl[idx(pidx)]);
            }
            for (int i = 0; i < n; ++i)
                if (passive[idx(i)] && bl[idx(i)] <= nnls_eps) {
                    passive[idx(i)] = 0; bl[idx(i)] = 0.0;
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
    std::vector<double> W(idx(np) * idx(ne), 0.0);
    for (int e = 0; e < ne; ++e)
        for (int r = 0; r < np; ++r) {
            double acc = 0.0;
            for (int c = 0; c < np; ++c)
                acc += ppinv[idx(r) + idx(np) * idx(c)] *
                       ppwts[idx(c) + idx(np) * idx(e)];
            W[idx(r) + idx(np) * idx(e)] = acc;
        }
    std::vector<double> cc(idx(ne) * idx(ne), 0.0), q(idx(ne), 0.0);
    for (int e1 = 0; e1 < ne; ++e1) {
        for (int e2 = 0; e2 < ne; ++e2) {
            double acc = 0.0;
            for (int r = 0; r < np; ++r)
                acc += ppwts[idx(r) + idx(np) * idx(e1)] *
                       W[idx(r) + idx(np) * idx(e2)];
            cc[idx(e1) + idx(ne) * idx(e2)] = acc;
        }
        double rr = 0.0;
        for (int r = 0; r < np; ++r) {
            double qf = 0.0;
            for (int c = 0; c < np; ++c)
                qf += ppinv[idx(r) + idx(np) * idx(c)] * f_obs[idx(c)];
            rr += ppwts[idx(r) + idx(np) * idx(e1)] * qf;
        }
        q[idx(e1)] = rr;
    }
    double trm = 0.0;
    for (int e = 0; e < ne; ++e) trm += cc[idx(e) + idx(ne) * idx(e)];
    const double ridge = fudge * trm / static_cast<double>(ne);
    for (int e = 0; e < ne; ++e) cc[idx(e) + idx(ne) * idx(e)] += ridge;
    std::vector<double> sc(idx(ne));
    for (int e = 0; e < ne; ++e) sc[idx(e)] = std::sqrt(cc[idx(e) + idx(ne) * idx(e)]);
    std::vector<double> ccs(idx(ne) * idx(ne)), q1(idx(ne));
    for (int e1 = 0; e1 < ne; ++e1) {
        q1[idx(e1)] = q[idx(e1)] / sc[idx(e1)];
        for (int e2 = 0; e2 < ne; ++e2)
            ccs[idx(e1) + idx(ne) * idx(e2)] =
                cc[idx(e1) + idx(ne) * idx(e2)] /
                (sc[idx(e1)] * sc[idx(e2)]);
    }
    std::vector<double> bls;
    if (constrained) {
        if (!nnls_active_set(ccs, q1, ne, bls)) return false;
    } else {
        const LinAlgStatus st = solve(ccs, ne, q1, bls);
        if (!st.ok) return false;
    }
    bl.assign(idx(ne), 0.0);
    for (int e = 0; e < ne; ++e) bl[idx(e)] = bls[idx(e)] / sc[idx(e)];
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
    std::vector<double> res(idx(np));
    for (int k = 0; k < np; ++k) {
        double fit = 0.0;
        for (int e = 0; e < ne; ++e)
            fit += ppwts[idx(k) + idx(np) * idx(e)] * bl[idx(e)];
        res[idx(k)] = f_obs[idx(k)] - fit;
        if (out_fit) (*out_fit)[idx(k)] = fit;
    }
    double score = 0.0;
    for (int a = 0; a < np; ++a) {
        double row = 0.0;
        for (int b = 0; b < np; ++b)
            row += ppinv[idx(a) + idx(np) * idx(b)] * res[idx(b)];
        score += res[idx(a)] * row;
    }
    if (out_bl) *out_bl = bl;
    return score;
}

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_QPGRAPH_OBJECTIVE_HPP

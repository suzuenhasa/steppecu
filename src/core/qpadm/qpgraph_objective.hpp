// src/core/qpadm/qpgraph_objective.hpp
//
// The qpGraph GLS objective + the IDEA-1 projected-Newton fleet — the HOST reference
// (the CpuBackend oracle body). This is the productization of bench_optimizers.cu's
// host_score / d_graph_gls_score + idea1_fleet_kernel, GENERALIZED to an arbitrary
// topology via the path-table model (fill_pwts_centered) and given the AT2 constrained
// edge solve (the box-constrained NNLS opt_edge_lengths, the spike's unconstrained
// d_solve was insufficient — the golden has a boundary edge at 0). The device kernel
// (qpgraph_fit_kernels.cu) mirrors this math per-thread.
//
// OBJECTIVE (AT2 optimweightsfun, verified):
//   pwts_c   = fill_pwts(theta) centered on base, base column dropped  [nedge x (npop-1)]
//   ppwts_2d[k,e] = pwts_c[e, cmb1[k]] * pwts_c[e, cmb2[k]]            [npair x nedge]
//   bl       = opt_edge_lengths(ppwts_2d, ppinv, f_obs, fudge, constrained)
//   f3_fit   = ppwts_2d * bl
//   score    = (f_obs - f3_fit)' ppinv (f_obs - f3_fit)
//
// opt_edge_lengths (AT2): pppp = ppwts_2d' ppinv; cc = pppp ppwts_2d; diag(cc) +=
//   fudge*mean(diag(cc)); sc=sqrt(diag(cc)); q1=(pppp f_obs)/sc; cc/=sc⊗sc; then
//   constrained ? NNLS(cc, q1, bl>=0)/sc : solve(cc,q1)/sc.
#ifndef STEPPE_CORE_QPADM_QPGRAPH_OBJECTIVE_HPP
#define STEPPE_CORE_QPADM_QPGRAPH_OBJECTIVE_HPP

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/internal/small_linalg.hpp"
#include "core/qpadm/qpgraph_model.hpp"

namespace steppe::core::qpadm {

/// Build ppwts_2d [npair x nedge] COLUMN-MAJOR (ppwts[k + npair*e]) from centered pwts_c.
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

/// Non-negative least squares min 0.5 bl' A bl - bl' q s.t. bl >= 0 (A SPD nedge x nedge
/// col-major, q length nedge). Lawson-Hanson active-set. Returns bl (>=0); ok=false on a
/// singular passive sub-system. This is the AT2 qpsolve(cc,q1,bl>=0) reduction with
/// lower=0/upper=+inf (the golden's constrained mode).
inline bool nnls_active_set(const std::vector<double>& A, const std::vector<double>& q,
                            int n, std::vector<double>& bl) {
    bl.assign(static_cast<std::size_t>(n), 0.0);
    std::vector<char> passive(static_cast<std::size_t>(n), 0);  // in the active (free) set?
    // Shared NNLS tolerance: the KKT gradient-free threshold AND the variable-magnitude
    // floor (passive-drop / ratio test). Parity-frozen value — name only, do NOT change
    // its magnitude (AT2 qpsolve(cc,q1,bl>=0) golden parity; NAMING-STYLE-STANDARD §3.2/§5).
    const double nnls_eps = 1e-12;
    // Active-set iteration cap, shared by the outer KKT loop and the inner passive solve.
    const int max_iter = 3 * n + 30;
    // gradient w = q - A bl; KKT: at the solution, for active(free) vars w==0; for the
    // bound(bl==0) vars w<=0.
    for (int outer = 0; outer < max_iter; ++outer) {
        // w = q - A bl
        std::vector<double> w(static_cast<std::size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            double s = q[static_cast<std::size_t>(i)];
            for (int j = 0; j < n; ++j)
                s -= A[static_cast<std::size_t>(i) + static_cast<std::size_t>(n) * static_cast<std::size_t>(j)] * bl[static_cast<std::size_t>(j)];
            w[static_cast<std::size_t>(i)] = s;
        }
        // pick the bound variable with the most-positive gradient to free.
        int t = -1; double best = nnls_eps;
        for (int i = 0; i < n; ++i)
            if (!passive[static_cast<std::size_t>(i)] && w[static_cast<std::size_t>(i)] > best) { best = w[static_cast<std::size_t>(i)]; t = i; }
        if (t < 0) return true;  // KKT satisfied
        passive[static_cast<std::size_t>(t)] = 1;
        // inner: solve the passive sub-system A_PP z_P = q_P; while any z_P <= 0, ratio-
        // test back toward feasibility and drop the binding var.
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
            // all positive? accept.
            bool all_pos = true;
            for (int a = 0; a < p; ++a) if (z[static_cast<std::size_t>(a)] <= nnls_eps) { all_pos = false; break; }
            if (all_pos) {
                for (int i = 0; i < n; ++i) bl[static_cast<std::size_t>(i)] = 0.0;
                for (int a = 0; a < p; ++a) bl[static_cast<std::size_t>(P[static_cast<std::size_t>(a)])] = z[static_cast<std::size_t>(a)];
                break;  // back to the outer KKT check
            }
            // ratio test: alpha = min over z<=0 of bl/(bl-z).
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
            // drop the now-zero passive vars.
            for (int i = 0; i < n; ++i)
                if (passive[static_cast<std::size_t>(i)] && bl[static_cast<std::size_t>(i)] <= nnls_eps) {
                    passive[static_cast<std::size_t>(i)] = 0; bl[static_cast<std::size_t>(i)] = 0.0;
                }
        }
    }
    return true;
}

/// opt_edge_lengths: returns the fitted edge lengths bl [nedge] for a given ppwts_2d.
/// ok=false on a singular system. `constrained` => NNLS(bl>=0); else the LU solve.
inline bool opt_edge_lengths(const std::vector<double>& ppwts, const std::vector<double>& ppinv,
                             const std::vector<double>& f_obs, int npair, int nedge,
                             double fudge, bool constrained, std::vector<double>& bl) {
    const int np = npair, ne = nedge;
    // pppp = ppwts' ppinv  [ne x np];  cc = pppp ppwts [ne x ne];  q = pppp f_obs [ne].
    std::vector<double> W(static_cast<std::size_t>(np) * static_cast<std::size_t>(ne), 0.0);  // W = ppinv ppwts
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
    // AT2 ridge: diag(cc) += fudge * mean(diag(cc)).
    double trm = 0.0;
    for (int e = 0; e < ne; ++e) trm += cc[static_cast<std::size_t>(e) + static_cast<std::size_t>(ne) * static_cast<std::size_t>(e)];
    const double ridge = fudge * trm / static_cast<double>(ne);
    for (int e = 0; e < ne; ++e) cc[static_cast<std::size_t>(e) + static_cast<std::size_t>(ne) * static_cast<std::size_t>(e)] += ridge;
    // AT2 scaling: sc=sqrt(diag(cc)); q1=q/sc; cc=cc/(sc⊗sc).
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

/// The full GLS objective score(theta) (AT2 optimweightsfun). Returns +inf on a singular
/// inner solve (rejected by the optimizer). Optionally returns bl + f3_fit (the final
/// extras for the result). PRECONDITION: if `out_fit` is non-null the CALLER must pre-size
/// it to `m.npair` — it is written element-wise via `(*out_fit)[k]`, never resized (unlike
/// `*out_bl`, which is assigned).
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

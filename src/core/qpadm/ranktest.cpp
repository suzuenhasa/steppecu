// src/core/qpadm/ranktest.cpp — the M(fit-2) rank-test / qpWave host orchestrator.
//
// The popdrop (AT2 res$popdrop) leave-one-LEFT-SOURCE-out table + the `feasible`
// predicate (§3), reproducing admixtools::drop_pops VERBATIM: subset the rows of
// the already-computed f4_est + subset the already-computed (FUDGED) qinv, then fit
// — NO re-gather, NO re-jackknife. Routes the reduced fit through the SAME backend
// rank_sweep + gls_weights seam, so the CpuBackend oracle and CudaBackend
// deliverable share one path.

#include "core/qpadm/ranktest.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "core/qpadm/qpadm_bounds.hpp"  // qpadm_dof — the single-source (nl-r)*(nr-r) dof
#include "steppe/error.hpp"  // Status

namespace steppe::core::qpadm {

bool popdrop_feasible(const std::vector<double>& weights) {
    bool has_surviving = false;
    for (double w : weights) {
        if (std::isnan(w)) continue;  // a dropped slot — not a constraint
        has_surviving = true;
        if (w < 0.0 || w > 1.0) return false;
    }
    return has_surviving;  // at least one surviving weight, all in [0,1]
}

namespace {

/// Build a REDUCED F4Blocks + JackknifeCov keeping only the left ROWS in `surv`
/// (subset of 0..nl-1), per admixtools::drop_pops: f4_est[surv,] and
/// qinv[ind,ind] with ind = the row-major block indices (surv_row*nr + j). The Q
/// sub-block is carried for rank_Q observability. x_total is row-major k = j+nr*i.
void reduce_rows(const F4Blocks& x, const JackknifeCov& cov,
                 const std::vector<int>& surv, F4Blocks& x_reduced, JackknifeCov& cov_reduced) {
    const int nr = x.nr;
    const int nl_red = static_cast<int>(surv.size());
    const int m_full = x.nl * nr;  // read below as the full-vec stride in the Qinv gather
    const int m_red = nl_red * nr;

    x_reduced.nl = nl_red;
    x_reduced.nr = nr;
    x_reduced.n_block = x.n_block;
    x_reduced.x_total.assign(static_cast<std::size_t>(m_red), 0.0);
    for (int ii = 0; ii < nl_red; ++ii) {
        const int i = surv[static_cast<std::size_t>(ii)];
        for (int j = 0; j < nr; ++j)
            x_reduced.x_total[static_cast<std::size_t>(j + nr * ii)] =
                x.x_total[static_cast<std::size_t>(j + nr * i)];
    }

    // ind[a] = the full-vec index of reduced-vec position a (a = j + nr*ii).
    std::vector<int> ind(static_cast<std::size_t>(m_red), 0);
    for (int ii = 0; ii < nl_red; ++ii) {
        const int i = surv[static_cast<std::size_t>(ii)];
        for (int j = 0; j < nr; ++j)
            ind[static_cast<std::size_t>(j + nr * ii)] = j + nr * i;
    }
    cov_reduced.m = m_red;
    cov_reduced.status = cov.status;
    cov_reduced.Qinv.assign(static_cast<std::size_t>(m_red) * static_cast<std::size_t>(m_red), 0.0);
    cov_reduced.Q.assign(static_cast<std::size_t>(m_red) * static_cast<std::size_t>(m_red), 0.0);
    for (int a = 0; a < m_red; ++a)
        for (int b = 0; b < m_red; ++b) {
            const std::size_t src = static_cast<std::size_t>(ind[static_cast<std::size_t>(a)]) +
                                    static_cast<std::size_t>(m_full) *
                                        static_cast<std::size_t>(ind[static_cast<std::size_t>(b)]);
            const std::size_t dst = static_cast<std::size_t>(a) +
                                    static_cast<std::size_t>(m_red) * static_cast<std::size_t>(b);
            cov_reduced.Qinv[dst] = cov.Qinv[src];
            if (!cov.Q.empty()) cov_reduced.Q[dst] = cov.Q[src];
        }
}

/// One popdrop row over the left rows in `surv` (the surviving sources, in source
/// order); `drop` is the dropped source index (< 0 ⇒ the full model "0...0").
PopDropRow popdrop_one(ComputeBackend& be, const F4Blocks& x, const JackknifeCov& cov,
                       int nl_full, int drop, const std::vector<int>& surv,
                       const QpAdmOptions& opts, const Precision& precision) {
    PopDropRow row;
    row.pat.assign(static_cast<std::size_t>(nl_full), '0');
    if (drop >= 0) row.pat[static_cast<std::size_t>(drop)] = '1';
    row.wt = (drop >= 0) ? 1 : 0;

    F4Blocks x_reduced; JackknifeCov cov_reduced;
    reduce_rows(x, cov, surv, x_reduced, cov_reduced);
    const int nl_red = static_cast<int>(surv.size());

    // AT2 res$popdrop fits EACH (sub-)model at its FULL rank f4rank = len(surv)-1 (the
    // default rank for a qpAdm fit, NOT a rank-decision): the probe over the rotation
    // shows the popdrop `f4rank` column is exactly len(surv)-1 for every row, the dof
    // is (nl_red-r)*(nr-r) with r=nl_red-1, and the weights are the rank-(nl_red-1)
    // fit (the same rank the top-level qpAdm weights use). The rank-DECISION (the
    // smallest non-rejected rank from the sweep) is a SEPARATE quantity, reported on
    // the top-level RankSweep.f4rank — NOT the popdrop column. Earlier this used
    // rs.f4rank, which COINCIDED with len(surv)-1 on the nl=2 goldens (9-pop / NRBIG)
    // but DIVERGED on nl>=3 rotation models (rank-decision 0/1 vs fitted rank 2),
    // mis-fitting the full popdrop row's weights/chisq/feasible. The run_rank_sweep is
    // still issued (its chisq/dof/p at r=nl_red-1 are the reported row values; the
    // f4rank column is the fitted rank).
    const int r_fit = nl_red - 1;
    const RankSweep rs = run_rank_sweep(be, x_reduced, cov_reduced, opts.rank_alpha, opts, precision);
    const std::size_t ri = static_cast<std::size_t>(r_fit < 0 ? 0 : r_fit);
    row.f4rank = r_fit;
    row.dof = (ri < rs.dof.size()) ? rs.dof[ri] : qpadm_dof(x_reduced.nl, x_reduced.nr, r_fit);
    row.chisq = (ri < rs.chisq.size()) ? rs.chisq[ri] : 0.0;
    row.p = (ri < rs.p.size()) ? rs.p[ri] : 0.0;
    row.status = rs.status;

    // Per-source weights at the FITTED rank (length nl_full; NaN for dropped slots).
    row.weight.assign(static_cast<std::size_t>(nl_full),
                      std::numeric_limits<double>::quiet_NaN());
    const GlsWeights gw = gls_weights(be, x_reduced, cov_reduced, r_fit, opts, precision);
    if (gw.status == Status::Ok && gw.w.size() == surv.size()) {
        for (std::size_t s = 0; s < surv.size(); ++s)
            row.weight[static_cast<std::size_t>(surv[s])] = gw.w[s];
    }
    row.feasible = popdrop_feasible(row.weight);
    return row;
}

}  // namespace

std::vector<PopDropRow> run_popdrop(ComputeBackend& be, const F4Blocks& x,
                                    const JackknifeCov& cov, const QpAdmOptions& opts,
                                    const Precision& precision) {
    const int nl_full = x.nl;
    std::vector<PopDropRow> rows;
    if (nl_full <= 0) return rows;
    rows.reserve(static_cast<std::size_t>(nl_full) + 1);

    // The FULL model "0..0" (all rows kept).
    std::vector<int> all;
    for (int i = 0; i < nl_full; ++i) all.push_back(i);
    rows.push_back(popdrop_one(be, x, cov, nl_full, -1, all, opts, precision));

    // Each single-source drop. AT2 row order (after the full row) drops the higher
    // source index first (pat "01" then "10" for nl=2 ⇒ drop source 1, then 0). A
    // single-source model (nl_full==1) has NO valid drop (dropping the only source
    // leaves 0 rows, which cannot be fit) — AT2's power_set empty set is not a
    // fittable model, so only the full row is emitted.
    if (nl_full >= 2) {
        for (int drop = nl_full - 1; drop >= 0; --drop) {
            std::vector<int> surv;
            for (int i = 0; i < nl_full; ++i)
                if (i != drop) surv.push_back(i);
            rows.push_back(popdrop_one(be, x, cov, nl_full, drop, surv, opts, precision));
        }
    }
    return rows;
}

}  // namespace steppe::core::qpadm

// src/core/qpadm/ranktest.cpp — builds the qpAdm popdrop table: refit the model
// with each source dropped in turn to see which sources are load-bearing.
//
// Works purely by subsetting the already-computed f4 estimates and inverse
// covariance and refitting through the shared backend seam — no genotype re-read,
// no re-jackknife.
//
// Reference: docs/reference/src_core_qpadm_ranktest.cpp.md

#include "core/qpadm/ranktest.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "core/internal/index_cast.hpp"
#include "core/qpadm/qpadm_bounds.hpp"
#include "steppe/error.hpp"

namespace steppe::core::qpadm {

// The feasibility test — reference §3
bool popdrop_feasible(const std::vector<double>& weights) {
    bool has_surviving = false;
    for (double w : weights) {
        if (std::isnan(w)) continue;
        has_surviving = true;
        if (w < 0.0 || w > 1.0) return false;
    }
    return has_surviving;
}

namespace {

// Building the reduced problem — reference §4
void reduce_rows(const F4Blocks& x, const JackknifeCov& cov,
                 const std::vector<int>& surv, F4Blocks& x_reduced, JackknifeCov& cov_reduced) {
    const int nr = x.nr;
    const int nl_red = static_cast<int>(surv.size());
    const int m_full = x.nl * nr;
    const int m_red = nl_red * nr;

    x_reduced.nl = nl_red;
    x_reduced.nr = nr;
    x_reduced.n_block = x.n_block;
    x_reduced.x_total.assign(idx(m_red), 0.0);

    std::vector<int> ind(idx(m_red), 0);
    for (int ii = 0; ii < nl_red; ++ii) {
        const int i = surv[idx(ii)];
        for (int j = 0; j < nr; ++j) {
            const int dst = j + nr * ii;
            const int src = j + nr * i;
            x_reduced.x_total[idx(dst)] =
                x.x_total[idx(src)];
            ind[idx(dst)] = src;
        }
    }
    cov_reduced.m = m_red;
    cov_reduced.status = cov.status;
    cov_reduced.Qinv.assign(idx(m_red) * idx(m_red), 0.0);
    cov_reduced.Q.assign(idx(m_red) * idx(m_red), 0.0);
    const std::size_t m_full_sz = idx(m_full);
    const std::size_t m_red_sz = idx(m_red);
    for (int a = 0; a < m_red; ++a) {
        const std::size_t ia = idx(ind[idx(a)]);
        const std::size_t a_sz = idx(a);
        for (int b = 0; b < m_red; ++b) {
            const std::size_t src = ia + m_full_sz * idx(ind[idx(b)]);
            const std::size_t dst = a_sz + m_red_sz * idx(b);
            cov_reduced.Qinv[dst] = cov.Qinv[src];
            if (!cov.Q.empty()) cov_reduced.Q[dst] = cov.Q[src];
        }
    }
}

// Fitting one dropped-source row — reference §5
PopDropRow popdrop_one(ComputeBackend& be, const F4Blocks& x, const JackknifeCov& cov,
                       int nl_full, int drop, const std::vector<int>& surv,
                       const QpAdmOptions& opts, const Precision& precision) {
    PopDropRow row;
    row.pat.assign(idx(nl_full), '0');
    if (drop >= 0) row.pat[idx(drop)] = '1';
    row.wt = (drop >= 0) ? 1 : 0;

    F4Blocks x_reduced; JackknifeCov cov_reduced;
    reduce_rows(x, cov, surv, x_reduced, cov_reduced);
    const int nl_red = static_cast<int>(surv.size());

    const int r_fit = nl_red - 1;
    const RankSweep rs = run_rank_sweep(be, x_reduced, cov_reduced, opts.rank_alpha, opts, precision);
    const std::size_t ri = idx(r_fit < 0 ? 0 : r_fit);
    row.f4rank = r_fit;
    row.dof = (ri < rs.dof.size()) ? rs.dof[ri] : qpadm_dof(x_reduced.nl, x_reduced.nr, r_fit);
    row.chisq = (ri < rs.chisq.size()) ? rs.chisq[ri] : 0.0;
    row.p = (ri < rs.p.size()) ? rs.p[ri] : 0.0;
    row.status = rs.status;

    row.weight.assign(idx(nl_full),
                      std::numeric_limits<double>::quiet_NaN());
    const GlsWeights gw = gls_weights(be, x_reduced, cov_reduced, r_fit, opts, precision);
    if (gw.status == Status::Ok && gw.w.size() == surv.size()) {
        for (std::size_t s = 0; s < surv.size(); ++s)
            row.weight[idx(surv[s])] = gw.w[s];
    }
    row.feasible = popdrop_feasible(row.weight);
    return row;
}

}  // namespace

// The whole popdrop table and its row order — reference §6
std::vector<PopDropRow> run_popdrop(ComputeBackend& be, const F4Blocks& x,
                                    const JackknifeCov& cov, const QpAdmOptions& opts,
                                    const Precision& precision) {
    const int nl_full = x.nl;
    std::vector<PopDropRow> rows;
    if (nl_full <= 0) return rows;
    rows.reserve(idx(nl_full) + 1);

    std::vector<int> all;
    for (int i = 0; i < nl_full; ++i) all.push_back(i);
    rows.push_back(popdrop_one(be, x, cov, nl_full, -1, all, opts, precision));

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

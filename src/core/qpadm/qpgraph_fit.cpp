// src/core/qpadm/qpgraph_fit.cpp
//
// Fits one fixed admixture graph to observed f-statistics (the qpGraph
// operation). Host-only and CUDA-free: it reaches the GPU only through the
// ComputeBackend seam, wiring model -> basis -> fleet -> result.
//
// Reference: docs/reference/src_core_qpadm_qpgraph_fit.cpp.md
#include "steppe/qpgraph.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "core/qpadm/f3_triples.hpp"
#include "core/qpadm/jackknife.hpp"
#include "core/qpadm/qpadm_fit.hpp"
#include "core/qpadm/qpgraph_model.hpp"
#include "device/backend.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace steppe {

namespace {

inline constexpr std::size_t kPrimaryGpu = 0;
[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}

// The fleet arena — reference §5
QpGraphTopoArena make_arena(const core::qpadm::QpGraphModel& m, const QpGraphOptions& opts) {
    QpGraphTopoArena a;
    a.npop = m.npop; a.nedge_norm = m.nedge_norm; a.nadmix = m.nadmix;
    a.npair = m.npair; a.npath = m.npath; a.base_leaf = m.base_leaf;
    a.pwts0 = m.pwts0;
    a.pe_edge = m.pe_edge; a.pe_leaf = m.pe_leaf; a.pe_path = m.pe_path;
    a.pae_path = m.pae_path; a.pae_admixedge = m.pae_admixedge;
    a.cmb1 = m.cmb1; a.cmb2 = m.cmb2;
    a.constrained = opts.constrained;
    a.fudge = opts.fudge;
    return a;
}

// The four-stage pipeline — reference §2
template <class F2Src>
QpGraphResult run_qpgraph_impl(ComputeBackend& be, const F2Src& f2,
                               const std::vector<QpGraphEdge>& edges,
                               const std::vector<std::string>& leaf_names,
                               const QpGraphOptions& opts) {
    QpGraphResult res;

    const core::qpadm::QpGraphModel m = core::qpadm::parse_qpgraph(edges, leaf_names);
    if (!m.ok()) {
        res.status = Status::InvalidConfig;
        return res;
    }
    res.leaves = m.leaves;

    const Precision prec = core::qpadm::default_fit_precision();
    res.precision_tag = core::qpadm::honored_tag(prec, be);

    const int base_f2 = m.leaf_to_f2[static_cast<std::size_t>(m.base_leaf)];
    std::vector<int> flat;
    flat.reserve(static_cast<std::size_t>(m.npair) * 3);
    for (int k = 0; k < m.npair; ++k) {
        const int la = m.centered_col_to_leaf(m.cmb1[static_cast<std::size_t>(k)]);
        const int lb = m.centered_col_to_leaf(m.cmb2[static_cast<std::size_t>(k)]);
        flat.push_back(base_f2);
        flat.push_back(m.leaf_to_f2[static_cast<std::size_t>(la)]);
        flat.push_back(m.leaf_to_f2[static_cast<std::size_t>(lb)]);
    }
    F4Blocks X = core::qpadm::assemble_f3_triples(be, f2, std::span<const int>(flat), prec);
    const int npair = X.nl * X.nr;
    if (npair <= 0 || X.n_block <= 0) { res.status = Status::NonSpdCovariance; return res; }

    const JackknifeCov cov =
        core::qpadm::jackknife_cov(be, X, std::span<const int>(X.block_sizes), opts.diag_f3, prec);
    if (cov.status != Status::Ok) { res.status = cov.status; return res; }

    std::span<const double> f_obs(X.x_total.data(), static_cast<std::size_t>(npair));
    std::span<const double> qinv(cov.Qinv.data(), static_cast<std::size_t>(npair) * npair);

    const QpGraphTopoArena arena = make_arena(m, opts);
    const QpGraphFleet fl =
        be.qpgraph_fit_fleet(arena, f_obs, qinv, opts.numstart, opts.maxit, opts.tol, prec);
    if (fl.status != Status::Ok) { res.status = fl.status; return res; }

    res.score = fl.score;
    res.restart_spread = fl.restart_spread;
    res.weight = fl.theta;
    res.weight_lo = fl.theta_lo;
    res.weight_hi = fl.theta_hi;
    res.admix_from = m.admix_from;
    res.admix_to = m.admix_to;
    res.edge_length = fl.edge_length;
    res.edge_from = m.edge_from;
    res.edge_to = m.edge_to;

    double worst = 0.0;
    int worst_k = -1;
    for (int k = 0; k < npair; ++k) {
        const double qkk = cov.Q[static_cast<std::size_t>(k) + static_cast<std::size_t>(npair) * static_cast<std::size_t>(k)];
        const double se = (qkk > 0.0) ? std::sqrt(qkk) : std::nan("");
        const double z = (X.x_total[static_cast<std::size_t>(k)] - fl.f3_fit[static_cast<std::size_t>(k)]) / se;
        if (std::isfinite(z) && std::fabs(z) > std::fabs(worst)) { worst = z; worst_k = k; }
    }
    res.worst_residual_z = worst;
    if (worst_k >= 0) {
        const int la = m.centered_col_to_leaf(m.cmb1[static_cast<std::size_t>(worst_k)]);
        const int lb = m.centered_col_to_leaf(m.cmb2[static_cast<std::size_t>(worst_k)]);
        res.worst_pop2 = m.leaves[static_cast<std::size_t>(la)];
        res.worst_pop3 = m.leaves[static_cast<std::size_t>(lb)];
    }
    res.status = Status::Ok;
    return res;
}

}  // namespace

// Public entry points — reference §10
QpGraphResult run_qpgraph(const device::DeviceF2Blocks& f2,
                          const std::vector<QpGraphEdge>& edges,
                          const std::vector<std::string>& leaf_names,
                          const QpGraphOptions& opts, device::Resources& resources) {
    return run_qpgraph_impl(primary_backend(resources), f2, edges, leaf_names, opts);
}

QpGraphResult run_qpgraph(const F2BlockTensor& f2_host,
                          const std::vector<QpGraphEdge>& edges,
                          const std::vector<std::string>& leaf_names,
                          const QpGraphOptions& opts, device::Resources& resources) {
    return run_qpgraph_impl(primary_backend(resources), f2_host, edges, leaf_names, opts);
}

}  // namespace steppe

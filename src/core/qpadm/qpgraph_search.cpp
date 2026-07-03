// src/core/qpadm/qpgraph_search.cpp
//
// Driver for the qpGraph topology search: scores every admixture-graph shape in a
// small bounded population space and returns the best-fitting one. Host-only and
// CUDA-free — it reaches the GPU through a single backend fleet-fit seam, reusing one
// statistical basis (pop-set-bound, assembled once) across every candidate.
//
// Reference: docs/reference/src_core_qpadm_qpgraph_search.cpp.md
#include "steppe/qpgraph_search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <unordered_map>
#include <vector>

#include "core/qpadm/f3_triples.hpp"
#include "core/qpadm/jackknife.hpp"
#include "core/qpadm/qpadm_fit.hpp"
#include "core/qpadm/qpgraph_enumerate.hpp"
#include "core/qpadm/qpgraph_model.hpp"
#include "device/backend.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace steppe {

namespace {

namespace cq = core::qpadm;

// Named constants — reference §3
inline constexpr std::size_t kPrimaryGpu = 0;
[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}

inline constexpr int kMaxHillClimbSteps = 1000;

inline constexpr double kRecoveryRtol = 1e-6;
inline constexpr double kRecoveryAtol = 1e-9;

// The canonical f3 basis over the search pop-set — reference §4
struct CanonicalBasis {
    int npop = 0, npair = 0;
    std::vector<int> flat;
    std::vector<int> pair_a_pop;
    std::vector<int> pair_b_pop;
};

[[nodiscard]] CanonicalBasis build_canonical_basis(const std::vector<std::string>& pops,
                                                   const std::vector<std::string>& leaf_names) {
    CanonicalBasis b;
    b.npop = static_cast<int>(pops.size());
    std::unordered_map<std::string, int> f2idx;
    for (int i = 0; i < static_cast<int>(leaf_names.size()); ++i)
        f2idx.emplace(leaf_names[static_cast<std::size_t>(i)], i);
    std::vector<int> pop_f2(static_cast<std::size_t>(b.npop), -1);
    for (int i = 0; i < b.npop; ++i) {
        auto it = f2idx.find(pops[static_cast<std::size_t>(i)]);
        pop_f2[static_cast<std::size_t>(i)] = (it == f2idx.end()) ? -1 : it->second;
    }
    const int base_f2 = pop_f2[0];
    std::vector<int> nonbase_f2;
    for (int i = 1; i < b.npop; ++i) nonbase_f2.push_back(pop_f2[static_cast<std::size_t>(i)]);
    const int n_nonbase = static_cast<int>(nonbase_f2.size());
    for (int a = 0; a < n_nonbase; ++a)
        for (int bb = a; bb < n_nonbase; ++bb) {
            b.flat.push_back(base_f2);
            b.flat.push_back(nonbase_f2[static_cast<std::size_t>(a)]);
            b.flat.push_back(nonbase_f2[static_cast<std::size_t>(bb)]);
            b.pair_a_pop.push_back(nonbase_f2[static_cast<std::size_t>(a)]);
            b.pair_b_pop.push_back(nonbase_f2[static_cast<std::size_t>(bb)]);
        }
    b.npair = static_cast<int>(b.pair_a_pop.size());
    return b;
}

// Build one topology's fleet arena, remapped to the canonical pair order — reference §5
[[nodiscard]] bool make_canonical_arena(const cq::QpGraphModel& m, const CanonicalBasis& basis,
                                        const QpGraphOptions& opts, QpGraphTopoArena& out) {
    std::unordered_map<int, int> f2_to_ccol;
    for (int c = 0; c < m.npop - 1; ++c) {
        const int leaf = m.centered_col_to_leaf(c);
        const int f2 = m.leaf_to_f2[static_cast<std::size_t>(leaf)];
        f2_to_ccol[f2] = c;
    }
    std::vector<int> cmb1(static_cast<std::size_t>(basis.npair));
    std::vector<int> cmb2(static_cast<std::size_t>(basis.npair));
    for (int k = 0; k < basis.npair; ++k) {
        auto ia = f2_to_ccol.find(basis.pair_a_pop[static_cast<std::size_t>(k)]);
        auto ib = f2_to_ccol.find(basis.pair_b_pop[static_cast<std::size_t>(k)]);
        if (ia == f2_to_ccol.end() || ib == f2_to_ccol.end()) return false;
        int ca = ia->second, cb = ib->second;
        if (ca > cb) std::swap(ca, cb);
        cmb1[static_cast<std::size_t>(k)] = ca;
        cmb2[static_cast<std::size_t>(k)] = cb;
    }
    out.npop = m.npop; out.nedge_norm = m.nedge_norm; out.nadmix = m.nadmix;
    out.npair = basis.npair; out.npath = m.npath; out.base_leaf = m.base_leaf;
    out.pwts0 = m.pwts0;
    out.pe_edge = m.pe_edge; out.pe_leaf = m.pe_leaf; out.pe_path = m.pe_path;
    out.pae_path = m.pae_path; out.pae_admixedge = m.pae_admixedge;
    out.cmb1 = std::move(cmb1); out.cmb2 = std::move(cmb2);
    out.constrained = opts.constrained;
    out.fudge = opts.fudge;
    return true;
}

}  // namespace

namespace {

// The shared search pipeline + hill-climb recovery — reference §6, §7
template <class F2Src>
QpGraphSearchResult run_search_impl(ComputeBackend& be, const F2Src& f2,
                                    const std::vector<std::string>& leaf_names,
                                    const QpGraphSearchOptions& opts) {
    QpGraphSearchResult res;
    if (opts.pops.size() < 3) { res.status = Status::InvalidConfig; return res; }
    const int max_nadmix = std::min(opts.max_nadmix, 1);

    const Precision prec = cq::default_fit_precision();
    res.precision_tag = cq::honored_tag(prec, be);

    const std::vector<cq::EnumeratedTopology> cands =
        cq::enumerate_bounded_space(opts.pops, max_nadmix);
    if (cands.empty()) { res.status = Status::InvalidConfig; return res; }
    for (const cq::EnumeratedTopology& c : cands)
        (c.nadmix == 0 ? res.n_trees : res.n_admix1) += 1;
    res.n_candidates = static_cast<int>(cands.size());

    const CanonicalBasis basis = build_canonical_basis(opts.pops, leaf_names);
    F4Blocks X = cq::assemble_f3_triples(be, f2, std::span<const int>(basis.flat), prec);
    const int npair = X.nl * X.nr;
    if (npair <= 0 || X.n_block <= 0) { res.status = Status::NonSpdCovariance; return res; }
    const JackknifeCov cov =
        cq::jackknife_cov(be, X, std::span<const int>(X.block_sizes), opts.fit.diag_f3, prec);
    if (cov.status != Status::Ok) { res.status = cov.status; return res; }
    std::span<const double> f_obs(X.x_total.data(), static_cast<std::size_t>(npair));
    std::span<const double> qinv(cov.Qinv.data(), static_cast<std::size_t>(npair) * npair);

    std::vector<cq::QpGraphModel> models(cands.size());
    std::vector<QpGraphTopoArena> arenas;
    arenas.reserve(cands.size());
    std::vector<int> arena_of(cands.size(), -1);
    for (std::size_t i = 0; i < cands.size(); ++i) {
        cq::QpGraphModel m = cq::parse_qpgraph(cands[i].edges, leaf_names, opts.pops.front());
        if (!m.ok()) continue;
        QpGraphTopoArena a;
        if (!make_canonical_arena(m, basis, opts.fit, a)) continue;
        models[i] = std::move(m);
        arena_of[i] = static_cast<int>(arenas.size());
        arenas.push_back(std::move(a));
    }
    if (arenas.empty()) { res.status = Status::InvalidConfig; return res; }

    const auto t0 = std::chrono::steady_clock::now();
    const QpGraphFleetBatch fb = be.qpgraph_fit_fleet_batch(
        arenas, f_obs, qinv, opts.fit.numstart, opts.fit.maxit, opts.fit.tol, prec);
    const auto t1 = std::chrono::steady_clock::now();
    if (fb.status != Status::Ok) { res.status = fb.status; return res; }
    res.fit_all_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.topologies_per_s =
        res.fit_all_wall_ms > 0.0 ? (static_cast<double>(arenas.size()) / (res.fit_all_wall_ms / 1000.0)) : 0.0;

    res.candidates.reserve(arenas.size());
    for (std::size_t i = 0; i < cands.size(); ++i) {
        const int a = arena_of[i];
        if (a < 0) continue;
        QpGraphCandidate c;
        c.nadmix = cands[i].nadmix;
        c.id = cands[i].id;
        c.hash = cands[i].hash;
        c.score = fb.best_score[static_cast<std::size_t>(a)];
        c.restart_spread = fb.restart_spread[static_cast<std::size_t>(a)];
        c.edges = cands[i].edges;
        res.candidates.push_back(std::move(c));
    }

    int best_arena = -1;
    double best_s = std::numeric_limits<double>::infinity();
    double second_s = std::numeric_limits<double>::infinity();
    for (int a = 0; a < static_cast<int>(arenas.size()); ++a) {
        const double s = fb.best_score[static_cast<std::size_t>(a)];
        if (!std::isfinite(s)) continue;
        if (s < best_s) { second_s = best_s; best_s = s; best_arena = a; }
        else if (s < second_s) { second_s = s; }
    }
    if (best_arena < 0) { res.status = Status::NonSpdCovariance; return res; }
    int best_cand = -1;
    for (std::size_t i = 0; i < cands.size(); ++i) if (arena_of[i] == best_arena) { best_cand = static_cast<int>(i); break; }
    res.best.nadmix = cands[static_cast<std::size_t>(best_cand)].nadmix;
    res.best.id = cands[static_cast<std::size_t>(best_cand)].id;
    res.best.hash = cands[static_cast<std::size_t>(best_cand)].hash;
    res.best.score = best_s;
    res.best.edges = cands[static_cast<std::size_t>(best_cand)].edges;
    res.second_best_score = second_s;

    {
        const QpGraphTopoArena& a = arenas[static_cast<std::size_t>(best_arena)];
        const QpGraphFleet fl =
            be.qpgraph_fit_fleet(a, f_obs, qinv, opts.fit.numstart, opts.fit.maxit, opts.fit.tol, prec);
        res.best_fit.status = fl.status;
        res.best_fit.score = fl.score;
        res.best_fit.weight = fl.theta;
        res.best_fit.weight_lo = fl.theta_lo;
        res.best_fit.weight_hi = fl.theta_hi;
        res.best_fit.restart_spread = fl.restart_spread;
        res.best_fit.edge_length = fl.edge_length;
        const cq::QpGraphModel& bm = models[static_cast<std::size_t>(best_cand)];
        res.best_fit.edge_from = bm.edge_from;
        res.best_fit.edge_to = bm.edge_to;
        res.best_fit.admix_from = bm.admix_from;
        res.best_fit.admix_to = bm.admix_to;
        res.best_fit.leaves = bm.leaves;
        res.best_fit.precision_tag = res.precision_tag;
    }

    if (opts.run_heuristic) {
        res.heuristic_recovered = true;
        std::unordered_map<std::uint64_t, double> score_cache;
        for (std::size_t i = 0; i < cands.size(); ++i)
            if (arena_of[i] >= 0)
                score_cache[cands[i].hash] = fb.best_score[static_cast<std::size_t>(arena_of[i])];

        auto score_of = [&](const cq::EnumeratedTopology& g) -> double {
            auto it = score_cache.find(g.hash);
            if (it != score_cache.end()) return it->second;
            cq::QpGraphModel m = cq::parse_qpgraph(g.edges, leaf_names, opts.pops.front());
            if (!m.ok()) return std::numeric_limits<double>::infinity();
            QpGraphTopoArena a;
            if (!make_canonical_arena(m, basis, opts.fit, a)) return std::numeric_limits<double>::infinity();
            const QpGraphFleetBatch one = be.qpgraph_fit_fleet_batch(
                std::vector<QpGraphTopoArena>{a}, f_obs, qinv, opts.fit.numstart, opts.fit.maxit,
                opts.fit.tol, prec);
            const double s = (one.status == Status::Ok && !one.best_score.empty())
                                 ? one.best_score.front()
                                 : std::numeric_limits<double>::infinity();
            score_cache[g.hash] = s;
            return s;
        };

        const int K = std::max(1, opts.heuristic_seeds);
        res.heuristic_seed_hashes.clear();
        std::uint64_t agg_hash = 0;
        double agg_s = std::numeric_limits<double>::infinity();
        for (int sidx = 0; sidx < K; ++sidx) {
            const std::size_t start =
                cands.size() > 1 ? (static_cast<std::size_t>(sidx) * cands.size()) / static_cast<std::size_t>(K) : 0;
            cq::EnumeratedTopology cur = cands[start];
            double cur_s = score_of(cur);
            for (int step = 0; step < kMaxHillClimbSteps; ++step) {
                const std::vector<cq::EnumeratedTopology> nb =
                    cq::topology_neighbors(cur, opts.pops, max_nadmix);
                double best_nb = cur_s;
                int best_j = -1;
                for (int j = 0; j < static_cast<int>(nb.size()); ++j) {
                    const double s = score_of(nb[static_cast<std::size_t>(j)]);
                    if (std::isfinite(s) && s < best_nb) { best_nb = s; best_j = j; }
                }
                if (best_j < 0) break;
                cur = nb[static_cast<std::size_t>(best_j)];
                cur_s = best_nb;
            }
            res.heuristic_seed_hashes.push_back(cur.hash);
            if (std::isfinite(cur_s) && cur_s < agg_s) { agg_s = cur_s; agg_hash = cur.hash; }
        }
        const bool same_hash = (agg_hash == res.best.hash);
        const bool same_score =
            std::fabs(agg_s - best_s) <= kRecoveryAtol + kRecoveryRtol * std::fabs(best_s);
        res.heuristic_recovered = same_hash && same_score;
    }

    res.status = Status::Ok;
    return res;
}

}  // namespace

// Entry points — reference §8
QpGraphSearchResult run_qpgraph_search(const device::DeviceF2Blocks& f2,
                                       const std::vector<std::string>& leaf_names,
                                       const QpGraphSearchOptions& opts,
                                       device::Resources& resources) {
    return run_search_impl(primary_backend(resources), f2, leaf_names, opts);
}

QpGraphSearchResult run_qpgraph_search(const F2BlockTensor& f2_host,
                                       const std::vector<std::string>& leaf_names,
                                       const QpGraphSearchOptions& opts,
                                       device::Resources& resources) {
    return run_search_impl(primary_backend(resources), f2_host, leaf_names, opts);
}

}  // namespace steppe

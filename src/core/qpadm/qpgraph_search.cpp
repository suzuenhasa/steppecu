// src/core/qpadm/qpgraph_search.cpp — the qpGraph TOPOLOGY SEARCH v1 driver (oracle C).
//
// Wires the host topology ENUMERATOR -> the resident f3 basis (assembled ONCE; pop-set-
// bound) -> the HETEROGENEOUS-TOPOLOGY fleet (one launch fits ALL candidates) -> the
// global-best argmin -> the mutation/hill-climb heuristic (host-proposes / fleet-fits) that
// must RECOVER the exhaustive global-best. HOST-PURE, CUDA-FREE; the GPU is reached ONLY via
// the ComputeBackend::qpgraph_fit_fleet_batch seam (the same resident-basis pattern as the
// single-graph run_qpgraph). The f3 basis depends ONLY on the pop-set, so it is computed
// ONCE and reread by every candidate (zero per-candidate D2H/H2D).
//
// THE CANONICAL BASIS REUSE: every candidate is fit against the SAME f_obs/qinv. The basis
// pair set is choose(npop,2) over the fixed pop-set with base = pops[0]; the pair ORDER is
// canonical (sorted (a,b) over the non-base leaves in pops-order). Each topology's parsed
// model may order leaves differently (node-first-seen), so its cmb is REMAPPED to the
// canonical pair order before the arena is built — so the SHARED f_obs[k] always matches
// every topology's ppwts[k] (the design's "basis is pop-set-bound, NOT topology-bound").
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
#include "core/qpadm/qpadm_fit.hpp"        // default_fit_precision(), honored_tag()
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

inline constexpr std::size_t kPrimaryGpu = 0;
[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}

/// The canonical f3 basis over the search pop-set: base = pops[0]; the pairs are the
/// choose(npop,2) (a<=b) over the npop-1 non-base leaves in pops order (incl the diagonal).
/// `pair_a_pop`/`pair_b_pop` are the f2 P-axis indices of pair k's two leaves; the flattened
/// triple {base_f2, a_f2, b_f2} feeds assemble_f3_triples. Independent of any topology.
struct CanonicalBasis {
    int npop = 0, npair = 0;
    std::vector<int> flat;          ///< [3*npair] {base, a, b} f2 indices per pair.
    std::vector<int> pair_a_pop;    ///< [npair] f2 index of pair k's leaf a.
    std::vector<int> pair_b_pop;    ///< [npair] f2 index of pair k's leaf b.
};

[[nodiscard]] CanonicalBasis build_canonical_basis(const std::vector<std::string>& pops,
                                                   const std::vector<std::string>& leaf_names) {
    CanonicalBasis b;
    b.npop = static_cast<int>(pops.size());
    std::unordered_map<std::string, int> f2idx;
    for (int i = 0; i < static_cast<int>(leaf_names.size()); ++i)
        f2idx.emplace(leaf_names[static_cast<std::size_t>(i)], i);
    // pops[i] -> f2 index.
    std::vector<int> pop_f2(static_cast<std::size_t>(b.npop), -1);
    for (int i = 0; i < b.npop; ++i) {
        auto it = f2idx.find(pops[static_cast<std::size_t>(i)]);
        pop_f2[static_cast<std::size_t>(i)] = (it == f2idx.end()) ? -1 : it->second;
    }
    const int base_f2 = pop_f2[0];
    // the non-base leaves in pops order (centered columns 0..npop-2).
    std::vector<int> ncol_f2;
    for (int i = 1; i < b.npop; ++i) ncol_f2.push_back(pop_f2[static_cast<std::size_t>(i)]);
    const int ncc = static_cast<int>(ncol_f2.size());
    for (int a = 0; a < ncc; ++a)
        for (int bb = a; bb < ncc; ++bb) {
            b.flat.push_back(base_f2);
            b.flat.push_back(ncol_f2[static_cast<std::size_t>(a)]);
            b.flat.push_back(ncol_f2[static_cast<std::size_t>(bb)]);
            b.pair_a_pop.push_back(ncol_f2[static_cast<std::size_t>(a)]);
            b.pair_b_pop.push_back(ncol_f2[static_cast<std::size_t>(bb)]);
        }
    b.npair = static_cast<int>(b.pair_a_pop.size());
    return b;
}

/// Build the fleet arena for one topology, REMAPPING its cmb to the CANONICAL pair order so
/// the shared f_obs/qinv aligns. For canonical pair k = (f2 pop A, f2 pop B), set the
/// topology's cmb1[k]/cmb2[k] = the centered-column indices (in THIS topology's leaf order)
/// of the leaves carrying f2 pops A and B. Returns false if the topology's leaves don't
/// cover the pop-set (a structural mismatch — should not happen for the enumerated set).
[[nodiscard]] bool make_canonical_arena(const cq::QpGraphModel& m, const CanonicalBasis& basis,
                                        const QpGraphOptions& opts, QpGraphTopoArena& out) {
    // f2 pop index -> this model's centered-column index (or -1 for the base / unknown).
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
        if (ca > cb) std::swap(ca, cb);  // ppwts[k] is symmetric; keep a<=b (cosmetic).
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

/// The shared search body over an f2 SOURCE (DeviceF2Blocks or F2BlockTensor).
template <class F2Src>
QpGraphSearchResult run_search_impl(ComputeBackend& be, const F2Src& f2,
                                    const std::vector<std::string>& leaf_names,
                                    const QpGraphSearchOptions& opts) {
    QpGraphSearchResult res;
    if (opts.pops.size() < 3) { res.status = Status::InvalidConfig; return res; }
    const int max_nadmix = std::min(opts.max_nadmix, 1);  // v1 bounded space.

    const Precision prec = cq::default_fit_precision();
    res.precision_tag = cq::honored_tag(prec, be);

    // ---- 1. ENUMERATE (cheap host combinatorial -> tiny arenas) -----------------
    const std::vector<cq::EnumeratedTopology> cands =
        cq::enumerate_bounded_space(opts.pops, max_nadmix);
    if (cands.empty()) { res.status = Status::InvalidConfig; return res; }
    for (const cq::EnumeratedTopology& c : cands)
        (c.nadmix == 0 ? res.n_trees : res.n_admix1) += 1;
    res.n_candidates = static_cast<int>(cands.size());

    // ---- 2. CANONICAL BASIS (assembled ONCE; pop-set-bound, resident) -----------
    const CanonicalBasis basis = build_canonical_basis(opts.pops, leaf_names);
    F4Blocks X = cq::assemble_f3_triples(be, f2, std::span<const int>(basis.flat), prec);
    const int npair = X.nl * X.nr;
    if (npair <= 0 || X.n_block <= 0) { res.status = Status::NonSpdCovariance; return res; }
    const JackknifeCov cov =
        cq::jackknife_cov(be, X, std::span<const int>(X.block_sizes), opts.fit.diag_f3, prec);
    if (cov.status != Status::Ok) { res.status = cov.status; return res; }
    std::span<const double> f_obs(X.x_total.data(), static_cast<std::size_t>(npair));
    std::span<const double> qinv(cov.Qinv.data(), static_cast<std::size_t>(npair) * npair);

    // ---- 3. build every candidate arena (cmb remapped to the canonical order) ---
    std::vector<cq::QpGraphModel> models(cands.size());
    std::vector<QpGraphTopoArena> arenas;
    arenas.reserve(cands.size());
    std::vector<int> arena_of(cands.size(), -1);  // candidate -> arena index (or -1 = bad parse).
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

    // ---- 4. the HETEROGENEOUS FLEET (ONE launch fits ALL candidates) ------------
    const auto t0 = std::chrono::steady_clock::now();
    const QpGraphFleetBatch fb = be.qpgraph_fit_fleet_batch(
        arenas, f_obs, qinv, opts.fit.numstart, opts.fit.maxit, opts.fit.tol, prec);
    const auto t1 = std::chrono::steady_clock::now();
    if (fb.status != Status::Ok) { res.status = fb.status; return res; }
    res.fit_all_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.topologies_per_s =
        res.fit_all_wall_ms > 0.0 ? (static_cast<double>(arenas.size()) / (res.fit_all_wall_ms / 1000.0)) : 0.0;

    // ---- 5. PER-CANDIDATE scored vector (additive exposure; NOT new compute) ----
    // Retain EVERY successfully-fit topology's {canonical hash, edges, best-of-restarts
    // score} in enumeration order — the same per-(topology) data the argmin below reduces
    // over. A candidate whose arena failed to build (a bad parse) is omitted (no score).
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

    // ---- 5b. GLOBAL-BEST argmin (deterministic reduction; NOT a fit) ------------
    int best_arena = -1, second_arena = -1;
    double best_s = std::numeric_limits<double>::infinity();
    double second_s = std::numeric_limits<double>::infinity();
    for (int a = 0; a < static_cast<int>(arenas.size()); ++a) {
        const double s = fb.best_score[static_cast<std::size_t>(a)];
        if (!std::isfinite(s)) continue;
        if (s < best_s) { second_s = best_s; second_arena = best_arena; best_s = s; best_arena = a; }
        else if (s < second_s) { second_s = s; second_arena = a; }
    }
    if (best_arena < 0) { res.status = Status::NonSpdCovariance; return res; }
    (void)second_arena;
    // map the best arena back to its candidate.
    int best_cand = -1;
    for (std::size_t i = 0; i < cands.size(); ++i) if (arena_of[i] == best_arena) { best_cand = static_cast<int>(i); break; }
    res.best.nadmix = cands[static_cast<std::size_t>(best_cand)].nadmix;
    res.best.id = cands[static_cast<std::size_t>(best_cand)].id;
    res.best.hash = cands[static_cast<std::size_t>(best_cand)].hash;
    res.best.score = best_s;
    res.best.edges = cands[static_cast<std::size_t>(best_cand)].edges;
    res.second_best_score = second_s;

    // The full single-graph FIT of the global-best (the cc9ff69-tier result the per-
    // candidate parity gate diffs against AT2 qpgraph(best_edges)). REUSES the fleet seam.
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

    // ---- 6. HEURISTIC: hill-climb that must RECOVER the exhaustive global-best ---
    // host-proposes a small local neighborhood per step; the SAME fleet seam scores the
    // batch; accept the argmin; iterate to a local minimum. From K deterministic seeds the
    // recovered local-min hash MUST equal the exhaustive best (the falsifiable v1 gate).
    if (opts.run_heuristic) {
        res.heuristic_recovered = true;
        // A score cache keyed by the canonical hash, SEEDED from the exhaustive run (every
        // bounded-space neighbor is already scored — the hill-climb re-fits nothing; an
        // out-of-set neighbor, should one ever arise, is re-fit via a 1-topology fleet batch).
        std::unordered_map<std::uint64_t, double> score_cache;
        for (std::size_t i = 0; i < cands.size(); ++i)
            if (arena_of[i] >= 0)
                score_cache[cands[i].hash] = fb.best_score[static_cast<std::size_t>(arena_of[i])];

        auto score_of = [&](const cq::EnumeratedTopology& g) -> double {
            auto it = score_cache.find(g.hash);
            if (it != score_cache.end()) return it->second;
            // not in the exhaustive set (shouldn't happen for the bounded space, but be safe):
            // fit it via the fleet (a 1-topology batch).
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

        // MULTI-START hill-climb (the standard heuristic-search semantics): from K
        // deterministic seeds (strided across the candidate list for basin diversity),
        // best-improvement descent to a local minimum each; the heuristic's ANSWER is the
        // best local-min over ALL seeds (argmin). RECOVERY = that aggregate answer equals the
        // exhaustive global-best (same canonical hash + score within the fit tolerance). A
        // single hill-climb finds a LOCAL min (some seeds land in a worse basin — expected);
        // the multi-start aggregate is what a topology search returns.
        const int K = std::max(1, opts.heuristic_seeds);
        res.heuristic_seed_hashes.clear();
        std::uint64_t agg_hash = 0;
        double agg_s = std::numeric_limits<double>::infinity();
        for (int sidx = 0; sidx < K; ++sidx) {
            const std::size_t start =
                cands.size() > 1 ? (static_cast<std::size_t>(sidx) * cands.size()) / static_cast<std::size_t>(K) : 0;
            cq::EnumeratedTopology cur = cands[start];
            double cur_s = score_of(cur);
            for (int step = 0; step < 1000; ++step) {  // best-improvement descent.
                const std::vector<cq::EnumeratedTopology> nb =
                    cq::topology_neighbors(cur, opts.pops, max_nadmix);
                double best_nb = cur_s;
                int best_j = -1;
                for (int j = 0; j < static_cast<int>(nb.size()); ++j) {
                    const double s = score_of(nb[static_cast<std::size_t>(j)]);
                    if (std::isfinite(s) && s < best_nb) { best_nb = s; best_j = j; }
                }
                if (best_j < 0) break;  // a local minimum.
                cur = nb[static_cast<std::size_t>(best_j)];
                cur_s = best_nb;
            }
            res.heuristic_seed_hashes.push_back(cur.hash);
            if (std::isfinite(cur_s) && cur_s < agg_s) { agg_s = cur_s; agg_hash = cur.hash; }
        }
        // recovery: the multi-start aggregate == the exhaustive global-best (hash + score).
        const double rtol = 1e-6, atol = 1e-9;
        const bool same_hash = (agg_hash == res.best.hash);
        const bool same_score = std::fabs(agg_s - best_s) <= atol + rtol * std::fabs(best_s);
        res.heuristic_recovered = same_hash && same_score;
        (void)score_cache;
    }

    res.status = Status::Ok;
    return res;
}

}  // namespace

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

// bindings/bind_qpgraph.cpp — the qpGraph Python bindings (steppe._core).
//
// Registers run_qpgraph (fit a fixed admixture graph) and run_qpgraph_search (the
// topology search). Both use the afprod=FALSE f2, unlike qpadm's afprod=TRUE.
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "internal/bind_common.hpp"

#include "steppe/qpgraph_search.hpp"

namespace steppe::pybind {
namespace {

nb::dict run_qpgraph_py(F2Handle& h,
                        const std::vector<std::array<std::string, 2>>& edges,
                        int numstart, double fudge, double diag_f3, bool constrained) {
    if (edges.empty()) raise_value("qpgraph: the graph edge list is empty");
    std::vector<steppe::QpGraphEdge> e;
    e.reserve(edges.size());
    for (const auto& pr : edges) e.push_back({pr[0], pr[1]});

    steppe::QpGraphOptions opts;
    opts.numstart = numstart;
    opts.fudge = fudge;
    opts.diag_f3 = diag_f3;
    opts.constrained = constrained;

    const steppe::QpGraphResult result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_qpgraph(dev_f2, e, h.pops, opts, resources);
        });
    if (result.status == steppe::Status::InvalidConfig)
        raise_value("qpgraph: the graph could not be fit (a leaf is not an f2 population, "
                    "or the topology is unrooted/cyclic/invalid)");
    return qpgraph_to_dict(result);
}

nb::dict run_qpgraph_search_py(F2Handle& h, const std::vector<std::string>& pops, int max_nadmix,
                               int numstart, double fudge, double diag_f3, bool constrained,
                               bool run_heuristic) {
    if (pops.size() < 3) raise_value("qpgraph-search: need >= 3 population labels");
    steppe::QpGraphSearchOptions opts;
    opts.pops = pops;
    opts.max_nadmix = max_nadmix;
    opts.run_heuristic = run_heuristic;
    opts.fit.numstart = numstart;
    opts.fit.fudge = fudge;
    opts.fit.diag_f3 = diag_f3;
    opts.fit.constrained = constrained;

    const steppe::QpGraphSearchResult r =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_qpgraph_search(dev_f2, h.pops, opts, resources);
        });
    if (r.status == steppe::Status::InvalidConfig)
        raise_value("qpgraph-search: invalid pop-set (a pop is not an f2 population, or < 3 leaves)");

    nb::dict d;
    d["n_trees"] = r.n_trees;
    d["n_admix1"] = r.n_admix1;
    d["n_candidates"] = r.n_candidates;
    d["best_score"] = r.best.score;
    d["second_best_score"] = r.second_best_score;
    d["best_nadmix"] = r.best.nadmix;
    d["best_hash"] = static_cast<std::uint64_t>(r.best.hash);
    d["heuristic_recovered"] = r.heuristic_recovered;
    d["fit_all_wall_ms"] = r.fit_all_wall_ms;
    d["topologies_per_s"] = r.topologies_per_s;
    nb::list be;
    for (const auto& e : r.best.edges) be.append(nb::make_tuple(e.from, e.to));
    d["best_edges"] = be;

    std::vector<int> cand_nadmix;
    std::vector<std::uint64_t> cand_hash;
    std::vector<double> cand_score;
    std::vector<double> cand_spread;
    cand_nadmix.reserve(r.candidates.size());
    cand_hash.reserve(r.candidates.size());
    cand_score.reserve(r.candidates.size());
    cand_spread.reserve(r.candidates.size());
    for (const auto& c : r.candidates) {
        cand_nadmix.push_back(c.nadmix);
        cand_hash.push_back(static_cast<std::uint64_t>(c.hash));
        cand_score.push_back(c.score);
        cand_spread.push_back(c.restart_spread);
    }
    d["cand_nadmix"] = cand_nadmix;
    d["cand_hash"] = cand_hash;
    d["cand_score"] = cand_score;
    d["cand_restart_spread"] = cand_spread;
    return d;
}

}  // namespace

void register_qpgraph(nb::module_& m) {
    m.def("run_qpgraph", &run_qpgraph_py, "f2"_a, "edges"_a, "numstart"_a = 10,
          "fudge"_a = 1e-4, "diag_f3"_a = 1e-5, "constrained"_a = true,
          "Single-graph qpGraph fit (GPU; the IDEA-1 fleet on-device). `edges` is a list "
          "of (parent, child) name pairs; the leaves must be f2 populations. NOTE: qpGraph "
          "uses the AT2 afprod=FALSE f2 (read_f2 of an afprod=FALSE dir). Returns a flat "
          "dict {score, weight, admix_from/to, edge_length, edge_from/to, ...}.");

    m.def("run_qpgraph_search", &run_qpgraph_search_py, "f2"_a, "pops"_a, "max_nadmix"_a = 1,
          "numstart"_a = 10, "fudge"_a = 1e-4, "diag_f3"_a = 1e-5, "constrained"_a = true,
          "run_heuristic"_a = true,
          "qpGraph TOPOLOGY SEARCH v1 (GPU; the heterogeneous-topology fleet, ONE launch fits "
          "ALL candidates). Exhaustively enumerates every rooted topology on the bounded `pops` "
          "leaf set (nadmix in {0..max_nadmix}; reproduces admixtools generate_all_graphs 1:1) "
          "and returns the deterministic global-best. Flat dict {n_trees, n_admix1, "
          "n_candidates, best_score, second_best_score, best_nadmix, best_hash, best_edges, "
          "heuristic_recovered, fit_all_wall_ms, topologies_per_s}.");
}

}  // namespace steppe::pybind

// include/steppe/qpgraph_search.hpp
//
// Public, CUDA-free interface to the qpGraph topology search: score every
// bounded admixture-graph shape on a fixed pop-set in one GPU launch and return
// the deterministic global-best. Declarations only — no device code, so it is
// includable by the core library, CLI, and bindings alike.
//
// Reference: docs/reference/include_steppe_qpgraph_search.hpp.md
#ifndef STEPPE_QPGRAPH_SEARCH_HPP
#define STEPPE_QPGRAPH_SEARCH_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/fstats.hpp"
#include "steppe/qpgraph.hpp"

namespace steppe {

namespace device {
class DeviceF2Blocks;
struct Resources;
}  // namespace device

// QpGraphSearchOptions — reference §2
struct QpGraphSearchOptions {
    std::vector<std::string> pops;
    int max_nadmix = 1;
    QpGraphOptions fit;
    bool run_heuristic = true;
    int heuristic_seeds = 8;
};

// QpGraphCandidate — reference §3
struct QpGraphCandidate {
    int nadmix = 0;
    int id = 0;
    std::uint64_t hash = 0;
    double score = 0.0;
    double restart_spread = 0.0;
    std::vector<QpGraphEdge> edges;
};

// QpGraphSearchResult — reference §4
struct QpGraphSearchResult {
    int n_trees = 0;
    int n_admix1 = 0;
    int n_candidates = 0;

    QpGraphCandidate best;
    double second_best_score = 0.0;

    std::vector<QpGraphCandidate> candidates;

    QpGraphResult best_fit;

    bool heuristic_recovered = false;
    std::vector<std::uint64_t> heuristic_seed_hashes;

    double fit_all_wall_ms = 0.0;
    double topologies_per_s = 0.0;

    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// Entry points — reference §5
[[nodiscard]] QpGraphSearchResult run_qpgraph_search(const device::DeviceF2Blocks& f2,
                                                     const std::vector<std::string>& leaf_names,
                                                     const QpGraphSearchOptions& opts,
                                                     device::Resources& resources);

[[nodiscard]] QpGraphSearchResult run_qpgraph_search(const F2BlockTensor& f2_host,
                                                     const std::vector<std::string>& leaf_names,
                                                     const QpGraphSearchOptions& opts,
                                                     device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_QPGRAPH_SEARCH_HPP

// include/steppe/qpgraph_search.hpp
//
// PUBLIC, CUDA-FREE qpGraph TOPOLOGY SEARCH v1 (oracle C: exhaustive bounded enumeration).
// The killer app of the heterogeneous-topology fleet — score EVERY topology in a bounded
// pop-set/nadmix space in ONE GPU launch and return the deterministic global-best.
//
// WHY (C) and not match-AT2's-search: AT2's find_graphs is stochastic (per-seed best
// varies SD~10, each a different overfit graph, never recovers the curated golden) — so a
// match-the-search oracle does NOT exist. (C) sidesteps it: in a BOUNDED enumerable space
// the GLOBAL BEST is exhaustively computable + deterministic, and each candidate is scored
// by the CLEAN single-graph fit (the cc9ff69 AT2-fit golden, rtol ~1e-6). The enumeration
// reproduces admixtools' OWN generate_all_trees/generate_all_graphs 1:1 (count + canonical
// graph_hash set), so AT2 is the exhaustive-coverage oracle.
//
// GPU SHAPE: the f3 basis (depends only on the pop-set) + Qinv are assembled ONCE and stay
// device-resident; the HETEROGENEOUS-TOPOLOGY fleet packs every topology's path-table arena
// into ONE device buffer + a per-topology index, flattens the launch over (topo,restart),
// and fits ALL candidates in one launch. The host does only the cheap enumeration + the
// global-best argmin (a reduction, not a fit). NEVER a per-candidate host fit.
#ifndef STEPPE_QPGRAPH_SEARCH_HPP
#define STEPPE_QPGRAPH_SEARCH_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/fstats.hpp"
#include "steppe/qpgraph.hpp"  // QpGraphEdge / QpGraphOptions / QpGraphResult

namespace steppe {

namespace device {
class DeviceF2Blocks;
struct Resources;
}  // namespace device

/// Per-call topology-search config. The per-candidate FIT tier reuses QpGraphOptions
/// (fudge / diag_f3 / numstart / constrained / maxit / tol == the cc9ff69 golden fit).
struct QpGraphSearchOptions {
    /// The bounded leaf pop-set (the search is over all rooted topologies on THESE leaves;
    /// each must be an f2 population). Empty ⇒ an InvalidConfig outcome.
    std::vector<std::string> pops;

    /// The maximum number of admixture nodes (the bounded nadmix axis). v1 supports {0,1}.
    int max_nadmix = 1;

    /// The per-candidate single-graph FIT options (the clean fit tier; defaults == golden).
    QpGraphOptions fit;

    /// Run the mutation/hill-climb heuristic AND verify it recovers the exhaustive global-
    /// best (the falsifiable v1 gate). Off ⇒ exhaustive-only.
    bool run_heuristic = true;

    /// The number of deterministic hill-climb seed starts (the recovery must hold from ALL).
    int heuristic_seeds = 8;
};

/// One scored candidate in the enumerated space (the per-topology best-of-restarts fit).
struct QpGraphCandidate {
    int nadmix = 0;             ///< the admixture-node count (0 = tree).
    int id = 0;                 ///< the stable enumeration index within its nadmix level.
    std::uint64_t hash = 0;     ///< the canonical graph_hash (the isomorphism / recovery key).
    double score = 0.0;         ///< the best (min over restarts) GLS fit score.
    std::vector<QpGraphEdge> edges;  ///< the topology edge list (parse_qpgraph-ready).
};

/// The topology-search result (oracle C, three parts).
struct QpGraphSearchResult {
    /// (2) EXHAUSTIVE-COVERAGE: the enumerated candidate counts (== AT2 numtrees /
    /// non-iso generate_all_graphs — the exhaustiveness witness).
    int n_trees = 0;            ///< nadmix=0 count.
    int n_admix1 = 0;           ///< nadmix=1 non-isomorphic count.
    int n_candidates = 0;       ///< total scored.

    /// (3) GLOBAL-BEST: the deterministic argmin over the enumerated scores.
    QpGraphCandidate best;
    double second_best_score = 0.0;  ///< the runner-up score (the identifiability gap witness).

    /// The full single-graph FIT of the global-best (the cc9ff69-tier result the per-
    /// candidate parity gate diffs against AT2 qpgraph(best_edges)).
    QpGraphResult best_fit;

    /// The HEURISTIC outcome: whether the hill-climb recovered the exhaustive global-best
    /// (same canonical hash + score within the fit tolerance) from ALL seeds. EMPTY
    /// heuristic_seed_hashes ⇒ the heuristic did not run.
    bool heuristic_recovered = false;
    std::vector<std::uint64_t> heuristic_seed_hashes;  ///< the local-min hash each seed found.

    /// WALL-CLOCK (the GPU-bound witness): the enumerate+fit-all wall (ms) and the derived
    /// topologies/s. The fleet launch is the work (host does only enumeration + argmin).
    double fit_all_wall_ms = 0.0;
    double topologies_per_s = 0.0;

    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// ---- Entry points -----------------------------------------------------------------
/// Topology search reading DEVICE-RESIDENT f2_blocks (the GPU-first primary). The f3 basis
/// is assembled device-resident ONCE; the heterogeneous fleet fits every candidate in one
/// launch. `leaf_names` maps the f2 P-axis (leaf_names[i] = the name of f2 pop i).
[[nodiscard]] QpGraphSearchResult run_qpgraph_search(const device::DeviceF2Blocks& f2,
                                                     const std::vector<std::string>& leaf_names,
                                                     const QpGraphSearchOptions& opts,
                                                     device::Resources& resources);

/// HOST-ORACLE / parity overload (the CpuBackend reads host memory).
[[nodiscard]] QpGraphSearchResult run_qpgraph_search(const F2BlockTensor& f2_host,
                                                     const std::vector<std::string>& leaf_names,
                                                     const QpGraphSearchOptions& opts,
                                                     device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_QPGRAPH_SEARCH_HPP

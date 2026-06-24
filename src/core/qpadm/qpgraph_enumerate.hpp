// src/core/qpadm/qpgraph_enumerate.hpp
//
// The qpGraph TOPOLOGY ENUMERATOR — the host-side combinatorial generator the topology
// SEARCH v1 (oracle C, exhaustive bounded enumeration) is built on. The ONE new host
// piece (the heterogeneous fleet + the global-best argmin + the hill-climb reuse this).
//
// SCHEME (admixtools' OWN enumerator, reproduced 1:1 against AT2 numtrees/numtreesadmix):
//   * nadmix=0 = ALL rooted bifurcating leaf-labeled topologies on the n leaves =
//                (2n-3)!! trees (generate_all_trees). Sequential leaf-insertion: start
//                with root->leaf0, then insert each subsequent leaf by SPLITTING an
//                existing edge with a fresh internal node; the recursion over every edge
//                choice enumerates the full set, deterministically (a canonical DFS).
//   * nadmix=1 = a base tree + ONE admixture node hosted on an UNORDERED pair of
//                distinct, INCOMPARABLE edges (the AT2 labeled construction T*C(2n-2,2)),
//                de-duplicated to NON-ISOMORPHIC graphs by a canonical graph hash
//                (graph_hash) — so the enumerated set == generate_all_graphs(leaves,1).
//
// EXHAUSTIVE-COVERAGE PROOF: the produced count == AT2 numtrees(n) / non-iso
// generate_all_graphs(n,1) EXACTLY (n=5 -> 105 trees, 1485 non-iso nadmix=1 graphs;
// box5090-verified). Each topology is a `std::vector<QpGraphEdge>` directly parseable by
// parse_qpgraph (the existing single-graph data model) — NO bolt-on.
//
// HOST-PURE, CUDA-FREE. Deterministic (no RNG): the topology IDs are stable across runs,
// the (C)-determinism property find_graphs lacks.
#ifndef STEPPE_CORE_QPADM_QPGRAPH_ENUMERATE_HPP
#define STEPPE_CORE_QPADM_QPGRAPH_ENUMERATE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "steppe/qpgraph.hpp"  // QpGraphEdge

namespace steppe::core::qpadm {

/// One enumerated candidate topology: the edge list (parent->child, parse_qpgraph-ready)
/// + its provenance (nadmix level + a stable index + the canonical graph hash used for
/// de-duplication). `id` is the deterministic enumeration index within (nadmix); `hash`
/// is the non-isomorphic canonical key (equal hashes => isomorphic graphs).
struct EnumeratedTopology {
    std::vector<QpGraphEdge> edges;
    int nadmix = 0;
    int id = 0;             ///< stable index within the nadmix level (enumeration order).
    std::uint64_t hash = 0; ///< canonical graph hash (the de-dup / isomorphism key).
};

/// Enumerate EVERY rooted bifurcating leaf-labeled tree on `leaves` (nadmix=0). The leaf
/// labels are used verbatim as the leaf node names; internal nodes get fresh deterministic
/// labels. Count == (2n-3)!! == AT2 numtrees(n). Order is a canonical DFS (deterministic).
[[nodiscard]] std::vector<EnumeratedTopology> enumerate_trees(
    const std::vector<std::string>& leaves);

/// Enumerate EVERY non-isomorphic nadmix=1 admixture graph on `leaves`: each base tree +
/// one admixture node on an unordered pair of distinct incomparable (non-ancestor) edges,
/// de-duplicated by the canonical graph hash. Count == AT2 non-iso generate_all_graphs(n,1).
[[nodiscard]] std::vector<EnumeratedTopology> enumerate_admix1(
    const std::vector<std::string>& leaves);

/// The full bounded v1 space: trees (nadmix=0) ++ admix1 (nadmix=1), in that order. The
/// exhaustive candidate set the heterogeneous fleet scores in one launch.
[[nodiscard]] std::vector<EnumeratedTopology> enumerate_bounded_space(
    const std::vector<std::string>& leaves, int max_nadmix);

/// The canonical isomorphism hash of an edge-list graph (leaf labels are the only fixed
/// vertices; internal/admix node labels are quotiented out). Two graphs are isomorphic iff
/// their hashes are equal (used for the de-dup + the heuristic's "same graph" recovery
/// check). A 1-WL-style color-refinement over the leaf-anchored DAG, order-independent.
[[nodiscard]] std::uint64_t graph_hash(const std::vector<QpGraphEdge>& edges);

/// neighbor MOVES for the hill-climb heuristic (the AT2 topology-move namespace,
/// restricted to the bounded n,nadmix<=1 space): all topologies one move away from
/// `current` (NNI/SPR-style tree rearrangements at nadmix=0; admix-edge relocations +
/// add/drop the single admix node across the nadmix boundary). De-duplicated by graph_hash
/// and filtered to the bounded space (nadmix<=max_nadmix). Each neighbor is re-scored by
/// the SAME fleet (host-proposes / fleet-fits).
[[nodiscard]] std::vector<EnumeratedTopology> topology_neighbors(
    const EnumeratedTopology& current, const std::vector<std::string>& leaves,
    int max_nadmix);

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_QPGRAPH_ENUMERATE_HPP

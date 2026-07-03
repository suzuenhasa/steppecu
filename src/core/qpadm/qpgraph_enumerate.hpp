// src/core/qpadm/qpgraph_enumerate.hpp
//
// Host-side generator of every candidate qpGraph topology the bounded search
// scores: all plain trees plus all one-admixture graphs (nadmix <= 1). Pure
// host code, CUDA-free and deterministic — the same call yields the same
// topologies in the same stable order on every run.
//
// Reference: docs/reference/src_core_qpadm_qpgraph_enumerate.hpp.md
#ifndef STEPPE_CORE_QPADM_QPGRAPH_ENUMERATE_HPP
#define STEPPE_CORE_QPADM_QPGRAPH_ENUMERATE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "steppe/qpgraph.hpp"

namespace steppe::core::qpadm {

// One candidate graph — reference §2
struct EnumeratedTopology {
    std::vector<QpGraphEdge> edges;
    int nadmix = 0;
    int id = 0;
    std::uint64_t hash = 0;
};

// All plain trees (nadmix=0) — reference §3
[[nodiscard]] std::vector<EnumeratedTopology> enumerate_trees(
    const std::vector<std::string>& leaves);

// All one-admixture graphs (nadmix=1) — reference §4
[[nodiscard]] std::vector<EnumeratedTopology> enumerate_admix1(
    const std::vector<std::string>& leaves);

// The whole bounded search set — reference §5
[[nodiscard]] std::vector<EnumeratedTopology> enumerate_bounded_space(
    const std::vector<std::string>& leaves, int max_nadmix);

// The shape identity (isomorphism hash) — reference §6
[[nodiscard]] std::uint64_t graph_hash(const std::vector<QpGraphEdge>& edges);

// Moves for the hill-climb — reference §7
[[nodiscard]] std::vector<EnumeratedTopology> topology_neighbors(
    const EnumeratedTopology& current, const std::vector<std::string>& leaves,
    int max_nadmix);

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_QPGRAPH_ENUMERATE_HPP

// src/core/qpadm/qpgraph_model.hpp
//
// The qpGraph topology data model: parses an admixtools-format edge list into a
// rooted DAG and precomputes a path-table model (the base matrix pwts0 plus the
// fill_pwts integer tables) that evaluates leaf weights for a given set of mixture
// weights. Host-only and CUDA-free; the evaluation is defined inline so the CPU
// oracle and the GPU fit run the identical code.
//
// Reference: docs/reference/src_core_qpadm_qpgraph_model.hpp.md
#ifndef STEPPE_CORE_QPADM_QPGRAPH_MODEL_HPP
#define STEPPE_CORE_QPADM_QPGRAPH_MODEL_HPP

#include <string>
#include <cstddef>
#include <vector>

#include "steppe/qpgraph.hpp"

namespace steppe::core::qpadm {

// QpGraphModel: dimensions and identity fields — reference §3
struct QpGraphModel {
    int npop = 0;
    int nedge_total = 0;
    int nedge_norm = 0;
    int nadmix = 0;
    int npair = 0;
    int npath = 0;
    int base_leaf = 0;

    std::vector<std::string> leaves;
    std::vector<int> leaf_to_f2;

    // Base leaf-weight matrix pwts0 — reference §4
    std::vector<double> pwts0;

    // fill_pwts path tables — reference §5
    std::vector<int> pe_edge;
    std::vector<int> pe_leaf;
    std::vector<int> pe_path;

    std::vector<int> pae_path;
    std::vector<int> pae_admixedge;

    // f3 basis pairs — reference §6
    std::vector<int> cmb1;
    std::vector<int> cmb2;

    // Labels and the error field — reference §7
    std::vector<std::string> edge_from;
    std::vector<std::string> edge_to;
    std::vector<std::string> admix_from;
    std::vector<std::string> admix_to;

    std::string error;
    [[nodiscard]] bool ok() const { return error.empty(); }

    // centered_col_to_leaf — reference §8
    [[nodiscard]] int centered_col_to_leaf(int c) const {
        int seen = 0;
        for (int li = 0; li < npop; ++li) {
            if (li == base_leaf) continue;
            if (seen == c) return li;
            ++seen;
        }
        return -1;
    }
};

// parse_qpgraph — reference §9
[[nodiscard]] QpGraphModel parse_qpgraph(const std::vector<QpGraphEdge>& edges,
                                         const std::vector<std::string>& leaf_names,
                                         const std::string& f3basepop = "");

// fill_pwts_centered — reference §10
inline void fill_pwts_centered(const QpGraphModel& m, const double* theta,
                               std::vector<double>& pwts_c) {
    const int ne = m.nedge_norm, np = m.npop;
    std::vector<double> path_w(static_cast<std::size_t>(m.npath), 1.0);
    for (std::size_t t = 0; t < m.pae_path.size(); ++t) {
        const int pi = m.pae_path[t];
        const int id = m.pae_admixedge[t];
        const int j = (id - 1) / 2;
        const double w = theta[j];
        const double v = (id % 2 == 1) ? w : (1.0 - w);
        path_w[static_cast<std::size_t>(pi)] *= v;
    }
    std::vector<double> pwts = m.pwts0;
    for (std::size_t a = 0; a < m.pe_edge.size(); ++a) {
        const int e = m.pe_edge[a], leaf = m.pe_leaf[a];
        bool first = true;
        for (std::size_t b = 0; b < a; ++b)
            if (m.pe_edge[b] == e && m.pe_leaf[b] == leaf) { first = false; break; }
        if (!first) continue;
        double sum = 0.0;
        for (std::size_t b = a; b < m.pe_edge.size(); ++b)
            if (m.pe_edge[b] == e && m.pe_leaf[b] == leaf) sum += path_w[static_cast<std::size_t>(m.pe_path[b])];
        pwts[static_cast<std::size_t>(e) + static_cast<std::size_t>(ne) * static_cast<std::size_t>(leaf)] = sum;
    }
    pwts_c.assign(static_cast<std::size_t>(ne) * static_cast<std::size_t>(np - 1), 0.0);
    int col = 0;
    for (int leaf = 0; leaf < np; ++leaf) {
        if (leaf == m.base_leaf) continue;
        for (int e = 0; e < ne; ++e) {
            const double v = pwts[static_cast<std::size_t>(e) + static_cast<std::size_t>(ne) * static_cast<std::size_t>(leaf)] -
                             pwts[static_cast<std::size_t>(e) + static_cast<std::size_t>(ne) * static_cast<std::size_t>(m.base_leaf)];
            pwts_c[static_cast<std::size_t>(e) + static_cast<std::size_t>(ne) * static_cast<std::size_t>(col)] = v;
        }
        ++col;
    }
}

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_QPGRAPH_MODEL_HPP

// src/core/qpadm/qpgraph_model.hpp
//
// The qpGraph TOPOLOGY DATA MODEL — the ONE genuinely-new piece (the spike's
// d_leafweights_n{1,2,3} were hard-coded per-nadmix; this GENERALIZES them to an
// arbitrary topology via a precomputed PATH-TABLE model, mirroring admixtools
// graph_to_pwts / graph_to_weightind / fill_pwts EXACTLY — verified file:line against
// the installed admixtools 2.0.10 R source on box5090).
//
// PARSE an admixtools-format edge list (parent->child rows; a node with 2 parents is an
// admixture node) into a rooted DAG, then enumerate every root->leaf simple path and
// build:
//   * pwts0       [nedge_norm x npop]  the BASE leaf-weight matrix (graph_to_pwts; the
//                 1/pathcount drift incidence over the NON-admixture edges).
//   * path tables (path_edge_table / path_admixedge_table) the fill_pwts inputs: per
//                 (edge2, leaf2) cell that VARIES with theta, the list of paths and the
//                 admixedges each path traverses. fill_pwts(theta) overwrites exactly
//                 those cells with sum_path prod_admixedge (theta or 1-theta).
//   * cmb         the f3 basis pair list (combn(0:npop-1, 2), the AT2 +(1:0) offset into
//                 the centered, base-dropped leaf columns).
//   * the base leaf index (f3basepop default = leaf 0) + the leaf->f2-index map.
//
// HOST-PURE, CUDA-FREE. The device fleet (qpgraph_fit_kernels.cu) uploads the int path
// arenas ONCE to VRAM (the qpAdm per-model index-arena pattern) and runs the SAME
// fill_pwts on-device. The CpuBackend oracle uses this struct directly.
#ifndef STEPPE_CORE_QPADM_QPGRAPH_MODEL_HPP
#define STEPPE_CORE_QPADM_QPGRAPH_MODEL_HPP

#include <string>
#include <cstddef>
#include <string>
#include <vector>

#include "steppe/qpgraph.hpp"  // QpGraphEdge

namespace steppe::core::qpadm {

/// The parsed, fit-ready topology. All arrays are 0-based and flat (the device-arena
/// shape). A non-empty `error` means the parse failed (an unrooted / cyclic /
/// malformed graph, an internal node marked sampled, a leaf not in the f2 set) — the
/// caller maps it to a Status domain outcome. Cyclicity is caught in two places (see
/// qpgraph_model.cpp): a no-root cycle (every node has in-degree>0, so no in-degree-0
/// root) AND a cycle reachable from a valid root (a back-edge found by the path-DFS
/// recursion-stack guard).
struct QpGraphModel {
    int npop = 0;            ///< leaf count (== the f3 npop).
    int nedge_total = 0;     ///< all edges (rows of the input edge list).
    int nedge_norm = 0;      ///< NON-admixture (drift) edges (the fitted edge lengths).
    int nadmix = 0;          ///< admixture nodes (theta dim).
    int npair = 0;           ///< choose(npop, 2) — the f3 basis dim.
    int npath = 0;           ///< total root->leaf simple paths.
    int base_leaf = 0;       ///< f3basepop leaf index (default 0 = first leaf).

    /// Leaf labels in graph leaf order (leaves[base_leaf] is the base population). Same
    /// order as the columns of pwts0 and the rows the f3 basis indexes.
    std::vector<std::string> leaves;
    /// leaf i's index into the f2 P axis (leaf_to_f2[i] = the f2 pop index). The f3
    /// basis triples use these.
    std::vector<int> leaf_to_f2;

    /// pwts0 [nedge_norm x npop] COLUMN-MAJOR: pwts0[e + nedge_norm*leaf] — the base
    /// (theta-independent) drift incidence (graph_to_pwts; 1/pathcount per traversed
    /// non-admix edge). fill_pwts OVERWRITES the (edge2,leaf2) cells listed below.
    std::vector<double> pwts0;

    /// fill_pwts PATH-EDGE table (the cells that vary with theta). Parallel arrays of
    /// length n_pe: cell (pe_edge[t], pe_leaf[t]) is contributed by path pe_path[t].
    /// 0-based: pe_edge in [0,nedge_norm), pe_leaf in [0,npop), pe_path in [0,npath).
    std::vector<int> pe_edge;
    std::vector<int> pe_leaf;
    std::vector<int> pe_path;

    /// fill_pwts PATH-ADMIXEDGE table: parallel arrays of length n_pae. Path pae_path[t]
    /// traverses admixedge pae_admixedge[t] (1-based admixedge id in [1, 2*nadmix], the
    /// AT2 wts2 index — odd => theta_j, even => 1-theta_j).
    std::vector<int> pae_path;
    std::vector<int> pae_admixedge;

    /// The f3 basis pairs over the centered, base-dropped leaf columns (the AT2 cmb with
    /// +(1:0)): pair k = (cmb1[k], cmb2[k]) are 0-based CENTERED-COLUMN indices in
    /// [0, npop-1) (the non-base leaves in leaf order, the direct pwts_c column index),
    /// 0 <= cmb1 <= cmb2 < npop-1. The diagonal cmb1==cmb2 is the f3(base;i,i)=f2(base,i)
    /// entry. Centered column c maps to the c-th NON-base leaf (skip base_leaf in leaf
    /// order) — see centered_col_to_leaf().
    std::vector<int> cmb1;
    std::vector<int> cmb2;

    /// The non-admixture edge labels (for the result echo), in pwts0 row order.
    std::vector<std::string> edge_from;
    std::vector<std::string> edge_to;
    /// The admixture-node first/second parent labels, one per admixture node (theta j).
    /// weight[j] is the mass on admix_from[j]->admix_to[j]; the second parent carries
    /// 1-weight[j].
    std::vector<std::string> admix_from;
    std::vector<std::string> admix_to;

    std::string error;  ///< non-empty => parse failure (the message).
    [[nodiscard]] bool ok() const { return error.empty(); }

    /// Map a centered column index c (0..npop-2) to its full leaf index (the c-th
    /// non-base leaf in leaf order). Used to build the f3 basis triples (leaf -> f2 idx).
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

/// Parse the edge list into the fit-ready path-table model. `leaf_names` maps every f2
/// P-axis index to its population name (leaf_names[i] = pop i); a graph leaf must be one
/// of these. `f3basepop` (empty => the first leaf in graph leaf order) selects the f3
/// base population. On any structural problem returns a model with a non-empty `error`.
[[nodiscard]] QpGraphModel parse_qpgraph(const std::vector<QpGraphEdge>& edges,
                                         const std::vector<std::string>& leaf_names,
                                         const std::string& f3basepop = "");

/// fill_pwts (HOST oracle): given theta (the nadmix mixture weights), produce the
/// per-edge centered, base-dropped leaf-weight matrix pwts_c [nedge_norm x (npop-1)]
/// COLUMN-MAJOR (pwts_c[e + nedge_norm*j], j over the npop-1 non-base leaves in
/// 0..npop-2 = the leaf order with base removed), the EXACT AT2 sequence:
///   pwts = fill_pwts(pwts0, theta);  pwts = pwts[,-base] - pwts[,base].
/// This is the host reference the device kernel mirrors. theta length == nadmix.
///
/// INLINE (header-defined) so it is available wherever the objective header is included
/// WITHOUT a steppe::core link dependency — the CudaBackend (steppe::device) uses it for
/// the final ONE-eval edge recovery, and steppe::device must not pull steppe::core
/// symbols (the §4 layering: many device-only tests link device but not core).
inline void fill_pwts_centered(const QpGraphModel& m, const double* theta,
                               std::vector<double>& pwts_c) {
    const int ne = m.nedge_norm, np = m.npop;
    std::vector<double> path_w(static_cast<std::size_t>(m.npath), 1.0);
    for (std::size_t t = 0; t < m.pae_path.size(); ++t) {
        const int pi = m.pae_path[t];
        const int id = m.pae_admixedge[t];  // 1-based
        const int j = (id - 1) / 2;
        const double w = theta[j];
        const double v = (id % 2 == 1) ? w : (1.0 - w);
        path_w[static_cast<std::size_t>(pi)] *= v;
    }
    // pwts = pwts0; overwrite the (edge,leaf) cells with sum_path path_w (AT2
    // group_by(edge2,leaf2) summarize(w=sum(w))). The path-edge table is small.
    std::vector<double> pwts = m.pwts0;  // [ne x np] col-major
    // accumulate per (edge,leaf) cell sum, then apply (matches the host/device behavior).
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
    // center + drop base.
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

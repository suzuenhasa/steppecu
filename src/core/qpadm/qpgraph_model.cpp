// src/core/qpadm/qpgraph_model.cpp — the qpGraph topology data model.
//
// A single function, parse_qpgraph, turns a parent-to-child edge list into the
// fit-ready tables for an admixture graph (plain host C++, no GPU). It reproduces
// ADMIXTOOLS 2's graph_to_pwts / graph_to_weightind; the one convention steppe owns
// is the input-edge-order rule for which admixture parent carries theta (§2).
//
// Reference: docs/reference/src_core_qpadm_qpgraph_model.cpp.md
#include "core/qpadm/qpgraph_model.hpp"

#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/internal/index_cast.hpp"

namespace steppe::core::qpadm {

namespace {

// Node id registry, first-seen order — reference §4
struct NodeReg {
    std::vector<std::string> names;
    std::unordered_map<std::string, int> idx;
    int get(const std::string& s) {
        auto it = idx.find(s);
        if (it != idx.end()) return it->second;
        const int id = static_cast<int>(names.size());
        names.push_back(s);
        idx.emplace(s, id);
        return id;
    }
    [[nodiscard]] int find(const std::string& s) const {
        auto it = idx.find(s);
        return it == idx.end() ? -1 : it->second;
    }
};

}  // namespace

// parse_qpgraph: builds the graph model from an edge list — reference §3
QpGraphModel parse_qpgraph(const std::vector<QpGraphEdge>& edges,
                           const std::vector<std::string>& leaf_names,
                           const std::string& f3basepop) {
    QpGraphModel m;
    if (edges.empty()) { m.error = "qpgraph: empty edge list"; return m; }

    NodeReg reg;
    const int E = static_cast<int>(edges.size());
    std::vector<int> e_parent(E), e_child(E);
    for (int e = 0; e < E; ++e) {
        if (edges[idx(e)].from.empty() ||
            edges[idx(e)].to.empty()) {
            m.error = "qpgraph: an edge has an empty endpoint";
            return m;
        }
        e_parent[idx(e)] = reg.get(edges[idx(e)].from);
        e_child[idx(e)] = reg.get(edges[idx(e)].to);
    }
    const int V = static_cast<int>(reg.names.size());
    m.nedge_total = E;

    std::vector<int> indeg(idx(V), 0), outdeg(idx(V), 0);
    std::vector<std::vector<int>> out_edges(idx(V));
    std::vector<std::vector<int>> in_edges(idx(V));
    for (int e = 0; e < E; ++e) {
        const int p = e_parent[idx(e)], c = e_child[idx(e)];
        ++outdeg[idx(p)];
        ++indeg[idx(c)];
        out_edges[idx(p)].push_back(e);
        in_edges[idx(c)].push_back(e);
    }
    int root = -1;
    for (int v = 0; v < V; ++v) {
        if (indeg[idx(v)] == 0) {
            if (root != -1) { m.error = "qpgraph: multiple roots (graph is not singly-rooted)"; return m; }
            root = v;
        }
        if (indeg[idx(v)] > 2) {
            m.error = "qpgraph: node '" + reg.names[idx(v)] +
                      "' has in-degree > 2 (not a valid admixture node)";
            return m;
        }
    }
    if (root == -1) { m.error = "qpgraph: no root (every node has a parent — graph is cyclic)"; return m; }

    std::vector<int> leaf_nodes;
    for (int v = 0; v < V; ++v)
        if (outdeg[idx(v)] == 0) leaf_nodes.push_back(v);
    if (leaf_nodes.empty()) { m.error = "qpgraph: no leaves"; return m; }
    m.npop = static_cast<int>(leaf_nodes.size());
    m.npair = m.npop * (m.npop - 1) / 2;

    std::vector<int> node_leaf(idx(V), -1);
    for (int li = 0; li < m.npop; ++li) node_leaf[idx(leaf_nodes[idx(li)])] = li;

    std::unordered_map<std::string, int> f2idx;
    for (int i = 0; i < static_cast<int>(leaf_names.size()); ++i)
        f2idx.emplace(leaf_names[idx(i)], i);
    m.leaves.resize(idx(m.npop));
    m.leaf_to_f2.assign(idx(m.npop), -1);
    for (int li = 0; li < m.npop; ++li) {
        const std::string& nm = reg.names[idx(leaf_nodes[idx(li)])];
        m.leaves[idx(li)] = nm;
        auto it = f2idx.find(nm);
        if (it == f2idx.end()) {
            m.error = "qpgraph: leaf '" + nm + "' is not in the f2 population set";
            return m;
        }
        m.leaf_to_f2[idx(li)] = it->second;
    }
    m.base_leaf = 0;
    if (!f3basepop.empty()) {
        bool found = false;
        for (int li = 0; li < m.npop; ++li)
            if (m.leaves[idx(li)] == f3basepop) { m.base_leaf = li; found = true; break; }
        if (!found) { m.error = "qpgraph: f3basepop '" + f3basepop + "' is not a leaf"; return m; }
    }

    std::vector<int> admix_nodes;
    for (int v = 0; v < V; ++v)
        if (indeg[idx(v)] == 2) admix_nodes.push_back(v);
    m.nadmix = static_cast<int>(admix_nodes.size());

    std::vector<int> admixedge_of(idx(E), -1);
    m.admix_from.resize(idx(m.nadmix));
    m.admix_to.resize(idx(m.nadmix));
    for (int j = 0; j < m.nadmix; ++j) {
        const int an = admix_nodes[idx(j)];
        const std::vector<int>& ie = in_edges[idx(an)];
        const int e0 = ie[0], e1 = ie[1];
        admixedge_of[idx(e0)] = 2 * j + 1;
        admixedge_of[idx(e1)] = 2 * j + 2;
        m.admix_from[idx(j)] = reg.names[idx(e_parent[idx(e0)])];
        m.admix_to[idx(j)] = reg.names[idx(e_child[idx(e0)])];
    }

    std::vector<int> norm_of(idx(E), -1);
    m.nedge_norm = 0;
    m.edge_from.clear(); m.edge_to.clear();
    for (int e = 0; e < E; ++e) {
        if (admixedge_of[idx(e)] == -1) {
            norm_of[idx(e)] = m.nedge_norm++;
            m.edge_from.push_back(reg.names[idx(e_parent[idx(e)])]);
            m.edge_to.push_back(reg.names[idx(e_child[idx(e)])]);
        }
    }

    struct Path { std::vector<int> edge_seq; int leaf = -1; };
    std::vector<Path> paths;
    {
        std::vector<int> stack_edges;
        std::vector<char> on_stack(idx(V), 0);
        bool cycle = false;
        std::function<void(int)> dfs = [&](int node) {
            if (cycle) return;
            if (outdeg[idx(node)] == 0) {
                paths.push_back(Path{stack_edges, node_leaf[idx(node)]});
                return;
            }
            on_stack[idx(node)] = 1;
            for (int e : out_edges[idx(node)]) {
                const int child = e_child[idx(e)];
                if (on_stack[idx(child)]) {
                    cycle = true;
                    break;
                }
                stack_edges.push_back(e);
                dfs(child);
                stack_edges.pop_back();
                if (cycle) break;
            }
            on_stack[idx(node)] = 0;
        };
        dfs(root);
        if (cycle) {
            m.error = "qpgraph: cycle reachable from root (graph is not a DAG)";
            return m;
        }
    }
    m.npath = static_cast<int>(paths.size());

    std::vector<int> pathcount(idx(m.npop), 0);
    for (const Path& p : paths) ++pathcount[idx(p.leaf)];

    m.pwts0.assign(idx(m.nedge_norm) * idx(m.npop), 0.0);
    for (const Path& p : paths) {
        const int leaf = p.leaf;
        const double inc = 1.0 / static_cast<double>(pathcount[idx(leaf)]);
        for (int e : p.edge_seq) {
            const int ne = norm_of[idx(e)];
            if (ne < 0) continue;
            m.pwts0[idx(ne) + idx(m.nedge_norm) * idx(leaf)] += inc;
        }
    }

    m.pae_path.clear(); m.pae_admixedge.clear();
    for (int pi = 0; pi < m.npath; ++pi) {
        for (int e : paths[idx(pi)].edge_seq) {
            const int ae = admixedge_of[idx(e)];
            if (ae > 0) { m.pae_path.push_back(pi); m.pae_admixedge.push_back(ae); }
        }
    }
    std::map<std::pair<int, int>, int> cell_cnt;
    for (int pi = 0; pi < m.npath; ++pi) {
        const int leaf = paths[idx(pi)].leaf;
        for (int e : paths[idx(pi)].edge_seq) {
            const int ne = norm_of[idx(e)];
            if (ne < 0) continue;
            ++cell_cnt[{ne, leaf}];
        }
    }
    m.pe_edge.clear(); m.pe_leaf.clear(); m.pe_path.clear();
    for (int pi = 0; pi < m.npath; ++pi) {
        const int leaf = paths[idx(pi)].leaf;
        for (int e : paths[idx(pi)].edge_seq) {
            const int ne = norm_of[idx(e)];
            if (ne < 0) continue;
            const int cnt = cell_cnt[{ne, leaf}];
            if (cnt < pathcount[idx(leaf)]) {
                m.pe_edge.push_back(ne);
                m.pe_leaf.push_back(leaf);
                m.pe_path.push_back(pi);
            }
        }
    }

    const int ncc = m.npop - 1;
    m.cmb1.clear(); m.cmb2.clear();
    for (int a = 0; a < ncc; ++a)
        for (int b = a; b < ncc; ++b) {
            m.cmb1.push_back(a);
            m.cmb2.push_back(b);
        }
    if (static_cast<int>(m.cmb1.size()) != m.npair) {
        m.error = "qpgraph: internal pair-count mismatch";
        return m;
    }
    return m;
}

}  // namespace steppe::core::qpadm

// src/core/qpadm/qpgraph_model.cpp — the qpGraph topology data model (the new piece).
//
// Replicates admixtools graph_to_pwts / graph_to_weightind / fill_pwts (verified
// file:line against admixtools 2.0.10 R on box5090). The ONE convention choice we own:
// igraph stores an admix node's two in-edges sorted by source-vertex id (an internal
// detail), so which parent is `theta` vs `1-theta` is implementation-defined; AT2's
// score is INVARIANT to that swap (swapping the parents complements theta), so we adopt
// the deterministic INPUT-EDGE-ORDER convention (first incident edge -> theta) and
// report the fitted weight keyed by PARENT NAME. The fit score + the per-named-parent
// weight then match AT2 exactly regardless of the in-edge order choice.
#include "core/qpadm/qpgraph_model.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace steppe::core::qpadm {

namespace {

// A simple node id registry preserving first-seen order (the node "vertex" order).
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

QpGraphModel parse_qpgraph(const std::vector<QpGraphEdge>& edges,
                           const std::vector<std::string>& leaf_names,
                           const std::string& f3basepop) {
    QpGraphModel m;
    if (edges.empty()) { m.error = "qpgraph: empty edge list"; return m; }

    // ---- 1. Register nodes (first-seen order) + the directed edges (input order) ----
    NodeReg reg;
    const int E = static_cast<int>(edges.size());
    std::vector<int> e_parent(E), e_child(E);
    for (int e = 0; e < E; ++e) {
        if (edges[static_cast<std::size_t>(e)].from.empty() ||
            edges[static_cast<std::size_t>(e)].to.empty()) {
            m.error = "qpgraph: an edge has an empty endpoint";
            return m;
        }
        e_parent[static_cast<std::size_t>(e)] = reg.get(edges[static_cast<std::size_t>(e)].from);
        e_child[static_cast<std::size_t>(e)] = reg.get(edges[static_cast<std::size_t>(e)].to);
    }
    const int V = static_cast<int>(reg.names.size());
    m.nedge_total = E;

    // ---- 2. in/out degree -> root, leaves, admixture nodes -------------------------
    std::vector<int> indeg(static_cast<std::size_t>(V), 0), outdeg(static_cast<std::size_t>(V), 0);
    std::vector<std::vector<int>> out_edges(static_cast<std::size_t>(V));  // child edges per node
    std::vector<std::vector<int>> in_edges(static_cast<std::size_t>(V));   // parent edges per node (input order)
    for (int e = 0; e < E; ++e) {
        const int p = e_parent[static_cast<std::size_t>(e)], c = e_child[static_cast<std::size_t>(e)];
        ++outdeg[static_cast<std::size_t>(p)];
        ++indeg[static_cast<std::size_t>(c)];
        out_edges[static_cast<std::size_t>(p)].push_back(e);
        in_edges[static_cast<std::size_t>(c)].push_back(e);
    }
    int root = -1;
    for (int v = 0; v < V; ++v) {
        if (indeg[static_cast<std::size_t>(v)] == 0) {
            if (root != -1) { m.error = "qpgraph: multiple roots (graph is not singly-rooted)"; return m; }
            root = v;
        }
        if (indeg[static_cast<std::size_t>(v)] > 2) {
            m.error = "qpgraph: node '" + reg.names[static_cast<std::size_t>(v)] +
                      "' has in-degree > 2 (not a valid admixture node)";
            return m;
        }
    }
    if (root == -1) { m.error = "qpgraph: no root (every node has a parent — graph is cyclic)"; return m; }

    // leaves = out-degree 0 nodes, in graph node (first-seen) order (AT2 get_leafnames).
    std::vector<int> leaf_nodes;
    for (int v = 0; v < V; ++v)
        if (outdeg[static_cast<std::size_t>(v)] == 0) leaf_nodes.push_back(v);
    if (leaf_nodes.empty()) { m.error = "qpgraph: no leaves"; return m; }
    m.npop = static_cast<int>(leaf_nodes.size());
    m.npair = m.npop * (m.npop - 1) / 2;

    // node -> leaf index (or -1)
    std::vector<int> node_leaf(static_cast<std::size_t>(V), -1);
    for (int li = 0; li < m.npop; ++li) node_leaf[static_cast<std::size_t>(leaf_nodes[static_cast<std::size_t>(li)])] = li;

    // ---- 3. Map leaves to f2 P-axis indices; resolve the base leaf -----------------
    std::unordered_map<std::string, int> f2idx;
    for (int i = 0; i < static_cast<int>(leaf_names.size()); ++i)
        f2idx.emplace(leaf_names[static_cast<std::size_t>(i)], i);
    m.leaves.resize(static_cast<std::size_t>(m.npop));
    m.leaf_to_f2.assign(static_cast<std::size_t>(m.npop), -1);
    for (int li = 0; li < m.npop; ++li) {
        const std::string& nm = reg.names[static_cast<std::size_t>(leaf_nodes[static_cast<std::size_t>(li)])];
        m.leaves[static_cast<std::size_t>(li)] = nm;
        auto it = f2idx.find(nm);
        if (it == f2idx.end()) {
            m.error = "qpgraph: leaf '" + nm + "' is not in the f2 population set";
            return m;
        }
        m.leaf_to_f2[static_cast<std::size_t>(li)] = it->second;
    }
    m.base_leaf = 0;
    if (!f3basepop.empty()) {
        bool found = false;
        for (int li = 0; li < m.npop; ++li)
            if (m.leaves[static_cast<std::size_t>(li)] == f3basepop) { m.base_leaf = li; found = true; break; }
        if (!found) { m.error = "qpgraph: f3basepop '" + f3basepop + "' is not a leaf"; return m; }
    }

    // ---- 4. Admixture nodes + admixedges (AT2 admixedgesfull / normedges) ----------
    // An admixture node has in-degree 2. admixedges = the in-edges of admix nodes, in
    // admix-node (node) order then INPUT-EDGE order (our deterministic convention).
    // admixedge local id (1-based) = position in this list; the AT2 wts2 index:
    //   id 2j-1 -> theta_j ; id 2j -> 1-theta_j  (for admix node j, 0-based j).
    std::vector<int> admix_nodes;
    for (int v = 0; v < V; ++v)
        if (indeg[static_cast<std::size_t>(v)] == 2) admix_nodes.push_back(v);
    m.nadmix = static_cast<int>(admix_nodes.size());

    std::vector<int> admixedge_of(static_cast<std::size_t>(E), -1);  // edge -> 1-based admixedge id, or -1
    m.admix_from.resize(static_cast<std::size_t>(m.nadmix));
    m.admix_to.resize(static_cast<std::size_t>(m.nadmix));
    for (int j = 0; j < m.nadmix; ++j) {
        const int an = admix_nodes[static_cast<std::size_t>(j)];
        const std::vector<int>& ie = in_edges[static_cast<std::size_t>(an)];  // input-edge order
        const int e0 = ie[0], e1 = ie[1];
        admixedge_of[static_cast<std::size_t>(e0)] = 2 * j + 1;  // -> theta_j
        admixedge_of[static_cast<std::size_t>(e1)] = 2 * j + 2;  // -> 1-theta_j
        m.admix_from[static_cast<std::size_t>(j)] = reg.names[static_cast<std::size_t>(e_parent[static_cast<std::size_t>(e0)])];
        m.admix_to[static_cast<std::size_t>(j)] = reg.names[static_cast<std::size_t>(e_child[static_cast<std::size_t>(e0)])];
    }

    // normedges = edges that are NOT admixedges, in input-edge order; norm local index
    // = position. nedge_norm = E - 2*nadmix.
    std::vector<int> norm_of(static_cast<std::size_t>(E), -1);  // edge -> 0-based norm index, or -1
    m.nedge_norm = 0;
    m.edge_from.clear(); m.edge_to.clear();
    for (int e = 0; e < E; ++e) {
        if (admixedge_of[static_cast<std::size_t>(e)] == -1) {
            norm_of[static_cast<std::size_t>(e)] = m.nedge_norm++;
            m.edge_from.push_back(reg.names[static_cast<std::size_t>(e_parent[static_cast<std::size_t>(e)])]);
            m.edge_to.push_back(reg.names[static_cast<std::size_t>(e_child[static_cast<std::size_t>(e)])]);
        }
    }

    // ---- 5. Enumerate every root->leaf simple path (DFS) ---------------------------
    // Each path: the ordered edge list root..leaf, and its terminal leaf. (A simple DAG
    // path; admixture nodes induce multiple paths to a leaf.)
    struct Path { std::vector<int> edge_seq; int leaf = -1; };
    std::vector<Path> paths;
    {
        std::vector<int> stack_edges;
        // Recursion-stack guard: a cycle REACHABLE from a valid root (e.g. an admix
        // node fed by its own descendant) passes the indeg/outdeg/root checks above
        // but would make this DFS recurse forever. on_stack marks nodes on the current
        // root..node path; revisiting one is a back-edge == cycle. We then short-circuit
        // (cycle flag) and return the clean m.error instead of overflowing the stack.
        std::vector<char> on_stack(static_cast<std::size_t>(V), 0);
        bool cycle = false;
        std::function<void(int)> dfs = [&](int node) {
            if (cycle) return;
            if (outdeg[static_cast<std::size_t>(node)] == 0) {
                paths.push_back(Path{stack_edges, node_leaf[static_cast<std::size_t>(node)]});
                return;
            }
            on_stack[static_cast<std::size_t>(node)] = 1;
            for (int e : out_edges[static_cast<std::size_t>(node)]) {
                const int child = e_child[static_cast<std::size_t>(e)];
                if (on_stack[static_cast<std::size_t>(child)]) {  // back-edge -> cycle
                    cycle = true;
                    break;
                }
                stack_edges.push_back(e);
                dfs(child);
                stack_edges.pop_back();
                if (cycle) break;
            }
            on_stack[static_cast<std::size_t>(node)] = 0;
        };
        dfs(root);
        if (cycle) {
            m.error = "qpgraph: cycle reachable from root (graph is not a DAG)";
            return m;
        }
    }
    m.npath = static_cast<int>(paths.size());

    // pathcount per leaf (number of root->leaf paths).
    std::vector<int> pathcount(static_cast<std::size_t>(m.npop), 0);
    for (const Path& p : paths) ++pathcount[static_cast<std::size_t>(p.leaf)];

    // ---- 6. graph_to_pwts: pwts0 [nedge_norm x npop], += 1/pathcount per norm edge ---
    m.pwts0.assign(static_cast<std::size_t>(m.nedge_norm) * static_cast<std::size_t>(m.npop), 0.0);
    for (const Path& p : paths) {
        const int leaf = p.leaf;
        const double inc = 1.0 / static_cast<double>(pathcount[static_cast<std::size_t>(leaf)]);
        for (int e : p.edge_seq) {
            const int ne = norm_of[static_cast<std::size_t>(e)];
            if (ne < 0) continue;  // admixedge rows are dropped from pwts
            m.pwts0[static_cast<std::size_t>(ne) + static_cast<std::size_t>(m.nedge_norm) * static_cast<std::size_t>(leaf)] += inc;
        }
    }

    // ---- 7. graph_to_weightind: the fill_pwts path tables --------------------------
    // path_admixedge_table: for each path, the admixedges it traverses (1-based ids).
    // path_edge_table: for each (norm-edge, leaf) cell, the paths through it — but ONLY
    // the cells that VARY with theta (AT2 keep = cnt < numpaths: the leaf has SOME but
    // not ALL of its paths through that edge). Cells where cnt == pathcount[leaf] are
    // theta-independent (already correct in pwts0) and are NOT in the table.
    m.pae_path.clear(); m.pae_admixedge.clear();
    for (int pi = 0; pi < m.npath; ++pi) {
        for (int e : paths[static_cast<std::size_t>(pi)].edge_seq) {
            const int ae = admixedge_of[static_cast<std::size_t>(e)];
            if (ae > 0) { m.pae_path.push_back(pi); m.pae_admixedge.push_back(ae); }
        }
    }
    // Count cnt[(norm-edge, leaf)] = number of paths through that cell.
    std::map<std::pair<int, int>, int> cell_cnt;
    for (int pi = 0; pi < m.npath; ++pi) {
        const int leaf = paths[static_cast<std::size_t>(pi)].leaf;
        for (int e : paths[static_cast<std::size_t>(pi)].edge_seq) {
            const int ne = norm_of[static_cast<std::size_t>(e)];
            if (ne < 0) continue;
            ++cell_cnt[{ne, leaf}];
        }
    }
    m.pe_edge.clear(); m.pe_leaf.clear(); m.pe_path.clear();
    for (int pi = 0; pi < m.npath; ++pi) {
        const int leaf = paths[static_cast<std::size_t>(pi)].leaf;
        for (int e : paths[static_cast<std::size_t>(pi)].edge_seq) {
            const int ne = norm_of[static_cast<std::size_t>(e)];
            if (ne < 0) continue;
            const int cnt = cell_cnt[{ne, leaf}];
            if (cnt < pathcount[static_cast<std::size_t>(leaf)]) {  // AT2 keep
                m.pe_edge.push_back(ne);
                m.pe_leaf.push_back(leaf);
                m.pe_path.push_back(pi);
            }
        }
    }

    // ---- 8. cmb: combn(0:npop-1, 2)+(1:0) — the f3 basis pairs over centered cols ---
    // AT2's centered, base-dropped pwts has npop-1 columns (the non-base leaves in leaf
    // order); cmb indexes THOSE columns. npair=choose(npop,2) over npop-1 columns WITH
    // the diagonal (choose(npop-1+1,2)=choose(npop,2)), so the pairs are (a<=b) over the
    // npop-1 centered columns (a==b => the diagonal f3(base;i,i)=f2(base,i)). We store
    // cmb1/cmb2 as 0-based CENTERED-COLUMN indices (0..npop-2), the direct pwts_c index.
    const int ncc = m.npop - 1;  // centered column count
    m.cmb1.clear(); m.cmb2.clear();
    for (int a = 0; a < ncc; ++a)
        for (int b = a; b < ncc; ++b) {
            m.cmb1.push_back(a);
            m.cmb2.push_back(b);
        }
    // npair sanity (choose(npop,2)).
    if (static_cast<int>(m.cmb1.size()) != m.npair) {
        m.error = "qpgraph: internal pair-count mismatch";
        return m;
    }
    return m;
}

}  // namespace steppe::core::qpadm

// src/core/qpadm/qpgraph_enumerate.cpp
//
// Enumerates admixture-graph topologies (all trees and all one-admixture graphs)
// on a set of leaves. Host-only, CUDA-free, and deterministic — no RNG anywhere.
//
// Reference: docs/reference/src_core_qpadm_qpgraph_enumerate.cpp.md
#include "core/qpadm/qpgraph_enumerate.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace steppe::core::qpadm {

namespace {

// Named constants — reference §2
inline constexpr int kWlExtraRounds = 4;
inline constexpr int kFreshNodeBase = 500000;

// Integer graph representation — reference §3
struct IEdge { int p, c; };
using IGraph = std::vector<IEdge>;

// Canonical graph hash (leaf-anchored 1-WL color refinement) — reference §4
std::uint64_t hash_igraph(const IGraph& g, int nleaf) {
    std::unordered_set<int> nodes;
    for (const IEdge& e : g) { nodes.insert(e.p); nodes.insert(e.c); }
    std::unordered_map<int, std::vector<int>> in_adj, out_adj;
    std::unordered_map<int, int> indeg, outdeg;
    for (int u : nodes) { indeg[u] = 0; outdeg[u] = 0; }
    for (const IEdge& e : g) {
        out_adj[e.p].push_back(e.c);
        in_adj[e.c].push_back(e.p);
        ++outdeg[e.p];
        ++indeg[e.c];
    }
    std::unordered_map<int, std::string> color;
    for (int u : nodes) {
        if (u >= 0 && u < nleaf)
            color[u] = "L" + std::to_string(u);
        else
            color[u] = "I" + std::to_string(indeg[u]) + "_" + std::to_string(outdeg[u]);
    }
    const int rounds = 2 * static_cast<int>(nodes.size()) + kWlExtraRounds;
    for (int r = 0; r < rounds; ++r) {
        std::unordered_map<int, std::string> nc;
        for (int u : nodes) {
            std::vector<std::string> ins, outs;
            for (int w : in_adj[u]) ins.push_back(color[w]);
            for (int w : out_adj[u]) outs.push_back(color[w]);
            std::sort(ins.begin(), ins.end());
            std::sort(outs.begin(), outs.end());
            std::string s = color[u] + "|<";
            for (const std::string& x : ins) { s += x; s += ','; }
            s += ">|<";
            for (const std::string& x : outs) { s += x; s += ','; }
            s += '>';
            nc[u] = s;
        }
        std::vector<std::string> uniq;
        uniq.reserve(nc.size());
        for (const auto& kv : nc) uniq.push_back(kv.second);
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        std::unordered_map<std::string, int> code;
        for (int i = 0; i < static_cast<int>(uniq.size()); ++i) code[uniq[i]] = i;
        for (int u : nodes) color[u] = std::to_string(code[nc[u]]);
    }
    std::vector<std::string> ncolors;
    ncolors.reserve(nodes.size());
    for (int u : nodes) ncolors.push_back(color[u]);
    std::sort(ncolors.begin(), ncolors.end());
    std::vector<std::string> ecolors;
    ecolors.reserve(g.size());
    for (const IEdge& e : g) ecolors.push_back(color[e.p] + ">" + color[e.c]);
    std::sort(ecolors.begin(), ecolors.end());
    std::string canon = "N:";
    for (const std::string& s : ncolors) { canon += s; canon += ';'; }
    canon += "E:";
    for (const std::string& s : ecolors) { canon += s; canon += ';'; }
    std::uint64_t h = 1469598103934665603ULL;
    for (char ch : canon) { h ^= static_cast<unsigned char>(ch); h *= 1099511628211ULL; }
    return h;
}

// Structural helpers over IGraph (root, outgroup, reachability) — reference §3
int root_of(const IGraph& g) {
    std::unordered_set<int> has_parent, all_nodes;
    for (const IEdge& e : g) { has_parent.insert(e.c); all_nodes.insert(e.p); all_nodes.insert(e.c); }
    for (int u : all_nodes) if (!has_parent.count(u)) return u;
    return -1;
}

int outpop_of(const IGraph& g, int nleaf) {
    const int r = root_of(g);
    int found = -1, count = 0;
    for (const IEdge& e : g)
        if (e.p == r && e.c >= 0 && e.c < nleaf) { found = e.c; ++count; }
    return count == 1 ? found : -1;
}

bool reachable_from(const std::unordered_map<int, std::vector<int>>& adj, int src, int target) {
    std::vector<int> st{src};
    std::unordered_set<int> seen;
    while (!st.empty()) {
        const int u = st.back();
        st.pop_back();
        const auto it = adj.find(u);
        if (it == adj.end()) continue;
        for (int v : it->second) {
            if (v == target) return true;
            if (!seen.count(v)) { seen.insert(v); st.push_back(v); }
        }
    }
    return false;
}

[[maybe_unused]] bool reachable_from(const IGraph& g, int src, int target) {
    std::unordered_map<int, std::vector<int>> adj;
    for (const IEdge& e : g) adj[e.p].push_back(e.c);
    return reachable_from(adj, src, target);
}

// Tree enumerator: sequential leaf insertion — reference §5
std::vector<IGraph> enumerate_trees_int(int nleaf) {
    std::vector<IGraph> trees{IGraph{{-1, 0}}};
    int next_id = nleaf;
    for (int k = 1; k < nleaf; ++k) {
        std::vector<IGraph> grown;
        const int w = next_id++;
        for (const IGraph& t : trees) {
            for (std::size_t ei = 0; ei < t.size(); ++ei) {
                const IEdge& e = t[ei];
                IGraph nt;
                nt.reserve(t.size() + 2);
                for (std::size_t j = 0; j < t.size(); ++j) if (j != ei) nt.push_back(t[j]);
                nt.push_back({e.p, w});
                nt.push_back({w, e.c});
                nt.push_back({w, k});
                grown.push_back(std::move(nt));
            }
        }
        trees.swap(grown);
    }
    for (IGraph& t : trees) {
        IGraph keep;
        keep.reserve(t.size());
        for (const IEdge& e : t) if (e.p != -1) keep.push_back(e);
        t.swap(keep);
    }
    return trees;
}

// Name assignment: int ids back to labeled edges — reference §3
std::vector<QpGraphEdge> to_named(const IGraph& g, const std::vector<std::string>& leaves) {
    const int nleaf = static_cast<int>(leaves.size());
    std::vector<QpGraphEdge> out;
    out.reserve(g.size());
    auto name = [&](int id) -> std::string {
        if (id >= 0 && id < nleaf) return leaves[static_cast<std::size_t>(id)];
        return "N" + std::to_string(id);
    };
    for (const IEdge& e : g) out.push_back({name(e.p), name(e.c)});
    return out;
}

void admix1_children_of(const IGraph& tree, int nleaf, const std::vector<std::string>& leaves,
                        std::unordered_set<std::uint64_t>& seen,
                        std::vector<EnumeratedTopology>& out);

}  // namespace

// Public graph hash (labeled edge list to isomorphism key) — reference §4
std::uint64_t graph_hash(const std::vector<QpGraphEdge>& edges) {
    std::unordered_set<std::string> parents;
    for (const QpGraphEdge& e : edges) parents.insert(e.from);
    std::unordered_set<std::string> all_nodes;
    for (const QpGraphEdge& e : edges) { all_nodes.insert(e.from); all_nodes.insert(e.to); }
    std::vector<std::string> leaf_labels;
    for (const std::string& n : all_nodes) if (!parents.count(n)) leaf_labels.push_back(n);
    std::sort(leaf_labels.begin(), leaf_labels.end());
    std::unordered_map<std::string, int> id;
    int next = 0;
    for (const std::string& l : leaf_labels) id[l] = next++;
    const int nleaf = next;
    for (const std::string& n : all_nodes) if (!id.count(n)) id[n] = next++;
    IGraph g;
    g.reserve(edges.size());
    for (const QpGraphEdge& e : edges) g.push_back({id[e.from], id[e.to]});
    return hash_igraph(g, nleaf);
}

// Public enumerators (whole-space) — reference §9
std::vector<EnumeratedTopology> enumerate_trees(const std::vector<std::string>& leaves) {
    const int nleaf = static_cast<int>(leaves.size());
    std::vector<EnumeratedTopology> out;
    int idx = 0;
    for (const IGraph& g : enumerate_trees_int(nleaf)) {
        EnumeratedTopology et;
        et.edges = to_named(g, leaves);
        et.nadmix = 0;
        et.id = idx++;
        et.hash = hash_igraph(g, nleaf);
        out.push_back(std::move(et));
    }
    return out;
}

std::vector<EnumeratedTopology> enumerate_admix1(const std::vector<std::string>& leaves) {
    const int nleaf = static_cast<int>(leaves.size());
    std::vector<EnumeratedTopology> out;
    std::unordered_set<std::uint64_t> seen;
    for (const IGraph& tree : enumerate_trees_int(nleaf))
        admix1_children_of(tree, nleaf, leaves, seen, out);
    int idx = 0;
    for (EnumeratedTopology& et : out) et.id = idx++;
    return out;
}

std::vector<EnumeratedTopology> enumerate_bounded_space(const std::vector<std::string>& leaves,
                                                        int max_nadmix) {
    std::vector<EnumeratedTopology> out = enumerate_trees(leaves);
    if (max_nadmix >= 1) {
        std::vector<EnumeratedTopology> a1 = enumerate_admix1(leaves);
        out.insert(out.end(), a1.begin(), a1.end());
    }
    return out;
}

namespace {

// Recover the base tree (inverse of admix wiring) — reference §7
IGraph base_tree_of(const IGraph& g, [[maybe_unused]] int nleaf) {
    std::unordered_map<int, int> indeg, outdeg;
    std::unordered_set<int> nodes;
    for (const IEdge& e : g) { ++indeg[e.c]; ++outdeg[e.p]; nodes.insert(e.p); nodes.insert(e.c); }
    int a = -1;
    for (int u : nodes) if (indeg[u] == 2) { a = u; break; }
    if (a < 0) return g;
    int a_child = -1;
    std::vector<int> a_par;
    for (const IEdge& e : g) {
        if (e.p == a) a_child = e.c;
        if (e.c == a) a_par.push_back(e.p);
    }
    int x = -1, dest_from = -1;
    if (a_par.size() == 2) {
        const int p0 = a_par[0], p1 = a_par[1];
        auto has_other_child = [&](int p) {
            for (const IEdge& e : g) if (e.p == p && e.c != a) return true;
            return false;
        };
        if (has_other_child(p0)) { x = p0; dest_from = p1; }
        else { x = p1; dest_from = p0; }
    } else if (!a_par.empty()) {
        x = a_par[0];
        dest_from = a_par.size() > 1 ? a_par[1] : -1;
    }
    int source_to = -1, source_from = -1;
    for (const IEdge& e : g) {
        if (e.p == x && e.c != a) source_to = e.c;
        if (e.c == x) source_from = e.p;
    }
    IGraph t;
    for (const IEdge& e : g) {
        if (e.p == a || e.c == a || e.p == x || e.c == x) continue;
        t.push_back(e);
    }
    if (source_from >= 0 && source_to >= 0) t.push_back({source_from, source_to});
    if (dest_from >= 0 && a_child >= 0) t.push_back({dest_from, a_child});
    return t;
}

// Relabel labeled edges to int ids — reference §3
IGraph relabel_to_int(const std::vector<QpGraphEdge>& edges, const std::vector<std::string>& leaves) {
    std::unordered_map<std::string, int> id;
    for (int i = 0; i < static_cast<int>(leaves.size()); ++i) id[leaves[static_cast<std::size_t>(i)]] = i;
    int next = static_cast<int>(leaves.size());
    IGraph g;
    g.reserve(edges.size());
    auto get = [&](const std::string& nm) {
        auto it = id.find(nm);
        if (it != id.end()) return it->second;
        const int v = next++;
        id.emplace(nm, v);
        return v;
    };
    for (const QpGraphEdge& e : edges) g.push_back({get(e.from), get(e.to)});
    return g;
}

// Admix-1 wiring: all one-admixture children of a base tree — reference §6
void admix1_children_of(const IGraph& tree, int nleaf, const std::vector<std::string>& leaves,
                        std::unordered_set<std::uint64_t>& seen,
                        std::vector<EnumeratedTopology>& out) {
    const int r = root_of(tree);
    const int op = outpop_of(tree, nleaf);
    IGraph base;
    for (const IEdge& e : tree) if (!(op >= 0 && e.p == r && e.c == op)) base.push_back(e);
    std::unordered_map<int, std::vector<int>> tree_adj;
    for (const IEdge& e : tree) tree_adj[e.p].push_back(e.c);
    int next_id = kFreshNodeBase;
    for (std::size_t si = 0; si < base.size(); ++si) {
        for (std::size_t di = 0; di < base.size(); ++di) {
            if (si == di) continue;
            const IEdge& s = base[si];
            const IEdge& d = base[di];
            if (s.p == d.p || s.c == d.c || s.c == d.p || d.c == s.p) continue;
            if (reachable_from(tree_adj, d.c, s.p)) continue;
            const int x = next_id, a = next_id + 1;
            IGraph ng;
            for (const IEdge& e : tree)
                if (!((e.p == s.p && e.c == s.c) || (e.p == d.p && e.c == d.c))) ng.push_back(e);
            ng.push_back({s.p, x}); ng.push_back({x, s.c});
            ng.push_back({d.p, a}); ng.push_back({a, d.c}); ng.push_back({x, a});
            const std::uint64_t h = hash_igraph(ng, nleaf);
            if (seen.insert(h).second) {
                EnumeratedTopology et;
                et.edges = to_named(ng, leaves);
                et.nadmix = 1; et.id = 0; et.hash = h;
                out.push_back(std::move(et));
            }
        }
        next_id += 2;
    }
}

std::vector<int> children_of(const IGraph& t, int u) {
    std::vector<int> ch;
    for (const IEdge& e : t) if (e.p == u) ch.push_back(e.c);
    return ch;
}

// Tree rearrangement moves (NNI neighbors) — reference §8
void nni_tree_neighbors(const IGraph& tree, int nleaf, const std::vector<std::string>& leaves,
                        std::unordered_set<std::uint64_t>& seen,
                        std::vector<EnumeratedTopology>& out) {
    const int E = static_cast<int>(tree.size());
    for (int ei = 0; ei < E; ++ei) {
        const int u = tree[static_cast<std::size_t>(ei)].p;
        const int v = tree[static_cast<std::size_t>(ei)].c;
        if (v < nleaf) continue;
        const std::vector<int> uch = children_of(tree, u);
        const std::vector<int> vch = children_of(tree, v);
        if (uch.size() != 2 || vch.size() != 2) continue;
        const int u_other = (uch[0] == v) ? uch[1] : uch[0];
        int idx_u_other = -1;
        for (int j = 0; j < E; ++j)
            if (tree[static_cast<std::size_t>(j)].p == u && tree[static_cast<std::size_t>(j)].c == u_other) { idx_u_other = j; break; }
        if (idx_u_other < 0) continue;
        for (int k = 0; k < 2; ++k) {
            const int v_child = vch[static_cast<std::size_t>(k)];
            int idx_v_child = -1;
            for (int j = 0; j < E; ++j)
                if (tree[static_cast<std::size_t>(j)].p == v && tree[static_cast<std::size_t>(j)].c == v_child) { idx_v_child = j; break; }
            if (idx_v_child < 0) continue;
            IGraph ng = tree;
            ng[static_cast<std::size_t>(idx_u_other)] = IEdge{u, v_child};
            ng[static_cast<std::size_t>(idx_v_child)] = IEdge{v, u_other};
            const std::uint64_t h = hash_igraph(ng, nleaf);
            if (seen.insert(h).second) {
                EnumeratedTopology et;
                et.edges = to_named(ng, leaves);
                et.nadmix = 0; et.id = 0; et.hash = h;
                out.push_back(std::move(et));
            }
        }
    }
}

}  // namespace

// Local-move neighborhood — reference §9
std::vector<EnumeratedTopology> topology_neighbors(const EnumeratedTopology& current,
                                                   const std::vector<std::string>& leaves,
                                                   int max_nadmix) {
    const int nleaf = static_cast<int>(leaves.size());
    std::vector<EnumeratedTopology> out;
    std::unordered_set<std::uint64_t> seen;
    seen.insert(current.hash);

    const IGraph cur = relabel_to_int(current.edges, leaves);
    const IGraph base = (current.nadmix == 0) ? cur : base_tree_of(cur, nleaf);

    nni_tree_neighbors(base, nleaf, leaves, seen, out);
    if (max_nadmix >= 1) {
        admix1_children_of(base, nleaf, leaves, seen, out);
        std::unordered_set<std::uint64_t> nni_seen;
        std::vector<EnumeratedTopology> nni_bases;
        nni_seen.insert(hash_igraph(base, nleaf));
        nni_tree_neighbors(base, nleaf, leaves, nni_seen, nni_bases);
        for (const EnumeratedTopology& nb : nni_bases) {
            const IGraph nbt = relabel_to_int(nb.edges, leaves);
            admix1_children_of(nbt, nleaf, leaves, seen, out);
        }
    }
    return out;
}

}  // namespace steppe::core::qpadm

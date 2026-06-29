// src/core/qpadm/qpgraph_enumerate.cpp — the qpGraph topology enumerator (host-pure).
//
// Reproduces admixtools' OWN enumerator 1:1 (verified file:line against admixtools 2.0.10
// R on box5090, and EXACT hash-set match against the committed generate_all_graphs(5,1)
// fixture):
//   * enumerate_trees    == generate_all_trees   (sequential leaf-insertion; (2n-3)!!).
//   * enumerate_admix1   == generate_all_graphs(.,1): for each base tree, the AT2
//                           find_newedges (source,dest) pairs -> insert_admix wiring,
//                           de-duplicated by graph_hash (the canonical isomorphism key).
//
// AT2 find_newedges (verified): drop the fixed outgroup edge root->outpop from the edge
// set ENTIRELY (both as source AND dest); over the remaining ordered (source,dest) pairs
// keep those with source!=dest endpoints disjoint in the 4 AT2 configs AND no directed
// path dest_to ~> source_from (the acyclicity guard).
// AT2 insert_admix (verified): add a split node x on the source edge and the admix node a
// on the dest edge; edges (source_from->x, x->source_to, dest_from->a, a->dest_to, x->a),
// delete the two original edges. a has in-degree 2 (dest_from, x), out-degree 1 (dest_to).
//
// HOST-PURE, CUDA-FREE, DETERMINISTIC (no RNG — the (C) property find_graphs lacks).
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

/// Slack rounds added to the 2*|V| Weisfeiler-Lehman color-refinement bound. 1-WL stabilizes
/// in at most |V|-1 iterations (each round either splits a color class or is the fixed point);
/// the 2*|V| factor already over-covers that, and the +4 is conservative headroom so the loop
/// is provably past the fixed point on every graph in the enumerated space. Tuning-only, not
/// correctness-load-bearing (the hash is exact on the AT2 1485-set with this bound).
inline constexpr int kWlExtraRounds = 4;

/// Base id for the synthetic split/admix nodes that the admix-1 wiring introduces. Sits well
/// above any leaf or tree-internal id; hash_igraph's structural (1-WL) coloring quotients these
/// synthetic labels out, so the concrete base is cosmetic and a single shared value is safe
/// (one base, not two, avoids the drift the prior 100000/500000 split risked).
inline constexpr int kFreshNodeBase = 500000;

// An int-labeled directed edge (parent,child). Leaves are 0..nleaf-1; internal/admix
// nodes are >= nleaf (fresh ids). The synthetic pre-root pendant uses parent == -1.
struct IEdge { int p, c; };
using IGraph = std::vector<IEdge>;

// ---- the canonical graph hash (1-WL color refinement, leaf-anchored) ---------------
// Leaves (id in [0,nleaf)) are colored by their label; every other node by (indeg,outdeg).
// Iterated refinement folds in sorted neighbor-color multisets until stable; the final
// (sorted node-color multiset, sorted colored-edge multiset) is the isomorphism key. A
// proven invariant on the AT2 canonical 1485-set (zero collisions, exact set match).
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
    // initial colors as comparable strings.
    std::unordered_map<int, std::string> color;
    for (int u : nodes) {
        if (u >= 0 && u < nleaf)
            color[u] = "L" + std::to_string(u);                 // leaf, anchored by label
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
        // compress to small integer codes (stable: sort the distinct strings).
        std::vector<std::string> uniq;
        uniq.reserve(nc.size());
        for (const auto& kv : nc) uniq.push_back(kv.second);
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        std::unordered_map<std::string, int> code;
        for (int i = 0; i < static_cast<int>(uniq.size()); ++i) code[uniq[i]] = i;
        for (int u : nodes) color[u] = std::to_string(code[nc[u]]);
    }
    // assemble the (sorted node-color multiset, sorted colored-edge multiset) -> a 64-bit
    // FNV-1a over the canonical string.
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

// ---- helpers over IGraph ------------------------------------------------------------
int root_of(const IGraph& g) {
    std::unordered_set<int> has_parent, all_nodes;
    for (const IEdge& e : g) { has_parent.insert(e.c); all_nodes.insert(e.p); all_nodes.insert(e.c); }
    for (int u : all_nodes) if (!has_parent.count(u)) return u;
    return -1;
}

// the unique LEAF child of the root, or -1 (AT2 get_outpop: only when there is exactly one).
int outpop_of(const IGraph& g, int nleaf) {
    const int r = root_of(g);
    int found = -1, count = 0;
    for (const IEdge& e : g)
        if (e.p == r && e.c >= 0 && e.c < nleaf) { found = e.c; ++count; }
    return count == 1 ? found : -1;
}

// is `target` reachable FROM `src` along out-edges (src excluded), reusing a prebuilt
// out-adjacency map. Overload for the loop-invariant case (admix1_children_of's (si,di)
// double loop), where rebuilding `adj` from the same graph on every probe is wasted O(E) work.
bool reachable_from(const std::unordered_map<int, std::vector<int>>& adj, int src, int target) {
    std::vector<int> st{src};
    std::unordered_set<int> seen;
    while (!st.empty()) {
        const int u = st.back();
        st.pop_back();
        const auto it = adj.find(u);
        if (it == adj.end()) continue;  // no out-edges (matches operator[]'s empty-vector case).
        for (int v : it->second) {
            if (v == target) return true;
            if (!seen.count(v)) { seen.insert(v); st.push_back(v); }
        }
    }
    return false;
}

// is `target` reachable FROM `src` along out-edges (src excluded)? Builds the out-adjacency
// from `g` once and delegates to the prebuilt-adjacency overload (the single source of the BFS).
// [[maybe_unused]]: the lone in-TU caller (admix1_children_of) now passes a hoisted prebuilt
// `adj` directly; this graph-taking convenience overload is kept as the documented entry point.
[[maybe_unused]] bool reachable_from(const IGraph& g, int src, int target) {
    std::unordered_map<int, std::vector<int>> adj;
    for (const IEdge& e : g) adj[e.p].push_back(e.c);
    return reachable_from(adj, src, target);
}

// ---- the tree enumerator (sequential leaf-insertion) --------------------------------
// Start with the synthetic pendant ROOT(-1)->leaf0; insert leaf k (k=1..nleaf-1) by
// SPLITTING every existing edge with a fresh internal node. The pendant edge is dropped at
// the end (its child IS the real root, out-degree 2). Count == (2n-3)!! == AT2 numtrees.
std::vector<IGraph> enumerate_trees_int(int nleaf) {
    std::vector<IGraph> trees{IGraph{{-1, 0}}};
    int next_id = nleaf;
    for (int k = 1; k < nleaf; ++k) {
        std::vector<IGraph> grown;
        const int w = next_id++;  // one fresh internal id per insertion STEP (per-tree
                                  // independence: each tree owns its copy of the ids).
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
    // drop the synthetic pendant ROOT(-1)->realroot from every tree.
    for (IGraph& t : trees) {
        IGraph keep;
        keep.reserve(t.size());
        for (const IEdge& e : t) if (e.p != -1) keep.push_back(e);
        t.swap(keep);
    }
    return trees;
}

// ---- name assignment (deterministic, parse_qpgraph-ready) ---------------------------
// Leaves keep their f2 labels; internal/admix nodes get a unique deterministic label.
// The names are quotiented by the hash, so only uniqueness + leaf-label fidelity matter.
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

// Forward decl: every admix-1 graph built on ONE base tree (defined below alongside the
// other local-move helpers). Shared by enumerate_admix1 (whole-space) and topology_neighbors
// (local moves) so the AT2 find_newedges/insert_admix wiring lives in exactly one place.
void admix1_children_of(const IGraph& tree, int nleaf, const std::vector<std::string>& leaves,
                        std::unordered_set<std::uint64_t>& seen,
                        std::vector<EnumeratedTopology>& out);

}  // namespace

std::uint64_t graph_hash(const std::vector<QpGraphEdge>& edges) {
    // Re-index the labeled graph to ints with leaves first (label-anchored), then hash.
    // Leaf detection: a node that is never a parent is a leaf; we anchor leaf colors by
    // their LABEL so isomorphism respects leaf identity (the AT2 graph_hash semantics).
    std::unordered_set<std::string> parents;
    for (const QpGraphEdge& e : edges) parents.insert(e.from);
    // Collect leaf labels (childless nodes) in sorted order -> stable leaf ids.
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
    // For every base tree, emit its admix-1 children (shared with topology_neighbors). The
    // per-base synthetic split/admix node ids inside admix1_children_of are quotiented out by
    // hash_igraph's structural (1-WL) coloring, so the emitted hashes + the `seen` de-dup are
    // independent of those ids; only the sequential `et.id` differs, which we assign below.
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

// Recover the BASE TREE of an admix-1 graph: delete the admix node a (in-deg 2, out-deg 1),
// reconnecting its single child to the x-side parent (the split node x feeding a, in-deg 1,
// out-deg 2) — the inverse of insert_admix — then suppress the two degree-2 split nodes by
// contracting them out, leaving the original bifurcating tree. Returns the int-labeled tree
// re-built canonically. For a nadmix=0 graph returns the graph itself.
IGraph base_tree_of(const IGraph& g, [[maybe_unused]] int nleaf) {
    // detect the admix node (the unique in-deg-2 node).
    std::unordered_map<int, int> indeg, outdeg;
    std::unordered_set<int> nodes;
    for (const IEdge& e : g) { ++indeg[e.c]; ++outdeg[e.p]; nodes.insert(e.p); nodes.insert(e.c); }
    int a = -1;
    for (int u : nodes) if (indeg[u] == 2) { a = u; break; }
    if (a < 0) return g;  // already a tree.
    // a's child + its two parents.
    int a_child = -1;
    std::vector<int> a_par;
    for (const IEdge& e : g) {
        if (e.p == a) a_child = e.c;
        if (e.c == a) a_par.push_back(e.p);
    }
    // the split node x = a parent of a that has another (non-a) child it can re-parent to
    // (the source_to). The OTHER parent is dest_from. (insert_admix: x->a + x->source_to;
    // dest_from->a.) When BOTH parents qualify (both out-deg 2 — the symmetric case), either
    // choice reconstructs a VALID base tree, so we pick the first qualifying as x and the
    // OTHER as dest_from (NOT leaving dest_from unset — the bug that dropped a leaf when both
    // parents had out-degree 2).
    int x = -1, dest_from = -1;
    if (a_par.size() == 2) {
        const int p0 = a_par[0], p1 = a_par[1];
        // x must have a non-a child (so source_to exists). Prefer p0 if it qualifies.
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
    // x's OTHER child (source_to) + x's parent (source_from).
    int source_to = -1, source_from = -1;
    for (const IEdge& e : g) {
        if (e.p == x && e.c != a) source_to = e.c;
        if (e.c == x) source_from = e.p;
    }
    // rebuild: drop edges touching a and x; restore source_from->source_to and
    // dest_from->a_child (the two edges insert_admix split).
    IGraph t;
    for (const IEdge& e : g) {
        if (e.p == a || e.c == a || e.p == x || e.c == x) continue;
        t.push_back(e);
    }
    if (source_from >= 0 && source_to >= 0) t.push_back({source_from, source_to});
    if (dest_from >= 0 && a_child >= 0) t.push_back({dest_from, a_child});
    return t;
}

// Re-index a labeled QpGraphEdge list to ints (leaves 0..nleaf-1 by `leaves` order, others
// fresh) so the int generators can operate on it.
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

// Every admix-1 graph built on ONE base tree (the AT2 find_newedges/insert_admix children),
// de-duplicated by hash. The single home for the admix-1 wiring: enumerate_admix1 maps this
// over ALL base trees (whole-space enumeration); topology_neighbors calls it on the base tree
// + the NNI-neighbor base trees (the local add-admix / relocate moves).
void admix1_children_of(const IGraph& tree, int nleaf, const std::vector<std::string>& leaves,
                        std::unordered_set<std::uint64_t>& seen,
                        std::vector<EnumeratedTopology>& out) {
    const int r = root_of(tree);
    const int op = outpop_of(tree, nleaf);
    IGraph base;
    for (const IEdge& e : tree) if (!(op >= 0 && e.p == r && e.c == op)) base.push_back(e);
    // `tree` is loop-invariant across the (si,di) double loop below, so its out-adjacency is
    // too; build it ONCE here and reuse it on every reachability probe (was an O(E) rebuild
    // per probe -> O(E^2) rebuilds; now O(E) once). Behavior-identical to the per-call build.
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

// children of a node in a tree (int graph).
std::vector<int> children_of(const IGraph& t, int u) {
    std::vector<int> ch;
    for (const IEdge& e : t) if (e.p == u) ch.push_back(e.c);
    return ch;
}

// NNI tree neighbors of a rooted binary tree (the genuine LOCAL tree move): for each
// internal edge (u->v) with v internal, the two interchanges swap one of u's OTHER child
// subtrees with one of v's child subtrees. Returns the distinct neighbor trees (by hash).
// The NNI graph on rooted binary trees is connected, so a monotone hill-climb over NNI +
// the add/drop-admix moves descends to the exhaustive optimum (the recovery gate).
void nni_tree_neighbors(const IGraph& tree, int nleaf, const std::vector<std::string>& leaves,
                        std::unordered_set<std::uint64_t>& seen,
                        std::vector<EnumeratedTopology>& out) {
    // For each internal edge (u->v) with v internal, the two interchanges swap u's OTHER
    // child subtree (u_other) with one of v's two child subtrees (v_child). We rebuild the
    // edge list by EDGE INDEX (not by (parent,child) value matching, which can be ambiguous
    // when the reconstructed base tree reuses ids) — detach exactly the two indexed edges
    // and re-attach the swapped pair, so EVERY leaf is preserved.
    const int E = static_cast<int>(tree.size());
    for (int ei = 0; ei < E; ++ei) {
        const int u = tree[static_cast<std::size_t>(ei)].p;
        const int v = tree[static_cast<std::size_t>(ei)].c;
        if (v < nleaf) continue;  // v must be internal.
        const std::vector<int> uch = children_of(tree, u);
        const std::vector<int> vch = children_of(tree, v);
        if (uch.size() != 2 || vch.size() != 2) continue;
        const int u_other = (uch[0] == v) ? uch[1] : uch[0];
        // the edge INDEX of u->u_other (the one we will redirect; NOT the (u,v) edge ei).
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
            ng[static_cast<std::size_t>(idx_u_other)] = IEdge{u, v_child};   // u now parents v_child
            ng[static_cast<std::size_t>(idx_v_child)] = IEdge{v, u_other};   // v now parents u_other
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

std::vector<EnumeratedTopology> topology_neighbors(const EnumeratedTopology& current,
                                                   const std::vector<std::string>& leaves,
                                                   int max_nadmix) {
    // The bounded-space LOCAL move set (host-proposes / fleet-fits a small batch per step):
    //   * nadmix=0 current: NNI tree neighbors (same level) + ADD-ADMIX children of THIS
    //     tree (cross the boundary to nadmix=1; the global optimum is a nadmix=1 graph).
    //   * nadmix=1 current: DROP-ADMIX to its base tree + the base tree's NNI neighbors,
    //     RELOCATE the admix edge (the base tree's other admix-1 children), AND re-add the
    //     admix on EACH NNI-neighbor base tree (so the nadmix=1 graphs form a CONNECTED move
    //     graph — a monotone descent over it reaches the global optimum from any seed).
    // De-duplicated by graph_hash; never proposes `current` itself. The recovery of the
    // exhaustive global-best from ALL seeds is the falsifiable v1 gate.
    const int nleaf = static_cast<int>(leaves.size());
    std::vector<EnumeratedTopology> out;
    std::unordered_set<std::uint64_t> seen;
    seen.insert(current.hash);

    const IGraph cur = relabel_to_int(current.edges, leaves);
    const IGraph base = (current.nadmix == 0) ? cur : base_tree_of(cur, nleaf);

    // tree-level NNI neighbors of the base tree.
    nni_tree_neighbors(base, nleaf, leaves, seen, out);
    if (max_nadmix >= 1) {
        // admix-1 children of the base tree (add-admix / relocate moves on the same base).
        admix1_children_of(base, nleaf, leaves, seen, out);
        // + admix-1 children of each NNI-neighbor base tree (the nadmix=1 connectivity: an
        // NNI on the base tree carried into the nadmix=1 level). Collect the NNI base trees
        // first (a separate seen-set so we generate THEIR admix children even if the tree
        // itself is already in `out`).
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

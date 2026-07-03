# `qpgraph_enumerate.cpp` reference

## 1. Purpose

`src/core/qpadm/qpgraph_enumerate.cpp` generates admixture-graph *topologies* —
the shapes a qpGraph model can take — so that a search can score every candidate
and pick the best. It answers two questions exhaustively for a given set of leaf
populations:

1. What are all the plain trees (no admixture) on these leaves?
2. What are all the graphs with exactly one admixture event on these leaves?

It also provides a "neighbors of this graph" move set used by a hill-climbing
search, and a canonical hash that decides when two graphs are really the same
shape (isomorphic) so duplicates can be thrown away.

The whole file is host-only C++ with no GPU code and no randomness. Because there
is no random-number generator anywhere, the same leaves always produce the same
topologies in the same order, run after run. That determinism is a deliberate
property: the reference tool's own random graph-finder does not have it, and this
code exists partly to replace that nondeterminism with a stable, reproducible
enumeration.

Every construction here is a one-to-one reproduction of ADMIXTOOLS 2's own
enumerator, checked line-by-line against version 2.0.10 and checked for an exact
match against a committed fixture:

- The tree enumeration matches ADMIXTOOLS 2's `generate_all_trees`, producing
  exactly `(2n-3)!!` trees for `n` leaves (for example, 105 trees for 5 leaves).
- The one-admixture enumeration matches `generate_all_graphs(leaves, 1)`,
  producing exactly the same set of non-isomorphic graphs (1485 of them for 5
  leaves), verified as an exact hash-set match against the committed fixture.

The results come out as ordinary edge lists (`parent -> child` string pairs) that
the existing single-graph parser can read directly — there is no separate data
model to bolt on.

---

## 2. Named constants

Two named constants sit at the top of the file's private section. Neither one
changes any reported result; both are documented as safe-to-tune or cosmetic.

| Constant | Value | What it's for |
|---|---|---|
| `kWlExtraRounds` | `4` | Extra safety rounds added to the graph-hash refinement loop (see section 4). The refinement is guaranteed to reach its fixed point in at most `|V|-1` passes, and the loop already runs `2*|V|` passes, which more than covers that. The `+4` is conservative headroom so the loop is provably past the fixed point on every graph in the enumerated space. Tuning-only — not load-bearing for correctness. The hash is exact (zero collisions) on the 1485-graph reference set with this bound. |
| `kFreshNodeBase` | `500000` | The starting id number for the synthetic split/admix nodes that the one-admixture wiring introduces (see section 6). It sits well above any real leaf id or tree-internal id so there is no collision. The exact value is cosmetic: the hash quotients these synthetic labels out (section 4), so the graph's identity does not depend on which base number is used. A single shared base is used on purpose — an earlier design split the base into two separate values (100000 and 500000) and that split risked the two drifting apart, so it was collapsed to one. |

---

## 3. How graphs are represented internally

The public interface speaks in string-labeled edges (`QpGraphEdge`, which is a
`from`/`to` pair of population or node names). Internally, the file works on a
lighter integer representation and converts back to strings only when emitting a
result.

### The integer edge and graph types

- `IEdge` is a single directed edge with an integer parent `p` and integer child
  `c`. Edges always point from parent (ancestor) to child (descendant).
- `IGraph` is just a `std::vector<IEdge>` — a graph is nothing more than its list
  of edges.

### The node-id convention

The integer ids follow a fixed convention that the rest of the file relies on:

- Leaf nodes use ids `0` to `nleaf-1`, in the order the caller passed their
  labels.
- Internal nodes and admix nodes use ids `>= nleaf` (fresh, never-reused-within-a-graph
  ids).
- One special case: during tree building there is a synthetic "pre-root" pendant
  whose parent id is `-1`. It is a scaffold that gets removed before the tree is
  finished (section 5).

### Small helpers over a graph

Three short functions read structural facts out of an `IGraph`:

- `root_of` finds the root — the one node that is never anyone's child. (If the
  graph is malformed and has no such node it returns `-1`.)
- `outpop_of` finds the outgroup population: the single leaf that hangs directly
  off the root. It returns that leaf only when there is *exactly one* such leaf,
  matching ADMIXTOOLS 2's `get_outpop`, and returns `-1` otherwise.
- `reachable_from` answers "can I get from node `src` to node `target` by
  following edges downward?" — a depth-first reachability test used as the
  acyclicity guard when adding admixture edges. There are two versions: one takes
  a pre-built adjacency map (used inside the hot double loop so the map isn't
  rebuilt on every probe), and a convenience one that builds the map from a graph
  and delegates to the first. The convenience version is kept as the documented
  entry point even though the current in-file caller passes a pre-built map
  directly.

### Turning ids back into names

- `to_named` converts an integer graph back to string-labeled edges. Leaves get
  their original labels; every other node gets a unique deterministic name like
  `N500000`. Because the hash ignores internal-node labels, only two things
  matter about these names: that they are unique within the graph, and that leaf
  labels are preserved exactly.
- `relabel_to_int` goes the other way — it takes a string-labeled edge list and
  assigns integer ids (leaves first, in the caller's order; everything else
  fresh) so the integer machinery can operate on a graph that came in as strings.

---

## 4. The canonical graph hash (the isomorphism key)

The single hardest piece in the file is deciding when two graphs are the *same
shape*. Two admixture graphs can be drawn with different internal node names, or
have their edges listed in a different order, and still be identical as
structures. The enumeration produces many such duplicates and must collapse them.

The answer is a canonical hash: a 64-bit number computed so that two graphs get
the same number if and only if they are isomorphic (respecting leaf identity).
The de-duplication and the search's "is this the same graph I already have?"
check both rely on it. Equal hashes mean isomorphic graphs.

### The algorithm: leaf-anchored color refinement

The core routine `hash_igraph` uses a standard technique known as 1-WL
(one-dimensional Weisfeiler–Lehman) color refinement, adapted so that leaves keep
their identity:

1. **Initial colors.** Every node is given a starting "color" (a short string).
   A leaf's color is anchored to its label (for example `L3`), so leaf identity
   is never washed out — swapping two leaves gives a genuinely different graph.
   Every non-leaf node starts with a color built only from its in-degree and
   out-degree (for example `I2_1`), because internal nodes have no identity of
   their own.
2. **Refinement rounds.** Repeatedly, each node's new color is formed from its
   own current color plus the *sorted* multiset of its parents' colors and the
   *sorted* multiset of its children's colors. Sorting is what makes the result
   independent of edge order. After computing the new color strings, they are
   compressed back to small integer codes (by sorting the distinct strings and
   numbering them) so the strings don't grow without bound.
3. **How many rounds.** The loop runs `2*|V| + kWlExtraRounds` times, which is
   provably more than enough to reach the stable fixed point (see section 2).
4. **Assembling the key.** Once colors are stable, the routine builds a canonical
   string from the sorted list of node colors plus the sorted list of
   edges-labeled-by-their-endpoint-colors, then folds that string into a 64-bit
   number with the FNV-1a hash. Sorting both lists is again what makes the key
   independent of how the graph happened to be laid out.

This matches ADMIXTOOLS 2's `graph_hash` semantics. It is an empirically proven
invariant on the reference set: zero collisions and an exact set match across all
1485 canonical one-admixture graphs on 5 leaves.

### The public entry point

The exported `graph_hash` takes a string-labeled edge list and does the same
thing from the outside. It first works out which nodes are leaves (a leaf is any
node that is never a parent), sorts those leaf labels to give them stable ids,
assigns ids to the rest, and then calls `hash_igraph`. Anchoring the leaf colors
by label is what makes the hash respect leaf identity, exactly as ADMIXTOOLS 2
does.

---

## 5. Enumerating all trees (sequential leaf insertion)

`enumerate_trees_int` builds every rooted, two-way-branching, leaf-labeled tree
on `nleaf` leaves. The method is sequential leaf insertion:

1. Start with a single tree that is just the synthetic pendant edge `(-1) -> leaf 0`.
2. For each further leaf `k` (from 1 up to `nleaf-1`), take every tree built so
   far and, for *every* edge in it, produce a new tree by splitting that edge:
   insert a fresh internal node in the middle of the edge and hang the new leaf
   `k` off that node. Trying every edge choice, over every existing tree, is what
   makes the enumeration exhaustive, and doing it in a fixed order makes it a
   deterministic depth-first walk.
3. At the very end, drop the synthetic `(-1) -> ...` pendant from every tree. The
   child of that pendant is the real root, which now has the correct out-degree
   of two.

Each insertion step allocates exactly one fresh internal id, and every tree
carries its own copy of these ids, so ids never collide across trees. The final
count is `(2n-3)!!`, exactly matching ADMIXTOOLS 2's `numtrees`.

The exported `enumerate_trees` wraps this: it runs the integer enumeration,
converts each tree to named edges, tags it with `nadmix = 0`, a sequential `id`,
and its canonical hash, and returns the list.

---

## 6. Adding one admixture node (the admix-1 wiring)

`admix1_children_of` is the single home for the one-admixture construction. Given
one base tree it produces every graph obtained by adding exactly one admixture
event to that tree. Both the whole-space enumerator and the local-move
neighborhood call it, so the wiring lives in exactly one place.

### Choosing the two edges

The construction adds an admixture node that draws from one edge (the *source*)
and lands on another (the *destination*). Following ADMIXTOOLS 2's `find_newedges`
exactly:

- The fixed outgroup edge (root to the single outgroup leaf) is removed from the
  candidate set entirely — it can be neither a source nor a destination.
- Over the remaining ordered pairs of distinct edges, a pair is a valid
  (source, destination) only when the two edges share none of their four
  endpoints, and when the destination is not an ancestor of the source. That last
  condition — checked with `reachable_from` — is the acyclicity guard: it keeps
  the new admixture edge from creating a cycle.

### Wiring the two new nodes

For each accepted pair, following ADMIXTOOLS 2's `insert_admix` exactly, two
fresh nodes are introduced:

- A split node `x` is inserted on the source edge, so `source_parent -> x` and
  `x -> source_child`.
- An admix node `a` is inserted on the destination edge, so `dest_parent -> a`
  and `a -> dest_child`.
- A new admixture edge `x -> a` connects them.
- The two original edges are deleted.

The result is that `a` has in-degree 2 (its two ancestries come from `dest_parent`
and from `x`) and out-degree 1 (leading to `dest_child`). Each finished graph is
hashed; only the first graph seen for a given hash is kept, so isomorphic
duplicates are discarded via a shared `seen` set.

### Two implementation notes

- **Adjacency is built once.** The base tree does not change across the inner
  double loop over edge pairs, so its downward-adjacency map is built a single
  time and reused on every reachability probe. Rebuilding it per probe was
  quadratic wasted work; building it once is linear.
- **Fresh ids are reused within a source.** All candidate destinations for a
  given source edge reuse the same two ids for `x` and `a`, and the base id is
  bumped by two only after each source edge. This is safe and cosmetic: internal
  ids only need to be locally distinct within a single graph, and the hash
  quotients internal labels out anyway.

`enumerate_admix1` maps this over *every* base tree, feeding all of them through a
single shared `seen` set so the whole one-admixture space is de-duplicated
together, then assigns each surviving graph a sequential `id`. The count matches
ADMIXTOOLS 2's non-isomorphic `generate_all_graphs(n, 1)`.

---

## 7. Recovering the base tree (the inverse of the admix wiring)

`base_tree_of` runs the section-6 construction backward: given a one-admixture
graph, it removes the admixture and returns the plain tree underneath. The
local-move search needs this so it can drop an admixture back to its base and
re-explore from there.

The procedure:

1. Find the admix node `a` — the unique node with in-degree 2.
2. Find `a`'s single child and its two parents. One parent is the split node `x`
   (which also feeds a real subtree); the other is the destination-side parent.
3. Delete the edges touching `a` and `x`, then restore the two edges that
   `insert_admix` had originally split — reconnecting the source parent to the
   source child, and the destination parent to `a`'s child. The two now-degenerate
   split points are contracted out, leaving the original bifurcating tree.

If the graph has no admix node (it is already a tree), it is returned unchanged.

### The symmetric-case fix

Choosing which of `a`'s two parents is `x` needs care. `x` must be the parent
that still has another (non-`a`) child to re-parent to. Sometimes *both* parents
qualify — the symmetric case where both have out-degree two. Either choice
reconstructs a valid base tree, so the code picks the first qualifying parent as
`x` and explicitly assigns the *other* one as the destination-side parent. Not
assigning that other parent was a real bug: it dropped a leaf whenever both
parents had out-degree two. The current code is careful to always set it.

---

## 8. Tree rearrangement moves (NNI)

`nni_tree_neighbors` produces the neighboring trees of a given tree under
Nearest-Neighbor Interchange, the standard small local rearrangement on rooted
binary trees. For each internal edge `u -> v` where `v` is itself internal, there
are two interchanges: swap one of `u`'s *other* child subtrees with one of `v`'s
two child subtrees. Each distinct resulting tree (by hash) is returned.

The important implementation detail is that the swap is done **by edge index, not
by matching parent/child id values**. After a base tree has been reconstructed by
section 7, its internal ids can be reused in ways that make value-matching
ambiguous — two edges could look the same by their `(parent, child)` numbers. So
the routine locates the exact two edges to move by their positions in the edge
list, detaches precisely those two, and re-attaches the swapped pair. This
guarantees every leaf is preserved and no edge is accidentally moved twice.

The set of trees reachable by repeated NNI moves is connected: any binary tree can
be reached from any other by a sequence of them. That connectivity is what lets a
hill-climb that only ever takes small steps still reach the global best.

---

## 9. The public enumerators and the local-move neighborhood

Three exported functions assemble the pieces above into what a search actually
calls.

### Whole-space enumeration

- `enumerate_trees` (section 5) returns the full `nadmix = 0` space.
- `enumerate_admix1` (section 6) returns the full de-duplicated `nadmix = 1`
  space.
- `enumerate_bounded_space` concatenates them: it returns the trees, and if
  `max_nadmix` is at least 1, appends the one-admixture graphs after them, in that
  order. This is the exhaustive candidate set an exhaustive search scores in one
  pass.

### The local-move neighborhood

`topology_neighbors` returns every topology that is one move away from a given
`current` graph — the move set a hill-climbing search steps through. It never
returns `current` itself, and it de-duplicates by hash (seeding the `seen` set
with `current`'s own hash up front). The moves depend on where `current` sits:

- **From a tree (`nadmix = 0`):** the NNI neighbors of that tree (staying at the
  same level), plus the add-admixture children of that tree (crossing up to
  `nadmix = 1`). Crossing up matters because the global best is generally a
  one-admixture graph.
- **From a one-admixture graph (`nadmix = 1`):** first the base tree is recovered
  (section 7). The neighbors are then the NNI neighbors of that base tree (which
  amounts to dropping the admixture and rearranging), the *other* add-admixture
  children of the same base tree (relocating the admixture edge), and — crucially
  — the add-admixture children of every NNI-neighbor base tree as well. That last
  group is what makes the one-admixture graphs form a *connected* move graph: an
  NNI applied to the base tree, carried up into the one-admixture level. Without
  it, a hill-climb could get stranded; with it, a monotone descent reaches the
  global optimum from any starting graph.

To generate the NNI-neighbor base trees' admixture children, the function collects
those neighbor trees using a separate local `seen` set (so their admixture
children are still generated even when the tree itself already appears in the
output), then re-labels each back to integers and runs the section-6 wiring on it.

One caller-facing caveat: neighbors returned here carry `id == 0`, which is *not*
a meaningful enumeration index — unlike the whole-space enumerators, where `id` is
a stable position. Neighbor results are identified and de-duplicated by their
`hash`, not by `id`.

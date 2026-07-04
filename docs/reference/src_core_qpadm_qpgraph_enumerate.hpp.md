# `qpgraph_enumerate.hpp` reference

## 1. Purpose

`src/core/qpadm/qpgraph_enumerate.hpp` is the host-side generator that produces
every candidate admixture-graph *topology* the qpGraph search is allowed to try. A
topology here is the shape of a population tree — which populations descend from
which, and (optionally) which population is a mix of two ancestral sources. The
search works by generating this whole set of shapes up front, then scoring each one
against the data on the GPU and keeping the best.

The header declares only plain data and free functions. It is pure host code with no
CUDA in it, and it uses no random numbers. Because it is deterministic, the same call
always yields the same topologies in the same order, and every topology carries a
stable index. This is a property the older randomized graph finder did not have.

Two concepts recur throughout:

- **Leaf** — a named population you supplied. Leaves are the only fixed, labeled
  points in a graph. Internal branch points and admixture nodes are unnamed
  structure that the code invents and labels deterministically.
- **`nadmix`** — the number of admixture nodes in the graph, i.e. how many
  populations are modeled as a two-source mixture. `nadmix = 0` is an ordinary tree;
  `nadmix = 1` adds exactly one admixed population. This file covers the bounded
  space of `nadmix ≤ 1`.

Everything is generated to exactly reproduce the reference enumeration[^at2]. The
counts this file produces match the `numtrees` / `numtreesadmix`
functions one-for-one, and each generated topology is an edge list that the existing
single-graph parser can consume directly, with no special-case handling.

---

## 2. `EnumeratedTopology` — one candidate graph

`EnumeratedTopology` is the result type every function in this file hands back (one
per candidate). It bundles the graph itself with enough provenance to identify and
de-duplicate it.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `edges` | `vector<QpGraphEdge>` | empty | The graph as a list of parent→child edges. This is exactly the format the existing single-graph parser reads, so an enumerated topology can be scored with no conversion. |
| `nadmix` | `int` | `0` | How many admixture nodes this graph has (0 for a plain tree, 1 for a one-admixture graph). |
| `id` | `int` | `0` | A stable index of this topology *within its `nadmix` level*, following the enumeration order. Because generation is deterministic, the same topology always gets the same `id` across runs. See the caveat in section 7: neighbor results do **not** carry a meaningful `id`. |
| `hash` | `uint64_t` | `0` | The canonical graph hash (see section 6). Two topologies are the same shape (isomorphic) exactly when their hashes are equal. This is the key used to remove duplicate shapes and to recognize when the search has revisited a graph it already saw. |

The important pairing to remember: `id` is a position in an ordered enumeration, while
`hash` is an identity that ignores how internal nodes happen to be labeled. When a
result was produced by whole-space enumeration, use `id` as a stable name; when it
came from a neighbor move, only `hash` is meaningful.

---

## 3. `enumerate_trees` — all plain trees

```
vector<EnumeratedTopology> enumerate_trees(const vector<string>& leaves);
```

Generates *every* rooted, bifurcating, leaf-labeled tree on the given populations —
the complete `nadmix = 0` set. "Bifurcating" means every internal branch point has
exactly two children; "leaf-labeled" means each supplied population sits at exactly
one leaf.

- **How it builds them.** It inserts leaves one at a time. It starts with just the
  root joined to the first leaf, then adds each further leaf by splitting one
  existing edge with a fresh internal node. Recursing over every possible edge choice
  visits the entire set of trees exactly once. The supplied leaf strings are used
  verbatim as the leaf node names; the invented internal nodes receive fresh,
  deterministic labels.
- **Order.** A canonical depth-first traversal, so the sequence — and therefore each
  topology's `id` — is identical on every run.
- **How many.** Exactly `(2n − 3)!!` trees for `n` leaves (the double factorial:
  `1 × 3 × 5 × …`). This equals `numtrees(n)`[^at2]. As a checked example,
  5 leaves produce 105 trees.

---

## 4. `enumerate_admix1` — all one-admixture graphs

```
vector<EnumeratedTopology> enumerate_admix1(const vector<string>& leaves);
```

Generates *every distinct-shaped* graph with exactly one admixture node — the
complete, de-duplicated `nadmix = 1` set.

- **How it builds them.** For each base tree from section 3, it places one admixture
  node so that it draws from an unordered pair of two distinct edges. The two edges
  must be *incomparable* — neither lies on the path from the root to the other (one
  is not an ancestor of the other) — because an admixture node genuinely mixes two
  separate lineages. This mirrors the labeled construction of choosing 2
  of the `2n − 2` edges[^at2].
- **Why de-duplication is needed.** That labeled construction generates the same
  underlying shape more than once (different edge labelings, one real graph). The
  function collapses these to distinct shapes using the canonical hash from section
  6, keeping one representative of each.
- **How many.** Exactly the count of non-isomorphic graphs
  `generate_all_graphs(n, 1)` yields[^at2]. As a checked example, 5 leaves produce 1485
  distinct one-admixture graphs.

---

## 5. `enumerate_bounded_space` — the whole search set

```
vector<EnumeratedTopology> enumerate_bounded_space(
    const vector<string>& leaves, int max_nadmix);
```

Produces the full candidate set the search scores, as a single list. It concatenates
the plain trees (`nadmix = 0`) followed by the one-admixture graphs (`nadmix = 1`),
in that order, with `max_nadmix` capping how far up the admixture ladder to go. This
is the complete, exhaustive space handed to the GPU scorer in one batch. Since the
two levels are simply the outputs of sections 3 and 4 in sequence, the ordering and
each topology's `id` remain stable and deterministic.

---

## 6. `graph_hash` — the shape identity

```
uint64_t graph_hash(const vector<QpGraphEdge>& edges);
```

Computes a canonical hash of a graph that depends only on its *shape*, not on how its
internal nodes happen to be named. The leaves (the named populations) are the only
fixed anchor points; the labels of internal and admixture nodes are ignored, since
those are arbitrary. Two graphs produce the same hash exactly when they are
isomorphic — the same shape drawn with possibly different internal labels.

- **How it works.** It runs a color-refinement pass (a 1-Weisfeiler-Leman-style
  scheme) over the graph as a directed acyclic graph anchored at the leaves. Each
  node's identity is refined from its neighbors' identities repeatedly until stable,
  then combined into one order-independent value. Because it is order-independent, the
  same shape hashes the same regardless of how its edges were listed.
- **Where it is used.** Two places: removing duplicate shapes during enumeration
  (sections 4 and 7), and letting the hill-climb heuristic recognize when it has
  wandered back to a graph it already visited.

Equal hashes mean isomorphic graphs; this is the identity the whole file relies on to
mean "the same graph."

---

## 7. `topology_neighbors` — moves for the hill-climb

```
vector<EnumeratedTopology> topology_neighbors(
    const EnumeratedTopology& current, const vector<string>& leaves,
    int max_nadmix);
```

Returns every topology that is one local move away from `current`. This is what a
hill-climbing search uses to explore outward from a good graph instead of scoring the
entire space: propose the neighbors on the host, let the GPU scorer re-score them,
step to the best, repeat.

The moves it considers:

- At `nadmix = 0`: nearest-neighbor and subtree-prune-and-regraft style tree
  rearrangements (small, local reshapings of the tree).
- Across the admixture boundary: relocating the single admixture edge, and adding or
  dropping the one admixture node — which moves the graph between the `nadmix = 0` and
  `nadmix = 1` levels.

The returned neighbors are de-duplicated by the canonical hash (section 6) and
filtered to stay inside the bounded space (`nadmix ≤ max_nadmix`).

**Caveat — `id` is not meaningful here.** Every returned neighbor carries `id == 0`.
Unlike the whole-space enumerators of sections 3–5, where `id` is a genuine, stable
enumeration index, neighbor results are **not** indexed. Identify, de-duplicate, and
compare neighbors by their `hash` (their shape identity), never by `id`.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

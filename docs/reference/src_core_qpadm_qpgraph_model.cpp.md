# `qpgraph_model.cpp` reference

## 1. Purpose

`src/core/qpadm/qpgraph_model.cpp` implements a single function, `parse_qpgraph`,
which turns a list of parent-to-child edges into a fit-ready description of an
admixture graph. An admixture graph is a family tree of populations that allows
mixing: most nodes have one parent (a plain split), but a node with two parents is
an *admixture node* — a population formed by mixing two ancestral sources in some
proportion.

The function takes the raw edge list and produces every table the later fitting
stages need: which edges are plain "drift" edges whose lengths get fitted, which
nodes are admixture nodes whose mixing proportions get fitted, how each leaf
population's ancestry flows down through the graph, and the list of population pairs
that form the statistical basis. All of that is packed into a `QpGraphModel` struct
(defined and field-documented in the companion header `qpgraph_model.hpp`).

The whole file is plain host C++ with no GPU code. It builds flat integer arrays
that the GPU fitting kernels later upload once into device memory, and the
CPU reference path reads the same struct directly. The math it computes is a
faithful reproduction of the corresponding functions in ADMIXTOOLS 2 (version
2.0.10), checked line by line against that source. Where ADMIXTOOLS 2 leaves a
choice up to an internal implementation detail, steppe pins down one deterministic
convention — described in section 2 — chosen so the final fitted numbers still
match ADMIXTOOLS 2 exactly.

---

## 2. The parent-order convention (which parent gets theta)

Every admixture node has exactly two incoming edges — its two parents. One parent
contributes a fraction called *theta* of the node's ancestry; the other contributes
the remaining `1 - theta`. Nothing in the graph itself says which parent is "the
theta one," so a convention is needed.

ADMIXTOOLS 2 uses a graph library (igraph) that stores an admixture node's two
in-edges sorted by the internal numeric id of their source vertex. That ordering is
an implementation detail of the library, not something a user controls, so treating
"the first stored parent" as the theta parent is not portable.

The key fact that makes this safe: ADMIXTOOLS 2's fit score does not change if you
swap the two parents. Swapping them simply replaces `theta` with `1 - theta`, which
describes the exact same mixture. Because the score is invariant to the swap, steppe
is free to pick its own rule.

The rule steppe owns is **input-edge order**: whichever of the node's two parent
edges appears first in the caller's edge list becomes the theta parent, and the
second becomes the `1 - theta` parent. To make the reported result independent of
this choice, the fitted mixture weight is reported keyed by the **parent's name**,
not by "first" or "second." So no matter which internal ordering either tool used,
the fit score and the per-named-parent weight steppe reports both match ADMIXTOOLS 2
exactly.

---

## 3. Inputs and outputs

`parse_qpgraph` takes three arguments:

| Argument | Meaning |
|---|---|
| `edges` | The graph as a list of directed parent-to-child rows. A node that appears as the `to` end of two different rows is an admixture node. |
| `leaf_names` | The population names on the f2 axis, in order: `leaf_names[i]` is the name of population `i`. Every graph leaf must be one of these names — this is how a leaf gets connected to its precomputed f2 statistics. |
| `f3basepop` | Which leaf to use as the base (reference) population for the f3 basis. Empty means "use the first leaf in graph order." |

It returns a `QpGraphModel`. On any structural problem the returned model carries a
non-empty `error` string and the caller treats that as a parse failure; the
individual failure cases are listed in section 10. The struct's fields (the leaf
list, the weight matrix `pwts0`, the path tables, the pair list, and the edge
labels) are documented in the header. Sections 4 through 9 explain how each of them
is built.

---

## 4. Registering nodes and edges

The graph is given by name (strings like `"Root"`, `"Mbuti"`), so the first step
assigns every distinct node name a small integer id. A helper registry (`NodeReg`)
hands out ids in **first-seen order** — the first new name encountered becomes node
0, the next new name node 1, and so on. This first-seen order is the same node
ordering ADMIXTOOLS 2 uses, and it becomes the leaf ordering later, so it must be
preserved exactly.

Each edge is then recorded as a pair of node ids (`parent`, `child`) in the same
order the caller supplied the edges. That original input order is deliberately kept,
because it is what drives the parent-order convention from section 2. If any edge
has an empty `from` or `to` name, the parse fails immediately.

---

## 5. Finding the root, leaves, and admixture nodes

With the edges in integer form, the code counts each node's in-degree (number of
parents) and out-degree (number of children), and remembers the list of parent
edges and child edges attached to each node. Those degree counts classify every
node:

- **Root** — the one node with in-degree 0 (no parents). There must be exactly one.
  Finding a second in-degree-0 node is an error ("multiple roots"); finding none is
  an error ("no root"), which also means every node has a parent and the graph must
  contain a cycle.
- **Admixture node** — a node with in-degree exactly 2. An in-degree greater than 2
  is rejected, because a mixture of more than two sources is not a valid admixture
  node in this model.
- **Leaf** — a node with out-degree 0 (no children). Leaves are collected in node
  (first-seen) order, matching ADMIXTOOLS 2's leaf-name routine. There must be at
  least one.

The number of leaves becomes `npop`, and the number of leaf pairs becomes
`npair = npop * (npop - 1) / 2`. A lookup array is also built mapping each node back
to its leaf index (or -1 if the node is not a leaf).

---

## 6. Mapping leaves to the f2 population set and the base leaf

Each graph leaf must correspond to a real population that has precomputed f2
statistics. This step builds a name-to-index lookup over `leaf_names` and, for every
leaf, records its position on the f2 axis in `leaf_to_f2`. A leaf whose name is not
found in the f2 population set is an error.

The base leaf is resolved here too. By default the base is leaf 0 (the first leaf in
graph order). If the caller passed a non-empty `f3basepop`, the code searches the
leaves for that name and uses it instead; a name that is not one of the leaves is an
error. The base leaf is the reference population that the f3 basis is centered on
(see section 9).

---

## 7. Admixedges, normedges, and the theta indexing

The edges are split into two kinds:

- **Admixedges** are the incoming edges of admixture nodes — the edges that carry a
  mixing proportion. They are collected admixture-node by admixture-node, and within
  each node in input-edge order (the convention from section 2). Each admixedge is
  given a **1-based id**. For admixture node `j` (0-based), its first parent edge
  gets id `2j + 1` and its second gets id `2j + 2`. The rule tying an id back to a
  proportion is: an **odd** id means `theta` of that admixture node, an **even** id
  means `1 - theta`. This is the same "wts2" index scheme ADMIXTOOLS 2 uses. For
  each admixture node the code also records the name of its first parent
  (`admix_from`) and the node itself (`admix_to`), so the fitted weight can be
  reported by parent name.

- **Normedges** ("normal" edges) are all the remaining edges — the plain drift edges
  whose lengths are what the fit actually estimates. They are numbered with a 0-based
  index in input-edge order, and their parent/child names are stored for echoing in
  the result. The count of them is `nedge_norm`, which equals the total edge count
  minus `2 * nadmix` (each admixture node consumes two edges as admixedges).

Only normedges get rows in the weight matrix `pwts0`; admixedges do not. The effect
of an admixedge on ancestry flow is captured instead through the path tables in
section 9, as a multiplier on the paths that pass through it.

---

## 8. Enumerating root-to-leaf paths (and catching cycles)

To know how ancestry reaches each leaf, the code walks the graph from the root and
records **every** simple path (a path that never repeats a node) from the root down
to a leaf. It does this with a depth-first search that keeps the current sequence of
edges on a stack; each time it reaches a leaf it saves that edge sequence together
with the leaf it ended at. Because an admixture node can be reached along two
different routes, a single leaf can have several root-to-leaf paths — that
multiplicity is exactly what the later weighting has to account for.

Cycle detection happens in two separate places, because a bad graph can fail in two
different ways:

1. A cycle that involves the whole graph having no entry point is caught back in
   section 5 as "no root" — if every node has a parent, there is no in-degree-0 node
   to start from.
2. A cycle that is reachable *from* a valid root — for example an admixture node fed
   by one of its own descendants — passes all the degree and root checks but would
   make the depth-first search recurse forever. The search guards against this with
   an "on the current path" marker on each node. If it ever tries to step to a node
   already on the current root-to-here path, that is a back-edge, i.e. a cycle; it
   stops and returns a clean "cycle reachable from root (graph is not a DAG)" error
   instead of overflowing the call stack.

The total number of paths becomes `npath`, and the code counts how many paths reach
each leaf (`pathcount` per leaf) for use in the next step.

---

## 9. The base weight matrix `pwts0` (graph_to_pwts)

`pwts0` is a matrix with one row per normedge and one column per leaf, stored
column-major (`pwts0[edge + nedge_norm * leaf]`). It captures the *theta-independent*
part of how much each drift edge contributes to each leaf's ancestry.

It is filled by walking each root-to-leaf path: the path's leaf gets an increment of
`1 / pathcount[leaf]` added to every normedge that path traverses. Splitting the
weight evenly across a leaf's paths this way is the direct reproduction of ADMIXTOOLS
2's `graph_to_pwts`. Admixedges are skipped here (they have no row); their
contribution is folded in later as a theta-dependent multiplier.

The name "base" matters: `pwts0` is the starting matrix that the theta-dependent step
(section 9's path tables, applied by `fill_pwts`) later *overwrites* in a small
number of cells. The cells `fill_pwts` leaves alone keep their `pwts0` value.

---

## 10. The theta-varying path tables (graph_to_weightind)

These two tables are the inputs to `fill_pwts`, the routine that, given a set of
mixing proportions `theta`, produces the final leaf-weight matrix. They exist so that
applying a new `theta` only has to touch the cells that actually change, rather than
recomputing the whole matrix.

- **Path-admixedge table** (`pae_path` / `pae_admixedge`): for every path, the list of
  admixedges (by 1-based id) that the path passes through. When `fill_pwts` is given a
  `theta`, it computes each path's weight as the product, over that path's admixedges,
  of the corresponding proportion (`theta` for an odd id, `1 - theta` for an even id).

- **Path-edge table** (`pe_edge` / `pe_leaf` / `pe_path`): for each (normedge, leaf)
  cell, which paths flow through it — but **only** for the cells whose value actually
  varies with `theta`. The test for "varies" is: count how many paths pass through the
  cell (`cnt`) and compare to the leaf's total path count. If `cnt` is **less than**
  `pathcount[leaf]`, the leaf reaches this edge on some but not all of its paths, so
  the cell's weight depends on the mixing proportions and is recorded in the table. If
  `cnt` equals `pathcount[leaf]`, every path to that leaf uses the edge, so the weight
  is already correct and constant in `pwts0` and the cell is left out of the table.

At fit time `fill_pwts` therefore only overwrites the recorded cells: for each such
cell it sums the weights of the paths listed for it, and writes that sum into the
matrix, leaving all other cells at their `pwts0` values.

---

## 11. The f3 basis pairs (cmb)

The fit works against a basis of f3 statistics, all sharing the base population from
section 6. After centering on the base and dropping the base's own column, the leaf
weight matrix has `npop - 1` columns — one per non-base leaf, in leaf order. The pair
list `cmb1` / `cmb2` indexes those centered columns.

The pairs enumerated are every `(a, b)` with `a <= b` over the `npop - 1` centered
columns, including the diagonal `a == b`. Counting pairs over `npop - 1` columns *with*
the diagonal gives exactly `choose(npop, 2)` pairs, which is `npair`. The diagonal
case `a == b` corresponds to the statistic f3(base; i, i), which is just f2(base, i).
This reproduces ADMIXTOOLS 2's pair construction (its `combn(0:npop-1, 2)` with the
`+(1:0)` offset), but stored directly as 0-based centered-column indices in the range
`0 .. npop-2`, so each entry is already the exact column index the later math uses —
no further remapping needed. A sanity check confirms the generated pair count equals
`npair`; a mismatch is an internal error.

To recover the actual leaf (and thus its f2-axis index) behind a centered column, the
header's `centered_col_to_leaf` helper walks the leaves skipping the base, so centered
column `c` maps to the `c`-th non-base leaf.

---

## 12. Failure modes

`parse_qpgraph` never throws for a malformed graph; it returns a model whose `error`
field is non-empty and whose `ok()` is false. The caller maps that message to its own
error outcome. The cases that set an error, in the order they are checked:

| Condition | Message (abridged) |
|---|---|
| Empty edge list | `empty edge list` |
| An edge with an empty `from` or `to` | `an edge has an empty endpoint` |
| More than one in-degree-0 node | `multiple roots (graph is not singly-rooted)` |
| A node with in-degree greater than 2 | `node '…' has in-degree > 2 (not a valid admixture node)` |
| No in-degree-0 node at all | `no root (… graph is cyclic)` |
| No leaves | `no leaves` |
| A leaf name not in the f2 population set | `leaf '…' is not in the f2 population set` |
| `f3basepop` given but not a leaf | `f3basepop '…' is not a leaf` |
| A cycle reachable from the root (found during the path search) | `cycle reachable from root (graph is not a DAG)` |
| Generated pair count does not equal `npair` | `internal pair-count mismatch` |

The first eight are user-facing "your graph is malformed" conditions; the cycle case
is the second half of the two-place cycle detection from section 8; and the last is an
internal consistency assertion that should never fire.

# `qpgraph_model.hpp` reference

## 1. Purpose

`src/core/qpadm/qpgraph_model.hpp` defines the data model that turns an
admixture-graph description into a form that can be fit to observed
f-statistics. It holds one struct, `QpGraphModel`, and two functions:
`parse_qpgraph` (which builds the struct from an edge list) and
`fill_pwts_centered` (which evaluates the struct for a given set of mixture
weights).

An admixture graph is a description of population history: a set of populations
(the leaves) descended from a common ancestor (the root) through a branching
tree, with a few extra edges where one population is a mixture of two ancestral
sources. Fitting the graph means finding the edge lengths and the mixture
proportions that best reproduce the measured relationships between the leaf
populations.

The one genuinely new idea in this file is how it handles an *arbitrary*
topology. An earlier prototype had the leaf-weight formulas hard-coded for a
fixed number of admixture events. This file replaces that with a general
recipe: parse the graph once, enumerate every path from the root to each leaf,
and precompute a set of small integer tables. Evaluating the graph for a
particular set of mixture weights is then just walking those tables — no
graph traversal at fit time.

The header is host-only and contains no CUDA code. The GPU fit code uploads the
integer tables to GPU memory once and runs the exact same evaluation there; the
CPU reference implementation uses this struct directly. Both must produce
identical numbers, which is why the evaluation logic is written once, in this
header, as an inline function.

The whole model is a deliberate, line-for-line reproduction of the equivalent
reference routines (`graph_to_pwts`, `graph_to_weightind`, and
`fill_pwts`), so that steppe's graph fit matches for numerical parity[^at2].

---

## 2. The model in plain terms

A few concepts recur throughout the struct. Understanding them first makes the
individual fields obvious.

**Leaves, root, and internal nodes.** The graph is a set of named nodes joined
by directed parent-to-child edges. The one node with no parent is the *root*.
Nodes with no children are *leaves* — these are the actual populations being
fit. Everything in between is an internal (unsampled) ancestral node.

**Drift edges vs. admixture nodes.** Most nodes have exactly one parent; the
edge into such a node is a plain *drift* edge, and its length is one of the
quantities being fit. A node with *two* parents is an *admixture node*: it
represents a population formed by mixing two ancestral sources. Each admixture
node introduces one *mixture weight* (called theta): the fraction of ancestry
coming from the first parent, with the remaining `1 - theta` coming from the
second. The two edges feeding an admixture node are *admixture edges* and are
not counted as fitted drift edges.

**Paths.** Because admixture nodes have two parents, there can be more than one
route from the root down to a given leaf. The model enumerates every such
simple path. A leaf reachable through an admixture node has several paths; a
leaf on a pure tree branch has exactly one.

**The leaf-weight matrix.** The quantity everything is built around is a matrix
that says, for each drift edge and each leaf, how much of that edge's drift the
leaf "inherits." A leaf inherits a share of every edge on its root-to-leaf
path(s), weighted by the mixture proportions along the way. This matrix is what
gets combined with edge lengths to predict the f-statistics.

**Centering and dropping the base.** The graph fit works with f3-statistics,
which are always measured relative to one chosen *base* population. So the
leaf-weight matrix is *centered*: every leaf's column has the base leaf's column
subtracted from it, and the base column itself (now all zeros) is dropped. A
graph with `npop` leaves therefore has `npop - 1` centered columns.

**Column-major storage.** Every matrix in this struct is stored flat in
column-major order, matching the layout the GPU expects. "Column-major" means
the entries of one column sit next to each other in memory; to read row `e`,
column `k` of a matrix with `nedge_norm` rows you index
`array[e + nedge_norm * k]`. All array indices in the struct are 0-based.

---

## 3. `QpGraphModel`: dimensions and identity fields

`QpGraphModel` is the parsed, fit-ready topology. Its scalar fields are the
dimensions every array is sized against, plus the fields that identify the
leaves.

| Field | Type | Meaning |
|---|---|---|
| `npop` | `int` | The number of leaf populations. Equals the f3 population count. |
| `nedge_total` | `int` | Every edge in the input list, drift edges and admixture edges together (the row count of the input). |
| `nedge_norm` | `int` | The number of non-admixture (drift) edges — the edges whose lengths are actually fit. Equals `nedge_total - 2 * nadmix`. This is the row count of the leaf-weight matrices. |
| `nadmix` | `int` | The number of admixture nodes, which is the number of mixture weights (the dimension of theta). |
| `npair` | `int` | `choose(npop, 2)` — the number of f3 basis pairs. |
| `npath` | `int` | The total number of distinct root-to-leaf simple paths across all leaves. |
| `base_leaf` | `int` | The 0-based index (in leaf order) of the base population for the f3-statistics. Defaults to leaf 0, the first leaf. |

Two parallel arrays identify the leaves:

| Field | Type | Meaning |
|---|---|---|
| `leaves` | `vector<string>` | The leaf labels, in graph leaf order. `leaves[base_leaf]` is the base population. This order matches the columns of the leaf-weight matrix and the rows the f3 basis indexes. |
| `leaf_to_f2` | `vector<int>` | For each leaf, its index into the f2 population axis. `leaf_to_f2[i]` is the f2 population index of leaf `i`. The f3 basis triples use these to look up the right f2 entries. |

Leaf order is "first-seen" order: leaves come out in the order their names first
appear while scanning the input edge list. This matches the parity leaf
ordering[^at2].

---

## 4. The base leaf-weight matrix: `pwts0`

`pwts0` is the *base* leaf-weight matrix — the part that does **not** depend on
the mixture weights.

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `pwts0` | `vector<double>` | `nedge_norm × npop`, column-major | For each drift edge and leaf, the base drift incidence: how much of that edge the leaf inherits before mixture weights are applied. Indexed `pwts0[e + nedge_norm * leaf]`. |

The value in cell `(edge, leaf)` is built by walking every root-to-leaf path for
that leaf and adding `1 / (number of paths to that leaf)` each time the path
crosses that drift edge. A leaf with a single path gets a weight of `1` on every
edge along it. A leaf reachable by two paths gets `1/2` on the edges each path
crosses, so its total still sums to a full share. Admixture edges are not rows
of this matrix and contribute nothing to it.

`pwts0` is only the starting point. For any cell whose weight actually depends
on the mixture proportions, the value stored here is a placeholder that the
evaluation step overwrites. Exactly which cells those are is described by the
path tables in the next section.

---

## 5. The `fill_pwts` path tables

These four integer arrays are the precomputed instructions for re-deriving the
mixture-weight-dependent cells of the leaf-weight matrix from a given theta.
They are the tables uploaded to the GPU once and reused for every evaluation.
They come in two parallel-array groups.

**Path-to-edge table** — which matrix cells each path contributes to. All three
arrays have the same length (call it `n_pe`); entry `t` describes one
contribution.

| Field | Type | Meaning |
|---|---|---|
| `pe_edge` | `vector<int>` | The drift-edge row of the cell, in `[0, nedge_norm)`. |
| `pe_leaf` | `vector<int>` | The leaf column of the cell, in `[0, npop)`. |
| `pe_path` | `vector<int>` | The path (in `[0, npath)`) that contributes to this cell. |

Only the cells that *vary* with the mixture weights appear in this table. A cell
that every one of its leaf's paths passes through is fully determined and
already correct in `pwts0`, so it is left out. A cell that only some of the
leaf's paths reach is theta-dependent and is listed here, once per contributing
path.

**Path-to-admixture-edge table** — which mixture factors each path carries. Both
arrays share a length (call it `n_pae`); entry `t` says path `pae_path[t]`
passes through admixture edge `pae_admixedge[t]`.

| Field | Type | Meaning |
|---|---|---|
| `pae_path` | `vector<int>` | The path, in `[0, npath)`. |
| `pae_admixedge` | `vector<int>` | The admixture edge it traverses, as a **1-based** id in `[1, 2 * nadmix]`. |

The admixture-edge id encodes which mixture factor to multiply in. For admixture
node `j` (0-based), id `2j + 1` is the first parent and means multiply by
`theta[j]`; id `2j + 2` is the second parent and means multiply by
`1 - theta[j]`. Equivalently: an **odd** id contributes `theta`, an **even** id
contributes `1 - theta`, and the node index is `(id - 1) / 2`.

Which of an admixture node's two parents is treated as "the theta parent" is an
arbitrary but fixed choice: the parent whose feeding edge appears first in the
input edge list. The graph's fit score is unaffected by this choice (swapping
the two parents just replaces theta with `1 - theta`), and the fitted weight is
reported keyed by parent name, so the result matches for parity regardless.

Together these tables let the evaluation compute each path's total mixture
factor (the product of the theta/`1 - theta` values along it) and then sum those
factors into the matrix cells the path touches.

---

## 6. The f3 basis pairs: `cmb`

`cmb1` and `cmb2` list the pairs of leaf columns that form the f3-statistic
basis. They are two parallel arrays of length `npair`; pair `k` is
`(cmb1[k], cmb2[k])`.

| Field | Type | Meaning |
|---|---|---|
| `cmb1` | `vector<int>` | First column of the pair. |
| `cmb2` | `vector<int>` | Second column of the pair. |

The indices are 0-based positions among the **centered columns** — the
`npop - 1` non-base leaves in leaf order — so each is in `[0, npop - 1)` and the
pairs satisfy `0 <= cmb1[k] <= cmb2[k] < npop - 1`. The diagonal case
`cmb1[k] == cmb2[k]` corresponds to the f3 entry `f3(base; i, i)`, which is just
`f2(base, i)`.

There are exactly `choose(npop, 2)` such pairs: `choose(npop - 1, 2)` off-diagonal
pairs plus the `npop - 1` diagonal ones sum to `choose(npop, 2)`, which is why
`cmb1.size()` equals `npair`. To turn a centered-column index back into a real
leaf (and from there into an f2 index), use `centered_col_to_leaf` (section 8).

---

## 7. Labels, the `error` field, and validation invariants

**Edge and admixture labels** are carried through purely so the fit result can
echo human-readable names.

| Field | Type | Meaning |
|---|---|---|
| `edge_from`, `edge_to` | `vector<string>` | The parent and child labels of each drift edge, in `pwts0` row order — one entry per drift edge. |
| `admix_from`, `admix_to` | `vector<string>` | For each admixture node (each theta), the labels of the first parent and of the admixture node itself. The fitted weight `weight[j]` is the mass on the `admix_from[j]` → `admix_to[j]` edge; the second parent carries `1 - weight[j]`. |

**The `error` field** is how the parse reports failure without throwing.

| Field / method | Type | Meaning |
|---|---|---|
| `error` | `string` | Empty on success; a human-readable message on failure. |
| `ok()` | `bool` | Convenience: true exactly when `error` is empty. |

A non-empty `error` means the input was not a well-formed rooted admixture
graph, and the caller maps the message to a status outcome. The conditions the
parse rejects are: an empty edge list, an edge with an empty endpoint, more than
one root, a node with in-degree greater than two (not a valid admixture node),
no root at all, no leaves, a leaf whose name is not in the f2 population set, a
requested base population that is not a leaf, and an internal consistency check
on the pair count.

Two of these deserve note because they both mean "the graph is not a valid
acyclic graph," but they are caught in different ways:

- **A graph with no root.** If every node has at least one parent, there is no
  in-degree-zero node to serve as the root. This is detected up front from the
  degree counts.
- **A cycle reachable from a valid root.** A graph can have a proper root and
  still contain a cycle further down (for example an admixture node fed by one
  of its own descendants). This passes the degree checks but would make the
  path enumeration recurse forever, so the path search carries a guard that
  marks the nodes on the current path and treats revisiting one as a cycle,
  returning a clean error instead of overflowing.

---

## 8. `centered_col_to_leaf`

```cpp
int centered_col_to_leaf(int c) const;
```

Maps a centered-column index `c` (in `0 .. npop - 2`) back to the full leaf
index it stands for — the `c`-th non-base leaf in leaf order. It walks the
leaves in order, skips `base_leaf`, and returns the leaf reached after `c`
non-base leaves. Returns `-1` if `c` is out of range.

This is the inverse of the base-dropping convention: the centered leaf-weight
matrix and the `cmb` pairs speak in centered-column indices, and this helper is
how the fit turns one of those indices into a real leaf so it can look up the
matching f2 index via `leaf_to_f2`.

---

## 9. `parse_qpgraph`

```cpp
QpGraphModel parse_qpgraph(const std::vector<QpGraphEdge>& edges,
                           const std::vector<std::string>& leaf_names,
                           const std::string& f3basepop = "");
```

The entry point that builds a `QpGraphModel` from a raw edge list.

- `edges` is the admixture graph as a list of parent-to-child rows. A node
  appearing as a child in two rows is an admixture node.
- `leaf_names` maps every f2 population-axis index to its population name
  (`leaf_names[i]` is the name of population `i`); every graph leaf must be one
  of these names.
- `f3basepop` selects the base population for the f3-statistics. An empty string
  (the default) selects the first leaf in graph leaf order.

It never throws. On any structural problem it returns a model whose `error`
field is set (and whose `ok()` is false); the caller is expected to check that
before using the model. On success it returns the fully populated, fit-ready
struct described in the sections above: the dimensions, the leaf identity
arrays, `pwts0`, the path tables, the `cmb` basis, and the echo labels.

---

## 10. `fill_pwts_centered`

```cpp
inline void fill_pwts_centered(const QpGraphModel& m, const double* theta,
                               std::vector<double>& pwts_c);
```

The host reference that evaluates the model for one set of mixture weights. Given
`theta` (an array of length `nadmix`, the mixture weight per admixture node), it
produces `pwts_c`, the centered, base-dropped leaf-weight matrix of shape
`nedge_norm × (npop - 1)`, column-major (`pwts_c[e + nedge_norm * j]`, with `j`
running over the `npop - 1` non-base leaves in leaf order). This is the matrix
the fit multiplies against edge lengths to predict f-statistics.

It reproduces the reference sequence[^at2]: fill the mixture-weight-dependent
cells of the base matrix, then center and drop the base column, i.e.
`pwts = fill_pwts(pwts0, theta)` followed by `pwts = pwts[, -base] - pwts[, base]`.
It works in three steps:

1. **Per-path mixture factor.** Start every path's factor at `1`, then walk the
   path-to-admixture-edge table and multiply in `theta[j]` (odd admixture-edge
   id) or `1 - theta[j]` (even id) for each admixture edge the path traverses.
2. **Overwrite the varying cells.** Copy `pwts0`, then for each distinct
   `(edge, leaf)` cell listed in the path-to-edge table, sum the mixture factors
   of the paths that reach it and store that sum in place of the placeholder.
3. **Center and drop the base.** Subtract the base leaf's column from every other
   leaf's column and pack the results into the `npop - 1` output columns.

This function is defined **inline in the header** on purpose. The GPU backend
(in the `steppe::device` layer) needs it for the final single evaluation that
recovers edge lengths once the fit has converged, but the device layer must not
link against the core library's symbols — many device-only tests link the device
layer without core. Defining the evaluation inline lets any translation unit
that includes this header use it directly, with no core link dependency, while
keeping the CPU reference and the GPU path running literally the same code.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

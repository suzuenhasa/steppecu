# `qpgraph_arena.hpp` reference

## 1. Purpose

`src/core/qpadm/qpgraph_arena.hpp` holds one small helper,
`fill_topo_arena_common()`, whose whole job is to copy the fields of a qpGraph
model that *don't change with the pair layout* out of a `QpGraphModel` and into
the device-side `QpGraphTopoArena` that the GPU fit reads.

A bit of background on what a "qpGraph" is. An **admixture graph** is a picture of
population history: it's a tree of populations where, on top of the ordinary
branching, some populations are drawn as a **mix** of two ancestral streams. qpGraph
takes such a graph shape (which populations, which edges, which mix nodes) plus the
measured **f-statistics** — summaries of how allele frequencies differ between
populations — and finds the edge lengths and mix proportions that best explain the
data. To evaluate a candidate graph on the GPU, steppe first flattens the graph's
fixed structure into a plain bundle of arrays: that bundle is the
`QpGraphTopoArena`. "Topo" is short for topology — the *shape* of the graph, the
part that stays put while the fitter tries different numbers.

That arena is built in **two different places**:

- the **fit** path (`qpgraph_fit.cpp`), which scores one specific graph, and
- the **search** path (`qpgraph_search.cpp`), which explores many graph shapes
  looking for a good one.

Most of the arena is identical in both cases — it's just a straight copy out of the
model. The reason that copy lives here, in one shared function, is the same reason
the f2 formulas live in one place: **if each caller re-typed the same dozen
assignments, they could quietly drift apart.** One path might add a field the other
forgot, and the two qpGraph code paths would then be feeding the GPU subtly
different topology data without anyone noticing. Homing the shared copy here means
there is exactly one list of "these fields come straight from the model," and both
callers are guaranteed to agree on it.

The one thing this helper deliberately does **not** touch is the pair layout —
`npair`, `cmb1`, `cmb2`. Those are left for each caller to fill in, because the two
paths genuinely disagree about them (see section 2).

---

## 2. `fill_topo_arena_common()`

```
fill_topo_arena_common(QpGraphTopoArena& out,
                       const QpGraphModel& m,
                       const QpGraphOptions& opts) -> void
```

It copies thirteen fields — eleven straight off the model, two off the options —
into `out`. Nothing is computed; it is pure assignment. The eleven model fields are:

| Field | What it is |
|---|---|
| `npop` | Number of populations (leaves of the graph). |
| `nedge_norm` | Number of normal (non-admixture) edges — the free branch lengths being fit. |
| `nadmix` | Number of admixture (mix) nodes in the graph. |
| `npath` | Number of root-to-leaf paths through the graph. |
| `base_leaf` | Index of the leaf used as the reference/base population that all others are measured against. |
| `pwts0` | The base leaf-weight matrix — how much each edge contributes to each leaf before mix weights are applied. |
| `pe_edge`, `pe_leaf`, `pe_path` | The path tables that say, for each path element, which edge / leaf / path it belongs to. |
| `pae_path`, `pae_admixedge` | The tables that connect admixture edges to the paths they sit on. |

Those are exactly the arrays a GPU kernel needs to walk the graph and turn a
candidate set of edge lengths and mix proportions into predicted f-statistics. They
are functions of the graph's *shape*, so they're the same no matter which pairs of
populations the fit happens to be scoring.

The two options fields are the fitter's knobs, not topology, but they ride along in
the same arena because the kernel needs them too:

- `constrained` — whether mix proportions are held to the valid range (they must sum
  sensibly and stay between 0 and 1) rather than allowed to roam free.
- `fudge` — a tiny regularization term (default `1e-4`) added to stabilize the fit's
  linear algebra so a near-singular system doesn't blow up.

### Why the pair fields are left to the caller

`npair`, `cmb1`, and `cmb2` describe the set of population **pairs** whose f2 values
feed the fit, and how each pair maps onto the model's centered columns. This helper
skips them on purpose, because the two callers build them differently:

- The **fit / fleet** path just copies the model's own pairs verbatim
  (`a.npair = m.npair; a.cmb1 = m.cmb1; a.cmb2 = m.cmb2`), because it is scoring
  that one model against its own basis.
- The **search** path instead *re-derives* `cmb1`/`cmb2` so the pairs line up with a
  **shared f2 basis** that many candidate graphs are scored against — it looks each
  pair's two populations up in the basis's column ordering and stores the remapped
  indices.

Because those two layouts are legitimately different, folding them into the shared
copy would be wrong; each caller sets its own right after calling this helper. So
the contract is simple: `fill_topo_arena_common()` fills the thirteen
topology-invariant fields, and **the caller is responsible for the three pair
fields.** The field-by-field breakdown of the arena itself lives with the qpGraph
fit and search references (the source header points at `qpgraph_fit §5` /
`qpgraph_search §5`), and the model's own fields are documented in the
`qpgraph_model.hpp` reference.

This helper is a plain host-side `inline` function — no GPU code, no allocation of
its own; it just populates a struct that later gets shipped to the device.

---

The qpGraph machinery reproduces the admixture-graph fitting of ADMIXTOOLS 2 for
numerical parity[^at2].

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

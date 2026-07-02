# `qpgraph.hpp` reference

## 1. Purpose

`include/steppe/qpgraph.hpp` is the public interface for fitting a single
admixture graph to observed genetic statistics — the operation ADMIXTOOLS 2
calls `qpgraph`. It defines the value types a caller passes in and gets back,
plus the two entry-point functions that run the fit.

The problem it solves: an admixture graph is a family tree of populations in
which some populations are mixtures of two ancestral sources. Given such a graph
whose *shape* is already decided, the fit finds the numbers that make the graph
best reproduce the population statistics measured from real data — the mixing
proportions at each mixture point and the amount of genetic drift along each
branch. It reports how good that best fit is and where it fits worst.

A few properties of this file are worth stating up front:

- **The graph shape is an input, not something this file searches for.** The
  caller supplies a fixed topology (which populations descend from which, and
  where the mixtures are). Searching over possible shapes is a separate,
  deferred capability. This file only fits the one shape it is handed.
- **It reads pairwise population statistics, not raw genotypes.** The fit
  consumes a precomputed "f2" object (the pairwise-distance statistics between
  populations), either already resident in GPU memory or supplied as host
  memory. It does not read genotype files itself.
- **The whole fit runs on the GPU.** The statistics the fit needs are assembled
  once and kept in GPU memory, and the entire search for the best numbers runs
  there — including every trial evaluation inside the optimizer's loop. The host
  launches the work once per graph and receives only the final answer. It never
  runs a per-iteration evaluation on the CPU.
- **The header is deliberately free of any CUDA code.** It uses only the C++
  standard library and forward-declares the two GPU-side types it references.
  That keeps it lightweight enough to include in the core library, the
  command-line tool, and the language bindings without forcing any of them to
  pull in the GPU toolkit.

---

## 2. QpGraphEdge

`QpGraphEdge` is one directed branch of the admixture graph, written as a
parent-to-child pair. The graph as a whole is just a list (`vector`) of these
edges.

| Field | Type | Meaning |
|---|---|---|
| `from` | `string` | The parent node's label. |
| `to` | `string` | The child node's label. |

The labels follow a simple rule:

- **Leaves** — nodes that are never a parent (they never appear in any `from`) —
  must be named exactly as the populations in the f2 statistics. These are the
  observed populations the graph is being fit to.
- **Internal nodes** are free-form labels; they name unobserved ancestral
  populations and can be called anything.
- **An admixture node** is any node that appears as the `to` of *two* different
  edges — that is, a node with two parents. Its two parents are the sources it
  is a mixture of, and the fit assigns it a single mixing weight that decides how
  much of it comes from each parent.

---

## 3. QpGraphOptions

`QpGraphOptions` holds the per-call settings. Its defaults are chosen to match
ADMIXTOOLS 2's `qpgraph()` defaults exactly, so a steppe fit reproduces an
ADMIXTOOLS 2 fit on the same graph and data. Each parity value is a named field
rather than a bare number typed into the code, and the reference results steppe
validates against were generated with exactly these defaults.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `fudge` | `double` | `1e-4` | A tiny ridge term added to the branch-length solve to keep it numerically stable. It nudges the diagonal of that solve by this fraction of the average diagonal value before solving, which prevents a near-singular system from producing wild branch lengths. Matches ADMIXTOOLS 2's `diag` option. |
| `diag_f3` | `double` | `1e-5` | A small regularization added to the statistics' covariance before it is inverted, so the inversion stays well-behaved. It bumps the covariance diagonal by this fraction of its total diagonal. Matches ADMIXTOOLS 2's `diag_f3` option. |
| `numstart` | `int` | `10` | How many independent restarts the optimizer runs from different starting points. Because the fit runs many restarts in parallel on the GPU (one per GPU thread), and because the restarts agreeing on the same answer is the evidence that the answer is trustworthy, this is a meaningful knob. Matches ADMIXTOOLS 2's `numstart`, and `10` is the value the reference results were generated with. |
| `constrained` | `bool` | `true` | Whether drift branch lengths are held at or above zero (a branch can't have negative length). `true` is both the ADMIXTOOLS 2 default and the mode the reference results use. Setting it `false` allows the unconstrained solve, which can return negative branch lengths. |
| `maxit` | `int` | `200` | The most iterations any single restart of the optimizer will run before stopping. The objective being minimized is cheap and well-behaved and converges in far fewer iterations than this, so the cap is deliberately generous. |
| `tol` | `double` | `1e-9` | The convergence tolerance. A restart stops once its step size and its improvement in the fit score both fall below this. |

---

## 4. QpGraphResult

`QpGraphResult` is everything the fit returns. It reports the fitted numbers,
several parallel label arrays that say what each number refers to, a handful of
quality diagnostics, and an outcome code.

The result never signals a problem by throwing an exception. Ordinary "this data
or graph can't be fit" outcomes — a covariance that can't be inverted, a graph
that can't be parsed, a degenerate graph — are reported through the `status`
field instead.

### The fitted mixing weights

These four arrays are all the same length, one entry per admixture node, in the
order the admixture nodes appear in the input edge list. For a graph with no
admixture at all (a pure tree) they are empty.

| Field | Type | Meaning |
|---|---|---|
| `weight` | `vector<double>` | The fitted mixing weight at each admixture node — the share of that node's ancestry coming from its first parent. |
| `weight_lo` | `vector<double>` | The smallest value each weight took across all the restarts. |
| `weight_hi` | `vector<double>` | The largest value each weight took across all the restarts. |
| `admix_from` | `vector<string>` | The first parent of each admixture node. `weight[j]` is the share of the mass on the branch from `admix_from[j]`. |
| `admix_to` | `vector<string>` | The admixture node itself for each weight. |

`weight_lo` and `weight_hi` together form a bracket per weight: a *tight* bracket
means every restart converged to the same value, which is the evidence that the
weight is well determined by the data. A wide bracket warns that the weight is
not uniquely pinned down. The second parent of an admixture node (not named in
these arrays) carries the remaining share, `1 - weight[j]`.

### The fitted branch lengths

| Field | Type | Meaning |
|---|---|---|
| `edge_length` | `vector<double>` | The fitted genetic-drift length of each ordinary (non-admixture) branch. |
| `edge_from` | `vector<string>` | The parent of each branch, in the same order as `edge_length`. |
| `edge_to` | `vector<string>` | The child of each branch, in the same order. |

These three arrays run in the input edge order, with the admixture edges removed.

### Fit quality and diagnostics

| Field | Type | Default | Meaning |
|---|---|---|---|
| `score` | `double` | `0.0` | The overall goodness-of-fit score at the best solution — the weighted residual between what the graph predicts and what the data shows. Lower is better. This is the same number ADMIXTOOLS 2 reports as the fit score. |
| `restart_spread` | `double` | `0.0` | How much the score varied across the restarts (the largest minus the smallest). A near-zero spread means every restart found the same optimum, which is the sign the fit converged. |
| `worst_residual_z` | `double` | `0.0` | The single worst-fitting statistic, expressed as a signed z-score (how many standard errors the prediction is off by). The sign is kept; the magnitude is what matters as a diagnostic — a large magnitude flags a part of the graph the data disagrees with. |
| `worst_pop2` | `string` | empty | One of the two populations naming the worst-fitting statistic. |
| `worst_pop3` | `string` | empty | The other population naming the worst-fitting statistic. (The third, shared, reference population is the first leaf — see `leaves` below.) |
| `leaves` | `vector<string>` | empty | The graph's leaf populations, in the order the fit used them. The first leaf serves as the shared base population for all the statistics. |

### Outcome and precision tags

| Field | Type | Default | Meaning |
|---|---|---|---|
| `status` | `Status` | `Status::Ok` | The outcome of this call: success, or a specific failure such as a covariance that can't be inverted, or a graph that can't be parsed or is degenerate. This is how the fit reports problems — it does not throw. |
| `precision_tag` | `Precision::Kind` | `Fp64` | Which arithmetic mode actually produced this result, recorded so the caller knows the numerical grade of the answer. |

---

## 5. Entry points

Both entry points fit one graph and return one `QpGraphResult`. They differ only
in where the input statistics live. Both are marked so that ignoring the returned
result is a compile-time warning.

### The primary, GPU-resident entry

```
QpGraphResult run_qpgraph(const device::DeviceF2Blocks& f2,
                          const std::vector<QpGraphEdge>& edges,
                          const std::vector<std::string>& leaf_names,
                          const QpGraphOptions& opts,
                          device::Resources& resources);
```

This is the fast path. It takes the pairwise statistics already sitting in GPU
memory, assembles the quantities the fit needs there, and runs the whole
optimization on the GPU using the first GPU in `resources`.

- `f2` — the pairwise population statistics, already resident on the GPU.
- `edges` — the admixture graph as a list of parent-to-child branches.
- `leaf_names` — the roster of population names in the order the statistics use
  them. Entry *i* is the name of population index *i* in the statistics, so this
  is what connects the leaf labels in `edges` to the rows and columns of `f2`.
- `opts` — the per-call settings (see section 3).
- `resources` — the GPU handles the fit runs on.

### The parity / host-memory overload

```
QpGraphResult run_qpgraph(const F2BlockTensor& f2_host,
                          const std::vector<QpGraphEdge>& edges,
                          const std::vector<std::string>& leaf_names,
                          const QpGraphOptions& opts,
                          device::Resources& resources);
```

This overload takes the same statistics as ordinary host memory (an
`F2BlockTensor`) instead of a GPU-resident object. It exists so the CPU
reference implementation, which reads host memory directly, can run the same fit
and serve as the correctness oracle the GPU path is checked against. Everything
else is identical, and `leaf_names` again gives the population order of the
host tensor.

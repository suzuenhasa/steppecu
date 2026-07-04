# `qpgraph_search.hpp` reference

## 1. Purpose

`include/steppe/qpgraph_search.hpp` is the public, CUDA-free interface to the
qpGraph *topology search*: given a fixed set of populations, score **every**
admixture-graph shape you can build on them and return the single best-fitting
one.

A little background. An admixture graph is a family tree of populations that also
allows for mixture events (a population formed by blending two ancestral
branches). qpGraph fits one such graph to the data and reports how well it
explains the observed relationships between populations. Normally you have to
guess a graph shape by hand. This search removes the guessing: it enumerates all
possible shapes within a bounded space and fits all of them at once.

The header defines three result-carrying structs, one options struct, and two
entry-point functions. It contains no GPU code — it only uses the C++ standard
library and other steppe headers — so it can be included by the core library, the
command-line tool, and the language bindings without pulling in CUDA.

### Why exhaustive enumeration instead of copying the reference search

The reference finds graphs with a *stochastic* search[^at2]: it starts from random
seeds and hill-climbs, and the best graph it lands on varies from run to run
(each seed tends to find a different over-fitted graph, and it rarely recovers a
known hand-curated answer). Because there is no single deterministic result to
match, there is no way to build a "reproduce the reference search exactly"
test against it.

This search sidesteps that problem. In a *bounded, enumerable* space — a fixed
list of populations and a small cap on the number of mixture events — the set of
possible graph shapes is finite, so the global best can be computed exhaustively
and is fully deterministic. Two facts make the reference still usable as an
oracle:

- **Each candidate is scored** by the ordinary single-graph fit, which *is*
  validated against the reference `qpgraph()` to a tight relative tolerance
  (about 1e-6)[^at2]. So every individual score is trustworthy.
- **The enumeration itself is checked** against the reference
  `generate_all_trees` / `generate_all_graphs`: the number of shapes produced and
  the set of canonical graph identities match one-for-one. So the coverage is
  provably complete.

### How the GPU work is shaped

The parts of the computation that depend only on the population set — the f3
statistics basis and the inverse covariance matrix `Qinv` — are assembled once
and kept resident in GPU memory. All the different graph shapes are then packed
into a single device buffer (a "fleet" of heterogeneous topologies, each with its
own lookup index), and the launch is flattened over every (shape, restart) pair
so that **all candidates are fit in one GPU launch**. The host only does the
cheap parts: enumerating the shapes and taking the final `argmin` over the scores
(a reduction, not a fit). There is never a per-candidate fit on the host.

---

## 2. QpGraphSearchOptions

The per-call configuration for one search. The per-candidate fitting tier reuses
the ordinary single-graph fit options (`QpGraphOptions`), so this struct only
adds the fields that describe the search *space* and the optional heuristic
check.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `pops` | `vector<string>` | empty | The set of leaf populations to search over. The search covers every rooted graph shape whose leaves are exactly these populations. Each name must be a population that has f2 statistics. An empty list is rejected with an `InvalidConfig` outcome. |
| `max_nadmix` | `int` | `1` | The largest number of admixture (mixture) nodes a candidate may have — the upper bound on the search's other axis. Version 1 supports `0` and `1`: pure trees, and graphs with a single mixture event. |
| `fit` | `QpGraphOptions` | golden defaults | The single-graph fit options applied to each candidate (things like the fudge factor, whether f3 is diagonalized, the number of restarts, and the fit tolerance). The defaults are the same ones used by the validated single-graph fit, so a searched candidate is scored identically to a standalone qpGraph fit of that same shape. |
| `run_heuristic` | `bool` | `true` | Also run the mutation / hill-climb heuristic and verify that it lands on the same global best the exhaustive enumeration found. This is a self-check on the heuristic, not a separate search. Set to `false` for exhaustive-only. |
| `heuristic_seeds` | `int` | `8` | How many deterministic hill-climb starts the heuristic runs from. The recovery check requires that *all* of these starting points converge to the exhaustive global best, not just one. |

---

## 3. QpGraphCandidate

One scored graph shape from the enumerated space — the best fit found for that one
topology across all its restarts.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `nadmix` | `int` | `0` | The number of admixture nodes in this shape. `0` means it is a plain tree. |
| `id` | `int` | `0` | A stable index for this candidate within its `nadmix` level, following the deterministic enumeration order. |
| `hash` | `uint64_t` | `0` | The canonical graph hash — a fingerprint that is identical for any two graphs that are the same shape up to relabeling (isomorphic). This is the key used to look a candidate up and to check that the heuristic recovered the right shape. |
| `score` | `double` | `0.0` | The fit score for this shape: the minimum generalized-least-squares (GLS) fit residual over all restarts. Lower is a better fit. |
| `restart_spread` | `double` | `0.0` | The gap between the best and worst restart scores for this one shape — a measure of how *identifiable* the shape is. A spread near zero means the restarts all found the same minimum, so the fit is sharp and reproducible (a well-identified graph). A large spread means the fit landscape is flat or ill-conditioned, so different restarts disagree — the mark of a poorly-identified graph that a search should reject. |
| `edges` | `vector<QpGraphEdge>` | empty | The edge list describing this shape, in a form ready to be parsed back into a qpGraph. |

---

## 4. QpGraphSearchResult

The full search result. It bundles three logically separate deliverables — proof
of exhaustive coverage, the winning graph, and the complete scored list — plus the
winning graph's detailed fit, the heuristic self-check, and timing.

### Exhaustive-coverage counts

These are the witnesses that the enumeration was complete; they are the numbers
compared against the reference enumeration.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `n_trees` | `int` | `0` | How many pure-tree shapes (`nadmix = 0`) were enumerated. |
| `n_admix1` | `int` | `0` | How many single-admixture (`nadmix = 1`) shapes were enumerated, counting only one representative per isomorphism class. |
| `n_candidates` | `int` | `0` | The total number of shapes actually scored. |

### The global best

| Field | Type | Default | Meaning |
|---|---|---|---|
| `best` | `QpGraphCandidate` | — | The deterministic winner: the candidate with the smallest score across the whole enumeration. |
| `second_best_score` | `double` | `0.0` | The runner-up's score. The gap between this and the best score is the witness for how clearly the winner stands out — an identifiability margin. |

### The full scored list

| Field | Type | Default | Meaning |
|---|---|---|---|
| `candidates` | `vector<QpGraphCandidate>` | empty | Every shape that was successfully fit, each with its canonical hash, edge list, and best-of-restarts score, in the deterministic enumeration order (all trees first, then the single-admixture shapes). This is exactly the per-shape data the global-best reduction ran over. Exposing it is purely additive — it does not change any fit, score, or enumeration math — and it lets a parity test look up a candidate by its canonical hash or edges and compare its score against the reference `qpgraph()` for the same shape. A candidate whose graph failed to build (a bad parse) is omitted, because it has no score. |

### The winner's detailed fit

| Field | Type | Default | Meaning |
|---|---|---|---|
| `best_fit` | `QpGraphResult` | — | The complete single-graph fit of the winning shape — the full-detail result the parity gate diffs against a standalone reference `qpgraph()` of the same edge list. |

### The heuristic self-check

| Field | Type | Default | Meaning |
|---|---|---|---|
| `heuristic_recovered` | `bool` | `false` | Whether the hill-climb heuristic found the exhaustive global best — the same canonical hash and a score within the fit tolerance — starting from *every* seed. |
| `heuristic_seed_hashes` | `vector<uint64_t>` | empty | The canonical hash of the local minimum each seed converged to. An empty vector means the heuristic did not run (`run_heuristic` was off). |

### Timing

| Field | Type | Default | Meaning |
|---|---|---|---|
| `fit_all_wall_ms` | `double` | `0.0` | Wall-clock time, in milliseconds, for the combined enumerate-and-fit-everything step. |
| `topologies_per_s` | `double` | `0.0` | Shapes scored per second, derived from the wall time. Because the single fleet launch is where essentially all the work happens (the host only enumerates and takes the `argmin`), this is the throughput witness that the search is GPU-bound. |

### Status and precision

| Field | Type | Default | Meaning |
|---|---|---|---|
| `status` | `Status` | `Ok` | The outcome of the call (for example `InvalidConfig` when `pops` is empty). |
| `precision_tag` | `Precision::Kind` | `Fp64` | Which arithmetic mode produced the result, recorded so a consumer knows the numerical tier the scores came from. |

---

## 5. Entry points

Both functions run the same search and return a `QpGraphSearchResult`; they differ
only in where the f2 statistics live. Both are marked `[[nodiscard]]`, so the
returned result must be used. In both, `leaf_names` maps the f2 population axis to
names — `leaf_names[i]` is the name of f2 population `i` — which is how the
population names in `QpGraphSearchOptions::pops` are resolved to the underlying
data.

### GPU-first primary

```cpp
QpGraphSearchResult run_qpgraph_search(const device::DeviceF2Blocks& f2,
                                       const std::vector<std::string>& leaf_names,
                                       const QpGraphSearchOptions& opts,
                                       device::Resources& resources);
```

Reads f2 statistics that are already resident in GPU memory. This is the
production path: the f3 basis is assembled device-resident once, and the whole
fleet of shapes is fit in a single launch.

### Host-oracle / parity overload

```cpp
QpGraphSearchResult run_qpgraph_search(const F2BlockTensor& f2_host,
                                       const std::vector<std::string>& leaf_names,
                                       const QpGraphSearchOptions& opts,
                                       device::Resources& resources);
```

Reads f2 statistics from host memory. This overload exists for the CPU reference
backend used to validate the GPU path against the reference[^at2]; it is not the runtime
users go through.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

# qpgraph and qpgraph-search

Fit admixture graphs to f2 statistics: `qpgraph` fits one graph whose shape you supply, and `qpgraph-search` enumerates every graph shape over a small pop set and returns the best.

An admixture graph is a family tree of populations that also allows mixture events — a population formed by blending two ancestral branches. Both commands read an f2-blocks directory (built by [`extract-f2`](./extract-f2.md)) and run the whole fit on the GPU. steppe reproduces ADMIXTOOLS 2's `qpgraph()` numerically, so the fitted scores, weights, and branch lengths match the reference on the same graph and data.

---

## qpgraph

Fit a single admixture-graph topology you provide and report how well it reproduces the data.

### What it does

You hand `qpgraph` a fixed graph *shape* — an edge list of parent-child branches — and it finds the numbers that make that shape best reproduce the population statistics: the mixing proportion at each mixture point and the amount of genetic drift along each branch. It reports an overall goodness-of-fit score (lower is better), the fitted weights and edge lengths, and the single worst-fitting statistic as a z-score so you can see where the data disagrees with your graph.

Reach for this when you already have a hypothesis about how a set of populations are related — for example a graph you drew by hand or one that came out of `qpgraph-search` — and you want to fit it and judge it. The graph shape is an input here; `qpgraph` does not search for it.

The edge-list file has one `parent child` pair per line. **Leaf** nodes (nodes that are never a parent) must be named exactly as populations in the f2 statistics. **Internal** nodes are free-form labels for unobserved ancestors. An **admixture** node is any node that appears as the child of two different edges — its two parents are the sources it mixes, and the fit assigns it one mixing weight.

### Flags

| flag | what it does | default |
|---|---|---|
| `--f2-dir` | The f2-blocks directory to read (`f2.bin` + `pops.txt`), as built by `extract-f2`. Required. | — |
| `--graph` | The admixture-graph edge-list file: one `parent child` pair per line. Required. | — |
| `--numstart` | How many independent optimizer restarts to run from different starting points. The restarts run in parallel on the GPU; them agreeing is the evidence the answer is trustworthy. | `10` |
| `--diag-f3` | Regularization added to the f3 covariance before it is inverted, as a fraction of its diagonal, so the inversion stays well-behaved. Matches the reference `diag_f3`. | `1e-5` |
| `--constrained` / `--no-constrained` | Hold drift branch lengths at or above zero (a branch can't have negative length). `--no-constrained` allows the unconstrained solve, which can return negative lengths. | on (`--constrained`) |
| `--fudge` | A tiny ridge added to the branch-length solve (as a fraction of its average diagonal) to keep a near-singular system from producing wild branch lengths. Matches the reference `diag`. | `1e-4` |
| `--out` | Write output to this file instead of stdout. | stdout |
| `--format` | Output format: `csv`, `tsv`, or `json`. | `csv` |
| `--device` | CUDA device(s): `auto`, an ordinal like `0`, or two ordinals. GPU-only — there is no `cpu`. | `auto` |
| `--precision` | Matmul precision tier: `emu40`, `emu32`, `fp64`, or `tf32`. `emu40` is the parity default. | `emu40` |
| `--config` | **Reserved — not yet supported** (passing one currently errors). | — |

### Examples

Fit the 9-pop golden admixture graph from its edge list, as JSON:

```
steppe qpgraph --f2-dir /path/to/qpgraph_9pop_f2 \
  --graph /path/to/golden_qpgraph_edges.csv \
  --format json --device 0
```

Expect a fit score, the fitted mixing weight(s) with a lo/hi bracket across restarts, the fitted edge lengths, and the worst-fitting statistic as a signed z-score.

---

## qpgraph-search

Enumerate every admixture-graph shape over a bounded set of populations, fit them all, and return the single best.

### What it does

Instead of you guessing a graph shape, `qpgraph-search` takes a fixed list of leaf populations and a small cap on the number of mixture events, then scores **every** graph shape you can build in that bounded space. Because the space is finite and enumerable, the global best is computed exhaustively and is fully deterministic — the same pops give the same winner every run. Every shape is scored by the same single-graph fit that `qpgraph` uses, so each score is trustworthy on its own.

Reach for this when you have a handful of populations and no strong prior on how they connect. Version 1 supports `--max-nadmix 0` (pure trees) and `--max-nadmix 1` (a single mixture event). All shapes are packed into one GPU launch, so the throughput scales; the host only enumerates the shapes and takes the final argmin.

The result gives you the winning shape's edge list (ready to feed back into `qpgraph`), its score, the runner-up's score as an identifiability margin, and the full scored list of candidates. It also runs a hill-climb heuristic and checks that the heuristic lands on the same global best — a self-check, not a separate search.

### Flags

| flag | what it does | default |
|---|---|---|
| `--f2-dir` | The f2-blocks directory to read (`f2.bin` + `pops.txt`), as built by `extract-f2`. Required. | — |
| `--pops` | The bounded leaf pop-set to enumerate topologies over (space- or comma-separated). Needs at least 3 names; each must have f2 statistics. Required. | — |
| `--max-nadmix` | The ceiling on admixture nodes a candidate may have. v1 supports `0` (pure trees) or `1` (one mixture event). | `1` |
| `--numstart` | Per-candidate optimizer restart count — how many starting points each shape is fit from. | `10` |
| `--diag-f3` | f3-covariance regularization before inversion, as a fraction of the diagonal. Matches the reference `diag_f3`. | `1e-5` |
| `--fudge` | Ridge on the per-candidate branch-length solve, to keep it stable. Matches the reference `diag`. | `1e-4` |
| `--constrained` / `--no-constrained` | Hold drift edges at or above zero for every candidate. `--no-constrained` allows negative branch lengths. | on (`--constrained`) |
| `--out` | Write output to this file instead of stdout. | stdout |
| `--format` | Output format: `csv`, `tsv`, or `json`. | `csv` |
| `--device` | CUDA device(s): `auto`, an ordinal like `0`, or two ordinals. GPU-only — no `cpu`. | `auto` |
| `--precision` | Matmul precision tier: `emu40`, `emu32`, `fp64`, or `tf32`. `emu40` is the parity default. | `emu40` |
| `--config` | **Reserved — not yet supported** (passing one currently errors). | — |

Note: `--diag-f3`, `--fudge`, `--numstart`, and `--constrained` here are the single-graph fit settings applied to each candidate — the same knobs `qpgraph` exposes — so a searched shape is scored identically to a standalone `qpgraph` fit of that shape.

### Examples

Search all shapes over six populations with up to one admixture event, as JSON:

```
steppe qpgraph-search --f2-dir /path/to/qpgraph_9pop_f2 \
  --pops England_BellBeaker,Czechia_EBA_CordedWare,Turkey_N,Mbuti,Han,Karitiana \
  --max-nadmix 1 --numstart 10 --format json --device 0
```

Expect the enumeration counts (how many trees and single-admixture shapes were scored), the winning shape with its edge list and score, the runner-up score, and a throughput number (shapes scored per second).

Restrict to pure trees only (no mixture events):

```
steppe qpgraph-search --f2-dir /path/to/qpgraph_9pop_f2 \
  --pops England_BellBeaker,Czechia_EBA_CordedWare,Turkey_N,Mbuti,Han,Karitiana \
  --max-nadmix 0 --device 0
```

---

## Gotchas

- **Both need an f2-dir first.** `--f2-dir` must point at a directory built by [`extract-f2`](./extract-f2.md) (`f2.bin` + `pops.txt`). Neither command reads genotype files directly.
- **Leaf names must match the f2 pops exactly.** In `qpgraph`, every leaf in your edge list (any node that is never a parent) must be spelled exactly as a population in `pops.txt`. In `qpgraph-search`, every name in `--pops` must have f2 statistics. Internal node labels in a graph edge list are free-form and do not need to match anything.
- **An admixture node is a node with two parents** — one that appears as the *child* of two edges. That is how you encode a mixture in the edge list; there is no separate admixture keyword.
- **`--max-nadmix` is capped at 1 in v1.** Values above 1 are not supported yet — the search covers pure trees (`0`) and single-mixture graphs (`1`) only.
- **`--pops` needs at least 3 names.** Fewer is rejected; an empty or too-small set is an invalid configuration.
- **GPU-only.** `--device` never accepts `cpu`. steppe is a GPU product; there is no CPU runtime path.
- **Wide restart brackets are a warning, not a bug.** In `qpgraph` output, a weight whose lo/hi bracket is wide (or a large restart spread) means the restarts disagreed — the weight or shape is not well pinned down by the data, not that the fit failed.

## See also

- [`extract-f2`](./extract-f2.md) — build the f2-blocks directory both commands read.
- [`qpfstats`](./qpfstats.md) — produce a smoothed f2 dir from genotypes to feed these commands.
- [`qpadm`](./qpadm.md) — a related model-fitting command on the same f2-blocks path.
- `docs/reference/include_steppe_qpgraph.hpp.md` — internals and result fields for the single-graph fit.
- `docs/reference/include_steppe_qpgraph_search.hpp.md` — internals of the topology search: enumeration, the fleet launch, and the heuristic self-check.

# `qpgraph_search.cpp` reference

## 1. Purpose

`src/core/qpadm/qpgraph_search.cpp` is the driver for the qpGraph *topology
search*: given a fixed set of populations, it scores every admixture-graph shape
in a small, bounded space and returns the single best-fitting one.

A useful mental model is a small assembly line:

1. A host-side **enumerator** lists every candidate graph shape.
2. A **shared statistical basis** (the observed f-statistics and their inverse
   covariance) is assembled **once** for the whole population set.
3. A **heterogeneous fleet** fits *all* the candidate shapes against that shared
   basis in a single GPU launch.
4. A deterministic **argmin** picks the global-best shape.
5. A **hill-climb heuristic** then re-derives that same global-best from a handful
   of starting points, as a falsifiable check that a local search would find the
   same answer an exhaustive scan does.

The file is host-only and contains no CUDA code. It reaches the GPU through exactly
one seam — the backend's fleet-fit call — the same resident-basis pattern the
single-graph qpGraph run uses. Because the statistical basis depends only on the
population set (never on the graph shape), it is computed once and reread by every
candidate, so there is no per-candidate copying of data to or from the GPU.

The heavy lifting (parsing graph shapes, assembling f-statistics, jackknife
covariance, the actual fit) all lives in other units. This file's own contribution
is the orchestration: how the shared basis is built, how each candidate is lined up
against it, and how the exhaustive answer is cross-checked by a local search.

---

## 2. The canonical basis reuse (the central design idea)

Every candidate graph is fit against the **same** observed f-statistics and the
**same** inverse covariance matrix. This is the design's core claim: the basis is
*population-set-bound*, not *topology-bound*. Assemble it once, and every candidate
reads it.

The basis is a set of population pairs. With `npop` populations, one population is
chosen as the fixed **base** (`pops[0]`), and the pairs are all
choose(`npop`, 2) combinations `(a, b)` with `a <= b` drawn from the `npop − 1`
non-base populations, taken in the order the populations were supplied (the
diagonal `a == b` is included). Each basis entry is a triple `{base, a, b}` of
f2-axis indices; that flattened list of triples is what the f3 assembly step
consumes. This ordering is fixed and independent of any graph shape — it is the
"canonical" pair order that gives the rest of the file its name.

The subtlety this file exists to handle: when a specific graph shape is parsed, its
leaves come out in a different order (the order nodes are first seen while walking
that shape), so *that shape's* internal pair-to-leaf mapping does not line up with
the canonical order. Before a shape can be fit against the shared basis, its
mapping is **remapped** into the canonical pair order (see §5). After the remap, the
shared observed value for pair `k` always corresponds to that shape's path weights
for pair `k`, for every shape. That alignment is what makes one shared basis usable
by all candidates at once.

---

## 3. Named constants

All of these are defined in an anonymous namespace at the top of the file. Note that
they are the constants of the topology *search* — the projected-Newton fit that
actually scores a single graph has its own separate constants elsewhere.

| Constant | Value | What it's for |
|---|---|---|
| `kPrimaryGpu` | `0` | The device index the search runs on. The search always uses a single, primary GPU; the entry points pull that GPU's backend out of the resource set through this index. |
| `kMaxHillClimbSteps` | `1000` | A safety cap on how many steps a single hill-climb descent may take before giving up. The bounded search space has finite neighborhoods and the climb is strictly downhill, so it reaches a local minimum in far fewer steps than this in practice. The cap exists only to guard against a pathological walk that never terminates — it is not expected to fire. |
| `kRecoveryRtol` | `1e-6` | The **relative** tolerance in the recovery check that compares the hill-climb's answer against the exhaustive global-best score. This mirrors the roughly 1e-6 relative tolerance used for the qpGraph fit's own reference comparison against ADMIXTOOLS 2. |
| `kRecoveryAtol` | `1e-9` | The **absolute** tolerance in that same recovery check. Two scores count as equal when `|a − b| <= kRecoveryAtol + kRecoveryRtol · |b|`. |

---

## 4. The `CanonicalBasis` struct

`CanonicalBasis` is the in-memory description of the shared pair set from §2. It is
built once, by `build_canonical_basis`, from the population list and the full list
of leaf names, and is independent of any graph shape.

| Field | Type | Meaning |
|---|---|---|
| `npop` | `int` | The number of populations in the search set. |
| `npair` | `int` | The number of basis pairs, equal to choose(`npop`, 2). |
| `flat` | `vector<int>` | A flattened list of `3 · npair` f2-axis indices: the triple `{base, a, b}` for each pair, in canonical order. This is the exact form the f3-triple assembly step reads. |
| `pair_a_pop` | `vector<int>` | For each of the `npair` pairs, the f2-axis index of that pair's first leaf `a`. |
| `pair_b_pop` | `vector<int>` | For each of the `npair` pairs, the f2-axis index of that pair's second leaf `b`. |

`build_canonical_basis` first maps each leaf name to its position on the f2 axis,
then maps each supplied population to that f2 index. The base population is `pops[0]`;
the remaining populations, in supplied order, become the non-base leaves. The double
loop over `a <= b` then emits the triples and records each pair's two leaf indices.
A population whose name is not found among the leaves is recorded as `-1`.

---

## 5. Building a candidate's arena (`make_canonical_arena`)

`make_canonical_arena` prepares one parsed graph shape to be fit against the shared
basis. Its whole job is the remap described in §2.

It first builds a lookup from f2-axis population index to *this shape's*
centered-column index, by walking the shape's own centered columns and reading which
leaf — and therefore which f2 population — each one carries.

Then, for every canonical pair `k`, it looks up the centered-column indices of the
two leaves that carry that pair's f2 populations `a` and `b`, and stores them as the
shape's remapped pair mapping (`cmb1[k]`, `cmb2[k]`). The two are ordered so the
smaller comes first; because the path-weight term for a pair is symmetric in its two
leaves, this ordering is only cosmetic. If either of the pair's populations is not
present in this shape, the function returns `false` — a structural mismatch that
should not occur for a correctly enumerated candidate.

The rest of the function copies the shape's structural description into the arena
(population and edge counts, the number of admixture events, the number of paths, the
base leaf, the initial path weights, and the several path-encoding tables) and adds
the fit options that vary per run (`constrained` and `fudge`). The result is a
self-contained arena the fleet can fit without touching the original parsed model.

---

## 6. The search pipeline (`run_search_impl`)

`run_search_impl` is the templated body shared by both public entry points; the
template parameter is just the source of the f2 blocks (on-device or host tensor).
It runs the assembly line in numbered stages that match the comments in the source:

1. **Validate and bound.** The search needs at least three populations. The number of
   admixture events is capped at `1` — the "v1" bounded space is trees plus
   single-admixture graphs only. The fit precision policy and its honored tag are
   resolved here.

2. **Enumerate.** The bounded space is listed by a cheap host-side combinatorial
   pass. Each candidate is tallied as either a tree (zero admixture events) or a
   single-admixture graph. An empty list is an invalid configuration.

3. **Assemble the shared basis.** The canonical basis (§2, §4) is built, the f3
   triples are assembled from it into the observed-statistics block tensor, and a
   block jackknife produces the covariance and its inverse. From these come the two
   shared inputs every candidate is fit against: the observed vector and the inverse
   covariance matrix. A degenerate basis or a non-positive-definite covariance ends
   the run early with the corresponding status.

4. **Build every candidate's arena.** Each enumerated shape is parsed and turned into
   a canonical arena (§5). Shapes that fail to parse or fail the remap are skipped;
   a per-candidate index records which arena, if any, each candidate produced. If no
   arena survives, the configuration is invalid.

5. **Fit the whole fleet in one launch.** All arenas are handed to the backend's
   single heterogeneous fleet-fit call, which fits every shape against the shared
   inputs at once. The wall time of that one launch, and a topologies-per-second
   rate, are recorded.

6. **Score vector and global-best.** Every successfully-fit shape's canonical hash,
   edges, and best-of-restarts score are retained in enumeration order. A single
   deterministic pass over the finite scores finds the best and second-best,
   ignoring any non-finite score, and maps the winning arena back to its candidate to
   fill in the reported best. If no finite score exists, the covariance is treated as
   non-positive-definite.

7. **Full fit of the winner.** The global-best is then re-fit on its own — a second,
   cheap, single-topology launch through the same seam — to obtain the complete result
   structure (score, admixture weights with confidence bounds, edge lengths, and the
   graph's edge and leaf tables). This is the reference-grade result that the
   per-candidate parity check diffs against ADMIXTOOLS 2's fit of the same edges. It
   is a genuine new fit, not a reuse of the batch scores.

8. **Heuristic recovery** (optional, see §7).

On success the status is set to OK and the fully-populated result is returned.

---

## 7. The multi-start hill-climb recovery heuristic

When enabled, this stage answers a falsifiable question: would a *local* search,
starting from a few points, arrive at the *same* global-best that the exhaustive
scan found? If the local search and the exhaustive scan agree, the recovery flag is
set true.

**Score cache.** A cache keyed by each shape's canonical hash is seeded from the
exhaustive run — every shape in the bounded space is already scored, so the
hill-climb re-fits nothing. If a neighbor ever appeared that was not in the
exhaustive set (it should not, for this bounded space), it would be parsed, built
into an arena, and fit through the seam as a one-topology batch, then cached.

**Multi-start descent.** The number of starting points is `max(1, heuristic_seeds)`.
The seeds are spread evenly across the candidate list (strided) so they land in
different basins. From each seed, the search does best-improvement descent: at each
step it lists that shape's neighbors, scores them from the cache, and moves to the
strictly-best-improving one; it stops when no neighbor improves (a local minimum) or
after `kMaxHillClimbSteps`. Each seed's final local minimum is recorded.

**Aggregate answer.** A single hill-climb only finds a local minimum, and some seeds
are expected to land in worse basins — that is normal. The heuristic's actual answer
is the best local minimum across all seeds.

**Recovery gate.** Recovery is declared only when that aggregate answer matches the
exhaustive global-best on **both** its canonical hash and its score. The score match
uses the combined tolerance from §3
(`|agg − best| <= kRecoveryAtol + kRecoveryRtol · |best|`). Both conditions must
hold; a matching score with a different hash, or vice versa, is not recovery.

---

## 8. Entry points

Two public overloads, both thin wrappers, differ only in where the f2 blocks live:

- `run_qpgraph_search(const device::DeviceF2Blocks& f2, …)` — f2 blocks already on
  the GPU.
- `run_qpgraph_search(const F2BlockTensor& f2_host, …)` — f2 blocks in a host
  tensor.

Each resolves the primary GPU's backend (via `kPrimaryGpu`) from the passed-in
resource set and forwards to the shared `run_search_impl` body. The result type,
options, and per-candidate structures they exchange are declared in the public
header and described in its own reference.

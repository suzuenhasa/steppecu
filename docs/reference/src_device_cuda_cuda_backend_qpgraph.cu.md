# `cuda_backend_qpgraph.cu` reference

## 1. Purpose

`src/device/cuda/cuda_backend_qpgraph.cu` holds the GPU-backend code that fits
qpGraph models — the admixture-graph models that explain a set of populations by a
tree of drift edges plus a handful of admixture (mixture) nodes. It contains the
out-of-line bodies of exactly two methods of the CUDA backend class:

- `qpgraph_fit_fleet` — fit **one** topology, and return its best fit.
- `qpgraph_fit_fleet_batch` — score **many** candidate topologies at once, and
  return one best score per candidate.

This file is only the *host-side orchestration*: it uploads the data the GPU needs,
launches a kernel, and reduces the small result arrays that come back. The actual
math — the per-restart optimizer loop and the objective it minimizes — lives in the
companion kernel file (`qpgraph_fit_kernels.cu`), and the helper types and launch
wrappers it uses are declared in `qpgraph_fit_kernels.cuh`. Splitting this out from
the main backend file changed nothing about the math, the precision, or the results;
it is purely a code-organization boundary.

The two methods are deliberately different in ambition. The single-topology method
does a *full fit*: it returns the mixture weights, the fitted drift-edge lengths, the
fitted statistics, and a fit score. The batch method does a *screening pass*: it
returns only a score per candidate topology, because it exists to rank a large space
of candidate graphs quickly, not to fully solve each one.

---

## 2. The fleet idea: one launch does the whole search

"The fleet" is the design that makes this fast. A qpGraph fit is a small nonlinear
optimization that has to be run from many random starting points ("restarts", also
called multistart) because the objective has local minima, and each restart runs an
iterative optimizer for up to a fixed number of iterations (`maxit`).

The naive way to do this on a GPU would launch a kernel per iteration and let the
host drive the loop, paying a round-trip to the GPU on every step. Instead, the fleet
launches **once**. Each restart is one GPU thread, and that single thread runs the
*entire* optimizer loop — all `maxit` iterations of the projected-Newton search — from
start to finish inside the kernel, without ever returning to the host mid-fit. The
host only sees the final per-restart answers. This is why the code paths here are
short: upload the inputs, fire one launch, read back a few small arrays.

The batch method takes the same idea one level up. It flattens all
(topology, restart) pairs onto a single thread grid and fits every restart of every
candidate topology in **one** launch.

---

## 3. Precision: the native double-precision carve-out

Both methods take a `precision` argument and both deliberately ignore it (the
`(void)precision;` line). The in-thread optimizer always runs in the GPU's native
double precision, and it uses **no** matrix-multiply library and **no** dense-solver
library — the whole objective is coded by hand in the kernel.

The reason is that this is exactly the kind of numerically delicate work that steppe
always keeps in true double precision: the objective involves small differences that
can cancel, and an inner edge-solve that can be ill-conditioned. The faster emulated
double-precision path that steppe uses elsewhere is for large, well-conditioned
matrix multiplies — a different shape of computation than this per-thread solver — so
the precision knob does not apply here and is intentionally not honored. It is
accepted only so this method matches the common signature of the backend's other
fit entry points.

---

## 4. The per-thread weight-stack cap

Inside the kernel, each restart thread stores the mixture weights it is optimizing
(and their small forward-difference perturbations) in **fixed-size** local arrays.
The length of those arrays is a compile-time constant, `kMaxThetaDev` (currently
**16**), declared in the kernel header. So a topology's admixture-node count
(`nadmix`) must never exceed that cap, or a thread would write past the end of its
local arrays.

Both methods guard against this **before** launching. If a topology's `nadmix`
exceeds `kMaxThetaDev`, the method throws a `std::runtime_error` naming the offending
count and the cap, rather than launching a kernel that would silently corrupt its own
stack. The batch method checks each candidate individually and names which candidate
failed. (The kernel additionally returns a large sentinel score on the over-cap path
as a second line of defense, but the host throw is the loud, intended failure.) A cap
of 16 comfortably covers real models, whose admixture-node counts are small.

---

## 5. Fitting one topology (`qpgraph_fit_fleet`)

This method fits a single topology and returns a full `QpGraphFleet` result. Its
steps are:

**Upload the shared basis and the topology description.** Two arrays describe the
target the fit is trying to match: `f_obs` (the observed statistics, one value per
population pair) and `qinv` (the inverse covariance matrix used to weight the fit,
stored column-major). Both are copied to the GPU once. The topology itself arrives as
a bundle of flat integer and floating-point arrays (the `QpGraphTopoArena`) — the
path tables that describe how each edge contributes to each statistic, the starting
edge weights, and the pair indices. These are all uploaded once and then referenced
by every restart; a single topology is uploaded exactly once and reused, never
re-sent per restart. A small struct of device pointers and scalars
(`QpGraphDeviceTopo`) is filled in to hand all of this to the kernel.

**Fit.** The number of restarts is taken from `numstart` (at least one). The launch
runs every restart, and writes back two small arrays: the converged weights for each
restart (`numstart × nadmix`) and each restart's final score (`numstart`).

The pure-tree case (`nadmix == 0`) short-circuits before the fleet launch — see
section 6. The best-restart selection and edge recovery that follow the launch are
described in section 7.

---

## 6. The pure-tree shortcut (no admixture)

When a topology has zero admixture nodes (`nadmix == 0`) it is a plain tree: there are
no mixture weights to optimize, so there is nothing for the multistart fleet to
search. The method skips the fleet entirely and does a single on-device objective
evaluation with a **null** weight pointer.

This works because the same objective routine the fleet uses also handles the tree
case directly: with no admixture nodes present it never dereferences the weight
pointer, so passing it nothing is safe, and the one evaluation both solves for the
edge lengths and computes the score. This lets the tree case reuse the exact same
device code as the admixture case instead of needing a separate host-side path.

If that single evaluation comes back with a non-finite or huge-sentinel score, the
inner solve was degenerate: the result is marked `NonSpdCovariance` and the edge
lengths and fitted statistics are zeroed. Otherwise the score, edges, and fitted
statistics are returned with status `Ok`.

---

## 7. Choosing the best restart and recovering the edges

After the fleet launch returns its per-restart weights and scores, a tiny host loop
over the (few) restart rows does three things:

- **Best-of-restarts.** Pick the restart with the smallest (best) score. Its weights
  become the reported mixture weights (`theta`), and its score becomes the reported
  fit score.
- **The per-weight bracket.** For each mixture weight, record the minimum and maximum
  value it took across all restarts (`theta_lo` / `theta_hi`). This is a cheap,
  informative sanity signal: if every restart converged to nearly the same weights,
  the bracket is tight.
- **The restart spread.** Record the gap between the best and worst finite scores
  (`restart_spread`). A small spread is evidence the restarts agreed on one optimum —
  a witness that the fit converged rather than getting stuck at scattered local minima.

**Recovering the edges at the winning weights.** The fleet returns only weights and
scores, not the fitted drift-edge lengths or fitted statistics. To get those, the
method uploads the winning weights and runs the objective **one more time** at exactly
those weights (`launch_qpgraph_eval_at_theta`), which exports the fitted edge lengths,
the fitted statistics, and the score at that point. This reuses the basis and topology
that are still resident on the GPU from the fit, so it is a single cheap extra
evaluation rather than a host-side re-solve. As in the tree case, a non-finite or
huge-sentinel score maps to `NonSpdCovariance` with the edges and fitted statistics
zeroed; otherwise status is `Ok`.

---

## 8. Fitting many topologies at once (`qpgraph_fit_fleet_batch`)

This method screens a whole list of candidate topologies that all share the same
population set — and therefore the same observed statistics and the same weighting
matrix (`f_obs` / `qinv`), uploaded once and read by every candidate. An empty list
returns immediately with status `Ok`.

**Packing.** Every candidate's flat arrays are concatenated end-to-end into a few big
host buffers (one per array kind), and a per-candidate *view* record
(`QpGraphDeviceTopoView`) stores where that candidate's slice begins in each buffer as
an element offset. The kernel rebuilds each candidate's pointers by adding these
offsets to the base pointers of the packed buffers. The upshot is that any number of
different-shaped topologies live in one set of device buffers plus one small index
table, so the batch is uploaded as a handful of copies rather than one upload per
candidate. Each candidate is also checked against the weight-stack cap here (section 4).

**The batch-maximum scratch slab.** Each restart thread needs per-thread scratch
memory, and different topologies need different amounts. Rather than size scratch per
candidate, the method computes the largest dimensions across all candidates and builds
a single scratch layout big enough for the biggest one (`make_layout` on the
batch-maximum sizes). Every thread gets an identically-sized slab from that layout, so
the largest candidate fits and the smaller ones simply leave part of their slab unused.
The total scratch is one slab per thread, and there are (number of topologies × number
of restarts) threads. Note the contrast with the single-topology method: there the
scratch is sized and allocated internally by its launcher, while here the caller must
size and allocate the shared batch-maximum slab because it has to cover a heterogeneous
mix.

**One launch, then a reduction — not a fit.** A single launch fits every restart of
every candidate and writes one score per (topology, restart). The host then reduces
those scores per topology into the best (minimum) score and the restart spread
(max minus min). That is deliberately *all* it produces: no weights, no edges, no
fitted statistics. This method is the fast ranking pass over a space of candidate
graphs; once a promising topology is identified, the single-topology method does the
full fit on it.

---

## 9. The result structures and status

Both methods return value types (defined in `src/device/backend.hpp`); a degenerate
fit is reported as a status value, never thrown.

**`QpGraphFleet`** (single-topology result):

| Field | Meaning |
|---|---|
| `theta` | The best restart's mixture weights, one per admixture node. |
| `theta_lo` / `theta_hi` | The per-weight minimum/maximum across restarts (the bracket from section 7). |
| `edge_length` | The fitted drift-edge lengths at the best weights. |
| `f3_fit` | The fitted statistics implied by those edges. |
| `score` | The best (minimum) fit score across restarts. |
| `restart_spread` | The best-minus-worst score gap — a convergence witness. |
| `status` | `Ok`, or `NonSpdCovariance` when the inner solve was degenerate (edges and fitted statistics zeroed). |

**`QpGraphFleetBatch`** (screening result), with one entry per candidate topology:

| Field | Meaning |
|---|---|
| `best_score` | The best (minimum over restarts) score for each candidate. |
| `restart_spread` | The per-candidate best-minus-worst score gap. |
| `status` | `Ok`. |

A large sentinel score (on the order of `1e30`) or a non-finite score always means
the fit's inner solve failed at that point; the single-topology path turns that into
`NonSpdCovariance`, and the batch path simply leaves that candidate's best score as
non-finite so the ranking pushes it to the bottom.

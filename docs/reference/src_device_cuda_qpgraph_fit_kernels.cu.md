# `qpgraph_fit_kernels.cu` reference

## 1. Purpose

`src/device/cuda/qpgraph_fit_kernels.cu` holds the GPU code that fits a qpGraph
model — an admixture-graph topology whose edge lengths and admixture weights are
chosen to best explain a set of observed f-statistics.

Fitting a qpGraph is an optimization problem. The unknowns are the admixture
weights (called *theta*), one per admixture node. For any candidate theta the code
solves for the best edge lengths and computes a goodness-of-fit *score* (lower is
better). Because that score surface has many local minima, the fit is run from many
random starting points ("restarts", also called a multistart), and the best
restart wins.

The file contains three GPU kernels and three host-callable launchers that wrap
them:

1. **The fleet** — fit one topology from `numstart` restarts, one GPU thread per
   restart, each thread running the entire optimizer loop by itself.
2. **The evaluate-at-theta pass** — take one already-chosen theta and export the
   fitted edge lengths and fitted f-statistics for it.
3. **The heterogeneous-topology batch fleet** — fit many *different* topologies in
   a single launch, so a whole space of candidate graphs can be scored at once.

Everything the optimizer needs lives on the GPU for the duration of a fit. The host
launches a kernel once and gets back only the small per-restart results; there is no
per-iteration round trip to the CPU.

The companion header `qpgraph_fit_kernels.cuh` documents the data structures this
file consumes (`QpGraphDeviceTopo`, `QpGraphDeviceTopoView`, `ScratchLayout`,
`make_layout`, and the `kMaxThetaDev` cap) and the launcher signatures. This
reference covers the algorithms and design decisions inside the `.cu` itself.

---

## 2. The fleet design: one thread per restart

The parallel axis is the restart, not the math inside a single fit. Each GPU thread
takes one restart index and runs the *whole* optimizer — the multistart
initialization, every optimizer iteration, and every objective evaluation — start to
finish, by itself. With `numstart` restarts the launch is simply `numstart` threads
(grouped 64 to a block), each an independent fit.

This is a deliberate choice over the more obvious "parallelize the linear algebra of
one fit across many threads" design. The reason: the classic way to write this
optimizer (the approach the reference uses[^at2]) is a host loop that calls the objective,
looks at the result on the CPU, decides the next step, and calls the objective
again. That per-iteration CPU round trip is a throughput trap — the GPU sits idle
waiting for the host between every evaluation. Here the host launches the kernel
exactly once and receives back only the per-restart `{theta, score}` arrays. The
optimizer's decision-making never leaves the device.

The best-of-restarts selection, any confidence bracket, and the final edge-length
recovery are all done on the host afterward, working over those small per-restart
arrays — none of that needs the GPU.

---

## 3. Precision policy

The numerically delicate parts of the fit run in **native double precision**: the
symmetric positive-definite edge-length solve and the final goodness-of-fit
quadratic form. These involve subtracting nearly-equal quantities (cancellation),
where lower-precision arithmetic would lose accuracy, so they are carved out to run
in native FP64. This exactly matches the CPU reference ("oracle") the GPU results
are validated against.

This carve-out is affordable because the objective evaluated inside a thread is
small — it works over at most a per-thread number of edges that fits the scratch
budget. The heavy, production-scale matrix multiplies that assemble the fit's inputs
(the "cc" assembly) run elsewhere, in the batched emulated-double-precision path;
that larger path was measured to be GPU-bound (over 90% utilization). The in-thread
optimizer body in this file is intentionally *not* that path.

---

## 4. Per-thread scratch slabs

Each fit needs several working arrays (the centered path weights, the assembled
normal-equation matrices, the solver's temporaries, and so on). Rather than trying
to keep these in registers or local memory — which would cap how large a graph can
be fit — every thread is given its own slab of global GPU memory to work in.

The launcher allocates `numstart` slabs (one per restart) up front and frees them
when the fit finishes. The size and internal layout of a slab come from
`make_layout` in the header, which computes the byte offset of every sub-array from
the topology's dimensions. Because the offsets are computed once and passed in, the
same slab layout is understood identically by the host (which sizes and allocates
it) and the kernel (which indexes into it).

Sizing the scratch to the topology at runtime is what lets the fit scale past the
per-thread register and local-memory envelope, which matters because production
graphs have many edges. The read-only integer arrays that describe the topology (the
path tables) are uploaded once and stay resident; they are shared by every thread
and never written.

---

## 5. The theta-stack cap (`kMaxThetaDev`)

A few small per-thread arrays — the current theta and its forward-difference
perturbations — are held in fixed-size stack arrays inside the kernel, of length
`kMaxThetaDev` (defined as **16** in the header). The number of admixture nodes,
`nadmix`, must therefore be at most 16. Sixteen comfortably covers real graphs;
admixture-node counts are small in practice.

The cap is guarded twice, so an oversized topology fails loudly rather than silently
overrunning those stack arrays and corrupting neighboring locals:

- **Before launch**, each host launcher checks `nadmix` against the cap and throws a
  clear `std::runtime_error` naming both the offending count and the cap.
- **Inside the kernel**, the per-restart fit re-checks and returns a singular
  sentinel score of `1e30` if the topology is over-cap, without touching the theta
  stack.

Because the host rejects an over-cap topology before launch, the reference tests
never exercise the device sentinel branch — it exists purely as defense in depth.
The read-back of the result also clamps to the stack length, so nothing reads past a
partially-filled theta array on the sentinel path.

---

## 6. Deterministic multistart initialization

Each restart's starting theta is generated deterministically from its restart index
and the dimension index using a splitmix-style hash (`d_init_theta`). Given the same
restart and dimension, the same starting value comes out every time — so a fit is
fully reproducible, and the GPU fleet starts from exactly the same points as the CPU
reference.

That bit-for-bit agreement with the CPU reference is why the hash constants (the
multipliers, the seed increment, the two mixing constants, and the mantissa
mask/divisor that map the hash into the `[0, 1)` range) are not written here. They
live in one shared constants header included by both the GPU kernel and the CPU
oracle, so neither can drift from the other. The starting theta is clamped into
`[0, 1]` before use.

---

## 7. The GLS objective

`d_qpgraph_score` is the objective the optimizer minimizes: given a candidate theta,
it returns the model's goodness-of-fit score. It mirrors the CPU reference
`core::qpadm::qpgraph_score` step for step, using the scratch slab for all its
intermediate arrays. The pipeline is:

1. **Centered path weights** — turn theta into the per-edge, per-population weight
   matrix `pwts_c` (see section 8).
2. **Predictor products** (`ppwts`) — for each population pair, multiply the two
   populations' centered weight columns edge by edge. These are the model's
   predictors, one column per edge.
3. **Weighting by the inverse covariance** — form `Wm = qinv · ppwts` and
   `qf = qinv · f_obs`, where `qinv` is the inverse covariance of the observed
   statistics. This is the generalized-least-squares weighting.
4. **Normal equations** — form the edge-by-edge matrix `cc = ppwts' · Wm` and the
   right-hand side `q = ppwts' · qf`.
5. **Ridge regularization**[^at2] — add `fudge · mean(diag(cc))` to
   the diagonal of `cc`. This trace-scaled ridge (the topology's `fudge` field, the
   parity equivalent of the `diag` term) keeps the solve stable.
6. **Diagonal scaling**[^at2] — rescale the system by the square
   roots of the diagonal of `cc`, so the solve operates on a well-conditioned matrix;
   the solution is un-scaled afterward.
7. **The edge-length solve** — solve for the edge lengths `bl`, either box-constrained
   or unconstrained (see section 9), then divide out the scaling from step 6.
8. **The score** — form the residual `res = f_obs − ppwts · bl` and return the
   quadratic form `res' · qinv · res`. Steps 7's solve and this quadratic form are
   the native-FP64 carve-out from section 3.

If the inner solve is singular, the objective returns the `1e30` sentinel, which the
host maps to a non-SPD-covariance failure status for that fit.

---

## 8. The centered path-weight fill

`d_fill_pwts_centered` turns a theta vector into the centered path-weight matrix
`pwts_c` (edges by populations, minus a base population). It mirrors the CPU
reference `fill_pwts_centered`, and it is the general form of what used to be a
hard-coded routine for a single fixed number of admixture nodes: it works for an
*arbitrary* topology by reading the device path tables instead of assuming a shape.

The idea: every population is reached from the graph's root along one or more paths.
Each path's weight starts at 1 and is multiplied down by the admixture weights of the
admixture edges it crosses — an admixture edge contributes `theta` on one side of the
split and `1 − theta` on the other (which side is encoded in the path table's
1-based admixture-edge id, odd versus even). The resulting per-path weights are then
summed into the `(edge, population)` cells the paths touch.

The routine produces the *centered* weights directly — each population column is
expressed relative to a chosen base population — because that is the form the rest of
the objective consumes. It starts each column from the base-relative starting weights
and then applies the path-weight deltas: a change to a non-base population's cell adds
into that population's column, while a change to the base population's cell shifts
*every* column at that edge by the negative of the delta. Because the path-edge table
is ordered by path rather than grouped by cell, the code accumulates each unique
`(edge, population)` cell's total by scanning the table for matching entries at the
cell's first occurrence.

---

## 9. The edge-length solve: constrained NNLS versus unconstrained LU

At the heart of the objective is a small linear solve for the edge lengths. The
topology's `constrained` flag selects between two solvers:

- **Box-constrained (non-negative) solve** — a non-negative least squares (NNLS)
  solver, `d_nnls`, which finds the edge lengths that best fit the data *subject to
  every edge length being ≥ 0*. This is the parity behavior[^at2] and it is required
  for correctness: real fits have edges that sit exactly at the zero boundary, and an
  unconstrained solve cannot represent that — it would hand back a negative length. The
  solver is the Lawson-Hanson active-set method and mirrors the CPU reference
  `nnls_active_set`: it maintains a "passive" set of edges allowed to be positive, adds
  the most-violating edge at each outer step, and solves the reduced system on the
  passive set, backing off along the segment to the boundary whenever a solution
  component would go negative. It stops when the optimality (KKT) conditions are met.

- **Unconstrained solve** — a straight linear solve, used when the topology is not
  flagged constrained.

Both rest on a small native-FP64 dense linear-algebra pair, `d_lu_factor` and
`d_solve`: an LU factorization with partial pivoting and its forward/back
substitution, working on column-major matrices, using caller-provided scratch. The
NNLS routine calls this repeatedly on the reduced passive-set system.

---

## 10. The projected-Newton per-restart loop

`d_fit_one_restart` is the optimizer a single thread runs for one restart. It seeds
theta from the deterministic initializer (section 6), evaluates the objective, and
then iterates a projected-Newton scheme up to `maxit` times. For each admixture
dimension in turn:

- **Gradient and curvature** by forward differences: perturb that one weight up and
  down by a small step, re-evaluate the objective at each, and form a first
  derivative (slope) and a diagonal second derivative (curvature) from the three
  scores. Near the `[0, 1]` boundary, where a two-sided difference doesn't fit, it
  falls back to a one-sided slope.
- **A trust-clamped step**: divide the slope by the curvature when the curvature is
  meaningfully positive (a Newton step), otherwise take a scaled gradient step; then
  clamp the step's magnitude to a fixed trust radius and project the new weight back
  into `[0, 1]`.
- **Backtracking line search**: if the trial score is worse, halve the move back
  toward the current point up to a fixed number of times, accepting the step only if
  it actually lowered the score.

After sweeping all dimensions, the loop stops early when both the largest weight
change and the largest score change fall below the tolerance (each scaled by its own
factor). All of the step sizes, thresholds, clamps, backtrack count, and tolerance
scales are the shared optimizer constants from the single-sourced constants header,
so this device loop stays bit-identical to the CPU oracle.

A pure tree — a topology with **no** admixture nodes (`nadmix == 0`) — has nothing to
optimize, so it takes a single objective evaluation and returns that score directly,
with no optimizer loop and no theta axis.

This function is the shared body of *both* fleet kernels: the single-topology fleet
and the heterogeneous-topology batch fleet call the exact same code, so the two can
never diverge.

---

## 11. The kernels and host launchers

### The single-topology fleet

`qpgraph_fleet_kernel` assigns one thread per restart, points that thread at its own
scratch slab, runs `d_fit_one_restart`, and writes the restart's `{theta, score}`
into the output arrays. `launch_qpgraph_fleet` wraps it: it rejects an over-cap
`nadmix` with a throw, builds the scratch layout, allocates the per-restart slabs,
launches with 64 threads per block, checks for launch errors, and synchronizes the
stream. It returns nothing itself — the caller reads the per-restart output arrays.

### Evaluate at a chosen theta

After the host has picked the winning restart, it needs that fit's fitted edge
lengths and fitted f-statistics — not just its score. `qpgraph_eval_at_theta_kernel`
does this with a single thread: it runs the *same* objective body one more time at
the host-supplied theta and exports `bl` (the fitted edge lengths) and
`f3_fit = f_obs − res` (the fitted statistics), along with the score.

This re-evaluation is done on the device, replacing what used to be a host-side
re-computation, for two reasons. First, the fleet's per-restart scratch has already
been freed by the time the host chooses the winner, so the fitted quantities are no
longer around to read. Second, the last objective evaluation inside the optimizer
loop is a trial probe, not necessarily the accepted step, so its leftover `bl`/`res`
are stale with respect to the theta actually returned — a clean single re-evaluation
at the converged theta is required regardless. Doing it on the identical device body
is at least as faithful as the host re-computation it replaces, and it removes the
host objective entirely. A singular solve here yields the `1e30` sentinel, which the
host maps to a non-SPD-covariance status with the outputs zeroed.
`launch_qpgraph_eval_at_theta` wraps it with the same over-cap check, one-slab
allocation, single-thread launch, and synchronize.

---

## 12. The heterogeneous-topology batch fleet

`qpgraph_fleet_batch_kernel` (and its wrapper `launch_qpgraph_fleet_batch`) fit many
*different* topologies in a single launch — the building block for searching over a
space of candidate graphs. This is the design's payoff: instead of one launch per
candidate graph, an entire batch of graphs is scored at once.

The launch axis is flattened over `(topology, restart)`: a thread's global index is
split as `topology_id = index / numstart` and `restart = index % numstart`, so every
topology gets the same `numstart` restarts. All the topologies' data is packed into a
handful of shared device buffers (the path-weight arena, the path tables, the pair
index tables), and a per-topology "view" (`QpGraphDeviceTopoView`) records that
topology's element offsets into each packed buffer. Each thread reconstructs its
topology's `QpGraphDeviceTopo` on the fly by adding the packed base pointers to its
view's offsets, then runs the same `d_fit_one_restart` inner loop as the single
fleet. So many heterogeneous graphs live in one set of buffers plus one small index
table.

Every thread reads the same resident observed statistics and inverse covariance (the
basis is fixed by the population set, which is shared across the batch). The
per-thread scratch is sized to the *batch maximum* layout — large enough for the
biggest topology in the batch — and each thread rebuilds its own topology's offsets
inside that maximum-sized slab, touching only the cells its topology actually uses.
The caller passes the batch-maximum per-thread double and integer totals so the
kernel strides the scratch with exactly the stride the host allocated.

The kernel writes one score per `(topology, restart)`. Turning that into a per-topology
best fit and the single global best graph is a reduction (an argument-minimum over the
score array), done on the host — it is a reduction over results, not another fit.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

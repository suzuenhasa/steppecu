# `qpgraph_fit.cpp` reference

## 1. Purpose

`src/core/qpadm/qpgraph_fit.cpp` is the driver that fits one fixed admixture
graph to a set of observed f-statistics. This is the qpGraph operation: the caller
hands in a graph (as a list of parent-to-child edges), the names of its leaf
populations, an f2 statistics source, and a small options struct, and gets back a
filled result — fitted admixture weights, edge lengths, a fit score, and a
worst-residual diagnostic.

The file is host-only and contains no CUDA code. It reaches the GPU exclusively
through one seam, the compute backend, so it compiles into the core library, the
command-line tool, and the language bindings without pulling in the device layer.
Every actual number-crunching step is delegated to a backend call; this file's job
is to wire those steps together in the right order and package the outcome.

Two design rules run through the whole file:

- **Reuse over reinvention.** The heavy lifting is done by steps that already exist
  and are shared with the other fit operations: an f3 triple-assembly step, a
  block-jackknife covariance step, and a productized fit "fleet." The only genuinely
  new piece behind this driver is the topology model that turns an edge list into a
  path table (built in a separate file). Everything else is a call into an existing,
  independently tested seam.
- **Problems are values, not exceptions.** A bad graph, a non-invertible covariance,
  or a failed fit is reported by setting the result's `status` field and returning.
  Nothing here throws. The caller (the CLI or a binding) inspects the status and
  surfaces the message.

---

## 2. The four-stage pipeline

The core work happens in a single templated helper, `run_qpgraph_impl`, which runs
four stages in order. Each stage can short-circuit by setting a status and
returning, so a failure in an early stage never reaches a later one.

1. **Model.** Parse the edge list plus leaf names into a topology model. This
   resolves the graph shape: how many populations, edges, and admixture nodes there
   are; which leaf is the "base" the f3 statistics are centered on; and the path
   tables that say how each leaf's allele frequency is built up from edge lengths and
   admixture weights. If the graph is malformed (a parse or structural problem), the
   result is stamped `InvalidConfig` and returned early, still carrying whatever leaf
   list the parser managed to build.

2. **Basis.** Assemble the observed f3 statistics and their covariance — the data the
   fit is measured against. This is where the reused triple-assembly and covariance
   seams are called. On the GPU path the results stay resident in GPU memory and are
   reused across every fit restart, with no copy back to the host in between. Stage 3
   below covers this in detail.

3. **Fleet.** Run the actual fit. This is a single backend call that launches many
   independent restarts at once, each doing the whole optimization loop on the GPU.
   Stage 5 covers this.

4. **Result.** Copy the fitted quantities into the result struct, attach the leaf and
   edge labels from the model, and compute the worst-residual diagnostic (stages 6
   through 8). Stamp the status `Ok` and return.

---

## 3. Building the f3 basis

The fit is measured against a set of observed f3 statistics. This section is stage 2
of the pipeline and is where most of the numerical setup lives.

### Which statistics form the basis

The basis is every f3 statistic of the form `f3(base; a, b)`, where `base` is the
fixed base leaf chosen by the model, and `a` and `b` range over all the *non-base*
leaves with `a <= b`. Including the case `a == b` on the diagonal matters: that
degenerate f3 is just `f3(base; i, i) = f2(base, i)`, so the diagonal entries bring
the plain f2 distances into the basis alongside the genuine three-way statistics.
The number of such pairs is `choose(npop, 2)` counted with the diagonal — i.e.
`npop * (npop - 1) / 2` where `npop` includes the base — and this count is called
`npair`.

### How the basis is handed to the backend

The f3 triple-assembly seam works on a flat array of index triples. This file builds
that array by walking the `npair` pairs and, for each, pushing three f2 column
indices in a fixed order: `{ base, leaf_a, leaf_b }`. The seam reads this array and
produces, for every triple, the assembled f3 value using the standard three-term
identity. The result is `f_obs` — the vector of observed f3 statistics, one per pair.
This exactly reproduces ADMIXTOOLS 2's weighted block-jackknife f3 estimate.

### The covariance and its inverse

The fit weights each residual by how noisy that f3 estimate is, so it needs the
covariance of the basis. This comes from the reused block-jackknife covariance seam,
which partitions the SNPs into blocks, recomputes the basis with each block left out,
and forms the covariance across those leave-one-out estimates. It returns two things
the fit uses:

- `Q` — the raw covariance matrix of the basis.
- `Qinv` — an inverse covariance used as the fit's weighting matrix.

`Qinv` is not the plain inverse of `Q`. It is a *regularized* inverse that matches
ADMIXTOOLS 2's `ppinv`: before inverting, a small ridge is added to the diagonal,
scaled by the trace of `Q`. Concretely, `Qinv = inverse(Q + diag_f3 * trace(Q) * I)`,
which is the same thing ADMIXTOOLS 2 computes when it solves against the f3 variance
with the diagonal bumped by `diag_f3 * sum(diag)`. The `diag_f3` factor is the ridge
strength and comes from the options struct (default `1e-5`). The raw, *unfudged* `Q`
is kept separately because the residual diagnostic in stage 6 needs the true
diagonal, not the regularized one.

There is a second, larger regularization knob, `fudge` (default `1e-4`), which is
carried into the fit fleet through the arena rather than applied here; it stabilizes
the fit's own internal solves.

### Residency and precision

On the GPU path, `f_obs` and `Qinv` live in GPU memory and are reused across all fit
restarts without a round trip to the host. The assembly runs under the library's
default precision policy: the matrix-multiply-heavy assembly uses the fast emulated
double-precision mode, while the small cancellation-prone f3 combine and the
symmetric positive-definite solve inside the covariance step fall back to native
double precision automatically. Section 9 covers this.

---

## 4. The basis-size invariant

After the triple-assembly seam returns, the code recomputes `npair` from the seam's
own output dimensions (`X.nl * X.nr`) rather than trusting the model's `m.npair` that
was used to build the input.

This looks redundant, but it is deliberate. The seam's contract guarantees that the
number of assembled entries equals the number of triples it was fed, so feeding it
exactly `npair` triples must yield `npair` results. Re-deriving the count from the
returned object means the basis dimension always tracks the seam's *actual* output,
which is what the covariance, `f_obs`, and `Qinv` are all sized against. If the two
ever drifted apart, everything downstream would be mis-sized; re-deriving from the
one authoritative source makes that drift impossible.

Immediately after, a guard rejects a degenerate basis: if the recomputed `npair` is
not positive, or the covariance came back with no jackknife blocks, the result is
stamped `NonSpdCovariance` and returned. The covariance step's own status is also
checked and propagated the same way.

---

## 5. The fit fleet

Stage 3 is a single backend call, `qpgraph_fit_fleet`. It runs the entire fit as a
fleet of independent restarts launched together, each restart carried by one GPU
thread that runs the whole optimization loop — a multi-start, bounded-iteration
projected-Newton search — start to finish on the GPU. Crucially, there is no host
involvement per iteration: the objective is never evaluated on the CPU inside the
loop. The host only sees the final fitted quantities once every restart has
converged or hit its iteration cap.

The call takes:

- **The arena** — a small, CUDA-free bundle of the topology (population, edge, and
  path-table sizes; the base leaf; the path-weight tables; the admixture-edge maps)
  copied out of the model, plus two options: whether the fit is constrained and the
  `fudge` regularization value. `make_arena` builds this. It is the only shape the
  device fit needs, kept free of any host-side objects so the backend can consume it
  directly.
- **`f_obs` and `Qinv`** — the observed basis and its weighting matrix from stage 2.
- **The search settings** — `numstart` (how many restarts, default 10), `maxit`
  (per-restart iteration cap, default 200), `tol` (convergence tolerance, default
  `1e-9`), and the precision policy.

The fleet returns, among other things, the best fit's score, the fitted admixture
weights with low/high bounds taken from the spread across restarts, the fitted edge
lengths, and the fitted f3 values (`f3_fit`) that stage 6 compares against the
observed ones. If the fleet reports a non-`Ok` status, that status is propagated and
the driver returns.

---

## 6. The worst f3-residual z-score

After a successful fit, stage 4 reports the single largest standardized residual over
the basis, as a quick "how well did this graph fit?" diagnostic.

For each basis pair `k`, the residual z-score is:

```
z[k] = (f_obs[k] - f3_fit[k]) / se[k]
se[k] = sqrt(Q[k, k])
```

`f_obs[k]` is the observed f3 statistic and `f3_fit[k]` is the value the fitted graph
predicts. The standard error `se[k]` is the square root of the *unfudged* covariance
diagonal — the true variance of that f3 estimate, not the ridge-regularized version
used for the fit weighting. If a diagonal entry is not strictly positive, its
standard error is set to not-a-number so that pair's z-score is non-finite and is
skipped.

The driver scans all `npair` pairs, keeps the one with the largest absolute z-score,
and records both the signed z-value (`worst_residual_z`) and the names of the two
non-base populations of that pair (`worst_pop2`, `worst_pop3`). The population names
are looked up from the model's leaf list. The reported z is signed, but the ranking
that selects it uses absolute value.

---

## 7. Why the residual scan runs on the host

The residual scan in section 6 is a plain host-side loop, and this is a conscious,
documented choice rather than an oversight. Across the codebase, a broad effort moved
per-block and per-model numeric work onto the GPU to protect throughput. This scan
was deliberately left on the host, for two concrete reasons:

1. **Its inputs are already on the host.** The scan needs the unfudged covariance
   diagonal (`Q`, a host-side vector produced by this CUDA-free driver) and the
   fitted f3 values (`f3_fit`, already brought down from the device as part of the
   fit result). Folding the scan onto the GPU would mean shipping `Q` back up to GPU
   memory and then bringing a single index back down — adding round trips to save a
   linear scan.

2. **The result must be resolved on the host anyway.** The useful output is a pair of
   population *name strings*, and those can only be produced by looking indices up in
   the model's host-side maps. The device cannot produce label strings, so a host
   step is unavoidable regardless.

The deciding factor is that this scan is a single, size-`npair`, bounded diagnostic
run exactly once per graph fit. It is not inside any batched, swept, or per-restart
inner loop — the hot paths that the on-device effort targeted. Because it carries no
throughput exposure, keeping it on the host is a net win, not a compromise. The
CPU-reference backend runs this identical host code, so the two paths cannot
disagree. By contrast, the fit's own rank test and the graph re-evaluation that
produces `f3_fit` *did* move onto the device, because those sit in hot loops.

---

## 8. The zero sentinel and the strict comparison

The scan starts with `worst = 0.0` and `worst_k = -1`, and its running-best test is a
*strict* greater-than on absolute value: a pair is recorded only if its `|z|` is
strictly larger than the current best `|z|`. This has two intended consequences worth
knowing before touching the loop.

- **`0.0` does double duty.** It is both the initial value of the running maximum and
  the value `worst_residual_z` keeps when no finite residual is ever found. If every
  pair's z is non-finite (for example, a fully degenerate all-zero-residual basis),
  `worst_k` stays `-1`, no population labels are set, and the reported worst z is the
  initial `0.0`. This is a benign outcome on real data.
- **An exactly-zero residual is never recorded as the worst.** Because the test is
  strict, a pair whose `|z|` is exactly `0` can never beat the initial `0.0`, so it is
  never selected and never populates the population labels.

**Do not relax the comparison to greater-than-or-equal.** That is a behavior change,
not a cleanup: it would start reporting population labels for an exactly-zero residual
that is left empty today.

---

## 9. Precision selection

The whole fit runs under one precision policy, chosen once by the shared
`default_fit_precision()` helper so that this driver and the other fit operations
stay consistent. That policy defaults to the fast emulated double-precision mode for
the matrix-multiply-heavy work (the f3 assembly and the covariance formation), while
the small, cancellation-prone f3 combine and the symmetric positive-definite solve
inside the covariance step drop to native double precision automatically — that
carve-out is handled inside the seams, not here.

The result records which precision was actually honored, via `honored_tag()`. This
matters because the emulated mode is only genuinely available when the GPU code was
built with the corresponding capability; if it was not, the backend quietly runs
native double precision instead, and the honored tag reflects what really ran rather
than what was requested.

---

## 10. Public entry points and the shared implementation

The file exposes two public `run_qpgraph` overloads, differing only in where the f2
statistics come from:

- one takes a device-resident f2 blocks object (the GPU-resident source), and
- one takes a host-side f2 block tensor.

Both are thin forwarders. Each selects the primary GPU's backend (device index 0,
via a fixed constant) from the passed-in resources and calls the single templated
`run_qpgraph_impl`, parameterized on the f2 source type. Keeping the real body in one
template means the two entry points can never drift apart in behavior — they are the
same pipeline fed from two different f2 sources. Selecting the primary GPU here
reflects that this operation runs on one device; the resources handle is still passed
through so the backend seam has everything it needs.

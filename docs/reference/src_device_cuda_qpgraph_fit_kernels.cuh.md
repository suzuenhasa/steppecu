# `qpgraph_fit_kernels.cuh` reference

## 1. Purpose

`src/device/cuda/qpgraph_fit_kernels.cuh` is the GPU-side interface for fitting
admixture-graph topologies (qpGraph). It declares the host-callable launchers that
the CUDA backend calls, plus the small data structures those launchers pass down to
the kernels.

An admixture graph is a model of population history: populations split (drift) and,
at a few places, mix together (admixture). A *topology* is the fixed shape of that
model — the branching pattern and where the mixing events sit. Fitting a topology
means finding the numeric parameters that make it best reproduce the measured
allele-frequency statistics (f-statistics) between populations. This header covers
the machinery that runs that fit on the GPU.

The header holds four kinds of thing:

1. **One named constant** (`kMaxThetaDev`) — the fixed cap on how many admixture
   mixing weights a single fit can hold, because those weights live in fixed-size
   per-thread stack arrays.
2. **Three data structures** (`QpGraphDeviceTopo`, `ScratchLayout`,
   `QpGraphDeviceTopoView`) — the flat, GPU-friendly descriptions of a topology and
   of the per-thread scratch memory a fit needs.
3. **One small inline helper** (`make_layout`) — computes the per-thread scratch
   layout from a topology's dimensions, callable from both host and device.
4. **Three launcher declarations** — one for fitting a single topology with many
   restarts, one for re-scoring a topology at a chosen set of weights and exporting
   the fitted quantities, and one for fitting many different topologies in a single
   launch.

The actual kernels and the device-side objective function live in the paired
`qpgraph_fit_kernels.cu`. This split keeps all CUDA code inside the `.cu`/`.cuh`
files so that the backend interface header the rest of the library includes stays
free of CUDA.

---

## 2. How a topology fit works

Every launcher in this file implements the same underlying fit, so it helps to know
what that fit is before reading the individual contracts.

For a *fixed* topology, the fit has two layers of parameters:

- **Drift edge lengths** — one non-negative number per graph edge, saying how much
  genetic drift happened along that branch.
- **Admixture mixing weights** (called `theta`) — the proportions at each mixing
  node, saying how much ancestry each incoming branch contributes.

The key simplification is that, for any *given* set of mixing weights, the best drift
edge lengths can be found directly by a weighted least-squares solve. So the
optimizer never has to search over the edge lengths — it only searches over the
mixing weights `theta`, and at each candidate `theta` it solves for the edge lengths
and measures how well the resulting graph reproduces the observed statistics. The
leftover mismatch (the residual) becomes the fit *score*: lower is a better fit.

Because that score surface has local minima, the fit uses **multiple random
restarts** (a "multistart"): it runs the same projected-Newton optimization from many
different starting weights and keeps the best result. The "fleet" idea in this file
is to run all of those restarts in parallel on the GPU — one thread per restart —
with the entire optimization loop running inside the kernel, so there is no trip back
to the host on every iteration.

Two inputs drive the fit and are the same for every restart:

- **`f_obs`** — the observed f-statistics, a vector of length `npair`.
- **`qinv`** — the inverse of the covariance matrix of those statistics, an
  `npair × npair` matrix stored column-major. It weights the least-squares fit so
  that noisier statistics count for less.

The final fitted edge lengths and the fitted statistics are recovered separately,
once, at the winning weights (see section 7).

---

## 3. The per-thread theta cap: `kMaxThetaDev`

| Constant | Value | What it's for |
|---|---|---|
| `kMaxThetaDev` | `16` | The maximum number of admixture mixing weights a single fit can hold. |

Each restart runs on one GPU thread and keeps its working copy of the mixing weights
`theta` — plus three forward-difference perturbation copies used to estimate
derivatives — in **fixed-size per-thread stack arrays** of this length. A fixed size
is what lets these arrays live in fast per-thread memory instead of a heap
allocation. The consequence is a hard precondition: a topology's admixture-node count
`nadmix` must be `<= kMaxThetaDev`.

The cap is enforced in two independent places so an oversized topology fails loudly
rather than silently overrunning the stack:

- The host launchers reject an over-cap topology *before* launching, with a clear
  thrown error.
- The kernel itself guards the over-cap path and returns a large sentinel score
  (`1e30`) instead of writing past the end of its stack arrays.

Both the host check and the device guard read this one constant, so they can never
disagree. The value `16` comfortably covers real topologies, because the number of
admixture nodes in a graph is always small.

---

## 4. `QpGraphDeviceTopo` — one topology, uploaded once

`QpGraphDeviceTopo` is the flat, GPU-friendly description of a single topology. It is
uploaded to the device **once per fit** and then read by every restart. All of its
pointer fields point into device memory; the scalars are passed by value. This is the
same "pack the index tables into device arenas once, then reuse them" pattern used
elsewhere in the library.

**Precondition:** `nadmix <= kMaxThetaDev` (see section 3), enforced by the host
launchers and re-checked by the kernel.

### Scalar dimensions

| Field | Type | Meaning |
|---|---|---|
| `npop` | `int` | Number of populations in the graph. |
| `nedge_norm` | `int` | Number of drift edges (the normalized edge count) — the length of the fitted edge-length vector. |
| `nadmix` | `int` | Number of admixture (mixing) nodes — the number of mixing weights `theta`. Must be `<= kMaxThetaDev`. |
| `npair` | `int` | Number of centered f-statistic pairs — the length of `f_obs` and the side of the `qinv` matrix. |
| `npath` | `int` | Number of paths through the graph used when building the path weights. |
| `base_leaf` | `int` | The base leaf index used to interpret the path tables. |
| `n_pe` | `int` | Length of the path-edge index tables (`pe_edge`, `pe_leaf`, `pe_path`). |
| `n_pae` | `int` | Length of the path-admixedge index tables (`pae_path`, `pae_admixedge`). |
| `constrained` | `int` | `1` selects the box-constrained (non-negative least squares) edge solve, which forbids negative edge lengths; otherwise an unconstrained solve is used. |
| `fudge` | `double` | A small ridge regularizer added to the solve, scaled by the trace of the coefficient matrix. This matches the reference `diag` term[^at2] and keeps the solve stable when the matrix is near-singular. |

### Device pointers (the topology's index tables)

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `pwts0` | `const double*` | `nedge_norm × npop`, column-major | The base path weights that map edge lengths to fitted statistics. |
| `pe_edge` | `const int*` | `n_pe` | Which edge each path-edge entry refers to. |
| `pe_leaf` | `const int*` | `n_pe` | Which leaf each path-edge entry refers to. |
| `pe_path` | `const int*` | `n_pe` | Which path each path-edge entry belongs to. |
| `pae_path` | `const int*` | `n_pae` | Which path each path-admixedge entry belongs to. |
| `pae_admixedge` | `const int*` | `n_pae` | Which admixture edge each path-admixedge entry refers to. |
| `cmb1` | `const int*` | `npair` | The first index of each centered-column statistic pair. |
| `cmb2` | `const int*` | `npair` | The second index of each centered-column statistic pair. |

The three `pe_*` tables together describe, for each path, which edges and leaves it
passes through; the two `pae_*` tables describe how admixture edges enter each path.
The `cmb1`/`cmb2` pair encodes which columns are differenced to form each centered
statistic.

---

## 5. `ScratchLayout` and `make_layout` — the per-thread scratch slab

A single restart needs a fair amount of temporary working memory: the expanded path
weights, the coefficient matrices for the edge-length solve, the residuals, and the
scratch buffers for the least-squares routine. Rather than allocate these
individually, each thread is given one contiguous slab of memory, and
`ScratchLayout` records the byte offset (measured in elements) of each named
sub-array inside that slab.

### `ScratchLayout` fields

All sub-arrays are `double` except the three int sub-arrays. The two `*_total` fields
give the slab size per thread.

| Field group | Fields | Role |
|---|---|---|
| Double sub-array offsets | `pwts_c`, `ppwts`, `Wm`, `cc`, `ccs`, `sc`, `q1`, `bl`, `res`, `path_w`, `qf` | Working buffers for the path-weight expansion, the coefficient matrices, the solved edge lengths (`bl`), the residual (`res`), and the fitted statistics. |
| Least-squares / solve scratch (double) | `nn_w`, `nn_Ap`, `nn_qp`, `nn_z`, `nn_lu`, `nn_y` | Scratch used by the (non-negative) least-squares edge solve, including its own LU factorization buffers. |
| Total doubles | `dbl_total` | Total number of `double`s one thread needs. |
| Int sub-array offsets | `nn_P`, `nn_piv`, `nn_pass` | The active-set / pivot / pass bookkeeping arrays for the solve. |
| Total ints | `int_total` | Total number of `int`s one thread needs. |

### `make_layout(npop, nedge, npair, npath)`

`make_layout` computes a `ScratchLayout` for a topology of the given dimensions. It is
marked `STEPPE_HD`, meaning the identical function is compiled for both host and
device: the **host** calls it to size the slab it allocates, and the **kernel** calls
it to reconstruct the same offsets for its topology. Because both sides run the same
code, the offsets can never drift apart.

The function simply walks the sub-arrays in a fixed order, assigning each one the next
run of elements and advancing a running offset. The sizes are:

| Sub-array | Size (in elements) |
|---|---|
| `pwts_c` | `nedge × (npop − 1)` |
| `ppwts` | `npair × nedge` |
| `Wm` | `npair × nedge` |
| `cc` | `nedge × nedge` |
| `ccs` | `nedge × nedge` |
| `sc` | `nedge` |
| `q1` | `nedge` |
| `bl` | `nedge` |
| `res` | `npair` |
| `path_w` | `npath` |
| `qf` | `npair` |
| `nn_w` | `nedge` |
| `nn_Ap` | `nedge × nedge` |
| `nn_qp` | `nedge` |
| `nn_z` | `nedge` |
| `nn_lu` | `nedge × nedge` |
| `nn_y` | `nedge` |
| `nn_P` (int) | `nedge` |
| `nn_piv` (int) | `nedge` |
| `nn_pass` (int) | `nedge` |

`dbl_total` is the sum of all the double sub-array sizes and `int_total` the sum of
the int sub-array sizes. When many topologies of different sizes are fit together
(section 9), the caller runs `make_layout` at the **largest** dimensions across the
batch and gives every thread a slab of that batch-maximum size, so one uniform slab
size serves every topology.

---

## 6. `launch_qpgraph_fleet` — multistart fleet for a single topology

```
void launch_qpgraph_fleet(const QpGraphDeviceTopo& topo, int numstart, int maxit,
                          double tol, const double* d_fobs, const double* d_qinv,
                          double* d_out_theta, double* d_out_score, cudaStream_t stream);
```

Launches the fleet for one topology: `numstart` restarts, **one GPU thread each**,
with the whole multistart optimization — up to `maxit` projected-Newton iterations per
restart, converged to tolerance `tol` — running entirely inside the kernel. There is
no host-side objective evaluation per iteration; the fit is GPU-bound.

**Inputs**

| Argument | Meaning |
|---|---|
| `topo` | The single-topology device description (section 4). |
| `numstart` | Number of random restarts (one thread per restart). |
| `maxit` | Maximum projected-Newton iterations per restart. |
| `tol` | Convergence tolerance for a restart. |
| `d_fobs` | The observed f-statistics, `npair`, device-resident. |
| `d_qinv` | The inverse covariance, `npair × npair` column-major, device-resident. |
| `stream` | The CUDA stream to launch on. |

**Outputs (one entry per restart)**

| Argument | Shape | Meaning |
|---|---|---|
| `d_out_theta` | `numstart × nadmix` | Each restart's converged mixing weights. |
| `d_out_score` | `numstart` | Each restart's final fit score (`1e30` marks a failed or over-cap restart). |

The launcher does **not** pick the winner. Choosing the best restart, refining the
bracket, and recovering the final edge lengths are all done on the host afterward,
working over these small per-restart arrays. The follow-up that recovers the fitted
quantities at the chosen restart is section 7.

---

## 7. `launch_qpgraph_eval_at_theta` — evaluate and export at a chosen theta

```
void launch_qpgraph_eval_at_theta(const QpGraphDeviceTopo& topo, const double* d_theta,
                                  const double* d_fobs, const double* d_qinv,
                                  double* d_out_bl, double* d_out_f3, double* d_out_score,
                                  cudaStream_t stream);
```

Runs the *identical* scoring computation the fleet uses, but exactly **once**, at a
given fixed set of mixing weights `d_theta` — normally the weights of the host-chosen
best restart from section 6. Its job is to export the fitted quantities that the fleet
does not bother to keep. It runs single-threaded in native double precision, because
this final scoring is the numerically delicate part and is deliberately carved out to
use full-precision arithmetic.

**Inputs**

| Argument | Meaning |
|---|---|
| `topo` | The single-topology device description (section 4). |
| `d_theta` | The fixed mixing weights to evaluate at (the winning restart's weights), device-resident. |
| `d_fobs`, `d_qinv` | Same observed statistics and inverse covariance as the fleet, device-resident. |

**Outputs**

| Argument | Shape | Meaning |
|---|---|---|
| `d_out_bl` | `nedge_norm` | The fitted drift edge lengths. |
| `d_out_f3` | `npair` | The fitted f-statistics (equal to `f_obs − residual`). |
| `d_out_score` | `1` | The fit score at these weights. |

If the covariance turns out to be non-positive-definite the score comes back as the
`1e30` sentinel, and the host maps that to a "non-SPD covariance" error. This launcher
replaces what used to be a re-evaluation of the objective back on the host at the
winning weights, moving that final step onto the device.

---

## 8. `QpGraphDeviceTopoView` — one topology inside a packed batch

When many *different* topologies are fit together, uploading each one as its own
`QpGraphDeviceTopo` would mean many separate buffers. Instead, all the topologies'
index tables are concatenated into a handful of single packed device buffers, and each
topology is described by a `QpGraphDeviceTopoView`: the same scalar dimensions as
`QpGraphDeviceTopo`, but with the pointer fields replaced by **base offsets** (element
counts) into the packed buffers. The kernel reconstructs a full `QpGraphDeviceTopo` for
a given topology by adding these offsets to the shared base device pointers. This lets
any number of heterogeneous topologies live in one set of buffers plus one index
table.

**Precondition:** `nadmix <= kMaxThetaDev` (the same per-thread cap as in section 3),
enforced by the batch host launcher and re-checked by the kernel.

### Scalars (same meaning as `QpGraphDeviceTopo`)

`npop`, `nedge_norm`, `nadmix`, `npair`, `npath`, `base_leaf`, `n_pe`, `n_pae`,
`constrained`, `fudge` — see section 4.

### Offsets into the packed buffers

| Field | Type | Points into | Length of this topology's slice |
|---|---|---|---|
| `off_pwts0` | `long` | `d_pwts0` | `nedge_norm × npop` |
| `off_pe` | `long` | `d_pe_edge`, `d_pe_leaf`, `d_pe_path` | `n_pe` each |
| `off_pae` | `long` | `d_pae_path`, `d_pae_admixedge` | `n_pae` each |
| `off_cmb` | `long` | `d_cmb1`, `d_cmb2` | `npair` each |

Each offset is an element index, not a byte offset, so the kernel indexes the packed
buffer directly.

---

## 9. `launch_qpgraph_fleet_batch` — fitting many topologies in one launch

```
void launch_qpgraph_fleet_batch(const QpGraphDeviceTopoView* d_views, int ntopo,
                                int numstart, int maxit, double tol, int dbl_per_thread,
                                int int_per_thread, const double* d_pwts0,
                                const int* d_pe_edge, const int* d_pe_leaf, const int* d_pe_path,
                                const int* d_pae_path, const int* d_pae_admixedge,
                                const int* d_cmb1, const int* d_cmb2,
                                const double* d_fobs, const double* d_qinv,
                                double* d_g_dbl, int* d_g_int, double* d_out_score,
                                cudaStream_t stream);
```

Fits **all `ntopo` packed topologies in a single kernel launch**. This is the path
used when searching over many candidate topologies at once. The launch grid is
flattened over a combined `(topology, restart)` axis: each thread's instance index is
`inst = blockIdx * blockDim + threadIdx`, and from that the kernel derives
`topo_id = inst / numstart` and `restart = inst % numstart`. Every thread reads the
same resident `d_fobs` / `d_qinv`, because a batch is bound to one fixed population set
and therefore one shared set of observed statistics.

**Per-thread scratch.** Because the topologies differ in size, the caller computes one
batch-maximum layout with `make_layout` (section 5) at the largest dimensions in the
batch and passes the resulting per-thread sizes as `dbl_per_thread` and
`int_per_thread`. `d_g_dbl` and `d_g_int` are the global scratch slabs, sized for all
threads at that per-thread stride.

**Tree topologies.** A topology with zero admixture nodes has no mixing weights to
search over, so it does a single objective evaluation instead of the multistart loop —
there is no separate host special case for it.

**Inputs**

| Argument | Meaning |
|---|---|
| `d_views` | Array of `ntopo` `QpGraphDeviceTopoView` descriptors. |
| `ntopo` | Number of topologies in the batch. |
| `numstart`, `maxit`, `tol` | Restart count, iteration cap, and tolerance (as in section 6). |
| `dbl_per_thread`, `int_per_thread` | The batch-maximum per-thread scratch sizes. |
| `d_pwts0`, `d_pe_*`, `d_pae_*`, `d_cmb1`, `d_cmb2` | The packed index-table buffers the views index into. |
| `d_fobs`, `d_qinv` | The shared observed statistics and inverse covariance, device-resident. |
| `d_g_dbl`, `d_g_int` | The global double and int scratch slabs. |
| `stream` | The CUDA stream to launch on. |

**Output**

| Argument | Shape | Meaning |
|---|---|---|
| `d_out_score` | `ntopo × numstart` | The score for every `(topology, restart)` pair. |

The launcher only produces raw scores. Reducing them to each topology's best restart,
and then to the single best topology across the whole batch (the global argmin), is a
plain reduction done afterward on the host — not part of the fit.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

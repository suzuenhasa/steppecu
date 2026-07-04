# `qpgraph_objective.hpp` reference

## 1. Purpose

`src/core/qpadm/qpgraph_objective.hpp` holds the host (CPU) reference
implementation of the qpGraph fit objective — the function that scores how well a
candidate admixture-graph topology, with a given set of mixture proportions,
reproduces the observed genetic statistics.

qpGraph works by proposing a graph (populations at the leaves, drift along the
edges, admixture at internal nodes), then asking: for this graph, what edge lengths
best explain the measured relationships between populations, and how large is the
leftover mismatch? The mismatch is a single number — the score. A search over
mixture proportions repeatedly calls this scorer, and the proportions that give the
lowest score are the fit.

Everything here is plain C++ using only the standard library and two small internal
headers. It serves two roles:

1. It is the **oracle** the CPU test backend runs, and the reference that every
   answer is validated against.
2. It is the exact math the GPU kernel reproduces per thread, so that a run on the
   GPU and a run on the CPU agree. The GPU version lives in a separate file and
   mirrors these same steps; this header is the readable, authoritative statement of
   what that math is.

The whole calculation reproduces the corresponding reference routine (the
`optimweightsfun` function), step for step, so that steppe's qpGraph fit
lands on the same numbers[^at2].

Four functions build up the objective, smallest to largest: `build_ppwts_2d`
(section 3) assembles a matrix of edge-weight products; `nnls_active_set`
(section 4) solves a constrained least-squares system; `opt_edge_lengths`
(section 5) uses both to find the best-fit edge lengths; and `qpgraph_score`
(section 6) ties everything together into the single score a caller asks for.

---

## 2. The objective, end to end

The score is computed in five stages. Reading them in order explains what every
later function is for.

1. **Per-edge leaf weights.** For the given mixture proportions (called `theta`),
   `fill_pwts_centered` produces a matrix `pwts_c` that says how much each graph
   edge contributes to the drift reaching each population. It is *centered* on one
   chosen base population and the base column is dropped, leaving one column per
   non-base population. Its shape is `nedge_norm × (npop − 1)`: one row per fittable
   (drift) edge, one column per non-base population.

2. **Pairwise edge products.** The observed statistics are `f3` values, one for each
   pair of non-base populations. Stage 2 turns the per-population edge weights into
   per-*pair* edge weights: for pair `k` (populations `cmb1[k]` and `cmb2[k]`) and
   edge `e`, multiply that edge's weight for the first population by its weight for
   the second. The result is `ppwts_2d`, shaped `npair × nedge`. This is
   `build_ppwts_2d` (section 3).

3. **Best-fit edge lengths.** Given `ppwts_2d`, the observed statistics `f_obs`, and
   a weight matrix `ppinv`, find the edge lengths `bl` that best explain the
   observations in a generalized-least-squares sense. This is `opt_edge_lengths`
   (section 5). In the mode steppe uses, edge lengths are held non-negative, which is
   why a constrained solver (section 4) is needed.

4. **Model-predicted statistics.** Multiply `ppwts_2d` by the fitted edge lengths to
   get the statistics the graph predicts: `f3_fit = ppwts_2d · bl`.

5. **Weighted mismatch.** The score is the weighted squared residual between what was
   observed and what the graph predicts:

   ```
   score = (f_obs − f3_fit)ᵀ · ppinv · (f_obs − f3_fit)
   ```

   `ppinv` is the inverse of the estimated covariance of the observed statistics, so
   the residual is weighted by how precisely each statistic was measured. A smaller
   score means a better-fitting graph.

### Inputs shared across the functions

| Name | Shape | Meaning |
|---|---|---|
| `theta` | length `nadmix` | The mixture proportions at the admixture nodes — the free parameters an outer search varies. |
| `f_obs` | length `npair` | The observed `f3` statistics, one per population pair. |
| `ppinv` | `npair × npair` | The weight (inverse-covariance) matrix for the generalized least squares. Symmetric. |
| `pwts_c` | `nedge_norm × (npop−1)` | Per-edge, per-non-base-population drift weights for a given `theta` (from `fill_pwts_centered`). |
| `ppwts_2d` | `npair × nedge` | Per-pair edge-weight products (from `build_ppwts_2d`). |
| `bl` | length `nedge` | The fitted edge (drift) lengths — the output of `opt_edge_lengths`. |
| `fudge` | scalar | A small ridge factor (see section 5) that stabilizes the edge-length solve. |

`npair` is the number of population pairs (choose(`npop`, 2)); `nedge` (the code
calls it `nedge_norm`) is the number of fittable drift edges.

### Matrix storage

Every matrix here is stored **column-major** and flat in a `std::vector<double>` —
element (row `i`, column `j`) of a matrix with `nrows` rows lives at index
`i + nrows·j`. This is the same layout the linear-algebra libraries expect, so the
host code and the GPU code index memory identically. `ppwts_2d`, for instance, is
`npair × nedge`, so its element for pair `k` and edge `e` is at `k + npair·e`.

---

## 3. `build_ppwts_2d` — the pair-weight matrix

```cpp
void build_ppwts_2d(const QpGraphModel& m,
                    const std::vector<double>& pwts_c,
                    std::vector<double>& ppwts);
```

Builds the `npair × nedge` matrix of edge-weight products (stage 2 in section 2).
For every population pair `k` and every edge `e`, it looks up that edge's weight for
the pair's first population (`cmb1[k]`) and its second population (`cmb2[k]`) inside
`pwts_c`, multiplies them, and stores the product at `ppwts[k + npair·e]`.

`cmb1` and `cmb2` come from the parsed graph model and list the two population
columns that make up each pair. The output vector `ppwts` is sized and zero-filled
inside the function, so the caller does not need to pre-size it.

A pair where `cmb1[k] == cmb2[k]` is allowed — it corresponds to the statistic
relating the base population to a single other population (that population's variance
term), and simply squares that population's edge weight.

---

## 4. `nnls_active_set` — non-negative least squares

```cpp
bool nnls_active_set(const std::vector<double>& A,
                     const std::vector<double>& q,
                     int n, std::vector<double>& bl);
```

Solves the constrained least-squares problem

```
minimize   ½·blᵀ·A·bl − blᵀ·q     subject to   bl ≥ 0
```

where `A` is an `n × n` symmetric positive-definite matrix (column-major) and `q`
is a length-`n` vector. It returns the solution in `bl`, with every entry ≥ 0.

This is the constrained mode `opt_edge_lengths` uses when edge lengths must not go
negative. It exists because the plain (unconstrained) solve was not enough: at least
one reference graph has a true edge length sitting exactly at the zero boundary, and
only a solver that respects the `≥ 0` constraint lands on the right answer there.

### The method

It is a Lawson–Hanson active-set solver. It keeps a *passive set* of variables that
are currently allowed to be nonzero (free), while all others are pinned at zero. The
outer loop checks the optimality (KKT) conditions: for a free variable the gradient
should be zero, and for a pinned variable the gradient should point in the direction
that would only make it negative. Each outer step frees the pinned variable whose
gradient is most positive — the one most eager to become nonzero — then re-solves.

The inner loop solves the unconstrained least-squares system restricted to just the
passive variables. If that solution is all-positive, it is accepted. If any passive
variable came out non-positive, a ratio test steps the solution partway toward it,
drops the variable that hits zero back into the pinned set, and re-solves. When no
pinned variable wants to be freed, the KKT conditions hold and the solver returns.

### Return value and failure

Returns `true` on success. Returns `false` if the restricted linear system is
singular (the small solver reported a zero pivot); the caller treats that as a failed
solve. On success `bl` holds the non-negative solution.

### Named constants

| Name | Value | What it's for |
|---|---|---|
| `nnls_eps` | `1e-12` | The single tolerance used for three related jobs: the threshold for deciding a gradient is "positive enough" to free a variable (the KKT test), the floor below which a solved variable counts as non-positive, and the guard in the ratio-test denominator. **This value is a frozen parity literal for exact agreement with the constrained solve[^at2]. Do not change its magnitude** — the name may be kept or renamed, but `1e-12` is what reproduces the reference results. |
| `max_iter` | `3·n + 30` | The iteration cap, shared by the outer KKT loop and the inner passive re-solve loop. It scales with the number of variables (`n`) so larger systems get proportionally more steps, plus a fixed cushion of 30 for very small systems. It bounds the work so a pathological case cannot loop forever. |

---

## 5. `opt_edge_lengths` — fitting the edge lengths

```cpp
bool opt_edge_lengths(const std::vector<double>& ppwts,
                      const std::vector<double>& ppinv,
                      const std::vector<double>& f_obs,
                      int npair, int nedge,
                      double fudge, bool constrained,
                      std::vector<double>& bl);
```

Finds the edge lengths `bl` that best explain the observed statistics for a fixed set
of pair-weights `ppwts` (stage 3 in section 2). It returns `true` and writes the
length-`nedge` result into `bl`; it returns `false` if the underlying linear system is
singular.

### The steps

1. **Form the normal equations.** Build the `nedge × nedge` system matrix
   `cc = ppwtsᵀ · ppinv · ppwts` and the right-hand side `q = ppwtsᵀ · ppinv · f_obs`.
   These are the standard generalized-least-squares normal equations for the edge
   lengths, weighted by `ppinv`. (Internally the intermediate `ppinv · ppwts` is
   computed once and reused for both.)

2. **Ridge stabilization.** Add a small amount to the diagonal of `cc`:
   `diag(cc) += fudge · mean(diag(cc))`. This is proportional to the average diagonal
   magnitude, so it scales with the problem and nudges a nearly-singular system toward
   being solvable without materially changing a well-conditioned one. The `fudge`
   factor is passed in by the caller. This is the parity ridge term[^at2].

3. **Symmetric scaling to a unit diagonal.** Compute `sc = sqrt(diag(cc))` and rescale
   the system so its diagonal becomes all ones: divide `q` by `sc`, and divide each
   `cc[i,j]` by `sc[i]·sc[j]`. Solving the scaled system and then dividing the answer
   back by `sc` improves numerical conditioning. This is a Jacobi (diagonal)
   preconditioner and again mirrors the reference.

4. **Solve.** If `constrained` is true, solve the scaled system with the non-negative
   solver from section 4 (edge lengths held ≥ 0). Otherwise solve it with a plain LU
   solve, which permits negative lengths. steppe's default is the constrained mode,
   matching the parity golden case[^at2].

5. **Unscale.** Divide the scaled solution back by `sc` to recover the true edge
   lengths and write them into `bl`.

### The `constrained` flag

`constrained == true` is the mode steppe uses to match the reference and is why the
non-negative solver exists — some real graphs have an edge length pinned at exactly
zero, which only the constrained path reproduces. The unconstrained path (`false`) is
available but permits negative lengths and is not the reference default.

---

## 6. `qpgraph_score` — the full objective

```cpp
double qpgraph_score(const QpGraphModel& m, const double* theta,
                     const std::vector<double>& f_obs,
                     const std::vector<double>& ppinv,
                     double fudge, bool constrained,
                     std::vector<double>* out_bl  = nullptr,
                     std::vector<double>* out_fit = nullptr);
```

This is the function an outer search calls. Given a graph `m` and mixture proportions
`theta`, it runs all five stages from section 2 and returns the single score. It is
the `optimweightsfun` objective[^at2].

Internally it: fills the centered per-edge weights for `theta`, builds `ppwts_2d`,
fits the edge lengths with `opt_edge_lengths`, forms the model-predicted statistics
and their residual against `f_obs`, and returns the `ppinv`-weighted squared residual.

### Return value

Returns the score — a non-negative number where smaller is a better fit. Returns
positive infinity if the inner edge-length solve was singular. Infinity is a
deliberate signal to the optimizer that this point in the search is invalid, so it is
rejected rather than crashing the search.

### Optional outputs

Both are optional (pass `nullptr` to skip). They let a caller retrieve the fitted
details after the search has settled on a `theta`, without recomputing.

| Argument | What it returns |
|---|---|
| `out_bl` | The fitted edge lengths `bl`. It is **assigned** (resized as needed), so the caller does not pre-size it. |
| `out_fit` | The model-predicted statistics `f3_fit`, one per population pair. |

### Precondition on `out_fit`

`out_fit` is treated differently from `out_bl`. If it is non-null, **the caller must
pre-size it to `m.npair` before the call.** It is written element by element
(`(*out_fit)[k] = …`) and never resized inside the function. Passing a non-null but
too-small `out_fit` is undefined behavior. If you do not want the predicted
statistics, pass `nullptr`.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

# `ranktest.cpp` reference

## 1. Purpose

`src/core/qpadm/ranktest.cpp` builds the **popdrop table** — the "leave one source
population out" feasibility report that a qpAdm run prints alongside its main result.

A qpAdm model explains one target population as a mixture of several *source*
populations, measured against a panel of *reference* (outgroup) populations. The
popdrop table answers a practical question about that model: **is each source
actually pulling its weight?** For the full model, and then for every way of dropping
exactly one source, it refits the model on the sources that remain and records
whether the refit is still statistically sane (all mixture weights land between 0 and
1). If dropping a source leaves a perfectly good model behind, that source may be
unnecessary; if dropping it wrecks the fit, that source is load-bearing.

The important design fact about this file is what it *does not* do. It never touches
raw genotype data, never recomputes the f4 statistics, and never re-runs the block
jackknife. Everything it needs — the full table of f4 estimates and the full inverse
covariance matrix — was already computed once by an earlier stage. Each dropped-source
model is produced purely by **selecting a subset of those already-computed numbers**
and refitting. This is a deliberate, exact reproduction of the `drop_pops`
routine[^at2], and it is the reason the table is cheap to produce.

The file is host-only C++ (no GPU code of its own). It hands the actual fitting math
to a compute backend through a narrow seam, so the CPU reference implementation and
the GPU implementation run through the very same code here.

---

## 2. What "popdrop" reproduces (the drop_pops contract)

The reference popdrop procedure[^at2] follows three rules, and this file follows them
exactly:

1. **Keep rows, do not re-gather.** The f4 estimates are laid out as a grid with one
   row per source population and one column per reference population. Dropping a
   source means keeping only the rows for the sources that survive — a subset of the
   existing grid, not a fresh computation.
2. **Subset the inverse covariance, do not re-invert.** Uncertainty in the f4 grid is
   carried as one large inverse-covariance matrix (called `Qinv` here) whose rows and
   columns are indexed by every (source, reference) cell of the grid. For a
   dropped-source model, the code carves out the sub-block of that existing matrix
   corresponding to the surviving cells. It does **not** build a smaller covariance
   from scratch and invert that — carving the sub-block of the already-inverted matrix
   is the parity behavior, and it gives a different (and matching) answer.
3. **Fit at the sub-model's full rank.** Each dropped-source model is fit at a rank of
   *(number of surviving sources − 1)*. See section 5 for why this specific rank
   matters and how it differs from the rank the top-level model reports.

Because both the row subset and the covariance sub-block are just index arithmetic
over data that already lives in memory, the whole table costs a handful of small
refits and no new statistics work.

---

## 3. `popdrop_feasible` — the feasibility test

`popdrop_feasible(weights)` decides whether one refit produced a usable model. It
returns `true` only when **every surviving mixture weight is between 0 and 1
inclusive, and at least one weight survived.**

The rules, in order:

- A weight equal to "not a number" (NaN) marks a *dropped* source slot. It is not a
  real constraint, so it is skipped — it neither passes nor fails the test.
- Any real weight below 0 or above 1 immediately fails the model (returns `false`). A
  negative weight or a weight over 1 is not a physically meaningful ancestry
  proportion.
- If, after skipping the NaN slots, no real weight was seen at all, the model also
  fails (returns `false`) — there is nothing to accept.

This matches the parity feasibility predicate for the "weights must be
non-negative" setting[^at2]. A model where only a single source survives (its one weight
equal to 1) is trivially feasible.

---

## 4. `reduce_rows` — building the reduced problem

`reduce_rows` is the internal routine that turns the full f4 grid and full inverse
covariance into the smaller versions a dropped-source model needs. It is where all the
index arithmetic lives, so it is worth understanding the layout it works over.

### The memory layout

- The f4 grid is stored flat, one row after another (row-major). With `nr` reference
  populations, the cell for source row `i` and reference column `j` sits at flat
  position `j + nr*i`. The whole grid has `m = nl*nr` cells, where `nl` is the number
  of sources.
- The inverse covariance `Qinv` is an `m × m` matrix, also stored flat. Its entry for
  the pair of grid cells `(a, b)` sits at `a + m*b`.

### What it produces

Given `surv` — the list of surviving source row indices, in source order — it writes:

- `x_reduced`: the f4 grid with only the surviving rows, so its size is
  `(number of survivors) × nr`.
- `cov_reduced`: the corresponding sub-block of `Qinv`, of size `m_red × m_red` where
  `m_red = (number of survivors) × nr`.

### How the gather works

The routine does two passes:

1. **Row gather.** For each surviving source (call its position in the reduced grid
   `ii`, and its original row `i = surv[ii]`) and each reference column `j`, it copies
   the full cell at `j + nr*i` into the reduced cell at `j + nr*ii`. In the *same* loop
   it records `ind[j + nr*ii] = j + nr*i` — a lookup table mapping each reduced cell
   back to the full cell it came from. Keeping the copy and the lookup table in one
   fused loop is intentional: they share the exact same source and destination index
   expressions, so an edit to one can never silently disagree with the other.
2. **Covariance sub-block gather.** For each pair of reduced cells `(a, b)`, it copies
   `Qinv[ind[a] + m_full*ind[b]]` into `cov_reduced.Qinv[a + m_red*b]`, using the
   lookup table to translate reduced positions back to full positions. A parallel `Q`
   matrix (the non-inverted covariance sub-block) is carried along when present, for
   later rank observability.

### A frozen correctness detail: the wide multiply

In the covariance gather, the source index `ind[a] + m_full*ind[b]` multiplies two
values that can both be large on a big model. The multiplication is deliberately done
in a wide unsigned integer type (`size_t`) so that the product cannot overflow a
32-bit `int` and wrap to a wrong address. This widening is a correctness invariant, not
a style choice — it must be preserved. The loop is also written so the source-row term
and destination-row term are computed once per row and reused across the inner column
loop, rather than recomputed on every iteration.

---

## 5. `popdrop_one` — fitting one dropped-source row

`popdrop_one` produces a single line of the table: either the full model, or one model
with exactly one source dropped. It assembles the reduced problem, fits it, and records
the result.

### The pattern and the weight column

- **`pat`** is a text string of length `nl_full` (the original source count), all `'0'`
  characters, with a `'1'` at the position of the dropped source. The full model, where
  nothing is dropped, is the all-zeros string.
- **`wt`** is the count of dropped sources — 0 for the full model, 1 for a
  single-source drop. This mirrors the parity popdrop "weight" column[^at2], which is a
  *count of dropped populations*, **not** a mixture weight. This is a common point of
  confusion: the mixture weights live in the separate `weight` field described below.

### The fitted rank — a subtle, correctness-critical choice

Every dropped-source model here is fit at rank `r_fit = (number of surviving sources) − 1`.

This is the key non-obvious decision in the file. There are two different "ranks" in
play and they must not be mixed up:

| Quantity | What it is | Where it belongs |
|---|---|---|
| **Fitted rank** = survivors − 1 | The default rank a qpAdm fit uses to actually solve for the weights. | The popdrop table's `f4rank` column, and the rank at which each popdrop row's weights, chi-squared, and p-value are computed. |
| **Rank decision** | The smallest rank the sweep could *not* reject as too simple — a model-selection outcome. | The top-level result's `f4rank`, reported once for the whole run — **not** the popdrop column. |

The parity popdrop table reports the fitted rank, not the rank decision[^at2]. An earlier
version of this code mistakenly used the rank-decision value from the sweep. On the
small two-source reference datasets the two happened to be equal, so the bug was
invisible. On models with three or more sources they diverged (a rank decision of 0 or
1 against a fitted rank of 2), which fit the full popdrop row's weights, chi-squared,
and feasibility at the wrong rank. Using `r_fit = survivors − 1` directly is the fix,
and it must stay that way.

### How the reported numbers are obtained

- The rank sweep is still run on the reduced problem (it is the shared seam that
  computes chi-squared, degrees of freedom, and p-value across ranks). The row then
  reads those three numbers **at the fitted rank** `r_fit`. If `r_fit` falls outside
  the sweep's returned arrays, the code falls back to computing the degrees of freedom
  directly from the single shared formula and reports zero chi-squared/p.
- The **degrees of freedom** come from the one canonical formula, *(survivors − rank) ×
  (references − rank)*, so the popdrop rows and the rest of qpAdm can never use two
  different definitions.
- The **per-source weights** are solved on the reduced problem at the fitted rank. The
  result is written into a full-length array (one slot per original source): each
  surviving source gets its solved weight scattered back into its original position,
  and every dropped slot is left as NaN. Marking dropped slots with NaN is what lets
  `popdrop_feasible` (section 3) tell "dropped" apart from "genuinely zero." If the
  weight solve fails or returns an unexpected length, the weights stay all-NaN.
- Finally `feasible` is set by running `popdrop_feasible` over that weight array.

---

## 6. `run_popdrop` — the whole table and its row order

`run_popdrop` is the public entry point. It emits the rows in a fixed order that
matches the reference[^at2]:

1. **The full model first** — all sources kept, pattern all-zeros.
2. **Then one row per single-source drop**, iterating the dropped source from the
   *highest* index down to the lowest. For a two-source model this yields the pattern
   `"01"` (drop source 1) before `"10"` (drop source 0). Matching this ordering is what
   makes steppe's table line up row-for-row with the reference.

Two edge cases are handled explicitly:

- **A single-source model** (`nl_full == 1`) emits only the full row and no drops.
  Dropping the one and only source would leave zero rows, which cannot be fit —
  the reference likewise does not treat the empty set as a fittable model.
- **A model with no sources** (`nl_full <= 0`) returns an empty table.

Every row is produced by `popdrop_one`, so the full model and the dropped-source models
share one code path and one set of rules.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

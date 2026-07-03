# `qpfstats_jackknife.hpp` reference

## 1. Purpose

`src/core/internal/qpfstats_jackknife.hpp` holds the two reference building
blocks that turn per-block statistics into a single jackknife estimate for
qpfstats. Both are small, header-only host functions written in pure C++20 with
no CUDA.

steppe estimates uncertainty with a **block jackknife**: it splits the SNPs into
blocks along the genome, and for each block it can recompute an estimate with
that one block left out. The spread of those "leave-one-out" estimates is what
feeds the standard errors. These two functions compute the central estimate that
the leave-one-out machinery is built around, one for each of the two quantities
qpfstats needs:

- `matrix_jackknife_est_col` — the **global per-combination estimate** (the
  target value for one column of the statistics output).
- `f2blocks_pair_est` — the **per-pair recentering estimate** (the value used to
  recenter one f2 pair series).

Both reproduce a specific ADMIXTOOLS 2 routine exactly, and both accumulate in
`long double` on purpose (see section 3). The file lives in `core/internal`
because it has no GPU dependency and is shared by two callers that must never
disagree (see section 2).

---

## 2. One definition shared by the CPU oracle and the GPU kernel

These helpers used to be file-local to `src/core/stats/qpfstats.cpp`. They were
lifted into this header so there is exactly **one** copy of the math, used by
both:

- the **CPU reference backend** — the `long double` oracle that qpfstats results
  are validated against; and
- the **qpfstats driver**, which is CUDA-free itself but drives the on-device
  jackknife.

Separately, the CUDA on-device jackknife kernel
(`qpfstats_jackknife_kernel.cu`) is written to reproduce the same math on the
GPU, and its output is diffed against these functions. So a single edit here
moves the reference for every path at once, and there is no second copy that
could quietly drift out of sync.

Each function corresponds to a named ADMIXTOOLS 2 routine:

| This function | ADMIXTOOLS 2 equivalent | What it produces |
|---|---|---|
| `matrix_jackknife_est_col` | `matrix_jackknife_est_full`, one population-combination column | the global per-combination estimate (the target the results are centered on) |
| `f2blocks_pair_est` | `f2(array)$est` for one pair series | the recentering value for one f2 pair |

---

## 3. Long double accumulation and the GPU parity contract

Every running sum in this file is a `long double`, even though the inputs and the
returned value are ordinary `double`. This is deliberate and should not be
"simplified" back to `double`.

The reason is that a jackknife estimate is a **leave-one-out difference**: it
subtracts one block's contribution from a whole-dataset total (`tot - numer*rel`
below). Subtracting two nearly equal numbers throws away low-order bits — the
classic cancellation problem. steppe applies extra precision exactly where
cancellation happens (this reduction) rather than everywhere; the heavy
matrix-multiply stages elsewhere in steppe use a faster emulated arithmetic, but
a cancellation-prone reduction like this one gets the extra guard digits of
`long double` instead.

The GPU kernel cannot use `long double` (GPUs have no such type), so it
reproduces this same computation in **native FP64** with two rules that keep it
bit-comparable to this reference:

- **Same operand order.** The loops here walk the blocks in ascending block
  index `b`. Floating-point addition is not associative, so the order in which
  the partial sums are added changes the last bits of the result. The GPU kernel
  sums in the identical ascending-`b` order.
- **Only the carry precision differs.** The single intended difference between
  this host reference and the GPU kernel is that the host carries the running
  sums in `long double` while the GPU carries them in FP64. Everything else —
  which values are included, the order, the formula — is identical.

Both the host reference and the GPU kernel are gated on the 9-population golden
test, which is what proves the two stay in agreement.

---

## 4. `matrix_jackknife_est_col` — the global per-combination estimate

```
double matrix_jackknife_est_col(const double* numer, const double* cnt,
                                int c, int n_block)
```

Computes the global jackknife estimate for one population combination — one
column of the statistics output. This is the value the final results are
centered on.

### Inputs and layout

- `numer` and `cnt` are both **row-major** arrays of shape
  `[n_population_combinations × n_block]`. The entry for combination `c`,
  block `b` lives at index `c * n_block + b`. This is the natural output shape of
  the upstream D-statistic computation.
- `numer[c, b]` is the per-block mean value for that combination (a sum divided
  by its count), and is `NaN` where the block is invalid.
- `cnt[c, b]` is that block's count (its weight).
- `c` selects the column; `n_block` is the number of blocks.

### What it computes

For the chosen column, over the blocks where `numer` is finite and `cnt > 0`:

1. **Total weight.** `sum_n_all` = sum of `cnt` over *all* blocks. If it is not
   positive, return `NaN`.
2. **Weighted total.** `tot` = weighted mean of `numer` over the valid blocks,
   `Σ(numer·cnt) / Σ(cnt)`. If the valid-block weight is not positive,
   return `NaN`.
3. **Leave-one-out per block.** For each valid block, with relative weight
   `rel = cnt / sum_n_all` (note: relative to the *all-block* total, not just the
   valid ones):
   - Skip the block if `rel ≥ 1`, because then `1 − rel` is zero and the
     leave-one-out value would not be finite.
   - `loo = (tot − numer·rel) / (1 − rel)`.
   - Apply the finiteness gate described below; skip non-finite `loo`.
   - Accumulate the running sums it needs: `Σloo`, `Σ(1−rel)`, `Σ(loo·(1−rel))`,
     `Σ(loo·cnt)`, `Σcnt`, and a count `n_finite` of surviving blocks.
4. **Guard.** If no block survived, or `Σ(1−rel)` is not positive, or the
   surviving-count sum is not positive, return `NaN`.
5. **Combine.** With `tot2 = Σ(loo·(1−rel)) / Σ(1−rel)` and
   `weighted_loo_mean = Σ(loo·cnt) / Σcnt`, the estimate is

   ```
   y = n_finite · tot2 − Σloo + weighted_loo_mean
   ```

   This is the ADMIXTOOLS 2 `matrix_jackknife_est_full` formula for the column.

### The deliberate narrow-to-`double` finiteness gate

The finiteness check on `loo` is done by narrowing it to `double` first:

```cpp
if (!std::isfinite(static_cast<double>(loo))) continue;
```

This is intentional and load-bearing for parity. ADMIXTOOLS 2 holds `loo` as an
R double, so a value that is finite as a `long double` but overflows to infinity
when stored as a `double` must be treated as non-finite here too — otherwise a
*different set of blocks* would survive and the result could diverge from the
oracle. The test therefore runs in `double`, not the `long double`
`std::isfinite` overload. Only the survivor test is narrowed; the running sums
themselves stay in `long double`.

---

## 5. `f2blocks_pair_est` — the per-pair recentering estimate

```
double f2blocks_pair_est(const std::vector<double>& arr,
                         const std::vector<int>& bl)
```

Computes the jackknife estimate for one f2 pair series — the value used to
recenter that pair. This reproduces ADMIXTOOLS 2's `f2(array)$est`.

### Inputs

- `arr` — the per-block estimate vector, one value per block. Non-finite entries
  mark blocks to skip.
- `bl` — the block lengths, used as weights. By contract `bl` has the same length
  as `arr`, and the two are indexed in lock-step. A debug-only assertion
  (`STEPPE_ASSERT`) checks that lengths match; it compiles out under `NDEBUG`, so
  the release hot path is unchanged. The block count is taken from `arr.size()`.

### What it computes

Over the blocks where `arr[b]` is finite (skipping non-finite ones throughout,
matching R's `na.rm = TRUE`):

1. **Total weight.** `sum_bl` = sum of `bl` over finite blocks. If it is not
   positive, return `0`.
2. **Weighted total.** `tot` = `Σ(arr·bl) / sum_bl` — the block-length-weighted
   mean of `arr`.
3. **Leave-one-out mean.** For each finite block, with `rel = bl / sum_bl`:
   - Skip if `rel ≥ 1` (again, `1 − rel` would be zero).
   - `loo = (tot − arr·rel) / (1 − rel)`.
   - Weight it by `w = 1 − 1/h`, where `h = sum_bl / bl`.
   - Accumulate `num += loo·w` and `den += w`.
4. **Result.** If `den` is not positive, return `0`; otherwise return
   `num / den` — the weighted mean of the leave-one-out values with weights
   `1 − 1/h`.

Unlike section 4, this function uses the simpler
`weighted.mean(loo, 1 − 1/h)` combine (not the three-term formula) and it checks
finiteness on the input block estimate `arr[b]` up front rather than narrowing
`loo`.

---

## 6. Return-value and edge-case contract

The two functions differ in what they return for degenerate inputs, so the
callers must handle each correctly:

| Situation | `matrix_jackknife_est_col` | `f2blocks_pair_est` |
|---|---|---|
| Total weight not positive (all blocks empty/invalid) | returns `NaN` | returns `0` |
| No block survives the leave-one-out step | returns `NaN` | returns `0` |
| A block would leave nothing out (`rel ≥ 1`, only one weighted block) | that block is skipped | that block is skipped |
| Non-finite input block | skipped (finiteness on `numer`, then on narrowed `loo`) | skipped (finiteness on `arr[b]`) |

Both are marked `[[nodiscard]]`, so a caller cannot accidentally drop the
returned estimate. Both are pure functions of their inputs — no global state, no
allocation of persistent buffers, no CUDA — which is what lets them serve as the
single trusted reference for the GPU path.

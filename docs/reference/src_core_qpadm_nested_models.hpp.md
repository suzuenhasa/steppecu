# `nested_models.hpp` reference

## 1. Purpose

`src/core/qpadm/nested_models.hpp` declares the small piece of the qpAdm fit
that turns raw mixture weights into **uncertainty numbers**: a standard error
(SE) for each estimated admixture weight, and a z-score (the weight divided by
its SE) alongside it.

It exposes exactly two things:

- `SeResult` — the little bundle of results (the per-weight SEs and z-scores).
- `se_from_loo` — the one function that produces that bundle.

The header itself is **host-only and CUDA-free**. It is pure orchestration: it
declares a function that decides *what* to compute and hands the heavy numerical
work off to the compute backend (CPU or GPU) through a single method on that
backend. No GPU code, and no math beyond the final divide, lives here.

---

## 2. How the standard errors are computed (the block jackknife)

steppe measures how uncertain each admixture weight is with a **block
jackknife**, the `get_weights_covariance` routine[^at2].

The idea in plain terms:

1. The genome is already partitioned into blocks (chunks of SNPs). Call the
   number of blocks `nb`.
2. The mixture weights are re-estimated `nb` times. Each re-estimation leaves
   **one** block out of the data — the "leave-one-out" or "delete-one"
   replicates. Each replicate produces its own full set of weights.
3. Those per-replicate weight vectors are stacked into a matrix (one row or
   column per replicate). How much the weights wobble from replicate to
   replicate is what the SE measures: a weight that barely moves when a block is
   removed is well-determined; one that swings a lot is not.
4. The spread is turned into a variance by taking the **diagonal of the sample
   covariance** of that stacked matrix, and the SE is the square root of that
   variance.

Two details matter for exact parity:

- **The full-data inverse is reused, not recomputed.** Each leave-one-out
  re-fit reuses the matrix inverse computed from the *full* dataset
  (`cov.Qinv`) rather than re-inverting the matrix for every replicate — no
  per-replicate re-inversion. This is a deliberate parity pin[^at2], not an
  approximation to "fix" later.
- **The delete-one scale factor.** Before the covariance is taken, the stacked
  weight matrix is scaled by `(nb − 1) / sqrt(nb)`. This is the standard
  delete-one jackknife scaling, and it is applied to reproduce the parity
  result[^at2].

The full-data weight vector itself is **not** changed by any of this. The
jackknife only supplies the SE; the reported weight stays the weight estimated
from all the data.

---

## 3. Where the work actually happens (the backend seam)

All of the expensive parts of section 2 — the per-block re-fits, the
`(nb − 1) / sqrt(nb)` scale, and the sample-covariance-diagonal variance
reduction — are delegated to a **single method on the compute backend**,
`se_from_wmat`. That method does everything and returns the finished, already
scaled SE array (one value per weight).

Because the whole reduction sits behind that one seam, each backend can
implement it in the way that suits it:

- **GPU (CUDA) backend.** The stacked per-replicate weight matrix stays
  **resident in GPU memory**. An on-device kernel performs the scale and the
  covariance-diagonal reduction right there. Nothing is copied back to the host
  and no reduction happens on the CPU.
- **CPU backend.** This is the test / parity oracle. It overrides `se_from_wmat`
  with a `long double` reference implementation of the same
  sample-covariance-diagonal computation, so its results can be trusted as the
  ground truth the GPU path is validated against.

`se_from_loo` itself is **backend-agnostic**. On every path it applies neither
the scale nor the variance reduction on its own — those live entirely inside
`se_from_wmat`. The only arithmetic `se_from_loo` performs directly is the final
`z = weight / se`.

---

## 4. Precision policy

The **SE reduction runs in native double precision (`Fp64`)**. This is a
deliberate carve-out: the covariance-diagonal step subtracts nearly-equal
quantities, and that kind of cancellation loses accuracy in the faster emulated
double-precision mode. So the reduction is always done in true FP64 regardless
of the mode selected elsewhere.

The underlying leave-one-out re-fits, by contrast, honor the `precision`
argument passed in — emulated double precision by default (the fast path), with
native double precision available as the same cancellation-sensitive carve-out.

---

## 5. `SeResult`

The result bundle returned by `se_from_loo`. Both vectors have one entry per
estimated weight.

| Field | Type | Meaning |
|---|---|---|
| `se` | `std::vector<double>` | The jackknife standard error of each admixture weight — the square root of the variance from section 2, already scaled. |
| `z` | `std::vector<double>` | The z-score of each weight: the full-data weight divided by its `se`. A large magnitude means the weight is many standard errors away from zero. |

---

## 6. `se_from_loo`

```cpp
[[nodiscard]] SeResult se_from_loo(ComputeBackend& be, const F4Blocks& x,
                                   const JackknifeCov& cov, int r,
                                   const QpAdmOptions& opts,
                                   const std::vector<double>& weight,
                                   const Precision& precision);
```

Computes the per-weight standard errors over the `nb` leave-one-out replicates
(the `get_weights_covariance` procedure[^at2]) and returns them together
with the z-scores. It delegates the re-fits, the `(nb − 1) / sqrt(nb)` scale,
and the variance reduction to `be.se_from_wmat`, then computes `z = weight / se`
from the returned SE array. It is marked `[[nodiscard]]` because the whole point
of calling it is the value it returns.

### Parameters

| Parameter | Type | Meaning |
|---|---|---|
| `be` | `ComputeBackend&` | The compute backend (CPU or GPU) that supplies the `se_from_wmat` seam described in section 3. |
| `x` | `const F4Blocks&` | The per-block f-statistics the re-fits are computed from. |
| `cov` | `const JackknifeCov&` | The jackknife covariance data. Notably it carries `cov.Qinv`, the **full-data inverse** that every leave-one-out replicate reuses (the parity pin from section 2). |
| `r` | `int` | The model rank — the number of admixture components / source populations. |
| `opts` | `const QpAdmOptions&` | The qpAdm options that govern the fit. |
| `weight` | `const std::vector<double>&` | The full-data weight vector. Used only as the numerator of `z = weight / se`; it is not re-estimated here. |
| `precision` | `const Precision&` | The precision policy for the underlying re-fits (emulated FP64 by default). The SE reduction itself is always native FP64, independent of this. |

### Returns

A `SeResult` whose `se` and `z` vectors each have one entry per weight (see
section 5).

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

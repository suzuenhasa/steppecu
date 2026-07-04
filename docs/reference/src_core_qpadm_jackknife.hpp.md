# `jackknife.hpp` reference

## 1. Purpose

`src/core/qpadm/jackknife.hpp` is the small driver that produces the
**block-jackknife covariance** used to put error bars on f-statistics and on the
qpAdm model fit. A block jackknife estimates uncertainty by partitioning the
genome into blocks, recomputing the statistic with each block left out in turn,
and measuring how much the answer wobbles across those leave-one-out runs.

The header exposes exactly two functions:

- `jackknife_cov` — the full covariance matrix `Q` and the inverse of a slightly
  regularized version of it. This is what the qpAdm fit needs, because the fit
  weights its residuals by that inverse.
- `jackknife_diag` — only the diagonal of `Q` (the per-item variance), for the
  per-item f-statistics (f4 and f3) that need a standard error but never invert
  the covariance.

The file contains no math and no GPU code of its own. Each function is a one-line
forwarder into a compute backend (see section 2). Its job is to be the stable,
host-side entry point that the rest of the qpAdm code calls, while the actual
numerical work happens behind the backend seam.

The header is deliberately **host-pure and CUDA-free**: it includes only the
backend interface and the precision policy, never any CUDA headers. That keeps it
cheap to include from ordinary host code that should not have to pull in the GPU
layer.

---

## 2. The backend seam

Both functions take a `ComputeBackend&` as their first argument and immediately
delegate to it:

- `jackknife_cov(be, ...)` calls `be.jackknife_cov(...)`.
- `jackknife_diag(be, ...)` calls `be.jackknife_diag(...)`.

`ComputeBackend` is an abstract interface with more than one implementation. The
CPU implementation is the reference that everything else is validated against; a
GPU implementation does the same computation on the device. This header does not
know or care which one it is handed — it just passes the call through.

This is why the header itself is free of any real logic. The whole
leave-one-out-to-covariance pipeline lives inside the backend. Splitting it this
way means the caller writes the same one line regardless of where the math runs,
and the pipeline has a single reference implementation to check against.

---

## 3. `jackknife_cov` — the full covariance and its inverse

```
JackknifeCov jackknife_cov(ComputeBackend& be, const F4Blocks& x,
                           std::span<const int> block_sizes,
                           double fudge, const Precision& precision);
```

This produces the full `m × m` covariance matrix over the `m` f-statistic entries
being fit, plus the inverse that the fit consumes.

Inside the backend, the pipeline runs in these stages[^at2]:

1. **Leave-one-out replicates.** For each block, form the statistic with that
   block removed. (These replicate values are carried alongside the input so the
   fit can reuse them later without recomputing this step.)
2. **Pseudo-values.** Turn the per-block replicates into jackknife pseudo-values —
   the weighted deviations that the covariance is built from.
3. **Raw covariance `Q`.** Average the outer product of those pseudo-values across
   blocks. The `Q` that comes out of this step is **unregularized** — this is the
   golden convention, so `Q` is stored exactly as computed with nothing added to
   it.
4. **Regularize, then invert.** Add a small ridge to the diagonal — `fudge` times
   the trace of `Q` — and invert that regularized matrix to get `Qinv`. The ridge
   is what keeps the inverse well-behaved when the covariance is nearly singular.
   `Q` itself is never modified; only the copy that gets inverted is.

The work is batched over the `m` axis, and this whole pipeline runs in **native
double precision**. It is one of the small, cancellation-prone parts of the
system that is deliberately kept in native FP64 for numerical safety rather than
the faster emulated arithmetic used for the large matrix multiplies elsewhere.
The `precision` argument is threaded through the seam for consistency with the
other backend calls, but this particular computation does not run in the emulated
mode.

The `fudge` argument is the parity ridge[^at2]. See section 6 for the output type
and its status field.

---

## 4. `jackknife_diag` — the diagonal-only variance

```
JackknifeDiag jackknife_diag(ComputeBackend& be, const F4Blocks& x,
                             std::span<const int> block_sizes,
                             const Precision& precision);
```

This computes **only the diagonal of `Q`** — the per-item variance — and nothing
else. It exists for the per-item f-statistics (f4 and f3) that need a standard
error but never form or invert the full covariance matrix.

It is the production-scale shape, and it exists to avoid an out-of-memory
failure. When a sweep enumerates tens of thousands of items or more, asking
`jackknife_cov` for the full `m × m` matrix would try to allocate something on the
order of tens of gigabytes and fall over. `jackknife_diag` sidesteps that
entirely:

- **Work is `O(m · nb)`** — for each item, a short sum over the `nb` blocks.
- **Memory is `O(m)`** — one variance number per item, never an `m × m` matrix.

Each diagonal entry is the same per-item sum-of-squares of pseudo-values that the
diagonal of `Q` would contain, computed directly without the dense matrix
formation or the inverse. It is **bit-for-bit equal** to the diagonal
`jackknife_cov` would produce, so the f4 and f3 reference results do not shift
when this path is used. The standard error is the square root of each variance;
the point estimate is read separately from the input.

Two things it deliberately does **not** do:

- **No fudge.** There is no `fudge` argument here, because a bare f-statistic
  standard error is the *unregularized* diagonal. The ridge only matters when you
  are going to invert the covariance, which this path never does.
- **No inversion.** Because nothing is inverted, this path can never report a
  non-invertible-covariance error.

Like `jackknife_cov`, this runs in native double precision.

---

## 5. The per-block weight (`block_sizes`)

Both functions take a `block_sizes` span, and its meaning is a specific,
non-obvious contract worth stating plainly:

**`block_sizes[b]` is the number of SNPs in block `b`** — the count that weights
that block in the jackknife (the `block_lengths` weight[^at2]). It is
**not** a count of how many population pairs had valid data in the block, and it
is not any pairwise-validity quantity. The caller is responsible for passing the
per-block SNP counts here.

There is one subtlety about *which* blocks these counts describe. A block in which
some population pair has no SNP that is jointly valid across both populations is a
**missing block**. steppe drops such blocks entirely before running the
jackknife[^at2] (rather than filling them with zeros, which would bias the statistic
toward zero and inflate the variance): the caller
compacts the block data down to the surviving blocks and passes the **survivor**
SNP counts here — so this driver and everything downstream see a clean survivor
set and need no further filtering. When there are no missing blocks (which is the
case for every dataset with no missing-data threshold), the survivor counts are
identical to the full block-size list and the path is byte-for-byte the same.

---

## 6. Output types

Both functions report success or failure through a status field on their return
value rather than by throwing. This "status as a value" style lets the caller
decide how to handle a degenerate covariance without exception handling.

### `JackknifeCov` (returned by `jackknife_cov`)

| Field | Meaning |
|---|---|
| `Q` | The `m × m` covariance matrix, stored **unregularized** (the golden convention). It is symmetric, so the storage layout is not significant. |
| `Qinv` | The inverse of the *regularized* covariance — the matrix with `fudge × trace(Q)` added to its diagonal, then inverted. |
| `m` | The number of f-statistic entries (the matrix dimension). |
| `status` | `Ok` on success, or a non-invertible-covariance status if the regularized `Q` could not be inverted. Reported as a value, never thrown. |

### `JackknifeDiag` (returned by `jackknife_diag`)

| Field | Meaning |
|---|---|
| `var` | The `m` per-item variances — exactly the diagonal of `Q`. The standard error of item `k` is `sqrt(var[k])`. |
| `m` | The number of items. |
| `status` | Always `Ok`. Because nothing is inverted, this path has no failure mode to report. |

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

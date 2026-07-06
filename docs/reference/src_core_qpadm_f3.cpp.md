# `f3.cpp` reference

## 1. Purpose

`src/core/qpadm/f3.cpp` is the GPU-side implementation behind the public
`include/steppe/f3.hpp` header. It computes the **f3 statistic** on its own — for
each three-population group ("triple") the caller supplies, it produces one point
estimate, a standard error, and the derived z-score and p-value. It fits no model,
weights no sources, and runs no rank test.

The file is a near-exact sibling of `f4.cpp` (the four-population version). It
reuses the two building blocks that already exist for the f4 and model-fit paths
and adds **no new infrastructure**:

- the routine that assembles the per-triple f3 identity from the stored per-block
  f2 values, and
- the block-jackknife routine that turns those per-block values into a standard
  error.

Only one small piece of math is specific to f3 (the three-term combine), and it
lives in the assemble routine, not here. Everything else in this file is glue:
flatten the triples, call the assemble step once, call the jackknife step once,
read the results out per triple.

The population indices in a triple are written `{C, A, B}` and map onto the result
as `p1 = C` (the apex/outgroup/target), `p2 = A`, `p3 = B`.

---

## 2. The pipeline

A run walks through a fixed, short sequence:

1. **Flatten the triples.** The N input `{C, A, B}` tuples are copied into a flat
   array of `3 * N` integers (the shape the assemble step consumes), and the three
   index columns are echoed onto the result so a downstream emitter can label each
   row without going back to the input.
2. **Assemble once.** A single call builds the batched per-triple f3 combine. Each
   triple is framed as a one-column problem — one "left" population, one "right"
   population, one output — so a batch of N triples is N such columns computed
   together. The block carrier used here is the same per-block container the f4
   path uses; f3 sets its "left" count to N and its "right" count to 1, which makes
   the number of output columns `N * 1 = N`.
3. **Jackknife once.** A single block-jackknife pass over the whole N-column batch
   produces the per-triple variance (see section 3 for why it is the diagonal-only
   form).
4. **Read out per triple.** For each triple `k`: the estimate is the assembled
   total for that column, the standard error is the square root of that triple's
   variance, `z = est / se`, and `p` is the two-sided normal p-value.

The p-value reuses the exact z-to-p routine the f4 path uses (`f4_two_sided_p`).
That routine matches the parity convention[^at2] (`erfc(|z| / sqrt(2))`), so f3 and
f4 p-values are produced by identical arithmetic.

---

## 3. Why the standard error uses the diagonal jackknife (the out-of-memory fix)

A bare f3 standard error needs, for each triple, only that triple's **own**
variance — the diagonal entry of what would be the full triple-by-triple
covariance matrix. It never needs the off-diagonal cross-triple terms, because
nothing here inverts a covariance (only the qpAdm model fit's generalized
least-squares step does that).

Forming the full dense covariance would be an `N × N` matrix. At an
all-triples sweep scale — where N is the number of triples, which can run into the
billions — that dense matrix does not fit in memory. So the file calls the
**diagonal-only** jackknife routine instead, which computes

```
var[k] = (1 / n_blocks) * sum over blocks b of  xtau[k, b]^2
```

directly. This is `O(N * n_blocks)` work and `O(N)` memory, and it never
materializes the dense matrix or its Cholesky inverse.

Two properties matter and are worth stating plainly:

- **It is bit-for-bit equal to the dense diagonal.** The diagonal-only path uses
  the same per-block leave-one-out values (`xtau`) and the same double-precision
  order of operations as the dense form once did, so switching to it does not move
  the f3 reference result. It is equal *by construction*, not merely close.
- **No non-SPD failure is possible.** Because f3 never inverts a covariance, there
  is no "the covariance is not symmetric-positive-definite" outcome for f3 — that
  failure only exists on the paths that actually invert. The only true f3 domain
  failure is a degenerate assemble (section 7).

---

## 4. The fudge factor is fixed at zero

The `QpAdmOptions` argument (`opts`) is accepted only for API symmetry with
`run_qpadm`, `run_qpwave`, and `run_f4`. Its `fudge` field is **deliberately not
consulted** here, and the code casts `opts` to void to satisfy `-Werror`.

The reason: the small `1e-4` ridge ("fudge") that the qpAdm fit applies exists only
to keep a matrix inversion well-conditioned. A bare f3 standard error has no matrix
inverse in it, so there is nothing to regularize. The jackknife variance is taken
completely **unfudged** (fudge is passed as 0), and that unfudged diagonal is the
value reported.

---

## 5. Precision: the estimate stays native double precision

The default arithmetic mode for the matrix-multiply-heavy stages is the faster
emulated double precision, and the heavier covariance step may run in it. The
result records which mode the covariance ran in via `precision_tag`.

The f3 estimate itself is different. The three-term combine is a small,
cancellation-prone subtraction of near-equal f2 values, so it is **always** carved
out to run in native double precision, regardless of the selected mode. The default
emulated precision is still passed down to the assemble step for one-policy
consistency, but the assemble step ignores it for this cancellation-sensitive
combine and uses native double precision anyway. As a result the reported estimate
does not vary with the precision mode; only the standard error's precision does.

---

## 6. The integer-overflow guard on the triple count

Before doing anything else, the file checks that the number of triples does not
exceed the largest value a signed 32-bit integer can hold, and returns
`Status::InvalidConfig` if it does.

This guard sits **before** the `std::size_t → int` narrowing cast that turns the
triple count into the internal `N`. At an all-triples sweep scale — for example
about 2500 populations, where the number of triples is roughly
`C(2500, 3) ≈ 2.6 billion`, well past the ~2.1-billion signed-32-bit limit — an
unguarded cast would silently wrap to a negative or truncated value. That would
defeat the `N <= 0` empty-batch guard and corrupt every downstream size that is
derived from `N` (the index-echo reserves, the `3 * N` flat-array reserve, the
column-count comparison). Surfacing an over-cap batch as a status value up front
avoids all of that.

This is behavior-preserving on real data: the reference datasets have tiny triple
counts and never approach the limit, so the guard never fires in practice and has
no effect on any checked result.

---

## 7. The three degenerate outcomes, all as status values

f3 never throws for a domain result. A batch that cannot produce numbers is a
**reported outcome**, so a caller sweeping many batches is never interrupted by an
exception. There are exactly three non-normal outcomes, handled in order:

| Situation | What is returned |
|---|---|
| The triple count exceeds the 32-bit limit (section 6). | `status = InvalidConfig`, no rows. |
| The batch is empty (`N <= 0`). | `status = Ok`, all result arrays empty. A clean, empty success — not a fault. |
| The assemble produced no usable data (zero columns or zero genome blocks — for example all blocks were missing). | `status = Ok`, but the estimate/standard-error/z/p arrays are filled with `NaN`, one per input triple, while the `p1`/`p2`/`p3` index columns are still populated. |

The third row is the important one: a degenerate assemble yields a `NaN` sentinel
per row rather than a crash or an empty result, so a caller that filters on status
still sees a well-formed table with a per-row `NaN` marking the bad triples, and
the row labels are intact.

The normal path returns `status = Ok` with real numbers. Because f3 has no
covariance inverse, the per-triple diagonal standard error is valid regardless of
whether the (never-formed) full covariance would have been positive-definite — so
`NonSpdCovariance` is not a possible f3 outcome.

---

## 8. The two overloads and the shared body

The public API has two `run_f3` overloads that differ only in where the f2 data
lives. Both are thin forwarders into a single templated body, `run_f3_impl`, which
is templated on the f2 source type so the two overloads do not duplicate the
pipeline:

- **`DeviceF2Blocks` overload** — the production path. The f2 values are already
  resident in GPU memory, so the assemble runs with **zero copy-down** from the
  device to the host.
- **`F2BlockTensor` overload** — the host-oracle / parity path. The f2 values come
  from ordinary host memory; the CPU reference backend reads them directly. This is
  how the reference-comparison test stages a known-good f2 tensor and checks
  steppe's f3 output against the parity oracle[^at2].

Both overloads target the first GPU via the shared `device::primary_backend`
accessor — the file brings it in with `using device::primary_backend;` rather than
defining any constant of its own. The `kPrimaryGpu` value that pins that choice now
lives in the shared header `core/internal/primary_backend.hpp`. Any spreading of
work across multiple GPUs happens **above** this seam —
the model-batched rotation drives the other GPUs — so this file always targets the
one primary device.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

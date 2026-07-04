# `f4.hpp` reference

## 1. Purpose

`include/steppe/f4.hpp` is the public entry point for computing a **standalone
f4 statistic** — the value `f4(p1, p2; p3, p4)` for a set of population
quartets, together with its standard error, z-score, and p-value.

It is a sibling of the qpAdm entry point in `qpadm.hpp`, not a stripped-down
copy of it. A standalone f4 does **none** of the extra work qpAdm does: there is
no model fitting (no alternating-least-squares solve) and no rank test. For each
quartet of populations it computes just two things — the point estimate and the
standard error — using a block jackknife over the genome.

It introduces no new math. It computes the estimate and error by reusing the two
internal building blocks the model-fit engine already uses:

1. the routine that assembles f4 values from the underlying per-block f2 values, and
2. the routine that computes the block-jackknife covariance.

The header is **CUDA-free by contract**: it uses only standard C++ and pulls in
no GPU code. Populations are identified by their integer **index into the
population axis** (the ordered list of populations behind the f2 data), never by
name — turning a name like `"French"` into an index is the caller's job (the
command-line tool or the language bindings). The GPU-side types it needs
(`device::DeviceF2Blocks`, `device::Resources`) are only forward-declared here;
the implementation file (`.cpp`) is what includes their real headers. This keeps
the header lightweight enough to include from the public API and the bindings
without dragging in the GPU layer.

The header depends on a few small pieces from elsewhere: `Precision` (the
arithmetic-mode tag), `Status` (the per-call outcome enum), `F2BlockTensor` (the
host-side f2 data used by the parity overload), and `QpAdmOptions` (the shared
per-call options struct, which carries the `fudge` field and the GPU
forward-declarations).

---

## 2. How a standalone f4 is computed

### The quartet-to-f4 mapping

Every quartet `(p1, p2; p3, p4)` is treated as a single column of a general f4
matrix whose left side is `{p1, p2}` and right side is `{p3, p4}`. Because that
means exactly one left pair, one right pair, and one output column, the general
four-term f4 identity collapses to a simple combination of four f2 values:

```
est = 0.5 * ( f2(p2,p3) + f2(p1,p4) - f2(p1,p3) - f2(p2,p4) )
    = f4(p1, p2; p3, p4)
```

Here `f2(a,b)` is the per-block f2 statistic between populations `a` and `b` (the
squared allele-frequency difference, computed block by block along the genome).
This matches the parity `f4(p1, p2; p3, p4)` definition[^at2]. It was validated against the
regenerated golden reference to a maximum relative difference of about
`1.36e-12` — i.e. agreement to roughly twelve significant figures.

### The point estimate and standard error

Both the estimate and its error come from the same **weighted block jackknife**
used for parity[^at2]. The genome is partitioned into blocks; the statistic is
recomputed with each block left out in turn; and the leave-one-out values are
combined (weighted by how much data each block holds) into a jackknife point
estimate and a jackknife variance. The standard error is the square root of that
variance.

- **`est`** is the jackknife point estimate of `f4`.
- **`se`** is the square root of the jackknife variance — specifically the
  diagonal entry of the jackknife covariance for that quartet.

### Batched over all quartets at once

All `N` requested quartets are computed together in one batch: the quartet index
becomes the batch axis, and a single covariance computation covers the whole
batch. For quartet `k`, the estimate is that batch entry's total, and the
standard error is the square root of the covariance's `k`-th diagonal entry.

### No ridge on the f4 standard error (`fudge = 0`)

The standard error uses the **unfudged** jackknife variance — no small value is
added to the diagonal. This is deliberate. qpAdm adds a tiny `1e-4` ridge
("fudge") to stabilize a matrix inverse inside its generalized-least-squares
solve. A bare f4 standard error involves no matrix inverse at all, so there is
nothing to regularize and the ridge is set to zero.

### z-score and p-value

From the estimate and error:

- **`z`** = `est / se`.
- **`p`** = the two-sided normal tail probability `2 * (1 - Phi(|z|))`, where
  `Phi` is the standard normal cumulative distribution. This is the two-sided
  normal convention used for f4[^at2].

---

## 3. F4Result

`F4Result` is the output table for one call. It is a set of parallel arrays: each
array has one slot per input quartet, and slot `k` in every array describes
quartet `k`, in the **same order the quartets were passed in**.

| Field | Type | Meaning |
|---|---|---|
| `p1`, `p2`, `p3`, `p4` | `vector<int>` | The population-axis indices of each quartet (length `N`). These simply echo the input, so that whatever emits the results (the CLI or a binding) can label each row without having to hold onto the original request. |
| `est` | `vector<double>` | The `f4(p1,p2;p3,p4)` point estimate per quartet (the jackknife estimate described above). |
| `se` | `vector<double>` | The standard error per quartet — the square root of the **unfudged** jackknife variance (the covariance diagonal). |
| `z` | `vector<double>` | `est / se` per quartet. |
| `p` | `vector<double>` | The two-sided normal tail probability `2 * (1 - Phi(|z|))` per quartet. |
| `status` | `Status` | The per-call outcome, default `Ok`. |
| `precision_tag` | `Precision::Kind` | Which arithmetic actually produced the result, default `Fp64` (native double precision). |

### Degenerate batches are a value, not an exception

If the batch is degenerate — for example an empty quartet list, or a covariance
matrix that comes out not positive-definite — that outcome is reported through the
`status` field (`Status::NonSpdCovariance` for the non-positive-definite case),
**not** by throwing. A domain outcome like this is always recorded and the call
returns normally, so a caller sweeping many batches can record the outcome and
keep going.

### What `precision_tag` reflects

`precision_tag` records the arithmetic mode of the **covariance matrix-multiply**,
which is the only heavy matrix step here and the one that honors the requested
precision mode. The step that assembles the f4 estimate from f2 values is a
cancellation-sensitive subtraction and is therefore always kept in native double
precision — so the **estimate is always native FP64** regardless of the tag.

---

## 4. f4_two_sided_p

```cpp
double f4_two_sided_p(double z);
```

Computes the two-sided normal tail probability `p = 2 * (1 - Phi(|z|))` — the
z-to-p conversion used for f4. It is host-only and runs in native double
precision.

This is the one special function the f4 path adds on top of what qpAdm already
provides (qpAdm uses a chi-squared tail instead). It is declared in the header so
that the implementation and any test that needs the same conversion all refer to
a single definition rather than re-deriving it.

---

## 5. run_f4 — the two entry points

There are two overloads of `run_f4`. Both take the same three trailing arguments —
the quartets to compute, the shared per-call options, and the runtime resources —
and both return an `F4Result` with one row per quartet in input order. They differ
only in where the f2 data comes from.

`quartets` is passed as a span of 4-tuples `(p1, p2, p3, p4)`, each a
population-axis index.

### The device-resident overload (the primary path)

```cpp
F4Result run_f4(const device::DeviceF2Blocks& f2,
                std::span<const std::array<int, 4>> quartets,
                const QpAdmOptions& opts,
                device::Resources& resources);
```

This is the GPU-first primary entry point. It reads f2 data that already lives in
GPU memory, so no data has to be copied back from the GPU to the host on the CUDA
path. The work runs on the first GPU in the provided resources.

### The host-oracle overload (the parity path)

```cpp
F4Result run_f4(const F2BlockTensor& f2_host,
                std::span<const std::array<int, 4>> quartets,
                const QpAdmOptions& opts,
                device::Resources& resources);
```

This overload takes f2 data as a plain host-memory tensor. It exists so the
CPU reference backend — which reads host memory — can be exercised for parity
checking. The parity test stages the golden f2 values as a host tensor and calls
this overload (or the device overload above when running on a real GPU), so the
two backends can be compared against the same reference.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

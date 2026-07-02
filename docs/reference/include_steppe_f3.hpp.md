# `f3.hpp` reference

## 1. Purpose

`include/steppe/f3.hpp` is the public entry point for computing the **f3
statistic** on its own. It is the three-population sibling of the f4 entry point
in `include/steppe/f4.hpp`: f4 works over a group of four populations (a
"quartet"), and f3 works over a group of three (a "triple").

An f3 run here is deliberately small in scope. It does not fit any model, weight
any sources, or run a rank test — it only produces, for each triple you give it:

- a single **point estimate** (the f3 value), and
- a **standard error** for that estimate, plus the derived z-score and p-value.

The header is intentionally free of any CUDA code. It uses only the C++ standard
library plus a few steppe headers, so it can be included by the core library, the
command-line tool, and the language bindings without any of them being forced to
pull in the GPU code. The actual GPU work happens in the matching `.cpp` file.

The whole computation is built by reusing machinery that already exists for the
model-fit and f4 paths — the routine that assembles the f3 identity from the
stored per-block f2 values, and the routine that turns those per-block values into
a jackknife standard error. Only one small piece of math is specific to f3 (the
three-term combine below); everything else is shared.

---

## 2. What an f3 value is and how it is computed

Each triple is written `f3(C; A, B)`, with three roles:

- **C** is the apex (also called the outgroup or the target, depending on how you
  are using it),
- **A** and **B** are the other two populations.

In the code and the result struct these map to `p1 = C`, `p2 = A`, `p3 = B`.

### The identity

An f3 value is computed from three **f2** values (an f2 value measures how far
apart two populations' allele frequencies are). The formula, per genome block, is:

```
f3(C; A, B) = 0.5 * ( f2(C, A) + f2(C, B) - f2(A, B) )
```

This is the same formula ADMIXTOOLS 2 uses, and steppe's result is checked
bit-for-bit against an ADMIXTOOLS 2 reference computed on matching data. Read it
as: the two "distances" from C out to A and out to B, minus the distance between A
and B — which isolates the shared history that A and B have in common relative to
C. Because the formula is symmetric in A and B, swapping A and B gives the same
value.

Internally each triple is treated as a one-column problem (one left population, one
right population, one output), so a batch of N triples is just N such columns
computed together.

---

## 3. The two ways f3 is used

The formula above is the only formula. What changes between the two common uses is
only which population you put in the apex role and how you read the sign of the
result.

- **Outgroup-f3** — `f3(Outgroup; A, B)`. Here C is a distant outgroup, and the
  value measures the amount of shared genetic drift between A and B (how closely
  related they are, seen from the outgroup). Larger means more shared history.

- **Admixture-f3** — `f3(Target; Src1, Src2)`. Here C is a target population and A
  and B are two candidate source populations. A **negative** value is evidence
  that the target is a mixture of (populations related to) the two sources.

Both are the same computation; only the apex population and the interpretation of
the number differ.

---

## 4. The standard error, z-score, and p-value

The standard error comes from a **block jackknife**: the genome is split into
blocks, the estimate is recomputed leaving out each block in turn, and the spread
of those leave-one-out estimates gives the variance. steppe uses the
diagonal-only form of that variance — for a batch of many triples it computes each
triple's own variance directly, rather than forming the full cross-triple variance
matrix it would never need. The standard error is the square root of that
per-triple variance.

Two specifics of the standard error are worth stating plainly:

- **No fudge factor.** The variance here is completely unfudged. The small `1e-4`
  ridge that the qpAdm model fit applies exists only to keep a matrix inversion
  well-conditioned, and a bare f3 standard error has no matrix inverse in it — so
  there is nothing to regularize, and the fudge is fixed at zero.

- **The estimate is always native double precision.** The heavier covariance step
  may run in the faster emulated-double-precision arithmetic (and the result
  records which, see `precision_tag`), but the f3 estimate itself is a small,
  cancellation-prone subtraction and is always carved out to run in native double
  precision. So the reported estimate does not vary with the precision mode.

From the estimate and standard error:

- **z-score** is simply `z = est / se`.
- **p-value** is the two-sided normal p-value, `p = 2 * pnorm_upper(|z|)`. This is
  computed by reusing the exact same z-to-p routine the f4 path uses
  (`f4_two_sided_p`), which is the same as ADMIXTOOLS 2's convention
  (`erfc(|z| / sqrt(2))`), so f3 and f4 p-values are produced identically.

---

## 5. `F3Result`

`F3Result` is the output of a run. It is a set of parallel arrays: every array has
one slot per input triple, in the **same order** you passed the triples in. Slot
`k` of every array describes triple `k`.

| Field | Type | Meaning |
|---|---|---|
| `p1` | `vector<int>` | The population index of C (the apex) for each triple. Echoed back so the caller can label rows. |
| `p2` | `vector<int>` | The population index of A for each triple. |
| `p3` | `vector<int>` | The population index of B for each triple. |
| `est` | `vector<double>` | The f3 estimate `f3(p1; p2, p3)` for each triple. |
| `se` | `vector<double>` | The standard error — the square root of the unfudged jackknife variance. |
| `z` | `vector<double>` | The z-score, `est / se`. |
| `p` | `vector<double>` | The two-sided normal p-value. |
| `status` | `Status` | The per-call outcome (see below). Defaults to `Status::Ok`. |
| `precision_tag` | `Precision::Kind` | Which arithmetic produced this result. Because the estimate is always carved out to native double precision, this really records the precision of the covariance/standard-error step. Defaults to `Fp64`. |

The population indices in `p1`/`p2`/`p3` are the same indices you supplied in the
input triples; they are copied into the result purely so a downstream emitter or
binding has everything it needs to label each row without going back to the input.

### Failures are values, not exceptions

A run that cannot produce numbers — an empty batch of triples, or a covariance that
is not a valid (symmetric positive-definite) matrix — does **not** throw. It
returns an `F3Result` whose `status` carries the reason (for example
`Status::NonSpdCovariance`). This is the record-and-continue rule: a bad batch is a
reported outcome, so a caller sweeping many batches is never interrupted by an
exception for a domain result.

---

## 6. The two `run_f3` overloads and the CUDA-free contract

There are two overloads of `run_f3`. Both take the **same** three-tuples of
population indices, the same per-call options, and the same GPU resources, and both
return an `F3Result`. They differ only in where the f2 data lives.

```cpp
// GPU-first primary: f2 already resident on the device.
F3Result run_f3(const device::DeviceF2Blocks& f2,
                std::span<const std::array<int, 3>> triples,
                const QpAdmOptions& opts,
                device::Resources& resources);

// Host-oracle / parity door: f2 supplied as a host tensor.
F3Result run_f3(const F2BlockTensor& f2_host,
                std::span<const std::array<int, 3>> triples,
                const QpAdmOptions& opts,
                device::Resources& resources);
```

- The **`DeviceF2Blocks`** overload is the production path. The f2 values are
  already sitting in GPU memory, so nothing has to be copied down from the device
  to the host to run f3. This is the normal way to call it. The work runs on the
  first GPU in `resources`.

- The **`F2BlockTensor`** (host) overload is the parity/testing path. It takes the
  f2 values from ordinary host memory, which is how the reference-comparison test
  stages a known-good ("golden") f2 tensor and checks steppe's f3 output against
  ADMIXTOOLS 2.

### What `triples` is

`triples` is a span (a non-owning view) of `{C, A, B}` index tuples. Each entry is
a `std::array<int, 3>` of **population indices**, not names. steppe deals in the
integer position of each population along its internal population axis; turning a
population *name* into that index is the job of the caller (the command-line tool or
a language binding), not this header.

### Why the header is CUDA-free

This header can be included anywhere without dragging in the GPU toolchain. It uses
only standard C++, and the two GPU-specific types it mentions
(`device::DeviceF2Blocks` and `device::Resources`) appear only as forward
declarations — names promised to exist, with their real definitions pulled in only
by the `.cpp` file that actually does the GPU work. That is what lets the public
API, the command-line tool, and the bindings all include `f3.hpp` cheaply.

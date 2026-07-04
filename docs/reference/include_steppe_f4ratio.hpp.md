# `f4ratio.hpp` reference

## 1. Purpose

`include/steppe/f4ratio.hpp` is the public entry point for computing **f4-ratios** —
the admixture-proportion statistic `qpf4ratio`[^at2]. It is the
sibling of the f4 entry point (`f4.hpp`) and the f3 entry point (`f3.hpp`), and it
follows the same shape: a small result struct and two `run_...` functions.

An f4-ratio is a single number, `alpha`, formed by dividing one f4 statistic by
another:

```
alpha = f4(p1,p2;p3,p4) / f4(p1,p2;p5,p4)
```

The two f4 statistics share three of their populations (`p1`, `p2`, `p4`); only the
third slot differs — `p3` in the numerator, `p5` in the denominator. Because it is
just a ratio of f4 values, this path is deliberately simple compared with a full
model fit: there is **no** least-squares admixture solve and **no** rank test. The
numerator and denominator are each an ordinary per-block f4, computed by reusing the
exact same four-slab quartet assembly that the f4 path and the model-fit path already
use, so no new "assemble" math is introduced. The one genuinely new piece of math is
the block-jackknife *of the ratio* — turning per-block ratios into a point estimate
and a standard error — and that lives entirely in the `.cpp`, not this header.

The header contains no CUDA code. It uses only the C++ standard library, plus a few
steppe types. Populations are passed as plain integer indices into the
device-resident f2 data (turning population names into indices is the caller's job,
handled by the app or the language bindings). The GPU-side types (`DeviceF2Blocks`,
`Resources`) are only forward-declared here; the `.cpp` includes their real headers.
This keeps the header lightweight enough to include anywhere without pulling in the
GPU stack.

---

## 2. What an f4-ratio measures (the parity convention)

The input for one f4-ratio is a **5-tuple** of populations, `(p1, p2, p3, p4, p5)`.
From that 5-tuple steppe forms two f4 statistics:

- **Numerator:** `f4(p1, p2; p3, p4)` — assembled with the four slots `a=p1`,
  `b=p2`, `c=p3`, `d=p4`.
- **Denominator:** `f4(p1, p2; p5, p4)` — assembled with `a=p1`, `b=p2`, `c=p5`,
  `d=p4`.

Only the third population differs between them. This convention was verified against
the reference R source (version 4.3.3)[^at2], where the input is a five-column matrix
`c(p1, p2, p3, p4, p5)`.

### How `alpha` is reported

The reported `alpha` is **not** simply the ratio of the two totals
(total-numerator ÷ total-denominator). Instead it is the point estimate produced by a
weighted block-jackknife over the *per-block ratios*. In other words, for each
genome block steppe forms a leave-one-out f4 for the numerator and for the
denominator, takes their ratio for that block, and then the jackknife combines those
per-block ratios into the final estimate. (The ratio-of-totals value still exists
internally, but only as the centering term inside the jackknife variance, not as the
reported answer.) The standard error `se` is the square root of that
jackknife-of-the-ratio variance, and `z = alpha / se`. This matches exactly what
the reference reports[^at2]: it emits only `alpha`, `se`, and `z` — there is no per-block
"p" column.

---

## 3. The one shared assemble (the essential invariant)

The numerator quartets and the denominator quartets are assembled together in a
**single** call, over one interleaved flat array holding both sets. For a batch of
`N` input 5-tuples this is a `2N`-quartet array. This is not an optimization detail —
it is a correctness requirement.

Here is why it matters. Before the f4 values are computed, steppe drops any genome
block that has no usable data and compacts the survivors. If the numerator and the
denominator were assembled separately, they could end up with *different* sets of
surviving blocks. The reference does not allow that[^at2]: when it marks data as missing, it
marks the numerator and the denominator missing **together**. Assembling both in one
call guarantees a single shared set of surviving blocks and a single shared set of
per-block sizes (the jackknife weights) — exactly the inputs the ratio jackknife
needs to line up.

The assembled result has `2N` rows arranged as two contiguous halves:

- Rows `0` through `N-1` are the **numerator** blocks.
- Rows `N` through `2N-1` are the **denominator** blocks.

Each row carries the three things the ratio jackknife consumes: the per-block f4
value, the per-block leave-one-out replicate, and the block size used as the
jackknife weight.

---

## 4. `F4RatioResult`

`F4RatioResult` is the result table for a batch of f4-ratios. It is a set of parallel
arrays: for a batch of `N` input 5-tuples, every array has length `N`, and index `k`
in each array refers to the same input tuple. Rows are returned in **input order**.

| Field | Type | Meaning |
|---|---|---|
| `p1, p2, p3, p4, p5` | `vector<int>` (each) | The population indices of each input 5-tuple, echoed back so the caller (emitter or binding) can label each row. `pX[k]` is the index of population X in tuple `k`. |
| `alpha` | `vector<double>` | The f4-ratio itself, `f4(p1,p2;p3,p4) / f4(p1,p2;p5,p4)` — the block-jackknife point estimate described in section 2. |
| `se` | `vector<double>` | The standard error: the square root of the jackknife-of-the-ratio variance. |
| `z` | `vector<double>` | The z-score, `alpha / se`. |
| `status` | `Status` | The **per-call** outcome. `Status::Ok` for a populated result. A degenerate batch — empty input, or blocks that are all missing — is reported as a status value with the affected rows filled by a NaN sentinel, **never** by throwing an exception. Domain outcomes like these are always values, not exceptions. |
| `precision_tag` | `Precision::Kind` | Which arithmetic produced the result. This is always `Fp64` (native double precision): the quartet assembly is the cancellation-prone step that stays in native double precision, so `alpha` is always computed natively. |

There is deliberately no per-row `p` column, because the `qpf4ratio`
output has none.

---

## 5. The two entry points

There are two overloads of `run_f4ratio`. They compute the same thing and differ only
in where the input f2 data lives. Both take the batch of 5-tuples as a span of
five-element integer arrays, a shared options struct (`QpAdmOptions`), and a
`Resources` handle, and both return one result row per input tuple in input order.
Both are marked `[[nodiscard]]`, so the returned result cannot be accidentally
ignored.

### Device-resident overload (the primary path)

```cpp
F4RatioResult run_f4ratio(const device::DeviceF2Blocks& f2,
                          std::span<const std::array<int, 5>> tuples,
                          const QpAdmOptions& opts,
                          device::Resources& resources);
```

This is the GPU-first primary entry point. Its f2 input already lives in GPU memory,
so there is no copy back to the host on the CUDA path. Work is routed through the
first GPU's backend.

### Host-oracle overload (the parity door)

```cpp
F4RatioResult run_f4ratio(const F2BlockTensor& f2_host,
                          std::span<const std::array<int, 5>> tuples,
                          const QpAdmOptions& opts,
                          device::Resources& resources);
```

This overload takes an f2 tensor that lives in **host** memory. It exists for
parity checking against the reference implementation: a test stages the known-good
("golden") f2 values as a host tensor and calls this overload, and the CPU reference
backend reads directly from that host memory. On a real GPU the device overload above
is used instead.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

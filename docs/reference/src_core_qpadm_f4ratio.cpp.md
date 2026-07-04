# `f4ratio.cpp` reference

## 1. Purpose

`src/core/qpadm/f4ratio.cpp` implements `run_f4ratio`, the standalone entry
point that computes an f4-ratio statistic and its standard error. It is the
sibling of `run_f4` (in `f4.cpp`) and `run_f3` (in `f3.cpp`) and is built the
same way: it reuses the exact same quartet-assembly step those two use — the
per-quartet four-slab identity that turns f2 block values into f4 block values —
with no new assembly math of its own. On top of that shared step it adds the one
piece of math an f4-ratio actually needs that an f4 or f3 does not: a weighted
block jackknife of a *ratio* rather than of a single statistic.

The file itself is deliberately thin. It does two things: it packs the caller's
population indices into the layout the assembly step expects, and it hands the
result to a shared backend routine that does the heavy lifting on the GPU. The
detailed jackknife and variance math does not live here; it lives next to that
backend routine. What *does* live here, and is documented below, is the layout
convention and the reasoning that lets the numerator and denominator of the ratio
be computed together correctly.

---

## 2. The f4-ratio statistic

An f4-ratio compares two f4 statistics that share most of their populations. The
caller supplies a five-population tuple `(p1, p2, p3, p4, p5)`, and the result is:

```
alpha = f4(p1, p2; p3, p4) / f4(p1, p2; p5, p4)
```

This matches the `qpf4ratio` convention[^at2] (admixtools 4.3.3). Three of the five
populations — `p1`, `p2`, `p4` — appear in both the numerator and the
denominator. Only the third slot changes: it is `p3` in the numerator and `p5`
in the denominator. Everything else is identical. Concretely, each tuple produces
two quartets:

| Role | Quartet | Which tuple slots |
|---|---|---|
| Numerator | `{p1, p2, p3, p4}` | slots 1, 2, 3, 4 |
| Denominator | `{p1, p2, p5, p4}` | slots 1, 2, 5, 4 |

`run_f4ratio` accepts a whole batch of these tuples at once (call the count `N`)
and returns one `alpha`, one standard error, and one z-score per tuple.

---

## 3. Why the numerator and denominator are assembled together

The central design decision in this file is that the `N` numerator quartets and
the `N` denominator quartets are packed into **one** flat array and assembled in
**one** call, rather than being computed as two separate batches. This is not an
optimization for its own sake — it is required for correctness.

The block jackknife works by partitioning the genome into blocks and computing the
statistic with each block left out in turn. Blocks can drop out of a computation
(become "missing") when the data does not support them. The parity handling of
missing data[^at2] forces the numerator and denominator of a ratio to go missing
*together*: a block that is unusable for one is treated as unusable for the other.
For the ratio's per-block values to line up correctly, the numerator and
denominator must therefore agree on exactly which blocks survived and how much
each surviving block weighs.

Assembling both halves in a single call is what guarantees that. One call produces
one shared set of surviving blocks and one shared set of block weights (the
`block_sizes`) that both the numerator and the denominator use. If the two halves
were assembled separately they could disagree on which blocks survived, and the
per-block ratios would be paired incorrectly.

### The flat array layout

The single flat array has length `8N`: it holds `2N` quartets back to back, and
each quartet is four integer population indices (`2N × 4 = 8N`). The two halves are
laid out as contiguous blocks, numerator first:

| Array region | Quartets | Contents |
|---|---|---|
| First half | quartets `[0, N)` | the `N` numerator quartets `{p1, p2, p3, p4}` |
| Second half | quartets `[N, 2N)` | the `N` denominator quartets `{p1, p2, p5, p4}` |

The five raw tuple indices are also copied straight onto the result (`p1` … `p5`)
so the output rows can be labeled by the caller or the language bindings.

---

## 4. The assembled block layout (`m = 2N`)

The single assembly call returns a block structure whose row count `m` is `2N` —
one row per quartet, in the same order as the flat array. So the two halves stay
contiguous on the row axis:

- Numerator rows are `k` for `k` in `[0, N)`.
- Denominator rows are `k` for `k` in `[N, 2N)`.

The pairing rule that the jackknife relies on falls straight out of this: **the
numerator for tuple `k` is row `k`, and its denominator is row `N + k`.** They are
the same offset into the two contiguous halves.

Each returned row carries, per genome block, the block's own f4 value and the
leave-one-out replicate of that f4 value (the statistic recomputed with that one
block removed), together with the shared `block_sizes` weights. That is exactly the
input the ratio jackknife consumes — nothing further needs to be derived in this
file.

---

## 5. The ratio block jackknife (delegated on-device)

Turning the paired per-block numerator and denominator values into a final
`alpha`, standard error, and z-score is **not** a host loop in this file. It is a
single call into a shared backend routine — `f4ratio_blocks_jackknife` — that both
assembles the blocks and runs the jackknife in one step. The same routine backs the
D-statistic path, so there is one engine, not two.

The call is:

```
be.f4ratio_blocks_jackknife(f2, flat, N, kSetmissThresh, prec)
```

and it returns the finished estimate, standard error, z-score, and a status. Those
map directly onto the four output fields (`alpha`, `se`, `z`, `status`).

Two backends implement the routine, and both compute the same reference math:

- **On the GPU path**, the assembled per-block f4 values and their leave-one-out
  replicates stay resident in GPU memory and feed the jackknife kernel directly.
  This deliberately avoids copying the full per-block arrays back to the host and
  running a per-tuple ratio loop there — that copy and that host loop were removed.
- **On the CPU path** (the parity oracle used for testing), the same entry point
  assembles the blocks and then hands them to an extended-precision reference
  implementation. The reference math is unchanged by the on-device version.

Inside the routine the numerator is row `k`, the denominator is row `N + k`, the
per-block weights are the shared `block_sizes`, and the total-mode selector is `0`
(the standard weighting). The routine also applies the near-zero-denominator skip
described in the next section.

---

## 6. Named constants

Two file-private constants are defined here. Both are conventions shared with the
sibling f4 and f3 entry points.

| Constant | Value | What it's for |
|---|---|---|
| `kPrimaryGpu` | `0` | The GPU index this single entry point runs on. Batching a model space across multiple GPUs is handled one layer above this file — a higher-level rotation drives the other GPUs — so at this level the work always targets GPU 0. Matches the same constant in `f4.cpp` and `f3.cpp`. |
| `kSetmissThresh` | `1e-6` | The near-zero-denominator threshold. Inside the jackknife, a per-block numerator or denominator whose absolute value is smaller than this is treated as missing for that block, so a vanishing denominator cannot produce a meaningless blown-up ratio. This matches the `qpf4ratio` `setmiss` threshold of `1e-6`[^at2]. |

---

## 7. The `fudge = 0` choice and precision

**The ridge term is always zero for an f4-ratio.** `run_f4ratio` accepts a
`QpAdmOptions` argument purely for signature symmetry with the other entry points
(`run_qpadm`, `run_qpwave`, `run_f4`, `run_f3`), but it deliberately ignores
`opts.fudge`. The small ridge that qpAdm adds is only meaningful when a matrix is
being inverted, to keep the inversion well-conditioned. An f4-ratio never inverts
anything — it is a bare ratio of two numbers — so there is nothing to stabilize and
the ridge is left at zero. The argument is explicitly discarded so the strict
compiler settings do not flag it as unused.

For precision, this file follows the same policy as the rest of the fit path:

- The arithmetic mode is whatever `default_fit_precision()` returns (the emulated
  double-precision default), and the mode that was actually honored is recorded on
  the result via `honored_tag(prec, be)` so callers can see whether the requested
  mode was available on this backend.
- The quartet assembly step itself stays in native double precision regardless of
  the selected mode. This is the same cancellation carve-out that `f4.cpp` and
  `f3.cpp` use: the f4 value is a small difference of larger numbers, and computing
  that subtraction in native double precision avoids losing significant digits.

---

## 8. Empty-batch and missing-data outcomes

Bad or degenerate inputs are reported as ordinary return values, never as thrown
exceptions.

- **An empty batch** (`N` at or below zero) returns immediately with an `Ok` status
  and no result rows. Asking for zero ratios is a clean, valid request, not a fault.
- **A tuple whose blocks all end up missing** does not throw either. The jackknife
  returns a not-a-number sentinel in that row's estimate and reports the condition
  through the per-call status, so the caller can detect it by inspecting the value
  and the status rather than by catching an error.

---

## 9. Public entry points and the shared body

There are two public overloads of `run_f4ratio`, differing only in where the input
f2 block values live:

| Overload | f2 source | Notes |
|---|---|---|
| `run_f4ratio(const device::DeviceF2Blocks&, …)` | f2 blocks already resident in GPU memory | The assembly runs on-device with no copy of the f2 data back to the host. |
| `run_f4ratio(const F2BlockTensor&, …)` | f2 blocks in host memory | Used by the CPU parity oracle, which reads host memory directly. |

Both overloads are thin forwarders. They resolve the primary GPU's backend and call
a single shared implementation, `run_f4ratio_impl`, which is templated on the f2
source type so the numerator/denominator packing and the jackknife call are written
once and not duplicated between the two overloads. This mirrors how `run_f4` and
`run_f3` share their bodies.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

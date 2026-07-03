# `qpadm_bounds.hpp` reference

## 1. Purpose

`src/core/qpadm/qpadm_bounds.hpp` is the single place that defines the size limits
of the fast qpAdm fit path — the "small path" — plus two shared formulas and one
shared status code. It is a tiny leaf header (a handful of `constexpr` values and
functions), but it sits at a spot where a mistake would be a correctness bug, so
the reasoning behind each value matters more than the line count.

qpAdm models a target population as a mixture of some **left** source populations,
measured against some **right** outgroup populations, fit at a chosen **rank**.
Throughout this file those three numbers are named:

- `nl` — the number of left sources,
- `nr` — the number of right outgroups,
- `r` — the fit rank.

There are two implementations of the fit. The **small path** runs one GPU thread
per model and keeps that model's entire working set in per-thread local memory; it
is fast when many small models are fit together (the common case when sweeping a
space of candidate models, and the small reference dataset). The **large path**
uses a full singular-value-decomposition solver and scratch memory in GPU global
memory (VRAM); it handles models too big for the small path. Both paths compute the
same math — the only question is which one a given model is routed to.

This header does two jobs:

1. It **sizes** the fixed per-thread arrays inside the small-path fit kernels (the
   dominant ones are a genotype-weight matrix and a coefficient matrix).
2. It **gates** which models are allowed onto the small path — the same predicate is
   consulted by the host code that partitions the model search and by the device
   backend.

### Why everything lives in one home

The host gate and the kernel array sizes must agree **exactly**. If the host gate
were ever widened to admit a model bigger than the arrays the kernels were compiled
for, that oversized model's per-thread arrays would be indexed past their fixed
length — a device buffer overflow, which is undefined behavior. Defining the
envelope here once, and having all three sites (the host gate, the device backend
gate, and the kernel array bounds) refer to these same names, makes that class of
drift impossible: you cannot widen the gate without also widening the arrays,
because both read the same constants. Changing the envelope is a single edit that
moves everything together.

### Why the header is CUDA-free

Everything here is a plain `constexpr int` (or a `constexpr` function over ints)
with no CUDA header included. That is deliberate. It lets this leaf be included both
by the parts of the core library that must stay free of GPU code and by the GPU
translation units, without dragging CUDA into the core. Because they are `constexpr
int`, the values can be used directly as C++ array bounds and as template
parameters — which is exactly how the kernels consume them (for example, a
fixed-length `double` array whose length is one of these constants, or a device
helper templated on three of them).

---

## 2. The small-path envelope constants

Five constants define the small-path size envelope. Three are the primary limits;
two are derived from them.

| Constant | Value | What it's for |
|---|---|---|
| `kQpMaxNl` | `5` | The most left sources the small-path per-thread arrays are sized for. |
| `kQpMaxNr` | `10` | The most right outgroups the small-path per-thread arrays are sized for. |
| `kQpMaxR` | `4` | The largest fit rank the small-path per-thread arrays are sized for. |
| `kQpMaxM` | `50` = `kQpMaxNl * kQpMaxNr` | Derived: the maximum length `m = nl * nr` of the f4 matrix. This is the row count of the internal weight and inverse matrices. |
| `kQpMaxT` | `40` = `max(kQpMaxNl, kQpMaxNr) * kQpMaxR` | Derived: the maximum coefficient dimension `t = max(nl, nr) * r` used by the alternating-least-squares solve. |

### Why these values

The small path runs one thread per model and holds that model's whole fit working
set in per-thread **local** memory. The two arrays that dominate that footprint are
the weight matrix (up to `m * t` doubles) and the coefficient matrix (up to
`t * t` doubles).

The reason the envelope is kept modest is a hard constraint of how the GPU reserves
local memory. The GPU reserves a kernel's per-thread local frame for the device's
**maximum resident thread count**, not for the number of threads you actually
launch. So an over-large fixed array makes the launch fail with an out-of-memory
error even when the kernel is run with a single thread. This was measured on real
hardware: a frame sized for a large rank/outgroup count runs out of memory at
launch.

With the values above the frame stays small:

- weight matrix: at most `50 * 40 = 2000` doubles = 16 KB,
- coefficient matrix: at most `40 * 40 = 1600` doubles = 12.5 KB.

A model that **exceeds** the envelope is not wrong or rejected — its math is
perfectly valid. It simply runs on the large path (the SVD-plus-VRAM-scratch
kernels) instead, because the host gate routes it there. For orientation: the small
reference dataset (`nl = 2`, `nr = 5`, `r = 1`, giving `m = 10` and `t` at most `5`)
is well inside the envelope, while a model with 39 right outgroups is outside it and
goes to the large path.

---

## 3. `model_fits_small_path` — the routing predicate

```cpp
constexpr bool model_fits_small_path(int nl, int nr, int r);
```

This is the one predicate that decides whether a model fits the small path. It
returns true exactly when all three of the model's dimensions are within the
envelope:

```
nl <= kQpMaxNl  &&  nr <= kQpMaxNr  &&  r <= kQpMaxR
```

A model that passes is fit by the fast model-batched small-path kernels; a model
that fails is sent to the large SVD path. This is the single home of that decision,
so the host code that partitions the model search, the device backend, and the
compiled-in kernel array bounds all agree by construction and cannot drift apart.

---

## 4. `qpadm_dof` — the chi-square degrees of freedom

```cpp
constexpr int qpadm_dof(int nl, int nr, int r);
```

This returns the number of degrees of freedom for the qpAdm chi-square test of a
rank-`r` fit of an `nl` × `nr` f4 matrix:

```
dof(r) = (nl - r) * (nr - r)
```

It is defined here once so that every consumer uses the identical formula. The
per-rank test sweep, the population-drop fallback, and the result and rank-drop
tables all call this same function instead of each re-deriving the expression (which
would risk them slowly disagreeing). It is used by the CPU reference implementation
(the parity oracle), by the GPU backend's rank sweep, result assembly, and
population-drop steps, and by the host-side rank-test fallback. Like the rest of the
file it is a plain `constexpr int`, so the device and host code share one copy.

---

## 5. `kQpStatusRankDeficient` — the rank-deficient status code

```cpp
inline constexpr int kQpStatusRankDeficient = 6;
```

This is the status code a fit kernel writes out when a per-model weight solve hits a
**singular** left-hand side — that is, a rank-deficient system with no unique
solution. The host reads this code back and turns it into a `RankDeficient` result
status. Homing the value here keeps the kernel side that emits the code and the host
side that decodes it from drifting apart.

There is one caveat recorded on the constant. The value is **frozen at `6`** because
that number is part of the kernel/host contract. Today the actual kernel emit sites
(in the small, large, and leave-one-out fit paths) are a deferred cleanup item and
still write the bare literal `6` rather than referencing this named symbol; they will
adopt the name when that cleanup happens. Until then, this constant standardizes the
name while the value stays fixed so the contract holds either way.

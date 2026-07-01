I read through `src/core/qpadm/qpgraph_fit.cpp` carefully. This is **not slop** — it's a tight, well-layered host driver for the qpGraph fit — but a senior reviewer would flag a handful of C++ polish issues and one real operational assumption.

## What's genuinely good

- **Clean architectural seams.** The pipeline is exactly what the header promises: parse model → assemble f3 basis → jackknife covariance → fleet → result. It reuses `assemble_f3_triples`, `jackknife_cov`, and `ComputeBackend::qpgraph_fit_fleet` rather than inlining a second copy of the math.
- **Status-as-value discipline.** Domain failures (`parse_qpgraph` returns an invalid model, the covariance is non-SPD, the fleet fails) are returned as `Status` values, not exceptions, matching the project's stated error-handling convention (line 69–75, 101, 108, 118).
- **Good default safety on the result struct.** `QpGraphResult` has in-class member initializers (`score = 0.0`, `status = Status::Ok`, etc.), so the early-return paths don't leave callers staring at uninitialized garbage.
- **The L4 residual-host explanation.** Lines 135–150 justify why the worst-residual scan stays on the host. It's over-long, but the reasoning is sound: the inputs are already host-resident and the output requires string labels that only the host model maps can resolve.
- **Template dispatch for the two overloads.** Lines 60–61 and 173–185 keep the public API thin and avoid duplicating the pipeline body between `DeviceF2Blocks` and `F2BlockTensor` inputs.

## What a senior developer would flag

**The unchecked `gpus[0]` assumption in `primary_backend`:**

```cpp
// line 39-42
inline constexpr std::size_t kPrimaryGpu = 0;
[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}
```

`.at(0)` will throw `std::out_of_range` if `resources.gpus` is empty. This is a CUDA-free host file that silently assumes a GPU-backed backend exists; it should return `Status::NoDevice` / `InvalidConfig` or at least assert with a clear message. The rest of the file treats domain problems as values, so a throw here is inconsistent.

**Heavy `static_cast<std::size_t>` noise and repeated model lookups:**

```cpp
// line 88-97
const int base_f2 = m.leaf_to_f2[static_cast<std::size_t>(m.base_leaf)];
...
const int la = m.centered_col_to_leaf(m.cmb1[static_cast<std::size_t>(k)]);
const int lb = m.centered_col_to_leaf(m.cmb2[static_cast<std::size_t>(k)]);
flat.push_back(m.leaf_to_f2[static_cast<std::size_t>(la)]);
```

A senior dev would ask: why not size the vector once and index-assign, or precompute the `(la, lb)` pairs? The casts are legal but visually noisy, and calling `centered_col_to_leaf` on every iteration for a constant mapping is slightly lazy.

**The residual-scan index expression is a mouthful:**

```cpp
// line 154
const double qkk = cov.Q[static_cast<std::size_t>(k) + static_cast<std::size_t>(npair) * static_cast<std::size_t>(k)];
```

Precomputing `const std::size_t npair_s = static_cast<std::size_t>(npair);` and using `k * (npair_s + 1)` would be clearer and cheaper to read.

**Idiosyncratic NaN:**

```cpp
// line 155
const double se = (qkk > 0.0) ? std::sqrt(qkk) : std::nan("");
```

`std::nan("")` works, but it looks C-ish. `std::numeric_limits<double>::quiet_NaN()` is the modern C++ idiom and doesn't require readers to remember that `nan("")` is defined behavior.

**The residual "max" initialization is fragile:**

```cpp
// line 151, 157
double worst = 0.0;
...
if (std::isfinite(z) && std::fabs(z) > std::fabs(worst)) { worst = z; worst_k = k; }
```

If every residual is exactly zero, `worst_k` stays `-1` and the labels remain empty. That's probably benign, but initializing `worst` to `0.0` instead of tracking the first finite `z` is a classic "what if the data is all zeros?" papercut.

**The L4 comment block is compensating slightly.** Lines 135–150 make a good point, but it reads like a preemptive defense. A shorter inline note plus a one-line reference to a design doc would be more confident.

**Minor const pessimization:**

```cpp
// line 115
const QpGraphTopoArena arena = make_arena(m, opts);
const QpGraphFleet fl = be.qpgraph_fit_fleet(...);
```

The `const` doesn't hurt correctness, but it can suppress move semantics if the compiler doesn't elide the return value. On a heavy `QpGraphTopoArena` (it owns several vectors), a senior dev would write `auto arena = make_arena(...)` and `auto fl = ...`.

## The "slop" test

**Not slop.** There are no unexplained magic numbers, no copy-pasted blocks with stale comments, no obviously wrong algorithms, and no missing error checks on the domain paths. The comments explain *why*, not just *what*. The math pipeline is coherent and the ownership is clear.

## What it actually looks like

This looks like **solid production code written by a domain expert who understands the admixture-graph fit and the project's layering conventions.** The CUDA-free host/driver split is correct, the value semantics are right, and the big design choices (reuse the seams, keep the residual scan on the host) are defensible. The remaining issues are C++ polish and defensive assumptions rather than algorithmic or architectural problems. A senior CUDA person has nothing to do here because there's no CUDA; a senior C++ person would spend twenty minutes tightening the `gpus[0]` guard and the casts.

## Verdict

**B+ — ship after hardening the `gpus[0]` assumption and trimming the casts/NaN/comment verbosity.** It would not embarrass anyone in a showcase, but the `primary_backend` throw and the C-style `std::nan("")` are exactly the kinds of "obvious gotcha" seniors notice first.
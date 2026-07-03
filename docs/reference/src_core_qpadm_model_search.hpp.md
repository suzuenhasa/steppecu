# `model_search.hpp` reference

## 1. Purpose

`src/core/qpadm/model_search.hpp` is the small public face of the code that
searches over many candidate admixture models at once — the "rotation" that tries
lots of different left/right population arrangements for the same target and fits
each one.

The header itself declares exactly one function, `fit_one_model_device`: the shared
routine that fits a *single* model on a given backend's device. Everything else the
matching source file provides is introduced by this header's top comment as context,
because those pieces are declared in neighbouring headers but implemented in the same
translation unit:

- the default per-model fit body that both backends inherit;
- the two public search entry points (`run_qpadm_search`, one taking GPU-resident
  data and one taking host data for the reference oracle);
- the multi-GPU fan-out that splits the model list across devices, plus the one-time
  copy of the shared statistics to every device.

So this header is the one declaration you include to fit a model on a device, and its
comment doubles as the map of the whole model-search unit.

---

## 2. Host-pure and CUDA-free

This header names only two things that touch the GPU world: the `ComputeBackend`
interface and the `Resources` bundle — and both of those are themselves free of any
CUDA code. There are no CUDA types, no kernel launches, and no GPU headers pulled in
here.

That is deliberate, and it matches the sibling fit code. Keeping the header CUDA-free
means the model-search entry points can be compiled and linked by code that has no
GPU toolchain in scope. All the actual device work happens *behind* the backend
interface: a caller hands in a backend, and the backend decides whether the numbers
are computed by GPU kernels or by the CPU reference implementation.

---

## 3. `fit_one_model_device` — the shared per-model fit body

```cpp
[[nodiscard]] QpAdmResult fit_one_model_device(
    ComputeBackend&                be,
    const device::DeviceF2Blocks&  f2,
    const QpAdmModel&              model,
    const QpAdmOptions&            opts);
```

This fits one candidate model, start to finish, on the device that `be` runs on, and
returns that model's result.

### Parameters and result

| Name | Type | Meaning |
|---|---|---|
| `be` | `ComputeBackend&` | The backend to run on. This also decides *where* the work happens: the GPU backend runs device kernels, the CPU backend runs the scalar reference implementation. |
| `f2` | `const device::DeviceF2Blocks&` | The precomputed pairwise (`f2`) statistics, already sitting in that backend's device memory. Nothing is copied off the device to feed this call. |
| `model` | `const QpAdmModel&` | The one candidate model: the target population, the left (source) populations, and the right (reference) populations, plus a stable `model_index` identity. |
| `opts` | `const QpAdmOptions&` | The fit settings (chosen rank, significance thresholds, jackknife policy, and so on). |
| *(return)* | `QpAdmResult` | The full result for this single model — weights, standard errors, z-scores, the chi-square fit, the rank test, and the nested "drop" analyses. It echoes back the caller's `model_index` so results can be re-ordered later. |

### What it does

For the one model, it builds the model's `f4` statistics from the resident `f2` data
(kept on the device — no round-trip to the host), then runs the fit pipeline on that:
solve for the admixture weights and their standard errors, run the rank test, and
compute the nested-model comparisons (dropping a rank, and dropping each population in
turn). The result carries all of those.

### The invariant that makes this the shared body

Both backends reach this exact routine through the *same* small set of backend
operations (the header calls them the five device virtuals). The GPU backend
implements them with device-resident, batched kernels; the CPU backend implements them
as a plain scalar reference. Because the two go through the identical operations for
the identical model, their results are directly comparable — bit-for-bit, model by
model. That is what lets the CPU backend serve as the trusted reference oracle the GPU
path is checked against, rather than being a separate, could-drift code path.

### Where feasibility lives

The population-drop analysis includes a row for the full (nothing-dropped) model, and
whether that row is feasible is reported inside the result, in the
`QpAdmResult::popdrop_feasible` field, not returned or passed as a separate argument
here. If you need that flag, read it off the result.

A model that simply fails to fit is not an error to this function: such domain
outcomes are recorded in the result's `status` field, not raised as an exception.

---

## 4. How it fits into the model-space search

The rest of the translation unit — declared in neighbouring headers, but living in the
same source file this header opens — wraps `fit_one_model_device` into a full search:

- **The inherited default fit body.** A default routine loops `fit_one_model_device`
  over a list of models. Both backends inherit it. The GPU backend also offers a
  genuinely *batched* override that fits many same-shape models in one dispatch; the
  search sends the common, small models to that batched override and the oversized
  "large" tail through the per-model default. The CPU oracle has no batched override,
  so every model goes through the per-model default — which is the correct shape for a
  reference.

- **The two public entry points.** `run_qpadm_search` comes in two forms: one takes
  the GPU-resident statistics (the real path), and one takes host-side statistics and
  routes every model through the CPU oracle chain (the reference the device path is
  diffed against).

- **The multi-GPU fan-out.** With more than one device, the search copies the shared
  `f2` statistics to every device once, splits the model list into contiguous shards
  (one worker thread per device), fits each shard wholly on its own device, and writes
  every result straight into its pre-sized slot by `model_index`. That pre-sized write
  *is* the deterministic re-ordering — no separate sort — and it fails fast if any
  index is out of range. If a worker faults, the lowest-numbered worker's error is the
  one re-thrown, so the failure is deterministic. Plain domain outcomes still ride in
  each result's `status`, not as throws.

- **Single-GPU is the supported path.** The multi-GPU fan-out is correct and produces
  results identical to a single-GPU run, but on the current hardware the one-time copy
  of the statistics to each device is a throughput bottleneck, so multi-GPU is
  deferred and the single-device path is the one to prefer.

# `bind_common.hpp` reference

## 1. Purpose

`bindings/internal/bind_common.hpp` is the shared toolbox for the compiled
`steppe._core` Python module. steppe's Python bindings are split into several
small compiled translation units — one per tool (`bind_qpadm.cpp`,
`bind_fstats.cpp`, and so on) — and this header is the one place all of them get
their common helpers from.

Everything here is **marshalling only**. That means converting values back and
forth between what C++ produces and what Python expects: numpy arrays to and from
C++ spans, population names to and from the integer indices the compute engine
uses, keyword arguments into option structs, and C++ result structs into flat
Python dictionaries. There is deliberately **no compute logic** in this layer and
**no pandas dependency**. The nicer, user-facing shaping — turning the flat
dictionaries into pandas DataFrames and dataclasses — happens in a separate,
pure-Python file, so the compiled module stays small and never has to link
against pandas.

The header is also a **CUDA-free host translation unit**. It includes only the
seams of the library that are themselves free of any CUDA toolkit header. It
never pulls in a GPU header like `<cuda_runtime.h>` or `<cublas_v2.h>`. If one
ever leaked in, the compile would fail — which turns the "the bindings don't
touch the GPU toolkit directly" rule into something the build enforces
structurally rather than something a reviewer has to remember to check.

---

## 2. Scope boundaries and the four interop risks

Handing GPU-backed results across the C++/Python boundary has four classic ways
to go wrong. Rather than solving each one with clever runtime machinery, this
layer sidesteps all four by keeping its scope narrow. Knowing why the scope is
drawn where it is explains several design choices further down.

| Risk | How it is avoided |
|---|---|
| **GPU memory ownership** | No GPU buffer is ever handed to Python. The only device-side object (`DeviceF2Blocks`) is created and destroyed *inside* each fit call. It never crosses into Python, so Python never owns or has to free GPU memory. |
| **Stream synchronization** | Every result returned to Python is already host-side (plain CPU memory). Python therefore never holds a handle that is secretly still being written by a GPU stream, and never has to wait on one. |
| **Silent precision loss** | The f2 tensor is double precision in every precision mode, and the numpy export is always `double` → numpy `float64`. There is no path that quietly downgrades to 32-bit. |
| **Wrong array layout** | The tensor is exported F-contiguous with shape `(P, P, n_block)`, so indexing out a single block as `arr[:,:,b]` gives that block with no silent transpose. |

There is intentionally no zero-copy GPU array sharing (no DLPack, no
`__cuda_array_interface__`) in this release. Copying host data across the
boundary is the conservative choice, and it is what makes all four risks above
simply not apply.

---

## 3. The `F2Handle`

`F2Handle` is the opaque object Python holds onto to represent a loaded f2
directory. It bundles three pieces of data plus a cached resource:

| Field | Type | Meaning |
|---|---|---|
| `tensor` | `F2BlockTensor` | The host-side f2 tensor, in double precision. |
| `pops` | `vector<string>` | The `P` population labels, in the same order as the tensor's P axis. This is the name-to-index map the compute engine itself does not carry — the engine works purely in integer indices, so the handle is where names get resolved. |
| `device` | `int` | The CUDA device ordinal that fits on this handle will target. Defaults to `0`. |
| `resources` | `unique_ptr<Resources>` | Lazily built and then cached (see below). |

Two small accessors, `P()` and `n_block()`, expose the tensor's dimensions.

### Precompute once, fit many

The `resources` field is the key design point. Building the GPU `Resources` for a
device is not free, so it is built the first time a fit runs and then cached on
the handle. Every later fit against the same handle reuses that one `Resources`.
This is the "precompute once, fit many times" pattern: load an f2 directory once,
then run many models against it cheaply.

Note that the *device upload* is separate from this caching. The `Resources` are
cached, but the host f2 tensor is uploaded to the GPU fresh inside each individual
fit call, and the resulting device object is freed at the end of that call. That
is what keeps GPU memory from crossing into Python.

---

## 4. Faults versus outcomes

This layer draws a firm line between two kinds of problems, and they are surfaced
differently to Python:

- A **fault** is a binding-layer failure — the caller asked for something that
  cannot be done, or the environment is wrong. Examples: no GPU is present, or a
  population name doesn't exist. Faults are raised as Python exceptions.
- An **outcome** is a normal domain result that happens to indicate a numerical
  problem (for example, a rank-deficient model). Outcomes are *not* exceptions;
  they ride back as a `status` field in the result dictionary.

Several small helpers implement the fault side:

- **`raise_value(msg)`** — throws a Python `ValueError` carrying a steppe reason
  string. This is the single primitive the other fault helpers build on.
- **`raise_no_device()`** — throws the one canonical "no CUDA device available"
  error. steppe is a GPU product, so an empty GPU list is a fault, not a fallback.
  Keeping this message in exactly one place stops the wording from drifting across
  the several entry points that need it.
- **`ensure_resources(handle)`** — returns the handle's cached `Resources`,
  building them on first use. It configures a single GPU (`--device <n>`; multi-GPU
  is parked, so the device list always has exactly one entry). If building the
  resources throws, that is a device fault and is re-raised as a
  `"device error: ..."` Python error. If the build succeeds but reports no GPUs,
  it raises the canonical no-device error. On success it caches the result on the
  handle and returns a reference to it.

---

## 5. Resolving population names and indices

The compute engine speaks only in integer P-axis indices; users speak in
population names. Three helpers bridge the two, and each raises a clean Python
`KeyError` naming the offending label on a miss.

- **`resolve_names(resolver, names, what)`** — resolves a variable-length list of
  names to a `vector<int>` of indices. `what` is a short label (like the argument
  name) that gets prefixed onto the error message so the failure says which
  argument was bad.
- **`resolve_tuple<N>(resolver, names, what)`** — the fixed-size counterpart:
  resolves a `std::array<std::string, N>` to a `std::array<int, N>`. This is the
  single home for the per-tuple resolve loop that the batched statistics share —
  f4 and qpDstat use `N = 4`, f3 uses `N = 3`, and f4-ratio uses `N = 5`.
- **`names_of(indices, pops)`** — the reverse direction, used when emitting
  results: it maps P-axis indices back to their labels against the `pops` map. It
  is bounds-checked. An out-of-range or `-1` index maps to the empty string, which
  is the honest sentinel for "no such population" rather than a crash or a wrong
  label. This is the single home for the indices-to-names step that every result
  emitter (f4, f3, D-statistic, f4-ratio) shares.

---

## 6. Parsing the precision string

**`parse_precision(precision, tool)`** turns the optional `precision=` keyword
argument into a `steppe::Precision` value. The accepted tokens are kept in
lockstep with the command-line `--precision` parser and with the precision tag
that gets emitted back in results, so a given precision is spelled the same way on
every surface. `tool` is the tool name, used only to make a good error message.

| Input | Result |
|---|---|
| *not given* | `EmulatedFp64{40}` — the default, tuned for the matrix-multiply-heavy f2 stages |
| `"emu40"` / `"emu"` / `"emulated_fp64"` | `EmulatedFp64{40}` |
| `"emu32"` / `"emulated_fp64_32"` | `EmulatedFp64{32}` — the faster, less accurate 32-bit emulation |
| `"fp64"` / `"native"` | native double precision — the validation oracle and fallback |
| `"tf32"` | TF32 tensor-core mode (approximate, for screening) |
| anything else | a `ValueError` naming the tool and the offending string |

The default (nothing given) is byte-identical to a default-constructed
`Precision`, so a bare call and an explicit `"emu40"` produce the exact same math.

---

## 7. The resident-f2 fit wrapper

**`with_device_f2(handle, run)`** is the shared prologue/epilogue that every fit
entry point on an `F2Handle` runs through. It captures the one place where a GPU
device object briefly exists, and does so in a way that guarantees that object
never leaks out to Python. In order, it:

1. Builds or reuses the cached `Resources` for the handle's device (via
   `ensure_resources`, so a no-GPU situation faults here).
2. Picks the bound device id, uploads the host f2 tensor to that device, and calls
   the caller-supplied `run` callable with the device tensor and the resources.
3. Returns whatever `run` returns — the return type is deduced from `run`, so each
   caller gets its own result struct back with no casting.

The uploaded device tensor is a local variable inside this function. It is freed
by its destructor the moment the call returns (or unwinds), which is exactly why
no GPU memory ever crosses into Python. Any exception thrown during the upload or
inside `run` — a CUDA error, an allocation failure, a seam error — is caught and
re-raised as a `"device error: ..."` Python error. Anything specific to a
particular tool (the exact `run` body, plus any check of the returned status) is
left to the caller.

---

## 8. Mapping keyword arguments to fit options

**`make_options(...)`** builds a `QpAdmOptions` struct from individual keyword
arguments, with every default chosen to match the option struct's own defaults so
that a bare call reproduces the reference results. It copies the numeric and
boolean knobs straight through (`fudge`, `als_iterations`, `rank`,
`allow_negative_weights`, `rank_alpha`) and translates the `jackknife` string into
the corresponding policy enum:

| `jackknife` string | Policy |
|---|---|
| `"all"` | jackknife everything |
| `"feasible_only"` | jackknife only the feasible models |
| `"none"` | no jackknife |
| anything else | a `ValueError` listing the valid choices |

---

## 9. Turning results into Python dictionaries

The bulk of the file is a family of `*_to_dict` functions. Each takes one C++
result struct and returns a flat `nb::dict` of parallel arrays and scalars. The
pure-Python facade then reshapes these flat dictionaries into DataFrames — this
layer only flattens.

Two shared enum-to-string converters underpin all of them:

- **`status_str(status)`** — maps the `Status` enum to a stable lower-case string
  such as `"ok"`, `"device_oom"`, `"rank_deficient"`, `"non_spd_covariance"`,
  `"chisq_undefined"`, or `"invalid_config"`. This is how a domain outcome (as
  opposed to a fault) reaches Python.
- **`precision_str(kind)`** — maps the precision kind to `"fp64"`,
  `"emulated_fp64"`, or `"tf32"`, so every result records which arithmetic mode
  produced it.

### Two sentinel conventions that must be preserved

Two "honest sentinel" rules run through all the emitters, and both exist to avoid
lying to the caller with a fake zero:

- **Empty means "not computed."** In `result_to_dict`, when standard errors were
  not computed the `se` (and `z`) arrays come back *empty*, which the facade turns
  into NaN columns. They are never filled with `0` — an empty list says "we didn't
  compute this," a zero would falsely say "we computed this and it's zero."
- **NaN means "degenerate."** In the f-statistic emitters (`f4_to_dict`,
  `f3_to_dict`, `dstat_to_dict`, `f4ratio_to_dict`), a degenerate combination of
  populations produces NaN estimate/standard-error/z/p values, and those NaNs ride
  straight through to numpy rather than being scrubbed to zero.

### The individual emitters

| Function | Produces |
|---|---|
| `result_to_dict` | The full qpAdm result: weights, standard errors, z, p, chi-squared, degrees of freedom, ranks, plus the parallel `rankdrop_*` and `popdrop_*` arrays. The `popdrop_feasible` field arrives as bytes (0/1) and is converted to a Python list of booleans. |
| `qpwave_to_dict` | The qpWave result: the rank test values plus the same `rankdrop_*` arrays. |
| `f4_to_dict` | An f4 table: four name columns (resolved from indices via `names_of`) plus est/se/z/p. |
| `dstat_to_dict` | A D-statistic table — identical in shape to the f4 table, because the genotype-path D result mirrors the f4 result. Four name columns plus est/se/z/p. |
| `f3_to_dict` | An f3 table — the three-column form (drop the fourth population). |
| `f4ratio_to_dict` | An f4-ratio table — the five-column form: five name columns, then `alpha` in place of `est`, plus se and z, and **no p column** (the reference f4-ratio emits only alpha/se/z). |
| `qpgraph_to_dict` | A flat dictionary of the qpGraph result: fit score, spread across restarts, the worst residual and which pair it belongs to, and the parallel arrays describing the fitted edges and admixture events. The facade reshapes this into a tidy edge table. |

The f-statistic emitters all take the handle's `pops` map as a second argument
precisely because their result structs carry integer indices, not names, so the
binding is the layer that resolves them back to labels.

---

## 10. Exporting the f2 tensor to numpy

**`f2_to_numpy(handle, src)`** exports the already-resident host f2 tensor as a
numpy `float64` array of shape `(P, P, n_block)`, F-contiguous. There is no
device-to-host copy involved — the tensor already lives in host memory.

It makes a **copy** into a fresh heap buffer rather than wrapping the tensor's
storage in place. That copy is the deliberate, conservative choice: the numpy
array owns its own buffer through a capsule deleter, so there is no lifetime
hazard tying the array's validity to the handle staying alive. The copy is a
single bulk `memcpy` (the source is one contiguous `double` array of length
`P × P × n_block`), which is a bit-identical double-precision copy with no
per-element loop.

The layout is the part that matters for correctness. Inside the tensor, the
element at row `i`, column `j`, block `b` lives at offset `i + P*j + P*P*b` — the
row index varies fastest (stride 1), then the column (stride `P`), then the block
(stride `P*P`). Those are exactly the Fortran strides for shape `(P, P, n_block)`,
so exporting the buffer as F-contiguous with that shape is correct with no
rearrangement. The practical payoff is that `arr[:,:,b]` gives block `b` as a
proper `P × P` slab with no silent per-slab transpose (this closes the
wrong-layout risk), and the values are `float64` throughout (closing the
precision-loss risk).

---

## 11. Registration entry points

The tail of the header declares the per-tool registration functions, one defined
in each `bind_<tool>.cpp` translation unit:

- `register_f2handle`
- `register_qpadm`
- `register_qpgraph`
- `register_fstats`
- `register_dates`

The module's setup code calls these in a specific order, and the order matters:
`register_f2handle` **must run first** so that the `F2Handle` nanobind type
exists before any fit entry point that takes an `F2Handle` as an argument is
registered.

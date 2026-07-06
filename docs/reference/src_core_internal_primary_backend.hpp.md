# `primary_backend.hpp` reference

## 1. Purpose

`src/core/internal/primary_backend.hpp` is a tiny helper header with one job: it
gives the single-GPU entry points **one place** to say "the compute backend on the
primary GPU."

steppe runs its work on a GPU through an object called a `ComputeBackend` — the seam
that actually launches the CUDA kernels. When a run is set up, every GPU on the box
gets its own backend, and they are all owned inside a `Resources` bundle as a vector
called `gpus` (each entry is one GPU's resources, including a `backend` pointer). A
single-GPU tool — the qpAdm fit, qpGraph, and the other entry points that fit on one
device — always wants the *first* GPU's backend, which is spelled
`*resources.gpus.at(0).backend`.

That little expression used to be copied out by hand at every such entry point —
around eleven near-identical copies. Copies like that are a quiet hazard: they all
have to agree on which index is "primary" and on how to reach through the bundle to
the backend, and nothing stops one copy from drifting (say, someone changes the
ownership layout and updates ten sites but misses the eleventh). Collapsing them onto
one named helper means there is a single source of truth. Change how you reach the
primary backend, and you change it here, once.

The header is deliberately **CUDA-free**: it names only the seam types
(`ComputeBackend`, `Resources`) and never touches a CUDA type or keyword, so it
compiles into the plain-C++ core and the tests without needing the CUDA toolkit
installed.

---

## 2. The primary-GPU index (`kPrimaryGpu`)

```
inline constexpr std::size_t kPrimaryGpu = 0;   // value: 0
```

This names the number `0` — the index of the first, "primary" GPU in the `gpus`
vector. The single-GPU entry points all fit their whole run on that one device, so
"primary" and "device 0" are the same thing here.

Giving the index a name rather than sprinkling the literal `0` around does two small
but real things: it says *why* the zero is there (it is the primary device, not an
accidental magic number), and it keeps the choice of primary device defined in one
spot instead of implied at every call site.

---

## 3. The accessor (`primary_backend`)

```
primary_backend(Resources& resources) -> ComputeBackend&
```

Returns a reference to the `ComputeBackend` living on the primary GPU:

```
return *resources.gpus.at(kPrimaryGpu).backend;
```

Reading that left to right: reach into the run's `Resources` bundle, take the primary
GPU's entry out of the `gpus` vector, and dereference its owned backend pointer to
hand back the backend itself by reference. The caller gets the real backend to launch
work on — nothing is copied, and ownership stays with `Resources`.

The `.at(kPrimaryGpu)` call is the bounds-checked vector access, not the unchecked
`[]`. If a run were ever misconfigured so that no GPU was set up, `.at(0)` throws a
`std::out_of_range` immediately instead of quietly reading past the end of an empty
vector and dereferencing a garbage pointer — a clear, loud failure at the exact spot
the mistake surfaces.

The function is marked `[[nodiscard]]`, so calling it and ignoring the returned
backend is flagged by the compiler — this accessor exists to *use* the backend, and
throwing the result away is almost certainly a bug.

It is a one-line `inline` function in a header so the compiler folds it away
entirely: using the helper costs nothing at runtime compared to the hand-written
expression it replaced, while giving every caller the single, consistent way to reach
the primary backend.

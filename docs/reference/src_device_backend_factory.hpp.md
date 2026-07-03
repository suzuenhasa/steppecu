# `backend_factory.hpp` reference

## 1. Purpose

`src/device/backend_factory.hpp` is the single place that declares how a compute
backend is constructed. steppe has two backends behind one abstract interface
(`ComputeBackend`): a CPU reference backend and a GPU backend. Rather than let each
part of the codebase hand-declare the functions that build these backends, all of
those construction declarations live here, once.

The header declares three free functions, all in `namespace steppe::device`:

1. `make_cpu_backend()` — build the CPU reference backend.
2. `make_cuda_backend(int device_id = 0)` — build the GPU backend, bound to one
   physical GPU.
3. `visible_device_count()` — ask how many CUDA devices this process can see,
   without building a backend at all.

Every one of them returns the abstract `ComputeBackend` (via `std::unique_ptr`), or
a plain `int`, so callers never have to name a concrete backend class.

---

## 2. The CUDA-free contract

This header contains no CUDA code and pulls in no part of the CUDA toolkit. It
includes only `<memory>` (for `std::unique_ptr`) and `device/backend.hpp` (for the
abstract `ComputeBackend` interface). Both of those are themselves CUDA-free.

That restriction is deliberate and load-bearing. Because constructing a backend is
declared here in CUDA-free terms, the core library, the command-line tool, and the
tests can all ask for a backend — either the CPU one or the GPU one — without any of
them having to compile against the CUDA toolkit. The actual CUDA code stays private
to the GPU library; only the two GPU-backed functions in this header
(`make_cuda_backend` and `visible_device_count`) are *defined* in a CUDA translation
unit, while every *caller* sees only these plain declarations.

The concrete backend classes are never named here either. `make_cpu_backend` is
declared here but the `CpuBackend` class stays a private detail of its own source
file, and likewise the GPU backend class is a private detail of the CUDA source
file. Callers depend on the interface, not the implementation.

---

## 3. Why the declarations live in one place

A backend's construction is part of that backend's public contract, so its
declaration belongs next to the interface — not copied into every place that happens
to build a backend.

Before this header existed, each consumer that needed a backend forward-declared the
factory function's prototype for itself. Those hand-copied prototypes lived in two
different namespaces, so a change to a factory's signature could silently break
linkage with no single source of truth to update. Collecting the declarations here
removes that hazard: there is now exactly one declaration of each function.

Consolidating the namespace was part of the same cleanup. The GPU factory always
lived in `namespace steppe::device`. The CPU factory used to live in
`namespace steppe::core`, which was a mismatch — both backends compile into the same
GPU library, so the CPU factory's placement in the core namespace did not reflect
where it actually belonged. Both factories now share the one namespace,
`steppe::device`: two constructors of one interface, in one place.

---

## 4. `make_cpu_backend()`

```cpp
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cpu_backend();
```

Constructs the CPU reference backend. This backend is the correctness oracle: it
computes in long-double precision on the host, and the GPU backend's output is
continuously compared against it to catch any divergence.

It is returned as the abstract `ComputeBackend` so callers depend only on the
CUDA-free interface. It is the backend that gets injected wherever a run wants the
reference path instead of the GPU path. The function is defined in the CPU backend's
own source file.

`[[nodiscard]]` marks the return value as not-to-be-discarded — constructing a
backend and immediately dropping it is always a mistake, so the compiler warns.

---

## 5. `make_cuda_backend(int device_id = 0)`

```cpp
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cuda_backend(int device_id = 0);
```

Constructs the GPU backend and returns it as the abstract interface, so the rest of
the code never names the concrete GPU class or touches a CUDA header. It is defined
in the CUDA source file.

### The `device_id` argument

`device_id` binds the returned backend instance to exactly one physical CUDA device.
This is the per-device-instance rule: in a multi-GPU run there is one backend
instance and one set of per-GPU resources for each device in use, and each backend
instance is constructed with its own device's id.

The backend selects its device (`cudaSetDevice`) at construction time — so that its
matrix-multiply library context is created on the right device — and again on every
compute call, so work always lands on the device the instance is bound to.

### The default of 0

The default `device_id = 0` keeps the single-GPU path bound to device 0, and it lets
every existing zero-argument call site (the reference tests and the early GPU spike
code) keep working unchanged.

### What this function does *not* do

Splitting SNPs across multiple GPUs, and combining each GPU's partial result back
together in a fixed, reproducible device order, is orchestrated one layer up — by the
resources object that owns all the per-device backends — not inside this factory.
This factory's only job is to hand back one backend bound to one device. The
multi-GPU combine that sits above it produces bit-identical results across the two
supported paths (gathering partials through the host, or copying them directly
GPU-to-GPU).

---

## 6. `visible_device_count()`

```cpp
[[nodiscard]] int visible_device_count();
```

Returns how many CUDA devices this process can see. It is a CUDA-free,
process-global count query — a single `cudaGetDeviceCount` call, defined in the CUDA
source file — that needs no backend instance to answer.

### Why this exists as its own function

The resources layer uses this count for auto-enumeration: to lay out the devices in
a dense `0 .. count-1` ordering and to validate any device ids the user configured.
Before this function existed, getting the count meant building a throwaway backend
bound to device 0 just to read the number off it — which was wasteful and had side
effects. A throwaway backend would allocate its scratch workspace, create and
destroy a matrix-multiply-library context, run a device-capability probe whose result
was then discarded, and leave the process's current device set to 0 as a lingering
side effect.

`visible_device_count` avoids all of that. Unlike `make_cuda_backend`, it creates no
library context, allocates no workspace, and does not change the current device. That
also keeps the resources code that calls it free of any direct CUDA dependency.

### Error behavior

`visible_device_count` throws `steppe::device::CudaError` if the CUDA runtime cannot
enumerate its devices at all. A process that has the GPU backend linked in is
expected to be able to enumerate, so an enumeration failure is treated as fail-fast.

A count of zero is **not** an error — the function simply returns `0`. Deciding what
to do about "no visible device" is left to the caller; the resources setup treats a
zero count as a fail-fast condition, but that policy lives there, not here.

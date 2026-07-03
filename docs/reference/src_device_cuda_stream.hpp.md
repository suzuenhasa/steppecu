# `stream.hpp` reference

## 1. Purpose

`src/device/cuda/stream.hpp` defines two small classes that own CUDA resources
and clean them up automatically: `Stream` (a CUDA stream) and `Event` (a CUDA
event). Both follow the "resource acquisition is initialization" idea — the
resource is created when the object is constructed and destroyed exactly once
when the object goes out of scope, so no caller ever has to remember to release
it by hand.

These wrappers exist to replace an older pattern where the code called
`cudaEventCreate` and `cudaEventDestroy` (or the stream equivalents) as raw,
manually paired calls — one pair per timed matrix multiply. That pattern had no
automatic cleanup, so every error path that bailed out early leaked the event it
had created. Tying the resource's lifetime to a scoped object fixes that class of
leak outright: whether the scope exits normally or by an exception, the
destructor runs and the resource is freed.

A stream is a queue of GPU work that runs in order; different streams can run
concurrently. An event is a marker you can drop into a stream to record "the work
up to here is done," which is then used either to time how long a span of work
took or to make one stream wait for another.

Despite the `.hpp` extension, this is a CUDA header — it includes
`cuda_runtime.h`. It is private to the GPU layer of steppe and must never be
compiled into the core library, the public API, or the command-line tool. Only
code that already depends on CUDA may include it.

---

## 2. The `Stream` class

`Stream` owns one CUDA stream. Constructing a `Stream` creates a fresh stream on
whatever GPU is current at that moment; destroying it frees the stream. You get
at the underlying handle with `get()` and hand that to kernel launches, to the
matrix-multiply library (cuBLAS), or to asynchronous memory copies.

### Non-blocking is the default, and it matters

By default a new `Stream` is created **non-blocking** (the
`cudaStreamNonBlocking` flag). This is deliberate and is the intended idiom: one
`Stream` per independent lane of work — a single statistics stream on the
reproducible compute path, and separate copy or search lanes when the goal is
throughput.

The reason non-blocking is the default comes down to how CUDA's legacy "default
stream" behaves. An ordinary stream implicitly synchronizes against that shared
legacy default stream, which means work on ordinary streams tends to serialize
through it. A stream created with the non-blocking flag does **not** implicitly
synchronize with the legacy default stream. That property is exactly what the
multi-GPU fan-out needs: when each per-device backend owns its own non-blocking
stream, the worker threads driving the two GPUs no longer implicitly serialize
against one shared stream, so the two GPUs' matrix multiplies can genuinely run
at the same time.

If you actually want the old legacy-default-synchronizing behavior, you can pass
`cudaStreamDefault` as the constructor's `flags` argument instead.

### The stream is bound to whichever device is current at construction

A CUDA stream is associated with the GPU that was current when it was created. So
the caller is responsible for selecting the correct device (with
`cudaSetDevice`) *before* constructing the `Stream`. This class does not pick a
device for you.

### Member functions

| Member | What it does |
|---|---|
| `Stream(flags = cudaStreamNonBlocking)` | Creates a new stream on the current device. Non-blocking by default; pass `cudaStreamDefault` for a legacy-synchronizing stream. Throws on failure. |
| `get()` | Returns the raw `cudaStream_t` handle. This is a non-owning view meant to be passed as a launch argument; the `Stream` object still owns it. |
| `synchronize()` | Blocks the calling host thread until every piece of work already queued on this stream has finished. Throws on failure. |

---

## 3. The `Event` class

`Event` owns one CUDA event. It is used for two distinct jobs — measuring elapsed
time, and ordering work across streams — and a single flag decides which job an
event is set up for.

### Timing is disabled by default

By default an `Event` is created with timing **disabled** (the
`cudaEventDisableTiming` flag). A timing-disabled event is cheaper to record and
has lower latency, which is what you want for the common case of using events
purely to order work between streams. Pass `enable_timing = true` to the
constructor only when you actually intend to measure elapsed time — the
matrix-multiply timing path.

There is a hard rule here: `elapsed_ms` only works between two events that were
both created with timing enabled. Calling it on a default (timing-disabled) event
will not give a valid duration.

### Cross-stream ordering is the preferred way to express dependencies

The `record` / `wait` pair is the primitive for making one stream depend on
another without stalling the whole GPU. You `record` an event on the stream that
produces some result, then call `wait` to make a second stream hold off until
that recorded point is reached. This is preferred over a device-wide
synchronization, which would block everything rather than just the one dependency
you care about.

### Member functions

| Member | What it does |
|---|---|
| `Event(enable_timing = false)` | Creates an event. Timing is disabled by default (the ordering use case); pass `true` to make `elapsed_ms` usable. Throws on failure. |
| `get()` | Returns the raw `cudaEvent_t` handle. |
| `record(stream)` | Drops this event into `stream`, capturing "all work queued on `stream` up to now." |
| `wait(stream)` | Makes `stream` wait until this event's recorded point completes. This is the cross-stream dependency primitive, preferred over a device-wide sync. |
| `synchronize()` | Blocks the calling host thread until this event completes. |
| `elapsed_ms(start)` | Returns the milliseconds elapsed from the `start` event to this event. Both events must have been created with timing enabled and already recorded. |

---

## 4. Move-only ownership and lifetime rules

Both `Stream` and `Event` are **move-only**, and both implement moving fully —
move-construction *and* move-assignment.

- **Copy is deleted.** Two objects must never think they both own the same
  underlying handle, because that would lead to it being destroyed twice. So you
  cannot copy either type.
- **Move is fully supported, on purpose.** Both move-construct (`Stream b =
  std::move(a);`) and move-assign (`b = std::move(a);`) are provided. Providing
  only move-construction and leaving move-assignment deleted would be a subtle
  trap: an assignment like `s = std::move(other)` would silently fail to compile.
  Supporting both makes these types behave the way an engineer expects.
- **Moving transfers ownership and empties the source.** After a move, the
  moved-from object owns nothing (its handle is set to null). A moved-from,
  empty object is still safe to destroy — its destructor simply does nothing.
- **Move-assignment frees the old resource first.** When you move-assign into an
  object that already owns a handle, it destroys its current handle before taking
  over the incoming one, so nothing leaks. It also guards against self-assignment.

### Destructors never throw

Both destructors are `noexcept` and never propagate an error. Freeing a CUDA
stream or event can, in principle, report a nonzero status at teardown — but
throwing from a destructor is dangerous (especially during stack unwinding). So
instead of throwing, a failed destroy is routed to a debug-only warning log and
otherwise ignored. This keeps teardown safe while still surfacing the problem to
anyone watching the logs in a debug build.

Note the asymmetry that is intentional: the *constructors* throw on failure (via
the project's CUDA-error check), because a resource that could not be created is
a real, up-front error worth stopping for. The *destructors* never throw,
because there is nothing useful a caller can do about a teardown failure and
throwing would be worse than swallowing it.

---

## 5. Why destruction is safe under any current device

The private `destroy()` helper in each class deliberately does **not** record the
device that was current at creation and restore it before freeing. That might
look like a missing safety step, but it is correct for streams and events.

The CUDA runtime resolves a stream's or event's own device association when you
destroy it. That means destroying one of these objects while a *different* GPU is
the current device is perfectly safe — it neither leaks the resource nor frees it
incorrectly (no use-after-free). This is the same "you can free it under any
current device" property that steppe's owning GPU-memory buffer type relies on,
so the rule is consistent across the codebase.

This is also what lets these RAII `Event` objects own the per-slot completion
events used elsewhere in the GPU pipeline: those events can be destroyed during a
teardown that happens under whatever device is current at the time, without any
special care about which device created them.

---

## 6. Header placement and layering

This file lives in the GPU-specific part of the tree and is a CUDA header because
it pulls in `cuda_runtime.h`. That places it firmly on the private, GPU-only side
of steppe's layering rule:

- It may be included by code that is already part of the GPU layer and already
  depends on CUDA.
- It must **not** be included by the core library, the public API surface, or the
  command-line tool. Those layers are intentionally kept free of any CUDA
  dependency so they can be built and consumed without a CUDA toolchain.

Its two dependencies beyond the CUDA runtime are small and purposeful: the
project's warning-log macro (the single sink used for the never-throw teardown
warning described above) and the project's CUDA-error-check macro (which turns a
failed CUDA call into a thrown error inside the constructors and the blocking
calls).

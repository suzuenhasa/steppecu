# `resources.hpp` reference

## 1. Purpose

`src/device/resources.hpp` defines the bundle of GPU device resources that the
multi-GPU "precompute" stage runs against — the stage that computes the f2 blocks
across one or more GPUs on a single machine. The bundle is built once, up front,
and then handed (injected) into the code that does the work. There is no global,
mutable "current device" state anywhere; everything a run needs travels inside this
bundle.

The header holds three kinds of things:

1. **Two small state records** — `CombinePath` (a tag naming which transport the
   last multi-GPU combine used) and `MultiGpuTimings` (wall-clock and byte totals
   for the last run). Both exist purely for observability: a benchmark or a test can
   look at them to see what happened, but they never feed into any number steppe
   reports.
2. **Two ownership structs** — `PerGpuResources` (everything for one GPU) and
   `Resources` (the collection of all the GPUs plus the run's settings and its
   observability state).
3. **Three free functions** — `validate_device_order` (check a device list is
   sane), `build_resources` (construct the whole bundle), and `device_fault_status`
   (decide whether a caught error was a genuine out-of-GPU-memory fault).

### CUDA-free by contract

This header deliberately contains no CUDA code. It names only CUDA-free types: the
abstract `ComputeBackend` interface (held through a pointer, never a concrete GPU
class), the plain `BackendCapabilities` record, and the `DeviceConfig` settings
struct. Because of that, the header compiles into the core library and the test
suite on a machine that does not even have the CUDA toolkit installed.

The one piece that actually touches the GPU — `build_resources`, whose body lives
in `resources.cpp` — is part of the device layer. It calls into the GPU factory to
create a concrete backend bound to a specific device. The declaration lives here;
the CUDA-using definition lives elsewhere.

### One backend, one device

Throughout this file the rule is: one `ComputeBackend` corresponds to exactly one
CUDA device. Each backend is constructed with its device's id and is bound to that
device. Everything a device needs to run — its CUDA stream, its cuBLAS matrix-math
handle, its cuSOLVER linear-algebra handle, and the fixed-size workspace the
emulated double-precision math requires — already lives *inside* that backend. The
structs in this header do not re-own or duplicate any of that. They add only the
things the backend itself does not carry: the device's ordinal number and its
probed capability record.

---

## 2. CombinePath — the which-transport tag

When steppe runs the f2 precompute across two or more GPUs, each GPU computes a
partial result and those partials must be summed together ("combined"). There are
two ways to move the partials for that sum:

- **Host-staged** — copy every GPU's partial back to host RAM and sum there. This is
  the portable baseline that works on any hardware.
- **Peer-to-peer, device-resident** — GPU 0 pulls each other GPU's partial directly
  over the GPU-to-GPU bus and sums them on the GPU, never staging through host RAM.
  This is the faster, opt-in path.

`CombinePath` is an enum that records **which of these the most recent multi-GPU run
actually used**. It is discovered runtime state, not a request: the run decides the
path itself, and this tag reports the outcome afterward. It is kept here, separate
from the numeric result, so that a caller or a test can confirm which transport ran
without having to inspect the result tensor. (The parity test relies on this: it
uses the tag to verify that the peer-to-peer path genuinely exercised peer-to-peer
copying rather than silently falling back to the host path.)

Both transports sum the partials in the same fixed device order, so they produce
bit-for-bit identical results. The tag therefore never signals a difference in
output — only a difference in how the bytes moved.

| Value | Meaning |
|---|---|
| `None` | No multi-GPU combine has run against this bundle yet — the default value. See the stale-read note below for an important subtlety. |
| `HostStaged` | The last run with two-or-more GPUs combined via the host-staged path. Chosen when the "prefer peer-to-peer" setting is off, **or** peer access was forbidden by the caller, **or** the hardware simply cannot do peer access. |
| `P2pDeviceResident` | The last run with two-or-more GPUs combined via the direct GPU-to-GPU path. Chosen only when all conditions line up: the caller permitted peer access, the caller preferred it, the hardware supports it, and there were at least two GPUs. |

### The stale-read subtlety of `None`

`None` is the value-initialized default, meaning "nothing has combined yet." There
is one wrinkle worth understanding: the single-GPU fast path (exactly one GPU, where
there is nothing to combine at all) does not touch this field. The multi-GPU entry
point only ever *assigns* `HostStaged` or `P2pDeviceResident`; it never writes
`None`.

The consequence: if you run a two-GPU job (which sets the tag) and then run a
one-GPU job against the same bundle, the tag still reads whatever the two-GPU run
left there — it is not reset to `None`. This is correct behavior, because a one-GPU
run combines nothing, so there is no "current transport" for it to report. The tag
always describes the last run that actually combined.

---

## 3. MultiGpuTimings

`MultiGpuTimings` is a plain record of how long the last multi-GPU run took and how
many bytes it moved. Like `CombinePath`, it exists only for observability — it lets
a benchmark or a test attribute a run's cost (compute versus combine, and the bus
traffic) without touching the numeric result. It never influences any reported
number.

The record is zeroed at the start of each two-or-more-GPU run. The single-GPU fast
path leaves it at its default, since it runs no combine.

The two wall-clock fields and the three byte-total fields are always filled by the
orchestrator, which is CUDA-free: it measures the wall times with an ordinary host
clock and computes the byte totals arithmetically from the problem dimensions. The
two finer, device-internal timing fields stay at zero unless a future combine
implementation reports them, because no GPU-side timer is threaded across the
CUDA-free boundary.

| Field | Type | Meaning |
|---|---|---|
| `compute_wall_ms` | `double` | Wall-clock time for the fan-out phase: planning the shards plus the concurrent per-GPU partial computation. Zero on the single-GPU fast path or before any multi-GPU run. |
| `combine_wall_ms` | `double` | Wall-clock time for the entire combine: placing the partials (peer or same-device copies) plus the single final copy of the full result back to the host. Zero if no combine ran. |
| `combine_peer_ms` | `double` | The cross-device copy portion of the combine, in milliseconds. Device-internal, so left at zero unless the combine reports it. |
| `combine_d2h_ms` | `double` | The single final copy-to-host portion of the combine, in milliseconds. Device-internal, so left at zero unless the combine reports it. |
| `h2d_bytes` | `size_t` | Total input bytes copied host-to-device across all GPUs for the run. The inputs partition the columns across the GPUs, so the total works out to three input arrays of size P×M in double precision. |
| `d2h_bytes` | `size_t` | Total bytes copied device-to-host, which is the one final full-result copy: two P×P-by-blocks arrays (the f2 result and its paired variance) in double precision. Under the device-resident combine there is exactly one such copy — no per-GPU partial copy-backs. |
| `peer_bytes` | `size_t` | Total cross-device copy bytes: the partials that GPU 0 pulls from the other GPUs. GPU 0's own partial is a same-device copy and is not counted as peer traffic, so this covers only the blocks that live on the non-root GPUs. |

---

## 4. PerGpuResources

`PerGpuResources` is everything the bundle owns for a single GPU. There is one of
these per device the run uses. It carries three things: the device's ordinal, the
compute backend bound to that device (which itself owns the device's stream, its
cuBLAS and cuSOLVER handles, and the emulated-double-precision workspace), and the
capability record probed from that device at build time.

Deliberately absent is any collective-communication library handle: the combine here
is always either a direct peer copy or a host-staged copy, never a library-driven
all-reduce. Also absent is any separate field for the cuSOLVER handle used by the
later model-fit stage — that handle lives one layer down, inside the backend,
alongside the stream and cuBLAS handle, so it is not re-declared here.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `device_id` | `int` | `0` | The physical CUDA device ordinal this entry owns. Equal to the corresponding entry in the run's device list. |
| `backend` | `unique_ptr<ComputeBackend>` | none | The compute backend bound to `device_id`. Owns that device's stream, handles, and workspace. Held through the abstract interface so this struct stays CUDA-free. |
| `caps` | `BackendCapabilities` | zeroed | The capability record probed from the backend once at build time: compute capability, total and free VRAM, whether the device can reach its peers, and whether the emulated double-precision mode is honorable on this device. Recorded here, separate from the numeric result; it drives the combine-path choice and the run's transport tag. |

### Move-only

Because the struct owns a `unique_ptr` to a backend, the whole struct is move-only:
it can be moved but not copied. This is intentional and matches the ownership model
for concrete backends. Copying would require cloning a backend, which is correctly
forbidden.

---

## 5. Resources

`Resources` is the top-level bundle — the collection of all the per-GPU records
plus the run's settings and its observability state. It is what gets injected into
the multi-GPU entry point. It is built once and treated as effectively constant
afterward (only the entry point mutates the two observability fields). Because it
holds a vector of move-only per-GPU records, the whole struct is itself move-only.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `gpus` | `vector<PerGpuResources>` | empty | One entry per device, in the fixed combine order (entry 0, entry 1, and so on). This ordering is the deterministic order in which partial results are summed. The first entry's device is the combine root — GPU 0 — for the peer-to-peer combine. |
| `config` | `DeviceConfig` | default | The resolved, frozen run settings: the fixed device set and order, plus the peer-access, prefer-peer-to-peer, and determinism levers. |
| `last_combine_path` | `CombinePath` | `None` | Which transport the most recent multi-GPU run used (see section 2). Discovered runtime state, written by the entry point, read by callers and tests. Stays `None` until the first multi-GPU run, and is left untouched by a single-GPU run. |
| `last_multigpu_timings` | `MultiGpuTimings` | default | Timing and byte totals for the most recent multi-GPU run (see section 3). Filled by the entry point's multi-GPU path, read by benchmarks and tests. Stays at its default after a single-GPU run, which has no combine to time. |

`device_count()` is a small convenience method returning the number of GPUs, which
is just the size of the `gpus` vector.

---

## 6. validate_device_order

```cpp
void validate_device_order(std::span<const int> order, int visible);
```

This is a pure check with no GPU access: given a list of device ordinals (`order`)
and the number of CUDA devices the process can actually see (`visible`), it confirms
the list is usable and throws if it is not. Because it touches no CUDA and no
device, it can be unit-tested on any machine.

It rejects two problems:

- **An out-of-range ordinal** — any device number that is negative or that is not
  below the visible device count. That means the caller asked for a device the
  process cannot see.
- **A duplicate ordinal** — the same device listed twice. The device order pins a
  distinct GPU at each position; a repeat would silently run two lanes one after the
  other on a single GPU, hiding a real second device and ignoring what the caller
  asked for.

On any bad ordinal it throws `std::runtime_error` (it is intentionally allowed to
throw). An **empty** list is *not* rejected here — it is simply a no-op — because the
"you must configure at least one device" check lives in `build_resources` instead,
where the error message can say so directly. This validator is factored out as its
own function specifically so this logic can be tested in isolation without any GPU.

---

## 7. build_resources

```cpp
[[nodiscard]] Resources build_resources(const DeviceConfig& config);
```

This constructs the whole `Resources` bundle for a multi-GPU precompute. The steps:

1. Read the visible CUDA device count **once**, through a CUDA-free factory query.
2. Resolve the device order. If the caller's device list is non-empty, that list is
   used as-is; if it is empty, it auto-enumerates every visible CUDA device in
   enumeration order.
3. Validate that resolved order against the visible count with
   `validate_device_order` (reject out-of-range and duplicate ordinals).
4. For each ordinal in order, construct a concrete CUDA backend bound to that device
   and probe its capabilities **once**, storing both in a `PerGpuResources`.

The capability probe does not throw merely because a device cannot reach its peers —
a "no peer access" device simply degrades to the host-staged combine baseline. But a
genuine fault — being unable to enumerate the devices, or to construct a backend on
a configured device — does throw, failing fast.

The resolved device list *is* the fixed combine order: position `g` is the GPU at
`gpus[g]`, and the combine sums those positions in order. A list of just `{0}` (or an
empty list on a one-GPU machine) yields a single-entry bundle whose one backend is
exactly the ordinary single-GPU path.

### Strongly exception-safe

If construction fails partway through — say the second of two devices cannot be bound
— everything already built is cleaned up automatically as the partial bundle unwinds.
No device handle and no GPU memory is leaked.

**Throws:**

- `std::runtime_error` on an empty resolved order, no visible device, or a duplicate
  or out-of-range ordinal.
- A typed device error (`steppe::device::CudaError`) if a configured device cannot be
  bound or a backend cannot be constructed on it.

---

## 8. device_fault_status

```cpp
[[nodiscard]] std::optional<Status> device_fault_status(
    const std::exception& e) noexcept;
```

This classifies a caught exception: it answers "was this a genuine
out-of-GPU-memory fault?" It returns the `DeviceOom` status only when the exception
is one of steppe's typed device exceptions (a CUDA, cuBLAS, or cuSOLVER error)
*and* that exception carries an allocation-failure status code. For anything else —
including a host-side `std::bad_alloc`, which is out of host RAM, not GPU memory — it
returns nothing.

### Why this function exists — the layering seam

The typed device exceptions and their CUDA status codes are private to the device
layer; they live in a CUDA-only header. A CUDA-free consumer, such as the
command-line app, therefore cannot inspect those exception types itself — it does not
have their definitions.

This declaration is the bridge. It is CUDA-free (it names only standard-library
types and steppe's own `Status`), so the app can call it, while its definition lives
in a device-layer translation unit that *can* see the private CUDA exception types.
The app's error-handling helper calls this so that a real GPU out-of-memory fault
exits with the dedicated out-of-memory exit code rather than the catch-all runtime
error code — all while the app itself stays CUDA-free, since it links against the
device library and the symbol resolves at link time.

The runtime-type inspection this function does is well-defined here because the whole
program is one statically-linked executable, so each type has a single identity. The
function does only that inspection and a few status comparisons, allocates nothing,
and is marked `noexcept`.

# `resources.cpp` reference

## 1. Purpose

`src/device/resources.cpp` is the builder that turns a caller's device settings
into a ready-to-use bundle of GPU resources.

The input is a `DeviceConfig` — the caller's *intent* about which GPUs to run on
and in what order. The output is a `Resources` object: one compute backend per
GPU, each one already bound to its physical device and already probed for what it
can do (its compute capability, how much memory it has, whether it can talk
directly to the other GPUs). The rest of the multi-GPU computation is handed this
bundle rather than discovering GPUs for itself, so there is no hidden global state
and the set of devices is fixed once, at construction.

There is one public entry point, `build_resources`, plus one public helper it
uses, `validate_device_order`, and one private helper, `resolve_device_order`,
that lives only in this file.

The single most surprising thing about this file is covered in the next section:
even though its whole job is to construct GPU backends, the file itself contains
no CUDA code at all.

---

## 2. Why this file contains no CUDA code

This file never includes the CUDA runtime header and never calls a CUDA function
directly. It reaches the GPU only through three narrow, CUDA-free entry points
that hide all the actual device code behind them:

| Entry point | What it does | Where the CUDA lives |
|---|---|---|
| `visible_device_count()` | Returns how many GPUs this process can see. | A one-line `cudaGetDeviceCount` inside the GPU code. |
| `make_cuda_backend(ordinal)` | Builds one backend object bound to one physical GPU. | The backend's constructor, in the GPU code. |
| `backend->capabilities()` | Reports what a built backend's GPU can do. | Probed inside the backend. |

Because it only names these CUDA-free functions and the plain data types they hand
back, this file compiles without the CUDA toolkit. It still gets *linked*
together with the real GPU code so those three functions resolve at link time, but
the file itself stays toolkit-free. That keeps the layering clean: the parts of
the codebase that must not depend on CUDA can still trigger a resource build
through this seam.

A practical payoff of using `visible_device_count()` for the device count is that
counting the GPUs no longer requires building a throwaway backend on GPU 0 just to
ask "how many are there?". The plain count query spins up no GPU context,
allocates no scratch memory, and — importantly — does not leave GPU 0 selected as
a lingering side effect the way constructing-and-discarding a backend used to.

---

## 3. The shared error-message prefix

Every failure this file reports starts with the same text, stored once in a
file-local constant:

| Constant | Value | What it's for |
|---|---|---|
| `kBuildErrPrefix` | `"steppe::device::build_resources: "` | The common opening of every error message thrown while building resources. Each specific failure appends its own cause after this prefix. |

Storing it once, rather than repeating the literal in each `throw`, means that if
the function is ever renamed the prefix updates in exactly one place and the four
error messages can't quietly drift apart from each other.

Two details are deliberate. The trailing space is part of the value, because each
message concatenates its specific cause immediately after the prefix and the space
keeps the combined text readable — and byte-for-byte identical to what the earlier
hand-copied messages produced. And it is a plain file-local constant, not a
promoted, project-wide tuning setting: it is a piece of message text, not a knob
anyone would ever want to configure, so it stays local to this file.

---

## 4. Choosing which devices to use, and in what order

`resolve_device_order` is a small private helper (visible only inside this file)
that decides the final, ordered list of GPU ordinals the run will use. It is a
pure decision: it works out the list and returns it, without touching any GPU or
leaving any side effect behind.

It has two cases:

- **The caller gave an explicit list.** If `config.devices` is non-empty, it is
  returned exactly as given. That list pins *both* which GPUs are used *and* the
  order they appear in. The order matters because it is the fixed order in which
  the multi-GPU partial results are later summed together, so it must be preserved
  verbatim.
- **The caller gave an empty list (auto-detect).** An empty list means "use every
  visible GPU." In that case the helper builds the dense list `0, 1, ... ,
  visible-1` in plain ordinal order, using the visible count that was already
  queried once and passed in.

One subtlety about the "no visible device" guard: the check that refuses to build
when zero GPUs are visible lives *only* on the auto-detect path. The explicit-list
case returns early, before that guard, and is never blocked by it here. An
explicit list on a machine with zero visible GPUs is not rejected in this helper;
it is caught later by the validation step (where every requested ordinal turns out
to be out of range because there are no valid ordinals at all). So this helper's
zero-visible error only fires for the auto-detect path, and validation handles the
explicit-list-on-empty-box case.

---

## 5. Checking the device list is valid

`validate_device_order` takes the resolved list of ordinals plus the visible
device count and refuses, up front, to proceed on a bad list. It is pure host
arithmetic — no CUDA, no GPU — which also makes it easy to unit-test on its own.
It rejects two things:

- **An out-of-range ordinal.** Any requested GPU number that is negative or is not
  below the visible count refers to a device the process cannot see, so the build
  is stopped with a clear message naming the bad ordinal and the visible count.
- **A duplicate ordinal.** The same GPU listed twice is rejected. The list defines
  the fixed order in which per-GPU partial results are summed, and that order must
  point at *distinct* physical devices. Silently allowing a duplicate would run two
  lanes on one physical GPU, hiding a genuinely-present second device and ignoring
  what the caller actually asked for.

An empty list is deliberately *not* rejected here — it is simply a no-op for an
empty span. The "you must have at least one device" error is raised by the caller
(`build_resources`) instead, so that the empty case carries its own, more specific
message.

This function is public (declared in the header) and is separated out from the
build routine specifically so this validation logic can be exercised on its own,
against plain `(list, count)` inputs, without any GPU present.

---

## 6. Building the bundle

`build_resources` is the public entry point that ties the previous pieces together
into the finished `Resources`. It runs the steps in a specific, deliberate order:

1. **Query the visible device count exactly once.** That single count is then
   handed to both the auto-detect sizing (`resolve_device_order`) and the
   validation step (`validate_device_order`), so the "one count query serves both"
   contract is literally true and the two can never disagree about how many GPUs
   exist.
2. **Resolve the device order** (Section 4) from the config and that count.
3. **Reject an empty resolved order.** A usable bundle must own at least one GPU,
   so an empty result stops the build with a clear message.
4. **Validate the order** (Section 5) *before binding any GPU*. Doing the check
   first is what turns two otherwise-nasty failures — two lanes silently landing on
   the same physical GPU, and a confusing low-level "invalid device ordinal" error
   thrown deep inside device selection — into an early, readable, fail-fast error.
5. **Freeze the config and build one backend per device.** The resolved config is
   copied into the bundle, then for each ordinal in order it constructs a backend
   bound to that GPU (`make_cuda_backend`) and probes its capabilities once,
   recording both in one per-GPU entry. The entries are stored in the given order,
   so the first entry is the combine root — GPU 0 — that later gathers and sums the
   others' partial results.

Two behaviors are worth calling out:

- **A genuine device fault fails fast.** If a configured GPU cannot be selected or
  a backend cannot be constructed on it, that throws and propagates out — the build
  does not limp along on a broken device.
- **Lacking direct GPU-to-GPU access is *not* a fault.** The capability probe does
  not throw when a GPU reports that it cannot do direct peer-to-peer memory copies.
  That is an expected, tagged downgrade: such a run simply uses the slower path that
  stages partial results through host memory instead. So a budget consumer GPU that
  can't do peer access still produces a perfectly valid `Resources` — its recorded
  capability just says peer access is unavailable.

The routine is strongly exception-safe. If a failure happens partway through the
bind loop — say the second GPU of a two-GPU request can't be bound — every backend
already built is cleaned up automatically as the partially-filled bundle unwinds,
so no GPU handle or GPU memory is leaked.

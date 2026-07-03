# `device_guard.cuh` reference

## 1. Purpose

`src/device/cuda/device_guard.cuh` defines one small class, `DeviceGuard`. Its
job is to remember which GPU was the "current" one when the guard was created,
and to put that GPU back as the current one when the guard goes out of scope —
automatically, including when the surrounding code exits early or throws an
exception.

CUDA has a notion of a *current device* per host thread. A function that needs to
touch a buffer living on GPU 1 has to switch the current device to GPU 1, do its
work, and then switch back to whatever the caller was using — otherwise the caller
silently finds itself running on the wrong GPU. Doing that switch-back by hand is
easy to get wrong: any `return` in the middle, or any exception, skips the manual
restore. `DeviceGuard` makes the restore happen no matter how the scope ends, using
the standard C++ "resource acquisition is initialization" (RAII) pattern — the
restore is tied to the lifetime of a local object, so the compiler guarantees it
runs at scope exit.

The header contains no computation and pulls in only the CUDA runtime, one small
logging helper, and `<utility>`. It lives inside the GPU layer and is meant to be
used only by other GPU code, not by the public API.

---

## 2. Why it exists — the duplicated guards it replaces

Before this class, the same tiny hand-written struct — a plain
`struct DeviceGuard { int dev; ~DeviceGuard(){ cudaSetDevice(dev); } }` — was
copy-pasted into several places in the device-resident "f2" pipeline. Each place
needed to temporarily switch to a buffer's owning GPU to run a copy, and then hand
the caller's GPU back:

- the code that copies f2 blocks from GPU to host and back,
- the two GPU-to-GPU ("peer") combine entry points,
- the tiered read-back path for f2 blocks.

Every one of those copies bound a buffer's owning device for a copy in one of three
directions — device-to-host, host-to-device, or GPU-to-GPU peer — and then had to
restore the caller's device afterward. Having the same guard pasted in many places
invites drift: one copy gets a fix or a tweak and the others don't. Collapsing them
into a single shared class gives the behavior exactly one home, so it can't fall out
of sync.

Two of the former restore sites did their restore with a bare `cudaSetDevice(...)`
whose return value was ignored, so a failed restore there disappeared with no trace.
Routing every site through the one shared class also fixes that (see section 4).

---

## 3. What the guard owns, and what the caller owns

`DeviceGuard` deliberately owns only *one half* of the device switch — the
**restore at the end**. It does **not** switch you to the working device for you.

The intended usage is:

1. Read the caller's current device (typically with `cudaGetDevice`).
2. Construct a `DeviceGuard` from that ordinal. From now on, whenever this scope
   ends, that device will be restored.
3. Explicitly switch to the device you actually want to work on, with a normal
   checked call (`STEPPE_CUDA_CHECK(cudaSetDevice(working_device))`). This is the
   caller's responsibility, not the guard's.
4. Do the work (a copy, a peer transfer, whatever needed the switch).
5. When the scope ends, the guard restores the device captured in step 2.

The reason the guard does not also perform the switch in step 3 is that the switch
*to* the working device is an operation that can meaningfully fail and that the
caller wants to check right away (with the throwing `STEPPE_CUDA_CHECK`). The
restore at the end, by contrast, happens during scope cleanup where throwing is not
allowed. Keeping those two on different error-handling policies is exactly why the
guard owns only the restore.

The constructor is marked `explicit`, so a bare integer can never silently turn
itself into a `DeviceGuard`. You always spell out that you are creating a guard.

---

## 4. Never-throwing teardown

A C++ destructor must never let an exception escape — if it did so while the stack
was already unwinding from another exception, the program would be terminated
outright. `DeviceGuard`'s restore runs inside its destructor, so the restore is not
allowed to throw.

That is a real constraint here, because the restore call (`cudaSetDevice`) can
return an error. In particular it can surface an error left behind by an *earlier*
asynchronous GPU launch — the failure may have nothing to do with the restore
itself, but CUDA reports it at the next call. So the restore cannot simply be
assumed to succeed.

To satisfy both requirements — report the problem, but never throw from the
destructor — the restore goes through `STEPPE_CUDA_WARN` instead of the throwing
`STEPPE_CUDA_CHECK`. `STEPPE_CUDA_WARN` logs one diagnostic line (in debug builds)
and then *yields* the error code rather than throwing on it. The yielded
`[[nodiscard]]` status is explicitly discarded with `(void)` so the strict
`-Werror` build stays happy. On the normal, everything-succeeded path this is
byte-for-byte the same work as a bare `cudaSetDevice`.

The practical upshot: routing the two former bare-`cudaSetDevice` restore sites
through this class means a failed restore now produces a warning instead of
vanishing silently, while a successful restore costs nothing extra.

---

## 5. Move-only, and the moved-from sentinel

`DeviceGuard` is **move-only**: you cannot copy it, but you can move it.

- **Copy is deleted.** Copying would mean two guards both believing they own the
  restore of the same device, and both would run the restore — one of them
  redundantly. Deleting the copy makes that impossible to write.
- **Move transfers ownership.** When a guard is moved from, the moved-*into* guard
  adopts the captured ordinal and the moved-*from* guard is parked so it owns
  nothing. That way exactly one guard ever performs the restore.

Parking is done with a sentinel value, `kNoDevice = -1`. A negative ordinal is never
a real CUDA device, so the internal `restore()` treats `kNoDevice` as "I own
nothing" and skips the `cudaSetDevice` entirely. The move constructor sets the
source's ordinal to `kNoDevice`; the move-assignment operator first restores its own
currently-held device (so that device isn't lost) and *then* adopts the other's
ordinal, again leaving the source parked at `kNoDevice`. Move operations are marked
`noexcept`.

At every place the guard is used today it is a plain named local variable that is
never actually moved, so the move-only design changes nothing about current
behavior. Move-only simply makes the class a well-formed reusable RAII owner rather
than a one-off.

---

## 6. Members at a glance

| Member | Kind | What it does |
|---|---|---|
| `DeviceGuard(int dev)` | constructor (`explicit`, `noexcept`) | Captures `dev` — the device ordinal to restore at scope exit. Does **not** switch to any device. |
| `~DeviceGuard()` | destructor | Restores the captured device, unless the guard was moved-from. Never throws (routes through `STEPPE_CUDA_WARN`). |
| `DeviceGuard(const DeviceGuard&)` | copy constructor | Deleted — the guard cannot be copied. |
| `operator=(const DeviceGuard&)` | copy assignment | Deleted. |
| `DeviceGuard(DeviceGuard&&)` | move constructor (`noexcept`) | Adopts the other guard's ordinal and parks the other at `kNoDevice`. |
| `operator=(DeviceGuard&&)` | move assignment (`noexcept`) | Restores this guard's own captured device first, then adopts the other's ordinal and parks the other at `kNoDevice`. |
| `kNoDevice` | private constant, `-1` | The "owns nothing" sentinel. A negative ordinal is never a valid CUDA device, so `restore()` skips the switch when the ordinal equals this. |
| `restore()` | private, `noexcept` | The single place the switch-back happens. Does nothing when the ordinal is `kNoDevice`; otherwise calls `cudaSetDevice` through `STEPPE_CUDA_WARN`. |
| `dev_` | private, `int` | The stored device ordinal to restore at scope exit; becomes `kNoDevice` once moved-from. |

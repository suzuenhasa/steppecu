# `pinned_buffer.cuh` reference

## 1. Purpose

`src/device/cuda/pinned_buffer.cuh` defines three small, move-only RAII helpers
that manage **page-locked (pinned) host memory** — ordinary system RAM that has
been locked so the GPU can copy to and from it directly by DMA:

1. **`PinnedBuffer<T>`** — an owning allocation of pinned host memory. Think of it
   as a reusable staging slot that the host fills (or the device fills on a copy
   back from the GPU) and then a GPU copy transfers.
2. **`RegisteredHostRegion`** — a guard that pins an *existing* buffer in place,
   without allocating anything new. It is the zero-copy alternative to
   `PinnedBuffer` for memory the backend does not own (for example a caller's input
   arrays, or the destination vectors a result is copied back into).
3. **`PinnedRegistryCache`** — a tiny fixed-size cache that remembers a handful of
   registrations so the (expensive) act of pinning the same buffer is paid once and
   reused across many calls, instead of every call.

All three are move-only (they can be moved but not copied), and each cleans itself
up in its destructor. None of them ever throws from a destructor.

This is a CUDA header and is private to the GPU layer of steppe — nothing outside
the device code includes it.

### The allocation allowlist

Pinning and unpinning host memory goes through a specific family of CUDA calls
(`cudaHostAlloc`/`cudaFreeHost` for allocating pinned memory, and
`cudaHostRegister`/`cudaHostUnregister` for pinning memory that already exists).
steppe restricts those calls to exactly three files. This header is one of them; the
other two own device (GPU-side) memory. Every other part of the codebase works only
with non-owning views and never calls the allocation family directly. Keeping the
raw calls in this one place is why the calls in this file are marked as being in an
"allowlisted" translation unit.

---

## 2. Why pinned host memory matters

A GPU copy issued as "asynchronous" (`cudaMemcpyAsync`) is only truly asynchronous —
only able to run in the background while the host thread and the GPU do other work —
when the host side of the copy is page-locked memory. If the host memory is ordinary
*pageable* memory instead, CUDA cannot DMA from it directly: it silently falls back
to a synchronous, host-blocking staging copy. In that case the copy that was
*supposed* to overlap with computation actually **blocks the thread that issued it**.

This matters most when steppe fans a workload out across more than one GPU. There, a
pageable copy blocks so much that one device's transfer cannot overlap with another
device's computation — measured at roughly 44% of time lost to blocked pageable
copies. Pinning the host memory turns the copy into a genuine background DMA that
runs concurrently with GPU kernels.

### Pin only the staging slots

Page-locked memory is a scarce operating-system resource. It is physical RAM that
can no longer be paged out, and the amount a process may lock is capped by a system
limit. Over-pinning degrades the whole machine, not just steppe. So the rule is to
pin **only** the buffers that are actually handed to an asynchronous copy, and no
more. The fixed-size cache in section 5 is a direct expression of this rule: it
bounds how many host ranges can be pinned at once.

Crucially, pinning is a performance optimization and never a correctness
requirement. If a pin cannot be done — most commonly because the system's locked-
memory limit is exhausted — steppe degrades gracefully to the pageable path (with a
debug warning) rather than failing. The copy still completes; it just runs
synchronously. See section 7.

---

## 3. `PinnedBuffer<T>`

`PinnedBuffer<T>` is an owning, move-only block of `n` page-locked elements of type
`T`, allocated with `cudaHostAlloc` and freed with `cudaFreeHost`. Use it as a
reusable staging slot: the host fills it (or a copy back from the GPU fills it) and
an asynchronous copy transfers it — the classic double-buffered pinned pipeline. It
hands out a raw host pointer through `data()` and never copies its bytes.

### `T` must be trivially copyable

The template is constrained so that `T` must be *trivially copyable*. A staging slot
is filled and drained with raw-byte copies, which are only well-defined for types
whose in-memory representation *is* their value (plain data with no custom copy
logic, no pointers-into-self, and so on). The constraint makes that a compiler-
enforced requirement: passing a non-trivially-copyable `T` is a clear compile error
rather than silent undefined behavior. Every type used with it today is a plain
arithmetic or plain-old-data type, so generated code is unaffected.

### Construction and the overflow guard

- The default constructor makes an empty buffer (null pointer, zero size).
- `PinnedBuffer(std::size_t n)` allocates `n` page-locked elements. `n == 0` is a
  valid no-op — it yields a null pointer and zero size, not an error.
- Before allocating, it checks that the byte request `n * sizeof(T)` does not
  overflow. That product is an unsigned multiply, and unsigned overflow wraps around
  silently rather than being caught by the hardware — so an unchecked multiply could
  quietly compute a *small* byte count and under-allocate. If the true byte count
  would exceed the maximum representable size, the constructor throws a typed error
  instead of allocating a too-small buffer. This mirrors the same guard on the GPU-
  side buffer type.
- On success it allocates with the default pinned-allocation flag, which yields
  ordinary page-locked memory usable by an asynchronous copy on any stream.

### Accessors

| Member | Returns |
|---|---|
| `data()` | The raw host pointer (a `const` overload returns a `const T*`). |
| `size()` | The element count. |
| `bytes()` | The byte count (`size() * sizeof(T)`). |

### Lifetime

Move-construct and move-assign transfer ownership and leave the moved-from buffer
empty. Copying is deleted. The destructor frees the buffer. Freeing (in the
destructor or when move-assigning over a live buffer) never throws: if
`cudaFreeHost` reports a nonzero status during teardown, that status is sent to a
debug-only warning sink, never raised as an exception.

The one non-obvious safety rule — do not destroy a `PinnedBuffer` while a copy into
or out of it is still in flight — is shared by all three types and is described in
section 6.

---

## 4. `RegisteredHostRegion`

`RegisteredHostRegion` is a move-only guard that page-locks an **existing** host
buffer *in place* (via `cudaHostRegister`) and unregisters it when the guard goes out
of scope. Nothing is allocated or copied: it takes a buffer that already exists —
for example a caller's input arrays being uploaded to the GPU, or the destination
vectors a result is copied back into — and turns the copy those buffers are already
part of into a true background DMA, with no extra host copy. The memory range does
not need to be page-aligned; the driver rounds to whole host pages.

### Construction

`RegisteredHostRegion(const void* ptr, std::size_t bytes)` pins the range
`[ptr, ptr+bytes)`. A null pointer or a zero length is a valid no-op — there is
nothing to pin. On *any* failure the region is simply left unregistered (see
graceful degradation, section 7); the constructor never throws.

### Registering a logically-const source

An upload to the GPU *reads* from its source buffer, so that source is naturally
`const`. `cudaHostRegister` takes a non-const `void*`, so the constructor casts the
constness away to pass it. This is sound: registering does not write any bytes, it
only changes the pages' locked state. The CUDA API uses `void*` for both read and
write ranges.

### `registered()`

`registered()` reports whether the pin actually took effect. `false` means the
pageable fallback is in use for this range. This lets the caller (and the cache in
section 5) know whether a real registration exists that will later need
unregistering.

### Lifetime

Move-construct and move-assign transfer the registration and leave the moved-from
guard empty; copying is deleted. The destructor unregisters the range. As with
`PinnedBuffer`, teardown never throws — a nonzero unregister status goes to the
debug-only warning sink. Internally, a non-null stored pointer is exactly the signal
"we registered this and must unregister it."

The destruction-while-a-copy-is-in-flight rule applies here too, and is especially
important because unregistering reverts a range to pageable — see section 6.

---

## 5. `PinnedRegistryCache`

`PinnedRegistryCache` is a small, fixed-capacity, move-only cache of *persistent*
host-page registrations. Its entire reason to exist is to **amortize the cost of
pinning**.

### Why a cache is needed

Pinning a large host range with `cudaHostRegister` is itself a heavyweight
operation: the driver has to walk and lock every page, which was measured at roughly
50 to 360 milliseconds for a multi-gigabyte range. If steppe re-pinned the *same*
caller buffer on every single call, that per-call pinning tax would dwarf the
overlap it buys — a net loss. The benefit only appears when the registration is paid
**once** and then reused across many calls.

Once amortized, the payoff on the two-GPU path is real: two concurrent *pinned*
uploads each run at close to full bus bandwidth (measured around 51 ms per iteration
per device), whereas two concurrent *pageable* uploads contend and serialize inside
the driver's internal staging (around 109 ms per iteration) — about a 2× per-device
copy speedup, but only after the pinning cost has been spread across the repeated
calls.

### `kSlots` — the fixed capacity

| Constant | Value | Meaning |
|---|---|---|
| `kSlots` | `3` | The number of host ranges the cache keeps pinned at once. |

The value 3 is deliberate: it is exactly the number of stable input arrays the f2
computation stages to the GPU per call (three of them). With three slots for three
inputs, all three stay pinned together across iterations and the cache never has to
evict anything in normal steady-state operation. A fourth distinct input would evict
the oldest — so the pinned footprint is always bounded to `kSlots` ranges and never
grows without limit. This is the "pin only the staging slots" rule (section 2) made
concrete.

### `ensure(ptr, bytes)` — the API

Call `ensure(ptr, bytes)` before each asynchronous upload of that range. Its
behavior:

- A null pointer or zero length is a no-op.
- If the same `(ptr, bytes)` is already registered in a slot, `ensure` returns
  immediately — this is the amortized fast path. A workload that reuses the same
  input buffers across iterations pays the pinning cost only on the first call.
- On a miss, it registers the new range into the next round-robin slot, evicting
  whatever that slot held. Eviction runs the old registration's destructor, which
  unregisters the old range.
- The cache only remembers a slot as holding a real pin if the registration actually
  succeeded; if the pin degraded to pageable, the slot is recorded as empty so a
  later lookup does not falsely believe that range is pinned.

All registrations are unregistered automatically when the cache is destroyed.

### Why the result destinations are not cached

The cache is used only for the *stable inputs*, whose base pointers are the same
call after call. The buffers a result is copied *back* into are freshly allocated on
every call, so they have a new base pointer each time. Caching would never hit for
them — they would pay the full pinning tax on every call with zero amortization, a
strict loss. The backend therefore leaves those result copies pageable and caches
only the stable inputs.

### The loud self-eviction guard

Evicting a slot unregisters whatever range it held, and unregistering a range that
still has a copy running over it is unsafe (see section 6). In the documented steady
state — three inputs, three slots — the round-robin victim slot is always empty, so
eviction never unregisters a live range. To make sure a subtle change can't quietly
violate that, `ensure` asserts (under debug and sanitizer builds) that the slot it is
about to evict is *not* still registered. Reaching a still-registered victim means a
fourth distinct stable input is being staged beyond the three slots — exactly the
situation where the eviction-versus-in-flight-copy hazard becomes live. Failing
loudly there is far safer than silently reverting a possibly-in-flight range to
pageable. In release builds the assert compiles out and the bounded round-robin
eviction still degrades gracefully.

---

## 6. The destruction-versus-in-flight-copy contract

All three types share one non-obvious safety rule, and it is the single most
important thing to understand before using them.

**Freeing or unregistering pinned memory does not wait for outstanding GPU copies.**
The teardown paths in this file call `cudaFreeHost` (for `PinnedBuffer`) or
`cudaHostUnregister` (for `RegisteredHostRegion`, including the cache's eviction
path) **synchronously, with no stream synchronization of their own.** That is a
deliberate design choice: adding a hidden synchronization inside a destructor would
serialize the hot copy path — the very overlap this file exists to enable.

The consequence is a caller obligation:

- Destroying (or move-assigning over, or evicting from the cache) a pinned buffer or
  a registered region **must not happen before every asynchronous copy touching that
  memory has completed.** The caller is responsible for synchronizing the stream, or
  gating reuse with an event, first.
- For `PinnedBuffer`, freeing a slot while a copy into or out of it is still running
  is a genuine use-after-free of pinned memory.
- For `RegisteredHostRegion` and the cache's eviction, unregistering a range while a
  copy is still running is just as unsafe in a subtler way: unregistering reverts the
  range to pageable, so an in-flight DMA would suddenly be reading from pageable
  memory mid-transfer — precisely the unsafe overlap these guards exist to prevent.

In practice this hazard stays latent. These owners and guards are long-lived members
of the backend: the staging buffer grows once, and the cache with `kSlots == 3`
never self-evicts in steady state, so an unregister never actually fires while a copy
is live. The self-eviction guard in section 5 exists to keep it that way even if the
workload shape changes.

---

## 7. Graceful degradation and parity

Pinning is always a performance lever, never a correctness precondition, and this
file is built so that a failed pin can never crash a run.

- A failed pin — most often because the system's locked-memory limit is exhausted,
  but also an already-registered or overlapping range, or any other status — is
  routed through a **non-throwing** warning path. The region is simply left
  unregistered. The subsequent copy still runs correctly; it just runs synchronously
  over pageable memory, which is exactly how steppe behaved before pinning was added.
  A low locked-memory limit therefore only forfeits the overlap speedup; it cannot
  bring the run down.
- When a registration fails, it may leave a "sticky" last-error flag set inside
  CUDA. This file **clears** that flag after a tolerated failure, so that a later,
  unrelated error check does not misread the tolerated pinning failure as, say, a
  kernel-launch failure.

**Parity-neutral.** Pinned and pageable copies move the *identical bytes* — pinning
changes only *how* the bytes travel, never *what* they are. So none of the machinery
in this file can change any number steppe reports. It is purely a data-movement
optimization.

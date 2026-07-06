# `device_buffer.cuh` reference

## 1. Purpose

`src/device/cuda/device_buffer.cuh` defines `DeviceBuffer<T>`, the owning,
move-only handle for a block of GPU memory. When some part of steppe needs a chunk
of device memory to hold `n` values of type `T`, it creates a `DeviceBuffer<T>`;
when that buffer goes out of scope, the memory is freed automatically. It replaces
the older pattern of pairing a raw allocate call with a matching free call by hand,
which is easy to get wrong (a missed free leaks GPU memory, a double free
corrupts it).

This is one of only three files in the whole codebase that is permitted to call
the raw CUDA allocate and free functions (`cudaMalloc` / `cudaFree`) directly. The
other two are the pooled allocator and the pinned host-buffer owner. Every other
part of steppe works with non-owning *views* into memory these three files own, and
never touches the allocate/free calls itself. Concentrating those calls in a few
audited places is what makes memory ownership easy to reason about.

The header pulls in the CUDA runtime, so it is private to the GPU layer of the
library. It is a template, so all of its logic lives in the header rather than a
separate implementation file.

The design goals baked into this file are worth stating up front, because the rest
of the file exists to serve them:

- **Automatic cleanup (RAII).** The buffer frees its own memory. You cannot forget.
- **Move-only, never copy.** Ownership of a device allocation transfers; it is never
  silently duplicated.
- **Fail fast, not silent corruption.** A bad size request throws immediately rather
  than quietly allocating something too small.
- **Fail fast must not become fail silent at teardown.** A problem freeing memory is
  reported (to a debug warning), not swallowed without a trace.

---

## 2. The trivially-copyable requirement

`DeviceBuffer<T>` only accepts a type `T` that is *trivially copyable* — roughly,
a plain-data type whose bytes fully describe its value, with no custom copy logic,
no pointers-to-self, no virtual functions. This is enforced at compile time with a
`requires` clause, so trying to instantiate the buffer with an unsuitable type is a
clear compiler error rather than a subtle runtime bug.

The reason is that every user of this buffer moves data to and from the GPU with a
raw byte copy (`cudaMemcpy` / `cudaMemcpyAsync`). A raw byte copy is only correct
for a type whose in-memory bytes *are* its value — if a type had, say, an internal
pointer, copying the bytes would copy the pointer rather than the thing it points
to, and the copy would be meaningless on the other side of the transfer. The
`requires` clause makes that unwritten rule a compiler-checked contract.

Every type actually used with this buffer today — `double`, `float`, `int`, and the
small plain-data layout structs — already satisfies the requirement, so adding the
constraint changed no generated code. It only closes the door on future misuse.

---

## 3. Constructing a buffer and the size-overflow guard

The allocating constructor takes a count `n` and reserves room for `n` elements of
`T` on the current GPU. A request of `n == 0` is treated as a legitimate empty
buffer, not an error: it holds a null pointer and a size of zero, and allocates
nothing.

The important detail here is the **size-overflow guard**. The number of bytes to
allocate is `n * sizeof(T)`. Both operands are unsigned integers, and unsigned
multiplication that grows too large does not crash or trap — by the rules of the
language it silently *wraps around* to a small value. Without a guard, a caller
asking for an enormous `n` could produce a byte count that wraps down to something
tiny; the allocate call would happily succeed with a buffer far smaller than the
buffer claims to be, and every later kernel or copy would run off the end of it.
That is silent heap corruption — exactly the failure mode the fail-fast design is
meant to prevent.

So before multiplying, the constructor checks whether `n * sizeof(T)` would exceed
the largest representable size. If it would, it throws a typed error immediately,
before any allocation happens. The error it throws is the same out-of-memory error
type the allocator itself would raise, which the public API surfaces as a
device-out-of-memory condition — a fair description, since a request that large is
by definition bigger than anything that could ever be allocated.

This buffer is the right single place to make that check because the sizes flowing
into it are formed by multiplying together several host-supplied values *before*
any higher-level memory-budget check ever sees them. Guarding once here protects
every caller.

---

## 4. Move-only semantics

A `DeviceBuffer` owns exactly one GPU allocation, and that ownership can be
transferred but never duplicated:

- **Move construction and move assignment are both provided.** You can write
  `buf = std::move(other)`, and it does the right thing: it releases whatever `buf`
  was holding, then takes over `other`'s pointer and size, leaving `other` empty.
  Providing move *assignment* (not just move *construction*) is deliberate — a
  buffer that could be move-constructed but not move-assigned would be a defect,
  because ordinary code relies on being able to reassign one.
- **Copying is deleted.** There is no copy constructor and no copy assignment.
  Copying a device allocation by handle would create two owners of the same memory,
  and whichever was destroyed first would free memory the other still thinks it
  owns. Making copies a compiler error removes that whole class of bug.

Both move operations are marked as never-throwing. Internally they use an
exchange idiom that atomically reads the source's pointer/size and resets the source
to empty in one step, so a moved-from buffer is always left in a valid, empty state
and its destructor becomes a harmless no-op.

---

## 5. Accessors and the exact byte footprint

The buffer exposes a small read-only surface:

- `data()` returns the raw device pointer, in both mutable and const forms. This is
  what gets handed to kernels and to the matrix-multiply library as an argument. The
  buffer hands out the pointer but keeps ownership; callers must not free it.
- `size()` returns the element count `n` the buffer was created with.
- `bytes()` returns the logical byte footprint, `size() * sizeof(T)`.

`bytes()` carries an invariant worth calling out. Because the constructor already
rejected any `n` whose byte product would overflow, a buffer that exists is
guaranteed to have a size small enough that `size() * sizeof(T)` *cannot* overflow.
So `bytes()` is always the exact, correct logical size — never a wrapped-around
small number.

That exactness matters because the memory-budget logic that decides whether the
whole working set fits in GPU memory adds up the `bytes()` of the resident buffers.
If `bytes()` could silently wrap to a too-small value, the budget total would come
out too small in a dangerous, safe-looking direction — the planner would conclude
everything fits when it does not. The overflow guard in the constructor is what lets
the budget code trust `bytes()`.

One nuance: `bytes()` is the *logical* size the caller asked for. The actual
allocation the GPU driver hands back is internally rounded up (to at least a
256-byte boundary). `bytes()` intentionally reports the logical request, not that
rounded-up physical size, because the logical size is what every consumer reasons
about.

---

## 6. Destruction and the device-agnostic free

All cleanup funnels through one private `reset()` helper, which the destructor and
move-assignment both call. It frees the allocation (guarding against freeing a null
pointer, which is a harmless no-op anyway) and then clears the pointer and size back
to the empty state.

Two design decisions in `reset()` are subtle enough to be the main reason this file
warrants a reference doc.

### The free is deliberately device-agnostic

In a multi-GPU run, a buffer can be allocated while one GPU is the "current" device,
then *escape* that context — it gets moved into a larger result structure and is
freed later, during a host-side combine step, while a possibly-different GPU (often
the entry/default one) is current. This is intentional and is central to how
multi-GPU results are gathered.

To support that, this owner records only the pointer and the size — **not** the
GPU that was current when the memory was allocated. And `reset()` issues a bare free
with **no** save-current-device / switch-to-alloc-device / restore-device dance
around it. This is sound because the free call is pointer-device-aware: it frees the
allocation correctly regardless of which GPU happens to be current at the time.
Adding a device switch here would be both unnecessary and wrong for the escape path
— and it would smuggle hidden global state into an object that is meant to be a
simple owner. Managing the current device is the caller's job, not this buffer's.

The comment in the source pins this down as a relied-upon property of the CUDA
runtime that the documentation permits but does not explicitly promise, so it is
recorded in one place rather than assumed silently in several.

### Re-trigger warning: this assumption breaks under async allocation

The device-agnostic free is only valid for the plain synchronous free. If steppe
ever switches this buffer to the newer *asynchronous, pool-based* allocation
scheme, the free becomes tied to a specific GPU's memory pool and stream ordering.
At that point the save/switch/restore-device dance would become mandatory: the
constructor would have to record which GPU it allocated on, and `reset()` would have
to select that GPU before freeing and restore the previous one afterward. The source
flags this explicitly so that a future change to async allocation does not quietly
inherit an assumption that no longer holds.

### The destructor never throws, but it is not silent

Cleanup must never throw an exception (throwing from a destructor is a recipe for
program termination). So `reset()` inspects the free's status code and, on a real
failure, routes it to a **debug-only warning** rather than throwing. This keeps the
fail-fast philosophy honest: a teardown problem is visible in a debug build instead
of vanishing without a trace.

Two specific status codes are treated as *benign* and deliberately do **not**
produce a warning: the codes that mean the CUDA context was already being torn down
when the free ran. These occur at normal process exit, when the runtime has already
unloaded and the operating system is reclaiming all the memory regardless. Warning
about those would just emit noise on every clean shutdown. steppe's buffers are
normally owned by the backend rather than being long-lived global objects, so this
end-of-process case rarely arises in the first place — but the guard keeps a clean
exit quiet.

---

## 7. The typed async copy helpers (`h2d_async` / `d2h_async`)

Alongside the buffer class, the header defines two small free functions that move
data between host and device: `h2d_async` (host to device) and `d2h_async` (device
to host). They are not members of `DeviceBuffer` — they are plain templates that
take a buffer plus a raw host pointer, a count, and a CUDA stream.

Each one does the same three small things that a hand-written copy would do, folded
into a single call:

- it computes the byte count as `n * sizeof(T)`, so the caller passes an *element*
  count and never has to spell out `sizeof` at the call site;
- it issues exactly one asynchronous copy in the requested direction on the given
  stream; and
- it wraps that copy in `STEPPE_CUDA_CHECK`, so a failed transfer throws the same
  typed error every other CUDA call in steppe does, rather than returning a status
  code that a caller might forget to inspect.

The reason these exist is the same "one source of truth" reasoning behind the shared
f2 numerics: before them, roughly two hundred places across the codebase hand-rolled
their own `cudaMemcpyAsync(dst, src, n * sizeof(T), direction, stream)` and their own
error check. Every one of those was a chance to get the byte count wrong (forget the
`sizeof`, or multiply by the wrong type's size), pass the wrong direction constant, or
skip the error check entirely. Converging all of them onto these two helpers means the
byte arithmetic and the error handling are written correctly once and reused, and the
call sites shrink to a clear statement of intent — *copy these `n` elements to the
device on this stream*.

One deliberate detail in the signatures: the *non-buffer* side of each copy is typed
as `void*` rather than `T*`. In `h2d_async` the host source is `const void*`; in
`d2h_async` the host destination is `void*`. This mirrors the raw `cudaMemcpyAsync`
prototype and lets the byte count derive solely from the `DeviceBuffer<T>`'s element
type. That matters at the sites where the host and device element types differ but
share a size — for example a `char` host array feeding a `std::uint8_t` device buffer
— where insisting the two pointer types match would force an extra cast for no real
benefit. Because the byte count comes from the buffer's `T`, the transfer stays
byte-identical to the hand-written copy it replaced.

These are `async` copies: they enqueue the transfer on the stream and return without
waiting for it to finish, exactly like the raw call. Ordering and completion are still
the caller's responsibility — the helpers only remove the repetitive, error-prone
boilerplate around each individual copy, not the surrounding stream synchronization.

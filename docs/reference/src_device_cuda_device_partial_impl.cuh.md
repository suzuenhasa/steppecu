# `device_partial_impl.cuh` reference

## 1. Purpose

`src/device/cuda/device_partial_impl.cuh` holds the single definition of one small
hidden struct: `DevicePartial::Impl`. That struct is where a `DevicePartial` handle
actually keeps its GPU memory — the two device buffers that hold one GPU's share of
the f2 results.

A `DevicePartial` is a move-only handle to the f2/variance result that a single GPU
computed and left sitting in that GPU's own memory (no copy back to the host, no
free until the results have been consumed). Its public class, declared in
`src/device/device_partial.hpp`, is deliberately written so that it names no CUDA
types at all. That lets host-only code carry the handle around — pass it between
worker threads, collect a vector of them — without having to include the CUDA
runtime headers. The trick that makes this possible is the "pointer to
implementation" idiom: everything that genuinely needs CUDA (the actual owning GPU
buffers) is pushed into a helper struct named `Impl`, which the public header only
forward-declares.

That helper struct has to be *defined* somewhere, and more than one CUDA source
file needs the complete definition. Rather than let each of them write its own copy
(which could silently drift apart), the definition lives here, in one tiny header
that every one of them includes. This file is CUDA code and is private to the device
library — host-only code never sees it.

---

## 2. The `Impl` struct and its two fields

`DevicePartial::Impl` contains exactly two members, both owning GPU-memory buffers:

| Field | Type | Layout | Meaning |
|---|---|---|---|
| `f2` | `DeviceBuffer<double>` | `P * P * n_block_local` elements | This GPU's share of the f2 statistics — a stack of per-block P×P slabs, one slab for each genome block this GPU owns. Stays resident in the GPU's own memory. |
| `vpair` | `DeviceBuffer<double>` | `P * P * n_block_local` elements | The matching paired-variance tensor, same shape and same residency as `f2`. |

`DeviceBuffer<double>` is an owning, move-only handle to a block of double-precision
GPU memory — it allocates on construction and frees on destruction, and it hands out
a raw device pointer (via `.data()`) for kernels and the matrix-multiply library to
use.

The two shape numbers in the layout come from the public handle (in
`device_partial.hpp`):

- `P` is the population count — the leading dimension of every slab, so each slab is
  a P-by-P matrix.
- `n_block_local` is how many genome blocks this particular GPU owns (its shard).
  When it is `0` the GPU owns nothing: that is an *empty shard*, and such a handle
  carries no `Impl` at all (the handle's `impl` pointer is null).

So the whole buffer is `n_block_local` slabs of `P × P` values laid out one after
another. `f2` and `vpair` always have the identical element count.

---

## 3. The extent-widening contract

This is the one invariant in the file that is easy to get wrong, and the source
comment calls it out explicitly. It concerns how the element count
`P * P * n_block_local` is computed.

The shape fields `P` and `n_block_local` are stored as plain 32-bit `int`. If you
multiply them together *as ints*, the product is computed in 32-bit arithmetic — and
for a realistically large problem that product overflows. Worked example from the
source: at roughly `P = 2500` populations and `n_block_local = 757` blocks, the
element count is about `4.7 billion`, which is well past the roughly `2.1 billion`
ceiling of a signed 32-bit integer. A silent overflow there would wrap the count
around to a small or negative number, the buffer would be *under-allocated*, and
every later access would read or write past the end of it.

The contract that prevents this: **whoever sizes the buffer must form the element
count in 64-bit `std::size_t`, not in the fields' `int` type.** In practice that
means casting the first factor before multiplying —
`std::size_t(P) * P * n_block_local` — so the entire product is done in 64-bit
arithmetic and cannot overflow. The sizing happens at the *consumer* (the code that
allocates the buffers, in `cuda_backend.cu`), and that consumer already computes a
`size_t` total. `DeviceBuffer`'s constructor also takes a `size_t`.

Two supporting points that are part of the same contract:

- **The shape fields stay `int` on purpose.** `P` and `n_block_local` are part of a
  plain-data boundary between the host orchestrator and the device layer, so they
  are kept as simple `int` scalars. The fix for the overflow is *not* to widen the
  fields; it is for the sizing call site to do its multiply in `size_t`. Widening is
  the caller's responsibility, not the fields'.
- **The buffer's own overflow guard is not enough by itself.** `DeviceBuffer`'s
  constructor fail-fast checks that the *byte* request (`n * sizeof(T)`) does not
  overflow. But that guard only sees the element count `n` after it has already been
  computed. If the `P * P * n_block_local` multiply overflowed in 32-bit arithmetic
  *before* the value reached the constructor, the constructor is handed an already
  wrong (small) `n` and cannot detect it. So the 64-bit widening must happen at the
  multiply site, upstream of the constructor.

---

## 4. Who shares this definition

The reason the `Impl` definition lives in its own tiny header — instead of inside one
`.cu` file — is that several CUDA source files each need the complete type. They all
include this header so they share one definition:

- **`device_partial.cu`** defines the out-of-line special members of `DevicePartial`
  (its constructor, destructor, and move operations). Those need the complete `Impl`
  type in order to construct, destroy, and move the `unique_ptr<Impl>` the handle
  owns.
- **`cuda_backend.cu` and `cuda_backend_f2_blocks.cu`** create the handles and move
  each GPU's freshly computed resident buffers into `impl->f2` and `impl->vpair`.
- **`p2p_combine.cu`** reads the resident device pointers (`impl->f2.data()`,
  `impl->vpair.data()`) so GPU 0 can pull each other GPU's partial result over a
  direct device-to-device copy and sum them; it also writes the combined output into
  a fresh output handle's own `impl->f2` / `impl->vpair`.
- **`device_f2_blocks.cu`** exposes the resident pointers and copies the tensor to or
  from host memory when needed.

Because they all compile against the same struct here, the two buffer fields have one
agreed-upon layout everywhere — the writers, the combiner, and the readers can never
disagree about what `impl->f2` and `impl->vpair` mean.

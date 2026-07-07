# `readv2_bitmatrix.cu` reference

## 1. Purpose

`src/device/cuda/readv2_bitmatrix.cu` is a four-line CUDA translation unit that does
one job: it gives the `Readv2Bitmatrix` handle a home for its special member
functions — the default constructor, the destructor, and the two move operations.

It looks almost empty, and that emptiness is the point. `Readv2Bitmatrix` is the
opaque handle to the resident `[sample × SNP-window]` bit-matrix that READv2 leaves
sitting in VRAM across the whole all-pairs relatedness sweep. The handle itself is
declared in `device/readv2_bitmatrix.hpp`, a header that is deliberately **free of
any CUDA code**. That header can only forward-declare the handle's nested `Impl`
type and hold a `std::unique_ptr<Impl>` to it — it never names the real GPU buffer.

Someone still has to compile the constructor, destructor, and move operations, and
those need to see the *complete* `Impl` (section 2). This file is that someone. It
is the only translation unit whose sole reason to exist is the special members, and
it is compiled by the CUDA toolchain because the complete `Impl` names a device
type. It is private to the `steppe_device` build target — nothing outside the GPU
layer includes it.

This file mirrors, deliberately and almost exactly, the same split already used for
`DeviceDecodeResult` (`device_decode_result.cu` / `device_decode_result_impl.cuh`).
If you understand one, you understand the other.

---

## 2. Why the special members can't be defaulted in the header (the PIMPL rule)

`Readv2Bitmatrix` follows the pointer-to-implementation (PIMPL) idiom. Its complete
state — the actual GPU allocation — lives in a nested struct:

```
struct Readv2Bitmatrix::Impl {
    DeviceBuffer<Readv2Word> words;  // [n_samples * words_per_sample], zeroed at alloc
};
```

That `Impl` is spelled out in `readv2_bitmatrix_impl.cuh`, not in the public header.
The public header only knows that an `Impl` *exists* (a forward declaration) and
holds a `std::unique_ptr<Impl>` to it.

Here is the subtlety that forces this file to exist. A `std::unique_ptr<Impl>`
cannot be destroyed — and therefore the enclosing object cannot be destructed or
move-assigned — unless the compiler can see the *complete* definition of `Impl` at
the point the destructor is generated. If the special members were `= default`ed
inside the CUDA-free header, the compiler would try to generate them there, where
`Impl` is still incomplete, and the build would fail (or, worse on some toolchains,
silently invoke undefined behavior deleting an incomplete type).

The fix is the standard PIMPL fix: **declare** the special members in the header,
but **define** them out of line in a translation unit that has included
`readv2_bitmatrix_impl.cuh` and so sees the complete `Impl`. That translation unit
is this file. Every one of the four members here is a simple `= default`:

| Member | Declaration site | Definition site |
|---|---|---|
| `Readv2Bitmatrix()` | header | here |
| `~Readv2Bitmatrix()` | header | here |
| `Readv2Bitmatrix(Readv2Bitmatrix&&) noexcept` | header | here |
| `operator=(Readv2Bitmatrix&&) noexcept` | header | here |

They are defaulted, but they are defaulted *here*, where `Impl` is complete, so the
compiler generates a correct `unique_ptr` teardown. The copy constructor and copy
assignment are `= delete`d in the header and need no out-of-line body — the handle
owns GPU memory and is move-only by design.

---

## 3. What the handle owns, and what stays CUDA-free

Understanding this file means knowing the shape of the thing whose lifetime it
manages. The handle carries two kinds of state:

- **Host scalars, in the public header.** The window geometry — `n_samples`,
  `window_snps`, `m0`, `wpw` (words per window), `n_win` (windows tiling the SNP
  axis), `words_per_sample`, and `device_id` — are plain fields on the handle. They
  are CUDA-free integers, so any host code that merely needs to *carry* a bit-matrix
  around (the core READv2 driver, the reduction/emit layer) can read the geometry
  without ever touching a GPU header.
- **The device payload, behind the `unique_ptr<Impl>`.** The single
  `DeviceBuffer<Readv2Word>` — a flat `[n_samples × words_per_sample]` array of
  128-bit `(allele, valid)` cells resident in the video memory of `device_id` — is
  the only CUDA-typed thing, and it hides entirely behind the opaque pointer.

The `empty()` predicate on the handle (`n_samples <= 0 || words_per_sample <= 0`)
lets host code test for a live matrix without dereferencing the pointer at all.

The producers and consumers of the payload live in `cuda_backend_readv2.cu`, not
here: that file allocates and zeroes the buffer (`readv2_alloc_bitmatrix`), packs
streamed 2-bit genotype chunks into it (`readv2_pack_chunk`), and runs the all-pairs
`__popc` windowed-mismatch reduction against it (`readv2_mismatch`). This file only
owns the *lifetime* of the handle, not any of that work.

---

## 4. Contracts and invariants

- **Move-only ownership.** After a move (construction or assignment), the source
  handle's `unique_ptr` is null. A null `impl` is the canonical representation of an
  empty or moved-from bit-matrix: it holds no resident GPU buffer. The destructor
  defined here frees the `DeviceBuffer` (and thus the VRAM) exactly once, when the
  owning handle dies; a moved-from handle frees nothing.
- **A default-constructed handle is empty.** `Readv2Bitmatrix()` leaves `impl` null
  and every geometry scalar at its header default (`n_samples = 0`, etc.), so
  `empty()` is true. The backend fills the scalars and the `impl` in
  `readv2_alloc_bitmatrix`; a handle whose allocation was skipped (any of
  `n_samples`, `window_snps`, `m0` non-positive) stays in exactly this default state,
  with the geometry set but `impl` still null.
- **`noexcept` moves.** Both move operations are `noexcept`, which is what a
  `unique_ptr` move is anyway. Marking them so lets containers and the rest of the
  pipeline move the handle without pessimizing to a copy (a copy is deleted, so a
  throwing move would be a hard error rather than a fallback).
- **Include discipline.** This file includes exactly `readv2_bitmatrix_impl.cuh`
  (which pulls in the public header, the `DeviceBuffer` definition, and the layout
  cell) plus `<memory>` for the `unique_ptr`. It launches no kernel, allocates
  nothing, and touches no stream. It must stay this thin — any real work belongs in
  `cuda_backend_readv2.cu`.

---

## 5. Edge cases and gotchas

- **The definitions must not migrate back into the header.** The single most
  important non-obvious rule: do not "simplify" by moving these `= default`
  definitions into `readv2_bitmatrix.hpp`. That would reintroduce the incomplete-type
  teardown bug section 2 exists to prevent, and it would drag CUDA types onto the
  CUDA-free seam. The two-file split is load-bearing, not stylistic.
- **Only two translation units include the `Impl`.** This file (the special members)
  and `cuda_backend_readv2.cu` (the producer/consumer) are the only includers of
  `readv2_bitmatrix_impl.cuh`. Keeping that list short is what keeps the CUDA
  dependency contained.
- **Destroying a handle runs no synchronization.** Freeing the handle frees the
  underlying `DeviceBuffer` through its own destructor. This file adds no explicit
  stream sync of its own; correctness of any in-flight async work against the buffer
  is the backend's responsibility (the backend synchronizes its stream after each
  pack/reduce), not the teardown's.
- **The file will keep looking trivial, and that is fine.** If `Impl` ever grows a
  second buffer, nothing here changes — the `= default` members regenerate correctly
  as long as they keep being compiled where `Impl` is complete. There is nothing to
  add to this file when the payload changes; that is the whole benefit of the
  pattern.

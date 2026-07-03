# `device_decode_result_impl.cuh` reference

## 1. Purpose

`src/device/cuda/device_decode_result_impl.cuh` defines the one small struct that
holds the actual GPU memory behind a `DeviceDecodeResult`. It is the CUDA half of a
deliberate two-file split:

- The public handle `DeviceDecodeResult` (in `device/device_decode_result.hpp`) is
  written to be free of any CUDA code. It only forward-declares a nested type named
  `Impl` and keeps a `std::unique_ptr<Impl>` to it. Anything that merely needs to
  carry the handle around — the CUDA-free orchestrator in the core library, for
  example — can include that header without pulling in the GPU code.
- This file is where `Impl` is actually spelled out. It is the only place that names
  the real GPU buffer type (`DeviceBuffer<double>`), so it is the only file in the
  pair that has to be compiled by the CUDA toolchain.

That split is the entire reason the file exists. It lets the decode result be a
device-resident thing (its data lives in GPU memory) while still being passable
through code that knows nothing about CUDA.

The struct carries a decode result that has already been "compacted": after decoding,
the SNPs that don't pass the keep rule are removed, and the surviving columns are
packed together. So the buffers below are sized to the *kept* SNP count, not the
original SNP count.

---

## 2. The three device buffers

`DeviceDecodeResult::Impl` is just three GPU buffers:

| Field | Type | Shape | What it holds |
|---|---|---|---|
| `q` | `DeviceBuffer<double>` | `P × M_kept` | The decoded per-population allele-frequency-style values, one column per kept SNP. |
| `v` | `DeviceBuffer<double>` | `P × M_kept` | The paired variance values, in lockstep with `q`. |
| `n` | `DeviceBuffer<double>` | `P × M_kept` | The per-population non-missing counts. **Present only on the filtered (regime-B) path — empty otherwise** (see section 3). |

Shared facts about all three:

- **Resident on the GPU.** They live in the video memory of the CUDA device that
  produced them (the handle's `device_id`), not in host RAM. Nothing is copied back
  to the host to build this struct.
- **Column-major layout.** `P` is the number of populations (the leading dimension)
  and `M_kept` is the compacted kept-SNP count (the number of columns). The value for
  population `i` at kept SNP `s` sits at flat index `i + P·s`. This is the same
  addressing the public handle documents for its borrowed device pointers.
- **`q` and `v` are always both present** on a non-empty result and always share the
  same shape. `n` is the conditional one.

---

## 3. The `n`-by-regime invariant

The one non-obvious rule in this file is that the `n` buffer is sometimes filled and
sometimes intentionally left empty. Which happens depends on *which* decode path
produced the result. There are two such paths, referred to as regimes:

- **Regime A — the autosome-only compact decode.** Its consumers (the qpfstats and
  D-statistic computations) read only `q` and `v`. They never look at per-SNP counts,
  so this path does not bother compacting `n`. On a regime-A result, `n` is empty and
  the handle's `n_device()` returns null.
- **Regime B — the filtered `extract_f2` decode.** This is the path that applies the
  frequency- and coverage-based SNP filters (minor-allele-frequency, max-missing,
  SNP-class). Its downstream step is the f2 matrix-multiply, which *does* need the
  per-population counts. So this path compacts `q`, `v`, and `n` together in lockstep
  onto the kept axis, and `n` comes out populated. On a regime-B result,
  `n_device()` is non-null.

So the presence of `n` is not decoration — it is the signal that tells the rest of
the code which kind of result it is holding. The read-back helper on the public
handle relies on this: it refuses to run on a result whose `n` is empty, because a
regime-A result simply has no counts to hand back.

---

## 4. Privacy and who may include this file

This header is private to the `steppe_device` build unit. It is not part of the
public API, and code outside the GPU layer must not include it — that is exactly what
the CUDA-free public handle protects against.

Two translation units include it, and only those two:

- `cuda/device_decode_result.cu` — defines the handle's special members
  (constructor, destructor, move operations) and the accessor functions
  (`q_device()`, `v_device()`, `n_device()`), all of which reach through the
  `unique_ptr` into this `Impl` to read the buffers.
- `cuda_backend.cu` — the producer side, which builds a result with its buffers
  already resident in GPU memory by filling in this `Impl`.

Because `Impl` is only ever handled through a `std::unique_ptr`, a `DeviceDecodeResult`
whose pointer is null represents an empty or moved-from result with no resident
buffers at all.

# `readv2_bitmatrix.hpp` reference

## 1. Purpose

`src/device/readv2_bitmatrix.hpp` declares two small host-facing types that the
READv2 kinship pipeline hands around: `Readv2Bitmatrix`, a move-only handle to the
packed genotype bit-matrix that lives in GPU memory for the whole all-pairs sweep,
and `Readv2Pairs`, the four host-resident arrays the mismatch reduction returns.

The important thing about this header is what it *doesn't* name. It has no CUDA in
it — no `__device__`, no `cudaStream_t`, no `DeviceBuffer`. It sits on the CUDA-free
seam so that the core READv2 driver (`src/core/readv2/readv2.cpp`) and the app layer
can include it and pass a `Readv2Bitmatrix` around without themselves becoming CUDA
translation units. The actual VRAM allocation is hidden behind an opaque pointer
(section 3).

Think of it as the READv2 twin of `DeviceDecodeResult`: same PIMPL shape, same
"host owns a handle, device owns the bytes" contract, built for the same reason.

---

## 2. What the bit-matrix is, and why it stays resident

READv2 estimates relatedness by walking the genome in fixed non-overlapping
windows of SNPs and, for every unordered pair of samples, counting the fraction of
sites in each window where the two samples' pseudo-haploid hardcalls disagree. That
is an all-pairs sweep: for `N` samples it visits `C(N,2)` pairs, and each pair reads
the whole genome.

Doing that from packed 2-bit genotypes on the host would be hopeless. Instead the
genome is packed once, on the GPU, into a `[sample x SNP-window]` bit-matrix and
left resident in VRAM. Then the mismatch sweep runs against that resident matrix.
The host never touches the packed bits — it streams genotype chunks *in* through the
backend and reads four small per-pair reductions *out*, and nothing in between.

A `Readv2Bitmatrix` is the handle to that resident matrix. Its whole reason to exist
is to keep the VRAM buffer alive across three backend calls — allocate, pack
(repeatedly, one streamed chunk at a time), then sweep — without ever materializing
the bytes on the host.

The on-device cell layout (the 128-bit allele/valid word, the LSB-first bit
convention, word-aligned windows) is documented separately in
`docs/reference/src_device_cuda_readv2_layout.cuh.md`; this header only carries the
*geometry* of that layout, not the bit format.

---

## 3. The PIMPL: a CUDA-free handle over a CUDA buffer

`Readv2Bitmatrix` names no CUDA type. The buffer that actually holds the bits — a
`DeviceBuffer<Readv2Word>` — lives in a forward-declared nested `struct Impl`, owned
through a `std::unique_ptr<Impl> impl`. `Impl` is defined only in the CUDA-side
`readv2_bitmatrix_impl.cuh`, and the special members (`~`, move ctor, move assign)
that need a complete `Impl` are defined out-of-line in `readv2_bitmatrix.cu`. That
is the standard PIMPL split, and it is exactly what lets this header stay on the
CUDA-free seam while still owning device memory.

An `Impl` is present only when the matrix has been allocated by the backend. A
default-constructed `Readv2Bitmatrix` (or one for a degenerate run — `N < 2`, empty
window) has a null `impl` and is `empty()`. The backend guards on `impl` before
packing or sweeping, so a null-impl handle is a safe no-op, not a crash.

The type is **move-only**: the copy constructor and copy assignment are deleted, and
move construction / move assignment are `noexcept`. That's the right contract for
something that owns a single VRAM allocation — you can hand ownership along (the
driver constructs it from `readv2_alloc_bitmatrix` and later passes it by reference
to pack and sweep), but you can't accidentally deep-copy gigabytes of device memory.

---

## 4. The window geometry (the public host scalars)

Everything about the matrix's shape is carried in plain host-side scalars on the
handle, so any host code can reason about sizes without touching the device:

| Field | Meaning |
|---|---|
| `n_samples` | `N`, the number of samples (matrix rows). |
| `window_snps` | The window width in SNPs — the READv2 window the mismatch fraction is averaged over. |
| `m0` | The total number of SNP sites the sweep runs over (the post-filter genome length). |
| `wpw` | Words per window = `ceil(window_snps / 64)`. Each window occupies a whole number of 64-bit plane words. |
| `n_win` | Windows tiling the SNP axis = `ceil(m0 / window_snps)`. The last window may be partial. |
| `words_per_sample` | `W = n_win * wpw`, the stride of one sample's row in the buffer. |
| `device_id` | The CUDA ordinal the buffer lives on (`-1` until allocated). |

These are the same quantities the layout helpers (`readv2_wpw`, `readv2_n_win`,
`readv2_words_per_sample` in `readv2_layout.cuh`) compute; the backend fills them in
at allocation time and they stay fixed for the life of the handle. The total buffer
is `n_samples * words_per_sample` cells of `Readv2Word`.

Two geometry decisions are load-bearing and worth calling out because the rest of
the pipeline leans on them:

- **Windows are word-aligned.** Window `g` owns words `[g*wpw, g*wpw+wpw)`. Because a
  window never starts mid-word, the mismatch kernel can popcount a whole window
  without ever splitting a count across a window boundary. That is the single biggest
  correctness simplifier in the subsystem.
- **The last window is kept even when partial** (`n_win` rounds up). The unused high
  bits of a partial window's tail word stay `valid = 0`, so they contribute nothing
  to any count — the padding is masked by the layout, not by special-casing.

`empty()` returns true when `n_samples <= 0` or `words_per_sample <= 0` — i.e. a
handle with no meaningful matrix. It is `noexcept` and touches only host scalars.

---

## 5. `Readv2Pairs`: the four per-pair reductions

`Readv2Pairs` is what comes back from the all-pairs mismatch sweep. It is entirely
host-resident — four parallel arrays, each with one entry per unordered pair:

| Array | Per-pair meaning |
|---|---|
| `sum_p0` (double) | Sum over used windows of that window's mismatch fraction P0. |
| `sum_p0_sq` (double) | Sum of the squared per-window P0 — feeds the window-jackknife standard error. |
| `n_win_used` (int) | Number of windows that actually contributed (had at least one both-valid overlapping site). |
| `tot_comp` (int64) | Total number of both-valid overlapping sites compared across the whole genome. |

### The pair indexing (the one contract a caller must honor)

The arrays are indexed by a flat unordered-pair rank `r` in `[0, C(N,2))`, and the
mapping is fixed:

```
r = C(j,2) + i        for the pair (i, j) with i < j
```

This is the standard lower-triangular enumeration, and it is shared bit-for-bit with
the on-device unrank the mismatch kernel uses and with the host mirror
`readv2_unrank_pair` in the core driver. Any code that reads these arrays must use
the same `r = C(j,2)+i` convention to recover which pair a row belongs to — that
agreement is the whole interface contract between the device sweep and the
host-side reduction/emit layer.

### What the emit layer does with them

The reduction/emit layer turns these raw sums into reported quantities. It forms the
per-pair mean mismatch `P0_mean = sum_p0 / n_win_used`, derives the window-jackknife
standard error from `sum_p0_sq`, and renames the two counts for the output table:
`n_win_used -> n_windows` and `tot_comp -> n_overlap_sites`. A pair with
`n_win_used == 0` (no overlapping sites anywhere — e.g. two samples that never share
a called site) is excluded from the background and reported as having no estimate,
rather than dividing by zero.

---

## 6. How the three backend calls use the handle

The handle threads through the three READv2 backend seams (declared on
`ComputeBackend`, implemented in `cuda_backend_readv2.cu`) in a fixed order:

1. **`readv2_alloc_bitmatrix(N, window_snps, m0)`** returns a fresh handle: it fills
   in the geometry scalars, allocates the `N * words_per_sample` buffer, and — this
   part is load-bearing — **zeroes it**. Every never-written padding cell (window
   tails, the genome tail beyond `m0`) must stay `valid = 0`, and the zero-fill is
   what guarantees that.
2. **`readv2_pack_chunk(bits, ...)`** is called repeatedly, once per streamed
   individual-major genotype chunk, each packing its SNP slice into the resident
   matrix. It takes `bits` by mutable reference and no-ops if `impl` is null.
3. **`readv2_mismatch(bits, n_pairs, tiled)`** runs the all-pairs windowed-mismatch
   reduction against the (now fully packed) matrix and returns the `Readv2Pairs`.

Because the buffer is resident the whole time, packing is incremental and the sweep
reads a matrix that's already entirely on the device. `n_pairs` passed to the sweep
is `C(N,2)`, matching the `Readv2Pairs` array length.

---

## 7. Edge cases and invariants

- **Degenerate runs produce an empty handle, not an error.** `N < 2`, a non-positive
  `window_snps`, or `m0 <= 0` all yield a handle with null `impl` and `empty() ==
  true`. Pack and sweep both guard on `impl`, so calling them on such a handle is a
  safe no-op that returns an empty `Readv2Pairs`.
- **Geometry is immutable after allocation.** The public scalars are set once by
  `readv2_alloc_bitmatrix` and are not meant to be edited afterward — the resident
  buffer's size and stride were computed from exactly those values.
- **The handle owns exactly one device allocation.** Move-only semantics keep that
  ownership single; a moved-from handle is left with a null `impl` and is safe to
  destroy.
- **This header carries geometry, not bit format.** The 128-bit `Readv2Word` cell,
  the allele/valid bit planes, and the LSB-first SNP-to-bit mapping live in
  `readv2_layout.cuh` (and its reference doc); nothing here needs to know how a cell
  is laid out, only how many of them there are.
- **The counts are sized for the genome, not the window.** `tot_comp` is a 64-bit
  count because it accumulates both-valid site comparisons across the entire genome
  for a pair, which can exceed 32 bits; `n_win_used` is a 32-bit window count, which
  cannot.

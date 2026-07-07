# `cuda_backend_readv2.cu` reference

## 1. Purpose

`src/device/cuda/cuda_backend_readv2.cu` is the CUDA-side body file for the three
`CudaBackend` methods that make up READv2's device path. It is deliberately thin:
each method is a few lines of allocation, host↔device copy, and a single kernel
launch. The interesting math lives in the two kernels next door
(`readv2_pack_kernel.cu` and `readv2_mismatch_kernel.cu`) and in the bit-matrix
geometry helpers (`readv2_layout.cuh`); this translation unit is the glue that wires
them into the backend and manages the memory around them.

READv2 measures pairwise genetic relatedness by, for every unordered pair of
samples, counting how often their pseudo-haploid alleles *disagree* across the SNP
axis, window by window. Doing that on the GPU needs three seams, and this file
supplies all three as out-of-line `CudaBackend` members:

1. `readv2_alloc_bitmatrix` — reserve and zero the resident `[sample × SNP-window]`
   packed bit-matrix that stays in VRAM for the whole run.
2. `readv2_pack_chunk` — stage one streamed genotype chunk to the device and pack it
   into that bit-matrix.
3. `readv2_mismatch` — run the all-pairs windowed-mismatch reduction over the packed
   matrix and hand back four per-pair arrays on the host.

The core driver (`src/core/readv2/readv2.cpp`) calls them in exactly that order:
allocate once, pack the whole SNP axis (currently one chunk — see section 4), then
sweep. It never touches a packed bit; the packed representation lives and dies on the
device, and only the four reduced arrays ever cross back to the host.

These three methods are split out of the main `CudaBackend` translation unit for the
same reason as the decode path — they include the READv2-private `.cuh` headers
(`readv2_layout.cuh`, the pack/mismatch kernel launchers, and the bit-matrix `Impl`),
so keeping them in their own TU stops those device-only headers from leaking into the
rest of the backend's compile.

---

## 2. The resident bit-matrix, in one paragraph

Everything here operates on the packed layout defined in `readv2_layout.cuh`. Each
`(sample, 64-SNP block)` cell is one 128-bit `Readv2Word` with two bit-planes — an
`allele` plane (the single pseudo-haploid allele) and a `valid` plane (1 exactly
where the site is a genuine 0/2 hardcall for that sample). SNPs are tiled into
non-overlapping, **word-aligned** windows: `wpw = ceil(window_snps / 64)` words per
window, `n_win = ceil(m0 / window_snps)` windows, and `words_per_sample = n_win *
wpw`. The whole matrix is `n_samples * words_per_sample` words, laid out
sample-major. Those three geometry scalars are computed once here (from the
`readv2_wpw` / `readv2_n_win` helpers) and stored on the `Readv2Bitmatrix` handle so
the pack and mismatch launchers can reuse them without recomputing.

---

## 3. Seam 1 — `readv2_alloc_bitmatrix`: allocate, and the load-bearing zero

`readv2_alloc_bitmatrix(n_samples, window_snps, m0)` builds a fresh
`Readv2Bitmatrix`, fills in its host-side geometry (`wpw`, `n_win`,
`words_per_sample`, `device_id`), allocates the `DeviceBuffer<Readv2Word>` that backs
it, and — the critical step — **zeroes every word** with a `cudaMemsetAsync` before
returning.

That zero is not hygiene, it is correctness. Not every bit in the matrix gets
written by the pack kernel: the last word of every window carries only
`window_snps % 64` real SNPs (its high bits are padding), and the final window of the
genome can run past `m0`. Those never-written padding bits must read back as
`valid = 0` so the mismatch kernel's both-valid AND mask silently ignores them. The
memset establishes that invariant up front, so any bit the pack kernel doesn't touch
is already a valid-0 "no data here" — window tails and the genome tail alike. This is
the layout's biggest correctness simplifier: because padding is pre-zeroed valid-0,
no kernel ever has to special-case a window edge or the end of the genome.

**Empty / degenerate inputs.** If any of `n_samples`, `window_snps`, or `m0` is
non-positive the method returns the default-constructed handle immediately, without
allocating. Its geometry scalars stay zero and `empty()` reports true, so a caller
that passes a degenerate shape gets a valid empty handle rather than a zero-byte
allocation or a crash — and the two later seams both short-circuit on that same empty
handle (they check `bits.impl`).

The allocation is `make_unique`'d into the opaque `Impl` (the PIMPL owner defined in
`readv2_bitmatrix_impl.cuh`), so the CUDA-free `Readv2Bitmatrix` handle in
`device/readv2_bitmatrix.hpp` never names a device type.

---

## 4. Seam 2 — `readv2_pack_chunk`: stage and pack one genotype chunk

`readv2_pack_chunk` takes one individual-major, 2-bit-packed genotype chunk sitting
in host memory and folds it into the resident bit-matrix at its correct SNP offset.
It:

1. Computes the chunk's total byte size (`n_samples * chunk_bytes_per_record`),
2. Allocates a device staging buffer and copies the chunk up with `h2d_async`,
3. Launches `launch_readv2_pack`, which fans each sample's 2-bit codes out into the
   `allele` / `valid` bit-planes at the right words, and
4. Synchronizes the stream.

The `snp0` / `snp_count` arguments say where in the global SNP axis this chunk lands;
the launcher uses `m0` and `window_snps` to place the bits and to pad the last window
correctly. Because windows are word-aligned and a chunk is always a whole number of
windows (except possibly the final one that ends at `m0`), a chunk never has to
straddle a word between two calls — each call writes complete words.

**Why the synchronize matters here.** The stream sync on step 4 is load-bearing for
lifetime, not just ordering: the staging buffer `d_chunk` is a local, and it frees
when the method returns. The kernel reads from it, so the method must block until the
launch has consumed the buffer before letting it go out of scope. Synchronizing per
chunk is what makes the host-side `d_chunk` safe to destroy.

**One chunk today, a loop by design.** The driver currently calls this exactly once —
it reads the whole SNP axis as a single tile because the canonical-tile readers only
support a SNP prefix beginning at index 0 (a nonzero `snp_begin` gather isn't
implemented yet). The 1240K panel's per-sample SNP axis is small and bounded, and the
device matrix holds all `m0` SNPs resident regardless, so one tile is both correct and
memory-safe. The signature is nonetheless written as "pack the chunk at `snp0`" so
that the moment the reader grows a nonzero-`snp_begin` gather, this seam becomes true
SNP-tile streaming with no change — each streamed chunk is already a whole number of
windows.

**Empty guard.** If the handle has no `Impl` (an empty matrix from section 3), or
`n_samples`/`snp_count` is non-positive, the method returns without copying or
launching.

---

## 5. Seam 3 — `readv2_mismatch`: the all-pairs windowed reduction

`readv2_mismatch(bits, n_pairs, tiled)` is where the actual READv2 statistic gets
computed. For every unordered sample pair it runs the AND-mask / XOR / `__popcll`
reduction over the packed matrix and returns a `Readv2Pairs` with four host-resident
arrays, one entry per pair:

| Field | Type | What it is | Downstream role |
|---|---|---|---|
| `sum_p0` | `double` | Σ over windows of (per-window mismatch / comparable) | numerator of `P0_mean` |
| `sum_p0_sq` | `double` | Σ over windows of that ratio squared | feeds the window-jackknife SE |
| `n_win_used` | `int` | count of windows with any comparable sites | becomes `n_windows` |
| `tot_comp` | `int64` | total comparable sites across all windows | becomes `n_overlap_sites` |

The reduction/emit layer forms `P0_mean = sum_p0 / n_win_used` and the jackknife SE
from `sum_p0_sq`; this file just produces the raw sums. Pairs are indexed in the
standard packed-triangle order — for the pair `i < j`, `r = C(j,2) + i`.

The body is mechanical: it allocates one device output buffer per array (each of
length `n_pairs`), launches `launch_readv2_mismatch` with the matrix geometry and the
four output pointers, then copies all four arrays back with `d2h_async` and
synchronizes once. The output vectors are pre-sized and zero-filled before the copies,
so an empty or short result still leaves the caller a well-formed, all-zero array
rather than an uninitialized one.

**The `tiled` switch.** The `tiled` bool is passed straight through to the launcher,
which chooses between two kernels behind one entry point (mirroring the D-statistic
kernel's dual): a baseline one-thread-per-pair reduction that is the correctness gate,
and a shared-memory tiled reduction that caches sample rows per window for throughput.
Both must produce bit-identical reductions; `tiled` only selects which one runs. The
driver threads this through from `opts.tiled`.

**Empty guard.** If the matrix has no `Impl`, or `n_pairs <= 0`, the method returns a
default-constructed (empty) `Readv2Pairs` without allocating or launching.

---

## 6. Contracts and invariants

A few rules hold across all three seams and are worth stating plainly:

- **The host never sees a packed bit.** The packed representation is created, filled,
  and reduced entirely on the device. Only the four reduced arrays of section 5 ever
  travel host-ward. This is what lets the `Readv2Bitmatrix` handle stay CUDA-free.
- **The matrix is allocated once and lives across the sweep.** Alloc happens a single
  time; every pack call writes into the same resident buffer; the mismatch sweep reads
  it in place. Nothing re-uploads the packed matrix.
- **Every method targets the backend's own device.** Each begins with
  `guard_device()` (a `cudaSetDevice` to `device_id_`), so the seams are safe to call
  on a multi-device backend without the caller having to set the current device first.
- **Every method synchronizes before returning.** Each of the three ends with a
  `cudaStreamSynchronize`. That makes them blocking, self-contained calls: alloc
  returns only once the zero has landed, pack returns only once its staging buffer has
  been consumed (section 4), and mismatch returns only once the four arrays are on the
  host. The seams are correct-by-construction rather than fast-by-overlap — READv2's
  time is dominated by the mismatch sweep, not by these copies.
- **Padding is valid-0, always.** Guaranteed by the alloc-time memset (section 3) and
  relied on by every window-tail and genome-tail read. No kernel special-cases an
  edge.
- **Every method NVTX-brackets itself** (`readv2_alloc_bitmatrix`,
  `readv2_pack_chunk`, `readv2_mismatch`) so the three seams show up as named ranges
  in a profile trace.

---

## 7. Edge cases

- **Degenerate shape → empty handle, no allocation.** A non-positive `n_samples`,
  `window_snps`, or `m0` makes alloc return an empty handle (section 3). Both later
  seams see `bits.impl == nullptr` and no-op, so a degenerate run threads through
  cleanly and produces an empty result rather than faulting.
- **Zero pairs → empty result.** `n_pairs <= 0` makes mismatch return an empty
  `Readv2Pairs` without touching the device.
- **The final window and the genome tail.** These carry fewer than `window_snps` real
  SNPs and fewer than 64 SNPs in their last word. The pack kernel writes only the real
  bits; the pre-zeroed padding does the rest. No caller has to trim `m0` to a window
  multiple.
- **Het / missing / diploid samples.** These have no encoding in the single-allele-bit
  layout. Missing (code 3) and het (code 1) sites are packed as `valid = 0` so the
  reduction skips them; a genuinely diploid sample is rejected upstream by the
  driver's ploidy gate before this file ever runs — this seam assumes pseudo-haploid
  hardcalls.

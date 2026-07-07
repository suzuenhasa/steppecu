# `readv2_pack_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/readv2_pack_kernel.cuh` is the one-function header that declares
`launch_readv2_pack` — the device-side entry point that fills the resident READv2
bit-matrix. READv2 is steppe's kinship / relatedness pass: for every pair of
samples it counts windowed allele mismatches, and to do that cheaply it first
packs the genotypes into the compact `[sample x SNP-window]` two-plane bit-matrix
described in `readv2_layout.cuh`. This launcher is the writer of that matrix; the
**mismatch kernel** is the reader.

The header is deliberately thin. It declares exactly one free function and the
contract every caller must honor, then gets out of the way. All the kernel
mechanics — the thread-per-word grid, the LSB-first bit building, the branch-free
padding, the coalescing argument — live in the paired `readv2_pack_kernel.cu` and
are written up in its own reference, `src_device_cuda_readv2_pack_kernel.cu.md`.
Read that one for *how* the packing happens; read this one for *what the launcher
promises* and *what a caller has to hand it*.

This is a `.cuh`, private to the `steppe_device` target. It pulls in
`readv2_layout.cuh` for the `Readv2Word` cell type and the window geometry, and
`<cuda_runtime.h>` for the stream handle. It never appears on the CUDA-free seam
that the host-only app translation units sit behind.

---

## 2. Where it sits in the pipeline

READv2 reads genotypes the same streamed, individual-major way the rest of steppe
does: one **chunk** of SNPs at a time, each chunk holding a block of samples'
2-bit codes. `launch_readv2_pack` is called once per chunk. It takes that chunk's
freshly-uploaded bytes and scatters them into the correct slice of the persistent,
device-resident bit-matrix `d_words`, which spans the whole genome. Chunk after
chunk lands in its own SNP-window range of the same matrix; when the last chunk is
packed, the full matrix is ready for the mismatch kernel to sweep.

So the launcher is a **partial writer**: each call fills the words for the SNP
range `[snp0, snp0 + snp_count)` and leaves every other sample-row region of
`d_words` untouched. It inverts `decode_af_kernel`, which collapses samples down to
per-population allele frequencies — here samples stay distinct and it is the SNP
axis that fans out, into the allele and valid bit planes.

---

## 3. The signature and its parameter contract

```cpp
void launch_readv2_pack(const std::uint8_t* d_chunk_packed,
                        std::size_t chunk_bytes_per_record,
                        int n_samples,
                        long snp0,
                        long snp_count,
                        long m0,
                        int window_snps,
                        Readv2Word* d_words,
                        long words_per_sample,
                        cudaStream_t stream);
```

Every argument carries a precondition the caller owns:

| Parameter | Meaning | The caller must guarantee |
|---|---|---|
| `d_chunk_packed` | This chunk's individual-major 2-bit bytes, already on the device. | A valid device pointer covering `n_samples * chunk_bytes_per_record` bytes — **this chunk only**, not the whole genome. |
| `chunk_bytes_per_record` | Per-sample byte stride inside the chunk = `ceil(snp_count / 4)`. | Matches the packing of `d_chunk_packed` exactly; four 2-bit codes per byte. |
| `n_samples` | Number of samples (rows of the bit-matrix). | The same sample count every chunk shares. |
| `snp0` | Global SNP index the chunk starts at. | A **multiple of `window_snps`** — chunks start on a window boundary so a window never straddles two chunks. |
| `snp_count` | SNPs in this chunk. | A multiple of `window_snps`, **except the final chunk**, whose tail may be shorter. |
| `m0` | Total SNPs tiled across the whole genome. | Constant across all chunks; used to clamp the genome-tail padding of the last window. |
| `window_snps` | SNPs per non-overlapping window. | Positive; the same value used to size `d_words` (`readv2_words_per_sample`). |
| `d_words` | The resident whole-genome bit-matrix, `n_samples * words_per_sample` cells. | Allocated for the full genome and persistent across chunks; the launcher writes only this chunk's slice. |
| `words_per_sample` | Row stride of `d_words` = `readv2_words_per_sample(m0, window_snps)`. | Computed from the *whole-genome* `m0`, not from `snp_count`. |
| `stream` | The CUDA stream the kernel launches on. | Any valid stream; the call is asynchronous on it. |

The pointers are raw and unowned — the launcher neither allocates nor frees. It
does not synchronize the stream; the launch is queued and returns. It runs no
host-to-device copy of its own: the chunk bytes are expected to already be
resident on the device before the call.

---

## 4. Why `snp0` must be window-aligned, and why `m0` is separate from `snp_count`

Two of the preconditions above are the load-bearing ones, and both exist to keep
the padding correct.

**`snp0` is a multiple of `window_snps`.** Windows are word-aligned and
non-overlapping (see `readv2_layout.cuh`): window `g` owns a fixed word range in
every sample row. If a chunk could begin mid-window, the same window's words would
be split across two `launch_readv2_pack` calls, and the shared high-bit padding of
its tail word would be ambiguous. Starting every chunk on a window boundary means
each window is packed start-to-finish by exactly one call, so the kernel can zero a
tail word's unused high bits once and be done.

**`m0` is passed even though the chunk only knows `snp_count`.** The very last
window of the genome may be partially filled — the genome may not divide evenly
into windows. The kernel needs the *global* SNP count `m0` to know where the real
SNPs stop, so it can clamp the last word to the genuine genome tail and leave the
padding SNPs beyond it with `valid = 0`. Deriving that from `snp_count` alone would
be wrong, because `snp_count` is a per-chunk quantity and the genome tail is a
whole-genome fact. This is exactly why the row stride `words_per_sample` is also a
whole-genome value: the slice a chunk writes must land at the right offset within
the full-genome matrix, so both must be computed from `m0`.

---

## 5. The bit-plane contract this launcher upholds

The launcher fills `d_words` to the layout the mismatch kernel depends on. For each
cell it writes:

- **`allele`** — the single pseudo-haploid allele bit per SNP, LSB-first (local SNP
  index `l` in `[0, 64)` maps to bit `l`). `0` for a ref-copy code of `0`, `1` for a
  ref-copy code of `2`. Meaningful only where the matching `valid` bit is set.
- **`valid`** — `1` exactly where the site is a genuine `0`/`2` pseudo-haploid
  hardcall for that sample. Everything else — a het (code `1`, which has no encoding
  in a 1-bit layout and is dropped), a missing call (code `3`), and any padding SNP
  beyond the window tail or the genome tail — is `valid = 0`.

Because the padding SNPs and the dropped calls all read `valid = 0`, the mismatch
kernel's both-valid AND doubles as the padding mask for free. That is the invariant
this writer exists to establish, and it is why the padding clamps of section 4 are
correctness, not cosmetics.

---

## 6. Edge cases and guarantees

- **No-op guards.** The launcher returns immediately, launching nothing, when
  `n_samples <= 0`, `snp_count <= 0`, or `window_snps <= 0`. An empty chunk is a
  clean no-op, not an error.
- **Grid overflow is checked, not silently truncated.** The word axis of the launch
  grid can be large; the launcher computes `gridDim.x` through the project's
  `grid_for_x` guard, which fails with a clear message
  ("reduce chunk size") rather than overflowing the CUDA grid limit. A chunk sized
  past that limit is a caller error surfaced at launch, not corrupted output.
- **Asynchronous.** The kernel is enqueued on `stream` and the call returns without
  synchronizing. The caller owns ordering — the chunk's upload must complete before
  the pack, and every chunk's pack must complete before the mismatch sweep reads the
  matrix.
- **Slice-local writes.** A call touches only the cells for SNP range
  `[snp0, snp0 + snp_count)`; it never clears or writes any other region of
  `d_words`. The whole matrix is correct only once every chunk covering `[0, m0)`
  has been packed.

---

## 7. Relationship to the rest of READv2

| File | Role |
|---|---|
| `readv2_layout.cuh` | Defines `Readv2Word` and the window geometry helpers this header consumes; the shared source of truth both the pack and mismatch kernels agree on. |
| `readv2_pack_kernel.cu` | Implements the kernel this header declares — the thread-per-word packing, documented in `src_device_cuda_readv2_pack_kernel.cu.md`. |
| `readv2_mismatch_kernel.cuh` / `.cu` | The reader: sweeps the packed matrix to count windowed mismatches for every sample pair. |
| `cuda_backend_readv2.cu` | The device backend that streams chunks in and calls `launch_readv2_pack` once per chunk, then the mismatch launcher. |

The division of labor is the point: this header is the stable contract, the `.cu`
is the mechanism, and the layout header is the shared vocabulary. A caller who
honors section 3's preconditions can treat the packing itself as a black box.

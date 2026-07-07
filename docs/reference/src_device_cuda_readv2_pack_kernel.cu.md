# `readv2_pack_kernel.cu` reference

## 1. Purpose

`src/device/cuda/readv2_pack_kernel.cu` is the GPU kernel that fills the resident
READv2 bit-matrix. It takes one streamed chunk of individual-major 2-bit genotype
bytes — the same on-disk packing every other decoder reads — and, sample by sample,
turns each sample's run of SNP codes into the packed `allele` / `valid` bit planes
that the windowed-mismatch kernel later popcounts.

It is the *inverse* of `decode_af_kernel`. That kernel collapses many samples down
into one per-population allele frequency; this one does the opposite — it keeps every
sample distinct and fans the SNP axis out into individual bits, one bit per SNP per
sample. The two share the exact same genotype-unpacking primitive
(`core::genotype_code`, from `decode_af.hpp`), so the codes they read can never
drift apart.

The bit-matrix layout it writes into — the 128-bit `Readv2Word` cell, the word-
aligned SNP windows, the LSB-first bit convention — is defined and documented in
`readv2_layout.cuh` (see `docs/reference/src_device_cuda_readv2_layout.cuh.md`).
This file is the *writer* for that layout; the mismatch kernel is the reader. The
host-side wrapper `launch_readv2_pack` and the one `__global__` kernel are the whole
translation unit, and it is private to the `steppe_device` target.

---

## 2. The thread geometry, and why it is fully coalesced

The kernel launches **one thread per (sample, chunk-local output word)**. The block
is `64 x 4` (`kPackBlockX = 64` walks output words along x, `kPackBlockY = 4` walks
samples along y), and the grid is sized to cover `chunk_out_words` words across x and
`n_samples` samples across y.

That axis assignment is deliberate and is the kernel's main performance decision.
Threads adjacent in x belong to the *same sample* and *adjacent output words*, which
means:

- **Reads are coalesced.** Adjacent words cover adjacent 64-SNP runs of one sample,
  which live in adjacent bytes of that sample's record. Neighbouring threads pull
  neighbouring bytes.
- **Writes are coalesced.** Adjacent output words are adjacent 16-byte `Readv2Word`
  cells in the bit-matrix, so neighbouring threads store to neighbouring cells.

Both ends of the memcpy-shaped work are contiguous, so the kernel gets full memory
throughput on both the load and the store. The `__restrict__` qualifiers on the
`chunk` input and the `d_words` output tell the compiler the two never alias, which
they cannot — one is a read-only host-uploaded chunk, the other is the resident
matrix.

---

## 3. Decomposing a thread's word index into a window and a word-in-window

Each thread owns a *chunk-local* output-word index `wl`. The first job is to turn
that flat index into the two coordinates the layout actually cares about:

```
g_local = wl / wpw          // which window within this chunk
wlocal  = wl % wpw          // which word within that window
```

`wpw` (words-per-window, `ceil(window_snps / 64)`) comes straight from
`readv2_wpw`. Because windows are word-aligned — window `g` owns exactly the word
range `[g*wpw, (g+1)*wpw)` — this division is exact and unambiguous. The chunk-local
window index is then lifted to a **global** window index by adding the chunk's window
offset:

```
g_global = snp0 / window_snps + g_local
```

This is where the contract that `snp0` is a multiple of `window_snps` (section 6)
earns its keep: `snp0 / window_snps` is the exact global index of the chunk's first
window, with no remainder to reason about. Windows never straddle a chunk boundary.

---

## 4. Window-relative SNP addressing (the one arithmetic trap)

The obvious way to find the first SNP of output word `wl` would be `wl * 64`. That is
**wrong**, and the code says so in a comment. The reason is padding: the last word of
every window can hold fewer than 64 real SNPs (when `window_snps` isn't a multiple of
64), and those unused high bits are padding that no later word "reclaims". So the SNP
axis is *not* a dense `wl*64` stream — there are gaps at every window tail.

The correct base is computed window-relative instead:

```
snp_base_global = g_global * window_snps + wlocal * 64
snp_base_local  = snp_base_global - snp0
```

The window contributes `g_global * window_snps` (windows *are* densely packed against
each other on the SNP axis), and within the window the word contributes `wlocal * 64`
real SNPs. Subtracting `snp0` re-expresses that as an offset into the chunk's own
byte record, which is what the read loop needs.

---

## 5. How many real SNPs a word holds (the three-way clamp)

A word nominally carries `kReadv2SnpsPerWord = 64` SNPs, but the actual count is
clamped down by two independent tails:

```
real = 64
real = min(real, window_snps - wlocal*64)   // the window tail
real = min(real, m0 - snp_base_global)       // the genome tail
```

- **The window tail** handles a `window_snps` that isn't a clean multiple of 64: the
  final word of each window holds `window_snps % 64` SNPs, and its high bits are left
  as padding.
- **The genome tail** handles the very last window of the whole dataset, which can be
  partial: `m0` is the total SNP count, so `m0 - snp_base_global` caps the last word
  at the true genome edge and stops the loop from reading past the record.

Whichever tail is tighter wins. Any bit position `l >= real` is simply never touched
by the loop, so it keeps whatever the bit-matrix was initialised to — which, by
contract, is zero (section 6). That is how padding stays `valid = 0` without the
kernel writing an explicit mask.

---

## 6. What actually gets encoded

For each real SNP `l` in `[0, real)`, the kernel unpacks that sample's 2-bit code and
folds it into the two bit planes, **LSB-first** — SNP local index `l` maps to bit `l`:

- The byte and in-byte position come from `local_snp = snp_base_local + l` via the
  shared `core::kCodesPerByte` (4 codes/byte) arithmetic, and the code itself from
  `core::genotype_code(rec[byte_idx], pos)` — the identical primitive the AF decoder
  uses.
- **Only genuine pseudo-haploid hardcalls survive.** A code of `0` (ref-allele copy)
  or `2` (alt-allele copy) sets the `valid` bit for position `l`; additionally, a `2`
  sets the `allele` bit. A het (code `1`) and a missing call (code `3`) set *nothing*
  — they drop out entirely. The 1-bit `allele` plane has no room to represent a het,
  so hets are simply not encoded (a documented scope decision in the layout header).
- `allele` bits are only meaningful where the matching `valid` bit is 1. Where a site
  is invalid, `allele` is left 0 but must never be read on its own — the mismatch
  kernel always gates on `valid` first.

The result is two 64-bit words, `allele` and `valid`, built up in registers across
the loop and written together.

---

## 7. Where the word lands (the output index)

The write target is computed in **global** coordinates, independent of which chunk
produced it:

```
out = s * words_per_sample + g_global * wpw + wlocal
d_words[out] = { allele, valid }
```

`s * words_per_sample` selects the sample's row; `g_global * wpw + wlocal` is that
sample's word offset for this global window and word. Because the index is fully
global, two different chunks writing to the same matrix never collide and never need
coordination — each `(sample, global word)` cell is written exactly once, by exactly
one thread, across the whole streaming pass. Packing is embarrassingly parallel and
order-independent.

---

## 8. The launch wrapper and its guards

`launch_readv2_pack` is the thin host entry point. It:

1. **Early-returns on a degenerate call.** If `n_samples <= 0`, `snp_count <= 0`, or
   `window_snps <= 0` it does nothing — a no-op chunk (for example, a streaming loop
   that ran out of SNPs) is a clean skip, not an error.
2. **Sizes the work.** `chunk_win = ceil(snp_count / window_snps)` is the number of
   windows this chunk covers, and `chunk_out_words = chunk_win * wpw` is the flat
   x-extent the kernel iterates.
3. **Caps the grid.** The x grid dimension goes through `core::grid_for_x`, which
   errors out with a clear message ("readv2 pack gridDim.x (word axis) exceeds
   kMaxGridX — reduce chunk size") if a single chunk is so large its word axis would
   overflow the CUDA grid-x limit. The sample (y) axis uses the ordinary `grid_for`.
   This is a real ceiling on per-chunk size and the remedy is to stream in smaller
   chunks.
4. **Launches and checks.** The kernel goes out on the caller's `stream`, followed by
   `STEPPE_CUDA_CHECK_KERNEL()` to surface a launch failure immediately.

The wrapper does **not** synchronize — that is the caller's job. In
`cuda_backend_readv2.cu`, `readv2_pack_chunk` uploads the chunk with an async H2D,
calls this launcher, and only then synchronizes, because the device chunk buffer must
stay alive until the kernel has consumed it.

---

## 9. Contracts and invariants

- **The matrix must be pre-zeroed.** This kernel only ever *sets* bits inside the
  `real`-SNP prefix of each word; it never clears anything. Every padding bit (window
  tails, the partial genome-tail window) relies on the allocation-time
  `cudaMemsetAsync` in `readv2_alloc_bitmatrix` having left it at zero. If the matrix
  weren't zeroed, stale `valid=1` bits in padding would corrupt every downstream
  popcount. This is load-bearing and is the reason the allocator zeroes.
- **`snp0` is a multiple of `window_snps`.** Chunks are cut on window boundaries, so
  `snp0 / window_snps` is exact and windows never straddle chunks (section 3). The
  header states this as the chunk contract.
- **`snp_count` is a multiple of `window_snps`, except for the last chunk.** The last
  chunk may end mid-window; the genome-tail clamp (section 5) absorbs the ragged end.
- **`m0` is the whole-dataset SNP total,** used only for the genome-tail clamp — it
  can be larger than `snp0 + snp_count` (there are more chunks to come) and the
  clamp still does the right thing, since `snp_base_global` for this chunk's words is
  always within `[snp0, snp0+snp_count)`.
- **Each cell is written exactly once** across the full streaming pass (section 7),
  so no atomics, no read-modify-write, no inter-chunk ordering.

---

## 10. Edge cases

- **Threads past the end.** The grid is rounded up to whole blocks, so some threads
  get `wl >= chunk_out_words` or `s >= n_samples`. Both are guarded by an immediate
  return before any memory is touched.
- **`window_snps` an exact multiple of 64.** Then `wpw = window_snps / 64`, every
  word is full, and the window-tail clamp never fires — the padding logic costs
  nothing in the common aligned case.
- **`window_snps` smaller than 64.** `wpw = 1`, and the single word per window holds
  only `window_snps` SNPs; its high bits are padding held at zero.
- **The final partial window of the genome.** The genome-tail clamp caps `real` at
  `m0 - snp_base_global`, so the last word stops exactly at the true SNP count and
  its remaining high bits stay `valid = 0`.
- **An all-missing or all-het sample word.** Every code drops, so both `allele` and
  `valid` are written as 0 — a legitimately empty word that the mismatch kernel's
  both-valid AND will correctly treat as contributing nothing.

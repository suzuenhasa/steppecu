# `readv2_layout.cuh` reference

## 1. Purpose

`src/device/cuda/readv2_layout.cuh` is the one small header that fixes the
on-device memory layout for READv2 — steppe's kinship/relatedness pass. It
describes the packed `[sample x SNP-window]` bit-matrix that two kernels agree on:
the **pack kernel** (`readv2_pack_kernel.cu`) that writes the matrix, and the
**mismatch kernel** (`readv2_mismatch_kernel.cu`) that reads it back to count
windowed allele mismatches for every pair of samples.

It is deliberately tiny — one struct, one constant, three inline
`__host__ __device__` geometry helpers, and a lot of comment. That is the point. There is no logic to run here;
the file's whole job is to be the single, shared source of truth for how the bit
planes are shaped so the writer and the reader can never drift apart. Everything
downstream — the coalesced 128-bit loads, the branch-free padding mask, the
word-aligned window boundaries — falls out of the decisions written down here.

This is a `.cuh`, private to the `steppe_device` target. It is never seen on the
CUDA-free seam that the host-only app translation units live behind; the helpers
are marked `__host__ __device__` only so the launch code on the host and the
kernels on the device compute identical geometry from the same function bodies.

---

## 2. The cell: one 128-bit word, two bit planes

The unit of storage is `Readv2Word`, a 16-byte struct of two `uint64_t` fields:

```
struct Readv2Word {
    std::uint64_t allele;  // the pseudo-haploid allele bit, per SNP
    std::uint64_t valid;   // 1 iff this SNP is a genuine 0/2 hardcall
};
```

One `Readv2Word` holds **64 SNPs** for **one sample** — a "64-SNP block". The two
fields are two bit planes over the same 64 SNPs, laid out **AoS** (array of
structs) so the `allele` and `valid` words for a block sit next to each other in
memory. That adjacency is what lets the mismatch kernel pull both planes in a
single coalesced 128-bit (`ulonglong2`) transaction — it needs both to compute a
mismatch, and packing them together means one load, not two.

Bits are **LSB-first**: a SNP with local index `l` in `[0, 64)` lives in bit `l`
of each word (`1ULL << l`). The two planes mean, per SNP:

| Plane | Meaning |
|---|---|
| `valid` bit | `1` exactly when this site is a genuine 0/2 pseudo-haploid hardcall for this sample. `0` for anything that isn't a clean hardcall (see below). |
| `allele` bit | The single pseudo-haploid allele: `0` for reference-copy code 0, `1` for reference-copy code 2. **Only meaningful where `valid == 1`.** Where `valid` is 0 the `allele` bit is left 0 and carries no information. |

`kReadv2SnpsPerWord = 64` is the constant that ties "one word" to "64 SNPs". It is
the width of a `uint64_t` bit plane, and it is what `__popcll` (the 64-bit
population count) counts over in the mismatch kernel.

### What sets `valid = 0`

The pack kernel only lights a `valid` bit for the two clean pseudo-haploid
hardcalls, codes `0` and `2`. Everything else stays `valid = 0`:

- a **missing** call (2-bit code `3`),
- a **heterozygous** call (code `1`) — a het has no encoding in a 1-bit allele
  layout, so READv2 drops it (scope T10),
- any **padding** SNP beyond the real window or beyond the end of the genome.

Because all three collapse to the same `valid = 0`, the reader never has to tell
them apart — a site either contributes to a comparison or it doesn't.

---

## 3. Windows, and why they are word-aligned

READv2 does not compute one genome-wide mismatch ratio; it works in
**non-overlapping SNP-count windows** and averages a per-window ratio across
windows. So the SNP axis is tiled into windows of `window_snps` SNPs each, and the
layout makes one decision that simplifies everything downstream: **windows start on
a word boundary.**

Window `g` owns a contiguous run of words `[g*wpw, g*wpw + wpw)`, where `wpw`
("words per window") is `ceil(window_snps / 64)`. Windows never share a word.

The catch this buys off is the window tail. When `window_snps` is not a multiple of
64, the last word of a window carries only `window_snps % 64` real SNPs; its unused
high bits are padding. Those padding bits are written with `valid = 0`, exactly like
missing/het/off-genome SNPs. That single fact — **padding is `valid = 0`, and
windows never straddle a word** — is called out in the source as "the single biggest
correctness simplifier", and it earns that:

- The mismatch kernel's per-window loop just runs `wl` from `0` to `wpw` and
  popcounts each word. It **never splits a popcount across a window edge**, because
  a word never spans two windows.
- The both-valid AND (`wi.valid & wj.valid`) that selects sites comparable in both
  samples **doubles as the padding mask** for free: padding bits are `valid = 0` in
  at least one operand, so they contribute nothing to either the comparable count or
  the mismatch count. There is no separate masking step and no special-casing of the
  last word (scope T2/T5).

So the alignment costs a little storage (the padding bits in each window's tail
word) and in return removes every boundary branch from the hot inner loop.

---

## 4. The geometry helpers

Three inline `__host__ __device__` functions compute the layout's shape. They are the arithmetic
that both the host launch code and the device kernels use so the two sides always
agree on where a given `(sample, window, word)` lands.

| Helper | Returns | Formula |
|---|---|---|
| `readv2_wpw(window_snps)` | words per window | `ceil(window_snps / 64)` |
| `readv2_n_win(m0, window_snps)` | number of windows tiling the SNP axis | `ceil(m0 / window_snps)` — the partial last window is **kept**, not dropped |
| `readv2_words_per_sample(m0, window_snps)` | total words in one sample's row | `n_win * wpw` |

`m0` is the SNP count of the genome (the count that survived any upstream QC).
`readv2_words_per_sample` is the stride between one sample's row of the bit-matrix
and the next: sample `s`'s first word lives at flat index `s * words_per_sample`,
so the full matrix is `n_samples * words_per_sample` `Readv2Word`s laid out
sample-major. Within a sample's row, window `g`'s word `wlocal` is at
`g * wpw + wlocal`.

Note the two ceilings are independent and both round **up**. `readv2_n_win` keeps a
short final window (fewer than `window_snps` real SNPs), and `readv2_wpw` keeps a
short final word inside every window (fewer than 64 real SNPs). The padding created
by each rounding is what `valid = 0` covers in section 3.

---

## 5. Contracts and invariants

These are the promises the writer makes and the reader relies on. They are not
checked at runtime here — the header is header-only geometry — so they are contract,
enforced by the two kernels being written against this one document.

1. **The reader computes geometry from the same helpers the writer does.** Both
   `launch_readv2_pack` and `launch_readv2_mismatch` derive `wpw`, `n_win`, and
   `words_per_sample` from the functions above rather than from independently
   re-derived arithmetic. This is the whole reason the helpers live in a shared
   header instead of in either `.cu`.

2. **Flat layout is sample-major, then window, then word within window.** A cell's
   flat index is `s * words_per_sample + g * wpw + wlocal`. The pack kernel writes to
   exactly this index; the mismatch kernel's row base for sample `s` is
   `d_words + s * words_per_sample`, and it strides windows by `wpw`.

3. **`allele` is only defined where `valid == 1`.** Consumers must AND with `valid`
   (their own and the other sample's) before trusting an `allele` bit. The mismatch
   kernel does exactly this: `mism = (wi.allele ^ wj.allele) & (wi.valid & wj.valid)`
   — the XOR of alleles is only counted at sites valid in both samples.

4. **Padding bits are `valid = 0`.** Every bit past the real SNP count of a word —
   whether it is a window tail, a genome tail, a dropped het, or a missing call — has
   `valid = 0` and `allele = 0`. Nothing downstream may assume a full 64 valid bits in
   any word, including the very first.

5. **One word never spans two windows.** Window boundaries fall on word boundaries,
   so any popcount over a whole word belongs to exactly one window.

---

## 6. Edge cases

- **Empty / degenerate inputs.** The helpers are pure integer arithmetic and are
  never called with the zero guards inline — the launch functions bail out early
  (`n_samples <= 0`, `snp_count <= 0`, `window_snps <= 0`) before computing geometry,
  so the ceilings are only ever evaluated on positive inputs. A `window_snps` of 1
  gives `wpw = 1` (one word holding a single valid bit); a `window_snps` that is an
  exact multiple of 64 gives a tail word with no padding.

- **A window larger than 64 SNPs** simply spans multiple words (`wpw > 1`); the
  per-window loop just runs longer. Only the very last word of that window can carry
  padding. (The mismatch kernel's tiled path additionally caps the per-window shared-
  memory footprint and falls back to its baseline kernel when a window is too wide to
  stage in shared memory — but that is the kernel's concern, not the layout's.)

- **Overflow.** The helpers return `long` for the window count and words-per-sample
  so a large genome (`m0` in the tens of millions of SNPs) times the window count
  does not overflow a 32-bit `int`. Callers keep the flat cell index and pair rank in
  `long` / `long long` for the same reason. `readv2_wpw` returns `int` deliberately —
  a single window's word count is small — but is widened to `long` at every
  multiplication site.

- **The `allele = 0` padding is not a "reference" call.** It is tempting to read a
  padding word's `allele = 0` as "everyone is reference here". It is not — the paired
  `valid = 0` means the site is invisible to every comparison. Reading `allele`
  without its `valid` gate would silently turn padding and missing data into fake
  agreement.

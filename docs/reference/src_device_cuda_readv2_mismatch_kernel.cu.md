# `readv2_mismatch_kernel.cu` reference

## 1. Purpose

`src/device/cuda/readv2_mismatch_kernel.cu` is the GPU heart of READv2 — the
all-pairs, windowed **mismatch reduction**. Given the packed
`[sample x SNP-window]` bit-matrix that the pack kernel leaves resident on the
device, it walks every unordered pair of samples and, for each pair, measures how
often the two samples *disagree* at sites where they can be compared. That
per-pair disagreement rate is READv2's raw signal for relatedness: closely related
individuals mismatch less often than unrelated ones.

For every pair it emits four reductions that the host later turns into a
`P0_mean` and a standard error:

| Output | Type | Meaning |
|---|---|---|
| `sum_p0` | `double` | Sum over windows of each window's `mismatch / comparable` ratio. |
| `sum_p0_sq` | `double` | Sum of that ratio *squared* — the raw material for the SE. |
| `n_win_used` | `int` | How many windows had at least one comparable site (becomes `n_windows`). |
| `tot_comp` | `long long` | Total comparable sites across all windows (becomes `n_overlap_sites`). |

The host (`src/core/readv2/readv2.cpp`) divides `sum_p0` by `n_win_used` to get a
pair's `P0_mean`, compares each pair against a background (the median or mean of
the surviving pairs), and uses `sum_p0_sq` for the z-score. This file's whole job
is producing those four numbers per pair, correctly and fast. It is steppe's
first `__popc`-family kernel — the reduction is fundamentally a population count
over bit-masks.

This file is private to the `steppe_device` target. Its only public surface is the
one launcher `launch_readv2_mismatch`; the two kernels behind it live in an
anonymous namespace.

---

## 2. The bit layout it reads

The reduction only makes sense against the cell format defined in
`readv2_layout.cuh`. Each `(sample, 64-SNP block)` is one 128-bit `Readv2Word`
with two bit planes packed together:

- **`allele`** — the single pseudo-haploid allele bit per SNP (0 for the
  ref-copy-0 hardcall, 1 for ref-copy-2). Meaningful only where `valid` is 1.
- **`valid`** — 1 exactly where this sample has a genuine 0/2 hardcall at that
  SNP. Missing calls, hets (dropped in a 1-bit layout), and padding SNPs beyond
  the real genome all set `valid = 0`.

Two geometry facts make the kernel simple and are load-bearing throughout:

1. **Windows are word-aligned.** A window of `window_snps` SNPs owns exactly
   `wpw = ceil(window_snps / 64)` consecutive words. No window boundary ever falls
   in the middle of a 64-bit word, so the kernel never has to split a popcount
   across a window edge. The last word of a window carries `window_snps % 64` real
   SNPs; its unused high bits are simply `valid = 0`.
2. **`valid` doubles as the padding mask.** Because every padding bit (end of a
   window, end of the genome, missing site, het) is `valid = 0`, the "both samples
   valid here" AND that section 3 computes automatically excludes all of them. The
   kernel needs no separate padding logic.

---

## 3. The per-window primitive (the whole point)

Everything reduces to three bitwise ops over one aligned word pair, in
`word_counts`:

```
comp = Wi.valid & Wj.valid;              // sites valid in BOTH samples
mism = (Wi.allele ^ Wj.allele) & comp;   // differing alleles among those
comparable += __popcll(comp);
mismatches += __popcll(mism);
```

- **`comp`** is the both-valid AND — the sites where the pair can actually be
  compared. Per section 2 this also masks away all padding.
- **`mism`** XORs the allele planes (1 wherever the two disagree) and re-masks with
  `comp`, so a disagreement only counts where *both* samples are valid. Masking the
  XOR is essential: an allele bit is meaningless where `valid = 0`, so an unmasked
  XOR would count garbage.
- **`__popcll`** — the 64-bit population count, never the 32-bit `__popc`. The words
  are 64 bits wide; using `__popc` would silently ignore the top 32 SNPs of every
  word.

The order — AND first, then popcount the *masked* word — is the correctness
contract. Popcounting a raw plane would count invalid or padded sites.

---

## 4. Per-window mean, not a genome-wide ratio

`reduce_pair` is where the four scalars for one pair come together. It loops over
all `n_win` windows; for each window it loops the window's `wpw` words, accumulates
`win_comp` and `win_mism`, and then — the key step — folds a **per-window ratio**
into the totals:

```
if (win_comp > 0) {
    p0 = win_mism / win_comp;   // this window's mismatch rate
    sum_p0    += p0;
    sum_p0_sq += p0 * p0;
    ++n_used;
    tot_comp  += win_comp;
}
```

`P0` is the **mean over windows** of each window's own `mismatch/comparable`, not
one big genome-wide `total_mismatch / total_comparable`. That is why the reduction
flushes `win_comp` / `win_mism` at every window boundary and forms a fresh ratio
per window. The distinction is deliberate and matters: window-averaging is what
makes READv2's estimate robust to uneven coverage across the genome.

**A window with zero comparable sites contributes nothing.** The `win_comp > 0`
guard skips it entirely — no ratio, no increment to `n_used`. This is both a
divide-by-zero guard and the definition of `n_win_used`: only windows the pair
could actually be scored on count. A pair with no overlap anywhere ends with all
four outputs at zero, which the host reads as "no usable data for this pair."

---

## 5. Two kernels behind one launcher

`launch_readv2_mismatch` picks between two kernels that compute the *same* four
numbers — mirroring the dual-kernel shape of `dstat_kernel`:

- **(a) Baseline — `readv2_mismatch_direct_kernel`.** One thread per pair. Each
  thread unranks its flat pair id `r` into `(i, j)`, points at the two samples'
  row bases, and calls `reduce_pair`. Dead simple, always correct — this is the
  correctness gate the tiled path is checked against.
- **(b) Tiled — `readv2_mismatch_tiled_kernel`.** A block owns a `TS x TS`
  tile-pair of samples (`TS = kTileSamples = 32`, so `32*32 = 1024` threads fit one
  block). For each window the block *cooperatively loads* that window's rows of
  both tiles into shared memory once, then every thread reads its pair's two rows
  out of shared. This is the dstat-style row-reuse: instead of each thread
  re-fetching sample rows from global memory, the block loads each row once and
  amortizes it across the whole tile. It is the throughput follow-up, not a
  different result.

Both write to the *same* global output slot for a given pair, indexed by the pair's
upper-triangular rank (section 6), so the two kernels are interchangeable.

### How the tiled kernel keeps its arithmetic correct

- **Tile-pair decode.** The block's flat id is walked down the upper-triangular
  row lengths (`n_tiles`, `n_tiles-1`, …) to recover `(tileA <= tileB)`. Only
  tile-pairs on or above the diagonal are enumerated.
- **Active mask.** A thread is active only when its global `(I, J)` are both in
  range and `I < J`. The `I < J` test discards the lower triangle; on a diagonal
  tile (`tileA == tileB`) the extra `a >= b` guard drops the within-tile lower half
  and the diagonal itself, so each unordered pair is scored exactly once.
- **Out-of-range rows load as zero.** When a tile straddles the end of the sample
  list, the cooperative load substitutes `Readv2Word{0, 0}` for rows past
  `n_samples`. A zero word has `valid = 0` everywhere, so it contributes zero
  comparable sites and can never corrupt a live pair's totals — but those padding
  lanes are inactive anyway.
- **`__syncthreads` bracketing.** There is a barrier after the load (so no thread
  reads shared before it's filled) and a barrier after the compute (so no thread
  overwrites shared for the next window before the current one is consumed). Both
  are required for correctness.

---

## 6. The pair rank — one enumeration, two derivations

Every pair has one canonical output slot: its rank in the upper-triangular
enumeration of `(i < j)`. The two kernels arrive at it two different ways but the
formula is identical, and that identity is what lets them share output buffers.

- **Baseline** unranks: given the flat rank `r`, `readv2_unrank_pair` (in
  `sweep_unrank.cuh`) recovers `(i, j)` in O(1) via the closed form
  `j = floor((1 + sqrt(1 + 8r)) / 2)`, `i = r - C(j,2)`.
- **Tiled** ranks the other direction: it already knows `(I, J)` from its tile
  position, so it computes `r = C(J,2) + I = J*(J-1)/2 + I` to find the slot.

Both agree because both use the same grouping: pairs are blocked by the larger
index `j`, the block for `j` starts at `C(j,2)` and holds `j` pairs (`i = 0..j-1`).
The comment on `readv2_unrank_pair` spells out the derivation.

**Overflow guard.** In the baseline the flat rank is formed as
`static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x` — the cast happens
*before* the multiply, so a pair count past 2^31 never overflows a 32-bit
intermediate. This was a review fix; the naive `int` product would silently wrap on
large sample sets.

---

## 7. The launcher's shared-memory decision

`launch_readv2_mismatch` is a thin dispatcher, but it makes one real decision: can
the tiled kernel's shared-memory footprint actually be allocated?

The tiled kernel needs `2 * TS * wpw` words of shared memory (both tiles, one
window's worth of rows each). For a wide window that can exceed the 48 KB default
static limit. So the launcher:

1. Computes the exact `smem` byte count.
2. If `smem` fits within the opt-in cap (`kOptinSmem = 99 KB`) but exceeds the
   48 KB default, it calls `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)`
   to raise the kernel's dynamic-shared ceiling first, then launches the tiled
   kernel with a 32x32 block.
3. If `smem` is larger than even the opt-in cap — a window too wide to tile —
   **it falls through to the baseline kernel.** The baseline needs no shared
   memory and is always correct, so an over-wide window degrades to slower-but-
   correct rather than failing.

The `tiled` argument only *requests* the tiled path; the launcher may still choose
the baseline when the geometry won't fit. Either way the caller gets the same four
outputs.

Grid sizing goes through `core::grid_for_x`, which carries a descriptive message
and enforces the `kMaxGridX` ceiling — so an enormous pair count (or tile-pair
count) fails with a clear "restrict --samples" error instead of a silently
truncated launch.

---

## 8. Contracts and invariants

- **`n_pairs <= 0` is a clean no-op.** The launcher returns immediately, leaving
  the output buffers untouched. The host pre-zeros them, so "no pairs" reads as
  all zeros.
- **Output buffers are sized `n_pairs`.** The caller
  (`cuda_backend_readv2.cu`) allocates `d_sum_p0`, `d_sum_p0_sq`, `d_n_win_used`,
  and `d_tot_comp` each with one slot per pair, indexed by the section-6 rank. Both
  kernels bounds-check `r < n_pairs` before writing.
- **Row bases assume a dense row-major matrix.** Sample `s`'s first word is at
  `d_words + s * words_per_sample`, and `words_per_sample = n_win * wpw`. The
  kernel does no bounds check on words within a sample — the pack kernel is trusted
  to have laid down exactly this geometry.
- **Every pair is scored exactly once.** The baseline enumerates `r` over
  `[0, n_pairs)` directly; the tiled path's `I < J` plus diagonal `a >= b` guards
  guarantee the same coverage with no double-counting.
- **The four outputs are consistent by construction.** `n_used`, `sum_p0`,
  `sum_p0_sq`, and `tot_comp` are all incremented together inside the single
  `win_comp > 0` branch, so `n_win_used == 0` iff `sum_p0 == 0` (no usable window).

---

## 9. Edge cases

- **A pair with no overlap.** If no window has a comparable site, every output is
  0. The host's `n_win_used` divisor would be zero, so the host applies its own
  min-overlap gate before dividing — this kernel simply reports the honest zeros.
- **A window with partial data.** A window where only some sites overlap still
  contributes its real `mismatch/comparable` ratio; the word-aligned padding
  (`valid = 0` on the tail bits) means the partial last word is handled with no
  special case.
- **Very wide windows.** Handled by the section-7 fallthrough: too wide to tile
  means the baseline runs instead. Correctness is never in question, only speed.
- **Sample counts past 32-bit pair ranks.** Guarded by the section-6
  cast-before-multiply and by `grid_for_x`'s `kMaxGridX` check, which surfaces an
  explicit error rather than wrapping.
- **Tiles at the end of the sample list.** Out-of-range rows load as zero words and
  their lanes are inactive, so a ragged final tile costs a little wasted load but
  produces no wrong answer.

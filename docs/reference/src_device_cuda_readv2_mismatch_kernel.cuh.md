# `readv2_mismatch_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/readv2_mismatch_kernel.cuh` declares a single function,
`launch_readv2_mismatch` — the launcher for READv2's all-pairs windowed-mismatch
reduction, and steppe's **first `__popc` kernel**. Given the packed
`[sample x SNP-window]` bit-matrix that the pack kernel wrote (see
`readv2_layout.cuh`), it walks every unordered pair of samples and, for each pair,
produces the four per-pair reductions the host later turns into a `P0_mean` and its
standard error.

This header is the public seam of the reduction: it is the one declaration the
device backend (`cuda_backend_readv2.cu`) calls, and it hides both kernels and all
the launch-shape arithmetic behind that one entry point. The kernel bodies and the
per-word AND/XOR/`__popcll` core live in the paired `.cu`
(`readv2_mismatch_kernel.cu` — its own reference doc covers the algorithm in full);
this file is the **contract** those kernels satisfy.

It is a `.cuh`, private to the `steppe_device` target. It includes `cuda_runtime.h`
and `readv2_layout.cuh` (for the `Readv2Word` cell type), so it never appears on the
CUDA-free seam the host-only app translation units live behind. There is no logic
here to run — just the declaration and the four output pointers it fills.

---

## 2. The four outputs

The launcher does not compute a mismatch ratio itself. For every pair it writes four
device arrays, each indexed by the pair's flat upper-triangular rank `r` (section 4),
and hands them back for the host to finish into `P0_mean` and its SE:

| Output pointer | Type | What it accumulates per pair |
|---|---|---|
| `d_sum_p0` | `double*` | Sum over windows of the per-window mismatch/comparable ratio. |
| `d_sum_p0_sq` | `double*` | Sum over windows of that ratio **squared** — the raw material for the standard error. |
| `d_n_win_used` | `int*` | Count of windows that had at least one comparable site (becomes `n_windows`). |
| `d_tot_comp` | `long long*` | Total comparable sites summed across all windows (becomes `n_overlap_sites`). |

The split into "sum of ratios" and "sum of squared ratios" is deliberate: `P0` is
the **mean over windows** of the per-window ratio, so the host divides `sum_p0` by
`n_win_used` for the mean, and combines `sum_p0_sq` with `sum_p0` and `n_win_used`
for the variance/SE — no second pass over the genome needed.

Each output array must be sized to hold one slot per pair — `C(n_samples, 2)` =
`n_pairs` entries. The caller allocates exactly `n_pairs` doubles/ints and reads them
straight back (see `cuda_backend_readv2.cu`, which D2H-copies all four into the host
result struct's `sum_p0` / `sum_p0_sq` / `n_win_used` / `tot_comp` vectors).

---

## 3. The parameter contract

```
void launch_readv2_mismatch(const Readv2Word* d_words,   // the resident bit-matrix
                            long words_per_sample,        // row stride (Readv2Words)
                            int  wpw,                     // words per window
                            long n_win,                   // windows tiling the SNP axis
                            int  n_samples,
                            long long n_pairs,            // = C(n_samples, 2)
                            double*    d_sum_p0,          // -- the four outputs --
                            double*    d_sum_p0_sq,
                            int*       d_n_win_used,
                            long long* d_tot_comp,
                            bool tiled,                   // pick the tiled kernel?
                            cudaStream_t stream);
```

The caller is expected to hold up its end:

- **`d_words` is the resident bit-matrix**, laid out exactly as `readv2_layout.cuh`
  fixes it: sample-major, `words_per_sample` `Readv2Word`s per row, window `g`'s word
  `wl` at `g*wpw + wl` within a row. `words_per_sample`, `wpw`, and `n_win` **must**
  come from that header's geometry helpers (`readv2_words_per_sample`, `readv2_wpw`,
  `readv2_n_win`) so the launcher's index arithmetic matches how the pack kernel
  wrote the matrix. This is the shared-layout invariant the two kernels are written
  against.
- **`n_pairs` must equal `C(n_samples, 2)`.** The launcher trusts it as the output
  bound and the direct kernel's thread count; a value that disagrees with
  `n_samples` would over- or under-fill the output arrays.
- **The four output pointers are device memory of `n_pairs` elements each**, and the
  launcher writes each pair's slot exactly once — no accumulation across calls, no
  atomics. A pair's four slots are all written together or (for a pair the tiled path
  skips as out of range) left as the caller initialized them.
- **`stream`** is the stream both kernels launch on; the launcher does not
  synchronize. The caller owns ordering and the subsequent D2H.

Nothing is copied host-side and nothing is allocated — the launcher only chooses a
kernel, computes its launch shape, and fires it on `stream`.

---

## 4. Why `tiled` is a hint, not a promise

`launch_readv2_mismatch` is the dual-kernel launcher pattern steppe uses elsewhere
(it mirrors `dstat_kernel`'s baseline/tiled split): a simple correctness-gate kernel
and a faster throughput kernel, both behind one call, selected by the `tiled` flag.

- **`tiled == false`** runs the **baseline** kernel: one thread per pair, each thread
  unranks its own `(i, j)` with the O(1) closed-form `readv2_unrank_pair` and reduces
  that pair's windows directly from global memory. This is the correctness gate — the
  always-valid reference the tiled path is checked against.
- **`tiled == true`** runs the **tiled** kernel: a block owns a `32 x 32` tile-pair
  of samples and cooperatively stages each window's rows into shared memory once,
  reusing them across the tile's pairs (the dstat-style row-reuse that cuts global
  traffic). It writes each pair to the same upper-triangular slot the baseline would.

The one subtlety the header's caller should know: **`tiled == true` is a request the
launcher may decline.** The tiled kernel stages `2 * kTileSamples * wpw` `Readv2Word`s
in shared memory; when a window is wide enough that this exceeds the opt-in
shared-memory ceiling (~99 KB), the launcher silently **falls through to the baseline
kernel** rather than fail. So `tiled` is a performance hint, and the result is
bit-identical either way — both kernels compute the same four reductions into the
same slots. (Between the 48 KB default and the 99 KB opt-in ceiling the launcher
raises the kernel's dynamic shared-memory attribute so the wider tile can run.)

The two paths agree on the output slot because both use the same upper-triangular
rank of `(i < j)`: `C(j, 2) + i`. The baseline gets `(i, j)` from
`readv2_unrank_pair`; the tiled kernel recomputes `r = C(J,2) + I` from its own
tile-derived `I < J`. Same enumeration, same slot — that identity is what makes the
tiled path a drop-in throughput swap.

---

## 5. Contracts and invariants

These are promises the launcher and its kernels keep; a caller reading this header
can rely on them.

1. **One write per pair, no atomics.** Each pair's four output slots are written
   exactly once by exactly one thread. There is no cross-pair accumulation, so the
   output arrays need no pre-zeroing for the pairs the launcher covers.

2. **The two kernels are numerically identical.** Baseline and tiled fold windows in
   the same order and use the same per-word primitive
   (`comp = vi & vj; mism = (ai ^ aj) & comp; __popcll` each), so switching `tiled`
   never changes a single output bit. The tiled path exists only to move less data,
   not to compute differently.

3. **P0 is the mean over windows, not one genome-wide ratio.** Both kernels flush a
   ratio at every word-aligned window boundary and only fold windows with
   `comparable > 0`, exactly as READv2 defines the statistic. A window with no
   comparable sites contributes nothing and does not count toward `n_win_used`.

4. **Geometry comes from the shared layout header.** `words_per_sample`, `wpw`, and
   `n_win` are the caller's, derived from `readv2_layout.cuh`; the launcher never
   re-derives them. This is what keeps the reader's indexing aligned with the
   writer's.

5. **`__popcll`, never `__popc`.** The counts run over full 64-bit `Readv2Word` bit
   planes, so the 64-bit population count is the correct primitive; using the 32-bit
   `__popc` would silently miss half of every word. (This is stated in the `.cu`, but
   it is a contract of the layout this header shares.)

---

## 6. Edge cases

- **`n_pairs <= 0`.** The launcher returns immediately — a run with zero or one
  sample has no pairs to reduce, and the output arrays are left untouched. Callers
  with `n_samples < 2` get a clean no-op.

- **A window too wide to tile.** Covered in section 4: `tiled == true` degrades to
  the baseline kernel when the per-window shared-memory footprint exceeds the opt-in
  ceiling, rather than launching a kernel that can't fit its tile. The caller sees
  the same outputs, just without the shared-memory speedup.

- **Grid too large for one launch axis.** Both paths route their grid size through
  `core::grid_for_x`, which throws a named error if the pair axis (baseline) or the
  tile-pair axis (tiled) would exceed the maximum `gridDim.x`. The message names the
  culprit and suggests restricting `--samples` — a very large sample count fails
  loudly at launch rather than silently truncating the sweep.

- **Padding and missing data need no special handling here.** Because
  `readv2_layout.cuh` writes every padding / missing / het / off-genome bit with
  `valid = 0`, the both-valid AND inside the kernel doubles as the padding mask.
  There is no separate boundary case for a window's short tail word or the genome's
  final short window — the launcher's contract inherits that simplification from the
  layout wholesale.

- **Diagonal (self) pairs never appear.** The enumeration is strictly upper-triangular
  (`i < j`); the tiled kernel additionally skips `a >= b` inside a same-tile block. No
  sample is ever compared with itself, so `n_pairs` is `C(n_samples, 2)` and the
  output arrays carry no self-comparison slot.

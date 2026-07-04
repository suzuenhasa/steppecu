# `qpfstats_jackknife_kernel.cu` reference

## 1. Purpose

`src/device/cuda/qpfstats_jackknife_kernel.cu` holds the two GPU kernels that run
the block-jackknife estimates for the qpfstats statistics on the device instead of
on the host. Both take per-block partial sums that already live in GPU memory and
reduce them, one output per population combination or per population pair, down the
block axis.

These kernels replace what used to be two serial host loops written in extended
(`long double`) precision. That host work was a real bottleneck: roughly 305,000
combinations, each reducing over about 711 jackknife blocks, comes to on the order
of 217 million iterations run on a single CPU core. In a profile of the run, the
GPU sat completely idle for the whole stretch while that one core ground through the
loops. Moving the loops onto the GPU makes them embarrassingly parallel — one thread
per combination or per pair — and, just as importantly, lets them read the partial
sums (each array is around 1.7 GB) straight from GPU memory rather than copying them
back to the host first.

Two kernels live here:

1. **The per-combination global estimate** (`qpfstats_numer_jackknife_kernel`) —
   turns the running numerator/count sums into per-block means and the single
   global block-jackknife estimate for each combination.
2. **The per-pair recenter shift** (`qpfstats_recenter_shift_kernel`) — computes,
   for each population pair, the small constant that recenters its smoothed
   per-block values against the global estimate.

Everything in the file except the two launch wrappers is private to the device
layer. The kernel bodies and the `<<<>>>` launches live only here; the rest of
steppe reaches them through the narrow wrappers `launch_qpfstats_numer_jackknife`
and `launch_qpfstats_recenter_shift`, declared in the companion `.cuh` header.

---

## 2. The block jackknife these kernels implement

Both kernels compute a **leave-one-out block jackknife**. The genome is split into
contiguous blocks of SNPs. For a given quantity you first form its value using all
blocks, then form it again with each block left out in turn, and combine those
leave-one-out values into a bias-corrected estimate. Leaving out one block at a time
(rather than one SNP) keeps nearby, linked SNPs together, which is what makes the
resulting uncertainty estimate honest.

A few pieces of vocabulary used throughout both kernels:

- **`rel`** (relative block weight) — a block's weight divided by the total weight
  across all valid blocks. It is the fraction of the whole that this one block
  represents.
- **leave-one-out value (`loo`)** — the quantity recomputed with one block removed,
  written as `(tot - value·rel) / (1 - rel)`, where `tot` is the all-blocks value.
- **`1 - rel`** — the leftover weight after removing a block. When a block is the
  only valid one its `rel` is 1, so `1 - rel` is 0 and the leave-one-out value would
  divide by zero; both kernels skip any block whose `rel` reaches 1 for exactly this
  reason.

The two kernels use the same jackknife machinery but combine the leave-one-out
values with different final formulas, because each reproduces a different reference
routine (see sections 3 and 4).

---

## 3. The per-combination global-estimate kernel

`qpfstats_numer_jackknife_kernel` assigns **one thread to one population
combination** `c`. It reproduces the `matrix_jackknife_est_col` routine
exactly[^at2] — the same missing-data masks, the same skip when a block's `rel`
reaches 1, the same guard that a leave-one-out value must be finite, the same
ascending-block accumulation order, and the same final formula.

### Inputs

Two arrays, both **row-major** and sized `[npopcomb × n_block]`. The cell for
combination `c` and block `b` sits at index `c * n_block + b`, so one thread reads a
single contiguous run.

- `numsum` — the running numerator sum for each (combination, block).
- `cnt` — how many SNPs contributed to that (combination, block) — the block's
  weight.

### Outputs

- `numer` — **row-major** `[npopcomb × n_block]`: the per-block mean
  `numsum / cnt`, or NaN where `cnt` is zero or negative.
- `ymat` — **column-major** `[npopcomb × n_block]`: the same per-block means, but
  laid out so that `ymat[c + npopcomb*b]` is the value for combination `c`, block
  `b`. This transposed copy is what a downstream stage consumes as a design matrix.
- `y` — `[npopcomb]`: the single global block-jackknife estimate per combination.
  This is the source of the right-hand side used later.

### What each thread does

1. Fill in the block means. For every block, `numer` and `ymat` get
   `numsum / cnt`, or NaN when `cnt <= 0`.
2. Total weight `sum_n_all = Σ cnt` over all blocks. If it is not positive, the
   estimate is NaN and the thread is done.
3. All-blocks value `tot = Σ(mean·cnt) / Σ(cnt)`, summed only over blocks whose mean
   is finite and whose count is positive. If the weight there is not positive, the
   estimate is NaN.
4. Walk the blocks once more, and for each valid block whose `rel = cnt/sum_n_all`
   is below 1, form the finite leave-one-out value and accumulate five running sums:
   the sum of leave-one-out values, the sum of leftover weights `1 - rel`, the
   weighted sum of leave-one-out values by leftover weight, the weighted sum by
   count, and the total count of finite blocks.
5. Combine into the final estimate:

   ```
   y = n_finite · (Σ loo·(1-rel) / Σ(1-rel))  −  Σ loo  +  (Σ loo·cnt / Σ cnt_finite)
   ```

   If no block was finite, or either weight sum is not positive, the estimate is
   NaN instead.

The NaN results and the various skip conditions are not error handling that could be
loosened — they are part of matching the reference routine bit-for-bit.

---

## 4. The per-pair recenter-shift kernel

`qpfstats_recenter_shift_kernel` assigns **one thread to one population pair** `p`.
It reproduces the `f2blocks_pair_est` routine[^at2] (the per-pair estimate of
an `f2` block array) exactly, then subtracts that estimate from the global value to
produce a recentering constant.

### Inputs

- `b` — the smoothed per-block values, **column-major** `[npairs × n_block]`, so
  the value for pair `p`, block `blk` is at `b[p + npairs*blk]`.
- `bglob` — `[npairs]`: the global value already computed for each pair.
- `block_sizes` — `[n_block]`: the number of SNPs in each block, used here as the
  block weight (integers, not counts of contributing SNPs).

### Output

- `shift` — `[npairs]`: `shift[p] = bglob[p] - est[p]`, the constant that recenters
  the pair's smoothed values onto the global scale.

### What each thread does

1. Total block weight `sum_bl = Σ block_sizes` over blocks whose value is finite. If
   that is not positive, the estimate defaults to 0 and the shift is just `bglob[p]`.
2. All-blocks value `tot = Σ(value·block_size) / sum_bl` (a weighted mean).
3. Walk the blocks and, for each finite one whose `rel = block_size/sum_bl` is below
   1, form the leave-one-out value and weight it by `1 - 1/h`, where `h = sum_bl /
   block_size`. Accumulate the weighted sum of leave-one-out values and the sum of
   weights.
4. The estimate is that weighted mean of leave-one-out values, or 0 if the weight
   sum is not positive. The shift is `bglob[p] - est`.

The two "default to 0" branches (empty total weight, zero weight sum) are the
reference routine's own return values, reproduced deliberately.

---

## 5. Precision: native double precision and exact operand order

Both kernels run in **native double precision**, and this is a firm rule, not a
default that may be swapped. steppe elsewhere uses a faster emulated form of
double-precision arithmetic for its heavy matrix multiplies, but that emulated mode
is **never** used here. A jackknife is built on differences of nearly equal numbers
(`tot - value·rel`, and the final `n_finite·tot2 - Σloo + …`), which are prone to
catastrophic cancellation. Cancellation-sensitive arithmetic like this is one of the
operations steppe always carves out to true double precision, exactly because the
emulated mode is accuracy-approximate. The rule of thumb: emulated precision is for
matrix multiplies, native precision is for cancellation-prone reductions like this
one.

The kernels are written as a direct transliteration of a host reference that is
written in extended `long double` precision (`core::matrix_jackknife_est_col` and
`core::f2blocks_pair_est`). Everything is kept identical — the same masks, the same
skips, and crucially **the same ascending-block accumulation order**. The only
deliberate difference is the width of the running total: the host reference carries
its sums in a 64-bit `long double` mantissa, while the GPU carries them in the
53-bit mantissa of standard double precision. Because the operand order is fixed and
the arithmetic is native FP64, each result is reproducible from run to run and
matches the reference to a tight tolerance.

This same leave-one-out jackknife has already been proven on the device in native
double precision by the qpAdm model-fit path, which holds its own reference result;
these two kernels are the same family of computation. The pair here is validated
against a nine-population genotype reference result at a relative tolerance of
`1e-6`, with the CPU reference backend and the GPU backend required to agree.

---

## 6. Thread layout, grid geometry, and the launch wrappers

### One thread per output, grid-stride down the block axis

Each kernel gives **one thread to one combination (or pair)**, and that thread owns
the whole short reduction over the block axis in its own registers. With only about
711 blocks per reduction, a single thread walking the run is the right granularity.
A cooperative per-block reduction across many threads (the kind used for very long
reductions) would only add launch and shared-memory overhead here without buying
anything. This mirrors the one-thread-per-row idiom already used by the qpAdm fit's
leave-one-out kernel.

Both kernels use a **grid-stride loop**: a thread processes combination (or pair) `i`,
then `i + gridDim.x·blockDim.x`, and so on. That lets a launch cover any number of
combinations with a bounded grid, and it means the combinations a single thread
touches are spaced apart rather than adjacent. Within one combination, though, the
block cells are contiguous in memory, so each individual reduction is a contiguous
read.

The loop index is carried as a 64-bit `long` and the array offsets are computed in
64-bit arithmetic, so the flat index into a multi-gigabyte `[npopcomb × n_block]`
array never overflows a 32-bit integer.

### The launch wrappers

Two thin wrappers configure and launch the kernels:

- `launch_qpfstats_numer_jackknife` — for the per-combination kernel.
- `launch_qpfstats_recenter_shift` — for the per-pair kernel.

Both follow the same shape:

- They return immediately and launch nothing when there is no work (a non-positive
  combination/pair count or block count).
- They use **128 threads per block** (`kThreads = 128`).
- They compute the grid as the ceiling of the work count divided by 128, and then
  **clamp that block count** to the single shared grid-dimension cap
  (`core::kMaxGridX`). Clamping is safe because the grid-stride loop inside the
  kernel covers any remaining work with the fixed grid. (The recenter wrapper also
  guards a minimum of one block.)
- After launching, each checks for a launch error through the shared kernel-check
  macro.

The wrappers are the only surface the device backend is allowed to call; the kernel
bodies and their launch syntax stay private to this translation unit.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

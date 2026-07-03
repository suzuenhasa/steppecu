# `decode_af_kernel.cu` reference

## 1. Purpose

`src/device/cuda/decode_af_kernel.cu` holds the GPU kernel that turns packed,
2-bit-per-genotype data into per-population allele frequencies, plus a thin host
wrapper that launches it.

It owns exactly two things:

1. **`decode_af_kernel`** — the GPU kernel. It runs one thread per
   `(population, SNP)` pair. Each thread walks over all the individuals belonging
   to its population, unpacks that SNP's 2-bit code from every individual, sums
   the codes into counts, and turns those counts into three output numbers: a
   frequency `Q`, a "valid?" flag `V`, and a sample-count `N`.
2. **`launch_decode_af`** — a small host-side function that computes the launch
   dimensions and starts the kernel. It exists so that host code never has to
   contain the `<<<...>>>` launch syntax directly; the kernel body and its launch
   both stay inside this one translation unit.

The actual bit-unpacking and counting math does **not** live in this file. It
comes from a shared routine (`core/internal/decode_af.hpp`) that both this GPU
kernel and the CPU reference implementation call. Sharing that one routine is what
guarantees the GPU path and the CPU reference can never disagree on how a code is
unpacked, how missing data is handled, or how the final division is done. This
file is the GPU-specific wrapper around that shared math: it adds the thread
layout, the memory-access pattern, and the launch geometry.

This is a CUDA source file and is private to the device layer. Nothing outside the
GPU backend includes it.

---

## 2. What the kernel reads and writes

The kernel's inputs and outputs form a fixed contract. Understanding the two
different memory layouts in play — one for the packed input, one for the outputs —
is the key to the rest of this file.

### Inputs

| Argument | Shape / meaning |
|---|---|
| `packed` | The packed genotype bytes, laid out one record per individual. Each record is `bytes_per_record` bytes, and four genotype codes are packed into each byte (2 bits each). Individuals are stored so that all individuals of one population sit next to each other. |
| `bytes_per_record` | How many bytes each individual's record occupies. Used to step from one individual to the next inside `packed`. |
| `pop_offsets` | An array of `P + 1` boundaries over the individual axis. Population `i` owns the individuals in the half-open range `[pop_offsets[i], pop_offsets[i+1])`. |
| `P` | Number of populations (the output row count). |
| `M` | Number of SNPs (the output column count). Typed as `long` because the SNP count is the one axis that can realistically exceed two billion. |
| `ploidy` | A single fallback ploidy value (2 = diploid, 1 = pseudo-haploid) used for every sample when no per-sample vector is supplied. |
| `sample_ploidy` | An optional per-sample ploidy array, one entry per individual. When present it overrides the scalar `ploidy` on a per-individual basis. When null, the kernel uses the scalar `ploidy` for everyone — this is the older all-diploid path and produces bit-for-bit identical results to before the per-sample vector existed. |

### Outputs

`Q`, `V`, and `N` are three separate arrays, each of size `P × M`, stored in
**column-major** order. That means element `(i, s)` — population `i`, SNP `s` —
lives at flat index `i + P·s`. The consequence that drives the whole kernel
design: **the population axis `i` is the unit-stride (contiguous) dimension**, and
stepping along the SNP axis `s` jumps by `P` elements.

| Output | Meaning |
|---|---|
| `Q` | The reference-allele frequency for that `(population, SNP)`. |
| `N` | The effective sample count (sum of ploidy over the non-missing individuals). |
| `V` | A validity flag: `1.0` when there was at least one non-missing individual, `0.0` otherwise. |

---

## 3. Decoding one (population, SNP): the compute phase

Each thread is assigned one output cell. The thread's `x` coordinate selects the
SNP `s`; its `y` coordinate selects the population `i`. If that `(i, s)` falls
outside the real `P × M` range (which happens at the edges of the grid), the
thread simply does no work in this phase.

For a valid `(i, s)`, the thread:

1. Looks up its population's individual range `[seg_begin, seg_end)` from
   `pop_offsets`.
2. Computes **which byte** inside each record holds SNP `s` (`s / 4`, since four
   codes pack into a byte) and **which of the four positions** within that byte
   (`s % 4`).
3. Loops over every individual `g` in its population's range. For each one it
   reads that individual's byte, extracts the 2-bit code, and folds the code into
   two running accumulators using the shared decode routine:
   - `ac` — a running reference-allele count. For diploid samples this adds the
     raw code (0, 1, or 2); for pseudo-haploid samples it adds a weighted value.
     This weighting is the same "adjust pseudo-haploid" accumulation the CPU
     reference uses, so the two match exactly.
   - `n` — a running sum of ploidy over the non-missing individuals.

   Missing genotypes are skipped by the shared routine and contribute to neither
   accumulator.
4. After the loop, forms the final result with a single division: `Q = ac / n`,
   `N = n`, and `V = 1` (or all zero when `n` came out as 0, i.e. the population
   had no non-missing individual at this SNP).

The per-individual ploidy used in step 3 is taken from `sample_ploidy[g]` when the
per-sample vector was supplied, and otherwise from the scalar `ploidy`.

At the end of this phase the thread does **not** write its result to global
memory yet. It writes into shared memory instead — the reason is explained in
section 5.

---

## 4. Precision: exact accumulation, a single divide

The decode is limited by how fast the GPU can read the packed bytes, not by how
fast it can do arithmetic — the per-individual loop is dominated by memory reads.
Because of that, there is nothing to gain from using a faster, lower-precision
form of arithmetic here; it would only risk breaking the exact match against the
reference.

So the accumulation is deliberately kept exact and the division is done exactly
once, at the very end:

- The running counts are summed as small exact values (whole numbers for diploid,
  exact halves for pseudo-haploid), so no rounding error builds up across the
  individual loop.
- The final frequency is a **single** native double-precision division of those
  accumulated counts. Doing the divide once, rather than dividing per individual
  and summing, is what makes the frequency reproducible to the last bit and keeps
  it identical to the CPU reference.

There is no emulated or reduced-precision math anywhere in this kernel. Everything
is native double precision and integer arithmetic.

---

## 5. Memory coalescing and the two-phase design

This is the most important design point in the file. It exists to make **both**
the input reads and the output writes fast, even though the input and the output
want opposite thread layouts.

### The input reads want the SNP axis on `x`

Threads are laid out so that consecutive threads in a warp (varying `x`) map to
consecutive SNPs. Because the byte that holds SNP `s` is `s / 4`, four neighboring
SNPs share a byte and neighboring groups of SNPs sit in neighboring bytes. So on
each step of the individual loop, a warp reads a small contiguous window of bytes
from the same individual's record. Contiguous reads are what the hardware wants
(they "coalesce" into a few wide memory transactions instead of many scattered
ones). This is the dominant memory traffic in the kernel, and keeping it fast is
why the SNP axis is pinned to the `x` dimension and never moved.

### The output writes want the population axis on `x`

But the outputs are column-major, so the contiguous output dimension is the
**population** axis, not the SNP axis. With the SNP axis on `x`, the naive
approach — each thread writing its own result straight to global memory — would
have neighboring threads writing to addresses `P` apart. That is a scattered
write: 32 threads touching 32 far-apart locations, which the hardware cannot
combine, so it is slow.

### The fix: stage in shared memory, then re-emit with a swapped mapping

The kernel resolves the conflict by splitting into two phases separated by a
barrier:

1. **Compute phase** (section 3): each thread computes its `Q`, `V`, `N` and
   writes them into three shared-memory tiles, indexed by the thread's local
   position. The coalesced input reads are untouched.
2. **`__syncthreads()`** so every thread's result is visible to the whole block.
3. **Store phase**: the block re-numbers its threads with a flat id and a new
   mapping in which consecutive threads now vary the **population** (unit-stride)
   axis. Each thread reads back the appropriate value from the shared tile and
   writes it to global memory. Because consecutive threads now hit consecutive
   global addresses (`i + P·s` with `i` varying), the three stores become fast,
   contiguous bursts.

The actual per-cell math and the final global addresses are exactly the same as
they would be without this scheme — only the *access pattern* of the stores
changes. So this optimization does not alter any reported number.

### Why the shared tiles are padded by one

The three shared tiles are declared as `[8][32 + 1]` — that trailing `+ 1` is
deliberate padding, not a spare slot. In the store phase, consecutive threads read
*down a column* of a tile (varying the first index). Shared memory is split into
32 banks, and if the tile's inner dimension were exactly 32, every entry of a
column would fall in the same bank — a 32-way bank conflict that serializes the
read. Widening the inner dimension to 33 shifts each row by one bank, so a column
now spreads across all banks and the conflict disappears. This is the standard
efficient-matrix-transpose padding trick.

---

## 6. A deferred optimization: within-warp duplicate reads

There is a known, deliberately-not-done optimization worth being aware of.

With four SNPs sharing a byte, a warp of 32 threads only touches 8 distinct bytes
per individual — each byte is fetched by 4 different threads, and that same 8-byte
window is re-fetched on every step of the individual loop. In principle a warp
could load each byte once and share it among the 4 threads that need it (via a
warp shuffle or shared memory), avoiding the duplicate fetches.

This is **not** done, on purpose. The duplicate reads are contiguous, so the GPU's
L1/L2 caches already absorb them cheaply — it is not a correctness or coalescing
bug, just the natural shape of the bandwidth-bound design. Cooperatively loading
and broadcasting would add barrier/shuffle complexity, and it would only pay off
if a profiler showed the cache was *not* already absorbing the duplicates. No such
measurement exists, so the code is left as-is.

---

## 7. Launch geometry and the host wrapper

`launch_decode_af` is the host-side entry point. It picks the thread-block and
grid dimensions and starts the kernel.

### Block shape

The block is a fixed `32 × 8` = 256 threads: 32 along the SNP (`x`) axis — one full
warp, aligned so the input reads coalesce — and 8 along the population (`y`) axis.
This is a **non-square** block, which is why the launch code sets the two axes
separately rather than using a square-block default helper.

The kernel is compiled with a launch-bound of `32 × 8` = 256, which caps the
registers per thread against the one and only block shape it is ever launched with.
This matters more now that the kernel uses three shared-memory tiles: occupancy is
limited by several resources at once, and the register cap is a safe guard because
the fixed block can never exceed the declared bound.

### Grid shape and the SNP-axis size guard

The grid is sized to cover the full output:

- The **SNP axis** rides the grid's `x` dimension. Because the SNP count `M` can
  exceed two billion, this axis is computed with `long` arithmetic. The `x`
  dimension is the only axis allowed to be that large, which is exactly why the
  SNP axis is put here.
- The **population axis** rides the grid's `y` dimension, computed from `P` with
  the ordinary integer helper.

Both grid dimensions have hardware maximums. The population axis is checked by its
helper. The SNP axis is checked explicitly in `launch_decode_af`: an assertion
verifies the computed `x`-grid extent does not exceed the maximum grid width, and
if it ever did the fix would be to tile the SNP axis into multiple launches. This
assertion compiles out of release builds, so it costs nothing at runtime and only
guards against a future dataset large enough to overflow the grid. After the
launch, the standard kernel-error check runs.

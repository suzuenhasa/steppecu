# `dstat_kernel.cu` reference

## 1. Purpose

`src/device/cuda/dstat_kernel.cu` holds the GPU reduction that turns per-SNP
allele frequencies into the block-level partial sums the normalized-D statistic
and the qpfstats genotype-path f4 numerator are built from. It is the
genotype-path counterpart to the f2-cache path: instead of precomputing an f2
tensor, it walks the raw per-SNP frequencies directly and accumulates the
statistic's numerator, denominator, and used-SNP count for every requested
population quadruple, split by jackknife block.

The file contains two kernels that compute exactly the same thing, plus the
launch wrapper that chooses between them:

1. A **tiled, shared-memory reduction** — the hot path. It loads each SNP's
   frequencies once, builds a small table of pairwise differences shared by all
   the population quadruples in a thread block, and reconstructs each
   quadruple's contribution from that table. This is where the speed comes from
   when there are hundreds of thousands of quadruples to evaluate.
2. A **legacy per-cell reduction** — the cold fallback, kept unchanged. One
   thread computes one output cell by re-reading the frequencies itself. It is
   used only when the population count is so large that the shared table no
   longer fits in GPU shared memory.

Everything in the file except the one launch wrapper is private to the device
layer. The kernel bodies and the `<<<>>>` launches live only here; the rest of
steppe reaches them through the narrow wrapper `launch_dstat_block_reduce`.

---

## 2. What the kernel computes

### Inputs

- `Q` — per-SNP allele frequencies, one value per (population, SNP). Stored
  **column-major** as a `[P × M]` array: the value for population `i` at SNP `s`
  lives at index `i + P*s`. `P` is the number of populations, `M` is the number
  of SNPs.
- `V` — a finiteness mask laid out identically to `Q`. A SNP is *valid* for a
  population when `V == 1` there and *missing* when `V == 0`.
- `quad` — the list of population quadruples to evaluate, four integers per
  quadruple: `(p1, p2, p3, p4)`. There are `N` of them. These are arbitrary —
  qpDstat may repeat an index inside one quadruple.
- `block_begin` / `block_size` — the jackknife blocks. Block `b` covers the
  contiguous SNP columns `[block_begin[b], block_begin[b] + block_size[b])`.
  There are `n_block` blocks.
- `M` is passed but not read for bounds — the block layout bounds every SNP
  walk. It is kept in the signature as the documented dimension.

### The per-SNP contribution

For one quadruple and one valid SNP, the kernel adds:

- **numerator**: `(Q[p1] - Q[p2]) * (Q[p3] - Q[p4])` — the f4-style
  difference-of-differences.
- **denominator**: `(Q[p1] + Q[p2] - 2·Q[p1]·Q[p2]) * (Q[p3] + Q[p4] - 2·Q[p3]·Q[p4])`
  — a product of two heterozygosity-like terms, used to normalize D.
- **count**: `+1` for each SNP that contributed.

The normalized D value (numerator over denominator) is *not* formed here. This
kernel emits only the per-block partial sums; the ratio and the block-jackknife
standard error are assembled downstream from those sums.

### The finiteness rule

A SNP contributes to a quadruple only if all four of its populations are valid
at that SNP — that is, `V != 0` for `p1`, `p2`, `p3`, and `p4`. If any one of
the four is missing, the SNP is skipped for that quadruple. This is the
per-(block, quadruple) "all SNPs" finiteness convention, matching ADMIXTOOLS 2.

### Outputs

Three arrays, each sized `[N × n_block]` and stored **row-major**: the cell for
quadruple `k` and block `b` is at index `k * n_block + b`.

- `numsum[k*n_block + b]` — the numerator sum over that block's valid SNPs.
- `densum[k*n_block + b]` — the denominator sum.
- `cnt[k*n_block + b]` — how many SNPs contributed.

---

## 3. The pairwise index encoding

Both the tiled kernel and its host-side counterpart address the unordered
population pairs `(i, j)` with `i < j` by a single flat integer, laid out as a
row-major upper triangle over all `C(P, 2)` pairs. The mapping is:

```
pair_index(lo, hi) = lo*P - lo*(lo+1)/2 + (hi - lo - 1)
```

where `lo` is the smaller index and `hi` the larger. Callers must pass the
smaller index first; the function is symmetric in its two arguments only in the
sense that the pair itself is unordered, so the caller is responsible for
ordering them.

This exact formula is shared with the host `pair_index` in
`src/core/stats/qpfstats.cpp` and reproduces ADMIXTOOLS 2's `indmat` ordering,
so the flat pair tables the GPU builds line up one-for-one with the host's
expectations.

---

## 4. The tiled pairwise-difference-reuse kernel (hot path)

`dstat_block_reduce_tiled_kernel` is the fast path. The idea is that most of the
work in evaluating hundreds of thousands of quadruples is recomputing the same
per-SNP pairwise differences over and over. Each difference `Q[i] - Q[j]`
depends only on the pair, not on which quadruple asked for it, so the kernel
computes each one once per SNP into shared memory and lets every quadruple read
it back.

### Grid and thread layout

- `gridDim.x = n_block` — **one grid block per jackknife block.** Because a whole
  grid block owns one jackknife block, it walks that block's SNP columns in
  ascending order and accumulates straight into per-thread registers. No SNP
  spans two grid blocks, so there is no cross-block reduction to stitch together
  afterward.
- `gridDim.y` — tiles over the quadruples.
- `blockDim.x = 256` threads. **Each thread owns exactly one quadruple**,
  identified by `k = blockIdx.y * blockDim.x + threadIdx.x`, and keeps three
  private double-precision accumulators (numerator, denominator, count) across
  the whole SNP walk.

### Per-SNP work, in three cooperative stages

For each SNP column `s` in the block, the threads of the grid block cooperate:

1. **Coalesced load.** The threads together read the SNP's `Q` and `V` values
   for all `P` populations into shared memory (`Qsh`, `Vsh`) — one contiguous,
   coalesced read of the column, reused by every thread.
2. **Build the shared tables.** The threads together fill two `C(P, 2)`-entry
   tables, computing each pair once:
   - `diff[ij] = Qsh[i] - Qsh[j]` for `i < j`.
   - `het[ij] = Qsh[i] + Qsh[j] - 2·Qsh[i]·Qsh[j]` for `i < j` (symmetric in
     `i`, `j`).

   To fill entry `idx` a thread decodes the flat upper-triangle index back into
   its `(i, j)` pair by walking the triangle rows.
3. **Reconstruct per quadruple.** Each active thread looks up its four
   populations' validity in `Vsh`, and if all four are valid, reconstructs its
   numerator and denominator contribution from the shared tables (see below),
   then adds them to its private accumulators.

The two tables are formed once per SNP and consumed by all 256 quadruples in the
grid block, which is what makes the kernel compute-bound rather than
memory-bound: the expensive shared reads amortize across the whole tile.

A `__syncthreads()` guards the boundary after each stage, and a final one at the
bottom of the SNP loop protects the tables from being overwritten by the next
SNP before every thread has finished reading them.

### Shared-memory layout

One dynamic `double[]` region, with `npairs = C(P, 2)`:

| Region | Range | Contents |
|---|---|---|
| `Qsh[i]` | `[0, P)` | this SNP's `Q` for each population |
| `Vsh[i]` | `[P, 2P)` | this SNP's `V` (validity) for each population |
| `diff[ij]` | `[2P, 2P + npairs)` | `Qsh[i] - Qsh[j]`, `i < j` |
| `het[ij]` | `[2P + npairs, 2P + 2·npairs)` | `Qsh[i] + Qsh[j] - 2·Qsh[i]·Qsh[j]`, `i < j` |

### Reconstructing a quadruple's contribution

The numerator needs two signed differences, `f1 = Q[p1] - Q[p2]` and
`f2 = Q[p3] - Q[p4]`. The shared table only stores the canonical `lo - hi`
direction, so the sign is recovered explicitly:

- if `p1 < p2`, `f1 = diff[pair_index(p1, p2)]`;
- if `p1 > p2`, `f1 = -diff[pair_index(p2, p1)]`;
- if `p1 == p2`, `f1 = 0` (a valid qpDstat quadruple may repeat an index, and
  `a - a` is exactly zero).

The same three cases give `f2`. The numerator contribution is `f1 * f2`.

The denominator needs two heterozygosity terms. `het` is symmetric, so a pair is
looked up with the smaller index first. When the two indices are equal, the term
is computed directly as `q + q - 2·q·q` rather than looked up, so the expression
stays byte-for-byte identical to what the legacy kernel would compute for the
same repeated-index quadruple.

### Threads with no quadruple

The last quadruple tile can extend past `N`. A thread whose `k` is out of range
is marked *inactive*, but it still participates in the cooperative load and
table-fill stages — it just skips its private accumulation and its final write.
This is deliberate: all `P` populations are loaded and all `C(P, 2)` pairs are
built regardless of which quadruples the tile happens to hold, so an arbitrary
set of qpDstat quadruples always finds the same complete shared tables.

---

## 5. The legacy per-cell kernel (cold fallback)

`dstat_block_reduce_legacy_kernel` is the original, unchanged reduction. It
launches one thread per output cell — `N × n_block` threads total — and each
thread decodes its own quadruple `k` and block `b`, walks that block's SNP
columns directly out of global `Q`/`V`, applies the four-population finiteness
mask, and accumulates the same numerator, denominator, and count. It uses no
shared memory and shares no work between quadruples.

Because it is byte-identical to the historical reduction, it is automatically
correct against the reference results — nothing about it changed, so it cannot
have regressed. It exists only to cover the population counts where the tiled
kernel's shared tables would not fit (see the next section).

---

## 6. Choosing a path and the shared-memory budget

`launch_dstat_block_reduce` sizes the tiled kernel's shared requirement and picks
the path. It launches nothing when there is no work (`N × n_block <= 0`). Both
kernels use `kThreads = 256` threads per block.

The tiled kernel's shared footprint is

```
smem = (2·P + 2·C(P,2)) · sizeof(double)   bytes
```

covering `Qsh[P]`, `Vsh[P]`, `diff[C(P,2)]`, and `het[C(P,2)]`.

Two named thresholds bound the decision:

| Threshold | Value | Meaning |
|---|---|---|
| `kDefaultSmem` | `48 KiB` (`49152`) | The dynamic shared a kernel gets with no opt-in. |
| `kOptinSmem` | `99 KiB` (`101376`) | The largest dynamic shared block available with opt-in on sm_120 / CUDA 13 (RTX 5090), reported as `sharedMemPerBlockOptin`. |

The path is chosen as follows:

- **`P >= 2` and `smem <= 99 KiB` → tiled hot path.**
  - When `smem` fits in the 48 KiB default, the kernel launches directly.
  - When `smem` is between 48 KiB and 99 KiB (roughly the `78 < P <= 112` band),
    the wrapper first opts into the larger limit with `cudaFuncSetAttribute(...,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem)`. On CUDA 13 / sm_120 this
    succeeds up to the 99 KiB cap. The call is idempotent per kernel-and-attribute
    and cheap, so it is simply issued whenever it is needed.
- **`smem > 99 KiB` (roughly `P > 112`) → legacy fallback.** Above this the
  `C(P, 2)` tables no longer fit even the opt-in budget, so the per-cell legacy
  kernel is launched instead over `ceil(N·n_block / 256)` one-dimensional blocks.

---

## 7. Grid geometry and the capped quadruple axis

The tiled launch maps the jackknife blocks onto `gridDim.x` directly
(`n_block`), and the quadruple axis `N` onto `gridDim.y`. CUDA caps a grid's Y
extent at 65,535, so `N` is turned into a tile count through the shared
launch-configuration helper `grid_for(N, kThreads)` rather than an open-coded
ceiling-divide.

Routing through that helper is what makes an over-limit extent fail fast: in a
debug build the helper's assert fires with a clear message instead of the launch
returning an opaque `cudaErrorInvalidConfiguration`. The computed value is the
same ceiling-divide the code used before, so the launch geometry is unchanged;
only the fail-fast check is added. This mirrors how the f2 feeder routes its own
population axis through the same helper.

The legacy fallback needs no such care — it is a flat one-dimensional grid of
`ceil(N·n_block / 256)` blocks.

---

## 8. Parity: bit-identical, native double precision

The tiled kernel is held to a strict guarantee: for every (quadruple, block),
its numerator sum and denominator sum are **bit-identical** to what the legacy
per-cell kernel produces — not merely within a relative tolerance, but the exact
same bits. This is why the tiled kernel deliberately reuses the legacy kernel's
exact subtraction and expression forms (including the direct `q + q - 2·q·q` for
a repeated index) rather than any algebraically equivalent but bit-different
rearrangement.

Both kernels run this reduction in **native double precision, never the emulated
double-precision mode** used elsewhere in steppe. The numerator is a small
difference of products and is prone to catastrophic cancellation, so it is one of
the operations carved out to always use true FP64. Because the accumulation order
is fixed (ascending SNP within a block, one grid block per block) and the
arithmetic is native FP64, the result is reproducible run to run and matches the
reference exactly.

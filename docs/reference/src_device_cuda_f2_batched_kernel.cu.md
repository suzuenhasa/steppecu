# `f2_batched_kernel.cu` reference

## 1. Purpose

This file implements the batched, per-block path for computing the f2 statistics on
the GPU. f2 is a pairwise statistic: for every ordered pair of populations `(i, j)`
it produces one number, so the full result for one block of SNPs is a `P × P`
matrix, where `P` is the number of populations.

The genome is partitioned into many small blocks of SNPs (used later for a block
jackknife that estimates uncertainty). This file computes a separate `P × P` f2
matrix for *every* block and writes all of them into two large resident tensors that
stay in GPU memory: the f2 tensor and its paired-variance (jackknife weight) tensor,
each shaped `P × P × n_block`.

The naive way to do that is to run one matrix multiply per block. That turns out to
be slow because each block is tiny, so the work is dominated by the fixed cost of
launching thousands of small operations. The way chosen here is **size-grouped
strided-batched**: blocks are sorted into buckets by how many SNPs they contain,
and every bucket is computed in a single batched matrix multiply. Section 2 explains
why this specific design was chosen over the two obvious alternatives.

This is GPU-internal code. The functions that host code calls are the three `launch_*`
/ `run_*` wrappers; the actual GPU kernels and the batched-matrix-multiply calls live
only inside this file. The per-pair f2 formula itself is not reimplemented here — it
comes from a shared primitive that the CPU reference path also calls, so the two can
never disagree on the arithmetic.

---

## 2. Why size-grouped strided-batched

Three designs were measured on real data (768 populations, ~584,000 SNPs). The
chosen one won on every axis. The numbers below are the measured spike results and
should be treated as the rationale of record, not re-derived guesses.

| Design | What it does | Verdict |
|---|---|---|
| **Loop of per-block matrix multiplies** | One small matrix multiply per block. | Launch-bound: ~2,271 tiny launches, 591 ms at 40-bit mantissa. Slow because each block is too small to hide the launch overhead. |
| **Naive global-max strided-batched** | One big batched call, every block padded to the size of the *largest* block. | Not viable on memory: needs `P² · s_max · n_block` ≈ 53.8 GB, which does not fit in 32 GB of GPU memory at `P = 768`. Also wasteful: 2.76× padding overhead. |
| **Size-grouped strided-batched (chosen)** | Bucket blocks by rounding their SNP count up to the next power of two; run one batched call per bucket, padding only to that bucket's width; keep only one bucket's slabs resident at a time. | Fastest **and** frugal: 317 ms (1.9× faster than the loop), only 1.43× padding overhead, and it fits in memory because just one bucket is resident. |

Rounding block sizes up to the next power of two is what keeps the padding waste
small: within a bucket, no block is more than 2× the size it was padded to, and the
number of buckets stays small (it grows only with the logarithm of the largest block
size). On the measured data, 768 populations produced 10 buckets.

The grouped path preserves the large speed advantage of doing these computations as
matrix multiplies: it runs about 7.2× faster than native double precision at a 40-bit
mantissa, and about 8.9× faster at 32-bit. Splitting the work per block does not throw
that win away.

---

## 3. The three-stage pipeline

Each bucket (size-group) is processed in three stages. Host code calls one wrapper
per stage, in this order, once per bucket:

1. **Gather** (`launch_gather_group`) — copy the SNP columns of every block in the
   bucket out of the big whole-genome input arrays into a compact, padded, batched
   layout ("slabs"), one slab per block, all padded to the bucket's common width.
2. **The three GEMMs** (`run_f2_gemms_group`) — run three batched matrix multiplies
   over the bucket's slabs to produce the raw building blocks of f2.
3. **Assemble and scatter** (`launch_assemble_blocks_group`) — combine those raw
   building blocks into the final f2 value for each pair, and write the results into
   the two large resident tensors at each block's own slot.

The earlier decoding stage that turns raw genotypes into per-population arrays over
*all* SNPs is not in this file. It is shared with the non-batched path and reused
unchanged, so there is only ever one copy of that per-element math.

The three inputs that stage 1 gathers from, computed by that earlier stage over all
`M` SNPs, are column-major arrays:

- `Q_all` and `V_all`, each shaped `P × M`. `Q` carries the per-population,
  per-SNP genotype term; `V` carries the corresponding weight (it doubles as the
  jackknife weight later).
- `S_all`, shaped `2P × M` — two stacked halves. The top `P` rows hold a
  sum-of-squares term per population; the bottom `P` rows hold a
  heterozygosity-correction term. Stacking them lets a single matrix multiply
  produce both.

---

## 4. Stage 1 — gather into padded slabs

`gather_group_kernel` (called through `launch_gather_group`) rearranges the bucket's
data into batched slabs so one strided-batched matrix multiply can sweep the whole
bucket.

**What each thread does.** Every thread owns one `(population row i, padded column c,
slab k)` position. Slab `k` corresponds to the global block whose id is
`block_ids_in_group[k]`. That block's SNPs occupy a contiguous run of columns in the
whole-genome input, starting at `block_offsets[id]` and running for `block_sizes[id]`
columns.

**The padding scheme.** Every slab in a bucket is `s_pad` columns wide, where `s_pad`
is the bucket's common (power-of-two) width. A block usually has fewer SNPs than that.

- For a real column (`c < block size`), the thread copies the matching column out of
  the whole-genome input into the slab — the `Q`/`V` value and both the top and bottom
  halves of the stacked `S` value.
- For a padding column (`c ≥ block size`), the thread writes **0** into `Q`, `V`, and
  both halves of `S`.

Writing zeros into the padding is what makes padding free of any effect on the answer.
The matrix multiplies in stage 2 all involve `V`, and a zero weight column contributes
nothing to the products. So a padded slab produces exactly the same `P × P` result as
if the block had been run at its true, unpadded width. The padding only wastes some
arithmetic, never the correctness of it.

**The slab layouts** (all column-major):

- `Qg`, `Vg`: shaped `P × s_pad × n_in_group`. Element `(i, c, k)` sits at
  `i + P·c + P·s_pad·k`.
- `Sg`: shaped `2P × s_pad × n_in_group`. Element `(i, c, k)` sits at
  `i + 2P·c + 2P·s_pad·k`. The top `P` rows and bottom `P` rows of each column hold
  the two stacked `S` terms.

The gather is entirely native double precision. It only moves memory around, so a
lower-precision copy would buy nothing.

---

## 5. Stage 2 — the three batched GEMMs

`run_f2_gemms_group` issues three strided-batched matrix multiplies over the bucket's
slabs. Each does the same shape of operation independently on all `n_in_group` slabs
(the "batch"). The contraction dimension is `s_pad` (the bucket width); the batch
count is `n_in_group` (the number of blocks in the bucket).

| Output | Formula (per slab) | Shape | Meaning |
|---|---|---|---|
| `Gg` | `Qg · Qgᵀ` | `P × P` | The main genotype cross-product. |
| `Vpairg` | `Vg · Vgᵀ` | `P × P` | The paired variance — the jackknife weight — which is retained and carried through to the final output unchanged. |
| `Rg` | `Sg · Vgᵀ` | `2P × P` | The two stacked `S` terms each crossed with `V`. Because `Sg` has `2P` rows, `Rg` has `2P` rows: the top `P` rows are the sum-of-squares term, the bottom `P` rows are the heterozygosity-correction term. |

The leading dimensions and per-slab strides are set so that each batched call steps
from one slab to the next: `Qg`/`Vg` slabs stride by `P·s_pad`, `Sg` slabs stride by
`2P·s_pad`, the `P × P` outputs stride by `P·P`, and the `2P × P` output strides by
`2P·P`.

**Precision.** Only these three matrix multiplies are governed by the precision knob
(see section 7). The selected precision is turned into a compute type and passed to
each call. Emulated double precision runs a fixed-slice scheme at the chosen mantissa
width; native double precision runs the standard 64-bit compute type.

**Stream and workspace contract — do not add a per-call stream set.** This routine
takes no stream and deliberately never calls the library's "set stream" function. The
matrix-multiply handle already has its stream and its fixed determinism workspace
bound to it once, when the backend was constructed. This routine runs once per bucket.
Calling "set stream" here would, as a side effect, reset the workspace back to the
default shared pool before every bucket's multiplies — which would break the exact
run-to-run reproducibility the fixed workspace exists to guarantee. So this routine
sets only the per-call compute type and leaves the handle's stream and math mode
exactly as the caller engaged them once, before the bucket loop began.

---

## 6. Stage 3 — fused assemble and scatter

`assemble_blocks_group_kernel` (called through `launch_assemble_blocks_group`) turns
the three raw GEMM outputs into the final f2 value for each population pair and writes
the results into the two large resident tensors.

**What each thread does.** Every thread owns one `(row i, column j, slab k)` output
entry. Slab `k` is the local result for global block `block_ids_in_group[k]`. The
thread reads that slab's `Gg`, `Vpairg`, and `Rg` entries, computes the numerator of
f2 through the shared primitive, divides it by the paired variance through a second
shared primitive to get the finished f2, and stores both the f2 value and the paired
variance into the resident tensors at the block's own `P × P` slot (base
`P·P·block_id`, offset `i + P·j`).

This step runs in **native double precision regardless of the selected precision
mode**. It is the numerically delicate part — a subtraction of nearly-equal large
quantities (catastrophic cancellation) — so it is always done in the highest-fidelity
arithmetic. The precision knob governs only the bulk matrix multiplies, never this
assembly.

**How the stacked `Rg` is read.** `Rg` for one slab is a `2P × P` column-major matrix.
Its top `P` rows hold the sum-of-squares term, its bottom `P` rows hold the
heterozygosity-correction term. Building the symmetric f2 for pair `(i, j)` needs the
`i`-side and the `j`-side of both terms, which are four different entries:

    sum-of-squares, i-side:   Rg(i,   j)
    sum-of-squares, j-side:   Rg(j,   i)
    het-correction, i-side:   Rg(P+i, j)
    het-correction, j-side:   Rg(P+j, i)

Notice the `i`-side entries are read from column `j`, and the `j`-side entries from
column `i` — the transpose of each other.

**The transpose is an accepted memory-access cost, not a bug.** Consecutive GPU threads
in a warp vary the row `i`. So the `i`-side reads (`Rg(i, j)` and `Rg(P+i, j)`), which
walk down a single column as `i` increases, are unit-stride and coalesced — efficient.
The `j`-side reads (`Rg(j, i)` and `Rg(P+j, i)`) instead vary the *column* as `i`
increases, so consecutive threads stride by `2P` (up to about 5,000 doubles at
`P = 2500`), which is uncoalesced — less efficient.

This transposed access is left as-is on purpose, for three reasons:

1. The symmetric f2 genuinely needs both the `i`-row and `j`-row column sums of the
   `2P × P` matrix, so the transpose is required by the math — it is not a layout
   mistake that a different storage order would fix.
2. It touches only the small per-slab `2P × P` matrix (`P` is at most a few thousand),
   not the SNP-scale arrays that dominate memory traffic elsewhere.
3. This assembly is not the hot part of the batched path — the three GEMMs are. Staging
   these reads through fast on-chip shared memory to make them coalesced would add real
   complexity (two off-diagonal tiles plus block-diagonal bounds and a new
   bank-conflict surface) for a bounded win on a step that is off the critical path.

If this assembly ever shows up as a bottleneck in profiling, the documented fix is to
stage the needed `Rg` rows in a padded shared-memory tile. The per-element math and
precision would be unchanged either way.

---

## 7. Precision: which stages use which arithmetic

There is a clean split, and it is deliberate:

- **The three batched matrix multiplies (stage 2) follow the selected precision.**
  The default is emulated double precision (fast, essentially native accuracy); native
  double precision is available as the reference and fallback. This is where almost all
  the arithmetic is, so this is where speed is bought.
- **The gather (stage 1) and the final assembly (stage 3) are always native double
  precision.** The gather just moves memory, so precision is irrelevant to it. The
  assembly performs the cancellation-prone subtraction that defines f2's accuracy, so it
  is pinned to the most faithful arithmetic no matter what mode the caller picked.

The single policy for turning the precision knob into a compute type, and for engaging
the emulated mode on the handle, is shared with the non-batched f2 path — there is one
source of the precision rules, not two.

---

## 8. Launch geometry and the degenerate-batch guards

**Thread-block shape.** All the kernels here launch with a fixed 16×16 (256-thread)
2-D block that tiles the output grid. Both kernels are marked so that their register
usage is capped to fit that exact block, which guarantees the launch cannot fail for
exceeding the register budget.

**Grid dimensions.** The grid's `x` and `y` axes tile the square output dimensions
(`P`, and for the gather also `s_pad`) through the shared launch-math helper, which
also asserts those axes stay under the hardware limit. The `z` axis is the batch
count (`n_in_group`) and is set directly, so it is routed through a dedicated guard
that both enforces the hardware limit on the `z` dimension and rejects a zero batch.

**Why the degenerate-batch asserts exist.** Several inputs must be at least 1, and a 0
would not simply do nothing — it would silently produce a wrong or empty result rather
than a clean failure:

- A **zero grid `y` or `z` extent** is an invalid launch the driver rejects. So the
  gather wrapper asserts `s_pad ≥ 1`, and the gather and assemble wrappers route their
  batch count through the guard that asserts `1 ≤ n_in_group ≤` the grid-`z` limit.
- The GEMM routine issues no kernel launch, so it cannot reuse that guard. But a
  **batch count of 0** is a degenerate empty batched multiply, and a **contraction
  width of 0** (`s_pad == 0`) is a scale-only operation with no actual product. Both
  quietly give the wrong answer. So this routine asserts `n_in_group ≥ 1` and
  `s_pad ≥ 1` directly.

In normal operation the backend always satisfies these: bucket widths are always at
least the power-of-two base, and the backend tiles a large batch into chunks no bigger
than the grid-`z` limit before calling in, so the count stays in range at every call.
The asserts are there to fail fast and loudly if a corrupted call ever reaches these
functions — for example, an empty SNP shard handed to a device under the sharded
multi-GPU path.

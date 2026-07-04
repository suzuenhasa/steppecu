# `f2_from_blocks.hpp` reference

## 1. Purpose

`src/core/fstats/f2_from_blocks.hpp` declares the two host-side functions that
assemble the **f2 statistic** from genotype data. f2 is a population-genetics
measure of how genetically different two populations are: roughly the average,
across genetic markers (SNPs), of the squared difference in allele frequency
between the two populations, with a correction applied so that a population
compared against a noisy estimate of itself does not look artificially far from
itself. f2 is the raw material that the higher-level f3 and f4 statistics — and
the qpAdm admixture models — are built from.

The functions in this header are the **orchestration layer**. They decide the
policy — how the work is organized, what the inputs must look like, which
precision to use — but they do not do the numerical work themselves. Instead they
hand the actual matrix math to an injected *backend*: either a CPU reference
implementation (the slow, exact oracle used to validate everything) or the GPU
implementation (the fast production path). The same orchestration code runs
unchanged against both, which is what lets the whole pipeline be tested without a
GPU present.

The header is deliberately **host-only and CUDA-free**. It names only plain C++
types and the CUDA-free backend interface, so it can be compiled into the core
library without dragging in the GPU toolkit. There are two entry points:

1. `compute_f2_block` — computes f2 for a **single block** of SNPs.
2. `compute_f2_blocks` — computes f2 for **every block at once**, producing a
   per-block tensor. This is the production path.

---

## 2. The Q/V/N inputs

Both functions take their genotype data as three matrices called **Q**, **V**,
and **N**. Together they are known as the Q/V/N contract, and they are the single
agreed-upon way genotype data is handed to the f2 math. Each is a column-major
`P × M` matrix, where `P` is the number of populations and `M` is the number of
SNPs. Column-major means the value for population `i` at SNP `s` lives at flat
index `i + P·s`.

| Matrix | What it holds |
|---|---|
| `Q` | The frequency of the fixed reference allele at each SNP, a number in `[0, 1]`. Entries that are invalid (missing) are filled with zero — that zero is deliberate, because it makes the masked matrix-multiply produce the correct result at invalid entries instead of contaminating them. |
| `V` | A validity mask: `1.0` where the population has a non-missing genotype at that SNP, `0.0` otherwise. |
| `N` | The non-missing **haploid** count — that is, alleles, not individuals: two per non-missing diploid sample, or one per non-missing pseudo-haploid sample (the ancient-DNA case). This count feeds only the bias-correction term. |

A guarantee the producer of these matrices makes, and that the f2 math relies on:
`V` is nonzero exactly when `N` is greater than zero.

These three matrices are passed as `MatView` objects — lightweight, non-owning
views that just describe where the data lives and how it is laid out. A `MatView`
does not own or free the memory; the caller keeps the data alive for the duration
of the call.

The f2 value that comes out for populations `i` and `j` is, per SNP that is valid
in both, `(p_i − p_j)² − hc_i − hc_j`, averaged over those jointly-valid SNPs.
Here `p` is the allele frequency and `hc` is the per-population heterozygosity
correction, `q(1−q)/max(N−1, 1)`, which removes the upward bias that sampling
noise would otherwise add. This is the unbiased estimator used for
parity[^at2].

---

## 3. `compute_f2_block` — one SNP block

```cpp
F2Result compute_f2_block(ComputeBackend& backend,
                          const MatView& Q, const MatView& V, const MatView& N,
                          const Precision& precision);
```

Computes the bias-corrected f2 matrix for a **single** block of SNPs. It takes
the Q/V/N matrices for that one block and returns an `F2Result`: a `P × P` f2
matrix plus a second `P × P` matrix of pairwise-valid counts (see section 8).

The `precision` argument only affects the GPU path's matrix multiplications; the
CPU reference oracle ignores it entirely and always computes in its own
high-precision mode. The result is marked `[[nodiscard]]`, so a caller cannot
accidentally throw it away.

---

## 4. `compute_f2_blocks` — the full per-block tensor

```cpp
F2BlockTensor compute_f2_blocks(ComputeBackend& backend,
                                const MatView& Q, const MatView& V, const MatView& N,
                                const BlockPartition& partition,
                                const Precision& precision);
```

This is the production entry point. Instead of one block, it takes the **full**
per-SNP Q/V/N (all `M` SNPs) plus a `partition` that says which block each SNP
belongs to, and it returns a `F2BlockTensor`: a stack of per-block f2 matrices,
one `P × P` slab per block, of shape `P × P × n_block`.

### Why per-block

The reason for computing f2 separately per block, rather than once over all SNPs,
is the **block jackknife** — the method steppe uses to estimate uncertainty. The
genome is partitioned into contiguous blocks of SNPs, f2 is computed within each
block, and the spread across blocks (leaving one out at a time) gives standard
errors. So the per-block tensor is not an optimization detail; it is the input the
uncertainty estimate needs.

### The partition

The `partition` argument (`BlockPartition`) carries two things: a per-SNP array
`block_id` giving each SNP's block number, and `n_block`, the number of distinct
blocks. This partition is produced elsewhere by a single shared rule that walks
the SNPs in file order and cuts a new block whenever the chromosome changes or the
cumulative genetic distance since the block started reaches the configured block
size. The orchestration here does not re-derive that rule; it just consumes the
partition. Keeping the block-assignment logic in exactly one place is what keeps
the CPU and GPU paths bit-for-bit consistent.

Under the hood the GPU backend batches these blocks efficiently (grouping
similarly-sized blocks and running them as padded batched matrix-multiplies),
while the CPU backend computes each block one at a time as the exact reference.
As with the single-block function, `precision` governs only the GPU matrix math.

---

## 5. The backend seam: policy vs implementation

Both functions take a `ComputeBackend&` as their first argument and route all the
heavy work through it. This is a deliberate split of responsibilities:

- The **orchestration** (this header's functions) owns the *policy*: what the
  inputs must satisfy, how the block structure is expressed, which precision is
  requested, and the up-front safety checks.
- The **backend** owns the *implementation*: the actual matrix multiplies and,
  on the GPU, the memory management and kernel launches.

Two backends implement the same interface. `CpuBackend` is the reference oracle —
slow but exact, used to validate the fast path. `CudaBackend` is the production
GPU path, which reformulates f2 as a sequence of matrix multiplies (a "3-GEMM"
formulation). Because the orchestration only ever talks to the abstract backend
interface, the exact same orchestration code drives both. That dependency
injection is what makes the pipeline testable without a GPU, and it is also why
this header stays CUDA-free: it names the backend interface but never any GPU
type.

---

## 6. Precision

Both functions take a `Precision` argument, but its reach is narrow: it controls
**only** the precision of the matrix-multiply-heavy stages on the GPU. It does not
affect the numerically delicate parts of the computation (which always run in high
precision regardless), and it does not affect the CPU oracle at all — the oracle
computes in its own exact mode and ignores the argument entirely.

The default precision is an emulated form of double precision that is much faster
than the GPU's native double precision at essentially the same accuracy. The full
meaning of the precision modes lives with the `Precision` type itself; here it is
enough to know that this knob steers the GPU's bulk arithmetic and nothing else.

---

## 7. Preconditions and fail-fast checks

Both functions document preconditions that the caller must satisfy, and both check
them up front in **debug builds only**. In a release build (`NDEBUG`) the checks
are compiled out entirely, so the hot path pays nothing; a violated precondition
in release is undefined behavior, and the caller is responsible for meeting the
contract.

The reason the checks live here, and not inside the backend, is that this
orchestration is the one place in the code that sees all three Q/V/N views
*together with* the full partition. Some of that information — notably the length
of the `block_id` array — is erased before the backend sees it (the backend
receives a bare pointer and length pair that no longer knows the original array's
size). So catching a malformed input here turns a silent out-of-bounds read or
write deep inside the backend into an immediate, located abort with a file and
line number.

**`compute_f2_block`** requires that Q, V, and N all share the same `P` and `M`
and have non-negative dimensions — the same shape contract the backend interface
documents.

**`compute_f2_blocks`** requires all of the above *plus* a well-formed partition:

- `partition.block_id.size()` must equal `Q.M` — the partition must describe
  exactly the `M` SNPs being passed. This is the check that catches a short or
  null `block_id`, which the backend cannot detect on its own.
- `0 < n_block <= M` — there is at least one block, and never more blocks than
  SNPs.
- `block_id` must be dense and non-decreasing over the range `[0, n_block)` — the
  block numbers go up (or stay equal) as you scan the SNPs and cover every value
  from `0` up to `n_block − 1` with no gaps.

---

## 8. Return types and the retained pairwise counts

### `F2Result` (from `compute_f2_block`)

A plain host struct holding three fields: the `f2` matrix (column-major `P × P`),
a `vpair` matrix (also column-major `P × P`), and the population count `P`. Being
a plain vector-backed struct is what lets it cross the CUDA-free boundary — the
GPU backend copies its device results into these host vectors before returning.

**The diagonal is filled, not zeroed.** `f2(i, i)` carries the full computation
rather than a forced `0`. Since `p_i − p_i` is zero, the per-SNP term reduces to
`−2·hc_i`, so the diagonal is minus twice the within-population heterozygosity
correction — a genuine (usually nonzero) within-population quantity, not a
between-population f2. Both backends produce the same diagonal by construction.
Downstream code never reads the diagonal (f3 and f4 use only off-diagonal f2), but
keeping it consistent means a comparison between the CPU and GPU results can check
the entire matrix, diagonal included.

### `F2BlockTensor` (from `compute_f2_blocks`)

A struct holding the full per-block stack: an `f2` vector of `P × P × n_block`
values, a matching `vpair` vector, a `block_sizes` array (the SNP count per
block), and `P` and `n_block`. It provides accessors to index an individual
element `(i, j, b)` or to grab one block's `P × P` slab as a span.

### Why the pairwise-valid counts are kept

Both result types carry a `vpair` (Vpair) matrix alongside the f2 values. Element
`(i, j)` is the number of SNPs valid in **both** population `i` and population
`j`. This is retained — not discarded after computing f2 — because it is the
**weight** the block jackknife needs later. The per-pair averaging done here and
the jackknife weighting done downstream must compose to exactly the parity
definition[^at2]; carrying Vpair forward is what lets the later stage weight each block
correctly instead of double-normalizing. The Vpair diagonal, like the f2 diagonal,
is filled (it is population `i`'s own valid-SNP count) and agrees across backends.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>

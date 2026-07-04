# `fstats.hpp` reference

## 1. Purpose

`include/steppe/fstats.hpp` defines the single public data structure that carries
steppe's f-statistics result: `F2BlockTensor`. An "f2" is a pairwise, distance-like
quantity between two populations, computed from allele frequencies; it is the raw
material that every downstream f-statistic (f3, f4, D, and the qpAdm model fit) is
built from. This structure holds one f2 value for every pair of populations, split
out separately for each jackknife block along the genome, together with the
bookkeeping needed to turn those per-block values into estimates with standard
errors.

The header holds exactly one struct plus its accessor methods. There are no free
functions and no algorithms here — it is a plain container that both computational
phases agree on.

The header is intentionally free of any CUDA code. It only uses the C++ standard
library (`<cstddef>`, `<span>`, `<vector>`). That keeps it lightweight enough to be
included everywhere — the core library, the public API, the command-line tool, and
the Python bindings — without forcing any of them to also pull in the GPU code.

---

## 2. Why this header lives in `include/`

`F2BlockTensor` is the hand-off point between steppe's two computational phases:
the GPU precompute engine that turns raw genotypes into per-block f2 values, and the
much smaller linear-algebra fit engine that consumes those values to fit models.
Because it sits on that boundary, it is also the cacheable, parity-compatible
interchange artifact[^at2] — the "compute the f2 values once, fit many models against them
later" cache.

It lives in `include/` rather than deep inside the GPU source for a deliberate
reason: it is plain, host-accessible, double-precision storage, so it crosses the
"no CUDA in this header" boundary cleanly. That lets the core library, the CLI, and
the Python bindings all read and pass around an f2 result without dragging in the
device toolkit. Any code can hold an `F2BlockTensor`; only the GPU engine needs the
CUDA compiler to *produce* one.

---

## 3. Storage is always FP64

steppe has a precision knob that chooses which flavor of floating-point arithmetic
its heavy matrix multiplications run in. That knob is an **operation** mode — it
governs *how the numbers are computed*, never *how they are stored*. The tensors in
`F2BlockTensor` are always native double precision (FP64), in every precision mode.

There is a memory consequence worth knowing. The GPU path keeps **both** large
tensors resident at once: the f2 values and the paired per-block valid-count
(`vpair`, described below). Each of those is `P × P × n_block × 8` bytes, so the two
together occupy `2 × P² × n_block × 8` bytes. Both of those terms have to be counted
when reserving GPU memory. Counting only the f2 tensor under-reserves the resident
working set by a factor of two and causes an out-of-memory failure partway through
the computation. The device memory-budget helper reserves for both tensors for
exactly this reason.

---

## 4. The `F2BlockTensor` memory layout

Both the `f2` array and the `vpair` array store the same shape: a stack of
`n_block` matrices, each `P × P`, where `P` is the number of populations. The value
for (population `i`, population `j`, block `b`) lives at the flat array index:

```
i + P·j + P·P·b
```

Reading that formula: the block index `b` is the slowest-varying (outermost) axis,
so all of block `b`'s data is contiguous; within one block the two population
indices are laid out column-major (`i` varies fastest, then `j`). This is the same
`[P × P × n_block]` tensor the fit engine works with, and the `n_block` axis is the
batch axis the jackknife contracts over — the fit engine loops over blocks to
estimate uncertainty.

Each individual `P × P` slab is **symmetric**: the f2 value for the pair (`i`, `j`)
equals the value for (`j`, `i`).

---

## 5. The `F2BlockTensor` fields

| Field | Type | Meaning |
|---|---|---|
| `f2` | `vector<double>` | The per-block, bias-corrected f2 values. Entry `f2[i + P·j + P·P·b]` is the unbiased f2 for the population pair (`i`, `j`), computed over just the SNPs in block `b` that have non-missing data in *both* population `i` and population `j` (the "pairwise-complete" path — each pair uses exactly the SNPs valid for that pair). Matches the parity unbiased f2 definition[^at2]. Length `P · P · n_block`. |
| `vpair` | `vector<double>` | The per-block pairwise-valid SNP count. Entry `vpair[i + P·j + P·P·b]` is *how many* SNPs in block `b` were valid in both population `i` and population `j`. This is kept, not discarded, because it is the weight the block jackknife needs — blocks with more valid SNPs count for more. Integer-valued, but stored as `double` to sit alongside `f2`. Length `P · P · n_block`. |
| `block_sizes` | `vector<int>` | The number of SNPs assigned to each block. `block_sizes[b]` is the SNP count of block `b`; summing over all blocks gives the total SNP count. Length `n_block`. |
| `P` | `int` | The number of populations — the side length of each `P × P` slab. Defaults to `0`. |
| `n_block` | `int` | The number of jackknife blocks — the length of the outer/batch axis. Defaults to `0`. |

### The `vpair` weighting invariant

`vpair` and `f2` interact in a way that is easy to get wrong. The per-block f2 value
is already divided by its per-block valid count when it is produced, and the block
jackknife *also* weights each block by its valid count. These two steps must
**compose** into the parity `f2_blocks` definition[^at2] — they must not both divide by
the count and thereby normalize twice. Keeping `vpair` around (rather than folding it
away early) is what lets the later jackknife apply exactly the one weighting the
reference expects.

### `size()`

`size()` returns the flat element count `P · P · n_block` — the length of both the
`f2` and `vpair` arrays — as a convenience. The multiplication is done in
`std::size_t`, not `int`, for the overflow reason described in section 7.

---

## 6. The diagonal convention

The diagonal entries of each slab — the (`i`, `i`) values, an f2 of a population with
itself — are **not** forced to zero. They carry the full self-computation, which
works out to minus two times that population's within-population heterozygosity
correction and is generally a nonzero number.

This is deliberate, and it is the same diagonal convention used by steppe's
single-block f2 result. Keeping the convention identical everywhere means three code
paths that could otherwise disagree — the batched GPU path, the per-block CPU
reference oracle, and the original single-block result — all produce the same
diagonal. The downstream f3 and f4 statistics only ever read *off-diagonal* f2
values, so the diagonal is never actually consumed by a calculation; it is kept
consistent purely so the different code paths remain bit-for-bit comparable.

---

## 7. Accessors and the flat-index helper

Rather than have callers compute `i + P·j + P·P·b` by hand, the struct exposes typed
accessors, and every one of them routes through a single private helper,
`flat_index(i, j, b)`. That helper is the one canonical home for the index formula,
so the layout convention cannot quietly drift apart from the code that writes the
tensor or the GPU kernel that fills it.

| Accessor | What it gives you |
|---|---|
| `f2_at(i, j, b)` | The f2 value for (`i`, `j`, `b`). Has a const read overload and a mutable overload that returns a reference you can assign into. No bounds checking — it is on the hot path. |
| `vpair_at(i, j, b)` | The valid-count counterpart of `f2_at`, indexing into `vpair` with the identical formula. Const and mutable overloads. |
| `block(b)` | The contiguous `P × P` f2 slab for block `b`, returned as a `std::span<const double>` of length `P · P` (column-major `i + P·j` within the slab). A read-only view — no copy. |
| `size()` | The flat element count (see section 5). |

### Why the index math uses `std::size_t`

Every one of these methods computes its offset in `std::size_t`, never plain `int`.
At production scale — on the order of 768 populations and 800 blocks — the flat index
`P · P · n_block` runs into the hundreds of millions and exceeds the largest value a
signed 32-bit `int` can hold. An `int` computation would overflow and produce a
wrong (often negative) offset, corrupting reads and writes silently. Casting the
operands to `std::size_t` before multiplying keeps the arithmetic correct at that
scale. This is the same cast pattern `size()` uses, and it is the reason the private
`flat_index` helper exists: get the formula and the wide arithmetic right in one
place, and every accessor inherits it.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
